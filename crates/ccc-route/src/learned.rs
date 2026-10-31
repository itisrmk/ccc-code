//! A small local classifier that decides whether a conversation can start on a cheaper model.
//!
//! It's logistic regression over a request's words, trained from labelled examples on this
//! machine. A conversation has to stay on one model, because caches and thinking blocks are
//! tied to it, so the choice is made once at the start and saved for every later call.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Duration;

use ccc_pipeline::{Context, Decisions, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::Request;
use serde_json::{Value, json};

use crate::classify::{CallKind, classify};
use crate::route::is_downgrade;

const LEARNING_RATE: f64 = 0.2;
const EPOCHS: usize = 200;
/// Keeps weights small, which matters with few examples and many distinct words.
const WEIGHT_DECAY: f64 = 0.001;
const TIME_CAP: Duration = Duration::from_millis(10);

/// Word-based logistic regression.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextClassifier {
    bias: f64,
    weights: BTreeMap<String, f64>,
}

fn features(text: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

impl TextClassifier {
    /// Trains on `(text, label)` pairs. The result depends only on the examples and their
    /// order, so training is repeatable.
    #[must_use]
    pub fn train(examples: &[(String, bool)]) -> Self {
        let mut model = Self::default();
        let prepared: Vec<(BTreeSet<String>, f64)> = examples
            .iter()
            .map(|(text, label)| (features(text), f64::from(u8::from(*label))))
            .collect();
        for _ in 0..EPOCHS {
            for (words, label) in &prepared {
                let error = label - model.score(words);
                model.bias += LEARNING_RATE * error;
                for word in words {
                    let weight = model.weights.entry(word.clone()).or_default();
                    *weight += LEARNING_RATE * (error - WEIGHT_DECAY * *weight);
                }
            }
        }
        model
    }

    fn score(&self, words: &BTreeSet<String>) -> f64 {
        let sum: f64 = words.iter().filter_map(|word| self.weights.get(word)).sum();
        sigmoid(self.bias + sum)
    }

    /// The probability that `text` has the positive label.
    #[must_use]
    pub fn probability(&self, text: &str) -> f64 {
        self.score(&features(text))
    }

    /// Writes the model as JSON.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the file can't be written.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        fs::write(
            path,
            json!({ "bias": self.bias, "weights": self.weights }).to_string(),
        )
    }

    /// Reads a saved model. Returns `None` if the file is missing or isn't one.
    #[must_use]
    pub fn load(path: &Path) -> Option<Self> {
        let saved: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
        let weights = saved
            .get("weights")?
            .as_object()?
            .iter()
            .filter_map(|(word, weight)| Some((word.clone(), weight.as_f64()?)))
            .collect();
        Some(Self {
            bias: saved.get("bias")?.as_f64()?,
            weights,
        })
    }
}

/// Accuracy on data held out in turn, compared with always guessing the commoner label.
/// Returns `(model accuracy, majority accuracy)`.
#[must_use]
pub fn cross_validate(examples: &[(String, bool)], folds: usize) -> (f64, f64) {
    if examples.is_empty() || folds < 2 {
        return (0.0, 0.0);
    }
    let mut correct = 0;
    for fold in 0..folds {
        let training: Vec<(String, bool)> = examples
            .iter()
            .enumerate()
            .filter(|(index, _)| index % folds != fold)
            .map(|(_, example)| example.clone())
            .collect();
        let model = TextClassifier::train(&training);
        correct += examples
            .iter()
            .enumerate()
            .filter(|(index, _)| index % folds == fold)
            .filter(|(_, (text, label))| (model.probability(text) >= 0.5) == *label)
            .count();
    }
    let total = examples.len() as f64;
    let positive = examples.iter().filter(|(_, label)| *label).count() as f64;
    (
        correct as f64 / total,
        positive.max(total - positive) / total,
    )
}

/// Starts a conversation on a cheaper model when a classifier is confident it's enough.
#[derive(Debug)]
pub struct LearnedRoute {
    classifier: TextClassifier,
    cheap_model: String,
    min_confidence: f64,
    decisions: Decisions,
}

impl LearnedRoute {
    /// A stage moving confident conversations to `cheap_model`, saving each choice in
    /// `decisions`.
    #[must_use]
    pub fn new(
        classifier: TextClassifier,
        cheap_model: String,
        min_confidence: f64,
        decisions: Decisions,
    ) -> Self {
        Self {
            classifier,
            cheap_model,
            min_confidence,
            decisions,
        }
    }
}

fn message_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            let parts: Vec<&str> = blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .filter(|text| !text.trim_start().starts_with('<'))
                .collect();
            parts.join("\n")
        }
        _ => String::new(),
    }
}

impl Stage for LearnedRoute {
    fn name(&self) -> &'static str {
        "route-learned"
    }

    fn time_cap(&self) -> Duration {
        TIME_CAP
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, _: &Context) -> Result<Vec<Edit>, StageError> {
        let Some(first) = request.messages().first() else {
            return Ok(Vec::new());
        };
        let current = request.model().unwrap_or_default();
        if !is_downgrade(current, &self.cheap_model) || classify(request) == CallKind::Background {
            return Ok(Vec::new());
        }
        let opening = message_text(first);
        let identity = [opening.as_str(), current];
        let chosen = match self.decisions.get(&identity) {
            Some(saved) => saved,
            None => {
                let starting = !request.messages().iter().any(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                });
                if !starting {
                    // Already under way on its own model, and moving it would lose its cache.
                    return Ok(Vec::new());
                }
                let cheap_enough = self.classifier.probability(&opening) >= self.min_confidence;
                let choice = cheap_enough.then_some(self.cheap_model.as_str());
                if !self.decisions.save(&identity, choice) {
                    return Ok(Vec::new());
                }
                choice.map(str::to_owned)
            }
        };
        Ok(chosen
            .map(|model| Edit {
                target: Target::Model,
                pointer: String::new(),
                value: json!(model),
                rule: "learned",
            })
            .into_iter()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn examples() -> Vec<(String, bool)> {
        let mut examples = Vec::new();
        for word in ["typo", "rename", "comment", "format", "spelling", "lint"] {
            examples.push((format!("fix the {word} in the readme"), true));
            examples.push((format!("quick {word} change please"), true));
        }
        for word in [
            "architecture",
            "migration",
            "concurrency",
            "deadlock",
            "rewrite",
            "security",
        ] {
            examples.push((format!("redesign the {word} across all services"), false));
            examples.push((format!("investigate the {word} problem deeply"), false));
        }
        examples
    }

    #[test]
    fn learns_to_separate_the_examples_and_is_repeatable() {
        let model = TextClassifier::train(&examples());
        assert!(model.probability("fix the typo") > 0.7);
        assert!(model.probability("investigate the deadlock") < 0.3);
        assert_eq!(model, TextClassifier::train(&examples()));
    }

    #[test]
    fn beats_guessing_on_held_out_examples() {
        let (accuracy, majority) = cross_validate(&examples(), 4);
        assert!(accuracy > majority, "{accuracy} vs {majority}");
        assert_eq!(cross_validate(&[], 4), (0.0, 0.0));
    }

    #[test]
    fn saved_models_load_identically() {
        let path = std::env::temp_dir().join(format!("ccc-router-{}.json", std::process::id()));
        let model = TextClassifier::train(&examples());
        model.save(&path).unwrap();
        let loaded = TextClassifier::load(&path).unwrap();
        assert!(
            (loaded.probability("fix the typo") - model.probability("fix the typo")).abs() < 1e-9
        );
        fs::remove_file(&path).unwrap();
        assert_eq!(TextClassifier::load(&path), None);
    }

    fn stage(name: &str) -> (LearnedRoute, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("ccc-learned-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let stage = LearnedRoute::new(
            TextClassifier::train(&examples()),
            "claude-haiku-5-5".to_owned(),
            0.7,
            Decisions::new(dir.clone()),
        );
        (stage, dir)
    }

    fn request(model: &str, messages: &Value) -> Request {
        let body = json!({ "model": model, "tools": [{ "name": "Bash" }], "messages": messages });
        Request::parse(body.to_string().as_bytes()).unwrap()
    }

    fn opened_with(text: &str) -> Value {
        json!([{ "role": "user", "content": text }])
    }

    #[test]
    fn a_conversation_stays_on_the_model_it_started_on() {
        let (stage, dir) = stage("pinned");
        let first = request(
            "claude-opus-5-5",
            &opened_with("fix the typo in the readme"),
        );
        let chosen = stage.plan(&first, &Context::default()).unwrap();
        assert_eq!(chosen[0].value, "claude-haiku-5-5");

        let later = request(
            "claude-opus-5-5",
            &json!([
                { "role": "user", "content": "fix the typo in the readme" },
                { "role": "assistant", "content": "ok" },
                { "role": "user", "content": "now redesign the architecture" },
            ]),
        );
        let again = stage.plan(&later, &Context::default()).unwrap();
        assert_eq!(again[0].value, "claude-haiku-5-5");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hard_work_stays_put_and_conversations_already_under_way_are_not_moved() {
        let (stage, dir) = stage("stays");
        let hard = request(
            "claude-opus-5-5",
            &opened_with("investigate the deadlock problem deeply"),
        );
        assert!(stage.plan(&hard, &Context::default()).unwrap().is_empty());

        let under_way = request(
            "claude-opus-5-5",
            &json!([
                { "role": "user", "content": "fix the typo please" },
                { "role": "assistant", "content": "ok" },
            ]),
        );
        assert!(
            stage
                .plan(&under_way, &Context::default())
                .unwrap()
                .is_empty()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn never_upgrades() {
        let (stage, dir) = stage("upgrade");
        let small = request("claude-haiku-5-5", &opened_with("fix the typo"));
        assert!(stage.plan(&small, &Context::default()).unwrap().is_empty());
        let _ = fs::remove_dir_all(dir);
    }
}

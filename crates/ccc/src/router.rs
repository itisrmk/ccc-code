//! Training the learned router and loading it into the pipeline.

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::Context as _;
use ccc_pipeline::Decisions;
use ccc_route::{LearnedRoute, TextClassifier, cross_validate};
use serde_json::Value;

/// Rounds of held-out testing.
const FOLDS: usize = 5;

/// Trains from JSON lines of `{prompt, cheap_enough}` and reports held-out accuracy.
pub(crate) fn train(labels: &Path, out: &Path) -> anyhow::Result<()> {
    let text =
        fs::read_to_string(labels).with_context(|| format!("reading {}", labels.display()))?;
    let examples = parse_examples(&text)?;
    let (accuracy, majority) = cross_validate(&examples, FOLDS);
    TextClassifier::train(&examples)
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    writeln!(
        std::io::stdout(),
        "{} examples; held-out accuracy {:.1}%, always guessing the commoner label {:.1}%",
        examples.len(),
        accuracy * 100.0,
        majority * 100.0
    )?;
    Ok(())
}

fn parse_examples(text: &str) -> anyhow::Result<Vec<(String, bool)>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(number, line)| {
            let described = || format!("label line {}", number + 1);
            let value: Value = serde_json::from_str(line).with_context(described)?;
            let prompt = value
                .get("prompt")
                .and_then(Value::as_str)
                .with_context(described)?;
            let label = value
                .get("cheap_enough")
                .and_then(Value::as_bool)
                .with_context(described)?;
            Ok((prompt.to_owned(), label))
        })
        .collect()
}

/// Loads the trained router as a stage.
pub(crate) fn stage(
    model: &Path,
    cheap_model: String,
    min_confidence: f64,
    state_dir: &Path,
) -> anyhow::Result<LearnedRoute> {
    let classifier = TextClassifier::load(model)
        .with_context(|| format!("reading router model {}", model.display()))?;
    Ok(LearnedRoute::new(
        classifier,
        cheap_model,
        min_confidence,
        Decisions::new(state_dir.join("router-decisions")),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_examples_and_names_the_bad_line() {
        let parsed = parse_examples(concat!(
            r#"{"prompt":"fix typo","cheap_enough":true}"#,
            "\n\n",
            r#"{"prompt":"rewrite it","cheap_enough":false}"#,
        ))
        .unwrap();
        assert_eq!(
            parsed,
            [
                ("fix typo".to_owned(), true),
                ("rewrite it".to_owned(), false)
            ]
        );

        let err = parse_examples(r#"{"prompt":"no label"}"#).unwrap_err();
        assert!(err.to_string().contains("line 1"), "{err}");
    }
}

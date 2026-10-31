//! Labels a user turn's intent and difficulty.
//!
//! Nothing here changes a request, so the result doesn't need to be reproducible. A label
//! that doesn't arrive in time is simply absent.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ccc_pipeline::{Context, Edit, OnFailure, Stage, StageError};
use ccc_wire::{Request, text_of_tool_result};
use serde_json::Value;

use crate::model::{CompletionRequest, ModelError, TextModel};

/// The spec's budget for classifying, since it runs on every user turn in the request path.
pub const DEFAULT_CLASSIFY_CAP: Duration = Duration::from_millis(80);
/// How much of the user's message to show the model.
const MAX_MESSAGE_CHARS: usize = 800;
/// How much of the last tool result to show it.
const MAX_RESULT_CHARS: usize = 400;
/// Enough tokens for the longest pair of labels.
const MAX_LABEL_TOKENS: u32 = 8;

const GRAMMAR: &str = r#"root ::= intent " " difficulty
intent ::= "explore" | "edit" | "debug" | "explain" | "plan"
difficulty ::= "easy" | "medium" | "hard"
"#;

/// Fixed text ahead of the request, so a model that caches a shared prompt prefix only has
/// to read the new part. These examples are deliberately not from the measuring set.
const INSTRUCTIONS: &str = "\
Label a request to a coding agent with an intent and a difficulty.

Intents:
explore = find out where something is or how it works, in this codebase
edit = change code or files
debug = find out why something is broken, or fix a reported failure
explain = teach the user what something means
plan = decide on an approach before doing the work

Difficulties:
easy = one small, contained step
medium = ordinary work across a few files
hard = broad, subtle or risky work

Request: which file sets up the logger?
Labels: explore easy
Request: trace how a payment moves through all the services
Labels: explore hard
Request: change the default port to 8080
Labels: edit easy
Request: rewrite the scheduler to use async tasks throughout
Labels: edit hard
Request: the login test started failing after the merge
Labels: debug medium
Request: production crashes randomly under load and nobody knows why
Labels: debug hard
Request: what does this error message mean?
Labels: explain easy
Request: walk me through why this lifetime annotation is needed
Labels: explain medium
Request: how should we structure the new plugin system? outline options first
Labels: plan hard
Request: before coding, sketch the steps to add dark mode
Labels: plan medium

";

/// What the user wants done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intent {
    /// Find out how something works or where it is.
    Explore,
    /// Change code.
    Edit,
    /// Find out why something is broken.
    Debug,
    /// Explain something to the user.
    Explain,
    /// Decide how to approach something.
    Plan,
}

/// How hard the request looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    /// A quick, contained change or question.
    Easy,
    /// Ordinary work.
    Medium,
    /// Broad or subtle work.
    Hard,
}

/// A model's read of a user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Labels {
    /// What the user wants.
    pub intent: Intent,
    /// How hard it looks.
    pub difficulty: Difficulty,
}

impl fmt::Display for Labels {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} {:?}", self.intent, self.difficulty)
    }
}

fn parse_labels(text: &str) -> Option<Labels> {
    let (intent, difficulty) = text.trim().split_once(' ')?;
    let intent = match intent {
        "explore" => Intent::Explore,
        "edit" => Intent::Edit,
        "debug" => Intent::Debug,
        "explain" => Intent::Explain,
        "plan" => Intent::Plan,
        _ => return None,
    };
    let difficulty = match difficulty {
        "easy" => Difficulty::Easy,
        "medium" => Difficulty::Medium,
        "hard" => Difficulty::Hard,
        _ => return None,
    };
    Some(Labels { intent, difficulty })
}

/// Labels `user_text`, with `last_result` as context if there is one.
///
/// # Errors
///
/// Returns [`ModelError`] if the model fails or misses `deadline`, or if its output can't
/// be read as labels.
pub fn classify_labels(
    model: &dyn TextModel,
    user_text: &str,
    last_result: Option<&str>,
    deadline: Duration,
) -> Result<Labels, ModelError> {
    let message: String = user_text.chars().take(MAX_MESSAGE_CHARS).collect();
    let context = last_result.map_or_else(String::new, |result| {
        let shown: String = result.chars().take(MAX_RESULT_CHARS).collect();
        format!("Last tool output:\n{shown}\n\n")
    });
    let prompt = format!("{INSTRUCTIONS}{context}Request: {message}\nLabels:");
    let completion = model.complete(&CompletionRequest {
        prompt: &prompt,
        grammar: Some(GRAMMAR),
        max_tokens: MAX_LABEL_TOKENS,
        deadline,
    })?;
    parse_labels(&completion.text)
        .ok_or_else(|| ModelError::Inference(format!("unreadable labels {:?}", completion.text)))
}

/// The newest user turn's text and the tool result nearest before it.
fn turn_to_label(request: &Request) -> Option<(String, Option<String>)> {
    let last = request
        .messages()
        .iter()
        .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"))?;
    let text = match request.messages()[last].get("content")? {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => {
            let parts: Vec<&str> = blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect();
            if parts.is_empty() {
                return None;
            }
            parts.join("\n")
        }
        _ => return None,
    };
    let result = request.messages()[..last]
        .iter()
        .rev()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
        .and_then(|block| block.get("content").and_then(text_of_tool_result))
        .map(|text| text.into_owned());
    Some((text, result))
}

/// Records labels for each user turn. Makes no edits; other parts read [`Classify::latest`].
#[derive(Debug)]
pub struct Classify {
    model: Arc<dyn TextModel>,
    cap: Duration,
    latest: Mutex<Option<Labels>>,
}

impl Classify {
    /// A stage using `model`, giving up on a label after `cap`.
    #[must_use]
    pub fn new(model: Arc<dyn TextModel>, cap: Duration) -> Self {
        Self {
            model,
            cap,
            latest: Mutex::default(),
        }
    }

    /// The most recent labels produced.
    #[must_use]
    pub fn latest(&self) -> Option<Labels> {
        *self.latest.lock().expect("lock poisoned")
    }
}

impl Stage for Classify {
    fn name(&self) -> &'static str {
        "classify"
    }

    fn time_cap(&self) -> Duration {
        // A little over the model's own deadline, so it reports the miss itself.
        self.cap + self.cap / 4
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, _: &Context) -> Result<Vec<Edit>, StageError> {
        let Some((text, result)) = turn_to_label(request) else {
            return Ok(Vec::new());
        };
        let labels = classify_labels(self.model.as_ref(), &text, result.as_deref(), self.cap)
            .map_err(|err| StageError(err.to_string()))?;
        tracing::debug!(%labels, "classified");
        *self.latest.lock().expect("lock poisoned") = Some(labels);
        Ok(Vec::new())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::Completion;
    use serde_json::json;

    /// A model that always says the same thing.
    #[derive(Debug)]
    pub(crate) struct Fixed(pub(crate) Result<String, ModelError>);

    impl TextModel for Fixed {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            self.0.clone().map(|text| Completion {
                text,
                ..Completion::default()
            })
        }
    }

    fn request(messages: &Value) -> Request {
        Request::parse(json!({ "messages": messages }).to_string().as_bytes()).unwrap()
    }

    #[test]
    fn reads_every_label_the_grammar_allows() {
        for (intent, want) in [
            ("explore", Intent::Explore),
            ("edit", Intent::Edit),
            ("debug", Intent::Debug),
            ("explain", Intent::Explain),
            ("plan", Intent::Plan),
        ] {
            let labels = parse_labels(&format!("{intent} hard")).unwrap();
            assert_eq!(labels.intent, want);
            assert_eq!(labels.difficulty, Difficulty::Hard);
        }
        assert_eq!(parse_labels("nonsense"), None);
        assert_eq!(parse_labels("edit impossible"), None);
    }

    #[test]
    fn labels_the_newest_user_text_and_never_edits() {
        let stage = Classify::new(
            Arc::new(Fixed(Ok("debug easy".to_owned()))),
            DEFAULT_CLASSIFY_CAP,
        );
        let messages = json!([{ "role": "user", "content": "why does this crash" }]);

        let edits = stage
            .plan(&request(&messages), &Context::default())
            .unwrap();

        assert!(edits.is_empty());
        assert_eq!(
            stage.latest(),
            Some(Labels {
                intent: Intent::Debug,
                difficulty: Difficulty::Easy
            })
        );
    }

    #[test]
    fn a_tool_loop_turn_with_no_new_text_is_skipped() {
        let stage = Classify::new(
            Arc::new(Fixed(Err(ModelError::Stopped))),
            DEFAULT_CLASSIFY_CAP,
        );
        let messages = json!([{ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": "a", "content": "output" },
        ] }]);
        assert!(stage.plan(&request(&messages), &Context::default()).is_ok());
    }

    #[test]
    fn a_failing_model_is_a_stage_failure_that_passes_through() {
        let stage = Classify::new(
            Arc::new(Fixed(Err(ModelError::DeadlineExceeded))),
            DEFAULT_CLASSIFY_CAP,
        );
        let messages = json!([{ "role": "user", "content": "hi" }]);
        assert!(
            stage
                .plan(&request(&messages), &Context::default())
                .is_err()
        );
        assert_eq!(stage.on_failure(), OnFailure::PassThrough);
    }

    #[test]
    fn prompts_stay_bounded_whatever_the_input_size() {
        #[derive(Debug)]
        struct CheckLength;
        impl TextModel for CheckLength {
            fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
                assert!(request.prompt.len() < 4_000, "{}", request.prompt.len());
                Ok(Completion {
                    text: "edit easy".to_owned(),
                    ..Completion::default()
                })
            }
        }
        let huge = "x".repeat(1_000_000);
        classify_labels(&CheckLength, &huge, Some(&huge), DEFAULT_CLASSIFY_CAP).unwrap();
    }
}

//! A gated rewrite of a vague prompt.
//!
//! Small rewriters can help, but rewriting isn't always an improvement, so a proposal only
//! goes out if it stays close to the original and a classifier agrees the intent is unchanged.

use std::collections::BTreeSet;
use std::time::Duration;

use ccc_index::terms;
use ccc_slm::{CompletionRequest, ModelError, TextModel, classify_labels};

/// A proposal longer than this many tokens has stopped being a clarification.
const MAX_REWRITE_TOKENS: u32 = 80;

/// How strict the gate is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateConfig {
    /// How much of the original's vocabulary a rewrite must share, from 0 to 1.
    pub min_similarity: f64,
    /// How long the classifier may take on each of the two checks.
    pub classify_deadline: Duration,
}

/// Asks the model to clarify `prompt`.
///
/// # Errors
///
/// Returns [`ModelError`] if the model fails or misses `deadline`.
pub fn propose_rewrite(
    model: &dyn TextModel,
    prompt: &str,
    deadline: Duration,
) -> Result<String, ModelError> {
    let instruction = format!(
        "Rewrite this request to a coding agent so it's clearer. Keep its meaning and every \
         specific name. Don't add requirements. Reply with only the rewritten request.\n\n{prompt}"
    );
    let completion = model.complete(&CompletionRequest {
        prompt: &instruction,
        grammar: None,
        max_tokens: MAX_REWRITE_TOKENS,
        deadline,
    })?;
    Ok(completion.text.trim().to_owned())
}

/// How much of `original`'s vocabulary `rewrite` kept.
fn vocabulary_kept(original: &str, rewrite: &str) -> f64 {
    let before: BTreeSet<String> = terms(original).into_iter().collect();
    let after: BTreeSet<String> = terms(rewrite).into_iter().collect();
    if before.is_empty() {
        return 0.0;
    }
    before.intersection(&after).count() as f64 / before.len() as f64
}

/// Whether `rewrite` may replace `original`. Any doubt, including a classifier failure,
/// keeps the original.
#[must_use]
pub fn gate_rewrite(
    model: &dyn TextModel,
    original: &str,
    rewrite: &str,
    config: GateConfig,
) -> bool {
    if rewrite.is_empty() || vocabulary_kept(original, rewrite) < config.min_similarity {
        return false;
    }
    let before = classify_labels(model, original, None, config.classify_deadline);
    let after = classify_labels(model, rewrite, None, config.classify_deadline);
    match (before, after) {
        (Ok(before), Ok(after)) => before.intent == after.intent,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_slm::Completion;

    /// Labels text by whether it mentions a crash.
    #[derive(Debug)]
    struct ByWord;

    impl TextModel for ByWord {
        fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            let asked = request.prompt.rsplit("Request:").next().unwrap_or_default();
            let text = if asked.contains("crash") {
                "debug easy"
            } else {
                "edit easy"
            };
            Ok(Completion {
                text: text.to_owned(),
                ..Completion::default()
            })
        }
    }

    fn config() -> GateConfig {
        GateConfig {
            min_similarity: 0.6,
            classify_deadline: Duration::from_secs(1),
        }
    }

    #[test]
    fn keeps_a_close_rewrite_with_the_same_intent() {
        assert!(gate_rewrite(
            &ByWord,
            "update parser config handling",
            "Update the parser config handling code",
            config()
        ));
    }

    #[test]
    fn rejects_a_rewrite_that_drops_the_user_s_words() {
        assert!(!gate_rewrite(
            &ByWord,
            "update parser config handling",
            "Refactor the entire module",
            config()
        ));
    }

    #[test]
    fn rejects_a_rewrite_that_changes_intent() {
        assert!(!gate_rewrite(
            &ByWord,
            "update parser config handling",
            "update parser config handling crash",
            config()
        ));
    }

    #[test]
    fn rejects_an_empty_rewrite_and_when_the_classifier_fails() {
        #[derive(Debug)]
        struct Broken;
        impl TextModel for Broken {
            fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
                Err(ModelError::DeadlineExceeded)
            }
        }
        assert!(!gate_rewrite(&ByWord, "update parser", "", config()));
        assert!(!gate_rewrite(
            &Broken,
            "update parser",
            "update parser",
            config()
        ));
    }
}

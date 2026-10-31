use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ccc_pipeline::{Context, Decisions, Edit, OnFailure, Stage, StageError, Target};
use ccc_slm::TextModel;
use ccc_wire::Request;
use serde_json::{Value, json};

use crate::rewrite::{GateConfig, gate_rewrite, propose_rewrite};
use crate::suggest::Suggester;

/// How many files to suggest. More would spend tokens on increasingly weak guesses.
const SUGGESTED_FILES: usize = 3;
/// Short prompts are the vague ones a rewrite might help. Long ones already say what they mean.
const MAX_WORDS_TO_REWRITE: usize = 15;
/// Extra time the pipeline allows, so this stage finishes writing down a decision itself.
const SAVE_MARGIN: Duration = Duration::from_millis(100);
/// Prefixes of the turn's parts that were never typed by the user, and aren't rewritten.
const INJECTED_PREFIX: char = '<';

/// How the stage behaves.
#[derive(Debug, Clone, PartialEq)]
pub struct EnrichConfig {
    /// Where decisions are saved. They must outlive this process.
    pub decisions_dir: PathBuf,
    /// How long the whole stage may take on the live path.
    pub time_cap: Duration,
    /// Where each proposed rewrite and its verdict is logged, for tuning the gate.
    pub rewrite_log: Option<PathBuf>,
    /// The rewrite gate's settings.
    pub gate: GateConfig,
}

/// The Enrich stage. See the crate docs.
#[derive(Debug)]
pub struct Enrich {
    suggester: Arc<Suggester>,
    rewriter: Option<Arc<dyn TextModel>>,
    decisions: Decisions,
    config: EnrichConfig,
}

impl Enrich {
    /// A stage suggesting from `suggester`, and rewriting with `rewriter` if there is one.
    #[must_use]
    pub fn new(
        suggester: Arc<Suggester>,
        rewriter: Option<Arc<dyn TextModel>>,
        config: EnrichConfig,
    ) -> Self {
        let decisions = Decisions::new(config.decisions_dir.clone());
        Self {
            suggester,
            rewriter,
            decisions,
            config,
        }
    }

    /// Decides what this turn's content should become, or `None` to leave it alone.
    fn decide(&self, prompt: &str, content: &Value) -> Option<Value> {
        let started = Instant::now();
        let hints = self.suggester.hints(prompt, SUGGESTED_FILES);
        let rewritten = self.rewritten(prompt, started);
        if hints.is_none() && rewritten.is_none() {
            return None;
        }
        Some(compiled_content(
            content,
            prompt,
            rewritten.as_deref(),
            hints.as_deref(),
        ))
    }

    fn rewritten(&self, prompt: &str, started: Instant) -> Option<String> {
        let model = self.rewriter.as_ref()?;
        if prompt.split_whitespace().count() > MAX_WORDS_TO_REWRITE {
            return None;
        }
        let proposal = propose_rewrite(model.as_ref(), prompt, self.config.time_cap)
            .inspect_err(|err| tracing::debug!(error = %err, "rewrite skipped"))
            .ok()?;
        let passed = gate_rewrite(model.as_ref(), prompt, &proposal, self.config.gate);
        let in_time = started.elapsed() <= self.config.time_cap;
        self.log_rewrite(prompt, &proposal, passed, in_time);
        (passed && in_time).then_some(proposal)
    }

    fn log_rewrite(&self, prompt: &str, proposal: &str, passed: bool, in_time: bool) {
        let Some(path) = &self.config.rewrite_log else {
            return;
        };
        let entry = json!({
            "original": prompt, "proposal": proposal, "passed_gate": passed, "in_time": in_time,
        });
        let written = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut file| writeln!(file, "{entry}"));
        if let Err(err) = written {
            tracing::warn!(error = %err, "could not log a rewrite");
        }
    }
}

/// The text a person typed in a user turn: every text part that wasn't injected, and
/// nothing if the turn carries tool results, since then the model is mid-task.
fn typed_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()).filter(|text| !text.starts_with(INJECTED_PREFIX)),
        Value::Array(blocks) => {
            let mut typed = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if !text.trim_start().starts_with(INJECTED_PREFIX) {
                            typed.push(text);
                        }
                    }
                    Some("tool_result") => return None,
                    _ => {}
                }
            }
            (!typed.is_empty()).then(|| typed.join("\n"))
        }
        _ => None,
    }
}

/// Builds the content to send: the same blocks, with the typed text replaced if rewritten
/// and the hints added at the end.
fn compiled_content(
    content: &Value,
    typed: &str,
    rewritten: Option<&str>,
    hints: Option<&str>,
) -> Value {
    let mut blocks: Vec<Value> = match content {
        Value::Array(blocks) => blocks.clone(),
        other => vec![json!({ "type": "text", "text": other.as_str().unwrap_or_default() })],
    };
    if let Some(rewritten) = rewritten {
        for block in &mut blocks {
            let is_typed = block.get("type").and_then(Value::as_str) == Some("text")
                && block.get("text").and_then(Value::as_str) == Some(typed);
            if is_typed {
                block["text"] = json!(rewritten);
            }
        }
    }
    if let Some(hints) = hints {
        blocks.push(json!({ "type": "text", "text": hints }));
    }
    Value::Array(blocks)
}

impl Stage for Enrich {
    fn name(&self) -> &'static str {
        "enrich"
    }

    fn time_cap(&self) -> Duration {
        self.config.time_cap + SAVE_MARGIN
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let newest_user = request
            .messages()
            .iter()
            .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"));
        let mut edits = Vec::new();
        for (index, message) in request.messages().iter().enumerate() {
            if index < context.protected_before
                || message.get("role").and_then(Value::as_str) != Some("user")
            {
                continue;
            }
            let Some(content) = message.get("content") else {
                continue;
            };
            let Some(typed) = typed_text(content) else {
                continue;
            };
            let identity = [typed.as_str(), &content.to_string()];
            let saved = match self.decisions.get(&identity) {
                Some(saved) => saved.and_then(|text| serde_json::from_str(&text).ok()),
                None if Some(index) == newest_user => {
                    let decided = self.decide(&typed, content);
                    let as_text = decided.as_ref().map(Value::to_string);
                    // A decision that can't be saved couldn't be repeated, so it's void.
                    if self.decisions.save(&identity, as_text.as_deref()) {
                        decided
                    } else {
                        None
                    }
                }
                // Sent before any decision existed, so it was sent plain and stays plain.
                None => None,
            };
            if let Some(compiled) = saved {
                edits.push(Edit {
                    target: Target::Message(index),
                    pointer: "/content".to_owned(),
                    value: compiled,
                    rule: "enrich",
                });
            }
        }
        Ok(edits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_index::Index;
    use ccc_slm::{Completion, CompletionRequest, ModelError};
    use std::fs;
    use std::path::Path;

    fn workspace(name: &str) -> (PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!("ccc-enrich-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join("repo");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("auth.rs"), "pub fn verify_token() {}\n").unwrap();
        (root, base.join("decisions"))
    }

    fn config(decisions_dir: PathBuf) -> EnrichConfig {
        EnrichConfig {
            decisions_dir,
            time_cap: Duration::from_secs(5),
            rewrite_log: None,
            gate: GateConfig {
                min_similarity: 0.5,
                classify_deadline: Duration::from_secs(1),
            },
        }
    }

    fn stage(root: &Path, decisions: PathBuf, rewriter: Option<Arc<dyn TextModel>>) -> Enrich {
        let suggester = Arc::new(Suggester::new(root.to_owned(), Index::default(), None));
        suggester.refresh();
        Enrich::new(suggester, rewriter, config(decisions))
    }

    fn request(messages: &Value) -> Request {
        Request::parse(json!({ "messages": messages }).to_string().as_bytes()).unwrap()
    }

    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": text })
    }

    fn assistant() -> Value {
        json!({ "role": "assistant", "content": "ok" })
    }

    #[test]
    fn adds_hints_to_the_newest_user_turn_keeping_the_original_words() {
        let (root, decisions) = workspace("hints");
        let enrich = stage(&root, decisions, None);

        let edits = enrich
            .plan(
                &request(&json!([user("verify the token please")])),
                &Context::default(),
            )
            .unwrap();

        assert_eq!(edits.len(), 1);
        let blocks = edits[0].value.as_array().unwrap();
        assert_eq!(blocks[0]["text"], "verify the token please");
        let hint = blocks[1]["text"].as_str().unwrap();
        assert!(hint.starts_with("<ccc-context>"), "{hint}");
        assert!(hint.contains("auth.rs (verify_token)"), "{hint}");
    }

    #[test]
    fn nothing_relevant_means_no_edit() {
        let (root, decisions) = workspace("nothing");
        let enrich = stage(&root, decisions, None);
        let edits = enrich
            .plan(
                &request(&json!([user("zebras and giraffes")])),
                &Context::default(),
            )
            .unwrap();
        assert!(edits.is_empty());
    }

    /// The core guarantee: the index changes, and what went out earlier doesn't.
    #[test]
    fn an_earlier_turn_compiles_the_same_even_after_the_repo_changes() {
        let (root, decisions) = workspace("stable");
        let enrich = stage(&root, decisions.clone(), None);
        let first = enrich
            .plan(
                &request(&json!([user("verify the token")])),
                &Context::default(),
            )
            .unwrap();

        fs::write(root.join("verify_token.rs"), "pub fn verify_token() {}\n").unwrap();
        let later = stage(&root, decisions, None);
        let messages = json!([
            user("verify the token"),
            assistant(),
            user("unrelated zebras")
        ]);
        let again = later
            .plan(&request(&messages), &Context::default())
            .unwrap();

        assert_eq!(again.len(), 1);
        assert_eq!(again[0].value, first[0].value);
    }

    #[test]
    fn an_older_turn_with_no_decision_stays_plain() {
        let (root, decisions) = workspace("older");
        let enrich = stage(&root, decisions, None);
        let messages = json!([user("verify the token"), assistant(), user("zebras")]);
        assert!(
            enrich
                .plan(&request(&messages), &Context::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn tool_results_injected_text_and_protected_history_are_skipped() {
        let (root, decisions) = workspace("skip");
        let enrich = stage(&root, decisions, None);
        let tool = json!([{ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": "a", "content": "verify token" },
            { "type": "text", "text": "verify token" },
        ] }]);
        assert!(
            enrich
                .plan(&request(&tool), &Context::default())
                .unwrap()
                .is_empty()
        );

        let injected = json!([user("<system-reminder>verify token</system-reminder>")]);
        assert!(
            enrich
                .plan(&request(&injected), &Context::default())
                .unwrap()
                .is_empty()
        );

        let protected = Context {
            protected_before: 5,
            ..Context::default()
        };
        let plain = json!([user("verify the token")]);
        assert!(
            enrich
                .plan(&request(&plain), &protected)
                .unwrap()
                .is_empty()
        );
    }

    #[derive(Debug)]
    struct Rewriter;

    impl TextModel for Rewriter {
        fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            let text = if request.grammar.is_some() {
                "edit easy"
            } else {
                "Please verify the token in auth"
            };
            Ok(Completion {
                text: text.to_owned(),
                ..Completion::default()
            })
        }
    }

    #[test]
    fn a_short_prompt_is_replaced_when_the_rewrite_passes_and_every_proposal_is_logged() {
        let (root, decisions) = workspace("rewrite");
        let log = root.join("rewrites.jsonl");
        let suggester = Arc::new(Suggester::new(root.clone(), Index::default(), None));
        suggester.refresh();
        let mut settings = config(decisions);
        settings.rewrite_log = Some(log.clone());
        let enrich = Enrich::new(suggester, Some(Arc::new(Rewriter)), settings);

        let body = json!([{ "role": "user", "content": [
            { "type": "text", "text": "<system-reminder>context</system-reminder>" },
            { "type": "text", "text": "verify token" },
        ] }]);
        let edits = enrich.plan(&request(&body), &Context::default()).unwrap();

        let blocks = edits[0].value.as_array().unwrap();
        assert_eq!(
            blocks[0]["text"],
            "<system-reminder>context</system-reminder>"
        );
        assert_eq!(blocks[1]["text"], "Please verify the token in auth");
        assert!(
            blocks[2]["text"]
                .as_str()
                .unwrap()
                .starts_with("<ccc-context>")
        );
        let logged = fs::read_to_string(log).unwrap();
        assert!(logged.contains("\"passed_gate\":true"), "{logged}");
    }

    #[test]
    fn a_long_prompt_is_never_rewritten() {
        let (root, decisions) = workspace("long");
        let enrich = stage(&root, decisions, Some(Arc::new(Rewriter)));
        let long = "verify the token in this very long request that already says exactly what it means and needs nothing else at all";
        let edits = enrich
            .plan(&request(&json!([user(long)])), &Context::default())
            .unwrap();
        assert_eq!(edits[0].value[0]["text"], long);
    }
}

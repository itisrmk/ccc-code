use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::Duration;

use ccc_pipeline::{Context, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::{Request, text_of_tool_result};
use serde_json::{Value, json};

use crate::rules::{self, Limits, Trimmed};

/// Results shorter than this are repeated rather than replaced by a longer stub.
const MIN_DUPLICATE_CHARS: usize = 200;
/// Live budget for the rule-based stage. Rules run on new results only; everything older
/// comes from the memo, so this is generous for one call's worth of new output.
const TIME_CAP: Duration = Duration::from_millis(50);
/// The memo is only an optimization, so it is simply emptied when it grows this large.
const MAX_MEMO_ENTRIES: usize = 100_000;

/// The Prune stage. See the crate docs for its rules.
#[derive(Debug, Default)]
pub struct Prune {
    limits: Limits,
    /// Trim decisions by result identity, so a long history isn't re-trimmed every call.
    memo: Mutex<HashMap<u64, Option<Trimmed>>>,
}

impl Prune {
    /// A stage with the default limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A stage with explicit limits, for comparing settings on a bench.
    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            limits,
            memo: Mutex::default(),
        }
    }

    fn trim(
        &self,
        id: &str,
        tool: &str,
        input: Option<&Value>,
        text: &str,
        text_hash: u64,
    ) -> Option<Trimmed> {
        let key = {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            text_hash.hash(&mut hasher);
            hasher.finish()
        };
        if let Some(known) = self.memo.lock().expect("lock poisoned").get(&key) {
            return known.clone();
        }
        let trimmed = rules::trim(&self.limits, tool, input, text);
        let mut memo = self.memo.lock().expect("lock poisoned");
        if memo.len() >= MAX_MEMO_ENTRIES {
            memo.clear();
        }
        memo.insert(key, trimmed.clone());
        trimmed
    }
}

fn hash_of(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl Stage for Prune {
    fn name(&self) -> &'static str {
        "prune"
    }

    fn time_cap(&self) -> Duration {
        TIME_CAP
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let mut edits = Vec::new();
        let mut calls: HashMap<&str, (&str, Option<&Value>)> = HashMap::new();
        let mut first_sent: HashMap<u64, &str> = HashMap::new();

        for (index, message) in request.messages().iter().enumerate() {
            let Some(blocks) = message.get("content").and_then(Value::as_array) else {
                continue;
            };
            for (position, block) in blocks.iter().enumerate() {
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        if let (Some(id), Some(name)) = (
                            block.get("id").and_then(Value::as_str),
                            block.get("name").and_then(Value::as_str),
                        ) {
                            calls.insert(id, (name, block.get("input")));
                        }
                    }
                    Some("tool_result") => {
                        let Some(id) = block.get("tool_use_id").and_then(Value::as_str) else {
                            continue;
                        };
                        let Some(text) = block.get("content").and_then(text_of_tool_result) else {
                            continue;
                        };
                        let text_hash = hash_of(&text);
                        // Protected results went out whole; anything newer may be trimmed.
                        let trimmed = if index < context.protected_before {
                            None
                        } else {
                            let (tool, input) = calls.get(id).copied().unwrap_or(("", None));
                            self.trim(id, tool, input, &text, text_hash)
                        };

                        // Compare what was actually sent, so a stub only ever points at
                        // something that really holds the same content.
                        let sent_hash = trimmed.as_ref().map_or(text_hash, |t| hash_of(&t.text));
                        let sent_len = trimmed.as_ref().map_or(text.len(), |t| t.text.len());
                        let earlier = first_sent.get(&sent_hash).copied();
                        first_sent.entry(sent_hash).or_insert(id);
                        if index < context.protected_before {
                            continue;
                        }

                        let replacement = match earlier {
                            Some(earlier) if sent_len >= MIN_DUPLICATE_CHARS => Some((
                                "duplicate",
                                format!(
                                    "[ccc: identical to the result of tool call {earlier} above]"
                                ),
                            )),
                            _ => trimmed.map(|trimmed| (trimmed.rule, trimmed.text)),
                        };
                        if let Some((rule, replacement)) = replacement {
                            edits.push(Edit {
                                target: Target::Message(index),
                                pointer: format!("/content/{position}/content"),
                                value: json!(replacement),
                                rule,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(edits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn context() -> Context {
        Context {
            protected_before: 0,
            enabled: BTreeSet::from(["prune".to_owned()]),
            spent_micros: 0,
        }
    }

    fn request(messages: &[Value]) -> Request {
        Request::parse(
            json!({ "model": "m", "messages": messages })
                .to_string()
                .as_bytes(),
        )
        .unwrap()
    }

    fn tool_use(id: &str, name: &str, input: Value) -> Value {
        json!({ "role": "assistant", "content": [
            { "type": "tool_use", "id": id, "name": name, "input": input }
        ] })
    }

    fn tool_result(id: &str, text: &str) -> Value {
        json!({ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": id, "content": text }
        ] })
    }

    fn bash(id: &str, command: &str) -> Value {
        tool_use(id, "Bash", json!({ "command": command }))
    }

    fn noisy_test_log() -> String {
        let mut log = String::new();
        for i in 0..400 {
            log.push_str(&format!("test module::case_{i} ... ok\n"));
        }
        log.push_str("test module::broken ... FAILED\n");
        log.push_str("test result: FAILED. 400 passed; 1 failed\n");
        log
    }

    #[test]
    fn repeated_output_points_to_the_earliest_copy() {
        let output = "same output line\n".repeat(30);
        let messages = [
            bash("a", "git status"),
            tool_result("a", &output),
            bash("b", "git status"),
            tool_result("b", &output),
        ];
        let edits = Prune::new().plan(&request(&messages), &context()).unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].target, Target::Message(3));
        assert_eq!(edits[0].rule, "duplicate");
        assert!(
            edits[0].value.as_str().unwrap().contains(" a "),
            "{edits:?}"
        );
    }

    /// Found by a live run: `make test` matched an earlier script's output exactly, so it
    /// became a stub pointing at a copy that had already lost the failure line.
    #[test]
    fn a_stub_never_points_at_a_copy_that_lost_content() {
        let mut lines: Vec<String> = (0..600).map(|i| format!("test case_{i} ... ok")).collect();
        lines[300] = "plain line that no rule treats as a failure".to_owned();
        let output = lines.join("\n");
        let messages = [
            bash("a", "python3 run_tests.py"),
            tool_result("a", &output),
            bash("b", "rg case"),
            tool_result("b", &output),
        ];
        let edits = Prune::new().plan(&request(&messages), &context()).unwrap();
        assert_eq!(edits.len(), 2, "{edits:?}");
        assert_eq!(edits[0].rule, "generic");
        assert_eq!(edits[1].rule, "listing", "kept its own view, not a stub");
    }

    #[test]
    fn identical_output_trimmed_the_same_way_still_dedupes() {
        let log = noisy_test_log();
        let messages = [
            bash("a", "cargo test"),
            tool_result("a", &log),
            bash("b", "cargo test"),
            tool_result("b", &log),
        ];
        let edits = Prune::new().plan(&request(&messages), &context()).unwrap();
        assert_eq!(edits[0].rule, "log");
        assert_eq!(edits[1].rule, "duplicate");
    }

    #[test]
    fn short_repeats_are_left_alone() {
        let messages = [
            bash("a", "pwd"),
            tool_result("a", "/tmp"),
            bash("b", "pwd"),
            tool_result("b", "/tmp"),
        ];
        assert!(
            Prune::new()
                .plan(&request(&messages), &context())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn trims_logs_but_never_source_reads() {
        let log = noisy_test_log();
        let test_run = [bash("a", "cargo test"), tool_result("a", &log)];
        let edits = Prune::new().plan(&request(&test_run), &context()).unwrap();
        assert_eq!(edits.len(), 1);
        let kept = edits[0].value.as_str().unwrap();
        assert!(kept.contains("module::broken ... FAILED"));
        assert!(kept.contains("test result: FAILED"));
        assert!(kept.len() < log.len() / 4, "{}", kept.len());

        let source = "fn main() {}\n".repeat(4000);
        for call in [
            bash("a", "cat src/main.rs"),
            bash("a", "sed -n '1,4000p' src/main.rs"),
            bash("a", "git diff"),
            tool_use("a", "Read", json!({ "file_path": "src/main.rs" })),
        ] {
            let messages = [call, tool_result("a", &source)];
            let edits = Prune::new().plan(&request(&messages), &context()).unwrap();
            assert!(edits.is_empty(), "{edits:?}");
        }
    }

    #[test]
    fn protected_history_is_not_edited_but_still_counts_as_seen() {
        let output = "same output line\n".repeat(30);
        let log = noisy_test_log();
        let messages = [
            bash("a", "cargo test"),
            tool_result("a", &log),
            bash("b", "git status"),
            tool_result("b", &output),
            bash("c", "git status"),
            tool_result("c", &output),
        ];
        let mut protected = context();
        protected.protected_before = 4;
        let edits = Prune::new().plan(&request(&messages), &protected).unwrap();
        assert_eq!(edits.len(), 1, "{edits:?}");
        assert_eq!(edits[0].target, Target::Message(5));
        assert_eq!(edits[0].rule, "duplicate");
    }

    #[test]
    fn only_rewrites_content_so_cache_control_and_is_error_survive() {
        let log = noisy_test_log();
        let result = json!({ "role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "a", "content": log,
            "is_error": true, "cache_control": { "type": "ephemeral" },
        }] });
        let messages = [bash("a", "cargo test"), result];
        let edits = Prune::new().plan(&request(&messages), &context()).unwrap();
        assert_eq!(edits[0].pointer, "/content/0/content");
    }

    #[test]
    fn non_text_results_are_skipped() {
        let result = json!({ "role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "a",
            "content": [{ "type": "image", "source": {} }],
        }] });
        let messages = [bash("a", "screenshot"), result];
        assert!(
            Prune::new()
                .plan(&request(&messages), &context())
                .unwrap()
                .is_empty()
        );
    }

    /// The core guarantee: appending to a conversation never changes how earlier messages
    /// compile. Otherwise the cached prefix and the thinking signatures would both break.
    #[test]
    fn compiling_a_longer_conversation_keeps_the_shared_prefix_identical() {
        use ccc_pipeline::Pipeline;
        use std::sync::Arc;

        let log = noisy_test_log();
        let repeated = "repeated line of output\n".repeat(40);
        let listing: String = (0..500)
            .map(|i| format!("src/file_{i}.rs:1: match\n"))
            .collect();
        let full = [
            json!({ "role": "user", "content": "go" }),
            bash("a", "cargo test"),
            tool_result("a", &log),
            bash("b", "ls"),
            tool_result("b", &repeated),
            bash("c", "ls"),
            tool_result("c", &repeated),
            bash("d", "rg match"),
            tool_result("d", &listing),
            bash("e", "cargo test"),
            tool_result("e", &log),
        ];
        let pipeline = Pipeline::new(vec![Arc::new(Prune::new())]);
        let mut previous: Option<Request> = None;
        for length in 1..=full.len() {
            let compiled = pipeline
                .compile_offline(&request(&full[..length]), &context())
                .unwrap()
                .request;
            if let Some(previous) = &previous {
                assert_eq!(
                    previous.messages(),
                    &compiled.messages()[..length - 1],
                    "prefix changed at length {length}"
                );
            }
            previous = Some(compiled);
        }
        let compiled = previous.unwrap();
        assert_ne!(
            compiled.messages()[2]["content"][0]["content"],
            log.as_str()
        );
    }

    #[test]
    fn the_memo_does_not_change_results() {
        let log = noisy_test_log();
        let messages = [bash("a", "cargo test"), tool_result("a", &log)];
        let prune = Prune::new();
        let first = prune.plan(&request(&messages), &context()).unwrap();
        let second = prune.plan(&request(&messages), &context()).unwrap();
        assert_eq!(first, second);
    }
}

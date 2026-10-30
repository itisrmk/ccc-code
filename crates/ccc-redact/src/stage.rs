use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ccc_pipeline::{Context, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::Request;
use serde_json::{Value, json};

use crate::detect::Detector;
use crate::vault::Vault;

/// Far above what detection needs, because exceeding it stops the request. A restart in a
/// long session re-scans its whole history once, and scanning is linear in its size.
const TIME_CAP: Duration = Duration::from_secs(2);
/// The memo is only an optimization, so it is simply emptied when it grows this large.
const MAX_MEMO_ENTRIES: usize = 200_000;

/// Secrets found in one piece of text, longest values first so that a secret containing
/// another is replaced whole.
#[derive(Debug, Default)]
struct KnownSecrets {
    by_length: BTreeMap<(std::cmp::Reverse<usize>, String), String>,
}

impl KnownSecrets {
    fn insert(&mut self, secret: &str, placeholder: String) {
        self.by_length.insert(
            (std::cmp::Reverse(secret.len()), secret.to_owned()),
            placeholder,
        );
    }

    fn replace_in(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for ((_, secret), placeholder) in &self.by_length {
            if text.contains(secret.as_str()) {
                text = text.replace(secret.as_str(), placeholder);
            }
        }
        text
    }
}

/// The Redact stage. See the crate docs for how it treats each kind of content.
#[derive(Debug)]
pub struct Redact {
    detector: Detector,
    vault: Arc<Vault>,
    /// Secret values found per text, by hash, so a long history isn't re-scanned every call.
    memo: Mutex<HashMap<u64, FoundSecrets>>,
}

/// Secret values found in one text, each with the rule that found it.
type FoundSecrets = Arc<Vec<(String, &'static str)>>;

impl Redact {
    /// A stage using `detector`, recording placeholders in `vault`.
    #[must_use]
    pub fn new(detector: Detector, vault: Arc<Vault>) -> Self {
        Self {
            detector,
            vault,
            memo: Mutex::default(),
        }
    }

    fn secrets_in(&self, text: &str) -> FoundSecrets {
        let key = {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };
        if let Some(known) = self.memo.lock().expect("lock poisoned").get(&key) {
            return Arc::clone(known);
        }
        let found: Vec<(String, &'static str)> = self
            .detector
            .find(text)
            .into_iter()
            .map(|(range, rule)| (text[range].to_owned(), rule))
            .collect();
        let found = Arc::new(found);
        let mut memo = self.memo.lock().expect("lock poisoned");
        if memo.len() >= MAX_MEMO_ENTRIES {
            memo.clear();
        }
        memo.insert(key, Arc::clone(&found));
        found
    }

    /// Redacts external text: anything the detectors find, and anything already known.
    fn redact_external(
        &self,
        text: &str,
        known: &mut KnownSecrets,
    ) -> Option<(String, &'static str)> {
        let mut rule = None;
        for (secret, found_by) in self.secrets_in(text).iter() {
            known.insert(secret, self.vault.placeholder_for(secret));
            rule = Some(*found_by);
        }
        let replaced = known.replace_in(text);
        (replaced != text).then(|| (replaced, rule.unwrap_or(RULE_KNOWN)))
    }
}

/// The rule name for text changed only because it contains an already known secret.
const RULE_KNOWN: &str = "known secret";

impl Stage for Redact {
    fn name(&self) -> &'static str {
        "redact"
    }

    fn time_cap(&self) -> Duration {
        TIME_CAP
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::Block
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let mut edits = Vec::new();
        let mut known = KnownSecrets::default();

        if context.protected_before == 0 {
            self.plan_system(request, &mut known, &mut edits);
        }
        for (index, message) in request.messages().iter().enumerate() {
            if index < context.protected_before {
                continue;
            }
            let target = Target::Message(index);
            if message.get("role").and_then(Value::as_str) == Some("assistant") {
                plan_assistant(message, target, &known, &mut edits);
            } else {
                self.plan_external(message, target, &mut known, &mut edits);
            }
        }
        Ok(edits)
    }
}

impl Redact {
    fn plan_system(&self, request: &Request, known: &mut KnownSecrets, edits: &mut Vec<Edit>) {
        match request.field("system") {
            Some(Value::String(text)) => {
                self.edit_external(text, Target::System, String::new(), known, edits);
            }
            Some(Value::Array(blocks)) => {
                for (position, block) in blocks.iter().enumerate() {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        let pointer = format!("/{position}/text");
                        self.edit_external(text, Target::System, pointer, known, edits);
                    }
                }
            }
            _ => {}
        }
    }

    fn plan_external(
        &self,
        message: &Value,
        target: Target,
        known: &mut KnownSecrets,
        edits: &mut Vec<Edit>,
    ) {
        match message.get("content") {
            Some(Value::String(text)) => {
                self.edit_external(text, target, "/content".to_owned(), known, edits);
            }
            Some(Value::Array(blocks)) => {
                for (position, block) in blocks.iter().enumerate() {
                    self.plan_external_block(block, position, target, known, edits);
                }
            }
            _ => {}
        }
    }

    fn plan_external_block(
        &self,
        block: &Value,
        position: usize,
        target: Target,
        known: &mut KnownSecrets,
        edits: &mut Vec<Edit>,
    ) {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    let pointer = format!("/content/{position}/text");
                    self.edit_external(text, target, pointer, known, edits);
                }
            }
            Some("tool_result") => match block.get("content") {
                Some(Value::String(text)) => {
                    let pointer = format!("/content/{position}/content");
                    self.edit_external(text, target, pointer, known, edits);
                }
                Some(Value::Array(parts)) => {
                    for (part_index, part) in parts.iter().enumerate() {
                        if part.get("type").and_then(Value::as_str) == Some("text")
                            && let Some(text) = part.get("text").and_then(Value::as_str)
                        {
                            let pointer = format!("/content/{position}/content/{part_index}/text");
                            self.edit_external(text, target, pointer, known, edits);
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn edit_external(
        &self,
        text: &str,
        target: Target,
        pointer: String,
        known: &mut KnownSecrets,
        edits: &mut Vec<Edit>,
    ) {
        if let Some((replaced, rule)) = self.redact_external(text, known) {
            edits.push(Edit {
                target,
                pointer,
                value: json!(replaced),
                rule,
            });
        }
    }
}

/// Restores what the model wrote: placeholders it saw and used, which rehydration turned
/// into real values on the way back. Never runs detectors, since a new match in the
/// model's own words would change bytes it actually produced.
fn plan_assistant(message: &Value, target: Target, known: &KnownSecrets, edits: &mut Vec<Edit>) {
    let Some(blocks) = message.get("content").and_then(Value::as_array) else {
        if let Some(text) = message.get("content").and_then(Value::as_str) {
            push_if_changed(text, known, target, "/content".to_owned(), edits);
        }
        return;
    };
    for (position, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    let pointer = format!("/content/{position}/text");
                    push_if_changed(text, known, target, pointer, edits);
                }
            }
            Some("tool_use") => {
                if let Some(input) = block.get("input") {
                    let replaced = replace_in_value(input, known);
                    if &replaced != input {
                        edits.push(Edit {
                            target,
                            pointer: format!("/content/{position}/input"),
                            value: replaced,
                            rule: RULE_KNOWN,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

fn push_if_changed(
    text: &str,
    known: &KnownSecrets,
    target: Target,
    pointer: String,
    edits: &mut Vec<Edit>,
) {
    let replaced = known.replace_in(text);
    if replaced != text {
        edits.push(Edit {
            target,
            pointer,
            value: json!(replaced),
            rule: RULE_KNOWN,
        });
    }
}

fn replace_in_value(value: &Value, known: &KnownSecrets) -> Value {
    match value {
        Value::String(text) => Value::String(known.replace_in(text)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| replace_in_value(item, known))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), replace_in_value(item, known)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_pipeline::Pipeline;
    use std::collections::BTreeSet;

    fn secret() -> String {
        format!("{}{}", "sk-ant-", "api03-abcdefghijklmnopqrstuvwxyz012345")
    }

    fn stage() -> (Redact, Arc<Vault>) {
        let vault = Arc::new(Vault::ephemeral().unwrap());
        (Redact::new(Detector::builtin(), Arc::clone(&vault)), vault)
    }

    fn context() -> Context {
        Context {
            protected_before: 0,
            enabled: BTreeSet::from(["redact".to_owned()]),
        }
    }

    fn compile(stage: Redact, body: &Value, context: &Context) -> Value {
        let request = Request::parse(body.to_string().as_bytes()).unwrap();
        let compiled = Pipeline::new(vec![Arc::new(stage)])
            .compile_offline(&request, context)
            .unwrap();
        serde_json::from_slice(&compiled.request.to_bytes()).unwrap()
    }

    #[test]
    fn nothing_external_leaves_with_a_secret_in_it() {
        let key = secret();
        let body = json!({
            "system": [{ "type": "text", "text": format!("env {key}") }],
            "messages": [
                { "role": "user", "content": format!("use {key}") },
                { "role": "user", "content": [
                    { "type": "text", "text": format!("again {key}") },
                    { "type": "tool_result", "tool_use_id": "a", "content": format!("out {key}") },
                    { "type": "tool_result", "tool_use_id": "b", "content": [
                        { "type": "text", "text": format!("part {key}") },
                    ] },
                ] },
                { "role": "system", "content": [{ "type": "text", "text": format!("note {key}") }] },
            ],
        });
        let (stage, _vault) = stage();
        let sent = compile(stage, &body, &context()).to_string();
        assert!(!sent.contains(&key), "{sent}");
        assert_eq!(sent.matches("<SECRET_").count(), 6, "{sent}");
    }

    #[test]
    fn the_same_secret_gets_the_same_placeholder_everywhere() {
        let key = secret();
        let body = json!({ "messages": [
            { "role": "user", "content": format!("a {key}") },
            { "role": "user", "content": format!("b {key}") },
        ] });
        let (stage, _vault) = stage();
        let sent = compile(stage, &body, &context());
        let first = sent["messages"][0]["content"].as_str().unwrap();
        let second = sent["messages"][1]["content"].as_str().unwrap();
        assert_eq!(first.strip_prefix("a "), second.strip_prefix("b "));
    }

    #[test]
    fn signed_thinking_and_image_data_are_never_changed() {
        let key = secret();
        let body = json!({ "messages": [
            { "role": "user", "content": [
                { "type": "text", "text": key },
                { "type": "image", "source": { "type": "base64", "data": key } },
            ] },
            { "role": "assistant", "content": [
                { "type": "thinking", "thinking": key, "signature": key },
                { "type": "redacted_thinking", "data": key },
            ] },
        ] });
        let (stage, _vault) = stage();
        let sent = compile(stage, &body, &context());
        assert_eq!(
            sent["messages"][0]["content"][1],
            body["messages"][0]["content"][1]
        );
        assert_eq!(sent["messages"][1], body["messages"][1]);
    }

    /// The model only ever saw the placeholder, so what comes back from Claude Code after
    /// rehydration must compile to exactly what the model produced.
    #[test]
    fn rehydrated_assistant_turns_compile_back_to_the_model_s_words() {
        let key = secret();
        let (stage, vault) = stage();
        let placeholder = vault.placeholder_for(&key);
        let body = json!({ "messages": [
            { "role": "user", "content": format!("key {key}") },
            { "role": "assistant", "content": [
                { "type": "text", "text": format!("I'll use {key}") },
                { "type": "tool_use", "id": "t", "name": "Bash", "input": {
                    "command": format!("curl -H '{key}' x"), "args": [key.clone()], "n": 3,
                } },
            ] },
        ] });
        let sent = compile(stage, &body, &context());
        let assistant = &sent["messages"][1]["content"];
        assert_eq!(assistant[0]["text"], format!("I'll use {placeholder}"));
        assert_eq!(
            assistant[1]["input"]["command"],
            format!("curl -H '{placeholder}' x")
        );
        assert_eq!(assistant[1]["input"]["args"][0], placeholder);
        assert_eq!(assistant[1]["input"]["n"], 3);
    }

    #[test]
    fn the_model_s_own_words_are_not_scanned_for_new_secrets() {
        let invented = secret();
        let body = json!({ "messages": [
            { "role": "user", "content": "write an example" },
            { "role": "assistant", "content": [{ "type": "text", "text": format!("eg {invented}") }] },
            { "role": "user", "content": "thanks" },
        ] });
        let (stage, _vault) = stage();
        let sent = compile(stage, &body, &context());
        assert_eq!(sent["messages"][1], body["messages"][1]);
    }

    #[test]
    fn an_assistant_mention_before_the_secret_appears_is_left_alone() {
        // If this used the secret known later, compiling a longer conversation would
        // rewrite an earlier turn.
        let key = secret();
        let body = json!({ "messages": [
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": [{ "type": "text", "text": format!("eg {key}") }] },
            { "role": "user", "content": format!("here it is {key}") },
        ] });
        let (stage, _vault) = stage();
        let sent = compile(stage, &body, &context());
        assert_eq!(sent["messages"][1], body["messages"][1]);
        assert!(!sent["messages"][2].to_string().contains(&key));
    }

    #[test]
    fn protected_history_and_its_system_prompt_stay_untouched() {
        let key = secret();
        let body = json!({
            "system": format!("env {key}"),
            "messages": [
                { "role": "user", "content": format!("old {key}") },
                { "role": "user", "content": format!("new {key}") },
            ],
        });
        let (stage, _vault) = stage();
        let protected = Context {
            protected_before: 1,
            ..context()
        };
        let sent = compile(stage, &body, &protected);
        assert_eq!(sent["system"], body["system"]);
        assert_eq!(sent["messages"][0], body["messages"][0]);
        assert!(!sent["messages"][1].to_string().contains(&key));
    }

    #[test]
    fn compiling_a_longer_conversation_keeps_the_shared_prefix_identical() {
        let key = secret();
        let full = [
            json!({ "role": "user", "content": format!("key {key} and jane@acme.io") }),
            json!({ "role": "assistant", "content": [
                { "type": "tool_use", "id": "t", "name": "Bash", "input": { "command": format!("use {key}") } },
            ] }),
            json!({ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "t", "content": format!("got {key}") },
            ] }),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "done" }] }),
            json!({ "role": "user", "content": format!("other {}", "ghp_abcdefghijklmnopqrstuvwxyz0123456789") }),
        ];
        let (stage, _vault) = stage();
        let pipeline = Pipeline::new(vec![Arc::new(stage)]);
        let mut previous: Option<Request> = None;
        for length in 1..=full.len() {
            let body = json!({ "system": "s", "messages": &full[..length] });
            let request = Request::parse(body.to_string().as_bytes()).unwrap();
            let compiled = pipeline
                .compile_offline(&request, &context())
                .unwrap()
                .request;
            if let Some(previous) = &previous {
                assert_eq!(
                    previous.messages(),
                    &compiled.messages()[..length - 1],
                    "length {length}"
                );
            }
            previous = Some(compiled);
        }
    }

    #[test]
    fn it_refuses_to_send_when_it_fails() {
        let (stage, _vault) = stage();
        assert_eq!(stage.on_failure(), OnFailure::Block);
    }
}

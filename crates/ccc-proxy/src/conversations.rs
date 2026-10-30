//! Remembers, per conversation, what the compiler may touch.
//!
//! The server binds thinking blocks to the exact history before them, so how a conversation
//! compiles has to stay fixed for its whole life, including across restarts of this proxy.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use ccc_pipeline::Context;
use ccc_wire::Request;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Identifies, and persists compile settings for, conversations.
#[derive(Debug)]
pub(crate) struct Conversations {
    /// Directory holding one small file per conversation. `None` keeps state in memory.
    dir: Option<PathBuf>,
    /// Stages that new conversations start with.
    enabled_for_new: BTreeSet<String>,
    known: Mutex<HashMap<String, Context>>,
}

impl Conversations {
    pub(crate) fn new(dir: Option<PathBuf>, enabled_for_new: BTreeSet<String>) -> Self {
        Self {
            dir,
            enabled_for_new,
            known: Mutex::default(),
        }
    }

    /// The context for `request`, creating it if this conversation is new here.
    ///
    /// A conversation first seen with assistant turns already in it was running before
    /// this proxy saw it, so everything it has so far is protected from edits.
    pub(crate) fn context_for(&self, session: Option<&str>, request: &Request) -> Context {
        let messages = request.messages();
        let key = conversation_key(session, messages.first().map(AsRef::as_ref));

        let remembered = self.known.lock().expect("lock poisoned").get(&key).cloned();
        let mut context = remembered.or_else(|| self.load(&key)).unwrap_or_else(|| {
            let has_history = messages
                .iter()
                .any(|message| message.get("role").and_then(Value::as_str) == Some("assistant"));
            let context = Context {
                protected_before: if has_history { messages.len() } else { 0 },
                enabled: self.enabled_for_new.clone(),
            };
            self.store(&key, &context);
            context
        });

        // Compaction shortens history. What remains is all the protected part can cover.
        if context.protected_before > messages.len() {
            context.protected_before = messages.len();
            self.store(&key, &context);
        }
        self.known
            .lock()
            .expect("lock poisoned")
            .insert(key, context.clone());
        context
    }

    fn load(&self, key: &str) -> Option<Context> {
        let path = self.dir.as_ref()?.join(format!("{key}.json"));
        let stored: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
        Some(Context {
            protected_before: stored
                .get("protected_before")?
                .as_u64()
                .and_then(|count| usize::try_from(count).ok())?,
            enabled: stored
                .get("enabled")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        })
    }

    /// Persists `context`. A failure is logged and not fatal: the worst outcome is that a
    /// restart mid-conversation misjudges this one conversation.
    fn store(&self, key: &str, context: &Context) {
        let Some(dir) = &self.dir else {
            return;
        };
        let record = json!({
            "protected_before": context.protected_before,
            "enabled": context.enabled,
        });
        let written = fs::create_dir_all(dir)
            .and_then(|()| fs::write(dir.join(format!("{key}.json")), record.to_string()));
        if let Err(err) = written {
            tracing::warn!(error = %err, "could not persist conversation state");
        }
    }
}

/// A stable name for a conversation: its session plus its first message. Subagents share
/// a session id with their parent but start from a different first message.
fn conversation_key(session: Option<&str>, first_message: Option<&Value>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(session.unwrap_or_default().as_bytes());
    hasher.update([0]);
    if let Some(message) = first_message {
        let mut message = message.clone();
        strip_cache_control(&mut message);
        hasher.update(message.to_string().as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Cache markers move between requests without the conversation changing.
fn strip_cache_control(value: &mut Value) {
    match value {
        Value::Array(items) => items.iter_mut().for_each(strip_cache_control),
        Value::Object(map) => {
            map.remove("cache_control");
            map.values_mut().for_each(strip_cache_control);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Value) -> Request {
        Request::parse(json!({ "messages": messages }).to_string().as_bytes()).unwrap()
    }

    fn enabled() -> BTreeSet<String> {
        BTreeSet::from(["prune".to_owned()])
    }

    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": text })
    }

    fn assistant() -> Value {
        json!({ "role": "assistant", "content": "ok" })
    }

    #[test]
    fn a_fresh_conversation_is_unprotected() {
        let conversations = Conversations::new(None, enabled());
        let context = conversations.context_for(Some("s"), &request(json!([user("hi")])));
        assert_eq!(context.protected_before, 0);
    }

    #[test]
    fn a_conversation_first_seen_with_history_protects_all_of_it() {
        let conversations = Conversations::new(None, enabled());
        let existing = json!([user("hi"), assistant(), user("more")]);
        let context = conversations.context_for(Some("s"), &request(existing));
        assert_eq!(context.protected_before, 3);

        let longer = json!([
            user("hi"),
            assistant(),
            user("more"),
            assistant(),
            user("x")
        ]);
        let context = conversations.context_for(Some("s"), &request(longer));
        assert_eq!(
            context.protected_before, 3,
            "the mark doesn't move as it grows"
        );
    }

    #[test]
    fn compaction_shrinks_the_protected_range() {
        let conversations = Conversations::new(None, enabled());
        let long = json!([user("hi"), assistant(), user("a"), assistant(), user("b")]);
        conversations.context_for(Some("s"), &request(long));
        let context =
            conversations.context_for(Some("s"), &request(json!([user("hi"), assistant()])));
        assert_eq!(context.protected_before, 2);
    }

    #[test]
    fn subagents_and_other_sessions_are_separate_conversations() {
        let conversations = Conversations::new(None, enabled());
        let with_history = json!([user("parent"), assistant()]);
        conversations.context_for(Some("s"), &request(with_history));

        let subagent = conversations.context_for(Some("s"), &request(json!([user("child")])));
        assert_eq!(subagent.protected_before, 0);

        let other = conversations.context_for(Some("t"), &request(json!([user("parent")])));
        assert_eq!(other.protected_before, 0);
    }

    #[test]
    fn moved_cache_markers_do_not_start_a_new_conversation() {
        let conversations = Conversations::new(None, enabled());
        let plain =
            json!([{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }, assistant()]);
        let marked = json!([{ "role": "user", "content": [
            { "type": "text", "text": "hi", "cache_control": { "type": "ephemeral" } }
        ] }, assistant(), user("x")]);
        conversations.context_for(Some("s"), &request(plain));
        let context = conversations.context_for(Some("s"), &request(marked));
        assert_eq!(context.protected_before, 2);
    }

    #[test]
    fn settings_survive_a_restart_and_stay_pinned() {
        let dir = std::env::temp_dir().join(format!(
            "ccc-conversations-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let history = json!([user("hi"), assistant()]);

        let before = Conversations::new(Some(dir.clone()), enabled());
        let original = before.context_for(Some("s"), &request(history.clone()));

        // A restart with the stage now disabled for new conversations.
        let after = Conversations::new(Some(dir.clone()), BTreeSet::new());
        let resumed = after.context_for(Some("s"), &request(history));
        assert_eq!(resumed, original);
        assert!(resumed.enabled.contains("prune"));

        let new_one = after.context_for(Some("other"), &request(json!([user("hello")])));
        assert!(new_one.enabled.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }
}

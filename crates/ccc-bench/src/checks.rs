//! Safety checks on a compile function, computed from transcripts alone.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use ccc_wire::{Request, text_of_tool_result};
use serde_json::Value;

use crate::transcript::Session;

/// Identifiers shorter than this are too common to tell us anything.
const MIN_IDENTIFIER_CHARS: usize = 8;

/// Counts calls whose compiled form changed a message the previous call also sent.
///
/// Any count above zero means compiling isn't append-only: on real traffic that restarts
/// the prompt cache and invalidates every later thinking block.
#[must_use]
pub fn append_only_violations(
    sessions: &[Session],
    compile: &dyn Fn(&Request) -> Request,
) -> usize {
    let mut violations = 0;
    for session in sessions {
        let mut previous: Option<(&Request, Request)> = None;
        for call in &session.calls {
            let compiled = compile(&call.request);
            if let Some((original, earlier)) = &previous {
                let shared = call
                    .request
                    .messages()
                    .iter()
                    .zip(original.messages())
                    .take_while(|(new, old)| Arc::ptr_eq(new, old) || new == old)
                    .count();
                let same = compiled.messages()[..shared]
                    .iter()
                    .zip(&earlier.messages()[..shared])
                    .all(|(new, old)| Arc::ptr_eq(new, old) || new == old);
                if !same {
                    violations += 1;
                }
            }
            previous = Some((&call.request, compiled));
        }
    }
    violations
}

/// How often pruning removed something a later turn went on to mention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fidelity {
    /// Tool results the compile changed.
    pub results_changed: usize,
    /// Of those, results with a removed identifier that a later assistant turn used
    /// while it was nowhere else in the compiled conversation before that turn.
    pub results_referenced_later: usize,
}

impl fmt::Display for Fidelity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let share = if self.results_changed == 0 {
            0.0
        } else {
            100.0 * self.results_referenced_later as f64 / self.results_changed as f64
        };
        write!(
            f,
            "{} of {} changed results ({share:.1}%) lost an identifier a later turn used",
            self.results_referenced_later, self.results_changed
        )
    }
}

/// Measures [`Fidelity`] on the last call of each session, which holds its full history.
///
/// This is deliberately pessimistic. The recorded model had seen the whole output, so it
/// could quote lines a pruned run wouldn't have seen, and that would have re-run the
/// command instead. A hit means "this removal might have cost a tool call", not a failure.
#[must_use]
pub fn fidelity(sessions: &[Session], compile: &dyn Fn(&Request) -> Request) -> Fidelity {
    let mut total = Fidelity::default();
    for session in sessions {
        let Some(last) = session.calls.last() else {
            continue;
        };
        let compiled = compile(&last.request);
        let session_fidelity = fidelity_of(&last.request, &compiled);
        total.results_changed += session_fidelity.results_changed;
        total.results_referenced_later += session_fidelity.results_referenced_later;
    }
    total
}

fn fidelity_of(original: &Request, compiled: &Request) -> Fidelity {
    let mut known: HashSet<String> = HashSet::new();
    let mut pending: Vec<HashSet<String>> = Vec::new();
    let mut referenced: HashSet<usize> = HashSet::new();

    for (before, after) in original.messages().iter().zip(compiled.messages()) {
        let is_assistant = after.get("role").and_then(Value::as_str) == Some("assistant");
        let (before_blocks, after_blocks) = (content_blocks(before), content_blocks(after));
        if is_assistant {
            let authored: HashSet<String> = after_blocks
                .iter()
                .flat_map(|block| identifiers(&block.to_string()))
                .collect();
            for (index, removed) in pending.iter().enumerate() {
                if removed
                    .iter()
                    .any(|id| authored.contains(id) && !known.contains(id))
                {
                    referenced.insert(index);
                }
            }
            known.extend(authored);
            continue;
        }
        for (old_block, new_block) in before_blocks.iter().zip(&after_blocks) {
            let new_text = result_text(new_block).unwrap_or_else(|| new_block.to_string());
            let kept = identifiers(&new_text);
            if old_block != new_block
                && let Some(old_text) = result_text(old_block)
            {
                let removed: HashSet<String> =
                    identifiers(&old_text).difference(&kept).cloned().collect();
                pending.push(removed);
            }
            known.extend(kept);
        }
    }
    Fidelity {
        results_changed: pending.len(),
        results_referenced_later: referenced.len(),
    }
}

fn content_blocks(message: &Value) -> Vec<Value> {
    match message.get("content") {
        Some(Value::Array(blocks)) => blocks.clone(),
        Some(text) => vec![text.clone()],
        None => Vec::new(),
    }
}

fn result_text(block: &Value) -> Option<String> {
    (block.get("type").and_then(Value::as_str) == Some("tool_result"))
        .then(|| block.get("content").and_then(text_of_tool_result))
        .flatten()
        .map(|text| text.into_owned())
}

/// Paths, symbols and similar tokens long enough to be distinctive.
fn identifiers(text: &str) -> HashSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '/' | '-' | ':')))
        .map(|word| word.trim_matches(|c: char| matches!(c, '.' | ':' | '-' | '/')))
        .filter(|word| {
            word.chars().count() >= MIN_IDENTIFIER_CHARS && word.chars().any(char::is_alphabetic)
        })
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{Call, Usage};
    use serde_json::json;
    use std::path::PathBuf;

    fn request(messages: &[Value]) -> Request {
        let body = json!({ "messages": messages }).to_string();
        Request::parse(body.as_bytes()).unwrap()
    }

    fn session_of(requests: Vec<Request>) -> Session {
        Session {
            path: PathBuf::from("s"),
            calls: requests
                .into_iter()
                .map(|request| Call {
                    at_millis: 0,
                    request,
                    usage: Usage::default(),
                })
                .collect(),
        }
    }

    #[test]
    fn detects_rewriting_a_message_the_previous_call_sent() {
        let a = json!({ "role": "user", "content": "aaa" });
        let b = json!({ "role": "user", "content": "bbb" });
        let sessions = [session_of(vec![
            request(std::slice::from_ref(&a)),
            request(&[a, b]),
        ])];

        let identity = |request: &Request| request.clone();
        assert_eq!(append_only_violations(&sessions, &identity), 0);

        let rewrite_once_longer = |request: &Request| {
            let mut compiled = request.clone();
            if compiled.messages().len() > 1 {
                compiled.set_in_message(0, "/content", json!("changed"));
            }
            compiled
        };
        assert_eq!(append_only_violations(&sessions, &rewrite_once_longer), 1);
    }

    #[test]
    fn counts_a_removed_identifier_only_if_used_later_and_absent_elsewhere() {
        let result = |text: &str| {
            json!({ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": "t", "content": text }
        ] })
        };
        let said = |text: &str| {
            json!({ "role": "assistant", "content": [
            { "type": "text", "text": text }
        ] })
        };
        let original = request(&[
            result("removed_symbol kept_symbol_name"),
            said("I will edit removed_symbol now"),
        ]);
        let compiled = {
            let mut compiled = original.clone();
            compiled.set_in_message(0, "/content/0/content", json!("kept_symbol_name"));
            compiled
        };
        assert_eq!(
            fidelity_of(&original, &compiled),
            Fidelity {
                results_changed: 1,
                results_referenced_later: 1
            }
        );

        let unused = request(&[
            result("removed_symbol kept_symbol_name"),
            said("I will edit kept_symbol_name now"),
        ]);
        let compiled = {
            let mut compiled = unused.clone();
            compiled.set_in_message(0, "/content/0/content", json!("kept_symbol_name"));
            compiled
        };
        assert_eq!(fidelity_of(&unused, &compiled).results_referenced_later, 0);

        let seen_again = request(&[
            result("removed_symbol kept_symbol_name"),
            result("a later read shows removed_symbol"),
            said("I will edit removed_symbol now"),
        ]);
        let compiled = {
            let mut compiled = seen_again.clone();
            compiled.set_in_message(0, "/content/0/content", json!("kept_symbol_name"));
            compiled
        };
        assert_eq!(
            fidelity_of(&seen_again, &compiled).results_referenced_later,
            0
        );
    }

    #[test]
    fn identifiers_keep_paths_and_skip_short_words() {
        let found = identifiers("see src/main.rs:12 and parse_config, not this or 12345678.");
        assert!(found.contains("src/main.rs:12"));
        assert!(found.contains("parse_config"));
        assert!(!found.contains("this"));
        assert!(!found.contains("12345678"), "numbers aren't identifiers");
    }
}

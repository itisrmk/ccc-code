//! Simulates Anthropic's prompt cache from request content.
//!
//! The cache keys on the exact prefix up to a breakpoint. Claude Code places breakpoints
//! on the preamble and on its newest messages, so each call caches its whole context,
//! and the next call reads back the longest prefix still alive.

use std::collections::HashMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// How long a cache entry lives after it is written or read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ttl {
    FiveMinutes,
    OneHour,
}

impl Ttl {
    const fn millis(self) -> i64 {
        match self {
            Self::FiveMinutes => 5 * 60 * 1000,
            Self::OneHour => 60 * 60 * 1000,
        }
    }
}

/// A 128-bit truncated hash: collisions are not a practical concern for one session.
pub(crate) type Fingerprint = [u8; 16];

/// Tokens read from and written to the cache by one simulated call.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct CacheOutcome {
    pub(crate) read: f64,
    pub(crate) write: f64,
}

/// How many of the newest prefixes each call caches, matching Claude Code's two
/// message breakpoints.
const MESSAGE_BREAKPOINTS: usize = 2;

/// One session's simulated cache.
#[derive(Debug, Default)]
pub(crate) struct CacheSim {
    live: HashMap<(String, Fingerprint), i64>,
}

impl CacheSim {
    /// Simulates a call. `prefix_tokens[i]` is the cumulative token count through
    /// element `i`, and `chain[i]` identifies the whole prefix through element `i`.
    pub(crate) fn step(
        &mut self,
        model: &str,
        at_millis: i64,
        ttl: Ttl,
        chain: &[Fingerprint],
        prefix_tokens: &[f64],
    ) -> CacheOutcome {
        assert_eq!(
            chain.len(),
            prefix_tokens.len(),
            "chain and sizes describe the same elements"
        );
        let Some(&total) = prefix_tokens.last() else {
            return CacheOutcome::default();
        };
        let expires = at_millis + ttl.millis();

        let hit = chain.iter().rposition(|fingerprint| {
            self.live
                .get(&(model.to_owned(), *fingerprint))
                .is_some_and(|expiry| *expiry > at_millis)
        });
        let read = hit.map_or(0.0, |index| {
            self.live.insert((model.to_owned(), chain[index]), expires);
            prefix_tokens[index]
        });
        let first_breakpoint = chain.len().saturating_sub(MESSAGE_BREAKPOINTS);
        let breakpoints = chain.first().into_iter().chain(&chain[first_breakpoint..]);
        for breakpoint in breakpoints {
            self.live.insert((model.to_owned(), *breakpoint), expires);
        }
        CacheOutcome {
            read,
            write: total - read,
        }
    }
}

/// The fingerprint of a JSON value, ignoring `cache_control` markers: moving a marker
/// doesn't change the content the cache compares.
pub(crate) fn fingerprint(value: &Value) -> Fingerprint {
    let mut hasher = Sha256::new();
    feed(value, &mut hasher);
    truncate(hasher.finalize().as_slice())
}

/// Extends a prefix's fingerprint with the next element's.
pub(crate) fn chain_next(previous: &Fingerprint, element: &Fingerprint) -> Fingerprint {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    hasher.update(element);
    truncate(hasher.finalize().as_slice())
}

fn truncate(digest: &[u8]) -> Fingerprint {
    let mut out = [0; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

/// The fingerprint of a whole message. The prompt is rendered from `content`, so plain
/// string content and a single text block with the same text are the same prefix.
pub(crate) fn message_fingerprint(message: &Value) -> Fingerprint {
    let mut hasher = Sha256::new();
    if let Some(role) = message.get("role") {
        feed(role, &mut hasher);
    }
    match message.get("content") {
        Some(Value::Array(blocks)) => match blocks.as_slice() {
            [only] if only.get("type").and_then(Value::as_str) == Some("text") => {
                if let Some(text) = only.get("text") {
                    feed(text, &mut hasher);
                }
            }
            _ => blocks.iter().for_each(|block| feed(block, &mut hasher)),
        },
        Some(text) => feed(text, &mut hasher),
        None => {}
    }
    truncate(hasher.finalize().as_slice())
}

/// The fingerprint of tool definitions, leaving out deferred tools: they aren't rendered
/// into the prompt until something loads them, so adding one doesn't change the prefix.
pub(crate) fn tools_fingerprint(tools: Option<&Value>) -> Fingerprint {
    let mut hasher = Sha256::new();
    if let Some(Value::Array(tools)) = tools {
        for tool in tools {
            if tool.get("defer_loading").and_then(Value::as_bool) != Some(true) {
                feed(tool, &mut hasher);
            }
        }
    }
    truncate(hasher.finalize().as_slice())
}

fn feed(value: &Value, hasher: &mut Sha256) {
    match value {
        Value::Null => hasher.update(b"n"),
        Value::Bool(flag) => hasher.update(if *flag { b"t" } else { b"f" }),
        Value::Number(number) => {
            hasher.update(b"#");
            hasher.update(number.to_string().as_bytes());
        }
        Value::String(text) => {
            hasher.update(b"s");
            hasher.update(text.len().to_le_bytes());
            hasher.update(text.as_bytes());
        }
        Value::Array(items) => {
            hasher.update(b"[");
            items.iter().for_each(|item| feed(item, hasher));
            hasher.update(b"]");
        }
        Value::Object(map) => {
            hasher.update(b"{");
            for (key, item) in map {
                if key != "cache_control" {
                    hasher.update(key.len().to_le_bytes());
                    hasher.update(key.as_bytes());
                    feed(item, hasher);
                }
            }
            hasher.update(b"}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chain_of(values: &[Value]) -> Vec<Fingerprint> {
        let mut chain = Vec::new();
        let mut previous = [0; 16];
        for value in values {
            previous = chain_next(&previous, &fingerprint(value));
            chain.push(previous);
        }
        chain
    }

    #[test]
    fn appending_reads_the_old_prefix_and_writes_the_rest() {
        let mut sim = CacheSim::default();
        let first = chain_of(&[json!("a"), json!("b")]);
        let second = chain_of(&[json!("a"), json!("b"), json!("c")]);

        let one = sim.step("m", 0, Ttl::OneHour, &first, &[10.0, 20.0]);
        let two = sim.step("m", 1000, Ttl::OneHour, &second, &[10.0, 20.0, 25.0]);

        assert_eq!(
            one,
            CacheOutcome {
                read: 0.0,
                write: 20.0
            }
        );
        assert_eq!(
            two,
            CacheOutcome {
                read: 20.0,
                write: 5.0
            }
        );
    }

    #[test]
    fn an_edit_in_the_prefix_misses_everything_after_it() {
        let mut sim = CacheSim::default();
        let first = chain_of(&[json!("a"), json!("b")]);
        let edited = chain_of(&[json!("X"), json!("b"), json!("c")]);

        sim.step("m", 0, Ttl::OneHour, &first, &[10.0, 20.0]);
        let outcome = sim.step("m", 1000, Ttl::OneHour, &edited, &[10.0, 20.0, 25.0]);

        assert_eq!(
            outcome,
            CacheOutcome {
                read: 0.0,
                write: 25.0
            }
        );
    }

    #[test]
    fn entries_expire_and_reads_refresh_them() {
        let mut sim = CacheSim::default();
        let chain = chain_of(&[json!("a")]);
        let five_minutes = Ttl::FiveMinutes.millis();

        sim.step("m", 0, Ttl::FiveMinutes, &chain, &[10.0]);
        let refreshed = sim.step("m", five_minutes - 1, Ttl::FiveMinutes, &chain, &[10.0]);
        assert_eq!(refreshed.read, 10.0);

        let alive = sim.step("m", 2 * five_minutes - 2, Ttl::FiveMinutes, &chain, &[10.0]);
        assert_eq!(alive.read, 10.0);

        let expired = sim.step("m", 10 * five_minutes, Ttl::FiveMinutes, &chain, &[10.0]);
        assert_eq!(
            expired,
            CacheOutcome {
                read: 0.0,
                write: 10.0
            }
        );
    }

    #[test]
    fn caches_are_per_model() {
        let mut sim = CacheSim::default();
        let chain = chain_of(&[json!("a")]);
        sim.step("m1", 0, Ttl::OneHour, &chain, &[10.0]);
        assert_eq!(sim.step("m2", 1, Ttl::OneHour, &chain, &[10.0]).read, 0.0);
    }

    #[test]
    fn string_content_and_one_text_block_are_the_same_prefix() {
        let plain = json!({ "role": "user", "content": "a" });
        let block = json!({ "role": "user", "content": [{ "type": "text", "text": "a" }] });
        let other = json!({ "role": "user", "content": "b" });
        assert_eq!(message_fingerprint(&plain), message_fingerprint(&block));
        assert_ne!(message_fingerprint(&plain), message_fingerprint(&other));
    }

    #[test]
    fn deferred_tools_do_not_change_the_tools_fingerprint() {
        let base = json!([{ "name": "Read" }]);
        let more = json!([{ "name": "Read" }, { "name": "X", "defer_loading": true }]);
        let real = json!([{ "name": "Read" }, { "name": "Y" }]);
        assert_eq!(
            tools_fingerprint(Some(&base)),
            tools_fingerprint(Some(&more))
        );
        assert_ne!(
            tools_fingerprint(Some(&base)),
            tools_fingerprint(Some(&real))
        );
    }

    #[test]
    fn the_second_newest_prefix_also_survives() {
        let mut sim = CacheSim::default();
        let first = chain_of(&[json!("a"), json!("b"), json!("reminder")]);
        let next = chain_of(&[json!("a"), json!("b"), json!("other"), json!("c")]);
        sim.step("m", 0, Ttl::OneHour, &first, &[10.0, 20.0, 21.0]);
        let outcome = sim.step("m", 1, Ttl::OneHour, &next, &[10.0, 20.0, 22.0, 30.0]);
        assert_eq!(outcome.read, 20.0);
    }

    #[test]
    fn moving_cache_control_does_not_change_a_fingerprint() {
        let plain = json!({ "type": "text", "text": "a" });
        let marked =
            json!({ "type": "text", "text": "a", "cache_control": { "type": "ephemeral" } });
        assert_eq!(fingerprint(&plain), fingerprint(&marked));
    }
}

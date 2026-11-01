//! What a session's next message will cost, worked out from its transcript.
//!
//! Hooks call this on the way to a prompt, so it reads only the end of the file.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use ccc_ledger::{Usage, rates};
use serde_json::Value;

use crate::timestamp::parse_millis;

/// How much of a transcript's end to read. Generous, since one record can hold a large file.
const TAIL_BYTES: u64 = 8 * 1024 * 1024;
const LIFETIME_1H_MILLIS: i64 = 3_600_000;
const LIFETIME_5M_MILLIS: i64 = 300_000;

/// Where a session stood after its last call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionState {
    /// When the last call finished, in milliseconds since the Unix epoch.
    pub last_at_millis: i64,
    /// Context of the last call, in tokens.
    pub context_tokens: u64,
    /// The model it used.
    pub model: String,
    /// How long the cache outlives a call. Assumed to be the longer lifetime when the
    /// transcript's end doesn't show, so a warning is never early.
    pub cache_lifetime_millis: i64,
}

/// Reads where a session stands, or `None` if its transcript shows no calls yet.
///
/// # Errors
///
/// Returns an I/O error if the file can't be read.
pub fn read_state(transcript: &Path) -> io::Result<Option<SessionState>> {
    let mut file = File::open(transcript)?;
    let length = file.metadata()?.len();
    let start = length.saturating_sub(TAIL_BYTES);
    file.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file);
    if start > 0 {
        // Starting mid-file lands in the middle of a line.
        let mut partial = Vec::new();
        reader.read_until(b'\n', &mut partial)?;
    }
    let (mut last, mut wrote_1h, mut wrote_5m) = (None, false, false);
    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if record.get("type").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(usage) = record.pointer("/message/usage").map(Usage::from_json) else {
            continue;
        };
        let at = record
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_millis);
        let model = record.pointer("/message/model").and_then(Value::as_str);
        let (Some(at), Some(model)) = (at, model) else {
            continue;
        };
        // Synthetic records carry no real usage.
        if usage.context() == 0 || rates(model, 0).is_none() {
            continue;
        }
        wrote_1h |= usage.cache_write_1h > 0;
        wrote_5m |= usage.cache_write_5m > 0;
        last = Some((at, usage.context(), model.to_owned()));
    }
    Ok(
        last.map(|(last_at_millis, context_tokens, model)| SessionState {
            last_at_millis,
            context_tokens,
            model,
            cache_lifetime_millis: if !wrote_1h && wrote_5m {
                LIFETIME_5M_MILLIS
            } else {
                LIFETIME_1H_MILLIS
            },
        }),
    )
}

impl SessionState {
    /// How long the cache has been gone, or `None` if it should still be there.
    #[must_use]
    pub fn expired_for_millis(&self, now_millis: i64) -> Option<i64> {
        let idle = now_millis - self.last_at_millis;
        (idle >= self.cache_lifetime_millis).then_some(idle)
    }

    /// What the next call pays to rewrite this context to the cache at list prices.
    #[must_use]
    pub fn rewrite_usd(&self) -> Option<f64> {
        let rates = rates(&self.model, self.context_tokens)?;
        let rate = if self.cache_lifetime_millis == LIFETIME_1H_MILLIS {
            rates.cache_write_1h
        } else {
            rates.cache_write_5m
        };
        Some(self.context_tokens as f64 * rate)
    }

    /// What one call pays to read this context from the cache at list prices.
    #[must_use]
    pub fn read_usd(&self) -> Option<f64> {
        let rates = rates(&self.model, self.context_tokens)?;
        Some(self.context_tokens as f64 * rates.cache_read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write as _;
    use std::path::PathBuf;

    fn write(name: &str, lines: &[String]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("ccc-advise-{}-{name}", std::process::id()));
        let mut file = File::create(&path).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        path
    }

    fn assistant(at: &str, usage: &str) -> String {
        format!(
            r#"{{"type":"assistant","timestamp":"{at}","message":{{"model":"claude-opus-5-5","usage":{usage}}}}}"#
        )
    }

    #[test]
    fn reads_the_last_real_call_and_its_cache_lifetime() {
        let long = assistant(
            "2026-10-30T12:00:00.000Z",
            r#"{"cache_creation_input_tokens":500,"cache_creation":{"ephemeral_1h_input_tokens":500},"output_tokens":1}"#,
        );
        let later = assistant(
            "2026-10-30T13:00:00.000Z",
            r#"{"cache_read_input_tokens":500000,"cache_creation_input_tokens":5,"cache_creation":{"ephemeral_1h_input_tokens":5},"output_tokens":1}"#,
        );
        let synthetic = r#"{"type":"assistant","timestamp":"2026-10-30T14:00:00.000Z","message":{"model":"<synthetic>","usage":{"input_tokens":0}}}"#;
        let path = write(
            "last",
            &[long, later, synthetic.to_owned(), "broken".to_owned()],
        );
        let state = read_state(&path).unwrap().unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(state.context_tokens, 500_005);
        assert_eq!(state.cache_lifetime_millis, LIFETIME_1H_MILLIS);
        assert_eq!(state.last_at_millis, 1_793_365_200_000);
    }

    #[test]
    fn short_lived_writes_mean_a_short_lifetime() {
        let line = assistant(
            "2026-10-30T12:00:00.000Z",
            r#"{"cache_creation_input_tokens":500,"output_tokens":1}"#,
        );
        let path = write("short", &[line]);
        let state = read_state(&path).unwrap().unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(state.cache_lifetime_millis, LIFETIME_5M_MILLIS);
    }

    #[test]
    fn a_transcript_with_no_calls_has_no_state() {
        let path = write("none", &[r#"{"type":"user"}"#.to_owned()]);
        assert_eq!(read_state(&path).unwrap(), None);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn only_reads_the_end_of_a_large_file() {
        let path = write("large", &[]);
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        let filler = format!("{}\n", "x".repeat(1024 * 1024));
        for _ in 0..9 {
            file.write_all(filler.as_bytes()).unwrap();
        }
        let line = assistant(
            "2026-10-30T12:00:00.000Z",
            r#"{"cache_read_input_tokens":7,"output_tokens":1}"#,
        );
        writeln!(file, "{line}").unwrap();
        let state = read_state(&path).unwrap().unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(state.context_tokens, 7);
    }

    #[test]
    fn prices_a_rewrite_against_a_read() {
        let state = SessionState {
            last_at_millis: 0,
            context_tokens: 500_000,
            model: "claude-opus-5-5".to_owned(),
            cache_lifetime_millis: LIFETIME_1H_MILLIS,
        };
        assert!((state.rewrite_usd().unwrap() - 4.0).abs() < 1e-9);
        assert!((state.read_usd().unwrap() - 0.1).abs() < 1e-9);
        assert_eq!(state.expired_for_millis(LIFETIME_1H_MILLIS - 1), None);
        assert_eq!(
            state.expired_for_millis(LIFETIME_1H_MILLIS),
            Some(LIFETIME_1H_MILLIS)
        );
    }
}

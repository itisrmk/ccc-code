//! Warns about what a long session is costing, at the moments a person can still act on it.
//!
//! Warnings go out as `systemMessage`, which Claude Code shows the person. Plain stdout would
//! join the model's context, and this is none of the model's business.

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use ccc_bench::{SessionState, read_state};
use serde_json::{Value, json};

/// Context sizes worth one warning each per session.
const LARGE_CONTEXT_BANDS: [u64; 2] = [300_000, 600_000];

/// How advice is delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdviseOptions {
    /// Stay quiet below this context size.
    pub(crate) min_context: u64,
    /// Hold back the first prompt after a cache expires, instead of only warning.
    pub(crate) block: bool,
}

/// What has already been said to a session, so each warning comes once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Warned {
    /// The last call a cache expiry was reported for.
    expiry_after_call: i64,
    /// The largest context band reported.
    band: u64,
}

/// The event a piece of advice answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moment {
    Resume,
    Prompt,
}

fn tokens(count: u64) -> String {
    format!("{}K", count / 1000)
}

fn expiry_message(state: &SessionState, idle_millis: i64) -> Option<String> {
    let idle = if idle_millis >= 3_600_000 {
        format!("{:.1} hours", idle_millis as f64 / 3_600_000.0)
    } else {
        format!("{} minutes", idle_millis / 60_000)
    };
    Some(format!(
        "ccc: this session's prompt cache expired {idle} ago. The next message rewrites {} \
         tokens: about ${:.2} at list prices, against ${:.2} with a live cache. /compact or a \
         fresh session with a short handoff would make that rewrite smaller.",
        tokens(state.context_tokens),
        state.rewrite_usd()?,
        state.read_usd()?,
    ))
}

fn large_message(state: &SessionState, band: u64) -> Option<String> {
    Some(format!(
        "ccc: context has passed {} tokens. Each call now pays about ${:.2} at list prices to \
         read it. `ccc whatif` shows what restarting at this size would have saved past sessions.",
        tokens(band),
        state.read_usd()?,
    ))
}

/// Decides what to say. Returns the hook's output, if any, and what's been said since.
fn decide(
    state: &SessionState,
    now_millis: i64,
    moment: Moment,
    options: AdviseOptions,
    warned: Warned,
) -> (Option<Value>, Warned) {
    let mut warned = warned;
    if state.context_tokens < options.min_context {
        return (None, warned);
    }
    if let Some(idle) = state.expired_for_millis(now_millis)
        && warned.expiry_after_call != state.last_at_millis
        && let Some(message) = expiry_message(state, idle)
    {
        warned.expiry_after_call = state.last_at_millis;
        let output = if options.block && moment == Moment::Prompt {
            json!({
                "decision": "block",
                "reason": format!("{message} Send the message again to go ahead."),
            })
        } else {
            json!({ "systemMessage": message })
        };
        return (Some(output), warned);
    }
    let band = LARGE_CONTEXT_BANDS
        .iter()
        .copied()
        .filter(|band| state.context_tokens >= *band)
        .max();
    if moment == Moment::Prompt
        && let Some(band) = band.filter(|band| *band > warned.band)
        && let Some(message) = large_message(state, band)
    {
        warned.band = band;
        return (Some(json!({ "systemMessage": message })), warned);
    }
    (None, warned)
}

fn marker_path(state_dir: &Path, session: &str) -> Option<std::path::PathBuf> {
    let safe = !session.is_empty()
        && session
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    safe.then(|| state_dir.join("advise").join(format!("{session}.json")))
}

fn read_marker(path: &Path) -> Warned {
    let Ok(text) = fs::read_to_string(path) else {
        return Warned::default();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return Warned::default();
    };
    Warned {
        expiry_after_call: value
            .get("expiry_after_call")
            .and_then(Value::as_i64)
            .unwrap_or(0),
        band: value.get("band").and_then(Value::as_u64).unwrap_or(0),
    }
}

/// Handles one hook event's advice, returning what Claude Code should be told.
pub(crate) fn advise(
    state_dir: &Path,
    session: &str,
    transcript: &str,
    moment: Moment,
    options: AdviseOptions,
) -> anyhow::Result<Option<Value>> {
    // A new session has no transcript yet, which is nothing to warn about.
    let Ok(Some(state)) = read_state(Path::new(transcript)) else {
        return Ok(None);
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("reading the clock")?
        .as_millis() as i64;
    let marker = marker_path(state_dir, session);
    let warned = marker.as_deref().map(read_marker).unwrap_or_default();
    let (output, warned) = decide(&state, now, moment, options, warned);
    if output.is_some()
        && let Some(path) = marker
    {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("creating the advice directory")?;
        }
        let body = json!({ "expiry_after_call": warned.expiry_after_call, "band": warned.band });
        fs::write(&path, body.to_string()).context("recording advice")?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: i64 = 3_600_000;
    const WARN: AdviseOptions = AdviseOptions {
        min_context: 100_000,
        block: false,
    };

    fn state(context_tokens: u64) -> SessionState {
        SessionState {
            last_at_millis: 1000,
            context_tokens,
            model: "claude-opus-5-5".to_owned(),
            cache_lifetime_millis: HOUR,
        }
    }

    #[test]
    fn warns_once_when_the_cache_has_expired_and_names_the_price() {
        let state = state(500_000);
        let (output, warned) = decide(
            &state,
            1000 + 2 * HOUR,
            Moment::Resume,
            WARN,
            Warned::default(),
        );
        let message = output.unwrap()["systemMessage"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(message.contains("2.0 hours"), "{message}");
        assert!(message.contains("500K"), "{message}");
        assert!(message.contains("$4.00"), "{message}");
        assert!(message.contains("$0.10"), "{message}");

        let (again, _) = decide(&state, 1000 + 3 * HOUR, Moment::Resume, WARN, warned);
        assert_eq!(again, None);
    }

    #[test]
    fn a_new_call_can_be_warned_about_again() {
        let warned = Warned {
            expiry_after_call: 1,
            band: 0,
        };
        let (output, _) = decide(&state(500_000), 2 * HOUR, Moment::Resume, WARN, warned);
        assert!(output.is_some());
    }

    #[test]
    fn says_nothing_with_a_live_cache_or_a_small_context() {
        let (live, _) = decide(
            &state(200_000),
            1000 + HOUR / 2,
            Moment::Prompt,
            WARN,
            Warned::default(),
        );
        assert_eq!(live, None);
        let (small, _) = decide(
            &state(50_000),
            1000 + 5 * HOUR,
            Moment::Prompt,
            WARN,
            Warned::default(),
        );
        assert_eq!(small, None);
    }

    #[test]
    fn blocking_holds_back_only_the_first_prompt_and_never_a_resume() {
        let block = AdviseOptions {
            block: true,
            ..WARN
        };
        let now = 1000 + 2 * HOUR;
        let (output, warned) = decide(
            &state(500_000),
            now,
            Moment::Prompt,
            block,
            Warned::default(),
        );
        let output = output.unwrap();
        assert_eq!(output["decision"], "block");
        assert!(output["reason"].as_str().unwrap().contains("again"));

        let (second, _) = decide(&state(500_000), now, Moment::Prompt, block, warned);
        assert_eq!(
            second.map(|value| value.get("decision").cloned()),
            Some(None)
        );

        let (resume, _) = decide(
            &state(500_000),
            now,
            Moment::Resume,
            block,
            Warned::default(),
        );
        assert!(resume.unwrap().get("systemMessage").is_some());
    }

    #[test]
    fn warns_once_per_size_band_while_the_cache_is_live() {
        let now = 1000 + 1000;
        let (first, warned) = decide(
            &state(350_000),
            now,
            Moment::Prompt,
            WARN,
            Warned::default(),
        );
        assert!(
            first.unwrap()["systemMessage"]
                .as_str()
                .unwrap()
                .contains("300K")
        );
        let (same, warned) = decide(&state(400_000), now, Moment::Prompt, WARN, warned);
        assert_eq!(same, None);
        let (next, _) = decide(&state(700_000), now, Moment::Prompt, WARN, warned);
        assert!(
            next.unwrap()["systemMessage"]
                .as_str()
                .unwrap()
                .contains("600K")
        );
    }

    #[test]
    fn a_session_id_that_is_not_a_plain_name_keeps_no_marker() {
        assert!(marker_path(Path::new("/s"), "../../etc/passwd").is_none());
        assert!(marker_path(Path::new("/s"), "").is_none());
        assert!(marker_path(Path::new("/s"), "abc-123_d").is_some());
    }

    #[test]
    fn a_missing_transcript_is_silent() {
        let output = advise(
            Path::new("/nonexistent"),
            "s",
            "/nonexistent/transcript.jsonl",
            Moment::Prompt,
            WARN,
        )
        .unwrap();
        assert_eq!(output, None);
    }
}

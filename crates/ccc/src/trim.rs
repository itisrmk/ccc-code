//! Shortens long command output before Claude sees it, with no proxy involved.
//!
//! The rules are the proxy's: they drop whole lines, never paraphrase, and say what they dropped.
//! Claude Code replaces the output through `updatedToolOutput`, so the command itself is untouched
//! and permission rules still see what was really asked for.

use std::path::Path;

use ccc_prune::trim_output;
use serde_json::{Value, json};

use crate::config::{JobMode, LocalConfig};
use crate::gain;

const CHARS_PER_TOKEN: usize = 4;

/// What the hook should say, given what Claude Code reports. `None` leaves the output alone.
fn trimmed_response(input: &Value) -> Option<(Value, usize)> {
    if input.get("tool_name").and_then(Value::as_str) != Some("Bash") {
        return None;
    }
    let response = input.get("tool_response")?;
    let stdout = response.get("stdout").and_then(Value::as_str)?;
    let trimmed = trim_output("Bash", input.get("tool_input"), stdout)?;
    let saved = stdout.len().checked_sub(trimmed.len())?;
    let mut updated = response.clone();
    updated["stdout"] = json!(trimmed);
    Some((
        json!({ "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": updated,
        } }),
        saved,
    ))
}

/// Handles the hook. It runs after every command, so with the job off it reads one small file.
pub(crate) fn hook(state_dir: &Path, input: &Value) -> Option<String> {
    if LocalConfig::load(state_dir).ok()?.jobs.trim != JobMode::On {
        return None;
    }
    let (output, saved) = trimmed_response(input)?;
    gain::record(
        state_dir,
        gain::Basis::Trimmed,
        (saved / CHARS_PER_TOKEN) as u64,
    );
    Some(output.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> String {
        let mut text = String::new();
        for index in 0..3000 {
            text.push_str(&format!("test module::case_{index} ... ok\n"));
        }
        text.push_str("test module::flaky ... FAILED\ntest result: FAILED. 1 failed\n");
        text
    }

    fn event(stdout: &str) -> Value {
        json!({
            "tool_name": "Bash",
            "tool_input": { "command": "cargo test" },
            "tool_response": { "stdout": stdout, "stderr": "warning", "interrupted": false },
        })
    }

    #[test]
    fn a_long_log_shrinks_keeps_its_failure_and_the_rest_of_the_response() {
        let text = log();
        let (output, saved) = trimmed_response(&event(&text)).unwrap();
        let updated = &output["hookSpecificOutput"]["updatedToolOutput"];
        let stdout = updated["stdout"].as_str().unwrap();
        assert!(saved > text.len() / 2, "{saved}");
        assert!(stdout.contains("module::flaky ... FAILED"), "{stdout}");
        assert_eq!(updated["stderr"], "warning");
        assert_eq!(updated["interrupted"], false);
        assert_eq!(output["hookSpecificOutput"]["hookEventName"], "PostToolUse");
    }

    #[test]
    fn short_output_and_other_tools_are_left_alone() {
        assert!(trimmed_response(&event("all good\n")).is_none());
        let mut other = event(&log());
        other["tool_name"] = json!("Read");
        assert!(trimmed_response(&other).is_none());
        assert!(trimmed_response(&json!({ "tool_name": "Bash" })).is_none());
    }

    #[test]
    fn with_the_job_off_nothing_happens() {
        let dir = std::env::temp_dir().join(format!("ccc-trim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(hook(&dir, &event(&log())), None);
        assert!(!dir.exists(), "it must not even create its state");
    }
}

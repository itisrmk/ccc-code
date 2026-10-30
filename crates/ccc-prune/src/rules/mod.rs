//! Chooses and applies a trimming rule for one tool result.

mod command;
mod generic;
mod listing;
mod logs;

use serde_json::Value;

use command::Kind;

/// Where each rule starts acting and how much it keeps. Thresholds are in bytes of output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Build and test logs shorter than this are kept whole.
    pub log_min_bytes: usize,
    /// Search results and listings shorter than this are kept whole.
    pub listing_min_bytes: usize,
    /// How many leading lines of a long listing to keep.
    pub listing_keep_lines: usize,
    /// Other command output shorter than this is kept whole.
    pub generic_min_bytes: usize,
    /// How many leading bytes of long generic output to keep.
    pub generic_head_bytes: usize,
    /// How many trailing bytes of long generic output to keep.
    pub generic_tail_bytes: usize,
}

impl Default for Limits {
    /// Chosen from replay on real sessions: tightening further saved little and removed
    /// more of what later turns went on to reference.
    fn default() -> Self {
        Self {
            log_min_bytes: 2_000,
            listing_min_bytes: 2_000,
            listing_keep_lines: 50,
            generic_min_bytes: 8_000,
            generic_head_bytes: 4_000,
            generic_tail_bytes: 2_000,
        }
    }
}

/// A trimmed result and the rule that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Trimmed {
    pub(crate) rule: &'static str,
    pub(crate) text: String,
}

/// Applies the rule for this tool, if any. Returns `None` when the output should stay
/// whole, which includes any trim that wouldn't actually make it shorter.
pub(crate) fn trim(
    limits: &Limits,
    tool: &str,
    input: Option<&Value>,
    text: &str,
) -> Option<Trimmed> {
    let kind = match tool {
        "Bash" => {
            let command = input?.get("command").and_then(Value::as_str)?;
            command::classify(command)
        }
        "Grep" | "Glob" => Kind::Listing,
        "BashOutput" | "TaskOutput" => Kind::Generic,
        // Reads and subagent reports are the point of the call, and unknown tools may
        // return anything, so they stay whole.
        _ => Kind::Verbatim,
    };
    let (rule, trimmed) = match kind {
        Kind::Verbatim => return None,
        Kind::Log => ("log", logs::trim(limits, text)?),
        Kind::Listing => ("listing", listing::trim(limits, text)?),
        Kind::Generic => ("generic", generic::trim(limits, text)?),
    };
    (trimmed.len() < text.len()).then_some(Trimmed {
        rule,
        text: trimmed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn big_output() -> String {
        (0..2000)
            .map(|i| format!("output line number {i}\n"))
            .collect()
    }

    #[test]
    fn dispatches_by_tool_and_command() {
        let limits = Limits::default();
        let output = big_output();
        let rule = |tool: &str, input: Value| {
            trim(&limits, tool, Some(&input), &output).map(|trimmed| trimmed.rule)
        };
        assert_eq!(
            rule("Bash", json!({ "command": "cargo test" })),
            Some("log")
        );
        assert_eq!(
            rule("Bash", json!({ "command": "rg foo" })),
            Some("listing")
        );
        assert_eq!(
            rule("Bash", json!({ "command": "curl x" })),
            Some("generic")
        );
        assert_eq!(rule("Bash", json!({ "command": "cat x" })), None);
        assert_eq!(rule("Grep", json!({})), Some("listing"));
        assert_eq!(rule("TaskOutput", json!({})), Some("generic"));
        assert_eq!(rule("Read", json!({})), None);
        assert_eq!(rule("Agent", json!({})), None);
        assert_eq!(rule("mcp__x__y", json!({})), None);
    }

    #[test]
    fn a_trim_that_would_not_shrink_output_is_dropped() {
        // Short lines just over the threshold, where the marker can outweigh the savings.
        let limits = Limits {
            listing_min_bytes: 1,
            listing_keep_lines: 1,
            ..Limits::default()
        };
        let input = json!({});
        assert!(trim(&limits, "Grep", Some(&input), "a\nb\n").is_none());
    }
}

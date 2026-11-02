//! Shrinks tool output the first time it is sent.
//!
//! The server binds thinking blocks to the exact history before them, so a result can't
//! be trimmed after it has been sent. Every decision here is therefore a pure function of
//! the result, the call that produced it and what came before it. Recompiling a longer
//! conversation always produces the same bytes for the messages it shares.
//!
//! Rules only ever drop whole lines of output, or cut one overlong line, and say they did.
//! They never paraphrase, and source code being read is left alone.

mod rules;
mod stage;

pub use rules::Limits;
pub use stage::Prune;

use serde_json::Value;

/// Shortens one tool's output, for callers outside the proxy. `None` means leave it whole.
#[must_use]
pub fn trim_output(tool: &str, input: Option<&Value>, text: &str) -> Option<String> {
    rules::trim(&Limits::default(), tool, input, text).map(|trimmed| trimmed.text)
}

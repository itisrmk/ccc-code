//! `ccc bench --export`: numbers only, for sharing a bench result if you choose to.
//!
//! Never sends anything itself. What it writes is meant to be read before being shared, and
//! it's built from totals, so no prompt, path, model output or session id can reach it.

use std::fs;
use std::path::Path;

use anyhow::Context as _;
use ccc_bench::{ReplayStats, Report};
use serde_json::{Value, json};

fn stats(stats: &ReplayStats) -> Value {
    json!({
        "name": stats.name,
        "context_tokens": stats.context_tokens.round(),
        "tool_result_tokens": stats.tool_result_tokens.round(),
        "cache_read_tokens": stats.cache_read_tokens.round(),
        "cache_write_tokens": stats.cache_write_tokens.round(),
        "cost_usd": (stats.cost_usd * 100.0).round() / 100.0,
    })
}

/// Writes `report` as JSON to `path`.
pub(crate) fn write(report: &Report, path: &Path) -> anyhow::Result<()> {
    let scenarios: Vec<Value> = report.scenarios().iter().map(stats).collect();
    let body = json!({
        "format": 1,
        "ccc_version": env!("CARGO_PKG_VERSION"),
        "sessions": report.session_count(),
        "calls": report.call_count(),
        "baseline_misses": report.baseline_misses(0.10),
        "scenarios": scenarios,
    });
    let text = serde_json::to_string_pretty(&body).context("rendering export")?;
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_bench::{Call, LoadSummary, Session, replay};
    use ccc_ledger::Usage;
    use ccc_wire::Request;
    use std::path::PathBuf;

    #[test]
    fn nothing_from_a_transcript_reaches_the_export() {
        let canary = "canary-secret-project-name";
        let body = json!({ "model": "claude-opus-5-5", "system": canary, "messages": [
            { "role": "user", "content": canary },
        ] });
        let session = Session {
            path: PathBuf::from(format!("/home/{canary}/session.jsonl")),
            calls: vec![Call {
                at_millis: 0,
                request: Request::parse(body.to_string().as_bytes()).unwrap(),
                usage: Usage {
                    cache_write_1h: 1000,
                    output: 10,
                    ..Usage::default()
                },
            }],
        };
        let report = replay(&[session], LoadSummary::default(), &[]);
        let path = std::env::temp_dir().join(format!("ccc-export-{}.json", std::process::id()));

        write(&report, &path).unwrap();

        let written = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(!written.contains("canary"), "{written}");
        assert!(written.contains("\"scenarios\""), "{written}");
    }
}

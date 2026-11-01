//! `ccc whatif`: where sessions' money went, and what restarting or refreshing would change.
//! Reads local transcripts only. The export holds totals, so no prompt, path or id reaches it.

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::Context as _;
use ccc_bench::{
    KeepWarmOutcome, RestartOutcome, RestartPolicy, Spend, Timeline, load_timelines,
    simulate_keep_warm, simulate_restarts, spend,
};
use serde_json::{Value, json};

/// Restart once context passes each of these.
const RESTART_ABOVE: [u64; 4] = [100_000, 200_000, 300_000, 500_000];
/// Context a fresh session has to bring back, from a short handoff to a lot of re-reading.
const RESTORED: [u64; 3] = [10_000, 50_000, 100_000];
/// How long to refresh a cache after a session's last call.
const KEEP_WARM_HOURS: [i64; 6] = [2, 4, 8, 12, 24, 48];
const HOUR_MILLIS: i64 = 3_600_000;

struct RestartRow {
    above_tokens: u64,
    outcomes: Vec<(u64, RestartOutcome)>,
}

struct Results {
    spend: Spend,
    restarts: Vec<RestartRow>,
    keep_warm: Vec<(i64, KeepWarmOutcome)>,
}

fn measure(timelines: &[Timeline]) -> Results {
    let restarts = RESTART_ABOVE
        .iter()
        .map(|&above_tokens| RestartRow {
            above_tokens,
            outcomes: RESTORED
                .iter()
                .map(|&restored_tokens| {
                    let policy = RestartPolicy {
                        above_tokens,
                        restored_tokens,
                    };
                    (restored_tokens, simulate_restarts(timelines, Some(policy)))
                })
                .collect(),
        })
        .collect();
    let keep_warm = KEEP_WARM_HOURS
        .iter()
        .map(|&hours| (hours, simulate_keep_warm(timelines, hours * HOUR_MILLIS)))
        .collect();
    Results {
        spend: spend(timelines),
        restarts,
        keep_warm,
    }
}

fn share(part: f64, total: f64) -> f64 {
    if total > 0.0 {
        100.0 * part / total
    } else {
        0.0
    }
}

fn top_share(report: &Spend, count: usize) -> f64 {
    share(
        report.session_costs.iter().take(count).sum(),
        report.total_usd,
    )
}

pub(crate) fn run(transcripts: &Path, export: Option<&Path>) -> anyhow::Result<()> {
    let timelines = load_timelines(transcripts)
        .with_context(|| format!("loading transcripts from {}", transcripts.display()))?;
    let results = measure(&timelines);
    print_report(&mut std::io::stdout(), &results)?;
    if let Some(path) = export {
        let text =
            serde_json::to_string_pretty(&export_json(&results)).context("rendering export")?;
        fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn print_report(out: &mut impl Write, results: &Results) -> anyhow::Result<()> {
    let report = &results.spend;
    let total = report.total_usd;
    writeln!(
        out,
        "{} calls in {} sessions, ${total:.2} at list prices",
        report.calls, report.sessions
    )?;
    writeln!(
        out,
        "The top session is {:.1}% of that and the top two are {:.1}%.\n",
        top_share(report, 1),
        top_share(report, 2),
    )?;
    writeln!(out, "Where the money went")?;
    for (label, usd) in [
        ("cache reads", report.read_usd),
        ("cache writes", report.write_usd),
        ("output", report.output_usd),
        ("uncached input", report.input_usd),
    ] {
        writeln!(out, "  {label:<36}{:>10.1}%", share(usd, total))?;
    }
    writeln!(
        out,
        "  {:<36}{:>10.1}%",
        "tool output, estimated (overlaps)",
        share(report.tool_output_usd, total)
    )?;
    writeln!(
        out,
        "  {:<36}{:>10.1}%  {} rewrites",
        "rewrites after idle (overlaps)",
        share(report.idle_rewrite_usd, total),
        report.idle_rewrites,
    )?;
    writeln!(
        out,
        "  {:<36}{:>10.1}%  {} rewrites\n",
        "other rewrites (overlaps)",
        share(report.other_rewrite_usd, total),
        report.other_rewrites,
    )?;
    writeln!(out, "Cost by context size")?;
    for band in &report.bands {
        writeln!(
            out,
            "  from {:>4}K tokens {:>8} calls {:>9.1}%",
            band.from_tokens / 1000,
            band.calls,
            share(band.cost_usd, total)
        )?;
    }
    writeln!(
        out,
        "\nRestarting at the next prompt above a context size: change in cost (restarts)"
    )?;
    write!(out, "  {:<12}", "above")?;
    for restored in RESTORED {
        write!(out, "{:>22}", format!("{}K restored", restored / 1000))?;
    }
    writeln!(out)?;
    for row in &results.restarts {
        write!(out, "  {:<12}", format!("{}K", row.above_tokens / 1000))?;
        for (_, outcome) in &row.outcomes {
            let cell = format!(
                "{:+.1}% ({})",
                share(outcome.cost_usd - total, total),
                outcome.restarts
            );
            write!(out, "{cell:>22}")?;
        }
        writeln!(out)?;
    }
    writeln!(
        out,
        "  Upper limits: rework beyond restoring that context isn't counted, nor any change in \
         the quality of the work.\n"
    )?;
    writeln!(
        out,
        "Refreshing the cache through idle gaps, assuming something stays running to do it"
    )?;
    writeln!(
        out,
        "  {:<8}{:>12}{:>10}{:>10}{:>10}{:>10}",
        "window", "prevented", "saved", "refresh", "wasted", "net"
    )?;
    for (hours, outcome) in &results.keep_warm {
        writeln!(
            out,
            "  {:<8}{:>12}{:>10}{:>10}{:>10}{:>9.1}%",
            format!("{hours}h"),
            format!("{}/{}", outcome.prevented, outcome.rewrites),
            format!("${:.2}", outcome.saved_usd),
            format!("${:.2}", outcome.refresh_usd),
            format!("${:.2}", outcome.wasted_usd),
            share(outcome.net_usd(), total),
        )?;
    }
    Ok(())
}

fn export_json(results: &Results) -> Value {
    let report = &results.spend;
    let total = report.total_usd;
    let bands: Vec<Value> = report
        .bands
        .iter()
        .map(|band| {
            json!({
                "from_tokens": band.from_tokens,
                "calls": band.calls,
                "cost_share": rounded(share(band.cost_usd, total)),
            })
        })
        .collect();
    let restarts: Vec<Value> = results
        .restarts
        .iter()
        .flat_map(|row| {
            row.outcomes.iter().map(|(restored, outcome)| {
                json!({
                    "above_tokens": row.above_tokens,
                    "restored_tokens": restored,
                    "restarts": outcome.restarts,
                    "cost_change": rounded(share(outcome.cost_usd - total, total)),
                })
            })
        })
        .collect();
    let keep_warm: Vec<Value> = results
        .keep_warm
        .iter()
        .map(|(hours, outcome)| {
            json!({
                "window_hours": hours,
                "rewrites": outcome.rewrites,
                "prevented": outcome.prevented,
                "net_share": rounded(share(outcome.net_usd(), total)),
            })
        })
        .collect();
    json!({
        "format": 1,
        "ccc_version": env!("CARGO_PKG_VERSION"),
        "calls": report.calls,
        "sessions": report.sessions,
        "total_usd": rounded(total),
        "top_session_share": rounded(top_share(report, 1)),
        "top_two_share": rounded(top_share(report, 2)),
        "read_share": rounded(share(report.read_usd, total)),
        "write_share": rounded(share(report.write_usd, total)),
        "output_share": rounded(share(report.output_usd, total)),
        "tool_output_share": rounded(share(report.tool_output_usd, total)),
        "idle_rewrites": report.idle_rewrites,
        "idle_rewrite_share": rounded(share(report.idle_rewrite_usd, total)),
        "other_rewrites": report.other_rewrites,
        "other_rewrite_share": rounded(share(report.other_rewrite_usd, total)),
        "bands": bands,
        "restarts": restarts,
        "keep_warm": keep_warm,
    })
}

fn rounded(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_bench::RecordedCall;
    use ccc_ledger::Usage;

    fn timeline(session: &str) -> Timeline {
        let mut calls = Vec::new();
        for index in 0..60_u64 {
            calls.push(RecordedCall {
                at_millis: index as i64 * 10_000,
                model: "claude-opus-5-5".to_owned(),
                usage: Usage {
                    cache_read: index * 15_000,
                    cache_write_1h: 15_000,
                    output: 200,
                    ..Usage::default()
                },
                starts_turn: true,
            });
        }
        Timeline {
            session: session.to_owned(),
            calls,
            ..Timeline::default()
        }
    }

    #[test]
    fn the_report_covers_every_section_and_the_export_holds_no_ids() {
        let canary = "canary-session-id";
        let results = measure(&[timeline(canary)]);
        let mut printed = Vec::new();
        print_report(&mut printed, &results).unwrap();
        let printed = String::from_utf8(printed).unwrap();
        for section in [
            "Where the money went",
            "Cost by context size",
            "Restarting",
            "Refreshing",
        ] {
            assert!(printed.contains(section), "{printed}");
        }
        let exported = export_json(&results).to_string();
        assert!(!exported.contains("canary"), "{exported}");
        assert!(!printed.contains("canary"), "{printed}");
        assert!(exported.contains("\"restarts\""), "{exported}");
    }
}

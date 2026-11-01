//! `ccc statusline`: the line Claude Code shows under the prompt. Leads with what someone on a
//! plan cares about, how much of their limit is left and whether the cache is about to cost them.
//!
//! Claude Code runs this constantly and cancels it if it's slow, so it reads only what it's handed
//! and one small file, and never touches a model.

use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use serde_json::Value;

use crate::gain;

const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";
/// A cache this close to expiring is worth a warning: the next message may have to rewrite it.
const CACHE_WARNING_SECONDS: i64 = 5 * 60;
/// Past this much of a limit, it's worth noticing.
const LIMIT_WARNING_PERCENT: f64 = 80.0;
/// From real runs: restarting only saved money once a session got this large.
const LARGE_CONTEXT_TOKENS: u64 = 300_000;
const SEPARATOR: &str = " · ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tone {
    Plain,
    Warn,
    Alarm,
}

fn paint(text: String, tone: Tone, color: bool) -> String {
    match (tone, color) {
        (Tone::Warn, true) => format!("{YELLOW}{text}{RESET}"),
        (Tone::Alarm, true) => format!("{RED}{text}{RESET}"),
        _ => text,
    }
}

fn tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1000 {
        format!("{}K", count / 1000)
    } else {
        count.to_string()
    }
}

/// The tightest limit, or spending if this account has no limits to report.
fn limit_part(input: &Value, color: bool) -> Option<String> {
    let used = |path: &str| input.pointer(path).and_then(Value::as_f64);
    let windows = [
        ("5h", used("/rate_limits/five_hour/used_percentage")),
        ("week", used("/rate_limits/seven_day/used_percentage")),
        ("spend", used("/rate_limits/spend_limit/used_percentage")),
    ];
    let tightest = windows
        .into_iter()
        .filter_map(|(label, percent)| Some((label, percent?)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((label, percent)) = tightest {
        let tone = if percent >= LIMIT_WARNING_PERCENT {
            Tone::Warn
        } else {
            Tone::Plain
        };
        return Some(paint(format!("{label} limit {percent:.0}%"), tone, color));
    }
    let cost = input
        .pointer("/cost/total_cost_usd")
        .and_then(Value::as_f64)?;
    Some(format!("${cost:.2}"))
}

fn cache_part(input: &Value, now: i64, color: bool) -> Option<String> {
    let cache = input.get("prompt_cache")?;
    let cold_cost = cache
        .get("recache_tokens_if_cold")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let expires = cache.get("expires_at").and_then(Value::as_i64);
    let warm = cache.get("warm").and_then(Value::as_bool).unwrap_or(false);
    match expires.map(|at| at - now) {
        Some(left) if warm && left > 0 => {
            let tone = if left <= CACHE_WARNING_SECONDS {
                Tone::Warn
            } else {
                Tone::Plain
            };
            let text = if left >= 3600 {
                format!("cache {}h{}m", left / 3600, left % 3600 / 60)
            } else {
                format!("cache {}m", (left / 60).max(1))
            };
            Some(paint(text, tone, color))
        }
        _ if cold_cost > 0 => Some(paint(
            format!("cache cold: next message rewrites {}", tokens(cold_cost)),
            Tone::Alarm,
            color,
        )),
        _ => None,
    }
}

fn context_part(input: &Value, color: bool) -> Option<String> {
    let usage = input.pointer("/context_window/current_usage")?;
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let total = count("input_tokens")
        + count("cache_creation_input_tokens")
        + count("cache_read_input_tokens");
    (total > 0).then(|| {
        let tone = if total >= LARGE_CONTEXT_TOKENS {
            Tone::Warn
        } else {
            Tone::Plain
        };
        paint(format!("context {}", tokens(total)), tone, color)
    })
}

/// Builds the line. Anything Claude Code didn't report is just left out.
fn render(input: &Value, now: i64, saved_today: u64, color: bool) -> String {
    let saved = (saved_today > 0).then(|| format!("saved {} today", tokens(saved_today)));
    [
        limit_part(input, color),
        cache_part(input, now, color),
        context_part(input, color),
        saved,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(SEPARATOR)
}

pub(crate) fn run(state_dir: &Path) -> anyhow::Result<()> {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .context("reading the status")?;
    // A status line that fails shows nothing, so unreadable input is just an empty line.
    let input: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(0));
    let color = std::env::var_os("NO_COLOR").is_none();
    let line = render(&input, now, gain::today_tokens(state_dir), color);
    writeln!(std::io::stdout(), "{line}").context("writing the status")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const NOW: i64 = 1_000_000;

    fn full() -> Value {
        json!({
            "cost": { "total_cost_usd": 1.5 },
            "context_window": { "current_usage": {
                "input_tokens": 500, "output_tokens": 9,
                "cache_creation_input_tokens": 500, "cache_read_input_tokens": 339_000,
            } },
            "prompt_cache": { "warm": true, "expires_at": NOW + 12 * 60, "recache_tokens_if_cold": 340_000 },
            "rate_limits": {
                "five_hour": { "used_percentage": 23.5 },
                "seven_day": { "used_percentage": 62.0 },
            },
        })
    }

    #[test]
    fn shows_the_tightest_limit_cache_context_and_savings() {
        let line = render(&full(), NOW, 18_400, false);
        assert_eq!(
            line,
            "week limit 62% · cache 12m · context 340K · saved 18K today"
        );
    }

    #[test]
    fn an_api_user_sees_spending_instead() {
        let mut input = full();
        input.as_object_mut().unwrap().remove("rate_limits");
        assert!(render(&input, NOW, 0, false).starts_with("$1.50 · cache 12m"));
    }

    #[test]
    fn a_spend_limit_counts_even_past_a_hundred() {
        let input = json!({ "rate_limits": { "spend_limit": { "used_percentage": 104.0 } } });
        assert_eq!(render(&input, NOW, 0, false), "spend limit 104%");
    }

    #[test]
    fn a_cold_cache_says_what_the_next_message_costs() {
        let mut input = full();
        input["prompt_cache"]["warm"] = json!(false);
        assert!(
            render(&input, NOW, 0, false).contains("cache cold: next message rewrites 340K"),
            "{}",
            render(&input, NOW, 0, false)
        );
        input["prompt_cache"]["warm"] = json!(true);
        input["prompt_cache"]["expires_at"] = json!(NOW - 5);
        assert!(render(&input, NOW, 0, false).contains("cache cold"));
    }

    #[test]
    fn warnings_get_color_only_when_it_matters_and_when_allowed() {
        let mut input = full();
        input["prompt_cache"]["expires_at"] = json!(NOW + 120);
        let line = render(&input, NOW, 0, true);
        assert!(
            line.contains(&format!("{YELLOW}cache 2m{RESET}")),
            "{line:?}"
        );
        assert!(
            line.contains(&format!("{YELLOW}context 340K{RESET}")),
            "{line:?}"
        );
        assert!(!line.contains(&format!("{YELLOW}week")), "{line:?}");
        assert!(!render(&input, NOW, 0, false).contains('\x1b'));
    }

    #[test]
    fn long_lives_and_early_sessions_read_well() {
        let mut input = full();
        input["prompt_cache"]["expires_at"] = json!(NOW + 3900);
        assert!(render(&input, NOW, 0, false).contains("cache 1h5m"));
        assert_eq!(render(&Value::Null, NOW, 0, false), "");
        assert_eq!(
            render(
                &json!({ "prompt_cache": { "expires_at": null } }),
                NOW,
                0,
                false
            ),
            ""
        );
    }
}

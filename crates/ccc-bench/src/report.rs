//! Runs the replay and renders the results.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use ccc_ledger::{Usage, cost_usd};
use ccc_wire::Request;
use serde_json::Value;

use crate::cache_sim::{
    CacheSim, Fingerprint, Ttl, chain_next, fingerprint, message_fingerprint, tools_fingerprint,
};
use crate::estimate::{self, Rates, Size, Source};
use crate::transcript::{Call, LoadSummary, Session};

/// How far the simulated baseline may stray from recorded usage.
const BASELINE_TOLERANCE: f64 = 0.10;

/// The number of [`Source`] variants, for per-source tallies.
const SOURCES: usize = 7;
const SOURCE_LABELS: [(Source, &str); SOURCES] = [
    (Source::SystemAndTools, "system prompt + tool definitions"),
    (Source::UserText, "user text and attachments"),
    (Source::SystemRoleText, "mid-conversation system messages"),
    (Source::AssistantText, "assistant text"),
    (Source::ToolUse, "tool_use inputs"),
    (Source::ToolResult, "tool_result text"),
    (Source::Other, "other blocks"),
];

/// A named way of compiling requests to compare against the unmodified baseline.
pub struct Scenario<'a> {
    /// Shown in the results table.
    pub name: &'a str,
    /// Produces the request that would be sent for a recorded one.
    pub compile: &'a dyn Fn(&Request) -> Request,
}

impl fmt::Debug for Scenario<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Scenario")
            .field("name", &self.name)
            .finish()
    }
}

/// Simulated totals for one scenario across every replayed call.
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    /// The scenario's name.
    pub name: String,
    /// Sum over calls of estimated context tokens.
    pub context_tokens: f64,
    /// Estimated tokens of `tool_result` text, summed over calls.
    pub tool_result_tokens: f64,
    /// Estimated tokens read from the cache.
    pub cache_read_tokens: f64,
    /// Estimated tokens written to the cache.
    pub cache_write_tokens: f64,
    /// Estimated cost at list prices, with output tokens taken from the recording.
    pub cost_usd: f64,
    by_source_chars: [f64; SOURCES],
    images: f64,
    thinking: f64,
    blocks: f64,
}

/// Everything `ccc bench` prints.
#[derive(Debug, Clone)]
pub struct Report {
    summary: LoadSummary,
    sessions: usize,
    calls: usize,
    rates: Rates,
    held_out_error: Option<f64>,
    held_out_worst: Option<f64>,
    actual_context: f64,
    actual_read: f64,
    actual_write: f64,
    actual_cost: f64,
    scenarios: Vec<ReplayStats>,
}

impl Report {
    /// The unmodified baseline's stats.
    #[must_use]
    pub fn baseline(&self) -> &ReplayStats {
        &self.scenarios[0]
    }

    /// Stats for the scenario called `name`, if it was run.
    #[must_use]
    pub fn scenario(&self, name: &str) -> Option<&ReplayStats> {
        self.scenarios.iter().find(|stats| stats.name == name)
    }

    /// The measures on which the baseline misses recorded usage by more than `tolerance`,
    /// as a fraction. Empty means the simulation reproduces what actually happened.
    #[must_use]
    pub fn baseline_misses(&self, tolerance: f64) -> Vec<&'static str> {
        let baseline = self.baseline();
        let checks = [
            (
                "context size",
                relative_error(baseline.context_tokens, self.actual_context),
            ),
            (
                "cache reads",
                relative_error(baseline.cache_read_tokens, self.actual_read),
            ),
            (
                "cache writes",
                relative_error(baseline.cache_write_tokens, self.actual_write),
            ),
            ("cost", relative_error(baseline.cost_usd, self.actual_cost)),
            (
                "held-out sessions",
                self.held_out_error.unwrap_or(f64::INFINITY),
            ),
        ];
        checks
            .into_iter()
            .filter(|(_, error)| error.abs() > tolerance)
            .map(|(label, _)| label)
            .collect()
    }
}

fn relative_error(estimate: f64, actual: f64) -> f64 {
    if actual == 0.0 {
        0.0
    } else {
        estimate / actual - 1.0
    }
}

/// Replays `sessions`, first unmodified and then through each of `scenarios`.
#[must_use]
pub fn replay(sessions: &[Session], summary: LoadSummary, scenarios: &[Scenario<'_>]) -> Report {
    let per_session: Vec<Vec<(Size, f64)>> = sessions.iter().map(estimate::samples).collect();
    let all_samples: Vec<(Size, f64)> = per_session.iter().flatten().copied().collect();
    let rates = Rates::fit(&all_samples);
    let (held_out_error, held_out_worst) = held_out(&per_session);

    let identity = |request: &Request| request.clone();
    let mut results = vec![replay_scenario(sessions, rates, "baseline", &identity)];
    for scenario in scenarios {
        results.push(replay_scenario(
            sessions,
            rates,
            scenario.name,
            scenario.compile,
        ));
    }

    let calls = || sessions.iter().flat_map(|session| &session.calls);
    Report {
        summary,
        sessions: sessions.len(),
        calls: all_samples.len(),
        rates,
        held_out_error,
        held_out_worst,
        actual_context: calls().map(|call| call.usage.context() as f64).sum(),
        actual_read: calls().map(|call| call.usage.cache_read as f64).sum(),
        actual_write: calls()
            .map(|call| (call.usage.cache_write_5m + call.usage.cache_write_1h) as f64)
            .sum(),
        actual_cost: calls()
            .filter_map(|call| cost_usd(call.model(), &call.usage))
            .sum(),
        scenarios: results,
    }
}

/// Fits rates on all sessions but one, then predicts that one. Returns the error over
/// all predictions, and the worst single session's error among those large enough to
/// matter (at least 1% of total tokens), so a fit can't merely memorize its data.
fn held_out(per_session: &[Vec<(Size, f64)>]) -> (Option<f64>, Option<f64>) {
    if per_session.len() < 2 {
        return (None, None);
    }
    let grand_total: f64 = per_session.iter().flatten().map(|(_, tokens)| tokens).sum();
    let mut predicted_total = 0.0;
    let mut worst: Option<f64> = None;
    for (held, samples) in per_session.iter().enumerate() {
        let training: Vec<(Size, f64)> = per_session
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != held)
            .flat_map(|(_, session)| session.iter().copied())
            .collect();
        let rates = Rates::fit(&training);
        let predicted: f64 = samples.iter().map(|(size, _)| rates.tokens(*size)).sum();
        let actual: f64 = samples.iter().map(|(_, tokens)| tokens).sum();
        predicted_total += predicted;
        if actual >= grand_total * 0.01 {
            let error = relative_error(predicted, actual);
            worst = Some(worst.map_or(
                error,
                |w: f64| if error.abs() > w.abs() { error } else { w },
            ));
        }
    }
    (Some(relative_error(predicted_total, grand_total)), worst)
}

fn replay_scenario(
    sessions: &[Session],
    rates: Rates,
    name: &str,
    compile: &dyn Fn(&Request) -> Request,
) -> ReplayStats {
    let mut stats = ReplayStats {
        name: name.to_owned(),
        ..ReplayStats::default()
    };
    // The cache is shared across an account: a subagent or resumed session reads prefixes
    // its parent wrote. So all calls replay in time order against a single cache.
    let mut calls: Vec<&Call> = sessions.iter().flat_map(|session| &session.calls).collect();
    calls.sort_by_key(|call| call.at_millis);

    let mut cache = CacheSim::default();
    let mut facts: HashMap<*const Value, MessageFacts> = HashMap::new();
    let mut ttl = Ttl::OneHour;
    for call in calls {
        ttl = recorded_ttl(call).unwrap_or(ttl);
        let compiled = compile(&call.request);
        replay_call(
            call, &compiled, rates, ttl, &mut cache, &mut facts, &mut stats,
        );
    }
    stats.tool_result_tokens =
        stats.by_source_chars[Source::ToolResult as usize] * rates.tokens_per_char;
    stats
}

/// Measurements of one message, kept with the message so its address stays unique
/// for as long as the entry exists.
struct MessageFacts {
    _keep_alive: Arc<Value>,
    size: Size,
    fingerprint: Fingerprint,
    by_source_chars: [f64; SOURCES],
}

fn replay_call(
    call: &Call,
    compiled: &Request,
    rates: Rates,
    ttl: Ttl,
    cache: &mut CacheSim,
    facts: &mut HashMap<*const Value, MessageFacts>,
    stats: &mut ReplayStats,
) {
    let preamble_size = estimate::measure_preamble(call);
    let preamble = chain_next(
        &fingerprint(compiled.field("system").unwrap_or(&Value::Null)),
        &tools_fingerprint(compiled.field("tools")),
    );
    let mut chain = vec![preamble];
    let mut cumulative = vec![rates.tokens(preamble_size)];
    stats.by_source_chars[Source::SystemAndTools as usize] += preamble_size.text;
    let mut call_size = preamble_size;

    for message in compiled.messages() {
        let entry = facts
            .entry(Arc::as_ptr(message))
            .or_insert_with(|| measure(message));
        call_size = call_size + entry.size;
        for (tally, chars) in stats.by_source_chars.iter_mut().zip(entry.by_source_chars) {
            *tally += chars;
        }
        let previous = chain.last().expect("the chain starts with the preamble");
        chain.push(chain_next(previous, &entry.fingerprint));
        let tokens_so_far = cumulative.last().expect("sizes start with the preamble");
        cumulative.push(tokens_so_far + rates.tokens(entry.size));
    }

    let outcome = cache.step(call.model(), call.at_millis, ttl, &chain, &cumulative);
    stats.context_tokens += cumulative.last().copied().unwrap_or(0.0);
    stats.cache_read_tokens += outcome.read;
    stats.cache_write_tokens += outcome.write;
    stats.images += call_size.images;
    stats.thinking += call_size.thinking;
    stats.blocks += call_size.blocks;

    // Estimates are fractional; rounding per call is negligible against the totals.
    let written = outcome.write.round() as u64;
    let simulated = Usage {
        input: 0,
        cache_read: outcome.read.round() as u64,
        cache_write_5m: if ttl == Ttl::FiveMinutes { written } else { 0 },
        cache_write_1h: if ttl == Ttl::OneHour { written } else { 0 },
        output: call.usage.output,
        thinking: call.usage.thinking,
    };
    stats.cost_usd += cost_usd(call.model(), &simulated).unwrap_or(0.0);
}

fn measure(message: &Arc<Value>) -> MessageFacts {
    let mut by_source_chars = [0.0; SOURCES];
    let size = estimate::measure_message(message, |source, chars| {
        by_source_chars[source as usize] += chars;
    });
    MessageFacts {
        _keep_alive: Arc::clone(message),
        size,
        fingerprint: message_fingerprint(message),
        by_source_chars,
    }
}

fn recorded_ttl(call: &Call) -> Option<Ttl> {
    if call.usage.cache_write_1h > 0 {
        Some(Ttl::OneHour)
    } else if call.usage.cache_write_5m > 0 {
        Some(Ttl::FiveMinutes)
    } else {
        None
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let summary = &self.summary;
        writeln!(f, "Transcripts")?;
        writeln!(f, "  files read                {:>12}", summary.files)?;
        writeln!(
            f,
            "  files too old to replay   {:>12}",
            summary.files_without_wire_records
        )?;
        writeln!(f, "  sessions replayed         {:>12}", self.sessions)?;
        writeln!(f, "  API calls seen            {:>12}", summary.calls_seen)?;
        writeln!(f, "  API calls replayed        {:>12}", self.calls)?;
        writeln!(
            f,
            "  calls left out, unresolved{:>12}",
            summary.calls_unresolved
        )?;
        writeln!(
            f,
            "  message refs unresolved   {:>11.2}%",
            percent(
                summary.message_refs_unresolved as f64,
                summary.message_refs as f64
            )
        )?;

        writeln!(f, "\nToken estimator (fitted to recorded usage)")?;
        if self.rates.tokens_per_char > 0.0 {
            writeln!(
                f,
                "  characters per token      {:>12.2}",
                1.0 / self.rates.tokens_per_char
            )?;
        }
        writeln!(
            f,
            "  tokens per image          {:>12.0}",
            self.rates.tokens_per_image
        )?;
        writeln!(
            f,
            "  tokens per thinking token {:>12.3}",
            self.rates.tokens_per_thinking_token
        )?;
        writeln!(
            f,
            "  tokens per content block  {:>12.1}",
            self.rates.tokens_per_block
        )?;

        let baseline = self.baseline();
        writeln!(f, "\nBaseline check: simulated vs recorded")?;
        writeln!(
            f,
            "  {:<26}{:>16}{:>16}{:>9}",
            "", "simulated", "recorded", "error"
        )?;
        let row = |f: &mut fmt::Formatter<'_>, label: &str, simulated: f64, actual: f64| {
            writeln!(
                f,
                "  {label:<26}{:>16}{:>16}{:>8.1}%",
                thousands(simulated),
                thousands(actual),
                100.0 * relative_error(simulated, actual)
            )
        };
        row(
            f,
            "context tokens",
            baseline.context_tokens,
            self.actual_context,
        )?;
        row(
            f,
            "cache read tokens",
            baseline.cache_read_tokens,
            self.actual_read,
        )?;
        row(
            f,
            "cache write tokens",
            baseline.cache_write_tokens,
            self.actual_write,
        )?;
        writeln!(
            f,
            "  {:<26}{:>16}{:>16}{:>8.1}%",
            "cost at list prices",
            format!("${:.2}", baseline.cost_usd),
            format!("${:.2}", self.actual_cost),
            100.0 * relative_error(baseline.cost_usd, self.actual_cost)
        )?;
        if let (Some(all), Some(worst)) = (self.held_out_error, self.held_out_worst) {
            writeln!(
                f,
                "  held-out sessions         overall {:+.1}%, worst large session {:+.1}%",
                100.0 * all,
                100.0 * worst
            )?;
        }
        let misses = self.baseline_misses(BASELINE_TOLERANCE);
        if misses.is_empty() {
            writeln!(f, "  verdict: every measure within 10% of recorded usage")?;
        } else {
            writeln!(f, "  verdict: over the 10% bar on {}", misses.join(", "))?;
        }

        writeln!(
            f,
            "\nWhat fills the context (share of estimated tokens, weighted by call)"
        )?;
        let tokens = |chars: f64| chars * self.rates.tokens_per_char;
        let image_tokens = baseline.images * self.rates.tokens_per_image;
        let thinking_tokens = baseline.thinking * self.rates.tokens_per_thinking_token;
        let framing_tokens = baseline.blocks * self.rates.tokens_per_block;
        let total = baseline.context_tokens.max(1.0);
        for (source, label) in SOURCE_LABELS {
            let share = tokens(baseline.by_source_chars[source as usize]);
            writeln!(f, "  {label:<36}{:>6.1}%", 100.0 * share / total)?;
        }
        writeln!(
            f,
            "  {:<36}{:>6.1}%",
            "images",
            100.0 * image_tokens / total
        )?;
        writeln!(
            f,
            "  {:<36}{:>6.1}%",
            "hidden thinking (signed blocks)",
            100.0 * thinking_tokens / total
        )?;
        writeln!(
            f,
            "  {:<36}{:>6.1}%",
            "per-block overhead (fitted)",
            100.0 * framing_tokens / total
        )?;

        writeln!(f, "\nScenarios")?;
        writeln!(
            f,
            "  {:<24}{:>12}{:>12}{:>12}{:>10}{:>10}",
            "", "tool_result", "context", "cache writes", "cost", "cost"
        )?;
        for stats in &self.scenarios {
            writeln!(
                f,
                "  {:<24}{:>11.1}%{:>11.1}%{:>11.1}%{:>10}{:>9.1}%",
                stats.name,
                100.0 * relative_error(stats.tool_result_tokens, baseline.tool_result_tokens),
                100.0 * relative_error(stats.context_tokens, baseline.context_tokens),
                100.0 * relative_error(stats.cache_write_tokens, baseline.cache_write_tokens),
                format!("${:.2}", stats.cost_usd),
                100.0 * relative_error(stats.cost_usd, baseline.cost_usd),
            )?;
        }
        Ok(())
    }
}

fn percent(part: f64, whole: f64) -> f64 {
    if whole == 0.0 {
        0.0
    } else {
        100.0 * part / whole
    }
}

fn thousands(value: f64) -> String {
    let digits = format!("{:.0}", value.max(0.0));
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn usage(read: u64, write: u64) -> Usage {
        Usage {
            input: 0,
            cache_read: read,
            cache_write_5m: 0,
            cache_write_1h: write,
            output: 10,
            thinking: 0,
        }
    }

    fn call(at_millis: i64, texts: &[&str], usage: Usage) -> Call {
        let messages: Vec<Arc<Value>> = texts
            .iter()
            .map(|text| Arc::new(json!({ "role": "user", "content": text })))
            .collect();
        let mut fields = serde_json::Map::new();
        fields.insert("model".to_owned(), json!("claude-opus-5-5"));
        Call {
            at_millis,
            request: Request::from_parts(fields, messages),
            usage,
        }
    }

    fn session(calls: Vec<Call>) -> Session {
        Session {
            path: PathBuf::from("test.jsonl"),
            calls,
        }
    }

    #[test]
    fn baseline_matches_recorded_usage_on_a_consistent_session() {
        let (long, longer) = ("x".repeat(4000), "y".repeat(8000));
        let (other, other_longer) = ("p".repeat(4000), "q".repeat(8000));
        let first = session(vec![
            call(0, &[&long], usage(0, 1000)),
            call(1000, &[&long, &longer], usage(1000, 2000)),
        ]);
        let second = session(vec![
            call(0, &[&other], usage(0, 1000)),
            call(1000, &[&other, &other_longer], usage(1000, 2000)),
        ]);
        let sessions = [first, second];

        let report = replay(&sessions, LoadSummary::default(), &[]);

        assert_eq!(report.baseline_misses(0.01), Vec::<&str>::new(), "{report}");
        assert_eq!(report.calls, 4);
    }

    #[test]
    fn sessions_read_prefixes_other_sessions_cached() {
        let shared = "s".repeat(4000);
        let parent = session(vec![call(0, &[&shared], usage(0, 1000))]);
        let child = session(vec![call(1000, &[&shared], usage(1000, 0))]);
        let report = replay(&[parent, child], LoadSummary::default(), &[]);
        assert!(report.baseline().cache_read_tokens > 0.0, "{report}");
    }

    #[test]
    fn a_scenario_that_shrinks_the_newest_message_keeps_cache_reads() {
        let old = "o".repeat(4000);
        let new = "n".repeat(8000);
        let sessions = [session(vec![
            call(0, &[&old], usage(0, 1000)),
            call(1000, &[&old, &new], usage(1000, 2000)),
        ])];
        let shrink_second = |request: &Request| {
            let mut compiled = request.clone();
            compiled.set_in_message(1, "/content", json!("short"));
            compiled
        };
        let scenarios = [Scenario {
            name: "shrink",
            compile: &shrink_second,
        }];

        let report = replay(&sessions, LoadSummary::default(), &scenarios);

        let baseline = report.baseline();
        let shrunk = report.scenario("shrink").unwrap();
        assert!(shrunk.context_tokens < baseline.context_tokens);
        assert_eq!(shrunk.cache_read_tokens, baseline.cache_read_tokens);
        assert!(shrunk.cost_usd < baseline.cost_usd);
    }

    #[test]
    fn a_scenario_that_edits_history_shows_the_cache_miss() {
        let first = "a".repeat(4000);
        let second = "b".repeat(100);
        let sessions = [session(vec![
            call(0, &[&first], usage(0, 1000)),
            call(1000, &[&first, &second], usage(1000, 25)),
        ])];
        // Rewrites message 0 only once message 1 exists, which is what a naive prune does.
        let rewrite_past = |request: &Request| {
            let mut compiled = request.clone();
            if compiled.messages().len() > 1 {
                let shorter = "a".repeat(3900);
                compiled.set_in_message(0, "/content", json!(shorter));
            }
            compiled
        };
        let scenarios = [Scenario {
            name: "rewrite",
            compile: &rewrite_past,
        }];

        let report = replay(&sessions, LoadSummary::default(), &scenarios);

        let baseline = report.baseline();
        let rewritten = report.scenario("rewrite").unwrap();
        assert!(rewritten.context_tokens < baseline.context_tokens);
        assert_eq!(rewritten.cache_read_tokens, 0.0);
        assert!(rewritten.cost_usd > baseline.cost_usd, "{report}");
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(1_234_567.0), "1,234,567");
        assert_eq!(thousands(12.0), "12");
        assert_eq!(thousands(1000.0), "1,000");
    }
}

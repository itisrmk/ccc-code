//! Counterfactuals over what sessions really cost, from recorded usage alone.
//!
//! The bench's replay needs exact wire requests. Asking "what if this session had restarted"
//! doesn't: recorded token counts are enough, and with no change the simulation reproduces
//! them exactly. What it can't see is what a change does to the work, so a restart's result
//! is an upper limit and the table of restored context shows how fast it shrinks.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ccc_ledger::{Rates, Usage, cost_usd, rates};
use serde_json::Value;

use crate::BenchError;
use crate::timestamp::parse_millis;

/// A call writes at least this much for its cache write to count as a lost prefix.
const LAPSE_MIN_WRITE_TOKENS: u64 = 20_000;
/// How much of a call's context a write must cover to count as a lost prefix.
const LAPSE_MIN_WRITE_SHARE: f64 = 0.5;
/// A context falling below this share of the last one means Claude Code compacted.
const COMPACTION_SHARE: f64 = 0.5;
/// Rough tokens per character of tool output.
const CHARS_PER_TOKEN: f64 = 4.0;
/// A session must have reached this size for keeping its cache warm to be considered.
const KEEP_WARM_MIN_CONTEXT: u64 = 50_000;
/// Refresh this long before a 1-hour cache lifetime runs out.
const PING_1H_MILLIS: i64 = 55 * 60_000;
/// Refresh this long before a 5-minute cache lifetime runs out.
const PING_5M_MILLIS: i64 = 270_000;
const LIFETIME_1H_MILLIS: i64 = 3_600_000;
const LIFETIME_5M_MILLIS: i64 = 300_000;

/// One API call as recorded, without its content.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedCall {
    /// When the response arrived, in milliseconds since the Unix epoch.
    pub at_millis: i64,
    /// The model that answered.
    pub model: String,
    /// What the API reported.
    pub usage: Usage,
    /// Whether a person's prompt came before this call, which is the only place a session
    /// could really have been restarted.
    pub starts_turn: bool,
}

impl RecordedCall {
    fn rates(&self) -> Rates {
        rates(&self.model, self.usage.context()).expect("timelines only hold priced models")
    }

    fn cost(&self) -> f64 {
        cost_usd(&self.model, &self.usage).expect("timelines only hold priced models")
    }
}

/// Tool output in a timeline, with the call it first went out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolOutput {
    /// Index of the first call that carried it.
    pub first_call: usize,
    /// Its length in characters.
    pub chars: usize,
}

/// The calls from one transcript file, oldest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timeline {
    /// The session these calls belong to.
    pub session: String,
    /// Whether the transcript belongs to a subagent, which can't be idle in its own right.
    pub subagent: bool,
    /// The calls, oldest first.
    pub calls: Vec<RecordedCall>,
    /// Tool output, oldest first.
    pub tool_outputs: Vec<ToolOutput>,
}

/// Loads every transcript under `dir`, counting a call only the first time it appears, since
/// resuming a session copies earlier records into a new file.
///
/// # Errors
///
/// Returns [`BenchError::Read`] if the directory tree or a file cannot be read.
pub fn load_timelines(dir: &Path) -> Result<Vec<Timeline>, BenchError> {
    let mut paths = Vec::new();
    collect(dir, &mut paths)?;
    paths.sort();
    let mut seen = HashSet::new();
    let mut timelines = Vec::new();
    for path in paths {
        let timeline = read_timeline(&path, &mut seen)?;
        if !timeline.calls.is_empty() {
            timelines.push(timeline);
        }
    }
    Ok(timelines)
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), BenchError> {
    let read_error = |source| BenchError::Read {
        path: dir.to_owned(),
        source,
    };
    for entry in fs::read_dir(dir).map_err(read_error)? {
        let path = entry.map_err(read_error)?.path();
        if path.is_dir() {
            collect(&path, found)?;
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            found.push(path);
        }
    }
    Ok(())
}

fn read_timeline(path: &Path, seen: &mut HashSet<String>) -> Result<Timeline, BenchError> {
    let file = File::open(path).map_err(|source| BenchError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut builder = TimelineBuilder::new(path, seen);
    for line in BufReader::new(file).lines() {
        // A session interrupted mid-write can end in a truncated or undecodable line.
        let Ok(line) = line else { continue };
        if let Ok(record) = serde_json::from_str::<Value>(&line) {
            builder.add(&record);
        }
    }
    Ok(builder.finish())
}

struct TimelineBuilder<'seen> {
    seen: &'seen mut HashSet<String>,
    timeline: Timeline,
    by_request: HashMap<String, usize>,
    prompt_pending: bool,
}

impl<'seen> TimelineBuilder<'seen> {
    fn new(path: &Path, seen: &'seen mut HashSet<String>) -> Self {
        let subagent = path
            .components()
            .any(|part| part.as_os_str() == "subagents");
        Self {
            seen,
            timeline: Timeline {
                subagent,
                ..Timeline::default()
            },
            by_request: HashMap::new(),
            prompt_pending: true,
        }
    }

    fn add(&mut self, record: &Value) {
        match record.get("type").and_then(Value::as_str) {
            Some("assistant") => self.add_response(record),
            Some("user") => self.add_user(record),
            _ => {}
        }
    }

    fn add_response(&mut self, record: &Value) {
        let (Some(usage), Some(id)) = (
            record.pointer("/message/usage"),
            record
                .get("requestId")
                .or_else(|| record.get("uuid"))
                .and_then(Value::as_str),
        ) else {
            return;
        };
        let usage = Usage::from_json(usage);
        if let Some(&index) = self.by_request.get(id) {
            self.timeline.calls[index].usage.update(&usage);
            return;
        }
        let model = record
            .pointer("/message/model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let at_millis = record
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_millis);
        let (Some(at_millis), true, true) = (
            at_millis,
            cost_usd(model, &usage).is_some(),
            self.seen.insert(id.to_owned()),
        ) else {
            return;
        };
        if self.timeline.session.is_empty() {
            record
                .get("sessionId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .clone_into(&mut self.timeline.session);
        }
        self.by_request
            .insert(id.to_owned(), self.timeline.calls.len());
        self.timeline.calls.push(RecordedCall {
            at_millis,
            model: model.to_owned(),
            usage,
            starts_turn: std::mem::take(&mut self.prompt_pending),
        });
    }

    fn add_user(&mut self, record: &Value) {
        let content = record.pointer("/message/content");
        let blocks = content.and_then(Value::as_array);
        let mut has_text = content.is_some_and(Value::is_string);
        let mut has_tool_result = false;
        for block in blocks.into_iter().flatten() {
            match block.get("type").and_then(Value::as_str) {
                Some("tool_result") => {
                    has_tool_result = true;
                    self.timeline.tool_outputs.push(ToolOutput {
                        first_call: self.timeline.calls.len(),
                        chars: tool_result_chars(block),
                    });
                }
                Some("text") => has_text = true,
                _ => {}
            }
        }
        // Text beside a tool result is a reminder added mid-turn, not a person's prompt.
        let meta = record.get("isMeta").and_then(Value::as_bool) == Some(true);
        if has_text && !has_tool_result && !meta {
            self.prompt_pending = true;
        }
    }

    fn finish(self) -> Timeline {
        self.timeline
    }
}

fn tool_result_chars(block: &Value) -> usize {
    match block.get("content") {
        Some(Value::String(text)) => text.len(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .map(str::len)
            .sum(),
        _ => 0,
    }
}

/// True when a call rewrote most of its context because the cached prefix had gone.
fn lost_prefix(previous: &RecordedCall, call: &RecordedCall) -> bool {
    let written = call.usage.cache_write_5m + call.usage.cache_write_1h;
    let context = call.usage.context();
    written > LAPSE_MIN_WRITE_TOKENS
        && written as f64 > LAPSE_MIN_WRITE_SHARE * context as f64
        && !compacted(previous, call)
}

fn compacted(previous: &RecordedCall, call: &RecordedCall) -> bool {
    (call.usage.context() as f64) < COMPACTION_SHARE * previous.usage.context() as f64
}

/// What a rewrite cost beyond reading the same tokens from a live cache.
fn rewrite_excess(call: &RecordedCall) -> f64 {
    let rates = call.rates();
    let written = call.usage.cache_write_5m + call.usage.cache_write_1h;
    call.usage.cache_write_5m as f64 * rates.cache_write_5m
        + call.usage.cache_write_1h as f64 * rates.cache_write_1h
        - written as f64 * rates.cache_read
}

fn lifetime_millis(call: &RecordedCall, long_lived: bool) -> i64 {
    if call.usage.cache_write_1h > 0 || (long_lived && call.usage.cache_write_5m == 0) {
        LIFETIME_1H_MILLIS
    } else {
        LIFETIME_5M_MILLIS
    }
}

impl Timeline {
    /// Whether most cache writes in this timeline use the 1-hour lifetime.
    fn long_lived(&self) -> bool {
        let long: u64 = self.calls.iter().map(|c| c.usage.cache_write_1h).sum();
        let short: u64 = self.calls.iter().map(|c| c.usage.cache_write_5m).sum();
        long >= short
    }
}

/// A span of context sizes and what calls in it cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextBand {
    /// Calls with at least this much context belong here, up to the next band.
    pub from_tokens: u64,
    /// How many calls.
    pub calls: usize,
    /// What they cost.
    pub cost_usd: f64,
}

/// Where the money went.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spend {
    /// Sessions seen.
    pub sessions: usize,
    /// Calls seen.
    pub calls: usize,
    /// Total at list prices.
    pub total_usd: f64,
    /// The part spent reading the cache.
    pub read_usd: f64,
    /// The part spent writing the cache.
    pub write_usd: f64,
    /// The part spent on output.
    pub output_usd: f64,
    /// The part spent on uncached input.
    pub input_usd: f64,
    /// The cost of each session, highest first.
    pub session_costs: Vec<f64>,
    /// Costs by context size.
    pub bands: Vec<ContextBand>,
    /// Calls that rewrote their whole context after sitting idle past the cache lifetime.
    pub idle_rewrites: usize,
    /// What those cost beyond reading the same tokens.
    pub idle_rewrite_usd: f64,
    /// Other calls that rewrote their whole context, such as after a changed prefix.
    pub other_rewrites: usize,
    /// What those cost beyond reading the same tokens.
    pub other_rewrite_usd: f64,
    /// An estimate of what tool output cost: writing it once and reading it on later calls.
    pub tool_output_usd: f64,
}

/// Context sizes the report breaks spend down by.
const BAND_STARTS: [u64; 4] = [0, 100_000, 300_000, 600_000];

/// Adds up what `timelines` cost.
#[must_use]
pub fn spend(timelines: &[Timeline]) -> Spend {
    let mut bands = BAND_STARTS.map(|from_tokens| ContextBand {
        from_tokens,
        calls: 0,
        cost_usd: 0.0,
    });
    let mut report = Spend::default();
    let mut by_session: HashMap<&str, f64> = HashMap::new();
    for timeline in timelines {
        let long_lived = timeline.long_lived();
        let mut previous: Option<&RecordedCall> = None;
        for call in &timeline.calls {
            let rates = call.rates();
            let usage = &call.usage;
            let cost = call.cost();
            report.calls += 1;
            report.total_usd += cost;
            report.read_usd += usage.cache_read as f64 * rates.cache_read;
            report.write_usd += usage.cache_write_5m as f64 * rates.cache_write_5m
                + usage.cache_write_1h as f64 * rates.cache_write_1h;
            report.output_usd += usage.output as f64 * rates.output;
            report.input_usd += usage.input as f64 * rates.input;
            *by_session.entry(&timeline.session).or_default() += cost;
            let band = bands
                .iter_mut()
                .rev()
                .find(|band| usage.context() >= band.from_tokens)
                .expect("the first band starts at zero");
            band.calls += 1;
            band.cost_usd += cost;
            if let Some(previous) = previous.filter(|previous| lost_prefix(previous, call)) {
                let idle = call.at_millis - previous.at_millis;
                if idle >= lifetime_millis(previous, long_lived) {
                    report.idle_rewrites += 1;
                    report.idle_rewrite_usd += rewrite_excess(call);
                } else {
                    report.other_rewrites += 1;
                    report.other_rewrite_usd += rewrite_excess(call);
                }
            }
            previous = Some(call);
        }
        report.tool_output_usd += tool_output_cost(timeline);
    }
    report.sessions = by_session.len();
    report.session_costs = by_session.into_values().collect();
    report.session_costs.sort_by(|a, b| b.total_cmp(a));
    report.bands = bands.to_vec();
    report
}

/// Each output is written once, then read on every later call until a compaction.
fn tool_output_cost(timeline: &Timeline) -> f64 {
    let calls = &timeline.calls;
    // What a token already in context costs from a given call on, built backward. A
    // compaction drops it, so nothing past that point counts.
    let mut carry = vec![0.0; calls.len() + 1];
    for index in (0..calls.len()).rev() {
        let compaction = index > 0 && compacted(&calls[index - 1], &calls[index]);
        carry[index] = if compaction {
            0.0
        } else {
            calls[index].rates().cache_read + carry[index + 1]
        };
    }
    let long_lived = timeline.long_lived();
    timeline
        .tool_outputs
        .iter()
        .filter_map(|output| {
            let call = calls.get(output.first_call)?;
            let rates = call.rates();
            let write = if long_lived {
                rates.cache_write_1h
            } else {
                rates.cache_write_5m
            };
            let tokens = output.chars as f64 / CHARS_PER_TOKEN;
            Some(tokens * (write + carry.get(output.first_call + 1).copied().unwrap_or(0.0)))
        })
        .sum()
}

/// When to restart, and how much a restart has to bring back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartPolicy {
    /// Start a fresh session at the next prompt once context passes this many tokens.
    pub above_tokens: u64,
    /// Context a fresh session has to restore, as a handoff or by re-reading, beyond its
    /// starting size.
    pub restored_tokens: u64,
}

/// What a restart policy would have cost.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RestartOutcome {
    /// Total at list prices.
    pub cost_usd: f64,
    /// Restarts the policy made.
    pub restarts: usize,
}

/// Replays `timelines` under `policy`, or with no restarts for `None`, which reproduces what
/// was recorded.
#[must_use]
pub fn simulate_restarts(timelines: &[Timeline], policy: Option<RestartPolicy>) -> RestartOutcome {
    let mut outcome = RestartOutcome::default();
    for timeline in timelines {
        let single = simulate_timeline(timeline, policy);
        outcome.cost_usd += single.cost_usd;
        outcome.restarts += single.restarts;
    }
    outcome
}

fn simulate_timeline(timeline: &Timeline, policy: Option<RestartPolicy>) -> RestartOutcome {
    let mut outcome = RestartOutcome::default();
    let long_lived = timeline.long_lived();
    let starting_size = timeline
        .calls
        .first()
        .map_or(0, |call| call.usage.context());
    let mut removed = 0_u64;
    let mut previous: Option<&RecordedCall> = None;
    for call in &timeline.calls {
        let usage = &call.usage;
        let context = usage.context();
        let compaction = previous.is_some_and(|previous| compacted(previous, call));
        if compaction {
            removed = 0;
        }
        let mut sized = context.saturating_sub(removed);
        let mut rewrite_all = removed > 0 && previous.is_some_and(|p| lost_prefix(p, call));
        if let (Some(policy), Some(before), true, false) =
            (policy, previous, call.starts_turn, compaction)
        {
            let fresh = starting_size
                + policy.restored_tokens
                + context.saturating_sub(before.usage.context());
            if sized > policy.above_tokens && fresh < sized {
                sized = fresh;
                removed = context - fresh;
                rewrite_all = true;
                outcome.restarts += 1;
            }
        }
        outcome.cost_usd += if removed == 0 {
            call.cost()
        } else {
            resized_cost(call, sized, rewrite_all, long_lived)
        };
        previous = Some(call);
    }
    outcome
}

/// What `call` costs with `sized` tokens of context in place of its recorded size.
fn resized_cost(call: &RecordedCall, sized: u64, rewrite_all: bool, long_lived: bool) -> f64 {
    let usage = &call.usage;
    let rates = rates(&call.model, sized).expect("timelines only hold priced models");
    let uncached = usage.input.min(sized);
    let cacheable = sized - uncached;
    let recorded_writes = usage.cache_write_5m + usage.cache_write_1h;
    let written = if rewrite_all {
        cacheable
    } else {
        recorded_writes.min(cacheable)
    };
    let write_rate = if rewrite_all {
        if long_lived {
            rates.cache_write_1h
        } else {
            rates.cache_write_5m
        }
    } else if recorded_writes == 0 {
        0.0
    } else {
        (usage.cache_write_5m as f64 * rates.cache_write_5m
            + usage.cache_write_1h as f64 * rates.cache_write_1h)
            / recorded_writes as f64
    };
    uncached as f64 * rates.input
        + (cacheable - written) as f64 * rates.cache_read
        + written as f64 * write_rate
        + usage.output as f64 * rates.output
}

/// What refreshing the cache through idle gaps would have done.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KeepWarmOutcome {
    /// Idle rewrites seen.
    pub rewrites: usize,
    /// Idle rewrites the refreshes would have prevented.
    pub prevented: usize,
    /// What those rewrites cost beyond a cache read.
    pub saved_usd: f64,
    /// What refreshing cost during gaps between calls.
    pub refresh_usd: f64,
    /// What refreshing cost after the last call of a session, for nothing.
    pub wasted_usd: f64,
}

impl KeepWarmOutcome {
    /// Savings after what the refreshes cost.
    #[must_use]
    pub fn net_usd(&self) -> f64 {
        self.saved_usd - self.refresh_usd - self.wasted_usd
    }
}

/// Replays `timelines` as if each session's cache were refreshed for up to `window_millis`
/// after its last call. Assumes something stays running to send the refreshes, so it's the
/// best such a feature could do.
#[must_use]
pub fn simulate_keep_warm(timelines: &[Timeline], window_millis: i64) -> KeepWarmOutcome {
    let mut outcome = KeepWarmOutcome::default();
    for timeline in timelines.iter().filter(|timeline| !timeline.subagent) {
        let long_lived = timeline.long_lived();
        for pair in timeline.calls.windows(2) {
            let (before, call) = (&pair[0], &pair[1]);
            let gap = call.at_millis - before.at_millis;
            let lifetime = lifetime_millis(before, long_lived);
            if gap < lifetime || !lost_prefix(before, call) {
                continue;
            }
            outcome.rewrites += 1;
            if gap <= window_millis {
                outcome.prevented += 1;
                outcome.saved_usd += rewrite_excess(call);
            }
            outcome.refresh_usd += refresh_cost(before, gap.min(window_millis), long_lived);
        }
        if let Some(last) = timeline.calls.last()
            && last.usage.context() >= KEEP_WARM_MIN_CONTEXT
        {
            outcome.wasted_usd += refresh_cost(last, window_millis, long_lived);
        }
    }
    outcome
}

fn refresh_cost(call: &RecordedCall, span_millis: i64, long_lived: bool) -> f64 {
    let every = if lifetime_millis(call, long_lived) == LIFETIME_1H_MILLIS {
        PING_1H_MILLIS
    } else {
        PING_5M_MILLIS
    };
    (span_millis / every) as f64 * call.usage.context() as f64 * call.rates().cache_read
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    const HOUR: i64 = 3_600_000;

    fn call(at_millis: i64, read: u64, write: u64, starts_turn: bool) -> RecordedCall {
        RecordedCall {
            at_millis,
            model: "claude-opus-5-5".to_owned(),
            usage: Usage {
                cache_read: read,
                cache_write_1h: write,
                output: 100,
                ..Usage::default()
            },
            starts_turn,
        }
    }

    fn timeline(calls: Vec<RecordedCall>) -> Timeline {
        Timeline {
            session: "s".to_owned(),
            subagent: false,
            calls,
            tool_outputs: Vec::new(),
        }
    }

    fn growing(calls: usize, step: u64) -> Timeline {
        let mut built = vec![call(0, 0, 20_000, true)];
        for index in 1..calls {
            let read = 20_000 + (index as u64 - 1) * step;
            built.push(call(index as i64 * 1000, read, step, true));
        }
        timeline(built)
    }

    #[test]
    fn with_no_change_the_simulation_reproduces_what_was_recorded() {
        let timeline = growing(40, 15_000);
        let recorded: f64 = timeline.calls.iter().map(RecordedCall::cost).sum();
        let outcome = simulate_restarts(std::slice::from_ref(&timeline), None);
        assert!((outcome.cost_usd - recorded).abs() < 1e-9, "{outcome:?}");
        assert_eq!(outcome.restarts, 0);
    }

    #[test]
    fn a_threshold_never_reached_changes_nothing() {
        let timeline = growing(5, 1000);
        let recorded: f64 = timeline.calls.iter().map(RecordedCall::cost).sum();
        let policy = RestartPolicy {
            above_tokens: 10_000_000,
            restored_tokens: 0,
        };
        let outcome = simulate_restarts(std::slice::from_ref(&timeline), Some(policy));
        assert!((outcome.cost_usd - recorded).abs() < 1e-9);
        assert_eq!(outcome.restarts, 0);
    }

    #[test]
    fn restarts_save_on_long_sessions_and_restoring_more_saves_less() {
        let timeline = growing(100, 15_000);
        let recorded = simulate_restarts(std::slice::from_ref(&timeline), None).cost_usd;
        let with = |restored_tokens| {
            simulate_restarts(
                std::slice::from_ref(&timeline),
                Some(RestartPolicy {
                    above_tokens: 300_000,
                    restored_tokens,
                }),
            )
        };
        let light = with(10_000);
        let heavy = with(150_000);
        assert!(light.restarts > 0);
        assert!(light.cost_usd < recorded, "{light:?} {recorded}");
        assert!(light.cost_usd < heavy.cost_usd, "{light:?} {heavy:?}");
    }

    #[test]
    fn restarting_a_short_session_costs_more_than_it_saves() {
        // Rewriting 110K tokens to save a few cache reads of slightly more is a bad trade.
        let timeline = timeline(vec![
            call(0, 0, 20_000, true),
            call(1000, 20_000, 100_000, false),
            call(2000, 120_000, 1000, true),
            call(3000, 121_000, 1000, false),
        ]);
        let recorded = simulate_restarts(std::slice::from_ref(&timeline), None).cost_usd;
        let policy = RestartPolicy {
            above_tokens: 100_000,
            restored_tokens: 90_000,
        };
        let outcome = simulate_restarts(std::slice::from_ref(&timeline), Some(policy));
        assert_eq!(outcome.restarts, 1);
        assert!(outcome.cost_usd > recorded, "{outcome:?} {recorded}");
    }

    #[test]
    fn restarts_only_happen_at_a_prompt() {
        let mut timeline = growing(50, 15_000);
        for call in &mut timeline.calls {
            call.starts_turn = false;
        }
        let policy = RestartPolicy {
            above_tokens: 100_000,
            restored_tokens: 0,
        };
        assert_eq!(simulate_restarts(&[timeline], Some(policy)).restarts, 0);
    }

    #[test]
    fn an_idle_rewrite_is_counted_and_priced_against_a_read() {
        let timeline = timeline(vec![
            call(0, 0, 100_000, true),
            call(1000, 100_000, 1000, false),
            call(1000 + 2 * HOUR, 0, 101_000, true),
        ]);
        let report = spend(std::slice::from_ref(&timeline));
        assert_eq!(report.idle_rewrites, 1);
        assert_eq!(report.other_rewrites, 0);
        // 101K tokens at $8 per million to write and $0.20 to read.
        assert!((report.idle_rewrite_usd - 101_000.0 * 7.8 / 1e6).abs() < 1e-9);
    }

    #[test]
    fn a_rewrite_with_no_idle_gap_is_reported_separately() {
        let timeline = timeline(vec![
            call(0, 0, 100_000, true),
            call(1000, 0, 100_000, false),
        ]);
        let report = spend(&[timeline]);
        assert_eq!((report.idle_rewrites, report.other_rewrites), (0, 1));
    }

    #[test]
    fn compaction_is_not_a_lost_prefix() {
        let timeline = timeline(vec![
            call(0, 0, 300_000, true),
            call(2 * HOUR, 0, 40_000, true),
        ]);
        let report = spend(&[timeline]);
        assert_eq!((report.idle_rewrites, report.other_rewrites), (0, 0));
    }

    #[test]
    fn spend_adds_up_and_bands_by_context() {
        let timeline = growing(60, 15_000);
        let report = spend(&[timeline]);
        let parts = report.read_usd + report.write_usd + report.output_usd + report.input_usd;
        assert!((parts - report.total_usd).abs() < 1e-9);
        let banded: f64 = report.bands.iter().map(|band| band.cost_usd).sum();
        assert!((banded - report.total_usd).abs() < 1e-9);
        assert_eq!(report.bands.iter().map(|b| b.calls).sum::<usize>(), 60);
        assert!(report.bands[3].calls > 0);
    }

    #[test]
    fn tool_output_is_charged_until_compaction() {
        let mut timeline = timeline(vec![
            call(0, 0, 100_000, true),
            call(1000, 100_000, 100, false),
            call(2000, 100_100, 100, false),
            call(3000, 0, 10_000, true),
            call(4000, 10_000, 100, false),
        ]);
        timeline.tool_outputs.push(ToolOutput {
            first_call: 1,
            chars: 4000,
        });
        let cost = tool_output_cost(&timeline);
        // 1,000 tokens written once at $8 and read on the next call only.
        let expected = 1000.0 * (8.0 + 0.2) / 1e6;
        assert!((cost - expected).abs() < 1e-12, "{cost} {expected}");
    }

    #[test]
    fn keep_warm_prevents_gaps_inside_its_window_and_pays_for_the_rest() {
        let timeline = timeline(vec![
            call(0, 0, 100_000, true),
            call(1000, 0, 500, false),
            call(1000 + 2 * HOUR, 0, 100_000, true),
            call(2000 + 2 * HOUR, 100_000, 100, false),
            call(2000 + 10 * HOUR, 0, 100_000, true),
        ]);
        let inside = simulate_keep_warm(std::slice::from_ref(&timeline), 3 * HOUR);
        assert_eq!((inside.rewrites, inside.prevented), (2, 1));
        assert!(inside.saved_usd > 0.0);
        assert!(inside.refresh_usd > 0.0);
        assert!(inside.wasted_usd > 0.0);

        let none = simulate_keep_warm(std::slice::from_ref(&timeline), 0);
        assert_eq!(none.prevented, 0);
        assert!(none.net_usd().abs() < 1e-9, "{none:?}");
    }

    #[test]
    fn reads_calls_prompts_and_tool_output_from_a_transcript() {
        let dir = std::env::temp_dir().join(format!("ccc-whatif-{}", std::process::id()));
        let subagents = dir.join("project").join("subagents");
        fs::create_dir_all(&subagents).unwrap();
        let lines = [
            r#"{"type":"user","message":{"content":"hello"}}"#,
            r#"{"type":"assistant","requestId":"r1","sessionId":"s1","timestamp":"2026-10-30T12:00:00.000Z","message":{"model":"claude-opus-5-5","usage":{"input_tokens":1,"cache_creation_input_tokens":100,"cache_creation":{"ephemeral_1h_input_tokens":100},"output_tokens":1}}}"#,
            r#"{"type":"assistant","requestId":"r1","sessionId":"s1","timestamp":"2026-10-30T12:00:00.100Z","message":{"model":"claude-opus-5-5","usage":{"input_tokens":1,"cache_creation_input_tokens":100,"output_tokens":9}}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","content":"abcd"},{"type":"text","text":"a reminder"}]}}"#,
            r#"{"type":"assistant","requestId":"r2","sessionId":"s1","timestamp":"2026-10-30T12:00:05.000Z","message":{"model":"claude-opus-5-5","usage":{"input_tokens":1,"cache_read_input_tokens":100,"output_tokens":1}}}"#,
            r#"{"type":"assistant","requestId":"r3","timestamp":"2026-10-30T12:00:06.000Z","message":{"model":"<synthetic>","usage":{"input_tokens":1}}}"#,
            "not json",
        ];
        let mut file = File::create(dir.join("project").join("a.jsonl")).unwrap();
        writeln!(file, "{}", lines.join("\n")).unwrap();
        // A resume repeats r1 in another file, which mustn't count twice.
        let mut other = File::create(subagents.join("b.jsonl")).unwrap();
        writeln!(other, "{}", lines[1]).unwrap();
        writeln!(other, r#"{{"type":"assistant","requestId":"r9","sessionId":"s1","timestamp":"2026-10-30T12:00:07.000Z","message":{{"model":"claude-opus-5-5","usage":{{"input_tokens":1,"output_tokens":1}}}}}}"#).unwrap();

        let timelines = load_timelines(&dir).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(timelines.len(), 2);
        let main = &timelines[0];
        assert_eq!(main.calls.len(), 2);
        assert_eq!(main.session, "s1");
        assert!(!main.subagent);
        assert!(main.calls[0].starts_turn);
        assert!(!main.calls[1].starts_turn);
        assert_eq!(main.calls[0].usage.output, 9);
        assert_eq!(main.calls[0].usage.cache_write_1h, 100);
        assert_eq!(
            main.tool_outputs,
            [ToolOutput {
                first_call: 1,
                chars: 4
            }]
        );
        assert_eq!(timelines[1].calls.len(), 1);
        assert!(timelines[1].subagent);
    }
}

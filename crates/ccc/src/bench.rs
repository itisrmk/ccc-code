//! `ccc bench`: replay local transcripts and print the results.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use ccc_bench::{Scenario, append_only_violations, fidelity};
use ccc_pipeline::{Context, Pipeline};
use ccc_prune::{Limits, Prune};
use ccc_wire::{Request, text_of_tool_result};
use serde_json::Value;

type CompileFn<'a> = Box<dyn Fn(&Request) -> Request + 'a>;

pub(crate) fn run(transcripts: &Path) -> anyhow::Result<()> {
    let (sessions, summary) = ccc_bench::load_dir(transcripts)
        .with_context(|| format!("loading transcripts from {}", transcripts.display()))?;

    let tight = Limits {
        log_min_bytes: 1_000,
        listing_min_bytes: 1_000,
        listing_keep_lines: 20,
        generic_min_bytes: 2_000,
        generic_head_bytes: 1_000,
        generic_tail_bytes: 500,
    };
    let loose = Limits {
        log_min_bytes: 8_000,
        listing_min_bytes: 8_000,
        listing_keep_lines: 200,
        generic_min_bytes: 32_000,
        generic_head_bytes: 16_000,
        generic_tail_bytes: 8_000,
    };
    let variants = [
        ("prune loose", loose),
        ("prune default", Limits::default()),
        ("prune tight", tight),
    ];

    let mut out = std::io::stdout();
    let compilers: Vec<(&str, Compiler)> = variants
        .into_iter()
        .map(|(name, limits)| (name, Compiler::new(limits)))
        .collect();
    let clearers = [
        (
            "clear old, 100K batches",
            ServerClearing::new(CLEAR_BATCH_CHARS),
        ),
        (
            "clear old, 300K batches",
            ServerClearing::new(3 * CLEAR_BATCH_CHARS),
        ),
    ];
    let mut scenarios: Vec<(&str, CompileFn<'_>)> = Vec::new();
    for (name, compiler) in &compilers {
        scenarios.push((
            *name,
            Box::new(|request: &Request| compiler.compile(request)),
        ));
    }
    for (name, clearing) in &clearers {
        scenarios.push((
            *name,
            Box::new(|request: &Request| clearing.compile(request)),
        ));
    }
    let scenarios: Vec<Scenario<'_>> = scenarios
        .iter()
        .map(|(name, compile)| Scenario {
            name,
            compile: compile.as_ref(),
        })
        .collect();

    let report = ccc_bench::replay(&sessions, summary, &scenarios);
    writeln!(out, "{report}").context("writing report")?;

    writeln!(out, "\nSafety checks").context("writing report")?;
    for (name, compiler) in &compilers {
        let exact = |request: &Request| compiler.compile_uncached(request);
        let violations = append_only_violations(&sessions, &exact);
        let kept = fidelity(&sessions, &exact);
        writeln!(
            out,
            "  {name:<16} append-only violations {violations}; {kept}"
        )
        .context("writing report")?;
    }
    Ok(())
}

/// Characters of old tool output that make up roughly 100K tokens at the fitted rate.
const CLEAR_BATCH_CHARS: usize = 290_000;
/// Newest messages left alone when clearing: about 25 tool rounds.
const CLEAR_KEEP_MESSAGES: usize = 50;

/// Simulates the server clearing old tool results (`clear_tool_uses_20250919`) in batches.
///
/// Optimistic by construction: it counts what clearing removes and the cache misses it
/// causes, but not the extra calls a model makes to re-read what it lost.
struct ServerClearing {
    batch_chars: usize,
    /// Per conversation, keyed by its first message: how far clearing has reached.
    reached: RefCell<HashMap<*const Value, usize>>,
    cleared: RefCell<HashMap<*const Value, Arc<Value>>>,
}

impl ServerClearing {
    fn new(batch_chars: usize) -> Self {
        Self {
            batch_chars,
            reached: RefCell::default(),
            cleared: RefCell::default(),
        }
    }

    fn compile(&self, request: &Request) -> Request {
        let messages = request.messages();
        let Some(first) = messages.first() else {
            return request.clone();
        };
        let mut reached = self.reached.borrow_mut();
        let reached = reached.entry(Arc::as_ptr(first)).or_insert(0);
        let eligible_end = messages.len().saturating_sub(CLEAR_KEEP_MESSAGES);
        if *reached < eligible_end {
            let clearable: usize = messages[*reached..eligible_end]
                .iter()
                .map(|message| tool_result_chars(message))
                .sum();
            if clearable >= self.batch_chars {
                *reached = eligible_end;
            }
        }
        let mut compiled = request.clone();
        let mut cleared = self.cleared.borrow_mut();
        for (index, message) in messages.iter().enumerate().take(*reached) {
            let replacement = cleared
                .entry(Arc::as_ptr(message))
                .or_insert_with(|| Arc::new(with_results_cleared(message)));
            compiled.set_message(index, Arc::clone(replacement));
        }
        compiled
    }
}

fn tool_result_blocks(message: &Value) -> impl Iterator<Item = &Value> {
    message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
}

fn tool_result_chars(message: &Value) -> usize {
    tool_result_blocks(message)
        .filter_map(|block| block.get("content").and_then(text_of_tool_result))
        .map(|text| text.chars().count())
        .sum()
}

fn with_results_cleared(message: &Value) -> Value {
    let mut message = message.clone();
    if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                block["content"] = Value::String("[cleared]".to_owned());
            }
        }
    }
    message
}

/// Compiles recorded requests with the Prune stage.
struct Compiler {
    pipeline: Pipeline,
    context: Context,
    /// Shares compiled messages between calls, as replay would otherwise hold a fresh
    /// copy of every edited message for every call that re-sends it.
    interned: RefCell<HashMap<*const Value, Arc<Value>>>,
}

impl Compiler {
    fn new(limits: Limits) -> Self {
        Self {
            pipeline: Pipeline::new(vec![Arc::new(Prune::with_limits(limits))]),
            context: Context {
                protected_before: 0,
                enabled: BTreeSet::from(["prune".to_owned()]),
            },
            interned: RefCell::default(),
        }
    }

    /// Prune never blocks a request, so failure here would be a bug worth stopping for.
    fn compile_uncached(&self, request: &Request) -> Request {
        self.pipeline
            .compile_offline(request, &self.context)
            .expect("prune passes through on failure")
            .request
    }

    fn compile(&self, request: &Request) -> Request {
        let mut compiled = self.compile_uncached(request);
        let mut interned = self.interned.borrow_mut();
        for (index, original) in request.messages().iter().enumerate() {
            let edited = &compiled.messages()[index];
            if !Arc::ptr_eq(original, edited) {
                let shared = interned
                    .entry(Arc::as_ptr(original))
                    .or_insert_with(|| Arc::clone(edited));
                let shared = Arc::clone(shared);
                compiled.set_message(index, shared);
            }
        }
        compiled
    }
}

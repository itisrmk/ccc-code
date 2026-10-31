//! `ccc model-bench`: measures a local model on this machine, using real tool output and a
//! small hand-labelled set of requests. Nothing leaves the machine.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use ccc_slm::{
    BlobPruneConfig, Completion, CompletionRequest, Difficulty, Intent, Labels, LlamaConfig,
    LlamaModel, TextModel, apply_keep_ranges, classify_labels, parse_keep_ranges,
};
use ccc_wire::text_of_tool_result;
use serde_json::Value;

/// The spec's budget for classifying one turn.
const CLASSIFY_CAP: Duration = Duration::from_millis(80);
/// The spec's budget for pruning one blob.
const BLOB_CAP: Duration = Duration::from_millis(600);
/// How long to let a call run here, far past the budget, so its real cost is measured.
const MEASURING_DEADLINE: Duration = Duration::from_secs(120);
/// Calls to measure, once models are warm.
const BLOB_SAMPLES: usize = 8;

const LABELLED: &[(&str, Intent, Difficulty)] = &[
    (
        "where is the retry logic for uploads defined?",
        Intent::Explore,
        Difficulty::Easy,
    ),
    (
        "how does auth flow through the middleware stack in this repo?",
        Intent::Explore,
        Difficulty::Medium,
    ),
    (
        "find every place we read the DATABASE_URL env var",
        Intent::Explore,
        Difficulty::Easy,
    ),
    (
        "map out how the billing, invoicing and ledger services talk to each other",
        Intent::Explore,
        Difficulty::Hard,
    ),
    (
        "rename get_user to fetch_user everywhere",
        Intent::Edit,
        Difficulty::Easy,
    ),
    (
        "add a --verbose flag to the cli",
        Intent::Edit,
        Difficulty::Easy,
    ),
    (
        "add pagination to the orders endpoint and update its tests",
        Intent::Edit,
        Difficulty::Medium,
    ),
    (
        "migrate the whole data layer from raw SQL to the new query builder",
        Intent::Edit,
        Difficulty::Hard,
    ),
    ("fix the typo in the README", Intent::Edit, Difficulty::Easy),
    (
        "why does the checkout test fail on CI but pass locally?",
        Intent::Debug,
        Difficulty::Medium,
    ),
    (
        "this panics with index out of bounds at parser.rs:41, fix it",
        Intent::Debug,
        Difficulty::Easy,
    ),
    (
        "users report a memory leak after about 6 hours in production, track it down",
        Intent::Debug,
        Difficulty::Hard,
    ),
    (
        "the build is broken since yesterday, what happened?",
        Intent::Debug,
        Difficulty::Medium,
    ),
    (
        "there's an intermittent deadlock between the queue workers",
        Intent::Debug,
        Difficulty::Hard,
    ),
    (
        "what does this regex do?",
        Intent::Explain,
        Difficulty::Easy,
    ),
    (
        "explain how the borrow checker error above arises",
        Intent::Explain,
        Difficulty::Medium,
    ),
    (
        "explain the difference between these two caching strategies",
        Intent::Explain,
        Difficulty::Medium,
    ),
    (
        "what does the -D flag do here?",
        Intent::Explain,
        Difficulty::Easy,
    ),
    (
        "how should we split this monolith into services? give me a plan",
        Intent::Plan,
        Difficulty::Hard,
    ),
    (
        "plan out adding SSO before we write any code",
        Intent::Plan,
        Difficulty::Medium,
    ),
    (
        "what's the best order to tackle these five refactors?",
        Intent::Plan,
        Difficulty::Medium,
    ),
    (
        "propose an approach for zero-downtime schema migrations",
        Intent::Plan,
        Difficulty::Hard,
    ),
];

pub(crate) fn run(model_path: &Path, transcripts: Option<&Path>) -> anyhow::Result<()> {
    let mut out = std::io::stdout();
    let loading = Instant::now();
    let model =
        LlamaModel::load(model_path, LlamaConfig::default()).context("loading the model")?;
    writeln!(
        out,
        "loaded {} in {:?}",
        model_path.display(),
        loading.elapsed()
    )?;

    classifier_report(&model, &mut out)?;
    if let Some(dir) = transcripts {
        blob_report(&model, dir, &mut out)?;
    }
    Ok(())
}

fn classifier_report(model: &LlamaModel, out: &mut impl Write) -> anyhow::Result<()> {
    // The first call pays for warming the GPU, which a real session pays once.
    let _ = classify_labels(model, "warm up", None, MEASURING_DEADLINE);

    let mut elapsed = Vec::with_capacity(LABELLED.len());
    let (mut intents_right, mut difficulties_right, mut failures) = (0, 0, 0);
    for (prompt, intent, difficulty) in LABELLED {
        let started = Instant::now();
        let labels: Result<Labels, _> = classify_labels(model, prompt, None, MEASURING_DEADLINE);
        elapsed.push(started.elapsed());
        match labels {
            Ok(labels) => {
                intents_right += usize::from(labels.intent == *intent);
                difficulties_right += usize::from(labels.difficulty == *difficulty);
            }
            Err(_) => failures += 1,
        }
    }
    elapsed.sort();
    let total = LABELLED.len();
    let within = elapsed.iter().filter(|time| **time <= CLASSIFY_CAP).count();
    writeln!(out, "\nClassifier, {total} hand-labelled requests")?;
    writeln!(
        out,
        "  latency p50 {:?}, p95 {:?}, max {:?}",
        at(&elapsed, 0.5),
        at(&elapsed, 0.95),
        at(&elapsed, 1.0)
    )?;
    writeln!(
        out,
        "  within the {CLASSIFY_CAP:?} cap: {within} of {total}"
    )?;
    writeln!(
        out,
        "  intent correct {intents_right} of {total}, difficulty correct {difficulties_right} of {total}, failed {failures}"
    )?;
    Ok(())
}

fn at(sorted: &[Duration], fraction: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * fraction) as usize]
}

fn blob_report(model: &LlamaModel, dir: &Path, out: &mut impl Write) -> anyhow::Result<()> {
    let limits = BlobPruneConfig {
        decisions_dir: std::env::temp_dir(),
        min_bytes: 11_500,
        max_bytes: 22_000,
        time_cap: BLOB_CAP,
    };
    let (sessions, _) = ccc_bench::load_dir(dir).context("loading transcripts")?;
    let mut seen = std::collections::HashSet::new();
    let mut blobs: Vec<String> = Vec::new();
    let mut eligible = 0_usize;
    let mut results = 0_usize;
    for session in &sessions {
        let Some(last) = session.calls.last() else {
            continue;
        };
        for message in last.request.messages() {
            let blocks = message.get("content").and_then(Value::as_array);
            for block in blocks.into_iter().flatten() {
                if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let Some(text) = block.get("content").and_then(text_of_tool_result) else {
                    continue;
                };
                results += 1;
                if text.len() >= limits.min_bytes && text.len() <= limits.max_bytes {
                    eligible += 1;
                    if seen.insert(text.to_string()) {
                        blobs.push(text.into_owned());
                    }
                }
            }
        }
    }
    writeln!(out, "\nBlob pruning, real tool output")?;
    writeln!(
        out,
        "  eligible results: {eligible} of {results} in final requests"
    )?;
    let (mut total_bytes, mut kept_bytes) = (0, 0);
    let mut times = Vec::new();
    let mut usable = 0;
    let mut prompt_rate = Vec::new();
    for blob in blobs.iter().take(BLOB_SAMPLES) {
        let numbered: String = blob
            .lines()
            .enumerate()
            .map(|(index, line)| format!("{} {line}\n", index + 1))
            .collect();
        let prompt = format!(
            "Below is numbered tool output from a coding session. List the line ranges a \
             programmer would still need later: definitions, signatures, errors. Reply with \
             ranges only, like 1-20,45,60-72.\n\n{numbered}"
        );
        let started = Instant::now();
        let completion: Result<Completion, _> = model.complete(&CompletionRequest {
            prompt: &prompt,
            grammar: Some("root ::= range (\",\" range)*\nrange ::= number (\"-\" number)?\nnumber ::= [1-9] [0-9]*\n"),
            max_tokens: 320,
            deadline: MEASURING_DEADLINE,
        });
        times.push(started.elapsed());
        let Ok(completion) = completion else {
            continue;
        };
        if !completion.prompt_time.is_zero() {
            prompt_rate
                .push(f64::from(completion.prompt_tokens) / completion.prompt_time.as_secs_f64());
        }
        total_bytes += blob.len();
        let kept = parse_keep_ranges(&completion.text)
            .and_then(|ranges| apply_keep_ranges(blob, &ranges, 0.8));
        if let Some(kept) = kept {
            usable += 1;
            kept_bytes += kept.len();
        } else {
            kept_bytes += blob.len();
        }
    }
    if times.is_empty() {
        writeln!(out, "  no eligible results to measure")?;
        return Ok(());
    }
    times.sort();
    let mean_rate = prompt_rate.iter().sum::<f64>() / prompt_rate.len().max(1) as f64;
    writeln!(
        out,
        "  measured {} results, p50 {:?}, max {:?}, cap {BLOB_CAP:?}",
        times.len(),
        at(&times, 0.5),
        at(&times, 1.0)
    )?;
    writeln!(out, "  prompt speed {mean_rate:.0} tokens/s")?;
    writeln!(
        out,
        "  usable answers {usable} of {}, bytes kept {kept_bytes} of {total_bytes}",
        times.len()
    )?;
    Ok(())
}

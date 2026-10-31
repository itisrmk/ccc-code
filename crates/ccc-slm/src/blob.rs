//! Shrinks oversized tool output with a model that may only choose which lines to keep.
//!
//! Output is rebuilt from the original lines, so the result is verbatim by construction:
//! the model can drop code but can't paraphrase it.
//!
//! A model isn't a pure function, and a result can't change once it has been sent. So only
//! the newest message is ever sent to the model, and the outcome is saved before the request
//! goes out, including "leave it alone" when the model is too slow. Older results replay
//! their saved decision and are never computed fresh.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ccc_pipeline::{Context, Decisions, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::{Request, text_of_tool_result};
use serde_json::{Value, json};

use crate::model::{CompletionRequest, ModelError, TextModel};

const KEEP_GRAMMAR: &str = r#"root ::= range ("," range)*
range ::= number ("-" number)?
number ::= [1-9] [0-9]*
"#;
/// Enough tokens for a long list of line ranges.
const MAX_RANGE_TOKENS: u32 = 320;
/// Extra time the pipeline allows, so this stage finishes writing down a decision itself.
const SAVE_MARGIN: Duration = Duration::from_millis(250);
/// A keep list covering more than this share isn't worth changing the output for.
const MAX_KEPT_SHARE: f64 = 0.8;

/// Lines in the prompt used to measure reading speed.
const PROBE_LINES: usize = 40;
/// How long measuring may take.
const PROBE_DEADLINE: Duration = Duration::from_secs(30);
/// Bytes of tool output per token, fitted from recorded usage on real transcripts.
const BYTES_PER_TOKEN: f64 = 2.89;

/// Settings for [`BlobPrune`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobPruneConfig {
    /// Where decisions are saved. They must outlive this process.
    pub decisions_dir: PathBuf,
    /// Results shorter than this many bytes are left alone. About 4K tokens by default.
    pub min_bytes: usize,
    /// Results longer than this are skipped, because the prompt wouldn't fit in the model.
    pub max_bytes: usize,
    /// How long the model may take on the live request path.
    pub time_cap: Duration,
}

/// The stage. See the module docs.
#[derive(Debug)]
pub struct BlobPrune {
    model: Arc<dyn TextModel>,
    decisions: Decisions,
    config: BlobPruneConfig,
}

impl BlobPrune {
    /// A stage using `model`.
    #[must_use]
    pub fn new(model: Arc<dyn TextModel>, config: BlobPruneConfig) -> Self {
        let decisions = Decisions::new(config.decisions_dir.clone());
        Self {
            model,
            decisions,
            config,
        }
    }

    /// Asks the model what to keep. Any problem, including slowness, means keep everything.
    fn decide(&self, text: &str) -> Option<String> {
        let started = Instant::now();
        let numbered: String = text
            .lines()
            .enumerate()
            .map(|(index, line)| format!("{} {line}\n", index + 1))
            .collect();
        let prompt = format!(
            "Below is numbered tool output from a coding session. List the line ranges a \
             programmer would still need later: definitions, signatures, errors. Reply with \
             ranges only, like 1-20,45,60-72.\n\n{numbered}"
        );
        let completion = self
            .model
            .complete(&CompletionRequest {
                prompt: &prompt,
                grammar: Some(KEEP_GRAMMAR),
                max_tokens: MAX_RANGE_TOKENS,
                deadline: self.config.time_cap,
            })
            .inspect_err(|err| tracing::debug!(error = %err, "model step skipped"))
            .ok()?;
        tracing::debug!(
            answer = %completion.text,
            prompt_tokens = completion.prompt_tokens,
            elapsed = ?started.elapsed(),
            "model answered"
        );
        if started.elapsed() > self.config.time_cap {
            return None;
        }
        let ranges = parse_keep_ranges(&completion.text)?;
        apply_keep_ranges(text, &ranges, MAX_KEPT_SHARE)
    }
}

impl Stage for BlobPrune {
    fn name(&self) -> &'static str {
        "prune-model"
    }

    fn time_cap(&self) -> Duration {
        self.config.time_cap + SAVE_MARGIN
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let newest_user = request
            .messages()
            .iter()
            .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"));
        let mut edits = Vec::new();
        for (index, message) in request.messages().iter().enumerate() {
            if index < context.protected_before {
                continue;
            }
            let Some(blocks) = message.get("content").and_then(Value::as_array) else {
                continue;
            };
            for (position, block) in blocks.iter().enumerate() {
                if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let Some(id) = block.get("tool_use_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(text) = block.get("content").and_then(text_of_tool_result) else {
                    continue;
                };
                if text.len() < self.config.min_bytes || text.len() > self.config.max_bytes {
                    continue;
                }
                let identity = [id, text.as_ref()];
                let kept = match self.decisions.get(&identity) {
                    Some(saved) => saved,
                    None if Some(index) == newest_user => {
                        let decided = self.decide(&text);
                        // A decision that can't be saved couldn't be repeated, so it's void.
                        if self.decisions.save(&identity, decided.as_deref()) {
                            decided
                        } else {
                            None
                        }
                    }
                    // Sent before any decision existed, so it was sent whole and stays whole.
                    None => None,
                };
                if let Some(kept) = kept {
                    edits.push(Edit {
                        target: Target::Message(index),
                        pointer: format!("/content/{position}/content"),
                        value: json!(kept),
                        rule: "model",
                    });
                }
            }
        }
        Ok(edits)
    }
}

/// Whether `model` reads fast enough to get through `min_bytes` of output within `cap`.
/// Below this floor the stage would add its full cap to every large result and change
/// nothing, so it shouldn't run at all.
///
/// # Errors
///
/// Returns [`ModelError`] if the model can't complete the measuring prompt.
pub fn reads_fast_enough(
    model: &dyn TextModel,
    min_bytes: usize,
    cap: Duration,
) -> Result<bool, ModelError> {
    // Different lines each time, so a cached prompt prefix can't flatter the speed.
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let sample: String = (0..PROBE_LINES)
        .map(|line| format!("{unique} fn measured_line_{line}() {{ value + {line} }}\n"))
        .collect();
    let completion = model.complete(&CompletionRequest {
        prompt: &sample,
        grammar: None,
        max_tokens: 1,
        deadline: PROBE_DEADLINE,
    })?;
    let needed = min_bytes as f64 / BYTES_PER_TOKEN;
    let seconds = completion.prompt_time.as_secs_f64().max(f64::EPSILON);
    let tokens_per_second = f64::from(completion.prompt_tokens) / seconds;
    tracing::info!(tokens_per_second, needed, "measured model speed");
    Ok(needed / tokens_per_second <= cap.as_secs_f64())
}

/// Reads a keep list such as `1-20,45`. Lines are numbered from one. An answer can stop at
/// the token limit partway through a range, so a dangling end is ignored: that only keeps
/// fewer lines. Returns `None` if any other part is unreadable, since a partly understood
/// list isn't safe to act on.
#[must_use]
pub fn parse_keep_ranges(text: &str) -> Option<Vec<(usize, usize)>> {
    let text = text.trim();
    let complete = match text.rsplit_once(',') {
        Some((before, last)) if last.is_empty() || last.ends_with('-') => before,
        _ => text,
    };
    complete
        .split(',')
        .map(|part| {
            let (start, end) = part.split_once('-').unwrap_or((part, part));
            let start: usize = start.trim().parse().ok()?;
            let end: usize = end.trim().parse().ok()?;
            (start >= 1 && start <= end).then_some((start, end))
        })
        .collect()
}

/// Rebuilds `text` from the lines `ranges` select, marking each gap. Returns `None` if that
/// would keep nothing, or more than `max_kept_share` of the lines.
#[must_use]
pub fn apply_keep_ranges(
    text: &str,
    ranges: &[(usize, usize)],
    max_kept_share: f64,
) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let selected = |number: usize| {
        ranges
            .iter()
            .any(|(start, end)| (*start..=*end).contains(&number))
    };
    let kept = (1..=lines.len()).filter(|number| selected(*number)).count();
    if kept == 0 || kept as f64 > max_kept_share * lines.len() as f64 {
        return None;
    }
    let mut out = String::new();
    let mut gap = 0;
    for (index, line) in lines.iter().enumerate() {
        if selected(index + 1) {
            if gap > 0 {
                out.push_str(&format!("[ccc: {gap} lines omitted]\n"));
                gap = 0;
            }
            out.push_str(line);
            out.push('\n');
        } else {
            gap += 1;
        }
    }
    if gap > 0 {
        out.push_str(&format!("[ccc: {gap} lines omitted]\n"));
    }
    out.push_str(&format!(
        "[ccc: kept {kept} of {} lines verbatim. Read the source again to see the rest.]\n",
        lines.len()
    ));
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::tests::Fixed;
    use crate::model::Completion;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn long_output() -> String {
        (1..=400)
            .map(|i| format!("line number {i} of the output\n"))
            .collect()
    }

    fn config(dir: PathBuf) -> BlobPruneConfig {
        BlobPruneConfig {
            decisions_dir: dir,
            min_bytes: 1_000,
            max_bytes: 100_000,
            time_cap: Duration::from_secs(5),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ccc-blob-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn request(results: &[(&str, &str)]) -> Request {
        let mut messages = vec![json!({ "role": "user", "content": "go" })];
        for (id, text) in results {
            messages.push(json!({ "role": "assistant", "content": [
                { "type": "tool_use", "id": id, "name": "Bash", "input": {} }
            ] }));
            messages.push(json!({ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": id, "content": text }
            ] }));
        }
        Request::parse(json!({ "messages": messages }).to_string().as_bytes()).unwrap()
    }

    #[test]
    fn parses_keep_lists_strictly() {
        assert_eq!(parse_keep_ranges("1-20,45"), Some(vec![(1, 20), (45, 45)]));
        assert_eq!(parse_keep_ranges("0-3"), None);
        assert_eq!(parse_keep_ranges("9-3"), None);
        assert_eq!(parse_keep_ranges("1-20,nope"), None);
        assert_eq!(
            parse_keep_ranges("1-20,45,279-"),
            Some(vec![(1, 20), (45, 45)]),
            "cut off"
        );
        assert_eq!(parse_keep_ranges("1-20,"), Some(vec![(1, 20)]), "cut off");
        assert_eq!(parse_keep_ranges(""), None);
    }

    #[test]
    fn output_is_verbatim_lines_in_order_plus_markers() {
        let text = long_output();
        let kept =
            apply_keep_ranges(&text, &[(1, 5), (200, 210), (390, 9999)], MAX_KEPT_SHARE).unwrap();
        let original: Vec<&str> = text.lines().collect();
        let mut remaining = original.iter();
        for line in kept.lines().filter(|line| !line.starts_with("[ccc:")) {
            assert!(
                remaining.any(|o| o == &line),
                "{line:?} isn't an in-order input line"
            );
        }
        assert!(kept.contains("line number 200 "));
        assert!(!kept.contains("line number 100 "));
        assert!(kept.contains("kept 27 of 400 lines"), "{kept}");
    }

    #[test]
    fn keeping_nothing_or_nearly_everything_means_no_change() {
        let text = long_output();
        assert_eq!(
            apply_keep_ranges(&text, &[(9000, 9001)], MAX_KEPT_SHARE),
            None
        );
        assert_eq!(apply_keep_ranges(&text, &[(1, 399)], MAX_KEPT_SHARE), None);
    }

    /// The property that makes a non-deterministic step safe: however the model behaves on
    /// later calls, what a message compiled to the first time never changes.
    #[test]
    fn a_decision_never_changes_even_if_the_model_would_now_answer_differently() {
        let dir = temp_dir("stable");
        let text = long_output();

        let first = BlobPrune::new(Arc::new(Fixed(Ok("1-10".to_owned()))), config(dir.clone()));
        let one = first
            .plan(&request(&[("a", &text)]), &Context::default())
            .unwrap();
        assert_eq!(one.len(), 1);

        // A new process, and a model that would now choose differently.
        let second = BlobPrune::new(Arc::new(Fixed(Ok("50-60".to_owned()))), config(dir.clone()));
        let again = second
            .plan(
                &request(&[("a", &text), ("b", "short")]),
                &Context::default(),
            )
            .unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].value, one[0].value);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_slow_model_is_recorded_so_the_result_stays_whole_for_good() {
        let dir = temp_dir("slow");
        let text = long_output();

        let slow = BlobPrune::new(
            Arc::new(Fixed(Err(ModelError::DeadlineExceeded))),
            config(dir.clone()),
        );
        assert!(
            slow.plan(&request(&[("a", &text)]), &Context::default())
                .unwrap()
                .is_empty()
        );

        let fast = BlobPrune::new(Arc::new(Fixed(Ok("1-10".to_owned()))), config(dir.clone()));
        let later = fast
            .plan(&request(&[("a", &text)]), &Context::default())
            .unwrap();
        assert!(
            later.is_empty(),
            "it went out whole once, so it must stay whole"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[derive(Debug, Default)]
    struct Counting(AtomicUsize);

    impl TextModel for Counting {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Completion {
                text: "1-10".to_owned(),
                ..Completion::default()
            })
        }
    }

    #[test]
    fn only_the_newest_message_ever_reaches_the_model() {
        let dir = temp_dir("newest");
        let text = long_output();
        let other = format!("{text}different");
        let model = Arc::new(Counting::default());
        let stage = BlobPrune::new(model.clone(), config(dir.clone()));

        // Both are long and undecided, but the first was already sent.
        let edits = stage
            .plan(
                &request(&[("a", &text), ("b", &other)]),
                &Context::default(),
            )
            .unwrap();

        assert_eq!(model.0.load(Ordering::Relaxed), 1);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].target, Target::Message(4));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn protected_history_and_out_of_range_sizes_are_skipped() {
        let dir = temp_dir("skip");
        let text = long_output();
        let model = Arc::new(Counting::default());
        let stage = BlobPrune::new(model.clone(), config(dir.clone()));

        let protected = Context {
            protected_before: 99,
            ..Context::default()
        };
        assert!(
            stage
                .plan(&request(&[("a", &text)]), &protected)
                .unwrap()
                .is_empty()
        );
        assert!(
            stage
                .plan(&request(&[("a", "tiny")]), &Context::default())
                .unwrap()
                .is_empty()
        );
        let huge = "x\n".repeat(100_000);
        assert!(
            stage
                .plan(&request(&[("a", &huge)]), &Context::default())
                .unwrap()
                .is_empty()
        );
        assert_eq!(model.0.load(Ordering::Relaxed), 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_speed_floor_compares_reading_speed_to_the_cap() {
        #[derive(Debug)]
        struct Speed(u32);
        impl TextModel for Speed {
            fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
                Ok(Completion {
                    prompt_tokens: self.0,
                    prompt_time: Duration::from_secs(1),
                    ..Completion::default()
                })
            }
        }
        let cap = Duration::from_millis(600);
        assert!(
            !reads_fast_enough(&Speed(373), 11_500, cap).unwrap(),
            "as measured on an M2 Pro"
        );
        assert!(reads_fast_enough(&Speed(10_000), 11_500, cap).unwrap());
        assert!(reads_fast_enough(&Speed(373), 11_500, Duration::from_secs(60)).unwrap());
    }

    #[test]
    fn an_unusable_answer_leaves_the_output_whole() {
        let dir = temp_dir("garbage");
        let stage = BlobPrune::new(
            Arc::new(Fixed(Ok("not ranges".to_owned()))),
            config(dir.clone()),
        );
        let text = long_output();
        assert!(
            stage
                .plan(&request(&[("a", &text)]), &Context::default())
                .unwrap()
                .is_empty()
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

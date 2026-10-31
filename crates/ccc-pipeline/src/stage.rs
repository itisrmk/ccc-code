use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ccc_wire::Request;
use serde_json::Value;

/// What an [`Edit`] changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The message at this index.
    Message(usize),
    /// The top-level `system` prompt.
    System,
    /// The top-level `model`, which isn't part of the history thinking blocks bind to.
    Model,
}

/// One replacement inside one message or the system prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Edit {
    /// What to change.
    pub target: Target,
    /// RFC 6901 pointer to the value within the target.
    pub pointer: String,
    /// The new value.
    pub value: Value,
    /// The rule that produced this edit, for tracing.
    pub rule: &'static str,
}

/// What a stage may do when it fails, chosen per stage because the stakes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnFailure {
    /// Send the request without this stage's edits.
    PassThrough,
    /// Refuse to send the request. For stages whose absence would be unsafe.
    Block,
}

/// A stage could not produce edits.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StageError(pub String);

/// Per-conversation facts stages and the pipeline must respect.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Context {
    /// Messages before this index were sent before this conversation came under the
    /// compiler. They must stay byte-identical.
    pub protected_before: usize,
    /// Stages that run for this conversation. Pinned when the conversation is first
    /// seen, since switching a stage mid-conversation would rewrite history.
    pub enabled: BTreeSet<String>,
    /// What this session has cost so far, in millionths of a dollar. Unlike the rest this
    /// changes between calls, so it may only drive decisions outside the bound history.
    pub spent_micros: u64,
}

/// One step of the compiler.
pub trait Stage: Send + Sync + Debug {
    /// A short stable name, used for flags and tracing.
    fn name(&self) -> &'static str;

    /// How long this stage may run on a live request before it is abandoned.
    fn time_cap(&self) -> Duration;

    /// What happens if this stage fails or exceeds its time cap.
    fn on_failure(&self) -> OnFailure;

    /// Computes the edits for `request`.
    ///
    /// The result must be a pure function of `request` and `context`, and must leave
    /// any message's compiled form unchanged as later messages are appended.
    ///
    /// # Errors
    ///
    /// Returns [`StageError`] if the stage can't do its job on this request.
    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError>;
}

/// What happened when a stage ran.
#[derive(Debug, Clone, PartialEq)]
pub enum StageOutcome {
    /// The stage isn't enabled for this conversation.
    Skipped,
    /// The stage ran and these edits were applied.
    Applied(Vec<Edit>),
    /// The stage failed, and its failure policy allowed sending without it.
    Failed(String),
    /// The stage ran past its time cap, and its failure policy allowed sending without it.
    TimedOut,
}

/// One stage's record in a compile.
#[derive(Debug, Clone, PartialEq)]
pub struct StageTrace {
    /// The stage's name.
    pub stage: &'static str,
    /// How long it took.
    pub elapsed: Duration,
    /// What happened.
    pub outcome: StageOutcome,
}

/// The result of compiling a request.
#[derive(Debug, Clone, PartialEq)]
pub struct Compiled {
    /// The request to send.
    pub request: Request,
    /// One entry per stage, in order.
    pub trace: Vec<StageTrace>,
}

impl Compiled {
    /// Whether any stage changed the request.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.trace.iter().any(
            |entry| matches!(&entry.outcome, StageOutcome::Applied(edits) if !edits.is_empty()),
        )
    }
}

/// A stage whose failure policy forbids sending failed or timed out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("stage {stage} could not run, and the request was not sent: {reason}")]
pub struct CompileError {
    /// The stage that stopped the request.
    pub stage: &'static str,
    /// What went wrong.
    pub reason: String,
}

/// An ordered list of stages.
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    stages: Vec<Arc<dyn Stage>>,
}

impl Pipeline {
    /// Builds a pipeline that runs `stages` in order.
    #[must_use]
    pub fn new(stages: Vec<Arc<dyn Stage>>) -> Self {
        Self { stages }
    }

    /// Whether any of the named stages refuses to let requests through unexamined.
    #[must_use]
    pub fn requires_inspection(&self, enabled: &BTreeSet<String>) -> bool {
        self.stages
            .iter()
            .any(|stage| enabled.contains(stage.name()) && stage.on_failure() == OnFailure::Block)
    }

    /// Names of all stages, in order.
    pub fn stage_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.stages.iter().map(|stage| stage.name())
    }

    /// Compiles a live request, abandoning any stage that exceeds its time cap.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if a stage whose policy is [`OnFailure::Block`] fails or
    /// times out. A failure of any other stage only omits that stage's edits.
    pub async fn compile(
        &self,
        request: Request,
        context: Context,
    ) -> Result<Compiled, CompileError> {
        let context = Arc::new(context);
        let mut current = Arc::new(request);
        let mut trace = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let started = Instant::now();
            let name = stage.name();
            if !context.enabled.contains(name) {
                trace.push(skipped(name));
                continue;
            }
            let task = {
                let (stage, request, context) = (
                    Arc::clone(stage),
                    Arc::clone(&current),
                    Arc::clone(&context),
                );
                tokio::task::spawn_blocking(move || stage.plan(&request, &context))
            };
            let result = match tokio::time::timeout(stage.time_cap(), task).await {
                Ok(Ok(planned)) => planned.map_err(|err| Abandoned::Failed(err.0)),
                Ok(Err(join)) => Err(Abandoned::Failed(format!("stage panicked: {join}"))),
                // The blocking task can't be cancelled, but its result is simply dropped.
                Err(_) => Err(Abandoned::TimedOut),
            };
            let outcome = settle(
                stage.as_ref(),
                result,
                &context,
                Arc::make_mut(&mut current),
            )?;
            trace.push(StageTrace {
                stage: name,
                elapsed: started.elapsed(),
                outcome,
            });
        }
        Ok(Compiled {
            request: Arc::unwrap_or_clone(current),
            trace,
        })
    }

    /// Compiles without time caps, for offline replay where latency doesn't matter.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if a stage whose policy is [`OnFailure::Block`] fails.
    pub fn compile_offline(
        &self,
        request: &Request,
        context: &Context,
    ) -> Result<Compiled, CompileError> {
        let mut current = request.clone();
        let mut trace = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let started = Instant::now();
            let name = stage.name();
            if !context.enabled.contains(name) {
                trace.push(skipped(name));
                continue;
            }
            let result = stage
                .plan(&current, context)
                .map_err(|err| Abandoned::Failed(err.0));
            let outcome = settle(stage.as_ref(), result, context, &mut current)?;
            trace.push(StageTrace {
                stage: name,
                elapsed: started.elapsed(),
                outcome,
            });
        }
        Ok(Compiled {
            request: current,
            trace,
        })
    }
}

enum Abandoned {
    Failed(String),
    TimedOut,
}

fn skipped(stage: &'static str) -> StageTrace {
    StageTrace {
        stage,
        elapsed: Duration::ZERO,
        outcome: StageOutcome::Skipped,
    }
}

fn settle(
    stage: &dyn Stage,
    result: Result<Vec<Edit>, Abandoned>,
    context: &Context,
    request: &mut Request,
) -> Result<StageOutcome, CompileError> {
    match (result, stage.on_failure()) {
        (Ok(edits), _) => Ok(StageOutcome::Applied(apply(stage, edits, context, request))),
        (Err(Abandoned::Failed(reason)), OnFailure::PassThrough) => {
            tracing::warn!(stage = stage.name(), %reason, "stage failed, passing through");
            Ok(StageOutcome::Failed(reason))
        }
        (Err(Abandoned::TimedOut), OnFailure::PassThrough) => {
            tracing::warn!(stage = stage.name(), "stage timed out, passing through");
            Ok(StageOutcome::TimedOut)
        }
        (Err(Abandoned::Failed(reason)), OnFailure::Block) => Err(CompileError {
            stage: stage.name(),
            reason,
        }),
        (Err(Abandoned::TimedOut), OnFailure::Block) => Err(CompileError {
            stage: stage.name(),
            reason: format!("exceeded its {:?} time cap", stage.time_cap()),
        }),
    }
}

/// Applies the edits the invariants allow and returns those that took effect.
fn apply(
    stage: &dyn Stage,
    edits: Vec<Edit>,
    context: &Context,
    request: &mut Request,
) -> Vec<Edit> {
    let mut applied = Vec::with_capacity(edits.len());
    for edit in edits {
        if is_protected(edit.target, context) {
            tracing::error!(stage = stage.name(), target = ?edit.target, "refused edit to protected history");
            continue;
        }
        if targets_thinking(request, &edit) {
            tracing::error!(stage = stage.name(), pointer = %edit.pointer, "refused edit to a signed thinking block");
            continue;
        }
        let found = match edit.target {
            Target::Message(index) => {
                request.set_in_message(index, &edit.pointer, edit.value.clone())
            }
            Target::System => request.set_in_field("system", &edit.pointer, edit.value.clone()),
            Target::Model => request.set_in_field("model", &edit.pointer, edit.value.clone()),
        };
        if found {
            applied.push(edit);
        } else {
            tracing::warn!(stage = stage.name(), pointer = %edit.pointer, "edit target not found");
        }
    }
    applied
}

/// The system prompt comes before all history, so it is protected whenever any history is.
fn is_protected(target: Target, context: &Context) -> bool {
    match target {
        Target::Message(index) => index < context.protected_before,
        Target::System => context.protected_before > 0,
        Target::Model => false,
    }
}

/// Whether `edit` lands in or on a `thinking` or `redacted_thinking` block.
fn targets_thinking(request: &Request, edit: &Edit) -> bool {
    let Target::Message(index) = edit.target else {
        return false;
    };
    let Some(message) = request.messages().get(index) else {
        return false;
    };
    // A block pointer is /content/<n>, optionally followed by a path within the block.
    let mut segments = edit.pointer.split('/').skip(1);
    let (Some("content"), Some(block)) = (segments.next(), segments.next()) else {
        return false;
    };
    message
        .pointer(&format!("/content/{block}/type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "thinking" | "redacted_thinking"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug)]
    enum Behavior {
        Edit(Vec<Edit>),
        Fail,
        Panic,
        Sleep(Duration),
    }

    #[derive(Debug)]
    struct Fake {
        name: &'static str,
        policy: OnFailure,
        behavior: Behavior,
    }

    impl Stage for Fake {
        fn name(&self) -> &'static str {
            self.name
        }

        fn time_cap(&self) -> Duration {
            Duration::from_millis(50)
        }

        fn on_failure(&self) -> OnFailure {
            self.policy
        }

        fn plan(&self, _: &Request, _: &Context) -> Result<Vec<Edit>, StageError> {
            match &self.behavior {
                Behavior::Edit(edits) => Ok(edits.clone()),
                Behavior::Fail => Err(StageError("boom".to_owned())),
                Behavior::Panic => panic!("stage bug"),
                Behavior::Sleep(duration) => {
                    std::thread::sleep(*duration);
                    Ok(vec![content_edit(0, "too late")])
                }
            }
        }
    }

    fn content_edit(message: usize, text: &str) -> Edit {
        Edit {
            target: Target::Message(message),
            pointer: "/content".to_owned(),
            value: json!(text),
            rule: "test",
        }
    }

    fn request() -> Request {
        Request::parse(
            br#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":[
                {"type":"thinking","thinking":"","signature":"s"},{"type":"text","text":"t"}]}]}"#,
        )
        .unwrap()
    }

    fn pipeline(stages: Vec<Fake>) -> (Pipeline, Context) {
        let context = Context {
            protected_before: 0,
            enabled: stages.iter().map(|s| s.name.to_owned()).collect(),
            spent_micros: 0,
        };
        let stages = stages
            .into_iter()
            .map(|stage| Arc::new(stage) as Arc<dyn Stage>)
            .collect();
        (Pipeline::new(stages), context)
    }

    fn fake(name: &'static str, policy: OnFailure, behavior: Behavior) -> Fake {
        Fake {
            name,
            policy,
            behavior,
        }
    }

    #[tokio::test]
    async fn applies_edits_in_order_with_later_stages_seeing_earlier_ones() {
        let (pipeline, context) = pipeline(vec![
            fake(
                "one",
                OnFailure::PassThrough,
                Behavior::Edit(vec![content_edit(0, "first")]),
            ),
            fake(
                "two",
                OnFailure::PassThrough,
                Behavior::Edit(vec![content_edit(0, "second")]),
            ),
        ]);
        let compiled = pipeline.compile(request(), context).await.unwrap();
        assert_eq!(compiled.request.messages()[0]["content"], "second");
        assert!(compiled.changed());
    }

    #[tokio::test]
    async fn pass_through_stages_that_fail_panic_or_time_out_change_nothing() {
        for behavior in [
            Behavior::Fail,
            Behavior::Panic,
            Behavior::Sleep(Duration::from_millis(400)),
        ] {
            let (pipeline, context) = pipeline(vec![fake("s", OnFailure::PassThrough, behavior)]);
            let original = request();
            let compiled = pipeline.compile(original.clone(), context).await.unwrap();
            assert_eq!(compiled.request, original);
            assert!(!compiled.changed());
            assert!(!matches!(
                compiled.trace[0].outcome,
                StageOutcome::Applied(_)
            ));
        }
    }

    #[tokio::test]
    async fn blocking_stages_that_fail_or_time_out_stop_the_request() {
        for behavior in [Behavior::Fail, Behavior::Sleep(Duration::from_millis(400))] {
            let (pipeline, context) = pipeline(vec![fake("guard", OnFailure::Block, behavior)]);
            let err = pipeline.compile(request(), context).await.unwrap_err();
            assert_eq!(err.stage, "guard");
        }
    }

    #[tokio::test]
    async fn a_failed_stage_does_not_undo_an_earlier_one() {
        let (pipeline, context) = pipeline(vec![
            fake(
                "one",
                OnFailure::PassThrough,
                Behavior::Edit(vec![content_edit(0, "kept")]),
            ),
            fake("two", OnFailure::PassThrough, Behavior::Fail),
        ]);
        let compiled = pipeline.compile(request(), context).await.unwrap();
        assert_eq!(compiled.request.messages()[0]["content"], "kept");
    }

    #[test]
    fn refuses_edits_to_protected_history() {
        let (pipeline, mut context) = pipeline(vec![fake(
            "s",
            OnFailure::PassThrough,
            Behavior::Edit(vec![content_edit(0, "rewritten"), content_edit(1, "ok")]),
        )]);
        context.protected_before = 1;
        let compiled = pipeline.compile_offline(&request(), &context).unwrap();
        assert_eq!(compiled.request.messages()[0]["content"], "a");
        assert_eq!(compiled.request.messages()[1]["content"], "ok");
    }

    #[test]
    fn refuses_edits_to_thinking_blocks_at_any_depth() {
        let edits = ["/content/0", "/content/0/signature"].map(|pointer| Edit {
            target: Target::Message(1),
            pointer: pointer.to_owned(),
            value: json!("x"),
            rule: "test",
        });
        let (pipeline, context) = pipeline(vec![fake(
            "s",
            OnFailure::PassThrough,
            Behavior::Edit(edits.to_vec()),
        )]);
        let original = request();
        let compiled = pipeline.compile_offline(&original, &context).unwrap();
        assert_eq!(compiled.request, original);
    }

    #[test]
    fn the_system_prompt_is_editable_only_without_protected_history() {
        let edit = Edit {
            target: Target::System,
            pointer: String::new(),
            value: json!("clean"),
            rule: "test",
        };
        let original = Request::parse(br#"{"system":"secret","messages":[]}"#).unwrap();
        let (pipeline, mut context) = pipeline(vec![fake(
            "s",
            OnFailure::PassThrough,
            Behavior::Edit(vec![edit]),
        )]);

        let compiled = pipeline.compile_offline(&original, &context).unwrap();
        assert_eq!(compiled.request.field("system"), Some(&json!("clean")));

        context.protected_before = 1;
        let compiled = pipeline.compile_offline(&original, &context).unwrap();
        assert_eq!(compiled.request, original);
    }

    #[test]
    fn skips_stages_not_enabled_for_the_conversation() {
        let (pipeline, mut context) = pipeline(vec![fake(
            "s",
            OnFailure::PassThrough,
            Behavior::Edit(vec![content_edit(0, "x")]),
        )]);
        context.enabled.clear();
        let compiled = pipeline.compile_offline(&request(), &context).unwrap();
        assert_eq!(compiled.trace[0].outcome, StageOutcome::Skipped);
        assert!(!compiled.changed());
    }

    const _: () = {
        const fn assert_send<T: Send>() {}
        assert_send::<Pipeline>();
    };
}

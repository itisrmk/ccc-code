//! Runs compiler stages over a request.
//!
//! A stage never mutates a request. It returns a list of edits, so a stage that fails
//! or runs out of time simply contributes nothing, and the same list explains what
//! changed. Edits to history the server has already seen are refused here regardless of
//! what a stage asks for, because they would invalidate signed thinking blocks.

mod stage;

pub use stage::{
    CompileError, Compiled, Context, Edit, OnFailure, Pipeline, Stage, StageError, StageOutcome,
    StageTrace,
};

//! Tests whether a Claude Code setup really helps, on tasks from a repository's own history.
//!
//! A task is a real commit with its source change undone and its tests kept. Each setup, a
//! `CLAUDE.md`, a skill, a hook, gets the same tasks, and the score comes from running the tests
//! afterward, never from anything the model says. Runs of one task share a task's difficulty, so
//! setups are compared task by task, and each task counts once however often it's repeated.

mod mine;
mod process;
mod run;
mod stats;
mod workspace;

pub use mine::{MineOptions, MineSummary, Task, TaskFile, mine};
pub use run::{Record, RunOptions, Setup, load_records, run};
pub use stats::{
    Comparison, Metric, MetricResult, PairedTest, SIGNIFICANCE, Verdict, compare, paired_test,
};
pub use workspace::Workspace;

use std::path::PathBuf;

/// Failure to mine tasks or run them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EvalError {
    /// A git command failed.
    #[error("git {command}: {detail}")]
    Git {
        /// What was asked of git.
        command: String,
        /// What it said.
        detail: String,
    },
    /// A file or process couldn't be used.
    #[error("{context}")]
    Io {
        /// What was being done.
        context: String,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },
    /// A stored file isn't what it should be.
    #[error("reading {path:?}")]
    Format {
        /// The file.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: serde_json::Error,
    },
}

impl EvalError {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

//! Replays Claude Code transcripts offline to measure token and cost effects.
//!
//! Recent transcripts hold every request as sent, so a replay uses exact wire content.
//! Recorded `usage` then calibrates the token estimator and checks the cache simulator,
//! which is what makes before-and-after numbers for compiled requests trustworthy.

mod cache_sim;
mod checks;
mod estimate;
mod report;
mod timestamp;
mod transcript;

pub use checks::{Fidelity, append_only_violations, fidelity};
pub use report::{ReplayStats, Report, Scenario, replay};
pub use transcript::{Call, LoadSummary, Session, load_dir};

use std::path::PathBuf;

/// Failure to read a transcript directory.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BenchError {
    /// A file or directory could not be read.
    #[error("reading {path:?}")]
    Read {
        /// The path that failed.
        path: PathBuf,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
}

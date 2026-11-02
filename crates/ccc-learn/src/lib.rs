//! Learns which files a repository's work tends to need, from local transcripts.
//!
//! Claude Code keeps its transcripts on this machine. This reads them, ties each user
//! prompt to the files the work on it went on to touch, and remembers the association.
//! Nothing leaves the machine.

mod memory;
mod rules;
mod turns;

pub use memory::{Memory, OpenMemoryError};
pub use rules::{
    Candidate, EVERYWHERE, Rule, RuleStatus, RuleStore, WorkingCommand, read_candidates,
    read_working_commands, rules_text,
};
pub use turns::{Turn, read_turns};

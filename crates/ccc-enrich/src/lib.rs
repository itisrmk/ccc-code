//! Adds likely relevant files to a user's turn, with an optional gated rewrite.
//!
//! Suggestions change as the index and memory grow, so each one is decided once, for the
//! newest message, and saved. A later request replays exactly what was sent.

mod rewrite;
mod stage;
mod suggest;

pub use rewrite::{GateConfig, gate_rewrite, propose_rewrite};
pub use stage::{Enrich, EnrichConfig};
pub use suggest::{Suggester, hint_text};

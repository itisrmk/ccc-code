//! Swaps secrets for stable placeholders before requests leave the machine, and swaps
//! them back in responses, so the model never sees the real value and Claude Code never
//! sees a placeholder.
//!
//! Placeholders derive from the secret and a per-install key, so a value always gets the
//! same placeholder, with no ordering state to lose across turns, subagents or restarts.
//!
//! Detectors run on what enters the conversation from outside. Text the model wrote only
//! has secrets seen earlier in the same request swapped back to their placeholders: the
//! model can only emit a real value through rehydration, so this restores its exact words,
//! which signed thinking blocks depend on.

mod detect;
mod rehydrate;
mod stage;
mod vault;

pub use detect::{Detector, InvalidRuleError};
pub use rehydrate::{ResponseRehydrator, rehydrate_json};
pub use stage::{NameFinder, Redact};
pub use vault::{OpenVaultError, Vault};

//! Checks whether a gateway is safe to put between Claude Code and the Anthropic API.
//!
//! A thinking block's signature binds `system`, `tools` and every earlier message, so a gateway
//! that edits sent history gets a 400 on accounts created on or after 2026-08-31 and restarts
//! the prompt cache on every account. This points a gateway at a recording stand-in for the API
//! and sends it a scripted conversation, so it needs no API key and spends nothing. Signatures
//! here are fake: the stand-in judges by comparing bytes, which is stricter than the real check.

mod checks;
mod script;
mod upstream;

pub use checks::{Check, Outcome, Report, VerifyError, VerifyOptions, run};
pub use upstream::FakeUpstream;

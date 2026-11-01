//! A small local model behind a narrow interface, and the stages that use it.
//!
//! Everything here is optional, off by default, and built to tolerate a slow or absent
//! model. A model isn't a pure function, which is what compiling otherwise depends on, so
//! anything that changes a request acts only on the newest message and persists its
//! decision. See [`BlobPrune`].

mod blob;
mod classify;
#[cfg(feature = "llama")]
mod llama;
mod model;

pub use blob::{
    BlobPrune, BlobPruneConfig, apply_keep_ranges, parse_keep_ranges, reads_fast_enough,
};
pub use classify::{Classify, DEFAULT_CLASSIFY_CAP, Difficulty, Intent, Labels, classify_labels};
#[cfg(feature = "llama")]
pub use llama::{ChatFormat, LlamaConfig, LlamaEmbedder, LlamaModel, LoadModelError};
pub use model::{Completion, CompletionRequest, Embedder, ModelError, TextModel};

//! An incremental index of a repository's files and symbols.
//!
//! Symbols come from tree-sitter, so a language's own grammar decides what a definition
//! is. Rebuilding only re-parses files whose size or modification time changed.

mod index;
mod symbols;

pub use index::{FileEntry, Index, IndexError, terms};
pub use symbols::{Symbol, extract_symbols};

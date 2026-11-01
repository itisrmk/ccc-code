//! Adds likely relevant files to a prompt as context, through a Claude Code hook.
//!
//! Which sources get used comes from `ccc enrich-eval` on real history. Past sessions plus
//! keywords found a touched file most often, and adding vectors to those made things worse, so
//! vectors only step in for a repository with no history, where they beat keywords alone. Asking
//! a model to choose among the files also made things worse, so none does.
//!
//! The person's own words are never changed: this only adds beside them. Going through a hook
//! needs no proxy, and Claude Code keeps what was added, so sent history stays stable.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use ccc_enrich::{Suggester, hint_text};
use ccc_index::{Index, fuse, terms};
use ccc_learn::Memory;

use crate::learn::{index_cache_path, memory_path};

/// Files to name.
const SUGGESTED_FILES: usize = 3;
/// How deep each ranking goes before combining them.
const RANKING_DEPTH: usize = 20;
/// Symbols to name per file.
const SYMBOLS_PER_FILE: usize = 3;
/// A prompt with fewer meaningful words than this gives search nothing to work with.
const MIN_PROMPT_TERMS: usize = 2;

/// The context for `prompt`, or `None` if nothing looks relevant. `vector_files` runs only when
/// vectors would be used, since embedding has a cost.
pub(crate) fn context_for(
    root: &Path,
    state_dir: &Path,
    prompt: &str,
    vector_files: &dyn Fn() -> Vec<String>,
) -> anyhow::Result<Option<String>> {
    if terms(prompt).len() < MIN_PROMPT_TERMS {
        return Ok(None);
    }
    let cache = index_cache_path(state_dir, root);
    let memory = Arc::new(Memory::open(&memory_path(state_dir)).context("opening memory")?);
    let repo = root.to_string_lossy();
    let has_history = !memory
        .suggest(&repo, prompt, 1)
        .context("reading memory")?
        .is_empty();
    let suggester = Suggester::new(
        root.to_owned(),
        Index::load(&cache),
        Some(Arc::clone(&memory)),
    );
    if suggester.refresh() > 0 {
        suggester.save_index(&cache).context("saving the index")?;
    }
    if has_history {
        return Ok(suggester.hints(prompt, SUGGESTED_FILES));
    }
    let keywords = suggester.suggest(prompt, RANKING_DEPTH);
    let vectors = vector_files();
    let files = fuse(&[&keywords, &vectors], SUGGESTED_FILES);
    if files.is_empty() {
        return Ok(None);
    }
    let described: Vec<(String, Vec<String>)> = files
        .into_iter()
        .map(|file| {
            let symbols = suggester.symbols_of(&file, SYMBOLS_PER_FILE);
            (file, symbols)
        })
        .collect();
    Ok(Some(hint_text(&described)))
}

fn vectors_for(state_dir: &Path, root: &Path, prompt: &str) -> Vec<String> {
    crate::local_work::vector_files(state_dir, root, prompt, RANKING_DEPTH)
}

/// Handles the hook. Nothing here may get in the way of a prompt, so any failure just adds
/// nothing. Plain output is what Claude Code adds as context.
pub(crate) fn hook(state_dir: &Path, cwd: &str, prompt: &str) -> Option<String> {
    let root = std::fs::canonicalize(cwd).ok()?;
    let vectors = || vectors_for(state_dir, &root, prompt);
    match context_for(&root, state_dir, prompt, &vectors) {
        Ok(context) => context,
        Err(err) => {
            tracing::warn!(error = %err, "could not build context");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_learn::Turn;
    use std::cell::Cell;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;

    struct Fixture {
        base: PathBuf,
        root: PathBuf,
        state: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn fixture(name: &str) -> Fixture {
        let base = std::env::temp_dir().join(format!("ccc-enhance-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join("repo");
        let state = base.join("state");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&state).unwrap();
        fs::write(root.join("auth.rs"), "pub fn verify_token() {}\n").unwrap();
        fs::write(root.join("session.rs"), "pub fn start() {}\n").unwrap();
        Fixture { base, root, state }
    }

    #[test]
    fn with_no_history_vectors_join_keywords() {
        let fixture = fixture("fresh");
        let vectors = || vec!["session.rs".to_owned()];
        let context = context_for(&fixture.root, &fixture.state, "verify the token", &vectors)
            .unwrap()
            .unwrap();
        assert!(context.contains("auth.rs (verify_token)"), "{context}");
        assert!(context.contains("session.rs (start)"), "{context}");
    }

    #[test]
    fn with_history_vectors_are_never_consulted() {
        let fixture = fixture("history");
        let repo = fixture.root.to_string_lossy().into_owned();
        Memory::open(&memory_path(&fixture.state))
            .unwrap()
            .learn(&[Turn {
                repo,
                at: "t".to_owned(),
                prompt: "verify the token".to_owned(),
                files: BTreeSet::from(["session.rs".to_owned()]),
            }])
            .unwrap();
        let asked = Cell::new(false);
        let vectors = || {
            asked.set(true);
            Vec::new()
        };
        let context = context_for(&fixture.root, &fixture.state, "verify the token", &vectors)
            .unwrap()
            .unwrap();
        assert!(!asked.get(), "vectors shouldn't run");
        assert!(context.contains("session.rs"), "{context}");
    }

    #[test]
    fn nothing_to_go_on_adds_nothing() {
        let fixture = fixture("nothing");
        let none = || Vec::new();
        for prompt in ["ok", "thanks", "", "zebra crossing"] {
            let context = context_for(&fixture.root, &fixture.state, prompt, &none).unwrap();
            assert_eq!(context, None, "{prompt}");
        }
    }

    #[test]
    fn a_bad_directory_never_gets_in_the_way() {
        let fixture = fixture("bad");
        assert_eq!(
            hook(&fixture.state, "/definitely/missing", "verify token"),
            None
        );
    }
}

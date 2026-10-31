//! The optional name-detection tier, and measuring it.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use ccc_ner::NerModel;
use ccc_redact::{NameFinder, Redact};

/// What counts as a name to hide.
const LABELS: [&str; 2] = ["person", "organization"];
/// The longest a scan of one text may take on the live path. Redaction stops the request if
/// it exceeds its own, longer limit, so this stays comfortably under that.
const NAME_SCAN_CAP: Duration = Duration::from_millis(1_000);
/// Pieces of text to measure, each about what the live tier would scan.
const BENCH_PIECES: usize = 200;
const PIECE_BYTES: usize = 2_000;
/// How many flagged strings to print from the bench.
const SHOWN_FLAGS: usize = 15;

#[derive(Debug)]
struct GlinerFinder(NerModel);

impl NameFinder for GlinerFinder {
    fn find_names(&self, text: &str) -> Result<Vec<String>, String> {
        let found = self.0.find(text, &LABELS).map_err(|err| err.to_string())?;
        Ok(found
            .into_iter()
            .map(|entity| entity.text)
            .filter(|name| looks_like_a_full_name(name))
            .collect())
    }
}

/// Whether a flagged string is safe to hide everywhere it appears. On source code the model
/// flags words such as `self`, `name` and `root`, and replacing those would wreck everything
/// the model reads. Requiring two or more capitalized words rules those out. A single-word
/// name such as a company's belongs in the project's redaction rules instead.
fn looks_like_a_full_name(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    words.len() >= 2
        && words
            .iter()
            .all(|word| word.chars().next().is_some_and(char::is_uppercase))
}

/// Adds the tier to `redact` if a model directory was given.
pub(crate) fn add_tier(
    redact: Redact,
    model_dir: Option<&Path>,
    state_dir: Option<&Path>,
) -> anyhow::Result<Redact> {
    let Some(model_dir) = model_dir else {
        return Ok(redact);
    };
    let state = state_dir.context("--ner-model needs a state directory")?;
    let model = NerModel::load(model_dir).context("loading the name model")?;
    Ok(redact.with_names(
        Arc::new(GlinerFinder(model)),
        state.join("name-decisions"),
        NAME_SCAN_CAP,
    ))
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .is_some_and(|name| name == "target" || name == ".git")
        {
            continue;
        }
        if path.is_dir() {
            collect(&path, found);
        } else {
            found.push(path);
        }
    }
}

/// Measures speed, and what's flagged in `files`. Source code holds few real names, so
/// running this on a code directory estimates the false positive rate.
pub(crate) fn bench(model_dir: &Path, files: &Path) -> anyhow::Result<()> {
    let model = NerModel::load(model_dir).context("loading the name model")?;
    let mut paths = Vec::new();
    collect(files, &mut paths);
    paths.sort();

    let mut pieces: Vec<String> = Vec::new();
    for path in paths {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let characters: Vec<char> = text.chars().collect();
        for piece in characters.chunks(PIECE_BYTES) {
            pieces.push(piece.iter().collect());
        }
    }
    pieces.truncate(BENCH_PIECES);

    let mut elapsed = Vec::with_capacity(pieces.len());
    let mut flagged: HashMap<String, usize> = HashMap::new();
    let mut pieces_flagged = 0;
    for piece in &pieces {
        let started = Instant::now();
        let found = model
            .find(piece, &LABELS)
            .context("running the name model")?;
        elapsed.push(started.elapsed());
        let kept: Vec<String> = found
            .into_iter()
            .map(|entity| entity.text)
            .filter(|name| looks_like_a_full_name(name))
            .collect();
        pieces_flagged += usize::from(!kept.is_empty());
        for name in kept {
            *flagged.entry(name).or_default() += 1;
        }
    }
    elapsed.sort();
    let at = |fraction: f64| elapsed[((elapsed.len() - 1) as f64 * fraction) as usize];

    let mut out = std::io::stdout();
    writeln!(
        out,
        "{} pieces of about {PIECE_BYTES} characters",
        pieces.len()
    )?;
    writeln!(
        out,
        "  latency p50 {:?}, p95 {:?}, max {:?}",
        at(0.5),
        at(0.95),
        at(1.0)
    )?;
    writeln!(
        out,
        "  pieces with a flag: {pieces_flagged}; distinct strings flagged: {}",
        flagged.len()
    )?;
    let mut ranked: Vec<(String, usize)> = flagged.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (text, count) in ranked.into_iter().take(SHOWN_FLAGS) {
        writeln!(out, "  {count:>4}  {text}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_multi_word_capitalized_names_count() {
        for name in ["Maria Gonzalez", "Northwind Traders", "Howard Hinnant"] {
            assert!(looks_like_a_full_name(name), "{name}");
        }
        // All flagged by the real model on this repository's own source.
        for word in [
            "self",
            "name",
            "root",
            "decisions",
            "Rates",
            "Conversations",
            "ccc-ledger",
        ] {
            assert!(!looks_like_a_full_name(word), "{word}");
        }
    }
}

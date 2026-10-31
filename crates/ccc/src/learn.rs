//! `ccc learn`, `ccc index` and `ccc enrich-eval`: building and judging local knowledge.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use ccc_enrich::Suggester;
use ccc_index::Index;
use ccc_learn::{Memory, Turn, read_turns};

/// Suggestions per prompt, matching what Enrich sends.
const SUGGESTED_FILES: usize = 3;
/// A repository needs some history before memory can be judged fairly on it.
const MIN_TURNS_TO_JUDGE: usize = 10;

/// Where a repository's index is cached, inside the state directory.
pub(crate) fn index_cache_path(state_dir: &Path, root: &Path) -> PathBuf {
    let safe: String = root
        .to_string_lossy()
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect();
    state_dir.join("index").join(format!("{safe}.json"))
}

pub(crate) fn memory_path(state_dir: &Path) -> PathBuf {
    state_dir.join("memory.sqlite")
}

/// Learns from every transcript under `transcripts`.
pub(crate) fn learn(state_dir: &Path, transcripts: &Path) -> anyhow::Result<()> {
    let turns = read_turns(transcripts).context("reading transcripts")?;
    let memory = Memory::open(&memory_path(state_dir)).context("opening memory")?;
    let learned = memory.learn(&turns).context("writing memory")?;
    writeln!(
        std::io::stdout(),
        "read {} turns, learned from {learned} new ones",
        turns.len()
    )?;
    Ok(())
}

/// Refreshes the index of `root` and saves it.
pub(crate) fn index(state_dir: &Path, root: &Path) -> anyhow::Result<()> {
    let cache = index_cache_path(state_dir, root);
    let mut index = Index::load(&cache);
    let started = Instant::now();
    let parsed = index.refresh(root);
    index.save(&cache).context("saving index")?;
    writeln!(
        std::io::stdout(),
        "{} files indexed, {parsed} parsed, in {:?}",
        index.len(),
        started.elapsed()
    )?;
    Ok(())
}

#[derive(Debug, Default)]
struct Score {
    turns: usize,
    hits: usize,
    suggested: usize,
    correct: usize,
    silent: usize,
}

impl Score {
    fn add(&mut self, suggested: &[String], actual: &BTreeSet<String>) {
        self.turns += 1;
        if suggested.is_empty() {
            self.silent += 1;
            return;
        }
        let correct = suggested
            .iter()
            .filter(|file| actual.contains(*file))
            .count();
        self.suggested += suggested.len();
        self.correct += correct;
        self.hits += usize::from(correct > 0);
    }

    fn row(&self, label: &str) -> String {
        let percent = |part: usize, whole: usize| {
            if whole == 0 {
                0.0
            } else {
                100.0 * part as f64 / whole as f64
            }
        };
        format!(
            "  {label:<26}{:>7}{:>11.1}%{:>13.1}%{:>11.1}%",
            self.turns,
            percent(self.hits, self.turns),
            percent(self.correct, self.suggested),
            percent(self.silent, self.turns),
        )
    }
}

/// Replays history in order: each prompt is judged using only what came before it, against
/// the files its work really touched. This is the evidence for whether hints are worth
/// their tokens.
pub(crate) fn enrich_eval(transcripts: &Path) -> anyhow::Result<()> {
    let turns = read_turns(transcripts).context("reading transcripts")?;
    let mut by_repo: BTreeMap<&str, Vec<&Turn>> = BTreeMap::new();
    for turn in turns.iter().filter(|turn| !turn.files.is_empty()) {
        by_repo.entry(turn.repo.as_str()).or_default().push(turn);
    }

    let (mut memory_score, mut index_score, mut both_score) =
        (Score::default(), Score::default(), Score::default());
    let mut repos_judged = 0;
    let mut repos_with_index = 0;
    for (repo, repo_turns) in &by_repo {
        if repo_turns.len() < MIN_TURNS_TO_JUDGE {
            continue;
        }
        repos_judged += 1;
        let root = PathBuf::from(repo);
        let has_checkout = root.is_dir();
        repos_with_index += usize::from(has_checkout);

        let memory = Arc::new(Memory::in_memory().context("creating memory")?);
        let memory_only = Suggester::new(root.clone(), Index::default(), Some(Arc::clone(&memory)));
        let index_only = Suggester::new(root.clone(), Index::default(), None);
        let both = Suggester::new(root.clone(), Index::default(), Some(Arc::clone(&memory)));
        if has_checkout {
            index_only.refresh();
            both.refresh();
        }
        for turn in repo_turns {
            memory_score.add(
                &memory_only.suggest(&turn.prompt, SUGGESTED_FILES),
                &turn.files,
            );
            if has_checkout {
                index_score.add(
                    &index_only.suggest(&turn.prompt, SUGGESTED_FILES),
                    &turn.files,
                );
                both_score.add(&both.suggest(&turn.prompt, SUGGESTED_FILES), &turn.files);
            }
            memory
                .learn(std::slice::from_ref(*turn))
                .context("updating memory")?;
        }
    }

    let mut out = std::io::stdout();
    writeln!(
        out,
        "{} turns touched files; judged {repos_judged} repositories, {repos_with_index} with a checkout here",
        turns.iter().filter(|turn| !turn.files.is_empty()).count()
    )?;
    writeln!(
        out,
        "  {:<26}{:>7}{:>12}{:>14}{:>12}",
        "suggesting from", "turns", "hit at 3", "precision", "no answer"
    )?;
    writeln!(out, "{}", memory_score.row("past sessions only"))?;
    writeln!(out, "{}", index_score.row("index only"))?;
    writeln!(out, "{}", both_score.row("both"))?;
    Ok(())
}

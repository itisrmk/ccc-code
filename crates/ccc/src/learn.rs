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
/// How deep each ranking goes before fusing. Rank fusion needs more than the final few.
#[cfg(feature = "local-model")]
const FUSION_DEPTH: usize = 20;
/// Files a model chooses among, and how much it's told about each.
#[cfg(feature = "local-model")]
const RERANK_CANDIDATES: usize = 8;
#[cfg(feature = "local-model")]
const RERANK_SYMBOLS: usize = 6;
/// A person is waiting, so a slow answer is no answer.
#[cfg(feature = "local-model")]
const RERANK_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);
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

/// Where a repository's vectors are cached, inside the state directory.
#[cfg(feature = "local-model")]
pub(crate) fn semantic_cache_path(state_dir: &Path, root: &Path) -> PathBuf {
    let safe = index_cache_path(state_dir, root);
    safe.with_extension("vectors.json")
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

/// What judging search by meaning needs.
#[cfg(feature = "local-model")]
pub(crate) struct VectorEval<'a> {
    /// The embedding model.
    pub(crate) model: &'a Path,
    /// A model to reorder candidates with, and how it wants its prompts.
    pub(crate) reranker: Option<(&'a Path, ccc_slm::ChatFormat)>,
    /// Where vectors are cached.
    pub(crate) state_dir: &'a Path,
}

/// Scores vector search and its combinations on repositories with a checkout here. Each row
/// sees the same turns as the keyword row, so they compare directly.
#[cfg(feature = "local-model")]
fn embedding_rows(
    vectors: &VectorEval<'_>,
    by_repo: &BTreeMap<&str, Vec<&Turn>>,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    use ccc_enrich::{Candidate, rerank};
    use ccc_index::{SemanticIndex, fuse, query_text};
    use ccc_slm::{Embedder as _, LlamaConfig, LlamaEmbedder, LlamaModel};

    let VectorEval {
        model,
        reranker,
        state_dir,
    } = *vectors;
    let reranker = reranker
        .map(|(path, format)| {
            LlamaModel::load(
                path,
                LlamaConfig {
                    format,
                    ..LlamaConfig::default()
                },
            )
        })
        .transpose()
        .context("loading the reranking model")?;
    let (mut reranked_score, mut rerank_time, mut rerank_failures) =
        (Score::default(), std::time::Duration::ZERO, 0_usize);

    let embedder = LlamaEmbedder::load(model).context("loading the embedding model")?;
    let (mut vector_score, mut keyword_score, mut fused_score, mut all_score) = (
        Score::default(),
        Score::default(),
        Score::default(),
        Score::default(),
    );
    let (mut chunks, mut build_time, mut query_time, mut queries) = (
        0,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        0_u32,
    );
    for (repo, repo_turns) in by_repo {
        let root = PathBuf::from(repo);
        if repo_turns.len() < MIN_TURNS_TO_JUDGE || !root.is_dir() {
            continue;
        }
        let cache = semantic_cache_path(state_dir, &root);
        let mut semantic = SemanticIndex::load(&cache);
        let started = Instant::now();
        semantic
            .refresh(&root, &mut |text: &str| embedder.embed(text))
            .with_context(|| format!("embedding {repo}"))?;
        build_time += started.elapsed();
        chunks += semantic.len();
        semantic.save(&cache).context("saving vectors")?;

        let memory = Arc::new(Memory::in_memory().context("creating memory")?);
        let keywords = Suggester::new(root.clone(), Index::default(), None);
        let past_sessions =
            Suggester::new(root.clone(), Index::default(), Some(Arc::clone(&memory)));
        keywords.refresh();
        let mut symbols = Index::default();
        symbols.refresh(&root);
        for turn in repo_turns {
            let asked = Instant::now();
            let query = embedder
                .embed(&query_text(&turn.prompt))
                .context("embedding a prompt")?;
            let vector: Vec<String> = semantic
                .search(&query, FUSION_DEPTH)
                .into_iter()
                .map(|hit| hit.file)
                .collect();
            query_time += asked.elapsed();
            queries += 1;
            let keyword = keywords.suggest(&turn.prompt, FUSION_DEPTH);
            let remembered = past_sessions.suggest(&turn.prompt, FUSION_DEPTH);
            let top = |files: &[String]| {
                files
                    .iter()
                    .take(SUGGESTED_FILES)
                    .cloned()
                    .collect::<Vec<_>>()
            };
            vector_score.add(&top(&vector), &turn.files);
            keyword_score.add(&top(&keyword), &turn.files);
            fused_score.add(&fuse(&[&keyword, &vector], SUGGESTED_FILES), &turn.files);
            all_score.add(
                &fuse(&[&keyword, &vector, &remembered], SUGGESTED_FILES),
                &turn.files,
            );
            if let Some(reranker) = &reranker {
                let candidates: Vec<Candidate> =
                    fuse(&[&keyword, &vector, &remembered], RERANK_CANDIDATES)
                        .into_iter()
                        .map(|file| Candidate {
                            symbols: symbols
                                .symbols_in(&file)
                                .iter()
                                .take(RERANK_SYMBOLS)
                                .map(|symbol| symbol.name.clone())
                                .collect(),
                            file,
                        })
                        .collect();
                let unranked: Vec<String> = candidates
                    .iter()
                    .take(SUGGESTED_FILES)
                    .map(|candidate| candidate.file.clone())
                    .collect();
                let started = Instant::now();
                let picked = rerank(
                    reranker,
                    &turn.prompt,
                    &candidates,
                    SUGGESTED_FILES,
                    RERANK_DEADLINE,
                );
                rerank_time += started.elapsed();
                rerank_failures += usize::from(picked.is_err());
                reranked_score.add(&picked.unwrap_or(unranked), &turn.files);
            }
            memory
                .learn(std::slice::from_ref(*turn))
                .context("updating memory")?;
        }
    }
    writeln!(out, "{}", keyword_score.row("keywords, these repos"))?;
    writeln!(out, "{}", vector_score.row("embeddings only"))?;
    writeln!(out, "{}", fused_score.row("keywords + embeddings"))?;
    writeln!(out, "{}", all_score.row("those + past sessions"))?;
    if reranker.is_some() {
        writeln!(out, "{}", reranked_score.row("those, reranked by model"))?;
        writeln!(
            out,
            "  reranking took {:?} per prompt, and failed {rerank_failures} times",
            rerank_time / queries.max(1)
        )?;
    }
    writeln!(
        out,
        "  {chunks} chunks, embedded or checked in {build_time:?}; {:?} per prompt",
        query_time / queries.max(1)
    )?;
    Ok(())
}

/// Replays history in order: each prompt is judged using only what came before it, against
/// the files its work really touched. This is the evidence for whether hints are worth
/// their tokens.
pub(crate) fn enrich_eval(
    transcripts: &Path,
    #[cfg(feature = "local-model")] with_vectors: Option<&VectorEval<'_>>,
) -> anyhow::Result<()> {
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
    #[cfg(feature = "local-model")]
    if let Some(vectors) = with_vectors {
        embedding_rows(vectors, &by_repo, &mut out)?;
    }
    Ok(())
}

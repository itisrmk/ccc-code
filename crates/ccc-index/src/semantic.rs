//! Finding code by meaning, for prompts that don't name what they're after.
//!
//! Vectors can't find an exact identifier the way the keyword [`Index`](crate::Index) can, and
//! keywords can't connect "login" to `verify_token`, so [`fuse`] combines their rankings. The
//! embedding model comes in as a closure, which keeps this crate free of native code.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use ignore::WalkBuilder;
use serde_json::{Value, json};

use crate::index::IndexError;

/// Lines per chunk: about a function's worth, and well inside the model's 2,048 tokens.
const CHUNK_LINES: usize = 40;
/// Lines shared between neighbouring chunks, so code at a boundary stays findable.
const CHUNK_OVERLAP: usize = 8;
/// Files larger than this are almost always generated.
const MAX_FILE_BYTES: u64 = 1 << 20;
/// The conventional constant in reciprocal rank fusion. It stops first place from dominating.
const FUSION_K: f64 = 60.0;
/// Files that hold no knowledge, however they match.
const SKIPPED_NAMES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "go.sum",
];
/// The model's own prefixes for code search. They're part of how it was trained.
const QUERY_PREFIX: &str = "task: code retrieval | query: ";

#[derive(Debug, Clone, PartialEq)]
struct Chunk {
    file: String,
    start_line: usize,
    /// Identifies the text, so an unchanged chunk keeps its vector.
    key: u64,
    vector: Vec<f32>,
}

/// One match.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticHit {
    /// The file, relative to the repository root.
    pub file: String,
    /// Where the matching chunk starts, from 1.
    pub start_line: usize,
    /// Its similarity to the query, at most 1.
    pub score: f32,
}

/// Vectors for a repository's code.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SemanticIndex {
    chunks: Vec<Chunk>,
}

fn chunk_key(file: &str, text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file.hash(&mut hasher);
    text.hash(&mut hasher);
    hasher.finish()
}

/// Splits `source` into overlapping runs of lines, with where each one starts. Blank runs go.
fn split_lines(source: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let text = lines[start..end].join("\n");
        if !text.trim().is_empty() {
            chunks.push((start + 1, text));
        }
        if end == lines.len() {
            break;
        }
        start = end - CHUNK_OVERLAP;
    }
    chunks
}

fn document_text(file: &str, text: &str) -> String {
    format!("title: {file} | text: {text}")
}

/// What a query should be embedded as.
#[must_use]
pub fn query_text(prompt: &str) -> String {
    format!("{QUERY_PREFIX}{prompt}")
}

impl SemanticIndex {
    /// Brings the index up to date with `root`, embedding only text it hasn't seen. Returns how
    /// many chunks were embedded. Respects `.gitignore`.
    ///
    /// # Errors
    ///
    /// Returns the embedder's first error. The index keeps what it had before the call.
    pub fn refresh<E>(
        &mut self,
        root: &Path,
        embed: &mut dyn FnMut(&str) -> Result<Vec<f32>, E>,
    ) -> Result<usize, E> {
        let known: HashMap<u64, &Chunk> = self.chunks.iter().map(|c| (c.key, c)).collect();
        let mut chunks = Vec::new();
        let mut embedded = 0;
        for entry in WalkBuilder::new(root).build().flatten() {
            let path = entry.path();
            if !entry.file_type().is_some_and(|kind| kind.is_file())
                || entry.metadata().map_or(true, |m| m.len() > MAX_FILE_BYTES)
            {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if SKIPPED_NAMES.contains(&name) {
                continue;
            }
            // Anything that isn't text fails here, and there's nothing to search in it.
            let (Ok(relative), Ok(source)) = (path.strip_prefix(root), fs::read_to_string(path))
            else {
                continue;
            };
            let file = relative.to_string_lossy().into_owned();
            for (start_line, text) in split_lines(&source) {
                let key = chunk_key(&file, &text);
                let vector = match known.get(&key) {
                    Some(chunk) => chunk.vector.clone(),
                    None => {
                        embedded += 1;
                        embed(&document_text(&file, &text))?
                    }
                };
                chunks.push(Chunk {
                    file: file.clone(),
                    start_line,
                    key,
                    vector,
                });
            }
        }
        self.chunks = chunks;
        Ok(embedded)
    }

    /// The number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether nothing is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// The chunks most similar to `query`, best first, with at most one per file.
    #[must_use]
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<SemanticHit> {
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .filter(|chunk| chunk.vector.len() == query.len())
            .map(|chunk| {
                let score = chunk.vector.iter().zip(query).map(|(a, b)| a * b).sum();
                (score, chunk)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut seen = HashSet::new();
        scored
            .into_iter()
            .filter(|(_, chunk)| seen.insert(chunk.file.as_str()))
            .take(limit)
            .map(|(score, chunk)| SemanticHit {
                file: chunk.file.clone(),
                start_line: chunk.start_line,
                score,
            })
            .collect()
    }

    /// Reads a saved index. A missing or damaged file just means starting over.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            return Self::default();
        };
        let chunks = value
            .get("chunks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|chunk| {
                Some(Chunk {
                    file: chunk.get("file")?.as_str()?.to_owned(),
                    start_line: usize::try_from(chunk.get("line")?.as_u64()?).ok()?,
                    key: chunk.get("key")?.as_str()?.parse().ok()?,
                    vector: chunk
                        .get("vector")?
                        .as_array()?
                        .iter()
                        .map(|number| number.as_f64().map(|n| n as f32))
                        .collect::<Option<Vec<f32>>>()?,
                })
            })
            .collect();
        Self { chunks }
    }

    /// Saves the index.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] if the file can't be written.
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        let chunks: Vec<Value> = self
            .chunks
            .iter()
            .map(|chunk| {
                json!({
                    "file": chunk.file,
                    "line": chunk.start_line,
                    // As text, since the largest values don't survive as JSON numbers.
                    "key": chunk.key.to_string(),
                    "vector": chunk.vector,
                })
            })
            .collect();
        let write = || -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, json!({ "chunks": chunks }).to_string())
        };
        write().map_err(|source| IndexError::new(path, source))
    }
}

/// Combines ranked lists of files by reciprocal rank fusion: a file does well by ranking high
/// in any list, and better by appearing in several. Works on ranks, so the lists' incomparable
/// scores don't matter.
#[must_use]
pub fn fuse(rankings: &[&[String]], limit: usize) -> Vec<String> {
    let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
    for ranking in rankings {
        for (rank, file) in ranking.iter().enumerate() {
            *scores.entry(file.as_str()).or_default() += 1.0 / (FUSION_K + rank as f64 + 1.0);
        }
    }
    let mut ordered: Vec<(&str, f64)> = scores.into_iter().collect();
    ordered.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ordered
        .into_iter()
        .take(limit)
        .map(|(file, _)| file.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    /// Stands in for a model: a vector saying which of a few topics the text mentions.
    fn fake_embed(text: &str) -> Result<Vec<f32>, Infallible> {
        let text = text.to_lowercase();
        let vector: Vec<f32> = [["login", "token"], ["cache", "evict"], ["parse", "syntax"]]
            .iter()
            .map(|topic| {
                if topic.iter().any(|word| text.contains(word)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Ok(vector)
    }

    fn repo(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("ccc-sem-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/auth.rs"), "fn verify_token() {}\n").unwrap();
        fs::write(
            root.join("src/store.rs"),
            "fn evict_oldest() { /* cache */ }\n",
        )
        .unwrap();
        fs::write(root.join("Cargo.lock"), "token token token\n").unwrap();
        root
    }

    #[test]
    fn finds_by_meaning_and_skips_lockfiles() {
        let root = repo("find");
        let mut index = SemanticIndex::default();
        let embedded = index.refresh(&root, &mut fake_embed).unwrap();
        assert_eq!(embedded, 2);
        let query = fake_embed("how does login work").unwrap();
        let hits = index.search(&query, 5);
        assert_eq!(hits[0].file, "src/auth.rs");
        assert_eq!(hits[0].start_line, 1);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn only_new_text_is_embedded_again() {
        let root = repo("refresh");
        let mut index = SemanticIndex::default();
        index.refresh(&root, &mut fake_embed).unwrap();
        assert_eq!(index.refresh(&root, &mut fake_embed).unwrap(), 0);
        fs::write(root.join("src/store.rs"), "fn parse_syntax() {}\n").unwrap();
        assert_eq!(index.refresh(&root, &mut fake_embed).unwrap(), 1);
        fs::remove_file(root.join("src/auth.rs")).unwrap();
        index.refresh(&root, &mut fake_embed).unwrap();
        assert_eq!(index.len(), 1, "deleted files leave the index");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_model_failure_leaves_the_index_as_it_was() {
        let root = repo("fail");
        let mut index = SemanticIndex::default();
        index.refresh(&root, &mut fake_embed).unwrap();
        let before = index.clone();
        fs::write(root.join("src/new.rs"), "fn parse() {}\n").unwrap();
        let mut failing = |_: &str| Err::<Vec<f32>, &str>("model gone");
        assert!(index.refresh(&root, &mut failing).is_err());
        assert_eq!(index, before);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn long_files_overlap_and_blank_runs_are_dropped() {
        let source = (1..=100)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = split_lines(&source);
        assert_eq!(chunks[0].0, 1);
        assert_eq!(chunks[1].0, CHUNK_LINES - CHUNK_OVERLAP + 1);
        assert!(chunks.last().unwrap().1.ends_with("line 100"));
        assert!(split_lines("\n\n\n").is_empty());
        assert!(split_lines("").is_empty());
    }

    #[test]
    fn at_most_one_hit_per_file() {
        let index = SemanticIndex {
            chunks: vec![
                Chunk {
                    file: "a".into(),
                    start_line: 1,
                    key: 1,
                    vector: vec![1.0],
                },
                Chunk {
                    file: "a".into(),
                    start_line: 40,
                    key: 2,
                    vector: vec![0.9],
                },
                Chunk {
                    file: "b".into(),
                    start_line: 1,
                    key: 3,
                    vector: vec![0.5],
                },
            ],
        };
        let files: Vec<String> = index
            .search(&[1.0], 5)
            .into_iter()
            .map(|h| h.file)
            .collect();
        assert_eq!(files, ["a", "b"]);
    }

    #[test]
    fn saving_and_loading_keeps_everything() {
        let root = repo("save");
        let mut index = SemanticIndex::default();
        index.refresh(&root, &mut fake_embed).unwrap();
        let path = root.join("saved/index.json");
        index.save(&path).unwrap();
        assert_eq!(SemanticIndex::load(&path), index);
        assert!(SemanticIndex::load(&root.join("missing.json")).is_empty());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn fusion_favours_what_both_lists_agree_on() {
        let keyword = ["a".to_owned(), "shared".to_owned()];
        let semantic = ["b".to_owned(), "shared".to_owned()];
        let fused = fuse(&[&keyword, &semantic], 3);
        assert_eq!(fused[0], "shared");
        assert_eq!(fused.len(), 3);
        assert_eq!(fuse(&[], 3), Vec::<String>::new());
    }
}

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use ignore::WalkBuilder;
use serde_json::{Value, json};

use crate::symbols::{Symbol, extract_symbols};

/// Files larger than this are almost always generated, and parsing them is slow.
const MAX_FILE_BYTES: u64 = 1 << 20;
/// Weight of a match in a file's path against one in a symbol's name. A path names what a
/// file is about, where one symbol is only part of it.
const PATH_WEIGHT: f64 = 3.0;
/// Word lengths below this are too common to discriminate.
const MIN_TERM_CHARS: usize = 3;
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "this", "that", "from", "what", "why", "how", "does", "are",
    "can", "you", "please", "fix", "add", "use", "make", "should", "would", "could", "into", "not",
    "but", "all", "any", "was", "has", "have", "when", "where", "which", "there", "here", "then",
    "them", "they", "will", "just", "like", "need", "want", "get", "let", "run",
];

/// The index couldn't be saved.
#[derive(Debug, thiserror::Error)]
#[error("saving index to {path:?}")]
pub struct IndexError {
    path: PathBuf,
    #[source]
    source: std::io::Error,
}

impl IndexError {
    pub(crate) fn new(path: &Path, source: std::io::Error) -> Self {
        Self {
            path: path.to_owned(),
            source,
        }
    }
}

/// What the index knows about one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Size when last parsed, which together with the time detects changes.
    pub bytes: u64,
    /// Modification time when last parsed, in milliseconds since the Unix epoch.
    pub modified_millis: u64,
    /// The definitions found in it.
    pub symbols: Vec<Symbol>,
}

/// An index of one repository.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Index {
    /// Files by path relative to the repository root.
    files: BTreeMap<String, FileEntry>,
}

impl Index {
    /// Brings the index up to date with `root`, re-parsing only changed files. Returns how
    /// many files were parsed. Respects `.gitignore`.
    pub fn refresh(&mut self, root: &Path) -> usize {
        let mut parsed = 0;
        let mut present = BTreeMap::new();
        for entry in WalkBuilder::new(root).build().flatten() {
            let path = entry.path();
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.len() > MAX_FILE_BYTES {
                continue;
            }
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            let key = relative.to_string_lossy().into_owned();
            let modified_millis = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |elapsed| {
                    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
                });
            if let Some(known) = self.files.get(&key)
                && known.bytes == metadata.len()
                && known.modified_millis == modified_millis
            {
                present.insert(key.clone(), known.clone());
                continue;
            }
            let extension = path.extension().map_or("", |e| e.to_str().unwrap_or(""));
            let symbols = fs::read_to_string(path)
                .map(|source| extract_symbols(extension, &source))
                .unwrap_or_default();
            parsed += 1;
            present.insert(
                key,
                FileEntry {
                    bytes: metadata.len(),
                    modified_millis,
                    symbols,
                },
            );
        }
        self.files = present;
        parsed
    }

    /// The number of files indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether no files are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The files best matching `query`, best first, each with a score above zero. Rare
    /// terms count for more, the same idea as BM25's inverse document frequency.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let wanted = terms(query);
        if wanted.is_empty() || self.files.is_empty() {
            return Vec::new();
        }
        let mut per_file: Vec<(&String, HashMap<String, f64>)> =
            Vec::with_capacity(self.files.len());
        let mut files_with: HashMap<String, usize> = HashMap::new();
        for (path, entry) in &self.files {
            let mut matched: HashMap<String, f64> = HashMap::new();
            for term in terms(path) {
                if wanted.contains(&term) {
                    *matched.entry(term).or_default() += PATH_WEIGHT;
                }
            }
            for symbol in &entry.symbols {
                for term in terms(&symbol.name) {
                    if wanted.contains(&term) {
                        *matched.entry(term).or_default() += 1.0;
                    }
                }
            }
            for term in matched.keys() {
                *files_with.entry(term.clone()).or_default() += 1;
            }
            if !matched.is_empty() {
                per_file.push((path, matched));
            }
        }
        let total = self.files.len() as f64;
        let mut scored: Vec<(String, f64)> = per_file
            .into_iter()
            .map(|(path, matched)| {
                let score: f64 = matched
                    .iter()
                    .map(|(term, weight)| {
                        let with = files_with.get(term).copied().unwrap_or(1) as f64;
                        // Diminishing returns, so one repeated word can't outweigh variety.
                        (1.0 + weight).ln() * (1.0 + total / with).ln()
                    })
                    .sum();
                (path.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(limit);
        scored
    }

    /// The definitions in `path`, if it's indexed.
    #[must_use]
    pub fn symbols_in(&self, path: &str) -> &[Symbol] {
        self.files
            .get(path)
            .map_or(&[], |entry| entry.symbols.as_slice())
    }

    /// Loads a saved index. A missing or unreadable file just gives an empty index, since
    /// the cache can always be rebuilt.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let Ok(raw) = fs::read(path) else {
            return Self::default();
        };
        let Ok(Value::Object(saved)) = serde_json::from_slice(&raw) else {
            return Self::default();
        };
        let files = saved
            .into_iter()
            .filter_map(|(name, entry)| Some((name, entry_from_json(&entry)?)))
            .collect();
        Self { files }
    }

    /// Saves the index to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] if the file can't be written.
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        let describe = |source| IndexError {
            path: path.to_owned(),
            source,
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(describe)?;
        }
        let saved: serde_json::Map<String, Value> = self
            .files
            .iter()
            .map(|(name, entry)| (name.clone(), entry_to_json(entry)))
            .collect();
        fs::write(path, Value::Object(saved).to_string()).map_err(describe)
    }
}

fn entry_to_json(entry: &FileEntry) -> Value {
    let symbols: Vec<Value> = entry
        .symbols
        .iter()
        .map(|symbol| json!([symbol.name, symbol.kind, symbol.line]))
        .collect();
    json!({ "bytes": entry.bytes, "modified": entry.modified_millis, "symbols": symbols })
}

fn entry_from_json(entry: &Value) -> Option<FileEntry> {
    let symbols = entry
        .get("symbols")?
        .as_array()?
        .iter()
        .filter_map(|symbol| {
            Some(Symbol {
                name: symbol.get(0)?.as_str()?.to_owned(),
                kind: symbol.get(1)?.as_str()?.to_owned(),
                line: usize::try_from(symbol.get(2)?.as_u64()?).ok()?,
            })
        })
        .collect();
    Some(FileEntry {
        bytes: entry.get("bytes")?.as_u64()?,
        modified_millis: entry.get("modified")?.as_u64()?,
        symbols,
    })
}

/// Splits text into lowercase search terms, breaking on punctuation, `snake_case` and
/// `camelCase`, and dropping very common words.
#[must_use]
pub fn terms(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for character in text.chars() {
        if !character.is_alphanumeric() {
            flush(&mut current, &mut words);
            previous_lower = false;
            continue;
        }
        if character.is_uppercase() && previous_lower {
            flush(&mut current, &mut words);
        }
        previous_lower = character.is_lowercase();
        current.extend(character.to_lowercase());
    }
    flush(&mut current, &mut words);
    words
}

fn flush(current: &mut String, words: &mut Vec<String>) {
    if current.chars().count() >= MIN_TERM_CHARS && !STOP_WORDS.contains(&current.as_str()) {
        words.push(current.clone());
    }
    current.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("ccc-index-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(
            root.join("src/auth.rs"),
            "pub fn verify_token() {}\nstruct Session;\n",
        )
        .unwrap();
        fs::write(
            root.join("src/billing.py"),
            "class Invoice:\n    def charge_card(self):\n        pass\n",
        )
        .unwrap();
        fs::write(root.join("README.md"), "# docs\n").unwrap();
        root
    }

    #[test]
    fn splits_terms_on_case_and_punctuation() {
        assert_eq!(
            terms("parseConfig_file src/main.rs"),
            ["parse", "config", "file", "src", "main"]
        );
        assert_eq!(terms("HTTPServer"), ["httpserver"]);
        assert!(terms("the and for").is_empty());
    }

    #[test]
    fn finds_files_by_symbol_and_path_with_paths_weighing_more() {
        let root = repo("search");
        let mut index = Index::default();
        assert_eq!(index.refresh(&root), 3);

        let by_symbol = index.search("where do we verify the token?", 3);
        assert_eq!(by_symbol[0].0, "src/auth.rs");

        fs::write(root.join("src/verify.rs"), "fn other() {}\n").unwrap();
        index.refresh(&root);
        let ranked = index.search("verify", 3);
        assert_eq!(ranked[0].0, "src/verify.rs", "{ranked:?}");

        assert!(index.search("zebra", 3).is_empty());
        assert_eq!(index.symbols_in("src/billing.py").len(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refreshing_only_parses_what_changed_and_drops_what_is_gone() {
        let root = repo("incremental");
        let mut index = Index::default();
        assert_eq!(index.refresh(&root), 3);
        assert_eq!(index.refresh(&root), 0, "nothing changed");

        fs::write(
            root.join("src/auth.rs"),
            "pub fn renamed_check_function() {}\n",
        )
        .unwrap();
        assert_eq!(index.refresh(&root), 1);
        assert_eq!(
            index.symbols_in("src/auth.rs")[0].name,
            "renamed_check_function"
        );

        fs::remove_file(root.join("README.md")).unwrap();
        assert_eq!(index.refresh(&root), 0);
        assert_eq!(index.len(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn respects_gitignore() {
        let root = repo("ignore");
        fs::write(root.join(".gitignore"), "generated/\n").unwrap();
        fs::create_dir_all(root.join("generated")).unwrap();
        fs::write(root.join("generated/big.rs"), "fn should_not_appear() {}\n").unwrap();
        let mut index = Index::default();
        index.refresh(&root);
        assert!(
            index.search("should appear", 5).is_empty(),
            "{:?}",
            index.search("appear", 5)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn saved_indexes_load_identically_and_bad_files_are_empty() {
        let root = repo("save");
        let mut index = Index::default();
        index.refresh(&root);
        let path = root.join("cache").join("index.json");
        index.save(&path).unwrap();

        let loaded = Index::load(&path);
        assert_eq!(loaded, index);
        let mut again = loaded;
        assert_eq!(again.refresh(&root), 1, "only the new cache file itself");

        fs::write(&path, "not json").unwrap();
        assert!(Index::load(&path).is_empty());
        assert!(Index::load(&root.join("missing")).is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ccc_index::Index;
use ccc_learn::Memory;

/// Marks hints in a prompt, so they read as context and not as the user's words.
const HINT_OPEN: &str = "<ccc-context>";
const HINT_CLOSE: &str = "</ccc-context>";
/// Memory reflects what work actually needed, which beats a word match in a file's name.
const MEMORY_WEIGHT: f64 = 2.0;
/// How many symbols to name per suggested file.
const SYMBOLS_PER_FILE: usize = 3;

/// Combines the repository index and past sessions into file suggestions.
#[derive(Debug)]
pub struct Suggester {
    root: PathBuf,
    index: Mutex<Index>,
    memory: Option<Arc<Memory>>,
}

impl Suggester {
    /// A suggester for the repository at `root`. Call [`Suggester::refresh`] to populate it.
    #[must_use]
    pub fn new(root: PathBuf, index: Index, memory: Option<Arc<Memory>>) -> Self {
        Self {
            root,
            index: Mutex::new(index),
            memory,
        }
    }

    /// Picks up changed files, returning how many were parsed.
    pub fn refresh(&self) -> usize {
        let mut index = self.index.lock().expect("lock poisoned").clone();
        let parsed = index.refresh(&self.root);
        *self.index.lock().expect("lock poisoned") = index;
        parsed
    }

    /// The likeliest files for `prompt`, best first.
    #[must_use]
    pub fn suggest(&self, prompt: &str, limit: usize) -> Vec<String> {
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        let index = self.index.lock().expect("lock poisoned");
        for (file, score) in index.search(prompt, limit * 2) {
            *scores.entry(file).or_default() += score;
        }
        if let Some(memory) = &self.memory {
            let repo = self.root.to_string_lossy();
            match memory.suggest(&repo, prompt, limit * 2) {
                Ok(remembered) => {
                    for (file, score) in remembered {
                        *scores.entry(file).or_default() += score * MEMORY_WEIGHT;
                    }
                }
                Err(err) => tracing::warn!(error = %err, "could not read memory"),
            }
        }
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(limit);
        ranked.into_iter().map(|(file, _)| file).collect()
    }

    /// Hints for `prompt`, or `None` if nothing looks relevant.
    #[must_use]
    pub fn hints(&self, prompt: &str, limit: usize) -> Option<String> {
        let files = self.suggest(prompt, limit);
        if files.is_empty() {
            return None;
        }
        let index = self.index.lock().expect("lock poisoned");
        let described: Vec<(String, Vec<String>)> = files
            .into_iter()
            .map(|file| {
                let symbols = index
                    .symbols_in(&file)
                    .iter()
                    .take(SYMBOLS_PER_FILE)
                    .map(|symbol| symbol.name.clone())
                    .collect();
                (file, symbols)
            })
            .collect();
        Some(hint_text(&described))
    }
}

/// Formats suggestions for the model, marked as possibly useful and not instructions.
#[must_use]
pub fn hint_text(files: &[(String, Vec<String>)]) -> String {
    let mut text = format!(
        "{HINT_OPEN}\nFiles that may be relevant, suggested automatically from a local index \
         and past sessions. They may be wrong; ignore them if they don't fit.\n"
    );
    for (file, symbols) in files {
        if symbols.is_empty() {
            text.push_str(&format!("- {file}\n"));
        } else {
            text.push_str(&format!("- {file} ({})\n", symbols.join(", ")));
        }
    }
    text.push_str(HINT_CLOSE);
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_learn::Turn;
    use std::collections::BTreeSet;
    use std::fs;

    fn repo() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ccc-suggest-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("auth.rs"),
            "pub fn verify_token() {}\nfn refresh_session() {}\n",
        )
        .unwrap();
        fs::write(root.join("deploy.sh"), "echo unrelated\n").unwrap();
        root
    }

    #[test]
    fn finds_by_index_and_names_symbols() {
        let root = repo();
        let suggester = Suggester::new(root.clone(), Index::default(), None);
        assert_eq!(suggester.refresh(), 2);

        let hints = suggester.hints("verify the token", 3).unwrap();

        assert!(
            hints.contains("auth.rs (verify_token, refresh_session)"),
            "{hints}"
        );
        assert!(!hints.contains("deploy.sh"), "{hints}");
        assert_eq!(suggester.hints("zebra", 3), None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn what_past_work_needed_outranks_a_name_match() {
        let root = repo();
        let memory = Arc::new(Memory::in_memory().unwrap());
        memory
            .learn(&[Turn {
                repo: root.to_string_lossy().into_owned(),
                at: "t".to_owned(),
                prompt: "deploy the auth change".to_owned(),
                files: BTreeSet::from(["deploy.sh".to_owned()]),
            }])
            .unwrap();
        let suggester = Suggester::new(root.clone(), Index::default(), Some(memory));
        suggester.refresh();

        assert_eq!(suggester.suggest("deploy auth", 1), ["deploy.sh"]);
        fs::remove_dir_all(root).unwrap();
    }
}

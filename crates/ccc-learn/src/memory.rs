use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ccc_index::terms;
use rusqlite::{Connection, params};

use crate::turns::Turn;

/// Files touched more often than this in one turn are a sweep, which says little about
/// what any one word in the prompt meant.
const MAX_FILES_PER_TURN: usize = 12;

/// The memory couldn't be opened or read.
#[derive(Debug, thiserror::Error)]
#[error("opening memory at {path:?}")]
pub struct OpenMemoryError {
    path: PathBuf,
    #[source]
    source: Source,
}

#[derive(Debug, thiserror::Error)]
enum Source {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
}

impl OpenMemoryError {
    pub(crate) fn read(path: &Path, source: std::io::Error) -> Self {
        Self {
            path: path.to_owned(),
            source: source.into(),
        }
    }
}

/// What past sessions in each repository needed.
pub struct Memory {
    connection: Mutex<Connection>,
}

impl fmt::Debug for Memory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Memory").finish_non_exhaustive()
    }
}

impl Memory {
    /// Opens the memory at `path`, creating it if needed.
    ///
    /// # Errors
    ///
    /// Returns [`OpenMemoryError`] if the database can't be opened or prepared.
    pub fn open(path: &Path) -> Result<Self, OpenMemoryError> {
        let describe = |source: Source| OpenMemoryError {
            path: path.to_owned(),
            source,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| describe(err.into()))?;
        }
        let connection = Connection::open(path).map_err(|err| describe(err.into()))?;
        Self::prepare(connection).map_err(|err| describe(err.into()))
    }

    /// A memory that lasts until it's dropped.
    ///
    /// # Errors
    ///
    /// Returns [`OpenMemoryError`] if the database can't be prepared.
    pub fn in_memory() -> Result<Self, OpenMemoryError> {
        let describe = |source: rusqlite::Error| OpenMemoryError {
            path: PathBuf::from(":memory:"),
            source: source.into(),
        };
        let connection = Connection::open_in_memory().map_err(describe)?;
        Self::prepare(connection).map_err(describe)
    }

    fn prepare(connection: Connection) -> Result<Self, rusqlite::Error> {
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS term_files (
                 repo TEXT NOT NULL, term TEXT NOT NULL, file TEXT NOT NULL, uses INTEGER NOT NULL,
                 PRIMARY KEY (repo, term, file)
             );
             CREATE TABLE IF NOT EXISTS learned_turns (repo TEXT NOT NULL, at TEXT NOT NULL, prompt TEXT NOT NULL,
                 PRIMARY KEY (repo, at, prompt));",
        )?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// Learns from `turns`, skipping any already learned so a job can run repeatedly.
    /// Stores only search terms and file paths, not prompts' words in order. Returns how many
    /// new turns it learned from.
    ///
    /// # Errors
    ///
    /// Returns the database error if writing fails.
    pub fn learn(&self, turns: &[Turn]) -> Result<usize, rusqlite::Error> {
        let mut connection = self.connection.lock().expect("lock poisoned");
        let transaction = connection.transaction()?;
        let mut learned = 0;
        for turn in turns {
            if turn.files.is_empty() || turn.files.len() > MAX_FILES_PER_TURN {
                continue;
            }
            let key = prompt_key(&turn.prompt);
            let new = transaction.execute(
                "INSERT OR IGNORE INTO learned_turns VALUES (?1, ?2, ?3)",
                params![turn.repo, turn.at, key],
            )?;
            if new == 0 {
                continue;
            }
            learned += 1;
            for term in terms(&turn.prompt) {
                for file in &turn.files {
                    transaction.execute(
                        "INSERT INTO term_files VALUES (?1, ?2, ?3, 1)
                         ON CONFLICT (repo, term, file) DO UPDATE SET uses = uses + 1",
                        params![turn.repo, term, file],
                    )?;
                }
            }
        }
        transaction.commit()?;
        Ok(learned)
    }

    /// The files past work in `repo` most associated with `prompt`, best first. Words used
    /// with many different files count for less, since they say little.
    ///
    /// # Errors
    ///
    /// Returns the database error if the query fails.
    pub fn suggest(
        &self,
        repo: &str,
        prompt: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, rusqlite::Error> {
        let connection = self.connection.lock().expect("lock poisoned");
        let mut statement = connection
            .prepare_cached("SELECT file, uses FROM term_files WHERE repo = ?1 AND term = ?2")?;
        let mut scores: HashMap<String, f64> = HashMap::new();
        for term in terms(prompt) {
            let rows: Vec<(String, i64)> = statement
                .query_map(params![repo, term], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<_, _>>()?;
            let spread = rows.len() as f64;
            for (file, uses) in rows {
                *scores.entry(file).or_default() += (1.0 + uses as f64).ln() / (1.0 + spread).ln();
            }
        }
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(limit);
        Ok(ranked)
    }
}

/// A stand-in for the prompt's text, so the record of what's been learned holds no prompts.
fn prompt_key(prompt: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prompt.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn turn(repo: &str, prompt: &str, files: &[&str]) -> Turn {
        Turn {
            repo: repo.to_owned(),
            at: format!("t-{prompt}"),
            prompt: prompt.to_owned(),
            files: files
                .iter()
                .map(|file| (*file).to_owned())
                .collect::<BTreeSet<_>>(),
        }
    }

    #[test]
    fn suggests_files_past_prompts_with_similar_words_needed() {
        let memory = Memory::in_memory().unwrap();
        memory
            .learn(&[
                turn(
                    "/r",
                    "fix the checkout flow",
                    &["src/checkout.ts", "src/cart.ts"],
                ),
                turn("/r", "checkout tax rounding", &["src/checkout.ts"]),
                turn("/r", "login page styling", &["src/login.css"]),
            ])
            .unwrap();

        let suggested = memory.suggest("/r", "checkout is broken", 3).unwrap();

        assert_eq!(suggested[0].0, "src/checkout.ts", "{suggested:?}");
        assert!(suggested.iter().all(|(file, _)| file != "src/login.css"));
        assert!(
            memory.suggest("/other", "checkout", 3).unwrap().is_empty(),
            "repos are separate"
        );
        assert!(memory.suggest("/r", "zebra", 3).unwrap().is_empty());
    }

    #[test]
    fn learning_twice_counts_once() {
        let memory = Memory::in_memory().unwrap();
        let turns = [turn("/r", "fix checkout", &["a.ts"])];
        assert_eq!(memory.learn(&turns).unwrap(), 1);
        assert_eq!(memory.learn(&turns).unwrap(), 0);
    }

    #[test]
    fn broad_sweeps_and_turns_without_files_teach_nothing() {
        let memory = Memory::in_memory().unwrap();
        let many: Vec<String> = (0..30).map(|i| format!("f{i}.rs")).collect();
        let many: Vec<&str> = many.iter().map(String::as_str).collect();
        memory
            .learn(&[
                turn("/r", "rename widget", &many),
                turn("/r", "question widget", &[]),
            ])
            .unwrap();
        assert!(memory.suggest("/r", "widget", 3).unwrap().is_empty());
    }

    #[test]
    fn the_store_keeps_no_prompts() {
        let memory = Memory::in_memory().unwrap();
        memory
            .learn(&[turn("/r", "secret-project-name launch", &["a.ts"])])
            .unwrap();
        let connection = memory.connection.lock().unwrap();
        let stored: String = connection
            .query_row("SELECT prompt FROM learned_turns", [], |row| row.get(0))
            .unwrap();
        assert!(!stored.contains("launch"), "{stored}");
    }
}

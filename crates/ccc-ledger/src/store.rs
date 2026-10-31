use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{Connection, params};

use crate::price::cost_usd;
use crate::usage::Usage;

/// The ledger's database couldn't be opened or prepared.
#[derive(Debug, thiserror::Error)]
#[error("opening ledger {path:?}")]
pub struct OpenLedgerError {
    path: PathBuf,
    #[source]
    source: LedgerSource,
}

#[derive(Debug, thiserror::Error)]
enum LedgerSource {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
}

/// One session's totals.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionTotal {
    /// The session's identifier.
    pub session: String,
    /// When its most recent call happened, in milliseconds since the Unix epoch.
    pub last_at_millis: i64,
    /// Number of API calls.
    pub calls: u64,
    /// Sum of context tokens across calls.
    pub context_tokens: u64,
    /// Tokens generated.
    pub output_tokens: u64,
    /// Cost at list prices. Calls to models without a known price count as zero.
    pub cost_usd: f64,
}

/// A local record of every call's usage and cost.
pub struct Ledger {
    connection: Mutex<Connection>,
}

impl fmt::Debug for Ledger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Ledger").finish_non_exhaustive()
    }
}

impl Ledger {
    /// Opens the ledger at `path`, creating the file and any parent directories.
    ///
    /// # Errors
    ///
    /// Returns [`OpenLedgerError`] if the database can't be opened or prepared.
    pub fn open(path: &Path) -> Result<Self, OpenLedgerError> {
        let describe = |source: LedgerSource| OpenLedgerError {
            path: path.to_owned(),
            source,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| describe(err.into()))?;
        }
        let connection = Connection::open(path).map_err(|err| describe(err.into()))?;
        Self::prepare(connection).map_err(|err| describe(err.into()))
    }

    /// A ledger that lasts until it is dropped, for tests and when no state directory exists.
    ///
    /// # Errors
    ///
    /// Returns [`OpenLedgerError`] if the database can't be prepared.
    pub fn in_memory() -> Result<Self, OpenLedgerError> {
        let describe = |source: rusqlite::Error| OpenLedgerError {
            path: PathBuf::from(":memory:"),
            source: source.into(),
        };
        let connection = Connection::open_in_memory().map_err(describe)?;
        Self::prepare(connection).map_err(describe)
    }

    fn prepare(connection: Connection) -> Result<Self, rusqlite::Error> {
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS calls (
                 session TEXT NOT NULL,
                 at_millis INTEGER NOT NULL,
                 model TEXT NOT NULL,
                 input INTEGER NOT NULL,
                 cache_read INTEGER NOT NULL,
                 cache_write_5m INTEGER NOT NULL,
                 cache_write_1h INTEGER NOT NULL,
                 output INTEGER NOT NULL,
                 cost_usd REAL NOT NULL
             );
             CREATE INDEX IF NOT EXISTS calls_by_session ON calls (session);",
        )?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// Records one call, returning its cost.
    ///
    /// # Errors
    ///
    /// Returns the database error if the row can't be written.
    pub fn record(
        &self,
        session: &str,
        at_millis: i64,
        model: &str,
        usage: &Usage,
    ) -> Result<f64, rusqlite::Error> {
        let cost = cost_usd(model, usage).unwrap_or(0.0);
        let as_stored = |count: u64| i64::try_from(count).unwrap_or(i64::MAX);
        self.connection.lock().expect("lock poisoned").execute(
            "INSERT INTO calls VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session,
                at_millis,
                model,
                as_stored(usage.input),
                as_stored(usage.cache_read),
                as_stored(usage.cache_write_5m),
                as_stored(usage.cache_write_1h),
                as_stored(usage.output),
                cost,
            ],
        )?;
        Ok(cost)
    }

    /// What `session` has cost so far.
    ///
    /// # Errors
    ///
    /// Returns the database error if the query fails.
    pub fn session_cost(&self, session: &str) -> Result<f64, rusqlite::Error> {
        self.connection.lock().expect("lock poisoned").query_row(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE session = ?1",
            params![session],
            |row| row.get(0),
        )
    }

    /// Totals for the most recently active sessions, newest first.
    ///
    /// # Errors
    ///
    /// Returns the database error if the query fails.
    pub fn recent_sessions(&self, limit: usize) -> Result<Vec<SessionTotal>, rusqlite::Error> {
        let connection = self.connection.lock().expect("lock poisoned");
        let mut statement = connection.prepare(
            "SELECT session, MAX(at_millis), COUNT(*),
                    SUM(input + cache_read + cache_write_5m + cache_write_1h),
                    SUM(output), SUM(cost_usd)
             FROM calls GROUP BY session ORDER BY MAX(at_millis) DESC LIMIT ?1",
        )?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows = statement.query_map(params![limit], |row| {
            let count = |index: usize| -> Result<u64, rusqlite::Error> {
                let stored: i64 = row.get(index)?;
                Ok(u64::try_from(stored).unwrap_or(0))
            };
            Ok(SessionTotal {
                session: row.get(0)?,
                last_at_millis: row.get(1)?,
                calls: count(2)?,
                context_tokens: count(3)?,
                output_tokens: count(4)?,
                cost_usd: row.get(5)?,
            })
        })?;
        rows.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(read: u64, output: u64) -> Usage {
        Usage {
            cache_read: read,
            output,
            ..Usage::default()
        }
    }

    #[test]
    fn totals_add_up_per_session_newest_first() {
        let ledger = Ledger::in_memory().unwrap();
        ledger
            .record("a", 1, "claude-opus-5-5", &usage(1_000_000, 0))
            .unwrap();
        ledger
            .record("a", 2, "claude-opus-5-5", &usage(0, 1_000_000))
            .unwrap();
        ledger
            .record("b", 3, "claude-opus-5-5", &usage(1_000_000, 0))
            .unwrap();

        assert!((ledger.session_cost("a").unwrap() - 20.2).abs() < 1e-9);
        assert_eq!(ledger.session_cost("absent").unwrap(), 0.0);

        let sessions = ledger.recent_sessions(10).unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session, "b");
        assert_eq!(sessions[1].calls, 2);
        assert_eq!(sessions[1].context_tokens, 1_000_000);
    }

    #[test]
    fn unpriced_models_are_recorded_at_zero() {
        let ledger = Ledger::in_memory().unwrap();
        assert_eq!(
            ledger.record("a", 1, "local:qwen3", &usage(5, 5)).unwrap(),
            0.0
        );
        assert_eq!(ledger.recent_sessions(1).unwrap()[0].calls, 1);
    }

    #[test]
    fn persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!("ccc-ledger-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("ledger.sqlite");
        Ledger::open(&path)
            .unwrap()
            .record("a", 1, "claude-opus-5-5", &usage(1_000_000, 0))
            .unwrap();
        assert!(Ledger::open(&path).unwrap().session_cost("a").unwrap() > 0.0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

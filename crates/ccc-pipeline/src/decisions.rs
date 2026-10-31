//! Saved outcomes for stages that aren't pure functions.
//!
//! Compiling must give the same bytes for a message every time. A model, a growing index or
//! a time limit can't promise that, so such a stage decides once, on the newest message, and
//! saves the outcome before the request goes out. After that it only replays the saved value.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use serde_json::{Value, json};

/// Decisions saved as one small file each, so they survive restarts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decisions {
    dir: PathBuf,
}

impl Decisions {
    /// A store in `dir`, which is created when first needed.
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Looks up a decision. The outer `Option` says whether one exists, and the inner one
    /// is the decision itself, where `None` means "leave it alone".
    #[must_use]
    pub fn get(&self, parts: &[&str]) -> Option<Option<String>> {
        let stored: Value = serde_json::from_slice(&fs::read(self.path(parts)).ok()?).ok()?;
        Some(
            stored
                .get("value")
                .and_then(Value::as_str)
                .map(str::to_owned),
        )
    }

    /// Saves a decision and returns whether that worked. If it didn't, the caller must not
    /// act on the decision, because a later request couldn't reproduce it.
    #[must_use]
    pub fn save(&self, parts: &[&str], value: Option<&str>) -> bool {
        let record = json!({ "value": value }).to_string();
        let saved =
            fs::create_dir_all(&self.dir).and_then(|()| fs::write(self.path(parts), record));
        if let Err(err) = &saved {
            tracing::warn!(error = %err, "could not save a decision");
        }
        saved.is_ok()
    }

    fn path(&self, parts: &[&str]) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        parts.hash(&mut hasher);
        self.dir.join(format!("{:016x}.json", hasher.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decisions_round_trip_and_distinguish_no_decision_from_leave_alone() {
        let dir = std::env::temp_dir().join(format!("ccc-decisions-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let decisions = Decisions::new(dir.clone());

        assert_eq!(decisions.get(&["a"]), None);
        assert!(decisions.save(&["a"], None));
        assert_eq!(decisions.get(&["a"]), Some(None));
        assert!(decisions.save(&["b", "c"], Some("kept")));
        assert_eq!(decisions.get(&["b", "c"]), Some(Some("kept".to_owned())));
        assert_eq!(decisions.get(&["bc"]), None, "parts stay distinct");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_unwritable_store_reports_failure() {
        let decisions = Decisions::new(PathBuf::from("/dev/null/not-a-dir"));
        assert!(!decisions.save(&["a"], Some("x")));
    }
}

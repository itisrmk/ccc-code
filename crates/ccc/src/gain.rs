//! `ccc gain`: what the local model has kept away from Claude, stated honestly.
//!
//! Only two things get recorded. Tokens removed from what would have been sent are measured.
//! Tokens a local answer read instead of Claude are an estimate, since nobody can know whether
//! Claude would have read them, so the two are never added together without saying so.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use serde_json::{Value, json};

const FILE: &str = "gain.jsonl";
const SECONDS_PER_DAY: u64 = 86_400;
const WEEK_DAYS: u64 = 7;

/// How a saving is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Basis {
    /// Output that was shortened before Claude saw it.
    Trimmed,
    /// Code a local model read, in place of Claude possibly reading it.
    LocalAnswer,
}

impl Basis {
    const fn key(self) -> &'static str {
        match self {
            Self::Trimmed => "trimmed",
            Self::LocalAnswer => "local_answer",
        }
    }
}

fn today() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() / SECONDS_PER_DAY)
}

fn path(state_dir: &Path) -> PathBuf {
    state_dir.join(FILE)
}

/// Records a saving. A failure here must never get in the way of the work it describes.
pub(crate) fn record(state_dir: &Path, basis: Basis, tokens: u64) {
    if tokens == 0 {
        return;
    }
    let entry = json!({ "day": today(), "basis": basis.key(), "tokens": tokens });
    let written = fs::create_dir_all(state_dir).and_then(|()| {
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path(state_dir))?;
        writeln!(log, "{entry}")
    });
    if let Err(err) = written {
        tracing::warn!(error = %err, "could not record a saving");
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Totals {
    trimmed: u64,
    local_answer: u64,
    uses: u64,
}

impl Totals {
    fn all(&self) -> u64 {
        self.trimmed + self.local_answer
    }
}

/// Adds up entries on or after `since_day`. Damaged lines are skipped.
fn totals(text: &str, since_day: u64) -> Totals {
    let mut totals = Totals::default();
    for entry in text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
    {
        let day = entry.get("day").and_then(Value::as_u64).unwrap_or(0);
        let tokens = entry.get("tokens").and_then(Value::as_u64).unwrap_or(0);
        if day < since_day {
            continue;
        }
        totals.uses += 1;
        match entry.get("basis").and_then(Value::as_str) {
            Some("trimmed") => totals.trimmed += tokens,
            Some("local_answer") => totals.local_answer += tokens,
            _ => {}
        }
    }
    totals
}

/// What's been kept away today, for the status line.
pub(crate) fn today_tokens(state_dir: &Path) -> u64 {
    fs::read_to_string(path(state_dir)).map_or(0, |text| totals(&text, today()).all())
}

fn describe(label: &str, totals: Totals) -> String {
    format!(
        "{label:<14}{:>10} trimmed{:>10} read locally (estimate){:>7} uses",
        totals.trimmed, totals.local_answer, totals.uses
    )
}

pub(crate) fn run(state_dir: &Path) -> anyhow::Result<()> {
    let text = match fs::read_to_string(path(state_dir)) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => return Err(err).context("reading the record"),
    };
    let mut out = std::io::stdout();
    if text.trim().is_empty() {
        writeln!(
            out,
            "Nothing yet. This fills in as the local model does work in your sessions.\n\
             For what your sessions have cost so far, run: ccc whatif"
        )?;
        return Ok(());
    }
    let day = today();
    writeln!(out, "Tokens kept out of Claude's context\n")?;
    writeln!(out, "{}", describe("today", totals(&text, day)))?;
    writeln!(
        out,
        "{}",
        describe(
            "last 7 days",
            totals(&text, day.saturating_sub(WEEK_DAYS - 1))
        )
    )?;
    writeln!(out, "{}", describe("all time", totals(&text, 0)))?;
    writeln!(
        out,
        "\nTrimmed tokens are measured. Tokens read locally are an estimate, since Claude might \
         not have read them anyway.\nNeither is a saving on your bill or limit in the same \
         proportion: most of what you pay for is re-reading context from the cache. `ccc whatif` \
         shows that breakdown."
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_up_by_basis_and_date_and_skips_damage() {
        let text = [
            r#"{"day":10,"basis":"trimmed","tokens":100}"#,
            r#"{"day":10,"basis":"local_answer","tokens":40}"#,
            "not json",
            r#"{"day":3,"basis":"trimmed","tokens":999}"#,
        ]
        .join("\n");
        let recent = totals(&text, 10);
        assert_eq!(
            (recent.trimmed, recent.local_answer, recent.uses),
            (100, 40, 2)
        );
        assert_eq!(recent.all(), 140);
        assert_eq!(totals(&text, 0).trimmed, 1099);
    }

    #[test]
    fn recording_shows_up_today_and_zero_is_ignored() {
        let dir = std::env::temp_dir().join(format!("ccc-gain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(today_tokens(&dir), 0, "no file yet");
        record(&dir, Basis::Trimmed, 0);
        assert!(!path(&dir).exists());
        record(&dir, Basis::Trimmed, 700);
        record(&dir, Basis::LocalAnswer, 300);
        assert_eq!(today_tokens(&dir), 1000);
        fs::remove_dir_all(&dir).unwrap();
    }
}

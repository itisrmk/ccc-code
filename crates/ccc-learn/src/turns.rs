//! Reads what each user prompt led to from transcript files.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::memory::OpenMemoryError;

/// Paths named in shell commands, so sessions that read files through Bash still count.
static PATH_IN_COMMAND: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[\s'\x22=])((?:\.{0,2}/)?[\w@.-]+(?:/[\w@.-]+)*\.[A-Za-z][A-Za-z0-9]{0,5})\b")
        .expect("path pattern is valid")
});

/// A user prompt and the files the work on it touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    /// The repository, taken from the session's working directory.
    pub repo: String,
    /// When the prompt was made, as recorded. Sorts chronologically.
    pub at: String,
    /// What the user asked.
    pub prompt: String,
    /// Files touched, relative to `repo` where possible.
    pub files: BTreeSet<String>,
}

/// Reads every turn from `.jsonl` transcripts under `dir`, oldest first.
///
/// # Errors
///
/// Returns [`OpenMemoryError`] if the directory tree can't be read.
pub fn read_turns(dir: &Path) -> Result<Vec<Turn>, OpenMemoryError> {
    let mut paths = Vec::new();
    collect(dir, &mut paths)?;
    let mut turns = Vec::new();
    for path in paths {
        // Subagent transcripts have no user prompts of their own worth learning from.
        if !path
            .components()
            .any(|part| part.as_os_str() == "subagents")
        {
            read_file(&path, &mut turns);
        }
    }
    turns.sort_by(|a, b| a.at.cmp(&b.at));
    Ok(turns)
}

fn collect(dir: &Path, found: &mut Vec<std::path::PathBuf>) -> Result<(), OpenMemoryError> {
    let describe = |source| OpenMemoryError::read(dir, source);
    for entry in fs::read_dir(dir).map_err(describe)? {
        let path = entry.map_err(describe)?.path();
        if path.is_dir() {
            collect(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            found.push(path);
        }
    }
    Ok(())
}

fn read_file(path: &Path, turns: &mut Vec<Turn>) {
    let Ok(file) = File::open(path) else {
        return;
    };
    let mut current: Option<Turn> = None;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match record.get("type").and_then(Value::as_str) {
            Some("user") => {
                if let Some(prompt) = typed_prompt(&record) {
                    turns.extend(current.take());
                    current = Some(Turn {
                        repo: text_field(&record, "cwd"),
                        at: text_field(&record, "timestamp"),
                        prompt,
                        files: BTreeSet::new(),
                    });
                }
            }
            Some("assistant") => {
                if let Some(turn) = &mut current {
                    add_touched_files(&record, turn);
                }
            }
            _ => {}
        }
    }
    turns.extend(current);
}

fn text_field(record: &Value, name: &str) -> String {
    record
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The words a person typed, if this record is one, skipping tool results and the
/// reminders and command output Claude Code wraps in tags.
fn typed_prompt(record: &Value) -> Option<String> {
    if record.get("isMeta").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let content = record.pointer("/message/content")?;
    let text = match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => {
            let parts: Vec<&str> = blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect();
            parts.join("\n")
        }
        _ => return None,
    };
    let text = text.trim();
    (!text.is_empty() && !text.starts_with('<')).then(|| text.to_owned())
}

fn add_touched_files(record: &Value, turn: &mut Turn) {
    let Some(blocks) = record.pointer("/message/content").and_then(Value::as_array) else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("tool_use") {
            continue;
        }
        let input = block.get("input");
        match block.get("name").and_then(Value::as_str) {
            Some("Read" | "Edit" | "Write" | "NotebookEdit") => {
                let path = input
                    .and_then(|input| input.get("file_path"))
                    .and_then(Value::as_str);
                if let Some(path) = path {
                    turn.files.insert(relative_to(&turn.repo, path));
                }
            }
            Some("Bash") => {
                let command = input
                    .and_then(|input| input.get("command"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let body = command.split("<<").next().unwrap_or(command);
                for captures in PATH_IN_COMMAND.captures_iter(body) {
                    turn.files.insert(relative_to(&turn.repo, &captures[1]));
                }
            }
            _ => {}
        }
    }
}

fn relative_to(repo: &str, path: &str) -> String {
    let without_dot = path.strip_prefix("./").unwrap_or(path);
    without_dot
        .strip_prefix(repo)
        .map_or(without_dot, |rest| rest.trim_start_matches('/'))
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn write_transcript(name: &str, records: &[Value]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ccc-turns-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut file = File::create(dir.join("s.jsonl")).unwrap();
        for record in records {
            writeln!(file, "{record}").unwrap();
        }
        dir
    }

    fn user(text: &str, at: &str) -> Value {
        json!({ "type": "user", "cwd": "/repo", "timestamp": at, "message": { "content": text } })
    }

    fn tool(name: &str, input: Value) -> Value {
        json!({ "type": "assistant", "message": { "content": [
            { "type": "tool_use", "name": name, "input": input }
        ] } })
    }

    #[test]
    fn ties_each_prompt_to_the_files_its_work_touched() {
        let dir = write_transcript(
            "basic",
            &[
                user("fix login", "2026-01-02"),
                tool("Read", json!({ "file_path": "/repo/src/auth.rs" })),
                tool("Edit", json!({ "file_path": "/elsewhere/x.rs" })),
                tool(
                    "Bash",
                    json!({ "command": "sed -n '1,9p' src/session.ts && cargo test" }),
                ),
                json!({ "type": "user", "cwd": "/repo", "message": { "content": [
                { "type": "tool_result", "tool_use_id": "a", "content": "output" }
            ] } }),
                tool("Write", json!({ "file_path": "/repo/README.md" })),
                user("now billing", "2026-01-01"),
                tool("Read", json!({ "file_path": "/repo/bill.py" })),
            ],
        );

        let turns = read_turns(&dir).unwrap();

        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].prompt, "now billing", "oldest first");
        assert_eq!(turns[1].prompt, "fix login");
        let files: Vec<&str> = turns[1].files.iter().map(String::as_str).collect();
        assert_eq!(
            files,
            [
                "/elsewhere/x.rs",
                "README.md",
                "src/auth.rs",
                "src/session.ts"
            ]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn skips_injected_text_and_heredoc_bodies() {
        let dir = write_transcript(
            "skip",
            &[
                user("<system-reminder>not typed</system-reminder>", "2026-01-01"),
                json!({ "type": "user", "isMeta": true, "cwd": "/repo", "message": { "content": "meta" } }),
                user("real prompt", "2026-01-02"),
                tool(
                    "Bash",
                    json!({ "command": "python3 - <<'EOF'\nopen('ghost.py')\nEOF" }),
                ),
            ],
        );
        let turns = read_turns(&dir).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].prompt, "real prompt");
        assert!(turns[0].files.is_empty(), "{:?}", turns[0].files);
        fs::remove_dir_all(dir).unwrap();
    }
}

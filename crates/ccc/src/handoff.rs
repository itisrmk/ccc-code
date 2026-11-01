//! `ccc handoff`: carries a session's essentials into a fresh one.
//!
//! Long sessions cost more per call, but restarting by hand means retyping everything that
//! mattered. This builds the summary from the transcript without calling a model, so it's free
//! and instant, and it says plainly that it's partial so the new session checks before it trusts.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context as _, bail};
use clap::Args;
use serde_json::Value;

/// Earlier requests to carry over, newest last.
const PROMPTS_KEPT: usize = 6;
/// Longest a single request can be before it's shortened.
const PROMPT_MAX_CHARS: usize = 1500;
/// Longest the last answer can be before it's shortened.
const ANSWER_MAX_CHARS: usize = 3000;
/// Most files to list.
const FILES_KEPT: usize = 40;

#[derive(Debug, Args)]
pub(crate) struct HandoffArgs {
    /// The transcript to read. Defaults to the newest session for this directory.
    #[arg(long, value_name = "FILE")]
    transcript: Option<PathBuf>,
    /// Write the handoff here instead of printing it.
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Start Claude Code with the handoff as its first message.
    #[arg(long, conflicts_with = "out")]
    start: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Handoff {
    first_prompt: Option<String>,
    recent_prompts: Vec<String>,
    /// Files changed, with how many times.
    files: BTreeMap<String, usize>,
    last_answer: Option<String>,
}

fn shortened(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    // Keeps the end of a long text as well, since that's usually where the point is.
    let head: String = text.chars().take(limit / 2).collect();
    let tail: Vec<char> = text.chars().rev().take(limit / 2).collect();
    let tail: String = tail.into_iter().rev().collect();
    format!("{head}\n[...]\n{tail}")
}

/// Claude Code adds its own messages to the record, and they aren't things the person asked for.
fn is_person_speaking(record: &Value, text: &str) -> bool {
    let flagged = ["isMeta", "isSidechain", "isCompactSummary"]
        .iter()
        .any(|key| record.get(*key).and_then(Value::as_bool) == Some(true));
    !flagged && !text.trim().is_empty() && !text.trim_start().starts_with('<')
}

fn add_user(handoff: &mut Handoff, record: &Value) {
    let text = match record.pointer("/message/content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            // A message holding a tool result is mid-turn, whatever text sits beside it.
            if blocks.iter().any(|b| b["type"] == "tool_result") {
                return;
            }
            blocks
                .iter()
                .filter(|block| block["type"] == "text")
                .filter_map(|block| block["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => return,
    };
    if !is_person_speaking(record, &text) {
        return;
    }
    if handoff.first_prompt.is_none() {
        handoff.first_prompt = Some(text);
    } else {
        handoff.recent_prompts.push(text);
    }
}

fn add_assistant(handoff: &mut Handoff, record: &Value) {
    if record.get("isSidechain").and_then(Value::as_bool) == Some(true) {
        return;
    }
    let Some(blocks) = record.pointer("/message/content").and_then(Value::as_array) else {
        return;
    };
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(text) = block["text"]
                    .as_str()
                    .filter(|text| !text.trim().is_empty())
                {
                    handoff.last_answer = Some(text.to_owned());
                }
            }
            Some("tool_use") if matches!(block["name"].as_str(), Some("Edit" | "Write")) => {
                if let Some(path) = block.pointer("/input/file_path").and_then(Value::as_str) {
                    *handoff.files.entry(path.to_owned()).or_default() += 1;
                }
            }
            _ => {}
        }
    }
}

fn read(path: &Path) -> anyhow::Result<Handoff> {
    let file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut handoff = Handoff::default();
    for line in BufReader::new(file).lines() {
        // A session interrupted mid-write can end in a truncated or undecodable line.
        let Ok(line) = line else { continue };
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match record.get("type").and_then(Value::as_str) {
            Some("user") => add_user(&mut handoff, &record),
            Some("assistant") => add_assistant(&mut handoff, &record),
            _ => {}
        }
    }
    Ok(handoff)
}

fn render(handoff: &Handoff) -> String {
    let mut text = String::from(
        "This continues earlier work from a session that got too long. This summary was built \
         automatically and is partial. Check the code and run the tests rather than trusting it, \
         and don't redo finished work.\n",
    );
    // Writing to a String can't fail.
    if let Some(first) = &handoff.first_prompt {
        let _ = write!(
            text,
            "\n## How the work began\n\n{}\n",
            shortened(first, PROMPT_MAX_CHARS)
        );
    }
    let skip = handoff.recent_prompts.len().saturating_sub(PROMPTS_KEPT);
    if handoff.recent_prompts.len() > PROMPTS_KEPT {
        let _ = write!(text, "\n({skip} earlier requests are left out.)\n");
    }
    if !handoff.recent_prompts.is_empty() {
        text.push_str("\n## Requests since, oldest first\n\n");
        for prompt in &handoff.recent_prompts[skip..] {
            let _ = writeln!(text, "- {}", shortened(prompt, PROMPT_MAX_CHARS));
        }
    }
    if !handoff.files.is_empty() {
        text.push_str("\n## Files changed so far\n\n");
        let mut files: Vec<(&String, &usize)> = handoff.files.iter().collect();
        files.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (path, _) in files.iter().take(FILES_KEPT) {
            let _ = writeln!(text, "- {path}");
        }
        if files.len() > FILES_KEPT {
            let _ = writeln!(text, "- and {} more", files.len() - FILES_KEPT);
        }
    }
    if let Some(answer) = &handoff.last_answer {
        let _ = write!(
            text,
            "\n## The last thing said before this handoff\n\n{}\n",
            shortened(answer, ANSWER_MAX_CHARS)
        );
    }
    text
}

/// Claude Code's folder name for a directory.
fn project_folder(directory: &Path) -> String {
    directory
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn newest_transcript(folder: &Path) -> anyhow::Result<PathBuf> {
    let entries = fs::read_dir(folder).with_context(|| {
        format!(
            "no sessions found for this directory in {}",
            folder.display()
        )
    })?;
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .filter_map(|path| Some((fs::metadata(&path).ok()?.modified().ok()?, path)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, path)| path)
        .with_context(|| format!("no sessions found in {}", folder.display()))
}

pub(crate) fn run(args: &HandoffArgs, claude_home: &Path) -> anyhow::Result<ExitCode> {
    let transcript = match &args.transcript {
        Some(path) => path.clone(),
        None => {
            let here = std::env::current_dir()
                .and_then(fs::canonicalize)
                .context("finding the current directory")?;
            newest_transcript(&claude_home.join("projects").join(project_folder(&here)))?
        }
    };
    let handoff = read(&transcript)?;
    if handoff.first_prompt.is_none() {
        bail!("{} holds no requests to carry over", transcript.display());
    }
    let text = render(&handoff);
    if args.start {
        // Only returns if starting it failed.
        let err = Command::new("claude").arg(&text).exec();
        return Err(err).context("starting claude");
    }
    match &args.out {
        Some(path) => {
            fs::write(path, &text).with_context(|| format!("writing {}", path.display()))?;
        }
        None => writeln!(std::io::stdout(), "{text}")?,
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(content: Value) -> Value {
        json!({ "type": "user", "message": { "content": content } })
    }

    fn transcript() -> Handoff {
        let mut handoff = Handoff::default();
        for prompt in [
            json!("Build the parser. Every error must use ParseError."),
            json!("<system-reminder>not the person</system-reminder>"),
            json!([{ "type": "tool_result", "content": "output" }, { "type": "text", "text": "x" }]),
            json!([{ "type": "text", "text": "Now add tests." }]),
        ] {
            add_user(&mut handoff, &user(prompt));
        }
        add_assistant(
            &mut handoff,
            &json!({ "type": "assistant", "message": { "content": [
                { "type": "thinking", "thinking": "private" },
                { "type": "tool_use", "name": "Edit", "input": { "file_path": "/a/parser.rs" } },
                { "type": "tool_use", "name": "Edit", "input": { "file_path": "/a/parser.rs" } },
                { "type": "tool_use", "name": "Read", "input": { "file_path": "/a/read.rs" } },
                { "type": "text", "text": "Tests are next." },
            ] } }),
        );
        handoff
    }

    #[test]
    fn keeps_what_the_person_asked_and_what_changed() {
        let handoff = transcript();
        assert_eq!(
            handoff.first_prompt.as_deref(),
            Some("Build the parser. Every error must use ParseError.")
        );
        assert_eq!(handoff.recent_prompts, ["Now add tests."]);
        assert_eq!(handoff.files["/a/parser.rs"], 2);
        assert!(!handoff.files.contains_key("/a/read.rs"));
        assert_eq!(handoff.last_answer.as_deref(), Some("Tests are next."));
    }

    #[test]
    fn the_text_says_it_is_partial_and_holds_no_thinking() {
        let text = render(&transcript());
        assert!(text.contains("partial"), "{text}");
        assert!(text.contains("ParseError"), "{text}");
        assert!(text.contains("/a/parser.rs"), "{text}");
        assert!(text.contains("Tests are next."), "{text}");
        assert!(!text.contains("private"), "{text}");
        assert!(!text.contains("not the person"), "{text}");
    }

    #[test]
    fn long_sessions_stay_short() {
        let mut handoff = Handoff {
            first_prompt: Some("x".repeat(100_000)),
            last_answer: Some("y".repeat(100_000)),
            ..Handoff::default()
        };
        for index in 0..500 {
            handoff
                .recent_prompts
                .push(format!("request {index} {}", "z".repeat(5000)));
            handoff.files.insert(format!("/f{index}"), 1);
        }
        let text = render(&handoff);
        assert!(text.len() < 30_000, "{}", text.len());
        assert!(text.contains("request 499"), "the newest request stays");
        assert!(!text.contains("request 100 "), "old ones go");
        assert!(text.contains("earlier requests are left out"));
        assert!(text.contains("and 460 more"));
    }

    #[test]
    fn shortening_keeps_both_ends_and_is_safe_on_multibyte_text() {
        let text = format!("start{}end", "é".repeat(100));
        let short = shortened(&text, 20);
        assert!(short.starts_with("start"), "{short}");
        assert!(short.ends_with("end"), "{short}");
        assert_eq!(shortened("short", 20), "short");
    }

    #[test]
    fn subagent_records_are_left_out() {
        let mut handoff = Handoff::default();
        add_user(
            &mut handoff,
            &json!({ "type": "user", "isSidechain": true, "message": { "content": "inner" } }),
        );
        assert_eq!(handoff.first_prompt, None);
    }

    #[test]
    fn names_a_project_folder_as_claude_code_does() {
        assert_eq!(project_folder(Path::new("/a/b.c d")), "-a-b-c-d");
    }
}

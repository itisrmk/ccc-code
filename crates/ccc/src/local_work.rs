//! Work a local model can do for free: answering questions about a repository, directly with
//! `ccc ask` or as a tool Claude can call through `ccc mcp`. Nothing leaves the machine, unless a
//! server you configured is elsewhere.

use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context as _;
use ccc_index::{SemanticHit, SemanticIndex, query_text};
use ccc_redact::Detector;
use ccc_slm::{CompletionRequest, Embedder, TextModel};
use clap::Args;
use serde_json::{Value, json};

use crate::daemon::DaemonClient;
use crate::gain;
use crate::learn::semantic_cache_path;

const EXCERPTS: usize = 6;
/// As many lines as a chunk holds.
const EXCERPT_LINES: usize = 40;
const MAX_ANSWER_TOKENS: u32 = 512;
const EXPLAIN_LOG: &str = "Say what went wrong in this log and quote the lines that show it \
    exactly, in under 120 words. If nothing went wrong, say so.";
/// A person may wait this long for a free answer, and no longer.
const ANSWER_DEADLINE: Duration = Duration::from_secs(90);
/// A person is waiting on a hook, so vectors arrive quickly or not at all.
const VECTOR_PATIENCE: Duration = Duration::from_millis(800);
const CHARS_PER_TOKEN: usize = 4;
/// Fits the model's 16K context with room for its answer, even if every character were a token.
/// Logs really are dense: 40,000 characters of one came to 18,318 tokens.
const MAX_DIGEST_CHARS: usize = 15_000;
const SEARCH_LIMIT: u64 = 5;
/// Enough of a match to judge it, and often enough to skip opening the file.
const HIT_LINES: usize = 8;
const MAX_SEARCH_LIMIT: u64 = 20;
/// The MCP version to offer a client that doesn't name one.
const MCP_VERSION: &str = "2025-06-18";

/// A repository's code, with models reached through the background service.
struct Models {
    root: PathBuf,
    state_dir: PathBuf,
    client: DaemonClient,
    index: OnceLock<SemanticIndex>,
}

impl Models {
    fn new(state_dir: &Path) -> anyhow::Result<Self> {
        let root = std::env::current_dir()
            .and_then(fs::canonicalize)
            .context("finding the current directory")?;
        Ok(Self {
            root,
            state_dir: state_dir.to_owned(),
            client: DaemonClient::new(state_dir),
            index: OnceLock::new(),
        })
    }

    fn index(&self) -> anyhow::Result<&SemanticIndex> {
        if self.index.get().is_none() {
            let built = refreshed_index(&self.state_dir, &self.root, &self.client)?;
            let _ = self.index.set(built);
        }
        self.index.get().context("the index went away")
    }
}

fn refreshed_index(
    state_dir: &Path,
    root: &Path,
    embedder: &dyn Embedder,
) -> anyhow::Result<SemanticIndex> {
    let cache = semantic_cache_path(state_dir, root);
    let mut index = SemanticIndex::load(&cache);
    let embedded = index
        .refresh(root, &mut |text: &str| embedder.embed(text))
        .context("embedding the repository")?;
    if embedded > 0 {
        index.save(&cache).context("saving vectors")?;
    }
    Ok(index)
}

fn search(
    index: &SemanticIndex,
    embedder: &dyn Embedder,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<SemanticHit>> {
    let vector = embedder
        .embed(&query_text(query))
        .context("embedding the question")?;
    Ok(index.search(&vector, limit))
}

fn excerpt(root: &Path, hit: &SemanticHit) -> Option<String> {
    lines_at(root, hit, EXCERPT_LINES)
}

fn lines_at(root: &Path, hit: &SemanticHit, count: usize) -> Option<String> {
    let source = fs::read_to_string(root.join(&hit.file)).ok()?;
    let body: Vec<&str> = source
        .lines()
        .skip(hit.start_line.saturating_sub(1))
        .take(count)
        .collect();
    Some(body.join("\n"))
}

fn answer_prompt(question: &str, excerpts: &[(String, usize, String)]) -> String {
    let mut text = String::from(
        "Answer the question using only these excerpts from a code repository. Name the file and \
         line you relied on. If they don't hold the answer, say that instead of guessing.\n",
    );
    for (file, line, body) in excerpts {
        // Writing to a String can't fail.
        let _ = write!(text, "\n--- {file}, from line {line} ---\n{body}\n");
    }
    let _ = write!(text, "\nQuestion: {question}");
    text
}

/// Answers `question` from `root`'s code, returning the answer and where it looked.
fn answer(
    root: &Path,
    index: &SemanticIndex,
    embedder: &dyn Embedder,
    model: &dyn TextModel,
    question: &str,
) -> anyhow::Result<(String, Vec<String>)> {
    let hits = search(index, embedder, question, EXCERPTS)?;
    let excerpts: Vec<(String, usize, String)> = hits
        .iter()
        .filter_map(|hit| Some((hit.file.clone(), hit.start_line, excerpt(root, hit)?)))
        .collect();
    let completion = model
        .complete(&CompletionRequest {
            prompt: &answer_prompt(question, &excerpts),
            grammar: None,
            max_tokens: MAX_ANSWER_TOKENS,
            deadline: ANSWER_DEADLINE,
        })
        .context("asking the local model")?;
    let sources = excerpts
        .iter()
        .map(|(file, line, _)| format!("{file}:{line}"))
        .collect();
    Ok((completion.text.trim().to_owned(), sources))
}

#[derive(Debug, Args)]
pub(crate) struct AskArgs {
    /// What to ask about the code in this directory.
    #[arg(value_name = "QUESTION")]
    question: String,
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Args)]
pub(crate) struct LocalArgs {
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Args)]
pub(crate) struct EmbedArgs {
    /// Embed at most this many new chunks, leaving the rest for next time. For working through a
    /// large repository in the background.
    #[arg(long, value_name = "CHUNKS")]
    budget: Option<usize>,
    #[command(flatten)]
    state: crate::StateArgs,
}

/// `ccc embed`: builds this directory's vectors, reusing any whose text hasn't changed.
pub(crate) fn embed(args: &EmbedArgs) -> anyhow::Result<()> {
    let state_dir = args.state.dir()?;
    let root = std::env::current_dir()
        .and_then(fs::canonicalize)
        .context("finding the current directory")?;
    let client = DaemonClient::new(&state_dir);
    let started = std::time::Instant::now();
    let cache = semantic_cache_path(&state_dir, &root);
    let mut index = SemanticIndex::load(&cache);
    let embedded = index
        .refresh_within(
            &root,
            args.budget.unwrap_or(usize::MAX),
            &mut |text: &str| client.embed(text),
        )
        .context("embedding the repository")?;
    if embedded > 0 {
        index.save(&cache).context("saving vectors")?;
    }
    writeln!(
        std::io::stdout(),
        "{} chunks ready, {embedded} new, in {:?}",
        index.len(),
        started.elapsed()
    )?;
    Ok(())
}

/// `ccc ask`: answers from this directory's code without calling any service.
pub(crate) fn ask(args: &AskArgs) -> anyhow::Result<()> {
    let models = Models::new(&args.state.dir()?)?;
    let (text, sources) = answer(
        &models.root,
        models.index()?,
        &models.client,
        &models.client,
        &args.question,
    )?;
    let mut out = std::io::stdout();
    writeln!(out, "{text}\n\nLooked at: {}", sources.join(", "))?;
    Ok(())
}

/// The files a query's vectors point to, for prompt enhancement. Only uses vectors that already
/// exist, since embedding a whole repository has no place on the way to a prompt.
pub(crate) fn vector_files(
    state_dir: &Path,
    root: &Path,
    prompt: &str,
    depth: usize,
) -> Vec<String> {
    let index = SemanticIndex::load(&semantic_cache_path(state_dir, root));
    if index.is_empty() {
        return Vec::new();
    }
    let client = DaemonClient::impatient(state_dir, VECTOR_PATIENCE);
    search(&index, &client, prompt, depth).map_or_else(
        |_| Vec::new(),
        |hits| hits.into_iter().map(|hit| hit.file).collect(),
    )
}

/// What the tools do, so the protocol can be tested without real weights.
trait Tools {
    fn search_code(&self, query: &str, limit: usize) -> anyhow::Result<String>;
    fn ask_local(&self, question: &str) -> anyhow::Result<String>;
    fn summarize_file(&self, path: &str) -> anyhow::Result<String>;
    fn explain_log(&self, log: &str) -> anyhow::Result<String>;
    fn explain_log_file(&self, path: &str) -> anyhow::Result<String>;
}

impl Tools for Models {
    fn search_code(&self, query: &str, limit: usize) -> anyhow::Result<String> {
        let hits = search(self.index()?, &self.client, query, limit)?;
        Ok(format_hits(&self.root, &hits))
    }

    fn ask_local(&self, question: &str) -> anyhow::Result<String> {
        let (text, sources) = answer(
            &self.root,
            self.index()?,
            &self.client,
            &self.client,
            question,
        )?;
        record_local_answer(&self.state_dir, &self.root, &sources, &text);
        Ok(format!("{text}\n\nLooked at: {}", sources.join(", ")))
    }

    fn summarize_file(&self, path: &str) -> anyhow::Result<String> {
        let file = file_inside(&self.root, path)?;
        let text = fs::read_to_string(&file).with_context(|| format!("reading {path}"))?;
        self.digest(
            "Summarize what this file does and name its main functions or types, in under 120 \
             words. Only state what the file shows.",
            first_chars(&text),
        )
    }

    fn explain_log(&self, log: &str) -> anyhow::Result<String> {
        // Claude already paid for this text by writing it out, so nothing was saved.
        self.complete(EXPLAIN_LOG, last_chars(log))
    }

    fn explain_log_file(&self, path: &str) -> anyhow::Result<String> {
        let file = file_inside(&self.root, path)?;
        let bytes = fs::read(&file).with_context(|| format!("reading {path}"))?;
        self.digest(EXPLAIN_LOG, last_chars(&String::from_utf8_lossy(&bytes)))
    }
}

impl Models {
    /// Has the local model read `text` so Claude doesn't have to, and records the difference.
    fn digest(&self, instruction: &str, shown: &str) -> anyhow::Result<String> {
        let answer = self.complete(instruction, shown)?;
        let saved = shown.len().saturating_sub(answer.len()) / CHARS_PER_TOKEN;
        gain::record(&self.state_dir, gain::Basis::LocalAnswer, saved as u64);
        Ok(answer)
    }

    fn complete(&self, instruction: &str, shown: &str) -> anyhow::Result<String> {
        let completion = self
            .client
            .complete(&CompletionRequest {
                prompt: &format!("{instruction}\n\n{shown}"),
                grammar: None,
                max_tokens: MAX_ANSWER_TOKENS,
                deadline: ANSWER_DEADLINE,
            })
            .context("asking the local model")?;
        Ok(completion.text.trim().to_owned())
    }
}

/// The start of `text`, where a file says what it is.
fn first_chars(text: &str) -> &str {
    let end = text
        .char_indices()
        .nth(MAX_DIGEST_CHARS)
        .map_or(text.len(), |(at, _)| at);
    &text[..end]
}

/// The end of `text`, since a log's failure is usually its last thing.
fn last_chars(text: &str) -> &str {
    let start = text
        .char_indices()
        .rev()
        .nth(MAX_DIGEST_CHARS.saturating_sub(1))
        .map_or(0, |(at, _)| at);
    &text[start..]
}

/// Search results with their code. A match holding a secret shows only its location.
fn format_hits(root: &Path, hits: &[SemanticHit]) -> String {
    let detector = Detector::builtin();
    let mut text = String::new();
    for hit in hits {
        let _ = writeln!(text, "{}:{} ({:.2})", hit.file, hit.start_line, hit.score);
        if let Some(body) =
            lines_at(root, hit, HIT_LINES).filter(|body| !detector.finds_secret(body))
        {
            let _ = writeln!(text, "{body}\n");
        }
    }
    text
}

/// `path` as a file inside `root`. Anything that leads outside is refused, since a model can be
/// talked into asking for anything.
fn file_inside(root: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let found = fs::canonicalize(root.join(path)).with_context(|| format!("no file at {path}"))?;
    if !found.starts_with(root) {
        anyhow::bail!("{path} is outside this repository");
    }
    Ok(found)
}

/// Counts what the local model read in place of Claude. It's an estimate, since Claude might not
/// have read all of it, and `ccc gain` says so.
fn record_local_answer(state_dir: &Path, root: &Path, sources: &[String], answer: &str) {
    let read: usize = sources
        .iter()
        .filter_map(|source| source.rsplit_once(':'))
        .filter_map(|(file, line)| {
            let hit = SemanticHit {
                file: file.to_owned(),
                start_line: line.parse().ok()?,
                score: 0.0,
            };
            excerpt(root, &hit).map(|body| body.len())
        })
        .sum();
    let tokens = read.saturating_sub(answer.len()) / CHARS_PER_TOKEN;
    gain::record(state_dir, gain::Basis::LocalAnswer, tokens as u64);
}

fn tool_list() -> Value {
    json!({ "tools": [
        {
            "name": "search_code",
            "description": "Find code in this repository by meaning, for when you don't know \
                what it's called. Returns file:line locations with the code there. Runs locally. Use Grep instead \
                when you know the exact name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What the code does" },
                    "limit": { "type": "integer", "description": "How many files, default 5" },
                },
                "required": ["query"],
            },
        },
        {
            "name": "summarize_file",
            "description": "Have a small local model summarize a large file, so you can decide \
                whether to read it. Free and private, but it can be wrong: read the file before \
                you change it or rely on the details.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Relative to the repository" } },
                "required": ["path"],
            },
        },
        {
            "name": "explain_log",
            "description": "Have a small local model pick out what went wrong in a long log. \
                Free and private, but it can miss things, so check what it quotes. Give a path \
                when the log is in a file, so you don't have to write it out.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Log file, relative to the repository" },
                    "log": { "type": "string", "description": "The log's text, if it isn't in a file" },
                },
            },
        },
        {
            "name": "ask_local",
            "description": "Ask a small local model a simple question about this repository. \
                It's free and private but much weaker than you, and it can be wrong, so check \
                anything you rely on.",
            "inputSchema": {
                "type": "object",
                "properties": { "question": { "type": "string" } },
                "required": ["question"],
            },
        },
    ] })
}

fn call_tool(tools: &dyn Tools, params: &Value) -> Value {
    let text_field = |key: &str| {
        params
            .pointer(&format!("/arguments/{key}"))
            .and_then(Value::as_str)
    };
    let result = match params.get("name").and_then(Value::as_str) {
        Some("search_code") => match text_field("query") {
            Some(query) => {
                let limit = params
                    .pointer("/arguments/limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(SEARCH_LIMIT)
                    .clamp(1, MAX_SEARCH_LIMIT);
                tools.search_code(query, limit as usize)
            }
            None => Err(anyhow::anyhow!("search_code needs a query")),
        },
        Some("ask_local") => match text_field("question") {
            Some(question) => tools.ask_local(question),
            None => Err(anyhow::anyhow!("ask_local needs a question")),
        },
        Some("summarize_file") => match text_field("path") {
            Some(path) => tools.summarize_file(path),
            None => Err(anyhow::anyhow!("summarize_file needs a path")),
        },
        Some("explain_log") => match (text_field("path"), text_field("log")) {
            (Some(path), _) => tools.explain_log_file(path),
            (None, Some(log)) => tools.explain_log(log),
            (None, None) => Err(anyhow::anyhow!("explain_log needs a path or a log")),
        },
        other => Err(anyhow::anyhow!("no tool called {other:?}")),
    };
    match result {
        Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
        // A tool's failure is for the model to read, not a protocol error.
        Err(err) => json!({
            "content": [{ "type": "text", "text": format!("{err:#}") }],
            "isError": true,
        }),
    }
}

/// Answers one message, or nothing if it's a notification.
fn handle_message(tools: &dyn Tools, message: &Value) -> Option<Value> {
    let id = message.get("id")?.clone();
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let result = match message.get("method").and_then(Value::as_str) {
        Some("initialize") => {
            let version = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(MCP_VERSION);
            json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "ccc", "version": env!("CARGO_PKG_VERSION") },
            })
        }
        Some("ping") => json!({}),
        Some("tools/list") => tool_list(),
        Some("tools/call") => call_tool(tools, &params),
        other => {
            return Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("no method {other:?}") },
            }));
        }
    };
    Some(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

/// `ccc mcp`: serves the tools over stdio for as long as the client stays connected.
pub(crate) fn mcp(args: &LocalArgs) -> anyhow::Result<()> {
    let models = Models::new(&args.state.dir()?)?;
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.context("reading from the client")?;
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(reply) = handle_message(&models, &message) {
            writeln!(out, "{reply}").context("writing to the client")?;
            out.flush().context("writing to the client")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_slm::{Completion, ModelError};
    use std::sync::Mutex;

    #[derive(Debug)]
    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
            Ok(vec![
                f32::from(u8::from(text.contains("token"))),
                f32::from(u8::from(text.contains("cache"))),
            ])
        }
    }

    #[derive(Debug, Default)]
    struct FakeModel {
        seen: Mutex<String>,
    }

    impl TextModel for FakeModel {
        fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            request.prompt.clone_into(&mut self.seen.lock().unwrap());
            Ok(Completion {
                text: " It's in auth.rs. ".to_owned(),
                ..Completion::default()
            })
        }
    }

    #[test]
    fn answers_from_the_code_that_matches_and_names_it() {
        let root = std::env::temp_dir().join(format!("ccc-ask-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("auth.rs"), "fn verify_token() {}\n").unwrap();
        let mut index = SemanticIndex::default();
        index
            .refresh(&root, &mut |text: &str| FakeEmbedder.embed(text))
            .unwrap();
        let model = FakeModel::default();

        let (text, sources) = answer(
            &root,
            &index,
            &FakeEmbedder,
            &model,
            "where is the token checked",
        )
        .unwrap();

        assert_eq!(text, "It's in auth.rs.");
        assert_eq!(sources, ["auth.rs:1"]);
        let seen = model.seen.lock().unwrap().clone();
        assert!(seen.contains("fn verify_token"), "{seen}");
        assert!(seen.contains("say that instead of guessing"), "{seen}");
        assert!(seen.ends_with("where is the token checked"), "{seen}");
        fs::remove_dir_all(&root).unwrap();
    }

    struct FakeTools;

    impl Tools for FakeTools {
        fn search_code(&self, query: &str, limit: usize) -> anyhow::Result<String> {
            Ok(format!("{query} {limit}"))
        }

        fn ask_local(&self, _: &str) -> anyhow::Result<String> {
            anyhow::bail!("no model here")
        }

        fn summarize_file(&self, path: &str) -> anyhow::Result<String> {
            Ok(format!("summary of {path}"))
        }

        fn explain_log(&self, log: &str) -> anyhow::Result<String> {
            Ok(format!("{} bytes", log.len()))
        }

        fn explain_log_file(&self, path: &str) -> anyhow::Result<String> {
            Ok(format!("log at {path}"))
        }
    }

    #[test]
    fn the_smaller_jobs_are_offered_and_need_their_input() {
        let tools = call(json!({ "id": 1, "method": "tools/list" }));
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["search_code", "summarize_file", "explain_log", "ask_local"]
        );
        let reply = call(json!({ "id": 2, "method": "tools/call", "params": {
            "name": "summarize_file", "arguments": { "path": "a.rs" } } }));
        assert_eq!(reply["result"]["content"][0]["text"], "summary of a.rs");
        let missing = call(json!({ "id": 3, "method": "tools/call", "params": {
            "name": "explain_log", "arguments": {} } }));
        assert_eq!(missing["result"]["isError"], true);
        let by_path = call(json!({ "id": 4, "method": "tools/call", "params": {
            "name": "explain_log", "arguments": { "path": "b.log", "log": "x" } } }));
        assert_eq!(by_path["result"]["content"][0]["text"], "log at b.log");
        let by_text = call(json!({ "id": 5, "method": "tools/call", "params": {
            "name": "explain_log", "arguments": { "log": "abc" } } }));
        assert_eq!(by_text["result"]["content"][0]["text"], "3 bytes");
    }

    #[test]
    fn a_long_log_keeps_its_end_and_a_long_file_its_start() {
        let text = format!("start{}end é", "x".repeat(MAX_DIGEST_CHARS));
        assert!(last_chars(&text).ends_with("end é"));
        assert_eq!(last_chars(&text).chars().count(), MAX_DIGEST_CHARS);
        assert!(first_chars(&text).starts_with("start"));
        assert_eq!(first_chars(&text).chars().count(), MAX_DIGEST_CHARS);
        assert_eq!(last_chars("short"), "short");
        assert_eq!(first_chars("short"), "short");
    }

    #[test]
    fn search_results_show_code_but_never_a_secret() {
        let root = std::env::temp_dir().join(format!("ccc-hits-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.py"), "skipped\ndef refund():\n    pass\n").unwrap();
        let key = format!("sk-ant-api03-{}", "a1B2c3D4e5F6".repeat(6));
        fs::write(root.join("b.py"), format!("KEY = \"{key}\"\n")).unwrap();
        let hit = |file: &str, start_line| SemanticHit {
            file: file.to_owned(),
            start_line,
            score: 0.5,
        };

        let text = format_hits(&root, &[hit("a.py", 2), hit("b.py", 1), hit("gone.py", 1)]);

        fs::remove_dir_all(&root).unwrap();
        assert!(text.contains("a.py:2 (0.50)\ndef refund():"), "{text}");
        assert!(!text.contains("skipped"), "{text}");
        assert!(text.contains("b.py:1 (0.50)"), "{text}");
        assert!(!text.contains("sk-ant"), "{text}");
        assert!(text.contains("gone.py:1 (0.50)"), "{text}");
    }

    #[test]
    fn a_file_outside_the_repository_is_refused() {
        let base = std::env::temp_dir().join(format!("ccc-inside-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("repo")).unwrap();
        let base = fs::canonicalize(&base).unwrap();
        let root = base.join("repo");
        fs::write(root.join("kept.rs"), "x").unwrap();
        fs::write(base.join("secret.env"), "x").unwrap();
        std::os::unix::fs::symlink(base.join("secret.env"), root.join("link.env")).unwrap();

        assert!(file_inside(&root, "kept.rs").is_ok());
        for path in ["../secret.env", "link.env", "/etc/passwd", "missing.rs"] {
            assert!(file_inside(&root, path).is_err(), "{path}");
        }
        fs::remove_dir_all(&base).unwrap();
    }

    fn call(message: Value) -> Value {
        handle_message(&FakeTools, &message).unwrap()
    }

    #[test]
    fn speaks_the_protocol() {
        let reply = call(json!({ "id": 1, "method": "initialize",
            "params": { "protocolVersion": "2025-03-26" } }));
        assert_eq!(reply["result"]["protocolVersion"], "2025-03-26");
        assert!(reply["result"]["capabilities"]["tools"].is_object());

        let tools = call(json!({ "id": 2, "method": "tools/list" }));
        assert_eq!(tools["result"]["tools"][0]["name"], "search_code");
        assert_eq!(tools["result"]["tools"][3]["name"], "ask_local");

        assert_eq!(
            call(json!({ "id": 3, "method": "nope" }))["error"]["code"],
            -32601
        );
        assert_eq!(
            handle_message(
                &FakeTools,
                &json!({ "method": "notifications/initialized" })
            ),
            None
        );
        assert_eq!(
            call(json!({ "id": 4, "method": "ping" }))["result"],
            json!({})
        );
    }

    #[test]
    fn tools_run_with_limits_and_report_failures_to_the_model() {
        let reply = call(json!({ "id": 1, "method": "tools/call", "params": {
            "name": "search_code", "arguments": { "query": "login", "limit": 999 } } }));
        assert_eq!(reply["result"]["content"][0]["text"], "login 20");

        let defaulted = call(json!({ "id": 2, "method": "tools/call", "params": {
            "name": "search_code", "arguments": { "query": "login" } } }));
        assert_eq!(defaulted["result"]["content"][0]["text"], "login 5");

        for params in [
            json!({ "name": "ask_local", "arguments": { "question": "x" } }),
            json!({ "name": "search_code", "arguments": {} }),
            json!({ "name": "missing" }),
        ] {
            let reply = call(json!({ "id": 3, "method": "tools/call", "params": params }));
            assert_eq!(reply["result"]["isError"], true, "{reply}");
        }
    }
}

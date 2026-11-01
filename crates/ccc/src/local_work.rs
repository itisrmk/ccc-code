//! Work a local model can do for free: answering questions about a repository, directly with
//! `ccc ask` or as a tool Claude can call through `ccc mcp`. Nothing leaves the machine.

use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context as _;
use ccc_index::{SemanticHit, SemanticIndex, query_text};
use ccc_slm::{
    ChatFormat, CompletionRequest, Embedder, LlamaConfig, LlamaEmbedder, LlamaModel, TextModel,
};
use clap::Args;
use serde_json::{Value, json};

use crate::learn::semantic_cache_path;

const EXCERPTS: usize = 6;
/// As many lines as a chunk holds.
const EXCERPT_LINES: usize = 40;
const MAX_ANSWER_TOKENS: u32 = 512;
/// A person may wait this long for a free answer, and no longer.
const ANSWER_DEADLINE: Duration = Duration::from_secs(90);
const DEFAULT_EMBEDDING_MODEL: &str = "embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_ANSWER_MODEL: &str = "gemma-4-E2B-it-Q4_0.gguf";
const SEARCH_LIMIT: u64 = 5;
const MAX_SEARCH_LIMIT: u64 = 20;
/// The MCP version to offer a client that doesn't name one.
const MCP_VERSION: &str = "2025-06-18";

fn model_path(state_dir: &Path, given: Option<&Path>, default: &str) -> PathBuf {
    given.map_or_else(|| state_dir.join("models").join(default), Path::to_owned)
}

/// Models for this directory, loaded when first needed since they take memory and a moment.
struct Models {
    root: PathBuf,
    state_dir: PathBuf,
    embedding: PathBuf,
    answering: PathBuf,
    embedder: OnceLock<LlamaEmbedder>,
    answerer: OnceLock<LlamaModel>,
    index: OnceLock<SemanticIndex>,
}

impl Models {
    fn new(
        state_dir: &Path,
        embedding: Option<&Path>,
        answering: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let root = std::env::current_dir()
            .and_then(fs::canonicalize)
            .context("finding the current directory")?;
        Ok(Self {
            root,
            state_dir: state_dir.to_owned(),
            embedding: model_path(state_dir, embedding, DEFAULT_EMBEDDING_MODEL),
            answering: model_path(state_dir, answering, DEFAULT_ANSWER_MODEL),
            embedder: OnceLock::new(),
            answerer: OnceLock::new(),
            index: OnceLock::new(),
        })
    }

    fn embedder(&self) -> anyhow::Result<&LlamaEmbedder> {
        if self.embedder.get().is_none() {
            let loaded = LlamaEmbedder::load(&self.embedding).with_context(|| {
                format!(
                    "loading {}. scripts/fetch-models.sh embedding downloads it",
                    self.embedding.display()
                )
            })?;
            let _ = self.embedder.set(loaded);
        }
        self.embedder.get().context("the embedding model went away")
    }

    fn answerer(&self) -> anyhow::Result<&LlamaModel> {
        if self.answerer.get().is_none() {
            let config = LlamaConfig {
                format: ChatFormat::Gemma4,
                ..LlamaConfig::default()
            };
            let loaded = LlamaModel::load(&self.answering, config).with_context(|| {
                format!(
                    "loading {}. scripts/fetch-models.sh gemma downloads it",
                    self.answering.display()
                )
            })?;
            let _ = self.answerer.set(loaded);
        }
        self.answerer.get().context("the answering model went away")
    }

    fn index(&self) -> anyhow::Result<&SemanticIndex> {
        if self.index.get().is_none() {
            let built = refreshed_index(&self.state_dir, &self.root, self.embedder()?)?;
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
    let source = fs::read_to_string(root.join(&hit.file)).ok()?;
    let body: Vec<&str> = source
        .lines()
        .skip(hit.start_line.saturating_sub(1))
        .take(EXCERPT_LINES)
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
pub(crate) struct LocalModelArgs {
    /// The GGUF embedding model. Defaults to EmbeddingGemma in the state directory's models.
    #[arg(long, value_name = "FILE")]
    embedding_model: Option<PathBuf>,
    /// The GGUF model that answers, in Gemma 4's format. Defaults to Gemma 4 E2B there.
    #[arg(long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[command(flatten)]
    state: crate::StateArgs,
}

impl LocalModelArgs {
    fn models(&self) -> anyhow::Result<Models> {
        Models::new(
            &self.state.dir()?,
            self.embedding_model.as_deref(),
            self.model.as_deref(),
        )
    }
}

#[derive(Debug, Args)]
pub(crate) struct AskArgs {
    /// What to ask about the code in this directory.
    #[arg(value_name = "QUESTION")]
    question: String,
    #[command(flatten)]
    local: LocalModelArgs,
}

#[derive(Debug, Args)]
pub(crate) struct EmbedArgs {
    #[command(flatten)]
    local: LocalModelArgs,
}

/// `ccc embed`: builds this directory's vectors, reusing any whose text hasn't changed.
pub(crate) fn embed(args: &EmbedArgs) -> anyhow::Result<()> {
    let models = args.local.models()?;
    let started = std::time::Instant::now();
    let index = models.index()?;
    writeln!(
        std::io::stdout(),
        "{} chunks ready in {:?}",
        index.len(),
        started.elapsed()
    )?;
    Ok(())
}

/// `ccc ask`: answers from this directory's code without calling any service.
pub(crate) fn ask(args: &AskArgs) -> anyhow::Result<()> {
    let models = args.local.models()?;
    let (text, sources) = answer(
        &models.root,
        models.index()?,
        models.embedder()?,
        models.answerer()?,
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
    let path = model_path(state_dir, None, DEFAULT_EMBEDDING_MODEL);
    let Ok(embedder) = LlamaEmbedder::load(&path) else {
        return Vec::new();
    };
    search(&index, &embedder, prompt, depth).map_or_else(
        |_| Vec::new(),
        |hits| hits.into_iter().map(|hit| hit.file).collect(),
    )
}

/// What the tools do, so the protocol can be tested without real weights.
trait Tools {
    fn search_code(&self, query: &str, limit: usize) -> anyhow::Result<String>;
    fn ask_local(&self, question: &str) -> anyhow::Result<String>;
}

impl Tools for Models {
    fn search_code(&self, query: &str, limit: usize) -> anyhow::Result<String> {
        let hits = search(self.index()?, self.embedder()?, query, limit)?;
        let mut text = String::new();
        for hit in hits {
            let _ = writeln!(text, "{}:{} ({:.2})", hit.file, hit.start_line, hit.score);
        }
        Ok(text)
    }

    fn ask_local(&self, question: &str) -> anyhow::Result<String> {
        let (text, sources) = answer(
            &self.root,
            self.index()?,
            self.embedder()?,
            self.answerer()?,
            question,
        )?;
        Ok(format!("{text}\n\nLooked at: {}", sources.join(", ")))
    }
}

fn tool_list() -> Value {
    json!({ "tools": [
        {
            "name": "search_code",
            "description": "Find code in this repository by meaning, for when you don't know \
                what it's called. Returns file:line locations. Runs locally. Use Grep instead \
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
pub(crate) fn mcp(args: &EmbedArgs) -> anyhow::Result<()> {
    let models = args.local.models()?;
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
        assert_eq!(tools["result"]["tools"][1]["name"], "ask_local");

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

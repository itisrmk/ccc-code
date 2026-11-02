//! `ccc`: make your Claude limit last longer, with a small model on your machine.

mod advise;
mod bench;
mod clients;
mod config;
mod daemon;
mod dashboard;
mod enhance;
mod eval;
mod export;
mod gain;
mod handoff;
mod hooks;
mod init;
mod learn;
#[cfg(feature = "local-model")]
mod local_models;
mod local_work;
#[cfg(feature = "local-model")]
mod model_bench;
#[cfg(feature = "ner")]
mod names;
mod policy;
mod router;
mod rules;
mod run;
mod setup;
mod sharpen;
mod statusline;
mod trim;
mod verify;
mod whatif;

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use crate::clients::Client;
use anyhow::Context;
use ccc_enrich::{Enrich, EnrichConfig, GateConfig, Suggester};
use ccc_index::Index;
use ccc_learn::Memory;
use ccc_ledger::Ledger;
use ccc_pipeline::{Pipeline, Stage};
use ccc_proxy::{Proxy, ProxyConfig};
use ccc_prune::Prune;
use ccc_redact::{Detector, Redact, Vault};
use ccc_route::{Route, RouteRules, StripNotices};
use clap::{Args, Parser, Subcommand};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

/// How often to pick up changed files while running.
const INDEX_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// The usage record, inside the state directory.
pub(crate) const LEDGER_FILE: &str = "ledger.sqlite";

/// Project redaction rules, relative to the directory `ccc` runs in.
const DEFAULT_REDACT_RULES: &str = ".ccc/redact-rules";

/// Where requests go when neither `--upstream` nor `ANTHROPIC_BASE_URL` is set.
const DEFAULT_UPSTREAM: &str = "https://api.anthropic.com";

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Set everything up: local models, cost warnings, status line and tools for Claude.
    Setup(setup::SetupArgs),
    /// Take everything back out again.
    Uninstall(setup::UninstallArgs),
    /// Check that everything's working.
    Doctor(setup::DoctorArgs),
    /// Show where things stand, without starting anything.
    Status(StateArgs),
    /// Show what the local model has kept out of Claude's context.
    Gain(StateArgs),
    /// See and decide about what the local model learned from your sessions.
    Rules(rules::RulesArgs),
    /// See or change which optional jobs the local model does.
    Jobs(sharpen::JobsArgs),
    /// List the other commands.
    Advanced,
    /// Print the status line. Called by Claude Code, not by hand.
    #[command(hide = true)]
    Statusline(StateArgs),
    /// Download a model. Called by setup, not by hand.
    #[command(hide = true)]
    FetchModel(setup::FetchModelArgs),
    #[command(hide = true)]
    /// Start the proxy and run a command, normally `claude`, pointed at it.
    Run(RunArgs),
    #[command(hide = true)]
    /// Start the proxy and wait; point ANTHROPIC_BASE_URL at it yourself.
    Serve(ServeArgs),
    #[command(hide = true)]
    /// Replay local transcripts and report token and cost effects. Nothing leaves the machine.
    Bench(BenchArgs),
    /// Show where sessions' money went, and what restarting or refreshing would change.
    /// Reads local transcripts; nothing leaves the machine.
    Whatif(WhatifArgs),
    /// Test whether a Claude Code setup really helps, on tasks from a repository's own history.
    Eval(eval::EvalArgs),
    #[command(hide = true)]
    /// Turn on cost warnings for Claude Code, by adding hooks to its settings.
    Init(init::InitArgs),
    /// Write what a fresh session needs from this one, and optionally start it. Costs nothing
    /// and calls no model until the new session starts.
    Handoff(handoff::HandoffArgs),
    #[command(hide = true)]
    /// Check whether a gateway is safe to put between Claude Code and the API. Needs no API
    /// key: the gateway forwards to a recording stand-in, and nothing leaves the machine.
    Verify(verify::VerifyArgs),
    #[command(hide = true)]
    /// Show what recent sessions cost at list prices.
    Stats(StatsArgs),
    #[command(hide = true)]
    /// Learn which files past work needed, from local transcripts. Nothing leaves the machine.
    Learn(LearnArgs),
    #[command(hide = true)]
    /// Build or refresh a repository's symbol index.
    Index(IndexArgs),
    #[command(hide = true)]
    /// Judge file suggestions against what past work really touched.
    EnrichEval(EnrichEvalArgs),
    #[command(hide = true)]
    /// Print the settings that register ccc's hooks, for `claude --settings`.
    HookSettings(HookSettingsArgs),
    #[command(hide = true)]
    /// Handle a Claude Code hook. Called by Claude Code, not by hand.
    Hook(HookArgs),
    #[command(hide = true)]
    /// Train the learned router from labelled requests, and report its held-out accuracy.
    RouterTrain(RouterTrainArgs),
    #[command(hide = true)]
    /// Serve a page of recent sessions' costs on this machine.
    Dashboard(DashboardArgs),
    #[command(hide = true)]
    /// Show the context prompt enhancement would add to a prompt. Costs nothing.
    Enhance(EnhanceArgs),
    /// Build this directory's vectors for search by meaning.
    Embed(local_work::EmbedArgs),
    /// Ask a local model about the code in this directory. Free, and nothing leaves the machine.
    Ask(local_work::AskArgs),
    #[command(hide = true)]
    /// Serve local code search and questions as tools for Claude, over MCP on stdio.
    Mcp(local_work::LocalArgs),
    #[command(hide = true)]
    /// Run the background service that holds local models. Started when needed.
    Daemon(DaemonArgs),
    #[command(hide = true)]
    /// Measure the name-detection model on files in a directory. Needs the ner build feature.
    #[cfg(feature = "ner")]
    NerBench(NerBenchArgs),
    #[command(hide = true)]
    /// Measure a local model on this machine. Needs the local-model build feature.
    #[cfg(feature = "local-model")]
    ModelBench(ModelBenchArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ProxyArgs {
    /// Base URL to forward to. Defaults to the Anthropic API, or the client's own service
    /// when `run` launches another client.
    #[arg(long, env = "ANTHROPIC_BASE_URL", default_value = DEFAULT_UPSTREAM)]
    upstream: String,
    /// Copy all traffic to this directory. Holds real prompts and tool output.
    #[arg(long, value_name = "DIR")]
    dump: Option<PathBuf>,
    /// Append logs to this file. Defaults to stderr for `serve` and off for `run`.
    #[arg(long, value_name = "FILE")]
    log_file: Option<PathBuf>,
    /// Log what each stage changed. Use with --log-file under `run`.
    #[arg(long)]
    debug: bool,
    /// Don't prune tool output in new conversations. Existing ones keep their settings.
    #[arg(long)]
    no_prune: bool,
    /// Don't redact secrets in new conversations. Existing ones keep their settings.
    #[arg(long)]
    no_redact: bool,
    /// Extra patterns to redact, one regular expression per line. Lines starting with #
    /// are ignored. Defaults to .ccc/redact-rules in the current directory, if present.
    #[arg(long, value_name = "FILE")]
    redact_rules: Option<PathBuf>,
    /// The model for background calls such as naming a session. `local:NAME` sends them
    /// to --local-url. Off by default; Claude Code already uses a small model for these.
    #[arg(long, value_name = "MODEL")]
    background_model: Option<String>,
    /// Base URL of a local model with an Anthropic-compatible API.
    #[arg(long, value_name = "URL")]
    local_url: Option<String>,
    /// Show a one-line notice when a session's cost at list prices passes this many dollars.
    #[arg(long, value_name = "USD")]
    budget_usd: Option<f64>,
    /// Move to this model once over --budget-usd. Its first call re-writes the whole
    /// context to that model's cache and drops thinking the new model can't read.
    #[arg(long, value_name = "MODEL", requires = "budget_usd")]
    budget_downshift: Option<String>,
    /// Label user turns with this GGUF model. Off by default: no decision depends on the
    /// labels yet, and on an M2 Pro a 4B model takes about 170 ms per turn.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "FILE")]
    pub(crate) classify_model: Option<PathBuf>,
    /// Give up on a label after this many milliseconds.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "MS", default_value_t = 80)]
    pub(crate) classify_cap_ms: u64,
    /// Let this GGUF model choose lines to keep from large tool output. Turns itself off at
    /// startup if the model is too slow for --prune-model-cap-ms.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "FILE")]
    pub(crate) prune_model: Option<PathBuf>,
    /// How long the prune model may hold up a request.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "MS", default_value_t = 600)]
    pub(crate) prune_model_cap_ms: u64,
    /// Also redact names of people and organizations, with the GLiNER model in this directory.
    /// Off by default: it's far slower than the regex tier and flags some code.
    #[cfg(feature = "ner")]
    #[arg(long, value_name = "DIR")]
    ner_model: Option<PathBuf>,
    /// Start conversations on --router-cheap-model when this trained model is confident
    /// that's enough. Off by default; see `ccc router-train`.
    #[arg(long, value_name = "FILE", requires = "router_cheap_model")]
    learned_router: Option<PathBuf>,
    /// The cheaper model for --learned-router.
    #[arg(long, value_name = "MODEL")]
    router_cheap_model: Option<String>,
    /// How sure the learned router must be, from 0 to 1.
    #[arg(long, value_name = "P", default_value_t = 0.9)]
    router_confidence: f64,
    /// Team rules to enforce. Defaults to .ccc/policy.toml in the current directory, if present.
    #[arg(long, value_name = "FILE")]
    policy: Option<PathBuf>,
    /// Suggest likely relevant files on each user turn, from this directory's index and
    /// `ccc learn`. Off by default: it spends tokens on every turn.
    #[arg(long)]
    enrich: bool,
    /// How long suggesting may hold up a request.
    #[arg(long, value_name = "MS", default_value_t = 400)]
    enrich_cap_ms: u64,
    /// Have the API reject any edit to history behind a thinking block. For testing: this
    /// is already the default for accounts created on or after 2026-08-31.
    #[arg(long)]
    enforce_thinking_check: bool,
    /// Where per-conversation state lives. Defaults to ~/.ccc.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
}

impl ProxyArgs {
    fn proxy(&self) -> anyhow::Result<Proxy> {
        let policy = policy::load(self.policy.as_deref())?;
        if let Some(policy) = &policy {
            policy
                .check(&self.settings())
                .context("this run breaks the team policy")?;
        }
        let state_dir = self.state_dir.clone().or_else(default_state_dir);
        let vault = Arc::new(match &state_dir {
            Some(dir) => Vault::open(&dir.join("redact.key")).context("opening redaction key")?,
            None => Vault::ephemeral().context("creating redaction key")?,
        });
        let extra_rules = policy
            .as_ref()
            .map_or(&[][..], |policy| policy.redact_rules());
        let redact = Redact::new(self.detector(extra_rules)?, Arc::clone(&vault));
        #[cfg(feature = "ner")]
        let redact = names::add_tier(redact, self.ner_model.as_deref(), state_dir.as_deref())?;
        let prune = Prune::new();

        let route = Route::new(RouteRules {
            background_model: self.background_model.clone(),
            budget_usd: self.budget_usd,
            budget_model: self.budget_downshift.clone(),
        });
        let ledger = match &state_dir {
            Some(dir) => Ledger::open(&dir.join(LEDGER_FILE)).context("opening ledger")?,
            None => Ledger::in_memory().context("creating ledger")?,
        };

        let mut disabled_stages = BTreeSet::new();
        if self.no_redact {
            disabled_stages.insert(redact.name().to_owned());
        }
        if self.no_prune {
            disabled_stages.insert(prune.name().to_owned());
        }
        // Secrets go first among the stages that edit, so nothing later handles or logs a
        // real value. Labelling stays on this machine, so it can read them.
        let mut stages: Vec<Arc<dyn Stage>> = Vec::new();
        #[cfg(feature = "local-model")]
        let local = local_models::load(self, state_dir.as_deref())?;
        #[cfg(feature = "local-model")]
        stages.extend(local.classify);
        stages.push(Arc::new(StripNotices));
        stages.push(Arc::new(redact));
        stages.push(Arc::new(prune));
        #[cfg(feature = "local-model")]
        stages.extend(local.prune);
        if self.enrich {
            let dir = state_dir
                .as_deref()
                .context("--enrich needs a state directory")?;
            stages.push(Arc::new(enrich_stage(dir, self.enrich_cap_ms)?));
        }
        if let Some(path) = &self.learned_router {
            let dir = state_dir
                .as_deref()
                .context("--learned-router needs a state directory")?;
            stages.push(Arc::new(router::stage(
                path,
                self.router_cheap_model.clone().unwrap_or_default(),
                self.router_confidence,
                dir,
            )?));
        }
        stages.push(Arc::new(route));

        let config = ProxyConfig {
            upstream: self.upstream.clone(),
            dump_dir: self.dump.clone(),
            pipeline: Pipeline::new(stages),
            disabled_stages,
            state_dir,
            vault: Some(vault),
            enforce_thinking_check: self.enforce_thinking_check,
            ledger: Some(Arc::new(ledger)),
            budget_usd: self.budget_usd,
            local_url: self.local_url.clone(),
        };
        Proxy::new(config).context("configuring proxy")
    }
}

/// Sets up Enrich for the current directory, keeping its index fresh in the background.
fn enrich_stage(state_dir: &std::path::Path, cap_ms: u64) -> anyhow::Result<Enrich> {
    let root = std::env::current_dir().context("finding the current directory")?;
    let cache = learn::index_cache_path(state_dir, &root);
    let memory = Memory::open(&learn::memory_path(state_dir)).context("opening memory")?;
    let suggester = Arc::new(Suggester::new(
        root,
        Index::load(&cache),
        Some(Arc::new(memory)),
    ));
    suggester.refresh();
    let refreshing = Arc::clone(&suggester);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(INDEX_REFRESH_INTERVAL);
            refreshing.refresh();
        }
    });
    let config = EnrichConfig {
        decisions_dir: state_dir.join("enrich-decisions"),
        time_cap: std::time::Duration::from_millis(cap_ms),
        rewrite_log: Some(state_dir.join("enrich-rewrites.jsonl")),
        gate: GateConfig {
            min_similarity: 0.6,
            classify_deadline: std::time::Duration::from_millis(cap_ms),
        },
    };
    Ok(Enrich::new(suggester, None, config))
}

/// Claude Code's own directory.
fn claude_home() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME isn't set")?;
    Ok(PathBuf::from(home).join(".claude"))
}

fn default_state_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".ccc"))
}

impl ProxyArgs {
    fn settings(&self) -> policy::RunSettings<'_> {
        policy::RunSettings {
            redact: !self.no_redact,
            prune: !self.no_prune,
            dump: self.dump.is_some(),
            upstream: &self.upstream,
            budget_usd: self.budget_usd,
        }
    }

    fn detector(&self, extra_rules: &[String]) -> anyhow::Result<Detector> {
        let path = self
            .redact_rules
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_REDACT_RULES));
        let rules = match fs::read_to_string(&path) {
            Ok(text) => text,
            // A rules file named on the command line has to exist; the default is optional.
            Err(err) if err.kind() == ErrorKind::NotFound && self.redact_rules.is_none() => {
                String::new()
            }
            Err(err) => {
                return Err(err).with_context(|| format!("reading {}", path.display()));
            }
        };
        let patterns: Vec<String> = rules
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_owned)
            .chain(extra_rules.iter().cloned())
            .collect();
        Detector::with_project_rules(&patterns)
            .with_context(|| format!("reading rules from {}", path.display()))
    }
}

#[derive(Debug, Args)]
struct RunArgs {
    #[command(flatten)]
    proxy: ProxyArgs,
    /// Print a one-line summary when the command exits.
    #[arg(long)]
    stats: bool,
    /// The command to run and its arguments.
    #[arg(last = true, required = true, value_name = "COMMAND")]
    command: Vec<String>,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[command(flatten)]
    proxy: ProxyArgs,
    /// Address to listen on. Stays on loopback unless you change it.
    #[arg(long, default_value = "127.0.0.1:3456")]
    listen: SocketAddr,
}

#[derive(Debug, Args)]
struct BenchArgs {
    /// Directory of Claude Code transcripts, normally ~/.claude/projects.
    #[arg(value_name = "DIR")]
    transcripts: PathBuf,
    /// Also write the results as numbers only, to read and share if you choose. Holds no
    /// prompts, paths, output or session ids, and nothing is sent anywhere.
    #[arg(long, value_name = "FILE")]
    export: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DaemonArgs {
    /// Exit after this long with nothing to do, which gives the memory back.
    #[arg(long, value_name = "SECONDS", default_value_t = daemon::DEFAULT_IDLE.as_secs())]
    idle_seconds: u64,
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct EnhanceArgs {
    /// The prompt to try.
    #[arg(value_name = "PROMPT")]
    prompt: String,
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct WhatifArgs {
    /// Directory of Claude Code transcripts. Defaults to ~/.claude/projects.
    #[arg(value_name = "DIR")]
    transcripts: Option<PathBuf>,
    /// Print only one line of totals, to share if you choose.
    #[arg(long)]
    share: bool,
    /// Also write the results as numbers only, to read and share if you choose. Holds no
    /// prompts, paths, output or session ids, and nothing is sent anywhere.
    #[arg(long, value_name = "FILE")]
    export: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct StateArgs {
    /// Where per-conversation state lives. Defaults to ~/.ccc.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
}

impl StateArgs {
    fn dir(&self) -> anyhow::Result<PathBuf> {
        self.state_dir
            .clone()
            .or_else(default_state_dir)
            .context("no state directory: set --state-dir or HOME")
    }
}

#[derive(Debug, Args)]
struct RouterTrainArgs {
    /// Labelled requests: JSON lines with `prompt` and `cheap_enough`.
    #[arg(value_name = "FILE")]
    labels: PathBuf,
    /// Where to write the trained model.
    #[arg(long, value_name = "FILE")]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct DashboardArgs {
    /// Address to listen on. Stays on loopback unless you change it.
    #[arg(long, default_value = "127.0.0.1:3457")]
    listen: SocketAddr,
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct LearnArgs {
    /// Directory of Claude Code transcripts, normally ~/.claude/projects.
    #[arg(value_name = "DIR")]
    transcripts: PathBuf,
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct IndexArgs {
    /// The repository to index.
    #[arg(value_name = "DIR", default_value = ".")]
    root: PathBuf,
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct EnrichEvalArgs {
    /// Directory of Claude Code transcripts, normally ~/.claude/projects.
    #[arg(value_name = "DIR")]
    transcripts: PathBuf,
    /// Also judge search by meaning, with this GGUF embedding model. Needs the local-model
    /// build feature.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "FILE")]
    embedding_model: Option<PathBuf>,
    /// Also judge letting this GGUF model order the candidates. Needs --embedding-model.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "FILE", requires = "embedding_model")]
    rerank_model: Option<PathBuf>,
    /// The reranking model's prompt format, qwen or gemma4.
    #[cfg(feature = "local-model")]
    #[arg(long, value_name = "FORMAT", default_value = "gemma4")]
    rerank_format: String,
    #[cfg(feature = "local-model")]
    #[command(flatten)]
    state: StateArgs,
}

#[derive(Debug, Args)]
struct AdviseArgs {
    /// Warn when a session's prompt cache has expired and when its context gets large. Warnings
    /// are shown to you and never reach the model.
    #[arg(long)]
    advise: bool,
    /// Stay quiet below this context size.
    #[arg(long, value_name = "TOKENS", default_value_t = 100_000)]
    advise_min_context: u64,
    /// Hold back the first prompt after the cache expires, so you can choose before paying.
    /// Sending it again goes ahead.
    #[arg(long, requires = "advise")]
    advise_block: bool,
}

impl AdviseArgs {
    fn options(&self) -> Option<advise::AdviseOptions> {
        self.advise.then_some(advise::AdviseOptions {
            min_context: self.advise_min_context,
            block: self.advise_block,
        })
    }
}

#[derive(Debug, Args)]
struct HookSettingsArgs {
    #[command(flatten)]
    state: StateArgs,
    #[command(flatten)]
    advice: AdviseArgs,
}

#[derive(Debug, Args)]
struct HookArgs {
    /// Which event fired: session-start, user-prompt-submit, pre-compact or stop.
    #[arg(value_name = "EVENT")]
    event: String,
    #[command(flatten)]
    state: StateArgs,
    #[command(flatten)]
    advice: AdviseArgs,
}

#[cfg(feature = "ner")]
#[derive(Debug, Args)]
struct NerBenchArgs {
    /// The directory holding the GLiNER model.
    #[arg(value_name = "MODEL_DIR")]
    model: PathBuf,
    /// Files to scan, recursively.
    #[arg(value_name = "DIR")]
    files: PathBuf,
}

#[cfg(feature = "local-model")]
#[derive(Debug, Args)]
struct ModelBenchArgs {
    /// A GGUF model file.
    #[arg(value_name = "FILE")]
    model: PathBuf,
    /// Transcripts to take real tool output from, normally ~/.claude/projects.
    #[arg(long, value_name = "DIR")]
    transcripts: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct StatsArgs {
    /// How many recent sessions to show.
    #[arg(long, default_value_t = 20)]
    limit: usize,
    /// Where per-conversation state lives. Defaults to ~/.ccc.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    match Cli::parse().command {
        Command::Run(mut args) => {
            // `run` shares the terminal with a full-screen app, so stderr is off limits.
            init_logging(&args.proxy, LogDefault::Silent)?;
            let client = Client::for_command(&args.command);
            if args.proxy.upstream == DEFAULT_UPSTREAM {
                client
                    .default_upstream()
                    .clone_into(&mut args.proxy.upstream);
            }
            run::run(args.proxy.proxy()?, client, &args.command, args.stats).await
        }
        Command::Serve(args) => {
            init_logging(&args.proxy, LogDefault::Stderr)?;
            serve(args).await?;
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(feature = "ner")]
        Command::NerBench(args) => {
            names::bench(&args.model, &args.files)?;
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(feature = "local-model")]
        Command::ModelBench(args) => {
            model_bench::run(&args.model, args.transcripts.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
        Command::RouterTrain(args) => {
            router::train(&args.labels, &args.out)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Dashboard(args) => {
            dashboard::serve(args.listen, &args.state.dir()?).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Learn(args) => {
            learn::learn(&args.state.dir()?, &args.transcripts)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Index(args) => {
            let root = args.root.canonicalize().context("finding the repository")?;
            learn::index(&args.state.dir()?, &root)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::EnrichEval(args) => {
            #[cfg(feature = "local-model")]
            {
                let state_dir = args.state.dir()?;
                let format = match args.rerank_format.as_str() {
                    "gemma4" => ccc_slm::ChatFormat::Gemma4,
                    "qwen" => ccc_slm::ChatFormat::Qwen,
                    other => anyhow::bail!("unknown prompt format {other:?}"),
                };
                let reranker = args.rerank_model.as_deref().map(|path| (path, format));
                let vectors = args
                    .embedding_model
                    .as_deref()
                    .map(|model| learn::VectorEval {
                        model,
                        reranker,
                        state_dir: &state_dir,
                    });
                learn::enrich_eval(&args.transcripts, vectors.as_ref())?;
            }
            #[cfg(not(feature = "local-model"))]
            learn::enrich_eval(&args.transcripts)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::HookSettings(args) => {
            let ccc = std::env::current_exe().context("finding this program")?;
            let settings = hooks::settings_json(&ccc, &args.state.dir()?, args.advice.options());
            writeln!(std::io::stdout(), "{settings}")?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Hook(args) => {
            hooks::handle(&args.event, &args.state.dir()?, args.advice.options())?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Stats(args) => {
            stats(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Bench(args) => {
            bench::run(&args.transcripts, args.export.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Eval(args) => {
            eval::command(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Setup(args) => {
            setup::setup(&args, &claude_home()?).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Uninstall(args) => {
            setup::uninstall(&args, &claude_home()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Doctor(args) => Ok(if setup::doctor(&args, &claude_home()?)? {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }),
        Command::Status(args) => {
            setup::status(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Gain(args) => {
            gain::run(&args.dir()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Rules(args) => {
            rules::run(&args, &claude_home()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Jobs(args) => {
            sharpen::jobs(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::FetchModel(args) => {
            setup::fetch_model_command(&args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Statusline(args) => {
            statusline::run(&args.dir()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Advanced => {
            let mut out = std::io::stdout();
            writeln!(out, "Other commands, each with its own --help:\n")?;
            let command = <Cli as clap::CommandFactory>::command();
            for sub in command.get_subcommands().filter(|sub| sub.is_hide_set()) {
                let about = sub
                    .get_about()
                    .map_or_else(String::new, ToString::to_string);
                writeln!(out, "  {:<15}{about}", sub.get_name())?;
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Enhance(args) => {
            let added = enhance::hook(&args.state.dir()?, ".", &args.prompt);
            let text = added.unwrap_or_else(|| "Nothing to add for that prompt.".to_owned());
            writeln!(std::io::stdout(), "{text}")?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Embed(args) => {
            local_work::embed(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Ask(args) => {
            local_work::ask(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Mcp(args) => {
            local_work::mcp(&args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Daemon(args) => {
            daemon::run(
                &args.state.dir()?,
                std::time::Duration::from_secs(args.idle_seconds),
            )?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Init(args) => {
            init::run(&args, &claude_home()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Handoff(args) => handoff::run(&args, &claude_home()?),
        Command::Verify(args) => verify::run(&args).await,
        Command::Whatif(args) => {
            let transcripts = match args.transcripts {
                Some(path) => path,
                None => claude_home()?.join("projects"),
            };
            whatif::run(&transcripts, args.export.as_deref(), args.share)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn stats(args: &StatsArgs) -> anyhow::Result<()> {
    let dir = args
        .state_dir
        .clone()
        .or_else(default_state_dir)
        .context("no state directory: set --state-dir or HOME")?;
    let ledger = Ledger::open(&dir.join(LEDGER_FILE)).context("opening ledger")?;
    let sessions = ledger
        .recent_sessions(args.limit)
        .context("reading ledger")?;
    let mut out = std::io::stdout();
    writeln!(
        out,
        "{:<38}{:>7}{:>16}{:>12}{:>11}",
        "session", "calls", "context tokens", "output", "cost"
    )?;
    for session in sessions {
        writeln!(
            out,
            "{:<38}{:>7}{:>16}{:>12}{:>11}",
            session.session,
            session.calls,
            session.context_tokens,
            session.output_tokens,
            format!("${:.4}", session.cost_usd),
        )?;
    }
    Ok(())
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let proxy = args.proxy.proxy()?;
    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    tracing::info!(address = %args.listen, upstream = %args.proxy.upstream, "listening");
    let shutdown = async {
        // A failed signal handler means no graceful stop, so shut down rather than hang.
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %err, "could not listen for ctrl-c");
        }
    };
    proxy
        .serve(listener, shutdown)
        .await
        .context("running proxy")
}

#[derive(Debug, Clone, Copy)]
enum LogDefault {
    Silent,
    Stderr,
}

fn init_logging(args: &ProxyArgs, default: LogDefault) -> anyhow::Result<()> {
    let default_filter = if args.debug { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_env("CCC_LOG").unwrap_or_else(|_| EnvFilter::new(default_filter));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match (args.log_file.as_deref(), default) {
        (Some(path), _) => {
            let log = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("opening log file {}", path.display()))?;
            builder
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(log))
                .init();
        }
        (None, LogDefault::Stderr) => builder.with_writer(std::io::stderr).init(),
        (None, LogDefault::Silent) => {}
    }
    Ok(())
}

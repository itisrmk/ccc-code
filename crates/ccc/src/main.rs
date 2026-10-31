//! `ccc`: a local-first prompt compiler for Claude Code.

mod bench;
mod export;
mod hooks;
mod learn;
#[cfg(feature = "local-model")]
mod local_models;
#[cfg(feature = "local-model")]
mod model_bench;
mod run;

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

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
    /// Start the proxy and run a command, normally `claude`, pointed at it.
    Run(RunArgs),
    /// Start the proxy and wait; point ANTHROPIC_BASE_URL at it yourself.
    Serve(ServeArgs),
    /// Replay local transcripts and report token and cost effects. Nothing leaves the machine.
    Bench(BenchArgs),
    /// Show what recent sessions cost at list prices.
    Stats(StatsArgs),
    /// Learn which files past work needed, from local transcripts. Nothing leaves the machine.
    Learn(LearnArgs),
    /// Build or refresh a repository's symbol index.
    Index(IndexArgs),
    /// Judge file suggestions against what past work really touched.
    EnrichEval(EnrichEvalArgs),
    /// Print the settings that register ccc's hooks, for `claude --settings`.
    HookSettings(StateArgs),
    /// Handle a Claude Code hook. Called by Claude Code, not by hand.
    Hook(HookArgs),
    /// Measure a local model on this machine. Needs the local-model build feature.
    #[cfg(feature = "local-model")]
    ModelBench(ModelBenchArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ProxyArgs {
    /// Base URL to forward to. Defaults to the Anthropic API.
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
        let state_dir = self.state_dir.clone().or_else(default_state_dir);
        let vault = Arc::new(match &state_dir {
            Some(dir) => Vault::open(&dir.join("redact.key")).context("opening redaction key")?,
            None => Vault::ephemeral().context("creating redaction key")?,
        });
        let redact = Redact::new(self.detector()?, Arc::clone(&vault));
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

fn default_state_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".ccc"))
}

impl ProxyArgs {
    fn detector(&self) -> anyhow::Result<Detector> {
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
}

#[derive(Debug, Args)]
struct HookArgs {
    /// Which event fired: session-start, pre-compact or stop.
    #[arg(value_name = "EVENT")]
    event: String,
    #[command(flatten)]
    state: StateArgs,
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
        Command::Run(args) => {
            // `run` shares the terminal with a full-screen app, so stderr is off limits.
            init_logging(&args.proxy, LogDefault::Silent)?;
            run::run(args.proxy.proxy()?, &args.command, args.stats).await
        }
        Command::Serve(args) => {
            init_logging(&args.proxy, LogDefault::Stderr)?;
            serve(args).await?;
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(feature = "local-model")]
        Command::ModelBench(args) => {
            model_bench::run(&args.model, args.transcripts.as_deref())?;
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
            learn::enrich_eval(&args.transcripts)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::HookSettings(args) => {
            let ccc = std::env::current_exe().context("finding this program")?;
            let settings = hooks::settings_json(&ccc, &args.dir()?);
            writeln!(std::io::stdout(), "{settings}")?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Hook(args) => {
            hooks::handle(&args.event, &args.state.dir()?)?;
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

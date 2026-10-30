//! `ccc`: a local-first prompt compiler for Claude Code.

mod bench;
mod run;

use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context;
use ccc_pipeline::{Pipeline, Stage};
use ccc_proxy::{Proxy, ProxyConfig};
use ccc_prune::Prune;
use clap::{Args, Parser, Subcommand};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

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
}

#[derive(Debug, Args)]
struct ProxyArgs {
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
    /// Where per-conversation state lives. Defaults to ~/.ccc.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
}

impl ProxyArgs {
    fn proxy(&self) -> anyhow::Result<Proxy> {
        let mut disabled_stages = BTreeSet::new();
        if self.no_prune {
            disabled_stages.insert(Prune::new().name().to_owned());
        }
        let config = ProxyConfig {
            upstream: self.upstream.clone(),
            dump_dir: self.dump.clone(),
            pipeline: Pipeline::new(vec![Arc::new(Prune::new())]),
            disabled_stages,
            state_dir: self.state_dir.clone().or_else(default_state_dir),
        };
        Proxy::new(config).context("configuring proxy")
    }
}

fn default_state_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".ccc"))
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
        Command::Bench(args) => {
            bench::run(&args.transcripts)?;
            Ok(ExitCode::SUCCESS)
        }
    }
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

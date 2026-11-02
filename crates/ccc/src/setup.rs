//! `ccc setup`, `ccc uninstall`, `ccc doctor` and `ccc status`: from nothing to working in one
//! command, and back again without leaving anything behind.

use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail};
use ccc_slm::Embedder as _;
use clap::Args;
use futures_util::StreamExt as _;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::config::{Backend, DEFAULT_ANSWERING, DEFAULT_EMBEDDING, Jobs, LocalConfig};
use crate::daemon::{DaemonClient, socket_path};
use crate::hooks::shell_quote;
use crate::{gain, init};

/// Below this, a 2.8 GB model plus everything else open makes the whole machine swap.
const MIN_MEMORY_FOR_ANSWERING_GB: u64 = 16;
const OLLAMA_URL: &str = "http://127.0.0.1:11434/v1";
const MODEL_HOST: &str = "https://huggingface.co";
/// Where to look for a model, its checksum, and roughly how big it is.
const MODELS: [(&str, &str, &str, u64); 2] = [
    (
        DEFAULT_EMBEDDING,
        "ggml-org/embeddinggemma-300M-GGUF",
        "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
        334,
    ),
    (
        DEFAULT_ANSWERING,
        "ggml-org/gemma-4-E2B-it-GGUF",
        "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52",
        2841,
    ),
];
const MCP_NAME: &str = "ccc";
const DOWNLOAD_PID: &str = "download.pid";
/// How often the status line refreshes when nothing else happens, for the cache countdown.
const STATUS_REFRESH_SECONDS: u32 = 60;
/// Marks a status line as ours, so it can be found again.
const OUR_STATUS: &str = " statusline";
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Args)]
pub(crate) struct SetupArgs {
    /// Use a model server you already run, such as Ollama or LM Studio, instead of downloading.
    /// Its OpenAI-compatible base URL, such as http://127.0.0.1:11434/v1.
    #[arg(long, value_name = "URL", requires = "embedding_model")]
    server: Option<String>,
    /// With --server, the embedding model's name there.
    #[arg(long, value_name = "NAME")]
    embedding_model: Option<String>,
    /// With --server, the model that answers questions.
    #[arg(long, value_name = "NAME")]
    answer_model: Option<String>,
    /// Skip the larger model, keeping search by meaning only. Chosen for you below 16 GB of memory.
    #[arg(long, conflicts_with = "full")]
    light: bool,
    /// Download the larger model even with less than 16 GB of memory.
    #[arg(long)]
    full: bool,
    /// Wait for the larger model to download, instead of letting it finish in the background.
    #[arg(long)]
    wait: bool,
    /// Don't offer local tools to Claude.
    #[arg(long)]
    no_mcp: bool,
    /// Change this settings file. Defaults to ~/.claude/settings.json.
    #[arg(long, value_name = "FILE")]
    settings_file: Option<PathBuf>,
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Args)]
pub(crate) struct FetchModelArgs {
    /// The model's file name.
    #[arg(value_name = "NAME")]
    name: String,
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Args)]
pub(crate) struct UninstallArgs {
    /// Also delete everything in the state directory: models, vectors, history and settings.
    #[arg(long)]
    purge: bool,
    /// Don't touch MCP registration.
    #[arg(long)]
    no_mcp: bool,
    /// Change this settings file. Defaults to ~/.claude/settings.json.
    #[arg(long, value_name = "FILE")]
    settings_file: Option<PathBuf>,
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    /// Check this settings file. Defaults to ~/.claude/settings.json.
    #[arg(long, value_name = "FILE")]
    settings_file: Option<PathBuf>,
    /// Don't start the service to test it.
    #[arg(long)]
    no_start: bool,
    #[command(flatten)]
    state: crate::StateArgs,
}

fn say(text: &str) {
    // Progress is a courtesy, so a closed pipe shouldn't stop the install.
    let _ = writeln!(std::io::stdout(), "{text}");
}

/// This machine's memory in whole gigabytes, or `None` if it can't be told.
fn memory_gb() -> Option<u64> {
    if cfg!(target_os = "macos") {
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let bytes: u64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;
        return Some(bytes / (1024 * 1024 * 1024));
    }
    let info = fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kilobytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kilobytes / (1024 * 1024))
}

/// Whether to fetch the answering model, and why not if it's skipped.
fn wants_answering(memory: Option<u64>, light: bool, full: bool) -> Result<(), String> {
    if light {
        return Err("skipped, as you asked".to_owned());
    }
    match memory {
        Some(gb) if gb < MIN_MEMORY_FOR_ANSWERING_GB && !full => Err(format!(
            "skipped: this machine has {gb} GB of memory, and it needs about 16. --full fetches it anyway"
        )),
        _ => Ok(()),
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).context("checking the file")?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Downloads to the side and renames only after the checksum matches, so a stopped or corrupted
/// download can never be loaded.
async fn download(url: &str, expected: &str, destination: &Path) -> anyhow::Result<()> {
    if destination.exists() && sha256_file(destination)? == expected {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).context("creating the models directory")?;
    }
    let partial = destination.with_extension("part");
    let response = reqwest::get(url)
        .await
        .and_then(reqwest::Response::error_for_status)
        .with_context(|| format!("downloading {url}"))?;
    let total = response.content_length();
    let mut file = fs::File::create(&partial).context("creating the download")?;
    let mut hasher = Sha256::new();
    let mut received = 0_u64;
    let mut shown = None;
    let mut stream = response.bytes_stream();
    while let Some(part) = stream.next().await {
        let part = part.with_context(|| format!("downloading {url}"))?;
        file.write_all(&part).context("saving the download")?;
        hasher.update(&part);
        received += part.len() as u64;
        if let Some(total) = total.filter(|total| *total > 0) {
            let percent = received * 100 / total;
            if shown != Some(percent) {
                shown = Some(percent);
                let mut err = std::io::stderr();
                let _ = write!(err, "\r  {percent}% of {} MB", total / 1_000_000);
                let _ = err.flush();
            }
        }
    }
    let _ = writeln!(std::io::stderr());
    drop(file);
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        fs::remove_file(&partial).ok();
        bail!(
            "the download of {url} doesn't match its checksum: expected {expected}, got {actual}"
        );
    }
    fs::rename(&partial, destination).context("finishing the download")
}

fn model_url(repository: &str, file: &str) -> String {
    let host = std::env::var("CCC_MODEL_HOST").unwrap_or_else(|_| MODEL_HOST.to_owned());
    format!("{host}/{repository}/resolve/main/{file}")
}

async fn fetch_model(state_dir: &Path, name: &str) -> anyhow::Result<()> {
    let (file, repository, checksum, megabytes) = MODELS
        .iter()
        .find(|(file, ..)| *file == name)
        .with_context(|| format!("no download known for {name}"))?;
    say(&format!("Getting {file} ({megabytes} MB)..."));
    download(
        &model_url(repository, file),
        checksum,
        &LocalConfig::model_file(state_dir, file),
    )
    .await
}

/// The larger model takes minutes, and nothing else needs to wait for it.
fn fetch_in_background(ccc: &Path, state_dir: &Path, name: &str) -> anyhow::Result<()> {
    fs::create_dir_all(state_dir).context("creating the state directory")?;
    let log = fs::File::create(state_dir.join("download.log")).context("creating a log")?;
    let child = Command::new(ccc)
        .args(["fetch-model", name, "--state-dir"])
        .arg(state_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .process_group(0)
        .spawn()
        .context("starting the download")?;
    fs::write(state_dir.join(DOWNLOAD_PID), child.id().to_string())
        .context("recording the download")
}

/// Stops a background download, so uninstalling doesn't leave one writing files.
fn stop_download(state_dir: &Path) {
    let file = state_dir.join(DOWNLOAD_PID);
    if let Some(pid) = fs::read_to_string(&file)
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok())
    {
        // Only if it's still ours: the id may since have gone to something else.
        let running_ours = Command::new("ps")
            .args(["-o", "command=", "-p", &pid.to_string()])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains("fetch-model"));
        if running_ours {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
    }
    fs::remove_file(file).ok();
}

/// `ccc fetch-model`: the background half of setup.
pub(crate) async fn fetch_model_command(args: &FetchModelArgs) -> anyhow::Result<()> {
    fetch_model(&args.state.dir()?, &args.name).await
}

/// How far along a background download is, if one is going.
fn download_progress(state_dir: &Path, name: &str) -> Option<String> {
    let (.., megabytes) = MODELS.iter().find(|(file, ..)| *file == name)?;
    let partial = LocalConfig::model_file(state_dir, name).with_extension("part");
    let done = fs::metadata(partial).ok()?.len() / 1_000_000;
    Some(format!("{name} is downloading: {done} of {megabytes} MB"))
}

/// Reads `path` as a settings object, or an empty one if it isn't there. Anything else is left
/// alone and reported, since someone's settings are at stake.
fn read_settings(path: &Path) -> anyhow::Result<(Option<String>, Map<String, Value>)> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok((None, Map::new()));
        }
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    if text.trim().is_empty() {
        return Ok((Some(text), Map::new()));
    }
    match serde_json::from_str(&text) {
        Ok(Value::Object(settings)) => Ok((Some(text), settings)),
        _ => bail!(
            "{} isn't a JSON object, so it was left unchanged",
            path.display()
        ),
    }
}

/// Keeps a copy of what was there and replaces it atomically.
fn write_settings(
    path: &Path,
    before: Option<&str>,
    settings: &Map<String, Value>,
) -> anyhow::Result<()> {
    let rendered = format!(
        "{}\n",
        serde_json::to_string_pretty(&Value::Object(settings.clone())).context("rendering")?
    );
    if before == Some(rendered.as_str()) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if before.is_some() {
        let backup = path.with_extension("json.ccc-backup");
        fs::copy(path, &backup).with_context(|| format!("writing {}", backup.display()))?;
    }
    let temporary = path.with_extension("json.ccc-new");
    fs::write(&temporary, rendered).with_context(|| format!("writing {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replacing {}", path.display()))
}

fn status_is_ours(settings: &Map<String, Value>) -> bool {
    settings
        .get("statusLine")
        .and_then(|line| line.get("command"))
        .and_then(Value::as_str)
        .is_some_and(|command| command.contains(OUR_STATUS))
}

/// Adds our status line, unless the person already has their own. Returns whether it's in place.
fn add_status_line(settings: &mut Map<String, Value>, ccc: &Path, state_dir: &Path) -> bool {
    if settings.contains_key("statusLine") && !status_is_ours(settings) {
        return false;
    }
    let command = format!(
        "{}{OUR_STATUS} --state-dir {}",
        shell_quote(&ccc.to_string_lossy()),
        shell_quote(&state_dir.to_string_lossy()),
    );
    settings.insert(
        "statusLine".to_owned(),
        json!({
            "type": "command",
            "command": command,
            "refreshInterval": STATUS_REFRESH_SECONDS,
        }),
    );
    true
}

fn remove_status_line(settings: &mut Map<String, Value>) {
    if status_is_ours(settings) {
        settings.remove("statusLine");
    }
}

fn claude_is_installed() -> bool {
    Command::new("claude")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn remove_mcp() {
    let _ = Command::new("claude")
        .args(["mcp", "remove", "--scope", "user", MCP_NAME])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn add_mcp(ccc: &Path, state_dir: &Path) -> anyhow::Result<()> {
    // Removing first is what makes running setup again safe after the binary moves.
    remove_mcp();
    let output = Command::new("claude")
        .args([
            "mcp",
            "add",
            "--transport",
            "stdio",
            "--scope",
            "user",
            MCP_NAME,
            "--",
        ])
        .arg(ccc)
        .arg("mcp")
        .arg("--state-dir")
        .arg(state_dir)
        .stdin(Stdio::null())
        .output()
        .context("running claude mcp add")?;
    if output.status.success() {
        Ok(())
    } else {
        bail!(
            "claude mcp add failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
}

/// Models another server already has, if one is running.
async fn running_server_models(url: &str) -> Option<Vec<String>> {
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
        .ok()?;
    let response = client.get(format!("{url}/models")).send().await.ok()?;
    let body: Value = serde_json::from_slice(&response.bytes().await.ok()?).ok()?;
    let names = body
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|model| model.get("id")?.as_str().map(str::to_owned))
        .collect();
    Some(names)
}

fn stop_service(state_dir: &Path) {
    if let Some(pid) = DaemonClient::existing_only(state_dir).ping() {
        let _ = Command::new("kill").arg(pid.to_string()).status();
    }
    let started = Instant::now();
    while DaemonClient::existing_only(state_dir).ping().is_some()
        && started.elapsed() < Duration::from_secs(5)
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    fs::remove_file(socket_path(state_dir)).ok();
}

pub(crate) async fn setup(args: &SetupArgs, claude_home: &Path) -> anyhow::Result<()> {
    let started = Instant::now();
    let state_dir = args.state.dir()?;
    let ccc = std::env::current_exe().context("finding this program")?;
    let settings_path = args
        .settings_file
        .clone()
        .unwrap_or_else(|| claude_home.join("settings.json"));

    // Running setup again mustn't quietly turn off what someone turned on.
    let earlier_jobs = LocalConfig::load(&state_dir).map_or_else(|_| Jobs::default(), |c| c.jobs);
    let config = if let (Some(url), Some(embedding)) = (&args.server, &args.embedding_model) {
        say(&format!("Using the model server at {url}."));
        LocalConfig {
            backend: Backend::Server { url: url.clone() },
            embedding: embedding.clone(),
            answering: args.answer_model.clone(),
            jobs: earlier_jobs,
        }
    } else {
        if let Some(models) = running_server_models(OLLAMA_URL).await {
            say(&format!(
                "Ollama is running here with {} models. To use it instead of downloading: \
                 ccc setup --server {OLLAMA_URL} --embedding-model NAME --answer-model NAME",
                models.len()
            ));
        }
        let memory = memory_gb();
        let answering = match wants_answering(memory, args.light, args.full) {
            Ok(()) => Some(DEFAULT_ANSWERING.to_owned()),
            Err(reason) => {
                say(&format!("Answering model {reason}."));
                None
            }
        };
        fetch_model(&state_dir, DEFAULT_EMBEDDING).await?;
        if let Some(name) = &answering {
            if args.wait {
                fetch_model(&state_dir, name).await?;
            } else if !LocalConfig::model_file(&state_dir, name).exists() {
                fetch_in_background(&ccc, &state_dir, name)?;
                say(&format!(
                    "Getting {name} in the background, so you can start now. ccc status shows how far along it is."
                ));
            }
        }
        LocalConfig {
            backend: Backend::Builtin,
            embedding: DEFAULT_EMBEDDING.to_owned(),
            answering,
            jobs: earlier_jobs,
        }
    };
    // A service already running holds the old settings, so the next use starts a fresh one.
    stop_service(&state_dir);
    config.save(&state_dir)?;

    let (before, mut settings) = read_settings(&settings_path)?;
    init::install_hooks(&mut settings, &ccc, &state_dir)?;
    let status_added = add_status_line(&mut settings, &ccc, &state_dir);
    write_settings(&settings_path, before.as_deref(), &settings)?;
    say(&format!(
        "Added cost warnings to {}.",
        settings_path.display()
    ));
    if status_added {
        say("Added the status line.");
    } else {
        say("You already have a status line, so it was left alone.");
    }

    if args.no_mcp {
        say("Skipped local tools for Claude, as you asked.");
    } else if claude_is_installed() {
        add_mcp(&ccc, &state_dir)?;
        say("Offered local tools to Claude.");
    } else {
        say(
            "Claude Code isn't installed, so local tools weren't offered. Run ccc setup again after installing it.",
        );
    }
    say(&format!(
        "\nReady in {}s. New Claude Code sessions will pick this up.\n\nTry:\n  ccc whatif         where your Claude usage has gone\n  ccc ask \"...\"       ask about this code, free and offline\n  ccc doctor         check everything's working",
        started.elapsed().as_secs()
    ));
    Ok(())
}

pub(crate) fn uninstall(args: &UninstallArgs, claude_home: &Path) -> anyhow::Result<()> {
    let state_dir = args.state.dir()?;
    let settings_path = args
        .settings_file
        .clone()
        .unwrap_or_else(|| claude_home.join("settings.json"));
    stop_service(&state_dir);
    stop_download(&state_dir);
    if settings_path.exists() {
        let (before, mut settings) = read_settings(&settings_path)?;
        init::remove_hooks(&mut settings)?;
        remove_status_line(&mut settings);
        write_settings(&settings_path, before.as_deref(), &settings)?;
        fs::remove_file(settings_path.with_extension("json.ccc-backup")).ok();
        say(&format!(
            "Removed hooks and status line from {}.",
            settings_path.display()
        ));
    }
    if !args.no_mcp && claude_is_installed() {
        remove_mcp();
        say("Removed local tools from Claude.");
    }
    if args.purge {
        match fs::remove_dir_all(&state_dir) {
            Ok(()) => say(&format!("Deleted {}.", state_dir.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| format!("deleting {}", state_dir.display()));
            }
        }
    } else {
        say(&format!(
            "Kept {} with your models and history. --purge deletes it.",
            state_dir.display()
        ));
    }
    Ok(())
}

/// One line of a health check.
struct Finding {
    ok: bool,
    /// Worth knowing, but nothing's broken.
    optional: bool,
    text: String,
}

fn finding(ok: bool, text: String) -> Finding {
    Finding {
        ok,
        optional: false,
        text,
    }
}

fn memory_of(pid: u32) -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let kilobytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(kilobytes / 1024)
}

fn check_models(state_dir: &Path, config: &LocalConfig) -> Vec<Finding> {
    if let Backend::Server { url } = &config.backend {
        return vec![finding(true, format!("Models are served by {url}"))];
    }
    let mut names = vec![config.embedding.as_str()];
    names.extend(config.answering.as_deref());
    let mut found: Vec<Finding> = names
        .into_iter()
        .map(|name| {
            let path = LocalConfig::model_file(state_dir, name);
            match fs::metadata(&path) {
                Ok(meta) => finding(true, format!("{name} ({} MB)", meta.len() / 1_000_000)),
                Err(_) => match download_progress(state_dir, name) {
                    Some(progress) => Finding {
                        ok: true,
                        optional: true,
                        text: progress,
                    },
                    None => finding(false, format!("{name} is missing: run ccc setup")),
                },
            }
        })
        .collect();
    if config.answering.is_none() {
        found.push(Finding {
            ok: true,
            optional: true,
            text: "No answering model, so ccc ask won't work: ccc setup --full adds it".to_owned(),
        });
    }
    found
}

fn check_service(state_dir: &Path, start: bool) -> Vec<Finding> {
    let existing = DaemonClient::existing_only(state_dir).ping();
    if existing.is_none() && !start {
        return vec![finding(
            true,
            "The background service isn't running, which is normal: it starts when needed"
                .to_owned(),
        )];
    }
    let client = DaemonClient::new(state_dir);
    let began = Instant::now();
    let probe = client.embed("task: code retrieval | query: is this working");
    let mut found = vec![match probe {
        Ok(vector) => finding(
            true,
            format!(
                "Search by meaning works ({} dimensions, {} ms)",
                vector.len(),
                began.elapsed().as_millis()
            ),
        ),
        Err(err) => finding(false, format!("Search by meaning failed: {err}")),
    }];
    if let Some(pid) = client.ping()
        && let Some(megabytes) = memory_of(pid)
    {
        found.push(Finding {
            ok: true,
            optional: true,
            text: format!(
                "The background service is using {megabytes} MB, until it's idle for 10 minutes"
            ),
        });
    }
    found
}

fn check_settings(path: &Path) -> Vec<Finding> {
    let settings = match read_settings(path) {
        Ok((_, settings)) => settings,
        Err(err) => return vec![finding(false, err.to_string())],
    };
    let hooks = init::hooks_installed(&settings);
    let mut found = vec![finding(
        hooks,
        if hooks {
            format!("Cost warnings are on in {}", path.display())
        } else {
            format!("Cost warnings aren't in {}: run ccc setup", path.display())
        },
    )];
    found.push(if status_is_ours(&settings) {
        finding(true, "The status line is on".to_owned())
    } else {
        Finding {
            ok: true,
            optional: true,
            text: "The status line isn't ours, so limits and cache won't show".to_owned(),
        }
    });
    found
}

fn check_mcp() -> Vec<Finding> {
    if !claude_is_installed() {
        return vec![finding(false, "Claude Code isn't installed".to_owned())];
    }
    let registered = Command::new("claude")
        .args(["mcp", "get", MCP_NAME])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    vec![Finding {
        ok: true,
        optional: !registered,
        text: if registered {
            "Local tools are offered to Claude".to_owned()
        } else {
            "Local tools aren't offered to Claude: ccc setup adds them".to_owned()
        },
    }]
}

/// Runs every check and prints them. Returns whether nothing's broken.
pub(crate) fn doctor(args: &DoctorArgs, claude_home: &Path) -> anyhow::Result<bool> {
    let state_dir = args.state.dir()?;
    let settings_path = args
        .settings_file
        .clone()
        .unwrap_or_else(|| claude_home.join("settings.json"));
    let config = LocalConfig::load(&state_dir)?;
    let mut findings = check_settings(&settings_path);
    findings.extend(check_mcp());
    findings.extend(check_models(&state_dir, &config));
    findings.extend(check_service(&state_dir, !args.no_start));
    if let Some(gb) = memory_gb() {
        findings.push(Finding {
            ok: true,
            optional: true,
            text: format!("{gb} GB of memory"),
        });
    }
    let mut text = String::new();
    for found in &findings {
        let mark = match (found.ok, found.optional) {
            (false, _) => "FAIL",
            (true, true) => "note",
            (true, false) => "ok  ",
        };
        // Writing to a String can't fail.
        let _ = writeln!(text, "{mark}  {}", found.text);
    }
    write!(std::io::stdout(), "{text}")?;
    Ok(findings.iter().all(|found| found.ok))
}

/// `ccc status`: where things stand, without starting anything.
pub(crate) fn status(state: &crate::StateArgs) -> anyhow::Result<()> {
    let state_dir = state.dir()?;
    let config = LocalConfig::load(&state_dir)?;
    let mut out = std::io::stdout();
    let models = match &config.backend {
        Backend::Builtin => "built in".to_owned(),
        Backend::Server { url } => format!("from {url}"),
    };
    writeln!(
        out,
        "Models: {} and {}, {models}",
        config.embedding,
        config.answering.as_deref().unwrap_or("no answering model"),
    )?;
    if let Some(progress) = config
        .answering
        .as_deref()
        .filter(|name| !LocalConfig::model_file(&state_dir, name).exists())
        .and_then(|name| download_progress(&state_dir, name))
    {
        writeln!(out, "{progress}")?;
    }
    match DaemonClient::existing_only(&state_dir).ping() {
        Some(pid) => {
            let memory = memory_of(pid).map_or_else(String::new, |mb| format!(", using {mb} MB"));
            writeln!(out, "Service: running{memory}")?;
        }
        None => writeln!(out, "Service: stopped, and starts when needed")?,
    }
    if let Ok(root) = std::env::current_dir().and_then(fs::canonicalize) {
        let vectors = crate::learn::semantic_cache_path(&state_dir, &root);
        let chunks = ccc_index::SemanticIndex::load(&vectors).len();
        if chunks > 0 {
            writeln!(out, "This directory: {chunks} chunks ready to search")?;
        } else {
            writeln!(
                out,
                "This directory: not indexed yet. ccc embed does it now"
            )?;
        }
    }
    writeln!(
        out,
        "Kept out of Claude's context today: {} tokens",
        gain::today_tokens(&state_dir)
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;
    use std::net::TcpListener;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ccc-setup-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Serves `body` to one request at a time, and returns its URL.
    fn serve(body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut buffer = [0; 2048];
                let _ = stream.read(&mut buffer).unwrap();
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(body).unwrap();
            }
        });
        url
    }

    fn sha(body: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(body);
        format!("{:x}", hasher.finalize())
    }

    #[tokio::test]
    async fn a_good_download_lands_and_a_bad_one_never_does() {
        let dir = dir("download");
        let url = serve(b"model bytes");
        let good = dir.join("models/good.gguf");

        download(&url, &sha(b"model bytes"), &good).await.unwrap();
        assert_eq!(fs::read(&good).unwrap(), b"model bytes");

        let bad = dir.join("models/bad.gguf");
        let err = download(&url, &"0".repeat(64), &bad).await.unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        assert!(
            !bad.exists(),
            "a file that failed its checksum must not be usable"
        );
        assert!(!bad.with_extension("part").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_file_already_right_isnt_fetched_again() {
        let dir = dir("cached");
        let path = dir.join("model.gguf");
        fs::write(&path, b"already here").unwrap();
        // The URL can't be reached, so this only passes if nothing's fetched.
        download("http://127.0.0.1:1/x", &sha(b"already here"), &path)
            .await
            .unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn memory_decides_about_the_larger_model_unless_told_otherwise() {
        assert!(wants_answering(Some(16), false, false).is_ok());
        assert!(
            wants_answering(Some(8), false, false)
                .unwrap_err()
                .contains("8 GB")
        );
        assert!(wants_answering(Some(8), false, true).is_ok());
        assert!(wants_answering(Some(64), true, false).is_err());
        assert!(wants_answering(None, false, false).is_ok());
    }

    #[test]
    fn someone_elses_status_line_is_left_alone() {
        let mut settings = Map::new();
        settings.insert(
            "statusLine".to_owned(),
            json!({ "type": "command", "command": "mine.sh" }),
        );
        assert!(!add_status_line(
            &mut settings,
            Path::new("/c"),
            Path::new("/s")
        ));
        remove_status_line(&mut settings);
        assert_eq!(settings["statusLine"]["command"], "mine.sh");
    }

    #[test]
    fn our_status_line_goes_in_and_comes_out() {
        let mut settings = Map::new();
        assert!(add_status_line(
            &mut settings,
            Path::new("/my tools/ccc"),
            Path::new("/s")
        ));
        let command = settings["statusLine"]["command"].as_str().unwrap();
        assert!(
            command.starts_with("'/my tools/ccc' statusline"),
            "{command}"
        );
        assert!(add_status_line(
            &mut settings,
            Path::new("/moved/ccc"),
            Path::new("/s")
        ));
        remove_status_line(&mut settings);
        assert!(settings.is_empty());
    }

    #[test]
    fn broken_settings_are_refused_and_not_overwritten() {
        let dir = dir("broken");
        let path = dir.join("settings.json");
        fs::write(&path, "{ not json").unwrap();
        assert!(read_settings(&path).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "{ not json");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn writing_keeps_a_copy_and_doing_nothing_touches_nothing() {
        let dir = dir("write");
        let path = dir.join("settings.json");
        fs::write(&path, "{\"a\":1}").unwrap();
        let (before, mut settings) = read_settings(&path).unwrap();
        settings.insert("b".to_owned(), json!(2));
        write_settings(&path, before.as_deref(), &settings).unwrap();
        assert_eq!(
            fs::read_to_string(path.with_extension("json.ccc-backup")).unwrap(),
            "{\"a\":1}"
        );
        let (before, settings) = read_settings(&path).unwrap();
        fs::remove_file(path.with_extension("json.ccc-backup")).unwrap();
        write_settings(&path, before.as_deref(), &settings).unwrap();
        assert!(!path.with_extension("json.ccc-backup").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}

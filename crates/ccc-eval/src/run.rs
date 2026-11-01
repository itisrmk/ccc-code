//! Runs an agent on every task under every setup, and scores it by running the tests.

use std::collections::{HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::EvalError;
use crate::mine::{Task, TaskFile};
use crate::process::{Ended, run_limited};
use crate::workspace::{TESTS_MARKER, Workspace};

const MICROS_PER_USD: f64 = 1_000_000.0;
/// What an agent may use, enough to do the work without asking.
const ALLOWED_TOOLS: &str = "Bash,Read,Edit,Write,Grep,Glob";

/// A configuration to test, as files placed over the repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setup {
    /// Its name in reports.
    pub name: String,
    /// Files to place over the repository, or `None` to add nothing.
    pub overlay: Option<PathBuf>,
}

/// How to run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// The setups, baseline first.
    pub setups: Vec<Setup>,
    /// Runs of each task under each setup.
    pub repetitions: usize,
    /// The model to run on.
    pub model: String,
    /// The agent program, normally `claude`.
    pub agent: PathBuf,
    /// Runs at a time.
    pub jobs: usize,
    /// Start no new run once this much has been spent at list prices.
    pub max_usd: f64,
    /// What one run may spend at list prices.
    pub run_max_usd: f64,
    /// How long one run may take.
    pub run_timeout: Duration,
    /// How long scoring may take.
    pub test_timeout: Duration,
    /// Where records go. Runs already recorded here are skipped, so a stopped run can continue.
    pub out: PathBuf,
    /// A directory for temporary copies.
    pub scratch: PathBuf,
    /// Where the agent keeps a folder per working directory, normally `~/.claude/projects`. It
    /// makes one for each throwaway workspace even when told to save no session.
    pub agent_projects: Option<PathBuf>,
}

/// One run's result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The task.
    pub task: String,
    /// The setup.
    pub setup: String,
    /// Which repetition.
    pub repetition: usize,
    /// The model.
    pub model: String,
    /// Whether the tests passed afterward, with the task's test files put back first.
    pub passed: bool,
    /// Whether the run changed a test file. It doesn't help: they're restored before scoring.
    pub tests_touched: bool,
    /// What the agent reported spending.
    pub cost_usd: f64,
    /// Turns the agent reported.
    pub turns: u64,
    /// How long the run took.
    pub seconds: f64,
    /// Why the agent stopped early, if it did.
    pub error: Option<String>,
}

/// Reads the records in `path`, or none if it doesn't exist yet. A line cut off by a crash
/// is skipped, and that run happens again.
///
/// # Errors
///
/// Returns an error if the file exists and can't be read.
pub fn load_records(path: &Path) -> Result<Vec<Record>, EvalError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(source) => Err(EvalError::io(format!("reading {}", path.display()), source)),
    }
}

fn prompt(task: &Task, test_command: &str) -> String {
    let command = test_command.replace(TESTS_MARKER, &task.test_paths.join(" "));
    format!(
        "A change to this repository is described below. Tests for it are already in place and \
         currently fail. Run them with: {command}\n\nChange the source code so they pass. Don't \
         change these test files: {}.\n\nThe change:\n{}",
        task.test_paths.join(", "),
        task.message,
    )
}

struct Job<'a> {
    task: &'a Task,
    setup: &'a Setup,
    repetition: usize,
}

struct AgentReport {
    cost_usd: f64,
    turns: u64,
    error: Option<String>,
}

fn read_agent_report(output: &Path, ended: Ended) -> AgentReport {
    let parsed = fs::read_to_string(output)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let field_error = parsed.as_ref().and_then(|value| {
        (value.get("is_error").and_then(Value::as_bool) == Some(true)).then(|| {
            value
                .get("subtype")
                .and_then(Value::as_str)
                .unwrap_or("agent error")
                .to_owned()
        })
    });
    let error = match (ended, &parsed) {
        (Ended::TimedOut, _) => Some("timed out".to_owned()),
        (_, None) => Some("no report from the agent".to_owned()),
        _ => field_error,
    };
    let number = |key: &str| parsed.as_ref().and_then(|value| value.get(key));
    AgentReport {
        cost_usd: number("total_cost_usd")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        turns: number("num_turns").and_then(Value::as_u64).unwrap_or(0),
        error,
    }
}

fn run_one(job: &Job<'_>, tasks: &TaskFile, options: &RunOptions) -> Result<Record, EvalError> {
    let started = Instant::now();
    let name = format!("{}-{}-{}", job.task.id, job.setup.name, job.repetition);
    let dir = options.scratch.join(&name);
    let workspace = Workspace::create(
        dir.clone(),
        &tasks.repo,
        job.task,
        job.setup.overlay.as_deref(),
    )?;
    let output = options.scratch.join(format!("{name}.json"));
    let stdout = File::create(&output)
        .map_err(|source| EvalError::io(format!("creating {}", output.display()), source))?;
    // Only the project's settings, so the person's own CLAUDE.md, hooks and MCP servers can't
    // affect a run, and no session gets added to their history.
    let ended = run_limited(
        Command::new(&options.agent)
            .arg("-p")
            .arg(prompt(job.task, &tasks.test_command))
            .args(["--model", &options.model])
            .args(["--allowedTools", ALLOWED_TOOLS])
            .args(["--output-format", "json"])
            .args(["--setting-sources", "project"])
            .arg("--no-session-persistence")
            .arg("--strict-mcp-config")
            .args(["--max-budget-usd", &options.run_max_usd.to_string()])
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(Stdio::null()),
        options.run_timeout,
    )
    .map_err(|source| EvalError::io("running the agent", source))?;
    let report = read_agent_report(&output, ended);
    let _ = fs::remove_file(&output);

    // A run can't pass by editing the tests, because they go back before scoring.
    let tests_touched = workspace.changed(&job.task.test_paths)?;
    workspace.restore(&tasks.repo, &job.task.commit, &job.task.test_paths)?;
    let scored = workspace.run_tests(job.task, &tasks.test_command, options.test_timeout)?;
    workspace.remove();
    if let Some(projects) = &options.agent_projects {
        forget_workspace(projects, &dir);
    }
    Ok(Record {
        task: job.task.id.clone(),
        setup: job.setup.name.clone(),
        repetition: job.repetition,
        model: options.model.clone(),
        passed: scored == Ended::Succeeded,
        tests_touched,
        cost_usd: report.cost_usd,
        turns: report.turns,
        seconds: started.elapsed().as_secs_f64(),
        error: report.error,
    })
}

/// Removes the folder the agent made for a workspace, so runs don't end up in the person's
/// history or skew anything that reads it. Only that one exactly named folder is touched.
fn forget_workspace(projects: &Path, workspace: &Path) {
    let name: String = workspace
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    // A leftover folder isn't worth failing a finished run over.
    let _ = fs::remove_dir_all(projects.join(name));
}

/// Runs everything not yet recorded in `options.out`, calling `progress` after each run, and
/// returns every record including earlier ones.
///
/// # Errors
///
/// Returns an error if records can't be written, or a run can't be set up.
pub fn run(
    tasks: &TaskFile,
    options: &RunOptions,
    progress: &(dyn Fn(&Record) + Sync),
) -> Result<Vec<Record>, EvalError> {
    let earlier = load_records(&options.out)?;
    let done: HashSet<(&str, &str, usize)> = earlier
        .iter()
        .map(|record| {
            (
                record.task.as_str(),
                record.setup.as_str(),
                record.repetition,
            )
        })
        .collect();
    let spent: f64 = earlier.iter().map(|record| record.cost_usd).sum();
    let spent = AtomicU64::new((spent * MICROS_PER_USD) as u64);
    let cap = (options.max_usd * MICROS_PER_USD) as u64;

    // One repetition at a time, so stopping early leaves every task equally covered.
    let mut queue = VecDeque::new();
    for repetition in 0..options.repetitions {
        for task in &tasks.tasks {
            for setup in &options.setups {
                if !done.contains(&(task.id.as_str(), setup.name.as_str(), repetition)) {
                    queue.push_back(Job {
                        task,
                        setup,
                        repetition,
                    });
                }
            }
        }
    }
    fs::create_dir_all(&options.scratch)
        .map_err(|source| EvalError::io("creating the scratch directory", source))?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&options.out)
        .map_err(|source| EvalError::io(format!("opening {}", options.out.display()), source))?;

    let queue = Mutex::new(queue);
    let collected = Mutex::new((file, earlier.clone(), None::<EvalError>));
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..options.jobs.max(1) {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) && spent.load(Ordering::Relaxed) < cap {
                    let Some(job) = queue.lock().expect("lock poisoned").pop_front() else {
                        return;
                    };
                    let result = run_one(&job, tasks, options);
                    let mut collected = collected.lock().expect("lock poisoned");
                    match result {
                        Ok(record) => {
                            spent.fetch_add(
                                (record.cost_usd * MICROS_PER_USD) as u64,
                                Ordering::Relaxed,
                            );
                            let line =
                                serde_json::to_string(&record).expect("records always serialize");
                            if let Err(source) = writeln!(collected.0, "{line}") {
                                collected.2 = Some(EvalError::io("writing a record", source));
                                stop.store(true, Ordering::Relaxed);
                                return;
                            }
                            progress(&record);
                            collected.1.push(record);
                        }
                        Err(err) => {
                            collected.2 = Some(err);
                            stop.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            });
        }
    });
    let (_, records, failure) = collected.into_inner().expect("lock poisoned");
    match failure {
        Some(err) => Err(err),
        None => Ok(records),
    }
}

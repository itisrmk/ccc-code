//! `ccc eval`: test whether a Claude Code setup really helps, on a repository's own history.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, bail};
use ccc_eval::{
    Comparison, Metric, MetricResult, MineOptions, Record, RunOptions, SIGNIFICANCE, Setup,
    TaskFile, Verdict, compare, load_records, mine, run,
};
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub(crate) struct EvalArgs {
    #[command(subcommand)]
    command: EvalCommand,
}

#[derive(Debug, Subcommand)]
enum EvalCommand {
    /// Find commits that make good tasks, and confirm their tests fail without the change and
    /// pass with it. Free: no model runs, though it does run the repository's tests.
    Mine(MineArgs),
    /// Run every task under every setup. Spends Claude quota, up to --max-usd at list prices.
    Run(RunArgs),
    /// Compare the setups in a results file. Free.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
struct MineArgs {
    /// The repository to read. It isn't changed.
    #[arg(value_name = "REPO")]
    repo: PathBuf,
    /// The shell command that runs the tests. `{tests}` becomes each task's test files.
    #[arg(long, value_name = "COMMAND")]
    test: String,
    /// Where to write the tasks.
    #[arg(long, value_name = "FILE")]
    out: PathBuf,
    /// Stop after this many tasks.
    #[arg(long, default_value_t = 12)]
    max_tasks: usize,
    /// How far back to look, in commits.
    #[arg(long, default_value_t = 500)]
    scan: usize,
    /// The largest source change to use, in lines added and removed.
    #[arg(long, default_value_t = 80)]
    max_lines: u64,
    /// The most source files a task may touch.
    #[arg(long, default_value_t = 3)]
    max_files: usize,
    /// How long one run of the tests may take, in seconds.
    #[arg(long, default_value_t = 120)]
    test_timeout: u64,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// The tasks from `ccc eval mine`.
    #[arg(value_name = "TASKS")]
    tasks: PathBuf,
    /// A setup to test, as NAME or NAME=DIR, where DIR holds files to place over the repository,
    /// such as a CLAUDE.md. The first is the baseline. Give at least two.
    #[arg(long = "setup", value_name = "NAME[=DIR]", required = true)]
    setups: Vec<String>,
    /// Where results go. Runs already recorded there are skipped, so a stopped run can continue.
    #[arg(long, value_name = "FILE")]
    out: PathBuf,
    /// Start no new run once this much has been spent at list prices.
    #[arg(long, value_name = "USD")]
    max_usd: f64,
    /// Runs of each task under each setup.
    #[arg(long, default_value_t = 2)]
    reps: usize,
    /// The model to run on.
    #[arg(long, default_value = "claude-haiku-5-5")]
    model: String,
    /// Runs at a time.
    #[arg(long, default_value_t = 4)]
    jobs: usize,
    /// What one run may spend at list prices.
    #[arg(long, value_name = "USD", default_value_t = 1.0)]
    run_max_usd: f64,
    /// How long one run may take, in seconds.
    #[arg(long, default_value_t = 420)]
    run_timeout: u64,
    /// How long scoring may take, in seconds.
    #[arg(long, default_value_t = 120)]
    test_timeout: u64,
    /// The agent program.
    #[arg(long, default_value = "claude", value_name = "PROGRAM")]
    agent: PathBuf,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// Results from `ccc eval run`.
    #[arg(value_name = "FILE")]
    results: PathBuf,
    /// The setup to compare the others to. Defaults to the first one in the file.
    #[arg(long, value_name = "NAME")]
    baseline: Option<String>,
}

/// A temporary directory, removed when a command ends however it ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!("ccc-eval-{}", std::process::id()));
        fs::create_dir_all(&dir).context("creating a temporary directory")?;
        // Canonical, since macOS's temporary path goes through a symlink.
        Ok(Self(
            fs::canonicalize(&dir).context("finding the temporary directory")?,
        ))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn command(args: &EvalArgs) -> anyhow::Result<()> {
    match &args.command {
        EvalCommand::Mine(args) => mine_command(args),
        EvalCommand::Run(args) => run_command(args),
        EvalCommand::Report(args) => report_command(args),
    }
}

fn mine_command(args: &MineArgs) -> anyhow::Result<()> {
    let scratch = Scratch::new()?;
    let (tasks, summary) = mine(&MineOptions {
        repo: args.repo.clone(),
        test_command: args.test.clone(),
        max_tasks: args.max_tasks,
        scan_commits: args.scan,
        max_source_lines: args.max_lines,
        max_source_files: args.max_files,
        test_timeout: Duration::from_secs(args.test_timeout),
        scratch: scratch.0.clone(),
    })
    .context("mining tasks")?;
    let text = serde_json::to_string_pretty(&tasks).context("rendering tasks")?;
    fs::write(&args.out, text).with_context(|| format!("writing {}", args.out.display()))?;
    let mut out = std::io::stdout();
    writeln!(
        out,
        "{} tasks from {} commits looked at. Passed over: {} without both tests and a small \
         source change, {} whose tests pass anyway, {} whose tests fail even with the real change.",
        tasks.tasks.len(),
        summary.scanned,
        summary.wrong_shape,
        summary.tests_already_pass,
        summary.real_change_fails,
    )?;
    for task in &tasks.tasks {
        let subject = task.message.lines().next().unwrap_or_default();
        writeln!(
            out,
            "  {}  {:>3} lines  {subject}",
            task.id, task.source_lines
        )?;
    }
    if tasks.tasks.len() < MIN_USEFUL_TASKS {
        writeln!(
            out,
            "Fewer than {MIN_USEFUL_TASKS} tasks can't show a difference at the level the report \
             uses, however clear it is."
        )?;
    }
    Ok(())
}

/// Below this many tasks, the smallest p-value possible is above [`SIGNIFICANCE`].
const MIN_USEFUL_TASKS: usize = 7;

fn parse_setup(text: &str) -> anyhow::Result<Setup> {
    let (name, overlay) = match text.split_once('=') {
        Some((name, dir)) => (name, Some(PathBuf::from(dir))),
        None => (text, None),
    };
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("setup name {name:?} should be letters, digits, - or _");
    }
    if let Some(dir) = &overlay
        && !dir.is_dir()
    {
        bail!("setup {name}: {} isn't a directory", dir.display());
    }
    Ok(Setup {
        name: name.to_owned(),
        overlay,
    })
}

fn run_command(args: &RunArgs) -> anyhow::Result<()> {
    let setups = args
        .setups
        .iter()
        .map(|text| parse_setup(text))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let names: BTreeSet<&str> = setups.iter().map(|setup| setup.name.as_str()).collect();
    if setups.len() < 2 || names.len() != setups.len() {
        bail!("give at least two setups with different names");
    }
    let tasks = TaskFile::load(&args.tasks)?;
    let scratch = Scratch::new()?;
    let total = tasks.tasks.len() * setups.len() * args.reps;
    writeln!(
        std::io::stdout(),
        "{} tasks x {} setups x {} repetitions = {total} runs on {}, stopping at ${:.2}",
        tasks.tasks.len(),
        setups.len(),
        args.reps,
        args.model,
        args.max_usd,
    )?;
    let options = RunOptions {
        setups,
        repetitions: args.reps,
        model: args.model.clone(),
        agent: args.agent.clone(),
        jobs: args.jobs,
        max_usd: args.max_usd,
        run_max_usd: args.run_max_usd,
        run_timeout: Duration::from_secs(args.run_timeout),
        test_timeout: Duration::from_secs(args.test_timeout),
        out: args.out.clone(),
        scratch: scratch.0.clone(),
        agent_projects: std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".claude").join("projects")),
    };
    let records = run(&tasks, &options, &|record| {
        // Progress is a courtesy, so a closed pipe shouldn't stop the run.
        let _ = writeln!(
            std::io::stdout(),
            "{:<12} {:<14} rep {}  {:<4}  {:>3} turns  ${:.4}{}",
            record.task,
            record.setup,
            record.repetition,
            if record.passed { "pass" } else { "FAIL" },
            record.turns,
            record.cost_usd,
            record
                .error
                .as_deref()
                .map_or_else(String::new, |error| format!("  ({error})")),
        );
    })
    .context("running tasks")?;
    if records.len() < total {
        writeln!(
            std::io::stdout(),
            "Stopped at the spending cap with {} of {total} runs done. Run again with a higher \
             --max-usd to continue.",
            records.len()
        )?;
    }
    // Records land in the order runs finish, so the file can't say which setup came first.
    let baseline = args
        .setups
        .first()
        .and_then(|text| text.split('=').next())
        .unwrap_or_default();
    print_report(&mut std::io::stdout(), &records, Some(baseline))
}

fn report_command(args: &ReportArgs) -> anyhow::Result<()> {
    let records = load_records(&args.results)?;
    print_report(&mut std::io::stdout(), &records, args.baseline.as_deref())
}

fn setups_in_order(records: &[Record]) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for record in records {
        if !names.contains(&record.setup.as_str()) {
            names.push(&record.setup);
        }
    }
    names
}

fn print_report(
    out: &mut impl Write,
    records: &[Record],
    baseline: Option<&str>,
) -> anyhow::Result<()> {
    let names = setups_in_order(records);
    let Some(baseline) = baseline.or_else(|| names.first().copied()) else {
        writeln!(out, "No results yet.")?;
        return Ok(());
    };
    if !names.contains(&baseline) {
        bail!("no results for a setup called {baseline:?}");
    }
    let spent: f64 = records.iter().map(|record| record.cost_usd).sum();
    writeln!(out, "\n{} runs, ${spent:.2} at list prices", records.len())?;
    for other in names.iter().filter(|name| **name != baseline) {
        print_comparison(out, &compare(records, baseline, other))?;
    }
    writeln!(
        out,
        "\nEach task counts once, however many times it ran. A difference counts when p is below \
         {SIGNIFICANCE:.3}. \"No difference detected\" isn't proof there is none, and tasks from \
         one repository may not represent your work."
    )?;
    Ok(())
}

fn format_value(metric: Metric, value: f64) -> String {
    match metric {
        Metric::PassRate => format!("{:.1}%", value * 100.0),
        Metric::Cost => format!("${value:.4}"),
        Metric::Turns => format!("{value:.1}"),
    }
}

fn format_change(metric: Metric, value: f64) -> String {
    match metric {
        Metric::PassRate => format!("{:+.1} pts", value * 100.0),
        Metric::Cost if value < 0.0 => format!("-${:.4}", -value),
        Metric::Cost => format!("+${value:.4}"),
        Metric::Turns => format!("{value:+.1}"),
    }
}

fn verdict_text(result: &MetricResult) -> &'static str {
    match result.verdict {
        Verdict::Better => "BETTER",
        Verdict::Worse => "WORSE",
        Verdict::NotDetected => "no difference detected",
        Verdict::TooFewDiffering => "can't tell: too few tasks differ",
    }
}

/// A run that gives up early is cheap, so lower cost alongside fewer passes isn't a saving.
fn cheaper_by_failing(comparison: &Comparison) -> Option<String> {
    let of = |wanted| {
        comparison
            .metrics
            .iter()
            .find(|result| result.metric == wanted)
    };
    let (passes, cost) = (of(Metric::PassRate)?, of(Metric::Cost)?);
    (cost.verdict == Verdict::Better && passes.other_mean < passes.baseline_mean).then(|| {
        format!(
            "Cheaper, but it also passed less often ({} against {}). Runs that give up early cost \
             less, so don't read this as a saving.",
            format_value(Metric::PassRate, passes.other_mean),
            format_value(Metric::PassRate, passes.baseline_mean),
        )
    })
}

fn print_comparison(out: &mut impl Write, comparison: &Comparison) -> anyhow::Result<()> {
    writeln!(
        out,
        "\n{} against {}: {} tasks, {} paired runs",
        comparison.other, comparison.baseline, comparison.tasks, comparison.pairs
    )?;
    writeln!(
        out,
        "  {:<10}{:>12}{:>12}{:>11}{:>22}{:>8}  verdict",
        "", comparison.baseline, comparison.other, "change", "95% interval", "p"
    )?;
    for result in &comparison.metrics {
        let (low, high) = result.test.interval;
        writeln!(
            out,
            "  {:<10}{:>12}{:>12}{:>11}{:>22}{:>8.4}  {}",
            result.metric.label(),
            format_value(result.metric, result.baseline_mean),
            format_value(result.metric, result.other_mean),
            format_change(result.metric, result.test.mean),
            format!(
                "{} to {}",
                format_change(result.metric, low),
                format_change(result.metric, high)
            ),
            result.test.p_value,
            verdict_text(result),
        )?;
    }
    if let Some(warning) = cheaper_by_failing(comparison) {
        writeln!(out, "  {warning}")?;
    }
    if comparison.tests_touched > 0 || comparison.errors > 0 {
        writeln!(
            out,
            "  {} runs changed a test file, which doesn't help them, and {} stopped on an error.",
            comparison.tests_touched, comparison.errors
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(task: &str, setup: &str, passed: bool) -> Record {
        Record {
            task: task.to_owned(),
            setup: setup.to_owned(),
            repetition: 0,
            model: "m".to_owned(),
            passed,
            tests_touched: false,
            cost_usd: 0.25,
            turns: 4,
            seconds: 1.0,
            error: None,
        }
    }

    #[test]
    fn setups_need_a_plain_name_and_a_real_directory() {
        assert_eq!(parse_setup("plain").unwrap().overlay, None);
        assert!(parse_setup("a=/definitely/missing").is_err());
        assert!(parse_setup("../x").is_err());
        assert!(parse_setup("").is_err());
        let dir = std::env::temp_dir().join(format!("ccc-eval-setup-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let text = format!("with-rules={}", dir.display());
        assert_eq!(
            parse_setup(&text).unwrap().overlay.as_deref(),
            Some(dir.as_path())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_report_names_a_clear_loss() {
        let mut records = Vec::new();
        for index in 0..8 {
            records.push(record(&format!("t{index}"), "plain", true));
            records.push(record(&format!("t{index}"), "harmful", false));
        }
        let mut printed = Vec::new();
        print_report(&mut printed, &records, None).unwrap();
        let printed = String::from_utf8(printed).unwrap();
        assert!(
            printed.contains("harmful against plain: 8 tasks"),
            "{printed}"
        );
        assert!(printed.contains("100.0%"), "{printed}");
        assert!(printed.contains("-100.0 pts"), "{printed}");
        assert!(printed.contains("WORSE"), "{printed}");
    }

    #[test]
    fn a_setup_that_is_cheap_because_it_fails_is_not_called_a_saving() {
        let mut records = Vec::new();
        for index in 0..8 {
            let task = format!("t{index}");
            records.push(record(&task, "plain", true));
            let mut gives_up = record(&task, "gives-up", index != 0);
            gives_up.cost_usd = 0.10;
            records.push(gives_up);
        }
        let mut printed = Vec::new();
        print_report(&mut printed, &records, None).unwrap();
        let printed = String::from_utf8(printed).unwrap();
        assert!(printed.contains("BETTER"), "{printed}");
        assert!(printed.contains("don't read this as a saving"), "{printed}");

        let mut cheaper = Vec::new();
        for index in 0..8 {
            let task = format!("t{index}");
            records_pair(&mut cheaper, &task);
        }
        let mut printed = Vec::new();
        print_report(&mut printed, &cheaper, None).unwrap();
        let printed = String::from_utf8(printed).unwrap();
        assert!(printed.contains("BETTER"), "{printed}");
        assert!(!printed.contains("don't read this"), "{printed}");
    }

    fn records_pair(records: &mut Vec<Record>, task: &str) {
        records.push(record(task, "plain", true));
        let mut lean = record(task, "lean", true);
        lean.cost_usd = 0.10;
        records.push(lean);
    }

    #[test]
    fn an_unknown_baseline_is_an_error_and_no_results_are_stated() {
        let records = vec![record("t", "plain", true)];
        assert!(print_report(&mut Vec::new(), &records, Some("other")).is_err());
        let mut printed = Vec::new();
        print_report(&mut printed, &[], None).unwrap();
        assert!(String::from_utf8(printed).unwrap().contains("No results"));
    }
}

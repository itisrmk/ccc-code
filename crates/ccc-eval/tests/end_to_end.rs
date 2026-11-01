//! Mining a real git repository, then running a scripted stand-in for the agent through the
//! whole loop. No model is involved, so this checks the machinery and costs nothing.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use ccc_eval::{
    MineOptions, RunOptions, Setup, TaskFile, Verdict, Workspace, compare, load_records, mine, run,
};

struct Fixture {
    base: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn git(repo: &Path, args: &[&str]) {
    let done = Command::new("git")
        .current_dir(repo)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(done.status.success(), "{done:?}");
}

fn commit(repo: &Path, files: &[(&str, &str)], message: &str) {
    for (path, content) in files {
        let target = repo.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, content).unwrap();
    }
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-qm", message]);
}

const TEST_SCRIPT: &str = "for f in tests/*.want; do n=$(basename $f .want); \
    [ \"$(cat lib/$n.val 2>/dev/null)\" = \"$(cat $f)\" ] || exit 1; done";

/// A repository holding good tasks and each kind of commit mining must pass over.
fn fixture(name: &str, good_tasks: usize) -> (Fixture, PathBuf) {
    let base = std::env::temp_dir().join(format!("ccc-eval-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    // Canonical as `ccc eval` makes it, since macOS's temporary path goes through a symlink and
    // the agent's folders are named after the real path.
    let base = fs::canonicalize(&base).unwrap();
    let repo = base.join("repo");
    fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q", "."]);
    commit(
        &repo,
        &[("tests/run.sh", TEST_SCRIPT), ("README.md", "hello")],
        "Start",
    );
    commit(&repo, &[("README.md", "more")], "Docs only");
    commit(&repo, &[("lib/untested.val", "1")], "Source only");
    commit(
        &repo,
        &[("lib/early.val", "1"), ("tests/early.want", "1")],
        "Both in one, which still fails first",
    );
    for index in 0..good_tasks {
        commit(
            &repo,
            &[
                (&format!("lib/v{index}.val"), "new"),
                (&format!("tests/v{index}.want"), "new"),
            ],
            &format!("Add value {index}"),
        );
    }
    // Tests that pass with or without the source change prove nothing.
    commit(
        &repo,
        &[
            ("lib/v0.val", "new"),
            ("lib/other.val", "x"),
            ("tests/note.sh", "#"),
        ],
        "Unrelated test change",
    );
    // A change whose own tests still fail isn't a usable answer.
    commit(
        &repo,
        &[("lib/broken.val", "no"), ("tests/broken.want", "yes")],
        "Broken at this commit",
    );
    commit(&repo, &[("lib/broken.val", "yes")], "Fix it, no tests");
    (Fixture { base }, repo)
}

fn mine_options(repo: &Path, base: &Path) -> MineOptions {
    MineOptions {
        repo: repo.to_owned(),
        test_command: "sh tests/run.sh".to_owned(),
        max_tasks: 20,
        scan_commits: 100,
        max_source_lines: 60,
        max_source_files: 3,
        test_timeout: Duration::from_secs(20),
        scratch: base.join("scratch"),
    }
}

/// Stands in for Claude Code: does the work, or not, according to files in the workspace, and
/// reports the way `claude -p --output-format json` does.
fn stand_in_agent(base: &Path) -> PathBuf {
    let projects = base.join("projects");
    let script = format!(
        "#!/bin/sh\nmkdir -p \"{}/$(pwd | sed 's/[^A-Za-z0-9]/-/g')\"\n",
        projects.display()
    ) + r#"
[ "$(git rev-list --count --all)" = "1" ] || { echo '{"is_error":true,"subtype":"saw_history"}'; exit 1; }
if [ -f LAZY ]; then
  echo '{"is_error":false,"total_cost_usd":0.01,"num_turns":1}'
elif [ -f CHEAT ]; then
  for f in tests/*.want; do n=$(basename $f .want); cat lib/$n.val > $f 2>/dev/null || rm $f; done
  echo '{"is_error":false,"total_cost_usd":0.01,"num_turns":1}'
elif [ -f HANG ]; then
  sleep 30
else
  for f in tests/*.want; do n=$(basename $f .want); cp $f lib/$n.val; done
  echo '{"is_error":false,"total_cost_usd":0.05,"num_turns":4}'
fi
"#;
    let path = base.join("agent.sh");
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn overlay(base: &Path, name: &str, marker: Option<&str>) -> Setup {
    let dir = base.join("overlays").join(name);
    fs::create_dir_all(&dir).unwrap();
    if let Some(marker) = marker {
        fs::write(dir.join(marker), "").unwrap();
    }
    Setup {
        name: name.to_owned(),
        overlay: Some(dir),
    }
}

fn run_options(base: &Path, setups: Vec<Setup>, name: &str) -> RunOptions {
    RunOptions {
        setups,
        repetitions: 1,
        model: "stand-in".to_owned(),
        agent: stand_in_agent(base),
        jobs: 4,
        max_usd: 100.0,
        run_max_usd: 1.0,
        run_timeout: Duration::from_secs(30),
        test_timeout: Duration::from_secs(20),
        out: base.join(format!("{name}.jsonl")),
        scratch: base.join("scratch"),
        agent_projects: Some(base.join("projects")),
    }
}

#[test]
fn mining_keeps_only_commits_whose_tests_depend_on_their_change() {
    let (fixture, repo) = fixture("mine", 3);
    let (tasks, summary) = mine(&mine_options(&repo, &fixture.base)).unwrap();

    let messages: Vec<&str> = tasks.tasks.iter().map(|t| t.message.as_str()).collect();
    assert!(messages.contains(&"Add value 0"), "{messages:?}");
    assert!(messages.contains(&"Add value 2"), "{messages:?}");
    assert!(!messages.iter().any(|m| m.contains("Docs only")));
    assert!(!messages.iter().any(|m| m.contains("Source only")));
    assert!(!messages.iter().any(|m| m.contains("Unrelated")));
    assert!(!messages.iter().any(|m| m.contains("Broken")));
    assert!(summary.tests_already_pass >= 1, "{summary:?}");
    assert!(summary.real_change_fails >= 1, "{summary:?}");
    assert!(summary.wrong_shape >= 2, "{summary:?}");
}

#[test]
fn a_workspace_has_the_tests_but_not_the_answer_or_any_way_to_find_it() {
    let (fixture, repo) = fixture("workspace", 1);
    let (tasks, _) = mine(&mine_options(&repo, &fixture.base)).unwrap();
    let task = tasks
        .tasks
        .iter()
        .find(|t| t.message == "Add value 0")
        .unwrap();

    let dir = fixture.base.join("one");
    let workspace = Workspace::create(dir.clone(), &tasks.repo, task, None).unwrap();

    assert!(dir.join("tests/v0.want").exists());
    assert!(!dir.join("lib/v0.val").exists());
    let history = Command::new("git")
        .current_dir(workspace.path())
        .args(["rev-list", "--count", "--all"])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&history.stdout).trim(), "1");
}

#[test]
fn a_good_setup_passes_a_bad_one_is_caught_and_cheating_does_not_count() {
    let (fixture, repo) = fixture("run", 8);
    let base = &fixture.base;
    let (tasks, _) = mine(&mine_options(&repo, base)).unwrap();
    assert!(tasks.tasks.len() >= 8, "{}", tasks.tasks.len());
    let setups = vec![
        overlay(base, "plain", None),
        overlay(base, "lazy", Some("LAZY")),
        overlay(base, "cheat", Some("CHEAT")),
    ];
    let records = run(&tasks, &run_options(base, setups, "good"), &|_| {}).unwrap();

    assert_eq!(records.len(), tasks.tasks.len() * 3);
    let of = |setup: &'static str| records.iter().filter(move |r| r.setup == setup);
    assert!(
        of("plain").all(|r| r.passed && r.error.is_none()),
        "{records:?}"
    );
    assert!(
        of("lazy").all(|r| !r.passed && !r.tests_touched),
        "{records:?}"
    );
    assert!(
        of("cheat").all(|r| !r.passed && r.tests_touched),
        "{records:?}"
    );

    let lazy = compare(&records, "plain", "lazy");
    assert_eq!(lazy.metrics[0].verdict, Verdict::Worse);
    let cheat = compare(&records, "plain", "cheat");
    assert_eq!(cheat.metrics[0].verdict, Verdict::Worse);
    assert_eq!(cheat.tests_touched, tasks.tasks.len());

    let projects = base.join("projects");
    assert!(projects.is_dir(), "the stand-in should have made folders");
    assert_eq!(
        fs::read_dir(&projects).unwrap().count(),
        0,
        "the agent's folders for throwaway workspaces should be gone"
    );
}

#[test]
fn stopping_early_at_the_spending_cap_and_continuing_later_loses_nothing() {
    let (fixture, repo) = fixture("cap", 8);
    let base = &fixture.base;
    let (tasks, _) = mine(&mine_options(&repo, base)).unwrap();
    let setups = vec![overlay(base, "a", None), overlay(base, "b", None)];
    let mut options = run_options(base, setups, "cap");
    options.jobs = 1;
    options.max_usd = 0.12;

    let first = run(&tasks, &options, &|_| {}).unwrap();
    assert_eq!(first.len(), 3, "stops once $0.12 is spent at $0.05 a run");

    options.max_usd = 100.0;
    let seen = std::sync::Mutex::new(0);
    let all = run(&tasks, &options, &|_| *seen.lock().unwrap() += 1).unwrap();
    assert_eq!(all.len(), tasks.tasks.len() * 2);
    assert_eq!(
        *seen.lock().unwrap(),
        all.len() - 3,
        "earlier runs aren't repeated"
    );
    assert_eq!(load_records(&options.out).unwrap().len(), all.len());
}

#[test]
fn a_hung_agent_is_stopped_and_recorded() {
    let (fixture, repo) = fixture("hang", 1);
    let base = &fixture.base;
    let (mut tasks, _): (TaskFile, _) = mine(&mine_options(&repo, base)).unwrap();
    tasks.tasks.truncate(1);
    let mut options = run_options(base, vec![overlay(base, "hang", Some("HANG"))], "hang");
    options.run_timeout = Duration::from_millis(500);

    let records = run(&tasks, &options, &|_| {}).unwrap();

    assert_eq!(records.len(), 1);
    assert!(!records[0].passed);
    assert_eq!(records[0].error.as_deref(), Some("timed out"));
}

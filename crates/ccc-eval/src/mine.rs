//! Finds commits that make good tasks: a small source change with tests that really depend on it.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::EvalError;
use crate::process::Ended;
use crate::workspace::{Workspace, git};

/// A commit turned into a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// A short name for reports.
    pub id: String,
    /// The commit whose change is the task.
    pub commit: String,
    /// The commit's message, which becomes the request.
    pub message: String,
    /// Test files the commit changed. These are in place when a run starts.
    pub test_paths: Vec<String>,
    /// Source files the commit changed.
    pub source_paths: Vec<String>,
    /// Every file the commit changed, so the real answer can be checked.
    pub all_paths: Vec<String>,
    /// Source lines added and removed.
    pub source_lines: u64,
}

/// A set of tasks and how to score them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskFile {
    /// The repository the commits belong to.
    pub repo: PathBuf,
    /// The shell command that runs the tests. `{tests}` becomes the task's test files.
    pub test_command: String,
    /// The tasks.
    pub tasks: Vec<Task>,
}

impl TaskFile {
    /// Reads a task file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be read or isn't a task file.
    pub fn load(path: &Path) -> Result<Self, EvalError> {
        let text = fs::read_to_string(path)
            .map_err(|source| EvalError::io(format!("reading {}", path.display()), source))?;
        serde_json::from_str(&text).map_err(|source| EvalError::Format {
            path: path.to_owned(),
            source,
        })
    }
}

/// What to look for.
#[derive(Debug, Clone)]
pub struct MineOptions {
    /// The repository to read.
    pub repo: PathBuf,
    /// The shell command that runs the tests. `{tests}` becomes the task's test files.
    pub test_command: String,
    /// Stop after this many tasks.
    pub max_tasks: usize,
    /// How far back to look, in commits.
    pub scan_commits: usize,
    /// The largest source change to use, in lines added and removed.
    pub max_source_lines: u64,
    /// The most source files a task may touch.
    pub max_source_files: usize,
    /// How long one run of the tests may take.
    pub test_timeout: Duration,
    /// A directory for temporary copies.
    pub scratch: PathBuf,
}

/// What mining found and why it passed over the rest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MineSummary {
    /// Commits looked at.
    pub scanned: usize,
    /// Commits without both tests and source, or too large.
    pub wrong_shape: usize,
    /// Commits whose new tests already pass without the source change.
    pub tests_already_pass: usize,
    /// Commits whose tests don't pass even with the real change.
    pub real_change_fails: usize,
}

fn is_test_path(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let folder = path.split('/').rev().skip(1).any(|part| {
        matches!(
            part,
            "test" | "tests" | "spec" | "specs" | "__tests__" | "testing"
        )
    });
    let stem = name.split('.').next().unwrap_or(name);
    folder
        || stem == "test"
        || stem == "conftest"
        || stem.starts_with("test_")
        || stem.ends_with("_test")
        || stem.ends_with("_spec")
        || name.contains(".test.")
        || name.contains(".spec.")
}

fn is_doc_path(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    let doc_folder = path
        .split('/')
        .rev()
        .skip(1)
        .any(|part| part == "doc" || part == "docs");
    let extension = name.rsplit_once('.').map(|(_, extension)| extension);
    doc_folder
        || matches!(extension, Some("md" | "rst" | "txt" | "adoc"))
        || [
            "license",
            "changelog",
            "changes",
            "authors",
            "news",
            "history",
        ]
        .iter()
        .any(|start| name.starts_with(start))
}

/// A commit's changed files, or `None` if it holds anything that can't be a text change.
fn changed_files(numstat: &str) -> Option<Vec<(String, u64)>> {
    numstat
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut fields = line.splitn(3, '\t');
            let added = fields.next()?.parse::<u64>().ok()?;
            let removed = fields.next()?.parse::<u64>().ok()?;
            Some((fields.next()?.to_owned(), added + removed))
        })
        .collect()
}

fn candidate(repo: &Path, commit: &str, options: &MineOptions) -> Result<Option<Task>, EvalError> {
    let numstat = git(
        repo,
        &["show", "--no-renames", "--numstat", "--format=", commit],
    )?;
    let Some(files) = changed_files(&String::from_utf8_lossy(&numstat)) else {
        return Ok(None);
    };
    let test_paths: Vec<String> = files
        .iter()
        .filter(|(path, _)| is_test_path(path))
        .map(|(path, _)| path.clone())
        .collect();
    let sources: Vec<&(String, u64)> = files
        .iter()
        .filter(|(path, _)| !is_test_path(path) && !is_doc_path(path))
        .collect();
    let source_lines: u64 = sources.iter().map(|(_, lines)| lines).sum();
    if test_paths.is_empty()
        || sources.is_empty()
        || sources.len() > options.max_source_files
        || source_lines > options.max_source_lines
    {
        return Ok(None);
    }
    let message = git(repo, &["show", "-s", "--format=%B", commit])?;
    Ok(Some(Task {
        id: commit.chars().take(10).collect(),
        commit: commit.to_owned(),
        message: String::from_utf8_lossy(&message).trim().to_owned(),
        test_paths,
        source_paths: sources.iter().map(|(path, _)| path.clone()).collect(),
        all_paths: files.into_iter().map(|(path, _)| path).collect(),
        source_lines,
    }))
}

/// Confirms the tests fail without the change and pass with it, which is what makes a pass mean
/// something later.
fn confirm(task: &Task, options: &MineOptions, dir: PathBuf) -> Result<Confirmed, EvalError> {
    let workspace = Workspace::create(dir, &options.repo, task, None)?;
    let before = workspace.run_tests(task, &options.test_command, options.test_timeout)?;
    let result = if before == Ended::Succeeded {
        Confirmed::AlreadyPasses
    } else {
        workspace.restore(&options.repo, &task.commit, &task.all_paths)?;
        match workspace.run_tests(task, &options.test_command, options.test_timeout)? {
            Ended::Succeeded => Confirmed::Good,
            _ => Confirmed::RealChangeFails,
        }
    };
    workspace.remove();
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Confirmed {
    Good,
    AlreadyPasses,
    RealChangeFails,
}

/// Looks back through history for tasks, newest first.
///
/// # Errors
///
/// Returns an error if the repository can't be read or a temporary copy can't be made.
pub fn mine(options: &MineOptions) -> Result<(TaskFile, MineSummary), EvalError> {
    let scan = options.scan_commits.to_string();
    let listed = git(
        &options.repo,
        &["log", "--no-merges", "--format=%H", "-n", &scan, "HEAD"],
    )?;
    let listed = String::from_utf8_lossy(&listed).into_owned();
    let mut tasks = Vec::new();
    let mut summary = MineSummary::default();
    for commit in listed.lines() {
        if tasks.len() >= options.max_tasks {
            break;
        }
        summary.scanned += 1;
        // A root commit has no parent to start from.
        if git(
            &options.repo,
            &["rev-parse", "--verify", &format!("{commit}^")],
        )
        .is_err()
        {
            summary.wrong_shape += 1;
            continue;
        }
        let Some(task) = candidate(&options.repo, commit, options)? else {
            summary.wrong_shape += 1;
            continue;
        };
        let dir = options.scratch.join(format!("mine-{}", task.id));
        match confirm(&task, options, dir)? {
            Confirmed::Good => tasks.push(task),
            Confirmed::AlreadyPasses => summary.tests_already_pass += 1,
            Confirmed::RealChangeFails => summary.real_change_fails += 1,
        }
    }
    let repo = fs::canonicalize(&options.repo)
        .map_err(|source| EvalError::io(format!("finding {}", options.repo.display()), source))?;
    Ok((
        TaskFile {
            repo,
            test_command: options.test_command.clone(),
            tasks,
        },
        summary,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tells_tests_from_source_and_docs() {
        for path in [
            "tests/foo.py",
            "src/foo_test.go",
            "test_foo.py",
            "a/spec/b.rb",
            "web/foo.test.ts",
            "conftest.py",
        ] {
            assert!(is_test_path(path), "{path}");
        }
        for path in ["src/foo.py", "contest.py", "attest/x.py", "latest.rs"] {
            assert!(!is_test_path(path), "{path}");
        }
        for path in ["README.md", "docs/api.py", "CHANGELOG", "notes.txt"] {
            assert!(is_doc_path(path), "{path}");
        }
        assert!(!is_doc_path("src/foo.py"));
    }

    #[test]
    fn reads_numstat_and_refuses_binary_changes() {
        assert_eq!(
            changed_files("1\t2\ta.py\n3\t0\tb c.py\n"),
            Some(vec![("a.py".to_owned(), 3), ("b c.py".to_owned(), 3)])
        );
        assert_eq!(changed_files("-\t-\timage.png\n"), None);
    }
}

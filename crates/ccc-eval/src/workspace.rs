//! The directory a run works in: the repository as it was just before a commit, with that
//! commit's tests in place and no way to look ahead.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::EvalError;
use crate::mine::Task;
use crate::process::{Ended, run_limited};

/// In a test command, replaced with the task's test files.
pub(crate) const TESTS_MARKER: &str = "{tests}";

/// A throwaway copy of a task's starting point.
#[derive(Debug)]
pub struct Workspace {
    dir: PathBuf,
}

pub(crate) fn git(repo: &Path, args: &[&str]) -> Result<Vec<u8>, EvalError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|source| EvalError::io("running git", source))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(EvalError::Git {
            command: args.join(" "),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

impl Workspace {
    /// Builds a workspace in `dir`, which must not exist yet: the source just before `task`'s
    /// commit plus that commit's tests, then `overlay` on top. It gets a new git history of one
    /// commit, because a clone would let a run read the answer out of later history.
    ///
    /// # Errors
    ///
    /// Returns an error if the repository can't be read or the files can't be written.
    pub fn create(
        dir: PathBuf,
        repo: &Path,
        task: &Task,
        overlay: Option<&Path>,
    ) -> Result<Self, EvalError> {
        fs::create_dir_all(&dir)
            .map_err(|source| EvalError::io(format!("creating {}", dir.display()), source))?;
        let workspace = Self { dir };
        workspace.unpack(repo, &format!("{}^", task.commit))?;
        workspace.restore(repo, &task.commit, &task.test_paths)?;
        if let Some(overlay) = overlay {
            copy_tree(overlay, &workspace.dir)?;
        }
        workspace.snapshot()?;
        Ok(workspace)
    }

    fn unpack(&self, repo: &Path, revision: &str) -> Result<(), EvalError> {
        let mut archive = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["archive", revision])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|source| EvalError::io("running git archive", source))?;
        let piped = archive.stdout.take().ok_or_else(|| EvalError::Git {
            command: "archive".to_owned(),
            detail: "no output".to_owned(),
        })?;
        let untar = Command::new("tar")
            .args(["-xf", "-", "-C"])
            .arg(&self.dir)
            .stdin(piped)
            .status()
            .map_err(|source| EvalError::io("running tar", source))?;
        let archived = archive
            .wait()
            .map_err(|source| EvalError::io("waiting for git archive", source))?;
        if untar.success() && archived.success() {
            Ok(())
        } else {
            Err(EvalError::Git {
                command: format!("archive {revision}"),
                detail: "couldn't unpack the source".to_owned(),
            })
        }
    }

    /// Makes `paths` match `commit`, removing any the commit deleted.
    pub(crate) fn restore(
        &self,
        repo: &Path,
        commit: &str,
        paths: &[String],
    ) -> Result<(), EvalError> {
        for path in paths {
            let target = self.dir.join(path);
            let write_error =
                |source| EvalError::io(format!("writing {}", target.display()), source);
            match git(repo, &["show", &format!("{commit}:{path}")]) {
                Ok(content) => {
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent).map_err(write_error)?;
                    }
                    fs::write(&target, content).map_err(write_error)?;
                }
                Err(EvalError::Git { .. }) => {
                    if target.exists() {
                        fs::remove_file(&target).map_err(write_error)?;
                    }
                }
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<(), EvalError> {
        git(&self.dir, &["init", "-q", "."])?;
        git(&self.dir, &["add", "-A"])?;
        git(
            &self.dir,
            &[
                "-c",
                "user.name=ccc-eval",
                "-c",
                "user.email=ccc-eval@localhost",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "--allow-empty",
                "--no-verify",
                "-m",
                "Starting point",
            ],
        )?;
        Ok(())
    }

    /// The workspace's directory.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Runs the task's tests.
    pub(crate) fn run_tests(
        &self,
        task: &Task,
        test_command: &str,
        limit: Duration,
    ) -> Result<Ended, EvalError> {
        let command = test_command.replace(TESTS_MARKER, &task.test_paths.join(" "));
        run_limited(
            Command::new("sh")
                .args(["-c", &command])
                .current_dir(&self.dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null()),
            limit,
        )
        .map_err(|source| EvalError::io("running the tests", source))
    }

    /// Whether any of `paths` differ from the starting point.
    pub(crate) fn changed(&self, paths: &[String]) -> Result<bool, EvalError> {
        if paths.is_empty() {
            return Ok(false);
        }
        let mut args = vec!["status", "--porcelain", "--"];
        args.extend(paths.iter().map(String::as_str));
        Ok(!git(&self.dir, &args)?.is_empty())
    }

    /// Deletes the workspace.
    pub(crate) fn remove(self) {
        // A leftover temporary directory isn't worth failing a finished run over.
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), EvalError> {
    let read_error = |source| EvalError::io(format!("reading {}", from.display()), source);
    for entry in fs::read_dir(from).map_err(read_error)? {
        let entry = entry.map_err(read_error)?;
        let target = to.join(entry.file_name());
        if entry.file_type().map_err(read_error)?.is_dir() {
            fs::create_dir_all(&target).map_err(|source| {
                EvalError::io(format!("creating {}", target.display()), source)
            })?;
            copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target).map_err(|source| {
                EvalError::io(format!("copying to {}", target.display()), source)
            })?;
        }
    }
    Ok(())
}

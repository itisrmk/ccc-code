//! Running commands with a time limit, including anything they start.

use std::io;
use std::os::unix::process::CommandExt as _;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How a limited command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ended {
    Succeeded,
    Failed,
    TimedOut,
}

/// Runs `command` in its own process group, so that killing it also kills anything it started.
pub(crate) fn run_limited(command: &mut Command, limit: Duration) -> io::Result<Ended> {
    let mut child = command.process_group(0).spawn()?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(if status.success() {
                Ended::Succeeded
            } else {
                Ended::Failed
            });
        }
        if started.elapsed() >= limit {
            let _ = Command::new("kill")
                .args(["-9", &format!("-{}", child.id())])
                .stderr(Stdio::null())
                .status();
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Ended::TimedOut);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[test]
    fn reports_how_a_command_ended() {
        let limit = Duration::from_secs(10);
        assert_eq!(
            run_limited(&mut shell("exit 0"), limit).unwrap(),
            Ended::Succeeded
        );
        assert_eq!(
            run_limited(&mut shell("exit 3"), limit).unwrap(),
            Ended::Failed
        );
    }

    #[test]
    fn stops_a_command_and_what_it_started_at_the_limit() {
        let started = Instant::now();
        let ended = run_limited(
            &mut shell("sleep 30 & sleep 30"),
            Duration::from_millis(200),
        )
        .unwrap();
        assert_eq!(ended, Ended::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}

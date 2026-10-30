//! `ccc run`: start the proxy on a free port and launch a child process against it.

use std::os::unix::process::ExitStatusExt;
use std::process::{ExitCode, ExitStatus};

use anyhow::Context;
use ccc_proxy::Proxy;
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::oneshot;

/// Added to a signal number to form a shell-style exit code.
const SIGNAL_EXIT_BASE: i32 = 128;

/// Runs `command` with `ANTHROPIC_BASE_URL` pointing at `proxy`, returning its exit code.
pub(crate) async fn run(proxy: Proxy, command: &[String]) -> anyhow::Result<ExitCode> {
    let (program, args) = command
        .split_first()
        .context("no command given to run after --")?;

    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .context("binding a local port")?;
    let address = listener.local_addr().context("reading the bound port")?;

    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(proxy.serve(listener, async {
        // The sender is dropped or fires when the child exits; either means stop.
        let _ = stopped.await;
    }));

    let mut child = Command::new(program)
        .args(args)
        .env("ANTHROPIC_BASE_URL", format!("http://{address}"))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting {program}"))?;

    let mut interrupt = signal(SignalKind::interrupt()).context("installing SIGINT handler")?;
    let mut terminate = signal(SignalKind::terminate()).context("installing SIGTERM handler")?;
    let status = loop {
        tokio::select! {
            status = child.wait() => break status.with_context(|| format!("waiting for {program}"))?,
            // The terminal delivers ctrl-c to the child too. It decides whether to exit.
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {
                child.start_kill().with_context(|| format!("stopping {program}"))?;
            }
        }
    };

    let _ = stop.send(());
    match server.await {
        Ok(result) => result.context("running proxy")?,
        Err(err) => tracing::error!(error = %err, "proxy task failed"),
    }

    Ok(exit_code(status))
}

fn exit_code(status: ExitStatus) -> ExitCode {
    let code = status
        .code()
        .or_else(|| status.signal().map(|signal| SIGNAL_EXIT_BASE + signal))
        .unwrap_or(1);
    // Exit codes are one byte on Unix, so anything out of range is reported as failure.
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}

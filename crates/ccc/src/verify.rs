//! `ccc verify`: runs the gateway checks from the command line.

use std::io::Write;
use std::process::{ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail};
use ccc_verify::{FakeUpstream, VerifyOptions};
use clap::Args;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};

/// How long a gateway started with --spawn gets to begin listening.
const GATEWAY_START_TIMEOUT: Duration = Duration::from_secs(30);
const GATEWAY_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// In --spawn, replaced with the stand-in's URL.
const UPSTREAM_MARKER: &str = "{upstream}";

#[derive(Debug, Args)]
pub(crate) struct VerifyArgs {
    /// The gateway's base URL.
    #[arg(long, value_name = "URL")]
    gateway: String,
    /// A shell command that starts the gateway. `{upstream}` is replaced with the URL of the
    /// stand-in for the API, which the gateway must forward to.
    #[arg(long, value_name = "COMMAND")]
    spawn: Option<String>,
    /// The port to run the stand-in on, for a gateway that's already running with this as its
    /// upstream. Needed without --spawn.
    #[arg(long, value_name = "PORT", required_unless_present = "spawn")]
    upstream_port: Option<u16>,
    /// An extra header for the gateway, such as its own key. Can be repeated.
    #[arg(long, value_name = "NAME: VALUE")]
    header: Vec<String>,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

fn parse_headers(headers: &[String]) -> anyhow::Result<Vec<(String, String)>> {
    headers
        .iter()
        .map(|header| {
            let (name, value) = header
                .split_once(':')
                .with_context(|| format!("header {header:?} should look like \"name: value\""))?;
            Ok((name.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

/// The host and port to wait for, from a gateway's URL.
fn authority(url: &str) -> anyhow::Result<String> {
    let rest = url
        .split_once("://")
        .map_or(url, |(_, rest)| rest)
        .split('/')
        .next()
        .unwrap_or_default();
    if rest.is_empty() {
        bail!("no host in gateway URL {url:?}");
    }
    let default_port = if url.starts_with("https://") { 443 } else { 80 };
    Ok(if rest.contains(':') {
        rest.to_owned()
    } else {
        format!("{rest}:{default_port}")
    })
}

async fn wait_for_gateway(address: &str, child: &mut Child) -> anyhow::Result<()> {
    let started = Instant::now();
    loop {
        if TcpStream::connect(address).await.is_ok() {
            return Ok(());
        }
        if let Some(status) = child.try_wait().context("checking on the gateway")? {
            bail!("the gateway exited before listening: {status}");
        }
        if started.elapsed() > GATEWAY_START_TIMEOUT {
            bail!("the gateway didn't start listening on {address}");
        }
        tokio::time::sleep(GATEWAY_POLL_INTERVAL).await;
    }
}

pub(crate) async fn run(args: &VerifyArgs) -> anyhow::Result<ExitCode> {
    let headers = parse_headers(&args.header)?;
    let port = args.upstream_port.unwrap_or(0);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("binding the stand-in to port {port}"))?;
    let upstream = FakeUpstream::start(listener);

    let mut child = None;
    if let Some(command) = &args.spawn {
        let command = command.replace(UPSTREAM_MARKER, &upstream.url());
        let mut spawned = Command::new("sh")
            .args(["-c", &command])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("starting the gateway")?;
        wait_for_gateway(&authority(&args.gateway)?, &mut spawned).await?;
        child = Some(spawned);
    }

    let report = ccc_verify::run(
        &upstream,
        &VerifyOptions {
            gateway: args.gateway.clone(),
            headers,
        },
    )
    .await
    .context("running the checks")?;
    drop(child);

    let mut out = std::io::stdout();
    if args.json {
        writeln!(out, "{}", report.to_json())?;
    } else {
        writeln!(out, "{report}")?;
    }
    Ok(if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_need_a_colon() {
        assert_eq!(
            parse_headers(&["x-key: abc:def".to_owned()]).unwrap(),
            [("x-key".to_owned(), "abc:def".to_owned())]
        );
        assert!(parse_headers(&["no separator".to_owned()]).is_err());
    }

    #[test]
    fn finds_where_to_wait_for_a_gateway() {
        assert_eq!(
            authority("http://127.0.0.1:4000/x").unwrap(),
            "127.0.0.1:4000"
        );
        assert_eq!(authority("https://gw.test/x").unwrap(), "gw.test:443");
        assert_eq!(authority("gw.test:9").unwrap(), "gw.test:9");
        assert!(authority("http://").is_err());
    }
}

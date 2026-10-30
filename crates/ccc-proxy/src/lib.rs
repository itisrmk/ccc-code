//! HTTP proxy that sits between Claude Code and the Anthropic API.
//!
//! Every method and path is forwarded. Messages API requests run through the compiler
//! pipeline, and anything no stage changed goes out byte for byte. Response bodies are
//! streamed chunk by chunk, so server-sent events reach the client as upstream produces them.

mod conversations;
mod dump;
mod forward;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use ccc_pipeline::Pipeline;
use ccc_redact::Vault;
use tokio::net::TcpListener;

use crate::conversations::Conversations;

use crate::dump::Dumper;

/// Failure to construct or run a [`Proxy`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProxyError {
    /// The upstream base URL could not be parsed or has an unsupported scheme.
    #[error("invalid upstream url {url:?}: {reason}")]
    InvalidUpstream {
        /// The value that was rejected.
        url: String,
        /// Why it was rejected.
        reason: String,
    },
    /// The HTTP client could not be built.
    #[error("building http client")]
    Client(#[source] reqwest::Error),
    /// The traffic dump directory could not be prepared.
    #[error("preparing dump directory {path:?}")]
    DumpDir {
        /// The directory that could not be prepared.
        path: PathBuf,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The server stopped with an I/O error.
    #[error("serving http")]
    Serve(#[source] std::io::Error),
}

/// Settings for a [`Proxy`].
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Base URL requests are forwarded to, such as `https://api.anthropic.com`.
    pub upstream: String,
    /// Directory that receives a copy of all traffic, when set.
    ///
    /// Dumps hold real prompts and tool output with auth headers removed.
    pub dump_dir: Option<PathBuf>,
    /// The stages to compile requests with.
    pub pipeline: Pipeline,
    /// Stages new conversations start without. Existing conversations keep the settings
    /// they started with, since changing them would rewrite history the server has seen.
    pub disabled_stages: BTreeSet<String>,
    /// Directory for per-conversation state. `None` keeps it in memory, in which case a
    /// restart in the middle of a conversation can invalidate its thinking blocks.
    pub state_dir: Option<PathBuf>,
    /// Resolves placeholders in responses. Must be the vault the Redact stage records in.
    pub vault: Option<Arc<Vault>>,
    /// Asks the API to reject any request that edits history behind a thinking block, as
    /// it already does for accounts created on or after 2026-08-31. For testing the
    /// compiler from an older account, where such an edit would otherwise go unreported.
    pub enforce_thinking_check: bool,
}

/// Running totals for a proxy, readable while it serves.
#[derive(Debug, Default)]
pub struct Stats {
    requests: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    edits: AtomicU64,
    stage_failures: AtomicU64,
    blocked: AtomicU64,
}

impl Stats {
    fn record(&self, bytes_in: usize, bytes_out: usize) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes_in.fetch_add(bytes_in as u64, Ordering::Relaxed);
        self.bytes_out
            .fetch_add(bytes_out as u64, Ordering::Relaxed);
    }

    /// A one-line summary suitable for a terminal.
    #[must_use]
    pub fn summary(&self) -> String {
        let bytes_in = self.bytes_in.load(Ordering::Relaxed);
        let bytes_out = self.bytes_out.load(Ordering::Relaxed);
        format!(
            "ccc: {} requests compiled, {} edits, {} KB saved of {} KB sent, {} stage failures, {} blocked",
            self.requests.load(Ordering::Relaxed),
            self.edits.load(Ordering::Relaxed),
            bytes_in.saturating_sub(bytes_out) / 1024,
            bytes_in / 1024,
            self.stage_failures.load(Ordering::Relaxed),
            self.blocked.load(Ordering::Relaxed),
        )
    }
}

/// A configured proxy, ready to serve on a listener.
#[derive(Debug, Clone)]
pub struct Proxy {
    state: Arc<forward::State>,
}

impl Proxy {
    /// Validates `config` and builds the upstream HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError`] if the upstream URL is invalid, the HTTP client cannot
    /// be built, or the dump directory cannot be created.
    pub fn new(config: ProxyConfig) -> Result<Self, ProxyError> {
        let upstream = normalize_upstream(&config.upstream)?;
        // No overall timeout: a long model response is a normal, healthy stream.
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(ProxyError::Client)?;
        let dumper = config.dump_dir.map(Dumper::create).transpose()?;
        let enabled_for_new: BTreeSet<String> = config
            .pipeline
            .stage_names()
            .filter(|name| !config.disabled_stages.contains(*name))
            .map(str::to_owned)
            .collect();
        let must_inspect = config.pipeline.requires_inspection(&enabled_for_new);
        let conversations = Conversations::new(
            config.state_dir.map(|dir| dir.join("conversations")),
            enabled_for_new,
        );
        Ok(Self {
            state: Arc::new(forward::State {
                upstream,
                client,
                dumper,
                pipeline: config.pipeline,
                conversations,
                must_inspect,
                vault: config.vault,
                enforce_thinking_check: config.enforce_thinking_check,
                stats: Arc::default(),
            }),
        })
    }

    /// A handle to this proxy's running totals.
    #[must_use]
    pub fn stats(&self) -> Arc<Stats> {
        Arc::clone(&self.state.stats)
    }

    /// Serves requests from `listener` until `shutdown` resolves.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::Serve`] if the server fails.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ProxyError> {
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(ProxyError::Serve)
    }

    fn router(self) -> Router {
        Router::new()
            .fallback(forward::handle)
            .with_state(self.state)
    }
}

fn normalize_upstream(raw: &str) -> Result<String, ProxyError> {
    let invalid = |reason: String| ProxyError::InvalidUpstream {
        url: raw.to_owned(),
        reason,
    };
    let url = reqwest::Url::parse(raw).map_err(|err| invalid(err.to_string()))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid(format!("unsupported scheme {:?}", url.scheme())));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(invalid("must not have a query or fragment".to_owned()));
    }
    Ok(raw.trim_end_matches('/').to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_upstream_strips_trailing_slash_and_keeps_path() {
        assert_eq!(
            normalize_upstream("https://example.test/base/").unwrap(),
            "https://example.test/base"
        );
    }

    #[test]
    fn normalize_upstream_rejects_bad_values() {
        for bad in ["not a url", "ftp://example.test", "http://x.test/?a=1"] {
            assert!(normalize_upstream(bad).is_err(), "accepted {bad:?}");
        }
    }
}

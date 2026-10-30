//! HTTP proxy that sits between Claude Code and the Anthropic API.
//!
//! Every method and path is forwarded. Request and response bodies pass through
//! byte for byte, and response bodies are streamed chunk by chunk so server-sent
//! events reach the client as soon as upstream produces them.

mod dump;
mod forward;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;

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
        Ok(Self {
            state: Arc::new(forward::State {
                upstream,
                client,
                dumper,
            }),
        })
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

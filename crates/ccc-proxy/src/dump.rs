//! Optional capture of traffic to disk for inspection and fixture building.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::Uri;
use axum::http::header::{self, HeaderMap, HeaderName};
use bytes::Bytes;
use ccc_pipeline::{StageOutcome, StageTrace};
use tokio::sync::mpsc;

use crate::ProxyError;

/// Headers that carry credentials and are never written to disk.
const SECRET_HEADERS: [HeaderName; 4] = [
    header::AUTHORIZATION,
    header::COOKIE,
    header::SET_COOKIE,
    HeaderName::from_static("x-api-key"),
];

/// Writes each exchange to files named `<unix-ms>-<seq>.{req.json,head.json,res.body}`.
#[derive(Debug)]
pub(crate) struct Dumper {
    dir: PathBuf,
    sequence: AtomicU64,
}

impl Dumper {
    pub(crate) fn create(dir: PathBuf) -> Result<Self, ProxyError> {
        DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)
            .map_err(|source| ProxyError::DumpDir {
                path: dir.clone(),
                source,
            })?;
        Ok(Self {
            dir,
            sequence: AtomicU64::new(0),
        })
    }

    pub(crate) fn record_request(
        &self,
        method: &str,
        uri: &Uri,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Exchange {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_millis());
        let stem = self.dir.join(format!("{millis}-{sequence:06}"));

        let head = serde_json::json!({
            "method": method,
            "uri": uri.to_string(),
            "headers": headers_to_json(headers),
        });
        write_private(
            &stem.with_extension("req-head.json"),
            head.to_string().as_bytes(),
        );
        write_private(&stem.with_extension("req-body.json"), body);

        Exchange { stem, body: None }.with_body_writer()
    }
}

/// One in-flight request/response pair being written to disk.
#[derive(Debug)]
pub(crate) struct Exchange {
    stem: PathBuf,
    body: Option<mpsc::UnboundedSender<Bytes>>,
}

impl Exchange {
    fn with_body_writer(mut self) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel::<Bytes>();
        let path = self.stem.with_extension("res-body");
        // Blocking file writes stay off the async runtime and out of the response path.
        tokio::task::spawn_blocking(move || {
            let Some(mut file) = create_private(&path) else {
                return;
            };
            while let Some(chunk) = receiver.blocking_recv() {
                if let Err(err) = file.write_all(&chunk) {
                    tracing::warn!(error = %err, path = %path.display(), "dump write failed");
                    return;
                }
            }
        });
        self.body = Some(sender);
        self
    }

    pub(crate) fn record_compiled_body(&self, body: &[u8]) {
        write_private(&self.stem.with_extension("req-compiled.json"), body);
    }

    pub(crate) fn record_trace(&self, trace: &[StageTrace]) {
        let entries: Vec<serde_json::Value> = trace
            .iter()
            .map(|entry| {
                let (outcome, edits) = match &entry.outcome {
                    StageOutcome::Skipped => ("skipped", Vec::new()),
                    StageOutcome::Failed(_) => ("failed", Vec::new()),
                    StageOutcome::TimedOut => ("timed_out", Vec::new()),
                    StageOutcome::Applied(edits) => (
                        "applied",
                        edits
                            .iter()
                            .map(|edit| {
                                serde_json::json!({
                                    "target": format!("{:?}", edit.target),
                                    "pointer": edit.pointer,
                                    "rule": edit.rule,
                                })
                            })
                            .collect(),
                    ),
                };
                serde_json::json!({
                    "stage": entry.stage,
                    "elapsed_micros": entry.elapsed.as_micros() as u64,
                    "outcome": outcome,
                    "edits": edits,
                })
            })
            .collect();
        write_private(
            &self.stem.with_extension("trace.json"),
            serde_json::Value::Array(entries).to_string().as_bytes(),
        );
    }

    pub(crate) fn record_response_head(&self, status: u16, headers: &HeaderMap) {
        let head = serde_json::json!({ "status": status, "headers": headers_to_json(headers) });
        write_private(
            &self.stem.with_extension("res-head.json"),
            head.to_string().as_bytes(),
        );
    }

    pub(crate) fn record_response_chunk(&self, chunk: Bytes) {
        if let Some(body) = &self.body {
            // A closed writer already logged its failure; capture must never affect traffic.
            let _ = body.send(chunk);
        }
    }
}

fn headers_to_json(headers: &HeaderMap) -> serde_json::Value {
    let entries: Vec<serde_json::Value> = headers
        .iter()
        .filter(|(name, _)| !SECRET_HEADERS.contains(name))
        .map(|(name, value)| {
            serde_json::json!([name.as_str(), String::from_utf8_lossy(value.as_bytes())])
        })
        .collect();
    serde_json::Value::Array(entries)
}

/// Writes a file readable only by the current user. Failures are logged, not returned,
/// because capture must never affect traffic.
fn write_private(path: &Path, contents: &[u8]) {
    if let Some(mut file) = create_private(path)
        && let Err(err) = file.write_all(contents)
    {
        tracing::warn!(error = %err, path = %path.display(), "dump write failed");
        let _ = fs::remove_file(path);
    }
}

fn create_private(path: &Path) -> Option<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .inspect_err(|err| tracing::warn!(error = %err, path = %path.display(), "dump open failed"))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn credentials_never_reach_the_dump() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("sk-secret"));
        headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer t"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let rendered = headers_to_json(&headers).to_string();

        assert!(!rendered.contains("sk-secret"), "{rendered}");
        assert!(!rendered.contains("Bearer"), "{rendered}");
        assert!(rendered.contains("anthropic-version"), "{rendered}");
    }
}

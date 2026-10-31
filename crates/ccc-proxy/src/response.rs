//! What happens to a compiled request's response on its way back to Claude Code.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::http::header::{self, HeaderMap};
use ccc_ledger::{Ledger, Usage};
use ccc_redact::{ResponseRehydrator, Vault, rehydrate_json};
use ccc_route::NoticeInjector;
use ccc_wire::sse::{event_json, frame_length};
use futures_util::{Stream, StreamExt, TryStreamExt, stream};
use serde_json::Value;

use crate::Stats;

/// Sessions that have already seen their budget notice, so it appears once.
#[derive(Debug, Default)]
pub(crate) struct NoticedSessions(Mutex<HashSet<String>>);

impl NoticedSessions {
    pub(crate) fn contains(&self, session: &str) -> bool {
        self.0.lock().expect("lock poisoned").contains(session)
    }

    fn insert(&self, session: String) {
        self.0.lock().expect("lock poisoned").insert(session);
    }
}

/// Everything a response needs done to it. Each part is optional.
#[derive(Debug, Default)]
pub(crate) struct ResponsePlan {
    pub(crate) vault: Option<Arc<Vault>>,
    pub(crate) notice: Option<String>,
    pub(crate) accounting: Option<Accounting>,
}

/// Where to record a call's usage.
#[derive(Debug, Clone)]
pub(crate) struct Accounting {
    pub(crate) ledger: Option<Arc<Ledger>>,
    pub(crate) session: String,
    pub(crate) model: String,
    pub(crate) stats: Arc<Stats>,
    pub(crate) noticed: Arc<NoticedSessions>,
}

impl Accounting {
    fn record(&self, usage: Usage, reported_model: Option<String>) {
        let model = reported_model.unwrap_or_else(|| self.model.clone());
        let cost = ccc_ledger::cost_usd(&model, &usage).unwrap_or(0.0);
        self.stats.add_cost(cost);
        let Some(ledger) = self.ledger.clone() else {
            return;
        };
        let session = self.session.clone();
        // Written off the async runtime: a commit can wait on the disk.
        tokio::task::spawn_blocking(move || {
            if let Err(err) = ledger.record(&session, now_millis(), &model, &usage) {
                tracing::warn!(error = %err, "could not record usage");
            }
        });
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Reads usage out of an event stream as it passes.
#[derive(Debug, Default)]
struct UsageTap {
    pending: Vec<u8>,
    usage: Usage,
    model: Option<String>,
}

impl UsageTap {
    fn push(&mut self, chunk: &[u8]) {
        self.pending.extend_from_slice(chunk);
        let mut consumed = 0;
        while let Some(length) = frame_length(&self.pending[consumed..]) {
            if let Some(event) = event_json(&self.pending[consumed..consumed + length]) {
                self.observe(&event);
            }
            consumed += length;
        }
        self.pending.drain(..consumed);
    }

    fn observe(&mut self, event: &Value) {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(message) = event.get("message") {
                    self.observe_message(message);
                }
            }
            Some("message_delta") => {
                if let Some(usage) = event.get("usage") {
                    self.usage.update(&Usage::from_json(usage));
                }
            }
            _ => {}
        }
    }

    fn observe_message(&mut self, message: &Value) {
        if let Some(usage) = message.get("usage") {
            self.usage.update(&Usage::from_json(usage));
        }
        if let Some(model) = message.get("model").and_then(Value::as_str) {
            self.model = Some(model.to_owned());
        }
    }
}

/// Applies a plan to one event stream, chunk by chunk.
#[derive(Debug)]
struct StreamProcessor {
    tap: UsageTap,
    rehydrator: Option<ResponseRehydrator>,
    notice: Option<NoticeInjector>,
    accounting: Option<Accounting>,
}

impl StreamProcessor {
    fn new(plan: ResponsePlan) -> Self {
        Self {
            tap: UsageTap::default(),
            rehydrator: plan.vault.map(ResponseRehydrator::new),
            notice: plan.notice.map(NoticeInjector::new),
            accounting: plan.accounting,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<u8> {
        self.tap.push(chunk);
        let rehydrated = match &mut self.rehydrator {
            Some(rehydrator) => rehydrator.push(chunk),
            None => chunk.to_vec(),
        };
        match &mut self.notice {
            Some(notice) => notice.push(&rehydrated),
            None => rehydrated,
        }
    }

    fn finish(&mut self) -> Vec<u8> {
        let mut out = match &mut self.rehydrator {
            Some(rehydrator) => rehydrator.finish(),
            None => Vec::new(),
        };
        if let Some(notice) = &mut self.notice {
            out = notice.push(&out);
            out.extend(notice.finish());
        }
        if let Some(accounting) = &self.accounting {
            accounting.record(self.tap.usage, self.tap.model.take());
            if self.notice.as_ref().is_some_and(NoticeInjector::delivered) {
                accounting.noticed.insert(accounting.session.clone());
            }
        }
        out
    }
}

/// Builds the body to return for an upstream response.
pub(crate) async fn processed_body<S>(stream: S, headers: &HeaderMap, plan: ResponsePlan) -> Body
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let nothing_to_do = plan.vault.is_none() && plan.notice.is_none() && plan.accounting.is_none();
    if nothing_to_do {
        return Body::from_stream(stream);
    }
    if headers.contains_key(header::CONTENT_ENCODING) {
        tracing::warn!("upstream compressed a response, so it can't be processed");
        return Body::from_stream(stream);
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if content_type.starts_with("text/event-stream") {
        return event_stream_body(stream, plan);
    }
    if content_type.starts_with("application/json") {
        return json_body(stream, plan).await;
    }
    Body::from_stream(stream)
}

fn event_stream_body<S>(events: S, plan: ResponsePlan) -> Body
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let processor = Arc::new(Mutex::new(StreamProcessor::new(plan)));
    let finishing = Arc::clone(&processor);
    let processed = events
        .map_ok(move |chunk| Bytes::from(processor.lock().expect("lock poisoned").push(&chunk)));
    let tail = stream::once(async move {
        Ok(Bytes::from(
            finishing.lock().expect("lock poisoned").finish(),
        ))
    });
    Body::from_stream(processed.chain(tail))
}

async fn json_body<S>(body: S, plan: ResponsePlan) -> Body
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let collected: Result<Vec<Bytes>, reqwest::Error> = body.try_collect().await;
    let chunks = match collected {
        Ok(chunks) => chunks,
        Err(err) => {
            tracing::warn!(error = %err, "upstream response failed partway");
            return Body::from(err.to_string());
        }
    };
    let whole = chunks.concat();
    if let Some(accounting) = &plan.accounting
        && let Ok(message) = serde_json::from_slice::<Value>(&whole)
        && let Some(usage) = message.get("usage")
    {
        let model = message
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        accounting.record(Usage::from_json(usage), model);
    }
    let rehydrated = plan
        .vault
        .as_ref()
        .and_then(|vault| rehydrate_json(&whole, vault));
    Body::from(rehydrated.unwrap_or(whole))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_tap_reads_input_from_the_start_and_output_from_the_end_across_chunks() {
        let start = json!({ "type": "message_start", "message": { "model": "m", "usage": {
            "input_tokens": 2, "cache_read_input_tokens": 100, "output_tokens": 1,
        } } });
        let delta = json!({ "type": "message_delta", "usage": { "output_tokens": 50 } });
        let stream = format!("event: a\ndata: {start}\n\nevent: b\ndata: {delta}\n\n");

        let mut tap = UsageTap::default();
        for chunk in stream.as_bytes().chunks(7) {
            tap.push(chunk);
        }

        assert_eq!(tap.usage.input, 2);
        assert_eq!(tap.usage.cache_read, 100);
        assert_eq!(tap.usage.output, 50);
        assert_eq!(tap.model.as_deref(), Some("m"));
    }
}

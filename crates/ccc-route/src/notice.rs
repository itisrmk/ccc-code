//! A one-line budget notice shown in the terminal, invisible to the model.
//!
//! The notice is added to a response as its own text block, so Claude Code displays and
//! stores it. The server never produced that block, so it has to come back out of the next
//! request, or the history would no longer match what thinking blocks were bound to.

use std::time::Duration;

use ccc_pipeline::{Context, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::Request;
use ccc_wire::sse::{event_json, frame_length};
use serde_json::{Value, json};

/// Starts every notice, and identifies notice blocks in later requests.
const NOTICE_PREFIX: &str = "[ccc budget] ";
const TIME_CAP: Duration = Duration::from_millis(50);

/// The notice for a session over its budget.
#[must_use]
pub fn budget_notice(spent_usd: f64, budget_usd: f64) -> String {
    format!(
        "{NOTICE_PREFIX}This session has used ${spent_usd:.2} of its ${budget_usd:.2} budget at \
         list prices. Consider a cheaper model with /model, or a new session."
    )
}

fn is_notice(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("text")
        && block
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.starts_with(NOTICE_PREFIX))
}

/// Removes notice blocks from history before a request goes out.
#[derive(Debug, Clone, Copy, Default)]
pub struct StripNotices;

impl Stage for StripNotices {
    fn name(&self) -> &'static str {
        "strip-notices"
    }

    fn time_cap(&self) -> Duration {
        TIME_CAP
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, _: &Context) -> Result<Vec<Edit>, StageError> {
        let mut edits = Vec::new();
        for (index, message) in request.messages().iter().enumerate() {
            let Some(blocks) = message
                .get("content")
                .and_then(Value::as_array)
                .filter(|_| message.get("role").and_then(Value::as_str) == Some("assistant"))
            else {
                continue;
            };
            if blocks.iter().any(is_notice) {
                let kept: Vec<Value> = blocks.iter().filter(|b| !is_notice(b)).cloned().collect();
                edits.push(Edit {
                    target: Target::Message(index),
                    pointer: "/content".to_owned(),
                    value: Value::Array(kept),
                    rule: "budget notice",
                });
            }
        }
        Ok(edits)
    }
}

/// Adds a notice as the first block of a response, moving the real blocks along by one.
///
/// It goes first because `claude -p` reports only the last text block as its result, so a
/// notice at the end would replace the model's answer. Events that carry no block index
/// stay byte-identical.
#[derive(Debug)]
pub struct NoticeInjector {
    notice: Option<String>,
    pending: Vec<u8>,
    shifting: bool,
}

impl NoticeInjector {
    /// An injector that will add `notice` to the next message it sees start.
    #[must_use]
    pub fn new(notice: String) -> Self {
        Self {
            notice: Some(notice),
            pending: Vec::new(),
            shifting: false,
        }
    }

    /// Whether the notice has been added to the stream.
    #[must_use]
    pub fn delivered(&self) -> bool {
        self.notice.is_none()
    }

    /// Processes the next chunk of the stream and returns the bytes to send on.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<u8> {
        self.pending.extend_from_slice(chunk);
        let mut out = Vec::with_capacity(self.pending.len());
        let mut consumed = 0;
        while let Some(length) = frame_length(&self.pending[consumed..]) {
            let frame = self.pending[consumed..consumed + length].to_vec();
            consumed += length;
            self.process_frame(&frame, &mut out);
        }
        self.pending.drain(..consumed);
        out
    }

    /// Returns anything left when the stream ends.
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }

    fn process_frame(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let Some(mut event) = event_json(frame) else {
            out.extend_from_slice(frame);
            return;
        };
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if kind == "message_start" {
            out.extend_from_slice(frame);
            if let Some(notice) = self.notice.take() {
                write_text_block(0, &notice, out);
                self.shifting = true;
            }
            return;
        }
        match event.get("index").and_then(Value::as_u64) {
            Some(index) if self.shifting => {
                event["index"] = json!(index + 1);
                out.extend_from_slice(format!("event: {kind}\ndata: {event}\n\n").as_bytes());
            }
            _ => out.extend_from_slice(frame),
        }
    }
}

fn write_text_block(index: u64, text: &str, out: &mut Vec<u8>) {
    let start = json!({ "type": "content_block_start", "index": index,
        "content_block": { "type": "text", "text": "" } });
    let delta = json!({ "type": "content_block_delta", "index": index,
        "delta": { "type": "text_delta", "text": text } });
    let stop = json!({ "type": "content_block_stop", "index": index });
    let events = format!(
        "event: content_block_start\ndata: {start}\n\n\
         event: content_block_delta\ndata: {delta}\n\n\
         event: content_block_stop\ndata: {stop}\n\n"
    );
    out.extend_from_slice(events.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(name: &str, data: Value) -> String {
        format!("event: {name}\ndata: {data}\n\n")
    }

    fn stream(stop_reason: &str) -> String {
        [
            event("message_start", json!({ "type": "message_start", "message": {} })),
            event("content_block_start", json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } })),
            event("content_block_delta", json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "hi" } })),
            event("content_block_stop", json!({ "type": "content_block_stop", "index": 0 })),
            event("message_delta", json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason } })),
            event("message_stop", json!({ "type": "message_stop" })),
        ]
        .concat()
    }

    fn run(chunk_size: usize, input: &str) -> (String, bool) {
        let mut injector = NoticeInjector::new(budget_notice(2.0, 1.0));
        let mut out = Vec::new();
        for chunk in input.as_bytes().chunks(chunk_size) {
            out.extend(injector.push(chunk));
        }
        out.extend(injector.finish());
        (String::from_utf8(out).unwrap(), injector.delivered())
    }

    fn events(stream: &str) -> Vec<Value> {
        stream
            .split("\n\n")
            .filter_map(|frame| event_json(frame.as_bytes()))
            .collect()
    }

    #[test]
    fn adds_a_first_block_and_moves_the_real_ones_along() {
        for chunk_size in [1, 13, 4096] {
            let (out, delivered) = run(chunk_size, &stream("tool_use"));
            assert!(delivered);
            let events = events(&out);
            let kinds: Vec<(&str, Option<u64>)> = events
                .iter()
                .map(|event| {
                    (
                        event["type"].as_str().unwrap(),
                        event.get("index").and_then(Value::as_u64),
                    )
                })
                .collect();
            assert_eq!(
                kinds,
                [
                    ("message_start", None),
                    ("content_block_start", Some(0)),
                    ("content_block_delta", Some(0)),
                    ("content_block_stop", Some(0)),
                    ("content_block_start", Some(1)),
                    ("content_block_delta", Some(1)),
                    ("content_block_stop", Some(1)),
                    ("message_delta", None),
                    ("message_stop", None),
                ]
            );
            assert!(
                events[2]["delta"]["text"]
                    .as_str()
                    .unwrap()
                    .starts_with(NOTICE_PREFIX)
            );
            assert_eq!(events[5]["delta"]["text"], "hi", "the answer stays last");
        }
    }

    #[test]
    fn a_stream_that_never_starts_gets_no_notice() {
        let input = event("error", json!({ "type": "error", "error": {} }));
        let (out, delivered) = run(4096, &input);
        assert!(!delivered);
        assert_eq!(out, input);
    }

    #[test]
    fn strips_exactly_the_notice_from_history() {
        let notice = budget_notice(2.0, 1.0);
        let body = json!({ "messages": [
            { "role": "user", "content": [{ "type": "text", "text": notice.clone() }] },
            { "role": "assistant", "content": [
                { "type": "thinking", "thinking": "", "signature": "s" },
                { "type": "text", "text": "answer" },
                { "type": "text", "text": notice },
            ] },
        ] });
        let request = Request::parse(body.to_string().as_bytes()).unwrap();

        let edits = StripNotices.plan(&request, &Context::default()).unwrap();

        assert_eq!(edits.len(), 1, "user text saying the same thing stays");
        let kept = edits[0].value.as_array().unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0]["type"], "thinking");
        assert_eq!(kept[1]["text"], "answer");
    }

    #[test]
    fn leaves_history_without_notices_alone() {
        let body = json!({ "messages": [{ "role": "assistant", "content": [{ "type": "text", "text": "hi" }] }] });
        let request = Request::parse(body.to_string().as_bytes()).unwrap();
        assert!(
            StripNotices
                .plan(&request, &Context::default())
                .unwrap()
                .is_empty()
        );
    }
}

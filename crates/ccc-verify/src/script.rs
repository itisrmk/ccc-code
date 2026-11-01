//! The conversation sent through a gateway, and the responses the stand-in gives back.

use serde_json::{Value, json};

pub(crate) const MODEL: &str = "claude-opus-5-5";
pub(crate) const BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";
pub(crate) const STREAM_THINKING: &str = "Checking the build output first.";
pub(crate) const STREAM_SIGNATURE: &str = "ccc-verify-stream-signature-Zm9vYmFy";
pub(crate) const STREAM_TEXT: &str = "The build is fixed.";
pub(crate) const ERROR_MESSAGE: &str = "ccc-verify scripted rejection";
pub(crate) const RETRY_AFTER_SECONDS: &str = "7";
/// Scripted cache figures, to see whether usage survives the trip.
pub(crate) const CACHE_READ_TOKENS: u64 = 123_456;
pub(crate) const CACHE_WRITE_1H_TOKENS: u64 = 789;

/// A long build log, the kind of output a compressing gateway is most tempted to touch.
fn build_log() -> String {
    let mut log = String::new();
    for index in 0..4000 {
        log.push_str(&format!("test module::case_{index} ... ok\n"));
    }
    log.push_str("test module::flaky ... FAILED\ntest result: FAILED. 4000 passed; 1 failed\n");
    log
}

fn thinking(text: &str, signature: &str) -> Value {
    json!({ "type": "thinking", "thinking": text, "signature": signature })
}

fn tool_use(id: &str, command: &str) -> Value {
    json!({ "type": "tool_use", "id": id, "name": "Bash", "input": { "command": command } })
}

fn tool_result(id: &str, content: &str) -> Value {
    json!({ "role": "user", "content": [
        { "type": "tool_result", "tool_use_id": id, "content": content },
    ] })
}

fn all_messages() -> Vec<Value> {
    let log = build_log();
    vec![
        json!({ "role": "user", "content": [{ "type": "text", "text": "Fix the failing test." }] }),
        json!({ "role": "assistant", "content": [
            thinking("I should run the tests.", "ccc-verify-signature-one"),
            tool_use("toolu_1", "cargo test"),
        ] }),
        tool_result("toolu_1", &log),
        json!({ "role": "assistant", "content": [
            { "type": "redacted_thinking", "data": "ccc-verify-redacted-data" },
            thinking("One test failed. Run it again to be sure.", "ccc-verify-signature-two"),
            tool_use("toolu_2", "cargo test"),
        ] }),
        tool_result("toolu_2", &log),
        json!({ "role": "assistant", "content": [
            thinking("It fails every time.", "ccc-verify-signature-three"),
            { "type": "text", "text": "The test named flaky fails on every run." },
        ] }),
        json!({ "role": "user", "content": [{ "type": "text", "text": "Please fix it." }] }),
    ]
}

/// How many requests the conversation grows through.
pub(crate) const TURNS: usize = 4;

/// The request a client would send on `turn`, from 0. As in Claude Code, the cache marker moves
/// to the newest block each time.
pub(crate) fn request(turn: usize, stream: bool) -> Value {
    let mut messages: Vec<Value> = all_messages();
    messages.truncate(2 * turn + 1);
    if let Some(block) = messages
        .last_mut()
        .and_then(|message| message["content"].as_array_mut())
        .and_then(|content| content.last_mut())
    {
        block["cache_control"] = json!({ "type": "ephemeral", "ttl": "1h" });
    }
    json!({
        "model": MODEL,
        "max_tokens": 1024,
        "stream": stream,
        "thinking": {
            "type": "adaptive",
            "block_binding": { "prefix_mismatch_behavior": "error" },
        },
        "system": [{
            "type": "text",
            "text": "You are a test fixture.",
            "cache_control": { "type": "ephemeral", "ttl": "1h" },
        }],
        "tools": [{
            "name": "Bash",
            "description": "Runs a command.",
            "input_schema": { "type": "object", "properties": { "command": { "type": "string" } } },
        }],
        "metadata": { "user_id": "ccc-verify" },
        "x_future_field": { "kept": true },
        "messages": messages,
    })
}

/// A single large message, to see whether a gateway refuses the sizes Claude Code really sends.
pub(crate) fn large_request(bytes: usize) -> Value {
    json!({
        "model": MODEL,
        "max_tokens": 16,
        "stream": false,
        "messages": [{ "role": "user", "content": "x ".repeat(bytes / 2) }],
    })
}

fn usage() -> Value {
    json!({
        "input_tokens": 3,
        "cache_read_input_tokens": CACHE_READ_TOKENS,
        "cache_creation_input_tokens": CACHE_WRITE_1H_TOKENS,
        "cache_creation": { "ephemeral_1h_input_tokens": CACHE_WRITE_1H_TOKENS },
        "output_tokens": 1,
    })
}

/// The whole-response form of an answer.
pub(crate) fn response_body() -> Value {
    json!({
        "id": "msg_verify",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            thinking(STREAM_THINKING, STREAM_SIGNATURE),
            { "type": "text", "text": STREAM_TEXT },
        ],
        "stop_reason": "end_turn",
        "usage": usage(),
    })
}

fn event(name: &str, data: &Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

/// The streamed form of an answer, in parts sent with a pause between them.
pub(crate) fn stream_chunks() -> Vec<String> {
    let opening = [
        event(
            "message_start",
            &json!({ "type": "message_start", "message": {
                "id": "msg_verify", "type": "message", "role": "assistant", "model": MODEL,
                "content": [], "usage": usage(),
            } }),
        ),
        event("ping", &json!({ "type": "ping" })),
        event(
            "content_block_start",
            &json!({ "type": "content_block_start", "index": 0,
                "content_block": { "type": "thinking", "thinking": "", "signature": "" } }),
        ),
    ]
    .concat();
    let thinking = [
        event(
            "content_block_delta",
            &json!({ "type": "content_block_delta", "index": 0,
                "delta": { "type": "thinking_delta", "thinking": STREAM_THINKING } }),
        ),
        event(
            "content_block_delta",
            &json!({ "type": "content_block_delta", "index": 0,
                "delta": { "type": "signature_delta", "signature": STREAM_SIGNATURE } }),
        ),
        event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 0 }),
        ),
        event(
            "future_event_type",
            &json!({ "type": "future_event_type", "kept": true }),
        ),
    ]
    .concat();
    let closing = [
        event(
            "content_block_start",
            &json!({ "type": "content_block_start", "index": 1,
                "content_block": { "type": "text", "text": "" } }),
        ),
        event(
            "content_block_delta",
            &json!({ "type": "content_block_delta", "index": 1,
                "delta": { "type": "text_delta", "text": STREAM_TEXT } }),
        ),
        event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 1 }),
        ),
        event(
            "message_delta",
            &json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 40 } }),
        ),
        event("message_stop", &json!({ "type": "message_stop" })),
    ]
    .concat();
    vec![opening, thinking, closing]
}

/// An API error body.
pub(crate) fn error_body(kind: &str) -> Value {
    json!({ "type": "error", "error": { "type": kind, "message": ERROR_MESSAGE } })
}

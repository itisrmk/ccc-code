//! Swaps placeholders back to real values in responses.
//!
//! Text arrives in arbitrary fragments, so a placeholder can be split across deltas. Each
//! content block holds back anything that might be the start of one until it can tell.
//! Thinking and signature deltas pass through untouched: thinking blocks are signed, and
//! the model only ever wrote placeholders there.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::vault::{PLACEHOLDER_HEX_DIGITS, PLACEHOLDER_PREFIX, Vault};

/// How substituted text must be encoded where it lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    /// Plain text, as in a `text_delta`.
    Text,
    /// Inside a JSON string, as in the raw fragments of an `input_json_delta`.
    JsonFragment,
}

/// Replaces placeholders in a text stream, whatever the fragment boundaries.
#[derive(Debug, Default)]
struct Matcher {
    held: String,
}

impl Matcher {
    fn push(&mut self, fragment: &str, vault: &Vault, encoding: Encoding) -> String {
        self.held.push_str(fragment);
        let buffered = std::mem::take(&mut self.held);
        let mut out = String::with_capacity(buffered.len());
        let mut rest = buffered.as_str();
        while let Some(start) = rest.find('<') {
            out.push_str(&rest[..start]);
            let candidate = &rest[start..];
            match classify(candidate) {
                Candidate::Complete(token) => {
                    match vault.secret_for(token) {
                        Some(secret) => out.push_str(&encode(&secret, encoding)),
                        None => out.push_str(token),
                    }
                    rest = &candidate[token.len()..];
                }
                Candidate::Partial => {
                    self.held.push_str(candidate);
                    return out;
                }
                Candidate::No => {
                    out.push('<');
                    rest = &candidate[1..];
                }
            }
        }
        out.push_str(rest);
        out
    }

    /// Whatever is still held, which can no longer become a placeholder.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

enum Candidate<'a> {
    Complete(&'a str),
    Partial,
    No,
}

/// Decides what the text starting at a `<` could be.
fn classify(text: &str) -> Candidate<'_> {
    let prefix_len = PLACEHOLDER_PREFIX.len();
    if text.len() < prefix_len {
        return if PLACEHOLDER_PREFIX.starts_with(text) {
            Candidate::Partial
        } else {
            Candidate::No
        };
    }
    if !text.starts_with(PLACEHOLDER_PREFIX) {
        return Candidate::No;
    }
    let token_len = prefix_len + PLACEHOLDER_HEX_DIGITS + 1;
    let digits = &text.as_bytes()[prefix_len..text.len().min(prefix_len + PLACEHOLDER_HEX_DIGITS)];
    if !digits
        .iter()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Candidate::No;
    }
    if text.len() < token_len {
        return Candidate::Partial;
    }
    if text.as_bytes()[token_len - 1] == b'>' {
        Candidate::Complete(&text[..token_len])
    } else {
        Candidate::No
    }
}

fn encode(secret: &str, encoding: Encoding) -> String {
    match encoding {
        Encoding::Text => secret.to_owned(),
        Encoding::JsonFragment => {
            let quoted = Value::String(secret.to_owned()).to_string();
            quoted[1..quoted.len() - 1].to_owned()
        }
    }
}

/// Which delta a block is streaming, needed to flush what it holds when it ends.
#[derive(Debug)]
struct BlockStream {
    matcher: Matcher,
    field: &'static str,
    delta_type: &'static str,
}

/// Rewrites a server-sent event stream, leaving events it doesn't change byte-identical.
#[derive(Debug)]
pub struct ResponseRehydrator {
    vault: Arc<Vault>,
    pending: Vec<u8>,
    blocks: HashMap<u64, BlockStream>,
}

impl ResponseRehydrator {
    /// A rehydrator resolving placeholders from `vault`.
    #[must_use]
    pub fn new(vault: Arc<Vault>) -> Self {
        Self {
            vault,
            pending: Vec::new(),
            blocks: HashMap::new(),
        }
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

    /// Returns anything left when the stream ends, such as a truncated final event.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut out = std::mem::take(&mut self.pending);
        let mut indexes: Vec<u64> = self.blocks.keys().copied().collect();
        indexes.sort_unstable();
        for index in indexes {
            flush_block(&mut self.blocks, index, &mut out);
        }
        out
    }

    fn process_frame(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let Some(event) = parse_event(frame) else {
            out.extend_from_slice(frame);
            return;
        };
        let index = event.get("index").and_then(Value::as_u64);
        match (event.get("type").and_then(Value::as_str), index) {
            (Some("content_block_delta"), Some(index)) => {
                self.process_delta(frame, &event, index, out);
            }
            (Some("content_block_stop"), Some(index)) => {
                flush_block(&mut self.blocks, index, out);
                out.extend_from_slice(frame);
            }
            _ => out.extend_from_slice(frame),
        }
    }

    fn process_delta(&mut self, frame: &[u8], event: &Value, index: u64, out: &mut Vec<u8>) {
        let delta = event.get("delta");
        let (field, delta_type, encoding) = match delta
            .and_then(|delta| delta.get("type"))
            .and_then(Value::as_str)
        {
            Some("text_delta") => ("text", "text_delta", Encoding::Text),
            Some("input_json_delta") => {
                ("partial_json", "input_json_delta", Encoding::JsonFragment)
            }
            _ => {
                out.extend_from_slice(frame);
                return;
            }
        };
        let Some(fragment) = delta
            .and_then(|delta| delta.get(field))
            .and_then(Value::as_str)
        else {
            out.extend_from_slice(frame);
            return;
        };
        let block = self.blocks.entry(index).or_insert_with(|| BlockStream {
            matcher: Matcher::default(),
            field,
            delta_type,
        });
        let produced = block.matcher.push(fragment, &self.vault, encoding);
        if produced == fragment {
            out.extend_from_slice(frame);
        } else if !produced.is_empty() {
            write_delta(index, delta_type, field, &produced, out);
        }
    }
}

fn flush_block(blocks: &mut HashMap<u64, BlockStream>, index: u64, out: &mut Vec<u8>) {
    if let Some(mut block) = blocks.remove(&index) {
        let held = block.matcher.finish();
        if !held.is_empty() {
            write_delta(index, block.delta_type, block.field, &held, out);
        }
    }
}

fn write_delta(index: u64, delta_type: &str, field: &str, text: &str, out: &mut Vec<u8>) {
    let mut delta = serde_json::Map::new();
    delta.insert("type".to_owned(), json!(delta_type));
    delta.insert(field.to_owned(), json!(text));
    let event = json!({ "type": "content_block_delta", "index": index, "delta": delta });
    out.extend_from_slice(format!("event: content_block_delta\ndata: {event}\n\n").as_bytes());
}

/// The length of the first complete event in `buffer`, including its blank line.
fn frame_length(buffer: &[u8]) -> Option<usize> {
    let lf = find(buffer, b"\n\n").map(|at| at + 2);
    let crlf = find(buffer, b"\r\n\r\n").map(|at| at + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The JSON payload of an event, if it has one.
fn parse_event(frame: &[u8]) -> Option<Value> {
    let text = std::str::from_utf8(frame).ok()?;
    let data = text
        .lines()
        .find_map(|line| line.strip_prefix("data:"))
        .map(str::trim)?;
    serde_json::from_str(data).ok()
}

/// Rehydrates a complete, non-streaming JSON response. Returns `None` if nothing changed.
#[must_use]
pub fn rehydrate_json(body: &[u8], vault: &Vault) -> Option<Vec<u8>> {
    let mut response: Value = serde_json::from_slice(body).ok()?;
    let blocks = response.get_mut("content")?.as_array_mut()?;
    let mut changed = false;
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get_mut("text") {
                    changed |= rehydrate_value(text, vault);
                }
            }
            Some("tool_use") => {
                if let Some(input) = block.get_mut("input") {
                    changed |= rehydrate_value(input, vault);
                }
            }
            _ => {}
        }
    }
    changed.then(|| response.to_string().into_bytes())
}

fn rehydrate_value(value: &mut Value, vault: &Vault) -> bool {
    match value {
        Value::String(text) => {
            let mut matcher = Matcher::default();
            let mut replaced = matcher.push(text, vault, Encoding::Text);
            replaced.push_str(&matcher.finish());
            let changed = replaced != *text;
            *text = replaced;
            changed
        }
        Value::Array(items) => items.iter_mut().fold(false, |changed, item| {
            rehydrate_value(item, vault) | changed
        }),
        Value::Object(map) => map.values_mut().fold(false, |changed, item| {
            rehydrate_value(item, vault) | changed
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "hunter2-hunter2";

    fn vault() -> (Arc<Vault>, String) {
        let vault = Arc::new(Vault::ephemeral().unwrap());
        let placeholder = vault.placeholder_for(SECRET);
        (vault, placeholder)
    }

    fn matched(matcher: &mut Matcher, vault: &Vault, parts: &[&str], encoding: Encoding) -> String {
        let mut out = String::new();
        for part in parts {
            out.push_str(&matcher.push(part, vault, encoding));
        }
        out.push_str(&matcher.finish());
        out
    }

    #[test]
    fn a_placeholder_split_at_every_byte_is_still_found() {
        let (vault, placeholder) = vault();
        let text = format!("start {placeholder} end");
        let want = format!("start {SECRET} end");
        for split in 0..=text.len() {
            let (left, right) = text.split_at(split);
            let got = matched(
                &mut Matcher::default(),
                &vault,
                &[left, right],
                Encoding::Text,
            );
            assert_eq!(got, want, "split at {split}");
        }
    }

    #[test]
    fn a_placeholder_delivered_one_byte_at_a_time_is_found() {
        let (vault, placeholder) = vault();
        let parts: Vec<String> = placeholder.chars().map(String::from).collect();
        let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
        let got = matched(&mut Matcher::default(), &vault, &parts, Encoding::Text);
        assert_eq!(got, SECRET);
    }

    #[test]
    fn text_that_only_looks_like_a_placeholder_passes_through() {
        let (vault, _) = vault();
        for text in [
            "plain",
            "a < b",
            "<div>tags</div>",
            "<SECRET",
            "<SECRET_",
            "<SECRET_zzzzzzzzzz>",
            "<SECRET_0123",
            "<SECRET_0000000000>",
            "<SECRET_0000000000 trailing",
            "<<<",
        ] {
            let got = matched(&mut Matcher::default(), &vault, &[text], Encoding::Text);
            assert_eq!(got, text);
        }
    }

    #[test]
    fn multibyte_text_near_placeholders_survives() {
        let (vault, placeholder) = vault();
        let text = format!("é<😀{placeholder}é");
        let got = matched(&mut Matcher::default(), &vault, &[&text], Encoding::Text);
        assert_eq!(got, format!("é<😀{SECRET}é"));
    }

    #[test]
    fn json_fragments_escape_substituted_values() {
        let vault = Arc::new(Vault::ephemeral().unwrap());
        let awkward = "quote\" back\\slash\nnewline";
        let placeholder = vault.placeholder_for(awkward);
        let fragment = format!(r#"{{"command":"use {placeholder}"}}"#);

        let got = matched(
            &mut Matcher::default(),
            &vault,
            &[&fragment],
            Encoding::JsonFragment,
        );

        let parsed: Value = serde_json::from_str(&got).unwrap();
        assert_eq!(parsed["command"], format!("use {awkward}"));
    }

    fn text_delta(index: u64, text: &str) -> String {
        let data = json!({ "type": "content_block_delta", "index": index,
            "delta": { "type": "text_delta", "text": text } });
        format!("event: content_block_delta\ndata: {data}\n\n")
    }

    fn stop(index: u64) -> String {
        let data = json!({ "type": "content_block_stop", "index": index });
        format!("event: content_block_stop\ndata: {data}\n\n")
    }

    /// The text of all deltas in a stream, per block.
    fn collect_text(stream: &[u8]) -> String {
        std::str::from_utf8(stream)
            .unwrap()
            .split("\n\n")
            .filter_map(|frame| parse_event(frame.as_bytes()))
            .filter_map(|event| {
                event
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect()
    }

    #[test]
    fn rewrites_a_split_placeholder_across_deltas_and_network_chunks() {
        let (vault, placeholder) = vault();
        let (left, right) = placeholder.split_at(7);
        let stream = format!(
            "{}{}{}",
            text_delta(0, &format!("use {left}")),
            text_delta(0, &format!("{right} now")),
            stop(0)
        );
        let want = format!("use {SECRET} now");
        for chunk_size in [1, 2, 7, 64, stream.len()] {
            let mut rehydrator = ResponseRehydrator::new(Arc::clone(&vault));
            let mut out = Vec::new();
            for chunk in stream.as_bytes().chunks(chunk_size) {
                out.extend(rehydrator.push(chunk));
            }
            out.extend(rehydrator.finish());
            assert_eq!(collect_text(&out), want, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn held_text_is_flushed_before_the_block_stops() {
        let (vault, _) = vault();
        let mut rehydrator = ResponseRehydrator::new(vault);
        let stream = format!("{}{}", text_delta(0, "ends with <SEC"), stop(0));
        let out = rehydrator.push(stream.as_bytes());
        let text = String::from_utf8(out).unwrap();
        let flushed = text.find("<SEC").unwrap();
        let stopped = text.find("content_block_stop").unwrap();
        assert!(flushed < stopped, "{text}");
        assert_eq!(collect_text(text.as_bytes()), "ends with <SEC");
    }

    #[test]
    fn blocks_are_tracked_independently() {
        let (vault, placeholder) = vault();
        let (left, right) = placeholder.split_at(10);
        let stream = format!(
            "{}{}{}{}{}",
            text_delta(0, left),
            text_delta(1, "other"),
            text_delta(0, right),
            stop(0),
            stop(1)
        );
        let mut rehydrator = ResponseRehydrator::new(vault);
        let out = rehydrator.push(stream.as_bytes());
        let text = collect_text(&out);
        assert!(text.contains(SECRET), "{text}");
        assert!(text.contains("other"), "{text}");
    }

    #[test]
    fn events_without_placeholders_stay_byte_identical() {
        let (vault, _) = vault();
        // Key order and spacing a re-serialization would change.
        let stream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",  \"message\":{\"id\":\"m\"}}\n\n",
            "event: ping\ndata: {\"type\": \"ping\"}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\",\"type\":\"text_delta\"},\"index\":0,\"type\":\"content_block_delta\"}\n\n",
            ": comment\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let mut rehydrator = ResponseRehydrator::new(vault);
        let mut out = rehydrator.push(stream.as_bytes());
        out.extend(rehydrator.finish());
        assert_eq!(String::from_utf8(out).unwrap(), stream);
    }

    #[test]
    fn signed_thinking_and_signature_deltas_are_not_touched() {
        let (vault, placeholder) = vault();
        let thinking = json!({ "type": "content_block_delta", "index": 0,
            "delta": { "type": "thinking_delta", "thinking": placeholder } });
        let signature = json!({ "type": "content_block_delta", "index": 0,
            "delta": { "type": "signature_delta", "signature": placeholder } });
        let stream = format!(
            "event: content_block_delta\ndata: {thinking}\n\nevent: content_block_delta\ndata: {signature}\n\n"
        );
        let mut rehydrator = ResponseRehydrator::new(vault);
        let out = rehydrator.push(stream.as_bytes());
        assert_eq!(String::from_utf8(out).unwrap(), stream);
    }

    #[test]
    fn tool_arguments_rehydrate_across_deltas_as_valid_json() {
        let vault = Arc::new(Vault::ephemeral().unwrap());
        let awkward = "pa\"ss\\word";
        let placeholder = vault.placeholder_for(awkward);
        let (left, right) = placeholder.split_at(5);
        let delta = |json: String| {
            let data = json!({ "type": "content_block_delta", "index": 1,
                "delta": { "type": "input_json_delta", "partial_json": json } });
            format!("event: content_block_delta\ndata: {data}\n\n")
        };
        let stream = format!(
            "{}{}{}",
            delta(format!("{{\"command\":\"{left}")),
            delta(format!("{right}\"}}")),
            stop(1)
        );
        let mut rehydrator = ResponseRehydrator::new(vault);
        let out = rehydrator.push(stream.as_bytes());
        let joined: String = std::str::from_utf8(&out)
            .unwrap()
            .split("\n\n")
            .filter_map(|frame| parse_event(frame.as_bytes()))
            .filter_map(|event| {
                event
                    .pointer("/delta/partial_json")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        let input: Value = serde_json::from_str(&joined).unwrap();
        assert_eq!(input["command"], awkward);
    }

    #[test]
    fn a_truncated_stream_returns_everything_it_held() {
        let (vault, _) = vault();
        let mut rehydrator = ResponseRehydrator::new(vault);
        let mut out = rehydrator.push(text_delta(0, "tail <SEC").as_bytes());
        out.extend(rehydrator.push(b"event: partial"));
        out.extend(rehydrator.finish());
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("event: partial"), "{text}");
        assert!(text.contains("<SEC"), "{text}");
    }

    #[test]
    fn rehydrates_non_streaming_responses() {
        let (vault, placeholder) = vault();
        let body = json!({ "type": "message", "content": [
            { "type": "thinking", "thinking": placeholder, "signature": "s" },
            { "type": "text", "text": format!("use {placeholder}") },
            { "type": "tool_use", "id": "t", "name": "Bash", "input": { "a": [placeholder] } },
        ] });
        let out = rehydrate_json(body.to_string().as_bytes(), &vault).unwrap();
        let parsed: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed["content"][0]["thinking"], placeholder);
        assert_eq!(parsed["content"][1]["text"], format!("use {SECRET}"));
        assert_eq!(parsed["content"][2]["input"]["a"][0], SECRET);

        let clean = json!({ "content": [{ "type": "text", "text": "fine" }] });
        assert_eq!(rehydrate_json(clean.to_string().as_bytes(), &vault), None);
    }
}

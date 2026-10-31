//! Rebuilds exact wire requests from `~/.claude/projects` JSONL files.
//!
//! An `api-request` record lists its messages as a kept prefix of an earlier request
//! plus a tail of content hashes. A hash names either an `api-request-blob` record or
//! a message that also appears as ordinary `user` and `assistant` records.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ccc_ledger::Usage;
use ccc_wire::Request;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::BenchError;
use crate::timestamp::parse_millis;

/// Fraction of a request's messages that may be unresolved while still replaying it.
/// Above this the size comparison against recorded usage stops being meaningful.
const MAX_UNRESOLVED_FRACTION: f64 = 0.02;

/// The field added to assistant messages to record their hidden thinking tokens.
pub(crate) const THINKING_TOKENS_KEY: &str = "ccc_thinking_tokens";

/// One API call rebuilt from a transcript.
#[derive(Debug, Clone)]
pub struct Call {
    /// When the request was sent, in milliseconds since the Unix epoch.
    pub at_millis: i64,
    /// The request as sent, with image data blanked to save memory.
    pub request: Request,
    /// What the API reported.
    pub usage: Usage,
}

impl Call {
    /// The model the request named, or an empty string if it named none.
    #[must_use]
    pub fn model(&self) -> &str {
        self.request.model().unwrap_or_default()
    }
}

/// The replayable calls from one transcript file, oldest first.
#[derive(Debug, Clone)]
pub struct Session {
    /// The transcript file.
    pub path: PathBuf,
    /// Calls whose messages were almost entirely resolved.
    pub calls: Vec<Call>,
}

/// What loading a directory found, including what could not be used.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadSummary {
    /// Transcript files read.
    pub files: usize,
    /// Files with calls but no wire records, from older Claude Code versions.
    pub files_without_wire_records: usize,
    /// Calls with recorded usage, whether or not they could be replayed.
    pub calls_seen: usize,
    /// Calls left out because too many of their messages could not be resolved.
    pub calls_unresolved: usize,
    /// Message references across replayed and unreplayed requests.
    pub message_refs: usize,
    /// Message references that matched no record.
    pub message_refs_unresolved: usize,
}

/// Loads every `.jsonl` transcript under `dir`.
///
/// # Errors
///
/// Returns [`BenchError::Read`] if the directory tree or a file cannot be read.
pub fn load_dir(dir: &Path) -> Result<(Vec<Session>, LoadSummary), BenchError> {
    let mut paths = Vec::new();
    collect_transcripts(dir, &mut paths)?;
    paths.sort();

    let mut summary = LoadSummary::default();
    let mut sessions = Vec::new();
    for path in paths {
        let session = load_file(&path, &mut summary)?;
        if !session.calls.is_empty() {
            sessions.push(session);
        }
    }
    Ok((sessions, summary))
}

fn collect_transcripts(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), BenchError> {
    let read_error = |source| BenchError::Read {
        path: dir.to_owned(),
        source,
    };
    for entry in fs::read_dir(dir).map_err(read_error)? {
        let path = entry.map_err(read_error)?.path();
        if path.is_dir() {
            collect_transcripts(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            found.push(path);
        }
    }
    Ok(())
}

#[derive(Default)]
struct RawTranscript {
    /// Wire messages by content hash.
    messages: HashMap<String, Arc<Value>>,
    /// `{system, tools}` by shape hash.
    shapes: HashMap<String, Value>,
    /// Records that matter to pairing, in file order.
    order: Vec<Ordered>,
    /// Assistant content blocks by request id, plus the final usage.
    responses: HashMap<String, Response>,
    /// Tool inputs as sent on the wire, where the stored form was normalized.
    wire_tool_inputs: Map<String, Value>,
}

#[derive(Default)]
struct Response {
    content: Vec<Value>,
    usage: Option<Usage>,
}

enum Ordered {
    Request(Box<WireRequest>),
    Response(String),
}

struct WireRequest {
    id: String,
    base: Option<String>,
    keep: usize,
    tail: Vec<String>,
    shape: Option<String>,
    at_millis: i64,
    model: Option<Value>,
}

fn load_file(path: &Path, summary: &mut LoadSummary) -> Result<Session, BenchError> {
    summary.files += 1;
    let raw = read_records(path)?;
    let (mut messages, order, shapes, responses) = raw.into_resolved();

    summary.calls_seen += responses.values().filter(|r| r.usage.is_some()).count();
    let has_requests = order.iter().any(|o| matches!(o, Ordered::Request(_)));
    if !has_requests {
        if !responses.is_empty() {
            summary.files_without_wire_records += 1;
        }
        return Ok(Session {
            path: path.to_owned(),
            calls: Vec::new(),
        });
    }

    let mut calls = Vec::new();
    let mut histories: HashMap<String, Vec<String>> = HashMap::new();
    let mut pending: Option<Box<WireRequest>> = None;
    let mut paired: HashSet<String> = HashSet::new();
    for item in order {
        match item {
            Ordered::Request(request) => {
                let mut hashes = request
                    .base
                    .as_ref()
                    .and_then(|base| histories.get(base))
                    .map_or_else(Vec::new, |base| {
                        base[..request.keep.min(base.len())].to_vec()
                    });
                hashes.extend(request.tail.iter().cloned());
                histories.insert(request.id.clone(), hashes);
                pending = Some(request);
            }
            Ordered::Response(id) => {
                if !paired.insert(id.clone()) {
                    continue;
                }
                let Some(request) = pending.take() else {
                    continue;
                };
                let Some(usage) = responses.get(&id).and_then(|response| response.usage) else {
                    continue;
                };
                let hashes = &histories[&request.id];
                let built = build_call(&request, hashes, usage, &mut messages, &shapes, summary);
                calls.extend(built);
            }
        }
    }
    Ok(Session {
        path: path.to_owned(),
        calls,
    })
}

fn build_call(
    request: &WireRequest,
    hashes: &[String],
    usage: Usage,
    messages: &mut HashMap<String, Arc<Value>>,
    shapes: &HashMap<String, Value>,
    summary: &mut LoadSummary,
) -> Option<Call> {
    let resolved: Vec<Option<Arc<Value>>> =
        hashes.iter().map(|h| messages.get(h).cloned()).collect();
    let missing = resolved.iter().filter(|m| m.is_none()).count();
    summary.message_refs += hashes.len();
    summary.message_refs_unresolved += missing;
    let shape = request.shape.as_ref().and_then(|hash| shapes.get(hash));
    let too_many_missing = missing as f64 > MAX_UNRESOLVED_FRACTION * hashes.len() as f64;
    let Some(shape) = shape.filter(|_| !too_many_missing) else {
        summary.calls_unresolved += 1;
        return None;
    };

    let mut fields = Map::new();
    if let Some(model) = &request.model {
        fields.insert("model".to_owned(), model.clone());
    }
    for key in ["system", "tools"] {
        if let Some(value) = shape.get(key) {
            fields.insert(key.to_owned(), value.clone());
        }
    }
    let placeholder = Arc::new(json!({ "role": "unresolved", "content": [] }));
    let wire: Vec<Arc<Value>> = resolved
        .into_iter()
        .map(|message| message.unwrap_or_else(|| Arc::clone(&placeholder)))
        .collect();
    Some(Call {
        at_millis: request.at_millis,
        request: Request::from_parts(fields, wire),
        usage,
    })
}

fn read_records(path: &Path) -> Result<RawTranscript, BenchError> {
    let read_error = |source| BenchError::Read {
        path: path.to_owned(),
        source,
    };
    let reader = BufReader::new(File::open(path).map_err(read_error)?);
    let mut raw = RawTranscript::default();
    let mut user_run: Vec<Value> = Vec::new();
    for line in reader.lines() {
        // A session interrupted mid-write can end in a truncated or undecodable line.
        let Ok(line) = line else { continue };
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        raw.add_record(record, &mut user_run);
    }
    Ok(raw)
}

impl RawTranscript {
    fn add_record(&mut self, mut record: Value, user_run: &mut Vec<Value>) {
        if let Some(Value::Object(inputs)) = record.get_mut("wireToolInputs").map(Value::take) {
            self.wire_tool_inputs.extend(inputs);
        }
        match record.get("type").and_then(Value::as_str) {
            Some("api-request-blob") => {
                let hash = record
                    .get("hash")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let (Some(hash), Some(mut message)) =
                    (hash, record.get_mut("message").map(Value::take))
                {
                    blank_image_data(&mut message);
                    self.messages.insert(hash, Arc::new(message));
                }
            }
            Some("api-request-shape") => {
                let hash = record
                    .get("shapeHash")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let (Some(hash), Some(shape)) = (hash, record.get_mut("shape").map(Value::take))
                {
                    self.shapes.insert(hash, shape);
                }
            }
            Some("api-request") => {
                if let Some(request) = wire_request(&record) {
                    self.order.push(Ordered::Request(Box::new(request)));
                }
            }
            Some("assistant") => {
                user_run.clear();
                self.add_assistant(&mut record);
            }
            Some("user") => self.add_user(&record, user_run),
            _ => {}
        }
    }

    fn add_assistant(&mut self, record: &mut Value) {
        let Some(id) = record
            .get("requestId")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            return;
        };
        let usage = record.pointer("/message/usage").map(Usage::from_json);
        let Some(Value::Array(blocks)) = record.pointer_mut("/message/content").map(Value::take)
        else {
            return;
        };
        if !self.responses.contains_key(&id) {
            self.order.push(Ordered::Response(id.clone()));
        }
        let response = self.responses.entry(id).or_default();
        response.content.extend(blocks);
        if usage.is_some() {
            response.usage = usage;
        }
    }

    /// Registers this record alone and merged with the `user` records before it, since
    /// Claude Code stores one record per tool result but sends them as one message.
    fn add_user(&mut self, record: &Value, user_run: &mut Vec<Value>) {
        let Some(content) = record.pointer("/message/content") else {
            return;
        };
        self.register_message(json!({ "role": "user", "content": content }));
        match content {
            Value::Array(blocks) => user_run.extend(blocks.iter().cloned()),
            text => user_run.push(json!({ "type": "text", "text": text })),
        }
        self.register_message(json!({ "role": "user", "content": user_run }));
    }

    fn register_message(&mut self, mut message: Value) {
        let hash = content_hash(&message);
        self.messages.entry(hash).or_insert_with(|| {
            blank_image_data(&mut message);
            Arc::new(message)
        });
    }

    /// Registers an assistant message under its wire hash, annotated with how many hidden
    /// thinking tokens it holds. Only the bench reads the annotation, and fingerprints
    /// look at `role` and `content` alone, so it can't affect a comparison.
    fn register_assistant(&mut self, mut message: Value, thinking_tokens: u64) {
        let hash = content_hash(&message);
        blank_image_data(&mut message);
        if thinking_tokens > 0 {
            message[THINKING_TOKENS_KEY] = json!(thinking_tokens);
        }
        self.messages
            .entry(hash)
            .or_insert_with(|| Arc::new(message));
    }

    #[expect(clippy::type_complexity, reason = "private destructuring helper")]
    fn into_resolved(
        mut self,
    ) -> (
        HashMap<String, Arc<Value>>,
        Vec<Ordered>,
        HashMap<String, Value>,
        HashMap<String, Response>,
    ) {
        let assembled: Vec<(Vec<Value>, u64)> = self
            .responses
            .values()
            .map(|response| {
                let thinking = response.usage.map_or(0, |usage| usage.thinking);
                (response.content.clone(), thinking)
            })
            .collect();
        for (content, thinking) in assembled {
            let as_stored = json!({ "role": "assistant", "content": content });
            let as_sent = with_wire_inputs(&as_stored, &self.wire_tool_inputs);
            if as_sent != as_stored {
                self.register_assistant(as_stored, thinking);
            }
            self.register_assistant(as_sent, thinking);
        }
        (self.messages, self.order, self.shapes, self.responses)
    }
}

fn with_wire_inputs(message: &Value, inputs: &Map<String, Value>) -> Value {
    let mut message = message.clone();
    if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
        for block in blocks {
            let wire = block
                .get("id")
                .and_then(Value::as_str)
                .and_then(|id| inputs.get(id))
                .cloned();
            if let (Some(wire), Some(input)) = (wire, block.get_mut("input")) {
                *input = wire;
            }
        }
    }
    message
}

fn wire_request(record: &Value) -> Option<WireRequest> {
    let tail = record.get("tail")?.as_array()?;
    Some(WireRequest {
        id: record.get("id")?.as_str()?.to_owned(),
        base: record
            .get("base")
            .and_then(Value::as_str)
            .map(str::to_owned),
        keep: record
            .get("keep")
            .and_then(Value::as_u64)
            .and_then(|keep| usize::try_from(keep).ok())
            .unwrap_or(0),
        tail: tail
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        shape: record
            .get("shapeHash")
            .and_then(Value::as_str)
            .map(str::to_owned),
        at_millis: record
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_millis)
            .unwrap_or(0),
        model: record.pointer("/params/model").cloned(),
    })
}

/// Hashes a message the way Claude Code does: SHA-256 of compact, key-sorted JSON.
fn content_hash(message: &Value) -> String {
    let mut canonical = Vec::new();
    write_canonical(message, &mut canonical);
    let digest = Sha256::digest(&canonical);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).expect("a nibble is a hex digit"));
        hex.push(char::from_digit(u32::from(byte & 15), 16).expect("a nibble is a hex digit"));
    }
    hex
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            out.push(b'{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_scalar(key, out);
                out.push(b':');
                write_canonical(item, out);
            }
            out.push(b'}');
        }
        scalar => write_scalar(scalar, out),
    }
}

fn write_scalar(value: &(impl serde::Serialize + ?Sized), out: &mut Vec<u8>) {
    serde_json::to_writer(out, value).expect("json values always serialize to a vec");
}

/// Replaces base64 image payloads with an empty string. Images dominate transcript
/// bytes but only their count affects token estimates.
fn blank_image_data(value: &mut Value) {
    match value {
        Value::Array(items) => items.iter_mut().for_each(blank_image_data),
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("image")
                && let Some(data) = map.get_mut("source").and_then(|s| s.get_mut("data"))
            {
                *data = Value::String(String::new());
            }
            map.values_mut().for_each(blank_image_data);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_transcript(lines: &[Value]) -> (PathBuf, PathBuf) {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ccc-transcript-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.jsonl");
        let mut file = File::create(&path).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        writeln!(file, "{{truncated").unwrap();
        (dir, path)
    }

    fn assistant(request_id: &str, block: Value, usage: Option<Value>) -> Value {
        let mut message = json!({ "role": "assistant", "content": [block] });
        if let Some(usage) = usage {
            message["usage"] = usage;
        }
        json!({ "type": "assistant", "requestId": request_id, "message": message })
    }

    #[test]
    fn content_hash_sorts_keys_at_every_depth() {
        let a = json!({ "b": 1, "a": { "y": 1, "x": [ { "q": 1, "p": 2 } ] } });
        let b = json!({ "a": { "x": [ { "p": 2, "q": 1 } ], "y": 1 }, "b": 1 });
        assert_eq!(content_hash(&a), content_hash(&b));
        assert_ne!(content_hash(&a), content_hash(&json!({ "b": 1 })));
    }

    #[test]
    fn content_hash_matches_a_digest_computed_independently() {
        // python3: hashlib.sha256(json.dumps(m, ensure_ascii=False,
        //   separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        let message = json!({ "role": "user", "content": "é" });
        assert_eq!(
            content_hash(&message),
            "e0db42f8f45e84d5254bd009b6846314445a94ad17346d740ca48c464aa92fbd"
        );
    }

    #[test]
    fn rebuilds_requests_from_blobs_merged_records_and_prior_history() {
        let first = json!({ "role": "user", "content": "hi" });
        let first_hash = content_hash(&first);
        let tool_a = json!({ "type": "tool_result", "tool_use_id": "a", "content": "A" });
        let tool_b = json!({ "type": "tool_result", "tool_use_id": "b", "content": "B" });
        let merged = json!({ "role": "user", "content": [tool_a, tool_b] });
        let wire_answer = json!({
            "role": "assistant",
            "content": [{ "type": "tool_use", "id": "a", "name": "Bash", "input": { "command": "wire" } }],
        });
        let usage = json!({ "input_tokens": 1, "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5, "output_tokens": 2,
            "cache_creation": { "ephemeral_1h_input_tokens": 4 } });

        let lines = [
            json!({ "type": "api-request-shape", "shapeHash": "s", "shape": { "system": [], "tools": [] } }),
            json!({ "type": "api-request-blob", "hash": first_hash, "message": first }),
            json!({ "type": "api-request", "id": "r1", "keep": 0, "tail": [first_hash],
                    "shapeHash": "s", "timestamp": "2026-01-01T00:00:00.000Z",
                    "params": { "model": "claude-opus-5-5" } }),
            assistant(
                "q1",
                json!({ "type": "tool_use", "id": "a", "name": "Bash", "input": { "command": "stored" } }),
                Some(usage),
            ),
            json!({ "type": "user", "message": { "role": "user", "content": [tool_a] },
                    "wireToolInputs": { "a": { "command": "wire" } } }),
            json!({ "type": "user", "message": { "role": "user", "content": [tool_b] } }),
            json!({ "type": "api-request", "id": "r2", "base": "r1", "keep": 1,
                    "tail": [content_hash(&wire_answer), content_hash(&merged)],
                    "shapeHash": "s", "timestamp": "2026-01-01T00:00:01.000Z",
                    "params": { "model": "claude-opus-5-5" } }),
            assistant(
                "q2",
                json!({ "type": "text", "text": "done" }),
                Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
            ),
        ];
        let (dir, _) = write_transcript(&lines);

        let (sessions, summary) = load_dir(&dir).unwrap();

        assert_eq!(summary.message_refs_unresolved, 0, "{summary:?}");
        assert_eq!(sessions.len(), 1);
        let calls = &sessions[0].calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].usage.cache_write_1h, 4);
        assert_eq!(calls[0].usage.cache_write_5m, 1);
        assert_eq!(calls[0].usage.context(), 16);
        let messages = calls[1].request.messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"][0]["input"]["command"], "wire");
        assert_eq!(messages[2]["content"][1]["tool_use_id"], "b");
        assert_eq!(calls[1].model(), "claude-opus-5-5");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn counts_files_without_wire_records_and_drops_unresolvable_calls() {
        let old = [assistant(
            "q",
            json!({ "type": "text", "text": "x" }),
            Some(json!({ "input_tokens": 1 })),
        )];
        let (dir, _) = write_transcript(&old);
        let (sessions, summary) = load_dir(&dir).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(summary.files_without_wire_records, 1);
        assert_eq!(summary.calls_seen, 1);
        fs::remove_dir_all(dir).unwrap();

        let unresolved = [
            json!({ "type": "api-request-shape", "shapeHash": "s", "shape": {} }),
            json!({ "type": "api-request", "id": "r", "keep": 0, "tail": ["nope"], "shapeHash": "s",
                    "timestamp": "2026-01-01T00:00:00.000Z" }),
            assistant(
                "q",
                json!({ "type": "text", "text": "x" }),
                Some(json!({ "input_tokens": 1 })),
            ),
        ];
        let (dir, _) = write_transcript(&unresolved);
        let (sessions, summary) = load_dir(&dir).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(summary.calls_unresolved, 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn image_data_is_blanked_but_the_block_stays_countable() {
        let mut message = json!({ "content": [{ "type": "tool_result", "content": [
            { "type": "image", "source": { "type": "base64", "data": "AAAA" } }
        ] }] });
        blank_image_data(&mut message);
        assert_eq!(message["content"][0]["content"][0]["source"]["data"], "");
        assert_eq!(message["content"][0]["content"][0]["type"], "image");
    }
}

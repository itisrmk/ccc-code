//! Sends the scripted conversation through a gateway and judges what came out each side.

use std::fmt;
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::script::{self, BINDING_BETA};
use crate::upstream::{FakeUpstream, Recorded, STREAM_PAUSE, Scripted};

/// How long the first streamed bytes may trail a stream's start before it counts as buffered.
const BUFFERING_MARGIN: Duration = STREAM_PAUSE;
/// Larger than most real requests, and small enough to send quickly.
const LARGE_REQUEST_BYTES: usize = 6 * 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How to reach the gateway.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// The gateway's base URL.
    pub gateway: String,
    /// Extra headers to send it, such as its own API key.
    pub headers: Vec<(String, String)>,
}

/// What stopped the checks from running at all.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// A header given on the command line isn't valid.
    #[error("invalid header {0:?}")]
    InvalidHeader(String),
    /// The HTTP client couldn't be set up.
    #[error("setting up the client")]
    Client(#[source] reqwest::Error),
}

/// One check's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Behaves as Claude Code needs.
    Pass,
    /// Will break conversations or restart the cache.
    Fail(String),
    /// Costs something or could break later, but won't break a conversation today.
    Warn(String),
    /// Worth knowing, and not a problem.
    Note(String),
}

/// A named check and how it went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// Short stable name.
    pub name: &'static str,
    /// What it found.
    pub outcome: Outcome,
}

/// Everything found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// Checks in the order they ran.
    pub checks: Vec<Check>,
}

impl Report {
    /// Whether nothing failed.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self
            .checks
            .iter()
            .any(|check| matches!(check.outcome, Outcome::Fail(_)))
    }

    /// The result of the check called `name`.
    #[must_use]
    pub fn outcome(&self, name: &str) -> Option<&Outcome> {
        self.checks
            .iter()
            .find(|check| check.name == name)
            .map(|check| &check.outcome)
    }

    /// The report as JSON.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let checks: Vec<Value> = self
            .checks
            .iter()
            .map(|check| {
                let (status, detail) = match &check.outcome {
                    Outcome::Pass => ("pass", ""),
                    Outcome::Fail(detail) => ("fail", detail.as_str()),
                    Outcome::Warn(detail) => ("warn", detail.as_str()),
                    Outcome::Note(detail) => ("note", detail.as_str()),
                };
                json!({ "name": check.name, "status": status, "detail": detail })
            })
            .collect();
        json!({ "passed": self.passed(), "checks": checks })
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for check in &self.checks {
            let (status, detail) = match &check.outcome {
                Outcome::Pass => ("pass", ""),
                Outcome::Fail(detail) => ("FAIL", detail.as_str()),
                Outcome::Warn(detail) => ("warn", detail.as_str()),
                Outcome::Note(detail) => ("note", detail.as_str()),
            };
            writeln!(f, "{status}  {:<20} {detail}", check.name)?;
        }
        let verdict = if self.passed() {
            "Safe for what these checks cover."
        } else {
            "Not safe: expect 400 errors on newer accounts, restarted caches, or both."
        };
        write!(f, "{verdict}")
    }
}

struct Answer {
    status: u16,
    headers: HeaderMap,
    body: Vec<u8>,
    first_byte: Duration,
    total: Duration,
}

struct Driver<'a> {
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    upstream: &'a FakeUpstream,
}

impl Driver<'_> {
    /// Sends one request. A connection failure comes back as text, since whether the gateway
    /// survives each request is part of what's being checked.
    async fn send(&self, body: &Value) -> Result<Answer, String> {
        let started = Instant::now();
        let response = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .body(body.to_string())
            .send()
            .await
            .map_err(|err| format!("no response: {err}"))?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let mut stream = response.bytes_stream();
        let mut received = Vec::new();
        let mut first_byte = None;
        while let Some(part) = stream.next().await {
            let part = part.map_err(|err| format!("response cut off: {err}"))?;
            first_byte.get_or_insert_with(|| started.elapsed());
            received.extend_from_slice(&part);
        }
        Ok(Answer {
            status,
            headers,
            body: received,
            first_byte: first_byte.unwrap_or_else(|| started.elapsed()),
            total: started.elapsed(),
        })
    }

    /// Sends a request, and returns what reached the upstream because of it.
    async fn forwarded(&self, body: &Value) -> Result<(Answer, Recorded), String> {
        let before = self.upstream.count();
        let answer = self.send(body).await?;
        if self.upstream.count() == before {
            return Err(format!(
                "nothing reached the upstream; the gateway answered {}",
                answer.status
            ));
        }
        let recorded = self
            .upstream
            .latest()
            .ok_or_else(|| "nothing reached the upstream".to_owned())?;
        Ok((answer, recorded))
    }
}

/// Removes cache markers, which legitimately move as a conversation grows.
fn without_cache_markers(value: &Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|(key, _)| *key != "cache_control")
                .map(|(key, value)| (key.clone(), without_cache_markers(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(without_cache_markers).collect()),
        other => other.clone(),
    }
}

fn parse(recorded: &Recorded) -> Result<Value, String> {
    serde_json::from_slice(&recorded.body)
        .map_err(|err| format!("the upstream received invalid JSON: {err}"))
}

fn messages(body: &Value) -> Vec<Value> {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(without_cache_markers).collect())
        .unwrap_or_default()
}

fn thinking_blocks(body: &Value) -> Vec<Value> {
    messages(body)
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|block| {
            matches!(
                block.get("type").and_then(Value::as_str),
                Some("thinking" | "redacted_thinking")
            )
        })
        .cloned()
        .collect()
}

fn block_types(body: &Value) -> Vec<Vec<String>> {
    messages(body)
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .map(|blocks| {
            blocks
                .iter()
                .map(|block| {
                    block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("?")
                        .to_owned()
                })
                .collect()
        })
        .collect()
}

fn pass_if(condition: bool, failure: impl FnOnce() -> String) -> Outcome {
    if condition {
        Outcome::Pass
    } else {
        Outcome::Fail(failure())
    }
}

/// Runs every check against the gateway at `options.gateway`, which must already be forwarding
/// to `upstream`.
///
/// # Errors
///
/// Returns an error only if the checks can't start. A gateway that misbehaves, or can't be
/// reached, is a failed check in the report.
pub async fn run(upstream: &FakeUpstream, options: &VerifyOptions) -> Result<Report, VerifyError> {
    let mut headers = HeaderMap::new();
    for (name, value) in [
        ("content-type", "application/json"),
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", BINDING_BETA),
        ("x-api-key", "ccc-verify-not-a-real-key"),
    ]
    .iter()
    .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
    .chain(options.headers.iter().cloned())
    {
        let invalid = || VerifyError::InvalidHeader(name.clone());
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid())?,
            HeaderValue::from_str(&value).map_err(|_| invalid())?,
        );
    }
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(VerifyError::Client)?;
    let driver = Driver {
        client,
        url: format!("{}/v1/messages", options.gateway.trim_end_matches('/')),
        headers,
        upstream,
    };

    let mut report = Report::default();
    let turns = match conversation(&driver, &mut report).await {
        Ok(turns) => turns,
        Err(detail) => {
            report.checks.push(Check {
                name: "forwarding",
                outcome: Outcome::Fail(detail),
            });
            return Ok(report);
        }
    };
    report.checks.insert(
        0,
        Check {
            name: "forwarding",
            outcome: Outcome::Pass,
        },
    );
    judge_requests(&turns, &mut report);
    retry(&driver, &turns, &mut report).await;
    streaming(&driver, &mut report).await;
    errors(&driver, &mut report).await;
    large(&driver, &mut report).await;
    Ok(report)
}

struct Turn {
    sent: Value,
    received: Value,
    headers: axum::http::HeaderMap,
}

async fn conversation(driver: &Driver<'_>, report: &mut Report) -> Result<Vec<Turn>, String> {
    driver.upstream.script(Scripted::Whole);
    let mut turns = Vec::new();
    let mut last_answer = None;
    for turn in 0..script::TURNS {
        let sent = script::request(turn, false);
        let (answer, recorded) = driver.forwarded(&sent).await?;
        if answer.status != 200 {
            return Err(format!(
                "turn {turn}: the gateway answered {}: {}",
                answer.status,
                String::from_utf8_lossy(&answer.body)
                    .chars()
                    .take(200)
                    .collect::<String>()
            ));
        }
        turns.push(Turn {
            sent,
            received: parse(&recorded)?,
            headers: recorded.headers,
        });
        last_answer = Some(answer);
    }
    let whole = last_answer
        .and_then(|answer| serde_json::from_slice::<Value>(&answer.body).ok())
        .unwrap_or(Value::Null);
    let expected = script::response_body();
    report.checks.push(Check {
        name: "response-whole",
        outcome: pass_if(whole.get("content") == expected.get("content"), || {
            "a whole response lost or changed content, such as a thinking signature".to_owned()
        }),
    });
    report.checks.push(Check {
        name: "response-usage",
        outcome: if whole.get("usage") == expected.get("usage") {
            Outcome::Pass
        } else {
            Outcome::Warn("usage changed, so cost tracking downstream will be off".to_owned())
        },
    });
    Ok(turns)
}

fn judge_requests(turns: &[Turn], report: &mut Report) {
    let mut check = |name, outcome| report.checks.push(Check { name, outcome });

    let lost = turns
        .iter()
        .find(|turn| thinking_blocks(&turn.sent) != thinking_blocks(&turn.received));
    check(
        "thinking-intact",
        pass_if(lost.is_none(), || {
            "thinking blocks or signatures were changed, dropped or reordered".to_owned()
        }),
    );

    let reshaped = turns
        .iter()
        .find(|turn| block_types(&turn.sent) != block_types(&turn.received));
    check(
        "block-order",
        pass_if(reshaped.is_none(), || {
            "messages or content blocks were merged, split, dropped or reordered".to_owned()
        }),
    );

    let edited = turns.windows(2).find_map(|pair| {
        let (before, after) = (messages(&pair[0].received), messages(&pair[1].received));
        let changed = before.iter().zip(&after).position(|(old, new)| old != new);
        match (changed, before.len() > after.len()) {
            (Some(index), _) => Some(format!("message {index} changed on a later turn")),
            (None, true) => Some("earlier messages disappeared on a later turn".to_owned()),
            (None, false) => None,
        }
    });
    check(
        "append-only",
        match edited {
            Some(detail) => Outcome::Fail(format!(
                "{detail}. History behind a thinking block must never change"
            )),
            None => Outcome::Pass,
        },
    );

    for (name, field) in [("system-stable", "system"), ("tools-stable", "tools")] {
        let parts: Vec<Value> = turns
            .iter()
            .map(|turn| {
                turn.received
                    .get(field)
                    .map(without_cache_markers)
                    .unwrap_or(Value::Null)
            })
            .collect();
        check(
            name,
            pass_if(parts.windows(2).all(|pair| pair[0] == pair[1]), || {
                format!("{field} changed between turns, which invalidates thinking and the cache")
            }),
        );
    }

    let models: Vec<Option<&str>> = turns
        .iter()
        .map(|turn| turn.received.get("model").and_then(Value::as_str))
        .collect();
    let model_stable = models.windows(2).all(|pair| pair[0] == pair[1]);
    check(
        "model-stable",
        if !model_stable {
            Outcome::Fail("the model changed partway through a conversation".to_owned())
        } else if models.first().copied().flatten() != Some(script::MODEL) {
            Outcome::Note(format!(
                "the model is rewritten to {:?}, consistently",
                models.first().copied().flatten().unwrap_or("nothing")
            ))
        } else {
            Outcome::Pass
        },
    );

    let thinking_kept = turns
        .iter()
        .all(|turn| turn.sent.get("thinking") == turn.received.get("thinking"));
    check(
        "thinking-config",
        pass_if(thinking_kept, || {
            "the thinking setting was changed or removed".to_owned()
        }),
    );

    let beta_kept = turns.iter().all(|turn| {
        turn.headers
            .get_all("anthropic-beta")
            .iter()
            .any(|value| value.to_str().is_ok_and(|text| text.contains(BINDING_BETA)))
    });
    check(
        "beta-header",
        pass_if(beta_kept, || {
            "anthropic-beta was dropped, and Claude Code relies on the features it enables"
                .to_owned()
        }),
    );

    let future_kept = turns
        .iter()
        .all(|turn| turn.sent.get("x_future_field") == turn.received.get("x_future_field"));
    check(
        "unknown-fields",
        if future_kept {
            Outcome::Pass
        } else {
            Outcome::Warn(
                "unknown request fields are dropped, so new API features may silently stop working"
                    .to_owned(),
            )
        },
    );

    let edited_on_arrival = turns
        .iter()
        .filter(|turn| messages(&turn.sent) != messages(&turn.received))
        .count();
    if edited_on_arrival > 0 {
        check(
            "edits-on-arrival",
            Outcome::Note(format!(
                "new content was changed on {edited_on_arrival} of {} turns, which is allowed \
                 while it stays the same on later turns",
                turns.len()
            )),
        );
    }
}

async fn retry(driver: &Driver<'_>, turns: &[Turn], report: &mut Report) {
    let Some(last) = turns.last() else { return };
    let outcome = match driver.forwarded(&last.sent).await {
        Ok((_, recorded)) => match parse(&recorded) {
            Ok(received) => pass_if(received == last.received, || {
                "sending the same request twice forwarded different bytes, so an output depends \
                 on something other than the request"
                    .to_owned()
            }),
            Err(detail) => Outcome::Fail(detail),
        },
        Err(detail) => Outcome::Fail(detail),
    };
    report.checks.push(Check {
        name: "repeatable",
        outcome,
    });
}

fn stream_events(body: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(body)
        .split("\n\n")
        .filter_map(|event| {
            let data = event
                .lines()
                .find_map(|line| line.strip_prefix("data:"))?
                .trim();
            serde_json::from_str(data).ok()
        })
        .collect()
}

fn delta_values(events: &[Value], kind: &str, field: &str) -> String {
    events
        .iter()
        .filter(|event| event.pointer("/delta/type").and_then(Value::as_str) == Some(kind))
        .filter_map(|event| {
            event
                .pointer(&format!("/delta/{field}"))
                .and_then(Value::as_str)
        })
        .collect()
}

async fn streaming(driver: &Driver<'_>, report: &mut Report) {
    driver.upstream.script(Scripted::Stream);
    let answer = driver.send(&script::request(script::TURNS - 1, true)).await;
    let mut check = |name, outcome| report.checks.push(Check { name, outcome });
    let answer = match answer {
        Ok(answer) if answer.status == 200 => answer,
        Ok(answer) => {
            let detail = format!("a streaming request got {}", answer.status);
            check("stream-thinking", Outcome::Fail(detail));
            return;
        }
        Err(detail) => {
            check("stream-thinking", Outcome::Fail(detail));
            return;
        }
    };
    let events = stream_events(&answer.body);
    let intact = delta_values(&events, "thinking_delta", "thinking") == script::STREAM_THINKING
        && delta_values(&events, "signature_delta", "signature") == script::STREAM_SIGNATURE;
    check(
        "stream-thinking",
        pass_if(intact, || {
            "streamed thinking or its signature was changed or dropped, so the next turn can't \
             send it back"
                .to_owned()
        }),
    );
    check(
        "stream-text",
        pass_if(
            delta_values(&events, "text_delta", "text") == script::STREAM_TEXT,
            || "streamed text was changed or dropped".to_owned(),
        ),
    );
    let started = events
        .iter()
        .find(|event| event.get("type").and_then(Value::as_str) == Some("message_start"));
    let cached = started
        .and_then(|event| event.pointer("/message/usage/cache_read_input_tokens"))
        .and_then(Value::as_u64);
    check(
        "stream-usage",
        if cached == Some(script::CACHE_READ_TOKENS) {
            Outcome::Pass
        } else {
            Outcome::Warn("cache usage in a stream was changed or dropped".to_owned())
        },
    );
    let future_kept = events
        .iter()
        .any(|event| event.get("type").and_then(Value::as_str) == Some("future_event_type"));
    check(
        "stream-unknown",
        if future_kept {
            Outcome::Pass
        } else {
            Outcome::Warn(
                "unknown stream events are dropped, so new API features may stop working"
                    .to_owned(),
            )
        },
    );
    let live = answer.first_byte + BUFFERING_MARGIN < answer.total;
    check(
        "stream-live",
        if live {
            Outcome::Pass
        } else {
            Outcome::Warn(format!(
                "the first bytes took {} ms of {} ms, so the stream looks buffered and long \
                 answers will seem frozen",
                answer.first_byte.as_millis(),
                answer.total.as_millis()
            ))
        },
    );
}

async fn errors(driver: &Driver<'_>, report: &mut Report) {
    let body = script::request(script::TURNS - 1, false);
    driver.upstream.script(Scripted::Rejected);
    let rejected = driver.send(&body).await;
    let outcome = match rejected {
        Ok(answer) => pass_if(
            answer.status == 400
                && String::from_utf8_lossy(&answer.body).contains(script::ERROR_MESSAGE),
            || {
                format!(
                    "a 400 came back as {}, or lost its message. Claude Code recovers from a \
                     thinking rejection by reading that error",
                    answer.status
                )
            },
        ),
        Err(detail) => Outcome::Fail(detail),
    };
    report.checks.push(Check {
        name: "error-400",
        outcome,
    });

    driver.upstream.script(Scripted::RateLimited);
    let limited = driver.send(&body).await;
    let outcome = match limited {
        Ok(answer) if answer.status != 429 => {
            Outcome::Fail(format!("a 429 came back as {}", answer.status))
        }
        Ok(answer) => {
            let waited = answer
                .headers
                .get("retry-after")
                .and_then(|value| value.to_str().ok());
            if waited == Some(script::RETRY_AFTER_SECONDS) {
                Outcome::Pass
            } else {
                Outcome::Warn("a 429 lost its retry-after header".to_owned())
            }
        }
        Err(detail) => Outcome::Fail(detail),
    };
    report.checks.push(Check {
        name: "error-429",
        outcome,
    });
    driver.upstream.script(Scripted::Whole);
}

async fn large(driver: &Driver<'_>, report: &mut Report) {
    let answer = driver
        .send(&script::large_request(LARGE_REQUEST_BYTES))
        .await;
    let limit = LARGE_REQUEST_BYTES / (1024 * 1024);
    let outcome = match answer {
        Ok(answer) if answer.status == 200 => Outcome::Pass,
        Ok(answer) => Outcome::Warn(format!(
            "a {limit} MB request got {}, and long sessions reach that size",
            answer.status
        )),
        Err(detail) => Outcome::Warn(format!("a {limit} MB request failed: {detail}")),
    };
    report.checks.push(Check {
        name: "large-request",
        outcome,
    });
}

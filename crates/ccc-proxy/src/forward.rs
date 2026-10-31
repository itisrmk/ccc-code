//! The forwarding path: one handler for every method and route.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::body::{Bytes, to_bytes};
use axum::extract::{Request, State as AxumState};
use axum::http::header::{self, HeaderMap, HeaderName, HeaderValue};
use axum::http::request::Parts;
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use ccc_ledger::Ledger;
use ccc_pipeline::{Compiled, Pipeline, StageOutcome};
use ccc_redact::Vault;
use ccc_route::{LOCAL_PREFIX, budget_notice};
use futures_util::TryStreamExt;

use crate::Stats;
use crate::conversations::Conversations;
use crate::dump::{Dumper, Exchange};
use crate::response::{Accounting, NoticedSessions, ResponsePlan, processed_body};

/// Largest request body the proxy will buffer. Claude Code bodies can carry many
/// base64 images on top of a 1M-token context, so this sits well above that.
const MAX_REQUEST_BYTES: usize = 1 << 30;

/// Paths whose bodies are Messages API requests.
const COMPILED_PATHS: [&str; 2] = ["/v1/messages", "/v1/messages/count_tokens"];

/// The name of the stage that puts placeholders into requests.
const REDACT_STAGE: &str = "redact";

const ANTHROPIC_BETA: &str = "anthropic-beta";
/// The beta that lets a request set `thinking.block_binding`.
const THINKING_CHECK_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The header Claude Code uses to name its session.
const SESSION_HEADER: &str = "x-claude-code-session-id";

/// Connection-scoped headers that must not cross a proxy (RFC 9110 section 7.6.1).
const HOP_BY_HOP: [HeaderName; 7] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

#[derive(Debug)]
pub(crate) struct State {
    pub(crate) upstream: String,
    pub(crate) client: reqwest::Client,
    pub(crate) dumper: Option<Dumper>,
    pub(crate) pipeline: Pipeline,
    pub(crate) conversations: Conversations,
    /// A stage forbids sending what the compiler can't read, so unreadable bodies stop here.
    pub(crate) must_inspect: bool,
    pub(crate) vault: Option<Arc<Vault>>,
    pub(crate) enforce_thinking_check: bool,
    pub(crate) ledger: Option<Arc<Ledger>>,
    pub(crate) budget_usd: Option<f64>,
    pub(crate) local_url: Option<String>,
    pub(crate) noticed: Arc<NoticedSessions>,
    pub(crate) stats: Arc<Stats>,
}

pub(crate) async fn handle(AxumState(state): AxumState<Arc<State>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, MAX_REQUEST_BYTES).await {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!(error = %err, "failed reading request body");
            return error_response(StatusCode::BAD_REQUEST, "could not read request body");
        }
    };

    let exchange = state.dumper.as_ref().map(|dumper| {
        dumper.record_request(parts.method.as_str(), &parts.uri, &parts.headers, &body)
    });

    let is_compiled = is_compiled_request(&parts);
    let outcome = if is_compiled {
        match compile_body(&state, &parts, body, exchange.as_ref()).await {
            Ok(outcome) => outcome,
            Err(response) => return *response,
        }
    } else {
        CompileOutcome::unchanged(body)
    };
    let CompileOutcome {
        body,
        local_body,
        plan,
    } = outcome;

    let mut request_headers = forwardable_headers(&parts.headers);
    if is_compiled {
        // Compressed event streams can't be read or rewritten on the way through.
        request_headers.insert(
            header::ACCEPT_ENCODING,
            HeaderValue::from_static("identity"),
        );
        if state.enforce_thinking_check {
            add_beta(&mut request_headers, THINKING_CHECK_BETA);
        }
    }

    let local_response = match local_body {
        Some(local_body) => send_to_local(&state, &parts, &request_headers, local_body).await,
        None => None,
    };
    let upstream_response = match local_response {
        Some(response) => response,
        None => {
            let upstream_url = upstream_url(&state.upstream, &parts.uri);
            let sent = state
                .client
                .request(parts.method, &upstream_url)
                .headers(request_headers)
                .body(body)
                .send()
                .await;
            match sent {
                Ok(response) => response,
                Err(err) => {
                    tracing::warn!(error = %err, url = %upstream_url, "upstream request failed");
                    return error_response(StatusCode::BAD_GATEWAY, "ccc could not reach upstream");
                }
            }
        }
    };

    let status = upstream_response.status();
    let headers = forwardable_headers(upstream_response.headers());
    if let Some(exchange) = &exchange {
        exchange.record_response_head(status.as_u16(), &headers);
    }

    let stream = upstream_response.bytes_stream().inspect_ok(move |chunk| {
        if let Some(exchange) = &exchange {
            exchange.record_response_chunk(chunk.clone());
        }
    });

    let mut response = Response::new(processed_body(stream, &headers, plan).await);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Tries the local model. Any failure falls back to the normal upstream, so a local model
/// that's down, or rejects a request shape, costs a retry rather than the user's turn.
async fn send_to_local(
    state: &State,
    parts: &Parts,
    headers: &HeaderMap,
    body: Bytes,
) -> Option<reqwest::Response> {
    let base = state.local_url.as_ref()?;
    let mut local_headers = headers.clone();
    // Anthropic credentials are for Anthropic.
    for secret in [header::AUTHORIZATION, HeaderName::from_static("x-api-key")] {
        local_headers.remove(secret);
    }
    let url = upstream_url(base, &parts.uri);
    match state
        .client
        .request(parts.method.clone(), &url)
        .headers(local_headers)
        .body(body)
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => Some(response),
        Ok(response) => {
            tracing::warn!(status = %response.status(), "local model refused, using upstream");
            state.stats.local_fallbacks.fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(err) => {
            tracing::warn!(error = %err, "local model unreachable, using upstream");
            state.stats.local_fallbacks.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// What compiling a request decided.
struct CompileOutcome {
    /// The bytes for the normal upstream.
    body: Bytes,
    /// The bytes for the local model, if routing chose it.
    local_body: Option<Bytes>,
    plan: ResponsePlan,
}

impl CompileOutcome {
    fn unchanged(body: Bytes) -> Self {
        Self {
            body,
            local_body: None,
            plan: ResponsePlan::default(),
        }
    }
}

fn is_compiled_request(parts: &Parts) -> bool {
    parts.method == Method::POST && COMPILED_PATHS.contains(&parts.uri.path())
}

/// Runs the pipeline over a request body and decides what the response needs. Forwards the
/// original bytes unless something actually changed.
async fn compile_body(
    state: &State,
    parts: &Parts,
    body: Bytes,
    exchange: Option<&Exchange>,
) -> Result<CompileOutcome, Box<Response>> {
    let refuse_or_forward = |body: Bytes, why: &str| {
        if state.must_inspect {
            tracing::error!(why, "refusing to send a request the compiler can't read");
            state.stats.blocked.fetch_add(1, Ordering::Relaxed);
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!("ccc blocked this request: {why}"),
            )));
        }
        Ok(CompileOutcome::unchanged(body))
    };
    if parts.headers.contains_key(header::CONTENT_ENCODING) {
        return refuse_or_forward(body, "its body is compressed");
    }
    let parsed = match ccc_wire::Request::parse(&body) {
        Ok(parsed) => parsed,
        Err(err) => {
            tracing::debug!(error = %err, "body isn't a messages request");
            return refuse_or_forward(body, "its body isn't a readable messages request");
        }
    };
    let session = parts
        .headers
        .get(SESSION_HEADER)
        .and_then(|value| value.to_str().ok());
    let mut context = state.conversations.context_for(session, &parsed);
    let spent_usd = session
        .zip(state.ledger.as_ref())
        .and_then(|(session, ledger)| ledger.session_cost(session).ok())
        .unwrap_or(0.0);
    // Whole micro-dollars are far finer than any budget, so rounding down is harmless.
    context.spent_micros = (spent_usd * 1_000_000.0) as u64;
    let vault = state
        .vault
        .clone()
        .filter(|_| context.enabled.contains(REDACT_STAGE));
    let original_model = parsed.model().map(str::to_owned);

    let compiled = match state.pipeline.compile(parsed, context).await {
        Ok(compiled) => compiled,
        Err(err) => {
            tracing::error!(error = %err, "request blocked");
            state.stats.blocked.fetch_add(1, Ordering::Relaxed);
            return Err(Box::new(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("ccc blocked this request: {err}"),
            )));
        }
    };
    record_trace(state, &compiled, exchange);

    let changed = compiled.changed();
    let mut request = compiled.request;
    let model = request.model().unwrap_or_default().to_owned();
    let plan = ResponsePlan {
        vault,
        notice: budget_notice_for(state, session, spent_usd),
        accounting: Some(Accounting {
            ledger: state.ledger.clone(),
            session: session.unwrap_or("unknown").to_owned(),
            model: model.clone(),
            stats: Arc::clone(&state.stats),
            noticed: Arc::clone(&state.noticed),
        }),
    };

    let enforcing = state.enforce_thinking_check && enforce_thinking_check(&mut request);

    // A local model gets its own copy. The upstream copy keeps the model the caller asked
    // for, ready as the fallback.
    let local_body = model.strip_prefix(LOCAL_PREFIX).map(|local_model| {
        let mut local = request.clone();
        local.set_in_field("model", "", serde_json::json!(local_model));
        Bytes::from(local.to_bytes())
    });
    if local_body.is_some()
        && let Some(original) = &original_model
    {
        request.set_in_field("model", "", serde_json::json!(original));
    }

    if !changed && !enforcing {
        state.stats.record(body.len(), body.len());
        return Ok(CompileOutcome {
            body,
            local_body: None,
            plan,
        });
    }
    let rewritten = Bytes::from(request.to_bytes());
    state.stats.record(body.len(), rewritten.len());
    if let Some(exchange) = exchange {
        exchange.record_compiled_body(&rewritten);
    }
    Ok(CompileOutcome {
        body: rewritten,
        local_body,
        plan,
    })
}

/// The notice to show, if this session has just gone over budget.
fn budget_notice_for(state: &State, session: Option<&str>, spent_usd: f64) -> Option<String> {
    let budget = state.budget_usd?;
    let session = session?;
    (spent_usd >= budget && !state.noticed.contains(session))
        .then(|| budget_notice(spent_usd, budget))
}

/// Opts a request into the strict history check. Returns whether the request changed;
/// without a `thinking` setting there's nothing to attach the option to.
fn enforce_thinking_check(request: &mut ccc_wire::Request) -> bool {
    let Some(mut thinking) = request.field("thinking").cloned() else {
        return false;
    };
    let Some(settings) = thinking.as_object_mut() else {
        return false;
    };
    settings.insert(
        "block_binding".to_owned(),
        serde_json::json!({ "prefix_mismatch_behavior": "error" }),
    );
    request.set_in_field("thinking", "", thinking)
}

/// Adds a value to the comma-separated `anthropic-beta` header.
fn add_beta(headers: &mut HeaderMap, beta: &'static str) {
    let combined = match headers
        .get(ANTHROPIC_BETA)
        .and_then(|value| value.to_str().ok())
    {
        Some(existing) if !existing.is_empty() => format!("{existing},{beta}"),
        _ => beta.to_owned(),
    };
    match HeaderValue::from_str(&combined) {
        Ok(value) => {
            headers.insert(ANTHROPIC_BETA, value);
        }
        Err(err) => tracing::warn!(error = %err, "could not add beta header"),
    }
}

fn record_trace(state: &State, compiled: &Compiled, exchange: Option<&Exchange>) {
    for entry in &compiled.trace {
        match &entry.outcome {
            StageOutcome::Skipped => {}
            StageOutcome::Applied(edits) => {
                state
                    .stats
                    .edits
                    .fetch_add(edits.len() as u64, Ordering::Relaxed);
                for edit in edits {
                    tracing::debug!(
                        stage = entry.stage,
                        rule = edit.rule,
                        target = ?edit.target,
                        pointer = %edit.pointer,
                        "edit"
                    );
                }
            }
            StageOutcome::Failed(_) | StageOutcome::TimedOut => {
                state.stats.stage_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
        tracing::debug!(stage = entry.stage, elapsed = ?entry.elapsed, "stage finished");
    }
    if let Some(exchange) = exchange {
        exchange.record_trace(&compiled.trace);
    }
}

fn upstream_url(base: &str, uri: &Uri) -> String {
    let path_and_query = uri.path_and_query().map_or("/", |value| value.as_str());
    format!("{base}{path_and_query}")
}

/// Copies `headers` without hop-by-hop fields, anything named by `Connection`, or
/// values the client recomputes for the new connection.
fn forwardable_headers(headers: &HeaderMap) -> HeaderMap {
    let named_by_connection: Vec<String> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|name| name.trim().to_ascii_lowercase())
        .collect();

    let mut forwarded = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let drop = HOP_BY_HOP.contains(name)
            || *name == header::HOST
            || *name == header::CONTENT_LENGTH
            || named_by_connection.iter().any(|n| n == name.as_str());
        if !drop {
            forwarded.append(name.clone(), value.clone());
        }
    }
    forwarded
}

/// Builds an error in the Anthropic API's shape so Claude Code reports it normally.
fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": { "type": "api_error", "message": message },
    });
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_beta_appends_to_an_existing_list() {
        let mut headers = HeaderMap::new();
        add_beta(&mut headers, "one");
        assert_eq!(headers[ANTHROPIC_BETA], "one");
        add_beta(&mut headers, "two");
        assert_eq!(headers[ANTHROPIC_BETA], "one,two");
    }

    #[test]
    fn enforcing_adds_to_thinking_settings_and_skips_requests_without_them() {
        let mut with =
            ccc_wire::Request::parse(br#"{"thinking":{"type":"adaptive"},"messages":[]}"#).unwrap();
        assert!(enforce_thinking_check(&mut with));
        let settings = with.field("thinking").unwrap();
        assert_eq!(settings["type"], "adaptive");
        assert_eq!(
            settings["block_binding"]["prefix_mismatch_behavior"],
            "error"
        );

        let mut without = ccc_wire::Request::parse(br#"{"messages":[]}"#).unwrap();
        assert!(!enforce_thinking_check(&mut without));
    }

    #[test]
    fn upstream_url_keeps_path_and_query() {
        let uri: Uri = "/v1/messages?beta=true".parse().unwrap();
        assert_eq!(
            upstream_url("https://x.test/base", &uri),
            "https://x.test/base/v1/messages?beta=true"
        );
    }

    #[test]
    fn forwardable_headers_drops_connection_scoped_and_keeps_the_rest() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONNECTION, HeaderValue::from_static("x-secret-hop"));
        headers.insert("x-secret-hop", HeaderValue::from_static("1"));
        headers.insert(header::HOST, HeaderValue::from_static("localhost"));
        headers.insert(header::CONTENT_LENGTH, HeaderValue::from_static("9"));
        headers.insert(
            header::TRANSFER_ENCODING,
            HeaderValue::from_static("chunked"),
        );
        headers.insert("x-api-key", HeaderValue::from_static("k"));
        headers.append("anthropic-beta", HeaderValue::from_static("a"));
        headers.append("anthropic-beta", HeaderValue::from_static("b"));

        let forwarded = forwardable_headers(&headers);

        assert_eq!(forwarded.len(), 3, "{forwarded:?}");
        assert_eq!(forwarded["x-api-key"], "k");
        assert_eq!(forwarded.get_all("anthropic-beta").iter().count(), 2);
    }
}

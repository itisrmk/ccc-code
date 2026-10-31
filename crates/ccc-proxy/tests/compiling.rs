//! Drives the compile path through a real proxy against a fake upstream.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Bytes, to_bytes};
use axum::extract::{Request as AxumRequest, State};
use axum::http::StatusCode;
use axum::routing::any;
use ccc_ledger::Ledger;
use ccc_pipeline::{Context, Edit, OnFailure, Pipeline, Stage, StageError, Target};
use ccc_proxy::{Proxy, ProxyConfig};
use ccc_redact::{Detector, Redact, Vault};
use ccc_route::{Route, RouteRules, StripNotices};
use ccc_wire::Request;
use serde_json::{Value, json};
use tokio::net::TcpListener;

type Bodies = Arc<Mutex<Vec<(String, Bytes)>>>;

async fn record(State(bodies): State<Bodies>, request: AxumRequest) -> &'static str {
    let path = request.uri().path().to_owned();
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    bodies.lock().unwrap().push((path, body));
    "ok"
}

async fn spawn_upstream() -> (SocketAddr, Bodies) {
    let bodies = Bodies::default();
    let app = Router::new()
        .fallback(any(record))
        .with_state(Arc::clone(&bodies));
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, bodies)
}

/// A stage that rewrites the newest message when it holds the word "shrink".
#[derive(Debug)]
struct Shrink;

impl Stage for Shrink {
    fn name(&self) -> &'static str {
        "shrink"
    }

    fn time_cap(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let edits = request
            .messages()
            .iter()
            .enumerate()
            .skip(context.protected_before)
            .filter(|(_, message)| message["content"] == "shrink")
            .map(|(message, _)| Edit {
                target: Target::Message(message),
                pointer: "/content".to_owned(),
                value: json!("small"),
                rule: "test",
            })
            .collect();
        Ok(edits)
    }
}

#[derive(Debug)]
struct Failing(OnFailure);

impl Stage for Failing {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn time_cap(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn on_failure(&self) -> OnFailure {
        self.0
    }

    fn plan(&self, _: &Request, _: &Context) -> Result<Vec<Edit>, StageError> {
        Err(StageError("boom".to_owned()))
    }
}

struct Setup {
    pipeline: Pipeline,
    disabled: BTreeSet<String>,
    state_dir: Option<PathBuf>,
    vault: Option<Arc<Vault>>,
    ledger: Option<Arc<Ledger>>,
    budget_usd: Option<f64>,
    local_url: Option<String>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            pipeline: Pipeline::new(vec![Arc::new(Shrink)]),
            disabled: BTreeSet::new(),
            state_dir: None,
            vault: None,
            ledger: None,
            budget_usd: None,
            local_url: None,
        }
    }
}

async fn spawn_proxy(upstream: SocketAddr, setup: Setup) -> SocketAddr {
    let proxy = Proxy::new(ProxyConfig {
        upstream: format!("http://{upstream}"),
        dump_dir: None,
        pipeline: setup.pipeline,
        disabled_stages: setup.disabled,
        state_dir: setup.state_dir,
        vault: setup.vault,
        enforce_thinking_check: false,
        ledger: setup.ledger,
        budget_usd: setup.budget_usd,
        local_url: setup.local_url,
    })
    .unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(proxy.serve(listener, std::future::pending()));
    address
}

async fn post(proxy: SocketAddr, path: &str, body: &str) -> reqwest::Response {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("http://{proxy}{path}"))
        .header("x-claude-code-session-id", "session-1")
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

fn sent(bodies: &Bodies) -> Value {
    let bodies = bodies.lock().unwrap();
    serde_json::from_slice(&bodies.last().unwrap().1).unwrap()
}

#[tokio::test]
async fn applies_edits_to_messages_requests_and_count_tokens() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(upstream, Setup::default()).await;
    let body = r#"{"model":"m","messages":[{"role":"user","content":"shrink"}],"stream":true}"#;

    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let response = post(proxy, path, body).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(sent(&bodies)["messages"][0]["content"], "small", "{path}");
        assert_eq!(sent(&bodies)["stream"], true);
    }
}

#[tokio::test]
async fn unchanged_requests_go_out_byte_for_byte() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(upstream, Setup::default()).await;
    // Spacing a re-serialization would normalize.
    let body = r#"{ "model": "m",  "messages": [ {"role":"user", "content":"leave"} ] }"#;

    post(proxy, "/v1/messages", body).await;

    assert_eq!(bodies.lock().unwrap()[0].1, body.as_bytes());
}

#[tokio::test]
async fn other_paths_and_unreadable_bodies_are_not_compiled() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(upstream, Setup::default()).await;
    let shrinkable = r#"{"messages":[{"role":"user","content":"shrink"}]}"#;

    post(proxy, "/v1/other", shrinkable).await;
    post(proxy, "/v1/messages", "not json").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies[0].1, shrinkable.as_bytes());
    assert_eq!(bodies[1].1, "not json".as_bytes());
}

#[tokio::test]
async fn a_conversation_with_existing_history_keeps_it_intact() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(upstream, Setup::default()).await;
    let first_seen = json!({ "messages": [
        { "role": "user", "content": "shrink" },
        { "role": "assistant", "content": "ok" },
    ] });
    post(proxy, "/v1/messages", &first_seen.to_string()).await;
    assert_eq!(sent(&bodies)["messages"][0]["content"], "shrink");

    let next = json!({ "messages": [
        { "role": "user", "content": "shrink" },
        { "role": "assistant", "content": "ok" },
        { "role": "user", "content": "shrink" },
    ] });
    post(proxy, "/v1/messages", &next.to_string()).await;
    let forwarded = sent(&bodies);
    assert_eq!(forwarded["messages"][0]["content"], "shrink", "old stays");
    assert_eq!(forwarded["messages"][2]["content"], "small", "new compiles");
}

#[tokio::test]
async fn a_disabled_stage_stays_off_only_for_new_conversations() {
    let dir = std::env::temp_dir().join(format!("ccc-compiling-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (upstream, bodies) = spawn_upstream().await;
    let started = json!({ "messages": [{ "role": "user", "content": "shrink" }] }).to_string();

    let first = spawn_proxy(
        upstream,
        Setup {
            state_dir: Some(dir.clone()),
            ..Setup::default()
        },
    )
    .await;
    post(first, "/v1/messages", &started).await;
    assert_eq!(sent(&bodies)["messages"][0]["content"], "small");

    // Restarted with the stage off: the running conversation must compile as it did.
    let second = spawn_proxy(
        upstream,
        Setup {
            state_dir: Some(dir.clone()),
            disabled: BTreeSet::from(["shrink".to_owned()]),
            ..Setup::default()
        },
    )
    .await;
    let resumed = json!({ "messages": [
        { "role": "user", "content": "shrink" },
        { "role": "assistant", "content": "ok" },
        { "role": "user", "content": "shrink" },
    ] });
    post(second, "/v1/messages", &resumed.to_string()).await;
    let forwarded = sent(&bodies);
    assert_eq!(forwarded["messages"][0]["content"], "small");
    assert_eq!(forwarded["messages"][2]["content"], "small");

    let new_conversation = json!({ "messages": [{ "role": "user", "content": "shrink" }, { "role": "user", "content": "x" }] });
    let other = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("http://{second}/v1/messages"))
        .header("x-claude-code-session-id", "session-2")
        .body(new_conversation.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(other.status(), StatusCode::OK);
    assert_eq!(sent(&bodies)["messages"][0]["content"], "shrink");
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn a_pass_through_failure_still_sends_the_request() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: Pipeline::new(vec![Arc::new(Failing(OnFailure::PassThrough))]),
            ..Setup::default()
        },
    )
    .await;
    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;

    let response = post(proxy, "/v1/messages", body).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(bodies.lock().unwrap()[0].1, body.as_bytes());
}

#[tokio::test]
async fn a_blocking_failure_never_reaches_upstream() {
    let (upstream, bodies) = spawn_upstream().await;
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: Pipeline::new(vec![Arc::new(Failing(OnFailure::Block))]),
            ..Setup::default()
        },
    )
    .await;

    let response = post(proxy, "/v1/messages", r#"{"messages":[]}"#).await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["type"], "error");
    assert!(bodies.lock().unwrap().is_empty());
}

fn fake_key() -> String {
    format!("{}{}", "sk-ant-", "api03-abcdefghijklmnopqrstuvwxyz012345")
}

/// A fake model: repeats whatever placeholder it finds in the request, streamed so that
/// the placeholder straddles two events, or as plain JSON for non-streaming requests.
async fn echo_placeholder(
    State(bodies): State<Bodies>,
    request: AxumRequest,
) -> axum::response::Response {
    let accept_encoding = request
        .headers()
        .get("accept-encoding")
        .map(|value| value.to_str().unwrap().to_owned());
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    bodies
        .lock()
        .unwrap()
        .push((accept_encoding.unwrap_or_default(), body.clone()));
    let text = String::from_utf8(body.to_vec()).unwrap();
    let start = text
        .find("<SECRET_")
        .expect("the request should carry a placeholder");
    let placeholder = &text[start..start + 19];
    let parsed: Value = serde_json::from_str(&text).unwrap();
    if parsed["stream"] == true {
        let (left, right) = placeholder.split_at(9);
        let delta = |value: &str| {
            let data = json!({ "type": "content_block_delta", "index": 0,
                "delta": { "type": "text_delta", "text": value } });
            format!("event: content_block_delta\ndata: {data}\n\n")
        };
        let stop =
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n";
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from(format!(
                "{}{}{stop}",
                delta(left),
                delta(right)
            )))
            .unwrap()
    } else {
        let answer = json!({ "content": [{ "type": "text", "text": placeholder }] });
        axum::response::Response::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(answer.to_string()))
            .unwrap()
    }
}

async fn spawn_redacting_proxy() -> (SocketAddr, Bodies) {
    let bodies = Bodies::default();
    let app = Router::new()
        .fallback(any(echo_placeholder))
        .with_state(Arc::clone(&bodies));
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let upstream = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let vault = Arc::new(Vault::ephemeral().unwrap());
    let stage = Redact::new(Detector::builtin(), Arc::clone(&vault));
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: Pipeline::new(vec![Arc::new(stage)]),
            vault: Some(vault),
            ..Setup::default()
        },
    )
    .await;
    (proxy, bodies)
}

#[tokio::test]
async fn a_secret_never_reaches_upstream_and_comes_back_real_when_streaming() {
    let (proxy, bodies) = spawn_redacting_proxy().await;
    let key = fake_key();
    let body = json!({ "stream": true, "messages": [
        { "role": "user", "content": format!("use {key}") },
    ] });

    let response = post(proxy, "/v1/messages", &body.to_string()).await;
    let text = response.text().await.unwrap();

    assert!(text.contains(&key), "{text}");
    assert!(!text.contains("<SECRET_"), "{text}");
    let (accept_encoding, sent) = bodies.lock().unwrap()[0].clone();
    assert!(!String::from_utf8_lossy(&sent).contains(&key));
    assert_eq!(accept_encoding, "identity");
}

#[tokio::test]
async fn a_secret_comes_back_real_in_a_non_streaming_response() {
    let (proxy, _bodies) = spawn_redacting_proxy().await;
    let key = fake_key();
    let body = json!({ "messages": [{ "role": "user", "content": format!("use {key}") }] });

    let response = post(proxy, "/v1/messages", &body.to_string()).await;
    let answer: Value = response.json().await.unwrap();

    assert_eq!(answer["content"][0]["text"], key);
}

#[tokio::test]
async fn with_redaction_on_unreadable_bodies_never_reach_upstream() {
    let (proxy, bodies) = spawn_redacting_proxy().await;
    let client = reqwest::Client::builder().no_proxy().build().unwrap();

    let unparseable = post(proxy, "/v1/messages", "not json").await;
    assert_eq!(unparseable.status(), StatusCode::BAD_REQUEST);

    let compressed = client
        .post(format!("http://{proxy}/v1/messages"))
        .header("content-encoding", "gzip")
        .body(vec![0x1f, 0x8b])
        .send()
        .await
        .unwrap();
    assert_eq!(compressed.status(), StatusCode::BAD_REQUEST);
    assert!(bodies.lock().unwrap().is_empty());
}

const MODEL_REPLY: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",",
    "\"usage\":{\"input_tokens\":0,\"cache_read_input_tokens\":1000000,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1000000}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

async fn model_reply(
    State(bodies): State<Bodies>,
    request: AxumRequest,
) -> axum::response::Response {
    let auth = request.headers().contains_key("x-api-key");
    let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
    bodies.lock().unwrap().push((format!("auth={auth}"), body));
    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(MODEL_REPLY))
        .unwrap()
}

async fn spawn_model() -> (SocketAddr, Bodies) {
    let bodies = Bodies::default();
    let app = Router::new()
        .fallback(any(model_reply))
        .with_state(Arc::clone(&bodies));
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, bodies)
}

#[tokio::test]
async fn usage_reaches_the_ledger_and_a_budget_notice_appears_once_then_is_stripped() {
    let (upstream, bodies) = spawn_model().await;
    let ledger = Arc::new(Ledger::in_memory().unwrap());
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: Pipeline::new(vec![Arc::new(StripNotices)]),
            ledger: Some(Arc::clone(&ledger)),
            budget_usd: Some(1.0),
            ..Setup::default()
        },
    )
    .await;
    let body =
        r#"{"model":"claude-opus-5-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // $20.20 at list prices: output of 1M tokens plus a read of 1M.
    let first = post(proxy, "/v1/messages", body)
        .await
        .text()
        .await
        .unwrap();
    assert!(first.contains("answer"));
    assert!(
        !first.contains("[ccc budget]"),
        "nothing spent yet: {first}"
    );

    let mut spent = 0.0;
    for _ in 0..100 {
        spent = ledger.session_cost("session-1").unwrap();
        if spent > 0.0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!((spent - 20.2).abs() < 1e-6, "{spent}");

    let second = post(proxy, "/v1/messages", body)
        .await
        .text()
        .await
        .unwrap();
    let notice_block = second.find("[ccc budget]").expect("over budget now");
    assert!(notice_block < second.find("answer").unwrap(), "{second}");

    let third = post(proxy, "/v1/messages", body)
        .await
        .text()
        .await
        .unwrap();
    assert!(!third.contains("[ccc budget]"), "only once: {third}");

    // Claude Code replays the notice as part of the assistant turn. The model mustn't see it.
    let replay = json!({ "model": "claude-opus-5-5", "messages": [
        { "role": "user", "content": "hi" },
        { "role": "assistant", "content": [
            { "type": "text", "text": "answer" },
            { "type": "text", "text": "[ccc budget] This session has used a lot." },
        ] },
        { "role": "user", "content": "more" },
    ] });
    post(proxy, "/v1/messages", &replay.to_string()).await;
    let (_, sent) = bodies.lock().unwrap().last().unwrap().clone();
    let sent: Value = serde_json::from_slice(&sent).unwrap();
    assert_eq!(
        sent["messages"][1]["content"],
        json!([{ "type": "text", "text": "answer" }])
    );
}

fn routing_to_local() -> Pipeline {
    Pipeline::new(vec![Arc::new(Route::new(RouteRules {
        background_model: Some("local:qwen3".to_owned()),
        ..RouteRules::default()
    }))])
}

#[tokio::test]
async fn background_calls_go_local_without_credentials() {
    let (upstream, upstream_bodies) = spawn_model().await;
    let (local, local_bodies) = spawn_model().await;
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: routing_to_local(),
            local_url: Some(format!("http://{local}")),
            ..Setup::default()
        },
    )
    .await;

    let response = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("http://{proxy}/v1/messages"))
        .header("x-api-key", "real-key")
        .body(r#"{"model":"claude-haiku-5-5","stream":true,"messages":[{"role":"user","content":"title"}]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(upstream_bodies.lock().unwrap().is_empty());
    let local_bodies = local_bodies.lock().unwrap();
    let (auth, sent) = &local_bodies[0];
    assert_eq!(
        auth, "auth=false",
        "Anthropic credentials stay away from local models"
    );
    let sent: Value = serde_json::from_slice(sent).unwrap();
    assert_eq!(sent["model"], "qwen3");
}

#[tokio::test]
async fn a_failing_local_model_falls_back_with_the_original_model() {
    let (upstream, upstream_bodies) = spawn_model().await;
    let dead = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let dead_address = dead.local_addr().unwrap();
    drop(dead);
    let proxy = spawn_proxy(
        upstream,
        Setup {
            pipeline: routing_to_local(),
            local_url: Some(format!("http://{dead_address}")),
            ..Setup::default()
        },
    )
    .await;

    let body = r#"{"model":"claude-haiku-5-5","stream":true,"messages":[{"role":"user","content":"title"}]}"#;
    let response = post(proxy, "/v1/messages", body).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.text().await.unwrap().contains("answer"));
    let bodies = upstream_bodies.lock().unwrap();
    let sent: Value = serde_json::from_slice(&bodies[0].1).unwrap();
    assert_eq!(sent["model"], "claude-haiku-5-5");
}

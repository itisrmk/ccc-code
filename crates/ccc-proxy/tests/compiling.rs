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
use ccc_pipeline::{Context, Edit, OnFailure, Pipeline, Stage, StageError};
use ccc_proxy::{Proxy, ProxyConfig};
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
                message,
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
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            pipeline: Pipeline::new(vec![Arc::new(Shrink)]),
            disabled: BTreeSet::new(),
            state_dir: None,
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

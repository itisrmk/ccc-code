//! Drives a real proxy against a fake upstream over loopback sockets.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::any;
use ccc_proxy::{Proxy, ProxyConfig};
use futures_util::StreamExt;
use futures_util::stream;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
struct Seen {
    method: Method,
    path_and_query: String,
    headers: HeaderMap,
    body: Bytes,
}

type SeenLog = Arc<Mutex<Vec<Seen>>>;

async fn record(State(log): State<SeenLog>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap();
    log.lock().unwrap().push(Seen {
        method: parts.method,
        path_and_query: parts.uri.path_and_query().unwrap().to_string(),
        headers: parts.headers,
        body,
    });
    Response::builder()
        .status(StatusCode::CREATED)
        .header("x-upstream", "yes")
        .body(Body::from("upstream-body"))
        .unwrap()
}

async fn spawn_upstream(app: Router) -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    address
}

struct RunningProxy {
    address: SocketAddr,
    _stop: oneshot::Sender<()>,
}

async fn spawn_proxy(upstream: String, dump_dir: Option<PathBuf>) -> RunningProxy {
    let proxy = Proxy::new(ProxyConfig { upstream, dump_dir }).unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(proxy.serve(listener, async {
        let _ = stopped.await;
    }));
    RunningProxy {
        address,
        _stop: stop,
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test]
async fn forwards_method_path_headers_and_body_unchanged() {
    let log = SeenLog::default();
    let app = Router::new()
        .fallback(any(record))
        .with_state(Arc::clone(&log));
    let upstream = spawn_upstream(app).await;
    let proxy = spawn_proxy(format!("http://{upstream}/base"), None).await;

    // Odd spacing, key order and escapes must all survive untouched.
    let body = "{ \"b\":1,  \"a\":\"\\u00e9\\/\", \"n\":1.0 }";
    let response = client()
        .post(format!("http://{}/v1/messages?beta=true", proxy.address))
        .header("x-api-key", "key-1")
        .header("authorization", "Bearer token-1")
        .header("anthropic-beta", "one")
        .header("anthropic-version", "2023-06-01")
        .body(body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-upstream"], "yes");
    assert_eq!(response.text().await.unwrap(), "upstream-body");

    let seen = log.lock().unwrap().pop().unwrap();
    assert_eq!(seen.method, Method::POST);
    assert_eq!(seen.path_and_query, "/base/v1/messages?beta=true");
    assert_eq!(seen.body, body.as_bytes());
    assert_eq!(seen.headers["x-api-key"], "key-1");
    assert_eq!(seen.headers["authorization"], "Bearer token-1");
    assert_eq!(seen.headers["anthropic-beta"], "one");
    assert_eq!(seen.headers["anthropic-version"], "2023-06-01");
    assert_ne!(seen.headers[header::HOST], proxy.address.to_string());
}

#[tokio::test]
async fn forwards_other_paths_and_methods() {
    let log = SeenLog::default();
    let app = Router::new()
        .fallback(any(record))
        .with_state(Arc::clone(&log));
    let upstream = spawn_upstream(app).await;
    let proxy = spawn_proxy(format!("http://{upstream}"), None).await;

    let response = client()
        .get(format!("http://{}/v1/models", proxy.address))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let seen = log.lock().unwrap().pop().unwrap();
    assert_eq!(seen.method, Method::GET);
    assert_eq!(seen.path_and_query, "/v1/models");
}

/// Upstream hands out chunks only as the test releases them, proving the proxy
/// delivers each one immediately instead of buffering the response.
#[tokio::test]
async fn streams_each_chunk_before_upstream_finishes() {
    let (release, released) = mpsc::unbounded_channel::<&'static str>();
    let released = Arc::new(tokio::sync::Mutex::new(Some(released)));
    let app = Router::new().fallback(any(move || {
        let released = Arc::clone(&released);
        async move {
            let receiver = released.lock().await.take().unwrap();
            let chunks = stream::unfold(receiver, |mut receiver| async move {
                let chunk = receiver.recv().await?;
                Some((
                    Ok::<_, Infallible>(Bytes::from_static(chunk.as_bytes())),
                    receiver,
                ))
            });
            Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(chunks))
                .unwrap()
        }
    }));
    let upstream = spawn_upstream(app).await;
    let proxy = spawn_proxy(format!("http://{upstream}"), None).await;

    let response = client()
        .post(format!("http://{}/v1/messages", proxy.address))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let mut body = response.bytes_stream();

    for chunk in ["event: a\ndata: {}\n\n", "event: b\ndata: {}\n\n"] {
        release.send(chunk).unwrap();
        let received = tokio::time::timeout(Duration::from_secs(5), body.next())
            .await
            .expect("chunk should arrive while upstream is still open")
            .unwrap()
            .unwrap();
        assert_eq!(received, chunk.as_bytes());
    }

    drop(release);
    assert!(
        body.next().await.is_none(),
        "stream should end with upstream"
    );
}

#[tokio::test]
async fn unreachable_upstream_returns_an_api_shaped_502() {
    // Bind then drop to get a port with nothing listening.
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let dead = listener.local_addr().unwrap();
    drop(listener);
    let proxy = spawn_proxy(format!("http://{dead}"), None).await;

    let response = client()
        .post(format!("http://{}/v1/messages", proxy.address))
        .body("{}")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "api_error");
}

#[tokio::test]
async fn dump_captures_traffic_without_credentials() {
    let log = SeenLog::default();
    let app = Router::new()
        .fallback(any(record))
        .with_state(Arc::clone(&log));
    let upstream = spawn_upstream(app).await;
    let dir = std::env::temp_dir().join(format!(
        "ccc-dump-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let proxy = spawn_proxy(format!("http://{upstream}"), Some(dir.clone())).await;

    let response = client()
        .post(format!("http://{}/v1/messages", proxy.address))
        .header("x-api-key", "sk-super-secret")
        .body("{\"hello\":1}")
        .send()
        .await
        .unwrap();
    assert_eq!(response.text().await.unwrap(), "upstream-body");

    let mut contents = String::new();
    let mut names = Vec::new();
    // The body writer runs on a background thread, so give it a moment to finish.
    for _ in 0..100 {
        names.clear();
        contents.clear();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            names.push(path.file_name().unwrap().to_string_lossy().into_owned());
            contents.push_str(&std::fs::read_to_string(&path).unwrap());
        }
        if contents.contains("upstream-body") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    names.sort();

    assert_eq!(names.len(), 4, "{names:?}");
    assert!(contents.contains("{\"hello\":1}"), "{contents}");
    assert!(contents.contains("upstream-body"), "{contents}");
    assert!(!contents.contains("sk-super-secret"), "{contents}");
    std::fs::remove_dir_all(&dir).unwrap();
}

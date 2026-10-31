//! Measures latency the proxy adds. Timing varies by machine, so this is opt-in:
//! `cargo test --release -p ccc-proxy --test overhead -- --ignored --nocapture`

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::routing::any;
use ccc_pipeline::Pipeline;
use ccc_proxy::{Proxy, ProxyConfig};
use ccc_prune::Prune;
use ccc_redact::{Detector, Redact, Vault};
use ccc_route::{Route, StripNotices};
use serde_json::json;
use tokio::net::TcpListener;

const ROUNDS: usize = 500;
/// Fewer rounds for whole-pipeline runs, since those bodies are far larger.
const GATE_2_ROUNDS: usize = 60;
/// Roughly the JSON size of a request with a large cached context.
const BODY_BYTES: usize = 4 << 20;
const OVERHEAD_P95_LIMIT: Duration = Duration::from_millis(5);

async fn p95(client: &reqwest::Client, address: SocketAddr, body: &[u8]) -> Duration {
    p95_of(client, address, body, ROUNDS).await
}

async fn p95_of(
    client: &reqwest::Client,
    address: SocketAddr,
    body: &[u8],
    rounds: usize,
) -> Duration {
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let started = Instant::now();
        let response = client
            .post(format!("http://{address}/v1/messages"))
            .body(body.to_vec())
            .send()
            .await
            .unwrap();
        response.bytes().await.unwrap();
        samples.push(started.elapsed());
    }
    samples.sort();
    samples[rounds * 95 / 100]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "timing measurement; run explicitly in release mode"]
async fn added_latency_stays_within_budget() {
    let app = Router::new().fallback(any(|request: axum::extract::Request| async move {
        axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .unwrap();
        "ok"
    }));
    let upstream_listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(upstream_listener, app).await.unwrap() });

    let proxy = Proxy::new(ProxyConfig {
        upstream: format!("http://{upstream}"),
        dump_dir: None,
        pipeline: Pipeline::default(),
        disabled_stages: BTreeSet::new(),
        state_dir: None,
        vault: None,
        enforce_thinking_check: false,
        ledger: None,
        budget_usd: None,
        local_url: None,
    })
    .unwrap();
    let proxy_listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let proxied = proxy_listener.local_addr().unwrap();
    tokio::spawn(proxy.serve(proxy_listener, std::future::pending()));

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let body = vec![b'x'; BODY_BYTES];

    let direct = p95(&client, upstream, &body).await;
    let through_proxy = p95(&client, proxied, &body).await;
    let added = through_proxy.saturating_sub(direct);
    println!("direct p95 {direct:?}, proxied p95 {through_proxy:?}, added {added:?}");

    assert!(added < OVERHEAD_P95_LIMIT, "added {added:?}");
}

/// Gate 2: what the full default pipeline adds on a real-shaped request.
const COMPILE_P95_LIMIT: Duration = Duration::from_millis(150);

/// A request shaped like a long session: thousands of tool rounds, some containing a secret
/// and some large enough for pruning to act on.
fn long_session_body(target_bytes: usize) -> Vec<u8> {
    let mut messages = vec![json!({ "role": "user", "content": "start" })];
    let log: String = (0..400)
        .map(|i| format!("test case_{i} ... ok\n"))
        .collect();
    let source: String = (0..120)
        .map(|i| format!("fn item_{i}() -> u32 {{ {i} }}\n"))
        .collect();
    let mut round = 0;
    let mut size = 0;
    while size < target_bytes {
        let (command, output) = if round % 5 == 0 {
            ("cargo test", format!("{log}round {round}\n"))
        } else {
            ("cat src/lib.rs", format!("{source}// round {round}\n"))
        };
        size += output.len() + 200;
        messages.push(json!({ "role": "assistant", "content": [
            { "type": "tool_use", "id": format!("t{round}"), "name": "Bash", "input": { "command": command } },
        ] }));
        messages.push(json!({ "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": format!("t{round}"), "content": output },
        ] }));
        round += 1;
    }
    json!({ "model": "claude-opus-5-5", "stream": true, "messages": messages })
        .to_string()
        .into_bytes()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "timing measurement; run explicitly in release mode"]
async fn gate_2_compiling_stays_under_150_ms_at_p95() {
    let app = Router::new().fallback(any(|request: axum::extract::Request| async move {
        axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .unwrap();
        "ok"
    }));
    let upstream_listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(upstream_listener, app).await.unwrap() });

    let vault = Arc::new(Vault::ephemeral().unwrap());
    let proxy = Proxy::new(ProxyConfig {
        upstream: format!("http://{upstream}"),
        dump_dir: None,
        pipeline: Pipeline::new(vec![
            Arc::new(StripNotices),
            Arc::new(Redact::new(Detector::builtin(), Arc::clone(&vault))),
            Arc::new(Prune::new()),
            Arc::new(Route::default()),
        ]),
        disabled_stages: BTreeSet::new(),
        state_dir: None,
        vault: Some(vault),
        enforce_thinking_check: false,
        ledger: None,
        budget_usd: None,
        local_url: None,
    })
    .unwrap();
    let proxy_listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let proxied = proxy_listener.local_addr().unwrap();
    tokio::spawn(proxy.serve(proxy_listener, std::future::pending()));

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for megabytes in [1, 4, 16] {
        let body = long_session_body(megabytes << 20);
        let direct = p95_of(&client, upstream, &body, GATE_2_ROUNDS).await;
        let through_proxy = p95_of(&client, proxied, &body, GATE_2_ROUNDS).await;
        let added = through_proxy.saturating_sub(direct);
        println!("{megabytes} MB request: added p95 {added:?}");
        if megabytes <= 4 {
            assert!(added < COMPILE_P95_LIMIT, "{megabytes} MB added {added:?}");
        }
    }
}

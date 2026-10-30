//! Measures latency the proxy adds. Timing varies by machine, so this is opt-in:
//! `cargo test --release -p ccc-proxy --test overhead -- --ignored --nocapture`

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use axum::Router;
use axum::routing::any;
use ccc_pipeline::Pipeline;
use ccc_proxy::{Proxy, ProxyConfig};
use tokio::net::TcpListener;

const ROUNDS: usize = 500;
/// Roughly the JSON size of a request with a large cached context.
const BODY_BYTES: usize = 4 << 20;
const OVERHEAD_P95_LIMIT: Duration = Duration::from_millis(5);

async fn p95(client: &reqwest::Client, address: SocketAddr, body: &[u8]) -> Duration {
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
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
    samples[ROUNDS * 95 / 100]
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

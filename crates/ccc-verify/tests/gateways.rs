//! Real HTTP gateways, honest and deliberately broken, each checked end to end. A check only
//! means something if a gateway that breaks the rule fails it.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers fail loudly on setup errors"
)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use ccc_pipeline::Pipeline;
use ccc_proxy::{Proxy, ProxyConfig};
use ccc_prune::Prune;
use ccc_redact::{Detector, Redact, Vault};
use ccc_verify::{FakeUpstream, Outcome, Report, VerifyOptions, run};
use futures_util::StreamExt as _;
use serde_json::{Value, json};
use tokio::net::TcpListener;

/// How a test gateway misbehaves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    DropsSignatures,
    MergesBlocks,
    RewritesHistory,
    StampsSystem,
    SwitchesModel,
    StripsBeta,
    DropsThinkingConfig,
    DropsUnknownFields,
    BuffersStream,
    DropsStreamSignature,
    MasksErrors,
    DropsRetryAfter,
    RejectsLarge,
}

struct Gateway {
    upstream: String,
    fault: Fault,
    calls: AtomicU64,
    client: reqwest::Client,
}

fn walk_blocks(body: &mut Value, mut visit: impl FnMut(&mut Value)) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages {
        if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
            for block in blocks {
                visit(block);
            }
        }
    }
}

fn mutate(body: &mut Value, fault: Fault, call: u64) {
    match fault {
        Fault::DropsSignatures => walk_blocks(body, |block| {
            if let Some(fields) = block.as_object_mut() {
                fields.remove("signature");
            }
        }),
        Fault::MergesBlocks => {
            let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
                return;
            };
            for message in messages.iter_mut() {
                if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
                    blocks.retain(|block| block["type"] != "redacted_thinking");
                }
            }
        }
        Fault::RewritesHistory => {
            // Shortens every tool result except the newest, which is how a naive compressor
            // "keeps recent context".
            let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
                return;
            };
            let count = messages.len();
            for message in messages.iter_mut().take(count.saturating_sub(1)) {
                if let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) {
                    for block in blocks {
                        if block["type"] == "tool_result" {
                            block["content"] = json!("[older output removed]");
                        }
                    }
                }
            }
        }
        Fault::StampsSystem => {
            body["system"] = json!(format!("Request number {call}"));
        }
        Fault::SwitchesModel => {
            if call >= 2 {
                body["model"] = json!("claude-haiku-5-5");
            }
        }
        Fault::DropsThinkingConfig => {
            body.as_object_mut().map(|fields| fields.remove("thinking"));
        }
        Fault::DropsUnknownFields => {
            body.as_object_mut()
                .map(|fields| fields.remove("x_future_field"));
        }
        _ => {}
    }
}

async fn relay(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    let fault = gateway.fault;
    let call = gateway.calls.fetch_add(1, Ordering::Relaxed);
    let path = request.uri().path().to_owned();
    let mut headers = HeaderMap::new();
    for (name, value) in request.headers() {
        let skip = matches!(name.as_str(), "host" | "content-length")
            || (fault == Fault::StripsBeta && name.as_str() == "anthropic-beta");
        if !skip {
            headers.append(name.clone(), value.clone());
        }
    }
    let raw = to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .expect("reading the request");
    if fault == Fault::RejectsLarge && raw.len() > 1024 * 1024 {
        return Response::builder()
            .status(StatusCode::PAYLOAD_TOO_LARGE)
            .body(Body::empty())
            .unwrap();
    }
    let body = if fault == Fault::None {
        raw.to_vec()
    } else {
        let mut value: Value = serde_json::from_slice(&raw).expect("a JSON request");
        mutate(&mut value, fault, call);
        value.to_string().into_bytes()
    };
    let answer = gateway
        .client
        .post(format!("{}{path}", gateway.upstream))
        .headers(headers)
        .body(body)
        .send()
        .await
        .expect("reaching the upstream");
    let status = answer.status();
    let mut builder =
        Response::builder().status(if fault == Fault::MasksErrors && !status.is_success() {
            StatusCode::BAD_GATEWAY
        } else {
            status
        });
    for (name, value) in answer.headers() {
        let dropped = matches!(name.as_str(), "content-length" | "transfer-encoding")
            || (fault == Fault::DropsRetryAfter && name.as_str() == "retry-after");
        if !dropped {
            builder = builder.header(name.clone(), value.clone());
        }
    }
    let body = match fault {
        Fault::BuffersStream => Body::from(answer.bytes().await.expect("buffering")),
        Fault::DropsStreamSignature => {
            let text =
                String::from_utf8(answer.bytes().await.expect("buffering").to_vec()).expect("text");
            let kept: Vec<&str> = text
                .split("\n\n")
                .filter(|event| !event.contains("signature_delta"))
                .collect();
            Body::from(kept.join("\n\n"))
        }
        _ => Body::from_stream(answer.bytes_stream().map(|part: Result<Bytes, _>| part)),
    };
    builder.body(body).expect("a valid response")
}

async fn check_gateway(fault: Fault) -> Report {
    let upstream = FakeUpstream::start(TcpListener::bind("127.0.0.1:0").await.unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway_url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(relay).with_state(Arc::new(Gateway {
        upstream: upstream.url(),
        fault,
        calls: AtomicU64::new(0),
        client: reqwest::Client::new(),
    }));
    tokio::spawn(async move { axum::serve(listener, app).await });
    run(
        &upstream,
        &VerifyOptions {
            gateway: gateway_url,
            headers: Vec::new(),
        },
    )
    .await
    .unwrap()
}

fn fails(report: &Report, name: &str) {
    assert!(
        matches!(report.outcome(name), Some(Outcome::Fail(_))),
        "{name} should fail:\n{report}"
    );
    assert!(!report.passed());
}

fn warns_without_failing(report: &Report, name: &str) {
    assert!(
        matches!(report.outcome(name), Some(Outcome::Warn(_))),
        "{name} should warn:\n{report}"
    );
    assert!(report.passed(), "{report}");
}

#[tokio::test]
async fn an_honest_gateway_passes_everything_cleanly() {
    let report = check_gateway(Fault::None).await;
    assert!(report.passed(), "{report}");
    let unclean: Vec<_> = report
        .checks
        .iter()
        .filter(|check| check.outcome != Outcome::Pass)
        .collect();
    assert!(unclean.is_empty(), "{report}");
    assert!(report.checks.len() >= 15, "{report}");
}

#[tokio::test]
async fn dropping_signatures_fails() {
    fails(
        &check_gateway(Fault::DropsSignatures).await,
        "thinking-intact",
    );
}

#[tokio::test]
async fn dropping_a_block_fails() {
    let report = check_gateway(Fault::MergesBlocks).await;
    fails(&report, "block-order");
}

#[tokio::test]
async fn rewriting_old_output_fails_append_only() {
    fails(&check_gateway(Fault::RewritesHistory).await, "append-only");
}

#[tokio::test]
async fn a_changing_system_prompt_fails() {
    let report = check_gateway(Fault::StampsSystem).await;
    fails(&report, "system-stable");
    fails(&report, "repeatable");
}

#[tokio::test]
async fn switching_model_midway_fails() {
    fails(&check_gateway(Fault::SwitchesModel).await, "model-stable");
}

#[tokio::test]
async fn stripping_the_beta_header_fails() {
    fails(&check_gateway(Fault::StripsBeta).await, "beta-header");
}

#[tokio::test]
async fn dropping_the_thinking_setting_fails() {
    fails(
        &check_gateway(Fault::DropsThinkingConfig).await,
        "thinking-config",
    );
}

#[tokio::test]
async fn dropping_a_streamed_signature_fails() {
    fails(
        &check_gateway(Fault::DropsStreamSignature).await,
        "stream-thinking",
    );
}

#[tokio::test]
async fn masking_errors_fails() {
    let report = check_gateway(Fault::MasksErrors).await;
    fails(&report, "error-400");
    fails(&report, "error-429");
}

#[tokio::test]
async fn lesser_problems_only_warn() {
    warns_without_failing(&check_gateway(Fault::BuffersStream).await, "stream-live");
    warns_without_failing(
        &check_gateway(Fault::DropsUnknownFields).await,
        "unknown-fields",
    );
    warns_without_failing(&check_gateway(Fault::DropsRetryAfter).await, "error-429");
    warns_without_failing(&check_gateway(Fault::RejectsLarge).await, "large-request");
}

#[tokio::test]
async fn an_unreachable_gateway_is_a_failed_check_and_not_a_crash() {
    let upstream = FakeUpstream::start(TcpListener::bind("127.0.0.1:0").await.unwrap());
    let report = run(
        &upstream,
        &VerifyOptions {
            gateway: "http://127.0.0.1:1".to_owned(),
            headers: Vec::new(),
        },
    )
    .await
    .unwrap();
    fails(&report, "forwarding");
}

#[tokio::test]
async fn a_bad_extra_header_is_reported() {
    let upstream = FakeUpstream::start(TcpListener::bind("127.0.0.1:0").await.unwrap());
    let result = run(
        &upstream,
        &VerifyOptions {
            gateway: upstream.url(),
            headers: vec![("bad header".to_owned(), "x".to_owned())],
        },
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cccs_own_proxy_passes_with_its_default_stages() {
    let upstream = FakeUpstream::start(TcpListener::bind("127.0.0.1:0").await.unwrap());
    let vault = Arc::new(Vault::ephemeral().unwrap());
    let redact = Redact::new(
        Detector::with_project_rules(&[]).unwrap(),
        Arc::clone(&vault),
    );
    let proxy = Proxy::new(ProxyConfig {
        upstream: upstream.url(),
        dump_dir: None,
        pipeline: Pipeline::new(vec![Arc::new(redact), Arc::new(Prune::new())]),
        disabled_stages: BTreeSet::new(),
        state_dir: None,
        vault: Some(vault),
        enforce_thinking_check: false,
        ledger: None,
        budget_usd: None,
        local_url: None,
    })
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(proxy.serve(listener, std::future::pending()));

    let report = run(
        &upstream,
        &VerifyOptions {
            gateway: url,
            headers: Vec::new(),
        },
    )
    .await
    .unwrap();

    assert!(report.passed(), "{report}");
    // Prune shortens the long log the first time it's seen. That's allowed, and is reported.
    assert!(
        matches!(report.outcome("edits-on-arrival"), Some(Outcome::Note(_))),
        "{report}"
    );
}

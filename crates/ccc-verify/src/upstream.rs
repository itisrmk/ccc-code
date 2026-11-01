//! A stand-in for the Anthropic API that records exactly what reaches it.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::stream;
use tokio::net::TcpListener;

use crate::script;

/// The pause between streamed parts, long enough to tell streaming from buffering.
pub(crate) const STREAM_PAUSE: Duration = Duration::from_millis(250);
/// The largest request the stand-in accepts, well above the sizes Claude Code sends.
const MAX_BODY_BYTES: usize = 128 * 1024 * 1024;

/// What the stand-in answers with next.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Scripted {
    #[default]
    Whole,
    Stream,
    Rejected,
    RateLimited,
}

/// A request as it arrived.
#[derive(Debug, Clone)]
pub(crate) struct Recorded {
    pub(crate) headers: HeaderMap,
    pub(crate) body: Bytes,
}

#[derive(Debug, Default)]
struct Shared {
    scripted: Scripted,
    recorded: Vec<Recorded>,
}

/// A running stand-in for the API. Point a gateway's upstream at [`FakeUpstream::url`].
#[derive(Debug, Clone)]
pub struct FakeUpstream {
    address: SocketAddr,
    shared: Arc<Mutex<Shared>>,
}

impl FakeUpstream {
    /// Starts serving on `listener` until the runtime stops.
    #[must_use]
    pub fn start(listener: TcpListener) -> Self {
        let address = listener
            .local_addr()
            .expect("a bound listener has an address");
        let shared = Arc::new(Mutex::new(Shared::default()));
        let app = Router::new()
            .fallback(answer)
            .with_state(Arc::clone(&shared));
        tokio::spawn(async move {
            // Nothing can act on a failure here: requests would simply stop arriving, and
            // the checks report that.
            let _ = axum::serve(listener, app).await;
        });
        Self { address, shared }
    }

    /// The base URL to give a gateway as its upstream.
    #[must_use]
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub(crate) fn script(&self, scripted: Scripted) {
        self.shared.lock().expect("lock poisoned").scripted = scripted;
    }

    /// The newest message request that has arrived.
    pub(crate) fn latest(&self) -> Option<Recorded> {
        self.shared
            .lock()
            .expect("lock poisoned")
            .recorded
            .last()
            .cloned()
    }

    pub(crate) fn count(&self) -> usize {
        self.shared.lock().expect("lock poisoned").recorded.len()
    }
}

async fn answer(State(shared): State<Arc<Mutex<Shared>>>, request: Request) -> Response {
    let is_messages = request.uri().path().ends_with("/messages");
    let headers = request.headers().clone();
    let Ok(body) = to_bytes(request.into_body(), MAX_BODY_BYTES).await else {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    };
    if !is_messages {
        return StatusCode::NOT_FOUND.into_response();
    }
    let scripted = {
        let mut shared = shared.lock().expect("lock poisoned");
        shared.recorded.push(Recorded { headers, body });
        shared.scripted
    };
    match scripted {
        Scripted::Whole => json(StatusCode::OK, &script::response_body()),
        Scripted::Rejected => json(
            StatusCode::BAD_REQUEST,
            &script::error_body("invalid_request_error"),
        ),
        Scripted::RateLimited => {
            let mut response = json(
                StatusCode::TOO_MANY_REQUESTS,
                &script::error_body("rate_limit_error"),
            );
            response.headers_mut().insert(
                header::RETRY_AFTER,
                header::HeaderValue::from_static(script::RETRY_AFTER_SECONDS),
            );
            response
        }
        Scripted::Stream => streamed(),
    }
}

fn json(status: StatusCode, body: &serde_json::Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn streamed() -> Response {
    let parts = stream::unfold(
        (script::stream_chunks().into_iter(), true),
        |(mut parts, first)| async move {
            let part = parts.next()?;
            if !first {
                tokio::time::sleep(STREAM_PAUSE).await;
            }
            Some((
                Ok::<_, std::convert::Infallible>(Bytes::from(part)),
                (parts, false),
            ))
        },
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(parts))
        .expect("a fixed status and headers are valid")
}

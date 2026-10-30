//! The forwarding path: one handler for every method and route.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::{Request, State as AxumState};
use axum::http::header::{self, HeaderMap, HeaderName};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use futures_util::TryStreamExt;

use crate::dump::Dumper;

/// Largest request body the proxy will buffer. Claude Code bodies can carry many
/// base64 images on top of a 1M-token context, so this sits well above that.
const MAX_REQUEST_BYTES: usize = 1 << 30;

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

    let upstream_url = upstream_url(&state.upstream, &parts.uri);
    let upstream_request = state
        .client
        .request(parts.method, &upstream_url)
        .headers(forwardable_headers(&parts.headers))
        .body(body)
        .send()
        .await;

    let upstream_response = match upstream_request {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(error = %err, url = %upstream_url, "upstream request failed");
            return error_response(StatusCode::BAD_GATEWAY, "ccc could not reach upstream");
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

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
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
    use axum::http::HeaderValue;

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

//! `ccc dashboard`: a page of recent sessions' costs, served from this machine only.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use ccc_ledger::{Ledger, SessionTotal};
use tokio::net::TcpListener;

use crate::LEDGER_FILE;

/// Sessions to show.
const SESSIONS_SHOWN: usize = 50;

pub(crate) async fn serve(address: SocketAddr, state_dir: &Path) -> anyhow::Result<()> {
    let ledger = Arc::new(Ledger::open(&state_dir.join(LEDGER_FILE)).context("opening ledger")?);
    let app = Router::new().route("/", get(page)).with_state(ledger);
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("binding {address}"))?;
    tracing::info!(%address, "serving dashboard");
    axum::serve(listener, app)
        .await
        .context("serving dashboard")
}

async fn page(State(ledger): State<Arc<Ledger>>) -> Response {
    match ledger.recent_sessions(SESSIONS_SHOWN) {
        Ok(sessions) => Html(render(&sessions)).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

/// Session ids come from request headers, so everything shown is escaped.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render(sessions: &[SessionTotal]) -> String {
    let total: f64 = sessions.iter().map(|session| session.cost_usd).sum();
    let highest = sessions
        .iter()
        .map(|session| session.cost_usd)
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    let mut rows = String::new();
    for session in sessions {
        let width = 100.0 * session.cost_usd / highest;
        rows.push_str(&format!(
            "<tr><td class=id>{}</td><td>{}</td><td>{}</td><td>{}</td>\
             <td class=cost>${:.4}<div class=bar style=\"width:{width:.1}%\"></div></td></tr>",
            escape(&session.session),
            session.calls,
            session.context_tokens,
            session.output_tokens,
            session.cost_usd,
        ));
    }
    if sessions.is_empty() {
        rows.push_str("<tr><td colspan=5>No sessions recorded yet.</td></tr>");
    }
    format!(
        "<!doctype html><html lang=en><meta charset=utf-8><title>ccc</title>\
         <meta name=viewport content=\"width=device-width,initial-scale=1\">\
         <style>\
         :root{{color-scheme:light dark;--bar:#4a7fd4;--line:#8884}}\
         body{{font:14px system-ui;margin:2rem;}}\
         table{{border-collapse:collapse;width:100%}}\
         th,td{{padding:.4rem .6rem;border-bottom:1px solid var(--line);text-align:right}}\
         th:first-child,td.id{{text-align:left;font-family:ui-monospace,monospace}}\
         td.cost{{min-width:9rem}}\
         .bar{{height:3px;background:var(--bar);margin-top:3px;margin-left:auto}}\
         </style>\
         <h1>Recent sessions</h1>\
         <p>${total:.2} across {} sessions, at API list prices. On a subscription these figures \
         are notional.</p>\
         <table><tr><th>Session</th><th>Calls</th><th>Context tokens</th><th>Output</th>\
         <th>Cost</th></tr>{rows}</table>",
        sessions.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: &str, cost: f64) -> SessionTotal {
        SessionTotal {
            session: id.to_owned(),
            calls: 3,
            cost_usd: cost,
            ..SessionTotal::default()
        }
    }

    #[test]
    fn shows_totals_and_scales_bars_to_the_highest_cost() {
        let page = render(&[session("a", 2.0), session("b", 1.0)]);
        assert!(page.contains("$3.00 across 2 sessions"), "{page}");
        assert!(page.contains("width:100.0%"), "{page}");
        assert!(page.contains("width:50.0%"), "{page}");
    }

    #[test]
    fn session_ids_are_escaped() {
        let page = render(&[session("<script>alert(1)</script>", 1.0)]);
        assert!(!page.contains("<script>"), "{page}");
        assert!(page.contains("&lt;script&gt;"), "{page}");
    }

    #[test]
    fn an_empty_ledger_says_so() {
        assert!(render(&[]).contains("No sessions recorded yet."));
    }
}

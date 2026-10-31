//! How `ccc run` points each coding agent at the proxy.
//!
//! Only Claude Code's traffic is compiled. Other clients speak other APIs, so they get
//! forwarding and traffic capture, and are told plainly that nothing is redacted or pruned.

use std::net::SocketAddr;
use std::path::Path;

use tokio::process::Command;

const ANTHROPIC_API: &str = "https://api.anthropic.com";
const GEMINI_API: &str = "https://generativelanguage.googleapis.com";
/// Includes `/v1` because Codex's base URL replaces that whole prefix. With a ChatGPT login,
/// pass `--upstream https://chatgpt.com/backend-api/codex` instead.
const OPENAI_API: &str = "https://api.openai.com/v1";

/// A coding agent `ccc run` knows how to configure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Client {
    /// Claude Code, or anything else that reads `ANTHROPIC_BASE_URL`.
    Anthropic,
    /// Gemini CLI.
    Gemini,
    /// OpenAI Codex CLI.
    Codex,
}

impl Client {
    /// Recognizes a client from the command about to run.
    pub(crate) fn for_command(command: &[String]) -> Self {
        let program = command
            .first()
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        match program {
            "gemini" => Self::Gemini,
            "codex" => Self::Codex,
            _ => Self::Anthropic,
        }
    }

    /// Where this client's traffic goes by default.
    pub(crate) fn default_upstream(self) -> &'static str {
        match self {
            Self::Anthropic => ANTHROPIC_API,
            Self::Gemini => GEMINI_API,
            Self::Codex => OPENAI_API,
        }
    }

    /// Points a command at the proxy at `address`. Runs before the user's own arguments.
    pub(crate) fn configure(self, command: &mut Command, address: SocketAddr) {
        let url = format!("http://{address}");
        match self {
            Self::Anthropic => {
                command.env("ANTHROPIC_BASE_URL", url);
            }
            Self::Gemini => {
                command.env("GOOGLE_GEMINI_BASE_URL", url);
            }
            // Codex documents no environment variable for this, only the config key.
            Self::Codex => {
                command.args(["-c", &format!("openai_base_url=\"{url}\"")]);
            }
        }
    }

    /// What the user should know about how this client's traffic is handled.
    pub(crate) fn notice(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => None,
            Self::Gemini | Self::Codex => Some(
                "ccc: forwarding this client's traffic unchanged. Redaction, pruning and cost \
                 tracking only apply to Claude Code.",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn recognizes_clients_by_program_name() {
        assert_eq!(
            Client::for_command(&command(&["claude", "-p"])),
            Client::Anthropic
        );
        assert_eq!(
            Client::for_command(&command(&["/usr/bin/gemini"])),
            Client::Gemini
        );
        assert_eq!(
            Client::for_command(&command(&["codex", "exec"])),
            Client::Codex
        );
        assert_eq!(Client::for_command(&command(&["other"])), Client::Anthropic);
        assert_eq!(Client::for_command(&[]), Client::Anthropic);
    }

    async fn output_of(client: Client, script: &str) -> String {
        let mut child = Command::new("sh");
        client.configure(&mut child, SocketAddr::from(([127, 0, 0, 1], 9)));
        let output = child.args(["-c", script]).output().await.unwrap();
        String::from_utf8(output.stdout).unwrap()
    }

    #[tokio::test]
    async fn gemini_and_claude_are_pointed_at_the_proxy_by_environment() {
        let gemini = output_of(Client::Gemini, "printf %s \"$GOOGLE_GEMINI_BASE_URL\"").await;
        assert_eq!(gemini, "http://127.0.0.1:9");
        let claude = output_of(Client::Anthropic, "printf %s \"$ANTHROPIC_BASE_URL\"").await;
        assert_eq!(claude, "http://127.0.0.1:9");
    }

    #[test]
    fn only_clients_that_are_not_compiled_get_a_notice() {
        assert!(Client::Anthropic.notice().is_none());
        assert!(Client::Codex.notice().unwrap().contains("unchanged"));
        assert!(Client::Gemini.notice().unwrap().contains("unchanged"));
    }
}

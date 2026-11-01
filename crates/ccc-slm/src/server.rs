//! Models served by something already running on this machine, such as Ollama or LM Studio,
//! through the OpenAI-compatible API they share.

use std::time::Duration;

use serde_json::{Value, json};

use crate::model::{Completion, CompletionRequest, Embedder, ModelError, TextModel, normalized};

/// How long an embedding may take. They're small, so this only catches a stuck server.
const EMBED_TIMEOUT: Duration = Duration::from_secs(30);

/// A model behind an OpenAI-compatible server. Grammars aren't part of that API, so callers
/// have to tolerate free-form answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerModel {
    base_url: String,
    model: String,
}

impl ServerModel {
    /// A model called `model` on the server at `base_url`, such as
    /// `http://127.0.0.1:11434/v1`.
    #[must_use]
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
        }
    }

    fn post(&self, path: &str, body: &Value, timeout: Duration) -> Result<Value, ModelError> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .into();
        let mut response = agent
            .post(format!("{}{path}", self.base_url))
            .header("content-type", "application/json")
            .send(body.to_string())
            .map_err(|err| match err {
                ureq::Error::Timeout(_) => ModelError::DeadlineExceeded,
                other => ModelError::Inference(other.to_string()),
            })?;
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|err| ModelError::Inference(err.to_string()))?;
        serde_json::from_str(&text).map_err(|err| ModelError::Inference(err.to_string()))
    }
}

impl TextModel for ServerModel {
    fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": request.prompt }],
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "stream": false,
        });
        let answer = self.post("/chat/completions", &body, request.deadline)?;
        let text = answer
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| ModelError::Inference(format!("unexpected answer: {answer}")))?;
        let count = |key: &str| {
            answer
                .pointer(&format!("/usage/{key}"))
                .and_then(Value::as_u64)
                .and_then(|tokens| u32::try_from(tokens).ok())
                .unwrap_or(0)
        };
        Ok(Completion {
            text: text.to_owned(),
            prompt_tokens: count("prompt_tokens"),
            generated_tokens: count("completion_tokens"),
            ..Completion::default()
        })
    }
}

impl Embedder for ServerModel {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        let body = json!({ "model": self.model, "input": text });
        let answer = self.post("/embeddings", &body, EMBED_TIMEOUT)?;
        let vector = answer
            .pointer("/data/0/embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| ModelError::Inference(format!("unexpected answer: {answer}")))?
            .iter()
            .map(|number| number.as_f64().map(|value| value as f32))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| ModelError::Inference("a non-number in an embedding".to_owned()))?;
        Ok(normalized(vector))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// Answers one request, and hands back what it was sent.
    fn one_answer(status: &str, body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap());
        let status = status.to_owned();
        let (sender, seen) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut received = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                received.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&received);
                if let Some((head, rest)) = text.split_once("\r\n\r\n") {
                    let length: usize = head
                        .lines()
                        .find_map(|line| {
                            let lower = line.to_lowercase();
                            lower
                                .strip_prefix("content-length: ")
                                .map(|value| value.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if rest.len() >= length {
                        break;
                    }
                }
            }
            let reply = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).unwrap();
            sender
                .send(String::from_utf8_lossy(&received).into_owned())
                .unwrap();
        });
        (url, seen)
    }

    fn request(prompt: &str) -> CompletionRequest<'_> {
        CompletionRequest {
            prompt,
            grammar: None,
            max_tokens: 9,
            deadline: Duration::from_secs(5),
        }
    }

    #[test]
    fn completes_through_the_shared_api() {
        let (url, seen) = one_answer(
            "200 OK",
            r#"{"choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#,
        );
        let answer = ServerModel::new(&url, "gemma")
            .complete(&request("hi"))
            .unwrap();
        assert_eq!(answer.text, "hello");
        assert_eq!((answer.prompt_tokens, answer.generated_tokens), (3, 1));
        let sent = seen.recv().unwrap();
        assert!(sent.starts_with("POST /v1/chat/completions"), "{sent}");
        assert!(sent.contains(r#""model":"gemma""#), "{sent}");
    }

    #[test]
    fn embeddings_come_back_at_length_one() {
        let (url, _seen) = one_answer("200 OK", r#"{"data":[{"embedding":[3.0,4.0]}]}"#);
        let vector = ServerModel::new(&url, "e").embed("x").unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6 && (vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn failures_are_errors_and_never_panics() {
        let (url, _seen) = one_answer("500 Internal Server Error", "{}");
        assert!(ServerModel::new(&url, "m").complete(&request("x")).is_err());
        let (url, _seen) = one_answer("200 OK", r#"{"unexpected":true}"#);
        assert!(ServerModel::new(&url, "m").complete(&request("x")).is_err());
        let (url, _seen) = one_answer("200 OK", r#"{"data":[{"embedding":["x"]}]}"#);
        assert!(ServerModel::new(&url, "m").embed("x").is_err());
        assert!(
            ServerModel::new("http://127.0.0.1:1/v1", "m")
                .embed("x")
                .is_err()
        );
    }
}

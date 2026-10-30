use std::borrow::Cow;
use std::sync::Arc;

use serde_json::{Map, Value};

/// The top-level key holding the conversation.
const MESSAGES: &str = "messages";

/// A request body that isn't a JSON object with a `messages` array.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParseRequestError {
    /// The body is not valid JSON.
    #[error("request body is not valid json")]
    Json(#[from] serde_json::Error),
    /// The body is JSON but has no `messages` array at the top level.
    #[error("request body has no messages array")]
    NoMessages,
}

/// A parsed `/v1/messages` request.
///
/// Messages are individually reference counted, so cloning a request is cheap and
/// editing one message copies only that message.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// All top-level fields, with `messages` left as an empty array to mark its position.
    fields: Map<String, Value>,
    messages: Vec<Arc<Value>>,
}

impl Request {
    /// Parses a request body.
    ///
    /// # Errors
    ///
    /// Returns [`ParseRequestError`] if `body` isn't JSON or has no `messages` array.
    pub fn parse(body: &[u8]) -> Result<Self, ParseRequestError> {
        let Value::Object(mut fields) = serde_json::from_slice(body)? else {
            return Err(ParseRequestError::NoMessages);
        };
        let Some(Value::Array(messages)) = fields.get_mut(MESSAGES).map(Value::take) else {
            return Err(ParseRequestError::NoMessages);
        };
        fields.insert(MESSAGES.to_owned(), Value::Array(Vec::new()));
        Ok(Self {
            fields,
            messages: messages.into_iter().map(Arc::new).collect(),
        })
    }

    /// Builds a request from already-parsed parts, placing `messages` after `fields`.
    #[must_use]
    pub fn from_parts(mut fields: Map<String, Value>, messages: Vec<Arc<Value>>) -> Self {
        fields.insert(MESSAGES.to_owned(), Value::Array(Vec::new()));
        Self { fields, messages }
    }

    /// Serializes the request to compact JSON, with keys in their original order.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4096);
        out.push(b'{');
        for (index, (key, value)) in self.fields.iter().enumerate() {
            if index > 0 {
                out.push(b',');
            }
            write_json(&mut out, key);
            out.push(b':');
            if key == MESSAGES {
                out.push(b'[');
                for (position, message) in self.messages.iter().enumerate() {
                    if position > 0 {
                        out.push(b',');
                    }
                    write_json(&mut out, message.as_ref());
                }
                out.push(b']');
            } else {
                write_json(&mut out, value);
            }
        }
        out.push(b'}');
        out
    }

    /// The conversation, oldest message first.
    #[must_use]
    pub fn messages(&self) -> &[Arc<Value>] {
        &self.messages
    }

    /// The requested model, if present.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.fields.get("model").and_then(Value::as_str)
    }

    /// The value of a top-level field other than `messages`.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name).filter(|_| name != MESSAGES)
    }

    /// Replaces the value at `pointer` (RFC 6901) within top-level field `name`.
    /// Returns whether the target existed.
    pub fn set_in_field(&mut self, name: &str, pointer: &str, value: Value) -> bool {
        if name == MESSAGES {
            return false;
        }
        match self
            .fields
            .get_mut(name)
            .and_then(|field| field.pointer_mut(pointer))
        {
            Some(target) => {
                *target = value;
                true
            }
            None => false,
        }
    }

    /// Replaces message `index` entirely. Returns whether it existed.
    pub fn set_message(&mut self, index: usize, message: Arc<Value>) -> bool {
        match self.messages.get_mut(index) {
            Some(slot) => {
                *slot = message;
                true
            }
            None => false,
        }
    }

    /// Replaces the value at `pointer` (RFC 6901) within message `index`.
    ///
    /// Returns whether the target existed. Only the edited message is copied, and only
    /// when another clone of this request still shares it.
    pub fn set_in_message(&mut self, index: usize, pointer: &str, value: Value) -> bool {
        let Some(message) = self.messages.get_mut(index) else {
            return false;
        };
        match Arc::make_mut(message).pointer_mut(pointer) {
            Some(target) => {
                *target = value;
                true
            }
            None => false,
        }
    }
}

fn write_json(out: &mut Vec<u8>, value: &(impl serde::Serialize + ?Sized)) {
    serde_json::to_writer(out, value).expect("json values always serialize to a vec");
}

/// Joins the text parts of a `tool_result` block's `content`, which may be a string or
/// a list of parts. Returns `None` if it holds anything other than text.
#[must_use]
pub fn text_of_tool_result(content: &Value) -> Option<Cow<'_, str>> {
    match content {
        Value::String(text) => Some(Cow::Borrowed(text)),
        Value::Array(parts) => {
            let mut joined = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return None;
                }
                joined.push_str(part.get("text").and_then(Value::as_str)?);
            }
            Some(Cow::Owned(joined))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_bodies_without_messages() {
        for body in [&b"[]"[..], b"{}", b"{\"messages\":3}", b"not json"] {
            assert!(Request::parse(body).is_err(), "{body:?}");
        }
    }

    /// Number spellings here are those `JSON.stringify` produces, which is what Claude
    /// Code sends. `serde_json` would normalize a spelling like `1e3` to `1e+3`.
    #[test]
    fn round_trips_order_escapes_numbers_and_unknown_shapes() {
        let body = concat!(
            r#"{"model":"m","z":1.0,"a":1e+21,"big":12345678901234567890123,"#,
            r#""messages":[{"role":"system","content":[{"type":"tool_addition","x":null}]},"#,
            r#"{"role":"user","content":"é \n \t \" \\ \u001b 😀"}],"#,
            r#""stream":true,"metadata":{"user_id":"{\"a\":1}"}}"#
        );
        let request = Request::parse(body.as_bytes()).unwrap();
        assert_eq!(String::from_utf8(request.to_bytes()).unwrap(), body);
    }

    #[test]
    fn messages_position_is_kept_when_not_first_or_last() {
        let request = Request::parse(br#"{"a":1,"messages":[],"b":2}"#).unwrap();
        assert_eq!(request.to_bytes(), br#"{"a":1,"messages":[],"b":2}"#);
    }

    #[test]
    fn editing_one_clone_leaves_the_other_and_untouched_messages_shared() {
        let original = Request::parse(
            br#"{"messages":[{"role":"user","content":"keep"},{"role":"user","content":"old"}]}"#,
        )
        .unwrap();
        let mut edited = original.clone();

        assert!(edited.set_in_message(1, "/content", json!("new")));

        assert_eq!(original.messages()[1]["content"], "old");
        assert_eq!(edited.messages()[1]["content"], "new");
        assert!(Arc::ptr_eq(&original.messages()[0], &edited.messages()[0]));
    }

    #[test]
    fn set_in_field_edits_fields_but_never_messages() {
        let mut request =
            Request::parse(br#"{"system":[{"text":"a"}],"messages":[{"role":"user"}]}"#).unwrap();
        assert!(request.set_in_field("system", "/0/text", json!("b")));
        assert_eq!(request.field("system").unwrap()[0]["text"], "b");
        assert!(!request.set_in_field("messages", "", json!([])));
        assert!(!request.set_in_field("absent", "", json!(1)));
        assert_eq!(request.messages().len(), 1);
    }

    #[test]
    fn set_in_message_reports_missing_targets() {
        let mut request = Request::parse(br#"{"messages":[{"role":"user"}]}"#).unwrap();
        assert!(!request.set_in_message(0, "/content/0", json!(1)));
        assert!(!request.set_in_message(9, "/role", json!(1)));
    }

    #[test]
    fn text_of_tool_result_handles_strings_lists_and_images() {
        assert_eq!(text_of_tool_result(&json!("a")).unwrap(), "a");
        let parts = json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]);
        assert_eq!(text_of_tool_result(&parts).unwrap(), "ab");
        assert!(text_of_tool_result(&json!([{"type":"image"}])).is_none());
    }
}

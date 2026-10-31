//! Tells what kind of call a request is from its shape alone.

use ccc_wire::Request;
use serde_json::Value;

/// How many messages a call can have and still count as having no real history. Claude
/// Code's one-shot calls send a user message and sometimes a system-role reminder.
const MAX_BACKGROUND_MESSAGES: usize = 2;

/// What a call is doing, judged by its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CallKind {
    /// A one-shot helper call with no tools and no history, such as naming a session.
    Background,
    /// The user just said something.
    UserTurn,
    /// The model is continuing after tool results.
    ToolLoop,
}

/// Classifies a request. A subagent's calls look like any other agent's here, because
/// nothing in a request's shape separates them from the main conversation.
#[must_use]
pub fn classify(request: &Request) -> CallKind {
    let no_tools = request
        .field("tools")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    if no_tools && request.messages().len() <= MAX_BACKGROUND_MESSAGES {
        return CallKind::Background;
    }
    let last_user = request
        .messages()
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"));
    let ends_with_tool_result = last_user
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks
                .iter()
                .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
        });
    if ends_with_tool_result {
        CallKind::ToolLoop
    } else {
        CallKind::UserTurn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(body: Value) -> Request {
        Request::parse(body.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn classifies_the_shapes_claude_code_sends() {
        let title = json!({ "messages": [{ "role": "user", "content": "name this" }] });
        assert_eq!(classify(&request(title)), CallKind::Background);

        let turn = json!({ "tools": [{ "name": "Bash" }],
            "messages": [{ "role": "user", "content": "fix it" }] });
        assert_eq!(classify(&request(turn)), CallKind::UserTurn);

        let tool = json!({ "tools": [{ "name": "Bash" }], "messages": [
            { "role": "user", "content": "fix it" },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "a" }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "a" }] },
            { "role": "system", "content": "reminder" },
        ] });
        assert_eq!(classify(&request(tool)), CallKind::ToolLoop);
    }

    #[test]
    fn a_long_conversation_without_tools_is_not_background() {
        let chat = json!({ "messages": [
            { "role": "user", "content": "a" },
            { "role": "assistant", "content": "b" },
            { "role": "user", "content": "c" },
        ] });
        assert_eq!(classify(&request(chat)), CallKind::UserTurn);
    }
}

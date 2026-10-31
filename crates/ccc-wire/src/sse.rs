//! Splitting a server-sent event stream into events.

use serde_json::Value;

/// The length of the first complete event in `buffer`, including its blank line.
#[must_use]
pub fn frame_length(buffer: &[u8]) -> Option<usize> {
    let lf = find(buffer, b"\n\n").map(|at| at + 2);
    let crlf = find(buffer, b"\r\n\r\n").map(|at| at + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The JSON payload of an event, if it has one.
#[must_use]
pub fn event_json(frame: &[u8]) -> Option<Value> {
    let text = std::str::from_utf8(frame).ok()?;
    let data = text
        .lines()
        .find_map(|line| line.strip_prefix("data:"))
        .map(str::trim)?;
    serde_json::from_str(data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_first_event_with_either_line_ending() {
        assert_eq!(frame_length(b"a\n\nb\n\n"), Some(3));
        assert_eq!(frame_length(b"a\r\n\r\nb"), Some(5));
        assert_eq!(frame_length(b"incomplete\n"), None);
    }

    #[test]
    fn reads_event_payloads() {
        let event = event_json(b"event: x\ndata: {\"type\":\"ping\"}\n\n").unwrap();
        assert_eq!(event["type"], "ping");
        assert_eq!(event_json(b": comment\n\n"), None);
        assert_eq!(event_json(b"data: not json\n\n"), None);
    }
}

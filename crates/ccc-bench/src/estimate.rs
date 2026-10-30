//! Estimates token counts from content, calibrated against recorded API usage.

use std::sync::Arc;

use serde_json::Value;

use crate::transcript::{Call, Session, THINKING_TOKENS_KEY};

/// Features of content that drive token counts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Size {
    /// Characters of visible text, tool inputs and definitions.
    pub(crate) text: f64,
    /// Number of images.
    pub(crate) images: f64,
    /// Hidden thinking tokens, as recorded when the model produced them.
    pub(crate) thinking: f64,
    /// Content blocks, each of which adds framing tokens beyond its characters.
    pub(crate) blocks: f64,
}

impl std::ops::Add for Size {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            text: self.text + other.text,
            images: self.images + other.images,
            thinking: self.thinking + other.thinking,
            blocks: self.blocks + other.blocks,
        }
    }
}

impl std::ops::Sub for Size {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            text: self.text - other.text,
            images: self.images - other.images,
            thinking: self.thinking - other.thinking,
            blocks: self.blocks - other.blocks,
        }
    }
}

impl std::iter::Sum for Size {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |total, size| total + size)
    }
}

/// Where a block of text came from, for reporting what fills the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Source {
    SystemAndTools,
    UserText,
    SystemRoleText,
    AssistantText,
    ToolUse,
    ToolResult,
    Other,
}

/// Calls `visit` with the character count of every piece of text in `message`, plus
/// image and signature totals in the returned [`Size`].
pub(crate) fn measure_message(message: &Value, mut visit: impl FnMut(Source, f64)) -> Size {
    let role = message.get("role").and_then(Value::as_str).unwrap_or("");
    let text_source = match role {
        "assistant" => Source::AssistantText,
        "system" => Source::SystemRoleText,
        _ => Source::UserText,
    };
    let mut size = Size {
        thinking: message
            .get(THINKING_TOKENS_KEY)
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        ..Size::default()
    };
    let mut add_text = |source: Source, chars: usize, size: &mut Size| {
        size.text += chars as f64;
        visit(source, chars as f64);
    };
    match message.get("content") {
        Some(Value::String(text)) => add_text(text_source, text.chars().count(), &mut size),
        Some(Value::Array(blocks)) => {
            size.blocks += blocks.len() as f64;
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                        add_text(text_source, text.chars().count(), &mut size);
                    }
                    Some("thinking") => {
                        let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                        add_text(text_source, text.chars().count(), &mut size);
                    }
                    Some("image") => size.images += 1.0,
                    Some("tool_use") => {
                        let chars = block.get("input").map_or(0, json_chars);
                        add_text(Source::ToolUse, chars, &mut size);
                    }
                    Some("tool_result") => match block.get("content") {
                        Some(Value::String(text)) => {
                            add_text(Source::ToolResult, text.chars().count(), &mut size);
                        }
                        Some(Value::Array(parts)) => {
                            for part in parts {
                                match part.get("type").and_then(Value::as_str) {
                                    Some("image") => size.images += 1.0,
                                    Some("text") => {
                                        let text =
                                            part.get("text").and_then(Value::as_str).unwrap_or("");
                                        add_text(
                                            Source::ToolResult,
                                            text.chars().count(),
                                            &mut size,
                                        );
                                    }
                                    _ => add_text(Source::ToolResult, json_chars(part), &mut size),
                                }
                            }
                        }
                        _ => {}
                    },
                    _ => add_text(Source::Other, json_chars(block), &mut size),
                }
            }
        }
        _ => {}
    }
    size
}

/// The size of everything before the messages: the system prompt and tool definitions.
pub(crate) fn measure_preamble(call: &Call) -> Size {
    let field_chars = |name: &str| call.request.field(name).map_or(0, json_chars);
    Size {
        text: (field_chars("system") + field_chars("tools")) as f64,
        ..Size::default()
    }
}

fn json_chars(value: &Value) -> usize {
    value.to_string().chars().count()
}

/// Token rates fitted from recorded usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rates {
    pub(crate) tokens_per_char: f64,
    pub(crate) tokens_per_image: f64,
    pub(crate) tokens_per_thinking_token: f64,
    pub(crate) tokens_per_block: f64,
}

/// The number of features in a [`Size`].
const FEATURES: usize = 4;

impl Rates {
    /// Used when there are too few calls to fit: 3.5 characters per token is a common
    /// rule of thumb for mixed code and prose, and everything else counts for nothing.
    pub(crate) const FALLBACK: Self = Self {
        tokens_per_char: 1.0 / 3.5,
        tokens_per_image: 0.0,
        tokens_per_thinking_token: 0.0,
        tokens_per_block: 0.0,
    };

    pub(crate) fn tokens(&self, size: Size) -> f64 {
        size.text * self.tokens_per_char
            + size.images * self.tokens_per_image
            + size.thinking * self.tokens_per_thinking_token
            + size.blocks * self.tokens_per_block
    }

    /// Fits rates to `(size, tokens)` pairs by least squares, keeping every rate
    /// non-negative by dropping any feature whose coefficient goes negative.
    pub(crate) fn fit(samples: &[(Size, f64)]) -> Self {
        let mut active = [true; FEATURES];
        for _ in 0..=FEATURES {
            let Some(solved) = solve_least_squares(samples, &mut active) else {
                return Self::FALLBACK;
            };
            let Some(negative) = solved.iter().position(|coefficient| *coefficient < 0.0) else {
                return Self {
                    tokens_per_char: solved[0],
                    tokens_per_image: solved[1],
                    tokens_per_thinking_token: solved[2],
                    tokens_per_block: solved[3],
                };
            };
            active[negative] = false;
        }
        Self::FALLBACK
    }
}

fn features(size: &Size) -> [f64; FEATURES] {
    [size.text, size.images, size.thinking, size.blocks]
}

fn solve_least_squares(
    samples: &[(Size, f64)],
    active: &mut [bool; FEATURES],
) -> Option<[f64; FEATURES]> {
    let mut normal = [[0.0_f64; FEATURES + 1]; FEATURES];
    for (size, target) in samples {
        let x = features(size);
        for row in 0..FEATURES {
            for column in 0..FEATURES {
                normal[row][column] += x[row] * x[column];
            }
            normal[row][FEATURES] += x[row] * target;
        }
    }
    // A feature that never occurs has no information, so it can't be solved for.
    for index in 0..FEATURES {
        if normal[index][index] == 0.0 {
            active[index] = false;
        }
    }
    if !active.iter().any(|is_active| *is_active) {
        return None;
    }
    for (index, is_active) in active.iter().enumerate() {
        if !is_active {
            normal[index] = [0.0; FEATURES + 1];
            normal[index][index] = 1.0;
            for (row, equation) in normal.iter_mut().enumerate() {
                if row != index {
                    equation[index] = 0.0;
                }
            }
        }
    }
    gauss_solve(normal)
}

fn gauss_solve(mut matrix: [[f64; FEATURES + 1]; FEATURES]) -> Option<[f64; FEATURES]> {
    for pivot in 0..FEATURES {
        let best = (pivot..FEATURES)
            .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))?;
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        if divisor.abs() < f64::EPSILON {
            return None;
        }
        let pivot_row = matrix[pivot];
        for (row, equation) in matrix.iter_mut().enumerate() {
            if row != pivot {
                let factor = equation[pivot] / divisor;
                for column in pivot..=FEATURES {
                    equation[column] -= factor * pivot_row[column];
                }
            }
        }
    }
    let mut solved = [0.0; FEATURES];
    for (index, value) in solved.iter_mut().enumerate() {
        *value = matrix[index][FEATURES] / matrix[index][index];
    }
    Some(solved)
}

/// One calibration sample per call: how its content changed against how its context grew.
///
/// The difference between consecutive recorded context sizes is the exact token count of
/// whatever changed, independent of how the cache behaved. Each sample isolates a small
/// piece of content, where whole contexts would overlap heavily and let a few long
/// sessions dominate. Differences also telescope, so fitted rates still sum to the right
/// total, which the report then checks against sessions held out of the fit.
pub(crate) fn samples(session: &Session) -> Vec<(Size, f64)> {
    let mut found = Vec::with_capacity(session.calls.len());
    let mut previous: Option<&Call> = None;
    for call in &session.calls {
        let messages = call.request.messages();
        let context = call.usage.context() as f64;
        let measure_all = |messages: &[Arc<Value>]| -> Size {
            messages.iter().map(|m| measure_message(m, |_, _| {})).sum()
        };
        let sample = match previous {
            None => Some((measure_preamble(call) + measure_all(messages), context)),
            // A model switch can drop thinking from the bill, which isn't a content change.
            Some(before) if same_preamble(before, call) && before.model() == call.model() => {
                let old = before.request.messages();
                let shared = messages
                    .iter()
                    .zip(old)
                    .take_while(|(new, old)| Arc::ptr_eq(new, old) || new == old)
                    .count();
                let change = measure_all(&messages[shared..]) - measure_all(&old[shared..]);
                Some((change, context - before.usage.context() as f64))
            }
            _ => None,
        };
        found.extend(sample);
        previous = Some(call);
    }
    found
}

fn same_preamble(a: &Call, b: &Call) -> bool {
    a.request.field("system") == b.request.field("system")
        && a.request.field("tools") == b.request.field("tools")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn measures_every_block_shape() {
        let message = json!({ "role": "assistant", "ccc_thinking_tokens": 5, "content": [
            { "type": "text", "text": "abc" },
            { "type": "thinking", "thinking": "", "signature": "12345" },
            { "type": "tool_use", "name": "Bash", "input": { "a": 1 } },
            { "type": "image", "source": {} },
            { "type": "tool_result", "content": "wxyz" },
            { "type": "tool_result", "content": [
                { "type": "text", "text": "pq" },
                { "type": "image", "source": {} },
            ] },
        ] });
        let mut by_source = std::collections::BTreeMap::new();
        let size = measure_message(&message, |source, chars| {
            *by_source.entry(source).or_insert(0.0) += chars;
        });
        assert_eq!(size.images, 2.0);
        assert_eq!(size.thinking, 5.0);
        assert_eq!(by_source[&Source::AssistantText], 3.0);
        assert_eq!(by_source[&Source::ToolResult], 6.0);
        assert_eq!(by_source[&Source::ToolUse], 7.0);
    }

    #[test]
    fn counts_characters_not_bytes() {
        let message = json!({ "role": "user", "content": "éé" });
        assert_eq!(measure_message(&message, |_, _| {}).text, 2.0);
    }

    #[test]
    fn fit_recovers_known_rates() {
        let truth = Rates {
            tokens_per_char: 0.25,
            tokens_per_image: 1500.0,
            tokens_per_thinking_token: 0.5,
            tokens_per_block: 9.0,
        };
        let samples: Vec<(Size, f64)> = (1..50)
            .map(|i| {
                let i = f64::from(i);
                let size = Size {
                    text: 1000.0 * i,
                    images: (i * 7.0) % 11.0,
                    thinking: 300.0 * ((i * 3.0) % 17.0),
                    blocks: (i * 5.0) % 13.0,
                };
                (size, truth.tokens(size))
            })
            .collect();
        let fitted = Rates::fit(&samples);
        assert!((fitted.tokens_per_char - 0.25).abs() < 1e-6, "{fitted:?}");
        assert!(
            (fitted.tokens_per_image - 1500.0).abs() < 1e-3,
            "{fitted:?}"
        );
        assert!(
            (fitted.tokens_per_thinking_token - 0.5).abs() < 1e-6,
            "{fitted:?}"
        );
        assert!((fitted.tokens_per_block - 9.0).abs() < 1e-3, "{fitted:?}");
    }

    #[test]
    fn fit_never_returns_a_negative_rate() {
        // Images anti-correlate with tokens here, which unconstrained least squares
        // would express as a negative price per image.
        let samples: Vec<(Size, f64)> = (1..30)
            .map(|i| {
                let i = f64::from(i);
                let size = Size {
                    text: 100.0 * i,
                    images: 30.0 - i,
                    thinking: 0.0,
                    blocks: 0.0,
                };
                (size, 25.0 * i)
            })
            .collect();
        let fitted = Rates::fit(&samples);
        assert!(fitted.tokens_per_image >= 0.0, "{fitted:?}");
        assert!(fitted.tokens_per_char >= 0.0, "{fitted:?}");
    }

    #[test]
    fn fit_handles_features_that_never_occur() {
        let samples = [(
            Size {
                text: 4000.0,
                ..Size::default()
            },
            1000.0,
        )];
        let fitted = Rates::fit(&samples);
        assert!((fitted.tokens_per_char - 0.25).abs() < 1e-9, "{fitted:?}");
    }

    #[test]
    fn samples_measure_content_change_against_context_growth() {
        use crate::transcript::Usage;
        use ccc_wire::Request;
        use serde_json::json;
        let message = |text: &str| Arc::new(json!({ "role": "user", "content": text }));
        let (old, new, reminder) = (message("aaaa"), message("bb"), message("c"));
        let call = |messages: Vec<Arc<Value>>, read: u64, write: u64| Call {
            at_millis: 0,
            request: Request::from_parts(serde_json::Map::new(), messages),
            usage: Usage {
                cache_read: read,
                cache_write_1h: write,
                ..Usage::default()
            },
        };
        let session = Session {
            path: "s".into(),
            calls: vec![
                call(vec![Arc::clone(&old), Arc::clone(&reminder)], 0, 12),
                // Cache behaviour differs on purpose: only the size change should matter.
                call(vec![Arc::clone(&old), Arc::clone(&new)], 1, 12),
            ],
        };

        let found = samples(&session);

        assert_eq!(found.len(), 2);
        assert_eq!(found[0].0.text, 5.0);
        assert_eq!(found[0].1, 12.0);
        assert_eq!(found[1].0.text, 1.0, "added two characters, removed one");
        assert_eq!(found[1].1, 1.0);
    }

    #[test]
    fn fit_falls_back_without_data() {
        assert_eq!(Rates::fit(&[]), Rates::FALLBACK);
    }
}

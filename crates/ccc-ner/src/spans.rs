//! The model-independent parts: splitting words and choosing among scored spans.

use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

/// How GLiNER's own tooling splits text into words.
static WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("word pattern is valid"));

/// A word and where it sits in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word<'a> {
    /// The word as written.
    pub text: &'a str,
    /// Its byte range in the original text.
    pub range: Range<usize>,
}

/// Splits `text` into words.
#[must_use]
pub fn split_words(text: &str) -> Vec<Word<'_>> {
    WORD.find_iter(text)
        .map(|found| Word {
            text: found.as_str(),
            range: found.range(),
        })
        .collect()
}

/// A candidate: a run of words given a label with some confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// The first word's index.
    pub start: usize,
    /// The last word's index, inclusive.
    pub end: usize,
    /// Which label.
    pub label: usize,
    /// Confidence from 0 to 1.
    pub score: f32,
}

/// Keeps the most confident spans that don't overlap one another.
#[must_use]
pub fn choose_spans(mut candidates: Vec<Span>) -> Vec<Span> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut chosen: Vec<Span> = Vec::new();
    for span in candidates {
        let overlaps = chosen
            .iter()
            .any(|kept| span.start <= kept.end && kept.start <= span.end);
        if !overlaps {
            chosen.push(span);
        }
    }
    chosen.sort_by_key(|span| span.start);
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_and_keeps_their_positions() {
        let text = "Ask Jane-Doe, please.";
        let words = split_words(text);
        let texts: Vec<&str> = words.iter().map(|word| word.text).collect();
        assert_eq!(texts, ["Ask", "Jane-Doe", ",", "please", "."]);
        assert_eq!(&text[words[1].range.clone()], "Jane-Doe");
    }

    #[test]
    fn handles_multibyte_text() {
        let text = "José visited Zürich";
        let words = split_words(text);
        assert_eq!(words.len(), 3);
        assert_eq!(&text[words[2].range.clone()], "Zürich");
    }

    #[test]
    fn the_more_confident_of_overlapping_spans_wins() {
        let span = |start, end, score| Span {
            start,
            end,
            label: 0,
            score,
        };
        let chosen = choose_spans(vec![span(0, 1, 0.6), span(1, 2, 0.9), span(4, 4, 0.7)]);
        assert_eq!(chosen.len(), 2);
        assert_eq!(chosen[0].start, 1);
        assert_eq!(chosen[1].start, 4);
    }
}

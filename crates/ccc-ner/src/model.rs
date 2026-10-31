use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::spans::{Span, Word, choose_spans, split_words};

/// The longest entity, in words, that this GLiNER model scores.
const MAX_WIDTH: usize = 12;
/// How many words to read at once. The model was trained on inputs up to about this long.
const CHUNK_WORDS: usize = 300;
/// Below this confidence a span isn't treated as a name.
const MIN_SCORE: f32 = 0.7;
const ENTITY_TOKEN: &str = "<<ENT>>";
const SEPARATOR_TOKEN: &str = "<<SEP>>";

/// The model or its tokenizer couldn't be loaded.
#[derive(Debug, thiserror::Error)]
#[error("loading NER model from {dir:?}: {reason}")]
pub struct LoadNerError {
    dir: PathBuf,
    reason: String,
}

/// Running the model failed.
#[derive(Debug, thiserror::Error)]
#[error("running NER: {0}")]
pub struct NerError(String);

/// A name found in text.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    /// The name as written.
    pub text: String,
    /// The label it matched.
    pub label: String,
    /// Confidence from 0 to 1.
    pub score: f32,
}

/// A loaded GLiNER model.
#[derive(Debug)]
pub struct NerModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl NerModel {
    /// Loads `model_quantized.onnx` and `tokenizer.json` from `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadNerError`] if either can't be loaded.
    pub fn load(dir: &Path) -> Result<Self, LoadNerError> {
        let failed = |reason: String| LoadNerError {
            dir: dir.to_owned(),
            reason,
        };
        let session = Session::builder()
            .map_err(|err| failed(err.to_string()))?
            .commit_from_file(dir.join("model_quantized.onnx"))
            .map_err(|err| failed(err.to_string()))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|err| failed(err.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Finds names in `text` matching any of `labels`.
    ///
    /// # Errors
    ///
    /// Returns [`NerError`] if tokenizing or the model fails.
    pub fn find(&self, text: &str, labels: &[&str]) -> Result<Vec<Entity>, NerError> {
        let words = split_words(text);
        let mut found = Vec::new();
        for chunk in words.chunks(CHUNK_WORDS) {
            let spans = self.score_chunk(chunk, labels)?;
            for span in choose_spans(spans) {
                let range = chunk[span.start].range.start..chunk[span.end].range.end;
                found.push(Entity {
                    text: text[range].to_owned(),
                    label: labels[span.label].to_owned(),
                    score: span.score,
                });
            }
        }
        Ok(found)
    }

    fn score_chunk(&self, words: &[Word<'_>], labels: &[&str]) -> Result<Vec<Span>, NerError> {
        let failed = |err: String| NerError(err);
        let mut pretokenized: Vec<&str> = Vec::with_capacity(words.len() + labels.len() * 2 + 1);
        for label in labels {
            pretokenized.push(ENTITY_TOKEN);
            pretokenized.push(label);
        }
        pretokenized.push(SEPARATOR_TOKEN);
        let prompt_words = pretokenized.len();
        pretokenized.extend(words.iter().map(|word| word.text));

        let encoding = self
            .tokenizer
            .encode(pretokenized, true)
            .map_err(|err| failed(err.to_string()))?;
        let tokens = encoding.get_ids().len();

        // Marks the first piece of each text word with its number, counted from one.
        let mut words_mask = vec![0_i64; tokens];
        let mut previous = None;
        for (position, word) in encoding.get_word_ids().iter().enumerate() {
            let Some(word) = word.map(|word| word as usize) else {
                continue;
            };
            if word >= prompt_words && previous != Some(word) {
                words_mask[position] = (word - prompt_words + 1) as i64;
            }
            previous = Some(word);
        }

        let count = words.len();
        let spans = count * MAX_WIDTH;
        let span_idx = Array3::from_shape_fn((1, spans, 2), |(_, index, side)| {
            let start = index / MAX_WIDTH;
            let width = index % MAX_WIDTH;
            let end = (start + width).min(count.saturating_sub(1));
            if side == 0 { start as i64 } else { end as i64 }
        });
        let span_mask = Array2::from_shape_fn((1, spans), |(_, index)| {
            index / MAX_WIDTH + index % MAX_WIDTH < count
        });

        let shape = |values: Vec<i64>| {
            Array2::from_shape_vec((1, tokens), values).map_err(|err| failed(err.to_string()))
        };
        let input_ids = shape(encoding.get_ids().iter().map(|id| i64::from(*id)).collect())?;
        let attention_mask = shape(
            encoding
                .get_attention_mask()
                .iter()
                .map(|m| i64::from(*m))
                .collect(),
        )?;
        let words_mask = shape(words_mask)?;
        let text_lengths = Array2::from_elem((1, 1), count as i64);

        let to_tensor = |err: ort::Error| failed(err.to_string());
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array(input_ids).map_err(to_tensor)?,
            "attention_mask" => Tensor::from_array(attention_mask).map_err(to_tensor)?,
            "words_mask" => Tensor::from_array(words_mask).map_err(to_tensor)?,
            "text_lengths" => Tensor::from_array(text_lengths).map_err(to_tensor)?,
            "span_idx" => Tensor::from_array(span_idx).map_err(to_tensor)?,
            "span_mask" => Tensor::from_array(span_mask).map_err(to_tensor)?,
        ];
        let mut session = self.session.lock().expect("lock poisoned");
        let outputs = session.run(inputs).map_err(to_tensor)?;
        let (dims, logits) = outputs[0].try_extract_tensor::<f32>().map_err(to_tensor)?;
        // Words by widths by labels, after the batch dimension.
        let (Some(&seen_words), Some(&seen_widths), Some(&seen_labels)) =
            (dims.get(1), dims.get(2), dims.get(3))
        else {
            return Err(NerError(format!("unexpected output shape {dims:?}")));
        };
        let (seen_words, seen_widths, seen_labels) = (
            seen_words as usize,
            seen_widths as usize,
            seen_labels as usize,
        );

        let mut candidates = Vec::new();
        for start in 0..seen_words.min(count) {
            for width in 0..seen_widths.min(count - start) {
                for label in 0..seen_labels.min(labels.len()) {
                    let at = (start * seen_widths + width) * seen_labels + label;
                    let score = sigmoid(logits[at]);
                    if score >= MIN_SCORE {
                        candidates.push(Span {
                            start,
                            end: start + width,
                            label,
                            score,
                        });
                    }
                }
            }
        }
        Ok(candidates)
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_maps_logits_to_probabilities() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    /// Needs the 180 MB model, so it only runs when pointed at one.
    /// `CCC_NER_DIR=~/.ccc/models/gliner cargo test -p ccc-ner -- --ignored`
    #[test]
    #[ignore = "needs a downloaded model named by CCC_NER_DIR"]
    fn finds_people_and_organizations() {
        let dir = std::env::var("CCC_NER_DIR").expect("CCC_NER_DIR should be set");
        let model = NerModel::load(Path::new(&dir)).unwrap();
        let found = model
            .find(
                "Please send the renewal to Maria Gonzalez at Northwind Traders by Friday.",
                &["person", "organization"],
            )
            .unwrap();
        let names: Vec<&str> = found.iter().map(|entity| entity.text.as_str()).collect();
        assert!(names.contains(&"Maria Gonzalez"), "{found:?}");
        assert!(names.contains(&"Northwind Traders"), "{found:?}");
    }
}

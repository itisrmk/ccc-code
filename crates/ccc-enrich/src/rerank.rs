//! Lets a small local model order candidate files, since finding candidates is cheap and telling
//! which ones matter is where search falls short.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::Duration;

use ccc_slm::{CompletionRequest, ModelError, TextModel};

/// A number or two is all it needs to say.
const MAX_ANSWER_TOKENS: u32 = 16;
/// Enough of a prompt to know what's being asked, without letting a pasted log crowd out the rest.
const MAX_PROMPT_CHARS: usize = 1200;
/// Limits the answer to up to three candidate numbers, so it can't ramble.
const ANSWER_GRAMMAR: &str = r#"root ::= num ("," num)? ("," num)?
num ::= [1-9] [0-9]?
"#;

/// A file to consider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The file's path.
    pub file: String,
    /// Names defined in it, as a clue to what it holds.
    pub symbols: Vec<String>,
}

fn rerank_prompt(prompt: &str, candidates: &[Candidate]) -> String {
    let asked: String = prompt.chars().take(MAX_PROMPT_CHARS).collect();
    let mut text = format!(
        "A developer asked a coding agent:\n\n{asked}\n\nWhich of these files will the agent most \
         likely need to read or change?\n\n"
    );
    for (index, candidate) in candidates.iter().enumerate() {
        // Writing to a String can't fail.
        let _ = write!(text, "{}. {}", index + 1, candidate.file);
        if !candidate.symbols.is_empty() {
            let _ = write!(text, " ({})", candidate.symbols.join(", "));
        }
        text.push('\n');
    }
    text.push_str("\nReply with up to three numbers, most likely first, separated by commas.");
    text
}

/// What an answer picks. Numbers out of range or repeated are dropped, since only the rest are
/// usable, and a model's slip shouldn't cost the whole answer.
fn parse_answer(answer: &str, count: usize) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    answer
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|number| (1..=count).contains(number))
        .map(|number| number - 1)
        .filter(|index| seen.insert(*index))
        .collect()
}

/// Orders `candidates` by what `model` thinks `prompt` needs, keeping `limit`. Anything it didn't
/// pick follows in the original order, so the result is never shorter than search alone gives.
///
/// # Errors
///
/// Returns [`ModelError`] if the model fails or misses `deadline`. Callers should then fall back
/// to the candidates as they were.
pub fn rerank(
    model: &dyn TextModel,
    prompt: &str,
    candidates: &[Candidate],
    limit: usize,
    deadline: Duration,
) -> Result<Vec<String>, ModelError> {
    let instruction = rerank_prompt(prompt, candidates);
    let completion = model.complete(&CompletionRequest {
        prompt: &instruction,
        grammar: Some(ANSWER_GRAMMAR),
        max_tokens: MAX_ANSWER_TOKENS,
        deadline,
    })?;
    let picked = parse_answer(&completion.text, candidates.len());
    let rest = (0..candidates.len()).filter(|index| !picked.contains(index));
    Ok(picked
        .iter()
        .copied()
        .chain(rest)
        .take(limit)
        .map(|index| candidates[index].file.clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_slm::Completion;

    #[derive(Debug)]
    struct Answers(Result<&'static str, ModelError>);

    impl TextModel for Answers {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            self.0.clone().map(|text| Completion {
                text: text.to_owned(),
                ..Completion::default()
            })
        }
    }

    fn candidates() -> Vec<Candidate> {
        ["a.rs", "b.rs", "c.rs", "d.rs"]
            .map(|file| Candidate {
                file: file.to_owned(),
                symbols: vec!["thing".to_owned()],
            })
            .to_vec()
    }

    fn ordered(answer: &'static str) -> Vec<String> {
        let second = Duration::from_secs(1);
        rerank(&Answers(Ok(answer)), "x", &candidates(), 3, second).unwrap()
    }

    #[test]
    fn follows_the_model_and_fills_in_from_search() {
        assert_eq!(ordered("3,1"), ["c.rs", "a.rs", "b.rs"]);
    }

    #[test]
    fn slips_cost_only_what_they_touch() {
        assert_eq!(ordered("9,2,2"), ["b.rs", "a.rs", "c.rs"]);
        assert_eq!(ordered("nonsense"), ["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn a_model_failure_is_reported_for_the_caller_to_fall_back() {
        let failed = rerank(
            &Answers(Err(ModelError::DeadlineExceeded)),
            "x",
            &candidates(),
            3,
            Duration::from_secs(1),
        );
        assert_eq!(failed, Err(ModelError::DeadlineExceeded));
    }

    #[test]
    fn the_prompt_names_each_file_and_stays_a_sensible_size() {
        let text = rerank_prompt(&"long ".repeat(10_000), &candidates());
        assert!(text.contains("1. a.rs (thing)"), "{text}");
        assert!(text.contains("4. d.rs"), "{text}");
        assert!(text.len() < 2000, "{}", text.len());
    }
}

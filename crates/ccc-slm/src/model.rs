use std::fmt::Debug;
use std::time::Duration;

/// What to ask a model for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionRequest<'a> {
    /// The instruction, as a user message.
    pub prompt: &'a str,
    /// A GBNF grammar the output must match.
    pub grammar: Option<&'a str>,
    /// The most tokens to generate.
    pub max_tokens: u32,
    /// How long the whole call may take before the model gives up.
    pub deadline: Duration,
}

/// What a model produced, with the timings needed to judge whether it's fast enough.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Completion {
    /// The generated text.
    pub text: String,
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Time spent reading the prompt.
    pub prompt_time: Duration,
    /// Tokens generated.
    pub generated_tokens: u32,
    /// Time spent generating.
    pub generation_time: Duration,
}

/// A model call that didn't produce a usable result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The call ran past its deadline.
    #[error("the model ran past its deadline")]
    DeadlineExceeded,
    /// The prompt doesn't fit in the model's context.
    #[error("the prompt has {tokens} tokens but the context holds {limit}")]
    PromptTooLong {
        /// Tokens in the prompt.
        tokens: usize,
        /// What the context can hold.
        limit: usize,
    },
    /// The model's worker thread has stopped.
    #[error("the model has stopped")]
    Stopped,
    /// The inference library reported a failure.
    #[error("inference failed: {0}")]
    Inference(String),
}

/// Anything that can complete a prompt. Calls block, so stages make them from their own
/// blocking task. Tests use a fake, since real weights are gigabytes.
pub trait TextModel: Send + Sync + Debug {
    /// Completes `request`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if the model fails or misses the deadline.
    fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError>;
}

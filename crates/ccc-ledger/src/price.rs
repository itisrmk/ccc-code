//! Dollar cost of a call at Anthropic's first-party list prices.

use crate::usage::Usage;

/// Cache write price for a 5-minute lifetime, as a multiple of the input price.
const WRITE_5M_MULTIPLE: f64 = 1.25;
/// Cache write price for a 1-hour lifetime, as a multiple of the input price.
const WRITE_1H_MULTIPLE: f64 = 2.0;
/// Haiku 5.5 charges a higher rate once a prompt passes this many tokens.
const HAIKU_LONG_PROMPT_TOKENS: u64 = 100_000;

/// List prices in dollars per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Prices {
    input: f64,
    cache_read: f64,
    output: f64,
}

/// Prices as of the Claude API docs bundled with Claude Code 2.1.308 (2026-10-06).
/// Returns `None` for models without a published first-party price here.
fn prices(model: &str, context_tokens: u64) -> Option<Prices> {
    let (input, cache_read, output) = match model {
        "claude-opus-5-5" => (4.0, 0.20, 20.0),
        "claude-opus-5" => (5.0, 0.50, 25.0),
        "claude-fable-5-1" => (10.0, 0.25, 50.0),
        "claude-fable-5" => (10.0, 1.00, 50.0),
        "claude-sonnet-5-5" | "claude-sonnet-5" => (2.0, 0.20, 10.0),
        "claude-haiku-5-5" if context_tokens > HAIKU_LONG_PROMPT_TOKENS => (0.50, 0.05, 2.50),
        "claude-haiku-5-5" => (0.10, 0.01, 0.50),
        _ => return None,
    };
    Some(Prices {
        input,
        cache_read,
        output,
    })
}

/// What one token costs on a model, in dollars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// An input token that wasn't cached.
    pub input: f64,
    /// An input token read from the cache.
    pub cache_read: f64,
    /// An input token written to the cache with a 5-minute lifetime.
    pub cache_write_5m: f64,
    /// An input token written to the cache with a 1-hour lifetime.
    pub cache_write_1h: f64,
    /// A generated token.
    pub output: f64,
}

/// Per-token rates for `model` at a given context size, or `None` if it has no known price.
#[must_use]
pub fn rates(model: &str, context_tokens: u64) -> Option<Rates> {
    let price = prices(model, context_tokens)?;
    let per_token = |dollars_per_million: f64| dollars_per_million / 1_000_000.0;
    Some(Rates {
        input: per_token(price.input),
        cache_read: per_token(price.cache_read),
        cache_write_5m: per_token(price.input * WRITE_5M_MULTIPLE),
        cache_write_1h: per_token(price.input * WRITE_1H_MULTIPLE),
        output: per_token(price.output),
    })
}

/// The dollar cost of `usage` on `model`, or `None` if the model has no known price.
#[must_use]
pub fn cost_usd(model: &str, usage: &Usage) -> Option<f64> {
    let rates = rates(model, usage.context())?;
    Some(
        usage.input as f64 * rates.input
            + usage.cache_read as f64 * rates.cache_read
            + usage.cache_write_5m as f64 * rates.cache_write_5m
            + usage.cache_write_1h as f64 * rates.cache_write_1h
            + usage.output as f64 * rates.output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_cost_claude_code_reported() {
        // A real one-word Opus 5.5 reply reported total_cost_usd 0.207576.
        let usage = Usage {
            input: 2,
            cache_read: 0,
            cache_write_5m: 0,
            cache_write_1h: 25_936,
            output: 4,
            thinking: 0,
        };
        let cost = cost_usd("claude-opus-5-5", &usage).unwrap();
        assert!((cost - 0.207_576).abs() < 1e-6, "{cost}");
    }

    #[test]
    fn unknown_models_have_no_price() {
        assert_eq!(cost_usd("<synthetic>", &Usage::default()), None);
    }

    #[test]
    fn haiku_switches_rate_above_its_prompt_threshold() {
        let short = Usage {
            input: 100_000,
            ..Usage::default()
        };
        let long = Usage {
            input: 100_001,
            ..Usage::default()
        };
        let short_cost = cost_usd("claude-haiku-5-5", &short).unwrap();
        let long_cost = cost_usd("claude-haiku-5-5", &long).unwrap();
        assert!(long_cost > short_cost * 4.0, "{short_cost} {long_cost}");
    }
}

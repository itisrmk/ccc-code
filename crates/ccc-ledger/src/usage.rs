use serde_json::Value;

/// Token counts the API reported for one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens that were neither read from nor written to the cache.
    pub input: u64,
    /// Input tokens read from the cache.
    pub cache_read: u64,
    /// Input tokens written to the cache with a 5-minute lifetime.
    pub cache_write_5m: u64,
    /// Input tokens written to the cache with a 1-hour lifetime.
    pub cache_write_1h: u64,
    /// Tokens generated.
    pub output: u64,
    /// The part of `output` that was hidden thinking.
    pub thinking: u64,
}

impl Usage {
    /// The size of the request's context in tokens.
    #[must_use]
    pub fn context(&self) -> u64 {
        self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h
    }

    /// Reads an API `usage` object. Missing fields count as zero.
    #[must_use]
    pub fn from_json(usage: &Value) -> Self {
        let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
        let written = count("cache_creation_input_tokens");
        let long = usage
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(written);
        Self {
            input: count("input_tokens"),
            cache_read: count("cache_read_input_tokens"),
            cache_write_5m: written - long,
            cache_write_1h: long,
            output: count("output_tokens"),
            thinking: usage
                .pointer("/output_tokens_details/thinking_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        }
    }

    /// Folds in a later report for the same call. Streams report input at the start and a
    /// running output total at the end. Input figures can't change once sent, and the final
    /// report repeats cache writes without their lifetimes, so only the first report counts.
    pub fn update(&mut self, later: &Self) {
        if self.context() == 0 {
            self.input = later.input;
            self.cache_read = later.cache_read;
            self.cache_write_5m = later.cache_write_5m;
            self.cache_write_1h = later.cache_write_1h;
        }
        if later.output > 0 {
            self.output = later.output;
        }
        if later.thinking > 0 {
            self.thinking = later.thinking;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn splits_cache_writes_by_lifetime() {
        let usage = Usage::from_json(&json!({
            "input_tokens": 1, "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5, "output_tokens": 2,
            "cache_creation": { "ephemeral_1h_input_tokens": 4 },
            "output_tokens_details": { "thinking_tokens": 1 },
        }));
        assert_eq!(usage.cache_write_1h, 4);
        assert_eq!(usage.cache_write_5m, 1);
        assert_eq!(usage.context(), 16);
        assert_eq!(usage.thinking, 1);
    }

    #[test]
    fn later_reports_update_without_erasing() {
        let mut usage = Usage::from_json(
            &json!({ "input_tokens": 3, "cache_read_input_tokens": 9, "output_tokens": 1 }),
        );
        usage.update(&Usage::from_json(&json!({ "output_tokens": 40 })));
        assert_eq!(usage.input, 3);
        assert_eq!(usage.cache_read, 9);
        assert_eq!(usage.output, 40);
    }
}

//! Finds secrets in text.

use std::ops::Range;
use std::sync::LazyLock;

use regex::{Regex, RegexSet};

/// A project rule isn't a valid regular expression.
#[derive(Debug, thiserror::Error)]
#[error("invalid redaction rule {rule:?}")]
pub struct InvalidRuleError {
    rule: String,
    #[source]
    source: regex::Error,
}

/// One detection rule. `group` selects the part of a match that is the secret.
#[derive(Debug, Clone)]
struct Rule {
    name: &'static str,
    pattern: String,
    group: usize,
}

/// Minimum Shannon entropy, in bits per character, for an assigned value to count as a
/// secret. English words and identifiers sit below it, random keys above.
const MIN_ASSIGNMENT_ENTROPY: f64 = 3.5;

/// Domains that appear in email-shaped strings without identifying anyone.
const HARMLESS_EMAIL_DOMAINS: &[&str] = &[
    "anthropic.com",
    "example.com",
    "example.org",
    "example.net",
    "test.com",
    "users.noreply.github.com",
    "github.com",
    "localhost",
];

const NAME_ASSIGNMENT: &str = "assignment";
const NAME_EMAIL: &str = "email";
const NAME_PROJECT: &str = "project rule";

const BUILTIN: &[(&str, &str, usize)] = &[
    (
        "private key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        0,
    ),
    ("anthropic key", r"\bsk-ant-[A-Za-z0-9_-]{20,}", 0),
    ("openai key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}", 0),
    (
        "github token",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b",
        0,
    ),
    ("aws access key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0),
    ("slack token", r"\bxox[abeoprs]-[A-Za-z0-9-]{10,}", 0),
    ("stripe key", r"\b[sr]k_live_[A-Za-z0-9]{16,}", 0),
    ("google api key", r"\bAIza[0-9A-Za-z_-]{35}\b", 0),
    ("npm token", r"\bnpm_[A-Za-z0-9]{36}\b", 0),
    (
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        0,
    ),
    (
        "connection string password",
        r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:([^\s:/@]{3,})@",
        1,
    ),
    (
        NAME_ASSIGNMENT,
        r#"(?i)\b[a-z0-9_.-]*(?:password|passwd|secret|token|api[_-]?key|auth[_-]?key|access[_-]?key)[a-z0-9_.-]*["']?\s*[:=]\s*["']?([^\s"',;{}()<>$]{12,})"#,
        1,
    ),
    (
        NAME_EMAIL,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
        0,
    ),
];

/// A compiled set of detection rules.
#[derive(Debug)]
pub struct Detector {
    prefilter: RegexSet,
    rules: Vec<(Rule, Regex)>,
}

/// The rules every detector includes, compiled once because they never change.
static BUILTIN_RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    BUILTIN
        .iter()
        .map(|(name, pattern, group)| Rule {
            name,
            pattern: (*pattern).to_owned(),
            group: *group,
        })
        .collect()
});

impl Detector {
    /// A detector with only the built-in rules.
    #[must_use]
    pub fn builtin() -> Self {
        Self::from_rules(BUILTIN_RULES.clone()).expect("built-in rules are valid")
    }

    /// A detector with the built-in rules plus project-specific regular expressions, such
    /// as customer names. Changing these mid-conversation changes what history compiles
    /// to, so they should only change between conversations.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidRuleError`] if a rule isn't a valid regular expression.
    pub fn with_project_rules(rules: &[String]) -> Result<Self, InvalidRuleError> {
        let mut all = BUILTIN_RULES.clone();
        all.extend(rules.iter().map(|pattern| Rule {
            name: NAME_PROJECT,
            pattern: pattern.clone(),
            group: 0,
        }));
        Self::from_rules(all)
    }

    fn from_rules(rules: Vec<Rule>) -> Result<Self, InvalidRuleError> {
        let invalid = |rule: &str, source| InvalidRuleError {
            rule: rule.to_owned(),
            source,
        };
        let compiled = rules
            .into_iter()
            .map(|rule| {
                let regex = Regex::new(&rule.pattern).map_err(|err| invalid(&rule.pattern, err))?;
                Ok((rule, regex))
            })
            .collect::<Result<Vec<_>, InvalidRuleError>>()?;
        let prefilter = RegexSet::new(compiled.iter().map(|(rule, _)| &rule.pattern))
            .map_err(|err| invalid("<rule set>", err))?;
        Ok(Self {
            prefilter,
            rules: compiled,
        })
    }

    /// Finds secrets in `text`: non-overlapping byte ranges in order, each with the name
    /// of the rule that found it. Time is linear in the length of `text`.
    pub(crate) fn find(&self, text: &str) -> Vec<(Range<usize>, &'static str)> {
        let mut found: Vec<(Range<usize>, &'static str)> = Vec::new();
        for index in self.prefilter.matches(text) {
            let (rule, regex) = &self.rules[index];
            for captures in regex.captures_iter(text) {
                let Some(secret) = captures.get(rule.group) else {
                    continue;
                };
                // `user:password@host` in a URL is shaped like an email but isn't one, and
                // the connection string rule already takes just the password.
                let inside_url = rule.name == NAME_EMAIL && text[..secret.start()].ends_with(':');
                if !inside_url && is_real_match(rule.name, secret.as_str()) {
                    found.push((secret.range(), rule.name));
                }
            }
        }
        found.sort_by_key(|(range, _)| (range.start, std::cmp::Reverse(range.end)));
        let mut merged: Vec<(Range<usize>, &'static str)> = Vec::with_capacity(found.len());
        for (range, name) in found {
            match merged.last_mut() {
                Some((last, _)) if range.start < last.end => last.end = last.end.max(range.end),
                _ => merged.push((range, name)),
            }
        }
        merged
    }
}

fn is_real_match(rule: &str, secret: &str) -> bool {
    match rule {
        NAME_EMAIL => secret.rsplit('@').next().is_none_or(|domain| {
            !HARMLESS_EMAIL_DOMAINS
                .iter()
                .any(|harmless| domain.eq_ignore_ascii_case(harmless))
        }),
        NAME_ASSIGNMENT => looks_random(secret),
        _ => true,
    }
}

/// Whether a value has the character spread of a generated key, not a word or a path.
fn looks_random(value: &str) -> bool {
    if value.contains('/') || value.contains('.') && !value.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    shannon_entropy(value) >= MIN_ASSIGNMENT_ENTROPY
}

fn shannon_entropy(value: &str) -> f64 {
    let mut counts = [0_u32; 256];
    let bytes = value.as_bytes();
    for byte in bytes {
        counts[usize::from(*byte)] += 1;
    }
    let total = bytes.len() as f64;
    counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let share = f64::from(*count) / total;
            -share * share.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secrets_in(text: &str) -> Vec<String> {
        Detector::builtin()
            .find(text)
            .into_iter()
            .map(|(range, _)| text[range].to_owned())
            .collect()
    }

    // Assembled at run time so these fake credentials never sit whole in the source
    // and trip secret scanners on this repository.
    fn fake(prefix: &str, body: &str) -> String {
        format!("{prefix}{body}")
    }

    #[test]
    fn finds_known_credential_formats() {
        let cases = [
            fake("sk-ant-", "api03-abcdefghijklmnopqrstuvwxyz012345"),
            fake("sk-proj-", "abcdefghijklmnopqrstuvwxyz0123456789"),
            fake("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789"),
            fake("github_pat_", "abcdefghijklmnopqrstuvwxyz012"),
            fake("AKIA", "ABCDEFGHIJKLMNOP"),
            fake("xoxb-", "1234567890-abcdefghij"),
            fake("sk_live_", "abcdefghijklmnop1234"),
            fake("AIza", "abcdefghijklmnopqrstuvwxyz012345678"),
            fake("npm_", "abcdefghijklmnopqrstuvwxyz0123456789"),
            fake("eyJhbGciOiJI", ".eyJzdWIiOiIx.abcdefghijkl"),
        ];
        for secret in cases {
            let text = format!("before {secret} after");
            assert_eq!(secrets_in(&text), std::slice::from_ref(&secret), "{secret}");
        }
    }

    #[test]
    fn finds_whole_private_key_blocks() {
        let block = format!(
            "-----BEGIN RSA {0}-----\nabc\ndef\n-----END RSA {0}-----",
            "PRIVATE KEY"
        );
        assert_eq!(secrets_in(&format!("x\n{block}\ny")), [block]);
    }

    #[test]
    fn finds_only_the_password_in_a_connection_string() {
        let text = "url=postgres://admin:s3cr3tpw@db.internal:5432/app";
        assert_eq!(secrets_in(text), ["s3cr3tpw"]);
    }

    #[test]
    fn finds_random_looking_assignments_only() {
        assert_eq!(secrets_in("API_KEY=9f8e7d6c5b4a3210"), ["9f8e7d6c5b4a3210"]);
        assert_eq!(
            secrets_in(r#""client_secret": "Zx81kQp02LmNv7Ty""#),
            ["Zx81kQp02LmNv7Ty"]
        );
        for harmless in [
            "token_count = compute_tokens",
            "password: aaaaaaaaaaaaaaaa",
            "auth_token = request.headers.authorization",
            "secret_path=/etc/app/credentials",
        ] {
            assert_eq!(secrets_in(harmless), Vec::<String>::new(), "{harmless}");
        }
    }

    #[test]
    fn finds_emails_except_harmless_domains() {
        assert_eq!(
            secrets_in("mail jane.doe@acme.io now"),
            ["jane.doe@acme.io"]
        );
        for harmless in ["noreply@anthropic.com", "git@github.com", "a@example.com"] {
            assert_eq!(secrets_in(harmless), Vec::<String>::new(), "{harmless}");
        }
    }

    #[test]
    fn leaves_ordinary_code_alone() {
        let code = "fn main() { let commit = \"86be2751\"; let id = uuid_v4(); println!(\"{id}\"); }\n\
                    git log --format=%H 86be275fe1234567890abcdef\nhttps://example.test/a/b?x=1";
        assert_eq!(secrets_in(code), Vec::<String>::new());
    }

    #[test]
    fn overlapping_matches_merge_into_one() {
        let key = fake("sk-ant-", "abcdefghijklmnopqrstuvwxyz012345");
        let text = format!("API_KEY={key}");
        assert_eq!(secrets_in(&text), [key]);
    }

    #[test]
    fn project_rules_add_to_the_builtins_and_reject_bad_patterns() {
        let detector = Detector::with_project_rules(&["Acme Corp".to_owned()]).unwrap();
        let text = "ticket from Acme Corp, jane@acme.io";
        assert_eq!(detector.find(text).len(), 2);
        assert!(Detector::with_project_rules(&["(".to_owned()]).is_err());
    }
}

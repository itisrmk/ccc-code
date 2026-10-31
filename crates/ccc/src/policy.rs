//! Team rules for how `ccc` may run, kept in a file a team can commit.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context as _, bail};
use toml::{Table, Value};

/// Where a team's policy lives, relative to the directory `ccc` runs in.
const DEFAULT_POLICY: &str = ".ccc/policy.toml";

/// The settings of this run that a policy can judge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RunSettings<'a> {
    pub(crate) redact: bool,
    pub(crate) prune: bool,
    pub(crate) dump: bool,
    pub(crate) upstream: &'a str,
    pub(crate) budget_usd: Option<f64>,
}

/// A team's rules.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Policy {
    require_redact: bool,
    require_prune: bool,
    forbid_dump: bool,
    max_budget_usd: Option<f64>,
    allowed_upstreams: Option<Vec<String>>,
    redact_rules: Vec<String>,
}

/// Loads the policy at `path`, or at the default location if none is given. Having no
/// default policy is fine, but a policy that was asked for by name has to exist.
pub(crate) fn load(path: Option<&Path>) -> anyhow::Result<Option<Policy>> {
    let location = path.unwrap_or_else(|| Path::new(DEFAULT_POLICY));
    match fs::read_to_string(location) {
        Ok(text) => parse(&text)
            .with_context(|| format!("reading policy {}", location.display()))
            .map(Some),
        Err(err) if err.kind() == ErrorKind::NotFound && path.is_none() => Ok(None),
        Err(err) => Err(err).with_context(|| format!("reading policy {}", location.display())),
    }
}

/// Reads a policy. An unknown key is an error, because a typo in a safety rule must not
/// silently turn that rule off.
fn parse(text: &str) -> anyhow::Result<Policy> {
    let table: Table = text.parse().context("not valid TOML")?;
    let mut policy = Policy::default();
    for (key, value) in &table {
        match key.as_str() {
            "require_redact" => policy.require_redact = as_bool(key, value)?,
            "require_prune" => policy.require_prune = as_bool(key, value)?,
            "forbid_dump" => policy.forbid_dump = as_bool(key, value)?,
            "max_budget_usd" => {
                let number = value
                    .as_float()
                    .or_else(|| value.as_integer().map(|whole| whole as f64));
                policy.max_budget_usd =
                    Some(number.with_context(|| format!("{key} must be a number"))?);
            }
            "allowed_upstreams" => policy.allowed_upstreams = Some(as_strings(key, value)?),
            "redact_rules" => policy.redact_rules = as_strings(key, value)?,
            other => bail!("unknown policy key {other:?}"),
        }
    }
    Ok(policy)
}

fn as_bool(key: &str, value: &Value) -> anyhow::Result<bool> {
    value
        .as_bool()
        .with_context(|| format!("{key} must be true or false"))
}

fn as_strings(key: &str, value: &Value) -> anyhow::Result<Vec<String>> {
    let items = value
        .as_array()
        .with_context(|| format!("{key} must be a list of strings"))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .with_context(|| format!("{key} must be a list of strings"))
        })
        .collect()
}

impl Policy {
    /// Extra patterns every run must redact.
    pub(crate) fn redact_rules(&self) -> &[String] {
        &self.redact_rules
    }

    /// Refuses a run that breaks a rule.
    pub(crate) fn check(&self, run: &RunSettings<'_>) -> anyhow::Result<()> {
        if self.require_redact && !run.redact {
            bail!("redaction is required, so --no-redact isn't allowed");
        }
        if self.require_prune && !run.prune {
            bail!("pruning is required, so --no-prune isn't allowed");
        }
        if self.forbid_dump && run.dump {
            bail!("traffic capture is forbidden, so --dump isn't allowed");
        }
        if let Some(allowed) = &self.allowed_upstreams {
            let upstream = run.upstream.trim_end_matches('/');
            if !allowed
                .iter()
                .any(|item| item.trim_end_matches('/') == upstream)
            {
                bail!("upstream {upstream} isn't in the allowed list");
            }
        }
        if let Some(limit) = self.max_budget_usd {
            match run.budget_usd {
                Some(budget) if budget <= limit => {}
                Some(budget) => bail!("a budget of ${budget} is over the ${limit} limit"),
                None => bail!("a budget of at most ${limit} is required: set --budget-usd"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed_run() -> RunSettings<'static> {
        RunSettings {
            redact: true,
            prune: true,
            dump: false,
            upstream: "https://api.anthropic.com",
            budget_usd: Some(5.0),
        }
    }

    const STRICT: &str = r#"
        require_redact = true
        require_prune = true
        forbid_dump = true
        max_budget_usd = 10
        allowed_upstreams = ["https://api.anthropic.com/"]
        redact_rules = ["Acme Corp"]
    "#;

    #[test]
    fn a_compliant_run_passes_and_rules_are_available() {
        let policy = parse(STRICT).unwrap();
        policy.check(&allowed_run()).unwrap();
        assert_eq!(policy.redact_rules(), ["Acme Corp"]);
    }

    #[test]
    fn each_rule_refuses_the_run_that_breaks_it() {
        let policy = parse(STRICT).unwrap();
        let cases = [
            (
                RunSettings {
                    redact: false,
                    ..allowed_run()
                },
                "--no-redact",
            ),
            (
                RunSettings {
                    prune: false,
                    ..allowed_run()
                },
                "--no-prune",
            ),
            (
                RunSettings {
                    dump: true,
                    ..allowed_run()
                },
                "--dump",
            ),
            (
                RunSettings {
                    upstream: "https://evil.test",
                    ..allowed_run()
                },
                "allowed list",
            ),
            (
                RunSettings {
                    budget_usd: Some(50.0),
                    ..allowed_run()
                },
                "over",
            ),
            (
                RunSettings {
                    budget_usd: None,
                    ..allowed_run()
                },
                "--budget-usd",
            ),
        ];
        for (run, expected) in cases {
            let err = policy.check(&run).unwrap_err().to_string();
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn an_empty_policy_allows_anything() {
        let policy = parse("").unwrap();
        let run = RunSettings {
            redact: false,
            prune: false,
            dump: true,
            upstream: "https://x.test",
            budget_usd: None,
        };
        policy.check(&run).unwrap();
    }

    #[test]
    fn a_typo_in_a_rule_is_an_error_and_not_a_silent_gap() {
        let err = parse("require_redcat = true").unwrap_err().to_string();
        assert!(err.contains("require_redcat"), "{err}");
        assert!(parse("require_redact = \"yes\"").is_err());
        assert!(parse("redact_rules = [1]").is_err());
        assert!(parse("not toml at all =").is_err());
    }

    #[test]
    fn a_named_policy_must_exist_but_the_default_is_optional() {
        assert!(load(Some(Path::new("/definitely/missing.toml"))).is_err());
    }
}

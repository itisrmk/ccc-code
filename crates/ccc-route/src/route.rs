use std::time::Duration;

use ccc_pipeline::{Context, Edit, OnFailure, Stage, StageError, Target};
use ccc_wire::Request;
use serde_json::json;

use crate::classify::{CallKind, classify};

/// Routing does no real work, so this only guards against a bug.
const TIME_CAP: Duration = Duration::from_millis(10);
/// A model named with this prefix is served by the local endpoint, not Anthropic.
pub const LOCAL_PREFIX: &str = "local:";

/// Which models to use. With nothing set, routing changes nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RouteRules {
    /// The model for background calls. Use [`LOCAL_PREFIX`] for a local model.
    pub background_model: Option<String>,
    /// Session cost, in dollars, at which [`RouteRules::budget_model`] takes over.
    pub budget_usd: Option<f64>,
    /// The model to move to at the budget. Set only on purpose: the first call on it
    /// re-writes the conversation's whole context to the new model's cache.
    pub budget_model: Option<String>,
}

/// The Route stage. Never moves a call to a more capable model.
#[derive(Debug, Default)]
pub struct Route {
    rules: RouteRules,
}

impl Route {
    /// A stage applying `rules`.
    #[must_use]
    pub fn new(rules: RouteRules) -> Self {
        Self { rules }
    }

    fn choose<'a>(
        &'a self,
        request: &Request,
        context: &Context,
    ) -> Option<(&'a str, &'static str)> {
        if let (Some(budget), Some(model)) = (self.rules.budget_usd, &self.rules.budget_model)
            && context.spent_micros as f64 / 1_000_000.0 >= budget
        {
            return Some((model, "budget downshift"));
        }
        if classify(request) == CallKind::Background
            && let Some(model) = &self.rules.background_model
        {
            return Some((model, "background"));
        }
        None
    }
}

impl Stage for Route {
    fn name(&self) -> &'static str {
        "route"
    }

    fn time_cap(&self) -> Duration {
        TIME_CAP
    }

    fn on_failure(&self) -> OnFailure {
        OnFailure::PassThrough
    }

    fn plan(&self, request: &Request, context: &Context) -> Result<Vec<Edit>, StageError> {
        let Some((model, rule)) = self.choose(request, context) else {
            return Ok(Vec::new());
        };
        let current = request.model().unwrap_or_default();
        if current == model || !is_downgrade(current, model) {
            return Ok(Vec::new());
        }
        Ok(vec![Edit {
            target: Target::Model,
            pointer: String::new(),
            value: json!(model),
            rule,
        }])
    }
}

/// Whether `to` is strictly less capable than `from`, judged by tier. A model whose tier
/// isn't known is never moved, and nothing moves to an unknown tier unless it's local.
pub(crate) fn is_downgrade(from: &str, to: &str) -> bool {
    let Some(from_tier) = tier(from) else {
        return false;
    };
    if to.starts_with(LOCAL_PREFIX) {
        return true;
    }
    tier(to).is_some_and(|to_tier| to_tier < from_tier)
}

fn tier(model: &str) -> Option<u8> {
    ["haiku", "sonnet", "opus", "fable"]
        .iter()
        .position(|family| model.contains(family))
        .and_then(|position| u8::try_from(position).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn background(model: &str) -> Request {
        let body = json!({ "model": model, "messages": [{ "role": "user", "content": "title" }] });
        Request::parse(body.to_string().as_bytes()).unwrap()
    }

    fn agent(model: &str) -> Request {
        let body = json!({ "model": model, "tools": [{ "name": "Bash" }],
            "messages": [{ "role": "user", "content": "go" }] });
        Request::parse(body.to_string().as_bytes()).unwrap()
    }

    fn chosen(rules: RouteRules, request: &Request, spent_micros: u64) -> Option<Value> {
        let context = Context {
            spent_micros,
            ..Context::default()
        };
        Route::new(rules)
            .plan(request, &context)
            .unwrap()
            .pop()
            .map(|edit| edit.value)
    }

    fn background_to(model: &str) -> RouteRules {
        RouteRules {
            background_model: Some(model.to_owned()),
            ..RouteRules::default()
        }
    }

    #[test]
    fn moves_background_calls_down_only() {
        let rules = background_to("claude-haiku-5-5");
        let moved = chosen(rules.clone(), &background("claude-opus-5-5"), 0);
        assert_eq!(moved, Some(json!("claude-haiku-5-5")));
        assert_eq!(
            chosen(rules.clone(), &background("claude-haiku-5-5"), 0),
            None
        );
        assert_eq!(chosen(rules, &background("unknown-model"), 0), None);
    }

    #[test]
    fn never_upgrades() {
        let rules = background_to("claude-opus-5-5");
        assert_eq!(chosen(rules, &background("claude-haiku-5-5"), 0), None);
    }

    #[test]
    fn leaves_agent_calls_alone_without_a_budget() {
        let rules = background_to("claude-haiku-5-5");
        assert_eq!(
            chosen(rules, &agent("claude-opus-5-5"), 1_000_000_000),
            None
        );
    }

    #[test]
    fn with_no_rules_nothing_changes() {
        assert_eq!(
            chosen(RouteRules::default(), &background("claude-opus-5-5"), 0),
            None
        );
    }

    #[test]
    fn local_models_count_as_a_downgrade_from_anything_known() {
        let moved = chosen(
            background_to("local:qwen3"),
            &background("claude-haiku-5-5"),
            0,
        );
        assert_eq!(moved, Some(json!("local:qwen3")));
    }

    #[test]
    fn a_budget_downshift_waits_for_the_budget() {
        let rules = RouteRules {
            budget_usd: Some(2.0),
            budget_model: Some("claude-sonnet-5-5".to_owned()),
            ..RouteRules::default()
        };
        assert_eq!(
            chosen(rules.clone(), &agent("claude-opus-5-5"), 1_999_999),
            None
        );
        assert_eq!(
            chosen(rules.clone(), &agent("claude-opus-5-5"), 2_000_000),
            Some(json!("claude-sonnet-5-5"))
        );
        assert_eq!(chosen(rules, &agent("claude-haiku-5-5"), 9_000_000), None);
    }

    #[test]
    fn a_budget_without_a_model_only_ever_warns() {
        let rules = RouteRules {
            budget_usd: Some(1.0),
            ..RouteRules::default()
        };
        assert_eq!(chosen(rules, &agent("claude-opus-5-5"), 9_000_000), None);
    }
}

//! `ccc rules`: learn lasting project rules from past sessions, for a person to approve.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use ccc_learn::{
    Candidate, Rule, RuleStatus, RuleStore, read_candidates, read_working_commands, rules_text,
};
use ccc_redact::Detector;
use ccc_slm::{CompletionRequest, ModelError, TextModel};
use clap::{Args, Subcommand};

use crate::daemon::DaemonClient;

const FILE: &str = "rules.json";
const MAX_RULE_TOKENS: u32 = 60;
const MAX_RULE_CHARS: usize = 240;
const NOT_A_RULE: &str = "NONE";
/// Long enough for a model to load the first time.
const EXTRACT_DEADLINE: Duration = Duration::from_secs(60);

#[derive(Debug, Args)]
pub(crate) struct RulesArgs {
    #[command(subcommand)]
    command: Option<RulesCommand>,
    #[command(flatten)]
    state: crate::StateArgs,
}

#[derive(Debug, Subcommand)]
enum RulesCommand {
    /// Read past sessions and propose rules. Runs by itself as sessions start.
    Learn(LearnArgs),
    /// Use a rule from now on. Takes an id, or the start of one.
    Approve {
        #[arg(value_name = "ID", required = true)]
        ids: Vec<String>,
    },
    /// Don't use a rule, and don't propose it again.
    Reject {
        #[arg(value_name = "ID", required = true)]
        ids: Vec<String>,
    },
}

#[derive(Debug, Args)]
struct LearnArgs {
    /// Look at no more than this many messages this time, leaving the rest for later.
    #[arg(long, value_name = "MESSAGES")]
    budget: Option<usize>,
    /// Transcripts to read. Defaults to ~/.claude/projects.
    #[arg(long, value_name = "DIR")]
    transcripts: Option<PathBuf>,
    #[command(flatten)]
    state: crate::StateArgs,
}

fn store_path(state_dir: &Path) -> PathBuf {
    state_dir.join(FILE)
}

fn extraction_prompt(candidate: &Candidate) -> String {
    format!(
        "A developer said this to a coding agent.\n\nAgent: {}\n\nDeveloper: {}\n\nWrite the \
         lasting project rule this states, as one short instruction naming the specific thing. If \
         there isn't one, reply with exactly {NOT_A_RULE}.",
        candidate.previous, candidate.message,
    )
}

/// A second opinion, asked separately. On real history, a single question turned task requests
/// into "rules" whichever way it was worded, and a narrow yes or no is easier to get right.
fn confirmation_prompt(rule: &str, message: &str) -> String {
    format!(
        "Developer's message: {message}\n\nProposed rule: {rule}\n\nAnswer no if the rule is only \
         about this piece of work, is general advice that applies to any project, or isn't what \
         the developer said. Answer yes only if it names a specific tool, command, path or \
         convention to follow in all future work on this project. Answer yes or no."
    )
}

/// Holds an answer to one word, so there's nothing to misread.
const YES_NO_GRAMMAR: &str = "root ::= \"yes\" | \"no\"\n";

/// What a model's answer amounts to. Anything that doesn't read as one short rule isn't one.
fn parse_rule(answer: &str) -> Option<String> {
    let line = answer.trim().lines().next()?.trim();
    let line = line.strip_prefix("- ").unwrap_or(line);
    let line = line.trim_matches(['"', '*', '`']).trim();
    let useless = line.is_empty()
        || line.to_uppercase().starts_with(NOT_A_RULE)
        || line.chars().count() > MAX_RULE_CHARS
        || !line.contains(' ');
    (!useless).then(|| line.to_owned())
}

/// Looks at each new message, up to `budget`. Marks one considered only once the model has
/// answered, so a model that's still loading or stops halfway just means trying again later.
fn learn_from(
    store: &mut RuleStore,
    candidates: &[Candidate],
    model: &dyn TextModel,
    detector: &Detector,
    budget: usize,
) -> Result<Vec<Rule>, ModelError> {
    let mut learned = Vec::new();
    let fresh = candidates
        .iter()
        .filter(|c| !store.was_considered(c))
        .take(budget);
    let fresh: Vec<Candidate> = fresh.cloned().collect();
    for candidate in &fresh {
        let answer = model.complete(&CompletionRequest {
            prompt: &extraction_prompt(candidate),
            grammar: None,
            max_tokens: MAX_RULE_TOKENS,
            deadline: EXTRACT_DEADLINE,
        })?;
        // A rule goes back into prompts, so nothing secret may ever become one.
        let rule = parse_rule(&answer.text).filter(|text| {
            !detector.finds_secret(text) && !detector.finds_secret(&candidate.message)
        });
        if let Some(text) = rule {
            let confirmed = model.complete(&CompletionRequest {
                prompt: &confirmation_prompt(&text, &candidate.message),
                grammar: Some(YES_NO_GRAMMAR),
                max_tokens: 2,
                deadline: EXTRACT_DEADLINE,
            })?;
            if confirmed.text.trim().eq_ignore_ascii_case("yes") {
                let rule = Rule::from_candidate(candidate, &text);
                if store.add(rule.clone()) {
                    learned.push(rule);
                }
            }
        }
        store.mark_considered(candidate);
    }
    Ok(learned)
}

fn learn(args: &LearnArgs, claude_home: &Path) -> anyhow::Result<()> {
    let state_dir = args.state.dir()?;
    let transcripts = args
        .transcripts
        .clone()
        .unwrap_or_else(|| claude_home.join("projects"));
    let path = store_path(&state_dir);
    let mut store = RuleStore::load(&path).context("reading rules")?;
    let candidates = read_candidates(&transcripts).context("reading transcripts")?;
    // Needs no model, so these still get learned if the model isn't there.
    let mut from_commands = 0;
    for found in read_working_commands(&transcripts).context("reading transcripts")? {
        from_commands += usize::from(store.add(Rule::from_command(&found)));
    }
    let client = DaemonClient::new(&state_dir);
    let result = learn_from(
        &mut store,
        &candidates,
        &client,
        &Detector::builtin(),
        args.budget.unwrap_or(usize::MAX),
    );
    // Whatever was learned before a failure is worth keeping.
    store.save(&path).context("saving rules")?;
    let learned = result.context("asking the local model")?;
    writeln!(
        std::io::stdout(),
        "{} new rules to review, {from_commands} from commands that keep working and {} from {} \
         things you said: ccc rules",
        from_commands + learned.len(),
        learned.len(),
        candidates.len(),
    )?;
    Ok(())
}

fn list(store: &RuleStore, repo: Option<&str>, out: &mut impl Write) -> anyhow::Result<()> {
    let pending: Vec<&Rule> = store
        .rules
        .iter()
        .filter(|rule| rule.status == RuleStatus::Pending)
        .collect();
    let approved: Vec<&Rule> = store
        .rules
        .iter()
        .filter(|rule| rule.status == RuleStatus::Approved)
        .collect();
    if pending.is_empty() && approved.is_empty() {
        writeln!(
            out,
            "No rules yet. They're learned from what you tell Claude, as sessions start. \
             ccc rules learn does it now."
        )?;
        return Ok(());
    }
    if !pending.is_empty() {
        writeln!(
            out,
            "Waiting for you. Nothing's used until you approve it.\n"
        )?;
        for rule in &pending {
            writeln!(out, "  {}  {}", rule.id, rule.text)?;
            let source = if rule.session.is_empty() {
                "evidence"
            } else {
                "you said"
            };
            writeln!(out, "            {source}: {}", one_line(&rule.quote))?;
            if repo != Some(rule.repo.as_str()) {
                writeln!(out, "            in {}", rule.repo)?;
            }
        }
        writeln!(out, "\nccc rules approve ID, or ccc rules reject ID\n")?;
    }
    if !approved.is_empty() {
        writeln!(out, "In use:\n")?;
        for rule in &approved {
            writeln!(out, "  {}  {}", rule.id, rule.text)?;
        }
    }
    Ok(())
}

fn one_line(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 110 {
        format!("{}...", flat.chars().take(110).collect::<String>())
    } else {
        flat
    }
}

pub(crate) fn run(args: &RulesArgs, claude_home: &Path) -> anyhow::Result<()> {
    if let Some(RulesCommand::Learn(learn_args)) = &args.command {
        return learn(learn_args, claude_home);
    }
    let path = store_path(&args.state.dir()?);
    let mut store = RuleStore::load(&path).context("reading rules")?;
    let mut decide = |ids: &[String], status| -> anyhow::Result<()> {
        for id in ids {
            if !store.decide(id, status) {
                anyhow::bail!("no rule starting with {id}");
            }
        }
        store.save(&path).context("saving rules")
    };
    let mut out = std::io::stdout();
    match &args.command {
        Some(RulesCommand::Approve { ids }) => {
            decide(ids, RuleStatus::Approved)?;
            writeln!(out, "In use from the next prompt.")?;
        }
        Some(RulesCommand::Reject { ids }) => {
            decide(ids, RuleStatus::Rejected)?;
            writeln!(out, "Rejected, and it won't be proposed again.")?;
        }
        _ => {
            let here = std::env::current_dir()
                .ok()
                .map(|dir| dir.to_string_lossy().into_owned());
            list(&store, here.as_deref(), &mut out)?;
        }
    }
    Ok(())
}

/// Adds approved rules that bear on this prompt. Any failure just adds nothing, since a broken
/// store must never get in the way of a prompt.
pub(crate) fn hook(state_dir: &Path, cwd: &str, prompt: &str) -> Option<String> {
    let store = RuleStore::load(&store_path(state_dir)).ok()?;
    let repo = std::fs::canonicalize(cwd).ok()?;
    let relevant = store.relevant(&repo.to_string_lossy(), prompt);
    let by_given = store.relevant(cwd, prompt);
    rules_text(if relevant.is_empty() {
        &by_given
    } else {
        &relevant
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_slm::Completion;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct Answers(Mutex<Vec<&'static str>>);

    impl TextModel for Answers {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            match self.0.lock().unwrap().pop() {
                Some(text) => Ok(Completion {
                    text: text.to_owned(),
                    ..Completion::default()
                }),
                None => Err(ModelError::Stopped),
            }
        }
    }

    fn candidate(message: &str) -> Candidate {
        Candidate {
            repo: "/repo".to_owned(),
            session: "s".to_owned(),
            at: "t".to_owned(),
            message: message.to_owned(),
            previous: "I'll use npm.".to_owned(),
        }
    }

    #[test]
    fn answers_become_one_clean_rule_or_nothing() {
        assert_eq!(
            parse_rule(" \"Use pnpm, never npm.\" \n more").as_deref(),
            Some("Use pnpm, never npm.")
        );
        assert_eq!(
            parse_rule("- **Run make lint first**").as_deref(),
            Some("Run make lint first")
        );
        for answer in [
            "NONE",
            "none.",
            "NONE - only this task",
            "",
            "Yes",
            &"x ".repeat(200),
        ] {
            assert_eq!(parse_rule(answer), None, "{answer}");
        }
    }

    #[test]
    fn the_prompt_carries_both_sides_and_a_way_to_say_no() {
        let prompt = extraction_prompt(&candidate("never npm"));
        assert!(prompt.contains("I'll use npm."), "{prompt}");
        assert!(prompt.contains("never npm"), "{prompt}");
        assert!(prompt.contains(NOT_A_RULE), "{prompt}");
    }

    #[test]
    fn rules_start_out_pending_and_nothing_is_looked_at_twice() {
        let mut store = RuleStore::default();
        // Popped from the end: a rule, its confirmation, then a message with no rule.
        let model = Answers(Mutex::new(vec!["NONE", "yes", "Use pnpm, never npm."]));
        let said = [candidate("no, use pnpm"), candidate("stop that")];
        let learned = learn_from(&mut store, &said, &model, &Detector::builtin(), 10).unwrap();
        assert_eq!(learned.len(), 1);
        assert_eq!(store.rules[0].status, RuleStatus::Pending);
        assert_eq!(store.rules[0].quote, "no, use pnpm");
        // The model has no answers left, so this only passes if it isn't asked again.
        assert!(
            learn_from(&mut store, &said, &model, &Detector::builtin(), 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_rule_that_fails_the_second_question_is_dropped_but_not_asked_again() {
        let mut store = RuleStore::default();
        let model = Answers(Mutex::new(vec!["no", "Use the blue one."]));
        let said = [candidate("never mind, always use the blue one")];
        assert!(
            learn_from(&mut store, &said, &model, &Detector::builtin(), 10)
                .unwrap()
                .is_empty()
        );
        assert!(store.rules.is_empty());
        assert!(store.was_considered(&said[0]));
    }

    #[test]
    fn a_budget_leaves_the_rest_for_later_and_a_stopped_model_loses_nothing() {
        let mut store = RuleStore::default();
        let said = [
            candidate("one must"),
            candidate("two must"),
            candidate("three must"),
        ];
        let one = Answers(Mutex::new(vec!["yes", "Do the first thing."]));
        learn_from(&mut store, &said, &one, &Detector::builtin(), 1).unwrap();
        assert!(store.was_considered(&said[0]));
        assert!(!store.was_considered(&said[1]));

        // Answers one message, then stops: that's kept, and the next tried again later.
        let stops = Answers(Mutex::new(vec!["yes", "Do the second thing."]));
        assert!(learn_from(&mut store, &said, &stops, &Detector::builtin(), 10).is_err());
        assert_eq!(store.rules.len(), 2);
        assert!(store.was_considered(&said[1]));
        assert!(!store.was_considered(&said[2]));
    }

    #[test]
    fn nothing_secret_becomes_a_rule() {
        let key = format!("sk-ant-{}", "abcdefghijklmnopqrstuvwx");
        let quoted = candidate(&format!("always use the key {key}"));
        let model = Answers(Mutex::new(vec!["yes", "Always use the deployment key."]));
        let mut store = RuleStore::default();
        learn_from(&mut store, &[quoted], &model, &Detector::builtin(), 1).unwrap();
        assert!(store.rules.is_empty(), "the message held a secret");
    }

    #[test]
    fn listing_shows_what_needs_deciding_and_where_it_came_from() {
        let mut store = RuleStore::default();
        store.add(Rule::from_candidate(
            &candidate("no, use pnpm"),
            "Use pnpm.",
        ));
        let mut out = Vec::new();
        list(&store, Some("/repo"), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Nothing's used until you approve"), "{text}");
        assert!(text.contains("you said: no, use pnpm"), "{text}");

        let mut empty = Vec::new();
        list(&RuleStore::default(), None, &mut empty).unwrap();
        assert!(String::from_utf8(empty).unwrap().contains("No rules yet"));
    }
}

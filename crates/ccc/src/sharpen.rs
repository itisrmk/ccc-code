//! Two optional prompt jobs, and `ccc jobs` to turn them on.
//!
//! A hook can't change a prompt. It can only add beside it, or hold it back and say why. So
//! catching a vague prompt means holding it back once, showing a sharper version, and letting the
//! same prompt through if it's sent again. A "rewrite" adds a clearer version as context, with the
//! person's own words still there and still in charge.

use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, bail};
use ccc_enrich::{GateConfig, gate_rewrite, propose_rewrite};
use ccc_slm::{CompletionRequest, ModelError, TextModel};
use clap::Args;
use serde_json::{Value, json};

use crate::config::{JobMode, LocalConfig};
use crate::daemon::DaemonClient;

/// A person is waiting, and the hook itself gets 10 seconds.
const SHARPEN_DEADLINE: Duration = Duration::from_secs(5);
const REWRITE_DEADLINE: Duration = Duration::from_secs(4);
const CLASSIFY_DEADLINE: Duration = Duration::from_millis(1500);
const MAX_SHARPER_TOKENS: u32 = 90;
/// How much of the original's words a rewrite has to keep.
const MIN_SIMILARITY: f64 = 0.6;
const MAX_VAGUE_WORDS: usize = 9;
const MIN_REWRITE_WORDS: usize = 4;
/// Words that often mean "you know what I mean", when nothing more specific is said.
const VAGUE_WORDS: [&str; 13] = [
    "this", "that", "it", "bug", "broken", "fix", "improve", "cleanup", "clean", "refactor",
    "better", "issue", "problem",
];

#[derive(Debug, Args)]
pub(crate) struct JobsArgs {
    /// Which job: sharpen, rewrite or trim. Shows them all if left out.
    #[arg(value_name = "JOB")]
    job: Option<String>,
    /// off, suggest (sharpen only) or on.
    #[arg(value_name = "MODE", requires = "job")]
    mode: Option<String>,
    #[command(flatten)]
    state: crate::StateArgs,
}

/// `ccc jobs`.
pub(crate) fn jobs(args: &JobsArgs) -> anyhow::Result<()> {
    let state_dir = args.state.dir()?;
    let mut config = LocalConfig::load(&state_dir)?;
    let mut out = std::io::stdout();
    if let (Some(job), Some(mode)) = (&args.job, &args.mode) {
        let mode = JobMode::parse(mode).context("the mode must be off, suggest or on")?;
        match (job.as_str(), mode) {
            ("sharpen", JobMode::On) => bail!("sharpen can only be off or suggest"),
            ("rewrite" | "trim", JobMode::Suggest) => bail!("{job} can only be off or on"),
            ("sharpen", _) => config.jobs.sharpen = mode,
            ("rewrite", _) => config.jobs.rewrite = mode,
            ("trim", _) => config.jobs.trim = mode,
            (other, _) => bail!("no job called {other:?}"),
        }
        config.save(&state_dir)?;
    }
    writeln!(
        out,
        "sharpen  {:<8} holds back a vague prompt once, with a sharper version\n\
         rewrite  {:<8} adds a clearer version of each prompt beside yours\n\
         trim     {:<8} shortens long command output before Claude sees it\n\n\
         Change one with: ccc jobs trim on\n\
         These are off by default until they've shown they help on real work, and the README has \
         what's been measured. Rewriting never passed more tasks when tested as context. Test any \
         of them on your own work with ccc eval.",
        config.jobs.sharpen.key(),
        config.jobs.rewrite.key(),
        config.jobs.trim.key(),
    )?;
    Ok(())
}

/// Whether a prompt leans on context it doesn't state. Cheap, since it runs before every prompt,
/// and cautious, since holding back a clear prompt would be worse than missing a vague one.
fn is_vague(prompt: &str) -> bool {
    let words: Vec<&str> = prompt.split_whitespace().collect();
    if words.is_empty() || words.len() > MAX_VAGUE_WORDS || prompt.starts_with('/') {
        return false;
    }
    let names_something = words.iter().any(|word| {
        let inner = &word[1..];
        word.contains(['/', '_', '`', '.', ':', '(', '=']) && !word.ends_with(['.', ':'])
            || inner.chars().any(char::is_uppercase)
            || word.chars().any(|c| c.is_ascii_digit())
    });
    if names_something {
        return false;
    }
    words.iter().any(|word| {
        let plain = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        VAGUE_WORDS.contains(&plain.as_str())
    })
}

fn sharpen_prompt(prompt: &str, context: Option<&str>) -> String {
    let files = context.map_or_else(String::new, |text| format!("\n\n{text}"));
    format!(
        "A developer asked a coding agent: \"{prompt}\"{files}\n\nRewrite that as one specific \
         request, naming the likely files if they're shown above, and saying how to check the \
         result. Keep its meaning and add no new requirements. Reply with only the request."
    )
}

fn hash_of(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn marker(state_dir: &Path, session: &str) -> Option<PathBuf> {
    let safe = !session.is_empty()
        && session
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    safe.then(|| state_dir.join("sharpen").join(session))
}

/// Holds back a vague prompt once, unless it's being sent again.
fn sharpen(
    model: &dyn TextModel,
    prompt: &str,
    context: Option<&str>,
    marker: Option<&Path>,
) -> Result<Option<Value>, ModelError> {
    if !is_vague(prompt) {
        return Ok(None);
    }
    let hash = hash_of(prompt);
    if let Some(path) = marker
        && fs::read_to_string(path).is_ok_and(|held| held == hash)
    {
        // The person saw the suggestion and sent it anyway, so it's theirs to send.
        fs::remove_file(path).ok();
        return Ok(None);
    }
    let answer = model.complete(&CompletionRequest {
        prompt: &sharpen_prompt(prompt, context),
        grammar: None,
        max_tokens: MAX_SHARPER_TOKENS,
        deadline: SHARPEN_DEADLINE,
    })?;
    let sharper = answer.text.trim().trim_matches('"').trim();
    if sharper.is_empty() {
        return Ok(None);
    }
    if let Some(path) = marker {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(path, hash).ok();
    }
    Ok(Some(json!({
        "decision": "block",
        "reason": format!(
            "ccc: that request is short on specifics, which can cost turns. A sharper version \
             from your local model, which may be wrong:\n\n{sharper}\n\nSend that, or send your \
             original again to go ahead with it."
        ),
    })))
}

/// A clearer version beside the person's words, if it passes the check that it means the same.
fn rewrite(model: &dyn TextModel, prompt: &str) -> Option<String> {
    if prompt.split_whitespace().count() < MIN_REWRITE_WORDS || prompt.starts_with('/') {
        return None;
    }
    let proposal = propose_rewrite(model, prompt, REWRITE_DEADLINE).ok()?;
    let gate = GateConfig {
        min_similarity: MIN_SIMILARITY,
        classify_deadline: CLASSIFY_DEADLINE,
    };
    gate_rewrite(model, prompt, &proposal, gate).then(|| {
        format!(
            "<ccc-rewrite>\nA local model's clearer reading of this request. It may be wrong, and \
             the person's own words above take priority:\n{}\n</ccc-rewrite>",
            proposal.replace(['<', '>'], "")
        )
    })
}

/// Handles the hook. Nothing here may get in the way of a prompt beyond what was asked for, so
/// any failure or missed deadline adds nothing.
pub(crate) fn hook(state_dir: &Path, session: &str, cwd: &str, prompt: &str) -> Option<String> {
    let config = LocalConfig::load(state_dir).ok()?;
    let (sharpening, rewriting) = (config.jobs.sharpen, config.jobs.rewrite);
    if sharpening == JobMode::Off && rewriting == JobMode::Off {
        return None;
    }
    let client = DaemonClient::new(state_dir);
    if sharpening == JobMode::Suggest {
        let context = crate::enhance::hook(state_dir, cwd, prompt);
        let held = sharpen(
            &client,
            prompt,
            context.as_deref(),
            marker(state_dir, session).as_deref(),
        );
        match held {
            Ok(Some(output)) => return Some(output.to_string()),
            Ok(None) => {}
            Err(err) => tracing::debug!(error = %err, "no suggestion"),
        }
    }
    if rewriting == JobMode::On {
        return rewrite(&client, prompt);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccc_slm::Completion;

    #[derive(Debug)]
    struct Says(&'static str);

    impl TextModel for Says {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            Ok(Completion {
                text: self.0.to_owned(),
                ..Completion::default()
            })
        }
    }

    #[derive(Debug)]
    struct Fails;

    impl TextModel for Fails {
        fn complete(&self, _: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            Err(ModelError::DeadlineExceeded)
        }
    }

    #[test]
    fn spots_prompts_that_lean_on_context() {
        for prompt in [
            "fix the bug",
            "fix this",
            "make it better",
            "clean this up",
            "it's broken",
        ] {
            assert!(is_vague(prompt), "{prompt}");
        }
    }

    #[test]
    fn leaves_specific_prompts_alone() {
        for prompt in [
            "fix the bug in src/auth.rs",
            "fix verify_token",
            "refactor the UserStore type",
            "fix issue 42",
            "add a dark mode toggle",
            "thanks",
            "/compact",
            "",
            "the login page throws an error when the password field is left empty on mobile",
        ] {
            assert!(!is_vague(prompt), "{prompt}");
        }
    }

    #[test]
    fn holds_back_once_and_then_it_is_the_persons_call() {
        let dir = std::env::temp_dir().join(format!("ccc-sharpen-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("sharpen/s");
        let model = Says("Fix the failing login test in tests/auth.rs and run it.");

        let first = sharpen(&model, "fix the bug", None, Some(&path))
            .unwrap()
            .unwrap();
        assert_eq!(first["decision"], "block");
        assert!(first["reason"].as_str().unwrap().contains("tests/auth.rs"));
        assert!(
            first["reason"]
                .as_str()
                .unwrap()
                .contains("send your original again")
        );

        assert_eq!(
            sharpen(&model, "fix the bug", None, Some(&path)).unwrap(),
            None
        );
        assert!(
            sharpen(&model, "fix the bug", None, Some(&path))
                .unwrap()
                .is_some()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_clear_prompt_never_reaches_the_model_and_a_failure_blocks_nothing() {
        assert_eq!(
            sharpen(&Fails, "add a dark mode toggle", None, None),
            Ok(None)
        );
        assert!(sharpen(&Fails, "fix the bug", None, None).is_err());
        assert_eq!(sharpen(&Says("  "), "fix the bug", None, None), Ok(None));
    }

    #[test]
    fn a_session_id_that_isnt_a_plain_name_keeps_no_marker() {
        assert!(marker(Path::new("/s"), "../../x").is_none());
        assert!(marker(Path::new("/s"), "abc-1").is_some());
    }

    #[test]
    fn a_rewrite_is_refused_when_it_drifts_or_the_model_fails() {
        assert_eq!(rewrite(&Fails, "please fix the login page now"), None);
        assert_eq!(
            rewrite(
                &Says("Something else entirely"),
                "please fix the login page now"
            ),
            None
        );
        assert_eq!(rewrite(&Says("x"), "short"), None);
    }
}

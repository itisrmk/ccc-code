//! Claude Code hooks: warm things up, add learned context after compaction, log each turn.
//!
//! Documented limits shape this. A `PreCompact` hook can only read or block a compaction, so
//! learned context goes in through `SessionStart`, which also fires right after one.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command as Process, Stdio};
use std::sync::Arc;

use anyhow::Context as _;
use ccc_enrich::Suggester;
use ccc_index::Index;
use ccc_learn::Memory;
use ccc_ledger::Ledger;
use serde_json::{Value, json};

use crate::LEDGER_FILE;
use crate::advise::{AdviseOptions, Moment, advise};
use crate::learn::{index_cache_path, memory_path};

/// Files to name when reminding a compacted session of its context.
const REMINDER_FILES: usize = 5;
/// How long Claude Code should wait for a hook, in seconds. They're meant to be quick.
const HOOK_TIMEOUT_SECONDS: u32 = 10;

/// The settings that register every hook, for `claude --settings`.
pub(crate) fn settings_json(ccc: &Path, state_dir: &Path, advice: Option<AdviseOptions>) -> Value {
    let advice_flags = advice.map_or_else(String::new, |options| {
        let block = if options.block { " --advise-block" } else { "" };
        format!(
            " --advise --advise-min-context {}{block}",
            options.min_context
        )
    });
    let command = |event: &str| {
        format!(
            "{} hook {event} --state-dir {}{advice_flags}",
            shell_quote(&ccc.to_string_lossy()),
            shell_quote(&state_dir.to_string_lossy()),
        )
    };
    let entry = |event: &str| {
        json!([{ "hooks": [{
            "type": "command",
            "command": command(event),
            "timeout": HOOK_TIMEOUT_SECONDS,
        }] }])
    };
    let mut hooks = json!({
        "SessionStart": entry("session-start"),
        "PreCompact": entry("pre-compact"),
        "Stop": entry("stop"),
    });
    if advice.is_some() {
        hooks["UserPromptSubmit"] = entry("user-prompt-submit");
    }
    json!({ "hooks": hooks })
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// Handles one hook event, reading its details from stdin.
pub(crate) fn handle(
    event: &str,
    state_dir: &Path,
    advice: Option<AdviseOptions>,
) -> anyhow::Result<()> {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .context("reading hook input")?;
    let input: Value = serde_json::from_str(&raw).context("parsing hook input")?;
    let field = |name: &str| input.get(name).and_then(Value::as_str).unwrap_or_default();

    log_event(state_dir, event, field("session_id"), &input)?;
    let give_advice = |moment| match advice {
        Some(options) => {
            let output = advise(
                state_dir,
                field("session_id"),
                field("transcript_path"),
                moment,
                options,
            )?;
            if let Some(output) = output {
                writeln!(std::io::stdout(), "{output}").context("writing advice")?;
            }
            Ok(())
        }
        None => Ok::<(), anyhow::Error>(()),
    };
    match event {
        "session-start" => {
            if field("source") == "resume" {
                give_advice(Moment::Resume)?;
            }
            session_start(state_dir, field("cwd"), field("source"))
        }
        "user-prompt-submit" => give_advice(Moment::Prompt),
        "stop" => stop(state_dir, field("session_id")),
        // Logged above; the docs offer no way to add compaction instructions from here.
        "pre-compact" => Ok(()),
        other => anyhow::bail!("unknown hook event {other:?}"),
    }
}

fn log_event(state_dir: &Path, event: &str, session: &str, input: &Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(state_dir).context("creating the state directory")?;
    let entry = json!({
        "event": event,
        "session": session,
        "source": input.get("source"),
        "trigger": input.get("trigger"),
    });
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_dir.join("hooks.log"))
        .context("opening the hook log")?;
    writeln!(log, "{entry}").context("writing the hook log")
}

/// Starts background work without holding up the session, and after a compaction prints
/// what past work in this repository tended to need. Claude Code adds that as context.
fn session_start(state_dir: &Path, cwd: &str, source: &str) -> anyhow::Result<()> {
    if !cwd.is_empty()
        && let Ok(ccc) = std::env::current_exe()
    {
        let spawned = Process::new(ccc)
            .args(["index", cwd, "--state-dir"])
            .arg(state_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Err(err) = spawned {
            tracing::warn!(error = %err, "could not start indexing");
        }
    }
    if source == "compact" {
        let root = Path::new(cwd);
        let memory = Memory::open(&memory_path(state_dir)).context("opening memory")?;
        let index = Index::load(&index_cache_path(state_dir, root));
        let suggester = Suggester::new(root.to_owned(), index, Some(Arc::new(memory)));
        // The folder's own name is the best query available without a prompt to go on.
        let name = root
            .file_name()
            .map_or(cwd.into(), |name| name.to_string_lossy());
        if let Some(hints) = suggester.hints(&name, REMINDER_FILES) {
            writeln!(std::io::stdout(), "{hints}").context("writing hints")?;
        }
    }
    Ok(())
}

/// Appends this session's cost so far to a per-turn log.
fn stop(state_dir: &Path, session: &str) -> anyhow::Result<()> {
    let ledger = Ledger::open(&state_dir.join(LEDGER_FILE)).context("opening ledger")?;
    let cost = ledger.session_cost(session).context("reading ledger")?;
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_dir.join("turns.log"))
        .context("opening the turn log")?;
    writeln!(log, "{}", json!({ "session": session, "cost_usd": cost }))
        .context("writing the turn log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn settings_register_every_hook_with_quoted_paths() {
        let settings = settings_json(
            &PathBuf::from("/opt/my tools/ccc"),
            &PathBuf::from("/st"),
            None,
        );
        assert!(settings["hooks"].get("UserPromptSubmit").is_none());
        for (event, name) in [
            ("SessionStart", "session-start"),
            ("PreCompact", "pre-compact"),
            ("Stop", "stop"),
        ] {
            let command = settings["hooks"][event][0]["hooks"][0]["command"]
                .as_str()
                .unwrap();
            assert!(
                command.starts_with("'/opt/my tools/ccc' hook "),
                "{command}"
            );
            assert!(command.contains(name), "{command}");
        }
    }

    #[test]
    fn advice_adds_the_prompt_hook_and_passes_its_options_along() {
        let options = AdviseOptions {
            min_context: 5000,
            block: true,
        };
        let settings = settings_json(&PathBuf::from("/ccc"), &PathBuf::from("/st"), Some(options));
        for event in ["UserPromptSubmit", "SessionStart"] {
            let command = settings["hooks"][event][0]["hooks"][0]["command"]
                .as_str()
                .unwrap();
            assert!(
                command.ends_with("--advise --advise-min-context 5000 --advise-block"),
                "{command}"
            );
        }
    }

    #[test]
    fn quoting_survives_an_apostrophe() {
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }
}

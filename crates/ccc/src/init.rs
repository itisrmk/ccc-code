//! `ccc init`: turns cost warnings on by adding hooks to Claude Code's settings.
//!
//! Someone else's settings file is at stake, so anything that doesn't parse is left alone, a copy
//! is kept, the write is atomic and running it twice changes nothing.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use clap::Args;
use serde_json::{Map, Value, json};

use crate::hooks::shell_quote;

/// Marks a hook as ours, so it can be found again.
const OUR_HOOKS: [&str; 2] = [" hook advise-", " hook enhance-"];
const HOOK_TIMEOUT_SECONDS: u32 = 10;
/// Claude Code event, and our handler for it.
const EVENTS: [(&str, &str); 2] = [
    ("SessionStart", "advise-session-start"),
    ("UserPromptSubmit", "advise-prompt"),
];

#[derive(Debug, Args)]
pub(crate) struct InitArgs {
    /// Change this settings file. Defaults to ~/.claude/settings.json.
    #[arg(long, value_name = "FILE")]
    settings_file: Option<PathBuf>,
    /// Also add likely relevant files to each prompt as context. Your words aren't changed. It
    /// spends tokens on every prompt, so run `ccc eval` to see whether it helps you.
    #[arg(long)]
    enhance: bool,
    /// Remove the hooks again.
    #[arg(long)]
    uninstall: bool,
    /// Show the result without writing anything.
    #[arg(long)]
    dry_run: bool,
    #[command(flatten)]
    state: crate::StateArgs,
}

fn is_ours(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|hooks| {
            hooks.iter().any(|hook| {
                hook.get("command")
                    .and_then(Value::as_str)
                    .is_some_and(|command| OUR_HOOKS.iter().any(|ours| command.contains(ours)))
            })
        })
}

/// Removes our hooks, along with any list or section that leaves empty.
fn remove_ours(settings: &mut Map<String, Value>) -> anyhow::Result<()> {
    let Some(hooks) = settings.get_mut("hooks") else {
        return Ok(());
    };
    let hooks = hooks
        .as_object_mut()
        .context("its \"hooks\" setting isn't an object")?;
    for (event, _) in EVENTS {
        let Some(entries) = hooks.get_mut(event).and_then(Value::as_array_mut) else {
            continue;
        };
        entries.retain(|entry| !is_ours(entry));
        if entries.is_empty() {
            hooks.remove(event);
        }
    }
    if hooks.is_empty() {
        settings.remove("hooks");
    }
    Ok(())
}

fn add_ours(
    settings: &mut Map<String, Value>,
    ccc: &Path,
    state_dir: &Path,
    enhance: bool,
) -> anyhow::Result<()> {
    let hooks = settings
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .context("its \"hooks\" setting isn't an object")?;
    let extra = enhance.then_some(("UserPromptSubmit", "enhance-prompt"));
    for (event, handler) in EVENTS.into_iter().chain(extra) {
        let command = format!(
            "{} hook {handler} --state-dir {}",
            shell_quote(&ccc.to_string_lossy()),
            shell_quote(&state_dir.to_string_lossy()),
        );
        let entries = hooks
            .entry(event)
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .with_context(|| format!("its {event} hooks aren't a list"))?;
        entries.push(json!({ "hooks": [{
            "type": "command",
            "command": command,
            "timeout": HOOK_TIMEOUT_SECONDS,
        }] }));
    }
    Ok(())
}

/// Adds the hooks to `settings`, replacing any earlier install.
pub(crate) fn install_hooks(
    settings: &mut Map<String, Value>,
    ccc: &Path,
    state_dir: &Path,
) -> anyhow::Result<()> {
    remove_ours(settings)?;
    add_ours(settings, ccc, state_dir, false)
}

/// Takes the hooks out of `settings`.
pub(crate) fn remove_hooks(settings: &mut Map<String, Value>) -> anyhow::Result<()> {
    remove_ours(settings)
}

/// Whether every hook is in place.
pub(crate) fn hooks_installed(settings: &Map<String, Value>) -> bool {
    EVENTS.iter().all(|(event, _)| {
        settings
            .get("hooks")
            .and_then(|hooks| hooks.get(*event))
            .and_then(Value::as_array)
            .is_some_and(|entries| entries.iter().any(is_ours))
    })
}

/// What the settings should become. An existing install is replaced, so a moved binary is fixed
/// by running this again.
fn updated(
    text: &str,
    uninstall: bool,
    enhance: bool,
    ccc: &Path,
    state_dir: &Path,
) -> anyhow::Result<Map<String, Value>> {
    let mut settings = if text.trim().is_empty() {
        Map::new()
    } else {
        match serde_json::from_str::<Value>(text).context("it isn't valid JSON")? {
            Value::Object(settings) => settings,
            _ => bail!("it isn't a JSON object"),
        }
    };
    remove_ours(&mut settings)?;
    if !uninstall {
        add_ours(&mut settings, ccc, state_dir, enhance)?;
    }
    Ok(settings)
}

pub(crate) fn run(args: &InitArgs, claude_home: &Path) -> anyhow::Result<()> {
    let path = args
        .settings_file
        .clone()
        .unwrap_or_else(|| claude_home.join("settings.json"));
    let text = match fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(err) if err.kind() == ErrorKind::NotFound => None,
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    let ccc = std::env::current_exe().context("finding this program")?;
    let state_dir = args.state.dir()?;
    let settings = updated(
        text.as_deref().unwrap_or(""),
        args.uninstall,
        args.enhance,
        &ccc,
        &state_dir,
    )
    .with_context(|| format!("leaving {} unchanged", path.display()))?;
    let rendered = format!(
        "{}\n",
        serde_json::to_string_pretty(&Value::Object(settings)).context("rendering settings")?
    );
    let mut out = std::io::stdout();
    if args.dry_run {
        writeln!(out, "{rendered}")?;
        return Ok(());
    }
    if text.as_deref() == Some(rendered.as_str()) {
        writeln!(out, "{} already has this.", path.display())?;
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if text.is_some() {
        let backup = path.with_extension("json.ccc-backup");
        fs::copy(&path, &backup).with_context(|| format!("writing {}", backup.display()))?;
        writeln!(out, "Kept a copy at {}.", backup.display())?;
    }
    let temporary = path.with_extension("json.ccc-new");
    fs::write(&temporary, rendered).with_context(|| format!("writing {}", temporary.display()))?;
    fs::rename(&temporary, &path).with_context(|| format!("replacing {}", path.display()))?;
    writeln!(
        out,
        "{} {}. It applies to new sessions.",
        if args.uninstall {
            "Removed cost warnings from"
        } else {
            "Added cost warnings to"
        },
        path.display()
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(text: &str, uninstall: bool) -> Value {
        Value::Object(
            updated(
                text,
                uninstall,
                false,
                Path::new("/bin/my ccc"),
                Path::new("/state"),
            )
            .unwrap(),
        )
    }

    #[test]
    fn adds_to_an_empty_file() {
        let settings = apply("", false);
        for (event, handler) in EVENTS {
            let command = settings["hooks"][event][0]["hooks"][0]["command"]
                .as_str()
                .unwrap();
            assert!(command.starts_with("'/bin/my ccc' hook "), "{command}");
            assert!(command.contains(handler), "{command}");
        }
    }

    #[test]
    fn keeps_everything_else_in_the_file() {
        let existing = json!({
            "model": "opus",
            "hooks": {
                "SessionStart": [{ "hooks": [{ "type": "command", "command": "mine" }] }],
                "Stop": [{ "hooks": [{ "type": "command", "command": "other" }] }],
            },
        })
        .to_string();
        let settings = apply(&existing, false);
        assert_eq!(settings["model"], "opus");
        assert_eq!(settings["hooks"]["Stop"][0]["hooks"][0]["command"], "other");
        assert_eq!(
            settings["hooks"]["SessionStart"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            settings["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            "mine"
        );
    }

    #[test]
    fn running_twice_changes_nothing() {
        let once = apply("", false);
        let twice = apply(&once.to_string(), false);
        assert_eq!(once, twice);
    }

    #[test]
    fn uninstalling_restores_what_was_there() {
        let existing = json!({
            "model": "opus",
            "hooks": { "SessionStart": [{ "hooks": [{ "command": "mine" }] }] },
        });
        let installed = apply(&existing.to_string(), false);
        assert_eq!(apply(&installed.to_string(), true), existing);

        let fresh = apply("", false);
        assert_eq!(apply(&fresh.to_string(), true), json!({}));
    }

    #[test]
    fn enhancing_adds_its_hook_and_uninstalling_removes_it() {
        let enhanced =
            Value::Object(updated("", false, true, Path::new("/c"), Path::new("/s")).unwrap());
        let commands = enhanced["hooks"]["UserPromptSubmit"].to_string();
        assert!(commands.contains("enhance-prompt"), "{commands}");
        assert!(commands.contains("advise-prompt"), "{commands}");
        assert_eq!(apply(&enhanced.to_string(), true), json!({}));
    }

    #[test]
    fn refuses_a_file_it_cannot_understand() {
        for text in ["{ not json", "[1, 2]", r#"{"hooks": 3}"#] {
            assert!(
                updated(text, false, false, Path::new("/c"), Path::new("/s")).is_err(),
                "{text}"
            );
        }
    }
}

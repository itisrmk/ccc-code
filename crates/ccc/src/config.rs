//! Which local models to use, kept in `config.toml` in the state directory.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use toml::Table;

const FILE: &str = "config.toml";
pub(crate) const DEFAULT_EMBEDDING: &str = "embeddinggemma-300M-Q8_0.gguf";
pub(crate) const DEFAULT_ANSWERING: &str = "gemma-4-E2B-it-Q4_0.gguf";

/// Where models run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Backend {
    /// Inside the background service, from files in the state directory's models folder.
    Builtin,
    /// In something else already running here, such as Ollama, by its OpenAI-compatible API.
    Server { url: String },
}

/// How far a job's allowed to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum JobMode {
    /// Does nothing. The default until a job has shown it helps on real work.
    #[default]
    Off,
    /// Offers something for the person to accept.
    Suggest,
    /// Acts by itself.
    On,
}

impl JobMode {
    pub(crate) const fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Suggest => "suggest",
            Self::On => "on",
        }
    }

    pub(crate) fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "suggest" => Some(Self::Suggest),
            "on" => Some(Self::On),
            _ => None,
        }
    }
}

/// The optional jobs a local model can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Jobs {
    /// Catches vague prompts. `suggest` holds one back once, with a sharper version.
    pub(crate) sharpen: JobMode,
    /// Adds a clearer version of every prompt beside the person's words.
    pub(crate) rewrite: JobMode,
    /// Shortens long command output before Claude sees it.
    pub(crate) trim: JobMode,
}

/// The local models this install uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalConfig {
    pub(crate) backend: Backend,
    /// A file name, or a server's model name.
    pub(crate) embedding: String,
    /// A file name, or a server's model name. `None` if this machine has too little memory.
    pub(crate) answering: Option<String>,
    /// What's been turned on.
    pub(crate) jobs: Jobs,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Builtin,
            embedding: DEFAULT_EMBEDDING.to_owned(),
            answering: Some(DEFAULT_ANSWERING.to_owned()),
            jobs: Jobs::default(),
        }
    }
}

impl LocalConfig {
    /// Reads the configuration. A missing file means the defaults. A broken one is an error,
    /// since guessing could mean loading gigabytes nobody asked for.
    pub(crate) fn load(state_dir: &Path) -> anyhow::Result<Self> {
        let path = state_dir.join(FILE);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
        };
        parse(&text).with_context(|| format!("reading {}", path.display()))
    }

    pub(crate) fn save(&self, state_dir: &Path) -> anyhow::Result<()> {
        let mut table = Table::new();
        if let Backend::Server { url } = &self.backend {
            table.insert("server_url".to_owned(), url.clone().into());
        }
        table.insert("embedding".to_owned(), self.embedding.clone().into());
        if let Some(answering) = &self.answering {
            table.insert("answering".to_owned(), answering.clone().into());
        }
        for (key, mode) in [
            ("sharpen", self.jobs.sharpen),
            ("rewrite", self.jobs.rewrite),
            ("trim", self.jobs.trim),
        ] {
            if mode != JobMode::Off {
                table.insert(key.to_owned(), mode.key().into());
            }
        }
        fs::create_dir_all(state_dir).context("creating the state directory")?;
        fs::write(state_dir.join(FILE), table.to_string()).context("writing the configuration")
    }

    pub(crate) fn model_file(state_dir: &Path, name: &str) -> PathBuf {
        state_dir.join("models").join(name)
    }
}

fn parse(text: &str) -> anyhow::Result<LocalConfig> {
    let table: Table = text.parse().context("not valid TOML")?;
    let mut config = LocalConfig {
        answering: None,
        ..LocalConfig::default()
    };
    for (key, value) in &table {
        let text = value
            .as_str()
            .with_context(|| format!("{key} must be text"))?
            .to_owned();
        match key.as_str() {
            "server_url" => config.backend = Backend::Server { url: text },
            "embedding" => config.embedding = text,
            "answering" => config.answering = Some(text),
            "sharpen" | "rewrite" | "trim" => {
                let mode = JobMode::parse(&text)
                    .with_context(|| format!("{key} must be off, suggest or on"))?;
                match key.as_str() {
                    "sharpen" => config.jobs.sharpen = mode,
                    "rewrite" => config.jobs.rewrite = mode,
                    _ => config.jobs.trim = mode,
                }
            }
            other => bail!("unknown setting {other:?}"),
        }
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ccc-cfg-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_missing_file_means_the_defaults() {
        let dir = dir("missing");
        assert_eq!(LocalConfig::load(&dir).unwrap(), LocalConfig::default());
    }

    #[test]
    fn saved_settings_come_back() {
        let dir = dir("round");
        let config = LocalConfig {
            backend: Backend::Server {
                url: "http://127.0.0.1:11434/v1".to_owned(),
            },
            embedding: "nomic".to_owned(),
            answering: None,
            jobs: Jobs {
                sharpen: JobMode::Suggest,
                rewrite: JobMode::On,
                trim: JobMode::Off,
            },
        };
        config.save(&dir).unwrap();
        assert_eq!(LocalConfig::load(&dir).unwrap(), config);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_typo_or_bad_value_is_an_error() {
        assert!(parse("embeding = \"x\"").is_err());
        assert!(parse("embedding = 3").is_err());
        assert!(parse("trim = \"maybe\"").is_err());
        assert!(parse("not toml =").is_err());
    }
}

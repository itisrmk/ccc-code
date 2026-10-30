use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use hmac::{Hmac, Mac};
use sha2::Sha256;

/// Size of the install's key.
const KEY_BYTES: usize = 32;
/// Hex digits of the keyed hash in a placeholder: 40 bits, enough that two secrets in one
/// install won't collide, and short enough not to waste tokens.
pub(crate) const PLACEHOLDER_HEX_DIGITS: usize = 10;
pub(crate) const PLACEHOLDER_PREFIX: &str = "<SECRET_";

/// The key file couldn't be read or created.
#[derive(Debug, thiserror::Error)]
#[error("opening redaction key {path:?}")]
pub struct OpenVaultError {
    path: PathBuf,
    #[source]
    source: std::io::Error,
}

/// Maps placeholders to the secrets they stand for. Secrets live only in memory.
pub struct Vault {
    key: [u8; KEY_BYTES],
    secrets: Mutex<HashMap<String, String>>,
}

impl fmt::Debug for Vault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.secrets.lock().expect("lock poisoned").len();
        formatter
            .debug_struct("Vault")
            .field("secrets", &format_args!("<{count} redacted>"))
            .finish_non_exhaustive()
    }
}

impl Vault {
    /// A vault with a key that lasts until the process exits.
    ///
    /// # Errors
    ///
    /// Returns [`OpenVaultError`] if the system's random source can't be read.
    pub fn ephemeral() -> Result<Self, OpenVaultError> {
        Ok(Self::with_key(random_key()?))
    }

    /// Opens the vault keyed by the file at `path`, creating the file readable only by
    /// this user if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`OpenVaultError`] if the key file can't be read or created.
    pub fn open(path: &Path) -> Result<Self, OpenVaultError> {
        let describe = |source| OpenVaultError {
            path: path.to_owned(),
            source,
        };
        if let Ok(existing) = fs::read(path)
            && let Ok(key) = <[u8; KEY_BYTES]>::try_from(existing.as_slice())
        {
            return Ok(Self::with_key(key));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(describe)?;
        }
        let key = random_key()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .map_err(describe)?;
        file.write_all(&key).map_err(describe)?;
        Ok(Self::with_key(key))
    }

    fn with_key(key: [u8; KEY_BYTES]) -> Self {
        Self {
            key,
            secrets: Mutex::default(),
        }
    }

    /// The placeholder for `secret`, remembering the pair for rehydration.
    pub(crate) fn placeholder_for(&self, secret: &str) -> String {
        let mut mac =
            <Hmac<Sha256>>::new_from_slice(&self.key).expect("hmac accepts keys of any length");
        mac.update(secret.as_bytes());
        let digest = mac.finalize().into_bytes();
        let hex: String = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            .chars()
            .take(PLACEHOLDER_HEX_DIGITS)
            .collect();
        let placeholder = format!("{PLACEHOLDER_PREFIX}{hex}>");
        self.secrets
            .lock()
            .expect("lock poisoned")
            .insert(placeholder.clone(), secret.to_owned());
        placeholder
    }

    /// The secret behind `placeholder`, if it is one of ours.
    pub(crate) fn secret_for(&self, placeholder: &str) -> Option<String> {
        self.secrets
            .lock()
            .expect("lock poisoned")
            .get(placeholder)
            .cloned()
    }

    /// Whether any secret has been seen, so responses are worth scanning.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.secrets.lock().expect("lock poisoned").is_empty()
    }
}

fn random_key() -> Result<[u8; KEY_BYTES], OpenVaultError> {
    let path = Path::new("/dev/urandom");
    let describe = |source| OpenVaultError {
        path: path.to_owned(),
        source,
    };
    let mut key = [0; KEY_BYTES];
    File::open(path)
        .and_then(|mut source| source.read_exact(&mut key))
        .map_err(describe)?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn placeholders_are_stable_within_a_key_and_differ_across_keys() {
        let vault = Vault::with_key([1; KEY_BYTES]);
        let first = vault.placeholder_for("hunter2hunter2");
        assert_eq!(first, vault.placeholder_for("hunter2hunter2"));
        assert_ne!(first, vault.placeholder_for("other-secret"));
        assert_ne!(
            first,
            Vault::with_key([2; KEY_BYTES]).placeholder_for("hunter2hunter2")
        );
        assert_eq!(
            first.len(),
            PLACEHOLDER_PREFIX.len() + PLACEHOLDER_HEX_DIGITS + 1
        );
    }

    #[test]
    fn rehydration_only_knows_placeholders_it_issued() {
        let vault = Vault::with_key([1; KEY_BYTES]);
        let placeholder = vault.placeholder_for("hunter2hunter2");
        assert_eq!(vault.secret_for(&placeholder).unwrap(), "hunter2hunter2");
        assert_eq!(vault.secret_for("<SECRET_0000000000>"), None);
    }

    #[test]
    fn debug_output_never_shows_a_secret() {
        let vault = Vault::with_key([1; KEY_BYTES]);
        vault.placeholder_for("hunter2hunter2");
        let shown = format!("{vault:?}");
        assert!(!shown.contains("hunter2"), "{shown}");
        assert!(shown.contains("1 redacted"), "{shown}");
    }

    #[test]
    fn the_key_file_persists_placeholders_and_is_private() {
        let dir = std::env::temp_dir().join(format!("ccc-vault-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("redact.key");

        let first = Vault::open(&path)
            .unwrap()
            .placeholder_for("hunter2hunter2");
        let second = Vault::open(&path)
            .unwrap()
            .placeholder_for("hunter2hunter2");

        assert_eq!(first, second);
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        fs::remove_dir_all(dir).unwrap();
    }
}

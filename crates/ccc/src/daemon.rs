//! The background service: holds local models so a hook can use them without loading gigabytes.
//!
//! It starts when first needed and exits when idle, which is the only reliable way to give the
//! memory back. Everything else treats it as optional: a hook that can't reach it in time just
//! adds nothing. Answers go over a Unix socket that only this user can open.

use std::fs;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail};
use ccc_slm::{Completion, CompletionRequest, Embedder, ModelError, ServerModel, TextModel};
use serde_json::{Value, json};

use crate::config::{Backend, LocalConfig};

const SOCKET: &str = "daemon.sock";
/// How long with nothing to do before giving the memory back.
pub(crate) const DEFAULT_IDLE: Duration = Duration::from_secs(10 * 60);
const ACCEPT_POLL: Duration = Duration::from_millis(50);
/// How long a real service gets to answer before its socket counts as abandoned.
const LIVENESS_TIMEOUT: Duration = Duration::from_millis(500);
/// Time for a freshly started service to open its socket. Models load later, on first use.
const START_WAIT: Duration = Duration::from_secs(3);
const START_POLL: Duration = Duration::from_millis(25);
/// Allows for what happens around a model call that has its own deadline.
const REPLY_MARGIN: Duration = Duration::from_secs(5);
/// Long enough for a model to load on its first use.
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);

pub(crate) fn socket_path(state_dir: &Path) -> PathBuf {
    state_dir.join(SOCKET)
}

/// What the service does when asked, so its plumbing can be tested without real weights.
pub(crate) trait Backends: Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError>;
    fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError>;
}

/// The configured models, each loaded on first use.
pub(crate) struct ConfiguredBackends {
    config: LocalConfig,
    state_dir: PathBuf,
    embedder: OnceLock<Box<dyn Embedder>>,
    answerer: OnceLock<Box<dyn TextModel>>,
    /// Loading is slow, so concurrent callers wait for one load instead of each starting one.
    loading: Mutex<()>,
}

impl ConfiguredBackends {
    pub(crate) fn new(config: LocalConfig, state_dir: &Path) -> Self {
        Self {
            config,
            state_dir: state_dir.to_owned(),
            embedder: OnceLock::new(),
            answerer: OnceLock::new(),
            loading: Mutex::new(()),
        }
    }

    fn load_embedder(&self) -> Result<Box<dyn Embedder>, String> {
        match &self.config.backend {
            Backend::Server { url } => Ok(Box::new(ServerModel::new(url, &self.config.embedding))),
            Backend::Builtin => self.builtin_embedder(),
        }
    }

    fn load_answerer(&self) -> Result<Box<dyn TextModel>, String> {
        let name = self
            .config
            .answering
            .as_deref()
            .ok_or("this install has no answering model; ccc setup can add one")?;
        match &self.config.backend {
            Backend::Server { url } => Ok(Box::new(ServerModel::new(url, name))),
            Backend::Builtin => self.builtin_answerer(name),
        }
    }

    #[cfg(feature = "local-model")]
    fn builtin_embedder(&self) -> Result<Box<dyn Embedder>, String> {
        let path = LocalConfig::model_file(&self.state_dir, &self.config.embedding);
        ccc_slm::LlamaEmbedder::load(&path)
            .map(|model| Box::new(model) as Box<dyn Embedder>)
            .map_err(|err| err.to_string())
    }

    #[cfg(feature = "local-model")]
    fn builtin_answerer(&self, name: &str) -> Result<Box<dyn TextModel>, String> {
        let path = LocalConfig::model_file(&self.state_dir, name);
        let config = ccc_slm::LlamaConfig {
            format: ccc_slm::ChatFormat::Gemma4,
            ..ccc_slm::LlamaConfig::default()
        };
        ccc_slm::LlamaModel::load(&path, config)
            .map(|model| Box::new(model) as Box<dyn TextModel>)
            .map_err(|err| err.to_string())
    }

    #[cfg(not(feature = "local-model"))]
    fn builtin_embedder(&self) -> Result<Box<dyn Embedder>, String> {
        Err(NO_BUILTIN.to_owned())
    }

    #[cfg(not(feature = "local-model"))]
    fn builtin_answerer(&self, _: &str) -> Result<Box<dyn TextModel>, String> {
        Err(NO_BUILTIN.to_owned())
    }
}

#[cfg(not(feature = "local-model"))]
const NO_BUILTIN: &str =
    "this build has no built-in model runner; point ccc setup --server at Ollama or LM Studio";

impl Backends for ConfiguredBackends {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        if self.embedder.get().is_none() {
            let _guard = self.loading.lock().expect("lock poisoned");
            if self.embedder.get().is_none() {
                let model = self.load_embedder().map_err(ModelError::Inference)?;
                let _ = self.embedder.set(model);
            }
        }
        self.embedder.get().ok_or(ModelError::Stopped)?.embed(text)
    }

    /// A failed load isn't remembered, so a model that finishes downloading gets picked up.
    fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
        if self.answerer.get().is_none() {
            let _guard = self.loading.lock().expect("lock poisoned");
            if self.answerer.get().is_none() {
                let model = self.load_answerer().map_err(ModelError::Inference)?;
                let _ = self.answerer.set(model);
            }
        }
        self.answerer
            .get()
            .ok_or(ModelError::Stopped)?
            .complete(request)
    }
}

fn answer(backends: &dyn Backends, request: &Value) -> Value {
    let text_of = |key: &str| request.get(key).and_then(Value::as_str);
    let result = match text_of("op") {
        Some("ping") => Ok(json!({ "pid": std::process::id() })),
        Some("embed") => match text_of("text") {
            Some(text) => backends
                .embed(text)
                .map(|vector| json!({ "vector": vector }))
                .map_err(|err| err.to_string()),
            None => Err("embed needs text".to_owned()),
        },
        Some("complete") => match text_of("prompt") {
            Some(prompt) => {
                let number = |key: &str| request.get(key).and_then(Value::as_u64).unwrap_or(0);
                backends
                    .complete(&CompletionRequest {
                        prompt,
                        grammar: text_of("grammar"),
                        max_tokens: u32::try_from(number("max_tokens")).unwrap_or(u32::MAX),
                        deadline: Duration::from_millis(number("deadline_ms")),
                    })
                    .map(|completion| json!({ "text": completion.text }))
                    .map_err(|err| err.to_string())
            }
            None => Err("complete needs a prompt".to_owned()),
        },
        other => Err(format!("unknown request {other:?}")),
    };
    result.unwrap_or_else(|error| json!({ "error": error }))
}

fn handle(backends: &dyn Backends, stream: UnixStream) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    let reply = serde_json::from_str::<Value>(&line).map_or_else(
        |_| json!({ "error": "not JSON" }),
        |request| answer(backends, &request),
    );
    let mut stream = stream;
    writeln!(stream, "{reply}")
}

/// Whether a real service is behind this socket. Connecting isn't enough: a process starting at
/// that moment can briefly inherit a dead service's handle, which accepts and then says nothing.
fn answers(path: &Path) -> bool {
    let Ok(stream) = UnixStream::connect(path) else {
        return false;
    };
    let timeouts = stream
        .set_read_timeout(Some(LIVENESS_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(LIVENESS_TIMEOUT)));
    let mut writer = &stream;
    if timeouts.is_err() || writeln!(writer, "{}", json!({ "op": "ping" })).is_err() {
        return false;
    }
    let mut line = String::new();
    if BufReader::new(&stream).read_line(&mut line).is_err() {
        return false;
    }
    serde_json::from_str::<Value>(&line).is_ok_and(|reply| reply.get("pid").is_some())
}

/// Claims the socket, clearing one left by a service that died. Fails if another is running.
fn bind(path: &Path) -> anyhow::Result<UnixListener> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("creating the state directory")?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).ok();
    }
    if path.exists() {
        if answers(path) {
            bail!("the service is already running");
        }
        fs::remove_file(path).context("clearing an old socket")?;
    }
    let listener = UnixListener::bind(path).context("opening the socket")?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .context("restricting the socket")?;
    listener
        .set_nonblocking(true)
        .context("configuring the socket")?;
    Ok(listener)
}

/// Serves until `idle` passes with nothing to do, then removes its socket. Models are dropped by
/// the caller after this returns, which matters: Metal aborts at exit if they're still held.
pub(crate) fn serve(
    backends: &dyn Backends,
    state_dir: &Path,
    idle: Duration,
) -> anyhow::Result<()> {
    let path = socket_path(state_dir);
    let listener = bind(&path)?;
    let started = Instant::now();
    let elapsed = || u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let last_used = AtomicU64::new(0);
    let active = AtomicUsize::new(0);
    std::thread::scope(|scope| -> anyhow::Result<()> {
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    active.fetch_add(1, Ordering::SeqCst);
                    let (active, last_used) = (&active, &last_used);
                    scope.spawn(move || {
                        if let Err(err) = handle(backends, stream) {
                            tracing::debug!(error = %err, "a client went away");
                        }
                        last_used.store(elapsed(), Ordering::SeqCst);
                        active.fetch_sub(1, Ordering::SeqCst);
                    });
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    let quiet = elapsed().saturating_sub(last_used.load(Ordering::SeqCst));
                    let idle_millis = u64::try_from(idle.as_millis()).unwrap_or(u64::MAX);
                    if active.load(Ordering::SeqCst) == 0 && quiet >= idle_millis {
                        return Ok(());
                    }
                    std::thread::sleep(ACCEPT_POLL);
                }
                Err(err) => return Err(err).context("accepting a client"),
            }
        }
    })?;
    fs::remove_file(&path).ok();
    Ok(())
}

/// Talks to the service, starting it if needed.
#[derive(Debug, Clone)]
pub(crate) struct DaemonClient {
    state_dir: PathBuf,
    /// Overrides how long to wait for an embedding, for callers with a person waiting.
    embed_timeout: Duration,
    start_if_needed: bool,
}

impl DaemonClient {
    pub(crate) fn new(state_dir: &Path) -> Self {
        Self {
            state_dir: state_dir.to_owned(),
            embed_timeout: EMBED_TIMEOUT,
            start_if_needed: true,
        }
    }

    /// For a hook: gives up quickly, since no help is better than a slow prompt.
    pub(crate) fn impatient(state_dir: &Path, timeout: Duration) -> Self {
        Self {
            embed_timeout: timeout,
            ..Self::new(state_dir)
        }
    }

    /// Only talks to a service that's already running.
    pub(crate) fn existing_only(state_dir: &Path) -> Self {
        Self {
            start_if_needed: false,
            ..Self::new(state_dir)
        }
    }

    fn connect(&self) -> Result<UnixStream, ModelError> {
        let path = socket_path(&self.state_dir);
        if let Ok(stream) = UnixStream::connect(&path) {
            return Ok(stream);
        }
        if !self.start_if_needed {
            return Err(ModelError::Stopped);
        }
        self.start();
        let started = Instant::now();
        while started.elapsed() < START_WAIT {
            if let Ok(stream) = UnixStream::connect(&path) {
                return Ok(stream);
            }
            std::thread::sleep(START_POLL);
        }
        Err(ModelError::Stopped)
    }

    /// Starts the service detached, so it outlives whichever short-lived hook needed it. If two
    /// race, the loser exits when it finds the socket taken.
    fn start(&self) {
        let Ok(program) = std::env::current_exe() else {
            return;
        };
        let spawned = Command::new(program)
            .arg("daemon")
            .arg("--state-dir")
            .arg(&self.state_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn();
        if let Err(err) = spawned {
            tracing::warn!(error = %err, "could not start the service");
        }
    }

    fn call(&self, request: &Value, timeout: Duration) -> Result<Value, ModelError> {
        let io_error = |err: std::io::Error| match err.kind() {
            ErrorKind::WouldBlock | ErrorKind::TimedOut => ModelError::DeadlineExceeded,
            _ => ModelError::Stopped,
        };
        let stream = self.connect()?;
        stream.set_read_timeout(Some(timeout)).map_err(io_error)?;
        stream.set_write_timeout(Some(timeout)).map_err(io_error)?;
        let mut writer = &stream;
        writeln!(writer, "{request}").map_err(io_error)?;
        let mut line = String::new();
        BufReader::new(&stream)
            .read_line(&mut line)
            .map_err(io_error)?;
        let reply: Value =
            serde_json::from_str(&line).map_err(|err| ModelError::Inference(err.to_string()))?;
        match reply.get("error").and_then(Value::as_str) {
            Some(error) => Err(ModelError::Inference(error.to_owned())),
            None => Ok(reply),
        }
    }

    /// The running service's process id, without starting one.
    pub(crate) fn ping(&self) -> Option<u32> {
        let client = Self::existing_only(&self.state_dir);
        let reply = client
            .call(&json!({ "op": "ping" }), Duration::from_secs(2))
            .ok()?;
        reply
            .get("pid")
            .and_then(Value::as_u64)
            .and_then(|pid| u32::try_from(pid).ok())
    }
}

impl Embedder for DaemonClient {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        let reply = self.call(&json!({ "op": "embed", "text": text }), self.embed_timeout)?;
        reply
            .get("vector")
            .and_then(Value::as_array)
            .and_then(|items| {
                items
                    .iter()
                    .map(|number| number.as_f64().map(|value| value as f32))
                    .collect()
            })
            .ok_or_else(|| ModelError::Inference("no vector in the reply".to_owned()))
    }
}

impl TextModel for DaemonClient {
    fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
        let reply = self.call(
            &json!({
                "op": "complete",
                "prompt": request.prompt,
                "grammar": request.grammar,
                "max_tokens": request.max_tokens,
                "deadline_ms": u64::try_from(request.deadline.as_millis()).unwrap_or(u64::MAX),
            }),
            request.deadline + REPLY_MARGIN + EMBED_TIMEOUT,
        )?;
        let text = reply
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| ModelError::Inference("no text in the reply".to_owned()))?;
        Ok(Completion {
            text: text.to_owned(),
            ..Completion::default()
        })
    }
}

/// `ccc daemon`: runs the service until it's been idle long enough.
pub(crate) fn run(state_dir: &Path, idle: Duration) -> anyhow::Result<()> {
    let config = LocalConfig::load(state_dir)?;
    let backends = ConfiguredBackends::new(config, state_dir);
    serve(&backends, state_dir, idle)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;

    impl Backends for Fake {
        fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
            if text == "fail" {
                return Err(ModelError::Inference("broken".to_owned()));
            }
            Ok(vec![0.5, 1.5])
        }

        fn complete(&self, request: &CompletionRequest<'_>) -> Result<Completion, ModelError> {
            if request.prompt == "slow" {
                std::thread::sleep(Duration::from_millis(600));
            }
            Ok(Completion {
                text: format!("{} {}", request.prompt, request.max_tokens),
                ..Completion::default()
            })
        }
    }

    fn state(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cccd-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn running(dir: &Path, idle: Duration) -> std::thread::JoinHandle<()> {
        let dir = dir.to_owned();
        std::thread::spawn(move || serve(&Fake, &dir, idle).unwrap())
    }

    fn wait_for_socket(dir: &Path) {
        let started = Instant::now();
        while UnixStream::connect(socket_path(dir)).is_err() {
            assert!(started.elapsed() < Duration::from_secs(5), "no socket");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn request(prompt: &str) -> CompletionRequest<'_> {
        CompletionRequest {
            prompt,
            grammar: None,
            max_tokens: 7,
            deadline: Duration::from_secs(5),
        }
    }

    #[test]
    fn answers_requests_then_exits_when_idle_and_cleans_up() {
        let dir = state("serve");
        let server = running(&dir, Duration::from_millis(300));
        wait_for_socket(&dir);
        let client = DaemonClient::existing_only(&dir);

        assert_eq!(client.embed("x").unwrap(), [0.5, 1.5]);
        assert_eq!(client.complete(&request("hi")).unwrap().text, "hi 7");
        assert_eq!(client.ping(), Some(std::process::id()));
        assert!(client.embed("fail").is_err(), "failures reach the caller");

        server.join().unwrap();
        assert!(!socket_path(&dir).exists(), "the socket is removed");
        assert_eq!(client.ping(), None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_client_with_no_service_fails_quickly_instead_of_hanging() {
        let dir = state("none");
        let started = Instant::now();
        assert_eq!(
            DaemonClient::existing_only(&dir).embed("x"),
            Err(ModelError::Stopped)
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_socket_left_by_a_dead_service_is_cleared() {
        let dir = state("stale");
        drop(UnixListener::bind(socket_path(&dir)).unwrap());
        assert!(socket_path(&dir).exists());
        let server = running(&dir, Duration::from_millis(100));
        wait_for_socket(&dir);
        server.join().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn something_that_accepts_but_says_nothing_isnt_a_service() {
        let dir = state("silent");
        let silent = UnixListener::bind(socket_path(&dir)).unwrap();
        assert!(!answers(&socket_path(&dir)));
        drop(silent);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_second_service_refuses_to_start() {
        let dir = state("second");
        let server = running(&dir, Duration::from_millis(400));
        wait_for_socket(&dir);
        assert!(serve(&Fake, &dir, Duration::from_millis(1)).is_err());
        server.join().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_socket_is_private_to_this_user() {
        let dir = state("private");
        let server = running(&dir, Duration::from_millis(200));
        wait_for_socket(&dir);
        let mode = fs::metadata(socket_path(&dir))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0, "{mode:o}");
        server.join().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_impatient_caller_stops_waiting_and_the_service_survives() {
        let dir = state("impatient");
        let server = running(&dir, Duration::from_millis(900));
        wait_for_socket(&dir);
        let mut client = DaemonClient::existing_only(&dir);
        client.embed_timeout = Duration::from_millis(50);
        let slow = client.call(
            &json!({ "op": "complete", "prompt": "slow", "max_tokens": 1, "deadline_ms": 1000 }),
            Duration::from_millis(50),
        );
        assert_eq!(slow, Err(ModelError::DeadlineExceeded));
        assert_eq!(
            DaemonClient::existing_only(&dir).embed("x").unwrap(),
            [0.5, 1.5]
        );
        server.join().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nonsense_gets_an_error_and_not_a_crash() {
        assert!(answer(&Fake, &json!({ "op": "what" }))["error"].is_string());
        assert!(answer(&Fake, &json!({ "op": "embed" }))["error"].is_string());
        assert!(answer(&Fake, &json!({ "op": "complete" }))["error"].is_string());
    }
}

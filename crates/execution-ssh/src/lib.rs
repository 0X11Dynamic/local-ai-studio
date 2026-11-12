//! SSH execution environment (spec section 8): commands over an exec
//! channel, files over SFTP, strict known_hosts checking. Unknown or changed
//! host keys are refused; the user trusts a new host explicitly, after
//! comparing its fingerprint, and it is recorded in the app's own
//! known_hosts file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use execution::{Capabilities, CommandSpec, DirEntry, ExecResult, ExecutionEnvironment, HARD_MAX_OUTPUT, Health, ProcessEvent, Shell, now};
use russh::ChannelMsg;
use russh::client;
use russh::keys::{HashAlg, PrivateKeyWithHashAlg, PublicKey, PublicKeyOrCertificate};
use russh_sftp::client::SftpSession;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SshAuth {
    /// Private key file (OpenSSH or PEM); passphrase from the vault.
    Key { path: String, passphrase: Option<String> },
    Password { password: String },
}

impl std::fmt::Debug for SshAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SshAuth::Key { path, .. } => write!(f, "Key({path})"),
            SshAuth::Password { .. } => write!(f, "Password(***)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SshConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub auth: SshAuth,
    /// The app's known_hosts file (e.g. %APPDATA%\LocalAIStudio\known_hosts).
    pub known_hosts: PathBuf,
    /// Remote folders file tools may use. Empty means the remote account's
    /// own permissions are the only limit.
    pub roots: Vec<String>,
    /// `Sh` for Linux/macOS hosts, `PowerShell` for Windows OpenSSH hosts.
    pub shell: Shell,
    pub connect_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostKeyProblem {
    Unknown { fingerprint: String, algorithm: String },
    Changed { fingerprint: String, algorithm: String, line: usize },
}

struct Client {
    host: String,
    port: u16,
    known_hosts: PathBuf,
    problem: Arc<StdMutex<Option<HostKeyProblem>>>,
    /// Capture mode (fetching a key to show the user) accepts nothing.
    seen: Arc<StdMutex<Option<PublicKey>>>,
    capture_only: bool,
}

pub fn fingerprint(k: &PublicKey) -> String {
    k.fingerprint(HashAlg::Sha256).to_string()
}

impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(&mut self, key: &PublicKeyOrCertificate) -> Result<bool, Self::Error> {
        let PublicKeyOrCertificate::PublicKey { key, .. } = key else {
            // Host certificates are not supported: refuse rather than guess.
            return Ok(false);
        };
        *self.seen.lock().unwrap() = Some(key.clone());
        if self.capture_only {
            return Ok(false);
        }
        let fp = fingerprint(key);
        let alg = key.algorithm().to_string();
        let verdict = if self.known_hosts.exists() { russh::keys::check_known_hosts_path(&self.host, self.port, key, &self.known_hosts) } else { Ok(false) };
        match verdict {
            Ok(true) => Ok(true),
            Ok(false) => {
                *self.problem.lock().unwrap() = Some(HostKeyProblem::Unknown { fingerprint: fp, algorithm: alg });
                Ok(false)
            }
            Err(russh::keys::Error::KeyChanged { line }) => {
                *self.problem.lock().unwrap() = Some(HostKeyProblem::Changed { fingerprint: fp, algorithm: alg, line });
                Ok(false)
            }
            Err(_) => {
                *self.problem.lock().unwrap() = Some(HostKeyProblem::Unknown { fingerprint: fp, algorithm: alg });
                Ok(false)
            }
        }
    }
}

fn client_config() -> Arc<client::Config> {
    Arc::new(client::Config { inactivity_timeout: None, keepalive_interval: Some(Duration::from_secs(30)), keepalive_max: 3, ..Default::default() })
}

fn host_err(p: &HostKeyProblem, host: &str, port: u16) -> AppError {
    match p {
        HostKeyProblem::Unknown { fingerprint, algorithm } => AppError::PermissionDenied(format!(
            "unknown host key for {host}:{port} ({algorithm} {fingerprint}). Compare it with the server's real fingerprint, then trust it in Execution Environments"
        )),
        HostKeyProblem::Changed { fingerprint, algorithm, line } => AppError::PermissionDenied(format!(
            "HOST KEY CHANGED for {host}:{port}: now {algorithm} {fingerprint}, which does not match known_hosts line {line}. This can mean someone is intercepting the connection. Not connecting"
        )),
    }
}

/// Connect without authenticating and return the server's host key, so the
/// user can compare its fingerprint before trusting it.
pub async fn fetch_host_key(host: &str, port: u16, timeout: Duration) -> AppResult<(String, String, String)> {
    let seen = Arc::new(StdMutex::new(None));
    let c = Client { host: host.into(), port, known_hosts: PathBuf::new(), problem: Default::default(), seen: seen.clone(), capture_only: true };
    let _ = tokio::time::timeout(timeout, client::connect(client_config(), (host, port), c))
        .await
        .map_err(|_| AppError::Runtime(format!("timed out connecting to {host}:{port}")))?;
    let k = seen.lock().unwrap().clone().ok_or_else(|| AppError::Runtime(format!("{host}:{port} did not offer a host key")))?;
    Ok((fingerprint(&k), k.algorithm().to_string(), k.to_openssh().map_err(|e| AppError::Runtime(e.to_string()))?))
}

/// Record a host key after the user confirmed `expected_fingerprint`. The key
/// is fetched again and must still match.
pub async fn trust_host(host: &str, port: u16, known_hosts: &Path, expected_fingerprint: &str) -> AppResult<()> {
    let seen = Arc::new(StdMutex::new(None));
    let c = Client { host: host.into(), port, known_hosts: PathBuf::new(), problem: Default::default(), seen: seen.clone(), capture_only: true };
    let _ = tokio::time::timeout(Duration::from_secs(20), client::connect(client_config(), (host, port), c)).await;
    let k = seen.lock().unwrap().clone().ok_or_else(|| AppError::Runtime(format!("{host}:{port} did not offer a host key")))?;
    if fingerprint(&k) != expected_fingerprint.trim() {
        return Err(AppError::Conflict(format!("the host key is now {}, not the {} you confirmed; not trusting it", fingerprint(&k), expected_fingerprint)));
    }
    if known_hosts.exists() {
        match russh::keys::check_known_hosts_path(host, port, &k, known_hosts) {
            Ok(true) => return Ok(()),
            Err(russh::keys::Error::KeyChanged { line }) => {
                return Err(AppError::Conflict(format!("known_hosts line {line} has a different key for this host; remove it first if the server was really re-keyed")));
            }
            _ => {}
        }
    }
    russh::keys::known_hosts::learn_known_hosts_path(host, port, &k, known_hosts).map_err(|e| AppError::Runtime(format!("could not write known_hosts: {e}")))
}

struct Conn {
    handle: client::Handle<Client>,
    sftp: Option<SftpSession>,
}

pub struct SshEnv {
    pub cfg: SshConfig,
    conn: Mutex<Option<Conn>>,
}

pub fn sq(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// POSIX lexical normalization (`.`/`..` removed; `..` cannot climb past `/`).
fn normalize(p: &str) -> String {
    let mut parts: Vec<&str> = vec![];
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    format!("/{}", parts.join("/"))
}

fn within(p: &str, root: &str) -> bool {
    let r = root.trim_end_matches('/');
    r.is_empty() || p == r || p.starts_with(&format!("{r}/"))
}

const PGID_MARK: &str = "__LAS_PGID=";

impl SshEnv {
    pub fn new(cfg: SshConfig) -> Self {
        Self { cfg, conn: Mutex::new(None) }
    }

    async fn connect(&self) -> AppResult<Conn> {
        let c = &self.cfg;
        let problem = Arc::new(StdMutex::new(None));
        let handler = Client { host: c.host.clone(), port: c.port, known_hosts: c.known_hosts.clone(), problem: problem.clone(), seen: Default::default(), capture_only: false };
        let connected = tokio::time::timeout(c.connect_timeout, client::connect(client_config(), (c.host.as_str(), c.port), handler))
            .await
            .map_err(|_| AppError::Runtime(format!("timed out connecting to {}:{}", c.host, c.port)))?;
        let mut handle = match connected {
            Ok(h) => h,
            Err(e) => {
                if let Some(p) = problem.lock().unwrap().as_ref() {
                    return Err(host_err(p, &c.host, c.port));
                }
                return Err(AppError::Runtime(format!("could not connect to {}:{}: {e}", c.host, c.port)));
            }
        };
        let ok = match &c.auth {
            SshAuth::Password { password } => handle.authenticate_password(&c.user, password).await.map_err(|e| AppError::Runtime(e.to_string()))?.success(),
            SshAuth::Key { path, passphrase } => {
                let key = russh::keys::load_secret_key(path, passphrase.as_deref()).map_err(|e| AppError::InvalidInput(format!("could not load the key {path}: {e}")))?;
                let hash = handle.best_supported_rsa_hash().await.map_err(|e| AppError::Runtime(e.to_string()))?.flatten();
                handle.authenticate_publickey(&c.user, PrivateKeyWithHashAlg::new(Arc::new(key), hash)).await.map_err(|e| AppError::Runtime(e.to_string()))?.success()
            }
        };
        if !ok {
            return Err(AppError::PermissionDenied(format!("{}@{}:{} rejected the login", c.user, c.host, c.port)));
        }
        Ok(Conn { handle, sftp: None })
    }

    /// Run `f` with a live connection, reconnecting once if it dropped.
    async fn with_conn(&self, need_sftp: bool) -> AppResult<tokio::sync::MutexGuard<'_, Option<Conn>>> {
        let mut g = self.conn.lock().await;
        if g.as_ref().is_none_or(|c| c.handle.is_closed()) {
            *g = Some(self.connect().await?);
        }
        if need_sftp && g.as_ref().is_some_and(|c| c.sftp.is_none()) {
            let ch = g.as_ref().unwrap().handle.channel_open_session().await.map_err(|e| AppError::Runtime(e.to_string()))?;
            ch.request_subsystem(true, "sftp").await.map_err(|e| AppError::Runtime(e.to_string()))?;
            let s = SftpSession::new(ch.into_stream()).await.map_err(|e| AppError::Runtime(format!("SFTP is not available: {e}")))?;
            g.as_mut().unwrap().sftp = Some(s);
        }
        Ok(g)
    }

    /// Forget the cached session (it died or misbehaved); the next call
    /// connects again.
    async fn drop_conn(&self) {
        let old = self.conn.lock().await.take();
        if let Some(c) = old {
            // A dead session may not answer: do not wait long for the goodbye.
            let _ = tokio::time::timeout(Duration::from_secs(2), c.handle.disconnect(russh::Disconnect::ByApplication, "", "en")).await;
        }
    }

    /// Open a session channel; if the cached connection turns out to be dead,
    /// reconnect once and try again.
    async fn open_channel(&self) -> AppResult<russh::Channel<client::Msg>> {
        let mut last = String::new();
        for attempt in 0..2 {
            let res = {
                let g = self.with_conn(false).await?;
                g.as_ref().unwrap().handle.channel_open_session().await
            };
            match res {
                Ok(ch) => return Ok(ch),
                Err(e) => {
                    last = e.to_string();
                    self.drop_conn().await;
                    if attempt == 0 {
                        tracing::info!("SSH channel to {} failed ({e}); reconnecting once", self.label());
                    }
                }
            }
        }
        Err(AppError::Runtime(format!("could not open a channel to {}: {last}", self.label())))
    }

    /// Check a remote path against the roots, resolving symlinks in the part
    /// that exists (SFTP realpath).
    async fn resolve(&self, path: &str) -> AppResult<String> {
        if path.trim().is_empty() || path.contains('\0') || path.contains('\n') {
            return Err(AppError::InvalidInput("path is empty or contains a control character".into()));
        }
        let g = self.with_conn(true).await?;
        let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
        let base = match self.cfg.roots.first() {
            Some(r) => r.clone(),
            None => sftp.canonicalize(".").await.map_err(|e| AppError::Runtime(e.to_string()))?,
        };
        let abs = if path.starts_with('/') { normalize(path) } else { normalize(&format!("{base}/{path}")) };
        // Realpath the longest existing prefix.
        let mut existing = abs.clone();
        let mut rest = vec![];
        let real = loop {
            match sftp.canonicalize(existing.clone()).await {
                Ok(r) => break r,
                Err(_) => match existing.rsplit_once('/') {
                    Some((parent, name)) => {
                        rest.push(name.to_string());
                        existing = if parent.is_empty() { "/".into() } else { parent.into() };
                    }
                    None => break existing.clone(),
                },
            }
        };
        let mut full = real.trim_end_matches('/').to_string();
        for n in rest.into_iter().rev() {
            full.push('/');
            full.push_str(&n);
        }
        let full = if full.is_empty() { "/".into() } else { full };
        if !self.cfg.roots.is_empty() && !self.cfg.roots.iter().any(|r| within(&full, &normalize(r))) {
            return Err(AppError::PermissionDenied(format!("{full} is outside the allowed folders ({})", self.cfg.roots.join(", "))));
        }
        Ok(full)
    }

    fn wrap_command(&self, spec: &CommandSpec) -> AppResult<String> {
        match spec.shell {
            Shell::Sh => {
                let mut s = format!("echo {PGID_MARK}$$ >&2\n");
                for (k, v) in &spec.env {
                    if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        return Err(AppError::InvalidInput(format!("bad environment variable name {k:?}")));
                    }
                    s.push_str(&format!("export {k}={}\n", sq(v)));
                }
                if let Some(cwd) = &spec.cwd {
                    s.push_str(&format!("cd {} || exit 126\n", sq(cwd)));
                }
                s.push_str(&spec.command);
                // `exec` keeps the session leader's pid, so $$ is the process
                // group to kill on timeout.
                Ok(format!("exec sh -c {}", sq(&s)))
            }
            Shell::PowerShell => {
                let mut s = String::new();
                for (k, v) in &spec.env {
                    s.push_str(&format!("$env:{k} = '{}'\n", v.replace('\'', "''")));
                }
                if let Some(cwd) = &spec.cwd {
                    s.push_str(&format!("Set-Location -LiteralPath '{}'\n", cwd.replace('\'', "''")));
                }
                s.push_str(&spec.command);
                Ok(format!("powershell -NoLogo -NoProfile -NonInteractive -EncodedCommand {}", execution::local::encode_ps(&s)))
            }
            Shell::Cmd => Err(AppError::InvalidInput("cmd is not supported over SSH; use sh or PowerShell".into())),
        }
    }

    async fn run_simple(&self, command: &str) -> AppResult<ExecResult> {
        let spec = CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new(command, self.cfg.shell) };
        self.execute(&spec, &CancellationToken::new(), &|_| {}).await
    }
}

struct Capture {
    kept: Vec<u8>,
    line: Vec<u8>,
    truncated: bool,
    cap: usize,
    is_err: bool,
}

impl Capture {
    fn new(cap: usize, is_err: bool) -> Self {
        Self { kept: vec![], line: vec![], truncated: false, cap, is_err }
    }
    fn push(&mut self, data: &[u8], on_event: &(dyn Fn(ProcessEvent) + Send + Sync), pgid: &mut Option<i64>) {
        for &b in data {
            self.line.push(b);
            if b == b'\n' || self.line.len() >= 4096 {
                self.flush_line(on_event, pgid);
            }
        }
    }
    fn flush_line(&mut self, on_event: &(dyn Fn(ProcessEvent) + Send + Sync), pgid: &mut Option<i64>) {
        if self.line.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(&self.line).into_owned();
        self.line.clear();
        if self.is_err && pgid.is_none() && text.starts_with(PGID_MARK) {
            *pgid = text.trim()[PGID_MARK.len()..].parse().ok();
            return;
        }
        for &b in text.as_bytes() {
            if self.kept.len() < self.cap {
                self.kept.push(b);
            } else {
                self.truncated = true;
            }
        }
        on_event(if self.is_err { ProcessEvent::Stderr(text) } else { ProcessEvent::Stdout(text) });
    }
}

fn sftp_err(e: russh_sftp::client::error::Error, what: &str) -> AppError {
    let s = e.to_string();
    if s.to_lowercase().contains("no such file") {
        AppError::NotFound(format!("{what} does not exist"))
    } else if s.to_lowercase().contains("permission denied") {
        AppError::PermissionDenied(format!("{what}: permission denied on the remote host"))
    } else {
        AppError::Runtime(format!("{what}: {s}"))
    }
}

#[async_trait]
impl ExecutionEnvironment for SshEnv {
    fn kind(&self) -> &'static str {
        "ssh"
    }

    fn label(&self) -> String {
        format!("{}@{}:{}", self.cfg.user, self.cfg.host, self.cfg.port)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { shells: vec![self.cfg.shell], files: true, local: false, os: if self.cfg.shell == Shell::PowerShell { "windows".into() } else { "posix".into() } }
    }

    async fn health(&self) -> Health {
        let cmd = if self.cfg.shell == Shell::PowerShell { "$PSVersionTable.PSVersion.ToString(); hostname" } else { "uname -sr; hostname" };
        match self.run_simple(cmd).await {
            Ok(r) if r.succeeded() => Health { ok: true, detail: r.stdout.split_whitespace().collect::<Vec<_>>().join(" "), checked_at: now() },
            Ok(r) => Health { ok: false, detail: format!("exit {:?}: {}", r.exit_code, r.stderr.trim()), checked_at: now() },
            Err(e) => Health { ok: false, detail: e.to_string(), checked_at: now() },
        }
    }

    async fn execute(&self, spec: &CommandSpec, cancel: &CancellationToken, on_event: &(dyn Fn(ProcessEvent) + Send + Sync)) -> AppResult<ExecResult> {
        let command = self.wrap_command(spec)?;
        let started_at = now();
        let t0 = Instant::now();
        let mut ch = self.open_channel().await?;
        ch.exec(true, command.into_bytes()).await.map_err(|e| AppError::Runtime(format!("exec failed: {e}")))?;
        let _ = ch.eof().await;
        let cap = spec.max_output_bytes.clamp(1024, HARD_MAX_OUTPUT);
        let (mut out, mut err) = (Capture::new(cap, false), Capture::new(cap, true));
        let mut pgid = None;
        let mut exit_code = None;
        let (mut timed_out, mut cancelled) = (false, false);
        let deadline = tokio::time::sleep(spec.timeout);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                msg = ch.wait() => match msg {
                    Some(ChannelMsg::Data { data }) => out.push(&data, on_event, &mut pgid),
                    Some(ChannelMsg::ExtendedData { data, ext: 1 }) => err.push(&data, on_event, &mut pgid),
                    Some(ChannelMsg::ExitStatus { exit_status }) => exit_code = Some(exit_status as i32),
                    Some(ChannelMsg::ExitSignal { .. }) => exit_code = exit_code.or(Some(-1)),
                    Some(_) => {}
                    None => break,
                },
                _ = &mut deadline => { timed_out = true; break; }
                _ = cancel.cancelled() => { cancelled = true; break; }
            }
        }
        out.flush_line(on_event, &mut pgid);
        err.flush_line(on_event, &mut pgid);
        if timed_out || cancelled {
            if let (Some(g), Shell::Sh) = (pgid, spec.shell) {
                // A fresh channel: kill the whole remote process group.
                if let Ok(k) = self.open_channel().await {
                    let _ = k.exec(true, format!("kill -KILL -- -{g} 2>/dev/null; kill -KILL {g} 2>/dev/null; true").into_bytes()).await;
                    let mut k = k;
                    let _ = tokio::time::timeout(Duration::from_secs(10), async { while k.wait().await.is_some() {} }).await;
                }
            }
            let _ = ch.close().await;
        } else if exit_code.is_none() {
            // The channel closed without an exit status: the connection (or
            // the server) went away mid-command.
            self.drop_conn().await;
            let so_far = String::from_utf8_lossy(&out.kept).trim().chars().take(500).collect::<String>();
            return Err(AppError::Runtime(format!(
                "the SSH connection to {} dropped while the command was running{}; the next command reconnects",
                self.label(),
                if so_far.is_empty() { String::new() } else { format!(" (output so far: {so_far})") }
            )));
        }
        Ok(ExecResult {
            exit_code,
            stdout: String::from_utf8_lossy(&out.kept).into_owned(),
            stderr: String::from_utf8_lossy(&err.kept).into_owned(),
            truncated: out.truncated || err.truncated,
            timed_out,
            cancelled,
            duration_ms: t0.elapsed().as_millis() as u64,
            started_at,
            ended_at: now(),
        })
    }

    async fn read_file(&self, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)> {
        let p = self.resolve(path).await?;
        let g = self.with_conn(true).await?;
        let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
        let meta = sftp.metadata(p.clone()).await.map_err(|e| sftp_err(e, &p))?;
        if meta.is_dir() {
            return Err(AppError::InvalidInput(format!("{p} is a folder")));
        }
        use tokio::io::AsyncReadExt;
        let f = sftp.open(p.clone()).await.map_err(|e| sftp_err(e, &p))?;
        let mut buf = Vec::new();
        f.take(max_bytes as u64).read_to_end(&mut buf).await.map_err(|e| AppError::Runtime(e.to_string()))?;
        Ok((buf, meta.len() as usize > max_bytes))
    }

    async fn write_file(&self, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()> {
        let p = self.resolve(path).await?;
        let (parent, _) = p.rsplit_once('/').unwrap_or(("", ""));
        let parent = if parent.is_empty() { "/".to_string() } else { parent.to_string() };
        // Unique per write: concurrent writes to one file must not share a temp file.
        static WRITE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tmp = format!("{p}.las-tmp-{}-{}", std::process::id(), WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        {
            let g = self.with_conn(true).await?;
            let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
            if !sftp.try_exists(parent.clone()).await.unwrap_or(false) {
                if !create_dirs {
                    return Err(AppError::NotFound(format!("folder {parent} does not exist")));
                }
                // Create each missing level.
                let mut cur = String::new();
                for part in parent.split('/').filter(|s| !s.is_empty()) {
                    cur.push('/');
                    cur.push_str(part);
                    if !sftp.try_exists(cur.clone()).await.unwrap_or(false) {
                        sftp.create_dir(cur.clone()).await.map_err(|e| sftp_err(e, &cur))?;
                    }
                }
            }
            if sftp.metadata(p.clone()).await.is_ok_and(|m| m.is_dir()) {
                return Err(AppError::InvalidInput(format!("{p} is a folder")));
            }
            {
                use russh_sftp::protocol::OpenFlags;
                use tokio::io::AsyncWriteExt;
                let mut f = sftp.open_with_flags(tmp.clone(), OpenFlags::CREATE | OpenFlags::TRUNCATE | OpenFlags::WRITE).await.map_err(|e| sftp_err(e, &tmp))?;
                f.write_all(bytes).await.map_err(|e| AppError::Runtime(format!("writing {tmp}: {e}")))?;
                f.shutdown().await.map_err(|e| AppError::Runtime(format!("closing {tmp}: {e}")))?;
            }
            if self.cfg.shell != Shell::Sh {
                // No POSIX rename over an existing file: remove then rename.
                let _ = sftp.remove_file(p.clone()).await;
                return sftp.rename(tmp.clone(), p.clone()).await.map_err(|e| sftp_err(e, &p));
            }
        }
        let r = self.run_simple(&format!("mv -f -- {} {}", sq(&tmp), sq(&p))).await?;
        if !r.succeeded() {
            return Err(AppError::Runtime(format!("could not replace {p}: {}", r.stderr.trim())));
        }
        Ok(())
    }

    async fn list_dir(&self, path: &str) -> AppResult<Vec<DirEntry>> {
        let p = self.resolve(path).await?;
        let g = self.with_conn(true).await?;
        let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
        let rd = sftp.read_dir(p.clone()).await.map_err(|e| sftp_err(e, &p))?;
        let mut out: Vec<DirEntry> = rd
            .filter(|e| e.file_name() != "." && e.file_name() != "..")
            .take(5000)
            .map(|e| {
                let m = e.metadata();
                DirEntry {
                    name: e.file_name(),
                    path: format!("{}/{}", p.trim_end_matches('/'), e.file_name()),
                    is_dir: m.is_dir(),
                    size: if m.is_dir() { 0 } else { m.len() },
                    modified: m.mtime.map(|t| execution::local::rfc3339_from_unix(t as u64)),
                }
            })
            .collect();
        out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.to_lowercase().cmp(&b.name.to_lowercase())));
        Ok(out)
    }

    async fn stat(&self, path: &str) -> AppResult<Option<DirEntry>> {
        let p = self.resolve(path).await?;
        let g = self.with_conn(true).await?;
        let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
        Ok(sftp.metadata(p.clone()).await.ok().map(|m| DirEntry {
            name: p.rsplit('/').next().unwrap_or("").to_string(),
            path: p.clone(),
            is_dir: m.is_dir(),
            size: if m.is_dir() { 0 } else { m.len() },
            modified: m.mtime.map(|t| execution::local::rfc3339_from_unix(t as u64)),
        }))
    }

    async fn rename(&self, from: &str, to: &str) -> AppResult<()> {
        let (a, b) = (self.resolve(from).await?, self.resolve(to).await?);
        let g = self.with_conn(true).await?;
        let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
        if !sftp.try_exists(a.clone()).await.unwrap_or(false) {
            return Err(AppError::NotFound(format!("{a} does not exist")));
        }
        if sftp.try_exists(b.clone()).await.unwrap_or(false) {
            return Err(AppError::Conflict(format!("{b} already exists")));
        }
        sftp.rename(a.clone(), b).await.map_err(|e| sftp_err(e, &a))
    }

    async fn remove(&self, path: &str, recursive: bool) -> AppResult<String> {
        let p = self.resolve(path).await?;
        if p == "/" || self.cfg.roots.iter().any(|r| normalize(r) == p) {
            return Err(AppError::PermissionDenied("an allowed root folder itself cannot be deleted".into()));
        }
        let is_dir = {
            let g = self.with_conn(true).await?;
            let sftp = g.as_ref().unwrap().sftp.as_ref().unwrap();
            let m = sftp.symlink_metadata(p.clone()).await.map_err(|e| sftp_err(e, &p))?;
            if !m.is_dir() {
                sftp.remove_file(p.clone()).await.map_err(|e| sftp_err(e, &p))?;
                return Ok(String::new());
            }
            if !recursive {
                return Err(AppError::InvalidInput(format!("{p} is a folder; deleting a folder needs recursive=true")));
            }
            true
        };
        debug_assert!(is_dir);
        let cmd = if self.cfg.shell == Shell::PowerShell {
            format!("Remove-Item -LiteralPath '{}' -Recurse -Force", p.replace('\'', "''"))
        } else {
            format!("rm -rf -- {}", sq(&p))
        };
        let r = self.run_simple(&cmd).await?;
        if !r.succeeded() {
            return Err(AppError::Runtime(format!("delete failed: {}", r.stderr.trim())));
        }
        Ok(String::new())
    }

    async fn cleanup(&self) -> AppResult<()> {
        if let Some(c) = self.conn.lock().await.take() {
            if let Some(s) = c.sftp {
                let _ = s.close().await;
            }
            let _ = c.handle.disconnect(russh::Disconnect::ByApplication, "", "en").await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_and_within() {
        assert_eq!(normalize("/home/u/../../etc/passwd"), "/etc/passwd");
        assert_eq!(normalize("/a/./b//c/"), "/a/b/c");
        assert_eq!(normalize("/.."), "/");
        assert!(within("/home/u/p/x", "/home/u/p"));
        assert!(!within("/home/u/project2", "/home/u/project"));
        assert_eq!(sq("a'b"), r"'a'\''b'");
        assert!(format!("{:?}", SshAuth::Password { password: "hunter2".into() }).contains("***"));
    }
}

//! Execution environments (spec section 8): one trait for running commands and
//! touching files, implemented for the local PC here (`local`), and for SSH,
//! Docker and the disposable WSL sandbox in their own crates.
//!
//! An agent run is bound to one environment. Every command and file operation
//! goes through it, so permissions, audit and cancellation are uniform.

pub mod fs_tools;
pub mod git;
pub mod local;
pub mod paths;
pub mod process;
pub mod sandbox;
pub mod shfs;
pub mod web_fetch;

use std::collections::BTreeMap;
use std::time::Duration;

use app_core::AppResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Default and hard limits for captured command output.
pub const DEFAULT_MAX_OUTPUT: usize = 256 * 1024;
pub const HARD_MAX_OUTPUT: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shell {
    /// Windows PowerShell 5.1, or PowerShell 7 (pwsh) when installed.
    PowerShell,
    /// cmd.exe (only when the user picks it explicitly).
    Cmd,
    /// POSIX sh (SSH, Docker, WSL sandbox).
    Sh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSpec {
    pub command: String,
    pub shell: Shell,
    pub cwd: Option<String>,
    pub env: BTreeMap<String, String>,
    #[serde(with = "secs")]
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

mod secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

impl CommandSpec {
    pub fn new(command: impl Into<String>, shell: Shell) -> Self {
        Self { command: command.into(), shell, cwd: None, env: BTreeMap::new(), timeout: Duration::from_secs(120), max_output_bytes: DEFAULT_MAX_OUTPUT }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stream", content = "text", rename_all = "snake_case")]
pub enum ProcessEvent {
    Stdout(String),
    Stderr(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Output beyond `max_output_bytes` was dropped.
    pub truncated: bool,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration_ms: u64,
    pub started_at: String,
    pub ended_at: String,
}

impl ExecResult {
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out && !self.cancelled
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub ok: bool,
    pub detail: String,
    pub checked_at: String,
}

/// What an environment can do (shown in the UI, used to pick tools).
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub shells: Vec<Shell>,
    pub files: bool,
    /// Commands run on this PC (as opposed to a remote host or container).
    pub local: bool,
    pub os: String,
}

#[async_trait]
pub trait ExecutionEnvironment: Send + Sync {
    /// local | ssh | docker | sandbox
    fn kind(&self) -> &'static str;
    /// Shown on every command, e.g. "This PC" or "user@host:22".
    fn label(&self) -> String;
    fn capabilities(&self) -> Capabilities;
    async fn health(&self) -> Health;
    async fn execute(
        &self,
        spec: &CommandSpec,
        cancel: &CancellationToken,
        on_event: &(dyn Fn(ProcessEvent) + Send + Sync),
    ) -> AppResult<ExecResult>;
    async fn read_file(&self, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)>;
    async fn write_file(&self, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()>;
    async fn list_dir(&self, path: &str) -> AppResult<Vec<DirEntry>>;
    async fn stat(&self, path: &str) -> AppResult<Option<DirEntry>>;
    async fn rename(&self, from: &str, to: &str) -> AppResult<()>;
    /// Remove a file or (with `recursive`) a folder. The local environment
    /// moves it to the app's trash instead of deleting it.
    async fn remove(&self, path: &str, recursive: bool) -> AppResult<String>;
    /// Release resources (remote sessions, containers, sandboxes).
    async fn cleanup(&self) -> AppResult<()>;
}

pub fn now() -> String {
    app_core::now_rfc3339()
}

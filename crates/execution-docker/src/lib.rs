//! Docker execution environment (spec section 8): one long-lived container
//! per environment, commands via `docker exec`, files via shell commands in
//! the container. Defaults are locked down: no network, all capabilities
//! dropped, no-new-privileges, memory/CPU/process limits, and host folders
//! mounted only when they pass the same folder policy as the local PC.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use bollard::Docker;
use bollard::exec::StartExecResults;
use bollard::models::{ContainerCreateBody, ExecConfig, HostConfig};
use bollard::query_parameters::{CreateContainerOptionsBuilder, CreateImageOptionsBuilder, RemoveContainerOptionsBuilder};
use execution::paths::RootPolicy;
use execution::shfs::{self, ShRunner, sq};
use execution::{Capabilities, CommandSpec, DirEntry, ExecResult, ExecutionEnvironment, HARD_MAX_OUTPUT, Health, ProcessEvent, Shell, now};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub const LABEL: &str = "local-ai-studio";
const WORKSPACE: &str = "/workspace";
const PGID_MARK: &str = "__LAS_PGID=";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    /// Folder on this PC.
    pub host: String,
    /// Where it appears in the container.
    pub container: String,
    pub read_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DockerConfig {
    /// `npipe:////./pipe/docker_engine` (Docker Desktop) or `tcp://host:port`.
    pub host: String,
    pub image: String,
    /// Container name suffix: the container is `las-<name>`.
    pub name: String,
    pub network: bool,
    pub mounts: Vec<Mount>,
    pub memory_mb: u64,
    pub cpus: f64,
    pub pids: i64,
    /// Keep the container (and what is installed in it) after cleanup.
    pub keep: bool,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            host: "npipe:////./pipe/docker_engine".into(),
            image: "alpine:3.24".into(),
            name: "default".into(),
            network: false,
            mounts: vec![],
            memory_mb: 2048,
            cpus: 2.0,
            pids: 512,
            keep: false,
        }
    }
}

pub struct DockerEnv {
    pub cfg: DockerConfig,
    docker: Docker,
    binds: Vec<String>,
    container: Mutex<Option<String>>,
}

fn derr(e: bollard::errors::Error) -> AppError {
    derr_ref(&e)
}

fn derr_ref(e: &bollard::errors::Error) -> AppError {
    let s = e.to_string();
    if s.contains("No such") || s.contains("404") {
        AppError::NotFound(s)
    } else if ["os error 2", "The system cannot find the file", "Connection refused", "error trying to connect", "client error (Connect)", "connection closed before message completed", "IncompleteMessage", "connection reset", "os error 10054", "os error 10061"]
        .iter()
        .any(|m| s.contains(m))
    {
        AppError::Runtime(format!("cannot reach Docker ({s}). Is Docker Desktop running?"))
    } else {
        AppError::Runtime(format!("Docker: {s}"))
    }
}

/// Host folders to bind, checked against the folder policy. The Docker
/// socket/pipe can never be mounted (it would give full control of Docker).
pub fn check_mounts(mounts: &[Mount], policy: &RootPolicy) -> AppResult<Vec<String>> {
    let mut out = vec![];
    for m in mounts {
        let low = m.host.to_lowercase().replace('/', "\\");
        if low.contains("docker_engine") || low.contains("docker.sock") || low.starts_with("\\\\") {
            return Err(AppError::PermissionDenied(format!("{} cannot be mounted into a container", m.host)));
        }
        let host = policy.resolve(&m.host)?;
        if !host.is_dir() {
            return Err(AppError::InvalidInput(format!("{} is not a folder", host.display())));
        }
        if !m.container.starts_with('/') || m.container.contains(':') || m.container.contains("..") || m.container.trim_end_matches('/').is_empty() {
            return Err(AppError::InvalidInput(format!("container path {} must be an absolute Linux path", m.container)));
        }
        out.push(format!("{}:{}:{}", host.display(), m.container, if m.read_only { "ro" } else { "rw" }));
    }
    Ok(out)
}

pub fn connect(host: &str) -> AppResult<Docker> {
    let d = if let Some(rest) = host.strip_prefix("npipe://") {
        Docker::connect_with_named_pipe(rest, 120, bollard::API_DEFAULT_VERSION)
    } else if host.starts_with("tcp://") || host.starts_with("http://") {
        Docker::connect_with_http(host, 120, bollard::API_DEFAULT_VERSION)
    } else {
        return Err(AppError::InvalidInput("Docker host must start with npipe:// or tcp://".into()));
    };
    d.map_err(derr)
}

impl DockerEnv {
    pub fn new(cfg: DockerConfig, policy: &RootPolicy) -> AppResult<Self> {
        if cfg.name.is_empty() || !cfg.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(AppError::InvalidInput("container name must be letters, digits, - and _".into()));
        }
        let binds = check_mounts(&cfg.mounts, policy)?;
        let docker = connect(&cfg.host)?;
        Ok(Self { cfg, docker, binds, container: Mutex::new(None) })
    }

    pub fn container_name(&self) -> String {
        format!("las-{}", self.cfg.name)
    }

    /// Docker version, or why Docker cannot be reached.
    pub async fn docker_version(&self) -> AppResult<String> {
        let v = self.docker.version().await.map_err(derr)?;
        Ok(format!("Docker {} (API {})", v.version.unwrap_or_default(), v.api_version.unwrap_or_default()))
    }

    async fn ensure_image(&self) -> AppResult<()> {
        if self.docker.inspect_image(&self.cfg.image).await.is_ok() {
            return Ok(());
        }
        let (repo, tag) = match self.cfg.image.rsplit_once(':') {
            Some((r, t)) if !t.contains('/') => (r.to_string(), t.to_string()),
            _ => (self.cfg.image.clone(), "latest".into()),
        };
        let mut s = self.docker.create_image(Some(CreateImageOptionsBuilder::default().from_image(&repo).tag(&tag).build()), None, None);
        while let Some(r) = s.next().await {
            r.map_err(|e| AppError::Runtime(format!("could not pull {}: {e}", self.cfg.image)))?;
        }
        Ok(())
    }

    /// Find or create the container and make sure it is running.
    async fn ensure_container(&self) -> AppResult<String> {
        let mut g = self.container.lock().await;
        if let Some(id) = g.as_ref()
            && let Ok(info) = self.docker.inspect_container(id, None).await
            && info.state.as_ref().and_then(|s| s.running).unwrap_or(false)
        {
            return Ok(id.clone());
        }
        let name = self.container_name();
        let id = match self.docker.inspect_container(&name, None).await {
            Ok(info) => {
                let is_ours = info.config.as_ref().and_then(|c| c.labels.as_ref()).is_some_and(|l| l.contains_key(LABEL));
                if !is_ours {
                    return Err(AppError::Conflict(format!("a container named {name} exists but was not made by Local AI Studio")));
                }
                info.id.unwrap_or(name.clone())
            }
            // Only "no such container" means create one; anything else (daemon
            // down, connection refused) is reported as it is.
            Err(e) if !matches!(derr_ref(&e), AppError::NotFound(_)) => return Err(derr(e)),
            Err(_) => {
                self.ensure_image().await?;
                let host = HostConfig {
                    network_mode: Some(if self.cfg.network { "bridge".into() } else { "none".into() }),
                    cap_drop: Some(vec!["ALL".into()]),
                    // Enough for package managers and file ownership changes.
                    cap_add: Some(vec!["CHOWN".into(), "DAC_OVERRIDE".into(), "FOWNER".into(), "SETUID".into(), "SETGID".into()]),
                    security_opt: Some(vec!["no-new-privileges".into()]),
                    privileged: Some(false),
                    memory: Some((self.cfg.memory_mb as i64) * 1024 * 1024),
                    nano_cpus: Some((self.cfg.cpus * 1e9) as i64),
                    pids_limit: Some(self.cfg.pids),
                    binds: if self.binds.is_empty() { None } else { Some(self.binds.clone()) },
                    ..Default::default()
                };
                let body = ContainerCreateBody {
                    image: Some(self.cfg.image.clone()),
                    cmd: Some(vec!["sh".into(), "-c".into(), "trap 'exit 0' TERM INT; while :; do sleep 3600 & wait $!; done".into()]),
                    working_dir: Some(WORKSPACE.into()),
                    labels: Some(HashMap::from([(LABEL.to_string(), "1".to_string())])),
                    network_disabled: Some(!self.cfg.network),
                    host_config: Some(host),
                    ..Default::default()
                };
                self.docker.create_container(Some(CreateContainerOptionsBuilder::default().name(&name).build()), body).await.map_err(derr)?.id
            }
        };
        self.docker.start_container(&id, None).await.or_else(|e| if e.to_string().contains("304") { Ok(()) } else { Err(e) }).map_err(derr)?;
        *g = Some(id.clone());
        Ok(id)
    }

    /// Run `sh -c script` in the container. The script's process group is
    /// reported on stderr first so a timeout can kill everything it started.
    async fn exec_sh(
        &self,
        script: &str,
        stdin: Option<Vec<u8>>,
        timeout: Duration,
        max_output: usize,
        cancel: &CancellationToken,
        on_event: &(dyn Fn(ProcessEvent) + Send + Sync),
    ) -> AppResult<ExecResult> {
        let id = self.ensure_container().await?;
        let inner = format!("echo {PGID_MARK}$$ >&2\n{script}");
        // runc starts each exec in its own session, so $$ is the process group.
        let wrapper = format!("exec sh -c {}", sq(&inner));
        let exec = self
            .docker
            .create_exec(
                &id,
                ExecConfig {
                    attach_stdin: Some(stdin.is_some()),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    cmd: Some(vec!["sh".into(), "-c".into(), wrapper]),
                    working_dir: Some("/".into()),
                    ..Default::default()
                },
            )
            .await
            .map_err(derr)?
            .id;
        let started_at = now();
        let t0 = Instant::now();
        let StartExecResults::Attached { mut output, mut input } = self.docker.start_exec(&exec, None).await.map_err(derr)? else {
            return Err(AppError::Runtime("docker exec did not attach".into()));
        };
        if let Some(data) = stdin {
            tokio::spawn(async move {
                let _ = input.write_all(&data).await;
                let _ = input.shutdown().await;
            });
        }
        let cap = max_output.clamp(1024, HARD_MAX_OUTPUT);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let (mut err_line, mut out_line) = (Vec::<u8>::new(), Vec::<u8>::new());
        let mut truncated = false;
        let mut pgid: Option<i64> = None;
        let (mut timed_out, mut cancelled) = (false, false);
        let deadline = tokio::time::sleep(timeout);
        tokio::pin!(deadline);
        let emit = |is_err: bool, line: &mut Vec<u8>, kept: &mut Vec<u8>, truncated: &mut bool, pgid: &mut Option<i64>| {
            let text = String::from_utf8_lossy(line).into_owned();
            line.clear();
            if is_err && pgid.is_none() && text.starts_with(PGID_MARK) {
                *pgid = text.trim()[PGID_MARK.len()..].parse().ok();
                return;
            }
            for &b in text.as_bytes() {
                if kept.len() < cap {
                    kept.push(b);
                } else {
                    *truncated = true;
                }
            }
            on_event(if is_err { ProcessEvent::Stderr(text) } else { ProcessEvent::Stdout(text) });
        };
        loop {
            tokio::select! {
                m = output.next() => match m {
                    Some(Ok(bollard::container::LogOutput::StdOut { message })) => {
                        for &b in message.iter() { out_line.push(b); if b == b'\n' || out_line.len() >= 4096 { emit(false, &mut out_line, &mut out, &mut truncated, &mut pgid); } }
                    }
                    Some(Ok(bollard::container::LogOutput::StdErr { message })) => {
                        for &b in message.iter() { err_line.push(b); if b == b'\n' || err_line.len() >= 4096 { emit(true, &mut err_line, &mut err, &mut truncated, &mut pgid); } }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(derr(e)),
                    None => break,
                },
                _ = &mut deadline => { timed_out = true; break; }
                _ = cancel.cancelled() => { cancelled = true; break; }
            }
        }
        if !out_line.is_empty() {
            emit(false, &mut out_line, &mut out, &mut truncated, &mut pgid);
        }
        if !err_line.is_empty() {
            emit(true, &mut err_line, &mut err, &mut truncated, &mut pgid);
        }
        drop(output);
        let mut exit_code = None;
        if (timed_out || cancelled)
            && let Some(g) = pgid
        {
            let kill = self
                .docker
                .create_exec(&id, ExecConfig { cmd: Some(vec!["sh".into(), "-c".into(), format!("kill -KILL -- -{g} 2>/dev/null; true")]), attach_stdout: Some(true), ..Default::default() })
                .await
                .map_err(derr)?
                .id;
            if let Ok(StartExecResults::Attached { mut output, .. }) = self.docker.start_exec(&kill, None).await {
                while output.next().await.is_some() {}
            }
        } else {
            // The stream ended: the exit code is ready (poll briefly).
            for _ in 0..50 {
                let info = self.docker.inspect_exec(&exec).await.map_err(derr)?;
                if info.running != Some(true) {
                    exit_code = info.exit_code.map(|c| c as i32);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            // A command that "failed" because its container stopped under it
            // (docker kill/stop, out of memory, daemon restart) is reported as such.
            if exit_code != Some(0) {
                let running = self.docker.inspect_container(&id, None).await.ok().and_then(|c| c.state).and_then(|st| st.running).unwrap_or(false);
                if !running {
                    *self.container.lock().await = None;
                    let so_far = String::from_utf8_lossy(&out).trim().chars().take(500).collect::<String>();
                    return Err(AppError::Runtime(format!(
                        "the container {} stopped while the command was running{}; the next command starts it again",
                        self.container_name(),
                        if so_far.is_empty() { String::new() } else { format!(" (output so far: {so_far})") }
                    )));
                }
            }
        }
        Ok(ExecResult {
            exit_code,
            stdout: String::from_utf8_lossy(&out).into_owned(),
            stderr: String::from_utf8_lossy(&err).into_owned(),
            truncated,
            timed_out,
            cancelled,
            duration_ms: t0.elapsed().as_millis() as u64,
            started_at,
            ended_at: now(),
        })
    }

    /// Remove the container (and everything installed in it).
    pub async fn remove_container(&self) -> AppResult<()> {
        let name = self.container_name();
        match self.docker.inspect_container(&name, None).await {
            Ok(info) => {
                if !info.config.as_ref().and_then(|c| c.labels.as_ref()).is_some_and(|l| l.contains_key(LABEL)) {
                    return Err(AppError::PermissionDenied(format!("refusing to remove {name}: not made by Local AI Studio")));
                }
                self.docker.remove_container(&name, Some(RemoveContainerOptionsBuilder::default().force(true).build())).await.map_err(derr)?;
            }
            Err(_) => {}
        }
        *self.container.lock().await = None;
        Ok(())
    }
}

#[async_trait]
impl ShRunner for DockerEnv {
    async fn sh(&self, script: &str, stdin: Option<Vec<u8>>) -> AppResult<ExecResult> {
        self.exec_sh(script, stdin, Duration::from_secs(120), HARD_MAX_OUTPUT, &CancellationToken::new(), &|_| {}).await
    }
    fn workspace(&self) -> &str {
        WORKSPACE
    }
    fn protected(&self) -> Vec<String> {
        let mut v = vec![WORKSPACE.to_string()];
        v.extend(self.cfg.mounts.iter().map(|m| m.container.trim_end_matches('/').to_string()));
        v
    }
}

#[async_trait]
impl ExecutionEnvironment for DockerEnv {
    fn kind(&self) -> &'static str {
        "docker"
    }

    fn label(&self) -> String {
        format!("Docker {} ({}, network {})", self.container_name(), self.cfg.image, if self.cfg.network { "on" } else { "off" })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { shells: vec![Shell::Sh], files: true, local: false, os: "linux (container)".into() }
    }

    async fn health(&self) -> Health {
        let v = match self.docker_version().await {
            Ok(v) => v,
            Err(e) => return Health { ok: false, detail: e.to_string(), checked_at: now() },
        };
        // Was the container we used removed outside the app?
        let had = self.container.lock().await.is_some();
        let gone = had && self.docker.inspect_container(&self.container_name(), None).await.is_err();
        let note = if gone { "the container was removed outside the app, so a new one was created (anything installed in it is gone); " } else { "" };
        if gone {
            *self.container.lock().await = None;
        }
        match self.sh("uname -sr", None).await {
            Ok(r) if r.succeeded() => Health { ok: true, detail: format!("{v}; {note}container {}: {}", self.container_name(), r.stdout.trim()), checked_at: now() },
            Ok(r) => Health { ok: false, detail: format!("{v}; {}", r.stderr.trim()), checked_at: now() },
            Err(e) => Health { ok: false, detail: format!("{v}; {e}"), checked_at: now() },
        }
    }

    async fn execute(&self, spec: &CommandSpec, cancel: &CancellationToken, on_event: &(dyn Fn(ProcessEvent) + Send + Sync)) -> AppResult<ExecResult> {
        if spec.shell != Shell::Sh {
            return Err(AppError::InvalidInput("containers run sh commands, not PowerShell or cmd".into()));
        }
        let script = shfs::command_script(spec, WORKSPACE)?;
        self.exec_sh(&script, None, spec.timeout, spec.max_output_bytes, cancel, on_event).await
    }

    async fn read_file(&self, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)> {
        shfs::read_file(self, path, max_bytes).await
    }

    async fn write_file(&self, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()> {
        shfs::write_file(self, path, bytes, create_dirs).await
    }

    async fn list_dir(&self, path: &str) -> AppResult<Vec<DirEntry>> {
        shfs::list_dir(self, path).await
    }

    async fn stat(&self, path: &str) -> AppResult<Option<DirEntry>> {
        shfs::stat(self, path).await
    }

    async fn rename(&self, from: &str, to: &str) -> AppResult<()> {
        shfs::rename(self, from, to).await
    }

    async fn remove(&self, path: &str, recursive: bool) -> AppResult<String> {
        shfs::remove(self, path, recursive).await
    }

    async fn cleanup(&self) -> AppResult<()> {
        if self.cfg.keep { Ok(()) } else { self.remove_container().await }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mounts_are_policy_checked() {
        let d = tempfile::tempdir().unwrap();
        let ok = d.path().join("proj");
        std::fs::create_dir_all(&ok).unwrap();
        let p = RootPolicy::new(vec![ok.clone()], vec![]);
        let m = |h: &str, c: &str| Mount { host: h.into(), container: c.into(), read_only: false };
        assert_eq!(check_mounts(&[m(&ok.to_string_lossy(), "/workspace/proj")], &p).unwrap().len(), 1);
        assert!(check_mounts(&[m(&d.path().to_string_lossy(), "/x")], &p).is_err(), "outside roots");
        assert!(check_mounts(&[m(r"\\.\pipe\docker_engine", "/var/run/docker.sock")], &p).is_err(), "docker pipe");
        assert!(check_mounts(&[m(r"C:\Windows", "/win")], &p).is_err(), "system folder");
        assert!(check_mounts(&[m(&ok.to_string_lossy(), "relative")], &p).is_err());
        assert!(check_mounts(&[m(&ok.to_string_lossy(), "/")], &p).is_err());
        assert!(connect("ftp://x").is_err());
    }
}

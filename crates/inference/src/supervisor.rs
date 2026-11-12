//! Supervises the single active llama-server process (V1: one generation model).
//!
//! Load: pick a free loopback port, start llama-server inside a kill-on-close
//! Job Object, stream its output into a bounded log ring, and wait for
//! `/health` to turn OK. A watcher task notices if the process dies on its own
//! and reports a crash with a diagnosis taken from the last log lines.
//! Unload: terminate the job (process tree) and wait for exit.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use regex::Regex;
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::client::{Health, LlamaClient, ServerProps};
use crate::config::LoadConfig;
use crate::runtime::InstalledRuntime;

const LOG_CAPACITY: usize = 4000;
const LOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);

static OFFLOAD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"offloaded (\d+)/(\d+) layers to GPU").unwrap());
/// e.g. "CUDA0 model buffer size =  1208.11 MiB", "CUDA0 KV buffer size = 112.00 MiB"
static BUFFER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<dev>[A-Za-z_]+[0-9]*(?:_Mapped|_Host)?)\s+(?P<kind>model|KV|compute|RS)\s+buffer size\s*=\s*(?P<mib>[\d.]+) MiB").unwrap()
});

/// e.g. "- CUDA0   : NVIDIA GeForce RTX 5060 (8123 MiB, 7031 MiB free)"
static DEVICE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-\s+(?P<dev>(?:CUDA|Vulkan|ROCm|SYCL)\d+)\s*:\s*(?P<name>.+?)\s*\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)").unwrap()
});

/// A GPU device as llama.cpp saw it at startup.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DeviceReport {
    pub device: String,
    pub name: String,
    pub total_mib: f64,
    pub free_mib: f64,
}

/// A buffer the backend reported allocating (measured, not estimated).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BufferReport {
    pub device: String,
    pub kind: String,
    pub mib: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    pub state: EngineState,
    pub model_id: Option<String>,
    pub instance_id: Option<String>,
    pub runtime_id: Option<String>,
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub config: Option<LoadConfig>,
    pub args: Vec<String>,
    /// Settings the runtime could not apply (shown to the user).
    pub unsupported_settings: Vec<String>,
    pub props: Option<ServerProps>,
    /// "offloaded N/M layers to GPU", parsed from the backend log.
    pub gpu_layers_offloaded: Option<(u32, u32)>,
    /// Buffer sizes from the backend log, per device.
    pub buffers: Vec<BufferReport>,
    /// GPU devices with their free memory at backend start.
    pub devices: Vec<DeviceReport>,
    /// Problems detected after load (e.g. VRAM spilled into shared memory).
    pub warnings: Vec<String>,
    pub load_seconds: Option<f64>,
    pub error: Option<String>,
    /// Actionable guidance for the last error.
    pub error_hint: Option<String>,
}

impl EngineStatus {
    fn unloaded() -> Self {
        Self {
            state: EngineState::Unloaded,
            model_id: None,
            instance_id: None,
            runtime_id: None,
            pid: None,
            port: None,
            config: None,
            args: vec![],
            unsupported_settings: vec![],
            props: None,
            gpu_layers_offloaded: None,
            buffers: vec![],
            devices: vec![],
            warnings: vec![],
            load_seconds: None,
            error: None,
            error_hint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineEvent {
    Status { status: Box<EngineStatus> },
    Log { line: String, stream: &'static str },
    /// A backend instance was stopped on purpose (unload, replaced by another
    /// load, cancelled load, app exit). Crashes are reported via Status=Failed.
    InstanceEnded { instance_id: String },
}

/// What to load. The caller resolves ids to paths before calling the engine.
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub model_id: String,
    pub instance_id: String,
    pub model_path: PathBuf,
    pub projector_path: Option<PathBuf>,
    pub runtime: InstalledRuntime,
    pub config: LoadConfig,
    /// Name reported as the model id by the backend's OpenAI API.
    pub alias: Option<String>,
}

struct Running {
    #[cfg(windows)]
    job: crate::job::JobObject,
    pid: u32,
    /// Set before an intentional stop so the watcher doesn't report a crash.
    stopping: Arc<std::sync::atomic::AtomicBool>,
    exited: CancellationToken,
}

struct Inner {
    status: EngineStatus,
    running: Option<Running>,
    load_cancel: Option<CancellationToken>,
    /// The first reply's speed has been checked for a VRAM spill (per load).
    speed_checked: bool,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<Inner>>,
    logs: Arc<Mutex<VecDeque<String>>>,
    events: broadcast::Sender<EngineEvent>,
    /// Serializes load/unload so two loads can never race.
    op_lock: Arc<tokio::sync::Mutex<()>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(1024);
        Self {
            inner: Arc::new(Mutex::new(Inner { status: EngineStatus::unloaded(), running: None, load_cancel: None, speed_checked: false })),
            logs: Arc::new(Mutex::new(VecDeque::with_capacity(LOG_CAPACITY))),
            events,
            op_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.events.subscribe()
    }

    pub fn status(&self) -> EngineStatus {
        self.inner.lock().unwrap().status.clone()
    }

    /// Client for the loaded backend, if one is loaded.
    pub fn client(&self) -> Option<LlamaClient> {
        let g = self.inner.lock().unwrap();
        (g.status.state == EngineState::Loaded).then(|| g.status.port.map(LlamaClient::new)).flatten()
    }

    pub fn backend_pid(&self) -> Option<u32> {
        self.inner.lock().unwrap().running.as_ref().map(|r| r.pid)
    }

    pub fn logs(&self, last: usize) -> Vec<String> {
        let l = self.logs.lock().unwrap();
        l.iter().skip(l.len().saturating_sub(last)).cloned().collect()
    }

    fn set_status(&self, f: impl FnOnce(&mut EngineStatus)) {
        let snapshot = {
            let mut g = self.inner.lock().unwrap();
            f(&mut g.status);
            g.status.clone()
        };
        let _ = self.events.send(EngineEvent::Status { status: Box::new(snapshot) });
    }

    fn push_log(logs: &Mutex<VecDeque<String>>, events: &broadcast::Sender<EngineEvent>, line: String, stream: &'static str) {
        {
            let mut l = logs.lock().unwrap();
            if l.len() >= LOG_CAPACITY {
                l.pop_front();
            }
            l.push_back(line.clone());
        }
        let _ = events.send(EngineEvent::Log { line, stream });
    }

    /// Cancel a load that is in progress (the process is killed).
    pub fn cancel_load(&self) {
        if let Some(c) = self.inner.lock().unwrap().load_cancel.clone() {
            c.cancel();
        }
    }

    pub async fn load(&self, req: LoadRequest) -> AppResult<EngineStatus> {
        req.config.validate().map_err(AppError::InvalidInput)?;
        if !req.model_path.is_file() {
            return Err(AppError::NotFound(format!("model file not found: {}", req.model_path.display())));
        }
        let _op = self.op_lock.lock().await;
        self.stop_inner().await;
        self.logs.lock().unwrap().clear();

        let port = free_port()?;
        let built = req.config.to_args_named(
            &req.model_path,
            req.projector_path.as_deref(),
            port,
            &req.runtime.capabilities,
            req.alias.as_deref(),
        );
        let cancel = CancellationToken::new();
        {
            let mut g = self.inner.lock().unwrap();
            g.load_cancel = Some(cancel.clone());
        }
        self.set_status(|s| {
            *s = EngineStatus::unloaded();
            s.state = EngineState::Loading;
            s.model_id = Some(req.model_id.clone());
            s.instance_id = Some(req.instance_id.clone());
            s.runtime_id = Some(req.runtime.id.clone());
            s.port = Some(port);
            s.config = Some(req.config.clone());
            s.args = built.args.clone();
            s.unsupported_settings = built.unsupported.clone();
        });
        tracing::info!(runtime = %req.runtime.id, args = ?built.args, "starting llama-server");

        let mut cmd = tokio::process::Command::new(&req.runtime.server_path);
        cmd.args(&built.args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(dir) = req.runtime.server_path.parent() {
            cmd.current_dir(dir);
        }
        #[cfg(windows)]
        cmd.creation_flags(crate::job::CREATE_NO_WINDOW);
        let started = Instant::now();
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("could not start {}: {e}", req.runtime.server_path.display());
                self.fail(&msg, Some("Reinstall the runtime from the Runtimes page.".into()));
                return Err(AppError::Runtime(msg));
            }
        };
        let pid = child.id().unwrap_or(0);
        #[cfg(windows)]
        let job = match crate::job::contain(&child) {
            Ok(j) => j,
            Err(e) => {
                let _ = child.start_kill();
                let msg = format!("could not contain backend process in a job object: {e}");
                self.fail(&msg, None);
                return Err(AppError::Runtime(msg));
            }
        };

        let offload: Arc<Mutex<LogFacts>> = Arc::new(Mutex::new(LogFacts::default()));
        if let Some(out) = child.stdout.take() {
            self.spawn_reader(out, "stdout", offload.clone());
        }
        if let Some(err) = child.stderr.take() {
            self.spawn_reader(err, "stderr", offload.clone());
        }

        let stopping = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let exited = CancellationToken::new();
        {
            let mut g = self.inner.lock().unwrap();
            g.running = Some(Running {
                #[cfg(windows)]
                job,
                pid,
                stopping: stopping.clone(),
                exited: exited.clone(),
            });
        }
        self.set_status(|s| s.pid = Some(pid));

        // Watcher: reports unexpected exits (crash, OOM) at any time.
        {
            let me = self.clone();
            let stopping = stopping.clone();
            let exited = exited.clone();
            tokio::spawn(async move {
                let code = child.wait().await.ok().and_then(|s| s.code());
                exited.cancel();
                if !stopping.load(std::sync::atomic::Ordering::SeqCst) {
                    // Give the log readers a moment to drain the final lines.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    let tail = me.logs(80).join("\n");
                    let (msg, hint) = diagnose(&tail, code);
                    tracing::warn!(?code, "llama-server exited unexpectedly: {msg}");
                    // Only clean up if this is still the current backend: a new
                    // load may already have started (dropping its Running would
                    // close its Job Object and kill the new process).
                    let current = {
                        let mut g = me.inner.lock().unwrap();
                        if g.running.as_ref().map(|r| r.pid) == Some(pid) {
                            g.running = None;
                            true
                        } else {
                            false
                        }
                    };
                    if current {
                        me.fail(&msg, hint);
                    }
                }
            });
        }

        // Wait for readiness.
        let client = LlamaClient::new(port);
        let deadline = Instant::now() + LOAD_TIMEOUT;
        loop {
            if cancel.is_cancelled() {
                drop(_op);
                self.unload().await;
                return Err(AppError::Cancelled);
            }
            if exited.is_cancelled() {
                // The watcher writes the diagnosis shortly after exit; wait for it.
                for _ in 0..20 {
                    if self.status().state == EngineState::Failed {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                let st = self.status();
                return Err(classify(st.error.unwrap_or_else(|| "backend exited during load".into())));
            }
            if Instant::now() > deadline {
                let msg = "model load timed out after 15 minutes".to_string();
                self.stop_inner().await;
                self.fail(&msg, None);
                return Err(AppError::Runtime(msg));
            }
            if client.health().await == Health::Ok {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(250)) => {}
                _ = exited.cancelled() => {}
                _ = cancel.cancelled() => {}
            }
        }
        let props = client.props().await.ok();
        let secs = started.elapsed().as_secs_f64();
        let facts = offload.lock().unwrap().clone();
        {
            let mut g = self.inner.lock().unwrap();
            g.load_cancel = None;
            g.speed_checked = false;
        }
        let fit = fits_layers(&req.config, &req.runtime);
        if let Some((n, m)) = facts.offloaded {
            tracing::info!("llama.cpp put {n} of {m} layers on the GPU{}", if fit { " (--fit)" } else { "" });
        }
        self.set_status(|s| {
            s.state = EngineState::Loaded;
            s.props = props;
            s.load_seconds = Some(secs);
            s.gpu_layers_offloaded = facts.offloaded;
            s.warnings = facts.spill_warnings(fit.then_some(req.config.fit_target_mib.unwrap_or(crate::config::DEFAULT_FIT_TARGET_MIB)));
            s.buffers = facts.buffers;
            s.devices = facts.devices;
        });
        Ok(self.status())
    }

    fn spawn_reader<R: AsyncRead + Unpin + Send + 'static>(
        &self,
        r: R,
        stream: &'static str,
        offload: Arc<Mutex<LogFacts>>,
    ) {
        let logs = self.logs.clone();
        let events = self.events.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                offload.lock().unwrap().observe(&line);
                // Cap individual line length so a runaway line can't bloat memory.
                let line = if line.len() > 4000 { format!("{}...", &line[..line.floor_char_boundary(4000)]) } else { line };
                Self::push_log(&logs, &events, line, stream);
            }
        });
    }

    fn fail(&self, msg: &str, hint: Option<String>) {
        let msg = msg.to_string();
        self.set_status(|s| {
            s.state = EngineState::Failed;
            s.error = Some(msg);
            s.error_hint = hint;
            s.pid = None;
        });
    }

    /// Kill the backend process tree and wait for it to exit.
    async fn stop_inner(&self) {
        let (running, instance_id) = {
            let mut g = self.inner.lock().unwrap();
            (g.running.take(), g.status.instance_id.clone())
        };
        let Some(r) = running else { return };
        r.stopping.store(true, std::sync::atomic::Ordering::SeqCst);
        #[cfg(windows)]
        if let Err(e) = r.job.terminate() {
            tracing::warn!("TerminateJobObject failed: {e}");
        }
        if tokio::time::timeout(Duration::from_secs(10), r.exited.cancelled()).await.is_err() {
            tracing::error!(pid = r.pid, "backend did not exit within 10s of termination");
        }
        // Dropping the job closes the last handle (kill-on-close as a backstop).
        drop(r);
        if let Some(instance_id) = instance_id {
            let _ = self.events.send(EngineEvent::InstanceEnded { instance_id });
        }
    }

    pub async fn unload(&self) {
        let _op = self.op_lock.lock().await;
        let had = self.inner.lock().unwrap().running.is_some();
        if had {
            self.set_status(|s| s.state = EngineState::Unloading);
        }
        self.stop_inner().await;
        let mut keep_error = None;
        {
            let g = self.inner.lock().unwrap();
            if g.status.state == EngineState::Failed && !had {
                keep_error = g.status.error.clone();
            }
        }
        self.set_status(|s| {
            *s = EngineStatus::unloaded();
            s.error = keep_error;
        });
    }
}

/// Facts parsed from backend log lines while loading.
#[derive(Debug, Clone, Default)]
struct LogFacts {
    offloaded: Option<(u32, u32)>,
    buffers: Vec<BufferReport>,
    devices: Vec<DeviceReport>,
}

impl LogFacts {
    fn observe(&mut self, line: &str) {
        if let Some(c) = OFFLOAD_RE.captures(line) {
            self.offloaded = Some((c[1].parse().unwrap_or(0), c[2].parse().unwrap_or(0)));
        }
        if let Some(c) = DEVICE_RE.captures(line) {
            let d = DeviceReport {
                device: c["dev"].to_string(),
                name: c["name"].trim().to_string(),
                total_mib: c["total"].parse().unwrap_or(0.0),
                free_mib: c["free"].parse().unwrap_or(0.0),
            };
            if !self.devices.iter().any(|x| x.device == d.device) {
                self.devices.push(d);
            }
        }
        if let Some(c) = BUFFER_RE.captures(line) {
            let r = BufferReport { device: c["dev"].to_string(), kind: c["kind"].to_string(), mib: c["mib"].parse().unwrap_or(0.0) };
            if !self.buffers.contains(&r) {
                self.buffers.push(r);
            }
        }
    }
}

impl LogFacts {
    /// On Windows the NVIDIA driver can silently place GPU allocations in shared
    /// system memory when VRAM runs out (sysmem fallback). Loading "succeeds"
    /// but generation becomes very slow. Detect it by comparing what was
    /// allocated on each device with the free memory it had at startup.
    /// `fit_margin_mib`: llama.cpp fitted the layers itself (`--fit`) with this
    /// margin, so the advice is a larger margin rather than fewer layers.
    fn spill_warnings(&self, fit_margin_mib: Option<u32>) -> Vec<String> {
        let mut out = Vec::new();
        for d in &self.devices {
            let used: f64 = self.buffers.iter().filter(|b| b.device == d.device).map(|b| b.mib).sum();
            if d.free_mib > 0.0 && used > d.free_mib + 64.0 {
                let advice = match fit_margin_mib {
                    Some(m) => format!("Raise the GPU memory margin for fitting (Advanced settings, now {m} MiB) and reload."),
                    None => "Offload fewer layers, lower the context, or use a smaller quantization.".to_string(),
                };
                out.push(format!(
                    "{} allocated {:.1} GB but had only {:.1} GB free, so about {:.1} GB is in shared system memory. Generation will be much slower. {advice}",
                    d.device,
                    used / 1024.0,
                    d.free_mib / 1024.0,
                    (used - d.free_mib) / 1024.0
                ));
            }
        }
        out
    }
}

/// llama.cpp picks the GPU layers itself for this load (`--fit` on, no
/// explicit `-ngl`), as Share with CPU does.
fn fits_layers(cfg: &LoadConfig, rt: &InstalledRuntime) -> bool {
    cfg.fit && cfg.gpu_layers == crate::config::GpuLayers::Auto && rt.capabilities.has("--fit")
}

/// Prompt processing below this multiple of the generation speed, with part
/// of the model on the GPU, points to a VRAM spill. Measured on an RTX 5060:
/// normal splits run prompts 60-75x faster than generation (Gemma-4-E4B
/// 1900 vs 25.6 t/s, Qwen3.8-27B 401 vs 5.6 t/s); the spill at ~7.7 GB used
/// collapsed prompts to 51 t/s at ~4 t/s generation (about 13x).
const SPILL_PROMPT_TO_GEN_RATIO: f64 = 20.0;
/// Too-short prompts and replies give noisy speeds; skip them.
const SPILL_CHECK_MIN_PROMPT_TOKENS: u64 = 512;
const SPILL_CHECK_MIN_GEN_TOKENS: u64 = 16;

/// Does a reply's speed suggest the GPU spilled into shared system memory?
/// `prompt_tokens` are the ones actually processed (not served from cache).
pub fn prompt_speed_suggests_spill(prompt_tokens: u64, prompt_tps: f64, gen_tokens: u64, gen_tps: f64) -> bool {
    prompt_tokens >= SPILL_CHECK_MIN_PROMPT_TOKENS
        && gen_tokens >= SPILL_CHECK_MIN_GEN_TOKENS
        && prompt_tps > 0.0
        && gen_tps > 0.0
        && prompt_tps < gen_tps * SPILL_PROMPT_TO_GEN_RATIO
}

impl Engine {
    /// Look at the first reply long enough to measure after a load: a prompt
    /// speed that collapsed next to the generation speed usually means VRAM
    /// spilled into shared system memory. Adds a warning to the status (no
    /// reload). Later replies are not checked again until the next load.
    pub fn note_reply_speed(&self, stats: &crate::chat::ChatStats) {
        let processed = stats.prompt_tokens.unwrap_or(0).saturating_sub(stats.cached_tokens.unwrap_or(0));
        let (Some(p), Some(g), Some(n)) = (stats.prompt_tokens_per_sec, stats.tokens_per_sec, stats.completion_tokens) else { return };
        if processed < SPILL_CHECK_MIN_PROMPT_TOKENS || n < SPILL_CHECK_MIN_GEN_TOKENS {
            return;
        }
        let warning = {
            let mut g_ = self.inner.lock().unwrap();
            let st = &g_.status;
            let on_gpu = st.gpu_layers_offloaded.is_some_and(|(n, _)| n > 0);
            if g_.speed_checked || st.state != EngineState::Loaded || !on_gpu {
                return;
            }
            g_.speed_checked = true;
            if !prompt_speed_suggests_spill(processed, p, n, g) {
                return;
            }
            let fit = g_.status.config.as_ref().is_some_and(|c| c.fit && c.gpu_layers == crate::config::GpuLayers::Auto && c.fit_target_mib.is_some());
            let advice = if fit {
                "Raise the GPU memory margin for fitting (Advanced settings) and reload."
            } else {
                "Offload fewer layers (or turn on Share with CPU) and reload."
            };
            format!(
                "The first reply processed its prompt at only {p:.0} tokens/s next to {g:.1} tokens/s generation. A prompt speed this low usually means GPU memory spilled into shared system memory. {advice}"
            )
        };
        tracing::warn!("{warning}");
        self.set_status(|s| {
            if !s.warnings.contains(&warning) {
                s.warnings.push(warning);
            }
        });
    }
}

fn classify(msg: String) -> AppError {
    if msg.to_lowercase().contains("out of memory") {
        AppError::OutOfMemory(msg)
    } else {
        AppError::Runtime(msg)
    }
}

/// Turn the last backend log lines into a readable error and a hint.
pub fn diagnose(tail: &str, code: Option<i32>) -> (String, Option<String>) {
    let t = tail.to_lowercase();
    let oom = ["out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate", "cuda error: out of memory"];
    if oom.iter().any(|p| t.contains(p)) {
        return (
            "the backend ran out of memory while loading the model".into(),
            Some("Lower the context length, offload fewer GPU layers, use a quantized KV cache (q8_0), or pick a smaller quantization.".into()),
        );
    }
    if t.contains("unknown model architecture") || t.contains("unknown architecture") {
        return (
            "this llama.cpp build does not support the model's architecture".into(),
            Some("Install a newer runtime from the Runtimes page.".into()),
        );
    }
    if t.contains("invalid magic") || t.contains("failed to read magic") || t.contains("not a gguf") || t.contains("gguf_init_from_file") {
        return ("the model file is not a valid GGUF file or is corrupt".into(), Some("Re-download or re-import the file.".into()));
    }
    if t.contains("couldn't bind") || t.contains("address already in use") || t.contains("failed to bind") {
        return ("the backend could not bind its local port".into(), Some("Try loading again; a new port is chosen each time.".into()));
    }
    if t.contains("failed to load model") || t.contains("error loading model") {
        let line = tail
            .lines()
            .rev()
            .find(|l| l.to_lowercase().contains("error") || l.to_lowercase().contains("failed"))
            .unwrap_or("");
        return (format!("the backend failed to load the model: {}", line.trim()), None);
    }
    if t.contains("cuda driver version is insufficient") || t.contains("no cuda-capable device") {
        return (
            "the CUDA runtime cannot use this GPU".into(),
            Some("Update the NVIDIA driver, or use the CPU or Vulkan runtime.".into()),
        );
    }
    let last = tail.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
    (
        format!(
            "the backend exited unexpectedly (exit code {}){}",
            code.map(|c| c.to_string()).unwrap_or_else(|| "unknown".into()),
            if last.is_empty() { String::new() } else { format!(": {last}") }
        ),
        Some("See the backend log for details.".into()),
    )
}

fn free_port() -> AppResult<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_offload_and_buffer_lines_from_real_log() {
        let mut f = LogFacts::default();
        for l in [
            "0.00.582.706 I load_tensors: offloaded 29/29 layers to GPU",
            "0.00.582.711 I load_tensors:   CPU_Mapped model buffer size =   182.57 MiB",
            "0.00.582.712 I load_tensors:        CUDA0 model buffer size =  1208.11 MiB",
            "llama_kv_cache:      CUDA0 KV buffer size =   112.00 MiB",
        ] {
            f.observe(l);
        }
        assert_eq!(f.offloaded, Some((29, 29)));
        assert_eq!(f.buffers.len(), 3);
        assert_eq!(f.buffers[1], BufferReport { device: "CUDA0".into(), kind: "model".into(), mib: 1208.11 });
        assert_eq!(f.buffers[2].kind, "KV");
    }

    #[test]
    fn detects_vram_spill_into_shared_memory() {
        let mut f = LogFacts::default();
        for l in [
            "0.00.202.140 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 5060 (8123 MiB, 7031 MiB free)",
            "load_tensors:        CUDA0 model buffer size = 15088.32 MiB",
            "llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB",
        ] {
            f.observe(l);
        }
        assert_eq!(f.devices[0].free_mib, 7031.0);
        let w = f.spill_warnings(None);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("shared system memory") && w[0].contains("Offload fewer layers"), "{}", w[0]);
        // When llama.cpp fitted the layers, the advice is a larger margin.
        let w = f.spill_warnings(Some(1024));
        assert!(w[0].contains("margin") && w[0].contains("1024 MiB") && !w[0].contains("lower the context"), "{}", w[0]);
        // A model that fits produces no warning.
        let mut ok = LogFacts::default();
        ok.observe("- CUDA0   : NVIDIA GeForce RTX 5060 (8123 MiB, 7031 MiB free)");
        ok.observe("CUDA0 model buffer size =  1208.11 MiB");
        assert!(ok.spill_warnings(None).is_empty());
    }

    #[test]
    fn prompt_speed_spill_heuristic_matches_the_measured_runs() {
        // Normal splits on the RTX 5060: no warning.
        assert!(!prompt_speed_suggests_spill(2000, 1900.0, 200, 25.6)); // Gemma-4-E4B, 12 layers
        assert!(!prompt_speed_suggests_spill(2000, 401.0, 200, 5.62)); // Qwen3.8-27B, 16 layers
        assert!(!prompt_speed_suggests_spill(2000, 6240.0, 200, 94.1)); // Gemma all on GPU
        // The -ngl 28 spill: prompts 51 t/s at ~4 t/s generation.
        assert!(prompt_speed_suggests_spill(2000, 51.0, 200, 3.9));
        // Too little to measure: never.
        assert!(!prompt_speed_suggests_spill(100, 51.0, 200, 3.9));
        assert!(!prompt_speed_suggests_spill(2000, 51.0, 4, 3.9));
        assert!(!prompt_speed_suggests_spill(2000, 0.0, 200, 0.0));
    }

    #[test]
    fn reply_speed_check_warns_once_per_load_and_only_with_gpu_layers() {
        let e = Engine::new();
        let slow = crate::chat::ChatStats {
            prompt_tokens: Some(3000),
            cached_tokens: Some(0),
            completion_tokens: Some(100),
            prompt_tokens_per_sec: Some(51.0),
            tokens_per_sec: Some(3.9),
            ..Default::default()
        };
        // Not loaded: nothing.
        e.note_reply_speed(&slow);
        assert!(e.status().warnings.is_empty());
        e.set_status(|s| {
            s.state = EngineState::Loaded;
            s.gpu_layers_offloaded = Some((28, 66));
            s.config = Some(LoadConfig { fit_target_mib: Some(1024), ..Default::default() });
        });
        // A cached prompt is too short to judge, and does not use up the check.
        e.note_reply_speed(&crate::chat::ChatStats { cached_tokens: Some(2900), ..slow.clone() });
        assert!(e.status().warnings.is_empty());
        e.note_reply_speed(&slow);
        let w = e.status().warnings;
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("51 tokens/s") && w[0].contains("margin"), "{}", w[0]);
        // Checked once per load.
        e.inner.lock().unwrap().status.warnings.clear();
        e.note_reply_speed(&slow);
        assert!(e.status().warnings.is_empty());
        // CPU only (0 layers on the GPU): CPU prompts are slow anyway, no check.
        let c = Engine::new();
        c.set_status(|s| {
            s.state = EngineState::Loaded;
            s.gpu_layers_offloaded = Some((0, 66));
        });
        c.note_reply_speed(&slow);
        assert!(c.status().warnings.is_empty());
    }

    #[test]
    fn diagnoses_oom_and_bad_files() {
        let (m, h) = diagnose("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: cudaMalloc failed: out of memory", Some(1));
        assert!(m.contains("out of memory") && h.unwrap().contains("context length"));
        let (m, _) = diagnose("gguf_init_from_file_impl: invalid magic characters: 'PK..'", Some(1));
        assert!(m.contains("not a valid GGUF"));
        let (m, _) = diagnose("something odd\nfinal line here", Some(3));
        assert!(m.contains("exit code 3") && m.contains("final line here"));
    }
}

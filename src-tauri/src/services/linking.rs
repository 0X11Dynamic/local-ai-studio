//! Linked PCs inside the app (no separate program): the helper's listener and
//! its ggml-rpc-server child, pairing, the main PC's control channels and RPC
//! bridges, and the split resolved for a load.
//!
//! The only extra process is llama.cpp's own `ggml-rpc-server.exe` from the
//! installed runtime folder, started with no console window inside a
//! kill-on-close Job Object while "Let this PC help" is on, bound to
//! 127.0.0.1. "Let this PC help" is off at every app start.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;

use app_core::{AppError, AppResult};
use linking::cache::{self, CacheStore};
use linking::client::{self, Control, ControlState, RpcBridge};
use linking::identity::{key_from_b64, key_to_b64};
use linking::reach::{self, ReachReport, UpnpOutcome};
use linking::split::{self, ListedDevice};
use linking::helper::{GuardSpec, HookFuture, RpcTarget};
use linking::{CodeSlot, CodeStatus, DevKind, DeviceInfo, HelperHooks, HelperServer, Identity, LinkError, PeerBudget, PeerInfo, Refusal, SessionEvent, ShareLimits};
use persistence::{Db, NewPeer, PeerRow, PeerShare};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::state::SharedState;

const SET_HELPER: &str = "linking.helper";
const SET_NAME: &str = "linking.name";
/// Measured generation speeds: "<model id>|alone" or "<model id>|split:<peer ids>" -> recent tok/s.
const SET_SPEEDS: &str = "linking.speeds";
const SPEED_SAMPLES: usize = 20;
const RPC_SERVER_EXE: &str = "ggml-rpc-server.exe";
const KEEPALIVE: Duration = Duration::from_secs(15);
const BW_TEST_BYTES: usize = 2 << 20;
/// The bandwidth test gives up after this; it runs in the background on its
/// own connection and never holds up linking or a load.
const BW_TEST_LIMIT: Duration = Duration::from_secs(10);
/// Connecting a control channel (dial, handshake, RTT) gives up after this.
const CONNECT_LIMIT: Duration = Duration::from_secs(20);
const LOG_LINES: usize = 200;

/// What the helper shares ("the helper decides what it shares").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HelperSettings {
    /// TCP port for linking and the tunnel.
    pub port: u16,
    /// rpc-server devices (`-d`), e.g. ["CUDA0"]; empty = all.
    pub devices: Vec<String>,
    /// CPU threads for rpc-server (`-t`) by default; None =
    /// [`default_rpc_threads`]. A linked PC's own setting overrides it.
    pub threads: Option<u32>,
    /// Older builds: "Most memory to lend" (MiB), applied by the main PC.
    /// Read once and moved to `share_gpu_mib` and `share_ram_mib` (both),
    /// then saved as None. Kept for one release.
    pub ram_cap_mib: Option<u64>,
    /// "Default limits for linked PCs": most GPU memory (MiB) a linked PC may
    /// use here; None = 75% of the shared GPUs' free memory when it connects.
    pub share_gpu_mib: Option<u64>,
    /// Most RAM (MiB) when the CPU is shared; None = 50% of available RAM
    /// after the RAM reserve.
    pub share_ram_mib: Option<u64>,
    /// "Speed up reloads": rpc-server `-c` keeps received weights on disk so
    /// the next load doesn't send them again (in `<data>\linking-cache`, see
    /// `linking::cache`). Off: rpc-server runs without `-c` and writes nothing.
    pub cache: bool,
    /// Size cap of that cache in GB; the oldest files go first.
    pub cache_cap_gb: u32,
    /// lz4 inside the tunnel (off by default; for slow links, not measured yet).
    pub compress: bool,
    /// Runtime whose rpc-server to run; None = the default runtime.
    pub runtime_id: Option<String>,
}

/// Said once after the stored port moved from the old default.
const PORT_MIGRATED_NOTICE: &str = "The linking port changed from 50070 to 47070: Windows can reserve ports around 50070 after a restart (Hyper-V, WSL, Docker). If you had set 50070 yourself, change the port back under \"Let this PC help\". The other PC can leave the port empty.";

impl Default for HelperSettings {
    fn default() -> Self {
        Self {
            port: linking::DEFAULT_PORT,
            devices: Vec::new(),
            threads: None,
            ram_cap_mib: None,
            share_gpu_mib: None,
            share_ram_mib: None,
            cache: true,
            cache_cap_gb: DEFAULT_CACHE_CAP_GB,
            compress: false,
            runtime_id: None,
        }
    }
}

const DEFAULT_CACHE_CAP_GB: u32 = (cache::DEFAULT_CAP_BYTES >> 30) as u32;

impl HelperSettings {
    fn cache_cap_bytes(&self) -> u64 {
        (self.cache_cap_gb as u64) << 30
    }

    /// Move the old default port (50070, inside Windows' dynamic range) to the
    /// new one. A stored 50070 can't be told from a chosen one, so every 50070
    /// moves (the user is told once). Returns whether it changed.
    fn migrate_port(&mut self) -> bool {
        if self.port == linking::LEGACY_PORT {
            self.port = linking::DEFAULT_PORT;
            true
        } else {
            false
        }
    }

    /// The single "memory to lend" of older builds becomes both default caps.
    fn migrate_legacy(mut self) -> Self {
        if let Some(c) = self.ram_cap_mib.take()
            && self.share_gpu_mib.is_none()
            && self.share_ram_mib.is_none()
        {
            self.share_gpu_mib = Some(c);
            self.share_ram_mib = Some(c);
        }
        self
    }
}

/// Most CPU threads a linked PC may be given: every logical processor.
fn max_threads() -> u32 {
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(512).clamp(1, 512)
}

fn validate_threads(t: Option<u32>) -> AppResult<()> {
    let max = max_threads();
    if t.is_some_and(|t| t == 0 || t > max) {
        return Err(AppError::InvalidInput(format!("CPU threads must be between 1 and {max} (this PC's logical processors)")));
    }
    Ok(())
}

impl HelperSettings {
    fn validate(&self) -> AppResult<()> {
        if self.port < 1024 {
            return Err(AppError::InvalidInput("pick a port from 1024 to 65535".into()));
        }
        validate_threads(self.threads)?;
        if !(1..=2000).contains(&self.cache_cap_gb) {
            return Err(AppError::InvalidInput("the cache size cap must be between 1 and 2000 GB".into()));
        }
        if self.devices.len() > 16 || !self.devices.iter().all(|d| !d.is_empty() && d.len() <= 32 && d.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')) {
            return Err(AppError::InvalidInput("device names are like CUDA0, Vulkan0 or CPU".into()));
        }
        Ok(())
    }
}

/// Live helper state, read by the listener's hooks.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HelperLive {
    pub helping: bool,
    pub rpc_endpoint: Option<String>,
    pub runtime_id: Option<String>,
    pub runtime_tag: Option<String>,
    pub runtime_variant: Option<String>,
    /// Devices the running rpc-server offers (its startup banner).
    pub devices: Vec<DeviceInfo>,
    /// Every device seen when rpc-server ran with all devices.
    pub available_devices: Vec<DeviceInfo>,
    pub ram_cap_mib: Option<u64>,
    /// Measured RAM read bandwidth (GB/s), reported for speed estimates.
    pub ram_bandwidth_gbs: Option<f64>,
    pub error: Option<String>,
}

/// Shared with the listener (implements [`HelperHooks`]).
struct Shared {
    db: Db,
    name: RwLock<String>,
    port: RwLock<u16>,
    /// Pinned keys -> peer name.
    pinned: RwLock<HashMap<[u8; 32], String>>,
    live: RwLock<HelperLive>,
    sessions: Mutex<HashMap<u64, SessionEvent>>,
    rpc_log: Arc<Mutex<VecDeque<String>>>,
    /// The helper's weight cache (set once the data folder is known).
    cache: RwLock<Option<Arc<CacheStore>>>,
    /// "Speed up reloads" is on for the running rpc-server.
    cache_on: AtomicBool,
    /// Its size cap in bytes.
    cache_cap: AtomicU64,
    /// Cache files at the moment each RPC session opened (session id -> names).
    rpc_before: Mutex<HashMap<u64, BTreeSet<String>>>,
    /// What each linked PC may use here, and uses (enforced in the tunnel).
    budgets: Mutex<HashMap<[u8; 32], Arc<PeerBudget>>>,
    /// Per-linked-PC limits as stored (None fields = the defaults).
    shares: RwLock<HashMap<[u8; 32], PeerShare>>,
    /// The helper settings (default limits, default threads).
    settings: RwLock<HelperSettings>,
    /// `-t` of the running rpc-server.
    rpc_threads: AtomicU32,
    /// The app, for free memory now and for restarting rpc-server with
    /// another thread count (set while helping).
    app: RwLock<Weak<crate::state::AppState>>,
}

impl Shared {
    fn cache(&self) -> Option<Arc<CacheStore>> {
        self.cache.read().unwrap().clone()
    }

    /// Kinds of the running rpc-server's devices, in its order.
    fn device_kinds(&self) -> Vec<DevKind> {
        self.live.read().unwrap().devices.iter().map(|d| DevKind::of(&d.name)).collect()
    }

    /// Free GPU memory (MiB) of the shared GPUs and (available, total) RAM, now.
    fn free_now(&self) -> (Option<u64>, u64, u64) {
        let banner_gpu: u64 = self.live.read().unwrap().devices.iter().filter(|d| DevKind::of(&d.name) == DevKind::Gpu).map(|d| d.free_mib).sum();
        match self.app.read().unwrap().upgrade() {
            Some(st) => {
                let hw = st.monitor.hardware_info();
                let gpu: u64 = hw.gpus.iter().map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes) >> 20).sum();
                (Some(if hw.gpus.is_empty() { banner_gpu } else { gpu }), hw.available_ram_bytes, hw.total_ram_bytes)
            }
            None => (Some(banner_gpu), 0, 0),
        }
    }

    /// The effective limits for a linked PC: its own settings, else the
    /// helper defaults, else the built-in defaults from free memory now.
    fn resolve_limits(&self, share: &PeerShare, kinds: &[DevKind]) -> ShareLimits {
        let s = self.settings.read().unwrap().clone();
        let (gpu_free, ram_avail, ram_total) = self.free_now();
        let d = default_share_limits(kinds, gpu_free.unwrap_or(0), ram_avail, ram_total, s.threads.unwrap_or_else(local_default_rpc_threads));
        let has = |k: DevKind| kinds.contains(&k);
        ShareLimits {
            gpu_mib: if has(DevKind::Gpu) { share.gpu_mib.or(s.share_gpu_mib).unwrap_or(d.gpu_mib) } else { 0 },
            ram_mib: if has(DevKind::Ram) { share.ram_mib.or(s.share_ram_mib).unwrap_or(d.ram_mib) } else { 0 },
            threads: share.threads.unwrap_or(d.threads).clamp(1, max_threads()),
            paused: share.paused,
        }
    }

    fn share_of(&self, key: &[u8; 32]) -> PeerShare {
        self.shares.read().unwrap().get(key).cloned().unwrap_or_default()
    }

    /// The limits a linked PC has right now: its budget's while it has a
    /// stream open (fixed for that session), else resolved fresh.
    fn limits_for(&self, key: &[u8; 32], kinds: &[DevKind]) -> ShareLimits {
        let share = self.share_of(key);
        match self.budgets.lock().unwrap().get(key) {
            Some(b) if b.open_streams() > 0 => ShareLimits { paused: share.paused, ..b.caps() },
            _ => self.resolve_limits(&share, kinds),
        }
    }

    /// Restart rpc-server with `threads` (another linked PC's thread limit).
    async fn restart_rpc(&self, threads: u32) -> Result<(), LinkError> {
        let Some(st) = self.app.read().unwrap().upgrade() else {
            return Err(LinkError::HelperOff);
        };
        st.linking.restart_rpc(&st, threads).await.map_err(|e| LinkError::Refused(format!("could not restart the helper's rpc-server: {e}")))
    }

    async fn rpc_target_inner(&self, key: &[u8; 32]) -> Result<RpcTarget, LinkError> {
        let me = self.name.read().unwrap().clone();
        let endpoint = |l: &HelperLive| match (&l.rpc_endpoint, l.helping) {
            (Some(ep), true) => ep.parse::<SocketAddr>().map_err(|_| LinkError::HelperOff),
            _ => Err(LinkError::HelperOff),
        };
        endpoint(&self.live.read().unwrap())?;
        let kinds = self.device_kinds();
        if kinds.is_empty() {
            // Fail closed: without the device list no limit can be enforced.
            return Err(LinkError::Refused(format!(
                "\"{me}\" couldn't read which devices its rpc-server shares, so it can't enforce its sharing limits; turn \"Let this PC help\" off and on there"
            )));
        }
        let share = self.share_of(key);
        if share.paused {
            return Err(LinkError::Refused(format!("\"{me}\" isn't sharing with this PC right now (paused by its owner).")));
        }
        // Busy check and this stream's claim in one step.
        let claim = {
            let mut budgets = self.budgets.lock().unwrap();
            if let Some((other, _)) = budgets.iter().find(|(k, b)| *k != key && b.open_streams() > 0) {
                let other = self.pinned.read().unwrap().get(other).cloned().unwrap_or_else(|| "another linked PC".into());
                return Err(LinkError::Refused(format!("\"{me}\" is busy helping \"{other}\" right now; try again when that model is unloaded.")));
            }
            let b = budgets.entry(*key).or_insert_with(|| PeerBudget::new(ShareLimits::default())).clone();
            if b.open_streams() == 0 {
                // A new session: limits from the settings and free memory now.
                b.set_caps(self.resolve_limits(&share, &kinds));
            }
            b.claim()
        };
        let caps = claim.budget().caps();
        // Thread limit: one rpc-server, restarted only between sessions.
        if caps.threads != self.rpc_threads.load(Ordering::Relaxed) && claim.budget().open_streams() == 1 {
            tracing::info!("linking: restarting rpc-server with -t {} for this linked PC", caps.threads);
            self.restart_rpc(caps.threads).await?;
        }
        let addr = endpoint(&self.live.read().unwrap())?;
        let kinds = Arc::new(self.device_kinds());
        Ok(RpcTarget { addr, limits: Some(GuardSpec { claim, kinds }) })
    }
}

/// The built-in limits for a linked PC (the owner's answers, 2026-11-23):
/// 75% of the shared GPUs' free memory, 50% of available RAM after the RAM
/// reserve (only when the CPU is shared), both rounded down to 0.5 GB; the
/// default thread count. `gpu_free_mib` is the shared GPUs' free memory.
pub fn default_share_limits(kinds: &[DevKind], gpu_free_mib: u64, ram_avail: u64, ram_total: u64, threads: u32) -> ShareLimits {
    let half_gb = |mib: u64| mib / 512 * 512;
    let gpu = if kinds.contains(&DevKind::Gpu) { half_gb(gpu_free_mib * 3 / 4) } else { 0 };
    let ram = if kinds.contains(&DevKind::Ram) {
        let usable = ram_avail.saturating_sub(crate::commands::models::ram_reserve(ram_total));
        half_gb((usable / 2) >> 20)
    } else {
        0
    };
    ShareLimits { gpu_mib: gpu, ram_mib: ram, threads: threads.max(1), paused: false }
}

fn local_default_rpc_threads() -> u32 {
    default_rpc_threads(system_monitor::core_topology(), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
}

/// Devices as a linked PC sees them through its limits: each reports at
/// most what is left of the limit for its kind.
/// The devices a helper lends this PC, from its info: the free memory it
/// reports (lowered by its limits on new helpers); an older helper's single
/// cap is spread over its devices as the split does.
pub fn shares_from_info(name: &str, endpoint: &str, info: &PeerInfo) -> Vec<LinkedShare> {
    let free: Vec<u64> = info.devices.iter().map(|d| d.free_mib << 20).collect();
    let total: u64 = free.iter().sum();
    let cap = if info.share_enforced { None } else { info.ram_cap_mib.map(|c| c << 20) };
    info.devices
        .iter()
        .zip(free)
        .map(|(d, f)| {
            let bytes = match cap {
                Some(c) if total > c && total > 0 => (f as u128 * c as u128 / total as u128) as u64,
                _ => f,
            };
            let kind = DevKind::of(&d.name);
            LinkedShare {
                peer: name.to_string(),
                endpoint: endpoint.to_string(),
                device: d.name.clone(),
                bytes,
                cpu: kind == DevKind::Ram,
                cap_bytes: info.share.map(|s| s.cap_bytes(kind)).or(cap),
            }
        })
        .collect()
}

fn capped_devices(devices: &[DeviceInfo], limits: &ShareLimits, used: impl Fn(DevKind) -> u64) -> Vec<DeviceInfo> {
    devices
        .iter()
        .map(|d| {
            let k = DevKind::of(&d.name);
            let cap = limits.cap_bytes(k) >> 20;
            let left = cap.saturating_sub(used(k) >> 20);
            DeviceInfo { free_mib: d.free_mib.min(left), total_mib: d.total_mib.min(cap), ..d.clone() }
        })
        .collect()
}

impl HelperHooks for Shared {
    fn peer_name(&self, key: &[u8; 32]) -> Option<String> {
        self.pinned.read().unwrap().get(key).cloned()
    }

    fn paired(&self, key: [u8; 32], info: &PeerInfo, addr: SocketAddr) {
        self.pinned.write().unwrap().insert(key, info.name.clone());
        let db = self.db.clone();
        let info = info.clone();
        tokio::spawn(async move {
            // Where it connected from is shown only ("connected from"); behind
            // NAT it is not an address this PC could dial back (e.g. 10.0.2.2).
            let from = addr.ip().to_string();
            let r = db
                .upsert_peer(&NewPeer {
                    name: &info.name,
                    public_key: &key_to_b64(&key),
                    last_address: None,
                    same_network: reach::is_lan_address(addr.ip()),
                    initiated: false,
                    runtime_tag: info.runtime_tag.as_deref(),
                    seen_from: Some(&from),
                })
                .await;
            if let Err(e) = r {
                tracing::warn!("could not save the linked PC: {e}");
            }
        });
    }

    fn unlinked(&self, key: [u8; 32]) {
        self.pinned.write().unwrap().remove(&key);
        // The weights this PC sent are deleted with the link.
        if let Some(c) = self.cache() {
            let freed = c.remove_peer(&key_to_b64(&key));
            tracing::info!("linked PC unlinked this PC: removed {} MB of its cached weights", freed >> 20);
        }
        let db = self.db.clone();
        tokio::spawn(async move {
            let _ = db.delete_peer_by_key(&key_to_b64(&key)).await;
        });
        tracing::info!("a linked PC unlinked this PC");
    }

    fn info(&self) -> PeerInfo {
        let l = self.live.read().unwrap();
        PeerInfo {
            name: self.name.read().unwrap().clone(),
            app_version: env!("CARGO_PKG_VERSION").into(),
            listen_port: Some(*self.port.read().unwrap()),
            helping: l.helping,
            runtime_tag: l.runtime_tag.clone(),
            runtime_variant: l.runtime_variant.clone(),
            devices: l.devices.clone(),
            ram_cap_mib: l.ram_cap_mib,
            ram_bandwidth_gbs: l.ram_bandwidth_gbs,
            share: None,
            share_enforced: true,
        }
    }

    /// What `peer` may use here: its limits, and each device's free memory
    /// lowered to what is left of them (as the guard reports it).
    fn info_for(&self, peer: Option<&[u8; 32]>) -> PeerInfo {
        let mut info = self.info();
        let Some(key) = peer else { return info };
        let kinds = self.device_kinds();
        let limits = self.limits_for(key, &kinds);
        let budget = self.budgets.lock().unwrap().get(key).cloned();
        info.devices = capped_devices(&info.devices, &limits, |k| budget.as_ref().map_or(0, |b| b.used_bytes(k)));
        // Older main PCs only know one total cap, which they apply to their split.
        info.ram_cap_mib = Some(limits.gpu_mib.saturating_add(limits.ram_mib));
        info.share = Some(limits);
        info
    }

    fn rpc_target<'a>(&'a self, key: &'a [u8; 32]) -> HookFuture<'a, Result<RpcTarget, LinkError>> {
        Box::pin(self.rpc_target_inner(key))
    }

    fn refused(&self, key: &[u8; 32], r: &Refusal) {
        let name = self.pinned.read().unwrap().get(key).cloned().unwrap_or_default();
        let line = match r {
            Refusal::OverLimit { kind, bytes, cap_bytes, used_bytes } => format!(
                "refused \"{name}\" {:.1} GB of {}: over its limit ({:.1} GB in use of {:.1} GB)",
                *bytes as f64 / 1e9,
                if *kind == DevKind::Gpu { "GPU memory" } else { "RAM" },
                *used_bytes as f64 / 1e9,
                *cap_bytes as f64 / 1e9
            ),
            Refusal::Incompatible { proto_major } => format!("refused \"{name}\": it uses ggml-rpc protocol {proto_major}, so this PC can't enforce its limits"),
        };
        tracing::info!("linking: {line}");
        push_log(&self.rpc_log, format!("[sharing] {line}"));
    }

    fn session(&self, ev: SessionEvent) {
        if ev.open && ev.mode == "control"
            && let Ok(a) = ev.remote.parse::<SocketAddr>()
        {
            let (db, key, ip) = (self.db.clone(), ev.peer_key.clone(), a.ip().to_string());
            tokio::spawn(async move {
                let _ = db.set_peer_seen_from(&key, &ip).await;
            });
        }
        // Weight cache: files that appear while a linked PC's stream is open
        // are that PC's; after each stream the size cap is enforced.
        if ev.mode == "rpc"
            && self.cache_on.load(Ordering::Relaxed)
            && let Some(c) = self.cache()
        {
            // (No lock guard is held across the cache's file work.)
            let before = if ev.open { None } else { self.rpc_before.lock().unwrap().remove(&ev.id) };
            if ev.open {
                let snap = c.snapshot();
                self.rpc_before.lock().unwrap().insert(ev.id, snap);
            } else if let Some(before) = before {
                c.attribute(&ev.peer_key, &before);
                let freed = c.enforce_cap(self.cache_cap.load(Ordering::Relaxed));
                if freed > 0 {
                    tracing::info!("weight cache over its cap: removed {} MB (oldest first)", freed >> 20);
                }
            }
        }
        let mut s = self.sessions.lock().unwrap();
        if ev.open {
            s.insert(ev.id, ev);
        } else {
            s.remove(&ev.id);
        }
    }
}

struct RpcChild {
    job: inference::job::JobObject,
    /// Set before an intentional stop, so the exit watcher stays quiet.
    stopping: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default)]
struct HelperRun {
    server: Option<HelperServer>,
    rpc: Option<RpcChild>,
    upnp_port: Option<u16>,
    /// The port the listener was asked for (it may listen on another one).
    asked_port: Option<u16>,
}

struct OutPeer {
    control: Arc<tokio::sync::Mutex<Control>>,
    /// Info (pushed by the helper) and whether the channel closed.
    state: Arc<ControlState>,
    address: String,
    key: [u8; 32],
    rtt_ms: f64,
    bandwidth_mbps: Option<f64>,
    bridge: Option<Arc<RpcBridge>>,
    /// Keepalive and close watcher.
    tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Which connection this is (a watcher only removes its own).
    generation: u64,
}

impl Drop for OutPeer {
    fn drop(&mut self) {
        for t in self.tasks.drain(..) {
            t.abort();
        }
        if let Some(b) = &self.bridge {
            b.stop();
        }
    }
}

/// This PC typed the code and knows an address to dial.
fn dialable(p: &PeerRow) -> bool {
    p.initiated != 0 && p.last_address.is_some()
}

pub struct LinkService {
    identity: Arc<Identity>,
    codes: Arc<CodeSlot>,
    shared: Arc<Shared>,
    run: tokio::sync::Mutex<HelperRun>,
    out: Arc<tokio::sync::Mutex<HashMap<String, OutPeer>>>,
    reach: Mutex<Option<ReachReport>>,
    settings: Mutex<HelperSettings>,
    /// Measured generation speeds by `speed_key` (persisted in SET_SPEEDS).
    speeds: Mutex<HashMap<String, Vec<f64>>>,
    /// Peer ids of the last split load (sorted), for its speed key.
    last_split_ids: Mutex<Vec<String>>,
    generation: std::sync::atomic::AtomicU64,
    /// Cancels running background bandwidth tests (a load starts).
    bw_cancel: Mutex<tokio_util::sync::CancellationToken>,
    /// Linked PCs taking part in the load being set up (their byte counters
    /// and expected shares), for the load progress.
    load_probes: Mutex<Vec<inference::PeerProbe>>,
    /// The stored port was moved to the new default at startup (shown until dismissed).
    port_notice: Mutex<Option<String>>,
    /// The listener is on another port than the one set, and why.
    bind_notice: Mutex<Option<String>>,
}

// ---------------------------------------------------------------- views

#[derive(Debug, Clone, Serialize)]
pub struct PeerView {
    pub id: String,
    pub name: String,
    pub fingerprint: String,
    /// Address this PC dials (only for PCs it linked to itself).
    pub address: Option<String>,
    /// For PCs that linked to this one: the IP they connected from (shown only).
    pub seen_from: Option<String>,
    /// This PC can connect to it (it typed the code and knows the address).
    pub dialable: bool,
    pub same_network: bool,
    pub split_layers: bool,
    /// Option B ("send whole jobs"), a later phase: stored only.
    pub send_jobs: bool,
    /// This PC typed the code (it can dial the other PC).
    pub initiated: bool,
    /// This PC has a live control connection to it.
    pub online: bool,
    pub rtt_ms: Option<f64>,
    pub bandwidth_mbps: Option<f64>,
    /// About how many ms a split adds per generated token (1 RTT, measured).
    pub added_ms_per_token: Option<f64>,
    pub runtime_tag: Option<String>,
    pub helping: bool,
    pub devices: Vec<DeviceInfo>,
    pub ram_cap_mib: Option<u64>,
    pub last_seen_at: Option<String>,
    /// It has connections open to this PC right now (it uses this PC).
    pub connected_here: bool,
    /// Bytes of this PC's weights in the helper cache here (it sent them).
    pub cache_bytes: u64,
    /// What this PC lends to it (enforced here) and what it uses now.
    pub lends: LendView,
    /// What it lets this PC use there (from its last info), None = unknown
    /// or an older helper.
    pub offers: Option<ShareLimits>,
    /// It enforces `offers` itself (false: an older version; only its single
    /// memory cap exists, applied by this PC's split).
    pub offers_enforced: bool,
    /// Its last refusal of this PC's memory request, in the user's words.
    pub refusal: Option<String>,
}

/// What this PC lends to one linked PC.
#[derive(Debug, Clone, Serialize)]
pub struct LendView {
    /// Effective limits (defaults resolved).
    pub limits: ShareLimits,
    /// As stored: None fields use the defaults.
    pub stored: PeerShare,
    pub used_gpu_mib: u64,
    pub used_ram_mib: u64,
    /// Kinds of memory this PC shares at all (its rpc-server devices).
    pub shares_gpu: bool,
    pub shares_ram: bool,
    pub refused: u64,
    pub refused_bytes: u64,
    /// Seconds since the last refusal.
    pub last_refused_secs_ago: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeView {
    pub code: String,
    pub state: linking::code::CodeState,
    pub seconds_left: u64,
    pub failures: u32,
}

impl From<CodeStatus> for CodeView {
    fn from(c: CodeStatus) -> Self {
        Self { code: c.code, state: c.state, seconds_left: c.seconds_left, failures: c.failures }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HelperView {
    pub enabled: bool,
    pub settings: HelperSettings,
    pub live: HelperLive,
    /// Address:port the listener is bound to (all interfaces), when listening.
    pub listening: Option<String>,
    /// The port actually listening (may differ from `settings.port`).
    pub listen_port: Option<u16>,
    /// Set when the chosen port was reserved or in use and another one is used.
    pub bind_notice: Option<String>,
    /// Set (once per start) when the stored port was moved to the new default.
    pub port_notice: Option<String>,
    pub log_tail: Vec<String>,
    pub cache: CacheView,
}

/// The helper's weight cache on disk.
#[derive(Debug, Clone, Serialize)]
pub struct CacheView {
    /// "Speed up reloads" is on.
    pub enabled: bool,
    pub bytes: u64,
    pub cap_bytes: u64,
    pub dir: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkStatus {
    pub name: String,
    pub fingerprint: String,
    pub local_ip: Option<String>,
    pub helper: HelperView,
    pub code: Option<CodeView>,
    pub reach: Option<ReachReport>,
    pub peers: Vec<PeerView>,
    pub sessions: Vec<SessionEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitEstimate {
    pub peer_id: String,
    pub name: String,
    pub online: bool,
    pub rtt_ms: Option<f64>,
    pub added_ms_per_token: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitPreview {
    /// Linked PCs with "Split model layers with this PC" on.
    pub peers: Vec<SplitEstimate>,
    /// This PC alone: median reply speed of the latest load without linked PCs.
    pub alone_tok_s: Option<f64>,
    /// Estimated generation speed with every split peer, as a range: each
    /// device reads its share of the weights per token at its memory
    /// bandwidth, plus one round trip per linked PC.
    pub est_low_tok_s: Option<f64>,
    pub est_high_tok_s: Option<f64>,
    /// What the estimate assumed, one line per device and the network.
    pub est_basis: Vec<String>,
    /// Why there is no estimate (e.g. a linked PC is offline).
    pub est_missing: Option<String>,
    /// Measured: median reply speed of the latest load with linked PCs.
    pub split_tok_s: Option<f64>,
    pub split_replies: usize,
    /// The latest split load was slower than this PC alone: suggest turning split off.
    pub suggest_off: bool,
    pub message: Option<String>,
}

/// This PC's RAM read bandwidth (GB/s), measured once (~0.3 s, off the async threads).
static RAM_GBS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

async fn ram_gbs() -> f64 {
    if let Some(v) = RAM_GBS.get() {
        return *v;
    }
    let v = tokio::task::spawn_blocking(split::probe_ram_bandwidth_gbs).await.unwrap_or(split::DEFAULT_RAM_GBS);
    *RAM_GBS.get_or_init(|| v)
}

/// This PC's GPU, for the split estimate.
#[derive(Debug, Clone, Default)]
pub struct LocalHw {
    pub gpu_name: Option<String>,
    pub free_vram: Option<u64>,
    pub total_vram: Option<u64>,
    /// Share of the model's layers that fit on this PC's GPU when it last
    /// loaded the model alone (llama.cpp's own fit), if it has.
    pub alone_gpu_layers: Option<f64>,
}

/// Share of a GPU's memory that holds weights when the rest goes to the
/// context and working buffers, when this PC hasn't loaded the model alone.
const DEFAULT_WEIGHT_SHARE: f64 = 0.6;

/// A linked PC ready for a split load.
pub struct SplitPeer {
    pub id: String,
    pub name: String,
    pub endpoint: String,
    pub ram_cap_mib: Option<u64>,
    /// What it lets this PC use (None: an older helper).
    pub share: Option<ShareLimits>,
    /// It enforces its limits itself.
    pub enforced: bool,
    /// Kind of each of its devices, in its rpc-server order.
    pub kinds: Vec<DevKind>,
}

/// A helper's `share_refused` message in the user's words (main PC side).
pub fn refusal_message(helper: &str, r: &serde_json::Value) -> String {
    let gb = |k: &str| r[k].as_u64().unwrap_or(0) as f64 / 1e9;
    match r["reason"].as_str() {
        Some("incompatible") => format!(
            "\"{helper}\" can't enforce its sharing limits with this PC's llama.cpp (RPC protocol {}), so it refused the link. Update Local AI Studio on \"{helper}\".",
            r["proto_major"].as_u64().unwrap_or(0)
        ),
        _ => format!(
            "\"{helper}\" refused {:.1} GB: its owner lets this PC use up to {:.1} GB of {}. Use a smaller context, or ask them to raise the limit on their Linked PCs page.",
            gb("bytes"),
            gb("cap_bytes"),
            if r["kind"] == "ram" { "RAM" } else { "GPU memory" }
        ),
    }
}

/// A device a linked PC lends for one load.
#[derive(Debug, Clone, Serialize)]
pub struct LinkedShare {
    pub peer: String,
    pub endpoint: String,
    /// e.g. "RPC0".
    pub device: String,
    /// Bytes this PC may use there (free through the helper's limit).
    pub bytes: u64,
    pub cpu: bool,
    /// The helper's limit for that kind of memory, when known.
    pub cap_bytes: Option<u64>,
}

/// What "Use linked PCs" resolved for one load, for sizing and the split.
#[derive(Debug, Clone, Default)]
pub struct LinkedLoad {
    pub devices: Vec<ListedDevice>,
    /// (endpoint, cap MiB for older helpers).
    pub caps: Vec<(String, Option<u64>)>,
    /// endpoint -> linked PC name.
    pub names: HashMap<String, String>,
    pub shares: Vec<LinkedShare>,
}

impl LinkedLoad {
    pub fn linked_devs(&self) -> Vec<model_registry::estimate::LinkedDev> {
        self.shares.iter().map(|s| model_registry::estimate::LinkedDev { bytes: s.bytes, cpu: s.cpu }).collect()
    }
}

fn key_err(e: LinkError) -> AppError {
    AppError::from(e)
}

impl LinkService {
    pub async fn new(db: Db, credentials: &security::CredentialStore) -> AppResult<Self> {
        let identity = Arc::new(Identity::load_or_create(credentials)?);
        let mut pinned = HashMap::new();
        let mut shares: HashMap<[u8; 32], PeerShare> = HashMap::new();
        for p in db.list_peers().await? {
            match key_from_b64(&p.public_key) {
                Ok(k) => {
                    shares.insert(k, p.share());
                    pinned.insert(k, p.name);
                }
                Err(_) => tracing::warn!("linked PC {} has an unreadable key; ignored", p.id),
            }
        }
        let name = match db.get_setting::<String>(SET_NAME).await? {
            Some(n) if !n.trim().is_empty() => n,
            _ => std::env::var("COMPUTERNAME").unwrap_or_else(|_| "This PC".into()),
        };
        let raw: Option<HelperSettings> = db.get_setting(SET_HELPER).await?;
        let legacy = raw.as_ref().is_some_and(|s| s.ram_cap_mib.is_some());
        let mut settings = raw.unwrap_or_default().migrate_legacy();
        let port_moved = settings.migrate_port();
        if legacy || port_moved {
            db.set_setting(SET_HELPER, &settings).await?;
        }
        let speeds: HashMap<String, Vec<f64>> = db.get_setting(SET_SPEEDS).await.ok().flatten().unwrap_or_default();
        let shared = Arc::new(Shared {
            db,
            name: RwLock::new(name),
            port: RwLock::new(settings.port),
            pinned: RwLock::new(pinned),
            live: RwLock::new(HelperLive::default()),
            sessions: Mutex::new(HashMap::new()),
            rpc_log: Arc::new(Mutex::new(VecDeque::new())),
            cache: RwLock::new(None),
            cache_on: AtomicBool::new(false),
            cache_cap: AtomicU64::new(cache::DEFAULT_CAP_BYTES),
            rpc_before: Mutex::new(HashMap::new()),
            budgets: Mutex::new(HashMap::new()),
            shares: RwLock::new(shares),
            settings: RwLock::new(settings.clone()),
            rpc_threads: AtomicU32::new(0),
            app: RwLock::new(Weak::new()),
        });
        Ok(Self {
            identity,
            codes: Arc::new(CodeSlot::new()),
            shared,
            run: Default::default(),
            out: Default::default(),
            reach: Mutex::new(None),
            settings: Mutex::new(settings),
            speeds: Mutex::new(speeds),
            last_split_ids: Mutex::new(Vec::new()),
            generation: Default::default(),
            bw_cancel: Mutex::new(tokio_util::sync::CancellationToken::new()),
            load_probes: Mutex::new(Vec::new()),
            port_notice: Mutex::new(port_moved.then(|| PORT_MIGRATED_NOTICE.to_string())),
            bind_notice: Mutex::new(None),
        })
    }

    /// Tell the service where the app's data folder is: the weight cache
    /// lives in `<data>\linking-cache` (also needed to delete a PC's files
    /// when it is unlinked while the helper is off).
    pub fn init_cache(&self, data_dir: &Path) {
        *self.shared.cache.write().unwrap() = Some(Arc::new(CacheStore::new(data_dir.join("linking-cache"))));
        // Older builds cached in `<data>\rpc-cache` with no cap or owner list.
        let _ = std::fs::remove_dir_all(data_dir.join("rpc-cache"));
    }

    /// Delete every cached weight file (the "Clear cache" button). Returns the bytes freed.
    pub fn clear_cache(&self) -> u64 {
        self.shared.cache().map(|c| c.clear()).unwrap_or(0)
    }

    fn cache_view(&self) -> CacheView {
        let s = self.settings.lock().unwrap().clone();
        let c = self.shared.cache();
        CacheView { enabled: s.cache, bytes: c.as_ref().map(|c| c.size()).unwrap_or(0), cap_bytes: s.cache_cap_bytes(), dir: c.map(|c| c.root().display().to_string()) }
    }

    /// Byte counters and expected shares of the linked PCs in the load being set up.
    pub fn load_probes(&self) -> Vec<inference::PeerProbe> {
        self.load_probes.lock().unwrap().clone()
    }

    /// The ggml-rpc counters (tensor data sent, cache hits) of a linked PC's bridge.
    async fn bridge_counters(&self, id: &str) -> Option<Arc<linking::rpc_tap::RpcCounters>> {
        self.out.lock().await.get(id).and_then(|o| o.bridge.as_ref()).map(|b| b.rpc_counters())
    }

    pub fn fingerprint(&self) -> String {
        linking::fingerprint(self.identity.public())
    }

    pub async fn status(&self) -> AppResult<LinkStatus> {
        let rows = self.shared.db.list_peers().await?;
        let peers = self.peer_views(rows).await;
        let run = self.run.lock().await;
        let listening = run.server.as_ref().map(|s| s.local_addr().to_string());
        let listen_port = run.server.as_ref().map(|s| s.local_addr().port());
        let enabled = run.rpc.is_some();
        drop(run);
        // One lock per statement: a guard taken inside the struct literal
        // below would live until the end of the whole expression, and
        // `cache_view()` locks `settings` again (self-deadlock that blocked a
        // runtime thread on every status poll).
        let name = self.shared.name.read().unwrap().clone();
        let settings = self.settings.lock().unwrap().clone();
        let live = self.shared.live.read().unwrap().clone();
        let log_tail: Vec<String> = self.shared.rpc_log.lock().unwrap().iter().rev().take(30).rev().cloned().collect();
        let cache = self.cache_view();
        let bind_notice = self.bind_notice.lock().unwrap().clone().filter(|_| listen_port.is_some());
        let port_notice = self.port_notice.lock().unwrap().clone();
        let reach_report = self.reach.lock().unwrap().clone();
        let sessions: Vec<SessionEvent> = self.shared.sessions.lock().unwrap().values().cloned().collect();
        Ok(LinkStatus {
            name,
            fingerprint: self.fingerprint(),
            local_ip: reach::local_ip().map(|i| i.to_string()),
            helper: HelperView { enabled, settings, live, listening, listen_port, bind_notice, port_notice, log_tail, cache },
            code: self.codes.status().map(CodeView::from),
            reach: reach_report,
            peers,
            sessions,
        })
    }

    async fn peer_views(&self, rows: Vec<PeerRow>) -> Vec<PeerView> {
        let out = self.out.lock().await;
        let sessions = self.shared.sessions.lock().unwrap().clone();
        let cached = self.shared.cache().map(|c| c.sizes_by_peer()).unwrap_or_default();
        rows.into_iter()
            .map(|p| {
                let o = out.get(&p.id).filter(|o| !o.state.is_closed());
                let info = o.map(|o| o.state.info());
                let rtt = o.map(|o| o.rtt_ms).or(p.last_rtt_ms);
                let can_dial = dialable(&p);
                // Older rows stored where a PC connected from as its address.
                let seen_from = p.seen_from.clone().or_else(|| {
                    (!can_dial).then(|| p.last_address.as_deref().map(|a| a.rsplit_once(':').map_or(a, |(h, _)| h).trim_matches(['[', ']']).to_string())).flatten()
                });
                PeerView {
                    fingerprint: key_from_b64(&p.public_key).map(|k| linking::fingerprint(&k)).unwrap_or_default(),
                    connected_here: sessions.values().any(|s| s.peer_key == p.public_key),
                    address: if can_dial { p.last_address.clone() } else { None },
                    seen_from: if can_dial { None } else { seen_from },
                    dialable: can_dial,
                    same_network: p.same_network != 0,
                    split_layers: p.split_layers != 0,
                    send_jobs: p.send_jobs != 0,
                    initiated: p.initiated != 0,
                    online: o.is_some(),
                    rtt_ms: rtt,
                    bandwidth_mbps: o.and_then(|o| o.bandwidth_mbps).or(p.last_bandwidth_mbps),
                    added_ms_per_token: rtt.map(split::added_ms_per_token),
                    runtime_tag: info.as_ref().and_then(|i| i.runtime_tag.clone()).or(p.runtime_tag.clone()),
                    helping: info.as_ref().is_some_and(|i| i.helping),
                    devices: info.as_ref().map(|i| i.devices.clone()).unwrap_or_default(),
                    ram_cap_mib: info.as_ref().and_then(|i| i.ram_cap_mib),
                    last_seen_at: p.last_seen_at.clone(),
                    cache_bytes: cached.get(&p.public_key).copied().unwrap_or(0),
                    lends: self.lend_view(&p),
                    offers: info.as_ref().and_then(|i| i.share),
                    offers_enforced: info.as_ref().is_some_and(|i| i.share_enforced),
                    refusal: o.and_then(|o| o.state.refusal()).map(|r| refusal_message(&p.name, &r)),
                    name: p.name,
                    id: p.id,
                }
            })
            .collect()
    }

    pub async fn set_name(&self, name: &str) -> AppResult<()> {
        let n = name.trim();
        if n.is_empty() || n.len() > 64 {
            return Err(AppError::InvalidInput("the name must be 1 to 64 characters".into()));
        }
        self.shared.db.set_setting(SET_NAME, &n).await?;
        *self.shared.name.write().unwrap() = n.to_string();
        Ok(())
    }

    // ------------------------------------------------------------ helper

    async fn ensure_listener(&self, run: &mut HelperRun, port: u16) -> AppResult<()> {
        if run.server.is_some() && run.asked_port == Some(port) {
            return Ok(());
        }
        if let Some(s) = run.server.take() {
            s.stop();
        }
        *self.bind_notice.lock().unwrap() = None;
        let hooks: Arc<dyn HelperHooks> = self.shared.clone();
        let bound = linking::helper::bind_with_fallback(port, |p| {
            let addr: SocketAddr = (std::net::Ipv4Addr::UNSPECIFIED, p).into();
            HelperServer::bind(addr, self.identity.clone(), self.codes.clone(), hooks.clone())
        })
        .await
        .map_err(AppError::Conflict)?;
        let server = bound.value;
        let actual = server.local_addr().port();
        // The port other PCs must type (also what the info exchange reports).
        *self.shared.port.write().unwrap() = actual;
        if let Some(n) = bound.notice {
            tracing::warn!("linking: {n}");
            *self.bind_notice.lock().unwrap() = Some(n);
        }
        tracing::info!("linking: listening on {}", server.local_addr());
        run.asked_port = Some(port);
        run.server = Some(server);
        Ok(())
    }

    /// Stop the listener when nothing needs it (helper off, no open code).
    async fn maybe_stop_listener(&self, run: &mut HelperRun) {
        if run.rpc.is_none() && !self.codes.is_open() {
            if let Some(s) = run.server.take() {
                s.stop();
                tracing::info!("linking: listener stopped");
            }
            if let Some(p) = run.upnp_port.take() {
                let _ = tokio::time::timeout(Duration::from_secs(5), reach::remove_upnp(p)).await;
            }
        }
    }

    /// Turn "Let this PC help" on (start rpc-server and the listener) or off.
    pub async fn set_helper(&self, st: &SharedState, enabled: bool, settings: HelperSettings) -> AppResult<()> {
        settings.validate()?;
        let settings = settings.migrate_legacy();
        st.db.set_setting(SET_HELPER, &settings).await?;
        *self.settings.lock().unwrap() = settings.clone();
        *self.shared.settings.write().unwrap() = settings.clone();
        *self.shared.app.write().unwrap() = Arc::downgrade(st);
        let mut run = self.run.lock().await;
        stop_rpc(&mut run, &self.shared);
        if !enabled {
            self.maybe_stop_listener(&mut run).await;
            // Still listening for a code: tell linked PCs helping is off.
            if let Some(s) = &run.server {
                s.notify_changed();
            }
            return Ok(());
        }
        let started = start_rpc_server(st, &settings, &self.shared).await;
        let child = match started {
            Ok(c) => c,
            Err(e) => {
                self.shared.live.write().unwrap().error = Some(e.to_string());
                self.maybe_stop_listener(&mut run).await;
                return Err(e);
            }
        };
        run.rpc = Some(child);
        if let Err(e) = self.ensure_listener(&mut run, settings.port).await {
            stop_rpc(&mut run, &self.shared);
            self.shared.live.write().unwrap().error = Some(e.to_string());
            return Err(e);
        }
        if let Some(s) = &run.server {
            s.notify_changed();
        }
        Ok(())
    }

    /// Show a new one-time code (and listen for it).
    pub async fn start_code(self: &Arc<Self>) -> AppResult<CodeView> {
        let port = self.settings.lock().unwrap().port;
        let mut run = self.run.lock().await;
        self.ensure_listener(&mut run, port).await?;
        drop(run);
        let c = self.codes.new_code();
        // Close the listener again once the code is used or expires (unless helping).
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if !me.codes.is_open() {
                    let mut run = me.run.lock().await;
                    me.maybe_stop_listener(&mut run).await;
                    return;
                }
            }
        });
        Ok(c.into())
    }

    pub async fn cancel_code(&self) {
        self.codes.cancel();
        let mut run = self.run.lock().await;
        self.maybe_stop_listener(&mut run).await;
    }

    /// Check whether other networks can reach this PC (UPnP only when asked).
    pub async fn check_reach(self: &Arc<Self>, try_upnp: bool) -> AppResult<ReachReport> {
        // The port actually listening when there is a listener (it may not be the one set).
        let port = match self.run.lock().await.server.as_ref() {
            Some(s) => s.local_addr().port(),
            None => self.settings.lock().unwrap().port,
        };
        let (local, outcome) = if try_upnp { reach::try_upnp(port).await } else { (reach::local_ip(), UpnpOutcome::NotTried) };
        if matches!(outcome, UpnpOutcome::Mapped { .. }) {
            let mut run = self.run.lock().await;
            if run.upnp_port.replace(port).is_none() {
                // Renew the lease while the mapping is wanted.
                let me = self.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_secs(reach::UPNP_LEASE_SECS as u64 / 2)).await;
                        let still = me.run.lock().await.upnp_port;
                        match still {
                            Some(p) => {
                                let _ = reach::try_upnp(p).await;
                            }
                            None => return,
                        }
                    }
                });
            }
        }
        let report = reach::classify(local, port, outcome);
        *self.reach.lock().unwrap() = Some(report.clone());
        Ok(report)
    }

    // ------------------------------------------------------------ main PC

    fn my_info(&self) -> PeerInfo {
        self.shared.info()
    }

    /// Link to the PC at `host` (+ port) that shows `code`.
    pub async fn pair(&self, host: &str, port: Option<u16>, code: &str) -> AppResult<PeerView> {
        // Only an IP typed: the default port, then the port older builds used.
        let (addr, p) = client::pair_typed(host, port, code, &self.identity, &self.my_info()).await.map_err(key_err)?;
        let key = key_to_b64(&p.key);
        let row = self
            .shared
            .db
            .upsert_peer(&NewPeer {
                name: &p.info.name,
                public_key: &key,
                last_address: Some(&addr),
                same_network: reach::is_lan_address(p.address.ip()),
                initiated: true,
                runtime_tag: p.info.runtime_tag.as_deref(),
                seen_from: None,
            })
            .await?;
        self.shared.pinned.write().unwrap().insert(p.key, p.info.name.clone());
        tracing::info!("linked to \"{}\" at {addr} (key {})", p.info.name, linking::fingerprint(&p.key));
        // Measure RTT and bandwidth now; a failure here doesn't undo the link.
        if let Err(e) = self.connect(&row.id).await {
            tracing::info!("linked, but the first connection failed: {e}");
        }
        self.view(&row.id).await
    }

    async fn view(&self, id: &str) -> AppResult<PeerView> {
        let row = self.shared.db.get_peer(id).await?;
        Ok(self.peer_views(vec![row]).await.remove(0))
    }

    /// Open (or re-open) the control connection: info, RTT, bandwidth.
    pub async fn connect(&self, id: &str) -> AppResult<PeerView> {
        self.connect_inner(id).await.map_err(key_err)?;
        self.view(id).await
    }

    /// Connect (bounded by CONNECT_LIMIT), then measure bandwidth in the background.
    async fn connect_inner(&self, id: &str) -> Result<(), LinkError> {
        tokio::time::timeout(CONNECT_LIMIT, self.connect_bounded(id)).await.map_err(|_| LinkError::Timeout)?
    }

    async fn connect_bounded(&self, id: &str) -> Result<(), LinkError> {
        let db_err = |e: AppError| LinkError::Refused(e.to_string());
        let row = self.shared.db.get_peer(id).await.map_err(db_err)?;
        if !dialable(&row) {
            return Err(LinkError::Refused(format!(
                "\"{}\" linked to this PC, so this PC doesn't have an address to connect to it. Link from this PC with a code shown there to use it as a helper",
                row.name
            )));
        }
        let address = row.last_address.clone().unwrap_or_default();
        let key = key_from_b64(&row.public_key)?;
        // Keep a running bridge: a loaded model may be using it.
        let old_bridge = self.out.lock().await.remove(id).and_then(|mut o| o.bridge.take());
        let mut c = Control::connect(&address, &self.identity, key).await?;
        let rtt_ms = c.rtt(3).await?.as_secs_f64() * 1000.0;
        // Bandwidth: measured afterwards in the background (spawn_bw_test).
        let bw = None;
        let info = c.info();
        self.shared
            .db
            .set_peer_seen(id, Some(&info.name), None, Some(rtt_ms), bw, info.runtime_tag.as_deref())
            .await
            .map_err(db_err)?;
        let state = c.state();
        let control = Arc::new(tokio::sync::Mutex::new(c));
        let generation = self.generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Keepalive: RTT, and a silent drop (no answer) closes the entry.
        let keepalive = {
            let (ctl, out, pid) = (control.clone(), self.out.clone(), id.to_string());
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(KEEPALIVE).await;
                    let r = ctl.lock().await.ping().await;
                    let mut out = out.lock().await;
                    match r {
                        Ok(d) => {
                            if let Some(o) = out.get_mut(&pid) {
                                o.rtt_ms = o.rtt_ms * 0.7 + d.as_secs_f64() * 1000.0 * 0.3;
                            }
                        }
                        Err(e) => {
                            tracing::info!("linked PC stopped answering: {e}");
                            if out.get(&pid).is_some_and(|o| o.generation == generation)
                                && let Some(mut o) = out.remove(&pid)
                            {
                                abort_others(&mut o);
                            }
                            return;
                        }
                    }
                }
            })
        };
        // The helper closed the channel (helping off, app closed): offline at once.
        let watcher = {
            let (st, out, pid) = (state.clone(), self.out.clone(), id.to_string());
            tokio::spawn(async move {
                let why = st.wait_closed().await;
                tracing::info!("link to a linked PC closed: {why}");
                let mut out = out.lock().await;
                if out.get(&pid).is_some_and(|o| o.generation == generation)
                    && let Some(mut o) = out.remove(&pid)
                {
                    abort_others(&mut o);
                }
            })
        };
        self.out.lock().await.insert(
            id.into(),
            OutPeer { control, state, address: address.clone(), key, rtt_ms, bandwidth_mbps: bw, bridge: old_bridge, tasks: vec![keepalive, watcher], generation },
        );
        self.spawn_bw_test(id, address, key);
        Ok(())
    }

    /// Measure upload bandwidth on a connection of its own, in the
    /// background, for at most BW_TEST_LIMIT. Cancelled when a load starts,
    /// so it never competes with the weights going over.
    fn spawn_bw_test(&self, id: &str, address: String, key: [u8; 32]) {
        let cancel = self.bw_cancel.lock().unwrap().clone();
        let (identity, out, db, pid) = (self.identity.clone(), self.out.clone(), self.shared.db.clone(), id.to_string());
        tokio::spawn(async move {
            let test = async {
                let mut c = Control::connect(&address, &identity, key).await?;
                let r = c.bandwidth(BW_TEST_BYTES, BW_TEST_LIMIT).await;
                drop(c);
                r
            };
            let r = tokio::select! {
                _ = cancel.cancelled() => return,
                r = tokio::time::timeout(BW_TEST_LIMIT + Duration::from_secs(5), test) => r,
            };
            match r {
                Ok(Ok(mbps)) => {
                    if let Some(o) = out.lock().await.get_mut(&pid) {
                        o.bandwidth_mbps = Some(mbps);
                    }
                    let _ = db.set_peer_seen(&pid, None, None, None, Some(mbps), None).await;
                }
                Ok(Err(e)) => tracing::info!("bandwidth test to {address} failed: {e}"),
                Err(_) => tracing::info!("bandwidth test to {address} timed out"),
            }
        });
    }

    /// Stop background bandwidth tests (a load is starting); later connects
    /// can test again.
    fn cancel_bw_tests(&self) {
        let mut c = self.bw_cancel.lock().unwrap();
        c.cancel();
        *c = tokio_util::sync::CancellationToken::new();
    }

    /// Last error of the RPC bridge to this peer, if any.
    pub async fn bridge_error(&self, id: &str) -> Option<String> {
        self.out.lock().await.get(id).and_then(|o| o.bridge.as_ref()).and_then(|b| b.snapshot().last_error)
    }

}

/// Abort an entry's background tasks except the one running this (it ends on
/// its own), before the entry is dropped.
fn abort_others(o: &mut OutPeer) {
    let me = tokio::task::try_id();
    for t in o.tasks.drain(..) {
        if Some(t.id()) != me {
            t.abort();
        }
    }
}

impl LinkService {
    pub async fn set_split(&self, id: &str, on: bool) -> AppResult<PeerView> {
        self.shared.db.set_peer_split(id, on).await?;
        self.view(id).await
    }

    pub async fn set_send_jobs(&self, id: &str, on: bool) -> AppResult<PeerView> {
        self.shared.db.set_peer_send_jobs(id, on).await?;
        self.view(id).await
    }

    /// Set what this PC lends to the linked PC `id` (None fields = the
    /// defaults). Memory limits and pause apply at once to new allocations
    /// and sessions; memory already lent is never taken back; a thread
    /// change applies from its next session.
    pub async fn set_peer_share(&self, id: &str, share: PeerShare) -> AppResult<PeerView> {
        validate_threads(share.threads)?;
        let row = self.shared.db.get_peer(id).await?;
        let live = self.shared.live.read().unwrap().clone();
        let devs = if live.devices.is_empty() { &live.available_devices } else { &live.devices };
        let total = |k: DevKind| devs.iter().filter(|d| DevKind::of(&d.name) == k).map(|d| d.total_mib).sum::<u64>();
        for (v, k, what) in [(share.gpu_mib, DevKind::Gpu, "GPU memory"), (share.ram_mib, DevKind::Ram, "RAM")] {
            if let Some(v) = v
                && total(k) > 0
                && v > total(k)
            {
                return Err(AppError::InvalidInput(format!("the {what} limit is more than this PC has ({:.1} GB)", total(k) as f64 / 1024.0)));
            }
        }
        self.shared.db.set_peer_share(id, &share).await?;
        if let Ok(key) = key_from_b64(&row.public_key) {
            self.shared.shares.write().unwrap().insert(key, share.clone());
            let kinds = self.shared.device_kinds();
            let budget = self.shared.budgets.lock().unwrap().get(&key).cloned();
            if let Some(b) = budget {
                let mut caps = self.shared.resolve_limits(&share, &kinds);
                // Threads change only with a new session (rpc-server restarts then).
                caps.threads = b.caps().threads;
                b.set_caps(caps);
            }
        }
        // Linked PCs see their new limits at once.
        if let Some(s) = &self.run.lock().await.server {
            s.notify_changed();
        }
        self.view(id).await
    }

    /// What this PC lends to `p` and what it uses now.
    fn lend_view(&self, p: &PeerRow) -> LendView {
        let stored = p.share();
        let live = self.shared.live.read().unwrap().clone();
        let devs = if live.devices.is_empty() { &live.available_devices } else { &live.devices };
        let kinds: Vec<DevKind> = devs.iter().map(|d| DevKind::of(&d.name)).collect();
        let key = key_from_b64(&p.public_key).ok();
        let limits = match &key {
            Some(k) => self.shared.limits_for(k, &kinds),
            None => self.shared.resolve_limits(&stored, &kinds),
        };
        let budget = key.and_then(|k| self.shared.budgets.lock().unwrap().get(&k).cloned());
        let (refused, refused_bytes, last) = budget.as_ref().map(|b| b.refusals()).unwrap_or((0, 0, None));
        LendView {
            limits,
            stored,
            used_gpu_mib: budget.as_ref().map_or(0, |b| b.used_bytes(DevKind::Gpu) >> 20),
            used_ram_mib: budget.as_ref().map_or(0, |b| b.used_bytes(DevKind::Ram) >> 20),
            shares_gpu: kinds.contains(&DevKind::Gpu),
            shares_ram: kinds.contains(&DevKind::Ram),
            refused,
            refused_bytes,
            last_refused_secs_ago: last.map(|(_, t)| t.elapsed().map(|d| d.as_secs()).unwrap_or(0)),
        }
    }

    /// Restart the helper's rpc-server with `threads` (a linked PC with
    /// another thread limit starts a session; none is open).
    pub async fn restart_rpc(&self, st: &SharedState, threads: u32) -> AppResult<()> {
        let mut run = self.run.lock().await;
        if run.rpc.is_none() {
            return Err(AppError::InvalidInput("\"Let this PC help\" is off".into()));
        }
        let settings = HelperSettings { threads: Some(threads), ..self.settings.lock().unwrap().clone() };
        stop_rpc(&mut run, &self.shared);
        match start_rpc_server(st, &settings, &self.shared).await {
            Ok(c) => {
                run.rpc = Some(c);
                if let Some(s) = &run.server {
                    s.notify_changed();
                }
                Ok(())
            }
            Err(e) => {
                self.shared.live.write().unwrap().error = Some(e.to_string());
                if let Some(s) = &run.server {
                    s.notify_changed();
                }
                Err(e)
            }
        }
    }

    /// What linked PCs with "Split model layers" on lend this PC now, from
    /// their last info (no process started): for the load panel's Auto
    /// context. Older helpers: their single cap, spread over their devices.
    pub async fn linked_capacity(&self) -> Vec<LinkedShare> {
        let rows: Vec<PeerRow> = match self.shared.db.list_peers().await {
            Ok(r) => r.into_iter().filter(|p| p.split_layers != 0 && dialable(p)).collect(),
            Err(_) => return Vec::new(),
        };
        let out = self.out.lock().await;
        let mut v = Vec::new();
        for p in rows {
            let Some(info) = out.get(&p.id).filter(|o| !o.state.is_closed()).map(|o| o.state.info()) else { continue };
            if !info.helping {
                continue;
            }
            v.extend(shares_from_info(&p.name, "", &info));
        }
        v
    }

    /// Forget the helpers' earlier refusals (a load with linked PCs starts).
    async fn clear_refusals(&self) {
        for o in self.out.lock().await.values() {
            o.state.clear_refusal();
        }
    }

    /// The helpers' refusals since the load started, in the user's words.
    pub async fn refusal_notes(&self) -> Vec<String> {
        let rows = self.shared.db.list_peers().await.unwrap_or_default();
        let out = self.out.lock().await;
        rows.iter().filter_map(|p| out.get(&p.id).and_then(|o| o.state.refusal()).map(|r| refusal_message(&p.name, &r))).collect()
    }

    /// Forget a linked PC here and (best effort) tell it to forget this one.
    pub async fn unlink(&self, id: &str) -> AppResult<bool> {
        let row = self.shared.db.get_peer(id).await?;
        let key = key_from_b64(&row.public_key).ok();
        if let Some(k) = &key {
            self.shared.pinned.write().unwrap().remove(k);
        }
        let existing = self.out.lock().await.remove(id);
        let told = match (existing, key, row.last_address.as_deref()) {
            (Some(o), _, _) => {
                let ctl = o.control.clone();
                drop(o);
                tokio::time::timeout(Duration::from_secs(5), async { ctl.lock().await.unlink().await }).await.is_ok_and(|r| r.is_ok())
            }
            (None, Some(k), Some(addr)) => tokio::time::timeout(Duration::from_secs(6), async {
                match Control::connect(addr, &self.identity, k).await {
                    Ok(mut c) => c.unlink().await.is_ok(),
                    Err(_) => false,
                }
            })
            .await
            .unwrap_or(false),
            _ => false,
        };
        // Whatever this PC cached for that PC goes with the link.
        if let Some(c) = self.shared.cache() {
            c.remove_peer(&row.public_key);
        }
        self.shared.db.delete_peer(id).await?;
        Ok(told)
    }

    /// Linked PCs for a split load with `runtime`: connected, helping, same
    /// llama.cpp build, each with a local bridge endpoint for `--rpc`.
    pub async fn split_peers(&self, runtime: &inference::InstalledRuntime) -> AppResult<Vec<SplitPeer>> {
        let rows: Vec<PeerRow> = self.shared.db.list_peers().await?.into_iter().filter(|p| p.split_layers != 0 && dialable(p)).collect();
        if rows.is_empty() {
            return Err(AppError::InvalidInput(
                "no linked PC has \"Split model layers with this PC\" on (only PCs this PC linked to with their code can help it). Turn it on in Linked PCs, or turn off \"Use linked PCs\"".into(),
            ));
        }
        let compress = self.settings.lock().unwrap().compress;
        // The load needs the link: no bandwidth test competing with it.
        self.cancel_bw_tests();
        let mut ready = Vec::new();
        let mut problems = Vec::new();
        for p in rows {
            match self.ready_peer(&p, runtime, compress).await {
                Ok(sp) => ready.push(sp),
                Err(msg) => problems.push(msg),
            }
        }
        if !problems.is_empty() {
            return Err(AppError::InvalidInput(format!(
                "can't split with linked PCs: {}. Or turn off \"Split model layers with this PC\" for that PC",
                problems.join("; ")
            )));
        }
        Ok(ready)
    }

    /// One linked PC, ready to split with, or why not (in the user's words).
    async fn ready_peer(&self, p: &PeerRow, runtime: &inference::InstalledRuntime, compress: bool) -> Result<SplitPeer, String> {
        let not_helping = || format!("\"{}\" isn't helping right now: turn on \"Let this PC help\" there", p.name);
        let explain = |e: &LinkError| match e {
            LinkError::NotListening(_) | LinkError::ClosedEarly(_) | LinkError::Disconnected(_) | LinkError::HelperOff | LinkError::Io(_) => not_helping(),
            LinkError::Unreachable(_) | LinkError::Timeout => format!("\"{}\" isn't reachable: {e}", p.name),
            other => format!("\"{}\": {other}", p.name),
        };
        // At most one (re)connect: each is bounded by CONNECT_LIMIT.
        let mut connected = false;
        for _ in 0..3 {
            let ctl = self.out.lock().await.get(&p.id).filter(|o| !o.state.is_closed()).map(|o| o.control.clone());
            let Some(ctl) = ctl else {
                if connected {
                    return Err(not_helping());
                }
                // No open control channel (never connected, or it closed): connect.
                self.connect_inner(&p.id).await.map_err(|e| explain(&e))?;
                self.cancel_bw_tests();
                connected = true;
                continue;
            };
            let refreshed = tokio::time::timeout(Duration::from_secs(10), async { ctl.lock().await.refresh_info().await }).await.unwrap_or(Err(LinkError::Timeout));
            let info = match refreshed {
                Ok(i) => i,
                Err(e) if !connected && e.is_connection_lost() => {
                    self.out.lock().await.remove(&p.id);
                    continue;
                }
                Err(e) => return Err(explain(&e)),
            };
            if !info.helping {
                return Err(not_helping());
            }
            match info.runtime_tag.as_deref() {
                Some(t) if t == runtime.tag => {}
                Some(t) => return Err(linking::helper::version_message(&runtime.tag, t, &p.name)),
                None => return Err(format!("\"{}\" didn't say which llama.cpp build it runs; update Local AI Studio there", p.name)),
            }
            let mut out = self.out.lock().await;
            let Some(o) = out.get_mut(&p.id) else { continue };
            if o.bridge.is_none() {
                let b = RpcBridge::start(o.address.clone(), self.identity.clone(), o.key, Some(runtime.tag.clone()), compress)
                    .await
                    .map_err(|e| format!("could not open a local endpoint for \"{}\": {e}", p.name))?;
                o.bridge = Some(Arc::new(b));
            }
            let ep = o.bridge.as_ref().unwrap().endpoint().to_string();
            return Ok(SplitPeer { id: p.id.clone(), name: p.name.clone(), endpoint: ep, ram_cap_mib: info.ram_cap_mib, share: info.share, enforced: info.share_enforced, kinds: info.devices.iter().map(|d| DevKind::of(&d.name)).collect() });
        }
        Err(not_helping())
    }

    /// Key for measured speeds: this model alone, or split with this set of PCs.
    fn speed_key(&self, model_id: &str, split: bool) -> String {
        if split {
            format!("{model_id}|split:{}", self.last_split_ids.lock().unwrap().join(","))
        } else {
            format!("{model_id}|alone")
        }
    }

    async fn record_speed(&self, key: &str, tok_s: f64) {
        let snapshot = {
            let mut m = self.speeds.lock().unwrap();
            let v = m.entry(key.to_string()).or_default();
            v.push(tok_s);
            if v.len() > SPEED_SAMPLES {
                v.remove(0);
            }
            m.clone()
        };
        if let Err(e) = self.shared.db.set_setting(SET_SPEEDS, &snapshot).await {
            tracing::warn!("could not save measured speed: {e}");
        }
    }

    /// Record the generation speed of every reply (any path: chat, tasks,
    /// local API) from llama-server's own timing line, per model and per
    /// set of linked PCs, for "this PC alone" vs "with linked PCs".
    pub fn watch_engine(self: &Arc<Self>, mut rx: tokio::sync::broadcast::Receiver<inference::EngineEvent>) {
        let me = self.clone();
        tauri::async_runtime::spawn(async move {
            use inference::{EngineEvent, EngineState};
            let mut current: Option<String> = None;
            loop {
                match rx.recv().await {
                    Ok(EngineEvent::Status { status }) => {
                        current = match (&status.model_id, status.state) {
                            (Some(m), EngineState::Loaded) => Some(me.speed_key(m, status.config.as_ref().is_some_and(|c| !c.rpc.is_empty()))),
                            _ => None,
                        };
                    }
                    Ok(EngineEvent::Log { line, .. }) => {
                        if let (Some(k), Some(tps)) = (&current, split::parse_eval_line(&line)) {
                            me.record_speed(k, tps).await;
                        }
                    }
                    Ok(_) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        });
    }

    /// Estimates and measured speeds for the load panel.
    pub async fn split_preview(&self, model_id: Option<&str>, local: &LocalHw) -> AppResult<SplitPreview> {
        let rows: Vec<PeerRow> = self.shared.db.list_peers().await?.into_iter().filter(|p| p.split_layers != 0 && dialable(p)).collect();
        let mut ids: Vec<String> = rows.iter().map(|p| p.id.clone()).collect();
        ids.sort();
        let views = self.peer_views(rows).await;
        let (alone, split_m, split_n) = match model_id {
            Some(m) => {
                let speeds = self.speeds.lock().unwrap().clone();
                let med = |v: &Vec<f64>| split::median(&mut v.clone());
                let alone = speeds.get(&format!("{m}|alone")).and_then(med);
                // This set of PCs first, else any split of this model.
                let exact = speeds.get(&format!("{m}|split:{}", ids.join(",")));
                let any = speeds.iter().filter(|(k, _)| k.starts_with(&format!("{m}|split:"))).max_by_key(|(_, v)| v.len()).map(|(_, v)| v);
                let sp = exact.or(any);
                let (mut alone, mut split_m, mut split_n) = (alone, sp.and_then(med), sp.map(|v| v.len()).unwrap_or(0));
                // Older loads: chat replies stored in the database.
                if alone.is_none() || split_m.is_none() {
                    let s = self.shared.db.model_instance_speeds(m, 50).await?;
                    if alone.is_none() {
                        alone = s.iter().find(|i| !i.split).map(|i| i.median_tok_s);
                    }
                    if split_m.is_none()
                        && let Some(i) = s.iter().find(|i| i.split)
                    {
                        (split_m, split_n) = (Some(i.median_tok_s), i.replies);
                    }
                }
                (alone, split_m, split_n)
            }
            None => (None, None, 0),
        };
        let peers: Vec<SplitEstimate> = views
            .iter()
            .map(|v| SplitEstimate {
                peer_id: v.id.clone(),
                name: v.name.clone(),
                online: v.online,
                rtt_ms: v.rtt_ms,
                added_ms_per_token: v.rtt_ms.map(split::added_ms_per_token),
            })
            .collect();
        let model_bytes = match model_id {
            Some(m) => self.shared.db.get_model(m).await.ok().map(|r| r.size as u64).filter(|b| *b > 0),
            None => None,
        };
        let (range, est_basis, est_missing) = match model_bytes {
            Some(b) if !views.is_empty() => self.estimate(b, local, &views, alone).await,
            Some(_) => (None, Vec::new(), None),
            None => (None, Vec::new(), model_id.map(|_| "model size unknown".to_string())),
        };
        let suggest_off = matches!((alone, split_m), (Some(a), Some(s)) if split::split_is_slower(a, s));
        let message = suggest_off.then(|| {
            format!(
                "With linked PCs this model ran at {:.1} tokens/s, slower than this PC alone ({:.1} tokens/s). Linked PCs can still give a longer context. If the model and the context you need fit on this PC, turn off \"Use linked PCs\" here (or \"Split model layers\" on the Linked PCs page).",
                split_m.unwrap_or(0.0),
                alone.unwrap_or(0.0)
            )
        });
        Ok(SplitPreview {
            peers,
            alone_tok_s: alone,
            est_low_tok_s: range.map(|r| r.0),
            est_high_tok_s: range.map(|r| r.1),
            est_basis,
            est_missing,
            split_tok_s: split_m,
            split_replies: split_n,
            suggest_off,
            message,
        })
    }

    /// Per-device estimate: this PC's GPU (rated bandwidth from its name,
    /// else from its measured speed alone), each linked PC's devices (GPU by
    /// name, CPU by the RAM bandwidth it measured), anything left over on
    /// this PC's CPU, and one round trip per linked PC per token.
    async fn estimate(&self, model_bytes: u64, local: &LocalHw, peers: &[PeerView], alone: Option<f64>) -> (Option<(f64, f64)>, Vec<String>, Option<String>) {
        const MARGIN: u64 = 1 << 30;
        let mut caps = Vec::new();
        let mut basis = Vec::new();
        let gb = |b: u64| b as f64 / 1e9;
        // Weights that fit on this PC's GPU alone, and the share of a GPU's
        // memory that is weights (the rest is context and working buffers),
        // applied to the linked PCs' GPUs too.
        let alone_gpu_bytes = local.alone_gpu_layers.map(|f| (model_bytes as f64 * f.clamp(0.0, 1.0)) as u64);
        let weight_share = match (alone_gpu_bytes, local.total_vram) {
            (Some(b), Some(t)) if t > MARGIN && b < model_bytes => (b as f64 / (t - MARGIN) as f64).clamp(0.2, 1.0),
            _ => DEFAULT_WEIGHT_SHARE,
        };
        let scaled = |b: u64| (b as f64 * weight_share) as u64;
        let mut alone_caps = Vec::new();
        if let (Some(name), Some(free)) = (&local.gpu_name, local.free_vram) {
            let short = name.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ").to_string();
            let (gbs, measured) = match (split::gpu_bandwidth_gbs(name), alone) {
                (Some(r), _) => (r, false),
                // Unknown GPU: what it achieved alone, as an effective bandwidth.
                (None, Some(a)) => (gb(model_bytes) * a / ((split::GPU_EFFICIENCY.0 + split::GPU_EFFICIENCY.1) / 2.0), true),
                (None, None) => (split::DEFAULT_GPU_GBS, false),
            };
            // What fit alone (also right while the model is loaded and
            // holding the memory), else a share of the free memory.
            let fits = alone_gpu_bytes.unwrap_or_else(|| scaled(free.saturating_sub(MARGIN)));
            let c = split::Capacity { label: format!("{short} on this PC"), free: fits, gb_s: gbs, cpu: false, measured };
            alone_caps.push(c.clone());
            caps.push(c);
        }
        let mut rtt_total = 0.0;
        let out = self.out.lock().await;
        for v in peers {
            let Some(info) = out.get(&v.id).filter(|o| !o.state.is_closed()).map(|o| o.state.info()) else {
                return (None, Vec::new(), Some(format!("{} is offline: connect it on the Linked PCs page to estimate", v.name)));
            };
            if info.devices.is_empty() {
                return (None, Vec::new(), Some(format!("{} hasn't reported its devices (turn on \"Let this PC help\" there)", v.name)));
            }
            rtt_total += v.rtt_ms.unwrap_or(1.0);
            let total_free: u64 = info.devices.iter().map(|d| (d.free_mib << 20).saturating_sub(512 << 20)).sum();
            let cap = info.ram_cap_mib.map(|c| c << 20).unwrap_or(u64::MAX);
            for d in &info.devices {
                let cpu = d.name.to_ascii_uppercase().starts_with("CPU");
                let (gbs, measured) = if cpu {
                    info.ram_bandwidth_gbs.map(|b| (b, true)).unwrap_or((split::DEFAULT_RAM_GBS, false))
                } else {
                    (split::gpu_bandwidth_gbs(&d.description).unwrap_or(split::DEFAULT_GPU_GBS), false)
                };
                let mut free = (d.free_mib << 20).saturating_sub(512 << 20);
                if total_free > cap && total_free > 0 {
                    free = (free as u128 * cap as u128 / total_free as u128) as u64;
                }
                if !cpu {
                    free = scaled(free);
                }
                caps.push(split::Capacity { label: format!("{} on {}", d.name, v.name), free, gb_s: gbs, cpu, measured });
            }
        }
        drop(out);
        let overflow = split::Capacity { label: "CPU on this PC".into(), free: 0, gb_s: ram_gbs().await, cpu: true, measured: true };
        let devs = split::place_weights(model_bytes, &caps, &overflow);
        for d in &devs {
            basis.push(format!(
                "{}: {:.1} GB at {:.0} GB/s ({})",
                d.label,
                gb(d.bytes),
                d.gb_s,
                if d.measured { "measured" } else if d.cpu { "assumed" } else { "rated" }
            ));
        }
        basis.push(format!("network: {:.1} ms round trip per token", rtt_total));
        // Scaled by this PC's measured speed alone when there is one.
        let range = match alone {
            Some(a) if !alone_caps.is_empty() => {
                basis.push(format!("scaled by this PC's measured {a:.1} tokens/s alone"));
                split::calibrated_range(&devs, &split::place_weights(model_bytes, &alone_caps, &overflow), a, rtt_total)
            }
            _ => None,
        };
        (range.or_else(|| split::estimate_range(&devs, rtt_total)), basis, None)
    }

    /// App exit: stop rpc-server, the listener, bridges and the UPnP mapping.
    pub async fn shutdown(&self) {
        self.codes.cancel();
        let mut run = self.run.lock().await;
        stop_rpc(&mut run, &self.shared);
        self.maybe_stop_listener(&mut run).await;
        drop(run);
        let peers: Vec<OutPeer> = self.out.lock().await.drain().map(|(_, o)| o).collect();
        for o in peers {
            if let Some(b) = &o.bridge {
                b.stop();
            }
        }
    }
}

fn stop_rpc(run: &mut HelperRun, shared: &Shared) {
    if let Some(c) = run.rpc.take() {
        c.stopping.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = c.job.terminate();
        tracing::info!("linking: rpc-server stopped");
    }
    let mut l = shared.live.write().unwrap();
    l.helping = false;
    l.rpc_endpoint = None;
    l.devices.clear();
}

fn push_log(log: &Mutex<VecDeque<String>>, line: String) {
    let mut l = log.lock().unwrap();
    if l.len() >= LOG_LINES {
        l.pop_front();
    }
    l.push_back(line);
}

async fn start_rpc_server(st: &SharedState, s: &HelperSettings, shared: &Arc<Shared>) -> AppResult<RpcChild> {
    let runtime = crate::commands::models::runtime_or_default(st, s.runtime_id.as_deref())
        .await
        .ok_or_else(|| AppError::NotFound("no llama.cpp runtime is installed yet; install one on the Runtimes page".into()))?;
    let exe = runtime.server_path.with_file_name(RPC_SERVER_EXE);
    if !exe.is_file() {
        return Err(AppError::NotFound(format!(
            "the {} runtime has no {RPC_SERVER_EXE}, which lending this PC needs. Install or update an official llama.cpp runtime on the Runtimes page (they include it)",
            runtime.id
        )));
    }
    inference::vcrt::ensure_next_to(&exe);
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0")?;
        l.local_addr()?.port()
    };
    let mut cmd = tokio::process::Command::new(&exe);
    // Loopback only: rpc-server has no authentication; the tunnel is the only way in.
    cmd.args(["-H", "127.0.0.1", "-p", &port.to_string()]);
    if !s.devices.is_empty() {
        cmd.args(["-d", &s.devices.join(",")]);
    }
    // Always pass -t: rpc-server's own default is half the logical
    // processors (2 on a 4-vCPU VM), which halved prompt speed on the VM
    // test (71.5 t/s with -t 2 vs 121.2 with -t 4, 2026-11-23).
    let threads = s.threads.unwrap_or_else(local_default_rpc_threads);
    cmd.args(["-t", &threads.to_string()]);
    shared.rpc_threads.store(threads, Ordering::Relaxed);
    // "Speed up reloads": rpc-server's disk cache goes to the app's own
    // folder (`LLAMA_CACHE` is where it puts `rpc\<hash>` files, checked on
    // b12250) and is capped. Off: no `-c`, so nothing is written to disk.
    shared.cache_on.store(s.cache, Ordering::Relaxed);
    shared.cache_cap.store(s.cache_cap_bytes(), Ordering::Relaxed);
    if s.cache {
        let store = shared.cache().unwrap_or_else(|| Arc::new(CacheStore::new(st.paths.data_dir.join("linking-cache"))));
        let _ = store.ensure_dir();
        store.enforce_cap(s.cache_cap_bytes());
        cmd.arg("-c").env("LLAMA_CACHE", store.llama_cache_dir());
    }
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        // Normal CPU, memory and I/O priority even when this app runs lower
        // (see `inference::job::normal_child_priority`).
        .creation_flags(inference::job::CREATE_NO_WINDOW | inference::job::NORMAL_PRIORITY_CLASS);
    shared.rpc_log.lock().unwrap().clear();
    let mut child = cmd.spawn().map_err(|e| AppError::Runtime(format!("could not start {RPC_SERVER_EXE}: {e}")))?;
    if let Err(e) = inference::job::normal_child_priority(&child) {
        tracing::warn!("could not give {RPC_SERVER_EXE} normal memory and I/O priority: {e}");
    }
    let job = inference::job::contain(&child).map_err(|e| AppError::Runtime(format!("could not contain {RPC_SERVER_EXE} in a job object: {e}")))?;
    for (out, name) in [
        (child.stdout.take().map(|o| Box::new(o) as Box<dyn tokio::io::AsyncRead + Unpin + Send>), "out"),
        (child.stderr.take().map(|o| Box::new(o) as Box<dyn tokio::io::AsyncRead + Unpin + Send>), "err"),
    ] {
        let Some(out) = out else { continue };
        let log = shared.rpc_log.clone();
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                tracing::debug!(target: "rpc_server", "[{name}] {l}");
                push_log(&log, l);
            }
        });
    }
    // Ready when the port answers; fail clearly if the process exits.
    let ep: SocketAddr = (std::net::Ipv4Addr::LOCALHOST, port).into();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait()? {
            let code = status.code().unwrap_or(-1);
            if code == inference::vcrt::STATUS_DLL_NOT_FOUND {
                let (msg, hint) = inference::vcrt::missing_dll_message();
                return Err(AppError::Runtime(format!("{RPC_SERVER_EXE}: {msg}. {hint}")));
            }
            let tail: Vec<String> = shared.rpc_log.lock().unwrap().iter().rev().take(5).rev().cloned().collect();
            return Err(AppError::Runtime(format!("{RPC_SERVER_EXE} exited with code {code}: {}", tail.join(" | "))));
        }
        if tokio::net::TcpStream::connect(ep).await.is_ok() {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            let _ = job.terminate();
            return Err(AppError::Runtime(format!("{RPC_SERVER_EXE} did not start listening within 30 s")));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // Give the banner a moment to print the device list.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let text = shared.rpc_log.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
    let devices: Vec<DeviceInfo> = split::parse_devices(&text)
        .into_iter()
        .map(|d| DeviceInfo { name: d.name, description: d.description, total_mib: d.total_mib, free_mib: d.free_mib })
        .collect();
    {
        let mut l = shared.live.write().unwrap();
        l.helping = true;
        l.rpc_endpoint = Some(ep.to_string());
        l.runtime_id = Some(runtime.id.clone());
        l.runtime_tag = Some(runtime.tag.clone());
        l.runtime_variant = Some(runtime.variant.clone());
        // Older main PCs read one total cap: the defaults' sum.
        l.ram_cap_mib = match (s.share_gpu_mib, s.share_ram_mib) {
            (None, None) => None,
            (g, r) => Some(g.unwrap_or(0).saturating_add(r.unwrap_or(0))),
        };
        l.error = None;
        l.ram_bandwidth_gbs = RAM_GBS.get().copied();
        if s.devices.is_empty() || l.available_devices.is_empty() {
            l.available_devices = devices.clone();
        }
        l.devices = devices;
    }
    tracing::info!("linking: {RPC_SERVER_EXE} ({}) listening on {ep}", runtime.id);
    // RAM bandwidth for the other PC's speed estimate (once per run).
    if RAM_GBS.get().is_none() {
        let sh = shared.clone();
        tokio::spawn(async move {
            let v = ram_gbs().await;
            sh.live.write().unwrap().ram_bandwidth_gbs = Some(v);
        });
    }
    let stopping = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (stop2, sh) = (stopping.clone(), shared.clone());
    tokio::spawn(async move {
        let status = child.wait().await;
        if !stop2.load(std::sync::atomic::Ordering::SeqCst) {
            let msg = format!("{RPC_SERVER_EXE} stopped unexpectedly ({status:?}); turn \"Let this PC help\" off and on again");
            tracing::warn!("{msg}");
            let mut l = sh.live.write().unwrap();
            l.helping = false;
            l.rpc_endpoint = None;
            l.error = Some(msg);
        }
    });
    Ok(RpcChild { job, stopping })
}

/// CPU threads a linked PC gets by default (the owner's answer,
/// 2026-11-23): every core but 2, at least 1, so lending never takes the
/// whole CPU. Cores are the physical cores (the logical count without a
/// topology); on a hybrid CPU also at most the P-cores plus 4, the
/// generation sweet spot measured for llama-server (`hybrid_cpu_threads`).
/// Per-PC settings may raise it up to every logical processor. On small CPUs
/// leaving 2 free costs speed on the layers the helper's CPU holds: on a
/// 4-vCPU VM, -t 2 gave 71.5 t/s prompt vs 121.2 with -t 4.
pub fn default_rpc_threads(topo: Option<system_monitor::CoreTopology>, logical: usize) -> u32 {
    let cores = match topo {
        Some(t) if t.physical_cores() > 0 => t.physical_cores(),
        _ => logical,
    };
    let mut n = cores.saturating_sub(2);
    if let Some(t) = topo
        && t.is_hybrid()
    {
        n = n.min(t.performance_cores + 4);
    }
    n.max(1) as u32
}

/// Devices llama-server sees with these RPC endpoints (`--list-devices`).
/// Runs the backend binary only to list devices; no model is loaded.
pub async fn list_devices(runtime: &inference::InstalledRuntime, endpoints: &[String]) -> AppResult<Vec<ListedDevice>> {
    inference::vcrt::ensure_next_to(&runtime.server_path);
    let mut cmd = tokio::process::Command::new(&runtime.server_path);
    cmd.args(["--rpc", &endpoints.join(","), "--list-devices"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .creation_flags(inference::job::CREATE_NO_WINDOW);
    let out = tokio::time::timeout(Duration::from_secs(60), cmd.output())
        .await
        .map_err(|_| AppError::Network("listing the linked PCs' devices took over 60 s: the link is stalled".into()))??;
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    Ok(split::parse_devices(&text))
}

/// llama.cpp's memory breakdown of a split load before it starts:
/// `llama-fit-params -v` with the load's model, devices, layers and context
/// allocates nothing (no weights are sent) and prints each device's model,
/// context and compute buffer. None when the runtime has no
/// `llama-fit-params` or it printed no breakdown.
pub async fn preflight(runtime: &inference::InstalledRuntime, model: &std::path::Path, cfg: &inference::LoadConfig) -> Option<Vec<split::DeviceUse>> {
    let exe = runtime.server_path.with_file_name("llama-fit-params.exe");
    if !exe.is_file() {
        return None;
    }
    inference::vcrt::ensure_next_to(&exe);
    let mut a: Vec<String> = vec!["-m".into(), model.to_string_lossy().into_owned(), "-v".into()];
    let mut push = |k: &str, v: String| {
        a.push(k.into());
        a.push(v);
    };
    push("--ctx-size", cfg.n_ctx.to_string());
    match cfg.gpu_layers {
        inference::GpuLayers::Count(n) => push("--n-gpu-layers", n.to_string()),
        _ => push("--n-gpu-layers", "all".into()),
    }
    if !cfg.rpc.is_empty() {
        push("--rpc", cfg.rpc.join(","));
    }
    if let Some(ts) = &cfg.tensor_split {
        push("--tensor-split", ts.clone());
    }
    if let Some(b) = cfg.batch_size {
        push("--batch-size", b.to_string());
    }
    if let Some(b) = cfg.ubatch_size {
        push("--ubatch-size", b.to_string());
    }
    let fa = match cfg.flash_attention {
        inference::FlashAttention::Auto => "auto",
        inference::FlashAttention::On => "on",
        inference::FlashAttention::Off => "off",
    };
    push("--flash-attn", fa.into());
    push("--cache-type-k", cfg.cache_type_k.clone());
    push("--cache-type-v", cfg.cache_type_v.clone());
    if let Some(n) = cfg.n_cpu_moe {
        push("--n-cpu-moe", n.to_string());
    }
    push("--parallel", cfg.parallel.to_string());
    if !cfg.kv_offload {
        a.push("--no-kv-offload".into());
    }
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.args(&a)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .creation_flags(inference::job::CREATE_NO_WINDOW);
    let out = tokio::time::timeout(Duration::from_secs(60), cmd.output()).await.ok()?.ok()?;
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let rows = split::parse_breakdown(&text);
    (!rows.is_empty()).then_some(rows)
}

/// Resolve "Use linked PCs" for one load: `--rpc` endpoints, a first
/// `--tensor-split` by free memory (replaced by [`net_split`] once the
/// context is known), and what each linked device lends for the Auto
/// context. Returns the config, a note and the linked devices.
pub async fn resolve_linked(
    st: &SharedState,
    runtime: &inference::InstalledRuntime,
    mut cfg: inference::LoadConfig,
) -> AppResult<(inference::LoadConfig, Option<String>, Option<LinkedLoad>)> {
    cfg.rpc.clear();
    cfg.tensor_split = None;
    st.linking.load_probes.lock().unwrap().clear();
    if !cfg.use_linked_pcs {
        return Ok((cfg, None, None));
    }
    st.linking.clear_refusals().await;
    if !runtime.capabilities.has("--rpc") {
        return Err(AppError::InvalidInput(format!("the {} runtime has no --rpc support; install an official llama.cpp runtime", runtime.id)));
    }
    if cfg.gpu_layers == inference::GpuLayers::CpuOnly {
        return Err(AppError::InvalidInput("\"Use linked PCs\" needs GPU layers (not CPU only)".into()));
    }
    let peers = st.linking.split_peers(runtime).await?;
    {
        let mut ids: Vec<String> = peers.iter().map(|p| p.id.clone()).collect();
        ids.sort();
        *st.linking.last_split_ids.lock().unwrap() = ids;
    }
    let eps: Vec<String> = peers.iter().map(|p| p.endpoint.clone()).collect();
    let devices = list_devices(runtime, &eps).await?;
    // New helpers lower the free memory they report to their limits
    // themselves; an older helper's single cap is applied here.
    let caps: Vec<(String, Option<u64>)> = peers.iter().map(|p| (p.endpoint.clone(), if p.enforced { None } else { p.ram_cap_mib })).collect();
    for p in &peers {
        if !devices.iter().any(|d| split::device_on_endpoint(d, &p.endpoint)) {
            // A helper that can't enforce its limits with this protocol says why.
            let refusal = st.linking.refusal_notes().await;
            if !refusal.is_empty() {
                return Err(AppError::Network(refusal.join(" ")));
            }
            let why = st.linking.bridge_error(&p.id).await.map(|e| format!(" (link: {e})")).unwrap_or_default();
            return Err(AppError::Network(format!(
                "llama-server couldn't reach \"{}\" through the link: its devices didn't show up{why}. Check that \"Let this PC help\" is on there and try again",
                p.name
            )));
        }
    }
    let plan = split::tensor_split(&devices, &caps).ok_or_else(|| AppError::InvalidInput("no device reported free memory for the split".into()))?;
    // What each linked device lends: its free memory as reported through
    // the helper's limit (the i-th RPC device of an endpoint is the i-th
    // device of that helper).
    let mut shares = Vec::new();
    for p in &peers {
        let info_devs: Vec<&ListedDevice> = devices.iter().filter(|d| split::device_on_endpoint(d, &p.endpoint)).collect();
        let total: u64 = info_devs.iter().map(|d| d.free_mib << 20).sum();
        let cap = if p.enforced { None } else { p.ram_cap_mib.map(|c| c << 20) };
        for (i, d) in info_devs.iter().enumerate() {
            let kind = p.kinds.get(i).copied().unwrap_or(DevKind::Gpu);
            let free = d.free_mib << 20;
            let bytes = match cap {
                Some(c) if total > c && total > 0 => (free as u128 * c as u128 / total as u128) as u64,
                _ => free,
            };
            shares.push(LinkedShare { peer: p.name.clone(), endpoint: p.endpoint.clone(), device: d.name.clone(), bytes, cpu: kind == DevKind::Ram, cap_bytes: p.share.map(|s| s.cap_bytes(kind)).or(cap) });
        }
    }
    let names: HashMap<String, String> = peers.iter().map(|p| (p.endpoint.clone(), p.name.clone())).collect();
    let mut note_parts: Vec<String> = Vec::new();
    for p in &peers {
        let lent: u64 = shares.iter().filter(|s| s.endpoint == p.endpoint).map(|s| s.bytes).sum();
        let mut line = format!("\"{}\" lends {:.1} GB", p.name, lent as f64 / 1e9);
        match p.share {
            Some(s) if p.enforced => {
                // Only the kinds of memory it shares (a CPU-only PC has no GPU limit to show).
                let has = |k: DevKind| p.kinds.contains(&k);
                let parts: Vec<String> = [(has(DevKind::Gpu), s.gpu_mib, "GPU memory"), (has(DevKind::Ram), s.ram_mib, "RAM")]
                    .into_iter()
                    .filter(|(on, _, _)| *on)
                    .map(|(_, mib, what)| format!("{:.1} GB {what}", (mib << 20) as f64 / 1e9))
                    .collect();
                if !parts.is_empty() {
                    line.push_str(&format!(" (its owner allows up to {})", parts.join(", ")));
                }
            }
            _ => line.push_str(" (it runs an older version: its limit isn't enforced there)"),
        }
        note_parts.push(line);
    }
    let note = format!("Linked PCs: {}.", note_parts.join("; "));
    // Each linked PC's progress row: tensor data really sent and what its
    // cache supplied, over its real share. The share is not guessed from the
    // split: the backend logs it ("RPCn model buffer size") once the layers
    // are assigned, before any weight is sent.
    let probes = {
        let mut v = Vec::new();
        for p in &peers {
            if let Some(c) = st.linking.bridge_counters(&p.id).await {
                v.push(inference::PeerProbe {
                    id: p.id.clone(),
                    name: p.name.clone(),
                    endpoint: p.endpoint.clone(),
                    expected_bytes: 0,
                    sent: c.tensor_bytes.clone(),
                    cached: c.cached_bytes.clone(),
                });
            }
        }
        v
    };
    *st.linking.load_probes.lock().unwrap() = probes;
    cfg.rpc = eps;
    cfg.tensor_split = Some(plan.tensor_split);
    Ok((cfg, Some(note), Some(LinkedLoad { devices, caps, names, shares })))
}

/// Why a linked PC holds no layers of this load, for the load panel and the
/// engine status: it lends less than a device's working buffer (a device
/// holds layers only with 90% of what it lends above its compute buffer, as
/// [`split::tensor_split_net`] and [`model_registry::estimate::linked_net`]
/// count it). `lent` is what it lends now, `cap` its owner's limit for this
/// PC when known.
pub fn left_out_note(name: &str, lent: u64, cap: Option<u64>, compute_buffer_bytes: u64) -> String {
    let gb = |b: u64| b as f64 / 1e9;
    // Smallest amount that holds anything, shown at least as 0.1 GB.
    let need = ((compute_buffer_bytes as f64 / 0.9) as u64).max(100_000_000);
    match cap {
        Some(c) if c < need => format!(
            "\"{name}\" left out: its owner lets this PC use {:.1} GB there, less than the {:.1} GB working buffer a device needs. Raise its limit for this PC on that PC's Linked PCs page.",
            gb(c),
            gb(need)
        ),
        Some(c) if lent < c => format!(
            "\"{name}\" left out: it has only {:.1} GB free (its limit allows {:.1} GB), less than the {:.1} GB working buffer a device needs. Free memory on that PC and load again.",
            gb(lent),
            gb(c),
            gb(need)
        ),
        _ => format!(
            "\"{name}\" left out: it lends {:.1} GB, less than the {:.1} GB working buffer a device needs. Raise its limit for this PC on that PC's Linked PCs page.",
            gb(lent),
            gb(need)
        ),
    }
}

/// The split for the final context: `--tensor-split` by net capacity
/// ([`split::tensor_split_net`]: what each device lends less its compute
/// buffer, this PC's GPU also less the projector and output layer), with linked PCs whose
/// devices all net 0 left out of `--rpc`. Returns (rpc, tensor split) when
/// it changes the load, and a note.
pub fn net_split(l: &LinkedLoad, compute_buffer_bytes: u64, projector_bytes: u64) -> (Option<(Vec<String>, Option<String>)>, Option<String>) {
    let (cb, proj) = (compute_buffer_bytes.div_ceil(1 << 20), projector_bytes.div_ceil(1 << 20));
    let Some(first) = split::tensor_split_net(&l.devices, &l.caps, cb, proj) else {
        return (None, None);
    };
    let idle = split::idle_endpoints(&first);
    let caps: Vec<(String, Option<u64>)> = l.caps.iter().filter(|(ep, _)| !idle.contains(ep)).cloned().collect();
    let devices: Vec<ListedDevice> = l.devices.iter().filter(|d| !idle.iter().any(|ep| split::device_on_endpoint(d, ep))).cloned().collect();
    let left_out: Vec<String> = idle
        .iter()
        .map(|ep| {
            let name = l.names.get(ep).cloned().unwrap_or_default();
            // What the split saw there: its devices' free memory (through its limits),
            // an older helper's single cap applied.
            let free: u64 = l.devices.iter().filter(|d| split::device_on_endpoint(d, ep)).map(|d| d.free_mib).sum();
            let old_cap = l.caps.iter().find(|(e, _)| e == ep).and_then(|(_, c)| *c);
            let lent = old_cap.map_or(free, |c| free.min(c)) << 20;
            // The owner's limit over the kinds it shares (one GPU limit covers all its GPUs).
            let mine: Vec<&LinkedShare> = l.shares.iter().filter(|s| &s.endpoint == ep).collect();
            let cap = [false, true]
                .into_iter()
                .filter_map(|cpu| mine.iter().filter(|s| s.cpu == cpu).filter_map(|s| s.cap_bytes).max())
                .reduce(|a, b| a.saturating_add(b));
            left_out_note(&name, lent, cap, compute_buffer_bytes)
        })
        .collect();
    let rpc: Vec<String> = caps.iter().map(|(ep, _)| ep.clone()).collect();
    let plan = if rpc.is_empty() { None } else { split::tensor_split_net(&devices, &caps, cb, proj) };
    let note = match &plan {
        Some(p) => {
            let parts: Vec<String> = p
                .shares
                .iter()
                .map(|(dev, mib, ep)| {
                    let owner = ep.as_ref().and_then(|e| l.names.get(e)).map(|n| format!("\"{n}\"")).unwrap_or_else(|| "this PC".into());
                    format!("{dev} on {owner} {:.1} GB", *mib as f64 / 1024.0)
                })
                .collect();
            let mut n = format!("Layers split by what each device can hold after its working buffer ({}), --tensor-split {}.", parts.join(", "), p.tensor_split);
            for x in &left_out {
                n.push(' ');
                n.push_str(x);
            }
            n
        }
        None => left_out.join(" "),
    };
    (Some((rpc, plan.map(|p| p.tensor_split))), Some(note).filter(|n| !n.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_lends_all_but_two_cores_by_default() {
        use system_monitor::CoreTopology;
        // The 4-vCPU VM: 2 (the owner's choice; the UI says what it costs there).
        assert_eq!(default_rpc_threads(Some(CoreTopology { performance_cores: 4, efficiency_cores: 0 }), 4), 2);
        // 8 cores / 16 threads: the cores minus 2.
        assert_eq!(default_rpc_threads(Some(CoreTopology { performance_cores: 8, efficiency_cores: 0 }), 16), 6);
        // Core Ultra 7 265F (8P + 12E): at most P-cores + 4.
        assert_eq!(default_rpc_threads(Some(CoreTopology { performance_cores: 8, efficiency_cores: 12 }), 20), 12);
        // Unknown topology: the logical count minus 2, at least 1.
        assert_eq!(default_rpc_threads(None, 6), 4);
        assert_eq!(default_rpc_threads(None, 2), 1);
        assert_eq!(default_rpc_threads(None, 0), 1);
    }

    #[test]
    fn default_limits_are_75_percent_of_free_gpu_and_half_of_usable_ram() {
        const GIB: u64 = 1 << 30;
        let gpu = [DevKind::Gpu];
        // 7031 MiB free -> 5273 -> rounded down to 0.5 GB: 5120 MiB.
        assert_eq!(default_share_limits(&gpu, 7031, 20 * GIB, 32 * GIB, 6), ShareLimits { gpu_mib: 5120, ram_mib: 0, threads: 6, paused: false });
        // CPU shared too: 32 GiB total -> reserve 4.8 GiB; 20 GiB available -> (20 - 4.8) / 2.
        let both = [DevKind::Gpu, DevKind::Ram];
        let l = default_share_limits(&both, 8000, 20 * GIB, 32 * GIB, 2);
        let usable = 20 * GIB - crate::commands::models::ram_reserve(32 * GIB);
        assert_eq!(l.ram_mib, (usable / 2 >> 20) / 512 * 512);
        assert_eq!(l.gpu_mib, 5632);
        // CPU only: no GPU memory, and less RAM than the reserve lends nothing.
        assert_eq!(default_share_limits(&[DevKind::Ram], 0, 3 * GIB, 16 * GIB, 0), ShareLimits { gpu_mib: 0, ram_mib: 0, threads: 1, paused: false });
    }

    #[test]
    fn legacy_memory_to_lend_becomes_both_default_caps() {
        let old = HelperSettings { ram_cap_mib: Some(6144), ..Default::default() };
        let m = old.migrate_legacy();
        assert_eq!((m.share_gpu_mib, m.share_ram_mib, m.ram_cap_mib), (Some(6144), Some(6144), None));
        // New settings win over a leftover legacy value.
        let both = HelperSettings { ram_cap_mib: Some(1), share_gpu_mib: Some(2048), ..Default::default() }.migrate_legacy();
        assert_eq!((both.share_gpu_mib, both.share_ram_mib), (Some(2048), None));
        // Old JSON (no share fields) still reads.
        let j: HelperSettings = serde_json::from_str(r#"{"port":50070,"devices":[],"threads":null,"ram_cap_mib":4096,"cache":true,"cache_cap_gb":50,"compress":false,"runtime_id":null}"#).unwrap();
        assert_eq!(j.migrate_legacy().share_ram_mib, Some(4096));
    }

    #[test]
    fn old_default_port_moves_to_the_new_one_once() {
        assert_eq!(HelperSettings::default().port, 47070);
        let mut old = HelperSettings { port: 50070, ..Default::default() };
        assert!(old.migrate_port());
        assert_eq!(old.port, 47070);
        // Now stored as 47070: nothing more to do.
        assert!(!old.migrate_port());
        // A port chosen by the user stays.
        let mut mine = HelperSettings { port: 51234, ..Default::default() };
        assert!(!mine.migrate_port());
        assert_eq!(mine.port, 51234);
        // Old stored JSON with 50070 migrates after reading.
        let mut j: HelperSettings = serde_json::from_str(r#"{"port":50070}"#).unwrap();
        assert!(j.migrate_port());
        assert_eq!(j.port, 47070);
    }

    /// Uses an in-memory database and a test-only Credential Manager namespace.
    #[tokio::test]
    async fn measured_speeds_feed_the_load_panel_and_suggest_turning_split_off() {
        let db = Db::open_in_memory().await.unwrap();
        let creds = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
        let svc = LinkService::new(db.clone(), &creds).await.unwrap();
        // Nothing measured yet.
        let p = svc.split_preview(Some("m1"), &LocalHw::default()).await.unwrap();
        assert!(p.alone_tok_s.is_none() && p.split_tok_s.is_none() && !p.suggest_off);
        // Two replies alone, one split (no split-on peers: empty peer set key).
        let (alone, split_key) = (svc.speed_key("m1", false), svc.speed_key("m1", true));
        for v in [120.0, 124.0, 122.0] {
            svc.record_speed(&alone, v).await;
        }
        svc.record_speed(&split_key, 23.0).await;
        let p = svc.split_preview(Some("m1"), &LocalHw::default()).await.unwrap();
        assert_eq!((p.alone_tok_s, p.split_tok_s, p.split_replies), (Some(122.0), Some(23.0), 1));
        assert!(p.suggest_off && p.message.as_deref().is_some_and(|m| m.contains("slower")));
        // Persisted: a new service (app restart) still has them.
        let again = LinkService::new(db, &creds).await.unwrap();
        assert_eq!(again.split_preview(Some("m1"), &LocalHw::default()).await.unwrap().alone_tok_s, Some(122.0));
        // Only the last SPEED_SAMPLES are kept.
        for _ in 0..30 {
            again.record_speed(&alone, 50.0).await;
        }
        assert_eq!(again.speeds.lock().unwrap()[&alone].len(), SPEED_SAMPLES);
        creds.delete(linking::identity::CRED_IDENTITY).unwrap();
    }

    fn rpc_event(id: u64, key: &str, open: bool) -> SessionEvent {
        SessionEvent { id, peer_key: key.into(), peer_name: "main".into(), remote: "127.0.0.1:1".into(), mode: "rpc", open }
    }

    /// Weight files written during a linked PC's stream belong to it, the cap
    /// is enforced when the stream ends, and unlinking deletes them.
    #[tokio::test]
    async fn weight_cache_is_attributed_capped_and_deleted_on_unlink() {
        let db = Db::open_in_memory().await.unwrap();
        let creds = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
        let svc = LinkService::new(db, &creds).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        svc.init_cache(dir.path());
        let store = svc.shared.cache().unwrap();
        store.ensure_dir().unwrap();
        let write = |name: &str, n: usize| std::fs::write(store.llama_cache_dir().join("rpc").join(name), vec![0u8; n]).unwrap();
        let key = [7u8; 32];
        let key_b64 = key_to_b64(&key);

        // "Speed up reloads" off: nothing is attributed.
        svc.shared.cache_on.store(false, Ordering::Relaxed);
        svc.shared.session(rpc_event(1, &key_b64, true));
        write("off", 10);
        svc.shared.session(rpc_event(1, &key_b64, false));
        assert!(svc.cache_view().bytes == 10 && store.sizes_by_peer().is_empty());
        store.clear();

        // On, cap 1000 bytes: two 600-byte files from one stream -> the older one is evicted.
        svc.shared.cache_on.store(true, Ordering::Relaxed);
        svc.shared.cache_cap.store(1000, Ordering::Relaxed);
        svc.shared.session(rpc_event(2, &key_b64, true));
        write("a", 600);
        std::thread::sleep(Duration::from_millis(30));
        write("b", 600);
        svc.shared.session(rpc_event(2, &key_b64, false));
        assert_eq!(store.size(), 600, "capped at 1000 bytes, oldest first");
        assert_eq!(store.sizes_by_peer().get(&key_b64), Some(&600));
        assert_eq!(svc.cache_view().bytes, 600);

        // The other PC unlinking this one deletes what it sent.
        svc.shared.unlinked(key);
        assert_eq!(store.size(), 0);
        assert_eq!(svc.cache_view().bytes, 0);

        // "Clear cache" empties it for good.
        write("c", 5);
        assert_eq!(svc.clear_cache(), 5);
        assert_eq!(store.size(), 0);
        creds.delete(linking::identity::CRED_IDENTITY).unwrap();
    }
}


#[cfg(test)]
mod stall_tests {
    use super::*;
    use linking::wire::{self, Frame, Mode};
    use std::time::Instant;

    /// A linked PC that pairs and answers pings/info, but never reads a
    /// bandwidth test (a stalled link, as seen on the VirtualBox NAT).
    async fn stalling_helper(id: Arc<Identity>, codes: Arc<CodeSlot>) -> SocketAddr {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = l.accept().await.unwrap();
                let (id, codes) = (id.clone(), codes.clone());
                tokio::spawn(async move {
                    let info = PeerInfo { name: "VM".into(), helping: true, runtime_tag: Some("b1".into()), ..Default::default() };
                    let mut h = Vec::new();
                    if wire::read_frame(&mut s, &mut h).await.is_err() {
                        return;
                    }
                    match wire::parse_hello(&h) {
                        Ok(Mode::Pair) => {
                            let i = info.clone();
                            let _ = linking::handshake::pair_respond(s, &id, &h, &codes, move |_, _| i).await;
                        }
                        Ok(Mode::Control) => {
                            let Ok(mut sess) = linking::handshake::link_respond(s, &id, &h).await else { return };
                            let _ = linking::handshake::send_accept(&mut sess.writer, serde_json::json!({ "info": info })).await;
                            loop {
                                match sess.reader.recv().await {
                                    Ok(Frame::Ping(n)) => {
                                        let _ = sess.writer.send(&Frame::Pong(n)).await;
                                    }
                                    Ok(Frame::Json(v)) if v["t"] == "info" => {
                                        let _ = sess.writer.send(&Frame::Json(serde_json::json!({ "t": "info", "info": info }))).await;
                                    }
                                    Ok(Frame::Json(v)) if v["t"] == "bw" => tokio::time::sleep(Duration::from_secs(3600)).await,
                                    Ok(_) => {}
                                    Err(_) => return,
                                }
                            }
                        }
                        _ => {}
                    }
                });
            }
        });
        a
    }

    /// The VM run: pairing's bandwidth test stalled ~150 s and the first
    /// load waited ~12 min. Now neither pairing nor the load waits for it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_stalled_bandwidth_test_delays_neither_pairing_nor_the_load() {
        let db = Db::open_in_memory().await.unwrap();
        let creds = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
        let svc = LinkService::new(db, &creds).await.unwrap();
        let (hid, codes) = (Arc::new(Identity::generate().unwrap()), Arc::new(CodeSlot::new()));
        let addr = stalling_helper(hid, codes.clone()).await;
        let t = Instant::now();
        let v = svc.pair("127.0.0.1", Some(addr.port()), &codes.new_code().code).await.unwrap();
        assert!(t.elapsed() < Duration::from_secs(5), "pairing took {:?}", t.elapsed());
        assert!(v.online && v.rtt_ms.is_some(), "{v:?}");
        // A load right away: ready while the bandwidth test is still stuck.
        let runtime = inference::InstalledRuntime {
            id: "b1-cpu".into(),
            tag: "b1".into(),
            variant: "cpu".into(),
            server_path: "llama-server.exe".into(),
            installed_at: String::new(),
            source: "official".into(),
            capabilities: Default::default(),
        };
        let t = Instant::now();
        let peers = svc.split_peers(&runtime).await.unwrap();
        assert!(t.elapsed() < Duration::from_secs(5), "load setup took {:?}", t.elapsed());
        assert_eq!(peers.len(), 1);
        assert!(peers[0].endpoint.starts_with("127.0.0.1:"));
        creds.delete(linking::identity::CRED_IDENTITY).unwrap();
    }
}


#[cfg(test)]
mod no_hang_tests {
    use super::*;
    use std::time::Instant;

    /// Regression (2026-11-23): `status()` held the `settings` lock (a
    /// temporary guard inside the struct literal) while `cache_view()` locked
    /// it again, so every Linked PCs poll blocked a runtime thread forever;
    /// `set_name` then waited on the `name` lock, and once the threads were
    /// gone every async command hung. This runs status, set_name, set_setting
    /// and create_conversation one after another and concurrently while the
    /// cache hooks and the speed recorder are busy. A deadlock can't be
    /// timed out from inside the blocked runtime, so the work runs on its own
    /// thread and the test waits for it from outside.
    #[test]
    fn linking_status_name_and_db_calls_never_hang() {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(String, Duration)>>();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async {
                let dir = tempfile::tempdir().unwrap();
                let db = Db::open(&dir.path().join("t.db")).await.unwrap();
                let creds = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
                let svc = Arc::new(LinkService::new(db.clone(), &creds).await.unwrap());
                svc.init_cache(dir.path());
                svc.shared.cache_on.store(true, Ordering::Relaxed);
                let mut times: Vec<(String, Duration)> = Vec::new();
                macro_rules! timed {
                    ($name:expr, $e:expr) => {{
                        let t = Instant::now();
                        let r = $e;
                        times.push(($name.to_string(), t.elapsed()));
                        r
                    }};
                }
                // One after another.
                timed!("linking_status", svc.status().await.unwrap());
                timed!("linking_set_name", svc.set_name("VM").await.unwrap());
                timed!("linking_status again", svc.status().await.unwrap());
                timed!("set_setting", db.set_setting("ui.theme", &"dark").await.unwrap());
                timed!("create_conversation", db.create_conversation("t", None, &serde_json::json!({})).await.unwrap());
                // Concurrently, with the cache hooks and the speed recorder busy.
                let busy = {
                    let svc = svc.clone();
                    tokio::spawn(async move {
                        for i in 0..50u64 {
                            svc.shared.session(SessionEvent { id: i, peer_key: "K".into(), peer_name: "main".into(), remote: "127.0.0.1:1".into(), mode: "rpc", open: true });
                            svc.record_speed("m|alone", 100.0 + i as f64).await;
                            svc.shared.session(SessionEvent { id: i, peer_key: "K".into(), peer_name: "main".into(), remote: "127.0.0.1:1".into(), mode: "rpc", open: false });
                        }
                    })
                };
                let mut workers = Vec::new();
                for w in 0..4 {
                    let (svc, db) = (svc.clone(), db.clone());
                    workers.push(tokio::spawn(async move {
                        let mut worst = Duration::ZERO;
                        for i in 0..10 {
                            let t = Instant::now();
                            svc.status().await.unwrap();
                            svc.set_name(&format!("PC {w}-{i}")).await.unwrap();
                            db.set_setting("ui.theme", &"light").await.unwrap();
                            db.create_conversation("c", None, &serde_json::json!({})).await.unwrap();
                            svc.split_preview(Some("m"), &LocalHw::default()).await.unwrap();
                            worst = worst.max(t.elapsed());
                        }
                        worst
                    }));
                }
                for (w, h) in workers.into_iter().enumerate() {
                    times.push((format!("worker {w} worst round"), h.await.unwrap()));
                }
                busy.await.unwrap();
                let _ = creds.delete(linking::identity::CRED_IDENTITY);
                tx.send(times).unwrap();
            });
        });
        let times = rx.recv_timeout(Duration::from_secs(30)).expect("linking status / set_name / database calls hung (deadlock)");
        for (name, d) in &times {
            assert!(*d < Duration::from_secs(2), "{name} took {d:?}");
        }
    }
}


/// Per-linked-PC sharing limits on the helper (2026-11-23).
#[cfg(test)]
mod share_tests {
    use super::*;

    async fn svc() -> (LinkService, security::CredentialStore) {
        let db = Db::open_in_memory().await.unwrap();
        let creds = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
        (LinkService::new(db, &creds).await.unwrap(), creds)
    }

    /// A helping PC with one 8000 MiB GPU (7000 MiB free) and two linked PCs.
    async fn helping(svc: &LinkService) -> (String, String) {
        {
            let mut l = svc.shared.live.write().unwrap();
            l.helping = true;
            l.rpc_endpoint = Some("127.0.0.1:1".into());
            l.devices = vec![DeviceInfo { name: "CUDA0".into(), description: "GPU".into(), total_mib: 8000, free_mib: 7000 }];
        }
        svc.shared.rpc_threads.store(local_default_rpc_threads(), Ordering::Relaxed);
        let mut ids = Vec::new();
        for (name, k) in [("Desk", [1u8; 32]), ("Laptop", [2u8; 32])] {
            let row = svc
                .shared
                .db
                .upsert_peer(&NewPeer { name, public_key: &key_to_b64(&k), last_address: None, same_network: true, initiated: false, runtime_tag: None, seen_from: None })
                .await
                .unwrap();
            svc.shared.pinned.write().unwrap().insert(k, name.into());
            ids.push(row.id);
        }
        (ids[0].clone(), ids[1].clone())
    }

    const A: [u8; 32] = [1u8; 32];
    const B: [u8; 32] = [2u8; 32];

    #[tokio::test]
    async fn limits_busy_pause_and_thread_restarts() {
        let (svc, creds) = svc().await;
        let (a_id, b_id) = helping(&svc).await;
        // Defaults: 75% of 7000 MiB free, rounded down to 0.5 GB.
        let t = svc.shared.rpc_target(&A).await.unwrap();
        let g = t.limits.as_ref().unwrap();
        assert_eq!(g.claim.budget().caps().gpu_mib, 5120);
        assert_eq!(*g.kinds, vec![DevKind::Gpu]);
        // A second PC while the first one is helped: the busy refusal.
        let e = svc.shared.rpc_target(&B).await.err().unwrap().to_string();
        assert!(e.contains("is busy helping \"Desk\""), "{e}");
        drop(t);
        assert!(svc.shared.rpc_target(&B).await.is_ok());
        // Paused: new sessions refused.
        svc.set_peer_share(&b_id, PeerShare { paused: true, ..Default::default() }).await.unwrap();
        let e = svc.shared.rpc_target(&B).await.err().unwrap().to_string();
        assert!(e.contains("paused by its owner"), "{e}");
        // Per-PC limit; info_for shows it and lowers the free memory reported.
        svc.set_peer_share(&a_id, PeerShare { gpu_mib: Some(2048), ..Default::default() }).await.unwrap();
        let i = svc.shared.info_for(Some(&A));
        assert_eq!(i.share.map(|s| s.gpu_mib), Some(2048));
        assert_eq!((i.devices[0].free_mib, i.devices[0].total_mib, i.ram_cap_mib), (2048, 2048, Some(2048)));
        assert!(i.share_enforced);
        // Another PC sees its own (default) limit.
        assert_eq!(svc.shared.info_for(Some(&B)).share.map(|s| s.gpu_mib), Some(5120));
        // Same threads as the running rpc-server: no restart.
        let t = svc.shared.rpc_target(&A).await.unwrap();
        assert_eq!(t.limits.as_ref().unwrap().claim.budget().caps().gpu_mib, 2048);
        drop(t);
        // Other threads: rpc-server restarts (no app here, so the restart itself fails).
        let other = if local_default_rpc_threads() == 1 { 2 } else { 1 };
        svc.set_peer_share(&a_id, PeerShare { threads: Some(other), ..Default::default() }).await.unwrap();
        assert!(matches!(svc.shared.rpc_target(&A).await, Err(LinkError::HelperOff)));
        // More than the PC has is refused.
        assert!(svc.set_peer_share(&a_id, PeerShare { gpu_mib: Some(9000), ..Default::default() }).await.is_err());
        assert!(svc.set_peer_share(&a_id, PeerShare { threads: Some(0), ..Default::default() }).await.is_err());
        // Stored and shown.
        let v = svc.view(&a_id).await.unwrap();
        assert_eq!((v.lends.stored.threads, v.lends.limits.threads, v.lends.shares_gpu, v.lends.shares_ram), (Some(other), other, true, false));
        creds.delete(linking::identity::CRED_IDENTITY).unwrap();
    }

    #[tokio::test]
    async fn no_device_list_or_helper_off_refuses() {
        let (svc, creds) = svc().await;
        assert!(matches!(svc.shared.rpc_target(&A).await, Err(LinkError::HelperOff)));
        helping(&svc).await;
        svc.shared.live.write().unwrap().devices.clear();
        let e = svc.shared.rpc_target(&A).await.err().unwrap().to_string();
        assert!(e.contains("can't enforce its sharing limits"), "{e}");
        creds.delete(linking::identity::CRED_IDENTITY).unwrap();
    }

    #[test]
    fn net_split_leaves_out_a_pc_that_lends_less_than_a_working_buffer() {
        let devices = split::parse_devices("  CUDA0: GPU (8000 MiB, 7000 MiB free)\n  RPC0: 127.0.0.1:1 (8000 MiB, 400 MiB free)\n  RPC1: 127.0.0.1:2 (8000 MiB, 4000 MiB free)\n");
        let names: HashMap<String, String> = [("127.0.0.1:1".to_string(), "Tiny".to_string()), ("127.0.0.1:2".to_string(), "Desk".to_string())].into();
        let l = LinkedLoad { devices, caps: vec![("127.0.0.1:1".into(), None), ("127.0.0.1:2".into(), None)], names, shares: Vec::new() };
        let (split, note) = net_split(&l, 500 << 20, 0);
        let (rpc, ts) = split.unwrap();
        assert_eq!(rpc, vec!["127.0.0.1:2".to_string()]);
        // 4000*0.9-500, 7000*0.9-500.
        assert_eq!(ts.as_deref(), Some("3100,5800"));
        let note = note.unwrap();
        assert!(note.contains("\"Tiny\" left out") && note.contains("--tensor-split 3100,5800"), "{note}");
    }

    #[test]
    fn a_helpers_refusal_reads_as_the_plan_says() {
        let r = serde_json::to_value(Refusal::OverLimit { kind: DevKind::Gpu, bytes: 3_000_000_000, cap_bytes: 2_147_483_648, used_bytes: 0 }).unwrap();
        assert_eq!(
            refusal_message("Desk PC", &r),
            "\"Desk PC\" refused 3.0 GB: its owner lets this PC use up to 2.1 GB of GPU memory. Use a smaller context, or ask them to raise the limit on their Linked PCs page."
        );
        let r = serde_json::to_value(Refusal::Incompatible { proto_major: 10 }).unwrap();
        assert!(refusal_message("Desk PC", &r).contains("RPC protocol 10"));
        // An older helper: its single cap is spread over its devices.
        let info = PeerInfo {
            devices: vec![DeviceInfo { name: "CUDA0".into(), free_mib: 6000, ..Default::default() }, DeviceInfo { name: "CPU".into(), free_mib: 2000, ..Default::default() }],
            ram_cap_mib: Some(4000),
            share_enforced: false,
            ..Default::default()
        };
        let s = shares_from_info("Old", "", &info);
        assert_eq!((s[0].bytes >> 20, s[1].bytes >> 20, s[1].cpu), (3000, 1000, true));
    }
}

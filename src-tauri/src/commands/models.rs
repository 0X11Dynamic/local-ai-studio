use std::path::{Path, PathBuf};

use app_core::AppError;
use inference::{GpuLayers, LoadConfig};
use model_registry::estimate::{self, AutoCtx, AutoCtxEnv, EstimateInput, MemoryEstimate, ShareSplit};
use model_registry::{ImportReport, ModelMetadata};
use persistence::ModelRow;
use serde::Serialize;
use tauri::State;

use super::Cmd;
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct ModelView {
    #[serde(flatten)]
    pub row: ModelRow,
    pub meta: ModelMetadata,
    pub shards: Vec<String>,
    /// True when this model is the one currently loaded (or loading).
    pub active: bool,
}

fn view(row: ModelRow, active_id: Option<&str>) -> ModelView {
    let meta = serde_json::from_str(&row.metadata).unwrap_or_default();
    let shards = serde_json::from_str(&row.shard_paths).unwrap_or_default();
    let active = active_id == Some(row.id.as_str());
    ModelView { row, meta, shards, active }
}

#[tauri::command]
pub async fn list_models(state: State<'_, SharedState>) -> Cmd<Vec<ModelView>> {
    let active = state.engine.status().model_id;
    Ok(state.db.list_models().await?.into_iter().map(|r| view(r, active.as_deref())).collect())
}

/// Import a GGUF file or a folder (scanned recursively). Files stay where they are.
#[tauri::command]
pub async fn import_models(state: State<'_, SharedState>, path: String) -> Cmd<ImportReport> {
    let p = PathBuf::from(&path);
    let report = model_registry::import_path(&state.db, &p, "import", None).await?;
    if p.is_file() && report.registered.is_empty() && report.invalid.is_empty() {
        let reason = report.skipped.first().map(|s| s.reason.clone()).unwrap_or_else(|| "not a model file".into());
        return Err(AppError::InvalidInput(format!("{}: {reason}", p.display())));
    }
    Ok(report)
}

#[tauri::command]
pub async fn scan_lmstudio(state: State<'_, SharedState>) -> Cmd<ImportReport> {
    let lm = state
        .paths
        .lmstudio_models_dir
        .clone()
        .ok_or_else(|| AppError::NotFound("LM Studio models folder not found".into()))?;
    model_registry::import_path(&state.db, &lm, "lmstudio", Some(&lm)).await
}

#[tauri::command]
pub async fn refresh_models(state: State<'_, SharedState>) -> Cmd<Vec<ModelView>> {
    model_registry::refresh_status(&state.db).await?;
    list_models(state).await
}

fn ensure_not_active(state: &SharedState, id: &str) -> Cmd<()> {
    if state.engine.status().model_id.as_deref() == Some(id) && state.engine.backend_pid().is_some() {
        return Err(AppError::Conflict("unload this model first".into()));
    }
    Ok(())
}

/// Remove from the library only; files on disk are untouched.
#[tauri::command]
pub async fn remove_model(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    ensure_not_active(state.inner(), &id)?;
    state.db.delete_model(&id).await
}

/// Move the model's files (and its vision projector, when no other model in
/// the library uses it) to the Windows Recycle Bin, then remove the model
/// from the library. Never deletes permanently: when the Recycle Bin of a
/// file's drive is off, missing (not a fixed local drive) or too small for
/// the files, nothing is touched and the error says why. The caller must
/// pass `confirm: true` (the user ticked the confirm box), so a call with
/// only an id can't do it. Returns the files moved.
#[tauri::command]
pub async fn delete_model_files(state: State<'_, SharedState>, id: String, confirm: Option<bool>) -> Cmd<Vec<String>> {
    if confirm != Some(true) {
        return Err(AppError::InvalidInput("confirm the delete first (confirm: true)".into()));
    }
    ensure_not_active(state.inner(), &id)?;
    let m = state.db.get_model(&id).await?;
    let mut files: Vec<String> = serde_json::from_str(&m.shard_paths).unwrap_or_default();
    if files.is_empty() {
        files.push(m.path.clone());
    }
    // The projector goes too, unless another model still uses it.
    if let Some(proj) = m.projector_path.as_ref().filter(|p| !p.trim().is_empty()) {
        let shared = state.db.list_models().await?.iter().any(|o| o.id != m.id && o.projector_path.as_deref().is_some_and(|q| Path::new(q) == Path::new(proj)));
        if !shared && !files.contains(proj) {
            files.push(proj.clone());
        }
    }
    let mut moving: Vec<PathBuf> = Vec::new();
    for f in &files {
        let p = Path::new(f);
        if p.extension().is_none_or(|e| !e.eq_ignore_ascii_case("gguf")) {
            return Err(AppError::InvalidInput(format!("refusing to delete non-GGUF file {f}")));
        }
        if p.is_file() {
            moving.push(std::path::absolute(p)?);
        }
    }
    if !moving.is_empty() {
        let list = moving.clone();
        tokio::task::spawn_blocking(move || recycle::check_and_send(&list)).await.map_err(|e| AppError::Runtime(e.to_string()))?.map_err(AppError::Conflict)?;
    }
    state.db.delete_model(&id).await?;
    Ok(moving.iter().map(|p| p.to_string_lossy().into_owned()).collect())
}

/// Sending files to the Windows Recycle Bin, refusing (rather than deleting
/// permanently) when the bin cannot hold them.
mod recycle {
    use std::path::{Path, PathBuf};

    /// Check every drive's Recycle Bin can take its files, then move them all.
    pub fn check_and_send(files: &[PathBuf]) -> Result<(), String> {
        let mut by_drive: Vec<(String, Option<String>, u64)> = vec![];
        for f in files {
            let size = std::fs::metadata(f).map(|m| m.len()).map_err(|e| format!("could not read {}: {e}", f.display()))?;
            let (root, guid) = volume_of(f)?;
            match by_drive.iter_mut().find(|d| d.0 == root) {
                Some(d) => d.2 += size,
                None => by_drive.push((root, guid, size)),
            }
        }
        for (root, guid, bytes) in &by_drive {
            check(root, guid.as_deref(), *bytes)?;
        }
        send(files)
    }

    fn gb(b: u64) -> String {
        format!("{:.1} GB", b as f64 / 1e9)
    }

    /// Can the Recycle Bin of `root` hold `bytes`? Err(why) when it cannot.
    pub fn check(root: &str, guid: Option<&str>, bytes: u64) -> Result<(), String> {
        if !is_fixed(root) {
            return Err(format!("{root} is not a fixed local drive, so it has no Recycle Bin; nothing was deleted (delete the files yourself if you are sure)"));
        }
        let key = guid.map(|g| format!(r"Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume\{g}"));
        if key.as_deref().and_then(|k| reg_dword(k, "NukeOnDelete")).unwrap_or(0) != 0 {
            return Err(format!("the Recycle Bin is turned off for {root} (\"Don't move files to the Recycle Bin\"), so the files would be deleted permanently; nothing was deleted"));
        }
        let max = match key.as_deref().and_then(|k| reg_dword(k, "MaxCapacity")) {
            Some(mb) => mb as u64 * 1024 * 1024,
            // No setting saved yet: Windows' default size for the drive.
            None => default_capacity(total_bytes(root).ok_or_else(|| format!("could not read the size of {root}; nothing was deleted"))?),
        };
        if bytes > max {
            return Err(format!(
                "the files ({}) are bigger than the Recycle Bin of {root} can hold ({}), so Windows would delete them permanently; nothing was deleted. Make the Recycle Bin bigger (its Properties) or delete the files yourself",
                gb(bytes),
                gb(max)
            ));
        }
        Ok(())
    }

    /// Windows' default Recycle Bin size: 10% of the first 40 GB of the
    /// drive plus 5% of the rest.
    pub fn default_capacity(total: u64) -> u64 {
        const G40: u64 = 40 * 1024 * 1024 * 1024;
        if total <= G40 { total / 10 } else { G40 / 10 + (total - G40) / 20 }
    }

    #[cfg(windows)]
    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    /// The drive root (e.g. "C:\") and its volume GUID ("{...}").
    #[cfg(windows)]
    fn volume_of(p: &Path) -> Result<(String, Option<String>), String> {
        use windows::Win32::Storage::FileSystem::{GetVolumeNameForVolumeMountPointW, GetVolumePathNameW};
        use windows::core::PCWSTR;
        let w = wide(p.as_os_str());
        let mut buf = [0u16; 1024];
        unsafe { GetVolumePathNameW(PCWSTR(w.as_ptr()), &mut buf) }.map_err(|e| format!("could not find the drive of {}: {e}", p.display()))?;
        let root = String::from_utf16_lossy(&buf[..buf.iter().position(|&c| c == 0).unwrap_or(0)]);
        let rw = wide(std::ffi::OsStr::new(&root));
        let mut name = [0u16; 128];
        let guid = unsafe { GetVolumeNameForVolumeMountPointW(PCWSTR(rw.as_ptr()), &mut name) }.ok().and_then(|_| {
            let s = String::from_utf16_lossy(&name[..name.iter().position(|&c| c == 0).unwrap_or(0)]);
            // \\?\Volume{GUID}\
            s.find('{').and_then(|a| s.find('}').map(|b| s[a..=b].to_string()))
        });
        Ok((root, guid))
    }

    #[cfg(windows)]
    fn is_fixed(root: &str) -> bool {
        use windows::Win32::Storage::FileSystem::GetDriveTypeW;
        use windows::core::PCWSTR;
        const DRIVE_FIXED: u32 = 3;
        let w = wide(std::ffi::OsStr::new(root));
        unsafe { GetDriveTypeW(PCWSTR(w.as_ptr())) == DRIVE_FIXED }
    }

    #[cfg(windows)]
    fn total_bytes(root: &str) -> Option<u64> {
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        use windows::core::PCWSTR;
        let w = wide(std::ffi::OsStr::new(root));
        let mut total = 0u64;
        unsafe { GetDiskFreeSpaceExW(PCWSTR(w.as_ptr()), None, Some(&mut total), None) }.ok().map(|_| total)
    }

    #[cfg(windows)]
    fn reg_dword(key: &str, value: &str) -> Option<u32> {
        use windows::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
        use windows::core::PCWSTR;
        let (k, v) = (wide(std::ffi::OsStr::new(key)), wide(std::ffi::OsStr::new(value)));
        let mut data = 0u32;
        let mut len = std::mem::size_of::<u32>() as u32;
        let r = unsafe { RegGetValueW(HKEY_CURRENT_USER, PCWSTR(k.as_ptr()), PCWSTR(v.as_ptr()), RRF_RT_REG_DWORD, None, Some(&mut data as *mut u32 as *mut _), Some(&mut len)) };
        r.is_ok().then_some(data)
    }

    /// Move the files to the Recycle Bin (one shell operation, no UI;
    /// FOF_WANTNUKEWARNING asks before anything would be destroyed instead).
    #[cfg(windows)]
    fn send(files: &[PathBuf]) -> Result<(), String> {
        use std::os::windows::ffi::OsStrExt;
        use windows::Win32::UI::Shell::{FO_DELETE, FOF_ALLOWUNDO, FOF_NO_UI, FOF_WANTNUKEWARNING, SHFILEOPSTRUCTW, SHFileOperationW};
        use windows::core::PCWSTR;
        // A list of full paths, each ending in NUL, the list in a second NUL.
        let mut from: Vec<u16> = vec![];
        for f in files {
            from.extend(f.as_os_str().encode_wide());
            from.push(0);
        }
        from.push(0);
        let mut op = SHFILEOPSTRUCTW { wFunc: FO_DELETE, pFrom: PCWSTR(from.as_ptr()), fFlags: (FOF_ALLOWUNDO.0 | FOF_NO_UI.0 | FOF_WANTNUKEWARNING.0) as u16, ..Default::default() };
        let r = unsafe { SHFileOperationW(&mut op) };
        if r != 0 {
            return Err(format!("Windows could not move the files to the Recycle Bin (error {r:#x}); check none of them is open in another program"));
        }
        if op.fAnyOperationsAborted.as_bool() {
            return Err("moving the files to the Recycle Bin was cancelled".into());
        }
        if let Some(left) = files.iter().find(|f| f.exists()) {
            return Err(format!("{} is still there after moving to the Recycle Bin", left.display()));
        }
        Ok(())
    }

    #[cfg(not(windows))]
    fn volume_of(_p: &Path) -> Result<(String, Option<String>), String> {
        Err("moving files to the Recycle Bin is only supported on Windows".into())
    }
    #[cfg(not(windows))]
    fn is_fixed(_root: &str) -> bool {
        false
    }
    #[cfg(not(windows))]
    fn total_bytes(_root: &str) -> Option<u64> {
        None
    }
    #[cfg(not(windows))]
    fn reg_dword(_key: &str, _value: &str) -> Option<u32> {
        None
    }
    #[cfg(not(windows))]
    fn send(_files: &[PathBuf]) -> Result<(), String> {
        Err("moving files to the Recycle Bin is only supported on Windows".into())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_capacity_follows_windows() {
            let g = 1024u64 * 1024 * 1024;
            assert_eq!(default_capacity(20 * g), 2 * g);
            assert_eq!(default_capacity(1040 * g), 4 * g + 50 * g);
        }

        #[cfg(windows)]
        #[test]
        fn a_drive_with_no_room_is_refused_without_touching_anything() {
            let d = std::env::temp_dir();
            let (root, guid) = volume_of(&d).unwrap();
            assert!(root.ends_with('\\'), "{root}");
            // Far more than any Recycle Bin holds: refused, with the reason.
            let e = check(&root, guid.as_deref(), u64::MAX / 4).unwrap_err();
            assert!(e.contains("nothing was deleted"), "{e}");
            if !e.contains("turned off") && !e.contains("not a fixed") {
                assert_eq!(check(&root, guid.as_deref(), 1), Ok(()), "a 1-byte file fits");
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EstimateView {
    pub estimate: MemoryEstimate,
    pub free_vram_bytes: Option<u64>,
    pub total_vram_bytes: Option<u64>,
    pub available_ram_bytes: u64,
    /// GPU layers that fit in free VRAM minus a 1 GiB margin (estimate).
    pub suggested_gpu_layers: Option<u64>,
    pub fits: bool,
    pub warnings: Vec<String>,
    /// What "Share with CPU" would pick right now (estimate).
    pub share_split: Option<ShareSplit>,
    /// The toggle is off and the full-GPU estimate does not fit free VRAM.
    pub suggest_share: bool,
    /// What "Auto (fits your hardware)" context picks right now (estimate).
    /// When the config's n_ctx is 0 the estimate above uses this size.
    pub auto_ctx: AutoCtx,
    /// What the "Auto" KV cache type picks right now (None: an explicit type).
    pub auto_kv: Option<AutoKv>,
    /// Share with CPU is on and the runtime fits the layers itself
    /// (`--fit`): `share_split` is only an estimate of its choice.
    pub share_fit: bool,
}

/// Inputs for an Auto context ([`estimate::auto_n_ctx`]) for a load with `cfg`.
/// `uses_gpu` is false for CPU-only runtimes.
#[allow(clippy::too_many_arguments)]
pub fn auto_ctx_for(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    available_ram: u64,
    has_n_cpu_moe: bool,
    uses_gpu: bool,
) -> AutoCtx {
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut input = estimate_input(cfg, meta, total, projector_bytes);
    if !uses_gpu {
        input.gpu_layers = Some(0);
    }
    let sharing = cfg.share_with_cpu && uses_gpu && !matches!(cfg.gpu_layers, GpuLayers::CpuOnly);
    let env = AutoCtxEnv {
        free_vram,
        available_ram,
        share_with_cpu: sharing,
        has_n_cpu_moe,
        auto_layers: matches!(cfg.gpu_layers, GpuLayers::Auto),
    };
    estimate::auto_n_ctx(meta, size, &input, &env)
}

/// Replace an Auto context (n_ctx 0) with the size [`auto_ctx_for`] picks.
/// Other configs are returned unchanged, with None.
#[allow(clippy::too_many_arguments)]
pub fn resolve_auto_ctx(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    available_ram: u64,
    has_n_cpu_moe: bool,
    uses_gpu: bool,
) -> (LoadConfig, Option<AutoCtx>) {
    if cfg.n_ctx != 0 {
        return (cfg.clone(), None);
    }
    let a = auto_ctx_for(cfg, meta, size, projector_bytes, free_vram, available_ram, has_n_cpu_moe, uses_gpu);
    let mut c = cfg.clone();
    c.n_ctx = a.n_ctx.min(u32::MAX as u64) as u32;
    (c, Some(a))
}

/// VRAM held by the loaded backend (its measured GPU buffers), which a new
/// load gets back because loading unloads it first.
pub fn loaded_backend_vram(st: &SharedState) -> u64 {
    let s = st.engine.status();
    if st.engine.backend_pid().is_none() {
        return 0;
    }
    let mib: f64 = s
        .buffers
        .iter()
        .filter(|b| !b.device.eq_ignore_ascii_case("cpu") && !b.device.to_ascii_lowercase().contains("host"))
        .map(|b| b.mib)
        .sum();
    (mib * 1024.0 * 1024.0) as u64
}

pub fn estimate_input(cfg: &LoadConfig, meta: &ModelMetadata, total_layers: u64, projector_bytes: u64) -> EstimateInput {
    EstimateInput {
        n_ctx: if cfg.n_ctx == 0 { meta.context_length.unwrap_or(4096) } else { cfg.n_ctx as u64 },
        gpu_layers: match cfg.gpu_layers {
            GpuLayers::All | GpuLayers::Auto => None,
            GpuLayers::Count(n) => Some((n as u64).min(total_layers)),
            GpuLayers::CpuOnly => Some(0),
        },
        cache_type_k: cfg.cache_type_k.clone(),
        cache_type_v: cfg.cache_type_v.clone(),
        kv_offload: cfg.kv_offload,
        n_ubatch: cfg.ubatch_size.unwrap_or(512) as u64,
        parallel: cfg.parallel as u64,
        projector_bytes,
    }
}

/// Bytes of the vision projector (mmproj) a load with `cfg` puts in memory:
/// its file size when `load_projector` is on and the file exists, else 0.
pub fn projector_bytes(cfg: &LoadConfig, projector_path: Option<&str>) -> u64 {
    if !cfg.load_projector {
        return 0;
    }
    projector_path.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0)
}

/// Free VRAM on the first GPU right now (NVML), if there is one.
pub fn free_vram(st: &SharedState) -> Option<u64> {
    st.monitor.hardware_info().gpus.first().map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes))
}

/// The default runtime: the saved setting, else the first installed.
pub async fn default_runtime(st: &SharedState) -> Option<inference::InstalledRuntime> {
    let installed = st.runtimes.list_installed();
    let want = st.db.get_setting::<String>("runtime.default").await.ok().flatten();
    match want {
        Some(w) => installed.into_iter().find(|r| r.id == w),
        None => installed.into_iter().next(),
    }
}

/// Can the runtime let llama.cpp fit the GPU layers itself with an explicit
/// margin (`--fit` and `--fit-target`)?
pub fn runtime_can_fit(caps: &inference::RuntimeCapabilities) -> bool {
    caps.has("--fit") && caps.has("--fit-target")
}

/// Turn the user's "Share with CPU" choice into concrete load settings.
/// Returns the config to load with (the user's own config is unchanged when
/// the toggle is off) and the split the estimate predicts.
///
/// With `can_fit` (the runtime has `--fit` and `--fit-target`), llama.cpp
/// picks the layer count itself from real allocation sizes: no `-ngl`,
/// `--fit on`, an explicit `--fit-target` margin (the config's, else 1024
/// MiB) and `--fit-ctx` equal to the chosen context, which is also passed
/// explicitly, so fitting never lowers the context. The estimate's split is
/// then only the UI preview. MoE models the estimate splits keep every layer
/// on the GPU with `--n-cpu-moe` (unchanged), and when nothing fits at all
/// the load is CPU only. Without `can_fit`, the estimate's layer count is
/// passed (`-ngl N`) as before.
/// `projector_bytes` is what [`projector_bytes`] returns for this load.
pub fn resolve_share(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    has_n_cpu_moe: bool,
    can_fit: bool,
) -> (LoadConfig, Option<ShareSplit>) {
    if !cfg.share_with_cpu || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) {
        return (cfg.clone(), None);
    }
    let mut c = cfg.clone();
    c.fit = true;
    let let_llama_fit = |c: &mut LoadConfig| {
        c.gpu_layers = GpuLayers::Auto;
        c.n_cpu_moe = None;
        if can_fit {
            c.fit_target_mib = Some(cfg.fit_target_mib.unwrap_or(inference::DEFAULT_FIT_TARGET_MIB));
            c.fit_ctx = (c.n_ctx > 0).then_some(c.n_ctx);
        }
    };
    let Some(free) = free_vram else {
        // No GPU information: let llama.cpp fit the layers itself.
        let_llama_fit(&mut c);
        return (c, None);
    };
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut input = estimate_input(cfg, meta, total, projector_bytes);
    input.gpu_layers = None;
    let split = estimate::share_split(meta, size, &input, free, has_n_cpu_moe);
    if let Some(n) = split.n_cpu_moe {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = Some(n as u32);
    } else if split.gpu_layers == 0 && !split.all_fit {
        c.gpu_layers = GpuLayers::CpuOnly;
        c.n_cpu_moe = None;
    } else if can_fit {
        let_llama_fit(&mut c);
    } else if split.all_fit {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = None;
    } else {
        c.gpu_layers = GpuLayers::Count(split.gpu_layers as u32);
        c.n_cpu_moe = None;
    }
    (c, Some(split))
}

/// What the runtime offers for an automatic KV cache type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSupport {
    /// `--cache-type-k` / `-v` accept q8_0.
    pub q8_0: bool,
    /// `--flash-attn` exists (a quantized V cache needs Flash Attention on).
    pub flash_attn: bool,
}

impl KvSupport {
    pub fn of(caps: &inference::RuntimeCapabilities) -> Self {
        Self {
            q8_0: caps.has("--cache-type-k") && caps.has("--cache-type-v") && caps.cache_types.iter().any(|t| t == "q8_0"),
            flash_attn: caps.has("--flash-attn"),
        }
    }
}

/// The KV cache type "Auto" picked, for the load panel.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutoKv {
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// Flash Attention was turned on for the quantized V cache.
    pub flash_attention_on: bool,
    /// KV cache bytes saved against f16 at the chosen context (estimate).
    pub saves_bytes: u64,
    /// Plain description for the UI.
    pub summary: String,
}

/// An f16 KV cache bigger than this is worth quantizing when the model has
/// to share VRAM with the CPU.
pub const AUTO_KV_Q8_THRESHOLD: u64 = 1 << 30;

/// The Auto KV cache rule: q8_0 when the f16 cache is over ~1 GiB and the
/// model shares the GPU with the CPU (Share with CPU on, or it does not fit
/// fully in VRAM), if the runtime takes q8_0. A q8_0 V cache needs Flash
/// Attention, so it is turned on; when the user turned Flash Attention off
/// (or the build lacks it), only K is quantized. Never q4_0.
/// Returns (K type, V type, turn Flash Attention on).
pub fn auto_kv_choice(cfg: &LoadConfig, f16_kv_bytes: u64, shares_with_cpu: bool, kv: KvSupport) -> Option<(&'static str, &'static str, bool)> {
    if !cfg.cache_type_auto || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) || !shares_with_cpu || f16_kv_bytes <= AUTO_KV_Q8_THRESHOLD || !kv.q8_0 {
        return None;
    }
    if kv.flash_attn && cfg.flash_attention != inference::FlashAttention::Off {
        Some(("q8_0", "q8_0", true))
    } else {
        Some(("q8_0", "f16", false))
    }
}

/// Resolve "Auto" KV cache type (`cache_type_auto`) to concrete types with
/// [`auto_kv_choice`], before the Auto context is sized (a q8_0 cache lets
/// Auto reach a larger context). The f16 size is measured at the trained
/// context for an Auto context, else at the configured one. Configs with an
/// explicit K/V choice are returned unchanged with None.
#[allow(clippy::too_many_arguments)]
pub fn resolve_auto_kv(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    uses_gpu: bool,
    kv: KvSupport,
) -> (LoadConfig, Option<AutoKv>) {
    if !cfg.cache_type_auto {
        return (cfg.clone(), None);
    }
    let mut c = cfg.clone();
    c.cache_type_k = "f16".into();
    c.cache_type_v = "f16".into();
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut input = estimate_input(&c, meta, total, projector_bytes);
    if cfg.n_ctx == 0 {
        input.n_ctx = meta.context_length.filter(|&t| t > 0).unwrap_or(estimate::AUTO_CTX_UNKNOWN_TRAINED);
    }
    input.gpu_layers = None;
    let full = estimate::estimate(meta, size, &input);
    let gpu = uses_gpu && !matches!(cfg.gpu_layers, GpuLayers::CpuOnly);
    let shares = gpu && (cfg.share_with_cpu || free_vram.is_some_and(|f| full.vram_bytes > (f as f64 * 0.90) as u64));
    let auto = match auto_kv_choice(&c, full.kv_cache_bytes, shares, kv) {
        Some((k, v, fa)) => {
            c.cache_type_k = k.into();
            c.cache_type_v = v.into();
            if fa {
                c.flash_attention = inference::FlashAttention::On;
            }
            AutoKv { cache_type_k: k.into(), cache_type_v: v.into(), flash_attention_on: fa, saves_bytes: 0, summary: String::new() }
        }
        None => AutoKv { cache_type_k: "f16".into(), cache_type_v: "f16".into(), flash_attention_on: false, saves_bytes: 0, summary: String::new() },
    };
    (c, Some(auto))
}

/// Fill in what the Auto KV choice saves at the final context of `cfg`
/// (the config after [`resolve_auto_kv`] and the Auto context).
pub fn finish_auto_kv(a: &mut AutoKv, cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64) {
    let total = meta.block_count.unwrap_or(0) + 1;
    let chosen = estimate_input(cfg, meta, total, projector_bytes);
    let f16 = EstimateInput { cache_type_k: "f16".into(), cache_type_v: "f16".into(), ..chosen.clone() };
    let kv = |i: &EstimateInput| estimate::estimate(meta, size, i).kv_cache_bytes;
    a.saves_bytes = kv(&f16).saturating_sub(kv(&chosen));
    let types = if a.cache_type_k == a.cache_type_v { a.cache_type_k.clone() } else { format!("{} K / {} V", a.cache_type_k, a.cache_type_v) };
    a.summary = if a.saves_bytes > 0 {
        format!("KV cache: {types} (auto, saves {:.1} GB)", a.saves_bytes as f64 / 1e9)
    } else {
        format!("KV cache: {types} (auto)")
    };
}

/// Generation / prompt threads for a hybrid CPU (P- and E-cores) when part
/// of the model runs on the CPU: generation on the P-cores plus 4 (measured
/// on a Core Ultra 7 265F, 8P+12E, Qwen3.8-27B split: -t 12 gave 5.79 t/s,
/// the default 20 gave 5.62, -t 8 5.04), prompt processing on every
/// physical core (-tb 20: 401 t/s, -tb 12: 373). None when everything is on
/// the GPU or the CPU is not hybrid: llama.cpp's defaults stay.
pub fn hybrid_cpu_threads(topo: Option<system_monitor::CoreTopology>, cpu_runs_layers: bool) -> Option<(u32, u32)> {
    let t = topo?;
    if !cpu_runs_layers || !t.is_hybrid() {
        return None;
    }
    let all = t.physical_cores();
    Some(((t.performance_cores + 4).min(all) as u32, all as u32))
}

/// Apply [`hybrid_cpu_threads`] to the thread counts the user left unset,
/// when the runtime has the flags.
pub fn apply_thread_policy(cfg: &LoadConfig, topo: Option<system_monitor::CoreTopology>, cpu_runs_layers: bool, caps: &inference::RuntimeCapabilities) -> LoadConfig {
    let mut c = cfg.clone();
    if let Some((t, tb)) = hybrid_cpu_threads(topo, cpu_runs_layers) {
        if c.threads.is_none() && caps.has("--threads") {
            c.threads = Some(t);
        }
        if c.threads_batch.is_none() && caps.has("--threads-batch") {
            c.threads_batch = Some(tb);
        }
    }
    c
}

/// Will part of the model (or its KV cache) run on the CPU with `cfg`?
/// `split` is the estimate's Share-with-CPU split, `full_gpu_fits` whether
/// the all-GPU estimate fits free VRAM (None without a reading).
pub fn cpu_runs_layers(cfg: &LoadConfig, total_layers: u64, uses_gpu: bool, split: Option<&ShareSplit>, full_gpu_fits: Option<bool>) -> bool {
    if !uses_gpu || !cfg.kv_offload || cfg.n_cpu_moe.is_some_and(|n| n > 0) {
        return true;
    }
    match cfg.gpu_layers {
        GpuLayers::CpuOnly => true,
        GpuLayers::Count(n) => (n as u64) < total_layers,
        GpuLayers::All => false,
        GpuLayers::Auto => match split {
            Some(s) => !s.all_fit,
            None => full_gpu_fits == Some(false),
        },
    }
}

#[tauri::command]
pub async fn estimate_memory(state: State<'_, SharedState>, id: String, config: LoadConfig) -> Cmd<EstimateView> {
    let m = state.db.get_model(&id).await?;
    let meta: ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    let hw = state.monitor.hardware_info();
    let gpu = hw.gpus.first();
    let total_layers = meta.block_count.unwrap_or(0) + 1;
    let proj = projector_bytes(&config, m.projector_path.as_deref());
    let free_vram = gpu.map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes));
    let rt = default_runtime(state.inner()).await;
    let has_moe_flag = rt.as_ref().is_some_and(|r| r.capabilities.has("--n-cpu-moe"));
    let kv_support = rt.as_ref().map(|r| KvSupport::of(&r.capabilities)).unwrap_or(KvSupport { q8_0: false, flash_attn: false });
    let share_fit = config.share_with_cpu && rt.as_ref().is_some_and(|r| runtime_can_fit(&r.capabilities));
    // Auto context, measured as a load would see it: a loaded model is
    // unloaded first, so its VRAM counts as free.
    let reclaim = loaded_backend_vram(state.inner());
    let free_at_load = free_vram.map(|f| f.saturating_add(reclaim).min(gpu.map(|g| g.vram_total_bytes).unwrap_or(u64::MAX)));
    let uses_gpu = !matches!(config.gpu_layers, GpuLayers::CpuOnly);
    // Auto KV cache type first: q8_0 lets the Auto context grow.
    let (config, mut auto_kv) = resolve_auto_kv(&config, &meta, m.size as u64, proj, free_at_load, uses_gpu, kv_support);
    let auto_ctx = auto_ctx_for(&config, &meta, m.size as u64, proj, free_at_load, hw.available_ram_bytes, has_moe_flag, uses_gpu);
    let mut input = estimate_input(&config, &meta, total_layers, proj);
    if config.n_ctx == 0 {
        input.n_ctx = auto_ctx.n_ctx;
    }
    if let Some(a) = auto_kv.as_mut() {
        let sized = LoadConfig { n_ctx: input.n_ctx.min(u32::MAX as u64) as u32, ..config.clone() };
        finish_auto_kv(a, &sized, &meta, m.size as u64, proj);
    }
    let est = estimate::estimate(&meta, m.size as u64, &input);
    let suggested = free_vram.map(|f| {
        estimate::max_gpu_layers_for_budget(&meta, m.size as u64, &input, f.saturating_sub(1 << 30))
    });
    let mut warnings = Vec::new();
    if let (Some(trained), true) = (meta.context_length, input.n_ctx > meta.context_length.unwrap_or(u64::MAX)) {
        warnings.push(format!(
            "Context {} is larger than the model's trained context ({trained}); quality may drop unless RoPE scaling is configured.",
            input.n_ctx
        ));
    }
    let mut fits = est.ram_bytes < hw.available_ram_bytes;
    if !fits {
        warnings.push("Estimated RAM use exceeds currently available RAM.".into());
    }
    // What "Share with CPU" would choose, and whether to suggest it.
    let share_split = free_vram.map(|f| {
        let mut all = input.clone();
        all.gpu_layers = None;
        estimate::share_split(&meta, m.size as u64, &all, f, has_moe_flag)
    });
    let full_gpu_too_big = share_split.as_ref().is_some_and(|s| !s.all_fit);
    let suggest_share = !config.share_with_cpu && !matches!(config.gpu_layers, GpuLayers::CpuOnly) && full_gpu_too_big;
    if config.share_with_cpu {
        if let Some(s) = &share_split {
            warnings.retain(|w| !w.contains("VRAM"));
            // What does not go to VRAM has to fit in RAM.
            let in_ram = (est.weights_bytes + est.kv_cache_bytes + est.projector_bytes).saturating_sub(s.vram_bytes);
            fits = in_ram < hw.available_ram_bytes;
            if !fits {
                warnings.push("Even when shared with the CPU, the part that does not fit in VRAM is more than the free RAM.".into());
            }
        }
    } else if let Some(f) = free_vram
        && est.vram_bytes > f
    {
        if matches!(config.gpu_layers, GpuLayers::Auto) {
            warnings.push("Not all layers fit in free VRAM; Auto will offload fewer layers.".into());
        } else {
            fits = false;
            warnings.push("Estimated VRAM use exceeds free VRAM; loading may fail with out of memory.".into());
        }
    }
    Ok(EstimateView {
        estimate: est,
        free_vram_bytes: free_vram,
        total_vram_bytes: gpu.map(|g| g.vram_total_bytes),
        available_ram_bytes: hw.available_ram_bytes,
        suggested_gpu_layers: suggested,
        fits,
        warnings,
        share_split,
        suggest_share,
        auto_ctx,
        auto_kv,
        share_fit,
    })
}

#[derive(Debug, Serialize)]
pub struct LoadDefaults {
    pub config: LoadConfig,
    /// Where the values came from: model_default | last_used | app_default
    pub source: String,
    pub has_projector: bool,
    pub trained_context: Option<u64>,
    pub total_layers: Option<u64>,
}

#[tauri::command]
pub async fn get_load_defaults(state: State<'_, SharedState>, id: String) -> Cmd<LoadDefaults> {
    let m = state.db.get_model(&id).await?;
    let meta: ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    let parse = |s: &Option<String>| s.as_deref().and_then(|t| serde_json::from_str::<LoadConfig>(t).ok());
    let (config, source) = if let Some(c) = parse(&m.default_config) {
        (c, "model_default")
    } else if let Some(c) = parse(&m.last_config) {
        (c, "last_used")
    } else {
        // No saved choice: Auto context and Auto KV cache type (both fit
        // the hardware at load time). Speculative decoding stays off.
        let c = LoadConfig { n_ctx: 0, cache_type_auto: true, ..LoadConfig::default() };
        (c, "app_default")
    };
    Ok(LoadDefaults {
        config,
        source: source.into(),
        has_projector: m.projector_path.is_some(),
        trained_context: meta.context_length,
        total_layers: meta.block_count.map(|b| b + 1),
    })
}

#[tauri::command]
pub async fn save_model_defaults(state: State<'_, SharedState>, id: String, config: LoadConfig) -> Cmd<()> {
    config.validate().map_err(AppError::InvalidInput)?;
    state.db.set_model_configs(&id, Some(&serde_json::to_value(&config)?), None).await
}

#[tauri::command]
pub async fn reset_model_defaults(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    state.db.clear_model_default_config(&id).await
}

#[cfg(test)]
mod perf_tests {
    use super::*;
    use inference::{FlashAttention, RuntimeCapabilities};
    use std::path::Path;
    use system_monitor::CoreTopology;

    const GIB: u64 = 1 << 30;

    fn dense() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("llama".into()),
            block_count: Some(32),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            vocab_size: Some(128_256),
            context_length: Some(131_072),
            ..Default::default()
        }
    }

    fn share(n_ctx: u32) -> LoadConfig {
        LoadConfig { n_ctx, share_with_cpu: true, ..Default::default() }
    }

    const HELP: &str = "\
-c,    --ctx-size N                     x
-t,    --threads N                      x
-tb,   --threads-batch N                x
-ngl,  --gpu-layers, --n-gpu-layers N   x
-fa,   --flash-attn [on|off|auto]       x
-ctk,  --cache-type-k TYPE              KV cache data type for K
                                        allowed values: f32, f16, bf16, q8_0, q4_0
-ctv,  --cache-type-v TYPE              x
-fit,  --fit [on|off]                   x
-fitt, --fit-target MiB0,MiB1,MiB2,...
-fitc, --fit-ctx N                      x
-ncmoe, --n-cpu-moe N                   x
";

    fn args(c: &LoadConfig, help: &str) -> Vec<String> {
        c.to_args(Path::new("m.gguf"), None, 1, &RuntimeCapabilities::parse(help)).args
    }

    fn val<'a>(a: &'a [String], f: &str) -> Option<&'a str> {
        a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).map(String::as_str)
    }

    #[test]
    fn share_with_fit_lets_llama_cpp_pick_the_layers_at_our_context() {
        // 16 GB dense model, 8 GiB free: the estimate splits it, llama.cpp fits it.
        let (c, s) = resolve_share(&share(32_768), &dense(), 16_000_000_000, 0, Some(8 * GIB), true, true);
        let s = s.expect("preview split");
        assert!(!s.all_fit && s.gpu_layers > 0 && s.gpu_layers < 33, "{s:?}");
        assert_eq!(c.gpu_layers, GpuLayers::Auto);
        assert_eq!((c.fit, c.fit_target_mib, c.fit_ctx, c.n_ctx, c.n_cpu_moe), (true, Some(1024), Some(32_768), 32_768, None));
        let a = args(&c, HELP);
        assert!(!a.iter().any(|x| x == "--n-gpu-layers"), "{a:?}");
        assert_eq!((val(&a, "--fit"), val(&a, "--fit-target"), val(&a, "--ctx-size"), val(&a, "--fit-ctx")), (Some("on"), Some("1024"), Some("32768"), Some("32768")));
        // The config's margin (the benchmark's A/B) wins over the default.
        let (c, _) = resolve_share(&LoadConfig { fit_target_mib: Some(512), ..share(32_768) }, &dense(), 16_000_000_000, 0, Some(8 * GIB), true, true);
        assert_eq!(val(&args(&c, HELP), "--fit-target"), Some("512"));
        // A model that fits is fitted the same way (llama.cpp keeps it all on the GPU).
        let (c, s) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, Some(12 * GIB), true, true);
        assert!(s.unwrap().all_fit && c.gpu_layers == GpuLayers::Auto && c.fit_target_mib == Some(1024));
        // No VRAM reading: fitted too.
        let (c, s) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, None, true, true);
        assert!(s.is_none() && c.gpu_layers == GpuLayers::Auto && c.fit_ctx == Some(4096));
        // Off: untouched.
        let off = LoadConfig { n_ctx: 4096, ..Default::default() };
        assert_eq!(resolve_share(&off, &dense(), 16_000_000_000, 0, Some(8 * GIB), true, true).0, off);
    }

    #[test]
    fn share_without_fit_flags_falls_back_to_the_estimated_layer_count() {
        let (c, s) = resolve_share(&share(32_768), &dense(), 16_000_000_000, 0, Some(8 * GIB), true, false);
        let s = s.unwrap();
        assert_eq!(c.gpu_layers, GpuLayers::Count(s.gpu_layers as u32));
        assert!(c.fit_target_mib.is_none() && c.fit_ctx.is_none());
        let old_help = "-c, --ctx-size N x\n-ngl,  --gpu-layers, --n-gpu-layers N   x\n-fit,  --fit [on|off]                   x\n";
        assert!(!runtime_can_fit(&RuntimeCapabilities::parse(old_help)) && runtime_can_fit(&RuntimeCapabilities::parse(HELP)));
        assert_eq!(val(&args(&c, old_help), "--n-gpu-layers"), Some(s.gpu_layers.to_string().as_str()));
        // Fits fully: all layers, as before.
        let (c, _) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, Some(12 * GIB), true, false);
        assert_eq!(c.gpu_layers, GpuLayers::All);
    }

    #[test]
    fn share_moe_path_is_unchanged() {
        let moe = ModelMetadata { expert_count: Some(64), ..dense() };
        for can_fit in [true, false] {
            let (c, s) = resolve_share(&share(8192), &moe, 16_000_000_000, 0, Some(8 * GIB), true, can_fit);
            let n = s.unwrap().n_cpu_moe.expect("moe split");
            assert_eq!((c.gpu_layers, c.n_cpu_moe, c.fit_target_mib), (GpuLayers::All, Some(n as u32), None), "can_fit {can_fit}");
            assert_eq!(val(&args(&c, HELP), "--n-cpu-moe"), Some(n.to_string().as_str()));
        }
        // Nothing fits at all: CPU only, as before.
        let (c, _) = resolve_share(&share(8192), &dense(), 16_000_000_000, 0, Some(0), true, true);
        assert_eq!(c.gpu_layers, GpuLayers::CpuOnly);
    }

    const KV: KvSupport = KvSupport { q8_0: true, flash_attn: true };

    #[test]
    fn auto_kv_q8_rules() {
        let auto = LoadConfig { n_ctx: 0, cache_type_auto: true, ..Default::default() };
        // Big cache, shared with the CPU: q8_0 for K and V with Flash Attention on.
        assert_eq!(auto_kv_choice(&auto, 2 * GIB, true, KV), Some(("q8_0", "q8_0", true)));
        // Small cache, fully on the GPU, or no q8_0 in the build: f16 stays.
        assert_eq!(auto_kv_choice(&auto, GIB, true, KV), None);
        assert_eq!(auto_kv_choice(&auto, 2 * GIB, false, KV), None);
        assert_eq!(auto_kv_choice(&auto, 2 * GIB, true, KvSupport { q8_0: false, flash_attn: true }), None);
        // Flash Attention off (or missing): only K is quantized.
        let fa_off = LoadConfig { flash_attention: FlashAttention::Off, ..auto.clone() };
        assert_eq!(auto_kv_choice(&fa_off, 2 * GIB, true, KV), Some(("q8_0", "f16", false)));
        assert_eq!(auto_kv_choice(&auto, 2 * GIB, true, KvSupport { q8_0: true, flash_attn: false }), Some(("q8_0", "f16", false)));
        // The user's explicit type always wins, and CPU only is left alone.
        let explicit = LoadConfig { cache_type_auto: false, ..auto.clone() };
        assert_eq!(auto_kv_choice(&explicit, 8 * GIB, true, KV), None);
        assert_eq!(auto_kv_choice(&LoadConfig { gpu_layers: GpuLayers::CpuOnly, ..auto.clone() }, 8 * GIB, true, KV), None);
    }

    #[test]
    fn auto_kv_on_the_27b_reaches_a_larger_context() {
        // Qwen3.8-27B Q4_K_M, 7 GiB free VRAM, 28 GiB available RAM, Share with CPU, Auto context.
        let m = ModelMetadata {
            architecture: Some("qwen35".into()),
            block_count: Some(65),
            context_length: Some(262_144),
            embedding_length: Some(5120),
            head_count: Some(24),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            vocab_size: Some(248_320),
            mtp_layers: Some(1),
            full_attention_interval: Some(4),
            ..Default::default()
        };
        let size = 16_810_714_336;
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_auto: true, ..Default::default() };
        let (c, a) = resolve_auto_kv(&cfg, &m, size, 0, Some(7 * GIB), true, KV);
        assert_eq!((c.cache_type_k.as_str(), c.cache_type_v.as_str(), c.flash_attention), ("q8_0", "q8_0", FlashAttention::On));
        let (sized, _) = resolve_auto_ctx(&c, &m, size, 0, Some(7 * GIB), 28 * GIB, true, true);
        let f16 = LoadConfig { cache_type_auto: false, ..cfg.clone() };
        let (sized16, _) = resolve_auto_ctx(&f16, &m, size, 0, Some(7 * GIB), 28 * GIB, true, true);
        assert!(sized.n_ctx >= sized16.n_ctx, "q8 {} vs f16 {}", sized.n_ctx, sized16.n_ctx);
        assert!(sized.n_ctx > 131_072, "{}", sized.n_ctx);
        let mut a = a.unwrap();
        finish_auto_kv(&mut a, &sized, &m, size, 0);
        assert!(a.saves_bytes > GIB && a.summary.starts_with("KV cache: q8_0 (auto, saves "), "{a:?}");
        // Manual KV type: untouched.
        let manual = LoadConfig { cache_type_auto: false, cache_type_k: "f16".into(), ..cfg.clone() };
        assert_eq!(resolve_auto_kv(&manual, &m, size, 0, Some(7 * GIB), true, KV), (manual.clone(), None));
        // Small model fully on the GPU: f16 (auto).
        let small = LoadConfig { n_ctx: 8192, cache_type_auto: true, ..Default::default() };
        let (c, a) = resolve_auto_kv(&small, &dense(), 4_900_000_000, 0, Some(12 * GIB), true, KV);
        assert_eq!((c.cache_type_k.as_str(), a.unwrap().cache_type_k.as_str()), ("f16", "f16"));
    }

    #[test]
    fn hybrid_cpu_thread_policy() {
        let ultra7 = CoreTopology { performance_cores: 8, efficiency_cores: 12 };
        assert_eq!(hybrid_cpu_threads(Some(ultra7), true), Some((12, 20)));
        // Everything on the GPU, a non-hybrid CPU or no reading: defaults.
        assert_eq!(hybrid_cpu_threads(Some(ultra7), false), None);
        assert_eq!(hybrid_cpu_threads(Some(CoreTopology { performance_cores: 8, efficiency_cores: 0 }), true), None);
        assert_eq!(hybrid_cpu_threads(None, true), None);
        // 6P + 2E: P + 4 capped at the physical cores.
        assert_eq!(hybrid_cpu_threads(Some(CoreTopology { performance_cores: 6, efficiency_cores: 2 }), true), Some((8, 8)));
        let caps = RuntimeCapabilities::parse(HELP);
        let c = apply_thread_policy(&LoadConfig::default(), Some(ultra7), true, &caps);
        assert_eq!((c.threads, c.threads_batch), (Some(12), Some(20)));
        let a = args(&c, HELP);
        assert_eq!((val(&a, "--threads"), val(&a, "--threads-batch")), (Some("12"), Some("20")));
        // The user's own counts win.
        let mine = apply_thread_policy(&LoadConfig { threads: Some(6), ..Default::default() }, Some(ultra7), true, &caps);
        assert_eq!((mine.threads, mine.threads_batch), (Some(6), Some(20)));
        // A build without the flags gets nothing (no "unsupported" noise).
        let none = apply_thread_policy(&LoadConfig::default(), Some(ultra7), true, &RuntimeCapabilities::parse("-c, --ctx-size N x\n"));
        assert_eq!((none.threads, none.threads_batch), (None, None));
    }

    #[test]
    fn which_loads_run_layers_on_the_cpu() {
        let split = |all_fit| ShareSplit { gpu_layers: 16, total_layers: 33, n_cpu_moe: None, all_fit, vram_bytes: 0, summary: String::new() };
        let auto = LoadConfig::default();
        assert!(cpu_runs_layers(&auto, 33, true, Some(&split(false)), None));
        assert!(!cpu_runs_layers(&auto, 33, true, Some(&split(true)), None));
        assert!(!cpu_runs_layers(&auto, 33, true, None, Some(true)));
        assert!(cpu_runs_layers(&auto, 33, true, None, Some(false)));
        assert!(!cpu_runs_layers(&auto, 33, true, None, None));
        assert!(!cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::All, ..auto.clone() }, 33, true, None, None));
        assert!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::Count(20), ..auto.clone() }, 33, true, None, None));
        assert!(!cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::Count(33), ..auto.clone() }, 33, true, None, None));
        assert!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::CpuOnly, ..auto.clone() }, 33, true, None, None));
        assert!(cpu_runs_layers(&LoadConfig { n_cpu_moe: Some(4), gpu_layers: GpuLayers::All, ..auto.clone() }, 33, true, None, None));
        assert!(cpu_runs_layers(&LoadConfig { kv_offload: false, ..auto.clone() }, 33, true, None, Some(true)));
        assert!(cpu_runs_layers(&auto, 33, false, None, None), "CPU runtime");
    }
}

#[cfg(test)]
mod auto_ctx_tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn meta() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("llama".into()),
            block_count: Some(32),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            vocab_size: Some(128_256),
            context_length: Some(131_072),
            ..Default::default()
        }
    }

    #[test]
    fn auto_ctx_is_resolved_before_launch_and_manual_is_untouched() {
        let auto = LoadConfig { n_ctx: 0, gpu_layers: GpuLayers::All, ..Default::default() };
        let (c, a) = resolve_auto_ctx(&auto, &meta(), 4_900_000_000, 0, Some(6 * GIB), 32 * GIB, true, true);
        let a = a.expect("auto resolved");
        assert_eq!((c.n_ctx, a.n_ctx), (4096, 4096), "{a:?}");
        assert!(c.validate().is_ok() && c.n_ctx != 0, "0 never reaches llama-server");
        // Bigger GPU: up to the trained context.
        let (c, _) = resolve_auto_ctx(&auto, &meta(), 4_900_000_000, 0, Some(24 * GIB), 64 * GIB, true, true);
        assert_eq!(c.n_ctx, 131_072);
        // CPU runtime: sized by RAM, VRAM ignored.
        let (c, a) = resolve_auto_ctx(&auto, &meta(), 4_900_000_000, 0, Some(0), 16 * GIB, true, false);
        assert!(c.n_ctx >= 65_536 && a.unwrap().fits, "{c:?}");
        // A projector counts.
        let (c, _) = resolve_auto_ctx(&auto, &meta(), 4_900_000_000, 400_000_000, Some(6 * GIB), 32 * GIB, true, true);
        assert!(c.n_ctx < 4096);
        // A manual size is kept as is.
        let manual = LoadConfig { n_ctx: 12_288, ..auto.clone() };
        let (c, a) = resolve_auto_ctx(&manual, &meta(), 4_900_000_000, 0, Some(6 * GIB), 32 * GIB, true, true);
        assert!(a.is_none() && c == manual);
        // Share with CPU: RAM counts, so a big model gets more than the minimum.
        let share = LoadConfig { share_with_cpu: true, ..auto };
        let (c, _) = resolve_auto_ctx(&share, &meta(), 16_000_000_000, 0, Some(8 * GIB), 32 * GIB, true, true);
        assert!(c.n_ctx > estimate::AUTO_CTX_MIN as u32, "{c:?}");
    }
}

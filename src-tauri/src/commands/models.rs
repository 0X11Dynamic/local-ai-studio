use std::path::{Path, PathBuf};

use app_core::AppError;
use inference::{GpuLayers, LoadConfig};
use model_registry::estimate::{self, AutoCtx, AutoCtxEnv, EstimateInput, LinkedDev, MemoryEstimate, ShareSplit};
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
    /// The `--fit-target` margin such a load passes (MiB), when `share_fit`.
    pub fit_target_mib: Option<u32>,
    /// RAM kept free for the system and other apps ([`ram_reserve`]).
    pub ram_reserve_bytes: u64,
    /// "Use linked PCs" is on and they lend memory: the Auto context this PC
    /// would get alone (the `auto_ctx` above counts the linked PCs).
    pub auto_ctx_alone: Option<u64>,
    /// What each linked PC lends this load (from their last info).
    pub linked_contrib: Vec<crate::services::linking::LinkedShare>,
}

/// Inputs for an Auto context ([`estimate::auto_n_ctx`]) for a load with `cfg`.
/// `meta` is the one the load uses ([`spec_meta`]); `uses_gpu` is false for
/// CPU-only runtimes.
#[cfg(test)]
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
    auto_ctx_linked(cfg, meta, size, projector_bytes, free_vram, available_ram, has_n_cpu_moe, uses_gpu, &[])
}

/// [`auto_ctx_for`] counting what linked PCs lend ([`estimate::LinkedDev`]).
#[allow(clippy::too_many_arguments)]
pub fn auto_ctx_linked(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    available_ram: u64,
    has_n_cpu_moe: bool,
    uses_gpu: bool,
    linked: &[LinkedDev],
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
        linked,
    };
    estimate::auto_n_ctx_with(meta, size, &input, &env, estimate::AutoCtxOpts { weights_may_page: weights_pageable(cfg) && meta.expert_count.unwrap_or(0) > 0 })
}

/// The weights are memory-mapped (load mode Auto or mmap, not locked or
/// read into memory), so Windows can page what does not fit in RAM from the
/// file instead of the load failing.
pub fn weights_pageable(cfg: &LoadConfig) -> bool {
    matches!(cfg.load_mode, inference::LoadMode::Auto | inference::LoadMode::Mmap)
}

/// Replace an Auto context (n_ctx 0) with the size [`auto_ctx_for`] picks.
/// Other configs are returned unchanged, with None.
#[cfg(test)]
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
    resolve_auto_ctx_linked(cfg, meta, size, projector_bytes, free_vram, available_ram, has_n_cpu_moe, uses_gpu, &[])
}

/// [`resolve_auto_ctx`] counting what linked PCs lend.
#[allow(clippy::too_many_arguments)]
pub fn resolve_auto_ctx_linked(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    available_ram: u64,
    has_n_cpu_moe: bool,
    uses_gpu: bool,
    linked: &[LinkedDev],
) -> (LoadConfig, Option<AutoCtx>) {
    if cfg.n_ctx != 0 {
        return (cfg.clone(), None);
    }
    let a = auto_ctx_linked(cfg, meta, size, projector_bytes, free_vram, available_ram, has_n_cpu_moe, uses_gpu, linked);
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

/// The runtime `id` names when it is installed, else the default runtime
/// (what a load without a runtime id uses).
pub async fn runtime_or_default(st: &SharedState, id: Option<&str>) -> Option<inference::InstalledRuntime> {
    if let Some(i) = id
        && let Some(r) = st.runtimes.list_installed().into_iter().find(|r| r.id == i)
    {
        return Some(r);
    }
    default_runtime(st).await
}

/// Can the runtime let llama.cpp fit the GPU layers itself with an explicit
/// margin (`--fit` and `--fit-target`)?
pub fn runtime_can_fit(caps: &inference::RuntimeCapabilities) -> bool {
    caps.has("--fit") && caps.has("--fit-target")
}

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The `--fit-target` margin (MiB) for a load llama.cpp fits: `base` (the
/// config's, else llama.cpp's 1024) plus the vision projector and the MTP
/// layer's KV cache. llama-server b12292's `--help` describes the margin only
/// as free memory kept per device and says nothing about counting either
/// (the old `-ngl` path budgeted both), so they are added. At most the
/// 65536 MiB `LoadConfig::validate` allows.
pub fn fit_margin_mib(base: u32, projector_bytes: u64, _mtp_kv_bytes: u64) -> u32 {
    // MTP: llama.cpp's --fit already counts the draft-mtp context (measured
    // 2026-11-12 on b12292: projected device memory 21742 MiB with MTP vs
    // 20334 MiB without), so adding it here double-counted it and could flip
    // speculative decoding off near a context step. The vision projector is
    // loaded separately (mtmd) and is still added.
    let extra = projector_bytes.div_ceil(MIB);
    (base as u64).saturating_add(extra).min(65_536) as u32
}

/// Turn the user's "Share with CPU" choice into concrete load settings.
/// Returns the config to load with (the user's own config is unchanged when
/// the toggle is off) and the split the estimate predicts.
///
/// When the estimate puts every layer on the GPU with the fit margin to
/// spare, all layers are offloaded (`-ngl all`), as before `--fit` was used:
/// fitting is only for loads that are split or tight. Then, with `can_fit`
/// (the runtime has `--fit` and `--fit-target`), llama.cpp picks the layer
/// count itself from real allocation sizes: no `-ngl`, `--fit on`, an
/// explicit `--fit-target` margin ([`fit_margin_mib`] of the config's margin,
/// else 1024 MiB) and `--fit-ctx` equal to the chosen context, which is also
/// passed explicitly, so fitting never lowers the context. The estimate's
/// split is then only the UI preview. MoE models the estimate splits keep
/// every layer on the GPU with `--n-cpu-moe` (unchanged), and when nothing
/// fits at all the load is CPU only. Without `can_fit`, the estimate's layer
/// count is passed (`-ngl N`) as before.
/// `meta` is the one the load uses ([`spec_meta`]); `projector_bytes` is what
/// [`projector_bytes`] returns for this load and `mtp_kv_bytes` the MTP
/// layer's KV cache in VRAM (0 without speculative MTP).
#[allow(clippy::too_many_arguments)]
pub fn resolve_share(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    mtp_kv_bytes: u64,
    free_vram: Option<u64>,
    has_n_cpu_moe: bool,
    can_fit: bool,
) -> (LoadConfig, Option<ShareSplit>) {
    if !cfg.share_with_cpu || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) {
        return (cfg.clone(), None);
    }
    let mut c = cfg.clone();
    c.fit = true;
    let base = cfg.fit_target_mib.unwrap_or(inference::DEFAULT_FIT_TARGET_MIB);
    let margin = fit_margin_mib(base, projector_bytes, mtp_kv_bytes);
    let let_llama_fit = |c: &mut LoadConfig| {
        c.gpu_layers = GpuLayers::Auto;
        c.n_cpu_moe = None;
        if can_fit {
            c.fit_target_mib = Some(margin);
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
    // Everything fits and the margin is left over too (the estimate already
    // holds the projector and the MTP KV cache).
    let roomy = split.all_fit && split.vram_bytes.saturating_add(base as u64 * MIB) <= free;
    if let Some(n) = split.n_cpu_moe {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = Some(n as u32);
    } else if split.gpu_layers == 0 && !split.all_fit {
        c.gpu_layers = GpuLayers::CpuOnly;
        c.n_cpu_moe = None;
    } else if roomy || (split.all_fit && !can_fit) {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = None;
    } else if can_fit {
        let_llama_fit(&mut c);
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
    /// The Auto context with an f16 cache, when q8_0 was chosen because it
    /// lets a model that runs fully on the GPU reach a larger step.
    pub raised_from: Option<u64>,
    /// Plain description for the UI.
    pub summary: String,
}

impl AutoKv {
    fn new(k: &str, v: &str, fa: bool) -> Self {
        Self { cache_type_k: k.into(), cache_type_v: v.into(), flash_attention_on: fa, saves_bytes: 0, raised_from: None, summary: String::new() }
    }
}

/// An f16 KV cache bigger than this is worth quantizing when the model is
/// split between the GPU and the CPU.
pub const AUTO_KV_Q8_THRESHOLD: u64 = 1 << 30;

/// The q8_0 rule for one f16 cache size: q8_0 when the f16 cache is over
/// ~1 GiB and `shares_with_cpu`, if the runtime takes q8_0. A q8_0 V cache
/// needs Flash Attention, so it is turned on; when the user turned Flash
/// Attention off (or the build lacks it), only K is quantized. Never q4_0.
/// Returns (K type, V type, turn Flash Attention on).
pub fn auto_kv_choice(cfg: &LoadConfig, f16_kv_bytes: u64, shares_with_cpu: bool, kv: KvSupport) -> Option<(&'static str, &'static str, bool)> {
    if !cfg.cache_type_auto || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) || !shares_with_cpu || f16_kv_bytes <= AUTO_KV_Q8_THRESHOLD || !kv.q8_0 {
        return None;
    }
    Some(q8_types(cfg, kv))
}

/// The q8_0 K / V types the runtime and the config allow: V too (with Flash
/// Attention turned on) unless Flash Attention is off or missing, then K only.
fn q8_types(cfg: &LoadConfig, kv: KvSupport) -> (&'static str, &'static str, bool) {
    if kv.flash_attn && cfg.flash_attention != inference::FlashAttention::Off { ("q8_0", "q8_0", true) } else { ("q8_0", "f16", false) }
}

/// For a model that runs fully on the GPU with an f16 cache at the Auto
/// context `f16` (`base`: the Auto config with f16 types): the q8_0 types and
/// the larger Auto context they reach, still fully on the GPU, when q8_0
/// reaches a larger step (the user's decision: ~3% slower, measured Gemma
/// 94 -> 91 t/s, for more context). None when q8_0 gives no larger step, f16
/// already reaches the trained context, the f16 load does not fit on the GPU
/// by the estimate, or the runtime lacks q8_0. Split loads never get here:
/// their context stays sized with f16.
pub fn full_gpu_q8_raise(base: &LoadConfig, f16: &AutoCtx, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> Option<(&'static str, &'static str, bool, AutoCtx)> {
    let kv = KvSupport::of(env.caps);
    let free = env.free_vram?;
    if !base.cache_type_auto || !kv.q8_0 || !env.uses_gpu || f16.limited_by == "trained" || f16.pages_weights || matches!(base.gpu_layers, GpuLayers::CpuOnly | GpuLayers::Count(_)) {
        return None;
    }
    let total = meta.block_count.unwrap_or(0) + 1;
    // The f16 load at its Auto context must be fully on the GPU by the estimate.
    let mut all = estimate_input(&LoadConfig { n_ctx: f16.n_ctx as u32, ..base.clone() }, meta, total, projector_bytes);
    all.gpu_layers = None;
    // Linked PCs' GPUs count as GPU memory (net of their compute buffers).
    let e = estimate::estimate(meta, size, &all);
    let gpu_budget = ((free as f64 * 0.90) as u64).saturating_add(linked_net(env.linked, e.compute_buffer_bytes));
    if e.vram_bytes > gpu_budget {
        return None;
    }
    let (k, v, fa) = q8_types(base, kv);
    // Sized fully on the GPU (no CPU split, no paging).
    let q8 = LoadConfig { cache_type_k: k.into(), cache_type_v: v.into(), gpu_layers: GpuLayers::All, share_with_cpu: false, ..base.clone() };
    let a = auto_ctx_linked(&q8, meta, size, projector_bytes, env.free_vram, env.available_ram, env.caps.has("--n-cpu-moe"), env.uses_gpu, env.linked);
    (a.fits && !a.pages_weights && a.n_ctx > f16.n_ctx).then_some((k, v, fa, a))
}

/// Is a load with `cfg` (context already chosen) known to be split between
/// the GPU and the CPU? From the config when it says so, else from what
/// llama.cpp did last time ([`PlanEnv::observed`]), else only when the
/// weights that can go to the GPU (`size`: the file minus the tensors
/// llama.cpp keeps on the host, [`estimate::offloadable_weight_bytes`]) plus
/// the projector are bigger than free VRAM. The all-GPU estimate still counts
/// the token embeddings, which llama.cpp keeps on the CPU, so it is not
/// trusted to call a model split.
fn runs_split(cfg: &LoadConfig, size: u64, projector_bytes: u64, total_layers: u64, compute_buffer_bytes: u64, env: &PlanEnv) -> bool {
    // A linked PC's CPU holds layers: split with a CPU (f16 Auto sizing).
    if cfg.n_cpu_moe.is_some_and(|n| n > 0) || env.linked.iter().any(|d| d.cpu && d.bytes > 0) {
        return true;
    }
    if !cfg.share_with_cpu {
        match cfg.gpu_layers {
            GpuLayers::Count(n) => return (n as u64) < total_layers,
            GpuLayers::All | GpuLayers::CpuOnly => return false,
            GpuLayers::Auto => {}
        }
    }
    if let Some((n, m)) = (env.observed)(cfg) {
        return n < m;
    }
    weights_over_vram(size, projector_bytes, pooled_free(env, compute_buffer_bytes))
}

/// The weights fit the GPU but weights + KV cache + buffers at this context
/// do not (by the same estimate [`ram_side`] uses for a "Share with CPU" or
/// GPU-offload-Auto load): llama.cpp then keeps layers (and their KV) on the
/// CPU, so the load is split even though [`runs_split`]'s weights test says
/// otherwise. Without this, Auto context dipped at free-VRAM sizes just above
/// the weights: the f16 KV was sized in RAM instead of q8_0 as at the
/// neighbouring sizes (27B, 23 GB RAM: 15 GiB 262144, 16 GiB 196608).
fn kv_overflows_gpu(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> bool {
    let Some(free) = env.free_vram else { return false };
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut all = estimate_input(cfg, meta, total, projector_bytes);
    all.gpu_layers = None;
    let full = estimate::estimate(meta, size, &all);
    let free = split_free(env, free, full.compute_buffer_bytes);
    // The all-GPU estimate counts the token embeddings llama.cpp keeps on
    // the CPU (Gemma-4-E4B fits 7 GiB at 131072 in practice), so use the
    // offloadable weights plus the KV cache and buffers, and call the load split
    // only when over 1 GiB of it would not fit (the size from which Auto
    // quantizes the KV cache anyway; Gemma at 7 GiB spills ~0.6 GB and keeps f16).
    let gpu_bytes = estimate::offloadable_weight_bytes(meta, size).saturating_add(projector_bytes).saturating_add(full.kv_cache_bytes).saturating_add(full.compute_buffer_bytes);
    (cfg.share_with_cpu || matches!(cfg.gpu_layers, GpuLayers::Auto)) && gpu_bytes.saturating_sub(free) > AUTO_KV_Q8_THRESHOLD
}

/// Linked PCs whose every device nets 0 at this compute buffer: the load
/// leaves them out of `--rpc`, so the load panel says so before the load
/// ([`crate::services::linking::left_out_note`]).
pub fn linked_left_out_notes(shares: &[crate::services::linking::LinkedShare], compute_buffer_bytes: u64) -> Vec<String> {
    let mut peers: Vec<&str> = shares.iter().map(|s| s.peer.as_str()).collect();
    peers.dedup();
    peers
        .into_iter()
        .filter_map(|name| {
            let mine: Vec<&crate::services::linking::LinkedShare> = shares.iter().filter(|s| s.peer == name).collect();
            if mine.iter().any(|s| estimate::linked_net(&LinkedDev { bytes: s.bytes, cpu: s.cpu }, compute_buffer_bytes) > 0) {
                return None;
            }
            let lent = mine.iter().map(|s| s.bytes).fold(0u64, |a, b| a.saturating_add(b));
            let cap = [false, true]
                .into_iter()
                .filter_map(|cpu| mine.iter().filter(|s| s.cpu == cpu).filter_map(|s| s.cap_bytes).max())
                .reduce(|a, b| a.saturating_add(b));
            Some(crate::services::linking::left_out_note(name, lent, cap, compute_buffer_bytes))
        })
        .collect()
}

/// Net bytes linked devices add at a context with this compute buffer
/// ([`estimate::linked_net`]).
fn linked_net(linked: &[LinkedDev], compute_buffer_bytes: u64) -> u64 {
    linked.iter().map(|d| estimate::linked_net(d, compute_buffer_bytes)).fold(0u64, |a, b| a.saturating_add(b))
}

/// Free VRAM here plus what linked PCs can hold, counted as [`split_free`]
/// counts it (the same rule decides "split" and sizes the split).
fn pooled_free(env: &PlanEnv, compute_buffer_bytes: u64) -> Option<u64> {
    env.free_vram.map(|f| split_free(env, f, compute_buffer_bytes))
}

/// Free VRAM to size a split with: this PC's plus linked PCs' net capacity
/// at a context with this compute buffer, as if it were VRAM here.
fn split_free(env: &PlanEnv, free: u64, compute_buffer_bytes: u64) -> u64 {
    free.saturating_add((linked_net(env.linked, compute_buffer_bytes) as f64 / 0.90) as u64)
}

/// The weights alone (plus the projector) are bigger than free VRAM: some
/// layers certainly run on the CPU.
fn weights_over_vram(size: u64, projector_bytes: u64, free_vram: Option<u64>) -> bool {
    free_vram.is_some_and(|f| size.saturating_add(projector_bytes) > f)
}

/// RAM kept free for the system and other apps when Auto sizes the context:
/// 15% of total RAM, at least 4 GiB. Measured: the 27B at 262144 (q8_0 KV)
/// left 0.45-1.0 GB free on a 31.6 GB PC and cold loads took 30-90 minutes
/// of paging.
pub fn ram_reserve(total_ram: u64) -> u64 {
    (4 * GIB).max(total_ram / 100 * 15)
}

/// RAM a load uses outside the GPU, split into the memory-mapped weights
/// and everything else (CPU-side KV cache, projector on the CPU, compute
/// buffers), which depends on the context and cannot be paged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamSide {
    pub weights: u64,
    pub other: u64,
}

impl RamSide {
    pub fn total(&self) -> u64 {
        self.weights.saturating_add(self.other)
    }
}

/// Split an estimate's RAM into weights and the rest.
fn ram_side_of(e: &MemoryEstimate) -> RamSide {
    let weights = e.weights_ram_bytes.min(e.ram_bytes);
    RamSide { weights, other: e.ram_bytes - weights }
}

/// [`ram_side_bytes`], split into weights and the rest ([`RamSide`]).
pub fn ram_side(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> RamSide {
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut i = estimate_input(cfg, meta, total, projector_bytes);
    if !env.uses_gpu || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) {
        i.gpu_layers = Some(0);
        return ram_side_of(&estimate::estimate(meta, size, &i));
    }
    let Some(free) = env.free_vram else { return ram_side_of(&estimate::estimate(meta, size, &i)) };
    let mut all = i.clone();
    all.gpu_layers = None;
    let full = estimate::estimate(meta, size, &all);
    // Share with CPU, or GPU offload Auto with a model too big for the GPU:
    // the split "Share with CPU" makes; the rest is in RAM.
    let free = split_free(env, free, full.compute_buffer_bytes);
    let split = cfg.share_with_cpu || (matches!(cfg.gpu_layers, GpuLayers::Auto) && full.vram_bytes > (free as f64 * 0.90) as u64);
    if !split {
        return ram_side_of(&estimate::estimate(meta, size, &i));
    }
    let s = estimate::share_split(meta, size, &all, free, env.caps.has("--n-cpu-moe"));
    let everything = full.weights_bytes.saturating_add(full.kv_cache_bytes).saturating_add(full.projector_bytes).saturating_add(full.compute_buffer_bytes);
    let ram = everything.saturating_sub(s.vram_bytes).saturating_add(256 * MIB);
    let weights = full.weights_bytes.saturating_sub(s.weights_vram_bytes).min(ram);
    RamSide { weights, other: ram - weights }
}

/// RAM a load with `cfg` (context set) uses outside the GPU (estimate): the
/// CPU-side weights, the CPU-side KV cache, the projector when it is on the
/// CPU, and the host compute buffers. Memory-mapped weights count: they are
/// paged in from the file and take RAM like anything else.
pub fn ram_side_bytes(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> u64 {
    ram_side(cfg, meta, size, projector_bytes, env).total()
}

/// How the RAM side of a load compares with available RAM minus [`ram_reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamFit {
    /// Weights and everything else fit.
    Fits,
    /// The memory-mapped weights do not fit, the rest does: Windows pages the
    /// weights from the file (slower; workable for mixture-of-experts models,
    /// which read only the active experts per token).
    PagesWeights,
    /// Even without the weights it does not fit (or the weights cannot page).
    TooBig,
}

pub fn ram_fit(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> RamFit {
    let budget = env.available_ram.saturating_sub(ram_reserve(env.total_ram));
    let r = ram_side(cfg, meta, size, projector_bytes, env);
    if r.total() <= budget {
        RamFit::Fits
    } else if weights_pageable(cfg) && meta.expert_count.unwrap_or(0) > 0 && r.other <= budget {
        // Dense models never page: every token reads all CPU-side weights.
        RamFit::PagesWeights
    } else {
        RamFit::TooBig
    }
}

/// The warning for a dense model whose CPU-side weights do not fit RAM (they
/// never page: every token reads all of them).
pub fn not_enough_ram_warning(in_ram: u64, ram_budget: u64, reserve: u64) -> String {
    format!(
        "Not enough RAM for this model with the given GPU memory: the {:.1} GB that does not fit on the GPU must stay in RAM (every token reads all of it), but only {:.1} GB of the available RAM is usable after keeping {:.1} GB free. Free up RAM, use a smaller quant, or give it more GPU memory.",
        in_ram as f64 / 1e9,
        ram_budget as f64 / 1e9,
        reserve as f64 / 1e9
    )
}

/// The warning for a load whose weights page ([`RamFit::PagesWeights`]).
pub fn paging_warning(meta: &ModelMetadata) -> String {
    if meta.expert_count.unwrap_or(0) > 0 {
        "This model is larger than free RAM; Windows will page it from disk, which is slower. It is a mixture-of-experts model, so only the active experts are read for each token and it stays usable.".into()
    } else {
        "This model is larger than free RAM; Windows will page it from disk, which is much slower: every token reads all of its weights.".into()
    }
}

/// Keep [`ram_reserve`] free: lower an Auto context, step by step, until the
/// RAM side of the load ([`ram_side_bytes`], with the KV type it really
/// uses) fits in available RAM minus the reserve. The largest step that
/// fits is kept (more context is preferred). Never raises the context.
/// When no step fits because the memory-mapped weights alone are too big
/// ([`RamFit::PagesWeights`] even at the minimum), the weights are not
/// allowed to drop the context to the minimum: the largest step whose
/// context-dependent RAM (CPU-side KV, compute buffers, projector) fits is
/// kept, as if the weights took whatever RAM is left, with `pages_weights`
/// set (see [`paging_warning`]).
pub fn apply_ram_reserve(c: LoadConfig, a: AutoCtx, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> (LoadConfig, AutoCtx) {
    let reserve = ram_reserve(env.total_ram);
    let budget = env.available_ram.saturating_sub(reserve);
    let side = |c: &LoadConfig| ram_side(c, meta, size, projector_bytes, env);
    let fits = |c: &LoadConfig| side(c).total() <= budget;
    if fits(&c) {
        return (c, a);
    }
    let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
    let cap = a.trained.unwrap_or(estimate::AUTO_CTX_UNKNOWN_TRAINED);
    let steps = estimate::ctx_steps(cap);
    let moe = meta.expert_count.unwrap_or(0) > 0;
    // Mixture-of-experts: the paged expert weights never count against the
    // reserve, whether or not they happen to fit, so more RAM never lowers the
    // context (lowering until weights + rest fit could undercut paging).
    let lower_with_weights = !(moe && weights_pageable(&c));
    for &n in steps.iter().rev().filter(|&&n| lower_with_weights && n < c.n_ctx as u64) {
        let t = LoadConfig { n_ctx: n as u32, ..c.clone() };
        if fits(&t) {
            let summary = format!("Largest size that leaves {} of RAM free for Windows and other apps.", gb(reserve));
            return (t, AutoCtx { n_ctx: n, trained: a.trained, limited_by: "ram_reserve".into(), fits: true, pages_weights: false, summary });
        }
    }
    if moe && weights_pageable(&c) {
        for &n in steps.iter().rev().filter(|&&n| n <= c.n_ctx as u64) {
            let t = LoadConfig { n_ctx: n as u32, ..c.clone() };
            if side(&t).other <= budget {
                let limited_by = if n == c.n_ctx as u64 { a.limited_by.clone() } else { "ram_reserve".into() };
                let summary = format!(
                    "Largest size whose KV cache and buffers leave {} of RAM free (trained: {}). The model is larger than free RAM, so Windows pages its weights from disk.",
                    gb(reserve),
                    a.trained.map(estimate::fmt_tokens).unwrap_or_else(|| "unknown".into())
                );
                let pages_weights = side(&t).total() > budget;
                return (t, AutoCtx { n_ctx: n, trained: a.trained, limited_by, fits: moe, pages_weights, summary });
            }
        }
    }
    let n = steps.first().copied().unwrap_or(c.n_ctx as u64).min(c.n_ctx as u64);
    let summary = format!("Even {n} tokens leave less than {} of RAM free, so Auto uses the minimum; see the warnings.", gb(reserve));
    (LoadConfig { n_ctx: n as u32, ..c }, AutoCtx { n_ctx: n, trained: a.trained, limited_by: "minimum".into(), fits: false, pages_weights: false, summary })
}

/// Resolve the "Auto" KV cache type (`cache_type_auto`) and an Auto context.
/// The Auto context is always sized with an f16 KV cache, so q8_0 never
/// raises the context step (the user's decision). q8_0 is then chosen at
/// that same context only when the load is known to be split with the CPU
/// ([`runs_split`]) and the f16 cache there is over ~1 GiB
/// ([`auto_kv_choice`]): it frees VRAM for more GPU layers and halves the
/// CPU-side KV (27B at 131072: q8_0 19 layers 4.93 t/s, f16 16 layers 4.82).
/// A model that fits on the GPU keeps f16 and the user's Flash Attention
/// setting. Last, an Auto context is lowered if needed to keep
/// [`ram_reserve`] free, measured with the KV type chosen. Configs with an
/// explicit K/V type get the Auto context (sized with their type) and the
/// reserve. Returns the sized config, the Auto context (None: a fixed size)
/// and the Auto KV choice (None: an explicit type; [`finish_auto_kv`] fills
/// in its savings).
///
/// With linked PCs, an Auto context is never smaller than this PC alone
/// gets (monotonic Auto): a model that fits here alone may take q8_0 to a
/// larger step, which the split sizing (f16, the CPU of a linked PC counted
/// as a split) does not; then this PC's own result is used, and the linked
/// PCs still take layers at that context (VM test 2026-11-24, Qwen3-4B:
/// 32768 with the VM, 49152 alone).
pub fn resolve_kv_and_ctx(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> (LoadConfig, Option<AutoCtx>, Option<AutoKv>) {
    let linked = resolve_kv_and_ctx_env(cfg, meta, size, projector_bytes, env);
    if env.linked.is_empty() || cfg.n_ctx != 0 {
        return linked;
    }
    let alone_env = PlanEnv {
        free_vram: env.free_vram,
        available_ram: env.available_ram,
        total_ram: env.total_ram,
        caps: env.caps,
        uses_gpu: env.uses_gpu,
        topo: env.topo,
        observed: env.observed,
        linked: &[],
    };
    let alone = resolve_kv_and_ctx_env(cfg, meta, size, projector_bytes, &alone_env);
    match (&linked.1, &alone.1) {
        (Some(l), Some(a)) if a.n_ctx > l.n_ctx && (a.fits || !l.fits) => alone,
        _ => linked,
    }
}

fn resolve_kv_and_ctx_env(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> (LoadConfig, Option<AutoCtx>, Option<AutoKv>) {
    let has_moe = env.caps.has("--n-cpu-moe");
    // A K/V type other than f16 is an explicit choice (e.g. an API or IPC
    // caller that set q8_0 on top of defaults that still say Auto): honour it.
    let explicit = cfg.cache_type_k != "f16" || cfg.cache_type_v != "f16";
    let cfg = &LoadConfig { cache_type_auto: cfg.cache_type_auto && !explicit, ..cfg.clone() };
    let mut base = cfg.clone();
    if cfg.cache_type_auto {
        base.cache_type_k = "f16".into();
        base.cache_type_v = "f16".into();
    }
    let (mut c, mut a) = resolve_auto_ctx_linked(&base, meta, size, projector_bytes, env.free_vram, env.available_ram, has_moe, env.uses_gpu, env.linked);
    let mut auto_kv = None;
    if cfg.cache_type_auto {
        auto_kv = Some(AutoKv::new("f16", "f16", false));
        let total = meta.block_count.unwrap_or(0) + 1;
        let gpu = env.uses_gpu && !matches!(cfg.gpu_layers, GpuLayers::CpuOnly);
        let f16_kv = estimate::estimate(meta, size, &estimate_input(&c, meta, total, projector_bytes)).kv_cache_bytes;
        let cb = estimate::estimate(meta, size, &estimate_input(&c, meta, total, projector_bytes)).compute_buffer_bytes;
        let split = gpu && (runs_split(&c, estimate::offloadable_weight_bytes(meta, size), projector_bytes, total, cb, env) || kv_overflows_gpu(&c, meta, size, projector_bytes, env));
        if let Some((k, v, fa)) = auto_kv_choice(&c, f16_kv, split, KvSupport::of(env.caps)) {
            c.cache_type_k = k.into();
            c.cache_type_v = v.into();
            if fa {
                c.flash_attention = inference::FlashAttention::On;
            }
            auto_kv = Some(AutoKv::new(k, v, fa));
        } else if gpu
            && !split
            && let Some(f16) = a.as_ref()
            && let Some((k, v, fa, q8)) = full_gpu_q8_raise(&base, f16, meta, size, projector_bytes, env)
        {
            // Fully on the GPU: q8_0 when it reaches a larger context step.
            c.n_ctx = q8.n_ctx.min(u32::MAX as u64) as u32;
            c.cache_type_k = k.into();
            c.cache_type_v = v.into();
            if fa {
                c.flash_attention = inference::FlashAttention::On;
            }
            let mut kv = AutoKv::new(k, v, fa);
            kv.raised_from = Some(f16.n_ctx);
            auto_kv = Some(kv);
            let summary = format!("{} With a q8_0 KV cache (f16 fits {} tokens).", q8.summary, estimate::fmt_tokens(f16.n_ctx));
            a = Some(AutoCtx { summary, ..q8 });
        }
    }
    match a {
        Some(a) => {
            let (rc, ra) = apply_ram_reserve(c.clone(), a.clone(), meta, size, projector_bytes, env);
            // The reserve lowered an f16 context that stayed f16 only because
            // the split is small (or the model looked fully on the GPU): a
            // q8_0 cache halves the CPU-side KV and keeps more of the context,
            // as it does at the neighbouring VRAM sizes. Without this, Auto
            // dipped just where the split became too small to quantize.
            if cfg.cache_type_auto
                && ra.limited_by == "ram_reserve"
                && c.cache_type_k == "f16"
                && env.uses_gpu
                && !matches!(cfg.gpu_layers, GpuLayers::CpuOnly)
                && let total = meta.block_count.unwrap_or(0) + 1
                && let f16_kv = estimate::estimate(meta, size, &estimate_input(&c, meta, total, projector_bytes)).kv_cache_bytes
                && let Some((k, v, fa)) = auto_kv_choice(&c, f16_kv, true, KvSupport::of(env.caps))
            {
                let mut q = c.clone();
                q.cache_type_k = k.into();
                q.cache_type_v = v.into();
                if fa {
                    q.flash_attention = inference::FlashAttention::On;
                }
                let (qc, qa) = apply_ram_reserve(q, a, meta, size, projector_bytes, env);
                if qa.n_ctx > ra.n_ctx {
                    return (qc, Some(qa), Some(AutoKv::new(k, v, fa)));
                }
            }
            (rc, Some(ra), auto_kv)
        }
        None => (c, None, auto_kv),
    }
}

/// Fill in what the Auto KV choice saves at the final context of `cfg`
/// (the config after [`resolve_kv_and_ctx`]). `meta` is the one the load uses.
pub fn finish_auto_kv(a: &mut AutoKv, cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64) {
    let total = meta.block_count.unwrap_or(0) + 1;
    let chosen = estimate_input(cfg, meta, total, projector_bytes);
    let f16 = EstimateInput { cache_type_k: "f16".into(), cache_type_v: "f16".into(), ..chosen.clone() };
    let kv = |i: &EstimateInput| estimate::estimate(meta, size, i).kv_cache_bytes;
    a.saves_bytes = kv(&f16).saturating_sub(kv(&chosen));
    let types = if a.cache_type_k == a.cache_type_v { a.cache_type_k.clone() } else { format!("{} K / {} V", a.cache_type_k, a.cache_type_v) };
    let ctx = cfg.n_ctx as u64;
    a.summary = if let Some(f16) = a.raised_from.filter(|&f| f > 0 && ctx > f) {
        let x = ctx as f64 / f16 as f64;
        let ratio = if (x - x.round()).abs() < 0.05 { format!("{}x", x.round()) } else { format!("{x:.1}x") };
        format!("KV cache: {types} (auto) gives {ratio} context: {} tokens, f16 fits {}", estimate::fmt_tokens(ctx), estimate::fmt_tokens(f16))
    } else if a.saves_bytes > 0 {
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

/// Apply [`hybrid_cpu_threads`] when the user left the thread count unset
/// and the runtime has the flags. A user who set the generation threads
/// keeps llama.cpp's prompt default (it follows `-t`) unless they set the
/// prompt threads too.
pub fn apply_thread_policy(cfg: &LoadConfig, topo: Option<system_monitor::CoreTopology>, cpu_runs_layers: bool, caps: &inference::RuntimeCapabilities) -> LoadConfig {
    let mut c = cfg.clone();
    if let Some((t, tb)) = hybrid_cpu_threads(topo, cpu_runs_layers)
        && c.threads.is_none()
    {
        if caps.has("--threads") {
            c.threads = Some(t);
        }
        if c.threads_batch.is_none() && caps.has("--threads-batch") {
            c.threads_batch = Some(tb);
        }
    }
    c
}

/// Will part of the model (or its KV cache) run on the CPU with `cfg`?
/// None when llama.cpp decides at load (GPU offload Auto / `--fit`) and
/// neither an earlier load of these settings (`observed`: layers on the GPU,
/// of all) nor the weights' size (`weights_over_vram`) settles it.
pub fn cpu_runs_layers(cfg: &LoadConfig, total_layers: u64, uses_gpu: bool, weights_over_vram: bool, observed: Option<(u32, u32)>) -> Option<bool> {
    if !uses_gpu || !cfg.kv_offload || cfg.n_cpu_moe.is_some_and(|n| n > 0) {
        return Some(true);
    }
    match cfg.gpu_layers {
        GpuLayers::CpuOnly => Some(true),
        GpuLayers::Count(n) => Some((n as u64) < total_layers),
        GpuLayers::All => Some(false),
        GpuLayers::Auto => match observed {
            Some((n, m)) => Some(n < m),
            None if weights_over_vram => Some(true),
            None => None,
        },
    }
}

/// The settings that decide how many layers llama.cpp fits, as a key for
/// remembering what it did ([`PlanEnv::observed`]).
pub fn offload_key(cfg: &LoadConfig) -> String {
    format!(
        "ctx={} k={} v={} spec={} proj={} share={} np={} ub={:?}",
        cfg.n_ctx,
        cfg.cache_type_k,
        cfg.cache_type_v,
        cfg.speculative.as_deref().unwrap_or(""),
        cfg.load_projector,
        cfg.share_with_cpu,
        cfg.parallel,
        cfg.ubatch_size
    )
}

/// What a load plan needs to know about the machine and the runtime.
pub struct PlanEnv<'a> {
    /// Free VRAM as the load will see it (None: no reading).
    pub free_vram: Option<u64>,
    /// Available RAM as the load will see it (after the old model is unloaded).
    pub available_ram: u64,
    pub total_ram: u64,
    /// The runtime the load uses.
    pub caps: &'a inference::RuntimeCapabilities,
    /// The runtime has a GPU backend.
    pub uses_gpu: bool,
    pub topo: Option<system_monitor::CoreTopology>,
    /// Layers llama.cpp put on the GPU (of all) the last time this model
    /// loaded with the same [`offload_key`], if the app saw it.
    pub observed: &'a (dyn Fn(&LoadConfig) -> Option<(u32, u32)> + Sync),
    /// What linked PCs lend this load ("Use linked PCs"); empty = this PC alone.
    pub linked: &'a [LinkedDev],
}

/// Everything the app decides for one load, in order: KV type and Auto
/// context (speculative MTP off), whether MTP still fits that context,
/// Share with CPU, then threads.
#[derive(Debug, Clone)]
pub struct LoadPlan {
    /// The config llama-server gets.
    pub config: LoadConfig,
    /// After the KV type, the Auto context and the speculative check, before
    /// Share with CPU and threads: what the memory estimate shows.
    pub sized: LoadConfig,
    /// Speculative MTP runs with this runtime, so its KV cache is counted.
    pub mtp_on: bool,
    pub auto_ctx: Option<AutoCtx>,
    pub auto_kv: Option<AutoKv>,
    pub split: Option<ShareSplit>,
    /// Part of the model runs on the CPU ([`cpu_runs_layers`]); None until
    /// llama.cpp has decided at load.
    pub cpu_layers: Option<bool>,
    /// Choices the app made that the user should know about.
    pub notes: Vec<String>,
}

/// Does speculative MTP decoding run with this runtime? The config turns
/// `draft-mtp` on, every listed type is one the runtime takes (else none is
/// passed), and the file has MTP tensors.
pub fn mtp_runs(cfg: &LoadConfig, meta: &ModelMetadata, caps: &inference::RuntimeCapabilities) -> bool {
    cfg.speculative.as_deref().is_some_and(|s| s.split(',').any(|t| t == "draft-mtp") && s.split(',').all(|t| caps.has_spec_type(t)))
        && caps.has("--spec-type")
        && meta.mtp_layers.unwrap_or(0) > 0
}

/// MTP keeps the context Auto chose without it: the context with the MTP KV
/// cache is at least as large, and fits whenever the one without it does.
pub fn spec_keeps_context(without: &AutoCtx, with: &AutoCtx) -> bool {
    with.n_ctx >= without.n_ctx && (with.fits || !without.fits)
}

/// llama-server's default host prompt cache (`--cache-ram`), MiB. It is RAM
/// on top of everything the estimate counts (measured on Gemma-4-E4B: RSS
/// 3547 MB after load, 5188 MB after two rounds of chats).
pub const DEFAULT_PROMPT_CACHE_MIB: u32 = 8192;
/// Below this the prompt cache is switched off instead.
const MIN_PROMPT_CACHE_MIB: u64 = 256;

/// `--cache-ram` for this load: the RAM left after the planned load
/// ([`ram_side`]) and [`ram_reserve`], at most llama.cpp's default 8192 MiB;
/// 0 (off) when under 256 MiB is left. None when the runtime lacks the flag.
/// The context is already fixed: the cache only takes what is left, it never
/// lowers n_ctx.
pub fn prompt_cache_mib(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> Option<u32> {
    if !env.caps.has("--cache-ram") {
        return None;
    }
    let left = env.available_ram.saturating_sub(ram_reserve(env.total_ram)).saturating_sub(ram_side_bytes(cfg, meta, size, projector_bytes, env)) / MIB;
    Some(if left < MIN_PROMPT_CACHE_MIB { 0 } else { left.min(DEFAULT_PROMPT_CACHE_MIB as u64) as u32 })
}

/// Plan a load ([`LoadPlan`]). The context is never traded for speed: an
/// Auto context is sized with speculative MTP off, and MTP stays on only if
/// that same context still fits with the MTP layer's KV cache added; else it
/// is off for this load, with a note.
pub fn plan_load(cfg: &LoadConfig, meta: &ModelMetadata, size: u64, projector_bytes: u64, env: &PlanEnv) -> LoadPlan {
    let caps = env.caps;
    let has_moe = caps.has("--n-cpu-moe");
    let meta_off = spec_meta(meta, false);
    let meta_on = spec_meta(meta, true);
    let mut notes = Vec::new();
    // 1. KV cache type and Auto context, sized without the MTP layer's cache.
    let (mut sized, mut auto_ctx, mut auto_kv) = resolve_kv_and_ctx(cfg, &meta_off, size, projector_bytes, env);
    // 2. Speculative MTP only where it costs no context.
    let mut mtp_on = mtp_runs(cfg, meta, caps);
    if mtp_on && let Some(off) = auto_ctx.clone() {
        let (s_on, a_on, kv_on) = resolve_kv_and_ctx(cfg, &meta_on, size, projector_bytes, env);
        let on = a_on.clone().unwrap_or_else(|| off.clone());
        if spec_keeps_context(&off, &on) {
            (sized, auto_ctx, auto_kv) = (s_on, a_on, kv_on);
        } else {
            mtp_on = false;
            sized.speculative = None;
            sized.spec_draft_n_max = None;
            sized.speculative_auto = false;
            notes.push(format!(
                "Speculative decoding off: it would have reduced your context (Auto fits {} tokens without it, {} with the MTP layer's KV cache). Set a fixed context size to use both.",
                off.n_ctx, on.n_ctx
            ));
        }
    }
    let meta_eff = if mtp_on { &meta_on } else { &meta_off };
    if let Some(a) = auto_kv.as_mut() {
        finish_auto_kv(a, &sized, meta_eff, size, projector_bytes);
    }
    let total = meta.block_count.unwrap_or(0) + 1;
    let input = estimate_input(&sized, meta_eff, total, projector_bytes);
    let kv = |m: &ModelMetadata| estimate::estimate(m, size, &input).kv_cache_bytes;
    let mtp_kv = if mtp_on && sized.kv_offload { kv(&meta_on).saturating_sub(kv(&meta_off)) } else { 0 };
    // 3. Share with CPU.
    let (config, split) = if cfg.share_with_cpu && env.uses_gpu {
        // Linked PCs hold layers before this PC's CPU does.
        let free = match env.free_vram {
            Some(f) if !env.linked.is_empty() => Some(split_free(env, f, estimate::estimate(meta_eff, size, &estimate_input(&sized, meta_eff, meta.block_count.unwrap_or(0) + 1, projector_bytes)).compute_buffer_bytes)),
            f => f,
        };
        // llama.cpp's --fit gives up once --tensor-split is set ("tensor_split
        // already set by user, abort") and then offloads every layer, so with
        // linked PCs the app picks the GPU layer count itself.
        resolve_share(&sized, meta_eff, size, projector_bytes, mtp_kv, free, has_moe, runtime_can_fit(caps) && env.linked.is_empty())
    } else {
        (sized.clone(), None)
    };
    // 4. Hybrid-CPU threads, only when the CPU is known to run layers.
    let cpu_layers = cpu_runs_layers(&config, total, env.uses_gpu, weights_over_vram(estimate::offloadable_weight_bytes(meta, size), projector_bytes, pooled_free(env, estimate::estimate(meta_eff, size, &input).compute_buffer_bytes)), (env.observed)(&config));
    let config = if cpu_layers == Some(true) { apply_thread_policy(&config, env.topo, true, caps) } else { config };
    // 5. Host-RAM prompt cache, from the RAM the load leaves. Never touches n_ctx.
    // No note: a smaller cache is normal on split loads and the value is shown in
    // the load arguments; a warning on every such load would only be noise.
    let mut config = config;
    if config.cache_ram_mib.is_none()
        && let Some(m) = prompt_cache_mib(&config, meta_eff, size, projector_bytes, env)
    {
        config.cache_ram_mib = Some(m);
    }
    LoadPlan { config, sized, mtp_on, auto_ctx, auto_kv, split, cpu_layers, notes }
}

/// After a load where llama.cpp chose the layers (`cpu_layers` None in the
/// plan): a note when it kept layers on the CPU and the hybrid-CPU thread
/// policy would have changed the threads. The measurement is remembered, so
/// the next load with these settings applies them.
pub fn after_load_note(
    launched: &LoadConfig,
    cpu_layers: Option<bool>,
    offloaded: Option<(u32, u32)>,
    topo: Option<system_monitor::CoreTopology>,
    caps: &inference::RuntimeCapabilities,
) -> Option<String> {
    let (n, m) = offloaded?;
    if cpu_layers.is_some() || n >= m {
        return None;
    }
    let with = apply_thread_policy(launched, topo, true, caps);
    if with == *launched {
        return None;
    }
    let threads = match (with.threads, with.threads_batch) {
        (Some(t), Some(tb)) => format!("{t} generation and {tb} prompt threads"),
        (Some(t), None) => format!("{t} generation threads"),
        (None, Some(tb)) => format!("{tb} prompt threads"),
        (None, None) => "different CPU threads".into(),
    };
    Some(format!(
        "llama.cpp kept {} of {m} layers on the CPU. The next load with these settings uses {threads} for that (hybrid CPU); this load uses llama.cpp's defaults.",
        m - n
    ))
}

/// Retry a failed load once without speculative decoding? Only when the app
/// turned it on by default (`speculative_auto`; never the user's own
/// choice), it was really passed to the backend, and the backend failed to
/// start (runtime error or out of memory, not a cancel or a rejected setting).
pub fn retry_without_spec(user: &LoadConfig, launched: &LoadConfig, caps: &inference::RuntimeCapabilities, err: &AppError) -> bool {
    user.speculative_auto
        && launched.speculative.as_deref().is_some_and(|s| s.split(',').all(|t| caps.has_spec_type(t)))
        && caps.has("--spec-type")
        && matches!(err, AppError::Runtime(_) | AppError::OutOfMemory(_))
}

/// The user's config with speculative decoding off: saved as "last used"
/// after the fallback, so the next load does not fail the same way first
/// (the user can turn MTP on again).
pub fn without_spec(user: &LoadConfig) -> LoadConfig {
    LoadConfig { speculative: None, speculative_auto: false, spec_draft_n_max: None, ..user.clone() }
}

/// The config for that retry: [`without_spec`], with the context and KV
/// cache type the failed load chose (dropping MTP only frees memory, so the
/// same context fits, and memory read right after the failed process exits
/// must not shrink it).
pub fn retry_config(user: &LoadConfig, launched: &LoadConfig) -> LoadConfig {
    LoadConfig {
        n_ctx: launched.n_ctx,
        cache_type_auto: false,
        cache_type_k: launched.cache_type_k.clone(),
        cache_type_v: launched.cache_type_v.clone(),
        flash_attention: launched.flash_attention,
        ..without_spec(user)
    }
}

#[tauri::command]
pub async fn estimate_memory(state: State<'_, SharedState>, id: String, config: LoadConfig, runtime_id: Option<String>) -> Cmd<EstimateView> {
    let st = state.inner();
    let m = st.db.get_model(&id).await?;
    let meta: ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    let hw = st.monitor.hardware_info();
    let gpu = hw.gpus.first();
    let total_layers = meta.block_count.unwrap_or(0) + 1;
    let proj = projector_bytes(&config, m.projector_path.as_deref());
    let free_vram = gpu.map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes));
    // The runtime the load would use, with the same rules as the load.
    let rt = runtime_or_default(st, runtime_id.as_deref()).await;
    let caps = rt.as_ref().map(|r| r.capabilities.clone()).unwrap_or_default();
    let uses_gpu = rt.as_ref().is_none_or(|r| r.uses_gpu());
    let has_moe_flag = caps.has("--n-cpu-moe");
    // Measured as a load would see it: a loaded model is unloaded first, so
    // its VRAM counts as free.
    let reclaim = loaded_backend_vram(st);
    let free_at_load = free_vram.map(|f| f.saturating_add(reclaim).min(gpu.map(|g| g.vram_total_bytes).unwrap_or(u64::MAX)));
    let observed = st.observed_offload.lock().unwrap().clone();
    let lookup = |c: &LoadConfig| observed.get(&(m.id.clone(), offload_key(c))).copied();
    // "Use linked PCs": what they lend, from their last info (nothing is started).
    let linked_contrib = if config.use_linked_pcs && uses_gpu && !matches!(config.gpu_layers, GpuLayers::CpuOnly) {
        st.linking.linked_capacity().await
    } else {
        Vec::new()
    };
    let linked: Vec<LinkedDev> = linked_contrib.iter().map(|s| LinkedDev { bytes: s.bytes, cpu: s.cpu }).collect();
    let env = PlanEnv {
        free_vram: free_at_load,
        available_ram: hw.available_ram_bytes,
        total_ram: hw.total_ram_bytes,
        caps: &caps,
        uses_gpu,
        topo: system_monitor::core_topology(),
        observed: &lookup,
        linked: &linked,
    };
    let plan = plan_load(&config, &meta, m.size as u64, proj, &env);
    // The same plan without the linked PCs: "32,768 on this PC alone".
    let auto_ctx_alone = if linked.is_empty() || config.n_ctx != 0 {
        None
    } else {
        plan_load(&config, &meta, m.size as u64, proj, &PlanEnv { linked: &[], ..env }).auto_ctx.map(|a| a.n_ctx)
    };
    let meta = spec_meta(&meta, plan.mtp_on);
    let auto_ctx = plan.auto_ctx.clone().unwrap_or_else(|| {
        let probe = LoadConfig { n_ctx: 0, ..plan.sized.clone() };
        auto_ctx_linked(&probe, &meta, m.size as u64, proj, free_at_load, hw.available_ram_bytes, has_moe_flag, uses_gpu, &linked)
    });
    let input = estimate_input(&plan.sized, &meta, total_layers, proj);
    let est = estimate::estimate(&meta, m.size as u64, &input);
    let suggested = free_vram.map(|f| estimate::max_gpu_layers_for_budget(&meta, m.size as u64, &input, f.saturating_sub(1 << 30)));
    let mut warnings = plan.notes.clone();
    warnings.extend(linked_left_out_notes(&linked_contrib, est.compute_buffer_bytes));
    if let (Some(trained), true) = (meta.context_length, input.n_ctx > meta.context_length.unwrap_or(u64::MAX)) {
        warnings.push(format!(
            "Context {} is larger than the model's trained context ({trained}); quality may drop unless RoPE scaling is configured.",
            input.n_ctx
        ));
    }
    // RAM outside the GPU must leave the reserve free.
    let reserve = ram_reserve(hw.total_ram_bytes);
    let ram_budget = hw.available_ram_bytes.saturating_sub(reserve);
    let in_ram = ram_side_bytes(&plan.sized, &meta, m.size as u64, proj, &env);
    let mut fits = in_ram <= ram_budget;
    if ram_fit(&plan.sized, &meta, m.size as u64, proj, &env) == RamFit::PagesWeights {
        // The weights page from the file; the context-dependent part fits.
        // Workable for mixture-of-experts models (active experts only).
        fits = meta.expert_count.unwrap_or(0) > 0;
        warnings.push(paging_warning(&meta));
    } else if !fits && meta.expert_count.unwrap_or(0) == 0 && ram_side(&plan.sized, &meta, m.size as u64, proj, &env).weights > 0 {
        warnings.push(not_enough_ram_warning(in_ram, ram_budget, reserve));
    } else if !fits {
        warnings.push(format!(
            "Estimated RAM use ({:.1} GB) leaves less than {:.1} GB of the available RAM free for Windows and other apps; the load may page to disk and be very slow.",
            in_ram as f64 / 1e9,
            reserve as f64 / 1e9
        ));
    }
    // What "Share with CPU" would choose (the load's own split when it is
    // on), and whether to suggest it.
    let share_split = plan.split.clone().or_else(|| {
        free_at_load.map(|f| {
            let mut all = input.clone();
            all.gpu_layers = None;
            estimate::share_split(&meta, m.size as u64, &all, f, has_moe_flag)
        })
    });
    let full_gpu_too_big = share_split.as_ref().is_some_and(|s| !s.all_fit);
    let suggest_share = !config.share_with_cpu && !matches!(config.gpu_layers, GpuLayers::CpuOnly) && full_gpu_too_big;
    if !config.share_with_cpu
        && let Some(f) = free_vram
        && est.vram_bytes > f
    {
        if matches!(config.gpu_layers, GpuLayers::Auto) {
            warnings.push("Not all layers fit in free VRAM; Auto will offload fewer layers.".into());
        } else {
            fits = false;
            warnings.push("Estimated VRAM use exceeds free VRAM; loading may fail with out of memory.".into());
        }
    }
    let share_fit = config.share_with_cpu && plan.config.gpu_layers == GpuLayers::Auto && plan.config.fit_target_mib.is_some();
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
        auto_kv: plan.auto_kv,
        share_fit,
        fit_target_mib: if share_fit { plan.config.fit_target_mib } else { None },
        ram_reserve_bytes: reserve,
        auto_ctx_alone,
        linked_contrib,
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

/// Default `--spec-type` for a model with no saved choice: "draft-mtp" when the
/// file really contains MTP tensors (`mtp_layers` is only set then) and the
/// runtime lists draft-mtp, else off.
pub fn default_speculative(meta: &ModelMetadata, caps: Option<&inference::RuntimeCapabilities>) -> Option<String> {
    (meta.mtp_layers.unwrap_or(0) > 0 && caps.is_some_and(|c| c.has_spec_type("draft-mtp"))).then(|| "draft-mtp".to_string())
}

/// `meta` with the MTP layer's own KV cache counted when `on` (speculative
/// MTP runs for the load, see [`mtp_runs`]).
pub fn spec_meta(meta: &ModelMetadata, on: bool) -> ModelMetadata {
    ModelMetadata { mtp_speculative: on, ..meta.clone() }
}

/// A "last used" config as saved, upgraded when it was saved before Auto KV
/// and speculative MTP existed (no such keys): it carries the old defaults,
/// not a choice, so the new defaults apply (MTP marked as the app's choice,
/// `speculative_auto`). Every load path that reads `last_config` uses this
/// (load panel defaults, the local API, reload with the projector); an
/// explicit per-model default is never changed. None when it does not parse.
pub fn upgrade_last_config(raw: &str, meta: &ModelMetadata, caps: Option<&inference::RuntimeCapabilities>) -> Option<LoadConfig> {
    let mut c: LoadConfig = serde_json::from_str(raw).ok()?;
    let keys: serde_json::Value = serde_json::from_str(raw).ok()?;
    if keys.get("cache_type_auto").is_none() && c.cache_type_k == "f16" && c.cache_type_v == "f16" {
        c.cache_type_auto = true;
    }
    if keys.get("speculative").is_none() {
        c.speculative = default_speculative(meta, caps);
        c.speculative_auto = c.speculative.is_some();
    }
    Some(c)
}

/// The model's "last used" config ([`upgrade_last_config`]) for a load with
/// runtime `runtime_id` (None: the default runtime).
pub async fn last_used_config(st: &SharedState, m: &ModelRow, runtime_id: Option<&str>) -> Option<LoadConfig> {
    let raw = m.last_config.as_deref()?;
    let meta: ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    let rt = runtime_or_default(st, runtime_id).await;
    upgrade_last_config(raw, &meta, rt.as_ref().map(|r| &r.capabilities))
}

#[tauri::command]
pub async fn get_load_defaults(state: State<'_, SharedState>, id: String) -> Cmd<LoadDefaults> {
    let m = state.db.get_model(&id).await?;
    let meta: ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    let parse = |s: &Option<String>| s.as_deref().and_then(|t| serde_json::from_str::<LoadConfig>(t).ok());
    let (config, source) = if let Some(c) = parse(&m.default_config) {
        (c, "model_default")
    } else if let Some(c) = last_used_config(state.inner(), &m, None).await {
        (c, "last_used")
    } else {
        // No saved choice: Auto context and Auto KV cache type (both fit
        // the hardware at load time). Speculative MTP decoding is on when the
        // file really has MTP tensors and the default runtime lists draft-mtp.
        let rt = default_runtime(state.inner()).await;
        let speculative = default_speculative(&meta, rt.as_ref().map(|r| &r.capabilities));
        let c = LoadConfig { n_ctx: 0, cache_type_auto: true, speculative_auto: speculative.is_some(), speculative, ..LoadConfig::default() };
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
mod spec_default_tests {
    use super::*;

    #[test]
    fn mtp_default_on_only_with_real_tensors_and_runtime_support() {
        let help = "--spec-type none,draft-mtp";
        let caps = inference::RuntimeCapabilities::parse(help);
        let none = inference::RuntimeCapabilities::parse("--spec-type none,ngram-simple");
        let real = ModelMetadata { mtp_layers: Some(1), mtp_layers_declared: Some(1), ..Default::default() };
        let stripped = ModelMetadata { mtp_layers: None, mtp_layers_declared: Some(1), ..Default::default() };
        let unknown = ModelMetadata { mtp_layers: None, mtp_layers_declared: Some(1), mtp_unknown: true, ..Default::default() };
        assert_eq!(default_speculative(&real, Some(&caps)).as_deref(), Some("draft-mtp"));
        assert_eq!(default_speculative(&stripped, Some(&caps)), None);
        assert_eq!(default_speculative(&unknown, Some(&caps)), None, "unknown tensors: not turned on");
        assert_eq!(default_speculative(&real, Some(&none)), None);
        assert_eq!(default_speculative(&real, None), None);
        // spec_meta counts the MTP KV only when on; mtp_runs follows the runtime.
        assert!(spec_meta(&real, true).mtp_speculative && !spec_meta(&real, false).mtp_speculative);
        let on = LoadConfig { speculative: Some("draft-mtp".into()), ..Default::default() };
        assert!(mtp_runs(&on, &real, &caps));
        assert!(!mtp_runs(&on, &real, &none), "runtime without draft-mtp: no MTP KV charged");
        assert!(!mtp_runs(&on, &stripped, &caps));
        assert!(!mtp_runs(&LoadConfig::default(), &real, &caps));
    }

    #[test]
    fn old_last_used_configs_get_the_new_defaults_on_every_path() {
        let caps = inference::RuntimeCapabilities::parse("--spec-type none,draft-mtp");
        let real = ModelMetadata { mtp_layers: Some(1), ..Default::default() };
        // Saved before Auto KV / MTP existed: the old defaults, not a choice.
        let c = upgrade_last_config(r#"{"n_ctx": 4096}"#, &real, Some(&caps)).unwrap();
        assert!(c.cache_type_auto && c.speculative.as_deref() == Some("draft-mtp") && c.speculative_auto, "{c:?}");
        assert_eq!(c.n_ctx, 4096);
        // An explicit choice is kept.
        let c = upgrade_last_config(r#"{"n_ctx": 4096, "speculative": null, "cache_type_auto": false}"#, &real, Some(&caps)).unwrap();
        assert!(!c.cache_type_auto && c.speculative.is_none() && !c.speculative_auto);
        let c = upgrade_last_config(r#"{"cache_type_k": "q8_0", "speculative": "draft-mtp"}"#, &real, Some(&caps)).unwrap();
        assert!(!c.cache_type_auto && c.speculative.is_some() && !c.speculative_auto, "user's own MTP choice");
        // Without MTP support: off, and not marked auto.
        let c = upgrade_last_config("{}", &real, None).unwrap();
        assert!(c.speculative.is_none() && !c.speculative_auto);
        assert!(upgrade_last_config("not json", &real, Some(&caps)).is_none());
    }

    #[test]
    fn a_failed_default_mtp_load_is_retried_once_without_it() {
        let caps = inference::RuntimeCapabilities::parse("--spec-type none,draft-mtp");
        let auto = LoadConfig { n_ctx: 0, cache_type_auto: true, speculative: Some("draft-mtp".into()), speculative_auto: true, spec_draft_n_max: Some(4), ..Default::default() };
        let launched = LoadConfig { n_ctx: 131_072, cache_type_auto: false, cache_type_k: "q8_0".into(), cache_type_v: "q8_0".into(), flash_attention: inference::FlashAttention::On, threads: Some(12), ..auto.clone() };
        let rt = AppError::Runtime("backend exited during load".into());
        assert!(retry_without_spec(&auto, &launched, &caps, &rt));
        assert!(retry_without_spec(&auto, &launched, &caps, &AppError::OutOfMemory("x".into())));
        // The user's own choice, a cancel, a rejected setting, or a runtime that never got the flag: no retry.
        let mine = LoadConfig { speculative_auto: false, ..auto.clone() };
        assert!(!retry_without_spec(&mine, &launched, &caps, &rt));
        assert!(!retry_without_spec(&auto, &launched, &caps, &AppError::Cancelled));
        assert!(!retry_without_spec(&auto, &launched, &caps, &AppError::InvalidInput("x".into())));
        let no_mtp = inference::RuntimeCapabilities::parse("--spec-type none,ngram-simple");
        assert!(!retry_without_spec(&auto, &launched, &no_mtp, &rt));
        assert!(!retry_without_spec(&auto, &LoadConfig { speculative: None, ..launched.clone() }, &caps, &rt));
        // The retry keeps the context and KV type the failed load chose; "last used" keeps Auto.
        let r = retry_config(&auto, &launched);
        assert_eq!((r.n_ctx, r.cache_type_k.as_str(), r.cache_type_auto, r.flash_attention), (131_072, "q8_0", false, inference::FlashAttention::On));
        assert!(r.speculative.is_none() && !r.speculative_auto && r.spec_draft_n_max.is_none() && r.threads.is_none());
        let saved = without_spec(&auto);
        assert!(saved.n_ctx == 0 && saved.cache_type_auto && saved.speculative.is_none());
    }
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

    /// Qwen3.8-27B Q4_K_M: hybrid, 65 blocks of which 1 is an MTP layer.
    fn qwen27b() -> ModelMetadata {
        ModelMetadata {
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
            mtp_layers_declared: Some(1),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }
    const Q27_SIZE: u64 = 16_810_714_336;

    /// Gemma-4-E4B Q4_K_M header (lmstudio-community).
    fn gemma4() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("gemma4".into()),
            block_count: Some(42),
            context_length: Some(131_072),
            embedding_length: Some(2560),
            head_count: Some(8),
            head_count_kv: Some(2),
            key_length: Some(512),
            value_length: Some(512),
            vocab_size: Some(262_144),
            sliding_window: Some(512),
            swa_layers: Some((0..42).map(|il| il % 6 != 5).collect()),
            key_length_swa: Some(256),
            value_length_swa: Some(256),
            shared_kv_layers: Some(18),
            ..Default::default()
        }
    }
    const GEMMA_SIZE: u64 = 5_335_291_936;

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
--spec-type none,draft-simple,draft-mtp,ngram-simple
";

    fn args(c: &LoadConfig, help: &str) -> Vec<String> {
        c.to_args(Path::new("m.gguf"), None, 1, &RuntimeCapabilities::parse(help)).args
    }

    fn val<'a>(a: &'a [String], f: &str) -> Option<&'a str> {
        a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).map(String::as_str)
    }

    fn no_obs(_: &LoadConfig) -> Option<(u32, u32)> {
        None
    }

    const ULTRA7: CoreTopology = CoreTopology { performance_cores: 8, efficiency_cores: 12 };
    /// The A/B PC: 31.6 GB RAM, about 23 GB available, 8 GB GPU with 7 GiB free.
    const TOTAL_RAM: u64 = 31_600_000_000;
    const AVAIL_RAM: u64 = 23_000_000_000;

    fn penv<'a>(caps: &'a RuntimeCapabilities, free: Option<u64>, available: u64, observed: &'a (dyn Fn(&LoadConfig) -> Option<(u32, u32)> + Sync)) -> PlanEnv<'a> {
        PlanEnv { free_vram: free, available_ram: available, total_ram: TOTAL_RAM, caps, uses_gpu: true, topo: Some(ULTRA7), observed, linked: &[] }
    }

    fn auto_for(meta: &ModelMetadata, size: u64, free_vram: u64, avail: u64) -> u64 {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = penv(&caps, Some(free_vram), avail, &no_obs);
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, ..Default::default() };
        resolve_kv_and_ctx(&cfg, meta, size, 0, &env).0.n_ctx as u64
    }

    #[test]
    fn auto_context_never_shrinks_when_ram_or_vram_grow() {
        // Dense models never page: both are swept from 8 GB RAM upward.
        for (name, meta, size, min_gb) in [("moe", moe35b(), MOE_SIZE, 8u64), ("dense", qwen27b(), Q27_SIZE, 8)] {
            // More RAM at fixed VRAM.
            for vram in [4u64, 7, 12, 24] {
                let mut prev = 0;
                for gb in min_gb..=64u64 {
                    let n = auto_for(&meta, size, vram * GIB, gb * 1_000_000_000);
                    assert!(n >= prev, "{name}: {gb} GB RAM / {vram} GiB VRAM gave {n} < {prev}");
                    prev = n;
                }
            }
            // More VRAM at fixed RAM.
            for gb in [8u64, 17, 24, 32, 48, 64].into_iter().filter(|&g| g >= min_gb) {
                let mut prev = 0;
                for vram in 2..=32u64 {
                    let n = auto_for(&meta, size, vram * GIB, gb * 1_000_000_000);
                    assert!(n >= prev, "{name}: {vram} GiB VRAM / {gb} GB RAM gave {n} < {prev}");
                    prev = n;
                }
            }
        }
    }

    #[test]
    fn dense_auto_context_never_pages_and_warns_when_ram_is_short() {
        let at = |gb: u64| auto_for(&qwen27b(), Q27_SIZE, 4 * GIB, gb * 1_000_000_000);
        eprintln!("dense auto ctx: 12={} 15={} 16={} 24={} 32={}", at(12), at(15), at(16), at(24), at(32));
        assert!(at(15) <= at(16) && at(16) <= at(24) && at(24) <= at(32));
        assert_eq!(at(15), 2048);
        let caps = RuntimeCapabilities::parse(HELP);
        let env = penv(&caps, Some(4 * GIB), 15_000_000_000, &no_obs);
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, ..Default::default() };
        let (c, a, _) = resolve_kv_and_ctx(&cfg, &qwen27b(), Q27_SIZE, 0, &env);
        let a = a.unwrap();
        assert!(!a.fits && !a.pages_weights && a.limited_by == "minimum", "{a:?}");
        assert_eq!(ram_fit(&c, &qwen27b(), Q27_SIZE, 0, &env), RamFit::TooBig);
        let w = not_enough_ram_warning(1, 1, 1);
        assert!(w.starts_with("Not enough RAM for this model"), "{w}");
    }

    #[test]
    fn moe_auto_context_at_17_gb_is_not_larger_than_at_24_gb() {
        let at = |gb: u64| auto_for(&moe35b(), MOE_SIZE, 7 * GIB, gb * 1_000_000_000);
        eprintln!("moe auto ctx: 17GB={} 24GB={} 32GB={}", at(17), at(24), at(32));
        assert_eq!(at(17), 262_144);
        assert!(at(24) >= at(17) && at(32) >= at(24));
    }

    #[test]
    fn share_with_fit_lets_llama_cpp_pick_the_layers_at_our_context() {
        // 16 GB dense model, 8 GiB free: the estimate splits it, llama.cpp fits it.
        let (c, s) = resolve_share(&share(32_768), &dense(), 16_000_000_000, 0, 0, Some(8 * GIB), true, true);
        let s = s.expect("preview split");
        assert!(!s.all_fit && s.gpu_layers > 0 && s.gpu_layers < 33, "{s:?}");
        assert_eq!(c.gpu_layers, GpuLayers::Auto);
        assert_eq!((c.fit, c.fit_target_mib, c.fit_ctx, c.n_ctx, c.n_cpu_moe), (true, Some(1024), Some(32_768), 32_768, None));
        let a = args(&c, HELP);
        assert!(!a.iter().any(|x| x == "--n-gpu-layers"), "{a:?}");
        assert_eq!((val(&a, "--fit"), val(&a, "--fit-target"), val(&a, "--ctx-size"), val(&a, "--fit-ctx")), (Some("on"), Some("1024"), Some("32768"), Some("32768")));
        // The config's margin (the benchmark's A/B) wins over the default.
        let (c, _) = resolve_share(&LoadConfig { fit_target_mib: Some(512), ..share(32_768) }, &dense(), 16_000_000_000, 0, 0, Some(8 * GIB), true, true);
        assert_eq!(val(&args(&c, HELP), "--fit-target"), Some("512"));
        // No VRAM reading: fitted too.
        let (c, s) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, 0, None, true, true);
        assert!(s.is_none() && c.gpu_layers == GpuLayers::Auto && c.fit_ctx == Some(4096));
        // Off: untouched.
        let off = LoadConfig { n_ctx: 4096, ..Default::default() };
        assert_eq!(resolve_share(&off, &dense(), 16_000_000_000, 0, 0, Some(8 * GIB), true, true).0, off);
    }

    #[test]
    fn share_when_everything_fits_offloads_all_layers_unless_it_is_tight() {
        // RT-4: fits with room to spare -> -ngl all (as before --fit), the UI's "All layers fit" is true.
        let (c, s) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, 0, Some(12 * GIB), true, true);
        let s = s.unwrap();
        assert!(s.all_fit && c.gpu_layers == GpuLayers::All && c.fit_target_mib.is_none() && c.fit_ctx.is_none(), "{c:?}");
        assert_eq!(val(&args(&c, HELP), "--n-gpu-layers"), Some("all"));
        // Fits the 90% budget but not with the 1024 MiB margin: llama.cpp fits it.
        let free = (s.vram_bytes as f64 / 0.90) as u64 + 64;
        assert!(s.vram_bytes + 1024 * MIB > free);
        let (c, s) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, 0, Some(free), true, true);
        assert!(s.unwrap().all_fit && c.gpu_layers == GpuLayers::Auto && c.fit_target_mib == Some(1024), "{c:?}");
        // The plan: all on the GPU is known, so no CPU thread settings.
        let caps = RuntimeCapabilities::parse(HELP);
        let p = plan_load(&share(4096), &dense(), 4_900_000_000, 0, &penv(&caps, Some(12 * GIB), AVAIL_RAM, &no_obs));
        assert_eq!((p.config.gpu_layers, p.cpu_layers, p.config.threads), (GpuLayers::All, Some(false), None));
        // Tight: llama.cpp decides; threads only once a load showed a split.
        let p = plan_load(&share(4096), &dense(), 4_900_000_000, 0, &penv(&caps, Some(free), AVAIL_RAM, &no_obs));
        assert_eq!((p.config.gpu_layers, p.cpu_layers, p.config.threads), (GpuLayers::Auto, None, None));
        let note = after_load_note(&p.config, p.cpu_layers, Some((30, 33)), Some(ULTRA7), &caps).expect("split seen after load");
        assert!(note.contains("3 of 33 layers on the CPU") && note.contains("12 generation and 20 prompt threads"), "{note}");
        assert!(after_load_note(&p.config, p.cpu_layers, Some((33, 33)), Some(ULTRA7), &caps).is_none());
        assert!(after_load_note(&p.config, Some(true), Some((30, 33)), Some(ULTRA7), &caps).is_none(), "already applied");
        // The next load with the same settings reads the measurement.
        let key = offload_key(&p.config);
        let seen = move |c: &LoadConfig| (offload_key(c) == key).then_some((30u32, 33u32));
        let p2 = plan_load(&share(4096), &dense(), 4_900_000_000, 0, &penv(&caps, Some(free), AVAIL_RAM, &seen));
        assert_eq!((p2.cpu_layers, p2.config.threads, p2.config.threads_batch), (Some(true), Some(12), Some(20)));
    }

    #[test]
    fn an_explicit_kv_type_wins_over_auto() {
        let caps = RuntimeCapabilities::parse(HELP);
        let cfg = LoadConfig { n_ctx: 8192, cache_type_auto: true, cache_type_k: "q8_0".into(), cache_type_v: "q8_0".into(), ..Default::default() };
        let (c, _, kv) = resolve_kv_and_ctx(&cfg, &dense(), 4_900_000_000, 0, &penv(&caps, Some(8 * GIB), AVAIL_RAM, &no_obs));
        assert_eq!((c.cache_type_k.as_str(), c.cache_type_v.as_str()), ("q8_0", "q8_0"));
        assert!(kv.is_none(), "not an Auto choice");
    }

    #[test]
    fn fit_margin_adds_the_projector_not_the_mtp_kv() {
        // RT-2: the projector is added; the MTP context is not (measured: --fit counts it).
        assert_eq!(fit_margin_mib(1024, 0, 0), 1024);
        assert_eq!(fit_margin_mib(1024, 900_000_000, 0), 1024 + 859, "858.3 MiB rounds up");
        assert_eq!(fit_margin_mib(512, 0, 512 * MIB), 512, "fit counts the MTP context itself");
        assert_eq!(fit_margin_mib(1024, 900_000_000, 512 * MIB + 1), 1024 + 859);
        assert_eq!(fit_margin_mib(65_000, 10 * GIB, 0), 65_536, "capped at what validate allows");
        assert_eq!(fit_margin_mib(u32::MAX, u64::MAX, u64::MAX), 65_536);
        let (c, _) = resolve_share(&share(32_768), &dense(), 16_000_000_000, 900_000_000, 512 * MIB, Some(8 * GIB), true, true);
        assert_eq!(c.fit_target_mib, Some(1024 + 859));
        assert!(c.validate().is_ok());
        // The 27B with MTP on: the margin stays the base (fit counts the MTP context).
        let caps = RuntimeCapabilities::parse(HELP);
        let cfg = LoadConfig { n_ctx: 32_768, share_with_cpu: true, speculative: Some("draft-mtp".into()), ..Default::default() };
        let p = plan_load(&cfg, &qwen27b(), Q27_SIZE, 0, &penv(&caps, Some(7 * GIB), AVAIL_RAM, &no_obs));
        let one_layer = 32_768 * 4 * 256 * 2 * 2u64; // 1 layer x n_ctx x 4 KiB, f16
        assert!(p.mtp_on);
        let _ = one_layer;
        assert_eq!(p.config.fit_target_mib, Some(1024), "{:?}", p.config);
    }

    #[test]
    fn share_without_fit_flags_falls_back_to_the_estimated_layer_count() {
        let (c, s) = resolve_share(&share(32_768), &dense(), 16_000_000_000, 0, 0, Some(8 * GIB), true, false);
        let s = s.unwrap();
        assert_eq!(c.gpu_layers, GpuLayers::Count(s.gpu_layers as u32));
        assert!(c.fit_target_mib.is_none() && c.fit_ctx.is_none());
        let old_help = "-c, --ctx-size N x\n-ngl,  --gpu-layers, --n-gpu-layers N   x\n-fit,  --fit [on|off]                   x\n";
        assert!(!runtime_can_fit(&RuntimeCapabilities::parse(old_help)) && runtime_can_fit(&RuntimeCapabilities::parse(HELP)));
        assert_eq!(val(&args(&c, old_help), "--n-gpu-layers"), Some(s.gpu_layers.to_string().as_str()));
        // Fits fully: all layers, as before.
        let (c, _) = resolve_share(&share(4096), &dense(), 4_900_000_000, 0, 0, Some(12 * GIB), true, false);
        assert_eq!(c.gpu_layers, GpuLayers::All);
    }

    #[test]
    fn share_moe_path_is_unchanged() {
        let moe = ModelMetadata { expert_count: Some(64), ..dense() };
        for can_fit in [true, false] {
            let (c, s) = resolve_share(&share(8192), &moe, 16_000_000_000, 0, 0, Some(8 * GIB), true, can_fit);
            let n = s.unwrap().n_cpu_moe.expect("moe split");
            assert_eq!((c.gpu_layers, c.n_cpu_moe, c.fit_target_mib), (GpuLayers::All, Some(n as u32), None), "can_fit {can_fit}");
            assert_eq!(val(&args(&c, HELP), "--n-cpu-moe"), Some(n.to_string().as_str()));
        }
        // Nothing fits at all: CPU only, as before.
        let (c, _) = resolve_share(&share(8192), &dense(), 16_000_000_000, 0, 0, Some(0), true, true);
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
    fn the_27b_gets_131072_sized_with_f16_and_q8_0_at_that_context() {
        // Qwen3.8-27B, 7 GiB free VRAM, 31.6 GB RAM with 23 GB available, Share with CPU, Auto context + Auto KV.
        let caps = RuntimeCapabilities::parse(HELP);
        let env = penv(&caps, Some(7 * GIB), AVAIL_RAM, &no_obs);
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_auto: true, ..Default::default() };
        let p = plan_load(&cfg, &qwen27b(), Q27_SIZE, 0, &env);
        // The f16 budget picks 131072 (q8_0 sizing would have reached 262144, the A/B's paging load).
        assert_eq!(p.sized.n_ctx, 131_072, "{:?}", p.auto_ctx);
        let f16 = LoadConfig { cache_type_auto: false, ..cfg.clone() };
        let (c, _) = resolve_auto_ctx(&f16, &qwen27b(), Q27_SIZE, 0, env.free_vram, AVAIL_RAM, true, true);
        assert_eq!(c.n_ctx, 131_072, "the f16 budget's step");
        // Split and the f16 KV at 131072 is ~8.6 GB: q8_0 at that same context, Flash Attention on.
        assert_eq!((p.sized.cache_type_k.as_str(), p.sized.cache_type_v.as_str(), p.sized.flash_attention), ("q8_0", "q8_0", FlashAttention::On));
        let kv = p.auto_kv.as_ref().unwrap();
        assert!(kv.saves_bytes > 3 * GIB && kv.summary.starts_with("KV cache: q8_0 (auto, saves "), "{kv:?}");
        // RT-15: the RAM side leaves the reserve (15% of 31.6 GB = 4.74 GB) free.
        assert_eq!(ram_reserve(TOTAL_RAM), 4_740_000_000);
        assert!(ram_side_bytes(&p.sized, &qwen27b(), Q27_SIZE, 0, &env) <= AVAIL_RAM - ram_reserve(TOTAL_RAM));
        // Known split (weights > VRAM): hybrid-CPU threads.
        assert_eq!((p.cpu_layers, p.config.threads, p.config.threads_batch), (Some(true), Some(12), Some(20)));
        assert_eq!(p.config.gpu_layers, GpuLayers::Auto, "llama.cpp fits the layers");
    }

    #[test]
    fn the_ram_reserve_lowers_an_auto_context_that_would_page() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = penv(&caps, Some(7 * GIB), AVAIL_RAM, &no_obs);
        // An explicit q8_0 cache sizes with q8_0: the old 90%-of-available budget allowed 262144.
        let q8 = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_k: "q8_0".into(), cache_type_v: "q8_0".into(), flash_attention: FlashAttention::On, ..Default::default() };
        let (unreserved, _) = resolve_auto_ctx(&q8, &qwen27b(), Q27_SIZE, 0, env.free_vram, AVAIL_RAM, true, true);
        assert_eq!(unreserved.n_ctx, 262_144);
        let (c, a, _) = resolve_kv_and_ctx(&q8, &qwen27b(), Q27_SIZE, 0, &env);
        let a = a.unwrap();
        assert!(c.n_ctx < 262_144 && a.limited_by == "ram_reserve" && a.fits, "{a:?}");
        let budget = AVAIL_RAM - ram_reserve(TOTAL_RAM);
        assert!(ram_side_bytes(&c, &qwen27b(), Q27_SIZE, 0, &env) <= budget);
        // The largest step that fits is kept.
        let next = estimate::ctx_steps(262_144).into_iter().find(|&n| n > c.n_ctx as u64).unwrap();
        assert!(ram_side_bytes(&LoadConfig { n_ctx: next as u32, ..c.clone() }, &qwen27b(), Q27_SIZE, 0, &env) > budget);
        // Plenty of RAM: no change.
        let big = penv(&caps, Some(7 * GIB), 120 * GIB, &no_obs);
        assert_eq!(resolve_kv_and_ctx(&q8, &qwen27b(), Q27_SIZE, 0, &big).0.n_ctx, 262_144);
        // Too little RAM for a dense model's weights: they never page, so the
        // minimum, flagged.
        let tiny = penv(&caps, Some(7 * GIB), 6 * GIB, &no_obs);
        let (c, a, _) = resolve_kv_and_ctx(&q8, &qwen27b(), Q27_SIZE, 0, &tiny);
        let a = a.unwrap();
        assert!(c.n_ctx == 2048 && !a.fits && !a.pages_weights, "{a:?}");
        assert_eq!(ram_fit(&c, &qwen27b(), Q27_SIZE, 0, &tiny), RamFit::TooBig);
        // Locked or read-in weights cannot page: the minimum, flagged, as before.
        let locked = LoadConfig { load_mode: inference::LoadMode::Mlock, ..q8.clone() };
        let (c, a, _) = resolve_kv_and_ctx(&locked, &qwen27b(), Q27_SIZE, 0, &tiny);
        assert!(c.n_ctx == 2048 && !a.unwrap().fits);
        // The reserve: 15% of total RAM, at least 4 GiB.
        assert_eq!(ram_reserve(16 * GIB), 4 * GIB);
        assert_eq!(ram_reserve(64_000_000_000), 9_600_000_000);
    }

    #[test]
    fn models_that_fit_on_the_gpu_keep_f16() {
        let caps = RuntimeCapabilities::parse(HELP);
        // Gemma-4-E4B, 7 GiB free, Share with CPU: f16 at its full 131072 (the A/B gave it q8_0 and its output changed).
        let env = penv(&caps, Some(7 * GIB), AVAIL_RAM, &no_obs);
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_auto: true, ..Default::default() };
        let p = plan_load(&cfg, &gemma4(), GEMMA_SIZE, 0, &env);
        assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str(), p.sized.cache_type_v.as_str()), (131_072, "f16", "f16"), "{:?}", p.auto_ctx);
        assert_eq!(p.sized.flash_attention, FlashAttention::Auto, "the user's Flash Attention setting stays");
        assert_eq!(p.auto_kv.as_ref().unwrap().summary, "KV cache: f16 (auto)");
        assert_eq!(p.cpu_layers, None, "llama.cpp decides; no split threads");
        // A long-context 8B on a 24 GiB GPU: the f16 cache is 16 GiB at 131072, all on the GPU -> f16.
        let big = penv(&caps, Some(24 * GIB), 64 * GIB, &no_obs);
        let p = plan_load(&cfg, &dense(), 4_900_000_000, 0, &big);
        assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str()), (131_072, "f16"));
        assert_eq!(p.config.gpu_layers, GpuLayers::All);
        // A known split (llama.cpp kept layers on the CPU last time) with a big f16 cache: q8_0.
        let seen = |_: &LoadConfig| Some((30u32, 43u32));
        let p = plan_load(&cfg, &gemma4(), GEMMA_SIZE, 0, &penv(&caps, Some(7 * GIB), AVAIL_RAM, &seen));
        assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str()), (131_072, "q8_0"));
        // Manual KV type: untouched, no Auto KV.
        let manual = LoadConfig { cache_type_auto: false, ..cfg.clone() };
        let p = plan_load(&manual, &qwen27b(), Q27_SIZE, 0, &env);
        assert!(p.auto_kv.is_none() && p.sized.cache_type_k == "f16");
    }

    #[test]
    fn the_prompt_cache_takes_the_ram_that_is_left_and_never_the_context() {
        let help_cram = format!("{HELP}-cram, --cache-ram N                    x
");
        let with = RuntimeCapabilities::parse(&help_cram);
        let without = RuntimeCapabilities::parse(HELP);
        let cfg = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_auto: true, ..Default::default() };
        let run = |caps: &RuntimeCapabilities, avail: u64| plan_load(&cfg, &gemma4(), GEMMA_SIZE, 0, &penv(caps, Some(7 * GIB), avail, &no_obs));
        // Plenty of RAM: llama.cpp's own 8192 MiB.
        assert_eq!(run(&with, 64 * GIB).config.cache_ram_mib, Some(8192));
        // Less RAM: only what is left after the load and the reserve.
        let p = run(&with, 10_000_000_000);
        let m = p.config.cache_ram_mib.unwrap();
        let used = ram_side_bytes(&p.config, &gemma4(), GEMMA_SIZE, 0, &penv(&with, Some(7 * GIB), 10_000_000_000, &no_obs));
        assert!(m > 0 && m < 8192 && used + m as u64 * MIB <= 10_000_000_000 - ram_reserve(TOTAL_RAM), "{m} MiB, used {used}");
        assert!(!p.notes.iter().any(|n| n.contains("--cache-ram")), "no warning for a normal, smaller cache: {:?}", p.notes);
        // Nothing left: off.
        let tight = run(&with, 4_500_000_000);
        assert_eq!(tight.config.cache_ram_mib, Some(0));
        // The context is the same with or without the flag, and at any RAM.
        for avail in [4_500_000_000u64, 10_000_000_000, 64 * GIB] {
            assert_eq!(run(&with, avail).config.n_ctx, run(&without, avail).config.n_ctx);
            assert_eq!(run(&with, avail).sized.n_ctx, run(&without, avail).sized.n_ctx);
        }
        // A runtime without --cache-ram: nothing set.
        assert_eq!(run(&without, 64 * GIB).config.cache_ram_mib, None);
    }

    #[test]
    fn speculative_mtp_never_lowers_the_auto_context() {
        // RT-1: a dense model with an MTP layer, sized by VRAM (GPU only).
        let m = ModelMetadata { block_count: Some(33), mtp_layers: Some(1), mtp_layers_declared: Some(1), ..dense() };
        let caps = RuntimeCapabilities::parse(HELP);
        let spec = LoadConfig { n_ctx: 0, gpu_layers: GpuLayers::All, speculative: Some("draft-mtp".into()), speculative_auto: true, ..Default::default() };
        let size = 4_900_000_000;
        let ctx = |free: u64, on: bool| auto_ctx_for(&spec, &spec_meta(&m, on), size, 0, Some(free), 64 * GIB, true, true).n_ctx;
        // A free-VRAM size where the MTP layer's KV cache would cost a context step.
        let free = (4 * GIB..12 * GIB).step_by(16 << 20).find(|&f| ctx(f, true) < ctx(f, false)).expect("such a size exists");
        let p = plan_load(&spec, &m, size, 0, &penv(&caps, Some(free), 64 * GIB, &no_obs));
        assert_eq!(p.sized.n_ctx as u64, ctx(free, false), "the context without MTP");
        assert!(!p.mtp_on && p.config.speculative.is_none(), "{:?}", p.config);
        assert!(p.notes.iter().any(|n| n.starts_with("Speculative decoding off: it would have reduced your context")), "{:?}", p.notes);
        // Room for both: MTP stays on at the same context.
        let p = plan_load(&spec, &m, size, 0, &penv(&caps, Some(24 * GIB), 64 * GIB, &no_obs));
        assert!(p.mtp_on && p.config.speculative.as_deref() == Some("draft-mtp") && p.notes.is_empty());
        assert_eq!(p.sized.n_ctx, 131_072);
        // A fixed context is never changed; MTP stays.
        let fixed = LoadConfig { n_ctx: 8192, ..spec.clone() };
        let p = plan_load(&fixed, &m, size, 0, &penv(&caps, Some(free), 64 * GIB, &no_obs));
        assert!(p.mtp_on && p.sized.n_ctx == 8192);
        // A runtime without draft-mtp: nothing charged, so the context is the plain one.
        let old = RuntimeCapabilities::parse("-c, --ctx-size N x\n--spec-type none,ngram-simple\n");
        let p = plan_load(&spec, &m, size, 0, &penv(&old, Some(free), 64 * GIB, &no_obs));
        assert!(!p.mtp_on && p.sized.n_ctx as u64 == ctx(free, false) && p.notes.is_empty());
        // The 27B: MTP on keeps 131072.
        let q = LoadConfig { n_ctx: 0, share_with_cpu: true, cache_type_auto: true, speculative: Some("draft-mtp".into()), speculative_auto: true, ..Default::default() };
        let p = plan_load(&q, &qwen27b(), Q27_SIZE, 0, &penv(&caps, Some(7 * GIB), AVAIL_RAM, &no_obs));
        assert!(p.mtp_on && p.sized.n_ctx == 131_072 && p.sized.cache_type_k == "q8_0", "{:?} {:?}", p.auto_ctx, p.notes);
        // spec_keeps_context itself.
        let a = |n: u64, fits: bool| AutoCtx { n_ctx: n, trained: None, limited_by: String::new(), fits, pages_weights: false, summary: String::new() };
        assert!(spec_keeps_context(&a(8192, true), &a(8192, true)));
        assert!(!spec_keeps_context(&a(8192, true), &a(6144, true)));
        assert!(!spec_keeps_context(&a(2048, true), &a(2048, false)));
        assert!(spec_keeps_context(&a(2048, false), &a(2048, false)));
    }

    // ---- Auto context fixes, 2026-11-13. The PC as measured: RTX 5060 with
    // 7418 MiB free, 20.3 GiB available of 31.6 GiB RAM.
    const FREE_7418: u64 = 7418 * MIB;
    const AVAIL_20_3: u64 = 21_797_000_000;
    const TOTAL_31_6: u64 = 33_930_000_000;

    fn pc<'a>(caps: &'a RuntimeCapabilities, observed: &'a (dyn Fn(&LoadConfig) -> Option<(u32, u32)> + Sync)) -> PlanEnv<'a> {
        PlanEnv { free_vram: Some(FREE_7418), available_ram: AVAIL_20_3, total_ram: TOTAL_31_6, caps, uses_gpu: true, topo: Some(ULTRA7), observed, linked: &[] }
    }

    fn auto(share: bool) -> LoadConfig {
        LoadConfig { n_ctx: 0, share_with_cpu: share, cache_type_auto: true, ..Default::default() }
    }

    /// Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q5_K_XL (qwen35moe, 256 experts).
    fn moe35b() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen35moe".into()),
            block_count: Some(41),
            context_length: Some(262_144),
            embedding_length: Some(2048),
            head_count: Some(16),
            head_count_kv: Some(2),
            key_length: Some(256),
            value_length: Some(256),
            expert_count: Some(256),
            vocab_size: Some(248_320),
            mtp_layers: Some(1),
            mtp_layers_declared: Some(1),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }
    const MOE_SIZE: u64 = 26_981_933_376;

    #[test]
    fn a_moe_bigger_than_free_ram_pages_instead_of_dropping_to_the_minimum() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        // The weights alone (27 GB file) exceed available RAM minus the reserve.
        assert!(MOE_SIZE > AVAIL_20_3 - ram_reserve(TOTAL_31_6));
        for share in [true, false] {
            let p = plan_load(&auto(share), &moe35b(), MOE_SIZE, 0, &env);
            let a = p.auto_ctx.clone().unwrap();
            assert!(p.sized.n_ctx > 2048 && a.limited_by != "minimum" && a.fits && a.pages_weights, "share {share}: {a:?}");
            assert_eq!(ram_fit(&p.sized, &moe35b(), MOE_SIZE, 0, &env), RamFit::PagesWeights, "share {share}");
            // Split, so sized with f16 and q8_0 only at that same context.
            let f16 = LoadConfig { cache_type_auto: false, ..auto(share) };
            let (c, a16) = resolve_auto_ctx(&f16, &moe35b(), MOE_SIZE, 0, env.free_vram, AVAIL_20_3, true, true);
            assert!(c.n_ctx >= p.sized.n_ctx && a16.unwrap().pages_weights, "share {share}");
            assert!(p.auto_kv.as_ref().unwrap().raised_from.is_none());
        }
        assert!(paging_warning(&moe35b()).starts_with("This model is larger than free RAM; Windows will page it from disk, which is slower."));
        // Locked weights cannot page: the minimum, flagged, as before.
        let locked = LoadConfig { load_mode: inference::LoadMode::Mlock, ..auto(true) };
        let p = plan_load(&locked, &moe35b(), MOE_SIZE, 0, &env);
        assert!(p.sized.n_ctx == 2048 && !p.auto_ctx.unwrap().fits);
        assert_eq!(ram_fit(&p.sized, &moe35b(), MOE_SIZE, 0, &env), RamFit::TooBig);
    }

    #[test]
    fn dense_27b_keeps_131072_and_a_dense_model_that_does_not_fit_is_flagged() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        let stripped = ModelMetadata { mtp_layers: None, ..qwen27b() };
        for (m, size) in [(qwen27b(), Q27_SIZE), (stripped, 16_547_400_512)] {
            let p = plan_load(&auto(true), &m, size, 0, &env);
            let a = p.auto_ctx.clone().unwrap();
            assert_eq!(p.sized.n_ctx, 131_072, "{a:?}");
            assert!(!a.pages_weights && a.fits, "{a:?}");
            assert_eq!(ram_fit(&p.sized, &m, size, 0, &env), RamFit::Fits);
            assert_eq!(p.sized.cache_type_k, "q8_0", "split: q8_0 at the f16 context");
            assert!(p.auto_kv.unwrap().raised_from.is_none());
        }
        // A 40 GB dense model never pages: Auto is the minimum, and it is flagged.
        let big = ModelMetadata { block_count: Some(80), ..dense() };
        let p = plan_load(&auto(true), &big, 40_000_000_000, 0, &env);
        let a = p.auto_ctx.unwrap();
        assert!(!a.pages_weights && !a.fits && p.sized.n_ctx == 2048, "{a:?}");
        assert_eq!(ram_fit(&p.sized, &big, 40_000_000_000, 0, &env), RamFit::TooBig);
    }

    #[test]
    fn gemma4_per_layer_embeddings_let_it_reach_131072_without_sharing() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        let ple = ModelMetadata { host_weight_bytes: Some(2_312_110_080), ..gemma4() };
        for share in [false, true] {
            let p = plan_load(&auto(share), &ple, GEMMA_SIZE, 0, &env);
            assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str(), p.sized.cache_type_v.as_str()), (131_072, "f16", "f16"), "share {share}: {:?}", p.auto_ctx);
            assert_eq!(p.auto_ctx.as_ref().unwrap().limited_by, "trained");
            assert_eq!(p.sized.flash_attention, FlashAttention::Auto);
        }
        // Fully on the GPU by the estimate: about 5.8 GB at 131072 f16.
        let e = estimate::estimate(&ple, GEMMA_SIZE, &estimate_input(&LoadConfig { n_ctx: 131_072, ..Default::default() }, &ple, 43, 0));
        assert!(e.vram_bytes < 6_000_000_000, "{}", e.vram_bytes);
        // Before (the whole file counted as GPU weights): 49152 with f16.
        let f16 = LoadConfig { cache_type_auto: false, ..auto(false) };
        let (c, _) = resolve_auto_ctx(&f16, &gemma4(), GEMMA_SIZE, 0, env.free_vram, AVAIL_20_3, true, true);
        assert_eq!(c.n_ctx, 49_152);
        // The weights that can go to the GPU exclude the table.
        assert_eq!(estimate::offloadable_weight_bytes(&ple, GEMMA_SIZE), GEMMA_SIZE - 2_312_110_080);
    }

    fn glm46v() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("glm4".into()),
            block_count: Some(40),
            context_length: Some(131_072),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(2),
            vocab_size: Some(151_552),
            ..Default::default()
        }
    }
    const GLM_SIZE: u64 = 6_166_577_888;

    /// Cyber-Ornith-1.5-9B Q4_K_S (qwen35 hybrid).
    fn ornith9b() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen35".into()),
            block_count: Some(32),
            context_length: Some(262_144),
            embedding_length: Some(4096),
            head_count: Some(16),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            vocab_size: Some(248_320),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }
    const ORNITH_SIZE: u64 = 5_351_632_448;

    #[test]
    fn full_gpu_models_use_q8_0_when_it_reaches_a_larger_step() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        for (name, m, size, f16_ctx) in [("glm", glm46v(), GLM_SIZE, 8192u64), ("ornith", ornith9b(), ORNITH_SIZE, 24_576)] {
            // The f16 Auto context (today's value).
            let f16 = LoadConfig { cache_type_auto: false, ..auto(false) };
            let (c, _) = resolve_auto_ctx(&f16, &m, size, 0, env.free_vram, AVAIL_20_3, true, true);
            assert_eq!(c.n_ctx as u64, f16_ctx, "{name}");
            // Auto KV, Share with CPU off: q8_0 K and V (Flash Attention on) at a larger step.
            let p = plan_load(&auto(false), &m, size, 0, &env);
            let kv = p.auto_kv.clone().unwrap();
            assert!(p.sized.n_ctx as u64 > f16_ctx, "{name}: {:?}", p.auto_ctx);
            assert_eq!((p.sized.cache_type_k.as_str(), p.sized.cache_type_v.as_str(), p.sized.flash_attention), ("q8_0", "q8_0", FlashAttention::On), "{name}");
            assert_eq!(kv.raised_from, Some(f16_ctx), "{name}");
            assert!(kv.summary.starts_with("KV cache: q8_0 (auto) gives "), "{name}: {}", kv.summary);
            assert!(p.auto_ctx.as_ref().unwrap().summary.contains("q8_0 KV cache"), "{name}");
            // Still fully on the GPU by the estimate at the new context.
            let e = estimate::estimate(&m, size, &estimate_input(&LoadConfig { gpu_layers: GpuLayers::All, ..p.sized.clone() }, &m, m.block_count.unwrap() + 1, 0));
            assert!(e.vram_bytes <= (FREE_7418 as f64 * 0.90) as u64, "{name}");
            assert_eq!(p.cpu_layers, None, "{name}: no split threads");
            // Flash Attention off: q8_0 K only, and only if that alone reaches a larger step.
            let fa_off = LoadConfig { flash_attention: FlashAttention::Off, ..auto(false) };
            let p = plan_load(&fa_off, &m, size, 0, &env);
            assert_eq!(p.sized.cache_type_v, "f16", "{name}");
            assert_eq!(p.sized.flash_attention, FlashAttention::Off, "{name}");
            if p.sized.cache_type_k == "q8_0" {
                assert!(p.sized.n_ctx as u64 > f16_ctx, "{name}");
            } else {
                assert_eq!(p.sized.n_ctx as u64, f16_ctx, "{name}");
            }
            // A runtime without q8_0: f16 at the f16 step.
            let no_q8 = RuntimeCapabilities::parse(&HELP.replace("q8_0, ", ""));
            let p = plan_load(&auto(false), &m, size, 0, &pc(&no_q8, &no_obs));
            assert_eq!((p.sized.n_ctx as u64, p.sized.cache_type_k.as_str()), (f16_ctx, "f16"), "{name}");
            // An explicit type or a fixed context: untouched.
            let fixed = LoadConfig { n_ctx: 8192, ..auto(false) };
            let p = plan_load(&fixed, &m, size, 0, &env);
            assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str()), (8192, "f16"), "{name}");
        }
        // Reaching the trained context with f16 already: f16 stays (DeepSeek 1.5B-style small model).
        let small = ModelMetadata { block_count: Some(28), embedding_length: Some(1536), head_count: Some(12), head_count_kv: Some(2), vocab_size: Some(151_936), ..dense() };
        let p = plan_load(&auto(false), &small, 1_464_179_808, 0, &env);
        assert_eq!((p.sized.n_ctx, p.sized.cache_type_k.as_str()), (131_072, "f16"), "{:?}", p.auto_ctx);
        assert!(p.auto_kv.unwrap().raised_from.is_none());
    }

    #[test]
    fn split_models_never_get_a_larger_step_from_q8_0() {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        // 16 GB dense, Share with CPU: the context is never above the f16 Auto step
        // (the RAM reserve is then checked with the KV type chosen).
        let f16 = LoadConfig { cache_type_auto: false, ..auto(true) };
        let (c16, _) = resolve_auto_ctx(&f16, &dense(), 16_000_000_000, 0, env.free_vram, AVAIL_20_3, true, true);
        let p = plan_load(&auto(true), &dense(), 16_000_000_000, 0, &env);
        assert!(p.sized.n_ctx <= c16.n_ctx, "{} > {}", p.sized.n_ctx, c16.n_ctx);
        assert!(p.auto_kv.unwrap().raised_from.is_none());
        // A known split (llama.cpp kept layers on the CPU last time) of a model that would fit: no raise either.
        let seen = |_: &LoadConfig| Some((30u32, 41u32));
        let (cg, _) = resolve_auto_ctx(&LoadConfig { cache_type_auto: false, ..auto(false) }, &glm46v(), GLM_SIZE, 0, env.free_vram, AVAIL_20_3, true, true);
        let p = plan_load(&auto(false), &glm46v(), GLM_SIZE, 0, &pc(&caps, &seen));
        assert_eq!(p.sized.n_ctx, cg.n_ctx);
        assert!(p.auto_kv.unwrap().raised_from.is_none());
    }

    #[test]
    fn the_27b_reserve_check_uses_the_kv_type_that_loads() {
        // Combined build, measured: 6766 MiB free VRAM, 31.6 GiB RAM; 17.0 and 20.3 GiB available.
        let caps = RuntimeCapabilities::parse(HELP);
        let total = (31.6 * GIB as f64) as u64;
        let stripped = ModelMetadata { mtp_layers: None, ..qwen27b() };
        for (name, m, size) in [("27b", qwen27b(), Q27_SIZE), ("27b-abl", stripped, 16_547_400_512)] {
            for avail_gib in [17.0f64, 20.3] {
                let avail = (avail_gib * GIB as f64) as u64;
                let env = PlanEnv { free_vram: Some(6766 * MIB), available_ram: avail, total_ram: total, caps: &caps, uses_gpu: true, topo: Some(ULTRA7), observed: &no_obs, linked: &[] };
                let budget = avail - ram_reserve(total);
                for share in [true, false] {
                    let p = plan_load(&auto(share), &m, size, 0, &env);
                    let a = p.auto_ctx.clone().unwrap();
                    let f16 = LoadConfig { cache_type_auto: false, ..auto(share) };
                    let (ceil, _) = resolve_auto_ctx(&f16, &m, size, 0, env.free_vram, avail, true, true);
                    let need = |n: u32, t: &str| ram_side(&LoadConfig { n_ctx: n, cache_type_k: t.into(), cache_type_v: t.into(), ..p.sized.clone() }, &m, size, 0, &env);
                    // The reserve is checked with the KV type that loads (q8_0, split), at most the f16 step.
                    assert_eq!(p.sized.cache_type_k, "q8_0", "{name}");
                    assert!(p.sized.n_ctx <= ceil.n_ctx, "{name}");
                    assert!(need(p.sized.n_ctx, "q8_0").total() <= budget, "{name} {avail_gib}");
                    if p.sized.n_ctx < ceil.n_ctx {
                        let next = estimate::ctx_steps(262_144).into_iter().find(|&n| n > p.sized.n_ctx as u64).unwrap();
                        assert!(need(next as u32, "q8_0").total() > budget, "{name} {avail_gib}: the next step really needs more than available - reserve with q8_0");
                    }
                    let want = match (avail_gib < 18.0, name) {
                        (true, _) => 32_768,
                        (false, "27b") => 98_304,
                        (false, _) => 131_072,
                    };
                    assert_eq!(p.sized.n_ctx, want, "{name} {avail_gib} share {share}: {a:?}");
                    println!(
                        "{name} avail {avail_gib} share {share}: Auto {} ({}) KV {} | f16 ceiling {} | budget {:.2} GB | q8 need 131072 {:.2} / 98304 {:.2} / 65536 {:.2} / 49152 {:.2} / 32768 {:.2} GB (weights {:.2}) | f16 need 131072 {:.2}",
                        p.sized.n_ctx, a.limited_by, p.sized.cache_type_k, ceil.n_ctx, budget as f64 / 1e9,
                        need(131_072, "q8_0").total() as f64 / 1e9, need(98_304, "q8_0").total() as f64 / 1e9, need(65_536, "q8_0").total() as f64 / 1e9,
                        need(49_152, "q8_0").total() as f64 / 1e9, need(32_768, "q8_0").total() as f64 / 1e9, need(32_768, "q8_0").weights as f64 / 1e9,
                        need(131_072, "f16").total() as f64 / 1e9,
                    );
                }
            }
        }
    }

    /// Auto values for the real model files (headers only), on the measured PC:
    /// `cargo test -p local-ai-studio real_files_auto_ctx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_files_auto_ctx() {
        let Ok(dir) = std::env::var("LAS_REAL_MODELS") else {
            eprintln!("skipping: set LAS_REAL_MODELS to a models directory");
            return;
        };
        let caps = RuntimeCapabilities::parse(HELP);
        let env = pc(&caps, &no_obs);
        for m in model_registry::scan::scan(Path::new(&dir)).models {
            if m.metadata.is_projector {
                continue;
            }
            for share in [false, true] {
                let p = plan_load(&auto(share), &m.metadata, m.total_size, 0, &env);
                let a = p.auto_ctx.unwrap();
                println!(
                    "{} share={share}: {} ({}, fits {}, pages {}) KV {}/{} | {}",
                    m.display_name,
                    a.n_ctx,
                    a.limited_by,
                    a.fits,
                    a.pages_weights,
                    p.sized.cache_type_k,
                    p.sized.cache_type_v,
                    p.auto_kv.map(|k| k.summary).unwrap_or_default()
                );
            }
        }
    }

    #[test]
    fn hybrid_cpu_thread_policy() {
        assert_eq!(hybrid_cpu_threads(Some(ULTRA7), true), Some((12, 20)));
        // Everything on the GPU, a non-hybrid CPU or no reading: defaults.
        assert_eq!(hybrid_cpu_threads(Some(ULTRA7), false), None);
        assert_eq!(hybrid_cpu_threads(Some(CoreTopology { performance_cores: 8, efficiency_cores: 0 }), true), None);
        assert_eq!(hybrid_cpu_threads(None, true), None);
        // 6P + 2E: P + 4 capped at the physical cores.
        assert_eq!(hybrid_cpu_threads(Some(CoreTopology { performance_cores: 6, efficiency_cores: 2 }), true), Some((8, 8)));
        let caps = RuntimeCapabilities::parse(HELP);
        let c = apply_thread_policy(&LoadConfig::default(), Some(ULTRA7), true, &caps);
        assert_eq!((c.threads, c.threads_batch), (Some(12), Some(20)));
        let a = args(&c, HELP);
        assert_eq!((val(&a, "--threads"), val(&a, "--threads-batch")), (Some("12"), Some("20")));
        // The user's own generation threads: prompt threads follow them (llama.cpp's default).
        let mine = apply_thread_policy(&LoadConfig { threads: Some(6), ..Default::default() }, Some(ULTRA7), true, &caps);
        assert_eq!((mine.threads, mine.threads_batch), (Some(6), None));
        assert!(!args(&mine, HELP).iter().any(|x| x == "--threads-batch"));
        let both = apply_thread_policy(&LoadConfig { threads: Some(6), threads_batch: Some(8), ..Default::default() }, Some(ULTRA7), true, &caps);
        assert_eq!((both.threads, both.threads_batch), (Some(6), Some(8)));
        // Only prompt threads set: generation gets the policy, prompt stays the user's.
        let tb = apply_thread_policy(&LoadConfig { threads_batch: Some(4), ..Default::default() }, Some(ULTRA7), true, &caps);
        assert_eq!((tb.threads, tb.threads_batch), (Some(12), Some(4)));
        // A build without the flags gets nothing (no "unsupported" noise).
        let none = apply_thread_policy(&LoadConfig::default(), Some(ULTRA7), true, &RuntimeCapabilities::parse("-c, --ctx-size N x\n"));
        assert_eq!((none.threads, none.threads_batch), (None, None));
    }

    #[test]
    fn which_loads_run_layers_on_the_cpu() {
        let auto = LoadConfig::default();
        assert_eq!(cpu_runs_layers(&auto, 33, true, true, None), Some(true), "weights bigger than VRAM");
        assert_eq!(cpu_runs_layers(&auto, 33, true, false, None), None, "llama.cpp decides at load");
        assert_eq!(cpu_runs_layers(&auto, 33, true, false, Some((20, 33))), Some(true), "seen split");
        assert_eq!(cpu_runs_layers(&auto, 33, true, true, Some((33, 33))), Some(false), "seen all on the GPU");
        assert_eq!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::All, ..auto.clone() }, 33, true, true, None), Some(false));
        assert_eq!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::Count(20), ..auto.clone() }, 33, true, false, None), Some(true));
        assert_eq!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::Count(33), ..auto.clone() }, 33, true, false, None), Some(false));
        assert_eq!(cpu_runs_layers(&LoadConfig { gpu_layers: GpuLayers::CpuOnly, ..auto.clone() }, 33, true, false, None), Some(true));
        assert_eq!(cpu_runs_layers(&LoadConfig { n_cpu_moe: Some(4), gpu_layers: GpuLayers::All, ..auto.clone() }, 33, true, false, None), Some(true));
        assert_eq!(cpu_runs_layers(&LoadConfig { kv_offload: false, ..auto.clone() }, 33, true, false, None), Some(true));
        assert_eq!(cpu_runs_layers(&auto, 33, false, false, None), Some(true), "CPU runtime");
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

/// Auto context with linked PCs ("update the context window limit depending
/// on linked computers", 2026-11-23).
#[cfg(test)]
mod linked_tests {
    use super::*;
    use inference::RuntimeCapabilities;

    const GIB: u64 = 1 << 30;
    const TOTAL_RAM: u64 = 31_600_000_000;
    const AVAIL_RAM: u64 = 23_000_000_000;
    const HELP: &str = "\
-c,    --ctx-size N                     x
-ngl,  --gpu-layers, --n-gpu-layers N   x
-fa,   --flash-attn [on|off|auto]       x
-ctk,  --cache-type-k TYPE              KV cache data type for K
                                        allowed values: f32, f16, bf16, q8_0, q4_0
-ctv,  --cache-type-v TYPE              x
-fit,  --fit [on|off]                   x
-fitt, --fit-target MiB0,MiB1,MiB2,...
-ncmoe, --n-cpu-moe N                   x
";

    fn no_obs(_: &LoadConfig) -> Option<(u32, u32)> {
        None
    }

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
    const DENSE_SIZE: u64 = 4_900_000_000;
    const DENSE_BIG: u64 = 16_000_000_000;

    fn moe() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen35moe".into()),
            block_count: Some(41),
            context_length: Some(262_144),
            embedding_length: Some(2048),
            head_count: Some(16),
            head_count_kv: Some(2),
            key_length: Some(256),
            value_length: Some(256),
            expert_count: Some(256),
            vocab_size: Some(248_320),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }
    const MOE_SIZE: u64 = 26_981_933_376;

    /// Qwen3.8-27B Q4_K_M (hybrid attention), as in the tests above.
    fn qwen27b() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen35".into()),
            block_count: Some(65),
            context_length: Some(262_144),
            embedding_length: Some(5120),
            head_count: Some(24),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            vocab_size: Some(248_320),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }
    const Q27_SIZE: u64 = 16_810_714_336;

    fn plan(meta: &ModelMetadata, size: u64, cfg: &LoadConfig, free: u64, ram: u64, linked: &[LinkedDev]) -> LoadPlan {
        let caps = RuntimeCapabilities::parse(HELP);
        let env = PlanEnv { free_vram: Some(free), available_ram: ram, total_ram: TOTAL_RAM, caps: &caps, uses_gpu: true, topo: None, observed: &no_obs, linked };
        plan_load(cfg, meta, size, 0, &env)
    }

    fn auto(share: bool) -> LoadConfig {
        LoadConfig { n_ctx: 0, share_with_cpu: share, cache_type_auto: true, ..Default::default() }
    }

    fn gpu(gb: u64) -> LinkedDev {
        LinkedDev { bytes: gb * GIB, cpu: false }
    }

    #[test]
    fn auto_context_never_shrinks_when_linked_capacity_grows() {
        for (name, meta, size) in [("dense 5 GB", dense(), DENSE_SIZE), ("dense 16 GB", dense(), DENSE_BIG), ("dense 27B", qwen27b(), Q27_SIZE), ("moe", moe(), MOE_SIZE)] {
            for share in [true, false] {
                for cpu in [false, true] {
                    let alone = plan(&meta, size, &auto(share), 7 * GIB, AVAIL_RAM, &[]).sized.n_ctx;
                    let mut prev = 0;
                    for gb in 0..=32u64 {
                        let dev = [LinkedDev { bytes: gb * GIB, cpu }];
                        let n = plan(&meta, size, &auto(share), 7 * GIB, AVAIL_RAM, &dev).sized.n_ctx;
                        assert!(n >= prev && n >= alone, "{name} share {share} cpu {cpu}: {gb} GB linked gave {n} (prev {prev}, alone {alone})");
                        prev = n;
                    }
                }
            }
        }
        // It really rises: a 16 GB dense model split on 7 GiB gets more with 16 GB lent.
        let alone = plan(&dense(), DENSE_BIG, &auto(true), 7 * GIB, AVAIL_RAM, &[]).sized.n_ctx;
        let linked = plan(&dense(), DENSE_BIG, &auto(true), 7 * GIB, AVAIL_RAM, &[gpu(16)]).sized.n_ctx;
        assert!(linked > alone, "alone {alone}, linked {linked}");
    }

    #[test]
    fn linked_split_counts_gpu_layers_itself() {
        // llama.cpp's --fit aborts once --tensor-split is set and then offloads
        // every layer (real 2-PC test, 2026-11-25: 25.2 GB asked of 12.2 GB
        // free). With linked PCs the plan gives a layer count instead.
        let total = qwen27b().block_count.unwrap() + 1;
        let alone = plan(&qwen27b(), Q27_SIZE, &auto(true), 7031 << 20, 21_600_000_000, &[]);
        assert_eq!(alone.config.gpu_layers, GpuLayers::Auto, "alone: llama.cpp fits the layers");
        let linked = plan(&qwen27b(), Q27_SIZE, &auto(true), 7031 << 20, 21_600_000_000, &[LinkedDev { bytes: 5120 << 20, cpu: false }]);
        match linked.config.gpu_layers {
            GpuLayers::Count(n) => assert!(n > 0 && u64::from(n) < total, "{n} of {total}"),
            other => panic!("linked: {other:?}"),
        }
    }

    #[test]
    fn dense_27b_auto_context_is_monotonic_around_the_weights_size() {
        // 27B Q4_K_M: the weights (16.8 GB) nearly fill 16 GiB. Auto used to dip
        // there (23 GB RAM: 15 GiB 262144, 16 GiB 196608, 18 GiB 262144).
        for ram in [17u64, 20, 23, 24, 32] {
            let at = |v: u64| plan(&qwen27b(), Q27_SIZE, &auto(true), v * GIB, ram * 1_000_000_000, &[]).sized.n_ctx;
            let row: Vec<u32> = [7u64, 12, 15, 16, 18, 24].iter().map(|&v| at(v)).collect();
            assert!(row.windows(2).all(|w| w[0] <= w[1]), "{ram} GB RAM: {row:?}");
        }
        assert_eq!(plan(&qwen27b(), Q27_SIZE, &auto(true), 15 * GIB, 23_000_000_000, &[]).sized.n_ctx, 262_144);
        assert_eq!(plan(&qwen27b(), Q27_SIZE, &auto(true), 16 * GIB, 23_000_000_000, &[]).sized.n_ctx, 262_144);
        // The owner's PC: 7309 MiB free VRAM, Share with CPU.
        for ram in [23_000_000_000u64, 24_000_000_000] {
            assert_eq!(plan(&qwen27b(), Q27_SIZE, &auto(true), 7309 << 20, ram, &[]).sized.n_ctx, 131_072, "{ram}");
        }
    }

    #[test]
    fn tiny_linked_device_never_lowers_context() {
        // Less than the compute buffer a device needs: adds nothing, takes nothing.
        for (meta, size) in [(dense(), DENSE_SIZE), (dense(), DENSE_BIG), (moe(), MOE_SIZE)] {
            for share in [true, false] {
                let alone = plan(&meta, size, &auto(share), 7 * GIB, AVAIL_RAM, &[]).sized.n_ctx;
                let tiny = plan(&meta, size, &auto(share), 7 * GIB, AVAIL_RAM, &[LinkedDev { bytes: 200 << 20, cpu: false }]).sized.n_ctx;
                assert_eq!(tiny, alone);
            }
        }
    }

    #[test]
    fn remote_cpu_device_keeps_f16_kv() {
        // A 5 GB model fully on a 7 GiB GPU: alone, q8_0 raises the context.
        let alone = plan(&dense(), DENSE_SIZE, &auto(false), 7 * GIB, AVAIL_RAM, &[]);
        let raised = alone.auto_kv.as_ref().and_then(|k| k.raised_from).expect("q8_0 raise alone");
        // A linked PC's CPU holds layers: split with a CPU, so the context is
        // sized with f16 and never raised by q8_0.
        let cpu = plan(&dense(), DENSE_SIZE, &auto(false), 7 * GIB, AVAIL_RAM, &[LinkedDev { bytes: 8 * GIB, cpu: true }]);
        let kv = cpu.auto_kv.as_ref().unwrap();
        assert!(kv.raised_from.is_none(), "{kv:?}");
        assert!(cpu.sized.n_ctx as u64 >= raised, "never below the f16 context alone");
        // Linked GPUs only: still all-GPU; never below alone.
        let g = plan(&dense(), DENSE_SIZE, &auto(false), 7 * GIB, AVAIL_RAM, &[gpu(8)]);
        assert!(g.sized.n_ctx >= alone.sized.n_ctx, "{} < {}", g.sized.n_ctx, alone.sized.n_ctx);
    }

    #[test]
    fn auto_context_never_shrinks_when_ram_or_vram_grow_with_a_6_gb_linked_gpu() {
        let six = [gpu(6)];
        for (name, meta, size) in [("moe", moe(), MOE_SIZE), ("dense 5 GB", dense(), DENSE_SIZE), ("dense 27B", qwen27b(), Q27_SIZE)] {
            for vram in [4u64, 7, 12] {
                let mut prev = 0;
                for gb in 8..=64u64 {
                    let n = plan(&meta, size, &auto(true), vram * GIB, gb * 1_000_000_000, &six).sized.n_ctx;
                    assert!(n >= prev, "{name}: {gb} GB RAM / {vram} GiB VRAM gave {n} < {prev}");
                    prev = n;
                }
            }
            for gb in [8u64, 17, 24, 32] {
                let mut prev = 0;
                for vram in 2..=24u64 {
                    let n = plan(&meta, size, &auto(true), vram * GIB, gb * 1_000_000_000, &six).sized.n_ctx;
                    assert!(n >= prev, "{name}: {vram} GiB VRAM / {gb} GB RAM gave {n} < {prev}");
                    prev = n;
                }
            }
        }
    }

    /// Qwen3-4B-Instruct-2507 Q4_K_M, the VM split test's model.
    fn qwen3_4b() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen3".into()),
            block_count: Some(36),
            context_length: Some(262_144),
            embedding_length: Some(2560),
            head_count: Some(32),
            head_count_kv: Some(8),
            key_length: Some(128),
            value_length: Some(128),
            vocab_size: Some(151_936),
            ..Default::default()
        }
    }
    const Q4B_SIZE: u64 = 2_497_280_256;

    /// The VM case (2026-11-24): "Use linked PCs", every layer on the GPU, a
    /// 4096 context, and a CPU-only linked PC. With a 3 GB limit it holds
    /// layers (in --rpc and --tensor-split); with a tiny one it is left out
    /// of --rpc, and the load panel and the engine status say why.
    #[test]
    fn cpu_only_linked_pc_is_split_with_at_3_gb_and_left_out_with_a_note_when_tiny() {
        use crate::services::linking::{net_split, LinkedLoad, LinkedShare};
        const MIB: u64 = 1 << 20;
        let ep = "127.0.0.1:47070".to_string();
        let names: std::collections::HashMap<String, String> = [(ep.clone(), "VM".to_string())].into();
        let cfg = LoadConfig { n_ctx: 4096, gpu_layers: GpuLayers::All, use_linked_pcs: true, rpc: vec![ep.clone()], tensor_split: Some("3072,7031".into()), ..Default::default() };
        let meta = qwen3_4b();
        for (lend_mib, included) in [(3072u64, true), (64, false)] {
            let share = LinkedShare { peer: "VM".into(), endpoint: ep.clone(), device: "RPC0".into(), bytes: lend_mib * MIB, cpu: true, cap_bytes: Some(lend_mib * MIB) };
            let linked = [LinkedDev { bytes: share.bytes, cpu: true }];
            let p = plan(&meta, Q4B_SIZE, &cfg, 7031 * MIB, AVAIL_RAM, &linked);
            // The planner keeps the linked PC and the fixed context.
            assert_eq!((p.config.rpc.clone(), p.config.n_ctx), (vec![ep.clone()], 4096), "{lend_mib} MiB");
            let total = meta.block_count.unwrap() + 1;
            let cb = estimate::estimate(&meta, Q4B_SIZE, &estimate_input(&p.config, &meta, total, 0)).compute_buffer_bytes;
            assert!(cb > 0 && cb < 1024 * MIB, "compute buffer {cb}");
            let devices = linking::split::parse_devices(&format!("  CUDA0: NVIDIA GeForce RTX 5060 (8123 MiB, 7031 MiB free)
  RPC0: {ep} ({lend_mib} MiB, {lend_mib} MiB free)
"));
            let l = LinkedLoad { devices, caps: vec![(ep.clone(), None)], names: names.clone(), shares: vec![share.clone()] };
            let (split, note) = net_split(&l, cb, 0);
            let (rpc, ts) = split.expect("a split");
            let note = note.expect("a note");
            // The load panel says the same before the load.
            let before = linked_left_out_notes(&[share], cb);
            if included {
                assert_eq!(rpc, vec![ep.clone()]);
                let ts = ts.expect("--tensor-split");
                let parts: Vec<u64> = ts.split(',').map(|x| x.parse().unwrap()).collect();
                // RPC device first, then this PC's GPU; both hold layers.
                assert_eq!(parts.len(), 2, "{ts}");
                assert!(parts[0] > 2000 && parts[1] > 5000, "{ts}");
                assert!(!note.contains("left out") && note.contains("\"VM\""), "{note}");
                assert!(before.is_empty(), "{before:?}");
            } else {
                assert!(rpc.is_empty() && ts.is_none(), "{rpc:?} {ts:?}");
                assert!(note.contains("\"VM\" left out") && note.contains("working buffer") && note.contains("Linked PCs page"), "{note}");
                assert_eq!(before.len(), 1);
                assert!(before[0].contains("\"VM\" left out") && before[0].contains("Linked PCs page"), "{before:?}");
            }
        }
        // A limit of 0 (the default RAM share of a 6 GB PC) names the limit.
        let n = crate::services::linking::left_out_note("VM", 0, Some(0), 300 * MIB);
        assert!(n.starts_with("\"VM\" left out: its owner lets this PC use 0.0 GB there, less than the 0.3 GB working buffer"), "{n}");
        // Limited by free memory, not by the limit.
        let n = crate::services::linking::left_out_note("VM", 100 * MIB, Some(3072 * MIB), 300 * MIB);
        assert!(n.contains("only 0.1 GB free (its limit allows 3.2 GB)"), "{n}");
    }

    /// The VM limits test (2026-11-24): Qwen3-4B fits this PC's GPU, so Auto
    /// alone takes q8_0 to 49152; with the VM's 2 GB CPU it gave 32768 (f16
    /// sizing as a split). Auto with linked PCs is never below alone.
    #[test]
    fn linked_cpu_never_lowers_auto_context_of_a_model_that_fits_here() {
        const MIB: u64 = 1 << 20;
        let cfg = LoadConfig { n_ctx: 0, gpu_layers: GpuLayers::All, cache_type_auto: true, use_linked_pcs: true, ..Default::default() };
        let alone = plan(&qwen3_4b(), Q4B_SIZE, &cfg, 7_725_641_728, 16_883_654_656, &[]).sized.n_ctx;
        for mib in [256u64, 1024, 2041, 3072, 8192] {
            let with = plan(&qwen3_4b(), Q4B_SIZE, &cfg, 7_725_641_728, 16_883_654_656, &[LinkedDev { bytes: mib * MIB, cpu: true }]).sized.n_ctx;
            assert!(with >= alone, "{mib} MiB linked CPU: {with} < alone {alone}");
        }
        assert!(alone > 24_576, "alone {alone}");
    }
}

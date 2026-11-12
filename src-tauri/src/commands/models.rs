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

/// Does the default runtime (setting, else the first installed) support `--n-cpu-moe`?
pub async fn default_runtime_has_n_cpu_moe(st: &SharedState) -> bool {
    let installed = st.runtimes.list_installed();
    let want = st.db.get_setting::<String>("runtime.default").await.ok().flatten();
    let rt = match want {
        Some(w) => installed.iter().find(|r| r.id == w),
        None => installed.first(),
    };
    rt.is_some_and(|r| r.capabilities.has("--n-cpu-moe"))
}

/// Turn the user's "Share with CPU" choice into concrete load settings.
/// Returns the config to load with (the user's own config is unchanged when
/// the toggle is off) and the split that was chosen.
/// `projector_bytes` is what [`projector_bytes`] returns for this load.
pub fn resolve_share(
    cfg: &LoadConfig,
    meta: &ModelMetadata,
    size: u64,
    projector_bytes: u64,
    free_vram: Option<u64>,
    has_n_cpu_moe: bool,
) -> (LoadConfig, Option<ShareSplit>) {
    if !cfg.share_with_cpu || matches!(cfg.gpu_layers, GpuLayers::CpuOnly) {
        return (cfg.clone(), None);
    }
    let mut c = cfg.clone();
    c.fit = true;
    let Some(free) = free_vram else {
        // No GPU information: let llama.cpp fit the layers itself.
        c.gpu_layers = GpuLayers::Auto;
        c.n_cpu_moe = None;
        return (c, None);
    };
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut input = estimate_input(cfg, meta, total, projector_bytes);
    input.gpu_layers = None;
    let split = estimate::share_split(meta, size, &input, free, has_n_cpu_moe);
    if split.all_fit {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = None;
    } else if let Some(n) = split.n_cpu_moe {
        c.gpu_layers = GpuLayers::All;
        c.n_cpu_moe = Some(n as u32);
    } else if split.gpu_layers == 0 {
        c.gpu_layers = GpuLayers::CpuOnly;
        c.n_cpu_moe = None;
    } else {
        c.gpu_layers = GpuLayers::Count(split.gpu_layers as u32);
        c.n_cpu_moe = None;
    }
    (c, Some(split))
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
    let has_moe_flag = default_runtime_has_n_cpu_moe(state.inner()).await;
    // Auto context, measured as a load would see it: a loaded model is
    // unloaded first, so its VRAM counts as free.
    let reclaim = loaded_backend_vram(state.inner());
    let auto_ctx = auto_ctx_for(
        &config,
        &meta,
        m.size as u64,
        proj,
        free_vram.map(|f| f.saturating_add(reclaim).min(gpu.map(|g| g.vram_total_bytes).unwrap_or(u64::MAX))),
        hw.available_ram_bytes,
        has_moe_flag,
        !matches!(config.gpu_layers, GpuLayers::CpuOnly),
    );
    let mut input = estimate_input(&config, &meta, total_layers, proj);
    if config.n_ctx == 0 {
        input.n_ctx = auto_ctx.n_ctx;
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
        // No saved choice: Auto context (fits the hardware at load time).
        let c = LoadConfig { n_ctx: 0, ..LoadConfig::default() };
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

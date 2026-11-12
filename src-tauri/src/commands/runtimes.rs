use std::path::PathBuf;

use app_core::AppError;
use inference::{AvailableRuntime, GpuHint, InstalledRuntime, driver_text, parse_driver_version};
use serde::Serialize;
use system_monitor::HardwareInfo;
use tauri::{AppHandle, Emitter, State};
use tokio_util::sync::CancellationToken;

use super::Cmd;
use crate::state::SharedState;

pub fn gpu_hint(state: &SharedState) -> GpuHint {
    hint_from(&state.monitor.hardware_info())
}

fn hint_from(hw: &HardwareInfo) -> GpuHint {
    let parse = |s: &str| {
        let (a, b) = s.split_once('.')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    };
    let g = hw.gpus.first();
    GpuHint {
        nvidia: g.is_some_and(|g| g.vendor == "NVIDIA"),
        compute_capability: g.and_then(|g| g.compute_capability.as_deref()).and_then(parse),
        cuda_driver: hw.cuda_driver_version.as_deref().and_then(parse),
        driver_version: g.and_then(|g| g.driver_version.as_deref()).and_then(parse_driver_version),
        vulkan_gpu: hw.adapters.iter().any(|a| a.is_vulkan_candidate()),
    }
}

// ---------------------------------------------------------------------------
// First-run setup
// ---------------------------------------------------------------------------

/// Setting written when the user finishes or skips first-run setup.
const SETUP_KEY: &str = "setup.completed";
/// Where NVIDIA drivers come from (shown and opened by the driver check).
pub const NVIDIA_DRIVER_URL: &str = "https://www.nvidia.com/en-us/drivers/";
/// CUDA version assumed when GitHub can't be reached (the b12250 build the app
/// was tested with is cuda-13.5).
const FALLBACK_CUDA: (u32, u32) = (13, 5);

/// One starter model the setup screen can suggest.
#[derive(Debug, Clone, Serialize)]
pub struct StarterModel {
    /// Hugging Face repo id.
    pub repo: String,
    /// Exact GGUF file in that repo (also Discover's variant key).
    pub file: String,
    pub name: String,
    pub size_bytes: u64,
    pub why: String,
}

/// The curated starter list, smallest first. Every repo and file was checked
/// against the Hugging Face API on 2026-11-11 (public, not gated, Apache-2.0).
/// `(repo, file, display name, size in bytes)`.
const STARTERS: &[(&str, &str, &str, u64)] = &[
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3 0.6B (Q4_K_M)", 396_705_472),
    ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf", "Qwen3 1.7B (Q4_K_M)", 1_107_409_472),
    (
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "Qwen3 4B Instruct 2507 (Q4_K_M)",
        2_497_281_120,
    ),
    ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "Qwen3 8B (Q4_K_M)", 5_027_783_488),
];

const GIB: u64 = 1 << 30;

/// Pick one starter for this hardware. `vram` is the GPU's total VRAM when the
/// runtime will use the GPU, else None. Leaves room for context and Windows.
pub fn pick_starter(vram: Option<u64>, ram: u64) -> StarterModel {
    let (i, why) = match vram {
        Some(v) if v >= 12 * GIB => (3, format!("Fits in your GPU's {} GB of VRAM with room for a long chat.", v / GIB)),
        Some(v) if v >= 6 * GIB => (2, format!("Fits in your GPU's {} GB of VRAM with room for a long chat.", v / GIB)),
        Some(v) if v >= 3 * GIB => (1, format!("Small enough for your GPU's {} GB of VRAM.", v / GIB)),
        _ if ram >= 6 * GIB => (1, "Runs on the CPU at a usable speed. Small and quick to download.".to_string()),
        _ => (0, "The smallest useful model; this PC has little free memory.".to_string()),
    };
    let (repo, file, name, size) = STARTERS[i];
    StarterModel { repo: repo.into(), file: file.into(), name: name.into(), size_bytes: size, why }
}

/// A GPU as the setup screen shows it (NVML for NVIDIA, DXGI for the rest).
#[derive(Debug, Clone, Serialize)]
pub struct SetupGpu {
    pub name: String,
    pub vendor: String,
    pub vram_total_bytes: u64,
    pub driver_version: Option<String>,
}

/// NVIDIA driver versus the CUDA build the app would install.
#[derive(Debug, Clone, Serialize)]
pub struct DriverCheck {
    pub gpu_name: String,
    pub driver_version: Option<String>,
    /// e.g. "13.5"
    pub cuda_version: String,
    pub minimum: String,
    pub recommended: String,
    /// ok | old_but_works | too_old | unknown
    pub status: String,
    pub message: String,
    pub download_url: String,
}

pub fn check_driver(gpu_name: &str, driver: Option<&str>, cuda: (u32, u32)) -> Option<DriverCheck> {
    let req = inference::cuda_driver_requirement(cuda)?;
    let cv = format!("{}.{}", cuda.0, cuda.1);
    let (min, rec) = (driver_text(req.minimum), driver_text(req.recommended));
    let parsed = driver.and_then(parse_driver_version);
    let (status, message) = match parsed {
        None => (
            "unknown",
            format!("Couldn't read the NVIDIA driver version. The CUDA {cv} build needs driver {min} or newer."),
        ),
        Some(d) if d < req.minimum => (
            "too_old",
            format!(
                "Your NVIDIA driver ({}) is too old for the CUDA {cv} build, which needs driver {min} or newer. \
                 Update the driver from NVIDIA, then open setup again from Settings, or use the Vulkan or CPU runtime now.",
                driver.unwrap_or_default()
            ),
        ),
        Some(d) if d < req.recommended => (
            "old_but_works",
            format!(
                "Your NVIDIA driver ({}) runs the CUDA {cv} build (needs {min} or newer). NVIDIA ships CUDA {cv} with driver {rec}; \
                 if a model fails to load, update the driver.",
                driver.unwrap_or_default()
            ),
        ),
        Some(_) => ("ok", format!("Driver {} is new enough for CUDA {cv}.", driver.unwrap_or_default())),
    };
    Some(DriverCheck {
        gpu_name: gpu_name.into(),
        driver_version: driver.map(str::to_string),
        cuda_version: cv,
        minimum: min,
        recommended: rec,
        status: status.into(),
        message,
        download_url: NVIDIA_DRIVER_URL.into(),
    })
}

/// Newest CUDA version among the offered builds that has kernels for this GPU
/// (what the app would install with an up-to-date driver).
fn best_cuda(available: &[AvailableRuntime], cc: Option<(u32, u32)>) -> Option<(u32, u32)> {
    available
        .iter()
        .filter_map(|r| r.variant.strip_prefix("cuda-"))
        .filter_map(|v| {
            let (a, b) = v.split_once('.').unwrap_or((v, "0"));
            Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
        })
        .filter(|v| !matches!(cc, Some(c) if c.0 >= 12 && *v < (12, 8)))
        .max()
}

#[derive(Debug, Serialize)]
pub struct SetupPlan {
    /// The user finished or skipped setup before.
    pub completed: bool,
    /// Open the setup screen by itself on launch.
    pub show: bool,
    pub installed: Vec<InstalledRuntime>,
    pub cpu_name: String,
    pub logical_cores: usize,
    pub total_ram_bytes: u64,
    pub gpus: Vec<SetupGpu>,
    pub gpu_note: Option<String>,
    /// From the same recommender the Runtimes page uses; recommended first.
    pub available: Vec<AvailableRuntime>,
    pub available_error: Option<String>,
    pub driver: Option<DriverCheck>,
    /// Starter when the runtime uses the GPU (None without a usable GPU).
    pub starter_gpu: Option<StarterModel>,
    pub starter_cpu: StarterModel,
}

/// Show on launch only for a profile that was never set up and has no runtime
/// yet (a fresh install). Existing profiles with a runtime are left alone.
pub fn should_show(completed: bool, runtimes_installed: usize) -> bool {
    !completed && runtimes_installed == 0
}

/// Everything the first-run screen needs: hardware, the recommended runtime,
/// the NVIDIA driver check and the starter model suggestions.
#[tauri::command]
pub async fn setup_plan(state: State<'_, SharedState>) -> Cmd<SetupPlan> {
    let st = state.inner().clone();
    let hw = tokio::task::spawn_blocking({
        let st = st.clone();
        move || st.monitor.hardware_info()
    })
    .await
    .map_err(|e| AppError::Other(e.to_string()))?;
    let hint = hint_from(&hw);
    let (available, available_error) = match st.runtimes.fetch_available(&hint).await {
        Ok(v) => (v, None),
        Err(e) => (Vec::new(), Some(e.to_string())),
    };
    let completed = st.db.get_setting::<serde_json::Value>(SETUP_KEY).await?.is_some();
    let installed = st.runtimes.list_installed();

    let mut gpus: Vec<SetupGpu> = hw
        .gpus
        .iter()
        .map(|g| SetupGpu {
            name: g.name.clone(),
            vendor: g.vendor.clone(),
            vram_total_bytes: g.vram_total_bytes,
            driver_version: g.driver_version.clone(),
        })
        .collect();
    for a in hw.adapters.iter().filter(|a| a.vendor != "NVIDIA" && a.vendor != "other") {
        gpus.push(SetupGpu {
            name: a.name.clone(),
            vendor: a.vendor.clone(),
            vram_total_bytes: a.dedicated_vram_bytes,
            driver_version: None,
        });
    }
    let driver = hw.gpus.iter().find(|g| g.vendor == "NVIDIA").and_then(|g| {
        let cuda = best_cuda(&available, hint.compute_capability).unwrap_or(FALLBACK_CUDA);
        check_driver(&g.name, g.driver_version.as_deref(), cuda)
    });
    // GPU starter: NVIDIA card (CUDA or Vulkan), else an AMD/Intel card with its own VRAM.
    let gpu_vram = hw
        .gpus
        .first()
        .map(|g| g.vram_total_bytes)
        .or_else(|| hw.adapters.iter().filter(|a| a.is_vulkan_candidate()).map(|a| a.dedicated_vram_bytes).max())
        .filter(|v| *v >= 3 * GIB);
    Ok(SetupPlan {
        completed,
        show: should_show(completed, installed.len()),
        installed,
        cpu_name: hw.cpu_name.clone(),
        logical_cores: hw.logical_cores,
        total_ram_bytes: hw.total_ram_bytes,
        gpus,
        gpu_note: hw.gpu_note.clone(),
        available,
        available_error,
        driver,
        starter_gpu: gpu_vram.map(|v| pick_starter(Some(v), hw.total_ram_bytes)),
        starter_cpu: pick_starter(None, hw.total_ram_bytes),
    })
}

/// Cheap check for the shell on launch (no network): open setup by itself?
#[tauri::command]
pub async fn setup_should_show(state: State<'_, SharedState>) -> Cmd<bool> {
    // The e2e suites launch isolated profiles with LOCALAISTUDIO_SKIP_SETUP=1
    // so setup never opens (or starts a runtime download) under them.
    if std::env::var_os("LOCALAISTUDIO_SKIP_SETUP").is_some_and(|v| v == "1") {
        return Ok(false);
    }
    let completed = state.db.get_setting::<serde_json::Value>(SETUP_KEY).await?.is_some();
    Ok(should_show(completed, state.runtimes.list_installed().len()))
}

/// Mark setup done (Finish) or skipped. Either way it won't open by itself again.
#[tauri::command]
pub async fn finish_setup(state: State<'_, SharedState>, skipped: bool) -> Cmd<()> {
    state.db.set_setting(SETUP_KEY, &serde_json::json!({ "skipped": skipped })).await
}

/// Open NVIDIA's driver download page in the default browser. The URL is fixed
/// here so the renderer can't use this to open arbitrary links.
#[tauri::command]
pub fn open_driver_download(app: AppHandle) -> Cmd<()> {
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_url(NVIDIA_DRIVER_URL, None::<&str>)
        .map_err(|e| AppError::Other(format!("couldn't open the browser: {e}")))
}

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn starter_follows_vram_and_ram() {
        let p = pick_starter(Some(8 * GIB), 32 * GIB);
        assert_eq!(p.file, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        assert_eq!(pick_starter(Some(16 * GIB), 32 * GIB).file, "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(pick_starter(Some(4 * GIB), 16 * GIB).file, "Qwen3-1.7B-Q4_K_M.gguf");
        assert_eq!(pick_starter(None, 16 * GIB).file, "Qwen3-1.7B-Q4_K_M.gguf");
        assert_eq!(pick_starter(None, 4 * GIB).file, "Qwen3-0.6B-Q4_K_M.gguf");
        // Files must be inside their repo's naming (guards copy-paste slips).
        for (repo, file, _, size) in STARTERS {
            let stem = file.split("-Q4_K_M").next().unwrap();
            assert!(repo.contains(stem), "{repo} / {file}");
            assert!(*size > 100_000_000);
        }
    }

    #[test]
    fn driver_check_messages() {
        let ok = check_driver("RTX 5060", Some("621.01"), (13, 5)).unwrap();
        assert_eq!(ok.status, "ok");
        let mid = check_driver("RTX 5060", Some("591.44"), (13, 5)).unwrap();
        assert_eq!(mid.status, "old_but_works");
        let old = check_driver("RTX 3060", Some("566.36"), (13, 5)).unwrap();
        assert_eq!(old.status, "too_old");
        assert!(old.message.contains("580.00"), "{}", old.message);
        assert!(old.message.contains("Vulkan or CPU"), "{}", old.message);
        assert_eq!(old.download_url, NVIDIA_DRIVER_URL);
        assert_eq!(check_driver("x", None, (13, 5)).unwrap().status, "unknown");
        assert!(check_driver("x", Some("500.00"), (11, 8)).is_none());
    }

    #[test]
    fn shows_only_for_a_fresh_profile() {
        assert!(should_show(false, 0));
        assert!(!should_show(true, 0), "skipped or finished: never by itself again");
        assert!(!should_show(false, 2), "existing profile with runtimes");
    }

    #[test]
    fn best_cuda_skips_builds_without_kernels() {
        let rt = |v: &str| AvailableRuntime {
            id: format!("b1-{v}"),
            tag: "b1".into(),
            variant: v.into(),
            published_at: String::new(),
            assets: vec![],
            total_size: 0,
            recommended: false,
            note: String::new(),
        };
        let a = vec![rt("cpu"), rt("cuda-12.4"), rt("cuda-13.5"), rt("vulkan")];
        assert_eq!(best_cuda(&a, Some((12, 0))), Some((13, 5)));
        assert_eq!(best_cuda(&a[..2], Some((12, 0))), None);
        assert_eq!(best_cuda(&a[..2], Some((8, 6))), Some((12, 4)));
    }
}

#[tauri::command]
pub async fn list_runtimes(state: State<'_, SharedState>) -> Cmd<Vec<InstalledRuntime>> {
    Ok(state.runtimes.list_installed())
}

#[tauri::command]
pub async fn fetch_available_runtimes(state: State<'_, SharedState>) -> Cmd<Vec<AvailableRuntime>> {
    let hint = gpu_hint(state.inner());
    state.runtimes.fetch_available(&hint).await
}

#[tauri::command]
pub async fn install_runtime(
    app: AppHandle,
    state: State<'_, SharedState>,
    runtime: AvailableRuntime,
) -> Cmd<InstalledRuntime> {
    let token = CancellationToken::new();
    {
        let mut m = state.installs.lock().unwrap();
        if m.contains_key(&runtime.id) {
            return Err(AppError::Conflict(format!("{} is already being installed", runtime.id)));
        }
        m.insert(runtime.id.clone(), token.clone());
    }
    let emit = move |p: inference::InstallProgress| {
        let _ = app.emit("runtime://install-progress", &p);
    };
    let r = state.runtimes.install(&runtime, &token, &emit).await;
    state.installs.lock().unwrap().remove(&runtime.id);
    let installed = r?;
    // First runtime installed becomes the default.
    if state.db.get_setting::<String>("runtime.default").await?.is_none() {
        state.db.set_setting("runtime.default", &installed.id).await?;
    }
    Ok(installed)
}

#[tauri::command]
pub fn cancel_runtime_install(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    if let Some(t) = state.installs.lock().unwrap().get(&id) {
        t.cancel();
    }
    Ok(())
}

#[tauri::command]
pub async fn register_custom_runtime(state: State<'_, SharedState>, path: String) -> Cmd<InstalledRuntime> {
    state.runtimes.register_custom(&PathBuf::from(path), None).await
}

#[tauri::command]
pub async fn remove_runtime(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    if state.engine.status().runtime_id.as_deref() == Some(id.as_str()) && state.engine.backend_pid().is_some() {
        return Err(AppError::Conflict("unload the model before removing the runtime it is using".into()));
    }
    state.runtimes.remove(&id)?;
    if state.db.get_setting::<String>("runtime.default").await?.as_deref() == Some(id.as_str()) {
        let next = state.runtimes.list_installed().into_iter().next().map(|r| r.id);
        state.db.set_setting("runtime.default", &next).await?;
    }
    Ok(())
}

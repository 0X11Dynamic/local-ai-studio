use std::path::{Path, PathBuf};

use app_core::AppError;
use inference::{GpuLayers, LoadConfig};
use model_registry::estimate::{self, EstimateInput, MemoryEstimate, ShareSplit};
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

/// Delete the model's files from disk (moved to nothing — this is permanent),
/// then remove it from the library. The caller must echo the display name as
/// confirmation, so a stray click or a scripted call with only an id can't do it.
#[tauri::command]
pub async fn delete_model_files(state: State<'_, SharedState>, id: String, confirm_name: String) -> Cmd<Vec<String>> {
    ensure_not_active(state.inner(), &id)?;
    let m = state.db.get_model(&id).await?;
    if confirm_name != m.display_name {
        return Err(AppError::InvalidInput("confirmation name does not match".into()));
    }
    let mut files: Vec<String> = serde_json::from_str(&m.shard_paths).unwrap_or_default();
    if files.is_empty() {
        files.push(m.path.clone());
    }
    let mut deleted = Vec::new();
    for f in &files {
        let p = Path::new(f);
        if p.extension().is_none_or(|e| !e.eq_ignore_ascii_case("gguf")) {
            return Err(AppError::InvalidInput(format!("refusing to delete non-GGUF file {f}")));
        }
        if p.exists() {
            std::fs::remove_file(p)?;
            deleted.push(f.clone());
        }
    }
    state.db.delete_model(&id).await?;
    Ok(deleted)
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
    let input = estimate_input(&config, &meta, total_layers, projector_bytes(&config, m.projector_path.as_deref()));
    let est = estimate::estimate(&meta, m.size as u64, &input);
    let free_vram = gpu.map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes));
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
    let has_moe_flag = default_runtime_has_n_cpu_moe(state.inner()).await;
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
        let mut c = LoadConfig::default();
        if let Some(tc) = meta.context_length {
            c.n_ctx = c.n_ctx.min(tc as u32);
        }
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

use std::path::PathBuf;
use std::time::{Duration, Instant};

use app_core::AppError;
use inference::{EngineStatus, LoadConfig, LoadRequest};
use tauri::State;

use super::Cmd;
use crate::state::SharedState;

/// Pick the runtime: explicit id, else the saved default, else the first installed.
async fn resolve_runtime(state: &SharedState, id: Option<String>) -> Cmd<inference::InstalledRuntime> {
    let installed = state.runtimes.list_installed();
    if installed.is_empty() {
        return Err(AppError::NotFound("no inference runtime installed yet; install one on the Runtimes page".into()));
    }
    let want = match id {
        Some(i) => Some(i),
        None => state.db.get_setting::<String>("runtime.default").await?,
    };
    if let Some(w) = want {
        return installed
            .into_iter()
            .find(|r| r.id == w)
            .ok_or_else(|| AppError::NotFound(format!("runtime {w} is not installed")));
    }
    Ok(installed.into_iter().next().unwrap())
}

#[tauri::command]
pub async fn load_model(
    state: State<'_, SharedState>,
    model_id: String,
    config: LoadConfig,
    runtime_id: Option<String>,
) -> Cmd<EngineStatus> {
    do_load(state.inner(), model_id, config, runtime_id).await
}

/// The one load path shared by the UI and the local API.
pub async fn do_load(st: &SharedState, model_id: String, config: LoadConfig, runtime_id: Option<String>) -> Cmd<EngineStatus> {
    let st = st.clone();
    // Validate before anything is recorded: a rejected config must never
    // become the model's "last used" settings.
    config.validate().map_err(AppError::InvalidInput)?;
    let m = st.db.get_model(&model_id).await?;
    if m.status != "ok" {
        return Err(AppError::InvalidInput(format!(
            "this model can't be loaded: {}",
            m.status_detail.unwrap_or_else(|| m.status.clone())
        )));
    }
    let runtime = resolve_runtime(&st, runtime_id).await?;
    // "Share with CPU": pick the GPU/CPU split from free VRAM right now. A
    // model that is already loaded would make VRAM look full, so unload it
    // first (a load replaces it anyway).
    let effective = if config.share_with_cpu && runtime.uses_gpu() {
        if let Some(pid) = st.engine.backend_pid() {
            do_unload(&st).await?;
            wait_for_vram_release(&st, pid).await;
        }
        let meta: model_registry::ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
        let projector = super::models::projector_bytes(&config, m.projector_path.as_deref());
        let (c, split) = super::models::resolve_share(
            &config,
            &meta,
            m.size as u64,
            projector,
            super::models::free_vram(&st),
            runtime.capabilities.has("--n-cpu-moe"),
        );
        if let Some(s) = &split {
            tracing::info!("Share with CPU: {}", s.summary);
        }
        c.validate().map_err(AppError::InvalidInput)?;
        c
    } else {
        config.clone()
    };
    // The model's "last used" keeps the user's choice (share_with_cpu), not the
    // resolved layer count, so the next load re-measures free VRAM.
    let cfg_json = serde_json::to_value(&config)?;
    let instance_id = st.db.insert_instance(&model_id, &runtime.id, &serde_json::to_value(&effective)?).await?;
    st.db.set_model_configs(&model_id, None, Some(&cfg_json)).await?;
    let req = LoadRequest {
        model_id,
        instance_id: instance_id.clone(),
        model_path: PathBuf::from(&m.path),
        projector_path: m.projector_path.map(PathBuf::from),
        runtime,
        config: effective,
        alias: Some(m.display_name.clone()),
    };
    match st.engine.load(req).await {
        Ok(s) => Ok(s),
        Err(e) => {
            let state = if matches!(e, AppError::Cancelled) { "unloaded" } else { "failed" };
            let _ = st.db.set_instance_state(&instance_id, state, Some(&e.to_string())).await;
            Err(e)
        }
    }
}

/// How long to wait after an unload for the driver to hand back the old
/// backend's VRAM, and how often to look.
const VRAM_RELEASE_TIMEOUT: Duration = Duration::from_secs(5);
const VRAM_POLL: Duration = Duration::from_millis(200);
/// Used VRAM counts as settled once it drops less than this between two polls.
const VRAM_SETTLE_TOLERANCE: u64 = 64 << 20;

/// After unloading backend `pid`, wait (up to [`VRAM_RELEASE_TIMEOUT`]) until
/// the process is gone and used VRAM stops dropping, so "Share with CPU"
/// measures free VRAM without the old model in it.
async fn wait_for_vram_release(st: &SharedState, pid: u32) {
    let deadline = Instant::now() + VRAM_RELEASE_TIMEOUT;
    let mut prev = None;
    loop {
        let snap = st.monitor.snapshot(Some(pid));
        let exited = snap.backend_memory_bytes.is_none();
        let used = snap.gpus.first().map(|g| g.vram_used_bytes);
        if exited && vram_settled(prev, used) {
            return;
        }
        if Instant::now() >= deadline {
            tracing::warn!(pid, exited, "VRAM still being released after unload; measuring free VRAM anyway");
            return;
        }
        prev = used;
        tokio::time::sleep(VRAM_POLL).await;
    }
}

/// True once used VRAM has stopped dropping: no GPU reading at all, or the
/// latest reading is within the tolerance of the one before. The first
/// reading never counts (there is nothing to compare it with).
fn vram_settled(prev: Option<u64>, now: Option<u64>) -> bool {
    match (prev, now) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(p), Some(n)) => n.saturating_add(VRAM_SETTLE_TOLERANCE) >= p,
    }
}

/// Reload the loaded model with its vision projector (for image attachments):
/// its last-used load settings with `load_projector` on, same runtime.
#[tauri::command]
pub async fn reload_with_projector(state: State<'_, SharedState>) -> Cmd<EngineStatus> {
    let st = state.inner();
    let status = st.engine.status();
    let model_id = status.model_id.clone().ok_or_else(|| AppError::InvalidInput("no model is loaded".into()))?;
    let m = st.db.get_model(&model_id).await?;
    if m.projector_path.is_none() {
        return Err(AppError::InvalidInput(format!("{} has no vision projector (mmproj) file next to it", m.display_name)));
    }
    let mut config: LoadConfig = m
        .last_config
        .as_deref()
        .and_then(|c| serde_json::from_str(c).ok())
        .or(status.config.clone())
        .unwrap_or_default();
    config.load_projector = true;
    do_load(st, model_id, config, status.runtime_id.clone()).await
}

#[tauri::command]
pub async fn unload_model(state: State<'_, SharedState>) -> Cmd<EngineStatus> {
    do_unload(state.inner()).await
}

pub async fn do_unload(st: &SharedState) -> Cmd<EngineStatus> {
    let st = st.clone();
    let inst = st.engine.status().instance_id;
    st.engine.unload().await;
    if let Some(i) = inst {
        st.db.set_instance_state(&i, "unloaded", None).await?;
    }
    Ok(st.engine.status())
}

#[tauri::command]
pub fn cancel_load(state: State<'_, SharedState>) -> Cmd<()> {
    state.engine.cancel_load();
    Ok(())
}

#[tauri::command]
pub fn engine_status(state: State<'_, SharedState>) -> EngineStatus {
    state.engine.status()
}

#[tauri::command]
pub fn engine_logs(state: State<'_, SharedState>, last: Option<usize>) -> Vec<String> {
    state.engine.logs(last.unwrap_or(500).min(4000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_settles_once_it_stops_dropping() {
        const MIB: u64 = 1 << 20;
        // No GPU metrics: nothing to wait for.
        assert!(vram_settled(None, None));
        // First reading: wait for a second one.
        assert!(!vram_settled(None, Some(9000 * MIB)));
        // Still dropping fast.
        assert!(!vram_settled(Some(9000 * MIB), Some(5000 * MIB)));
        // Flat, small jitter, or rising (another app): settled.
        assert!(vram_settled(Some(5000 * MIB), Some(5000 * MIB)));
        assert!(vram_settled(Some(5000 * MIB), Some(4990 * MIB)));
        assert!(vram_settled(Some(5000 * MIB), Some(5100 * MIB)));
    }
}

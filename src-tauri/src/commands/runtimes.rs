use std::path::PathBuf;

use app_core::AppError;
use inference::{AvailableRuntime, GpuHint, InstalledRuntime};
use tauri::{AppHandle, Emitter, State};
use tokio_util::sync::CancellationToken;

use super::Cmd;
use crate::state::SharedState;

pub fn gpu_hint(state: &SharedState) -> GpuHint {
    let hw = state.monitor.hardware_info();
    let parse = |s: &str| {
        let (a, b) = s.split_once('.')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    };
    let g = hw.gpus.first();
    GpuHint {
        nvidia: g.is_some_and(|g| g.vendor == "NVIDIA"),
        compute_capability: g.and_then(|g| g.compute_capability.as_deref()).and_then(parse),
        cuda_driver: hw.cuda_driver_version.as_deref().and_then(parse),
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

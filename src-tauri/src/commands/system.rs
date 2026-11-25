use app_core::AppError;
use system_monitor::{HardwareInfo, ResourceSnapshot};
use tauri::State;

use super::Cmd;
use crate::state::SharedState;

#[tauri::command]
pub async fn get_paths(state: State<'_, SharedState>) -> Cmd<app_core::AppPaths> {
    // The folders can change while the app runs (Settings -> Storage).
    let mut p = state.paths.clone();
    p.runtimes_dir = state.runtimes.dir();
    p.default_models_dir = state.models_dir().await;
    Ok(p)
}

/// The user guide, embedded at build time so it works offline after install.
#[tauri::command]
pub fn user_guide() -> &'static str {
    include_str!("../../../docs/USER_GUIDE.md")
}

#[tauri::command]
pub async fn get_hardware(state: State<'_, SharedState>) -> Cmd<HardwareInfo> {
    let st = state.inner().clone();
    tokio::task::spawn_blocking(move || st.monitor.hardware_info())
        .await
        .map_err(|e| AppError::Other(e.to_string()))
}

#[tauri::command]
pub async fn get_resources(state: State<'_, SharedState>) -> Cmd<ResourceSnapshot> {
    let st = state.inner().clone();
    let pid = st.engine.backend_pid();
    tokio::task::spawn_blocking(move || st.monitor.snapshot(pid))
        .await
        .map_err(|e| AppError::Other(e.to_string()))
}

/// Settings the UI may change. Anything else is rejected so the renderer can't
/// write arbitrary keys (later phases add security settings that must not be
/// writable from here).
const UI_SETTINGS: &[&str] = &[
    "ui.theme",
    "ui.sidebar_collapsed",
    "runtime.default",
    "library.models_dir",
    "load.defaults",
    "downloads.concurrency",
];

#[tauri::command]
pub async fn get_settings(state: State<'_, SharedState>) -> Cmd<serde_json::Map<String, serde_json::Value>> {
    state.db.all_settings().await
}

#[tauri::command]
pub async fn set_setting(state: State<'_, SharedState>, key: String, value: serde_json::Value) -> Cmd<()> {
    if !UI_SETTINGS.contains(&key.as_str()) {
        return Err(AppError::PermissionDenied(format!("setting '{key}' cannot be changed from the UI")));
    }
    if key == "downloads.concurrency" && !value.as_u64().is_some_and(|n| (1..=8).contains(&n)) {
        return Err(AppError::InvalidInput("concurrent downloads must be 1-8".into()));
    }
    if key == "library.models_dir" {
        // Kept for older callers; storage.json is the single source of truth.
        let p = value.as_str().ok_or_else(|| AppError::InvalidInput("models_dir must be a path".into()))?;
        let target = if p.trim().is_empty() {
            super::storage::default_models_dir(&state.paths).to_string_lossy().into_owned()
        } else {
            p.to_string()
        };
        super::storage::apply(&state, Some(target), None, false).await?;
        if p.trim().is_empty() {
            return state.db.set_setting(&key, &"").await;
        }
        return Ok(());
    }
    state.db.set_setting(&key, &value).await
}

use std::sync::Arc;

use app_core::AppError;
use local_api::{ApiConfig, ApiStatus};
use persistence::ApiLogRow;
use tauri::State;

use super::Cmd;
use crate::services::api_ops::AppOps;
use crate::state::{CRED_API_TOKEN, SharedState};

pub async fn saved_config(st: &SharedState) -> ApiConfig {
    st.db.get_setting::<ApiConfig>("api.config").await.ok().flatten().unwrap_or_default()
}

pub async fn start_with(st: &SharedState, cfg: ApiConfig) -> Cmd<ApiStatus> {
    let token = st.credentials.get(CRED_API_TOKEN)?;
    st.api
        .start(cfg, token, st.engine.clone(), st.db.clone(), Arc::new(AppOps(Arc::downgrade(st))))
        .await
}

#[tauri::command]
pub async fn api_status(state: State<'_, SharedState>) -> Cmd<ApiStatus> {
    let mut s = state.api.status();
    if !s.running {
        s.config = saved_config(state.inner()).await;
    }
    Ok(s)
}

/// Save the config and (re)start the server with it.
#[tauri::command]
pub async fn api_start(state: State<'_, SharedState>, config: ApiConfig) -> Cmd<ApiStatus> {
    config.validate().map_err(AppError::InvalidInput)?;
    state.db.set_setting("api.config", &config).await?;
    start_with(state.inner(), config).await
}

#[tauri::command]
pub async fn api_save_config(state: State<'_, SharedState>, config: ApiConfig) -> Cmd<()> {
    config.validate().map_err(AppError::InvalidInput)?;
    state.db.set_setting("api.config", &config).await
}

#[tauri::command]
pub async fn api_stop(state: State<'_, SharedState>) -> Cmd<ApiStatus> {
    state.api.stop().await;
    api_status(state).await
}

#[tauri::command]
pub async fn api_logs(state: State<'_, SharedState>, limit: Option<i64>) -> Cmd<Vec<ApiLogRow>> {
    state.db.recent_api_logs(limit.unwrap_or(200).clamp(1, 2000)).await
}

/// Create a new random API token (replacing any old one). Returned once so the
/// user can copy it; afterwards it lives only in Windows Credential Manager.
#[tauri::command]
pub async fn api_generate_token(state: State<'_, SharedState>) -> Cmd<String> {
    let mut bytes = [0u8; 32];
    getrandom(&mut bytes)?;
    let token = format!("las_{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
    state.credentials.set(CRED_API_TOKEN, &token)?;
    // A running server keeps its old token until restarted.
    if state.api.status().running {
        let cfg = saved_config(state.inner()).await;
        start_with(state.inner(), cfg).await?;
    }
    Ok(token)
}

#[tauri::command]
pub fn api_has_token(state: State<'_, SharedState>) -> Cmd<bool> {
    Ok(state.credentials.get(CRED_API_TOKEN)?.is_some())
}

/// Explicit "reveal" action for copying the token into a client.
#[tauri::command]
pub fn api_reveal_token(state: State<'_, SharedState>) -> Cmd<Option<String>> {
    state.credentials.get(CRED_API_TOKEN)
}

fn getrandom(buf: &mut [u8]) -> Cmd<()> {
    // Windows CSPRNG (BCryptGenRandom via the system-preferred RNG).
    use windows::Win32::Security::Cryptography::{BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom};
    let r = unsafe { BCryptGenRandom(None, buf, BCRYPT_USE_SYSTEM_PREFERRED_RNG) };
    if r.is_ok() { Ok(()) } else { Err(AppError::Other(format!("random generator failed: {r:?}"))) }
}

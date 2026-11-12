//! Execution Environments commands: This PC folders, SSH (with host-key
//! trust), Docker, WSL sandboxes, health checks, built-in tool permissions,
//! and running a built-in tool by hand.

use std::time::Duration;

use app_core::AppError;
use execution::Health;
use execution_docker::DockerConfig;
use tauri::State;

use super::Cmd;
use crate::services::exec::{self as svc, EnvView, SshProfile, SshSecrets};
use crate::state::SharedState;

#[tauri::command]
pub async fn list_environments(state: State<'_, SharedState>) -> Cmd<Vec<EnvView>> {
    svc::list(state.inner()).await
}

#[tauri::command]
pub async fn set_local_roots(state: State<'_, SharedState>, roots: Vec<String>) -> Cmd<Vec<String>> {
    svc::set_local_roots(state.inner(), roots).await
}

#[tauri::command]
pub async fn environment_health(state: State<'_, SharedState>, id: String) -> Cmd<Health> {
    svc::health(state.inner(), &id).await
}

#[tauri::command]
pub async fn add_ssh_environment(state: State<'_, SharedState>, name: String, profile: SshProfile, secrets: SshSecrets) -> Cmd<String> {
    svc::add_ssh(state.inner(), &name, &profile, &secrets).await
}

/// `secrets: None` keeps the stored password/passphrase.
#[tauri::command]
pub async fn update_ssh_environment(state: State<'_, SharedState>, id: String, name: String, profile: SshProfile, secrets: Option<SshSecrets>) -> Cmd<()> {
    svc::update_ssh(state.inner(), &id, &name, &profile, secrets.as_ref()).await
}

#[derive(serde::Serialize)]
pub struct HostKeyView {
    pub fingerprint: String,
    pub algorithm: String,
    /// Already in the app's known_hosts with this key.
    pub trusted: bool,
}

/// Fetch a server's host key so the user can compare its fingerprint.
#[tauri::command]
pub async fn ssh_host_key(state: State<'_, SharedState>, host: String, port: u16) -> Cmd<HostKeyView> {
    let (fingerprint, algorithm, _) = execution_ssh::fetch_host_key(host.trim(), port, Duration::from_secs(15)).await?;
    let kh = svc::known_hosts(state.inner());
    // trust_host is a no-op success when the same key is already recorded;
    // check without writing by reading the file.
    let trusted = std::fs::read_to_string(&kh).ok().is_some_and(|t| {
        let tag = if port == 22 { host.trim().to_string() } else { format!("[{}]:{port}", host.trim()) };
        t.lines().any(|l| l.split_whitespace().next() == Some(tag.as_str()))
    });
    Ok(HostKeyView { fingerprint, algorithm, trusted })
}

/// Record a host key the user confirmed (it is fetched again and must match).
#[tauri::command]
pub async fn ssh_trust_host(state: State<'_, SharedState>, host: String, port: u16, fingerprint: String) -> Cmd<()> {
    execution_ssh::trust_host(host.trim(), port, &svc::known_hosts(state.inner()), &fingerprint).await
}

#[tauri::command]
pub async fn add_docker_environment(state: State<'_, SharedState>, name: String, config: DockerConfig) -> Cmd<String> {
    svc::add_docker(state.inner(), &name, config).await
}

#[tauri::command]
pub async fn update_docker_environment(state: State<'_, SharedState>, id: String, name: String, config: DockerConfig) -> Cmd<()> {
    svc::update_docker(state.inner(), &id, &name, config).await
}

#[tauri::command]
pub async fn docker_status(host: String) -> Cmd<String> {
    let d = execution_docker::connect(&host)?;
    let v = d.version().await.map_err(|e| AppError::Runtime(format!("cannot reach Docker at {host} ({e}). Is Docker Desktop running?")))?;
    Ok(format!("Docker {} (API {})", v.version.unwrap_or_default(), v.api_version.unwrap_or_default()))
}

#[tauri::command]
pub async fn wsl_status() -> Cmd<String> {
    execution::sandbox::wsl_status().await
}

#[tauri::command]
pub async fn create_sandbox(state: State<'_, SharedState>, name: String, network: bool) -> Cmd<String> {
    svc::create_sandbox(state.inner(), &name, network).await
}

#[tauri::command]
pub async fn delete_environment(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    svc::delete(state.inner(), &id).await
}

#[tauri::command]
pub async fn set_builtin_tool_pref(state: State<'_, SharedState>, env_id: String, tool: String, enabled: bool, policy: String) -> Cmd<()> {
    if !svc::BUILTINS.iter().any(|b| b.name == tool) {
        return Err(AppError::NotFound(format!("no built-in tool {tool}")));
    }
    if env_id != svc::LOCAL_ID {
        state.db.get_exec_profile(&env_id).await?;
    }
    state.db.set_builtin_pref(&env_id, &tool, enabled, &policy).await
}

/// Run a built-in tool by hand through the normal dispatcher (schema check,
/// permission, approval, audit). Used by "Try it" and the e2e suite.
#[tauri::command]
pub async fn run_builtin_tool(
    app: tauri::AppHandle,
    state: State<'_, SharedState>,
    env_id: String,
    tool: String,
    arguments: serde_json::Value,
    conversation_id: Option<String>,
) -> Cmd<agent_runtime::DispatchOutcome> {
    let st = state.inner().clone();
    let (tools, _) = svc::exposed(&st, &env_id).await?;
    let t = tools.iter().find(|t| t.tool_name == tool).ok_or_else(|| AppError::NotFound(format!("{tool} is turned off or set to deny for this environment")))?;
    let call = inference::ToolCall { id: format!("manual-{}", app_core::new_id()), name: t.model_name.clone(), arguments: arguments.to_string() };
    let host = crate::services::mcp::AppToolHost::chat(st.clone(), app, conversation_id.clone());
    Ok(agent_runtime::dispatch(&host, &tools, &call, conversation_id.as_deref(), &tokio_util::sync::CancellationToken::new()).await)
}

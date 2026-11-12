//! MCP Connections and tool permission commands.

use std::path::PathBuf;

use app_core::AppError;
use mcp_client::{Secrets, ServerConfig, ServerStatus};
use persistence::{GrantRow, ToolExecRow};
use security::policy::{ApprovalAnswer, MCP_DEFAULT_POLICY};
use serde::{Deserialize, Serialize};
use tauri::State;

use super::Cmd;
use crate::services::mcp as svc;
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct ToolView {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub input_schema: serde_json::Value,
    /// Server-provided hints; shown as unverified, never used for policy.
    pub annotations: serde_json::Value,
    pub enabled: bool,
    pub policy: String,
}

#[derive(Debug, Serialize)]
pub struct ServerView {
    pub id: String,
    pub name: String,
    pub transport: String,
    pub enabled: bool,
    pub config: ServerConfig,
    /// Which secret fields have a stored value (values are never returned).
    pub secret_env_set: Vec<String>,
    pub secret_headers_set: Vec<String>,
    pub bearer_set: bool,
    pub status: ServerStatus,
    pub tools: Vec<ToolView>,
}

fn validate_name(name: &str) -> Cmd<String> {
    let n = name.trim();
    if n.is_empty() || n.len() > 64 || !n.chars().all(|c| c.is_ascii_alphanumeric() || "-_. ".contains(c)) {
        return Err(AppError::InvalidInput("server name: 1-64 letters, digits, spaces, '-', '_' or '.'".into()));
    }
    Ok(n.to_string())
}

async fn view(st: &SharedState, id: &str) -> Cmd<ServerView> {
    let row = st.db.get_mcp_server(id).await?;
    let config = svc::config_of(&row);
    let secrets = svc::secrets_of(st, &row);
    let status = st.mcp.status(id);
    let prefs = st.db.tool_prefs(id).await?;
    let tools = status
        .tools
        .iter()
        .map(|t| {
            let p = prefs.iter().find(|p| p.tool_name == t.name);
            ToolView {
                name: t.name.clone(),
                title: t.title.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
                annotations: t.annotations.clone(),
                enabled: p.map(|p| p.enabled != 0).unwrap_or(true),
                policy: p.map(|p| p.policy.clone()).unwrap_or_else(|| MCP_DEFAULT_POLICY.as_str().into()),
            }
        })
        .collect();
    Ok(ServerView {
        id: row.id,
        name: row.name,
        transport: row.transport,
        enabled: row.enabled != 0,
        secret_env_set: secrets.env.keys().cloned().collect(),
        secret_headers_set: secrets.headers.keys().cloned().collect(),
        bearer_set: secrets.bearer.as_deref().is_some_and(|b| !b.is_empty()),
        config,
        status,
        tools,
    })
}

#[tauri::command]
pub async fn list_mcp_servers(state: State<'_, SharedState>) -> Cmd<Vec<ServerView>> {
    let st = state.inner().clone();
    let mut out = vec![];
    for r in st.db.list_mcp_servers().await? {
        out.push(view(&st, &r.id).await?);
    }
    Ok(out)
}

/// New secret values. `None` keeps what is stored; `Some("")` clears it.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct SecretsPatch {
    pub env: std::collections::BTreeMap<String, Option<String>>,
    pub headers: std::collections::BTreeMap<String, Option<String>>,
    pub bearer: Option<String>,
}

fn apply_patch(mut cur: Secrets, cfg: &mut ServerConfig, patch: SecretsPatch) -> Secrets {
    for (k, v) in patch.env {
        match v {
            Some(val) if !val.is_empty() => {
                cur.env.insert(k.clone(), val);
            }
            Some(_) => {
                cur.env.remove(&k);
            }
            None => {}
        }
    }
    for (k, v) in patch.headers {
        match v {
            Some(val) if !val.is_empty() => {
                cur.headers.insert(k.clone(), val);
            }
            Some(_) => {
                cur.headers.remove(&k);
            }
            None => {}
        }
    }
    if let Some(b) = patch.bearer {
        cur.bearer = (!b.is_empty()).then_some(b);
    }
    // Keep only secrets the config still declares.
    cur.env.retain(|k, _| cfg.secret_env.contains(k));
    cur.headers.retain(|k, _| cfg.secret_headers.contains(k));
    cfg.bearer = cur.bearer.is_some();
    cur
}

#[tauri::command]
pub async fn add_mcp_server(state: State<'_, SharedState>, name: String, mut config: ServerConfig, secrets: Option<SecretsPatch>, enabled: Option<bool>) -> Cmd<ServerView> {
    let st = state.inner().clone();
    let name = validate_name(&name)?;
    config.validate().map_err(AppError::InvalidInput)?;
    let id = st.db.insert_mcp_server(&name, &config.transport, &serde_json::to_value(&config)?, None, enabled.unwrap_or(true)).await?;
    let sec = apply_patch(Secrets::default(), &mut config, secrets.unwrap_or_default());
    let cref = svc::store_secrets(&st, &id, &sec)?;
    st.db.update_mcp_server(&id, &name, &config.transport, &serde_json::to_value(&config)?, cref.as_deref(), enabled.unwrap_or(true)).await?;
    view(&st, &id).await
}

#[tauri::command]
pub async fn update_mcp_server(state: State<'_, SharedState>, id: String, name: String, mut config: ServerConfig, secrets: Option<SecretsPatch>, enabled: bool) -> Cmd<ServerView> {
    let st = state.inner().clone();
    let name = validate_name(&name)?;
    config.validate().map_err(AppError::InvalidInput)?;
    let row = st.db.get_mcp_server(&id).await?;
    let sec = apply_patch(svc::secrets_of(&st, &row), &mut config, secrets.unwrap_or_default());
    let cref = svc::store_secrets(&st, &id, &sec)?;
    st.db.update_mcp_server(&id, &name, &config.transport, &serde_json::to_value(&config)?, cref.as_deref(), enabled).await?;
    if !enabled {
        st.mcp.disconnect(&id).await;
    }
    view(&st, &id).await
}

#[tauri::command]
pub async fn delete_mcp_server(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    let st = state.inner().clone();
    st.mcp.disconnect(&id).await;
    st.mcp.forget(&id);
    st.credentials.delete(&svc::cred_ref(&id))?;
    st.db.delete_mcp_server(&id).await
}

#[tauri::command]
pub async fn connect_mcp_server(state: State<'_, SharedState>, id: String) -> Cmd<ServerView> {
    let st = state.inner().clone();
    let row = st.db.get_mcp_server(&id).await?;
    if row.enabled == 0 {
        return Err(AppError::InvalidInput("this server is turned off; turn it on first".into()));
    }
    // A failed connect is reported in the returned status, not as an error.
    let _ = svc::connect(&st, &id).await;
    view(&st, &id).await
}

#[tauri::command]
pub async fn disconnect_mcp_server(state: State<'_, SharedState>, id: String) -> Cmd<ServerView> {
    let st = state.inner().clone();
    st.mcp.disconnect(&id).await;
    view(&st, &id).await
}

#[tauri::command]
pub fn mcp_server_logs(state: State<'_, SharedState>, id: String) -> Vec<String> {
    state.mcp.logs(&id)
}

#[tauri::command]
pub async fn set_mcp_tool_pref(state: State<'_, SharedState>, server_id: String, tool: String, enabled: bool, policy: String) -> Cmd<()> {
    state.db.get_mcp_server(&server_id).await?;
    state.db.set_tool_pref(&server_id, &tool, enabled, &policy).await
}

#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub added: Vec<String>,
    pub skipped: Vec<String>,
    pub secrets_stored: usize,
}

async fn import_doc(st: &SharedState, json: &str, connect: bool) -> Cmd<ImportReport> {
    let servers = mcp_client::parse_mcp_servers(json).map_err(AppError::InvalidInput)?;
    let existing: Vec<String> = st.db.list_mcp_servers().await?.into_iter().map(|r| r.name).collect();
    let mut report = ImportReport { added: vec![], skipped: vec![], secrets_stored: 0 };
    for s in servers {
        let Ok(name) = validate_name(&s.name) else {
            report.skipped.push(format!("{} (name not allowed)", s.name));
            continue;
        };
        if existing.iter().any(|e| e.eq_ignore_ascii_case(&name)) {
            report.skipped.push(format!("{name} (already exists)"));
            continue;
        }
        let mut cfg = s.config.clone();
        // Imported servers start turned off unless asked: nothing runs until the user connects it.
        let id = st.db.insert_mcp_server(&name, &cfg.transport, &serde_json::to_value(&cfg)?, None, connect && !s.disabled).await?;
        let sec = s.secrets.clone();
        if !sec.is_empty() {
            report.secrets_stored += sec.env.len() + sec.headers.len() + usize::from(sec.bearer.is_some());
        }
        cfg.bearer = sec.bearer.is_some() || cfg.bearer;
        let cref = svc::store_secrets(st, &id, &sec)?;
        st.db.update_mcp_server(&id, &name, &cfg.transport, &serde_json::to_value(&cfg)?, cref.as_deref(), connect && !s.disabled).await?;
        report.added.push(name);
    }
    Ok(report)
}

/// Import an `mcpServers` JSON document (pasted text). Imported servers are
/// off until the user turns them on and connects.
#[tauri::command]
pub async fn import_mcp_servers(state: State<'_, SharedState>, json: String) -> Cmd<ImportReport> {
    import_doc(state.inner(), &json, false).await
}

/// One-click import from another app's config file: "lmstudio" or "claude".
#[tauri::command]
pub async fn import_mcp_from(state: State<'_, SharedState>, source: String) -> Cmd<ImportReport> {
    let path: PathBuf = match source.as_str() {
        "lmstudio" => dirs_home()?.join(".lmstudio").join("mcp.json"),
        "claude" => PathBuf::from(std::env::var("APPDATA").map_err(|_| AppError::NotFound("APPDATA not set".into()))?).join("Claude").join("claude_desktop_config.json"),
        "file" => return Err(AppError::InvalidInput("use import_mcp_file for files".into())),
        _ => return Err(AppError::InvalidInput("source must be lmstudio or claude".into())),
    };
    let text = tokio::fs::read_to_string(&path).await.map_err(|_| AppError::NotFound(format!("{} was not found", path.display())))?;
    import_doc(state.inner(), &text, false).await
}

#[tauri::command]
pub async fn import_mcp_file(state: State<'_, SharedState>, path: String) -> Cmd<ImportReport> {
    let meta = tokio::fs::metadata(&path).await?;
    if meta.len() > 2_000_000 {
        return Err(AppError::InvalidInput("file is too large for an MCP config".into()));
    }
    let text = tokio::fs::read_to_string(&path).await?;
    import_doc(state.inner(), &text, false).await
}

fn dirs_home() -> Cmd<PathBuf> {
    std::env::var("USERPROFILE").map(PathBuf::from).map_err(|_| AppError::NotFound("home folder not found".into()))
}

/// Export all servers as `mcpServers` JSON. Secret values are placeholders.
#[tauri::command]
pub async fn export_mcp_servers(state: State<'_, SharedState>) -> Cmd<String> {
    let rows = state.db.list_mcp_servers().await?;
    let list: Vec<(String, ServerConfig, bool)> = rows.iter().map(|r| (r.name.clone(), svc::config_of(r), r.enabled != 0)).collect();
    Ok(serde_json::to_string_pretty(&mcp_client::export_mcp_servers(&list))?)
}

// ---------- permissions and audit ----------

#[derive(Debug, Serialize)]
pub struct GrantView {
    #[serde(flatten)]
    pub grant: GrantRow,
    pub server_name: Option<String>,
    pub tool_name: Option<String>,
    pub conversation_title: Option<String>,
    /// Title of the agent task for "allow for this task" grants.
    pub task_title: Option<String>,
}

#[tauri::command]
pub async fn list_tool_grants(state: State<'_, SharedState>) -> Cmd<Vec<GrantView>> {
    let st = state.inner().clone();
    let servers = st.db.list_mcp_servers().await?;
    let mut out = vec![];
    for g in st.db.list_grants().await? {
        let (server_name, tool) = if let Some((env, tool)) = g.subject.strip_prefix(agent_runtime::BUILTIN_PREFIX).and_then(|s| s.split_once('/')) {
            // Built-in tools: name the execution environment.
            let label = crate::services::exec::meta(&st, env).await.map(|m| m.label).unwrap_or_else(|_| "removed environment".into());
            (Some(label), Some(tool.to_string()))
        } else {
            let (sid, tool) = g.subject.strip_prefix("mcp:").and_then(|s| s.split_once('/')).map(|(a, b)| (a.to_string(), b.to_string())).unzip();
            (sid.as_ref().and_then(|s| servers.iter().find(|r| &r.id == s).map(|r| r.name.clone())), tool)
        };
        let conversation_title = match &g.conversation_id {
            Some(c) => st.db.get_conversation(c).await.ok().map(|c| c.title),
            None => None,
        };
        let task_title = match &g.run_id {
            Some(r) => st.db.get_run(r).await.ok().map(|r| r.title),
            None => None,
        };
        out.push(GrantView { grant: g, server_name, tool_name: tool, conversation_title, task_title });
    }
    Ok(out)
}

#[tauri::command]
pub async fn revoke_tool_grant(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    state.db.delete_grant(&id).await
}

#[tauri::command]
pub async fn list_tool_activity(state: State<'_, SharedState>, conversation_id: Option<String>, limit: Option<i64>) -> Cmd<Vec<ToolExecRow>> {
    state.db.list_tool_execs(conversation_id.as_deref(), limit.unwrap_or(200).clamp(1, 2000)).await
}

#[tauri::command]
pub fn pending_tool_approvals(state: State<'_, SharedState>) -> Vec<agent_runtime::ApprovalRequest> {
    state.approvals.lock().unwrap().values().map(|(r, _)| r.clone()).collect()
}

#[tauri::command]
pub fn answer_tool_approval(state: State<'_, SharedState>, id: String, answer: ApprovalAnswer) -> Cmd<()> {
    if answer == ApprovalAnswer::TimedOut {
        return Err(AppError::InvalidInput("not a valid answer".into()));
    }
    svc::answer(state.inner(), &id, answer)
}

/// The tools a chat with these settings would be offered, plus the generated
/// prompt section (for the chat settings panel and the instruction preview).
#[tauri::command]
pub async fn preview_chat_tools(
    state: State<'_, SharedState>,
    tool_servers: Vec<String>,
    builtin_tools: Option<bool>,
    environment_id: Option<String>,
) -> Cmd<serde_json::Value> {
    let settings = crate::services::chat::ChatSettings { agent_mode: true, tool_servers, builtin_tools: builtin_tools.unwrap_or(true), environment_id, ..Default::default() };
    let (tools, section) = crate::services::chat::agent_tools(state.inner(), &settings).await?;
    Ok(serde_json::json!({ "tools": tools, "section": section }))
}

/// Run a tool by hand (MCP page "Try it", and the e2e suite). Goes through the
/// same dispatcher as a model's call: schema check, policy, approval, audit.
#[tauri::command]
pub async fn run_tool_manually(
    app: tauri::AppHandle,
    state: State<'_, SharedState>,
    server_id: String,
    tool: String,
    arguments: serde_json::Value,
    conversation_id: Option<String>,
) -> Cmd<agent_runtime::DispatchOutcome> {
    let st = state.inner().clone();
    let tools = svc::exposed_tools(&st, std::slice::from_ref(&server_id)).await?;
    let t = tools
        .iter()
        .find(|t| t.tool_name == tool)
        .ok_or_else(|| AppError::NotFound(format!("{tool} is not available (server not connected, tool turned off, or policy deny)")))?;
    let call = inference::ToolCall { id: format!("manual-{}", app_core::new_id()), name: t.model_name.clone(), arguments: arguments.to_string() };
    let host = svc::AppToolHost::chat(st.clone(), app, conversation_id.clone());
    Ok(agent_runtime::dispatch(&host, &tools, &call, conversation_id.as_deref(), &tokio_util::sync::CancellationToken::new()).await)
}

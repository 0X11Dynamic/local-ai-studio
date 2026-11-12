//! MCP servers inside the app: config + vault secrets, connecting, the tools a
//! chat may use, and the `ToolHost` that gives the dispatcher approvals,
//! grants, audit rows and real tool calls.

use std::collections::HashSet;
use std::time::Duration;

use agent_runtime::{ApprovalRequest, CallResult, ExposedTool, ToolEvent, ToolHost};
use app_core::{AppError, AppResult};
use async_trait::async_trait;
use inference::ToolCall;
use mcp_client::{Secrets, ServerConfig, ServerStatus};
use persistence::{McpServerRow, NewToolExec};
use security::policy::{ApprovalAnswer, Grant, GrantScope, MCP_DEFAULT_POLICY, Policy};
use tauri::{AppHandle, Emitter};
use tokio_util::sync::CancellationToken;

use crate::state::SharedState;

pub const APPROVAL_TIMEOUT: Duration = Duration::from_secs(10 * 60);

pub fn cred_ref(server_id: &str) -> String {
    format!("mcp/{server_id}")
}

pub fn config_of(row: &McpServerRow) -> ServerConfig {
    serde_json::from_str(&row.nonsecret_config).unwrap_or_default()
}

pub fn secrets_of(st: &SharedState, row: &McpServerRow) -> Secrets {
    row.credential_ref
        .as_deref()
        .and_then(|r| st.credentials.get(r).ok().flatten())
        .and_then(|j| serde_json::from_str(&j).ok())
        .unwrap_or_default()
}

/// Store (or clear) a server's secrets in Credential Manager. Returns the ref.
pub fn store_secrets(st: &SharedState, server_id: &str, secrets: &Secrets) -> AppResult<Option<String>> {
    let r = cred_ref(server_id);
    if secrets.is_empty() {
        st.credentials.delete(&r)?;
        Ok(None)
    } else {
        st.credentials.set(&r, &serde_json::to_string(secrets)?)?;
        Ok(Some(r))
    }
}

pub async fn connect(st: &SharedState, id: &str) -> AppResult<ServerStatus> {
    // One connect per server at a time (a manual Connect and a background
    // retry must not start two processes for the same server).
    let lock = connect_lock(id);
    let _guard = lock.lock().await;
    let row = st.db.get_mcp_server(id).await?;
    let cfg = config_of(&row);
    let secrets = secrets_of(st, &row);
    st.mcp.connect(id, &cfg, &secrets).await
}

// ---------- auto-connect (app start, imports, turning a server on) ----------

/// Attempts per server before giving up (the first try plus retries).
pub const AUTO_ATTEMPTS: u32 = 4;
/// Wait before retry n (seconds): 5, 15, 45.
const AUTO_BACKOFF_SECS: [u64; 3] = [5, 15, 45];

/// Background connect progress for one server (shown on the MCP page).
#[derive(Debug, Clone, serde::Serialize)]
pub struct AutoConnect {
    /// Attempt in progress or last made (1-based).
    pub attempt: u32,
    pub max_attempts: u32,
    /// Seconds until the next try (None while connecting or after giving up).
    pub retry_in_secs: Option<u64>,
    /// All attempts failed; connect by hand.
    pub gave_up: bool,
}

struct AutoEntry {
    generation: u64,
    attempt: u32,
    next_at: Option<std::time::Instant>,
    gave_up: bool,
}

static AUTO: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, AutoEntry>>> = std::sync::LazyLock::new(Default::default);
static AUTO_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
type ConnectLocks = std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>;
static CONNECT_LOCKS: std::sync::LazyLock<std::sync::Mutex<ConnectLocks>> = std::sync::LazyLock::new(Default::default);

fn connect_lock(id: &str) -> std::sync::Arc<tokio::sync::Mutex<()>> {
    CONNECT_LOCKS.lock().unwrap().entry(id.to_string()).or_default().clone()
}

/// Background connect state for the MCP page (None = no background connect).
pub fn auto_status(id: &str) -> Option<AutoConnect> {
    AUTO.lock().unwrap().get(id).map(|e| AutoConnect {
        attempt: e.attempt,
        max_attempts: AUTO_ATTEMPTS,
        retry_in_secs: e.next_at.map(|t| t.saturating_duration_since(std::time::Instant::now()).as_secs().max(1)),
        gave_up: e.gave_up,
    })
}

/// Stop a server's background retries (a manual connect, disconnect, edit
/// or removal takes over).
pub fn cancel_auto(id: &str) {
    AUTO.lock().unwrap().remove(id);
}

fn auto_current(id: &str, generation: u64) -> bool {
    AUTO.lock().unwrap().get(id).is_some_and(|e| e.generation == generation)
}

fn auto_update(id: &str, generation: u64, f: impl FnOnce(&mut AutoEntry)) {
    if let Some(e) = AUTO.lock().unwrap().get_mut(id).filter(|e| e.generation == generation) {
        f(e);
    }
}

/// Connect one server in the background, retrying with backoff. Stops when
/// it connects, the server is turned off or removed, or the user connects or
/// disconnects it by hand (see `cancel_auto`).
pub fn auto_connect(st: SharedState, id: String) {
    let generation = AUTO_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    AUTO.lock().unwrap().insert(id.clone(), AutoEntry { generation, attempt: 0, next_at: None, gave_up: false });
    tokio::spawn(async move {
        for attempt in 1..=AUTO_ATTEMPTS {
            if !auto_current(&id, generation) {
                return;
            }
            let on = st.db.get_mcp_server(&id).await.is_ok_and(|r| r.enabled != 0);
            if !on || st.mcp.status(&id).state == "connected" {
                if auto_current(&id, generation) {
                    cancel_auto(&id);
                }
                return;
            }
            auto_update(&id, generation, |e| {
                e.attempt = attempt;
                e.next_at = None;
            });
            let r = connect(&st, &id).await;
            if !auto_current(&id, generation) {
                return;
            }
            match r {
                Ok(_) => {
                    cancel_auto(&id);
                    return;
                }
                Err(e) if attempt == AUTO_ATTEMPTS => {
                    tracing::warn!("MCP auto-connect of {id} gave up after {attempt} attempts: {e}");
                    auto_update(&id, generation, |e| e.gave_up = true);
                    return;
                }
                Err(e) => {
                    let wait = Duration::from_secs(AUTO_BACKOFF_SECS[(attempt - 1) as usize]);
                    tracing::warn!("MCP auto-connect of {id} failed (attempt {attempt} of {AUTO_ATTEMPTS}), retrying in {}s: {e}", wait.as_secs());
                    auto_update(&id, generation, |e| e.next_at = Some(std::time::Instant::now() + wait));
                    tokio::time::sleep(wait).await;
                }
            }
        }
    });
}

/// Connect every turned-on server in the background (app start). Servers
/// connect in parallel, so a slow one doesn't hold up the others.
pub fn autoconnect(st: SharedState) {
    tokio::spawn(async move {
        let Ok(rows) = st.db.list_mcp_servers().await else { return };
        for r in rows.into_iter().filter(|r| r.enabled != 0) {
            auto_connect(st.clone(), r.id);
        }
    });
}

/// Tools a chat may be offered: connected, enabled servers (optionally only
/// `only`), tools that are enabled and not set to deny.
pub async fn exposed_tools(st: &SharedState, only: &[String]) -> AppResult<Vec<ExposedTool>> {
    let rows = st.db.list_mcp_servers().await?;
    let prefs = st.db.all_tool_prefs().await?;
    let mut taken = HashSet::new();
    let mut out = Vec::new();
    for row in rows.iter().filter(|r| r.enabled != 0 && (only.is_empty() || only.contains(&r.id))) {
        let status = st.mcp.status(&row.id);
        if status.state != "connected" {
            continue;
        }
        let cfg = config_of(row);
        for t in &status.tools {
            let pref = prefs.iter().find(|p| p.server_id == row.id && p.tool_name == t.name);
            let enabled = pref.map(|p| p.enabled != 0).unwrap_or(true);
            let policy = pref.and_then(|p| Policy::parse(&p.policy)).unwrap_or(MCP_DEFAULT_POLICY);
            if !enabled || policy == Policy::Deny {
                continue;
            }
            out.push(ExposedTool {
                tool_id: format!("mcp:{}/{}", row.id, t.name),
                model_name: agent_runtime::model_tool_name(&row.name, &t.name, &mut taken),
                server_id: row.id.clone(),
                server_name: row.name.clone(),
                tool_name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
                policy,
                timeout_secs: cfg.timeout_secs,
            });
        }
    }
    Ok(out)
}

/// The app side of tool dispatch for one conversation or agent task.
pub struct AppToolHost {
    pub st: SharedState,
    pub app: AppHandle,
    pub conversation_id: Option<String>,
    /// Set for agent tasks: the task, and the subagent (None = root agent).
    pub run_id: Option<String>,
    pub agent_id: Option<String>,
    /// Shown on approval prompts, e.g. "Subagent: count lines".
    pub agent_label: Option<String>,
}

impl AppToolHost {
    pub fn chat(st: SharedState, app: AppHandle, conversation_id: Option<String>) -> Self {
        Self { st, app, conversation_id, run_id: None, agent_id: None, agent_label: None }
    }
}

/// `server_id` prefix of the app's own agent tools (spawn_subagents,
/// subagent_status, wait_subagents, stop_subagent, ask_user, skill_propose_edit).
pub const AGENT_PREFIX: &str = "agent:";

#[async_trait]
impl ToolHost for AppToolHost {
    async fn grants(&self, tool_id: &str) -> AppResult<Vec<Grant>> {
        Ok(self
            .st
            .db
            .grants_for_tool(tool_id)
            .await?
            .into_iter()
            .filter_map(|g| {
                let scope = match g.scope.as_str() {
                    "session" => GrantScope::Session,
                    "persistent" => GrantScope::Persistent,
                    _ => return None,
                };
                // Task grants are scoped by run id (the scope dispatch passes for tasks).
                Some(Grant { id: g.id, scope, tool_id: g.subject, conversation_id: g.conversation_id.or(g.run_id) })
            })
            .collect())
    }

    async fn add_grant(&self, scope: GrantScope, tool_id: &str, conversation_id: Option<&str>) -> AppResult<()> {
        let s = match scope {
            GrantScope::Session => "session",
            GrantScope::Persistent => "persistent",
        };
        match (&self.run_id, scope) {
            (Some(run), GrantScope::Session) => {
                self.st.db.insert_run_grant(tool_id, run).await?;
            }
            _ => {
                self.st.db.insert_grant(s, tool_id, conversation_id).await?;
            }
        }
        Ok(())
    }

    async fn current_policy(&self, tool: &ExposedTool) -> AppResult<(bool, Policy, bool)> {
        if tool.server_id.starts_with(AGENT_PREFIX) {
            // Starting subagents and proposing skill edits do nothing risky by
            // themselves: subagents' own tools still ask, proposals need approval.
            return Ok((true, Policy::Allow, true));
        }
        if let Some(env_id) = tool.server_id.strip_prefix(agent_runtime::BUILTIN_PREFIX) {
            // Live preference; the environment must still exist.
            let exists = env_id == crate::services::exec::LOCAL_ID || crate::services::exec::is_adhoc(env_id) || self.st.db.get_exec_profile(env_id).await.is_ok();
            let (enabled, policy) = crate::services::exec::tool_pref(&self.st, env_id, &tool.tool_name).await?;
            return Ok((enabled, policy, exists));
        }
        let row = self.st.db.get_mcp_server(&tool.server_id).await;
        let Ok(row) = row else { return Ok((false, Policy::Deny, false)) };
        let pref = self.st.db.tool_prefs(&row.id).await?.into_iter().find(|p| p.tool_name == tool.tool_name);
        let enabled = row.enabled != 0 && pref.as_ref().map(|p| p.enabled != 0).unwrap_or(true);
        let policy = pref.and_then(|p| Policy::parse(&p.policy)).unwrap_or(MCP_DEFAULT_POLICY);
        let status = self.st.mcp.status(&row.id);
        let available = status.state == "connected" && status.tools.iter().any(|t| t.name == tool.tool_name);
        Ok((enabled, policy, available))
    }

    async fn request_approval(&self, mut req: ApprovalRequest, cancel: &CancellationToken) -> ApprovalAnswer {
        if self.run_id.is_some() {
            req.run_id = self.run_id.clone();
            // Tasks have no conversation; a chat's subagents keep theirs, so
            // the request shows in that chat.
            req.conversation_id = self.conversation_id.clone();
            req.agent = Some(self.agent_label.clone().unwrap_or_else(|| "Main agent".into()));
            req.agent_id = self.agent_id.clone();
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        let id = req.id.clone();
        self.st.approvals.lock().unwrap().insert(id.clone(), (req.clone(), tx));
        let _ = self.app.emit("tool://approval", &req);
        let answer = tokio::select! {
            r = tokio::time::timeout(APPROVAL_TIMEOUT, rx) => match r {
                Ok(Ok(a)) => a,
                Ok(Err(_)) => ApprovalAnswer::Deny,
                Err(_) => ApprovalAnswer::TimedOut,
            },
            _ = cancel.cancelled() => ApprovalAnswer::Deny,
        };
        self.st.approvals.lock().unwrap().remove(&id);
        let _ = self.app.emit("tool://approval-closed", &serde_json::json!({ "id": id, "answer": answer }));
        answer
    }

    async fn audit_start(&self, tool: &ExposedTool, call: &ToolCall, args_redacted: &str, conversation_id: Option<&str>) -> AppResult<String> {
        self.st
            .db
            .start_tool_exec(&NewToolExec {
                conversation_id: if self.run_id.is_some() { None } else { conversation_id },
                tool: &tool.tool_id,
                source: &if tool.server_id.starts_with(agent_runtime::BUILTIN_PREFIX) { "builtin".to_string() } else { format!("mcp:{}", tool.server_name) },
                args_redacted,
                tool_call_id: Some(&call.id),
                model_tool_name: Some(&tool.model_name),
                environment_id: tool.server_id.strip_prefix(agent_runtime::BUILTIN_PREFIX),
                run_id: self.run_id.as_deref(),
                agent_id: self.agent_id.as_deref(),
            })
            .await
    }

    async fn audit_status(&self, audit_id: &str, status: &str, approval: &serde_json::Value) -> AppResult<()> {
        self.st.db.set_tool_exec_approval(audit_id, status, approval).await
    }

    async fn audit_end(&self, audit_id: &str, status: &str, result: Option<&str>, approval: &serde_json::Value, duration_ms: u64) -> AppResult<()> {
        // Results are stored truncated and redacted (already done by dispatch).
        self.st.db.finish_tool_exec(audit_id, status, result, approval, duration_ms).await
    }

    async fn call(&self, tool: &ExposedTool, call: &ToolCall, args: serde_json::Value, timeout: Duration, cancel: &CancellationToken) -> AppResult<CallResult> {
        if tool.server_id.starts_with(AGENT_PREFIX) {
            return match tool.tool_name.as_str() {
                "spawn_subagents" => crate::services::tasks::spawn_subagents(self, call, args, timeout, cancel).await,
                "subagent_status" => crate::services::tasks::subagent_status(self, args).await,
                "wait_subagents" => crate::services::tasks::wait_subagents(self, args, timeout, cancel).await,
                "stop_subagent" => crate::services::tasks::stop_subagent_tool(self, args).await,
                "ask_user" => crate::services::tasks::ask_user(self, call, args, cancel).await,
                "skill_propose_edit" => match tokio::time::timeout(timeout, crate::services::skills::propose_from_agent(&self.st, &args, self.run_id.as_deref().or(self.conversation_id.as_deref()))).await {
                    Ok(r) => r,
                    Err(_) => Err(AppError::Runtime(format!("the tool did not finish within {} s", timeout.as_secs()))),
                },
                other => Err(AppError::NotFound(format!("no agent tool {other}"))),
            };
        }
        if let Some(env_id) = tool.server_id.strip_prefix(agent_runtime::BUILTIN_PREFIX) {
            let emit = |stream: &str, text: &str| self.event(ToolEvent::Output { call_id: call.id.clone(), stream: stream.into(), text: text.into() });
            return crate::services::exec::call(&self.st, env_id, &tool.tool_name, args, timeout, cancel, &emit).await;
        }
        let out = self.st.mcp.call_tool(&tool.server_id, &tool.tool_name, args, timeout, cancel).await?;
        Ok(CallResult { text: out.text, is_error: out.is_error })
    }

    fn event(&self, ev: ToolEvent) {
        let _ = self.app.emit("tool://event", &serde_json::json!({ "conversation_id": self.conversation_id, "run_id": self.run_id, "agent_id": self.agent_id, "event": ev }));
    }

    // --- built-in tool approvals: places outside the allowed folders, Plan mode, local network ---
    async fn call_facts(&self, tool: &ExposedTool, args: &serde_json::Value) -> security::policy::CallFacts {
        crate::services::exec::call_facts(&self.st, tool, args).await
    }

    async fn call_approved(&self, tool: &ExposedTool, call: &ToolCall, args: serde_json::Value, timeout: Duration, cancel: &CancellationToken, facts: &security::policy::CallFacts) -> AppResult<CallResult> {
        match tool.server_id.strip_prefix(agent_runtime::BUILTIN_PREFIX) {
            Some(env_id) if !facts.approved.is_empty() => {
                let emit = |stream: &str, text: &str| self.event(ToolEvent::Output { call_id: call.id.clone(), stream: stream.into(), text: text.into() });
                crate::services::exec::call_approved(&self.st, env_id, &tool.tool_name, args, timeout, cancel, &emit, &facts.approved).await
            }
            _ => self.call(tool, call, args, timeout, cancel).await,
        }
    }
    // --- end built-in tool approvals ---
}

/// Answer a pending approval from the UI.
pub fn answer(st: &SharedState, id: &str, answer: ApprovalAnswer) -> AppResult<()> {
    let entry = st.approvals.lock().unwrap().remove(id);
    match entry {
        Some((_, tx)) => {
            let _ = tx.send(answer);
            Ok(())
        }
        None => Err(AppError::NotFound("that approval request is no longer waiting".into())),
    }
}

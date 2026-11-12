//! MCP client: connections to stdio and Streamable HTTP MCP servers.
//!
//! Stdio servers run inside a kill-on-close Windows Job Object (via
//! process-wrap: the child is created suspended, assigned, then resumed, so
//! there is no window where it runs outside the job). Anything they start
//! (e.g. `npx` -> node) is in the same job and dies with it.
//!
//! The manager tracks connection state, captures each server's stderr in a
//! bounded log, notices when a server stops, and calls tools with a timeout
//! and cancellation. Tool descriptions and results are untrusted data: this
//! crate only transports them; policy lives in the `security` crate.

pub mod config;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde::Serialize;
use tokio::io::AsyncBufReadExt;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

pub use config::{ImportedServer, Secrets, ServerConfig, export_mcp_servers, parse_mcp_servers};

const STDERR_LINES: usize = 500;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ToolInfo {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub input_schema: serde_json::Value,
    /// Server-provided hints (readOnlyHint, destructiveHint, ...). UNVERIFIED.
    pub annotations: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub id: String,
    /// disconnected | connecting | connected | error
    pub state: String,
    pub error: Option<String>,
    pub server_name: Option<String>,
    pub server_version: Option<String>,
    pub tools: Vec<ToolInfo>,
    pub resources: Vec<serde_json::Value>,
    pub prompts: Vec<serde_json::Value>,
    pub pid: Option<u32>,
    pub connected_at: Option<String>,
}

impl ServerStatus {
    fn new(id: &str, state: &str) -> Self {
        Self {
            id: id.into(),
            state: state.into(),
            error: None,
            server_name: None,
            server_version: None,
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            pid: None,
            connected_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CallOutput {
    /// Text of all text content blocks, joined.
    pub text: String,
    pub is_error: bool,
    /// The full result as returned by the server.
    pub raw: serde_json::Value,
    pub elapsed_ms: u64,
}

struct Conn {
    service: Arc<RunningService<RoleClient, ()>>,
    stop: CancellationToken,
    /// Set before an intentional disconnect so the watcher does not report a crash.
    closing: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Clone, Default)]
pub struct McpManager {
    conns: Arc<tokio::sync::Mutex<HashMap<String, Conn>>>,
    status: Arc<Mutex<HashMap<String, ServerStatus>>>,
    logs: Arc<Mutex<HashMap<String, VecDeque<String>>>>,
    events: Arc<Mutex<Option<broadcast::Sender<ServerStatus>>>>,
}

impl McpManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self { events: Arc::new(Mutex::new(Some(tx))), ..Default::default() }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ServerStatus> {
        self.events.lock().unwrap().as_ref().expect("events").subscribe()
    }

    fn set(&self, st: ServerStatus) {
        self.status.lock().unwrap().insert(st.id.clone(), st.clone());
        if let Some(tx) = self.events.lock().unwrap().as_ref() {
            let _ = tx.send(st);
        }
    }

    pub fn status(&self, id: &str) -> ServerStatus {
        self.status.lock().unwrap().get(id).cloned().unwrap_or_else(|| ServerStatus::new(id, "disconnected"))
    }

    pub fn all_status(&self) -> Vec<ServerStatus> {
        self.status.lock().unwrap().values().cloned().collect()
    }

    pub fn logs(&self, id: &str) -> Vec<String> {
        self.logs.lock().unwrap().get(id).map(|l| l.iter().cloned().collect()).unwrap_or_default()
    }

    fn log(logs: &Mutex<HashMap<String, VecDeque<String>>>, id: &str, line: String) {
        let mut g = logs.lock().unwrap();
        let q = g.entry(id.to_string()).or_default();
        if q.len() >= STDERR_LINES {
            q.pop_front();
        }
        q.push_back(line);
    }

    /// Connect (or reconnect) a server and fetch its tools, resources and prompts.
    pub async fn connect(&self, id: &str, cfg: &ServerConfig, secrets: &Secrets) -> AppResult<ServerStatus> {
        cfg.validate().map_err(AppError::InvalidInput)?;
        self.disconnect(id).await;
        self.logs.lock().unwrap().remove(id);
        self.set(ServerStatus::new(id, "connecting"));
        let started = Instant::now();
        let result = tokio::time::timeout(CONNECT_TIMEOUT, self.open(id, cfg, secrets)).await;
        let (service, pid) = match result {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => return Err(self.fail(id, e)),
            Err(_) => return Err(self.fail(id, format!("the server did not finish starting within {}s", CONNECT_TIMEOUT.as_secs()))),
        };
        let service = Arc::new(service);
        let info = service.peer_info().map(|i| serde_json::to_value(&*i).unwrap_or_default()).unwrap_or_default();
        let tools = match service.list_all_tools().await {
            Ok(t) => t.iter().map(tool_info).collect(),
            Err(e) => {
                let _ = service.cancellation_token().cancel();
                return Err(self.fail(id, format!("connected, but listing tools failed: {e}")));
            }
        };
        let caps = &info["capabilities"];
        // Resources and prompts are optional capabilities; failures are not fatal.
        let resources = if caps.get("resources").is_some() {
            service.list_all_resources().await.map(|r| r.iter().map(|x| serde_json::to_value(x).unwrap_or_default()).collect()).unwrap_or_default()
        } else {
            vec![]
        };
        let prompts = if caps.get("prompts").is_some() {
            service.list_all_prompts().await.map(|r| r.iter().map(|x| serde_json::to_value(x).unwrap_or_default()).collect()).unwrap_or_default()
        } else {
            vec![]
        };
        let stop = CancellationToken::new();
        let closing = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let st = ServerStatus {
            id: id.into(),
            state: "connected".into(),
            error: None,
            server_name: info["serverInfo"]["name"].as_str().map(str::to_string),
            server_version: info["serverInfo"]["version"].as_str().map(str::to_string),
            tools,
            resources,
            prompts,
            pid,
            connected_at: Some(app_core::now_rfc3339()),
        };
        Self::log(&self.logs, id, format!("[app] connected in {} ms", started.elapsed().as_millis()));
        self.set(st.clone());
        // Watcher: notice the server going away on its own.
        {
            let me = self.clone();
            let svc = service.clone();
            let id = id.to_string();
            let stop = stop.clone();
            let closing = closing.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = stop.cancelled() => return,
                        _ = tokio::time::sleep(Duration::from_millis(400)) => {}
                    }
                    if svc.is_transport_closed() || svc.is_closed() {
                        if !closing.load(std::sync::atomic::Ordering::SeqCst) {
                            let tail = me.logs(&id).into_iter().rev().take(3).collect::<Vec<_>>().join(" | ");
                            let msg = format!("the server stopped unexpectedly{}", if tail.is_empty() { String::new() } else { format!(" (last output: {tail})") });
                            tracing::warn!(server = %id, "MCP {msg}");
                            me.conns.lock().await.remove(&id);
                            let mut s = ServerStatus::new(&id, "error");
                            s.error = Some(msg);
                            me.set(s);
                        }
                        return;
                    }
                }
            });
        }
        self.conns.lock().await.insert(id.to_string(), Conn { service, stop, closing });
        Ok(st)
    }

    fn fail(&self, id: &str, msg: String) -> AppError {
        let mut s = ServerStatus::new(id, "error");
        s.error = Some(msg.clone());
        self.set(s);
        AppError::Runtime(msg)
    }

    async fn open(&self, id: &str, cfg: &ServerConfig, secrets: &Secrets) -> Result<(RunningService<RoleClient, ()>, Option<u32>), String> {
        if cfg.transport == "http" {
            let mut c = StreamableHttpClientTransportConfig::with_uri(cfg.url.clone());
            if let Some(tok) = secrets.bearer.as_deref().filter(|t| !t.is_empty()) {
                // rmcp adds the "Bearer " prefix itself (reqwest bearer_auth).
                c = c.auth_header(tok.to_string());
            }
            let mut headers = HashMap::new();
            for (k, v) in cfg.headers.iter().chain(secrets.headers.iter()) {
                let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| format!("bad header name {k}: {e}"))?;
                let val = reqwest::header::HeaderValue::from_str(v).map_err(|e| format!("bad header value for {k}: {e}"))?;
                headers.insert(name, val);
            }
            c.custom_headers = headers;
            let transport = StreamableHttpClientTransport::from_config(c);
            let svc = ().serve(transport).await.map_err(|e| format!("could not connect to {}: {e}", cfg.url))?;
            return Ok((svc, None));
        }
        // Stdio. On Windows, .cmd/.bat shims (npx, uvx wrappers) need cmd.exe.
        let mut cmd = build_command(&cfg.command, &cfg.args);
        for (k, v) in cfg.env.iter().chain(secrets.env.iter()) {
            cmd.env(k, v);
        }
        if let Some(cwd) = cfg.cwd.as_deref().filter(|c| !c.is_empty()) {
            cmd.current_dir(cwd);
        }
        let mut wrap = process_wrap::tokio::CommandWrap::from(cmd);
        #[cfg(windows)]
        {
            use windows::Win32::System::Threading::CREATE_NO_WINDOW;
            wrap.wrap(process_wrap::tokio::CreationFlags(CREATE_NO_WINDOW));
            wrap.wrap(process_wrap::tokio::JobObject);
        }
        wrap.wrap(process_wrap::tokio::KillOnDrop);
        let (transport, stderr) = TokioChildProcess::builder(wrap)
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start '{}': {e}", cfg.command))?;
        let pid = transport.id();
        if let Some(err) = stderr {
            let logs = self.logs.clone();
            let id = id.to_string();
            tokio::spawn(async move {
                let mut lines = tokio::io::BufReader::new(err).lines();
                while let Ok(Some(l)) = lines.next_line().await {
                    let l = if l.len() > 2000 { format!("{}...", &l[..l.floor_char_boundary(2000)]) } else { l };
                    Self::log(&logs, &id, l);
                }
            });
        }
        match ().serve(transport).await {
            Ok(svc) => Ok((svc, pid)),
            Err(e) => {
                // Give stderr a moment so the error can include the reason.
                tokio::time::sleep(Duration::from_millis(300)).await;
                let tail = self.logs(id).into_iter().rev().take(5).collect::<Vec<_>>();
                let tail: Vec<String> = tail.into_iter().rev().collect();
                Err(format!(
                    "the server started but the MCP handshake failed: {e}{}",
                    if tail.is_empty() { String::new() } else { format!(". Server output: {}", tail.join(" | ")) }
                ))
            }
        }
    }

    pub async fn disconnect(&self, id: &str) {
        let c = self.conns.lock().await.remove(id);
        if let Some(c) = c {
            c.closing.store(true, std::sync::atomic::Ordering::SeqCst);
            c.stop.cancel();
            c.service.cancellation_token().cancel();
            // Dropping the last Arc drops the transport, which kills the job.
            drop(c);
            Self::log(&self.logs, id, "[app] disconnected".into());
            self.set(ServerStatus::new(id, "disconnected"));
        } else if self.status.lock().unwrap().get(id).is_some_and(|s| s.state != "disconnected") {
            self.set(ServerStatus::new(id, "disconnected"));
        }
    }

    pub async fn disconnect_all(&self) {
        let ids: Vec<String> = self.conns.lock().await.keys().cloned().collect();
        for id in ids {
            self.disconnect(&id).await;
        }
    }

    pub fn forget(&self, id: &str) {
        self.status.lock().unwrap().remove(id);
        self.logs.lock().unwrap().remove(id);
    }

    /// Call a tool. `arguments` must be a JSON object (validated by the caller).
    pub async fn call_tool(
        &self,
        id: &str,
        tool: &str,
        arguments: serde_json::Value,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> AppResult<CallOutput> {
        let service = {
            let g = self.conns.lock().await;
            g.get(id).map(|c| c.service.clone()).ok_or_else(|| AppError::Conflict("the MCP server is not connected".into()))?
        };
        let args = match arguments {
            serde_json::Value::Object(m) => m,
            serde_json::Value::Null => serde_json::Map::new(),
            _ => return Err(AppError::InvalidInput("tool arguments must be a JSON object".into())),
        };
        let params = rmcp::model::CallToolRequestParams::new(tool.to_string()).with_arguments(args);
        let started = Instant::now();
        let fut = service.call_tool(params);
        let res = tokio::select! {
            r = tokio::time::timeout(timeout, fut) => r,
            _ = cancel.cancelled() => return Err(AppError::Cancelled),
        };
        let result = match res {
            Err(_) => return Err(AppError::Runtime(format!("the tool did not finish within {}s", timeout.as_secs()))),
            Ok(Err(e)) => return Err(AppError::Runtime(format!("tool call failed: {e}"))),
            Ok(Ok(r)) => r,
        };
        let raw = serde_json::to_value(&result).unwrap_or_default();
        let mut text = String::new();
        if let Some(items) = raw["content"].as_array() {
            for c in items {
                match c["type"].as_str() {
                    Some("text") => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(c["text"].as_str().unwrap_or(""));
                    }
                    Some(other) => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&format!("[{other} content omitted]"));
                    }
                    None => {}
                }
            }
        }
        if text.is_empty()
            && let Some(sc) = raw.get("structuredContent")
        {
            text = sc.to_string();
        }
        Ok(CallOutput {
            text,
            is_error: raw["isError"].as_bool().unwrap_or(false),
            raw,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

fn tool_info(t: &rmcp::model::Tool) -> ToolInfo {
    let v = serde_json::to_value(t).unwrap_or_default();
    ToolInfo {
        name: v["name"].as_str().unwrap_or("").to_string(),
        title: v["title"].as_str().map(str::to_string),
        description: v["description"].as_str().unwrap_or("").to_string(),
        input_schema: v.get("inputSchema").cloned().unwrap_or_else(|| serde_json::json!({ "type": "object" })),
        annotations: v.get("annotations").cloned().unwrap_or(serde_json::Value::Null),
    }
}

/// Build the process command. On Windows, commands that are not .exe files
/// (npx, uvx, .cmd/.bat shims, bare names that resolve to .cmd) go through
/// `cmd.exe /d /s /c` so PATHEXT resolution works like in a terminal.
fn build_command(command: &str, args: &[String]) -> tokio::process::Command {
    #[cfg(windows)]
    {
        let lower = command.to_ascii_lowercase();
        let is_exe = lower.ends_with(".exe") || (!lower.contains('.') && which_exe(command));
        if !is_exe {
            let mut c = tokio::process::Command::new("cmd.exe");
            c.arg("/d").arg("/s").arg("/c").arg(command).args(args);
            return c;
        }
    }
    let mut c = tokio::process::Command::new(command);
    c.args(args);
    c
}

/// True when `name` resolves to a real .exe on PATH (so no cmd.exe is needed).
#[cfg(windows)]
fn which_exe(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    std::env::split_paths(&path).any(|d| d.join(format!("{name}.exe")).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real stdio server: the reference "everything" server via npx (network
    /// on first run to fetch the package).
    #[tokio::test]
    #[ignore = "needs node/npx and network on first run"]
    async fn live_everything_server_lists_and_calls_tools() {
        let m = McpManager::new();
        let cfg = ServerConfig {
            command: "npx".into(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-everything@2.2.1".into()],
            timeout_secs: 30,
            ..Default::default()
        };
        let st = m.connect("t1", &cfg, &Secrets::default()).await.unwrap();
        assert_eq!(st.state, "connected");
        assert!(st.tools.iter().any(|t| t.name == "echo"), "{:?}", st.tools.iter().map(|t| &t.name).collect::<Vec<_>>());
        let out = m
            .call_tool("t1", "echo", serde_json::json!({ "message": "hello mcp" }), Duration::from_secs(20), &CancellationToken::new())
            .await
            .unwrap();
        assert!(out.text.contains("hello mcp"), "{}", out.text);
        let pid = st.pid.unwrap();
        m.disconnect("t1").await;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let alive = std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]).output().unwrap();
        assert!(!String::from_utf8_lossy(&alive.stdout).contains(&pid.to_string()), "server process survived disconnect");
    }

    #[tokio::test]
    async fn bad_command_reports_a_clear_error() {
        let m = McpManager::new();
        let cfg = ServerConfig { command: "definitely-not-a-real-program-xyz".into(), ..Default::default() };
        let e = m.connect("bad", &cfg, &Secrets::default()).await.unwrap_err();
        assert!(e.to_string().contains("handshake failed") || e.to_string().contains("could not start"), "{e}");
        assert_eq!(m.status("bad").state, "error");
    }
}

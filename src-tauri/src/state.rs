use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use app_core::AppPaths;
use inference::{Engine, RuntimeManager};
use local_api::ApiServer;
use mcp_client::McpManager;
use model_registry::downloads::DownloadManager;
use persistence::Db;
use security::CredentialStore;
use system_monitor::SystemMonitor;
use tokio_util::sync::CancellationToken;

pub const CRED_HF_TOKEN: &str = "hf/token";
pub const CRED_API_TOKEN: &str = "api/token";

/// Everything the commands need. Held in Tauri's managed state.
pub struct AppState {
    pub paths: AppPaths,
    pub db: Db,
    pub monitor: SystemMonitor,
    pub runtimes: RuntimeManager,
    pub engine: Engine,
    pub credentials: CredentialStore,
    pub http: reqwest::Client,
    pub downloads: DownloadManager,
    pub api: ApiServer,
    pub mcp: McpManager,
    /// Live execution environments (SSH sessions, containers, sandboxes) by id.
    pub envs: crate::services::exec::EnvCache,
    /// Tool calls waiting for the user's answer, by audit id.
    pub approvals: Mutex<HashMap<String, (agent_runtime::ApprovalRequest, tokio::sync::oneshot::Sender<security::policy::ApprovalAnswer>)>>,
    /// Cancellation tokens for running runtime installs, by runtime id.
    pub installs: Mutex<HashMap<String, CancellationToken>>,
    /// Running chat generations, by conversation id.
    pub generations: Mutex<HashMap<String, CancellationToken>>,
    /// Token counts per (backend port, text hash) so the context meter does
    /// not re-tokenize unchanged history on every keystroke.
    pub token_cache: Mutex<HashMap<(u16, u64), usize>>,
    /// One generation at a time per loaded model slot (chats, tasks and
    /// subagents share it), keyed by backend instance id.
    pub gen_gate: Mutex<(Option<String>, agent_runtime::Gate)>,
    /// Running agent tasks, by run id.
    pub tasks: Mutex<HashMap<String, Arc<crate::services::tasks::RunCtx>>>,
    /// Tasks between "start/resume" and running (reserved so they start once).
    pub tasks_starting: Mutex<std::collections::HashSet<String>>,
    /// Stop tokens for running subagents, by subagent id.
    pub subagent_stops: Mutex<HashMap<String, CancellationToken>>,
    /// Attachments added to a chat's composer but not sent yet, by conversation id.
    /// Their files are already in the attachments folder; the next send takes them.
    pub pending_attachments: Mutex<HashMap<String, Vec<crate::services::attachments::Attachment>>>,
    /// Layers llama.cpp put on the GPU (of all) when it chose them itself
    /// (GPU offload Auto / `--fit`), by (model id, `models::offload_key`):
    /// the next load with the same settings knows whether it is split.
    pub observed_offload: Mutex<HashMap<(String, String), (u32, u32)>>,
    /// Linked PCs: helper listener + rpc-server, pairing, tunnels to helpers.
    pub linking: Arc<crate::services::linking::LinkService>,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    /// Where new downloads go: the folder recorded in storage.json (the one
    /// source of truth), else the folder resolved at startup.
    pub async fn models_dir(&self) -> PathBuf {
        app_core::storage::read(&self.paths.data_dir)
            .map(|c| c.models_dir)
            .unwrap_or_else(|| self.paths.default_models_dir.clone())
    }

    /// The generation gate for the loaded model: as many permits as the
    /// backend has parallel slots (1 by default).
    pub fn gate(&self) -> agent_runtime::Gate {
        let status = self.engine.status();
        let mut g = self.gen_gate.lock().unwrap();
        if g.0 != status.instance_id {
            let slots = status.props.as_ref().and_then(|p| p.total_slots).unwrap_or(1).clamp(1, 8) as usize;
            *g = (status.instance_id.clone(), agent_runtime::Gate::new(slots));
        }
        g.1.clone()
    }

    pub fn hf_client(&self) -> model_registry::hf::HfClient {
        let token = self.credentials.get(CRED_HF_TOKEN).ok().flatten();
        model_registry::hf::HfClient::new(self.http.clone(), token)
    }
}

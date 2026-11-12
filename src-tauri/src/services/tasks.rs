//! Agent tasks in the app: start / pause / resume / cancel, the step log and
//! live events, and subagents (the `spawn_subagents` tool).
//!
//! A task runs `agent_runtime::task::run_task` with the root agent's tools.
//! Subagents run the same loop with a subset of the parent's tools, a budget
//! carved out of the parent's, and child cancellation tokens, so stopping
//! the task reaches every child (and their commands die with their Job
//! Objects / process groups).

use std::sync::{Arc, Mutex};

use agent_runtime::task::{Budgets, TaskEnd, TaskObserver, Usage, run_task};
use agent_runtime::{CallResult, DispatchOutcome, ExposedTool, LoopHooks, ToolMode, TurnCtx};
use app_core::{AppError, AppResult};
use async_trait::async_trait;
use inference::{ChatDelta, ChatStats, LlamaClient, SamplingParams, ToolCall};
use persistence::{AgentRunSummary, AgentStepRow, SubagentRow};
use security::policy::Policy;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use tokio_util::sync::CancellationToken;

use crate::services::mcp::{AGENT_PREFIX, AppToolHost};
use crate::services::{chat, exec, instructions, mcp, skills};
use crate::state::SharedState;

pub const MAX_PARALLEL_HARD: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TaskSettings {
    /// Execution environment for the built-in tools (None = This PC).
    pub environment_id: Option<String>,
    pub builtin_tools: bool,
    /// Offer MCP tools; `tool_servers` empty = every connected server.
    pub mcp_tools: bool,
    pub tool_servers: Vec<String>,
    /// auto | off | manual (exactly `skills`)
    pub skills_mode: String,
    pub skills: Vec<String>,
    pub subagents: bool,
    pub max_parallel: u32,
    /// auto | native | xml
    pub tool_mode: String,
    pub sampling: SamplingParams,
    /// Run a read-only planning phase first and wait for the user to approve
    /// the plan (state `awaiting_plan_approval`).
    pub plan_first: bool,
    /// The plan the user approved (set by `approve_plan`); the task then runs
    /// with full tools and this plan in its brief.
    pub approved_plan: Option<String>,
    /// Thinking on/off for every model call of the task and its subagents
    /// (None = the model's default; only sent to models with a toggle).
    pub thinking: Option<bool>,
    /// Reasoning effort level (None = default; only levels the model lists are sent).
    pub reasoning_effort: Option<String>,
}

impl Default for TaskSettings {
    fn default() -> Self {
        Self {
            environment_id: None,
            builtin_tools: true,
            mcp_tools: false,
            tool_servers: vec![],
            skills_mode: "auto".into(),
            skills: vec![],
            subagents: true,
            max_parallel: 2,
            tool_mode: "auto".into(),
            sampling: SamplingParams::default(),
            plan_first: false,
            approved_plan: None,
            thinking: None,
            reasoning_effort: None,
        }
    }
}

/// Live state of a running task, shared with the spawn tool.
pub struct RunCtx {
    pub run_id: String,
    pub cancel: CancellationToken,
    pub pause: CancellationToken,
    pub settings: TaskSettings,
    pub budgets: Budgets,
    /// The root agent's usage after its last turn.
    pub usage: Mutex<Usage>,
    /// Usage of finished subagents, folded into the root before its next turn.
    pub child_usage: Mutex<Usage>,
    /// Root tools (the spawn tool included).
    pub tools: Vec<ExposedTool>,
    pub mode: ToolMode,
    pub client: LlamaClient,
    /// Base + agent instruction + environment and skills text, reused for subagents.
    pub shared_prompt: String,
    pub env_label: String,
    pub children: tokio::sync::Semaphore,
    /// Set for a chat reply's run (hidden agent_runs row): the conversation.
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskDetail {
    pub run: serde_json::Value,
    pub steps: Vec<AgentStepRow>,
    pub subagents: Vec<SubagentRow>,
    pub running: bool,
}

fn emit(app: &AppHandle, payload: serde_json::Value) {
    let _ = app.emit("task://event", &payload);
}

fn task_section(objective: &str, budgets: &Budgets, usage: &Usage, subagents: bool) -> String {
    let left = budgets.remaining(usage);
    let mut s = format!(
        "## Task\nYou are running an agent task, not a chat. Objective: {objective}\n\
Work step by step toward the objective using your tools. Keep going without asking the user questions: they are not watching each step. If something blocks you, explain it in your final answer.\n\
Check your work (run the tests, read the file back) before saying it is done.\n\
Your final answer is a reply with no tool call: start with the result, then say what you changed and how you checked it.\n\
Budget left: {} model turns, {} tool calls, about {} minutes.",
        left.max_turns,
        left.max_tool_calls,
        (left.max_wall_secs / 60).max(1)
    );
    if subagents {
        s.push('\n');
        s.push_str(SUBAGENT_HINT);
    }
    s
}

/// How to use spawn_subagents (tasks, and chats that allow subagents).
pub const SUBAGENT_HINT: &str = "You can delegate independent parts of the work with spawn_subagents. Each subagent gets its own objective and a share of your budget, works in parallel, and reports back as that tool's result. Only delegate work that is independent and worth it; do small things yourself. \
For a subagent that edits files, set \"isolation\": \"worktree\": it then works in its own git worktree and branch (on This PC, when the first allowed folder is inside a git repository), so subagents never edit the same files at once. Its result names the branch and folder; review its changes and merge them yourself with the shell tool.";

fn subagent_section(label: &str, objective: &str, budgets: &Budgets) -> String {
    format!(
        "## Subagent task\nYou are subagent \"{label}\", working for a main agent. Your objective: {objective}\n\
Do only this. You cannot start other subagents. Do not ask questions; work with what you have.\n\
When done, reply with no tool call: your result first, concise, with the facts the main agent needs.\n\
Budget: {} model turns, {} tool calls, about {} minutes.",
        budgets.max_turns,
        budgets.max_tool_calls,
        (budgets.max_wall_secs / 60).max(1)
    )
}

pub fn spawn_tool(run_id: &str) -> ExposedTool {
    ExposedTool {
        tool_id: format!("{AGENT_PREFIX}{run_id}/spawn_subagents"),
        model_name: "spawn_subagents".into(),
        server_id: format!("{AGENT_PREFIX}{run_id}"),
        server_name: "Local AI Studio".into(),
        tool_name: "spawn_subagents".into(),
        description: "Start subagents that work in parallel on independent parts of the task and report back. Each gets a label, its own objective, optionally a subset of your tools, and a share of your budget. Returns each subagent's final answer.".into(),
        input_schema: serde_json::json!({ "type": "object", "properties": {
            "subagents": { "type": "array", "minItems": 1, "maxItems": 6, "items": { "type": "object", "properties": {
                "label": { "type": "string", "description": "Short name, e.g. \"tests\"" },
                "objective": { "type": "string", "description": "What this subagent must do and report" },
                "tools": { "type": "array", "items": { "type": "string" }, "description": "Tool names it may use (default: all of yours except spawn_subagents)" },
                "max_turns": { "type": "integer", "minimum": 1 },
                "max_tool_calls": { "type": "integer", "minimum": 0 },
                "isolation": { "type": "string", "enum": ["none", "worktree"], "description": "worktree: work in its own git worktree and branch (use it for subagents that edit files). Default none" }
            }, "required": ["label", "objective"], "additionalProperties": false } }
        }, "required": ["subagents"], "additionalProperties": false }),
        policy: Policy::Allow,
        timeout_secs: 4 * 3600,
    }
}

// ---------- hooks: step log + live events ----------

struct StepHooks {
    st: SharedState,
    app: AppHandle,
    run_id: String,
    agent_id: Option<String>,
    current: Mutex<Option<String>>,
    /// Streamed text not yet sent to the UI (content, reasoning, last send):
    /// deltas go out at most every 40 ms, like chat, not once per token.
    pending: Mutex<(String, String, std::time::Instant)>,
}

impl StepHooks {
    fn new(st: SharedState, app: AppHandle, run_id: String, agent_id: Option<String>) -> Self {
        Self { st, app, run_id, agent_id, current: Mutex::new(None), pending: Mutex::new((String::new(), String::new(), std::time::Instant::now())) }
    }

    fn flush(&self, force: bool) {
        let (c, r) = {
            let mut p = self.pending.lock().unwrap();
            if !force && p.2.elapsed() < std::time::Duration::from_millis(40) {
                return;
            }
            p.2 = std::time::Instant::now();
            (std::mem::take(&mut p.0), std::mem::take(&mut p.1))
        };
        if c.is_empty() && r.is_empty() {
            return;
        }
        let id = self.current.lock().unwrap().clone();
        emit(&self.app, serde_json::json!({ "run_id": self.run_id, "agent_id": self.agent_id, "type": "delta", "step_id": id, "content": c, "reasoning": r }));
    }

    fn clear_pending(&self) {
        let mut p = self.pending.lock().unwrap();
        p.0.clear();
        p.1.clear();
    }
}

#[async_trait]
impl LoopHooks for StepHooks {
    async fn begin_segment(&self, _segment: usize) -> AppResult<()> {
        self.clear_pending();
        let step = self.st.db.add_step(&self.run_id, self.agent_id.as_deref(), "model", &serde_json::json!({ "text": "", "reasoning": "" }), "running").await?;
        *self.current.lock().unwrap() = Some(step.id.clone());
        emit(&self.app, serde_json::json!({ "run_id": self.run_id, "agent_id": self.agent_id, "type": "step", "step": step }));
        Ok(())
    }

    fn reset_segment(&self, _segment: usize) {
        self.clear_pending();
        let id = self.current.lock().unwrap().clone();
        emit(&self.app, serde_json::json!({ "run_id": self.run_id, "agent_id": self.agent_id, "type": "reset", "step_id": id }));
    }

    fn on_delta(&self, _segment: usize, d: &ChatDelta) {
        {
            let mut p = self.pending.lock().unwrap();
            match d {
                ChatDelta::Content { text } => p.0.push_str(text),
                ChatDelta::Reasoning { text } => p.1.push_str(text),
            }
        }
        self.flush(false);
    }

    async fn on_segment(&self, _segment: usize, content: &str, reasoning: &str, tool_calls: &[ToolCall], stats: &ChatStats) -> AppResult<String> {
        self.flush(true);
        let id = self.current.lock().unwrap().clone().unwrap_or_default();
        let calls: Vec<serde_json::Value> = tool_calls.iter().map(|c| serde_json::json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments } })).collect();
        let data = serde_json::json!({ "text": content, "reasoning": reasoning, "tool_calls": calls, "stats": stats });
        self.st.db.update_step(&id, &data, "done").await?;
        emit(&self.app, serde_json::json!({ "run_id": self.run_id, "agent_id": self.agent_id, "type": "step_done", "step_id": id, "data": data }));
        Ok(id)
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &DispatchOutcome) -> AppResult<()> {
        let data = serde_json::json!({ "call_id": call.id, "name": call.name, "arguments": call.arguments, "status": outcome.status, "content": outcome.content, "audit_id": outcome.audit_id });
        let step = self.st.db.add_step(&self.run_id, self.agent_id.as_deref(), "tool", &data, "done").await?;
        emit(&self.app, serde_json::json!({ "run_id": self.run_id, "agent_id": self.agent_id, "type": "step", "step": step }));
        Ok(())
    }
}

/// Saves progress after each turn; keeps the transcript inside the context.
struct Observer {
    st: SharedState,
    app: AppHandle,
    run: Arc<RunCtx>,
    /// None = root agent (saves the run), Some = a subagent.
    agent_id: Option<String>,
    n_ctx: u64,
}

/// Rough token estimate; the exact count is not needed to decide when to trim.
fn approx_tokens(v: &[serde_json::Value]) -> u64 {
    (serde_json::to_string(v).map(|s| s.len()).unwrap_or(0) / 3) as u64
}

fn is_tool_result(m: &serde_json::Value) -> bool {
    m["role"] == "tool" || (m["role"] == "user" && m["content"].as_str().is_some_and(|c| c.starts_with("<result")))
}

/// Keep the transcript under `limit` (approximate tokens): first remove
/// older tool outputs (oldest first); if the newest outputs alone are still
/// too big (e.g. several large reads in one turn), shorten them, keeping
/// their start and end. The system message and the objective are never
/// touched. Returns (removed, shortened).
fn fit_transcript(transcript: &mut [serde_json::Value], limit: u64) -> (usize, usize) {
    const PLACEHOLDER: &str = "[older tool output removed to save context]";
    let mut size = approx_tokens(transcript);
    if size <= limit {
        return (0, 0);
    }
    // Results after the last assistant message are the newest batch.
    let last_assistant = transcript.iter().rposition(|m| m["role"] == "assistant").unwrap_or(0);
    let mut removed = 0;
    for i in 2..last_assistant {
        if size <= limit {
            break;
        }
        let m = &mut transcript[i];
        if is_tool_result(m)
            && let Some(len) = m["content"].as_str().map(str::len).filter(|l| *l > 200)
        {
            m["content"] = PLACEHOLDER.into();
            size = size.saturating_sub(((len - PLACEHOLDER.len()) / 3) as u64);
            removed += 1;
        }
    }
    let mut shortened = 0;
    if size > limit {
        let newest: Vec<usize> = (last_assistant.max(2)..transcript.len()).filter(|&i| is_tool_result(&transcript[i])).collect();
        if !newest.is_empty() {
            // Characters to drop, spread over the newest outputs.
            let excess_chars = ((size - limit) * 3) as usize;
            let total: usize = newest.iter().map(|&i| transcript[i]["content"].as_str().map(str::len).unwrap_or(0)).sum();
            for &i in &newest {
                let Some(c) = transcript[i]["content"].as_str().map(str::to_string) else { continue };
                let cut = excess_chars * c.len() / total.max(1) + 200;
                if c.len() <= cut + 800 {
                    continue;
                }
                let keep = c.len() - cut;
                let head_end = (0..=keep * 2 / 3).rev().find(|&b| c.is_char_boundary(b)).unwrap_or(0);
                let tail_start = (c.len() - keep / 3..=c.len()).find(|&b| c.is_char_boundary(b)).unwrap_or(c.len());
                let new = format!(
                    "{}\n[... {} characters removed to fit the context. Read a smaller part (for example fs_read with start_line and max_lines) if you need them ...]\n{}",
                    &c[..head_end],
                    tail_start - head_end,
                    &c[tail_start..]
                );
                size = size.saturating_sub(((c.len().saturating_sub(new.len())) / 3) as u64);
                transcript[i]["content"] = new.into();
                shortened += 1;
            }
        }
    }
    (removed, shortened)
}

#[async_trait]
impl TaskObserver for Observer {
    async fn prepare(&self, transcript: &mut Vec<serde_json::Value>, usage: &mut Usage) -> AppResult<()> {
        if self.agent_id.is_none() {
            let done = std::mem::take(&mut *self.run.child_usage.lock().unwrap());
            usage.add(&done);
        }
        let (removed, shortened) = fit_transcript(transcript, self.n_ctx * 7 / 10);
        if removed + shortened > 0 {
            let mut text = vec![];
            if removed > 0 {
                text.push(format!("{removed} older tool output(s) removed from the model's context to make room"));
            }
            if shortened > 0 {
                text.push(format!("{shortened} recent tool output(s) shortened to fit the context"));
            }
            let step = self
                .st
                .db
                .add_step(&self.run.run_id, self.agent_id.as_deref(), "note", &serde_json::json!({ "text": format!("{}.", text.join("; ")) }), "done")
                .await?;
            emit(&self.app, serde_json::json!({ "run_id": self.run.run_id, "agent_id": self.agent_id, "type": "step", "step": step }));
        }
        Ok(())
    }

    async fn checkpoint(&self, transcript: &[serde_json::Value], usage: &Usage) -> AppResult<()> {
        let t = serde_json::Value::Array(transcript.to_vec());
        let u = serde_json::to_value(usage)?;
        match &self.agent_id {
            None => {
                *self.run.usage.lock().unwrap() = *usage;
                self.st.db.save_run_progress(&self.run.run_id, &t, &u).await?;
            }
            Some(a) => self.st.db.save_subagent_progress(a, &t, &u).await?,
        }
        emit(&self.app, serde_json::json!({ "run_id": self.run.run_id, "agent_id": self.agent_id, "type": "usage", "usage": usage }));
        Ok(())
    }
}

// ---------- starting and running ----------

pub async fn start(st: &SharedState, app: &AppHandle, title: &str, objective: &str, settings: TaskSettings, budgets: Budgets) -> AppResult<String> {
    let (_, model_id, _, _) = chat::loaded(st)?;
    if settings.max_parallel == 0 || settings.max_parallel > MAX_PARALLEL_HARD {
        return Err(AppError::InvalidInput(format!("subagents at once must be 1 to {MAX_PARALLEL_HARD}")));
    }
    chat::validate_effort(settings.reasoning_effort.as_deref())?;
    if let Some(env) = &settings.environment_id {
        exec::meta(st, env).await?;
    }
    let title = if title.trim().is_empty() { objective.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect::<String>() } else { title.to_string() };
    let id = st.db.create_run(&title, objective, Some(&model_id), &serde_json::to_value(&settings)?, &serde_json::to_value(budgets)?).await?;
    launch(st, app, &id).await?;
    Ok(id)
}

/// Build the root tools and prompt, then run (or resume) the task in the background.
pub async fn launch(st: &SharedState, app: &AppHandle, run_id: &str) -> AppResult<()> {
    // Reserve the task before any await, so two quick Start/Resume clicks
    // can never start two runners on one transcript.
    {
        let tasks = st.tasks.lock().unwrap();
        let mut starting = st.tasks_starting.lock().unwrap();
        if tasks.contains_key(run_id) || !starting.insert(run_id.to_string()) {
            return Err(AppError::Conflict("this task is already running".into()));
        }
    }
    let r = launch_reserved(st, app, run_id).await;
    st.tasks_starting.lock().unwrap().remove(run_id);
    r
}

async fn launch_reserved(st: &SharedState, app: &AppHandle, run_id: &str) -> AppResult<()> {
    let row = st.db.get_run(run_id).await?;
    let mut settings: TaskSettings = serde_json::from_str(&row.settings).unwrap_or_default();
    let budgets: Budgets = serde_json::from_str(&row.budgets).unwrap_or_default();
    let usage: Usage = serde_json::from_str(&row.usage).unwrap_or_default();
    let (client, model_id, n_ctx, supports_tools) = chat::loaded(st)?;
    // Thinking controls for every model call (subagents use these settings too).
    if let Ok(m) = st.db.get_model(&model_id).await {
        settings.sampling.template_kwargs = chat::thinking_request(&chat::caps_of(&m), settings.thinking, settings.reasoning_effort.as_deref()).kwargs;
    }
    // "Plan first": a read-only planning phase until the user approves a plan.
    let planning = settings.plan_first && settings.approved_plan.is_none();

    // Tools: built-ins for the environment, MCP, skill proposals, spawn.
    let env_id = settings.environment_id.clone().unwrap_or_else(|| exec::LOCAL_ID.to_string());
    let mut tools = vec![];
    let mut env_text = String::new();
    if settings.builtin_tools {
        let (t, sec) = exec::exposed(st, &env_id).await?;
        tools.extend(t);
        env_text = sec;
    }
    if settings.mcp_tools {
        tools.extend(mcp::exposed_tools(st, &settings.tool_servers).await?);
    }
    let picked = skills::pick(st, &row.objective, &settings.skills_mode, &settings.skills).await?;
    if planning {
        // Only reads while planning: no shell, writes, proposals or subagents.
        tools = chat::plan_filter(st, tools);
    } else {
        if !picked.is_empty() {
            tools.push(skills::propose_tool(run_id));
        }
        if settings.subagents {
            tools.push(spawn_tool(run_id));
        }
    }
    let mode = match settings.tool_mode.as_str() {
        "xml" => ToolMode::Xml,
        "native" => ToolMode::Native,
        _ if supports_tools => ToolMode::Native,
        _ => ToolMode::Xml,
    };

    // Prompt: base + agent profile + generated sections.
    let base = st.db.get_setting::<String>("instructions.base").await?.unwrap_or_default();
    let resolved = instructions::resolve_profile_mode(&st.db, None, Some(&model_id), if planning { "plan" } else { "agent" }).await?;
    let sections = resolved.as_ref().map(|(_, p)| instructions::sections_of(p)).unwrap_or_default();
    let skills_text = skills::section(st, &picked).await?;
    let mut shared = String::new();
    if !env_text.is_empty() {
        shared.push_str(&env_text);
    }
    if let Some(s) = &skills_text {
        shared.push_str("\n\n");
        shared.push_str(s);
    }
    if mode == ToolMode::Xml {
        shared.push_str("\n\n");
        shared.push_str(&agent_runtime::xml::format_section());
    }
    let generated = format!(
        "{}\n\n{}\n\n{}",
        agent_runtime::tools_section(&tools, &agent_runtime::Limits { max_tool_calls: budgets.max_tool_calls, max_wall_secs: budgets.max_wall_secs }),
        shared.trim(),
        if planning { plan_section(&row.objective, &budgets, &usage) } else { task_section(&row.objective, &budgets, &usage, settings.subagents) }
    );
    let (system, _) = instructions::compose(&base, &sections, "agent", Some(&generated));
    let (shared_prompt, _) = instructions::compose(&base, &sections, "agent", Some(shared.trim()));

    let mut transcript: Vec<serde_json::Value> = serde_json::from_str(&st.db.get_run_transcript(run_id).await?).unwrap_or_default();
    if transcript.is_empty() {
        let brief = match &settings.approved_plan {
            Some(plan) => format!("{}\n\nApproved plan (follow it; if a step turns out to be wrong, adapt and say why):\n{plan}", row.objective),
            None => row.objective.clone(),
        };
        transcript = vec![serde_json::json!({ "role": "system", "content": system }), serde_json::json!({ "role": "user", "content": brief })];
        if !picked.is_empty() {
            let text = picked.iter().map(|m| format!("{} ({})", m.name, m.reason)).collect::<Vec<_>>().join("; ");
            st.db.add_step(run_id, None, "note", &serde_json::json!({ "text": format!("Skills picked: {text}"), "skills": picked }), "done").await?;
        }
    } else if transcript.first().is_some_and(|m| m["role"] == "system") {
        // Resume: refresh the instructions (budget left, tools may have changed).
        transcript[0]["content"] = system.into();
    }

    let run = Arc::new(RunCtx {
        run_id: run_id.to_string(),
        cancel: CancellationToken::new(),
        pause: CancellationToken::new(),
        settings: settings.clone(),
        budgets,
        usage: Mutex::new(usage),
        child_usage: Mutex::new(Usage::default()),
        tools,
        mode,
        client,
        shared_prompt,
        env_label: exec::meta(st, &env_id).await.map(|m| m.label).unwrap_or_default(),
        children: tokio::sync::Semaphore::new(settings.max_parallel.clamp(1, MAX_PARALLEL_HARD) as usize),
        conversation_id: None,
    });
    st.tasks.lock().unwrap().insert(run_id.to_string(), run.clone());
    st.db.set_run_state(run_id, "running", None, None).await?;
    emit(app, serde_json::json!({ "run_id": run_id, "type": "state", "state": "running" }));

    let (st2, app2) = (st.clone(), app.clone());
    tokio::spawn(async move {
        let host = AppToolHost { st: st2.clone(), app: app2.clone(), conversation_id: None, run_id: Some(run.run_id.clone()), agent_id: None, agent_label: Some("Main agent".into()) };
        let hooks = StepHooks::new(st2.clone(), app2.clone(), run.run_id.clone(), None);
        let observer = Observer { st: st2.clone(), app: app2.clone(), run: run.clone(), agent_id: None, n_ctx };
        let gate = st2.gate();
        let ctx = TurnCtx {
            client: &run.client,
            tools: &run.tools,
            params: &run.settings.sampling,
            host: &host,
            hooks: &hooks,
            scope: Some(&run.run_id),
            cancel: &run.cancel,
            mode: run.mode,
            gate: Some(&gate),
        };
        let mut usage = *run.usage.lock().unwrap();
        let end = run_task(&ctx, &mut transcript, &run.budgets, &mut usage, &run.pause, &observer).await;
        if planning && let TaskEnd::Completed { answer } = &end {
            let step = st2.db.add_step(&run.run_id, None, "plan", &serde_json::json!({ "text": answer }), "done").await;
            if let Ok(step) = step {
                emit(&app2, serde_json::json!({ "run_id": run.run_id, "type": "step", "step": step }));
            }
            // Leave the running set first, so an Approve click right after the
            // state change can launch at once.
            st2.tasks.lock().unwrap().remove(&run.run_id);
            if let Err(e) = st2.db.set_run_state(&run.run_id, "awaiting_plan_approval", Some(answer), None).await {
                tracing::error!("could not save the plan of task {}: {e}", run.run_id);
            }
            emit(&app2, serde_json::json!({ "run_id": run.run_id, "type": "state", "state": "awaiting_plan_approval", "plan": answer }));
            return;
        }
        let (answer, error) = match &end {
            TaskEnd::Completed { answer } => (Some(answer.clone()), None),
            TaskEnd::BudgetExhausted(which) => (None, Some(format!("the {which} budget ran out"))),
            TaskEnd::Failed(e) => (None, Some(chat::explain_generation_error(&st2, e).await)),
            _ => (None, None),
        };
        if let Err(e) = st2.db.set_run_state(&run.run_id, end.state(), answer.as_deref(), error.as_deref()).await {
            // The runner is gone either way; say so instead of leaving "running".
            tracing::error!("could not save the end state of task {}: {e}", run.run_id);
            emit(&app2, serde_json::json!({ "run_id": run.run_id, "type": "error", "error": format!("the task ended ({}) but its state could not be saved: {e}", end.state()) }));
        }
        st2.tasks.lock().unwrap().remove(&run.run_id);
        emit(&app2, serde_json::json!({ "run_id": run.run_id, "type": "state", "state": end.state(), "error": error, "answer": answer }));
    });
    Ok(())
}

fn plan_section(objective: &str, budgets: &Budgets, usage: &Usage) -> String {
    let left = budgets.remaining(usage);
    format!(
        "## Task (planning)\nMake a plan for this objective: {objective}\nLook around with the read-only tools, then reply with a numbered step-by-step plan (what to change, where, how to check it) and stop. Do not change anything: the user approves the plan before the task runs.\nBudget left: {} model turns, {} tool calls, about {} minutes (planning counts toward it).\n\n{}",
        left.max_turns,
        left.max_tool_calls,
        (left.max_wall_secs / 60).max(1),
        instructions::PLAN_RULES
    )
}

/// Approve a task's plan (optionally edited) and run it with full tools. The
/// planning transcript is dropped; its usage still counts toward the budget.
pub async fn approve_plan(st: &SharedState, app: &AppHandle, run_id: &str, plan: Option<String>) -> AppResult<()> {
    let row = st.db.get_run(run_id).await?;
    refuse_chat_run(&row, "approved")?;
    if row.state != "awaiting_plan_approval" {
        return Err(AppError::InvalidInput(format!("this task is {}, not waiting for a plan approval", row.state)));
    }
    let plan = plan.filter(|p| !p.trim().is_empty()).or(row.final_answer.clone()).ok_or_else(|| AppError::InvalidInput("there is no plan to approve".into()))?;
    let mut settings: TaskSettings = serde_json::from_str(&row.settings).unwrap_or_default();
    settings.approved_plan = Some(plan.trim().to_string());
    st.db.set_run_plan_approved(run_id, &serde_json::to_value(&settings)?).await?;
    let step = st.db.add_step(run_id, None, "note", &serde_json::json!({ "text": "Plan approved; running it now." }), "done").await?;
    emit(app, serde_json::json!({ "run_id": run_id, "type": "step", "step": step }));
    launch(st, app, run_id).await
}

pub fn pause(st: &SharedState, run_id: &str) -> AppResult<()> {
    let r = st.tasks.lock().unwrap().get(run_id).cloned().ok_or_else(|| AppError::InvalidInput("the task is not running".into()))?;
    r.pause.cancel();
    Ok(())
}

/// Resume a paused, interrupted or budget-exhausted task. `budgets` (if
/// given) replaces the old ones, e.g. to give it more turns.
pub async fn resume(st: &SharedState, app: &AppHandle, run_id: &str, budgets: Option<Budgets>) -> AppResult<()> {
    let row = st.db.get_run(run_id).await?;
    refuse_chat_run(&row, "resumed")?;
    if !matches!(row.state.as_str(), "paused" | "interrupted" | "budget_exhausted" | "failed") {
        return Err(AppError::InvalidInput(format!("a {} task cannot be resumed", row.state)));
    }
    if let Some(b) = budgets {
        st.db.set_run_budgets(run_id, &serde_json::to_value(b)?).await?;
    }
    launch(st, app, run_id).await
}

pub async fn cancel(st: &SharedState, app: &AppHandle, run_id: &str) -> AppResult<()> {
    let running = st.tasks.lock().unwrap().get(run_id).cloned();
    match running {
        Some(r) => r.cancel.cancel(),
        None => {
            let row = st.db.get_run(run_id).await?;
            if persistence::agents::FINAL_STATES.contains(&row.state.as_str()) {
                return Err(AppError::InvalidInput("the task has already finished".into()));
            }
            st.db.set_run_state(run_id, "cancelled", None, None).await?;
            emit(app, serde_json::json!({ "run_id": run_id, "type": "state", "state": "cancelled" }));
        }
    }
    Ok(())
}

pub fn stop_subagent(st: &SharedState, id: &str) -> AppResult<()> {
    let t = st.subagent_stops.lock().unwrap().get(id).cloned().ok_or_else(|| AppError::InvalidInput("that subagent is not running".into()))?;
    t.cancel();
    Ok(())
}

/// A subagent's git worktree, as reported when its task (or chat) is deleted.
#[derive(Debug, Clone, Serialize)]
pub struct WorktreeReport {
    pub subagent_id: String,
    pub label: String,
    pub path: String,
    pub branch: String,
    /// Why it was kept (None when removed).
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DeleteResult {
    /// Worktrees (and their las/<id> branches) that were removed.
    pub removed_worktrees: Vec<WorktreeReport>,
    /// Worktrees left on disk because they have changes or unmerged commits.
    pub kept_worktrees: Vec<WorktreeReport>,
}

/// The hidden run that holds a chat reply's subagents (not a task).
fn is_chat_run(row: &persistence::AgentRunRow) -> bool {
    serde_json::from_str::<serde_json::Value>(&row.settings).ok().is_some_and(|v| v.get("chat_conversation_id").is_some_and(|c| !c.is_null()))
}

/// Task commands work on tasks only; a chat's run is managed by its chat.
fn refuse_chat_run(row: &persistence::AgentRunRow, what: &str) -> AppResult<()> {
    if is_chat_run(row) {
        return Err(AppError::InvalidInput(format!("this run belongs to a chat, so it cannot be {what} as a task; use the chat instead")));
    }
    Ok(())
}

pub async fn delete(st: &SharedState, run_id: &str) -> AppResult<DeleteResult> {
    let row = st.db.get_run(run_id).await?;
    if is_chat_run(&row) {
        return Err(AppError::InvalidInput("this run belongs to a chat; delete the chat instead".into()));
    }
    if st.tasks.lock().unwrap().contains_key(run_id) || st.tasks_starting.lock().unwrap().contains(run_id) {
        return Err(AppError::Conflict("stop the task before deleting it".into()));
    }
    let result = cleanup_worktrees(st, run_id).await?;
    st.db.delete_run(run_id).await?;
    Ok(result)
}

/// Remove the run's subagent worktrees that are clean and merged (or have
/// no commits); keep and report the rest. Never forced.
async fn cleanup_worktrees(st: &SharedState, run_id: &str) -> AppResult<DeleteResult> {
    let mut out = DeleteResult::default();
    for sub in st.db.list_subagents(run_id).await? {
        let settings: serde_json::Value = serde_json::from_str(&sub.settings).unwrap_or_default();
        let Ok(wt) = serde_json::from_value::<execution::git::Worktree>(settings["worktree"].clone()) else { continue };
        let outcome = execution::git::cleanup_worktree(&wt).await;
        let mut r = WorktreeReport { subagent_id: sub.id.clone(), label: sub.label.clone(), path: wt.path.clone(), branch: wt.branch.clone(), reason: None };
        match outcome {
            execution::git::Cleanup::Removed => out.removed_worktrees.push(r),
            execution::git::Cleanup::Kept { reason } => {
                r.reason = Some(reason);
                out.kept_worktrees.push(r);
            }
        }
    }
    Ok(out)
}

/// Conversation deleted: stop its reply's subagents, clean up their
/// worktrees and delete the hidden chat run.
pub async fn delete_chat_run(st: &SharedState, conversation_id: &str) -> AppResult<DeleteResult> {
    let Some(run_id) = st.db.find_chat_run(conversation_id).await? else { return Ok(DeleteResult::default()) };
    let running = st.tasks.lock().unwrap().get(&run_id).cloned();
    if let Some(r) = running {
        r.cancel.cancel();
    }
    // The reply ends once its children have stopped (their commands are
    // killed). Rows of a child that still runs are never deleted under it:
    // after 5 s each one still running is stopped directly and waited for.
    let mut waited = 0u32;
    loop {
        let children: Vec<String> = st.db.list_subagents(&run_id).await?.into_iter().map(|s| s.id).collect();
        let stops: Vec<CancellationToken> = {
            let stops = st.subagent_stops.lock().unwrap();
            children.iter().filter_map(|id| stops.get(id).cloned()).collect()
        };
        let reply_running = st.tasks.lock().unwrap().contains_key(&run_id);
        if stops.is_empty() && (!reply_running || waited >= 50) {
            // No child runs any more (queued ones end at once when cancelled).
            st.tasks.lock().unwrap().remove(&run_id);
            break;
        }
        if waited == 50 {
            for t in &stops {
                t.cancel();
            }
        }
        if waited >= 600 {
            return Err(AppError::Conflict("the chat's subagents are still stopping; delete the chat again in a moment".into()));
        }
        waited += 1;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let result = cleanup_worktrees(st, &run_id).await?;
    st.db.delete_run(&run_id).await?;
    Ok(result)
}

// ---------- chat runs ----------

/// May this chat offer spawn_subagents? Only in Agent mode (never Plan or
/// plain chat) with "Allow subagents" on.
pub fn chat_offers_subagents(s: &chat::ChatSettings) -> bool {
    s.subagents && s.effective_mode() == "agent"
}

/// The chat's per-reply limits become the budget its subagents share.
pub fn chat_budgets(limits: &agent_runtime::Limits) -> Budgets {
    let d = Budgets::default();
    Budgets { max_turns: d.max_turns, max_tool_calls: limits.max_tool_calls, max_wall_secs: limits.max_wall_secs, max_tokens: d.max_tokens }
}

/// What a chat reply needs to start its run context.
pub struct ChatRunInit {
    pub conversation_id: String,
    pub title: String,
    pub model_id: String,
    pub settings: chat::ChatSettings,
    /// The reply's tools (the spawn tool is re-pointed at the run).
    pub tools: Vec<ExposedTool>,
    pub mode: ToolMode,
    pub client: LlamaClient,
    /// Instruction base for the children (no tools section).
    pub shared_prompt: String,
    /// The reply's stop token: stopping the reply stops every child.
    pub cancel: CancellationToken,
}

/// Start a chat reply's run context, recorded in the conversation's hidden
/// agent_runs row ("Chat: <title>") so subagent rows and steps link to it.
pub async fn begin_chat_run(st: &SharedState, app: &AppHandle, init: ChatRunInit) -> AppResult<Arc<RunCtx>> {
    let s = &init.settings;
    let budgets = chat_budgets(&s.limits);
    let settings = TaskSettings {
        environment_id: s.environment_id.clone(),
        builtin_tools: s.builtin_tools,
        mcp_tools: true,
        tool_servers: s.tool_servers.clone(),
        skills_mode: if s.skills { "auto".into() } else { "off".into() },
        skills: vec![],
        subagents: true,
        max_parallel: s.max_subagents.clamp(1, MAX_PARALLEL_HARD),
        tool_mode: s.tool_mode.clone(),
        sampling: s.sampling.clone(),
        plan_first: false,
        approved_plan: None,
        thinking: s.thinking,
        reasoning_effort: s.reasoning_effort.clone(),
    };
    let title = format!("Chat: {}", init.title);
    let run_id = st.db.ensure_chat_run(&init.conversation_id, &title, Some(&init.model_id), &serde_json::to_value(&settings)?, &serde_json::to_value(budgets)?).await?;
    let tools = init.tools.into_iter().map(|t| if t.server_id.starts_with(AGENT_PREFIX) && t.tool_name == "spawn_subagents" { spawn_tool(&run_id) } else { t }).collect();
    let env_id = settings.environment_id.clone().unwrap_or_else(|| exec::LOCAL_ID.to_string());
    let run = Arc::new(RunCtx {
        run_id: run_id.clone(),
        cancel: init.cancel,
        pause: CancellationToken::new(),
        budgets,
        usage: Mutex::new(Usage::default()),
        child_usage: Mutex::new(Usage::default()),
        tools,
        mode: init.mode,
        client: init.client,
        shared_prompt: init.shared_prompt,
        env_label: exec::meta(st, &env_id).await.map(|m| m.label).unwrap_or_default(),
        children: tokio::sync::Semaphore::new(settings.max_parallel as usize),
        settings,
        conversation_id: Some(init.conversation_id),
    });
    st.tasks.lock().unwrap().insert(run_id.clone(), run.clone());
    if let Err(e) = st.db.set_run_state(&run_id, "running", None, None).await {
        st.tasks.lock().unwrap().remove(&run_id);
        return Err(e);
    }
    emit(app, serde_json::json!({ "run_id": run_id, "type": "state", "state": "running" }));
    Ok(run)
}

/// The chat reply ended (its children have ended too).
pub async fn end_chat_run(st: &SharedState, app: &AppHandle, run: &RunCtx, stop_reason: &str, error: Option<&str>) {
    st.tasks.lock().unwrap().remove(&run.run_id);
    let state = if error.is_some() { "failed" } else if stop_reason == "cancelled" { "cancelled" } else { "completed" };
    let u = { let mut u = *run.usage.lock().unwrap(); u.add(&run.child_usage.lock().unwrap()); u };
    let _ = st.db.save_run_progress(&run.run_id, &serde_json::json!([]), &serde_json::to_value(u).unwrap_or_default()).await;
    if let Err(e) = st.db.set_run_state(&run.run_id, state, None, error).await {
        tracing::error!("could not save the end state of chat run {}: {e}", run.run_id);
    }
    emit(app, serde_json::json!({ "run_id": run.run_id, "type": "state", "state": state, "error": error }));
}

impl RunCtx {
    /// A chat reply's own turn or tool call, so subagent budgets are carved
    /// out of what the reply has left.
    pub fn note_chat_usage(&self, turns: u32, tool_calls: u32, tokens: u64, wall_ms: u64) {
        let mut u = self.usage.lock().unwrap();
        u.turns += turns;
        u.tool_calls += tool_calls;
        u.tokens += tokens;
        u.wall_ms = u.wall_ms.max(wall_ms);
    }
}

pub async fn list(st: &SharedState) -> AppResult<Vec<AgentRunSummary>> {
    st.db.list_runs(200).await
}

pub async fn detail(st: &SharedState, run_id: &str) -> AppResult<TaskDetail> {
    let row = st.db.get_run(run_id).await?;
    let mut run = serde_json::to_value(&row)?;
    if let Some(o) = run.as_object_mut() {
        o.remove("transcript");
    }
    Ok(TaskDetail {
        run,
        steps: st.db.list_steps(run_id).await?,
        subagents: st.db.list_subagents(run_id).await?,
        running: st.tasks.lock().unwrap().contains_key(run_id),
    })
}

// ---------- subagents ----------

#[derive(Debug, Deserialize)]
struct SpawnArgs {
    subagents: Vec<SpawnOne>,
}

#[derive(Debug, Deserialize)]
struct SpawnOne {
    label: String,
    objective: String,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    max_turns: Option<u32>,
    #[serde(default)]
    max_tool_calls: Option<u32>,
    /// "none" (default) | "worktree"
    #[serde(default)]
    isolation: Option<String>,
}

/// A subagent that works in its own git worktree.
struct Isolation {
    wt: execution::git::Worktree,
    env_id: String,
    /// The worktree environment's prompt paragraph.
    section: String,
    /// Where the first allowed folder is inside the repo ("" = the repo top).
    rel: String,
}

/// Can this run give children worktrees? This PC with built-in tools, and
/// the first allowed folder inside a git repo. Returns (repo top, the first
/// folder relative to it).
async fn worktree_repo(st: &SharedState, run: &RunCtx) -> AppResult<(std::path::PathBuf, String)> {
    let env = run.settings.environment_id.as_deref().unwrap_or(exec::LOCAL_ID);
    if env != exec::LOCAL_ID {
        return Err(AppError::InvalidInput(format!("worktree isolation only works on This PC; this run's tools run on {}", run.env_label)));
    }
    let local_server = format!("{}{}", agent_runtime::BUILTIN_PREFIX, exec::LOCAL_ID);
    if !run.tools.iter().any(|t| t.server_id == local_server) {
        return Err(AppError::InvalidInput("worktree isolation needs the built-in shell and file tools on This PC, which this run does not have".into()));
    }
    let roots = exec::local_roots(st).await;
    let first = roots.first().ok_or_else(|| AppError::InvalidInput("This PC has no allowed folders; add one in Execution Environments".into()))?;
    let first = std::path::PathBuf::from(first);
    let repo = execution::git::repo_root(&first).await?;
    // The worktrees go inside the repository's top folder, so it must itself
    // be a folder the agent may use (not a parent of the allowed one).
    if let Err(e) = exec::check_local_path(st, &repo).await {
        return Err(AppError::PermissionDenied(format!(
            "the git repository that holds {} starts at {}, which is not inside the allowed folders ({e}); add the repository's top folder as an allowed folder to use worktrees",
            first.display(),
            repo.display()
        )));
    }
    let rel =std::fs::canonicalize(&first)
        .ok()
        .zip(std::fs::canonicalize(&repo).ok())
        .and_then(|(f, r)| f.strip_prefix(&r).ok().map(|p| p.to_string_lossy().into_owned()))
        .unwrap_or_default();
    Ok((repo, rel))
}

/// One child of a spawn call, saved before its worktree is made.
struct Child {
    id: String,
    label: String,
    objective: String,
    tools: Vec<ExposedTool>,
    budgets: Budgets,
    iso: Option<Isolation>,
}

/// Make a recorded child's worktree and its tools there.
async fn make_isolated(st: &SharedState, iso: &mut Isolation, tools: &[ExposedTool]) -> AppResult<Vec<ExposedTool>> {
    execution::git::create_worktree(&iso.wt).await?;
    let (_, section) = exec::exposed(st, &iso.env_id).await?;
    iso.section = section;
    worktree_tools(st, iso, tools.to_vec()).await
}

/// Undo a spawn that failed part-way: worktrees (and their las/ branches)
/// that are still empty are removed, the ad-hoc environments end, and the
/// children's rows say why they never started (so nothing is left behind
/// that the task's or chat's delete would not find).
async fn roll_back(st: &SharedState, app: &AppHandle, run_id: &str, made: &[Child], why: &str) {
    for c in made {
        if let Some(i) = &c.iso {
            exec::unregister_worktree_env(st, &i.env_id).await;
            if let execution::git::Cleanup::Kept { reason } = execution::git::cleanup_worktree(&i.wt).await {
                tracing::warn!("worktree {} was kept after a failed start: {reason}", i.wt.path);
            }
        }
        let _ = st.db.set_subagent_state(&c.id, "failed", None, Some(&format!("not started: {why}"))).await;
        emit(app, serde_json::json!({ "run_id": run_id, "agent_id": c.id, "type": "subagent_state", "state": "failed" }));
    }
}

async fn not_started(st: &SharedState, app: &AppHandle, run_id: &str, made: &[Child], why: &str, isolated: bool) -> AppResult<CallResult> {
    roll_back(st, app, run_id, made, why).await;
    let hint = if isolated { " You can call spawn_subagents again without \"isolation\" (the subagents then share your folders)." } else { "" };
    Ok(CallResult { text: format!("Not started: no subagent was started because {why}.{hint}"), is_error: true })
}

/// The child's tools with This PC's built-ins swapped for the same tools in
/// its worktree environment.
async fn worktree_tools(st: &SharedState, iso: &Isolation, tools: Vec<ExposedTool>) -> AppResult<Vec<ExposedTool>> {
    let (wt_tools, _) = exec::exposed(st, &iso.env_id).await?;
    let local_server = format!("{}{}", agent_runtime::BUILTIN_PREFIX, exec::LOCAL_ID);
    Ok(tools
        .into_iter()
        .filter_map(|t| if t.server_id == local_server { wt_tools.iter().find(|w| w.tool_name == t.tool_name).cloned() } else { Some(t) })
        .collect())
}

fn worktree_section(iso: &Isolation) -> String {
    let start = if iso.rel.is_empty() { iso.wt.path.clone() } else { format!("{}\\{}", iso.wt.path, iso.rel) };
    format!(
        "## Your git worktree\nYou work in your own git worktree, {path}, on branch {branch}. It is the only folder you may use: the allowed folders named above do not apply to you, so other subagents' edits never collide with yours. The folder the main agent works in corresponds to {start}.\n\
When your edits are done and checked, commit them on your branch with the shell tool (git add -A, then git commit -m \"...\") so the main agent can review and merge them. Do not merge into or touch the main working tree.\n\n{section}",
        path = iso.wt.path,
        branch = iso.wt.branch,
        section = iso.section
    )
}

/// What the parent reads about an isolated child's worktree.
fn worktree_note(wt: &execution::git::Worktree) -> String {
    format!(
        "Worktree: branch {branch} in {path}. Its changes are only there. Review them (for example git -C \"{path}\" log --oneline {base_short}..HEAD and git -C \"{path}\" status), then merge into the main tree with the shell tool, for example git -C \"{repo}\" merge {branch}.",
        branch = wt.branch,
        path = wt.path,
        repo = wt.repo,
        base_short = wt.base.chars().take(10).collect::<String>()
    )
}

/// The run a spawn call belongs to: the task's, or the chat reply's.
fn run_of(host: &AppToolHost) -> Option<Arc<RunCtx>> {
    let tasks = host.st.tasks.lock().unwrap();
    match (&host.run_id, &host.conversation_id) {
        (Some(r), _) => tasks.get(r).cloned(),
        (None, Some(c)) => tasks.values().find(|r| r.conversation_id.as_deref() == Some(c.as_str())).cloned(),
        _ => None,
    }
}

/// The `spawn_subagents` tool: run the children (up to the task's
/// concurrency cap at once), wait for all, return their answers.
pub async fn spawn_subagents(host: &AppToolHost, call: &ToolCall, args: serde_json::Value, timeout: std::time::Duration, cancel: &CancellationToken) -> AppResult<CallResult> {
    let st = &host.st;
    if host.agent_id.is_some() {
        return Ok(CallResult { text: "Subagents cannot start other subagents.".into(), is_error: true });
    }
    let Some(run) = run_of(host) else {
        let text = if host.run_id.is_some() {
            "The task is no longer running."
        } else {
            "Subagents are not allowed in this chat. They can be turned on in Chat settings (Allow subagents, Agent mode only)."
        };
        return Ok(CallResult { text: text.into(), is_error: true });
    };
    let run_id = run.run_id.clone();
    let a: SpawnArgs = serde_json::from_value(args).map_err(|e| AppError::InvalidInput(e.to_string()))?;
    if let Some(bad) = a.subagents.iter().find_map(|s| s.isolation.as_deref().filter(|i| !matches!(*i, "none" | "worktree"))) {
        return Ok(CallResult { text: format!("Not started: isolation must be \"none\" or \"worktree\", not {bad:?}."), is_error: true });
    }
    // What the parent may hand out: its tools minus the app's agent tools.
    let allowed: Vec<ExposedTool> = run.tools.iter().filter(|t| !t.server_id.starts_with(AGENT_PREFIX)).cloned().collect();
    let parent_used = {
        let mut u = *run.usage.lock().unwrap();
        u.add(&run.child_usage.lock().unwrap());
        u
    };
    // Each child gets an equal part of what is left (a third at most), with
    // no floors, so the children together never exceed the parent's limits.
    let n = a.subagents.len();
    let fraction = 1.0 / (n.max(3) as f64);
    let left = run.budgets.remaining(&parent_used);
    if let Some(which) = run.budgets.share(&parent_used, fraction, None).too_small() {
        return Ok(CallResult {
            text: format!(
                "Not started: too little of your budget is left to give {n} subagent(s) a useful share (not enough {which}; left in all: {} model turns, {} tool calls, {} s, {} tokens). Do the rest yourself, or start fewer subagents.",
                left.max_turns, left.max_tool_calls, left.max_wall_secs, left.max_tokens
            ),
            is_error: true,
        });
    }
    let limit = timeout.min(std::time::Duration::from_secs(left.max_wall_secs));
    let mut picked = vec![];
    for s in &a.subagents {
        let tools: Vec<ExposedTool> = match &s.tools {
            None => allowed.clone(),
            Some(names) => {
                let unknown: Vec<&String> = names.iter().filter(|n| !allowed.iter().any(|t| &t.model_name == *n)).collect();
                if !unknown.is_empty() {
                    let ok: Vec<&str> = allowed.iter().map(|t| t.model_name.as_str()).collect();
                    return Ok(CallResult {
                        text: format!("Not started: a subagent can only get tools you have. Unknown or not allowed: {:?}. Allowed: {}.", unknown, ok.join(", ")),
                        is_error: true,
                    });
                }
                allowed.iter().filter(|t| names.contains(&t.model_name)).cloned().collect()
            }
        };
        picked.push(tools);
    }
    // Worktrees: the repository is checked first; if isolation is not
    // available, no child starts.
    let isolated = a.subagents.iter().filter(|s| s.isolation.as_deref() == Some("worktree")).count();
    let repo = if isolated > 0 {
        match worktree_repo(st, &run).await {
            Ok(r) => Some(r),
            Err(e) => return not_started(st, &host.app, &run_id, &[], &format!("worktree isolation is not available: {e}"), true).await,
        }
    } else {
        None
    };
    // Each child's row (with its planned worktree) is saved before the
    // worktree is made; any failure rolls back every child made so far.
    let mut made: Vec<Child> = vec![];
    for (s, tools) in a.subagents.iter().zip(picked) {
        let asked = Budgets { max_turns: s.max_turns.unwrap_or(u32::MAX), max_tool_calls: s.max_tool_calls.unwrap_or(u32::MAX), max_wall_secs: u64::MAX, max_tokens: u64::MAX };
        let budgets = run.budgets.share(&parent_used, fraction, Some(&asked));
        let mut iso = None;
        if s.isolation.as_deref() == Some("worktree")
            && let Some((repo, rel)) = &repo
        {
            let wid: String = app_core::new_id().replace('-', "").chars().take(12).collect();
            let wt = match execution::git::plan_worktree(repo, &wid).await {
                Ok(w) => w,
                Err(e) => return not_started(st, &host.app, &run_id, &made, &format!("worktree isolation is not available: {e}"), true).await,
            };
            let env_id = format!("{}{wid}", exec::WORKTREE_ENV_PREFIX);
            exec::register_worktree_env(st, &env_id, &wt.path, &wt.branch).await;
            iso = Some(Isolation { wt, env_id, section: String::new(), rel: rel.clone() });
        }
        let environment = match &iso {
            Some(i) => exec::meta(st, &i.env_id).await.map(|m| m.label).unwrap_or_else(|_| format!("{}, git worktree {}", run.env_label, i.wt.branch)),
            None => tools.iter().find(|t| t.server_id.starts_with(agent_runtime::BUILTIN_PREFIX)).map(|t| t.server_name.clone()).unwrap_or_else(|| run.env_label.clone()),
        };
        let settings = serde_json::json!({
            "tools": tools.iter().map(|t| t.model_name.clone()).collect::<Vec<_>>(),
            "environment": environment,
            "call_id": call.id,
            "conversation_id": run.conversation_id,
            "worktree": iso.as_ref().map(|i| &i.wt),
        });
        let budgets_json = serde_json::to_value(budgets)?;
        let id = match st.db.create_subagent(&run_id, s.label.trim(), s.objective.trim(), &budgets_json, &settings).await {
            Ok(id) => id,
            Err(e) => {
                if let Some(i) = &iso {
                    exec::unregister_worktree_env(st, &i.env_id).await;
                }
                return not_started(st, &host.app, &run_id, &made, &format!("the subagent could not be saved: {e}"), false).await;
            }
        };
        let mut child = Child { id, label: s.label.trim().to_string(), objective: s.objective.trim().to_string(), tools, budgets, iso };
        let ready = match child.iso.as_mut() {
            None => Ok(None),
            Some(i) => make_isolated(st, i, &child.tools).await.map(Some),
        };
        match ready {
            Ok(Some(t)) => child.tools = t,
            Ok(None) => {}
            Err(e) => {
                made.push(child);
                return not_started(st, &host.app, &run_id, &made, &format!("worktree isolation is not available: {e}"), true).await;
            }
        }
        made.push(child);
    }
    let step = st
        .db
        .add_step(
            &run_id,
            None,
            "subagents",
            &serde_json::json!({ "call_id": call.id, "children": made.iter().map(|c| serde_json::json!({ "id": c.id, "label": c.label, "objective": c.objective, "budgets": c.budgets, "worktree": c.iso.as_ref().map(|i| &i.wt) })).collect::<Vec<_>>() }),
            "done",
        )
        .await;
    let step = match step {
        Ok(s) => s,
        Err(e) => return not_started(st, &host.app, &run_id, &made, &format!("the step could not be saved: {e}"), false).await,
    };
    emit(&host.app, serde_json::json!({ "run_id": run_id, "type": "step", "step": step }));

    // The whole call ends within the parent's time left (and the tool's
    // timeout): then unfinished children are stopped and waited for.
    let group = cancel.child_token();
    let futs = made.into_iter().map(|c| {
        let run = run.clone();
        let cancel = group.child_token();
        async move {
            let r = run_child(host, &run, &c.id, &c.label, &c.objective, c.tools, c.budgets, c.iso.as_ref(), cancel).await;
            if let Some(i) = c.iso {
                exec::unregister_worktree_env(&host.st, &i.env_id).await;
            }
            r
        }
    });
    let all = futures::future::join_all(futs);
    tokio::pin!(all);
    let (results, timed_out) = match tokio::time::timeout(limit, &mut all).await {
        Ok(r) => (r, false),
        Err(_) => {
            group.cancel();
            ((&mut all).await, true)
        }
    };
    let all_failed = results.iter().all(|r| r.1 != "completed");
    let mut text = String::new();
    if timed_out {
        text.push_str(&format!("The subagents' time ran out ({} s, what you had left); unfinished ones were stopped.\n\n", limit.as_secs()));
    }
    for (label, state, body) in results {
        text.push_str(&format!("### Subagent \"{label}\" ({state})\n{}\n\n", body.trim()));
    }
    Ok(CallResult { text: text.trim_end().to_string(), is_error: all_failed })
}

#[allow(clippy::too_many_arguments)]
async fn run_child(
    host: &AppToolHost,
    run: &Arc<RunCtx>,
    id: &str,
    label: &str,
    objective: &str,
    tools: Vec<ExposedTool>,
    budgets: Budgets,
    iso: Option<&Isolation>,
    cancel: CancellationToken,
) -> (String, String, String) {
    let st = &host.st;
    // Queue for a slot (the task's concurrency cap).
    let _slot = tokio::select! {
        p = run.children.acquire() => p.ok(),
        _ = cancel.cancelled() => {
            let _ = st.db.set_subagent_state(id, "cancelled", None, None).await;
            emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": id, "type": "subagent_state", "state": "cancelled" }));
            let mut body = "Stopped before it started.".to_string();
            if let Some(i) = iso {
                body = format!("{body}
{}", worktree_note(&i.wt));
            }
            return (label.to_string(), "cancelled".into(), body);
        }
    };
    st.subagent_stops.lock().unwrap().insert(id.to_string(), cancel.clone());
    let _ = st.db.set_subagent_state(id, "running", None, None).await;
    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": id, "type": "subagent_state", "state": "running" }));

    let child_host = AppToolHost {
        st: st.clone(),
        app: host.app.clone(),
        // A chat's children keep the chat, so their approvals show there.
        conversation_id: run.conversation_id.clone(),
        run_id: Some(run.run_id.clone()),
        agent_id: Some(id.to_string()),
        agent_label: Some(format!("Subagent \"{label}\"")),
    };
    let hooks = StepHooks::new(st.clone(), host.app.clone(), run.run_id.clone(), Some(id.to_string()));
    let n_ctx = st.engine.status().props.as_ref().and_then(|p| p.n_ctx).unwrap_or(4096);
    let observer = Observer { st: st.clone(), app: host.app.clone(), run: run.clone(), agent_id: Some(id.to_string()), n_ctx };
    let mut system = run.shared_prompt.clone();
    system.push_str("\n\n");
    system.push_str(&agent_runtime::tools_section(&tools, &agent_runtime::Limits { max_tool_calls: budgets.max_tool_calls, max_wall_secs: budgets.max_wall_secs }));
    system.push_str("\n\n");
    system.push_str(&subagent_section(label, objective, &budgets));
    if let Some(i) = iso {
        system.push_str("

");
        system.push_str(&worktree_section(i));
    }
    let mut transcript = vec![serde_json::json!({ "role": "system", "content": system }), serde_json::json!({ "role": "user", "content": objective })];
    let gate = st.gate();
    let pause = run.pause.child_token();
    let ctx = TurnCtx {
        client: &run.client,
        tools: &tools,
        params: &run.settings.sampling,
        host: &child_host,
        hooks: &hooks,
        scope: Some(&run.run_id),
        cancel: &cancel,
        mode: run.mode,
        gate: Some(&gate),
    };
    let mut usage = Usage::default();
    let end = run_task(&ctx, &mut transcript, &budgets, &mut usage, &pause, &observer).await;
    st.subagent_stops.lock().unwrap().remove(id);
    run.child_usage.lock().unwrap().add(&usage);
    let (state, body, result, error) = match &end {
        TaskEnd::Completed { answer } => ("completed", answer.clone(), Some(answer.clone()), None),
        TaskEnd::BudgetExhausted(w) => ("budget_exhausted", format!("Stopped: its {w} budget ran out before it finished."), None, Some(format!("the {w} budget ran out"))),
        TaskEnd::Cancelled => ("cancelled", "Stopped by the user.".into(), None, None),
        TaskEnd::Paused => ("cancelled", "Stopped because the task was paused; start it again if still needed.".into(), None, Some("task paused".into())),
        TaskEnd::Failed(e) => {
            let why = chat::explain_generation_error(st, e).await;
            ("failed", format!("Failed: {why}"), None, Some(why))
        }
    };
    // An isolated child's result always names its branch and folder.
    let (body, result) = match iso {
        Some(i) => {
            let note = worktree_note(&i.wt);
            (format!("{}\n\n{note}", body.trim_end()), result.map(|r| format!("{}\n\n{note}", r.trim_end())))
        }
        None => (body, result),
    };
    if let Err(e) = st.db.set_subagent_state(id, state, result.as_deref(), error.as_deref()).await {
        tracing::error!("could not save the end state of subagent {id}: {e}");
        emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": id, "type": "error", "error": format!("the subagent ended ({state}) but its state could not be saved: {e}") }));
    }
    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": id, "type": "subagent_state", "state": state, "usage": usage }));
    (label.to_string(), state.to_string(), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> serde_json::Value {
        serde_json::json!({ "role": role, "content": content })
    }

    #[test]
    fn trimming_removes_old_outputs_then_shortens_new_ones() {
        let big = "x".repeat(30_000);
        let mut t = vec![msg("system", "sys"), msg("user", "objective"), msg("assistant", "a1"), msg("tool", &big), msg("assistant", "a2"), msg("tool", &big), msg("tool", &big), msg("tool", &big), msg("tool", &big)];
        // ~50k tokens of tool output against an 8k context at 70%.
        let (removed, shortened) = fit_transcript(&mut t, 8192 * 7 / 10);
        assert_eq!(removed, 1, "the older batch is removed first");
        assert_eq!(shortened, 4, "the newest batch is shortened, not dropped");
        assert!(approx_tokens(&t) <= 8192 * 7 / 10 + 100, "fits: {}", approx_tokens(&t));
        assert_eq!(t[0]["content"], "sys");
        assert_eq!(t[1]["content"], "objective");
        for m in &t[5..] {
            let c = m["content"].as_str().unwrap();
            assert!(c.contains("characters removed to fit the context") && c.starts_with('x') && c.ends_with('x'), "{}", &c[..80]);
        }
        // Small transcripts are left alone.
        let mut s = vec![msg("system", "sys"), msg("user", "hi"), msg("assistant", "a"), msg("tool", "small")];
        assert_eq!(fit_transcript(&mut s, 8192), (0, 0));
    }

    #[test]
    fn task_commands_refuse_a_chats_hidden_run() {
        let row = |settings: &str| -> persistence::AgentRunRow {
            serde_json::from_value(serde_json::json!({
                "id": "r", "objective": "o", "model_id": null, "env_id": null, "state": "paused", "budgets": "{}", "usage": "{}",
                "final_answer": null, "error": null, "created_at": "", "started_at": null, "ended_at": null, "title": "t", "settings": settings
            }))
            .unwrap()
        };
        let chat = row(r#"{"chat_conversation_id":"c1","subagents":true}"#);
        assert!(is_chat_run(&chat));
        assert!(refuse_chat_run(&chat, "resumed").unwrap_err().to_string().contains("belongs to a chat"));
        let task = row(r#"{"subagents":true}"#);
        assert!(!is_chat_run(&task) && refuse_chat_run(&task, "resumed").is_ok());
        assert!(!is_chat_run(&row(r#"{"chat_conversation_id":null}"#)));
        assert!(!is_chat_run(&row("not json")));
    }

    #[test]
    fn plan_first_settings_default_off_and_plan_section_is_read_only() {
        // Tasks saved before Plan mode have neither field.
        let old: TaskSettings = serde_json::from_str(r#"{"subagents":true}"#).unwrap();
        assert!(!old.plan_first && old.approved_plan.is_none());
        let s = TaskSettings { plan_first: true, approved_plan: Some("1. do it".into()), ..Default::default() };
        let back: TaskSettings = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        let text = plan_section("fix the bug", &Budgets::default(), &Usage::default());
        assert!(text.contains("Make a plan for this objective: fix the bug"));
        assert!(text.contains(instructions::PLAN_RULES));
    }
}

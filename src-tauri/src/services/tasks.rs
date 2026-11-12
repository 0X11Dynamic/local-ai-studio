//! Agent tasks in the app: start / pause / resume / cancel, the step log and
//! live events, and subagents (the `spawn_subagents` tool).
//!
//! A task runs `agent_runtime::task::run_task` with the root agent's tools.
//! Subagents run the same loop with a subset of the parent's tools, a budget
//! carved out of the parent's, and child cancellation tokens, so stopping
//! the task reaches every child (and their commands die with their Job
//! Objects / process groups).

use std::sync::{Arc, Mutex};

use agent_runtime::ask::{self, AskBook, Question};
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
    /// MCP server ids turned off for this task (the start form's "Task
    /// tools" picker), applied after `tool_servers`. A list of what is off,
    /// like a chat's `disabled_servers`, so servers added later are on.
    pub disabled_servers: Vec<String>,
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
            // Every connected server is on unless the picker turns it off.
            mcp_tools: true,
            tool_servers: vec![],
            disabled_servers: vec![],
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
    /// Every subagent this run started (waited for or in the background),
    /// for subagent_status / wait_subagents / stop_subagent.
    pub kids: Mutex<Vec<Kid>>,
    /// Why the main agent stopped a child (stop_subagent's reason), by id.
    pub stop_notes: Mutex<std::collections::HashMap<String, String>>,
    /// Questions agents asked the user (`ask_user`) that are still waiting.
    pub asks: AskBook,
    /// Messages the user sent that the main agent has not read yet.
    pub inbox: Mutex<Vec<Queued>>,
}

/// A message from the user waiting for the main agent's next turn.
#[derive(Debug, Clone)]
pub struct Queued {
    /// The timeline entry (kind `user`) shown for it.
    pub step_id: String,
    pub text: String,
}

/// One subagent of a run.
#[derive(Clone, Debug)]
pub struct Kid {
    pub id: String,
    pub label: String,
    pub budgets: Budgets,
    /// Stops this child (and its commands); the same token as the UI's Stop.
    pub cancel: CancellationToken,
    /// Set when the child has ended: (state, what the parent reads).
    pub done: tokio::sync::watch::Receiver<Option<(String, String)>>,
}

impl Kid {
    fn ended(&self) -> Option<(String, String)> {
        self.done.borrow().clone()
    }
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

/// The tools section for a task or subagent: the tool list from
/// `tools_section`, with this work's own limits (which may be "no limit").
fn task_tools_section(tools: &[ExposedTool], b: &Budgets) -> String {
    let s = agent_runtime::tools_section(tools, &agent_runtime::Limits { max_tool_calls: b.max_tool_calls, max_wall_secs: b.max_wall_secs });
    let list = s.split("\n\n## Limits\n").next().unwrap_or(&s).to_string();
    format!("{list}\n\n## Limits\nEach tool call times out after its server's limit (usually 60 seconds). Limits for this work: {}.", b.describe())
}

/// "Budget left: ..." for the prompts.
fn budget_line(left: &Budgets) -> String {
    if *left == Budgets::default() {
        "There is no limit on model turns, tool calls, time or tokens, but stop as soon as the objective is done.".into()
    } else {
        format!("Budget left: {}.", left.describe())
    }
}

fn task_section(objective: &str, budgets: &Budgets, usage: &Usage, subagents: bool) -> String {
    let left = budgets.remaining(usage);
    let mut s = format!(
        "## Task\nYou are running an agent task, not a chat. Objective: {objective}\n\
Work step by step toward the objective using your tools. Keep going on your own: the user is not watching each step. Use ask_user when a decision or missing information really blocks you; otherwise proceed with sensible defaults and say what you assumed. If something else blocks you, explain it in your final answer.\n\
Check your work (run the tests, read the file back) before saying it is done.\n\
Your final answer is a reply with no tool call: start with the result, then say what you changed and how you checked it.\n\
{}",
        budget_line(&left)
    );
    if subagents {
        s.push('\n');
        s.push_str(SUBAGENT_HINT);
    }
    s
}

/// How to use the subagent tools (tasks, and chats that allow subagents).
pub const SUBAGENT_HINT: &str = "You can delegate independent parts of the work with spawn_subagents. Each subagent gets its own objective and a share of your budget, works in parallel, and reports back as that tool's result. Only delegate work that is independent and worth it; do small things yourself. \
With \"wait\": false, spawn_subagents returns at once with each subagent's id while they keep working: check them with subagent_status, block until they finish with wait_subagents, and stop one that is stuck or no longer needed with stop_subagent (it returns what that subagent had so far). If you answer while subagents are still working, the app waits for them and hands you their results before you finish. \
For a subagent that edits files, set \"isolation\": \"worktree\": it then works in its own git worktree and branch (on This PC, when the first allowed folder is inside a git repository), so subagents never edit the same files at once. Its result names the branch and folder; review its changes and merge them yourself with the shell tool.";

fn subagent_section(label: &str, objective: &str, budgets: &Budgets, can_ask: bool) -> String {
    let ask = if can_ask {
        "Use ask_user when a decision or missing information really blocks you (the question shows with your name); otherwise proceed with sensible defaults."
    } else {
        "Do not ask questions; work with what you have."
    };
    format!(
        "## Subagent task\nYou are subagent \"{label}\", working for a main agent. Your objective: {objective}\n\
Do only this. You cannot start other subagents. {ask}\n\
When done, reply with no tool call: your result first, concise, with the facts the main agent needs.\n\
{}",
        if *budgets == Budgets::default() { "You have no fixed budget, but finish as soon as your objective is done.".to_string() } else { format!("Budget: {}.", budgets.describe()) }
    )
}

/// Names of the app's subagent tools (all `AGENT_PREFIX` tools).
pub const SUBAGENT_TOOLS: [&str; 4] = ["spawn_subagents", "subagent_status", "wait_subagents", "stop_subagent"];

/// Is this one of the subagent tools (spawn, status, wait, stop)?
pub fn is_subagent_tool(t: &ExposedTool) -> bool {
    t.server_id.starts_with(AGENT_PREFIX) && SUBAGENT_TOOLS.contains(&t.tool_name.as_str())
}

/// Everything offered where subagents are allowed: spawn, status, wait, stop.
pub fn subagent_tools(run_id: &str) -> Vec<ExposedTool> {
    SUBAGENT_TOOLS.iter().filter_map(|n| subagent_tool(run_id, n)).collect()
}

fn agent_tool(run_id: &str, name: &str, description: &str, input_schema: serde_json::Value, timeout_secs: u64) -> ExposedTool {
    ExposedTool {
        tool_id: format!("{AGENT_PREFIX}{run_id}/{name}"),
        model_name: name.into(),
        server_id: format!("{AGENT_PREFIX}{run_id}"),
        server_name: "Local AI Studio".into(),
        tool_name: name.into(),
        description: description.into(),
        input_schema,
        policy: Policy::Allow,
        timeout_secs,
    }
}

/// One subagent tool by name, pointed at `run_id`.
pub fn subagent_tool(run_id: &str, name: &str) -> Option<ExposedTool> {
    let ids = serde_json::json!({ "type": "array", "items": { "type": "string" }, "description": "Subagent ids (or labels) from spawn_subagents. Default: every subagent you started" });
    Some(match name {
        "spawn_subagents" => spawn_tool(run_id),
        "subagent_status" => agent_tool(
            run_id,
            name,
            "The state of your subagents (running, completed, cancelled, failed), what each has used, and the result or error of those that ended. Returns at once.",
            serde_json::json!({ "type": "object", "properties": { "ids": ids }, "additionalProperties": false }),
            30,
        ),
        "wait_subagents" => agent_tool(
            run_id,
            name,
            "Wait until the given subagents (default: all) have finished, or until timeout_secs, then return their results. Subagents still running at the timeout keep running.",
            serde_json::json!({ "type": "object", "properties": { "ids": ids, "timeout_secs": { "type": "integer", "minimum": 1, "description": "Longest wait in seconds (default and maximum: the time you have left, at most an hour)" } }, "additionalProperties": false }),
            3600,
        ),
        "stop_subagent" => agent_tool(
            run_id,
            name,
            "Stop one of your running subagents (its commands are stopped too) and return its final state and what it had so far. Use it for a subagent that is stuck or no longer needed.",
            serde_json::json!({ "type": "object", "properties": {
                "id": { "type": "string", "description": "The subagent's id or label" },
                "reason": { "type": "string", "description": "Why it is stopped (kept with the subagent)" }
            }, "required": ["id"], "additionalProperties": false }),
            120,
        ),
        _ => return None,
    })
}

/// The `ask_user` tool, pointed at `run_id`. Offered in agent tasks (planning
/// included) and to their subagents, never in chat.
pub fn ask_user_tool(run_id: &str) -> ExposedTool {
    agent_tool(
        run_id,
        "ask_user",
        "Ask the user a question and wait for their answer. Use it only when a decision or missing information really blocks you and you cannot choose a sensible default. Give 2 to 6 short options when the answer is a choice. Waiting does not use up your time limit.",
        serde_json::json!({ "type": "object", "properties": {
            "question": { "type": "string", "minLength": 1, "description": "One clear question" },
            "options": { "type": "array", "minItems": 2, "maxItems": 6, "items": { "type": "string", "minLength": 1, "maxLength": 120 }, "description": "Short choices to offer as buttons (2 to 6)" },
            "allow_text": { "type": "boolean", "description": "Also let the user type their own answer (default true). false needs options" }
        }, "required": ["question"], "additionalProperties": false }),
        24 * 3600,
    )
}

pub fn spawn_tool(run_id: &str) -> ExposedTool {
    agent_tool(
        run_id,
        "spawn_subagents",
        "Start subagents that work in parallel on independent parts of the task and report back. Each gets a label, its own objective, optionally a subset of your tools, and a share of your budget. By default waits and returns each subagent's final answer; with \"wait\": false returns their ids at once.",
        serde_json::json!({ "type": "object", "properties": {
            "subagents": { "type": "array", "minItems": 1, "maxItems": 6, "items": { "type": "object", "properties": {
                "label": { "type": "string", "description": "Short name, e.g. \"tests\"" },
                "objective": { "type": "string", "description": "What this subagent must do and report" },
                "tools": { "type": "array", "items": { "type": "string" }, "description": "Tool names it may use (leave out or [] for all of yours except the subagent tools)" },
                "max_turns": { "type": "integer", "minimum": 1 },
                "max_tool_calls": { "type": "integer", "minimum": 0 },
                "isolation": { "type": "string", "enum": ["none", "worktree"], "description": "worktree: work in its own git worktree and branch (use it for subagents that edit files). Default none" }
            }, "required": ["label", "objective"], "additionalProperties": false } },
            "wait": { "type": "boolean", "description": "true (default): wait for all of them and return their answers. false: return at once with their ids; then use subagent_status, wait_subagents or stop_subagent" }
        }, "required": ["subagents"], "additionalProperties": false }),
        4 * 3600,
    )
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

    async fn take_user_messages(&self) -> Vec<String> {
        // Messages are for the main agent; subagents never read them.
        if self.agent_id.is_some() {
            return vec![];
        }
        let queued = std::mem::take(&mut *self.run.inbox.lock().unwrap());
        for m in &queued {
            let data = serde_json::json!({ "text": m.text, "delivered": true });
            let _ = self.st.db.update_step(&m.step_id, &data, "done").await;
            emit(&self.app, serde_json::json!({ "run_id": self.run.run_id, "agent_id": null, "type": "step_done", "step_id": m.step_id, "data": data }));
        }
        queued.into_iter().map(|m| m.text).collect()
    }

    fn take_waited_ms(&self) -> u64 {
        self.run.asks.take_waited(&self.agent_id)
    }

    async fn wait_children(&self, remaining: std::time::Duration) -> Option<String> {
        if self.agent_id.is_some() {
            return None;
        }
        await_children(&self.run, remaining).await
    }
}

/// The main agent answered while its subagents still run: wait for them (at
/// most `remaining`), then return their reports as a message to the model.
/// None = none were running, or the time ran out (they are then stopped).
pub async fn await_children(run: &RunCtx, remaining: std::time::Duration) -> Option<String> {
    let live: Vec<Kid> = run.kids.lock().unwrap().iter().filter(|k| k.ended().is_none()).cloned().collect();
    if live.is_empty() {
        return None;
    }
    let results = wait_all_credited(run, &live, remaining).await?;
    let mut text = String::from("Your subagents finished (you answered while they were still working). Their results:\n\n");
    for (label, state, body) in results {
        text.push_str(&format!("### Subagent \"{label}\" ({state})\n{}\n\n", body.trim()));
    }
    text.push_str("Now give your final answer, including what they found.");
    Some(text)
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
        // Servers turned off in the start form's "Task tools" picker are left out.
        tools.extend(mcp::exposed_tools(st, &settings.tool_servers).await?.into_iter().filter(|t| !settings.disabled_servers.contains(&t.server_id)));
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
            tools.extend(subagent_tools(run_id));
        }
    }
    // The agent may ask the user when it is really blocked, planning included.
    tools.push(ask_user_tool(run_id));
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
        task_tools_section(&tools, &budgets.remaining(&usage)),
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
        kids: Mutex::new(vec![]),
        stop_notes: Mutex::new(Default::default()),
        asks: AskBook::default(),
        inbox: Mutex::new(vec![]),
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
        // Subagents started with "wait": false and still running end with the task.
        stop_children(&run).await;
        let late = std::mem::take(&mut *run.child_usage.lock().unwrap());
        if late != Usage::default() {
            // Children that ended after the main agent's last turn count too.
            usage.add(&late);
            let _ = st2.db.save_run_progress(&run.run_id, &serde_json::Value::Array(transcript.clone()), &serde_json::to_value(usage).unwrap_or_default()).await;
        }
        // Messages sent too late for this run (paused, cancelled, failed): keep
        // them in the transcript so a Resume reads them.
        let left = std::mem::take(&mut *run.inbox.lock().unwrap());
        if !left.is_empty() {
            let texts: Vec<String> = left.iter().map(|m| m.text.clone()).collect();
            agent_runtime::task::inject_user_messages(&mut transcript, &texts);
            let _ = st2.db.save_run_progress(&run.run_id, &serde_json::Value::Array(transcript.clone()), &serde_json::to_value(usage).unwrap_or_default()).await;
            for m in &left {
                let data = serde_json::json!({ "text": m.text, "delivered": false, "note": "The task ended before it read this; Resume will include it." });
                let _ = st2.db.update_step(&m.step_id, &data, "undelivered").await;
                emit(&app2, serde_json::json!({ "run_id": run.run_id, "agent_id": null, "type": "step_done", "step_id": m.step_id, "data": data }));
            }
        }
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
        "## Task (planning)\nMake a plan for this objective: {objective}\nLook around with the read-only tools, then reply with a numbered step-by-step plan (what to change, where, how to check it) and stop. You may ask the user clarifying questions with ask_user before writing the plan, but only when the answer would change it. Do not change anything: the user approves the plan before the task runs.\n{} (Planning counts toward the task's limits.)\n\n{}",
        budget_line(&left),
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

/// May a task in this state be resumed? A task waiting for the user's answer
/// is not: it continues when the question is answered.
pub fn resume_check(state: &str) -> AppResult<()> {
    match state {
        "paused" | "interrupted" | "budget_exhausted" | "failed" => Ok(()),
        "waiting_for_user" => Err(AppError::InvalidInput("this task is waiting for your answer; answer its question (or cancel the task) instead of resuming it".into())),
        "awaiting_plan_approval" => Err(AppError::InvalidInput("this task is waiting for a plan approval; approve the plan or cancel the task".into())),
        other => Err(AppError::InvalidInput(format!("a {other} task cannot be resumed"))),
    }
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
    resume_check(&row.state)?;
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
            st.db.close_waiting_questions(run_id, "cancelled").await?;
            emit(app, serde_json::json!({ "run_id": run_id, "type": "question_closed", "state": "cancelled" }));
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
/// Turns and tokens are unlimited (the 500-turn backstop still applies), and
/// so are tool calls and time unless the chat sets them.
pub fn chat_budgets(limits: &agent_runtime::Limits) -> Budgets {
    Budgets { max_turns: None, max_tool_calls: limits.max_tool_calls, max_wall_secs: limits.max_wall_secs, max_tokens: None }
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
        disabled_servers: s.disabled_servers.clone(),
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
    let tools = init.tools.into_iter().map(|t| if is_subagent_tool(&t) { subagent_tool(&run_id, &t.tool_name).unwrap_or(t) } else { t }).collect();
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
        kids: Mutex::new(vec![]),
        stop_notes: Mutex::new(Default::default()),
        asks: AskBook::default(),
        inbox: Mutex::new(vec![]),
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
    // Children started with "wait": false and still running end with the reply.
    stop_children(run).await;
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
    /// true (default): wait for all and return their answers; false: return
    /// their ids at once and let them work in the background.
    #[serde(default = "yes")]
    wait: bool,
}

fn yes() -> bool {
    true
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
    // Used so far, plus what children still running in the background were
    // given (their share stays theirs until they end).
    let parent_used = {
        let mut u = *run.usage.lock().unwrap();
        u.add(&run.child_usage.lock().unwrap());
        u.add(&reserved(&run));
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
                "Not started: too little of your budget is left to give {n} subagent(s) a useful share (not enough {which}; left in all: {}). Do the rest yourself, or start fewer subagents.",
                left.describe()
            ),
            is_error: true,
        });
    }
    let limit = timeout.min(std::time::Duration::from_secs(left.secs_or_max()));
    let mut picked = vec![];
    for s in &a.subagents {
        let tools: Vec<ExposedTool> = match &s.tools {
            // An empty list almost always means "the usual tools", not "none"
            // (a child with no tools can't do anything): same as leaving it out.
            None => allowed.clone(),
            Some(names) if names.is_empty() => allowed.clone(),
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
        let mut tools = tools;
        // A subagent may ask the user too (never a chat's subagents: the
        // chat has no place to answer).
        if run.conversation_id.is_none() && !tools.iter().any(|t| t.tool_name == "ask_user") {
            tools.push(ask_user_tool(&run_id));
        }
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
        let asked = Budgets { max_turns: s.max_turns, max_tool_calls: s.max_tool_calls, max_wall_secs: None, max_tokens: None };
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

    // Each child runs in its own task, so one can be stopped on its own and
    // the main agent can go on without waiting. Waited-for children stop
    // with this call; background ones when the run ends (or when stopped).
    let group = if a.wait { cancel.child_token() } else { run.cancel.child_token() };
    let started: Vec<Kid> = made.into_iter().map(|c| launch_child(host, &run, c, group.child_token())).collect();
    if !a.wait {
        let list: Vec<serde_json::Value> = started.iter().map(|k| serde_json::json!({ "id": k.id, "label": k.label, "state": "running", "budgets": k.budgets })).collect();
        let text = serde_json::json!({
            "started": list,
            "note": "They work in the background. Use subagent_status to check them, wait_subagents to wait for their results, and stop_subagent to stop one. If you answer first, the app waits for them and gives you their results."
        });
        return Ok(CallResult { text: serde_json::to_string_pretty(&text).unwrap_or_default(), is_error: false });
    }
    // The whole call ends within the parent's time left (and the tool's
    // timeout): then unfinished children are stopped and waited for.
    let (results, timed_out) = match wait_all_credited(&run, &started, limit).await {
        Some(r) => (r, false),
        None => {
            group.cancel();
            (wait_all(&started).await, true)
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

/// Start one recorded child in its own task and list it on the run.
fn launch_child(host: &AppToolHost, run: &Arc<RunCtx>, c: Child, cancel: CancellationToken) -> Kid {
    let (tx, rx) = tokio::sync::watch::channel(None);
    let kid = Kid { id: c.id.clone(), label: c.label.clone(), budgets: c.budgets, cancel: cancel.clone(), done: rx };
    run.kids.lock().unwrap().push(kid.clone());
    let h = AppToolHost {
        st: host.st.clone(),
        app: host.app.clone(),
        conversation_id: host.conversation_id.clone(),
        run_id: host.run_id.clone(),
        agent_id: host.agent_id.clone(),
        agent_label: host.agent_label.clone(),
    };
    let run = run.clone();
    tokio::spawn(async move {
        let (_, state, body) = run_child(&h, &run, &c.id, &c.label, &c.objective, c.tools, c.budgets, c.iso.as_ref(), cancel).await;
        if let Some(i) = &c.iso {
            exec::unregister_worktree_env(&h.st, &i.env_id).await;
        }
        tx.send_replace(Some((state, body)));
    });
    kid
}

/// Wait until a child has ended: (state, what the parent reads).
async fn wait_one(k: &Kid) -> (String, String) {
    let mut rx = k.done.clone();
    match rx.wait_for(|v| v.is_some()).await {
        Ok(v) => (*v).clone().unwrap_or_default(),
        Err(_) => ("failed".into(), "The subagent stopped unexpectedly.".into()),
    }
}

/// Wait for all of them: (label, state, body) each.
async fn wait_all(kids: &[Kid]) -> Vec<(String, String, String)> {
    futures::future::join_all(kids.iter().map(|k| async move {
        let (state, body) = wait_one(k).await;
        (k.label.clone(), state, body)
    }))
    .await
}

/// Like `wait_all`, but gives up after `limit` of working time: time the
/// children spend waiting for the user's answer does not count, and is
/// credited to the waiting parent's usage too. None = the limit passed.
async fn wait_all_credited(run: &RunCtx, kids: &[Kid], limit: std::time::Duration) -> Option<Vec<(String, String, String)>> {
    let started = std::time::Instant::now();
    let base = run.asks.children_waited_ms();
    let fut = wait_all(kids);
    tokio::pin!(fut);
    let result = loop {
        let waited = std::time::Duration::from_millis(run.asks.children_waited_ms().saturating_sub(base));
        let deadline = started + limit + waited;
        let now = std::time::Instant::now();
        if now >= deadline {
            break None;
        }
        tokio::select! {
            r = &mut fut => break Some(r),
            _ = tokio::time::sleep((deadline - now).min(std::time::Duration::from_secs(1))) => {}
        }
    };
    run.asks.credit(&None, run.asks.children_waited_ms().saturating_sub(base));
    result
}

/// What running background children were given (wall time is not added:
/// children run during the parent's own time).
fn reserved(run: &RunCtx) -> Usage {
    let mut u = Usage::default();
    for k in run.kids.lock().unwrap().iter().filter(|k| k.ended().is_none()) {
        u.turns = u.turns.saturating_add(k.budgets.max_turns.unwrap_or(0));
        u.tool_calls = u.tool_calls.saturating_add(k.budgets.max_tool_calls.unwrap_or(0));
        u.tokens = u.tokens.saturating_add(k.budgets.max_tokens.unwrap_or(0));
    }
    u
}

/// Stop the run's children that are still running and wait (up to 30 s)
/// for them to end. Returns whether any was running.
pub async fn stop_children(run: &RunCtx) -> bool {
    let live: Vec<Kid> = run.kids.lock().unwrap().iter().filter(|k| k.ended().is_none()).cloned().collect();
    if live.is_empty() {
        return false;
    }
    for k in &live {
        k.cancel.cancel();
    }
    if tokio::time::timeout(std::time::Duration::from_secs(30), wait_all(&live)).await.is_err() {
        tracing::warn!("run {}: subagents still stopping after 30 s", run.run_id);
    }
    true
}

/// The run whose main agent may manage subagents (never a subagent itself).
fn managing_run(host: &AppToolHost) -> Result<Arc<RunCtx>, CallResult> {
    if host.agent_id.is_some() {
        return Err(CallResult { text: "Only the main agent can manage subagents.".into(), is_error: true });
    }
    run_of(host).ok_or_else(|| CallResult {
        text: if host.run_id.is_some() { "The task is no longer running.".into() } else { "Subagents are not allowed in this chat.".into() },
        is_error: true,
    })
}

/// Find this run's children by id or label. Unknown or ambiguous keys are
/// refused (only the run's own children can be named).
fn resolve_kids(kids: &[Kid], keys: &[String]) -> Result<Vec<Kid>, String> {
    let known = || kids.iter().map(|k| format!("\"{}\" ({})", k.label, k.id)).collect::<Vec<_>>().join(", ");
    if kids.is_empty() {
        return Err("You have not started any subagents in this run.".into());
    }
    if keys.is_empty() {
        return Ok(kids.to_vec());
    }
    let mut out: Vec<Kid> = vec![];
    for key in keys {
        let key = key.trim();
        let found = match kids.iter().find(|k| k.id == key) {
            Some(k) => k.clone(),
            None => {
                let by_label: Vec<&Kid> = kids.iter().filter(|k| k.label.eq_ignore_ascii_case(key)).collect();
                let running: Vec<&&Kid> = by_label.iter().filter(|k| k.ended().is_none()).collect();
                match (by_label.len(), running.len()) {
                    (0, _) => return Err(format!("No subagent {key:?} in this run. Your subagents: {}.", known())),
                    (1, _) => by_label[0].clone(),
                    (_, 1) => (*running[0]).clone(),
                    _ => return Err(format!("{} subagents are labelled {key:?}; use the id. Your subagents: {}.", by_label.len(), known())),
                }
            }
        };
        if !out.iter().any(|k| k.id == found.id) {
            out.push(found);
        }
    }
    Ok(out)
}

fn keys_of(args: &serde_json::Value) -> Vec<String> {
    args["ids"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// One child as the parent reads it: state, usage, result or error.
async fn kid_view(st: &SharedState, k: &Kid) -> serde_json::Value {
    let row = st.db.get_subagent(&k.id).await.ok();
    let ended = k.ended();
    let state = match (&ended, &row) {
        (Some((s, _)), _) => s.clone(),
        (None, Some(r)) => r.state.clone(),
        (None, None) => "running".into(),
    };
    let usage: Usage = row.as_ref().and_then(|r| serde_json::from_str(&r.usage).ok()).unwrap_or_default();
    let mut v = serde_json::json!({ "id": k.id, "label": k.label, "state": state, "usage": usage, "budgets": k.budgets });
    if let Some((_, body)) = ended {
        v["result"] = body.into();
    }
    if let Some(e) = row.and_then(|r| r.error) {
        v["error"] = e.into();
    }
    v
}

/// The `subagent_status` tool: `{ "ids"?: [id or label] }`.
pub async fn subagent_status(host: &AppToolHost, args: serde_json::Value) -> AppResult<CallResult> {
    let run = match managing_run(host) {
        Ok(r) => r,
        Err(c) => return Ok(c),
    };
    let kids = run.kids.lock().unwrap().clone();
    let picked = match resolve_kids(&kids, &keys_of(&args)) {
        Ok(k) => k,
        Err(e) => return Ok(CallResult { text: e, is_error: true }),
    };
    let mut list = vec![];
    for k in &picked {
        list.push(kid_view(&host.st, k).await);
    }
    Ok(CallResult { text: serde_json::to_string_pretty(&serde_json::json!({ "subagents": list })).unwrap_or_default(), is_error: false })
}

/// The `wait_subagents` tool: `{ "ids"?: [id or label], "timeout_secs"?: n }`.
/// Waits at most the parent's time left (and the tool's timeout); children
/// still running then keep running.
pub async fn wait_subagents(host: &AppToolHost, args: serde_json::Value, timeout: std::time::Duration, cancel: &CancellationToken) -> AppResult<CallResult> {
    let run = match managing_run(host) {
        Ok(r) => r,
        Err(c) => return Ok(c),
    };
    let kids = run.kids.lock().unwrap().clone();
    let picked = match resolve_kids(&kids, &keys_of(&args)) {
        Ok(k) => k,
        Err(e) => return Ok(CallResult { text: e, is_error: true }),
    };
    let used = {
        let mut u = *run.usage.lock().unwrap();
        u.add(&run.child_usage.lock().unwrap());
        u
    };
    let mut limit = timeout.min(std::time::Duration::from_secs(run.budgets.remaining(&used).secs_or_max()));
    if let Some(t) = args["timeout_secs"].as_u64() {
        limit = limit.min(std::time::Duration::from_secs(t.max(1)));
    }
    let finished = tokio::select! {
        r = wait_all_credited(&run, &picked, limit) => r.is_some(),
        _ = cancel.cancelled() => false,
    };
    let mut list = vec![];
    for k in &picked {
        list.push(kid_view(&host.st, k).await);
    }
    let note = if finished { "All of them have ended.".to_string() } else { format!("Stopped waiting after {} s; those still running keep running.", limit.as_secs()) };
    Ok(CallResult { text: serde_json::to_string_pretty(&serde_json::json!({ "note": note, "subagents": list })).unwrap_or_default(), is_error: false })
}

/// The `stop_subagent` tool: `{ "id": id or label, "reason"?: text }`.
/// Stops that child (its commands die with the cancel tree) and returns its
/// final state and what it had so far.
pub async fn stop_subagent_tool(host: &AppToolHost, args: serde_json::Value) -> AppResult<CallResult> {
    let run = match managing_run(host) {
        Ok(r) => r,
        Err(c) => return Ok(c),
    };
    let Some(key) = args["id"].as_str().or(args["label"].as_str()).map(str::to_string).filter(|k| !k.trim().is_empty()) else {
        return Ok(CallResult { text: "Say which subagent to stop: \"id\" (its id or label).".into(), is_error: true });
    };
    let kids = run.kids.lock().unwrap().clone();
    let k = match resolve_kids(&kids, std::slice::from_ref(&key)) {
        Ok(mut k) => k.remove(0),
        Err(e) => return Ok(CallResult { text: e, is_error: true }),
    };
    if let Some((state, _)) = k.ended() {
        let mut v = kid_view(&host.st, &k).await;
        v["note"] = format!("It had already ended ({state}); nothing was stopped.").into();
        return Ok(CallResult { text: serde_json::to_string_pretty(&v).unwrap_or_default(), is_error: false });
    }
    let reason = args["reason"].as_str().map(str::trim).filter(|r| !r.is_empty()).unwrap_or("no reason given").to_string();
    run.stop_notes.lock().unwrap().insert(k.id.clone(), reason);
    k.cancel.cancel();
    let ended = tokio::time::timeout(std::time::Duration::from_secs(60), wait_one(&k)).await.is_ok();
    let mut v = kid_view(&host.st, &k).await;
    if !ended {
        v["note"] = "Asked to stop; it is still stopping (check again with subagent_status).".into();
    }
    Ok(CallResult { text: serde_json::to_string_pretty(&v).unwrap_or_default(), is_error: false })
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
    system.push_str(&task_tools_section(&tools, &budgets));
    system.push_str("\n\n");
    system.push_str(&subagent_section(label, objective, &budgets, tools.iter().any(|t| t.tool_name == "ask_user")));
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
    // Stopped by the main agent (stop_subagent), with its reason.
    let stopped_by_parent = run.stop_notes.lock().unwrap().remove(id);
    let (state, mut body, result, error) = match &end {
        TaskEnd::Completed { answer } => ("completed", answer.clone(), Some(answer.clone()), None),
        TaskEnd::BudgetExhausted(w) => ("budget_exhausted", format!("Stopped: its {w} budget ran out before it finished."), None, Some(format!("the {w} budget ran out"))),
        TaskEnd::Cancelled => match &stopped_by_parent {
            Some(why) => ("cancelled", format!("Stopped by the main agent ({why})."), None, Some(format!("stopped by the main agent: {why}"))),
            None => ("cancelled", "Stopped by the user.".into(), None, None),
        },
        TaskEnd::Paused => ("cancelled", "Stopped because the task was paused; start it again if still needed.".into(), None, Some("task paused".into())),
        TaskEnd::Failed(e) => {
            let why = chat::explain_generation_error(st, e).await;
            ("failed", format!("Failed: {why}"), None, Some(why))
        }
    };
    if !matches!(end, TaskEnd::Completed { .. })
        && let Some(last) = transcript.iter().skip(2).rev().find_map(|m| m["content"].as_str().filter(|c| m["role"] == "assistant" && !c.trim().is_empty()))
    {
        // What it had so far, so stopping it does not lose its work.
        body = format!("{body}\nIts last message before it stopped:\n{}", last.trim());
    }
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

// ---------- asking the user, and messages to a running task ----------

fn question_data(q: &Question) -> serde_json::Value {
    serde_json::json!({
        "question_id": q.id, "question": q.question, "options": q.options, "allow_text": q.allow_text,
        "call_id": q.call_id, "agent_label": q.agent_label, "answer": null
    })
}

/// The `ask_user` tool: park this agent until the user answers. The question
/// is saved as a timeline entry (kind `question`, state `waiting`) and the run
/// (or the subagent) goes to `waiting_for_user`; the answer arrives through
/// `answer_question` (or a message sent to the task) and is the tool result.
pub async fn ask_user(host: &AppToolHost, call: &ToolCall, args: serde_json::Value, cancel: &CancellationToken) -> AppResult<CallResult> {
    let fail = |text: &str| Ok(CallResult { text: text.into(), is_error: true });
    let Some(run) = run_of(host).filter(|r| r.conversation_id.is_none()) else {
        return fail("ask_user only works in an agent task. In a chat, ask in your reply.");
    };
    let (question, options, allow_text) = match ask::parse_args(&args) {
        Ok(x) => x,
        Err(e) => return fail(&format!("Not asked: {e}. Call ask_user again with valid arguments.")),
    };
    let st = &host.st;
    let q = Question {
        id: app_core::new_id(),
        agent_id: host.agent_id.clone(),
        agent_label: host.agent_label.clone().unwrap_or_else(|| "Main agent".into()),
        call_id: call.id.clone(),
        question,
        options,
        allow_text,
    };
    let step = st.db.add_step(&run.run_id, q.agent_id.as_deref(), "question", &question_data(&q), "waiting").await?;
    let rx = run.asks.ask(q.clone());
    match &q.agent_id {
        None => {
            st.db.set_run_state(&run.run_id, "waiting_for_user", None, None).await?;
            emit(&host.app, serde_json::json!({ "run_id": run.run_id, "type": "state", "state": "waiting_for_user" }));
        }
        Some(a) => {
            st.db.set_subagent_state(a, "waiting_for_user", None, None).await?;
            emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": a, "type": "subagent_state", "state": "waiting_for_user" }));
        }
    }
    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": q.agent_id, "type": "step", "step": step }));
    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": q.agent_id, "type": "question", "question_id": q.id }));
    let t0 = std::time::Instant::now();
    let waited = ask::wait(rx, cancel).await;
    run.asks.finish_wait(&q.agent_id, t0.elapsed().as_millis() as u64);
    // Back to work (the answer command already closed the timeline entry).
    match waited {
        ask::Waited::Answer(a) => {
            match &q.agent_id {
                None => {
                    let _ = st.db.set_run_state(&run.run_id, "running", None, None).await;
                    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "type": "state", "state": "running" }));
                }
                Some(id) => {
                    let _ = st.db.set_subagent_state(id, "running", None, None).await;
                    emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": id, "type": "subagent_state", "state": "running" }));
                }
            }
            Ok(CallResult { text: ask::answer_result(&a), is_error: false })
        }
        ask::Waited::Cancelled => {
            run.asks.forget(&q.id);
            let mut d = question_data(&q);
            d["cancelled"] = true.into();
            let _ = st.db.update_step(&step.id, &d, "cancelled").await;
            emit(&host.app, serde_json::json!({ "run_id": run.run_id, "agent_id": q.agent_id, "type": "question_closed", "question_id": q.id, "state": "cancelled" }));
            fail("Not answered: the work was stopped before the user answered.")
        }
    }
}

fn parse_data(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap_or_default()
}

/// The first user message of a run: the objective, with the approved plan.
fn brief_of(row: &persistence::AgentRunRow, settings: &TaskSettings) -> String {
    match &settings.approved_plan {
        Some(plan) => format!("{}\n\nApproved plan (follow it; if a step turns out to be wrong, adapt and say why):\n{plan}", row.objective),
        None => row.objective.clone(),
    }
}

/// Save `text` as the next user message of a stopped run's transcript (joined
/// to a trailing user message), so launching it reads the text first.
async fn add_to_transcript(st: &SharedState, row: &persistence::AgentRunRow, text: &str) -> AppResult<()> {
    let mut t: Vec<serde_json::Value> = serde_json::from_str(&st.db.get_run_transcript(&row.id).await?).unwrap_or_default();
    if t.is_empty() {
        // Nothing saved yet: start it (launch refreshes the system message).
        let settings: TaskSettings = serde_json::from_str(&row.settings).unwrap_or_default();
        t = vec![serde_json::json!({ "role": "system", "content": "" }), serde_json::json!({ "role": "user", "content": brief_of(row, &settings) })];
    }
    match t.last_mut() {
        Some(last) if last["role"] == "user" && last["content"].as_str().is_some() => {
            let c = last["content"].as_str().unwrap_or_default().to_string();
            last["content"] = format!("{c}\n\n{text}").into();
        }
        _ => t.push(serde_json::json!({ "role": "user", "content": text })),
    }
    let usage: serde_json::Value = serde_json::from_str(&row.usage).unwrap_or_else(|_| serde_json::json!({}));
    st.db.save_run_progress(&row.id, &serde_json::Value::Array(t), &usage).await
}

/// Deliver the user's answer to a waiting question. The asking agent continues
/// with it as the tool result. If the app was restarted since (no agent is
/// waiting any more), a main-agent question resumes the task with the answer.
pub async fn answer_question(st: &SharedState, app: &AppHandle, run_id: &str, question_id: &str, answer: &str) -> AppResult<()> {
    let row = st.db.get_run(run_id).await?;
    refuse_chat_run(&row, "answered")?;
    let steps = st.db.list_steps(run_id).await?;
    let step = steps
        .iter()
        .find(|s| s.kind == "question" && s.state == "waiting" && parse_data(&s.data)["question_id"] == question_id)
        .ok_or_else(|| AppError::NotFound("that question is no longer waiting for an answer".into()))?;
    let mut data = parse_data(&step.data);
    let live = st.tasks.lock().unwrap().get(run_id).cloned();
    let answer = match &live {
        Some(run) => run.asks.answer(question_id, answer).map(|_| answer.trim().to_string()).map_err(|e| match e {
            ask::AskError::Unknown => AppError::NotFound(e.to_string()),
            _ => AppError::InvalidInput(e.to_string()),
        })?,
        None => {
            // After a restart only the main agent's question is still open.
            if step.parent_agent_id.is_some() || row.state != "waiting_for_user" {
                return Err(AppError::NotFound("that question is no longer waiting for an answer".into()));
            }
            let options: Vec<String> = data["options"].as_array().into_iter().flatten().filter_map(|o| o.as_str().map(str::to_string)).collect();
            ask::check_answer(&options, data["allow_text"].as_bool().unwrap_or(true), answer).map_err(|e| AppError::InvalidInput(e.to_string()))?
        }
    };
    data["answer"] = answer.clone().into();
    data["answered_at"] = app_core::now_rfc3339().into();
    st.db.update_step(&step.id, &data, "done").await?;
    emit(app, serde_json::json!({ "run_id": run_id, "type": "question_closed", "question_id": question_id, "state": "done" }));
    if live.is_none() {
        let text = format!("Earlier you asked the user: \"{}\"\n{}\nContinue the task from there.", data["question"].as_str().unwrap_or_default(), ask::answer_result(&answer));
        add_to_transcript(st, &row, &text).await?;
        launch(st, app, run_id).await?;
    }
    Ok(())
}

/// States from which a message can restart the task ("Resume with this message").
pub const RESUME_WITH_MESSAGE: [&str; 6] = ["paused", "interrupted", "budget_exhausted", "failed", "completed", "cancelled"];

/// Send the user's message to a task. Returns what happened: `answered` (it
/// answered the waiting question), `queued` (the agent reads it before its
/// next model turn) or `resumed` (the task was stopped; `resume` restarted it
/// with the message).
pub async fn send_message(st: &SharedState, app: &AppHandle, run_id: &str, text: &str, resume: bool, budgets: Option<Budgets>) -> AppResult<String> {
    let text = text.trim();
    if text.is_empty() {
        return Err(AppError::InvalidInput("the message is empty".into()));
    }
    let row = st.db.get_run(run_id).await?;
    refuse_chat_run(&row, "sent a message")?;
    let live = st.tasks.lock().unwrap().get(run_id).cloned();
    let main_question = |steps: &[AgentStepRow]| {
        steps.iter().find_map(|s| {
            let d = parse_data(&s.data);
            (s.kind == "question" && s.state == "waiting" && s.parent_agent_id.is_none() && d["allow_text"].as_bool().unwrap_or(true)).then(|| d["question_id"].as_str().unwrap_or_default().to_string())
        })
    };
    if let Some(run) = live {
        if let Some(qid) = run.asks.main_text_question() {
            answer_question(st, app, run_id, &qid, text).await?;
            return Ok("answered".into());
        }
        let step = st.db.add_step(run_id, None, "user", &serde_json::json!({ "text": text, "delivered": false }), "queued").await?;
        emit(app, serde_json::json!({ "run_id": run_id, "agent_id": null, "type": "step", "step": step }));
        run.inbox.lock().unwrap().push(Queued { step_id: step.id, text: text.to_string() });
        return Ok("queued".into());
    }
    match row.state.as_str() {
        "waiting_for_user" => {
            let steps = st.db.list_steps(run_id).await?;
            let qid = main_question(&steps).ok_or_else(|| AppError::InvalidInput("the task is asking you to pick one of its options; use the buttons".into()))?;
            answer_question(st, app, run_id, &qid, text).await?;
            Ok("answered".into())
        }
        "awaiting_plan_approval" | "queued" => Err(AppError::InvalidInput("approve or cancel the plan first; a message can be sent once the task runs".into())),
        state if RESUME_WITH_MESSAGE.contains(&state) => {
            if !resume {
                return Err(AppError::Conflict(format!("the task is {state}, not running; send the message with resume to continue it")));
            }
            if let Some(b) = budgets {
                st.db.set_run_budgets(run_id, &serde_json::to_value(b)?).await?;
            }
            add_to_transcript(st, &row, &agent_runtime::task::user_message_text(text)).await?;
            let step = st.db.add_step(run_id, None, "user", &serde_json::json!({ "text": text, "delivered": true }), "done").await?;
            emit(app, serde_json::json!({ "run_id": run_id, "agent_id": null, "type": "step", "step": step }));
            launch(st, app, run_id).await?;
            Ok("resumed".into())
        }
        other => Err(AppError::InvalidInput(format!("a {other} task cannot take a message"))),
    }
}

/// Questions waiting for the user in any task (the app-wide indicator).
pub async fn waiting_questions(st: &SharedState) -> AppResult<Vec<AgentStepRow>> {
    st.db.waiting_questions().await
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

    fn kid(id: &str, label: &str, ended: Option<&str>) -> Kid {
        let (tx, rx) = tokio::sync::watch::channel(ended.map(|s| (s.to_string(), "body".to_string())));
        std::mem::forget(tx);
        Kid { id: id.into(), label: label.into(), budgets: Budgets::default(), cancel: CancellationToken::new(), done: rx }
    }

    #[test]
    fn subagent_tool_schemas() {
        let tools = subagent_tools("r1");
        assert_eq!(tools.iter().map(|t| t.tool_name.as_str()).collect::<Vec<_>>(), SUBAGENT_TOOLS.to_vec());
        assert!(tools.iter().all(|t| is_subagent_tool(t) && t.server_id == "agent:r1" && t.tool_id == format!("agent:r1/{}", t.tool_name)));
        let ok = |name: &str, v: serde_json::Value| agent_runtime::validate_args(&subagent_tool("r", name).unwrap().input_schema, &v).is_ok();
        assert!(ok("spawn_subagents", serde_json::json!({ "subagents": [{ "label": "a", "objective": "b" }], "wait": false })));
        assert!(!ok("spawn_subagents", serde_json::json!({ "subagents": [{ "label": "a", "objective": "b" }], "wait": "no" })));
        assert!(ok("subagent_status", serde_json::json!({})) && ok("subagent_status", serde_json::json!({ "ids": ["x"] })));
        assert!(ok("wait_subagents", serde_json::json!({ "ids": ["x"], "timeout_secs": 30 })));
        assert!(!ok("wait_subagents", serde_json::json!({ "timeout_secs": 0 })));
        assert!(ok("stop_subagent", serde_json::json!({ "id": "w1", "reason": "stuck" })));
        assert!(!ok("stop_subagent", serde_json::json!({ "reason": "no id" })));
        assert!(subagent_tool("r", "format_disk").is_none());
        // Old saved spawn arguments (no "wait") still wait.
        let a: SpawnArgs = serde_json::from_value(serde_json::json!({ "subagents": [] })).unwrap();
        assert!(a.wait);
    }

    #[test]
    fn subagents_are_found_by_id_or_label_and_unknown_ones_refused() {
        assert!(resolve_kids(&[], &[]).unwrap_err().contains("not started any"));
        let kids = vec![kid("id1", "w1", Some("completed")), kid("id2", "w2", None), kid("id3", "w1", None), kid("id4", "dup", None), kid("id5", "dup", None)];
        assert_eq!(resolve_kids(&kids, &[]).unwrap().len(), 5, "no ids = all");
        assert_eq!(resolve_kids(&kids, &["id2".into()]).unwrap()[0].id, "id2");
        assert_eq!(resolve_kids(&kids, &["W2".into()]).unwrap()[0].id, "id2", "labels ignore case");
        // Two "w1": the one still running is meant.
        assert_eq!(resolve_kids(&kids, &["w1".into()]).unwrap()[0].id, "id3");
        assert!(resolve_kids(&kids, &["dup".into()]).unwrap_err().contains("use the id"));
        let e = resolve_kids(&kids, &["nope".into()]).unwrap_err();
        assert!(e.contains("No subagent \"nope\"") && e.contains("(id2)"), "{e}");
        assert_eq!(resolve_kids(&kids, &["id2".into(), "w2".into()]).unwrap().len(), 1, "no duplicates");
    }

    #[test]
    fn prompts_say_no_limit_when_there_is_none() {
        let free = task_section("x", &Budgets::default(), &Usage::default(), false);
        assert!(free.contains("no limit on model turns, tool calls, time or tokens"), "{free}");
        let tight = task_section("x", &Budgets::limited(10, 5, 600, 9000), &Usage { turns: 4, ..Default::default() }, true);
        assert!(tight.contains("Budget left: 6 model turns, 5 tool calls, about 10 minutes, 9000 tokens.") && tight.contains("stop_subagent"), "{tight}");
        let tools = task_tools_section(&[], &Budgets::default());
        assert!(tools.contains("Limits for this work: no limit on model turns") && !tools.contains("4294967295"), "{tools}");
        // MCP servers are on unless turned off; old settings load without the list.
        let d = TaskSettings::default();
        assert!(d.mcp_tools && d.disabled_servers.is_empty());
        let old: TaskSettings = serde_json::from_str(r#"{"mcp_tools":false,"tool_servers":["s1"]}"#).unwrap();
        assert!(!old.mcp_tools && old.disabled_servers.is_empty() && old.tool_servers == ["s1"]);
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

    #[test]
    fn ask_user_tool_schema_and_where_it_is_offered() {
        let t = ask_user_tool("r1");
        assert!(t.server_id == "agent:r1" && t.tool_id == "agent:r1/ask_user" && !is_subagent_tool(&t), "not a subagent tool, so chats never get it");
        let ok = |v: serde_json::Value| agent_runtime::validate_args(&t.input_schema, &v).is_ok();
        assert!(ok(serde_json::json!({ "question": "Which color?" })));
        assert!(ok(serde_json::json!({ "question": "Which?", "options": ["red", "blue"], "allow_text": false })));
        assert!(!ok(serde_json::json!({})), "question is required");
        assert!(!ok(serde_json::json!({ "question": "q", "options": ["only one"] })));
        assert!(!ok(serde_json::json!({ "question": "q", "options": ["1", "2", "3", "4", "5", "6", "7"] })));
        assert!(!ok(serde_json::json!({ "question": "q", "allow_text": "yes" })));
        assert!(!ok(serde_json::json!({ "question": "q", "extra": 1 })));
        // The prompts tell the model when to ask.
        let task = task_section("x", &Budgets::default(), &Usage::default(), false);
        assert!(task.contains("ask_user") && task.contains("sensible defaults"), "{task}");
        assert!(plan_section("x", &Budgets::default(), &Usage::default()).contains("clarifying questions with ask_user"));
        assert!(subagent_section("a", "b", &Budgets::default(), true).contains("ask_user"));
        assert!(subagent_section("a", "b", &Budgets::default(), false).contains("Do not ask questions"));
    }

    #[test]
    fn a_task_waiting_for_the_user_cannot_be_resumed_but_stopped_ones_can() {
        let e = resume_check("waiting_for_user").unwrap_err().to_string();
        assert!(e.contains("waiting for your answer"), "{e}");
        assert!(resume_check("awaiting_plan_approval").is_err() && resume_check("running").is_err() && resume_check("completed").is_err());
        for ok in ["paused", "interrupted", "budget_exhausted", "failed"] {
            assert!(resume_check(ok).is_ok(), "{ok}");
        }
        // A message can restart finished tasks too, but never a waiting or planning one.
        assert!(RESUME_WITH_MESSAGE.contains(&"completed") && RESUME_WITH_MESSAGE.contains(&"cancelled"));
        assert!(!RESUME_WITH_MESSAGE.contains(&"waiting_for_user") && !RESUME_WITH_MESSAGE.contains(&"awaiting_plan_approval"));
    }
}

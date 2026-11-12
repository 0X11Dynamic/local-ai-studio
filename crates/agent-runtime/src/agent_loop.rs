//! The tool-call loop. One `turn` = generate one assistant segment, then run
//! the tool calls it made (native `tool_calls`, or a strict `<command>` in
//! XML mode) and append their results. `run_reply` (a chat reply) and
//! `task::run_task` (an agent task) both loop over `turn`.

use std::time::{Duration, Instant};

use app_core::AppResult;
use async_trait::async_trait;
use inference::{ChatDelta, ChatStats, LlamaClient, SamplingParams, ToolCall};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::dispatch::{DispatchOutcome, ToolHost, dispatch};
use crate::tools::{ExposedTool, Limits, openai_tools};

/// Persistence and UI hooks for the loop (implemented by chat and tasks).
#[async_trait]
pub trait LoopHooks: Send + Sync {
    /// Called before each assistant segment is generated (create its message).
    async fn begin_segment(&self, segment: usize) -> AppResult<()>;
    /// The segment is being generated again after a broken stream: drop its partial text.
    fn reset_segment(&self, _segment: usize) {}
    /// Streaming text for the assistant segment currently being generated.
    fn on_delta(&self, segment: usize, delta: &ChatDelta);
    /// A finished assistant segment (may carry tool calls). Returns its message id.
    async fn on_segment(&self, segment: usize, content: &str, reasoning: &str, tool_calls: &[ToolCall], stats: &ChatStats) -> AppResult<String>;
    /// A tool result, stored as a `tool` message for `call`.
    async fn on_tool_result(&self, call: &ToolCall, outcome: &DispatchOutcome) -> AppResult<()>;
}

#[derive(Debug, Clone, Default)]
pub struct LoopOutcome {
    pub segments: usize,
    pub tool_calls: u32,
    /// stop | length | cancelled | limit | time_limit | error
    pub stop_reason: String,
    pub final_message_id: Option<String>,
    /// Stats of the final segment.
    pub stats: ChatStats,
    pub error: Option<String>,
}

/// How tools reach the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolMode {
    /// OpenAI-style `tools` / `tool_calls` (the chat template supports them).
    #[default]
    Native,
    /// The strict `<command>` text format (see `xml.rs`).
    Xml,
}

fn is_transient(msg: &str) -> bool {
    msg.contains("stream interrupted") || msg.contains("could not reach the model backend") || msg.contains("error decoding response body")
}

/// Everything a turn needs that stays the same across turns.
pub struct TurnCtx<'a> {
    pub client: &'a LlamaClient,
    pub tools: &'a [ExposedTool],
    pub params: &'a SamplingParams,
    pub host: &'a dyn ToolHost,
    pub hooks: &'a dyn LoopHooks,
    /// Permission scope for "allow for this chat/task" (conversation or run id).
    pub scope: Option<&'a str>,
    pub cancel: &'a CancellationToken,
    pub mode: ToolMode,
    /// Shared limit on concurrent generations (one GPU, one model).
    pub gate: Option<&'a Semaphore>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnEnd {
    /// The model answered without calling a tool.
    Final,
    /// Tools ran (or a malformed command was reported); generate again.
    Continue,
    /// stop reason (cancelled | time_limit | error) and the error text.
    Stop(String, Option<String>),
}

#[derive(Debug, Clone)]
pub struct TurnResult {
    pub end: TurnEnd,
    pub message_id: Option<String>,
    pub content: String,
    pub stats: ChatStats,
    /// Tool calls that counted against the limit (including refused ones).
    pub calls_counted: u32,
    /// The call limit was hit during this turn.
    pub limit_hit: bool,
}

impl TurnResult {
    fn stop(reason: &str, err: Option<String>) -> Self {
        Self { end: TurnEnd::Stop(reason.into(), err), message_id: None, content: String::new(), stats: ChatStats::default(), calls_counted: 0, limit_hit: false }
    }
}

/// Generate one segment and run its tool calls, appending everything to
/// `messages`. `offer_tools: false` asks for a written answer.
pub async fn turn(ctx: &TurnCtx<'_>, messages: &mut Vec<serde_json::Value>, segment: usize, offer_tools: bool, calls_left: u32, time_left: Duration) -> TurnResult {
    let xml = ctx.mode == ToolMode::Xml;
    let offer_tools = offer_tools && !ctx.tools.is_empty();
    let defs = openai_tools(ctx.tools);
    if let Err(e) = ctx.hooks.begin_segment(segment).await {
        return TurnResult::stop("error", Some(e.to_string()));
    }
    let msgs = serde_json::Value::Array(messages.clone());
    let res = {
        // Wait for a free generation slot (the wait counts against time).
        let _permit = match ctx.gate {
            Some(g) => tokio::select! {
                p = g.acquire() => p.ok(),
                _ = ctx.cancel.cancelled() => return TurnResult::stop("cancelled", None),
                _ = tokio::time::sleep(time_left) => return TurnResult::stop("time_limit", None),
            },
            None => None,
        };
        // A broken stream (connection dropped mid-reply) is retried once. This
        // is safe: tool calls only run after a complete reply, so nothing has
        // happened yet for this segment.
        let mut attempt = 0;
        loop {
            let tools_arg = if offer_tools && !xml { Some(&defs) } else { None };
            let fut = inference::chat::stream_chat_tools(ctx.client, &msgs, tools_arg, ctx.params, ctx.cancel, |d| ctx.hooks.on_delta(segment, &d));
            let r = match tokio::time::timeout(time_left, fut).await {
                Ok(r) => r,
                Err(_) => return TurnResult::stop("time_limit", None),
            };
            match &r {
                Err(e) if attempt == 0 && !ctx.cancel.is_cancelled() && is_transient(&e.to_string()) => {
                    tracing::warn!("agent reply stream broke ({e}); retrying once");
                    attempt += 1;
                    ctx.hooks.reset_segment(segment);
                    continue;
                }
                _ => break r,
            }
        }
    };
    let reply = match res {
        Ok(g) => g,
        Err(e) => return TurnResult::stop("error", Some(e.to_string())),
    };
    // XML mode: the reply text is the only place a command can come from.
    let mut note: Option<String> = None;
    let mut malformed = false;
    let calls = if !offer_tools {
        vec![]
    } else if xml {
        match crate::xml::parse(&reply.content, &format!("xml-{}", app_core::new_id())) {
            crate::xml::Parsed::None => vec![],
            crate::xml::Parsed::Command { call, ignored } => {
                if ignored > 0 {
                    note = Some(format!("Only the first command ran; {ignored} more in the same reply were ignored. Send one command per reply."));
                }
                vec![call]
            }
            crate::xml::Parsed::Malformed(why) => {
                malformed = true;
                note = Some(format!("Your command was not run: {why}. Write it exactly in the format from your instructions, or answer without a command."));
                vec![]
            }
        }
    } else if reply.tool_calls.is_empty() {
        // Safety net for a tool call the server left in the text.
        let offered: Vec<&str> = ctx.tools.iter().map(|t| t.model_name.as_str()).collect();
        crate::xml::parse_tool_call_text(&reply.content, &format!("txt-{}", app_core::new_id()), &offered).into_iter().collect()
    } else {
        reply.tool_calls.clone()
    };
    // A reply cut off at the length limit (often a model repeating itself
    // until the context is full) is not trusted: none of its tool calls run.
    let cut_off = reply.stats.stop_reason == "length" && !calls.is_empty();
    let calls = if cut_off {
        note = Some(format!(
            "Your last reply was cut off because it was too long, so none of its {} tool call(s) ran. Reply again, shorter, with only the next tool call(s) you need.",
            calls.len()
        ));
        vec![]
    } else {
        calls
    };
    let message_id = match ctx.hooks.on_segment(segment, &reply.content, &reply.reasoning, &calls, &reply.stats).await {
        Ok(id) => Some(id),
        Err(e) => return TurnResult::stop("error", Some(e.to_string())),
    };
    let mut result = TurnResult { end: TurnEnd::Final, message_id, content: reply.content.clone(), stats: reply.stats.clone(), calls_counted: 0, limit_hit: false };
    if reply.stats.stop_reason == "cancelled" {
        result.end = TurnEnd::Stop("cancelled".into(), None);
        return result;
    }
    if malformed || cut_off {
        // Tell the model and let it try again; this counts as a call so a
        // model that keeps writing broken commands cannot loop forever.
        result.calls_counted = 1;
        result.limit_hit = calls_left <= 1;
        messages.push(serde_json::json!({ "role": "assistant", "content": reply.content }));
        messages.push(serde_json::json!({ "role": "user", "content": note.unwrap_or_default() }));
        result.end = TurnEnd::Continue;
        return result;
    }
    if calls.is_empty() {
        return result;
    }
    // Record the assistant turn with its tool calls, then run each call.
    if xml {
        messages.push(serde_json::json!({ "role": "assistant", "content": reply.content }));
    } else {
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": calls.iter().map(|c| serde_json::json!({
                "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments }
            })).collect::<Vec<_>>(),
        }));
    }
    for call in &calls {
        let outcome = if result.calls_counted >= calls_left {
            result.limit_hit = true;
            DispatchOutcome { status: "limit".into(), content: "Not run: the tool call limit was reached. Answer with what you have so far.".into(), audit_id: None }
        } else if ctx.cancel.is_cancelled() {
            DispatchOutcome { status: "cancelled".into(), content: "Not run: the work was stopped.".into(), audit_id: None }
        } else {
            result.calls_counted += 1;
            dispatch(ctx.host, ctx.tools, call, ctx.scope, ctx.cancel).await
        };
        let _ = ctx.hooks.on_tool_result(call, &outcome).await;
        if xml {
            let mut text = crate::xml::result_message(&call.name, &outcome.status, &outcome.content);
            if let Some(n) = note.take() {
                text.push_str(&format!("\n{n}"));
            }
            messages.push(serde_json::json!({ "role": "user", "content": text }));
        } else {
            messages.push(serde_json::json!({ "role": "tool", "tool_call_id": call.id, "content": outcome.content }));
        }
    }
    result.end = if ctx.cancel.is_cancelled() { TurnEnd::Stop("cancelled".into(), None) } else { TurnEnd::Continue };
    result
}

/// One chat reply in Agent Mode: turns until a final answer or a limit.
#[allow(clippy::too_many_arguments)]
pub async fn run_reply(
    client: &LlamaClient,
    mut messages: Vec<serde_json::Value>,
    tools: &[ExposedTool],
    params: &SamplingParams,
    limits: &Limits,
    host: &dyn ToolHost,
    hooks: &dyn LoopHooks,
    conversation_id: Option<&str>,
    cancel: &CancellationToken,
    mode: ToolMode,
    gate: Option<&Semaphore>,
) -> LoopOutcome {
    let started = Instant::now();
    let ctx = TurnCtx { client, tools, params, host, hooks, scope: conversation_id, cancel, mode, gate };
    let mut out = LoopOutcome::default();
    let mut offer_tools = !tools.is_empty();
    let mut limit_seen = false;
    for segment in 0.. {
        if cancel.is_cancelled() {
            out.stop_reason = "cancelled".into();
            return out;
        }
        let remaining = Duration::from_secs(limits.max_wall_secs).saturating_sub(started.elapsed());
        if remaining.is_zero() {
            out.stop_reason = "time_limit".into();
            return out;
        }
        let calls_left = limits.max_tool_calls.saturating_sub(out.tool_calls);
        let t = turn(&ctx, &mut messages, segment, offer_tools, calls_left, remaining).await;
        out.segments = segment + 1;
        out.tool_calls += t.calls_counted;
        if t.message_id.is_some() {
            out.final_message_id = t.message_id.clone();
            out.stats = t.stats.clone();
        }
        if t.limit_hit {
            limit_seen = true;
            // Next generation gets no tools, forcing a written answer.
            offer_tools = false;
        }
        match t.end {
            TurnEnd::Final => {
                out.stop_reason = if limit_seen { "limit".into() } else { t.stats.stop_reason.clone() };
                return out;
            }
            TurnEnd::Continue => {}
            TurnEnd::Stop(reason, err) => {
                out.stop_reason = reason;
                out.error = err;
                return out;
            }
        }
    }
    out
}

//! The tool-call loop. One `turn` = generate one assistant segment, then run
//! the tool calls it made (native `tool_calls`, or a strict `<command>` in
//! XML mode) and append their results. `run_reply` (a chat reply) and
//! `task::run_task` (an agent task) both loop over `turn`.

use std::time::{Duration, Instant};

use app_core::AppResult;
use async_trait::async_trait;
use inference::{ChatDelta, ChatStats, LlamaClient, SamplingParams, ToolCall};
use tokio_util::sync::CancellationToken;

use crate::dispatch::{DispatchOutcome, ToolHost, dispatch};
use crate::gate::{Gate, GateStop, Seat};
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
    /// The model gave a final answer. If subagents it started are still running,
    /// wait for them (at most `remaining` working time) and return what they
    /// reported, as text to hand back to the model; None = none were running
    /// (or the time ran out). The loop then generates again.
    async fn wait_children(&self, _remaining: Duration) -> Option<String> {
        None
    }
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
    /// The final segment had no visible answer (see `TurnResult::empty_answer`).
    pub empty_answer: bool,
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
    /// This agent's seat at the shared generation gate (one GPU, one model).
    /// The seat keeps its permit across consecutive fast steps (see `gate.rs`).
    pub gate: Option<&'a Seat>,
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
    /// The reply had no visible text and no tool call (only reasoning, or
    /// nothing), even after the one nudge when it followed tool results.
    pub empty_answer: bool,
}

/// Sent (as a user message) once when a reply after tool results is empty.
pub const NUDGE: &str = "Continue: give your answer now.";

/// Shown instead of a silent empty reply.
pub const NO_ANSWER_NOTE: &str = "The model returned no answer.";

/// No visible text and no tool call to run (a refused malformed or cut-off
/// command is not an empty answer: the model is told and tries again).
fn is_empty_answer(reply: &inference::ChatOutput, calls: &[ToolCall], refused: bool) -> bool {
    calls.is_empty() && !refused && reply.content.trim().is_empty() && reply.stats.stop_reason != "cancelled"
}

/// The last message is a tool result (native `tool`, or the XML-mode result message).
fn follows_tool_results(messages: &[serde_json::Value], xml: bool) -> bool {
    let Some(last) = messages.last() else { return false };
    match last["role"].as_str() {
        Some("tool") => true,
        Some("user") if xml => last["content"].as_str().is_some_and(|c| c.starts_with("<result tool=")),
        _ => false,
    }
}

/// Stream one reply, retrying once after a broken stream. This is safe: tool
/// calls only run after a complete reply, so nothing has happened yet.
async fn generate(
    ctx: &TurnCtx<'_>,
    messages: &[serde_json::Value],
    send_tools: bool,
    defs: &serde_json::Value,
    segment: usize,
    time_left: Duration,
) -> Result<inference::ChatOutput, TurnResult> {
    let msgs = serde_json::Value::Array(messages.to_vec());
    let mut attempt = 0;
    loop {
        let tools_arg = if send_tools { Some(defs) } else { None };
        let fut = inference::chat::stream_chat_tools(ctx.client, &msgs, tools_arg, ctx.params, ctx.cancel, |d| ctx.hooks.on_delta(segment, &d));
        let r = match tokio::time::timeout(time_left, fut).await {
            Ok(r) => r,
            Err(_) => return Err(TurnResult::stop("time_limit", None)),
        };
        match r {
            Err(e) if attempt == 0 && !ctx.cancel.is_cancelled() && is_transient(&e.to_string()) => {
                tracing::warn!("agent reply stream broke ({e}); retrying once");
                attempt += 1;
                ctx.hooks.reset_segment(segment);
            }
            Err(e) => return Err(TurnResult::stop("error", Some(e.to_string()))),
            Ok(g) => return Ok(g),
        }
    }
}

impl TurnResult {
    fn stop(reason: &str, err: Option<String>) -> Self {
        Self { end: TurnEnd::Stop(reason.into(), err), message_id: None, content: String::new(), stats: ChatStats::default(), calls_counted: 0, limit_hit: false, empty_answer: false }
    }
}

/// Generate one segment and run its tool calls, appending everything to
/// `messages`. `offer_tools: false` asks for a written answer.
pub async fn turn(ctx: &TurnCtx<'_>, messages: &mut Vec<serde_json::Value>, segment: usize, offer_tools: bool, calls_left: u32, time_left: Duration) -> TurnResult {
    let r = turn_inner(ctx, messages, segment, offer_tools, calls_left, time_left).await;
    // Only a turn that goes straight on to the next step keeps the gate.
    if !matches!(r.end, TurnEnd::Continue)
        && let Some(seat) = ctx.gate
    {
        seat.release();
    }
    r
}

async fn turn_inner(ctx: &TurnCtx<'_>, messages: &mut Vec<serde_json::Value>, segment: usize, offer_tools: bool, calls_left: u32, time_left: Duration) -> TurnResult {
    let xml = ctx.mode == ToolMode::Xml;
    let offer_tools = offer_tools && !ctx.tools.is_empty();
    // Deferred loading: only the tools not deferred, plus the ones loaded so
    // far, are sent in full (re-read each step: load_tools may have added some).
    let offer = ctx.host.tool_offer();
    let defs = match &offer {
        Some(o) => openai_tools(&o.sent(ctx.tools)),
        None => openai_tools(ctx.tools),
    };
    if let Err(e) = ctx.hooks.begin_segment(segment).await {
        return TurnResult::stop("error", Some(e.to_string()));
    }
    let started = Instant::now();
    // A reply right after tool results that says nothing (only reasoning, or
    // nothing at all) and calls no tool is nudged once, then generated again.
    let after_tools = follows_tool_results(messages, xml);
    let mut nudged = false;
    let mut earlier_tokens = 0u64;
    // Wait for a free generation slot (the wait counts against time). An agent
    // that kept its slot from the last step goes straight on.
    if let Some(seat) = ctx.gate
        && let Err(stop) = seat.acquire(ctx.cancel, time_left).await
    {
        return TurnResult::stop(if stop == GateStop::Cancelled { "cancelled" } else { "time_limit" }, None);
    }
    let (mut reply, calls, mut note, malformed, cut_off) = loop {
        let left = time_left.saturating_sub(started.elapsed());
        let reply = match generate(ctx, messages, offer_tools && !xml, &defs, segment, left).await {
            Ok(r) => r,
            Err(t) => return t,
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
        let empty = is_empty_answer(&reply, &calls, malformed || cut_off);
        if empty && after_tools && !nudged && !ctx.cancel.is_cancelled() && !time_left.saturating_sub(started.elapsed()).is_zero() {
            tracing::info!("segment {segment}: empty reply after tool results ({} reasoning chars); nudging once", reply.reasoning.len());
            nudged = true;
            earlier_tokens += reply.stats.completion_tokens.unwrap_or(0);
            match messages.last_mut() {
                // XML mode: the tool result is already a user message; add to it
                // rather than sending two user messages in a row.
                Some(last) if last["role"] == "user" => {
                    let c = last["content"].as_str().unwrap_or_default().to_string();
                    last["content"] = format!("{c}

{NUDGE}").into();
                }
                _ => messages.push(serde_json::json!({ "role": "user", "content": NUDGE })),
            }
            ctx.hooks.reset_segment(segment);
            continue;
        }
        break (reply, calls, note, malformed, cut_off);
    };
    if earlier_tokens > 0 {
        // The nudged-away attempt's tokens count too (budgets, usage).
        reply.stats.completion_tokens = Some(reply.stats.completion_tokens.unwrap_or(0) + earlier_tokens);
    }
    let empty_answer = is_empty_answer(&reply, &calls, malformed || cut_off);
    let message_id = match ctx.hooks.on_segment(segment, &reply.content, &reply.reasoning, &calls, &reply.stats).await {
        Ok(id) => Some(id),
        Err(e) => return TurnResult::stop("error", Some(e.to_string())),
    };
    let mut result = TurnResult { end: TurnEnd::Final, message_id, content: reply.content.clone(), stats: reply.stats.clone(), calls_counted: 0, limit_hit: false, empty_answer };
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
            // A listed-but-not-loaded tool called directly: it is validated
            // against its real schema as usual, and loaded for later steps.
            if let Some(o) = offer.as_ref().filter(|o| o.is_hidden(&call.name)) {
                o.load(std::slice::from_ref(&call.name));
            }
            let run = dispatch(ctx.host, ctx.tools, call, ctx.scope, ctx.cancel);
            match ctx.gate {
                // A call that waits on something external lets go of the gate.
                Some(seat) => seat.during_tool(&call.name, run).await,
                None => run.await,
            }
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
    gate: Option<&Gate>,
) -> LoopOutcome {
    let started = Instant::now();
    let seat = gate.map(Gate::seat);
    let ctx = TurnCtx { client, tools, params, host, hooks, scope: conversation_id, cancel, mode, gate: seat.as_ref() };
    let mut out = LoopOutcome::default();
    let mut offer_tools = !tools.is_empty();
    let mut limit_seen = false;
    for segment in 0.. {
        if cancel.is_cancelled() {
            out.stop_reason = "cancelled".into();
            return out;
        }
        if segment as u32 >= crate::task::MAX_TURNS_BACKSTOP {
            out.stop_reason = "turn_limit".into();
            return out;
        }
        let remaining = Duration::from_secs(limits.secs_or_max()).saturating_sub(started.elapsed());
        if remaining.is_zero() {
            out.stop_reason = "time_limit".into();
            return out;
        }
        let calls_left = limits.calls_or_max().saturating_sub(out.tool_calls);
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
                // Subagents still working are waited for, not cancelled: their
                // results go back to the model, which then answers again.
                if !t.empty_answer
                    && let Some(text) = hooks.wait_children(Duration::from_secs(limits.secs_or_max()).saturating_sub(started.elapsed())).await
                {
                    messages.push(serde_json::json!({ "role": "assistant", "content": t.content }));
                    messages.push(serde_json::json!({ "role": "user", "content": text }));
                    continue;
                }
                out.stop_reason = if limit_seen { "limit".into() } else { t.stats.stop_reason.clone() };
                out.empty_answer = t.empty_answer;
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

//! Agent tasks: turns until the model gives a final answer, a budget runs
//! out, or the task is paused or cancelled. The app saves the transcript and
//! usage after every turn (`TaskObserver::checkpoint`), so a paused or
//! interrupted task resumes exactly where it stopped.

use std::time::{Duration, Instant};

use app_core::AppResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{TurnCtx, TurnEnd, turn};

/// A task's limits. Each one is optional: `None` (JSON `null`) = no limit,
/// which is the default. Old runs saved with numbers still load (a number
/// reads as `Some`). Whatever the limits, a run never loops forever silently:
/// Stop/Cancel always works, old tool output is trimmed to fit the context,
/// an empty answer ends the run with a note, and [`MAX_TURNS_BACKSTOP`] model
/// turns per start is a hard stop even with no turn limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Budgets {
    /// Model turns (assistant segments).
    pub max_turns: Option<u32>,
    pub max_tool_calls: Option<u32>,
    /// Active time (paused time does not count).
    pub max_wall_secs: Option<u64>,
    /// Completion tokens generated.
    pub max_tokens: Option<u64>,
}

/// Hard stop: at most this many model turns each time a run is started or
/// resumed, even when it has no turn limit (Resume goes on for another 500).
/// Keeps a model that never gives a final answer from looping forever.
pub const MAX_TURNS_BACKSTOP: u32 = 500;

/// Why a run hit [`MAX_TURNS_BACKSTOP`] (shown as "the ... budget ran out").
pub const BACKSTOP_REASON: &str = "model turns (safety stop after 500 turns per start; Resume to continue)";

/// Used where a duration is needed for "no time limit" (30 days).
pub const NO_TIME_LIMIT_SECS: u64 = 30 * 24 * 3600;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub turns: u32,
    pub tool_calls: u32,
    pub wall_ms: u64,
    pub tokens: u64,
}

impl Usage {
    pub fn add(&mut self, o: &Usage) {
        self.turns += o.turns;
        self.tool_calls += o.tool_calls;
        self.tokens += o.tokens;
        // Wall time is not added: children run during the parent's own time.
    }
}

/// The smaller of two optional limits (None = no limit).
fn min_opt<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl Budgets {
    /// A number on every limit (old-style budgets, tests).
    pub fn limited(max_turns: u32, max_tool_calls: u32, max_wall_secs: u64, max_tokens: u64) -> Self {
        Self { max_turns: Some(max_turns), max_tool_calls: Some(max_tool_calls), max_wall_secs: Some(max_wall_secs), max_tokens: Some(max_tokens) }
    }

    /// Which budget is used up, if any (a limit that is None never is).
    pub fn exhausted(&self, u: &Usage) -> Option<&'static str> {
        if self.max_turns.is_some_and(|m| u.turns >= m) {
            Some("model turns")
        } else if self.max_tokens.is_some_and(|m| u.tokens >= m) {
            Some("tokens")
        } else if self.max_wall_secs.is_some_and(|m| u.wall_ms >= m.saturating_mul(1000)) {
            Some("time")
        } else {
            None
        }
    }

    /// What is left of each limit (None stays None).
    pub fn remaining(&self, u: &Usage) -> Budgets {
        Budgets {
            max_turns: self.max_turns.map(|m| m.saturating_sub(u.turns)),
            max_tool_calls: self.max_tool_calls.map(|m| m.saturating_sub(u.tool_calls)),
            max_wall_secs: self.max_wall_secs.map(|m| m.saturating_sub(u.wall_ms / 1000)),
            max_tokens: self.max_tokens.map(|m| m.saturating_sub(u.tokens)),
        }
    }

    /// A child's budget: what it asked for, capped by `fraction` of what the
    /// parent has left. No floors: children together never get more than the
    /// parent has (with `fraction` = 1 / number of children or less). A limit
    /// the parent does not have is not invented for the child (it gets what
    /// it asked for, or no limit); a parent's time limit always caps its
    /// children's time. Check the result with `too_small` before starting a child.
    pub fn share(&self, used: &Usage, fraction: f64, asked: Option<&Budgets>) -> Budgets {
        let left = self.remaining(used);
        let f = fraction.clamp(0.0, 1.0);
        let cap = |l: u64| (l as f64 * f) as u64;
        let capped = Budgets {
            max_turns: left.max_turns.map(|l| cap(l as u64) as u32),
            max_tool_calls: left.max_tool_calls.map(|l| cap(l as u64) as u32),
            max_wall_secs: left.max_wall_secs.map(cap),
            max_tokens: left.max_tokens.map(cap),
        };
        match asked {
            None => capped,
            Some(a) => Budgets {
                max_turns: min_opt(a.max_turns, capped.max_turns),
                max_tool_calls: min_opt(a.max_tool_calls, capped.max_tool_calls),
                max_wall_secs: min_opt(a.max_wall_secs, capped.max_wall_secs),
                max_tokens: min_opt(a.max_tokens, capped.max_tokens),
            },
        }
    }

    /// Why a child's share is too small to do anything useful (None = fine):
    /// at least 2 model turns (one to act, one to answer), 1 tool call,
    /// 10 seconds and 500 tokens. A limit that is None is never too small.
    pub fn too_small(&self) -> Option<&'static str> {
        if self.max_turns.is_some_and(|m| m < 2) {
            Some("model turns")
        } else if self.max_tool_calls.is_some_and(|m| m < 1) {
            Some("tool calls")
        } else if self.max_wall_secs.is_some_and(|m| m < 10) {
            Some("time")
        } else if self.max_tokens.is_some_and(|m| m < 500) {
            Some("tokens")
        } else {
            None
        }
    }

    /// The tool-call limit as a count (no limit = u32::MAX).
    pub fn calls_or_max(&self) -> u32 {
        self.max_tool_calls.unwrap_or(u32::MAX)
    }

    /// The time limit in seconds (no limit = [`NO_TIME_LIMIT_SECS`]).
    pub fn secs_or_max(&self) -> u64 {
        self.max_wall_secs.unwrap_or(NO_TIME_LIMIT_SECS)
    }

    /// Plain text for prompts, e.g. "12 model turns, no limit on tool calls,
    /// about 5 minutes, no token limit".
    pub fn describe(&self) -> String {
        let turns = self.max_turns.map_or("no limit on model turns".to_string(), |n| format!("{n} model turns"));
        let calls = self.max_tool_calls.map_or("no limit on tool calls".to_string(), |n| format!("{n} tool calls"));
        let time = self.max_wall_secs.map_or("no time limit".to_string(), |s| format!("about {} minutes", (s / 60).max(1)));
        let tokens = self.max_tokens.map_or("no token limit".to_string(), |n| format!("{n} tokens"));
        format!("{turns}, {calls}, {time}, {tokens}")
    }
}

#[async_trait]
pub trait TaskObserver: Send + Sync {
    /// Before each turn: the app may shorten old tool outputs to fit the
    /// context, and fold finished subagents' usage into `usage`.
    async fn prepare(&self, _transcript: &mut Vec<serde_json::Value>, _usage: &mut Usage) -> AppResult<()> {
        Ok(())
    }
    /// After each turn: save the transcript and usage.
    async fn checkpoint(&self, transcript: &[serde_json::Value], usage: &Usage) -> AppResult<()>;
    /// Messages the user sent while the agent worked, oldest first. They are
    /// added to the transcript as user messages before the next model turn.
    async fn take_user_messages(&self) -> Vec<String> {
        vec![]
    }
    /// Milliseconds the last turn spent waiting for the user (`ask_user`);
    /// that time does not count toward the time limit.
    fn take_waited_ms(&self) -> u64 {
        0
    }
    /// The main agent gave a final answer: wait for subagents still running (at
    /// most `remaining`) and return their results as a message for the model;
    /// None = none were running (or the time ran out).
    async fn wait_children(&self, _remaining: Duration) -> Option<String> {
        None
    }
}

/// How a message from the user reads to the model.
pub fn user_message_text(text: &str) -> String {
    format!("Message from the user (their own words): {}", text.trim())
}

/// Add the user's queued messages to the transcript as user messages. Where
/// the last message is already a user message (the objective, or an XML-mode
/// tool result) they join it, so two user messages never follow each other.
pub fn inject_user_messages(transcript: &mut Vec<serde_json::Value>, msgs: &[String]) {
    for m in msgs {
        let text = user_message_text(m);
        match transcript.last_mut() {
            Some(last) if last["role"] == "user" && last["content"].as_str().is_some() => {
                let c = last["content"].as_str().unwrap_or_default().to_string();
                last["content"] = format!("{c}

{text}").into();
            }
            _ => transcript.push(serde_json::json!({ "role": "user", "content": text })),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskEnd {
    Completed { answer: String },
    Paused,
    Cancelled,
    BudgetExhausted(&'static str),
    Failed(String),
}

impl TaskEnd {
    pub fn state(&self) -> &'static str {
        match self {
            TaskEnd::Completed { .. } => "completed",
            TaskEnd::Paused => "paused",
            TaskEnd::Cancelled => "cancelled",
            TaskEnd::BudgetExhausted(_) => "budget_exhausted",
            TaskEnd::Failed(_) => "failed",
        }
    }
}

/// Run (or resume) a task. `pause` is a separate token: pausing takes effect
/// after the current turn (including its tool calls) finishes.
pub async fn run_task(
    ctx: &TurnCtx<'_>,
    transcript: &mut Vec<serde_json::Value>,
    budgets: &Budgets,
    usage: &mut Usage,
    pause: &CancellationToken,
    observer: &dyn TaskObserver,
) -> TaskEnd {
    let mut segment = usage.turns as usize;
    // Turns this start: the backstop counts per start, so Resume can go on.
    let mut this_start = 0u32;
    loop {
        if ctx.cancel.is_cancelled() {
            return TaskEnd::Cancelled;
        }
        if pause.is_cancelled() {
            return TaskEnd::Paused;
        }
        if let Some(which) = budgets.exhausted(usage) {
            return TaskEnd::BudgetExhausted(which);
        }
        let msgs = observer.take_user_messages().await;
        inject_user_messages(transcript, &msgs);
        if let Err(e) = observer.prepare(transcript, usage).await {
            return TaskEnd::Failed(e.to_string());
        }
        if let Some(which) = budgets.exhausted(usage) {
            return TaskEnd::BudgetExhausted(which);
        }
        if this_start >= MAX_TURNS_BACKSTOP {
            return TaskEnd::BudgetExhausted(BACKSTOP_REASON);
        }
        let left = budgets.remaining(usage);
        let offer_tools = left.max_tool_calls != Some(0);
        // One reply never generates more than the token budget has left.
        let mut params = ctx.params.clone();
        if let Some(t) = left.max_tokens {
            let cap = u32::try_from(t).unwrap_or(u32::MAX);
            params.max_tokens = Some(params.max_tokens.map_or(cap, |m| m.min(cap)));
        }
        let capped = TurnCtx { params: &params, ..*ctx };
        let t0 = Instant::now();
        let t = turn(&capped, transcript, segment, offer_tools, left.calls_or_max(), Duration::from_secs(left.secs_or_max().max(1))).await;
        this_start += 1;
        usage.turns += 1;
        usage.tool_calls += t.calls_counted;
        // Time spent waiting for the user's answer is not working time.
        usage.wall_ms += (t0.elapsed().as_millis() as u64).saturating_sub(observer.take_waited_ms());
        usage.tokens += t.stats.completion_tokens.unwrap_or(0);
        segment += 1;
        if let Err(e) = observer.checkpoint(transcript, usage).await {
            return TaskEnd::Failed(format!("could not save progress: {e}"));
        }
        match t.end {
            TurnEnd::Final if t.empty_answer => {
                // Not a silent empty "result": say so, and let Resume try again
                // (the transcript ends before this reply, so it is generated anew).
                return TaskEnd::Failed(format!("{} It wrote only reasoning, or nothing, even after being asked to answer. Resume to try again.", crate::agent_loop::NO_ANSWER_NOTE));
            }
            TurnEnd::Final => {
                // Keep the final answer in the transcript too (resume, history).
                transcript.push(serde_json::json!({ "role": "assistant", "content": t.content }));
                // The user wrote while this answer was generated: go on with
                // their message instead of ending without reading it.
                let late = observer.take_user_messages().await;
                if !late.is_empty() {
                    inject_user_messages(transcript, &late);
                    let _ = observer.checkpoint(transcript, usage).await;
                    continue;
                }
                // Subagents still running are waited for, not cancelled: their
                // results go back to the agent, which then answers again.
                let w0 = Instant::now();
                let kids = observer.wait_children(Duration::from_secs(budgets.remaining(usage).secs_or_max().max(1))).await;
                usage.wall_ms += (w0.elapsed().as_millis() as u64).saturating_sub(observer.take_waited_ms());
                if let Some(text) = kids {
                    transcript.push(serde_json::json!({ "role": "user", "content": text }));
                    let _ = observer.checkpoint(transcript, usage).await;
                    continue;
                }
                let _ = observer.checkpoint(transcript, usage).await;
                return TaskEnd::Completed { answer: t.content };
            }
            TurnEnd::Continue => {}
            TurnEnd::Stop(reason, err) => {
                return match reason.as_str() {
                    "cancelled" => TaskEnd::Cancelled,
                    "time_limit" => TaskEnd::BudgetExhausted("time"),
                    _ => TaskEnd::Failed(err.unwrap_or(reason)),
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_exhaust_and_share() {
        let b = Budgets::limited(10, 30, 600, 9000);
        let mut u = Usage { turns: 4, tool_calls: 12, wall_ms: 60_000, tokens: 3000 };
        assert_eq!(b.exhausted(&u), None);
        // A plain third of what is left (6 turns, 18 calls, 540 s, 6000 tokens).
        // Turns used to be raised to a floor of 3, so three children could get
        // 9 turns between them out of 6: shares have no floors any more.
        let child = b.share(&u, 1.0 / 3.0, None);
        assert_eq!(child, Budgets::limited(2, 6, 180, 2000));
        assert_eq!(child.too_small(), None);
        // A child asking for more than its share gets the share; less is kept.
        let asked = Budgets::limited(100, 2, 10_000, 100);
        let c2 = b.share(&u, 1.0 / 3.0, Some(&asked));
        assert_eq!((c2.max_turns, c2.max_tool_calls, c2.max_tokens), (Some(2), Some(2), Some(100)));
        // Children together never get more than the parent has left.
        for n in 1..=6u32 {
            let f = 1.0 / (n.max(3) as f64);
            let one = b.share(&u, f, None);
            let left = b.remaining(&u);
            let (t, l) = (|x: Option<u32>| x.unwrap(), |x: Option<u64>| x.unwrap());
            assert!(t(one.max_turns) * n <= t(left.max_turns) && t(one.max_tool_calls) * n <= t(left.max_tool_calls), "{n}: {one:?}");
            assert!(l(one.max_wall_secs) * n as u64 <= l(left.max_wall_secs) && l(one.max_tokens) * n as u64 <= l(left.max_tokens), "{n}: {one:?}");
        }
        // Too little left to share: the caller refuses to start children.
        let low = Usage { turns: 8, tool_calls: 28, wall_ms: 580_000, tokens: 8000 };
        assert_eq!(b.share(&low, 1.0 / 3.0, None).too_small(), Some("model turns"));
        let late = Usage { turns: 0, tool_calls: 0, wall_ms: 590_000, tokens: 0 };
        assert_eq!(b.share(&late, 1.0 / 3.0, None).too_small(), Some("time"));
        // Never more than the parent has left.
        let tight = Usage { turns: 9, tool_calls: 29, wall_ms: 590_000, tokens: 8900 };
        let c3 = b.share(&tight, 1.0, None);
        assert!(c3.max_turns <= Some(1) && c3.max_tool_calls <= Some(1) && c3.max_wall_secs <= Some(10) && c3.max_tokens <= Some(100), "{c3:?}");
        // Roll-up.
        u.add(&Usage { turns: 2, tool_calls: 3, wall_ms: 99_999, tokens: 500 });
        assert_eq!((u.turns, u.tool_calls, u.wall_ms, u.tokens), (6, 15, 60_000, 3500));
        u.turns = 10;
        assert_eq!(b.exhausted(&u), Some("model turns"));
    }

    #[test]
    fn no_limits_by_default_and_old_numeric_budgets_still_load() {
        let d = Budgets::default();
        assert_eq!(d, Budgets { max_turns: None, max_tool_calls: None, max_wall_secs: None, max_tokens: None });
        let huge = Usage { turns: 400, tool_calls: 1_000_000, wall_ms: u64::MAX / 2, tokens: u64::MAX / 2 };
        assert_eq!(d.exhausted(&huge), None, "no limit is never used up");
        assert_eq!(d.remaining(&huge), d);
        assert_eq!((d.calls_or_max(), d.secs_or_max()), (u32::MAX, NO_TIME_LIMIT_SECS));
        // Old saved runs: plain numbers.
        let old: Budgets = serde_json::from_str(r#"{"max_turns":40,"max_tool_calls":60,"max_wall_secs":1800,"max_tokens":60000}"#).unwrap();
        assert_eq!(old, Budgets::limited(40, 60, 1800, 60_000));
        // New: null or missing = no limit; mixed is fine.
        let mixed: Budgets = serde_json::from_str(r#"{"max_turns":null,"max_tool_calls":5}"#).unwrap();
        assert_eq!(mixed, Budgets { max_tool_calls: Some(5), ..Default::default() });
        assert_eq!(serde_json::to_value(mixed).unwrap(), serde_json::json!({ "max_turns": null, "max_tool_calls": 5, "max_wall_secs": null, "max_tokens": null }));
        assert_eq!(serde_json::from_str::<Budgets>("{}").unwrap(), d);
        assert!(d.describe().contains("no limit on model turns") && d.describe().contains("no time limit"), "{}", d.describe());
        assert!(old.describe().starts_with("40 model turns, 60 tool calls, about 30 minutes"), "{}", old.describe());
    }

    #[test]
    fn sharing_a_budget_with_no_limits() {
        let used = Usage { turns: 3, tool_calls: 5, wall_ms: 30_000, tokens: 2000 };
        // A parent with no limits: children get none (or what they asked for).
        let free = Budgets::default();
        assert_eq!(free.share(&used, 1.0 / 3.0, None), Budgets::default());
        assert_eq!(free.share(&used, 1.0 / 3.0, None).too_small(), None);
        let asked = Budgets { max_turns: Some(8), ..Default::default() };
        assert_eq!(free.share(&used, 1.0 / 3.0, Some(&asked)), asked);
        // Only a time limit: the children's time is capped by their share of
        // what the parent has left; the other limits stay off.
        let timed = Budgets { max_wall_secs: Some(630), ..Default::default() };
        let c = timed.share(&used, 1.0 / 3.0, Some(&asked));
        assert_eq!(c, Budgets { max_turns: Some(8), max_wall_secs: Some(200), ..Default::default() });
        // The parent's time running out still refuses children.
        let late = Usage { wall_ms: 620_000, ..used };
        assert_eq!(timed.share(&late, 1.0 / 3.0, None).too_small(), Some("time"));
    }

    // ---- a scripted model server and fake tools, for run_task end to end ----

    use crate::agent_loop::LoopHooks;
    use crate::dispatch::{ApprovalRequest, CallResult, DispatchOutcome, ToolEvent, ToolHost};
    use crate::tools::ExposedTool;
    use inference::{ChatDelta, ChatStats, LlamaClient, SamplingParams, ToolCall};
    use security::policy::{ApprovalAnswer, Grant, GrantScope, Policy};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One scripted reply: text, tool calls (name, arguments), finish reason.
    type Reply = (&'static str, Vec<(&'static str, &'static str)>, &'static str);

    /// Serves the replies in order (one per request) and records each request body.
    async fn model_server(replies: Vec<Reply>) -> (u16, Arc<Mutex<Vec<serde_json::Value>>>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(vec![]));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            for (text, calls, finish) in replies {
                let Ok((mut sock, _)) = l.accept().await else { return };
                let mut buf = vec![];
                let body = loop {
                    let mut chunk = [0u8; 8192];
                    let n = sock.read(&mut chunk).await.unwrap_or(0);
                    buf.extend_from_slice(&chunk[..n]);
                    let s = String::from_utf8_lossy(&buf).to_string();
                    if let Some(i) = s.find("\r\n\r\n") {
                        let len = s[..i]
                            .lines()
                            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                            .unwrap_or(0);
                        if buf.len() >= i + 4 + len {
                            break String::from_utf8_lossy(&buf[i + 4..i + 4 + len]).to_string();
                        }
                    }
                    if n == 0 {
                        break String::new();
                    }
                };
                seen2.lock().unwrap().push(serde_json::from_str(&body).unwrap_or_default());
                let tcs: Vec<serde_json::Value> = calls
                    .iter()
                    .enumerate()
                    .map(|(i, (name, args))| serde_json::json!({ "index": i, "id": format!("call{i}"), "function": { "name": name, "arguments": args } }))
                    .collect();
                let mut delta = serde_json::json!({ "content": text });
                if !tcs.is_empty() {
                    delta["tool_calls"] = tcs.into();
                }
                let event = serde_json::json!({ "choices": [{ "delta": delta, "finish_reason": finish }] });
                let sse = format!("data: {event}\n\ndata: [DONE]\n\n");
                let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", sse.len());
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(sse.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        (port, seen)
    }

    #[derive(Default)]
    struct Host {
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl ToolHost for Host {
        async fn grants(&self, _t: &str) -> AppResult<Vec<Grant>> {
            Ok(vec![])
        }
        async fn add_grant(&self, _s: GrantScope, _t: &str, _c: Option<&str>) -> AppResult<()> {
            Ok(())
        }
        async fn current_policy(&self, _t: &ExposedTool) -> AppResult<(bool, Policy, bool)> {
            Ok((true, Policy::Allow, true))
        }
        async fn request_approval(&self, _r: ApprovalRequest, _c: &CancellationToken) -> ApprovalAnswer {
            ApprovalAnswer::Deny
        }
        async fn audit_start(&self, _t: &ExposedTool, _c: &ToolCall, _a: &str, _conv: Option<&str>) -> AppResult<String> {
            Ok("a".into())
        }
        async fn audit_status(&self, _id: &str, _s: &str, _a: &serde_json::Value) -> AppResult<()> {
            Ok(())
        }
        async fn audit_end(&self, _id: &str, _s: &str, _r: Option<&str>, _a: &serde_json::Value, _ms: u64) -> AppResult<()> {
            Ok(())
        }
        async fn call(&self, _t: &ExposedTool, _call: &ToolCall, args: serde_json::Value, _to: Duration, _c: &CancellationToken) -> AppResult<CallResult> {
            self.calls.lock().unwrap().push(args.to_string());
            Ok(CallResult { text: format!("content of {}", args["path"].as_str().unwrap_or("")), is_error: false })
        }
        fn event(&self, _ev: ToolEvent) {}
    }

    struct Hooks;

    #[async_trait]
    impl LoopHooks for Hooks {
        async fn begin_segment(&self, _s: usize) -> AppResult<()> {
            Ok(())
        }
        fn on_delta(&self, _s: usize, _d: &ChatDelta) {}
        async fn on_segment(&self, _s: usize, _c: &str, _r: &str, _t: &[ToolCall], _st: &ChatStats) -> AppResult<String> {
            Ok("m".into())
        }
        async fn on_tool_result(&self, _c: &ToolCall, _o: &DispatchOutcome) -> AppResult<()> {
            Ok(())
        }
    }

    struct Saver;

    #[async_trait]
    impl TaskObserver for Saver {
        async fn checkpoint(&self, _t: &[serde_json::Value], _u: &Usage) -> AppResult<()> {
            Ok(())
        }
    }

    fn read_tool() -> Vec<ExposedTool> {
        vec![ExposedTool {
            tool_id: "builtin:local/fs_read".into(),
            model_name: "fs_read".into(),
            server_id: "builtin:local".into(),
            server_name: "This PC".into(),
            tool_name: "fs_read".into(),
            description: "read a file".into(),
            input_schema: serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
            policy: Policy::Allow,
            timeout_secs: 5,
        }]
    }

    async fn run(port: u16, host: &Host, transcript: &mut Vec<serde_json::Value>, budgets: &Budgets, usage: &mut Usage) -> TaskEnd {
        let client = LlamaClient::new(port);
        let tools = read_tool();
        let params = SamplingParams::default();
        let cancel = CancellationToken::new();
        let ctx = TurnCtx { client: &client, tools: &tools, params: &params, host, hooks: &Hooks, scope: Some("run"), cancel: &cancel, mode: crate::agent_loop::ToolMode::Native, gate: None };
        run_task(&ctx, transcript, budgets, usage, &CancellationToken::new(), &Saver).await
    }

    #[tokio::test]
    async fn max_turns_is_honoured_after_parallel_calls_and_resume_finishes() {
        let three = vec![("fs_read", r#"{"path":"f1.txt"}"#), ("fs_read", r#"{"path":"f2.txt"}"#), ("fs_read", r#"{"path":"f3.txt"}"#)];
        let (port, seen) = model_server(vec![("", three, "tool_calls"), ("f1, f2 and f3 read.", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "read three files" })];
        let budgets = Budgets::limited(1, 40, 900, 40_000);
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &budgets, &mut usage).await;
        assert_eq!(end, TaskEnd::BudgetExhausted("model turns"));
        assert_eq!(seen.lock().unwrap().len(), 1, "one model turn only");
        assert_eq!(host.calls.lock().unwrap().len(), 3, "the turn's three calls ran");
        assert_eq!((usage.turns, usage.tool_calls), (1, 3));
        assert_eq!(seen.lock().unwrap()[0]["max_tokens"], 40_000, "a reply never generates more than the token budget left");
        // Resumed with a bigger budget: the next turn answers.
        let more = Budgets { max_turns: Some(12), ..budgets };
        let end = run(port, &host, &mut transcript, &more, &mut usage).await;
        assert_eq!(end, TaskEnd::Completed { answer: "f1, f2 and f3 read.".into() });
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert!(seen.lock().unwrap()[1]["messages"].to_string().contains("content of f3.txt"), "the resumed turn sees the tool results");
    }

    #[tokio::test]
    async fn no_turn_limit_still_stops_at_the_backstop() {
        let replies: Vec<Reply> = (0..MAX_TURNS_BACKSTOP + 1).map(|_| ("", vec![("fs_read", r#"{"path":"a.txt"}"#)], "tool_calls")).collect();
        let (port, seen) = model_server(replies).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "loop" })];
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &Budgets::default(), &mut usage).await;
        assert_eq!(end, TaskEnd::BudgetExhausted(BACKSTOP_REASON));
        assert_eq!((usage.turns, seen.lock().unwrap().len()), (MAX_TURNS_BACKSTOP, MAX_TURNS_BACKSTOP as usize));
        assert!(seen.lock().unwrap()[0].get("max_tokens").is_none_or(|v| v.is_null()), "no token limit: no max_tokens cap is sent");
    }

    #[tokio::test]
    async fn tool_calls_of_a_cut_off_reply_never_run() {
        let runaway = vec![("fs_read", r#"{"path":"a.txt"}"#), ("fs_read", r#"{"path":"a.txt"}"#), ("fs_read", r#"{"path":"#)];
        let (port, seen) = model_server(vec![("Here is", runaway, "length"), ("Done without tools.", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &Budgets::default(), &mut usage).await;
        assert_eq!(end, TaskEnd::Completed { answer: "Done without tools.".into() });
        assert!(host.calls.lock().unwrap().is_empty(), "no call of the cut-off reply ran: {:?}", host.calls.lock().unwrap());
        let second = seen.lock().unwrap()[1]["messages"].to_string();
        assert!(second.contains("was cut off") && second.contains("none of its 3 tool call(s) ran"), "{second}");
        assert_eq!(usage.tool_calls, 1, "the refused reply counts as one call, so a looping model still ends");
    }

    #[tokio::test]
    async fn an_empty_reply_after_tool_results_is_nudged_once() {
        let read = vec![("fs_read", r#"{"path":"a.txt"}"#)];
        let (port, seen) = model_server(vec![("", read, "tool_calls"), ("", vec![], "stop"), ("a.txt says hello.", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "read a.txt" })];
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &Budgets::default(), &mut usage).await;
        assert_eq!(end, TaskEnd::Completed { answer: "a.txt says hello.".into() });
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3, "the empty reply was generated again");
        let last = seen[2]["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!((last["role"].as_str(), last["content"].as_str()), (Some("user"), Some(crate::agent_loop::NUDGE)), "{last}");
        assert_eq!(usage.turns, 2, "the nudge is part of the same turn");
    }

    #[tokio::test]
    async fn a_reply_still_empty_after_the_nudge_says_so() {
        let read = vec![("fs_read", r#"{"path":"a.txt"}"#)];
        let (port, seen) = model_server(vec![("", read, "tool_calls"), ("", vec![], "stop"), ("  ", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "read a.txt" })];
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &Budgets::default(), &mut usage).await;
        match end {
            TaskEnd::Failed(why) => assert!(why.starts_with(crate::agent_loop::NO_ANSWER_NOTE), "{why}"),
            other => panic!("expected a no-answer failure, got {other:?}"),
        }
        assert_eq!(seen.lock().unwrap().len(), 3, "nudged once only");
    }

    /// An observer that hands the loop scripted user messages (one batch per
    /// ask) and reports waiting time.
    struct Inbox {
        batches: Mutex<std::collections::VecDeque<Vec<String>>>,
        waited: std::sync::atomic::AtomicU64,
    }

    impl Inbox {
        fn new(batches: Vec<Vec<&str>>) -> Self {
            Self { batches: Mutex::new(batches.into_iter().map(|b| b.into_iter().map(String::from).collect()).collect()), waited: Default::default() }
        }
    }

    #[async_trait]
    impl TaskObserver for Inbox {
        async fn checkpoint(&self, _t: &[serde_json::Value], _u: &Usage) -> AppResult<()> {
            Ok(())
        }
        async fn take_user_messages(&self) -> Vec<String> {
            self.batches.lock().unwrap().pop_front().unwrap_or_default()
        }
        fn take_waited_ms(&self) -> u64 {
            self.waited.swap(0, std::sync::atomic::Ordering::SeqCst)
        }
    }

    async fn run_with(port: u16, host: &Host, transcript: &mut Vec<serde_json::Value>, usage: &mut Usage, obs: &Inbox) -> TaskEnd {
        let client = LlamaClient::new(port);
        let tools = read_tool();
        let params = SamplingParams::default();
        let cancel = CancellationToken::new();
        let ctx = TurnCtx { client: &client, tools: &tools, params: &params, host, hooks: &Hooks, scope: Some("run"), cancel: &cancel, mode: crate::agent_loop::ToolMode::Native, gate: None };
        run_task(&ctx, transcript, &Budgets::default(), usage, &CancellationToken::new(), obs).await
    }

    #[tokio::test]
    async fn a_message_sent_while_working_reaches_the_model_before_its_next_turn() {
        let (port, seen) = model_server(vec![("", vec![("fs_read", r#"{"path":"a.txt"}"#)], "tool_calls"), ("Used blue.", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        // Nothing before turn 1; the message arrives while the tool ran.
        let obs = Inbox::new(vec![vec![], vec!["use blue"], vec![]]);
        let end = run_with(port, &host, &mut transcript, &mut Usage::default(), &obs).await;
        assert_eq!(end, TaskEnd::Completed { answer: "Used blue.".into() });
        let seen = seen.lock().unwrap();
        let second = seen[1]["messages"].as_array().unwrap();
        let roles: Vec<&str> = second.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"], "the message follows the tool result");
        assert_eq!(second.last().unwrap()["content"], "Message from the user (their own words): use blue");
        assert!(!seen[0]["messages"].to_string().contains("use blue"), "not before it was sent");
    }

    #[tokio::test]
    async fn a_message_sent_during_the_last_answer_is_not_lost() {
        let (port, seen) = model_server(vec![("First answer.", vec![], "stop"), ("Second answer, with your note.", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        let obs = Inbox::new(vec![vec![], vec!["one more thing"]]);
        let end = run_with(port, &host, &mut transcript, &mut Usage::default(), &obs).await;
        assert_eq!(end, TaskEnd::Completed { answer: "Second answer, with your note.".into() });
        let msgs = seen.lock().unwrap()[1]["messages"].as_array().unwrap().clone();
        assert_eq!(msgs[msgs.len() - 2]["content"], "First answer.");
        assert!(msgs.last().unwrap()["content"].as_str().unwrap().ends_with("one more thing"));
    }

    /// Reports running subagents once, then none.
    struct Kids {
        texts: Mutex<Vec<String>>,
        asked: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl TaskObserver for Kids {
        async fn checkpoint(&self, _t: &[serde_json::Value], _u: &Usage) -> AppResult<()> {
            Ok(())
        }
        async fn wait_children(&self, _remaining: Duration) -> Option<String> {
            self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.texts.lock().unwrap().pop()
        }
    }

    #[tokio::test]
    async fn a_final_answer_with_running_subagents_waits_and_continues() {
        let (port, seen) = model_server(vec![("All done.", vec![], "stop"), ("All done, and the subagent found 42.", vec![], "stop")]).await;
        let client = LlamaClient::new(port);
        let tools = read_tool();
        let params = SamplingParams::default();
        let cancel = CancellationToken::new();
        let host = Host::default();
        let ctx = TurnCtx { client: &client, tools: &tools, params: &params, host: &host, hooks: &Hooks, scope: Some("run"), cancel: &cancel, mode: crate::agent_loop::ToolMode::Native, gate: None };
        let obs = Kids { texts: Mutex::new(vec!["Your subagents finished:\n### Subagent \"w\" (completed)\n42".into()]), asked: Default::default() };
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        let end = run_task(&ctx, &mut transcript, &Budgets::default(), &mut Usage::default(), &CancellationToken::new(), &obs).await;
        assert_eq!(end, TaskEnd::Completed { answer: "All done, and the subagent found 42.".into() });
        assert_eq!(obs.asked.load(std::sync::atomic::Ordering::SeqCst), 2, "asked again after the results arrived");
        let second = seen.lock().unwrap()[1]["messages"].as_array().unwrap().clone();
        assert_eq!(second[second.len() - 2], serde_json::json!({ "role": "assistant", "content": "All done." }));
        assert!(second.last().unwrap()["content"].as_str().unwrap().contains("Your subagents finished") && second.last().unwrap()["role"] == "user");
    }

    #[test]
    fn messages_join_a_trailing_user_message_and_keep_their_order() {
        let mut t = vec![serde_json::json!({ "role": "system", "content": "s" }), serde_json::json!({ "role": "user", "content": "objective" })];
        inject_user_messages(&mut t, &["first".into(), "second".into()]);
        assert_eq!(t.len(), 2, "joined the objective, no two user messages in a row");
        let c = t[1]["content"].as_str().unwrap();
        assert!(c.starts_with("objective") && c.find("first").unwrap() < c.find("second").unwrap(), "{c}");
        let mut t = vec![serde_json::json!({ "role": "assistant", "content": "a" })];
        inject_user_messages(&mut t, &["hi".into()]);
        assert_eq!(t[1], serde_json::json!({ "role": "user", "content": "Message from the user (their own words): hi" }));
        inject_user_messages(&mut t, &[]);
        assert_eq!(t.len(), 2);
    }

    #[tokio::test]
    async fn time_spent_waiting_for_the_user_does_not_count() {
        let (port, _seen) = model_server(vec![("done", vec![], "stop")]).await;
        let host = Host::default();
        let mut transcript = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        let obs = Inbox::new(vec![]);
        obs.waited.store(3_600_000, std::sync::atomic::Ordering::SeqCst);
        let mut usage = Usage::default();
        run_with(port, &host, &mut transcript, &mut usage, &obs).await;
        assert!(usage.wall_ms < 60_000, "an hour of waiting is not working time: {}", usage.wall_ms);
    }

    #[test]
    fn cancelling_a_parent_reaches_every_child() {
        let root = CancellationToken::new();
        let a = root.child_token();
        let b = root.child_token();
        let grandchild = a.child_token();
        root.cancel();
        assert!(a.is_cancelled() && b.is_cancelled() && grandchild.is_cancelled());
        // A child can be stopped alone without touching its parent.
        let root2 = CancellationToken::new();
        let c = root2.child_token();
        c.cancel();
        assert!(!root2.is_cancelled());
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Budgets {
    /// Model turns (assistant segments).
    pub max_turns: u32,
    pub max_tool_calls: u32,
    /// Active time (paused time does not count).
    pub max_wall_secs: u64,
    /// Completion tokens generated.
    pub max_tokens: u64,
}

impl Default for Budgets {
    fn default() -> Self {
        Self { max_turns: 40, max_tool_calls: 60, max_wall_secs: 1800, max_tokens: 60_000 }
    }
}

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

impl Budgets {
    /// Which budget is used up, if any.
    pub fn exhausted(&self, u: &Usage) -> Option<&'static str> {
        if u.turns >= self.max_turns {
            Some("model turns")
        } else if u.tokens >= self.max_tokens {
            Some("tokens")
        } else if u.wall_ms >= self.max_wall_secs * 1000 {
            Some("time")
        } else {
            None
        }
    }

    pub fn remaining(&self, u: &Usage) -> Budgets {
        Budgets {
            max_turns: self.max_turns.saturating_sub(u.turns),
            max_tool_calls: self.max_tool_calls.saturating_sub(u.tool_calls),
            max_wall_secs: self.max_wall_secs.saturating_sub(u.wall_ms / 1000),
            max_tokens: self.max_tokens.saturating_sub(u.tokens),
        }
    }

    /// A child's budget: what it asked for, capped by `fraction` of what the
    /// parent has left. No floors: children together never get more than the
    /// parent has (with `fraction` = 1 / number of children or less). Check
    /// the result with `too_small` before starting a child.
    pub fn share(&self, used: &Usage, fraction: f64, asked: Option<&Budgets>) -> Budgets {
        let left = self.remaining(used);
        let f = fraction.clamp(0.0, 1.0);
        let cap = |l: u64| (l as f64 * f) as u64;
        let capped = Budgets {
            max_turns: cap(left.max_turns as u64) as u32,
            max_tool_calls: cap(left.max_tool_calls as u64) as u32,
            max_wall_secs: cap(left.max_wall_secs),
            max_tokens: cap(left.max_tokens),
        };
        match asked {
            None => capped,
            Some(a) => Budgets {
                max_turns: a.max_turns.min(capped.max_turns),
                max_tool_calls: a.max_tool_calls.min(capped.max_tool_calls),
                max_wall_secs: a.max_wall_secs.min(capped.max_wall_secs),
                max_tokens: a.max_tokens.min(capped.max_tokens),
            },
        }
    }

    /// Why a child's share is too small to do anything useful (None = fine):
    /// at least 2 model turns (one to act, one to answer), 1 tool call,
    /// 10 seconds and 500 tokens.
    pub fn too_small(&self) -> Option<&'static str> {
        if self.max_turns < 2 {
            Some("model turns")
        } else if self.max_tool_calls < 1 {
            Some("tool calls")
        } else if self.max_wall_secs < 10 {
            Some("time")
        } else if self.max_tokens < 500 {
            Some("tokens")
        } else {
            None
        }
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
        if let Err(e) = observer.prepare(transcript, usage).await {
            return TaskEnd::Failed(e.to_string());
        }
        if let Some(which) = budgets.exhausted(usage) {
            return TaskEnd::BudgetExhausted(which);
        }
        let left = budgets.remaining(usage);
        let offer_tools = left.max_tool_calls > 0;
        // One reply never generates more than the token budget has left.
        let mut params = ctx.params.clone();
        let cap = u32::try_from(left.max_tokens).unwrap_or(u32::MAX);
        params.max_tokens = Some(params.max_tokens.map_or(cap, |m| m.min(cap)));
        let capped = TurnCtx { params: &params, ..*ctx };
        let t0 = Instant::now();
        let t = turn(&capped, transcript, segment, offer_tools, left.max_tool_calls, Duration::from_secs(left.max_wall_secs.max(1))).await;
        usage.turns += 1;
        usage.tool_calls += t.calls_counted;
        usage.wall_ms += t0.elapsed().as_millis() as u64;
        usage.tokens += t.stats.completion_tokens.unwrap_or(0);
        segment += 1;
        if let Err(e) = observer.checkpoint(transcript, usage).await {
            return TaskEnd::Failed(format!("could not save progress: {e}"));
        }
        match t.end {
            TurnEnd::Final => {
                // Keep the final answer in the transcript too (resume, history).
                transcript.push(serde_json::json!({ "role": "assistant", "content": t.content }));
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
        let b = Budgets { max_turns: 10, max_tool_calls: 30, max_wall_secs: 600, max_tokens: 9000 };
        let mut u = Usage { turns: 4, tool_calls: 12, wall_ms: 60_000, tokens: 3000 };
        assert_eq!(b.exhausted(&u), None);
        // A plain third of what is left (6 turns, 18 calls, 540 s, 6000 tokens).
        // Turns used to be raised to a floor of 3, so three children could get
        // 9 turns between them out of 6: shares have no floors any more.
        let child = b.share(&u, 1.0 / 3.0, None);
        assert_eq!((child.max_turns, child.max_tool_calls, child.max_wall_secs, child.max_tokens), (2, 6, 180, 2000));
        assert_eq!(child.too_small(), None);
        // A child asking for more than its share gets the share; less is kept.
        let asked = Budgets { max_turns: 100, max_tool_calls: 2, max_wall_secs: 10_000, max_tokens: 100 };
        let c2 = b.share(&u, 1.0 / 3.0, Some(&asked));
        assert_eq!((c2.max_turns, c2.max_tool_calls, c2.max_tokens), (2, 2, 100));
        // Children together never get more than the parent has left.
        for n in 1..=6u32 {
            let f = 1.0 / (n.max(3) as f64);
            let one = b.share(&u, f, None);
            let left = b.remaining(&u);
            assert!(one.max_turns * n <= left.max_turns && one.max_tool_calls * n <= left.max_tool_calls, "{n}: {one:?}");
            assert!(one.max_wall_secs * n as u64 <= left.max_wall_secs && one.max_tokens * n as u64 <= left.max_tokens, "{n}: {one:?}");
        }
        // Too little left to share: the caller refuses to start children.
        let low = Usage { turns: 8, tool_calls: 28, wall_ms: 580_000, tokens: 8000 };
        assert_eq!(b.share(&low, 1.0 / 3.0, None).too_small(), Some("model turns"));
        let late = Usage { turns: 0, tool_calls: 0, wall_ms: 590_000, tokens: 0 };
        assert_eq!(b.share(&late, 1.0 / 3.0, None).too_small(), Some("time"));
        // Never more than the parent has left.
        let tight = Usage { turns: 9, tool_calls: 29, wall_ms: 590_000, tokens: 8900 };
        let c3 = b.share(&tight, 1.0, None);
        assert!(c3.max_turns <= 1 && c3.max_tool_calls <= 1 && c3.max_wall_secs <= 10 && c3.max_tokens <= 100, "{c3:?}");
        // Roll-up.
        u.add(&Usage { turns: 2, tool_calls: 3, wall_ms: 99_999, tokens: 500 });
        assert_eq!((u.turns, u.tool_calls, u.wall_ms, u.tokens), (6, 15, 60_000, 3500));
        u.turns = 10;
        assert_eq!(b.exhausted(&u), Some("model turns"));
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
        let budgets = Budgets { max_turns: 1, max_tool_calls: 40, max_wall_secs: 900, max_tokens: 40_000 };
        let mut usage = Usage::default();
        let end = run(port, &host, &mut transcript, &budgets, &mut usage).await;
        assert_eq!(end, TaskEnd::BudgetExhausted("model turns"));
        assert_eq!(seen.lock().unwrap().len(), 1, "one model turn only");
        assert_eq!(host.calls.lock().unwrap().len(), 3, "the turn's three calls ran");
        assert_eq!((usage.turns, usage.tool_calls), (1, 3));
        assert_eq!(seen.lock().unwrap()[0]["max_tokens"], 40_000, "a reply never generates more than the token budget left");
        // Resumed with a bigger budget: the next turn answers.
        let more = Budgets { max_turns: 12, ..budgets };
        let end = run(port, &host, &mut transcript, &more, &mut usage).await;
        assert_eq!(end, TaskEnd::Completed { answer: "f1, f2 and f3 read.".into() });
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert!(seen.lock().unwrap()[1]["messages"].to_string().contains("content of f3.txt"), "the resumed turn sees the tool results");
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

//! Deferred tool loading. With many tools (an MCP server with more than
//! [`SERVER_THRESHOLD`] tools, or more than [`TOTAL_THRESHOLD`] in all), the
//! big servers' tools are offered as a short name index in the prompt, and
//! the model asks for the full definitions it needs with the `load_tools`
//! meta-tool. Loaded tools stay offered for the rest of the conversation
//! (never removed, so the prompt prefix only grows). Every call is still
//! validated against the tool's real schema by the dispatcher.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Mutex;

use crate::tools::{BUILTIN_PREFIX, ExposedTool};

/// Name the model calls to load tool definitions.
pub const LOAD_TOOLS: &str = "load_tools";
/// A server with more tools than this is deferred.
pub const SERVER_THRESHOLD: usize = 30;
/// More tools than this in all: the biggest servers are deferred until the rest fits.
pub const TOTAL_THRESHOLD: usize = 60;
/// Most tools one `load_tools` call loads.
pub const MAX_LOAD: usize = 20;
/// Tools a search query loads.
pub const SEARCH_HITS: usize = 5;
/// `server_id` prefix of the app's own agent tools (never deferred).
pub(crate) const APP_PREFIX: &str = "agent:";

/// May this tool be deferred? Only MCP tools: built-ins, the app's agent
/// tools and `load_tools` itself are always sent in full.
pub fn is_deferrable(t: &ExposedTool) -> bool {
    !t.server_id.starts_with(BUILTIN_PREFIX) && !t.server_id.starts_with(APP_PREFIX) && t.model_name != LOAD_TOOLS
}

/// Which servers to defer, from (server id, tool count) of the deferrable
/// servers and the total tool count. Servers over the per-server threshold
/// first, then the biggest remaining ones while the total is still over the
/// total threshold. Deterministic (count, then id).
pub fn deferred_servers(groups: &[(String, usize)], total: usize) -> Vec<String> {
    if !groups.iter().any(|(_, n)| *n > SERVER_THRESHOLD) && total <= TOTAL_THRESHOLD {
        return vec![];
    }
    let mut out: Vec<String> = groups.iter().filter(|(_, n)| *n > SERVER_THRESHOLD).map(|(id, _)| id.clone()).collect();
    let mut left = total - groups.iter().filter(|(_, n)| *n > SERVER_THRESHOLD).map(|(_, n)| n).sum::<usize>();
    let mut rest: Vec<&(String, usize)> = groups.iter().filter(|(_, n)| *n <= SERVER_THRESHOLD).collect();
    rest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (id, n) in rest {
        if left <= TOTAL_THRESHOLD {
            break;
        }
        out.push(id.clone());
        left -= n;
    }
    out
}

/// The model names to defer for this tool list (empty = send everything).
pub fn pick_deferred(tools: &[ExposedTool]) -> Vec<String> {
    let mut groups: BTreeMap<&str, usize> = BTreeMap::new();
    for t in tools.iter().filter(|t| is_deferrable(t)) {
        *groups.entry(t.server_id.as_str()).or_default() += 1;
    }
    let groups: Vec<(String, usize)> = groups.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let total = tools.iter().filter(|t| t.model_name != LOAD_TOOLS).count();
    let servers: HashSet<String> = deferred_servers(&groups, total).into_iter().collect();
    tools.iter().filter(|t| is_deferrable(t) && servers.contains(&t.server_id)).map(|t| t.model_name.clone()).collect()
}

/// Input schema of `load_tools`.
pub fn load_tools_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "names": { "type": "array", "items": { "type": "string" }, "maxItems": MAX_LOAD, "description": "Exact tool names from the \"More tools\" list" },
            "query": { "type": "string", "description": "A few words to search the listed tools' names and descriptions for (loads the best matches)" }
        },
        "additionalProperties": false
    })
}

pub const LOAD_TOOLS_DESCRIPTION: &str = "Load the full definitions of tools from the \"More tools\" list in your instructions so you can call them. Give exact names, or a search query. Loaded tools stay available for the rest of this conversation.";

#[derive(Debug, Clone)]
struct Entry {
    name: String,
    tool_id: String,
    description: String,
    schema: serde_json::Value,
    /// Everything about the tool that can change on a reconnect (name, id,
    /// description, schema, policy, timeout), as text.
    sig: String,
}

type OnLoad = Box<dyn Fn(Vec<String>) + Send + Sync>;

/// One conversation's (or agent's) deferred tools and which were loaded.
pub struct ToolOffer {
    /// Deferred tools, in tool order.
    entries: Vec<Entry>,
    /// Loaded model names, in load order.
    loaded: Mutex<Vec<String>>,
    /// XML mode: load_tools returns the definitions as text.
    xml: bool,
    /// Called with the tool ids newly loaded (to persist them).
    on_load: Mutex<Option<OnLoad>>,
}

impl std::fmt::Debug for ToolOffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolOffer").field("deferred", &self.entries.len()).field("loaded", &*self.loaded.lock().unwrap()).field("xml", &self.xml).finish()
    }
}

/// What one load request did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadReport {
    pub loaded: Vec<String>,
    pub already: Vec<String>,
    pub unknown: Vec<String>,
}

/// Fingerprint text of a tool's full definition.
fn tool_sig(t: &ExposedTool) -> String {
    format!("{}|{}|{}|{}|{:?}|{}", t.model_name, t.tool_id, t.description, t.input_schema, t.policy, t.timeout_secs)
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| w.len() >= 2).map(|w| w.to_ascii_lowercase()).collect()
}

/// First line (or sentence) of a description, at most `max` characters.
pub fn one_line(desc: &str, max: usize) -> String {
    let first = desc.trim().lines().next().unwrap_or("").trim();
    let first = match first.find(". ") {
        Some(i) => &first[..=i],
        None => first,
    };
    if first.chars().count() <= max {
        first.to_string()
    } else {
        format!("{}...", first.chars().take(max).collect::<String>().trim_end())
    }
}

impl ToolOffer {
    /// `deferred`: model names to defer (from [`pick_deferred`]); `loaded_ids`:
    /// tool ids loaded earlier in this conversation (kept, in that order).
    pub fn new(tools: &[ExposedTool], deferred: &[String], loaded_ids: &[String], xml: bool) -> Self {
        let want: HashSet<&str> = deferred.iter().map(String::as_str).collect();
        let entries: Vec<Entry> = tools
            .iter()
            .filter(|t| want.contains(t.model_name.as_str()))
            .map(|t| Entry { name: t.model_name.clone(), tool_id: t.tool_id.clone(), description: t.description.clone(), schema: t.input_schema.clone(), sig: tool_sig(t) })
            .collect();
        let mut loaded = vec![];
        for id in loaded_ids {
            if let Some(e) = entries.iter().find(|e| &e.tool_id == id)
                && !loaded.contains(&e.name)
            {
                loaded.push(e.name.clone());
            }
        }
        Self { entries, loaded: Mutex::new(loaded), xml, on_load: Mutex::new(None) }
    }

    pub fn set_on_load(&self, f: impl Fn(Vec<String>) + Send + Sync + 'static) {
        *self.on_load.lock().unwrap() = Some(Box::new(f));
    }

    pub fn xml(&self) -> bool {
        self.xml
    }

    /// Model names of the deferred tools, in tool order.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    pub fn is_deferred(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Deferred and not loaded yet: only its name is in the prompt.
    pub fn is_hidden(&self, name: &str) -> bool {
        self.is_deferred(name) && !self.loaded.lock().unwrap().iter().any(|n| n == name)
    }

    /// Tool ids loaded so far, in load order.
    pub fn loaded_ids(&self) -> Vec<String> {
        let l = self.loaded.lock().unwrap();
        l.iter().filter_map(|n| self.entries.iter().find(|e| &e.name == n).map(|e| e.tool_id.clone())).collect()
    }

    /// The same deferred tools as `other`, in order, with identical full
    /// definitions (names, ids, descriptions, schemas, policies): a server
    /// that reconnects with a changed schema is a different catalog.
    pub fn same_catalog(&self, other: &ToolOffer) -> bool {
        self.entries.len() == other.entries.len() && self.entries.iter().zip(&other.entries).all(|(a, b)| a.sig == b.sig)
    }

    /// The tools whose full definitions are sent: everything not deferred, in
    /// tool order, then the loaded ones in load order (so loading appends).
    pub fn sent(&self, tools: &[ExposedTool]) -> Vec<ExposedTool> {
        let loaded = self.loaded.lock().unwrap().clone();
        let mut out: Vec<ExposedTool> = tools.iter().filter(|t| !self.is_deferred(&t.model_name)).cloned().collect();
        for n in &loaded {
            if let Some(t) = tools.iter().find(|t| &t.model_name == n) {
                out.push(t.clone());
            }
        }
        out
    }

    /// Load tools by exact model name (or tool id). Newly loaded ones are
    /// reported to `on_load`.
    pub fn load(&self, names: &[String]) -> LoadReport {
        let mut r = LoadReport::default();
        let mut new_ids = vec![];
        {
            let mut loaded = self.loaded.lock().unwrap();
            for n in names.iter().map(|n| n.trim()).filter(|n| !n.is_empty()) {
                let Some(e) = self.entries.iter().find(|e| e.name == n || e.tool_id == n) else {
                    if !r.unknown.iter().any(|u| u == n) {
                        r.unknown.push(n.to_string());
                    }
                    continue;
                };
                if loaded.contains(&e.name) {
                    if !r.already.contains(&e.name) {
                        r.already.push(e.name.clone());
                    }
                } else {
                    loaded.push(e.name.clone());
                    r.loaded.push(e.name.clone());
                    new_ids.push(e.tool_id.clone());
                }
            }
        }
        if !new_ids.is_empty()
            && let Some(f) = self.on_load.lock().unwrap().as_ref()
        {
            f(new_ids);
        }
        r
    }

    /// Best matches for a query among the deferred tools (name words weigh 2,
    /// description words 1), at most `max`.
    pub fn search(&self, query: &str, max: usize) -> Vec<String> {
        let want: BTreeSet<String> = words(query).into_iter().collect();
        if want.is_empty() {
            return vec![];
        }
        let mut scored: Vec<(usize, usize, &str)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let name: HashSet<String> = words(&e.name).into_iter().collect();
                let desc: HashSet<String> = words(&e.description).into_iter().collect();
                let score: usize = want.iter().map(|w| if name.contains(w) { 2 } else if desc.contains(w) { 1 } else { 0 }).sum();
                (score > 0).then_some((score, i, e.name.as_str()))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(max).map(|(_, _, n)| n.to_string()).collect()
    }

    /// The prompt section: one short paragraph, then one line per deferred tool.
    pub fn index_section(&self) -> String {
        let how = if self.xml {
            "load_tools returns its description and arguments"
        } else {
            "its full definition is then sent with every later request"
        };
        let mut s = format!(
            "## More tools (load before use)\nThese {} tools are available, but only their names are listed here to keep this prompt short. To use one, first call {LOAD_TOOLS} with its exact name (up to {MAX_LOAD} at once) or a few search words; {how}, and it stays available for the rest of this conversation. Every call is checked against the tool's real definition.\n",
            self.entries.len()
        );
        for e in &self.entries {
            let d = one_line(&e.description.replace('\n', " "), 100);
            if d.is_empty() {
                s.push_str(&format!("- {}\n", e.name));
            } else {
                s.push_str(&format!("- {}: {d}\n", e.name));
            }
        }
        s.trim_end().to_string()
    }

    /// Run a `load_tools` call (arguments already validated against
    /// [`load_tools_schema`]). Returns the result text and whether it is an error.
    pub fn run_load_tools(&self, args: &serde_json::Value) -> (String, bool) {
        let mut names: Vec<String> = args["names"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let query = args["query"].as_str().unwrap_or("").trim().to_string();
        if names.is_empty() && query.is_empty() {
            return ("Give \"names\" (exact tool names from the More tools list) or a \"query\".".into(), true);
        }
        let mut matched = vec![];
        if !query.is_empty() {
            matched = self.search(&query, SEARCH_HITS);
            for m in &matched {
                if !names.contains(m) {
                    names.push(m.clone());
                }
            }
        }
        names.truncate(MAX_LOAD);
        let r = self.load(&names);
        let mut out = vec![];
        if !r.loaded.is_empty() {
            out.push(format!("Loaded {}: {}.", r.loaded.len(), r.loaded.join(", ")));
        }
        if !r.already.is_empty() {
            out.push(format!("Already loaded: {}.", r.already.join(", ")));
        }
        if !r.unknown.is_empty() {
            out.push(format!("Not in the More tools list: {}. Use the exact names listed there.", r.unknown.join(", ")));
        }
        if !query.is_empty() && matched.is_empty() {
            out.push(format!("No listed tool matched \"{query}\"."));
        }
        let usable: Vec<&String> = r.loaded.iter().chain(&r.already).collect();
        if usable.is_empty() {
            return (out.join(" "), true);
        }
        if self.xml {
            out.push("Their definitions:".into());
            for n in usable {
                if let Some(e) = self.entries.iter().find(|e| &e.name == n) {
                    out.push(format!("\n### {}\n{}\nArguments (JSON Schema): {}", e.name, e.description.trim(), e.schema));
                }
            }
        } else {
            out.push("Their full definitions are sent from your next step on; call them directly.".into());
        }
        (out.join("\n"), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use security::policy::Policy;

    fn tool(server: &str, name: &str) -> ExposedTool {
        ExposedTool {
            tool_id: format!("mcp:{server}/{name}"),
            model_name: format!("mcp__{server}__{name}"),
            server_id: server.into(),
            server_name: server.into(),
            tool_name: name.into(),
            description: format!("Does {name} things.\nMore text."),
            input_schema: serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
            policy: Policy::Allow,
            timeout_secs: 60,
        }
    }

    fn many(server: &str, n: usize) -> Vec<ExposedTool> {
        (0..n).map(|i| tool(server, &format!("t{i}"))).collect()
    }

    #[test]
    fn thresholds_pick_big_servers_only() {
        let mut tools = many("small", 10);
        assert!(pick_deferred(&tools).is_empty(), "10 tools: nothing deferred");
        tools.extend(many("big", 31));
        let d = pick_deferred(&tools);
        assert_eq!(d.len(), 31);
        assert!(d.iter().all(|n| n.starts_with("mcp__big__")));
        // Exactly 30 on one server and under 60 in all: nothing.
        assert!(pick_deferred(&many("edge", 30)).is_empty());
        // Built-ins are never deferred, even with a big total.
        let mut b: Vec<ExposedTool> = (0..12).map(|i| ExposedTool { server_id: "builtin:local".into(), model_name: format!("fs_{i}"), ..tool("x", "y") }).collect();
        b.extend(many("a", 25));
        b.extend(many("b", 20));
        b.extend(many("c", 10));
        // 67 in all, no server over 30: the biggest server (a) is deferred until <= 60.
        let d = pick_deferred(&b);
        assert!(d.iter().all(|n| n.starts_with("mcp__a__")) && d.len() == 25, "{d:?}");
        assert_eq!(deferred_servers(&[("a".into(), 25), ("b".into(), 25)], 70), vec!["a".to_string()], "ties: by id");
    }

    #[test]
    fn load_search_and_sent_order_with_persisted_ids() {
        let mut tools = many("small", 2);
        tools.extend(many("big", 40));
        let d = pick_deferred(&tools);
        let saved = Mutex::new(vec![]);
        let saved = std::sync::Arc::new(saved);
        let o = ToolOffer::new(&tools, &d, &["mcp:big/t5".into(), "mcp:gone/x".into()], false);
        let s2 = saved.clone();
        o.set_on_load(move |ids| s2.lock().unwrap().extend(ids));
        assert!(!o.is_hidden("mcp__big__t5") && o.is_hidden("mcp__big__t6") && !o.is_hidden("mcp__small__t0"));
        let names = |v: Vec<ExposedTool>| v.into_iter().map(|t| t.model_name).collect::<Vec<_>>();
        assert_eq!(names(o.sent(&tools)), vec!["mcp__small__t0", "mcp__small__t1", "mcp__big__t5"]);
        let r = o.load(&["mcp__big__t9".into(), "mcp__big__t5".into(), "nope".into()]);
        assert_eq!(r, LoadReport { loaded: vec!["mcp__big__t9".into()], already: vec!["mcp__big__t5".into()], unknown: vec!["nope".into()] });
        // Appended after the earlier load, never reordered or removed.
        assert_eq!(names(o.sent(&tools)), vec!["mcp__small__t0", "mcp__small__t1", "mcp__big__t5", "mcp__big__t9"]);
        assert_eq!(*saved.lock().unwrap(), vec!["mcp:big/t9".to_string()], "only new ids are persisted");
        assert_eq!(o.loaded_ids(), vec!["mcp:big/t5".to_string(), "mcp:big/t9".to_string()]);
        // Search by name words.
        assert_eq!(o.search("t12 tool", 5), vec!["mcp__big__t12".to_string()]);
        let (text, err) = o.run_load_tools(&serde_json::json!({ "query": "t12" }));
        assert!(!err && text.contains("Loaded 1: mcp__big__t12") && text.contains("next step"), "{text}");
        let (text, err) = o.run_load_tools(&serde_json::json!({}));
        assert!(err && text.contains("names"), "{text}");
        let (text, err) = o.run_load_tools(&serde_json::json!({ "names": ["zzz"] }));
        assert!(err && text.contains("Not in the More tools list"), "{text}");
        // The index lists every deferred tool (loaded or not), one line each.
        let idx = o.index_section();
        assert!(idx.starts_with("## More tools") && idx.contains("- mcp__big__t9: Does t9 things.") && idx.contains("- mcp__big__t39:"), "{idx}");
        assert_eq!(idx.lines().count(), 2 + 40);
        assert!(!idx.contains("mcp__small__"));
        // Same catalog check.
        assert!(o.same_catalog(&ToolOffer::new(&tools, &d, &[], false)));
        assert!(!o.same_catalog(&ToolOffer::new(&tools, &d[1..], &[], false)));
        let mut changed = tools.clone();
        changed[3].input_schema = serde_json::json!({ "type": "object", "required": ["x"] });
        assert!(!o.same_catalog(&ToolOffer::new(&changed, &d, &[], false)), "a changed schema is a new catalog");
    }

    #[test]
    fn xml_mode_returns_the_definitions() {
        let tools = many("big", 31);
        let o = ToolOffer::new(&tools, &pick_deferred(&tools), &[], true);
        let (text, err) = o.run_load_tools(&serde_json::json!({ "names": ["mcp__big__t3"] }));
        assert!(!err && text.contains("### mcp__big__t3") && text.contains("Arguments (JSON Schema)") && text.contains("\"required\":[\"path\"]"), "{text}");
        assert!(o.index_section().contains("returns its description and arguments"));
    }

    // ---- the whole flow through the agent loop, against a fake model server ----

    use crate::dispatch::{ApprovalRequest, CallResult, DispatchOutcome, ToolEvent, ToolHost};
    use crate::{LoopHooks, Limits, ToolMode, run_reply};
    use app_core::AppResult;
    use async_trait::async_trait;
    use inference::{ChatDelta, ChatStats, LlamaClient, SamplingParams, ToolCall};
    use security::policy::{ApprovalAnswer, Grant, GrantScope};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_util::sync::CancellationToken;

    /// Replies in order: (tool name, arguments) or a final text.
    async fn model_server(replies: Vec<Option<(&'static str, &'static str)>>) -> (u16, Arc<Mutex<Vec<serde_json::Value>>>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(vec![]));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            for r in replies {
                let Ok((mut sock, _)) = l.accept().await else { return };
                let mut buf = vec![];
                let body = loop {
                    let mut chunk = [0u8; 65536];
                    let n = sock.read(&mut chunk).await.unwrap_or(0);
                    buf.extend_from_slice(&chunk[..n]);
                    let s = String::from_utf8_lossy(&buf).to_string();
                    if let Some(i) = s.find("\r\n\r\n") {
                        let len = s[..i].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                        if buf.len() >= i + 4 + len {
                            break String::from_utf8_lossy(&buf[i + 4..i + 4 + len]).to_string();
                        }
                    }
                    if n == 0 {
                        break String::new();
                    }
                };
                seen2.lock().unwrap().push(serde_json::from_str(&body).unwrap_or_default());
                let (delta, finish) = match r {
                    Some((name, args)) => (serde_json::json!({ "content": "", "tool_calls": [{ "index": 0, "id": "c0", "function": { "name": name, "arguments": args } }] }), "tool_calls"),
                    None => (serde_json::json!({ "content": "done" }), "stop"),
                };
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

    struct Host {
        offer: Arc<ToolOffer>,
        calls: Mutex<Vec<(String, serde_json::Value)>>,
        results: Mutex<Vec<String>>,
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
        async fn call(&self, t: &ExposedTool, _call: &ToolCall, args: serde_json::Value, _to: Duration, _c: &CancellationToken) -> AppResult<CallResult> {
            self.calls.lock().unwrap().push((t.model_name.clone(), args.clone()));
            if t.model_name == LOAD_TOOLS {
                let (text, is_error) = self.offer.run_load_tools(&args);
                return Ok(CallResult { text, is_error });
            }
            Ok(CallResult { text: "ran".into(), is_error: false })
        }
        fn event(&self, _ev: ToolEvent) {}
        fn tool_offer(&self) -> Option<Arc<ToolOffer>> {
            Some(self.offer.clone())
        }
    }

    struct Hooks<'a>(&'a Host);

    #[async_trait]
    impl LoopHooks for Hooks<'_> {
        async fn begin_segment(&self, _s: usize) -> AppResult<()> {
            Ok(())
        }
        fn on_delta(&self, _s: usize, _d: &ChatDelta) {}
        async fn on_segment(&self, _s: usize, _c: &str, _r: &str, _t: &[ToolCall], _st: &ChatStats) -> AppResult<String> {
            Ok("m".into())
        }
        async fn on_tool_result(&self, _c: &ToolCall, o: &DispatchOutcome) -> AppResult<()> {
            self.0.results.lock().unwrap().push(format!("{}: {}", o.status, o.content));
            Ok(())
        }
    }

    fn sent_names(req: &serde_json::Value) -> Vec<String> {
        req["tools"].as_array().map(|a| a.iter().map(|t| t["function"]["name"].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default()
    }

    #[tokio::test]
    async fn load_tools_flow_validates_against_the_real_schema_and_keeps_the_prefix() {
        let mut tools = vec![ExposedTool { server_id: "builtin:local".into(), model_name: "fs_read".into(), tool_id: "builtin:local/fs_read".into(), ..tool("x", "fs_read") }];
        tools.extend(many("big", 31));
        tools.push(ExposedTool { tool_id: "agent:r/load_tools".into(), model_name: LOAD_TOOLS.into(), server_id: "agent:r".into(), server_name: "app".into(), tool_name: LOAD_TOOLS.into(), description: LOAD_TOOLS_DESCRIPTION.into(), input_schema: load_tools_schema(), policy: Policy::Allow, timeout_secs: 5 });
        let d = pick_deferred(&tools);
        assert_eq!(d.len(), 31);
        let persisted = Arc::new(Mutex::new(vec![]));
        let offer = Arc::new(ToolOffer::new(&tools, &d, &[], false));
        let p2 = persisted.clone();
        offer.set_on_load(move |ids| p2.lock().unwrap().extend(ids));
        let host = Host { offer: offer.clone(), calls: Mutex::new(vec![]), results: Mutex::new(vec![]) };
        let (port, seen) = model_server(vec![
            Some((LOAD_TOOLS, r#"{"names":["mcp__big__t3"]}"#)),
            Some(("mcp__big__t3", r#"{}"#)),
            Some(("mcp__big__t3", r#"{"path":"a"}"#)),
            Some(("mcp__big__t7", r#"{"path":"b"}"#)),
            None,
            None,
        ])
        .await;
        let client = LlamaClient::new(port);
        let msgs = vec![serde_json::json!({ "role": "system", "content": "sys" }), serde_json::json!({ "role": "user", "content": "go" })];
        let out = run_reply(&client, msgs, &tools, &SamplingParams::default(), &Limits::default(), &host, &Hooks(&host), Some("c"), &CancellationToken::new(), ToolMode::Native, None).await;
        assert_eq!(out.stop_reason, "stop", "{out:?}");
        let seen = seen.lock().unwrap().clone();
        assert_eq!(sent_names(&seen[0]), vec!["fs_read", LOAD_TOOLS], "first request: only built-ins and load_tools");
        assert_eq!(sent_names(&seen[1]), vec!["fs_read", LOAD_TOOLS, "mcp__big__t3"], "the loaded tool is appended");
        // A direct call to a listed tool loads it too (appended after t3).
        assert_eq!(sent_names(&seen[4]), vec!["fs_read", LOAD_TOOLS, "mcp__big__t3", "mcp__big__t7"]);
        // The earlier tool definitions are byte-identical between steps (stable prefix).
        let s1 = serde_json::to_string(&seen[1]["tools"]).unwrap();
        let s4 = serde_json::to_string(&seen[4]["tools"]).unwrap();
        assert!(s4.starts_with(&s1[..s1.len() - 1]), "definitions only grow at the end");
        let results = host.results.lock().unwrap().clone();
        assert!(results[0].starts_with("ok: Loaded 1: mcp__big__t3"), "{results:?}");
        assert!(results[1].starts_with("invalid_arguments:") && results[1].contains("path"), "checked against the real schema: {results:?}");
        let calls = host.calls.lock().unwrap().clone();
        assert_eq!(calls.iter().map(|c| c.0.as_str()).collect::<Vec<_>>(), vec![LOAD_TOOLS, "mcp__big__t3", "mcp__big__t7"], "the invalid call never ran");
        assert_eq!(*persisted.lock().unwrap(), vec!["mcp:big/t3".to_string(), "mcp:big/t7".to_string()]);
        // Next user turn in the same chat: a new offer from the persisted ids sends the same tools.
        let again = ToolOffer::new(&tools, &d, &offer.loaded_ids(), false);
        assert_eq!(crate::openai_tools(&again.sent(&tools)).to_string(), crate::openai_tools(&offer.sent(&tools)).to_string());
    }

    #[test]
    fn the_same_child_request_is_byte_identical_on_every_step() {
        let mut tools = vec![ExposedTool { server_id: "builtin:local".into(), model_name: "shell_execute".into(), tool_id: "builtin:local/shell_execute".into(), ..tool("x", "shell_execute") }];
        tools.extend(many("big", 40));
        let d = pick_deferred(&tools);
        let request = |o: &ToolOffer| {
            let sys = crate::tools_prompt(&tools, &crate::Limits::default(), crate::agent_loop::ToolMode::Native, Some(o));
            (sys, crate::openai_tools(&o.sent(&tools)).to_string())
        };
        let a = ToolOffer::new(&tools, &d, &[], false);
        let first = request(&a);
        // Step 2 of the same child (same offer) and a fresh offer for a sibling.
        assert_eq!(request(&a), first);
        assert_eq!(request(&ToolOffer::new(&tools, &d, &[], false)), first);
        // Loading appends to the tools and leaves the system prompt alone.
        a.load(&["mcp__big__t3".to_string()]);
        let after = request(&a);
        assert_eq!(after.0, first.0);
        assert!(after.1.starts_with(&first.1[..first.1.len() - 1]));
    }

    #[test]
    fn load_tools_schema_is_valid_and_strict() {
        let s = load_tools_schema();
        assert!(crate::validate_args(&s, &serde_json::json!({ "names": ["a"] })).is_ok());
        assert!(crate::validate_args(&s, &serde_json::json!({ "names": "a" })).is_err());
        assert!(crate::validate_args(&s, &serde_json::json!({ "other": 1 })).is_err());
    }
}

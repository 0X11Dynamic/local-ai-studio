//! Deferred tool loading in the app (see `agent_runtime::deferred`): the
//! `load_tools` tool, which chat / task / subagent has which offer, and
//! saving a chat's loaded tools in its settings so the prompt stays the same
//! on the next message.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use agent_runtime::deferred::{self, LOAD_TOOLS, ToolOffer};
use agent_runtime::{CallResult, ExposedTool};
use security::policy::Policy;

use crate::services::mcp::AGENT_PREFIX;
use crate::state::SharedState;

/// Live offers by scope key (see [`key`]). Chats keep theirs (their loads are
/// also saved in the chat settings); tasks and subagents only while they run.
static OFFERS: LazyLock<Mutex<Offers>> = LazyLock::new(Default::default);

/// An offer and when it was last used (a tick of [`CLOCK`]), for pruning the least recently used.
type Offers = HashMap<String, (Arc<ToolOffer>, u64)>;

static CLOCK: AtomicU64 = AtomicU64::new(0);

fn tick() -> u64 {
    CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Chat offers kept in memory (beyond this, the least recently used idle chats
/// are dropped; they rebuild theirs from their settings).
const MAX_CHAT_OFFERS: usize = 32;

/// The scope of a tool host: a subagent, else a task (or a chat's run), else a chat.
pub fn key(conversation_id: Option<&str>, run_id: Option<&str>, agent_id: Option<&str>) -> Option<String> {
    agent_id.map(|a| format!("agent:{a}")).or_else(|| run_id.map(|r| format!("run:{r}"))).or_else(|| conversation_id.map(|c| format!("chat:{c}")))
}

pub fn chat_key(conversation_id: &str) -> String {
    format!("chat:{conversation_id}")
}

/// The `load_tools` meta-tool (an app tool: runs without asking).
pub fn load_tools_tool(scope: &str) -> ExposedTool {
    ExposedTool {
        tool_id: format!("{AGENT_PREFIX}{scope}/{LOAD_TOOLS}"),
        model_name: LOAD_TOOLS.into(),
        server_id: format!("{AGENT_PREFIX}{scope}"),
        server_name: "Local AI Studio".into(),
        tool_name: LOAD_TOOLS.into(),
        description: deferred::LOAD_TOOLS_DESCRIPTION.into(),
        input_schema: deferred::load_tools_schema(),
        policy: Policy::Allow,
        timeout_secs: 30,
    }
}

pub fn is_load_tools(t: &ExposedTool) -> bool {
    t.server_id.starts_with(AGENT_PREFIX) && t.tool_name == LOAD_TOOLS
}

/// Decide deferred loading for these tools. Adds `load_tools` (or removes a
/// stale one) and returns the offer when tools are deferred; None = every
/// tool is sent in full (`enabled` off, or the tool set is small).
pub fn plan(tools: &mut Vec<ExposedTool>, enabled: bool, loaded_ids: &[String], xml: bool, scope: &str) -> Option<ToolOffer> {
    tools.retain(|t| !is_load_tools(t));
    if !enabled {
        return None;
    }
    let d = deferred::pick_deferred(tools);
    if d.is_empty() {
        return None;
    }
    tools.push(load_tools_tool(scope));
    Some(ToolOffer::new(tools, &d, loaded_ids, xml))
}

/// Register `offer` under `key`. When the same tools are already registered
/// (a later message in the same chat, or the context meter), the live offer
/// is kept, with any loads from `offer` added: loaded tools are never dropped.
pub fn install(st: &SharedState, key: &str, offer: ToolOffer) -> Arc<ToolOffer> {
    let mut map = OFFERS.lock().unwrap();
    prune(st, &mut map);
    install_in(&mut map, key, offer)
}

fn install_in(map: &mut Offers, key: &str, offer: ToolOffer) -> Arc<ToolOffer> {
    if let Some((old, used)) = map.get_mut(key) {
        *used = tick();
        if old.same_catalog(&offer) && old.xml() == offer.xml() {
            old.load(&offer.loaded_ids());
            return old.clone();
        }
        // Tools changed: carry the earlier loads over (those still offered).
        offer.load(&old.loaded_ids());
    }
    let a = Arc::new(offer);
    map.insert(key.to_string(), (a.clone(), tick()));
    a
}

/// The offer `install` would give for `offer`, without registering it or
/// changing the live one (the context meter runs on every typing pause and
/// must not add, replace or refresh offers). Loads of the live offer are
/// carried over, so the meter counts what the next message will send.
pub fn peek(key: &str, offer: ToolOffer) -> Arc<ToolOffer> {
    if let Some((old, _)) = OFFERS.lock().unwrap().get(key) {
        if old.same_catalog(&offer) && old.xml() == offer.xml() {
            return old.clone();
        }
        offer.load(&old.loaded_ids());
    }
    Arc::new(offer)
}

pub fn get(key: &str) -> Option<Arc<ToolOffer>> {
    OFFERS.lock().unwrap().get_mut(key).map(|(o, used)| {
        *used = tick();
        o.clone()
    })
}

/// Forget a scope's offer (deferred loading turned off, or the tools shrank).
pub fn remove(key: &str) {
    OFFERS.lock().unwrap().remove(key);
}

/// Drop offers of tasks and subagents that are no longer running, and the
/// least recently used idle chats beyond `MAX_CHAT_OFFERS`.
fn prune(st: &SharedState, map: &mut Offers) {
    let running: std::collections::HashSet<String> = st.tasks.lock().unwrap().keys().cloned().chain(st.tasks_starting.lock().unwrap().iter().cloned()).collect();
    let agents: std::collections::HashSet<String> = st.subagent_stops.lock().unwrap().keys().cloned().collect();
    map.retain(|k, _| match k.split_once(':') {
        Some(("run", id)) => running.contains(id),
        Some(("agent", id)) => agents.contains(id),
        _ => true,
    });
    let busy = st.generations.lock().unwrap();
    trim_chats(map, |id| busy.contains_key(id));
}

/// Keep at most `MAX_CHAT_OFFERS` chat offers: drop the least recently used
/// ones, never a chat that `busy` says is generating.
fn trim_chats(map: &mut Offers, busy: impl Fn(&str) -> bool) {
    let mut chats: Vec<(u64, String)> = map.iter().filter(|(k, _)| k.starts_with("chat:")).map(|(k, (_, used))| (*used, k.clone())).collect();
    if chats.len() <= MAX_CHAT_OFFERS {
        return;
    }
    chats.sort();
    let excess = chats.len() - MAX_CHAT_OFFERS;
    for (_, k) in chats.iter().filter(|(_, k)| !busy(&k["chat:".len()..])).take(excess) {
        map.remove(k);
    }
}

/// Run a `load_tools` call for the host's scope.
pub fn call(scope_key: Option<&str>, args: &serde_json::Value) -> CallResult {
    match scope_key.and_then(get) {
        Some(o) => {
            let (text, is_error) = o.run_load_tools(args);
            CallResult { text, is_error }
        }
        None => CallResult { text: "Nothing to load: every tool you have is already available.".into(), is_error: true },
    }
}

/// A chat's offer, rebuilt from its settings and registered; its new loads
/// are saved in the chat settings (`loaded_tools`).
pub fn install_for_chat(st: &SharedState, conversation_id: &str, offer: ToolOffer) -> Arc<ToolOffer> {
    let a = install(st, &chat_key(conversation_id), offer);
    let (st2, id) = (st.clone(), conversation_id.to_string());
    a.set_on_load(move |ids| {
        let (st, id) = (st2.clone(), id.clone());
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                if let Err(e) = crate::services::chat::add_settings_lists(&st, &id, &ids, &[]).await {
                    tracing::warn!("could not save loaded tools of chat {id}: {e}");
                }
            });
        }
    });
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(server: &str, i: usize) -> ExposedTool {
        ExposedTool {
            tool_id: format!("mcp:{server}/t{i}"),
            model_name: format!("mcp__{server}__t{i}"),
            server_id: server.into(),
            server_name: server.into(),
            tool_name: format!("t{i}"),
            description: "d".into(),
            input_schema: serde_json::json!({ "type": "object" }),
            policy: Policy::Allow,
            timeout_secs: 5,
        }
    }

    #[test]
    fn keys_and_plan() {
        assert_eq!(key(Some("c"), Some("r"), Some("a")).as_deref(), Some("agent:a"));
        assert_eq!(key(Some("c"), Some("r"), None).as_deref(), Some("run:r"));
        assert_eq!(key(Some("c"), None, None).as_deref(), Some("chat:c"));
        assert_eq!(key(None, None, None), None);
        let mut small: Vec<ExposedTool> = (0..5).map(|i| t("s", i)).collect();
        assert!(plan(&mut small, true, &[], false, "chat").is_none());
        assert_eq!(small.len(), 5, "no load_tools below the threshold");
        let mut big: Vec<ExposedTool> = (0..31).map(|i| t("b", i)).collect();
        assert!(plan(&mut big.clone(), false, &[], false, "chat").is_none(), "setting off: everything sent");
        let o = plan(&mut big, true, &["mcp:b/t2".into()], false, "chat").unwrap();
        assert_eq!(big.iter().filter(|x| is_load_tools(x)).count(), 1);
        assert_eq!(o.loaded_ids(), vec!["mcp:b/t2".to_string()]);
        // Planning again never adds a second load_tools.
        plan(&mut big, true, &[], false, "chat").unwrap();
        assert_eq!(big.iter().filter(|x| is_load_tools(x)).count(), 1);
        assert!(call(Some("chat:none"), &serde_json::json!({ "names": ["x"] })).is_error);
    }

    #[test]
    fn reinstalling_keeps_live_loads_and_never_drops_one() {
        let mut map = HashMap::new();
        let mut tools: Vec<ExposedTool> = (0..31).map(|i| t("b", i)).collect();
        let first = install_in(&mut map, "chat:c", plan(&mut tools, true, &[], false, "chat").unwrap());
        first.load(&["mcp__b__t4".into()]);
        // The context meter (or the next message) rebuilds from settings that
        // were not saved yet: the live offer, with its load, is kept.
        let again = install_in(&mut map, "chat:c", plan(&mut tools, true, &["mcp:b/t1".into()], false, "chat").unwrap());
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(again.loaded_ids(), vec!["mcp:b/t4".to_string(), "mcp:b/t1".to_string()]);
        // The tool set changed (a tool added): a new offer that keeps the loads.
        tools.push(t("b", 31));
        let changed = install_in(&mut map, "chat:c", plan(&mut tools, true, &[], false, "chat").unwrap());
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(changed.loaded_ids(), vec!["mcp:b/t4".to_string(), "mcp:b/t1".to_string()]);
        // The load_tools call goes to the registered offer.
        OFFERS.lock().unwrap().insert("chat:test-call".into(), (changed.clone(), tick()));
        let r = call(Some("chat:test-call"), &serde_json::json!({ "names": ["mcp__b__t9"] }));
        assert!(!r.is_error && r.text.contains("Loaded 1: mcp__b__t9"), "{}", r.text);
        remove("chat:test-call");
    }

    /// Red-team regression: an MCP server reconnects with the same tool names
    /// and ids but a new argument schema. `same_catalog` compares only names
    /// and ids, so the old offer (old schemas) is kept, and in XML mode
    /// `load_tools` hands the model the stale schema, while dispatch validates
    /// against the new one, so every call the model builds from it is refused.
    #[test]
    fn reinstall_after_schema_change_serves_the_new_schema() {
        let mut map = HashMap::new();
        let mut before: Vec<ExposedTool> = (0..31).map(|i| t("b", i)).collect();
        install_in(&mut map, "chat:schema", plan(&mut before, true, &[], true, "chat").unwrap());
        let mut after: Vec<ExposedTool> = (0..31)
            .map(|i| ExposedTool {
                input_schema: serde_json::json!({ "type": "object", "properties": { "new_arg": { "type": "string" } }, "required": ["new_arg"] }),
                description: "new description".into(),
                ..t("b", i)
            })
            .collect();
        let live = install_in(&mut map, "chat:schema", plan(&mut after, true, &[], true, "chat").unwrap());
        let (text, err) = live.run_load_tools(&serde_json::json!({ "names": ["mcp__b__t3"] }));
        assert!(!err, "{text}");
        assert!(text.contains("new_arg") && text.contains("new description"), "load_tools returned the pre-reconnect definition: {text}");
    }

    #[test]
    fn peek_changes_nothing_and_carries_live_loads() {
        let mut tools: Vec<ExposedTool> = (0..31).map(|i| t("b", i)).collect();
        let live = install_in(&mut OFFERS.lock().unwrap(), "chat:peek", plan(&mut tools, true, &[], false, "chat").unwrap());
        live.load(&["mcp__b__t4".into()]);
        // Same tools: the meter sees the live offer, nothing is registered anew.
        let seen = peek("chat:peek", plan(&mut tools, true, &[], false, "chat").unwrap());
        assert!(Arc::ptr_eq(&live, &seen));
        // A changed tool set: a separate offer with the live loads; the live one is untouched.
        tools.push(t("b", 31));
        let other = peek("chat:peek", plan(&mut tools, true, &[], false, "chat").unwrap());
        assert!(!Arc::ptr_eq(&live, &other));
        assert_eq!(other.loaded_ids(), vec!["mcp:b/t4".to_string()]);
        assert!(Arc::ptr_eq(&live, &get("chat:peek").unwrap()), "peek left the registered offer alone");
        // Nothing registered for the chat: peek does not register it.
        let none = peek("chat:peek-none", plan(&mut tools, true, &[], false, "chat").unwrap());
        assert!(get("chat:peek-none").is_none() && none.loaded_ids().is_empty());
        remove("chat:peek");
    }

    #[test]
    fn trimming_drops_the_least_recently_used_idle_chats() {
        let mut tools: Vec<ExposedTool> = (0..31).map(|i| t("b", i)).collect();
        let mut map = Offers::new();
        for i in 0..MAX_CHAT_OFFERS + 3 {
            install_in(&mut map, &format!("chat:c{i}"), plan(&mut tools, true, &[], false, "chat").unwrap());
        }
        // c0 is the oldest, but it is used again; c1 is oldest now. c2 is generating.
        install_in(&mut map, "chat:c0", plan(&mut tools, true, &[], false, "chat").unwrap());
        install_in(&mut map, "run:r", plan(&mut tools, true, &[], false, "chat").unwrap());
        trim_chats(&mut map, |id| id == "c2");
        assert_eq!(map.keys().filter(|k| k.starts_with("chat:")).count(), MAX_CHAT_OFFERS);
        for gone in ["chat:c1", "chat:c3", "chat:c4"] {
            assert!(!map.contains_key(gone), "{gone} should be dropped");
        }
        assert!(map.contains_key("chat:c0") && map.contains_key("chat:c2") && map.contains_key("run:r"));
    }
}

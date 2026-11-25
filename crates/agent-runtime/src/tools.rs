//! Tools exposed to the model: naming, OpenAI tool definitions, and the
//! app-generated "Available tools" section of the system instruction.

use std::collections::HashSet;

use security::policy::Policy;
use serde::{Deserialize, Serialize};

/// `server_id` of built-in tools: "builtin:<environment id>".
pub const BUILTIN_PREFIX: &str = "builtin:";

/// A tool the model may be offered in this chat (Deny-policy and disabled
/// tools are never exposed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExposedTool {
    /// Stable id for policy and audit: "mcp:<server-id>/<tool>".
    pub tool_id: String,
    /// Name the model sees and calls, e.g. "mcp__filesystem__read_text_file".
    pub model_name: String,
    pub server_id: String,
    pub server_name: String,
    /// The tool's name on its server.
    pub tool_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub policy: Policy,
    pub timeout_secs: u64,
}

fn sanitize(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    while out.contains("__") && out.ends_with('_') {
        out.pop();
    }
    out
}

/// `mcp__<server>__<tool>`, limited to `[A-Za-z0-9_-]{1,64}`, unique within `taken`.
pub fn model_tool_name(server: &str, tool: &str, taken: &mut HashSet<String>) -> String {
    let base = format!("mcp__{}__{}", sanitize(server), sanitize(tool));
    let mut name: String = base.chars().take(64).collect();
    let mut n = 2;
    while taken.contains(&name) {
        let suffix = format!("_{n}");
        name = format!("{}{}", base.chars().take(64 - suffix.len()).collect::<String>(), suffix);
        n += 1;
    }
    taken.insert(name.clone());
    name
}

/// OpenAI `tools` array for llama-server.
pub fn openai_tools(tools: &[ExposedTool]) -> serde_json::Value {
    serde_json::Value::Array(
        tools
            .iter()
            .map(|t| {
                let mut schema = t.input_schema.clone();
                // A server's invalid schema must not break every request that
                // includes the tool (llama-server rejects it when building the
                // template parser): send a plain object schema instead. The
                // dispatcher validates such tools as "any JSON object".
                if !schema.is_object() || !crate::dispatch::schema_compiles(&schema) {
                    schema = serde_json::json!({ "type": "object", "properties": {} });
                }
                if schema.get("type").is_none() {
                    schema["type"] = "object".into();
                }
                serde_json::json!({
                    "type": "function",
                    "function": { "name": t.model_name, "description": truncate(&t.description, 1024), "parameters": schema }
                })
            })
            .collect(),
    )
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    format!("{}...", s.chars().take(max).collect::<String>())
}

/// Limits for one agent reply. `None` (JSON `null`) = no limit, the default.
/// The reply still stops after `MAX_TURNS_BACKSTOP` model turns, and the
/// context window caps each generation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, from = "LimitsRaw")]
pub struct Limits {
    pub max_tool_calls: Option<u32>,
    pub max_wall_secs: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct LimitsRaw {
    max_tool_calls: Option<u32>,
    max_wall_secs: Option<u64>,
}

impl From<LimitsRaw> for Limits {
    /// Chats saved before limits became optional hold the old defaults
    /// (10 calls, 300 s): those read as "no limit". Any other number is the
    /// user's own choice and stays.
    fn from(r: LimitsRaw) -> Self {
        if (r.max_tool_calls, r.max_wall_secs) == (Some(10), Some(300)) {
            return Self::default();
        }
        Self { max_tool_calls: r.max_tool_calls, max_wall_secs: r.max_wall_secs }
    }
}

impl Limits {
    /// Tool-call limit as a number (no limit = `u32::MAX`).
    pub fn calls_or_max(&self) -> u32 {
        self.max_tool_calls.unwrap_or(u32::MAX)
    }
    /// Time limit in seconds (no limit = 30 days).
    pub fn secs_or_max(&self) -> u64 {
        self.max_wall_secs.unwrap_or(crate::task::NO_TIME_LIMIT_SECS)
    }
}

const NO_TOOLS: &str = "None. No tools are enabled for this chat, so do not try to call any. If the user asks for something that needs a tool, explain that tools can be turned on in Chat settings and MCP Connections.\n";

fn limits_paragraph(limits: &Limits) -> String {
    let calls = limits.max_tool_calls.map_or("No limit on tool calls per reply.".to_string(), |n| format!("At most {n} tool calls per reply."));
    let time = limits.max_wall_secs.map_or("No time limit on the reply.".to_string(), |s| format!("The whole reply stops after {} minutes.", (s / 60).max(1)));
    format!("\n## Limits\n{calls} Each call times out after its server's limit (usually 60 seconds). {time}")
}

/// The app-generated part of the Agent Mode instruction for XML (text
/// format) mode, and the default: every tool with its origin, description
/// and permission. Read-only for the user; describes exactly what this chat
/// exposes, nothing more.
pub fn tools_section(tools: &[ExposedTool], limits: &Limits) -> String {
    tools_prompt(tools, limits, crate::agent_loop::ToolMode::Xml, None)
}

/// The tools part of the agent instruction.
/// - XML mode: the per-tool list (the model gets nothing else about the tools).
/// - Native mode: the JSON tool definitions already carry names, descriptions
///   and arguments, so only what they lack: where tools run, which run
///   without asking, and the limits (`native_overview`), plus the per-tool
///   lines of the app's own tools (subagents, ask_user), which small models
///   need to use their optional arguments.
/// - With `offer` (deferred loading): deferred tools are left out of the
///   list and named in a short index instead (`ToolOffer::index_section`).
pub fn tools_prompt(tools: &[ExposedTool], limits: &Limits, mode: crate::agent_loop::ToolMode, offer: Option<&crate::deferred::ToolOffer>) -> String {
    let mut s = String::from("## Available tools\n");
    if tools.is_empty() {
        s.push_str(NO_TOOLS);
        s.push_str(&limits_paragraph(limits));
        return s;
    }
    let listed: Vec<&ExposedTool> = tools.iter().filter(|t| offer.is_none_or(|o| !o.is_deferred(&t.model_name))).collect();
    if mode == crate::agent_loop::ToolMode::Native {
        s.push_str(&native_overview(tools, listed.len()));
        // The app's own tools (spawn_subagents, ask_user, ...) keep their
        // line even in native mode: without it small models skip their
        // optional arguments (Qwen3-4B left out spawn_subagents' "isolation"
        // in 6 of 6 tries; with these lines it sent it in 5 of 6). They are
        // few and short, unlike MCP servers, so the overview stays compact.
        let app: Vec<&&ExposedTool> = listed.iter().filter(|t| t.server_id.starts_with(crate::deferred::APP_PREFIX) && t.model_name != crate::deferred::LOAD_TOOLS).collect();
        if !app.is_empty() {
            s.push_str("The app's own tools:\n");
            for t in app {
                s.push_str(&tool_line(t));
            }
        }
    } else {
        for t in &listed {
            s.push_str(&tool_line(t));
        }
    }
    if let Some(o) = offer.filter(|o| !o.names().is_empty()) {
        s.push('\n');
        s.push_str(&o.index_section());
        s.push('\n');
    }
    s.push_str(&limits_paragraph(limits));
    s
}

/// One tool's line in the per-tool list: name, origin, description, permission.
fn tool_line(t: &ExposedTool) -> String {
    let perm = match t.policy {
        Policy::Allow => "runs without asking",
        _ => "needs the user's approval each time",
    };
    let desc = truncate(&t.description.replace('\n', " "), 200);
    let origin = if t.server_id.starts_with(BUILTIN_PREFIX) { format!("built-in, runs on {}", t.server_name) } else { format!("server \"{}\"", t.server_name) };
    format!("- {} ({origin}): {} Permission: {}.\n", t.model_name, if desc.is_empty() { "No description.".to_string() } else { desc }, perm)
}

/// `mcp__<server>__*` when every name shares that prefix, else None.
fn wildcard(names: &[&str]) -> Option<String> {
    if names.len() < 2 {
        return None;
    }
    let first = names[0];
    let mut len = first.len();
    for n in &names[1..] {
        len = len.min(first.bytes().zip(n.bytes()).take_while(|(a, b)| a == b).count());
    }
    let rest = first[..len].strip_prefix("mcp__")?;
    let end = rest.find("__")?;
    Some(format!("mcp__{}__*", &rest[..end]))
}

/// Native-mode overview: tool count, where the tools come from, and which
/// run without asking (a whole MCP server with one permission is written as
/// `mcp__<server>__*`).
fn native_overview(tools: &[ExposedTool], sent: usize) -> String {
    let mut s = if sent == 1 {
        "Your 1 tool, with its description and arguments, is in the tool definitions sent with this request.\n".to_string()
    } else {
        format!("Your {sent} tools, with their descriptions and arguments, are in the tool definitions sent with this request.\n")
    };
    let mut envs: Vec<&str> = vec![];
    let mut servers: Vec<(&str, &str)> = vec![];
    for t in tools {
        if t.server_id.starts_with(BUILTIN_PREFIX) {
            if !envs.contains(&t.server_name.as_str()) {
                envs.push(&t.server_name);
            }
        } else if crate::deferred::is_deferrable(t) && !servers.iter().any(|(id, _)| *id == t.server_id) {
            servers.push((&t.server_id, &t.server_name));
        }
    }
    let mut origin = vec![];
    if !envs.is_empty() {
        origin.push(format!("Built-in tools run on {}.", envs.join(", ")));
    }
    let mut allow: Vec<String> = vec![];
    let mut ask: Vec<String> = vec![];
    let is_allow = |t: &ExposedTool| t.policy == Policy::Allow;
    // Built-ins and the app's own tools: by name.
    for t in tools.iter().filter(|t| !crate::deferred::is_deferrable(t)) {
        if is_allow(t) {
            allow.push(t.model_name.clone())
        } else {
            ask.push(t.model_name.clone())
        }
    }
    let mut mcp = vec![];
    for (id, name) in &servers {
        let mine: Vec<&ExposedTool> = tools.iter().filter(|t| t.server_id == *id).collect();
        let names: Vec<&str> = mine.iter().map(|t| t.model_name.as_str()).collect();
        let w = wildcard(&names);
        mcp.push(match &w {
            Some(w) => format!("\"{name}\" ({w})"),
            None => format!("\"{name}\" ({})", names.join(", ")),
        });
        match (&w, mine.iter().all(|t| is_allow(t)), mine.iter().all(|t| !is_allow(t))) {
            (Some(w), true, _) => allow.push(w.clone()),
            (Some(w), _, true) => ask.push(w.clone()),
            _ => {
                for t in mine {
                    if is_allow(t) {
                        allow.push(t.model_name.clone())
                    } else {
                        ask.push(t.model_name.clone())
                    }
                }
            }
        }
    }
    if !mcp.is_empty() {
        origin.push(format!("MCP servers: {}.", mcp.join(", ")));
    }
    if !origin.is_empty() {
        s.push_str(&origin.join(" "));
        s.push('\n');
    }
    if !allow.is_empty() {
        s.push_str(&format!("Run without asking: {}.\n", allow.join(", ")));
    }
    if !ask.is_empty() {
        s.push_str(&format!("Ask the user for approval each time: {}.\n", ask.join(", ")));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(policy: Policy) -> ExposedTool {
        ExposedTool {
            tool_id: "mcp:s/read".into(),
            model_name: "mcp__fs__read".into(),
            server_id: "s".into(),
            server_name: "fs".into(),
            tool_name: "read".into(),
            description: "Read a file.\nSecond line.".into(),
            input_schema: serde_json::json!({ "properties": { "path": { "type": "string" } } }),
            policy,
            timeout_secs: 60,
        }
    }

    #[test]
    fn names_are_sanitized_unique_and_bounded() {
        let mut taken = HashSet::new();
        assert_eq!(model_tool_name("file system", "read.file", &mut taken), "mcp__file_system__read_file");
        assert_eq!(model_tool_name("file system", "read.file", &mut taken), "mcp__file_system__read_file_2");
        let long = model_tool_name(&"s".repeat(80), &"t".repeat(80), &mut taken);
        assert!(long.len() <= 64 && long.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn default_limits_are_unlimited() {
        let l = Limits::default();
        assert_eq!((l.max_tool_calls, l.max_wall_secs), (None, None));
        // run_reply derives its time and call allowance from these: huge, never zero.
        assert_eq!((l.calls_or_max(), l.secs_or_max()), (u32::MAX, crate::task::NO_TIME_LIMIT_SECS));
        assert!(std::time::Duration::from_secs(l.secs_or_max()) > std::time::Duration::from_secs(24 * 3600));
        assert_eq!(serde_json::to_value(l).unwrap(), serde_json::json!({ "max_tool_calls": null, "max_wall_secs": null }));
    }

    #[test]
    fn openai_definitions_and_prompt_section() {
        let t = openai_tools(&[tool(Policy::Ask)]);
        assert_eq!(t[0]["function"]["name"], "mcp__fs__read");
        assert_eq!(t[0]["function"]["parameters"]["type"], "object", "type added when missing");
        let bad = ExposedTool { input_schema: serde_json::json!({ "type": "object", "properties": { "x": { "type": "not-a-real-type" } } }), ..tool(Policy::Ask) };
        assert_eq!(openai_tools(&[bad])[0]["function"]["parameters"], serde_json::json!({ "type": "object", "properties": {} }), "invalid schema replaced");
        let s = tools_section(&[tool(Policy::Allow), ExposedTool { model_name: "mcp__fs__write".into(), ..tool(Policy::Ask) }], &Limits::default());
        assert!(s.contains("mcp__fs__read (server \"fs\"): Read a file. Second line. Permission: runs without asking."), "{s}");
        assert!(s.contains("mcp__fs__write") && s.contains("needs the user's approval each time"));
        assert!(s.contains("No limit on tool calls") && s.contains("No time limit") && !s.contains("At most"), "{s}");
        let l = tools_section(&[], &Limits { max_tool_calls: Some(10), max_wall_secs: Some(300) });
        assert!(l.contains("At most 10 tool calls") && l.contains("after 5 minutes"), "{l}");
        assert!(tools_section(&[], &Limits::default()).contains("do not try to call any"));
    }

    fn named(server: &str, name: &str, policy: Policy) -> ExposedTool {
        ExposedTool { tool_id: format!("mcp:{server}/{name}"), model_name: format!("mcp__{server}__{name}"), server_id: server.into(), server_name: server.into(), tool_name: name.into(), ..tool(policy) }
    }

    fn builtin(name: &str, policy: Policy) -> ExposedTool {
        ExposedTool { tool_id: format!("builtin:local/{name}"), model_name: name.into(), server_id: "builtin:local".into(), server_name: "This PC".into(), tool_name: name.into(), description: format!("{name} desc"), ..tool(policy) }
    }

    #[test]
    fn native_section_has_no_tool_list_but_says_what_runs_without_asking() {
        use crate::agent_loop::ToolMode;
        let tools = vec![
            builtin("fs_read", Policy::Allow),
            builtin("shell_execute", Policy::Ask),
            named("rea", "open", Policy::Allow),
            named("rea", "close", Policy::Allow),
            named("mix", "read", Policy::Allow),
            named("mix", "write", Policy::Ask),
        ];
        let lim = Limits { max_tool_calls: Some(5), max_wall_secs: None };
        let n = tools_prompt(&tools, &lim, ToolMode::Native, None);
        assert!(!n.contains("Read a file") && !n.contains("Permission:") && !n.contains("fs_read desc"), "no per-tool descriptions: {n}");
        assert!(n.contains("Your 6 tools, with their descriptions and arguments, are in the tool definitions"), "{n}");
        assert!(n.contains("Built-in tools run on This PC."), "{n}");
        assert!(n.contains("MCP servers: \"rea\" (mcp__rea__*), \"mix\" (mcp__mix__*)."), "{n}");
        assert!(n.contains("Run without asking: fs_read, mcp__rea__*, mcp__mix__read.\n"), "{n}");
        assert!(n.contains("Ask the user for approval each time: shell_execute, mcp__mix__write.\n"), "{n}");
        assert!(n.contains("## Limits\nAt most 5 tool calls per reply."), "{n}");
        // XML mode is the old per-tool list, byte for byte.
        let x = tools_prompt(&tools, &lim, ToolMode::Xml, None);
        assert_eq!(x, tools_section(&tools, &lim));
        assert!(x.contains("- fs_read (built-in, runs on This PC): fs_read desc Permission: runs without asking.\n"), "{x}");
        assert!(x.contains("- mcp__mix__write (server \"mix\"): Read a file. Second line. Permission: needs the user's approval each time.\n"), "{x}");
        // No tools: the same text in both modes.
        assert_eq!(tools_prompt(&[], &lim, ToolMode::Native, None), tools_section(&[], &lim));
        // The JSON definitions do not depend on the prompt mode.
        assert_eq!(openai_tools(&tools).to_string(), openai_tools(&tools).to_string());
    }

    #[test]
    fn native_section_lists_the_apps_own_tools() {
        use crate::agent_loop::ToolMode;
        let app = |name: &str| ExposedTool { tool_id: format!("agent:r/{name}"), model_name: name.into(), server_id: "agent:r".into(), server_name: "Local AI Studio".into(), tool_name: name.into(), description: format!("{name} does things."), ..tool(Policy::Allow) };
        let tools = vec![builtin("fs_read", Policy::Allow), named("mix", "read", Policy::Allow), app("spawn_subagents"), app("ask_user"), app(crate::deferred::LOAD_TOOLS)];
        let n = tools_prompt(&tools, &Limits::default(), ToolMode::Native, None);
        assert!(
            n.contains("The app's own tools:\n- spawn_subagents (server \"Local AI Studio\"): spawn_subagents does things. Permission: runs without asking.\n- ask_user (server \"Local AI Studio\"): ask_user does things."),
            "{n}"
        );
        // Built-ins, MCP tools and load_tools keep the compact form.
        assert!(!n.contains("- fs_read (") && !n.contains("- mcp__mix__read (") && !n.contains("- load_tools ("), "{n}");
        assert!(n.find("The app's own tools:").unwrap() < n.find("\n\n## Limits\n").unwrap());
        // No app tools: no heading.
        assert!(!tools_prompt(&tools[..2], &Limits::default(), ToolMode::Native, None).contains("The app's own tools"));
    }

    #[test]
    fn deferred_tools_go_to_the_index_in_both_modes() {
        use crate::agent_loop::ToolMode;
        let mut tools = vec![builtin("fs_read", Policy::Allow)];
        tools.extend((0..31).map(|i| named("big", &format!("t{i}"), Policy::Allow)));
        let d = crate::deferred::pick_deferred(&tools);
        for mode in [ToolMode::Native, ToolMode::Xml] {
            let o = crate::deferred::ToolOffer::new(&tools, &d, &[], mode == ToolMode::Xml);
            let s = tools_prompt(&tools, &Limits::default(), mode, Some(&o));
            assert!(s.contains("## More tools (load before use)\nThese 31 tools") && s.contains("- mcp__big__t30: Read a file."), "{s}");
            assert!(s.find("## More tools").unwrap() < s.find("\n\n## Limits\n").unwrap(), "index before the limits");
            if mode == ToolMode::Xml {
                assert!(s.contains("- fs_read (built-in") && !s.contains("- mcp__big__t3 (server"), "deferred tools are not in the full list: {s}");
            } else {
                assert!(s.contains("Your 1 tool, with its") && s.contains("Run without asking: fs_read, mcp__big__*."), "{s}");
            }
        }
    }
}

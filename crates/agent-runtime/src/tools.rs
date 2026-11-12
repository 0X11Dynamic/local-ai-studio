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
                if !schema.is_object() || jsonschema::validator_for(&schema).is_err() {
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

/// Limits for one agent reply.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Limits {
    pub max_tool_calls: u32,
    pub max_wall_secs: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_tool_calls: 10, max_wall_secs: 300 }
    }
}

/// The app-generated part of the Agent Mode instruction. Read-only for the
/// user; describes exactly what this chat exposes, nothing more.
pub fn tools_section(tools: &[ExposedTool], limits: &Limits) -> String {
    let mut s = String::from("## Available tools\n");
    if tools.is_empty() {
        s.push_str("None. No tools are enabled for this chat, so do not try to call any. If the user asks for something that needs a tool, explain that tools can be turned on in Chat settings and MCP Connections.\n");
    } else {
        for t in tools {
            let perm = match t.policy {
                Policy::Allow => "runs without asking",
                _ => "needs the user's approval each time",
            };
            let desc = truncate(&t.description.replace('\n', " "), 200);
            let origin = if t.server_id.starts_with(BUILTIN_PREFIX) { format!("built-in, runs on {}", t.server_name) } else { format!("server \"{}\"", t.server_name) };
            s.push_str(&format!("- {} ({origin}): {} Permission: {}.\n", t.model_name, if desc.is_empty() { "No description.".to_string() } else { desc }, perm));
        }
    }
    s.push_str(&format!(
        "\n## Limits\nAt most {} tool calls per reply. Each call times out after its server's limit (usually 60 seconds). The whole reply stops after {} minutes.",
        limits.max_tool_calls,
        (limits.max_wall_secs / 60).max(1)
    ));
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
    fn openai_definitions_and_prompt_section() {
        let t = openai_tools(&[tool(Policy::Ask)]);
        assert_eq!(t[0]["function"]["name"], "mcp__fs__read");
        assert_eq!(t[0]["function"]["parameters"]["type"], "object", "type added when missing");
        let bad = ExposedTool { input_schema: serde_json::json!({ "type": "object", "properties": { "x": { "type": "not-a-real-type" } } }), ..tool(Policy::Ask) };
        assert_eq!(openai_tools(&[bad])[0]["function"]["parameters"], serde_json::json!({ "type": "object", "properties": {} }), "invalid schema replaced");
        let s = tools_section(&[tool(Policy::Allow), ExposedTool { model_name: "mcp__fs__write".into(), ..tool(Policy::Ask) }], &Limits::default());
        assert!(s.contains("mcp__fs__read (server \"fs\"): Read a file. Second line. Permission: runs without asking."), "{s}");
        assert!(s.contains("mcp__fs__write") && s.contains("needs the user's approval each time"));
        assert!(s.contains("At most 10 tool calls"));
        assert!(tools_section(&[], &Limits::default()).contains("do not try to call any"));
    }
}

//! Tool permission policy (spec sections 6, 7, 14, 18).
//!
//! The decision for a tool call depends only on app-owned state: the tool's
//! enabled flag and policy (set by the user in the UI) and the user's grants.
//! Nothing a model writes, and nothing a tool returns, is an input here, so
//! prompt injection cannot change a decision. Server-provided annotations such
//! as `readOnlyHint` are displayed as unverified hints and are ignored.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    /// Runs without asking.
    Allow,
    /// Asks the user every time unless a grant covers it.
    Ask,
    /// Never runs; the tool is not offered to the model.
    Deny,
}

impl Policy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "allow" => Some(Self::Allow),
            "ask" => Some(Self::Ask),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }
}

/// The default for every tool from an MCP server: ask, whatever the server
/// claims about itself.
pub const MCP_DEFAULT_POLICY: Policy = Policy::Ask;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantScope {
    /// For one conversation (a chat session).
    Session,
    /// Until revoked.
    Persistent,
}

/// A user grant: "this tool may run without asking" within a scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub id: String,
    pub scope: GrantScope,
    /// Stable tool id, e.g. "mcp:<server-id>/<tool-name>".
    pub tool_id: String,
    /// For session grants: the conversation it belongs to.
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    Allow { reason: String },
    Ask { reason: String },
    Deny { reason: String },
}

impl Decision {
    pub fn reason(&self) -> &str {
        match self {
            Decision::Allow { reason } | Decision::Ask { reason } | Decision::Deny { reason } => reason,
        }
    }
}

/// What the policy needs to know about a tool (all app-owned).
#[derive(Debug, Clone)]
pub struct ToolPolicyInput<'a> {
    pub tool_id: &'a str,
    pub enabled: bool,
    pub policy: Policy,
    /// The server is connected and offers this tool right now.
    pub available: bool,
}

/// Decide whether a call may run. Order: disabled/deny > unavailable >
/// grant > policy allow > ask.
pub fn decide(tool: &ToolPolicyInput<'_>, grants: &[Grant], conversation_id: Option<&str>) -> Decision {
    if !tool.enabled {
        return Decision::Deny { reason: "the tool is turned off in MCP Connections".into() };
    }
    if tool.policy == Policy::Deny {
        return Decision::Deny { reason: "the tool's policy is deny".into() };
    }
    if !tool.available {
        return Decision::Deny { reason: "the tool's server is not connected".into() };
    }
    let granted = grants.iter().find(|g| {
        g.tool_id == tool.tool_id
            && match g.scope {
                GrantScope::Persistent => true,
                GrantScope::Session => g.conversation_id.is_some() && g.conversation_id.as_deref() == conversation_id,
            }
    });
    if let Some(g) = granted {
        let scope = match g.scope {
            GrantScope::Session => "for this chat",
            GrantScope::Persistent => "always",
        };
        return Decision::Allow { reason: format!("you allowed this tool {scope}") };
    }
    if tool.policy == Policy::Allow {
        return Decision::Allow { reason: "the tool's policy is allow".into() };
    }
    Decision::Ask { reason: "the tool's policy is ask".into() }
}

/// The user's answer to an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalAnswer {
    AllowOnce,
    AllowSession,
    AllowAlways,
    Deny,
    /// No answer within the time limit (treated as deny).
    TimedOut,
}

impl ApprovalAnswer {
    pub fn allows(self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowSession | Self::AllowAlways)
    }
    pub fn grant_scope(self) -> Option<GrantScope> {
        match self {
            Self::AllowSession => Some(GrantScope::Session),
            Self::AllowAlways => Some(GrantScope::Persistent),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(policy: Policy) -> ToolPolicyInput<'static> {
        ToolPolicyInput { tool_id: "mcp:s1/write_file", enabled: true, policy, available: true }
    }

    fn grant(scope: GrantScope, conv: Option<&str>) -> Grant {
        Grant { id: "g".into(), scope, tool_id: "mcp:s1/write_file".into(), conversation_id: conv.map(str::to_string) }
    }

    #[test]
    fn default_is_ask_and_policies_apply() {
        assert!(matches!(decide(&tool(Policy::Ask), &[], Some("c1")), Decision::Ask { .. }));
        assert!(matches!(decide(&tool(Policy::Allow), &[], Some("c1")), Decision::Allow { .. }));
        assert!(matches!(decide(&tool(Policy::Deny), &[], Some("c1")), Decision::Deny { .. }));
        assert_eq!(MCP_DEFAULT_POLICY, Policy::Ask);
    }

    #[test]
    fn deny_and_disabled_beat_grants() {
        let g = [grant(GrantScope::Persistent, None)];
        assert!(matches!(decide(&tool(Policy::Deny), &g, Some("c1")), Decision::Deny { .. }));
        let mut off = tool(Policy::Ask);
        off.enabled = false;
        assert!(matches!(decide(&off, &g, Some("c1")), Decision::Deny { .. }));
        let mut gone = tool(Policy::Allow);
        gone.available = false;
        assert!(matches!(decide(&gone, &g, Some("c1")), Decision::Deny { .. }));
    }

    #[test]
    fn session_grants_are_scoped_to_their_chat() {
        let g = [grant(GrantScope::Session, Some("c1"))];
        assert!(matches!(decide(&tool(Policy::Ask), &g, Some("c1")), Decision::Allow { .. }));
        assert!(matches!(decide(&tool(Policy::Ask), &g, Some("c2")), Decision::Ask { .. }));
        assert!(matches!(decide(&tool(Policy::Ask), &g, None), Decision::Ask { .. }));
        let broken = [grant(GrantScope::Session, None)];
        assert!(matches!(decide(&tool(Policy::Ask), &broken, None), Decision::Ask { .. }), "a session grant without a chat matches nothing");
        let other_tool = [Grant { tool_id: "mcp:s1/delete".into(), ..grant(GrantScope::Persistent, None) }];
        assert!(matches!(decide(&tool(Policy::Ask), &other_tool, Some("c1")), Decision::Ask { .. }));
    }

    #[test]
    fn answers() {
        assert!(ApprovalAnswer::AllowOnce.allows() && !ApprovalAnswer::Deny.allows() && !ApprovalAnswer::TimedOut.allows());
        assert_eq!(ApprovalAnswer::AllowOnce.grant_scope(), None);
        assert_eq!(ApprovalAnswer::AllowAlways.grant_scope(), Some(GrantScope::Persistent));
    }
}

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

// ---------- per-call facts (paths outside the allowed folders, Plan mode, local network) ----------

/// What the app worked out about one particular call from its arguments
/// (never from the model's words): where it reaches and whether that needs
/// the user. Built by the app for built-in tools; empty for everything else.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallFacts {
    /// Refused whatever the policy and grants say (a Windows system folder,
    /// the app's own data, a UNC/device path).
    pub blocked: Option<String>,
    /// The call must be approved even when the tool's policy is allow. The
    /// text is shown on the approval card ("Outside your allowed folders: ...").
    pub must_ask: Option<String>,
    /// One entry per place the call reaches that needs approval: the grant
    /// subjects, any one of which covers that place (the folder and each of
    /// its parents). A grant skips the question only when every place is
    /// covered. Empty: no grant can skip it (ask every time).
    pub covered_by: Vec<Vec<String>>,
    /// Subjects recorded when the user answers "allow for this chat/task"
    /// (one per place). Empty: any answer counts for this call only.
    pub grant_to: Vec<String>,
    /// What the user approves (resolved paths, a URL), given back to the
    /// tool when it runs so it reaches exactly that and nothing else.
    pub approved: Vec<String>,
}

impl CallFacts {
    /// Which answers the approval card should offer: "any" (once, chat,
    /// always), "session" (once, chat) or "once".
    pub fn grant_choice(&self) -> &'static str {
        match (&self.must_ask, self.grant_to.is_empty()) {
            (None, _) => "any",
            (Some(_), false) => "session",
            (Some(_), true) => "once",
        }
    }
}

fn grant_matches(g: &Grant, conversation_id: Option<&str>) -> bool {
    match g.scope {
        GrantScope::Persistent => true,
        GrantScope::Session => g.conversation_id.is_some() && g.conversation_id.as_deref() == conversation_id,
    }
}

/// `decide`, then the per-call facts. Order: disabled/deny/unavailable >
/// blocked > (must ask: grants for exactly these places, else ask) > the
/// usual grant/policy decision. `place_grants` are the grants loaded for the
/// subjects in `facts.covered_by`; a grant for the tool as a whole never
/// covers a place outside the allowed folders.
pub fn decide_call(tool: &ToolPolicyInput<'_>, grants: &[Grant], facts: &CallFacts, place_grants: &[Grant], conversation_id: Option<&str>) -> Decision {
    let base = decide(tool, grants, conversation_id);
    if matches!(base, Decision::Deny { .. }) {
        return base;
    }
    if let Some(b) = &facts.blocked {
        return Decision::Deny { reason: b.clone() };
    }
    let Some(why) = &facts.must_ask else { return base };
    let covered = !facts.covered_by.is_empty()
        && facts.covered_by.iter().all(|subjects| place_grants.iter().any(|g| subjects.contains(&g.tool_id) && grant_matches(g, conversation_id)));
    if covered {
        return Decision::Allow { reason: "you allowed this tool there for this chat".into() };
    }
    Decision::Ask { reason: why.clone() }
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

    fn outside(path_subjects: &[&str], grant_to: &[&str]) -> CallFacts {
        CallFacts {
            must_ask: Some("Outside your allowed folders: C:\\Users\\me\\Documents\\a.txt".into()),
            covered_by: vec![path_subjects.iter().map(|s| s.to_string()).collect()],
            grant_to: grant_to.iter().map(|s| s.to_string()).collect(),
            approved: vec!["C:\\Users\\me\\Documents\\a.txt".into()],
            ..Default::default()
        }
    }

    fn place_grant(subject: &str, scope: GrantScope, conv: Option<&str>) -> Grant {
        Grant { id: "p".into(), scope, tool_id: subject.into(), conversation_id: conv.map(str::to_string) }
    }

    const DOCS: &str = "builtin:local/fs_read outside c:\\users\\me\\documents";
    const ME: &str = "builtin:local/fs_read outside c:\\users\\me";

    #[test]
    fn inside_the_folders_the_usual_policy_applies() {
        let none = CallFacts::default();
        assert!(matches!(decide_call(&tool(Policy::Allow), &[], &none, &[], Some("c1")), Decision::Allow { .. }));
        assert!(matches!(decide_call(&tool(Policy::Ask), &[], &none, &[], Some("c1")), Decision::Ask { .. }));
        let g = [grant(GrantScope::Session, Some("c1"))];
        assert!(matches!(decide_call(&tool(Policy::Ask), &g, &none, &[], Some("c1")), Decision::Allow { .. }), "tool grant inside the folders");
        assert_eq!(none.grant_choice(), "any");
    }

    #[test]
    fn outside_the_folders_always_asks_whatever_the_policy() {
        let f = outside(&[DOCS, ME], &[DOCS]);
        for p in [Policy::Allow, Policy::Ask] {
            let d = decide_call(&tool(p), &[], &f, &[], Some("c1"));
            assert!(matches!(&d, Decision::Ask { reason } if reason.contains("Outside your allowed folders")), "{p:?}: {d:?}");
        }
        // A grant for the whole tool (even "always") does not cover outside places.
        let tool_grants = [grant(GrantScope::Persistent, None), grant(GrantScope::Session, Some("c1"))];
        assert!(matches!(decide_call(&tool(Policy::Ask), &tool_grants, &f, &[], Some("c1")), Decision::Ask { .. }));
        // Deny and disabled still win.
        assert!(matches!(decide_call(&tool(Policy::Deny), &[], &f, &[], Some("c1")), Decision::Deny { .. }));
        assert_eq!(f.grant_choice(), "session");
    }

    #[test]
    fn blocked_places_are_refused_even_with_grants() {
        let f = CallFacts { blocked: Some("C:\\Windows\\win.ini is inside a protected folder".into()), ..outside(&[DOCS], &[DOCS]) };
        let pg = [place_grant(DOCS, GrantScope::Persistent, None)];
        let g = [grant(GrantScope::Persistent, None)];
        assert!(matches!(decide_call(&tool(Policy::Allow), &g, &f, &pg, Some("c1")), Decision::Deny { reason } if reason.contains("protected")));
    }

    #[test]
    fn place_grants_are_scoped_to_the_folder_and_the_chat() {
        let f = outside(&[DOCS, ME], &[DOCS]);
        // A grant for the folder or a parent, in this chat: runs.
        for subject in [DOCS, ME] {
            let pg = [place_grant(subject, GrantScope::Session, Some("c1"))];
            assert!(matches!(decide_call(&tool(Policy::Ask), &[], &f, &pg, Some("c1")), Decision::Allow { .. }), "{subject}");
            assert!(matches!(decide_call(&tool(Policy::Ask), &[], &f, &pg, Some("c2")), Decision::Ask { .. }), "another chat: {subject}");
            assert!(matches!(decide_call(&tool(Policy::Ask), &[], &f, &pg, None), Decision::Ask { .. }), "no chat: {subject}");
        }
        // A grant for a sibling folder or another tool covers nothing.
        let other = [
            place_grant("builtin:local/fs_read outside c:\\users\\me\\music", GrantScope::Session, Some("c1")),
            place_grant("builtin:local/fs_write outside c:\\users\\me\\documents", GrantScope::Session, Some("c1")),
        ];
        assert!(matches!(decide_call(&tool(Policy::Ask), &[], &f, &other, Some("c1")), Decision::Ask { .. }));
        // Two places (a move): both must be covered.
        let two = CallFacts { covered_by: vec![vec![DOCS.into()], vec!["builtin:local/fs_read outside d:\\x".into()]], ..f.clone() };
        let one = [place_grant(DOCS, GrantScope::Session, Some("c1"))];
        assert!(matches!(decide_call(&tool(Policy::Ask), &[], &two, &one, Some("c1")), Decision::Ask { .. }));
    }

    #[test]
    fn ask_every_time_ignores_all_grants() {
        // Plan mode shell commands and local-network fetches: no subjects at all.
        let f = CallFacts { must_ask: Some("Plan mode: read-only command requested".into()), ..Default::default() };
        let g = [grant(GrantScope::Persistent, None)];
        let pg = [place_grant(DOCS, GrantScope::Persistent, None)];
        assert!(matches!(decide_call(&tool(Policy::Allow), &g, &f, &pg, Some("c1")), Decision::Ask { reason } if reason.starts_with("Plan mode")));
        assert_eq!(f.grant_choice(), "once");
    }

    #[test]
    fn answers() {
        assert!(ApprovalAnswer::AllowOnce.allows() && !ApprovalAnswer::Deny.allows() && !ApprovalAnswer::TimedOut.allows());
        assert_eq!(ApprovalAnswer::AllowOnce.grant_scope(), None);
        assert_eq!(ApprovalAnswer::AllowAlways.grant_scope(), Some(GrantScope::Persistent));
    }
}

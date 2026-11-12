// Typed wrappers for phase 3 (MCP servers, tool permissions, approvals).
import { invoke } from "@tauri-apps/api/core";

export interface ServerConfig {
  transport: "stdio" | "http"; command: string; args: string[]; env: Record<string, string>; cwd: string | null;
  url: string; headers: Record<string, string>; secret_env: string[]; secret_headers: string[]; bearer: boolean; timeout_secs: number;
}
export interface ToolInfo { name: string; title: string | null; description: string; input_schema: unknown; annotations: unknown }
export interface ServerStatus {
  id: string; state: "disconnected" | "connecting" | "connected" | "error"; error: string | null;
  server_name: string | null; server_version: string | null; tools: ToolInfo[]; resources: unknown[]; prompts: unknown[];
  pid: number | null; connected_at: string | null;
}
export interface ToolView extends ToolInfo { enabled: boolean; policy: "allow" | "ask" | "deny" }
export interface ServerView {
  id: string; name: string; transport: string; enabled: boolean; config: ServerConfig;
  secret_env_set: string[]; secret_headers_set: string[]; bearer_set: boolean; status: ServerStatus; tools: ToolView[];
  /** Background connect (app start, import, turned on); null when none is running or waiting. */
  auto?: AutoConnect | null;
}
export interface AutoConnect { attempt: number; max_attempts: number; retry_in_secs: number | null; gave_up: boolean }
/** One tool source in a chat's Tools menu: "builtin" or an MCP server. */
export interface ToolCostEntry {
  id: string; name: string; kind: "builtin" | "mcp"; state: string; enabled: boolean; in_chat: boolean;
  tool_count: number; tokens: number; error: string | null;
}
export interface ToolCosts { exact: boolean; mode: "chat" | "plan" | "agent"; entries: ToolCostEntry[] }
export interface SecretsPatch { env?: Record<string, string | null>; headers?: Record<string, string | null>; bearer?: string | null }
export interface ImportReport { added: string[]; skipped: string[]; secrets_stored: number }
export interface GrantView {
  id: string; scope: "session" | "persistent"; subject: string; capability: string; created_at: string;
  conversation_id: string | null; server_name: string | null; tool_name: string | null; conversation_title: string | null;
  /** Set for "Allow for this task" grants. */
  run_id?: string | null; task_title?: string | null;
}
export interface ToolExecRow {
  id: string; conversation_id: string | null; tool: string; source: string; args_redacted: string; result: string | null;
  status: string; approval: string; started_at: string; ended_at: string | null; duration_ms: number | null;
  tool_call_id: string | null; model_tool_name: string | null;
}
export interface ApprovalRequest {
  id: string; call_id: string; conversation_id: string | null; tool_id: string; model_name: string; server_name: string; tool_name: string;
  arguments: unknown; reason: string; run_id?: string | null; agent?: string | null;
}
export type ApprovalAnswer = "allow_once" | "allow_session" | "allow_always" | "deny";
export type ToolEvent =
  | { phase: "started"; audit_id: string; call_id: string; model_name: string; server_name: string; tool_name: string; arguments: unknown }
  | { phase: "awaiting_approval"; audit_id: string; call_id: string }
  | { phase: "finished"; audit_id: string; call_id: string; status: string; output: string; decision: string }
  | { phase: "output"; call_id: string; stream: string; text: string };
/** Payload of tool://event: which chat or task (and subagent) the call belongs to. */
export interface ToolEventPayload { conversation_id: string | null; run_id?: string | null; agent_id?: string | null; event: ToolEvent }

export const BLANK_CONFIG: ServerConfig = {
  transport: "stdio", command: "", args: [], env: {}, cwd: null, url: "", headers: {}, secret_env: [], secret_headers: [], bearer: false, timeout_secs: 60,
};

export const api3 = {
  listServers: () => invoke<ServerView[]>("list_mcp_servers"),
  addServer: (name: string, config: ServerConfig, secrets: SecretsPatch, enabled = true) => invoke<ServerView>("add_mcp_server", { name, config, secrets, enabled }),
  updateServer: (id: string, name: string, config: ServerConfig, secrets: SecretsPatch, enabled: boolean) => invoke<ServerView>("update_mcp_server", { id, name, config, secrets, enabled }),
  deleteServer: (id: string) => invoke<void>("delete_mcp_server", { id }),
  connect: (id: string) => invoke<ServerView>("connect_mcp_server", { id }),
  disconnect: (id: string) => invoke<ServerView>("disconnect_mcp_server", { id }),
  logs: (id: string) => invoke<string[]>("mcp_server_logs", { id }),
  setToolPref: (serverId: string, tool: string, enabled: boolean, policy: string) => invoke<void>("set_mcp_tool_pref", { serverId, tool, enabled, policy }),
  importJson: (json: string) => invoke<ImportReport>("import_mcp_servers", { json }),
  importFrom: (source: "lmstudio" | "claude") => invoke<ImportReport>("import_mcp_from", { source }),
  importFile: (path: string) => invoke<ImportReport>("import_mcp_file", { path }),
  exportJson: () => invoke<string>("export_mcp_servers"),
  grants: () => invoke<GrantView[]>("list_tool_grants"),
  revokeGrant: (id: string) => invoke<void>("revoke_tool_grant", { id }),
  activity: (conversationId?: string | null, limit?: number) => invoke<ToolExecRow[]>("list_tool_activity", { conversationId, limit }),
  pendingApprovals: () => invoke<ApprovalRequest[]>("pending_tool_approvals"),
  answerApproval: (id: string, answer: ApprovalAnswer) => invoke<void>("answer_tool_approval", { id, answer }),
  runToolManually: (serverId: string, tool: string, args: unknown, conversationId: string | null = null) =>
    invoke<{ status: string; content: string; audit_id: string | null }>("run_tool_manually", { serverId, tool, arguments: args, conversationId }),
  /** Per-source tool count and token cost for one chat (composer Tools menu). */
  chatToolCosts: (id: string) => invoke<ToolCosts>("chat_tool_costs", { id }),
};

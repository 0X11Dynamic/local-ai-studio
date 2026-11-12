// Typed wrappers for the phase 2 IPC commands (discover, downloads, chat,
// system instructions, local API). Types mirror the Rust serde output.
import { invoke } from "@tauri-apps/api/core";

// ---------- discover & downloads ----------
export interface SearchResult {
  id: string; author: string; downloads: number; likes: number; created_at: string | null;
  pipeline_tag: string | null; tags: string[]; license: string | null; base_model: string | null; capabilities: string[];
}
export interface RepoDetails {
  id: string; author: string; sha: string; last_modified: string | null; license: string | null;
  pipeline_tag: string | null; tags: string[]; gated: boolean; downloads: number; likes: number;
  base_model: string | null; architecture: string | null; context_length: number | null; parameters: number | null;
  capabilities: string[];
}
export interface RepoFile { path: string; size: number; sha256: string | null }
export interface VariantView {
  key: string; quantization: string | null; files: RepoFile[]; total_size: number; shard_count: number;
  complete: boolean; is_projector: boolean;
  status: "installed" | "partial" | "downloading" | "paused" | "failed" | "loaded" | "available" | "incomplete";
  model_id: string | null; download_id: string | null; est_memory_bytes: number; fit: "gpu" | "partial" | "cpu" | "too_big";
}
export interface RepoView {
  details: RepoDetails; variants: VariantView[]; recommended: string | null;
  free_vram_bytes: number | null; available_ram_bytes: number; models_dir: string;
}
export interface DownloadEvent {
  id: string; repo: string; variant_key: string;
  state: "queued" | "running" | "paused" | "completed" | "failed" | "cancelled";
  downloaded: number; total: number; bytes_per_sec: number; eta_secs: number | null; current_file: string | null;
  files_done: number; files_total: number; error: string | null; model_id: string | null;
}

// ---------- chat ----------
export interface SamplingParams {
  temperature: number | null; top_p: number | null; top_k: number | null; min_p: number | null;
  repeat_penalty: number | null; max_tokens: number | null; seed: number | null; stop: string[];
}
export interface Limits { max_tool_calls: number | null; max_wall_secs: number | null }
export interface ChatSettings {
  profile_id: string | null; sampling: SamplingParams; overflow: "warn" | "drop_oldest"; reserve_tokens: number;
  agent_mode: boolean; tool_servers: string[]; limits: Limits; builtin_tools: boolean; environment_id: string | null; skills: boolean; tool_mode: string;
  /** "chat" | "plan" | "agent"; empty in older chats (then agent_mode decides). Plan = agent_mode + mode "plan". */
  mode?: string;
  /** MCP server ids turned off for this chat only (Tools menu); servers added later stay on. */
  disabled_servers?: string[];
  /** Agent mode only: offer spawn_subagents (default off). */
  subagents?: boolean;
  /** Subagents running at once, 1-4 (default 2). */
  max_subagents?: number;
  /** Thinking on/off for models with a toggle; null = the model's default. */
  thinking?: boolean | null;
  /** Reasoning effort level (only levels the loaded model lists are sent); null = default. */
  reasoning_effort?: string | null;
}
export interface ConversationSummary { id: string; title: string; model_id: string | null; updated_at: string; message_count: number }
export interface MessageStats {
  prompt_tokens?: number | null; completion_tokens?: number | null; cached_tokens?: number | null;
  prompt_tokens_per_sec?: number | null; tokens_per_sec?: number | null; time_to_first_token_ms?: number | null;
  total_ms?: number; stop_reason?: string; context_tokens?: number; n_ctx?: number; dropped_messages?: number;
  model?: string; profile?: string | null; error?: string;
  /** chat | plan | agent: the mode the reply was generated in. */
  mode?: string;
  /** Thinking on/off that was requested (null = model default / no toggle). */
  thinking_requested?: boolean | null;
  reasoning_effort?: string | null;
  thinking_control?: string;
  chat_template_kwargs?: Record<string, unknown> | null;
  /** Files attached to the message this replies to. */
  attachments?: number;
  /** The reply had no visible answer even after one nudge (Agent/Plan mode); `note` says so. */
  empty_answer?: boolean;
  note?: string;
}
export interface Message {
  id: string; conversation_id: string; role: "user" | "assistant" | "system" | "tool"; content: string;
  reasoning: string | null; stats: string | null; created_at: string; model_name: string | null;
  tool_calls: string | null; tool_call_id: string | null;
  /** JSON list of Attachment records (see api6). */
  attachments?: string;
}
export interface ConversationView {
  id: string; title: string; model_id: string | null; settings: string; created_at: string; updated_at: string;
  settings_parsed: ChatSettings; messages: Message[]; generating: boolean;
  /** Attached on the composer, sent with the next message. */
  pending_attachments?: import("./api6").Attachment[];
}
export interface ResolvedProfile { profile_id: string; title: string; source: "conversation" | "model" | "default" | "agent_default" | "plan_default" }
export interface ContextReport {
  n_ctx: number; total: number; system: number; history: number; draft: number; template: number;
  tools: number; skills: number; reserved: number; remaining: number; fits: boolean;
  profile: ResolvedProfile | null; messages_counted: number;
  agent_mode: boolean; tool_count: number; tools_warning: boolean; supports_tools: boolean;
  /** Attached files (text blocks counted, images/audio estimated). */
  attachments?: number; pending_attachments?: number;
}
export type ChatEvent =
  | { type: "delta"; conversation_id: string; message_id: string; content: string; reasoning: string }
  | { type: "segment_start"; conversation_id: string; message: Message }
  | { type: "segment_done"; conversation_id: string; message: Message }
  | { type: "tool_result"; conversation_id: string; message: Message }
  | { type: "segment_reset"; conversation_id: string; message_id: string }
  | { type: "done"; conversation_id: string; message_id: string; message: Message }
  | { type: "error"; conversation_id: string; message_id: string | null; error: string };
export interface SendStarted { user_message: Message | null; assistant_message_id: string }

// ---------- system instructions ----------
export interface Sections { mode: "sections" | "raw"; identity: string; task: string; tool_use: string; formatting: string; raw: string }
export interface InstructionProfile { id: string; title: string; builtin: boolean; sections: Sections; updated_at: string }
export interface InstructionSettings { base: string; chat_profile_id: string | null; agent_profile_id: string | null; plan_profile_id?: string | null }
export interface InstructionPart { label: string; text: string; editable: boolean }
export interface InstructionPreview {
  text: string; parts: InstructionPart[]; chars: number; tokens: number | null; templated_tokens: number | null;
  model: string | null; n_ctx: number | null;
}
export interface ChatStatsOut {
  prompt_tokens: number | null; completion_tokens: number | null; tokens_per_sec: number | null;
  time_to_first_token_ms: number | null; total_ms: number; stop_reason: string;
}

// ---------- local API ----------
export interface ApiConfig {
  host: string; port: number; require_auth: boolean; cors_origins: string[]; max_concurrent: number;
  rate_per_minute: number; log_retention_days: number; autostart: boolean;
}
export interface ApiStatus { running: boolean; address: string | null; started_at: string | null; config: ApiConfig; last_error: string | null }
export interface ApiLog {
  id: string; endpoint: string; method: string; status: number; model: string | null; latency_ms: number;
  prompt_tokens: number | null; completion_tokens: number | null; error: string | null; created_at: string;
}

export const api2 = {
  hfSearch: (query: string, sort: string) => invoke<SearchResult[]>("hf_search", { query, sort }),
  hfRepo: (repo: string) => invoke<RepoView>("hf_repo", { repo }),
  downloadVariant: (repo: string, variantKey: string, projectorKey: string | null) =>
    invoke<string>("download_variant", { repo, variantKey, projectorKey }),
  listDownloads: () => invoke<DownloadEvent[]>("list_downloads"),
  pauseDownload: (id: string) => invoke<void>("pause_download", { id }),
  resumeDownload: (id: string) => invoke<void>("resume_download", { id }),
  cancelDownload: (id: string) => invoke<void>("cancel_download", { id }),
  clearFinishedDownloads: () => invoke<number>("clear_finished_downloads"),
  setHfToken: (token: string | null) => invoke<boolean>("set_hf_token", { token }),
  hasHfToken: () => invoke<boolean>("has_hf_token"),

  listConversations: (query?: string) => invoke<ConversationSummary[]>("list_conversations", { query }),
  createConversation: () => invoke<ConversationView>("create_conversation"),
  getConversation: (id: string) => invoke<ConversationView>("get_conversation", { id }),
  renameConversation: (id: string, title: string) => invoke<void>("rename_conversation", { id, title }),
  duplicateConversation: (id: string) => invoke<ConversationView>("duplicate_conversation", { id }),
  /** Also deletes the chat's hidden subagent run; lists git worktrees it had to keep. */
  deleteConversation: (id: string) => invoke<import("./api5").DeleteResult>("delete_conversation", { id }),
  updateChatSettings: (id: string, settings: ChatSettings) => invoke<void>("update_chat_settings", { id, settings }),
  /** Pending attachments go with it: `attachments` picks some by id, omitted = all. */
  chatSend: (id: string, content: string, attachments?: string[]) => invoke<SendStarted>("chat_send", { id, content, attachments }),
  chatStop: (id: string) => invoke<boolean>("chat_stop", { id }),
  chatRegenerate: (id: string) => invoke<SendStarted>("chat_regenerate", { id }),
  chatEdit: (id: string, messageId: string, content: string) => invoke<SendStarted>("chat_edit", { id, messageId, content }),
  chatContext: (id: string, draft: string) => invoke<ContextReport>("chat_context", { id, draft }),
  exportConversation: (id: string, path: string) => invoke<void>("export_conversation", { id, path }),

  listProfiles: () => invoke<InstructionProfile[]>("list_instruction_profiles"),
  createProfile: (title: string, sections: Sections) => invoke<string>("create_instruction_profile", { title, sections }),
  updateProfile: (id: string, title: string, sections: Sections) => invoke<void>("update_instruction_profile", { id, title, sections }),
  deleteProfile: (id: string) => invoke<void>("delete_instruction_profile", { id }),
  getInstructionSettings: () => invoke<InstructionSettings>("get_instruction_settings"),
  setInstructionSettings: (settings: InstructionSettings) => invoke<void>("set_instruction_settings", { settings }),
  resetInstructionBase: () => invoke<string>("reset_instruction_base"),
  setModelProfile: (modelId: string, profileId: string | null) => invoke<void>("set_model_instruction_profile", { modelId, profileId }),
  getModelProfile: (modelId: string) => invoke<string | null>("get_model_instruction_profile", { modelId }),
  previewInstructions: (mode: "chat" | "agent" | "plan", sections: Sections, base: string | null) =>
    invoke<InstructionPreview>("preview_instructions", { mode, sections, base }),
  testInstructions: (sections: Sections, base: string | null, prompt: string) =>
    invoke<{ content: string; reasoning: string; stats: ChatStatsOut }>("test_instructions", { sections, base, prompt }),
  exportProfile: (id: string, path: string) => invoke<void>("export_instruction_profile", { id, path }),
  importProfile: (path: string) => invoke<string>("import_instruction_profile", { path }),

  apiStatus: () => invoke<ApiStatus>("api_status"),
  apiStart: (config: ApiConfig) => invoke<ApiStatus>("api_start", { config }),
  apiSaveConfig: (config: ApiConfig) => invoke<void>("api_save_config", { config }),
  apiStop: () => invoke<ApiStatus>("api_stop"),
  apiLogs: (limit?: number) => invoke<ApiLog[]>("api_logs", { limit }),
  apiGenerateToken: () => invoke<string>("api_generate_token"),
  apiHasToken: () => invoke<boolean>("api_has_token"),
  apiRevealToken: () => invoke<string | null>("api_reveal_token"),
};

// Typed wrappers for phase 5 (agent tasks, subagents, skills).
import { invoke } from "@tauri-apps/api/core";
import type { SamplingParams } from "./api2";
import type { ApprovalRequest } from "./api3";

/** A task's limits; null = no limit (the default). Old runs hold numbers. */
export interface Budgets { max_turns: number | null; max_tool_calls: number | null; max_wall_secs: number | null; max_tokens: number | null }
export interface Usage { turns: number; tool_calls: number; wall_ms: number; tokens: number }
export interface TaskSettings {
  environment_id: string | null; builtin_tools: boolean; mcp_tools: boolean; tool_servers: string[];
  /** MCP server ids turned off for this task ("Task tools" picker); servers added later are on. */
  disabled_servers?: string[];
  skills_mode: "auto" | "off" | "manual"; skills: string[]; subagents: boolean; max_parallel: number;
  tool_mode: "auto" | "native" | "xml"; sampling: SamplingParams;
  /** Read-only planning phase first; the task waits in awaiting_plan_approval. */
  plan_first?: boolean;
  /** Set by approveTaskPlan. */
  approved_plan?: string | null;
  /** Thinking on/off for every model call of the task (null = model default). */
  thinking?: boolean | null;
  reasoning_effort?: string | null;
}
export interface RunSummary { id: string; title: string; objective: string; state: string; budgets: string; usage: string; created_at: string; ended_at: string | null }
export interface RunRow extends RunSummary {
  model_id: string | null; env_id: string | null; final_answer: string | null; error: string | null; started_at: string | null; settings: string;
}
export interface StepRow { id: string; run_id: string; parent_agent_id: string | null; seq: number; kind: string; data: string; state: string; started_at: string; ended_at: string | null }
export interface SubagentRow {
  id: string; parent_run_id: string; parent_agent_id: string | null; objective: string; state: string; budgets: string; usage: string;
  result: string | null; created_at: string; ended_at: string | null; label: string; settings: string; transcript?: string; error: string | null; started_at: string | null;
}
/** A subagent's git worktree, reported when its task or chat is deleted. */
export interface WorktreeReport { subagent_id: string; label: string; path: string; branch: string; reason: string | null }
export interface DeleteResult { removed_worktrees: WorktreeReport[]; kept_worktrees: WorktreeReport[] }
/** Stored in a subagent's settings when it ran with isolation "worktree". */
export interface WorktreeInfo { repo: string; path: string; branch: string; base: string }
export interface SubagentSettings { tools?: string[]; environment?: string; call_id?: string; conversation_id?: string | null; worktree?: WorktreeInfo | null }
/** The hidden run that holds a chat's subagents. */
export interface ChatRun { id: string; title: string; conversation_id: string }
export interface ChatSubagents { run_id: string | null; running: boolean; subagents: SubagentRow[] }
/** An approval request as a subagent sends it: agent_id is the subagent's id (null for the main agent or a plain chat). */
export type SubagentApproval = ApprovalRequest & { agent_id?: string | null };
export interface TaskDetail { run: RunRow; steps: StepRow[]; subagents: SubagentRow[]; running: boolean }
export type TaskEvent =
  | { run_id: string; agent_id: string | null; type: "step"; step: StepRow }
  | { run_id: string; agent_id: string | null; type: "step_done"; step_id: string; data: Record<string, unknown> }
  | { run_id: string; agent_id: string | null; type: "delta"; step_id: string | null; content: string; reasoning: string }
  | { run_id: string; agent_id: string | null; type: "reset"; step_id: string | null }
  | { run_id: string; agent_id: string | null; type: "usage"; usage: Usage }
  | { run_id: string; agent_id?: string | null; type: "state"; state: string; error?: string | null; answer?: string | null }
  | { run_id: string; agent_id: string; type: "subagent_state"; state: string; usage?: Usage }
  /** An agent called ask_user (its entry is a step of kind "question", state "waiting"). */
  | { run_id: string; agent_id?: string | null; type: "question"; question_id: string }
  /** A question was answered, cancelled or closed (state done | cancelled). */
  | { run_id: string; agent_id?: string | null; type: "question_closed"; question_id?: string; state: string };

/** The data of a step of kind "question" (an agent's ask_user call). */
export interface QuestionData {
  question_id: string; question: string; options: string[]; allow_text: boolean; call_id: string; agent_label: string;
  answer: string | null; answered_at?: string; cancelled?: boolean;
}
/** The data of a step of kind "user" (a message sent to the task). */
export interface UserMessageData { text: string; delivered: boolean; note?: string }
/** What send_task_message did: answered the waiting question, queued the message, or resumed a stopped task. */
export type MessageOutcome = "answered" | "queued" | "resumed";

export interface FileView { path: string; sha256: string; size: number; enabled: boolean; changed: boolean }
export interface SkillView {
  id: string; name: string; description: string; version: string; keywords: string[]; mode: "auto" | "always" | "off"; source: string;
  content: string; path: string; files: FileView[]; pending: number; error: string | null;
}
export interface SkillRevision { id: string; skill_id: string; version: string; content_hash: string; content: string; note: string | null; created_at: string; state: string; proposed_by: string | null }
export interface SkillMatch { name: string; score: number; reason: string }

/** New tasks: no limits (Stop always works; a 500-turn safety stop per start still applies). */
export const DEFAULT_BUDGETS: Budgets = { max_turns: null, max_tool_calls: null, max_wall_secs: null, max_tokens: null };
/** The numbers a limit starts at when its "Limit" box is ticked. */
export const SUGGESTED_LIMITS = { max_turns: 40, max_tool_calls: 60, max_wall_secs: 1800, max_tokens: 60000 } as const;
/** "12/40" with a limit, "12" with none (null). */
export const usedOf = (used: number, max: number | null | undefined, fmt: (n: number) => string = (n) => n.toLocaleString()) =>
  max == null ? fmt(used) : `${fmt(used)}/${fmt(max)}`;

export const api5 = {
  startTask: (title: string, objective: string, settings: TaskSettings, budgets: Budgets) => invoke<string>("start_task", { title, objective, settings, budgets }),
  listTasks: () => invoke<RunSummary[]>("list_tasks"),
  getTask: (id: string) => invoke<TaskDetail>("get_task", { id }),
  pauseTask: (id: string) => invoke<void>("pause_task", { id }),
  resumeTask: (id: string, budgets: Budgets | null = null) => invoke<void>("resume_task", { id, budgets }),
  cancelTask: (id: string) => invoke<void>("cancel_task", { id }),
  /** Approve a "Plan first" task's plan (null = as written) and run it. */
  approveTaskPlan: (id: string, plan: string | null = null) => invoke<void>("approve_task_plan", { id, plan }),
  deleteTask: (id: string) => invoke<DeleteResult>("delete_task", { id }),
  /** Answer an agent's ask_user question; that agent (or only that subagent) continues. */
  answerTaskQuestion: (runId: string, questionId: string, answer: string) => invoke<void>("answer_task_question", { runId, questionId, answer }),
  /** Message a task: read before its next model turn; answers a waiting question; with resume, restarts a stopped task. */
  sendTaskMessage: (runId: string, text: string, resume = false, budgets: Budgets | null = null) => invoke<MessageOutcome>("send_task_message", { runId, text, resume, budgets }),
  /** Questions waiting for the user in any task (steps of kind "question"). */
  pendingTaskQuestions: () => invoke<StepRow[]>("pending_task_questions"),
  listChatRuns: () => invoke<ChatRun[]>("list_chat_runs"),
  /** A chat's subagents (settings.call_id says which spawn call started each). */
  chatSubagents: (conversationId: string) => invoke<ChatSubagents>("chat_subagents", { id: conversationId }),
  stopSubagent: (id: string) => invoke<void>("stop_subagent", { id }),
  recentSubagents: () => invoke<SubagentRow[]>("recent_subagents"),
  listSkills: () => invoke<SkillView[]>("list_skills"),
  saveSkill: (id: string | null, content: string) => invoke<SkillView>("save_skill", { id, content }),
  deleteSkill: (id: string) => invoke<void>("delete_skill", { id }),
  setSkillMode: (id: string, mode: string) => invoke<void>("set_skill_mode", { id, mode }),
  setSkillScript: (id: string, path: string, enabled: boolean) => invoke<void>("set_skill_script", { id, path, enabled }),
  revisions: (id: string) => invoke<SkillRevision[]>("skill_revisions", { id }),
  revert: (revisionId: string) => invoke<SkillView>("revert_skill", { revisionId }),
  approve: (revisionId: string) => invoke<SkillView>("approve_skill_proposal", { revisionId }),
  reject: (revisionId: string) => invoke<void>("reject_skill_proposal", { revisionId }),
  importSkill: (path: string) => invoke<SkillView>("import_skill", { path }),
  exportSkill: (id: string, dest: string) => invoke<string>("export_skill", { id, dest }),
  matchSkills: (text: string) => invoke<SkillMatch[]>("match_skills", { text }),
};

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Parse a JSON column. Rows can hold partial shapes (a new run's usage is "{}"
 * until its first turn is saved; "null"; older rows lack newer fields), so an
 * object fallback is merged under the parsed object: missing or null fields
 * keep the fallback's value. A value of the wrong kind gives the fallback.
 */
export function parseJson<T>(s: string | null | undefined, fallback: T): T {
  let v: unknown;
  try {
    v = s ? JSON.parse(s) : null;
  } catch {
    return fallback;
  }
  if (v == null) return fallback;
  if (isObj(fallback)) {
    if (!isObj(v)) return fallback;
    const out: Record<string, unknown> = { ...fallback };
    for (const [k, x] of Object.entries(v)) if (x != null || !(k in fallback)) out[k] = x;
    return out as T;
  }
  if (Array.isArray(fallback) && !Array.isArray(v)) return fallback;
  return v as T;
}

export const STATE_LABEL: Record<string, string> = {
  queued: "Queued", running: "Running", waiting: "Waiting for approval", waiting_for_user: "Needs your answer", paused: "Paused", completed: "Completed", failed: "Failed",
  cancelled: "Cancelled", budget_exhausted: "Budget used up", interrupted: "Interrupted", awaiting_plan_approval: "Plan ready for review",
};

/** Ask the Agent Tasks page to open a task (used by the Subagents page and the approvals indicator). */
export const OPEN_TASK_KEY = "las.openTask";
/** Ask the Chat page to open a conversation. */
export const OPEN_CHAT_KEY = "las.openChat";

// When the page is already open it will not remount, so also tell it directly.
export const openTask = (id: string) => window.dispatchEvent(new CustomEvent<string>("las:open-task", { detail: id }));
export const openChat = (id: string) => window.dispatchEvent(new CustomEvent<string>("las:open-chat", { detail: id }));

/** Subscribe to openTask/openChat requests; returns the unsubscribe function. */
export function onOpen(kind: "task" | "chat", fn: (id: string) => void): () => void {
  const h = (e: Event) => fn((e as CustomEvent<string>).detail);
  window.addEventListener(`las:open-${kind}`, h);
  return () => window.removeEventListener(`las:open-${kind}`, h);
}

/** Plain text about git worktrees a delete had to keep (null when none were kept). */
export function keptNote(r: DeleteResult | null | undefined): string | null {
  const kept = r?.kept_worktrees ?? [];
  if (kept.length === 0) return null;
  const lines = kept.map((w) => `${w.branch} in ${w.path}${w.label ? ` (subagent "${w.label}")` : ""}: ${w.reason ?? "kept"}`);
  return `Kept ${kept.length} git worktree${kept.length === 1 ? "" : "s"} so no work is lost (never forced):\n${lines.join("\n")}`;
}

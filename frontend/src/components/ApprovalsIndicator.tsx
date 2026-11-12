import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api3, type ApprovalRequest } from "../lib/api3";
import { OPEN_CHAT_KEY, OPEN_TASK_KEY, openChat, openTask } from "../lib/api5";
import type { PageId } from "../App";

/**
 * Tool calls waiting for approval anywhere in the app (any chat, any task,
 * any subagent). Without this, a task you are not looking at would wait
 * unseen and be denied after 10 minutes.
 */
export function ApprovalsIndicator({ go }: { go: (p: PageId) => void }) {
  const [pending, setPending] = useState<ApprovalRequest[]>([]);

  useEffect(() => {
    let cancelled = false;
    api3.pendingApprovals().then((p) => !cancelled && setPending(p)).catch(() => {});
    const a = listen<ApprovalRequest>("tool://approval", (e) => {
      setPending((x) => (x.some((p) => p.id === e.payload.id) ? x : [...x, e.payload]));
    });
    const c = listen<{ id: string }>("tool://approval-closed", (e) => {
      setPending((x) => x.filter((p) => p.id !== e.payload.id));
    });
    return () => {
      cancelled = true;
      a.then((f) => f());
      c.then((f) => f());
    };
  }, []);

  if (pending.length === 0) return null;
  const first = pending[0];
  const where = (p: ApprovalRequest) => (p.run_id ? `${p.agent ?? "a task"}${p.conversation_id ? " in a chat" : ""}` : p.conversation_id ? "a chat" : "a manual try");
  const title = pending.map((p) => `${p.tool_name} on ${p.server_name}, asked by ${where(p)}`).join("\n");
  const goTo = () => {
    // A chat's subagents carry both ids: their approvals are answered in the chat.
    if (first.conversation_id) {
      localStorage.setItem(OPEN_CHAT_KEY, first.conversation_id);
      openChat(first.conversation_id);
      go("chat");
    } else if (first.run_id) {
      localStorage.setItem(OPEN_TASK_KEY, first.run_id);
      openTask(first.run_id);
      go("agents");
    }
  };
  return (
    <button className="btn sm approvals-pill" onClick={goTo} title={title} aria-label={`${pending.length} waiting for approval`}>
      {pending.length} waiting for approval
    </button>
  );
}

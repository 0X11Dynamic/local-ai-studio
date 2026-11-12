import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api3, type ApprovalRequest } from "../lib/api3";
import { OPEN_CHAT_KEY, OPEN_TASK_KEY, api5, openChat, openTask, parseJson, type QuestionData, type StepRow, type TaskEvent } from "../lib/api5";
import type { PageId } from "../App";

/**
 * Things waiting for the user anywhere in the app: tool calls waiting for
 * approval (any chat, any task, any subagent) and questions an agent task
 * asked with ask_user. Without this, a task you are not looking at would
 * wait unseen (approvals are denied after 10 minutes).
 */
export function ApprovalsIndicator({ go }: { go: (p: PageId) => void }) {
  const [pending, setPending] = useState<ApprovalRequest[]>([]);
  const [questions, setQuestions] = useState<StepRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    api3.pendingApprovals().then((p) => !cancelled && setPending(p)).catch(() => {});
    const refreshQuestions = () => api5.pendingTaskQuestions().then((q) => !cancelled && setQuestions(q)).catch(() => {});
    refreshQuestions();
    const a = listen<ApprovalRequest>("tool://approval", (e) => {
      setPending((x) => (x.some((p) => p.id === e.payload.id) ? x : [...x, e.payload]));
    });
    const c = listen<{ id: string }>("tool://approval-closed", (e) => {
      setPending((x) => x.filter((p) => p.id !== e.payload.id));
    });
    // A question appears with a "question" event and goes away when answered,
    // cancelled or its task ends.
    const t = listen<TaskEvent>("task://event", (e) => {
      if (e.payload.type === "question" || e.payload.type === "question_closed" || e.payload.type === "state") refreshQuestions();
    });
    return () => {
      cancelled = true;
      a.then((f) => f());
      c.then((f) => f());
      t.then((f) => f());
    };
  }, []);

  if (pending.length === 0 && questions.length === 0) return null;
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
  const q = questions[0];
  const qTitle = questions
    .map((s) => {
      const d = parseJson<QuestionData>(s.data, { question_id: "", question: "", options: [], allow_text: true, call_id: "", agent_label: "A task", answer: null });
      return `${d.agent_label}: ${d.question}`;
    })
    .join("\n");
  const goToQuestion = () => {
    localStorage.setItem(OPEN_TASK_KEY, q.run_id);
    openTask(q.run_id);
    go("agents");
  };
  return (
    <>
      {pending.length > 0 && (
        <button className="btn sm approvals-pill" onClick={goTo} title={title} aria-label={`${pending.length} waiting for approval`}>
          {pending.length} waiting for approval
        </button>
      )}
      {questions.length > 0 && (
        <button className="btn sm approvals-pill" onClick={goToQuestion} title={qTitle} aria-label={`${questions.length} task question${questions.length === 1 ? "" : "s"} waiting for your answer`}>
          {questions.length === 1 ? "Task needs your answer" : `${questions.length} task questions need your answer`}
        </button>
      )}
    </>
  );
}

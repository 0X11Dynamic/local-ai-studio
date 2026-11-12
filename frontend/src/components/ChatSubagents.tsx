import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { errorMessage } from "../lib/api";
import type { ApprovalAnswer } from "../lib/api3";
import { api5, STATE_LABEL, parseJson, usedOf, type Budgets, type SubagentApproval, type SubagentRow, type SubagentSettings, type TaskEvent, type Usage } from "../lib/api5";

const ACTIVE = ["queued", "running", "waiting"];

/**
 * Live children of one spawn_subagents call in a chat: state, budget use,
 * worktree, result, and any approval they wait for (answered right here).
 */
export function ChatSubagents({
  conversationId, callId, approvals, onAnswer, onOpenPage,
}: {
  conversationId: string;
  callId: string;
  /** Pending approvals of this chat (the children's carry run_id and agent_id). */
  approvals: SubagentApproval[];
  onAnswer: (id: string, a: ApprovalAnswer) => void;
  onOpenPage: () => void;
}) {
  const [runId, setRunId] = useState<string | null>(null);
  const [subs, setSubs] = useState<SubagentRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await api5.chatSubagents(conversationId);
      setRunId(r.run_id);
      setSubs(r.subagents.filter((s) => parseJson<SubagentSettings>(s.settings, {}).call_id === callId));
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, [conversationId, callId]);

  useEffect(() => {
    refresh();
    // State changes only (not streamed text), at most every 300 ms.
    let timer: ReturnType<typeof setTimeout> | null = null;
    const un = listen<TaskEvent>("task://event", (e) => {
      if (!["state", "subagent_state", "step", "usage"].includes(e.payload.type) || timer) return;
      timer = setTimeout(() => {
        timer = null;
        refresh();
      }, 300);
    });
    return () => {
      if (timer) clearTimeout(timer);
      un.then((f) => f());
    };
  }, [refresh]);

  if (err) return <div className="faint" style={{ fontSize: 12 }}>Subagents: {err}</div>;
  if (subs == null) return null;
  if (subs.length === 0) return <div className="faint" style={{ fontSize: 12 }}>No subagent was started by this call.</div>;
  return (
    <div className="chat-subagents" aria-label="Subagents of this call">
      <div className="row">
        <span className="faint" style={{ fontSize: 12 }}>{subs.length} subagent{subs.length === 1 ? "" : "s"}</span>
        <span className="spacer" />
        <button className="btn sm ghost" onClick={onOpenPage}>Open Subagents page</button>
      </div>
      <ul className="sub-tree">
        {subs.map((s) => {
          const u = parseJson<Usage>(s.usage, { turns: 0, tool_calls: 0, wall_ms: 0, tokens: 0 });
          const b = parseJson<Budgets>(s.budgets, { max_turns: null, max_tool_calls: null, max_wall_secs: null, max_tokens: null });
          const wt = parseJson<SubagentSettings>(s.settings, {}).worktree;
          // By the subagent's id: two children may share a label.
          const waiting = approvals.filter((a) => a.run_id === runId && a.agent_id === s.id);
          return (
            <li key={s.id} aria-label={`Subagent ${s.label}`} data-state={s.state}>
              <div className="row">
                <span className="mono">{s.label || "subagent"}</span>
                <span className={`tag${ACTIVE.includes(s.state) ? " accent" : ""}`} aria-label="Subagent state">{waiting.length ? "Waiting for approval" : STATE_LABEL[s.state] ?? s.state}</span>
                <span className="faint mono" style={{ fontSize: 11.5 }}>
                  {usedOf(u.turns, b.max_turns)} turns, {usedOf(u.tool_calls, b.max_tool_calls)} calls
                </span>
                <span className="spacer" />
                {ACTIVE.includes(s.state) && (
                  <button className="btn sm ghost danger" aria-label={`Stop subagent ${s.label}`} onClick={() => api5.stopSubagent(s.id).then(refresh).catch((e) => setErr(errorMessage(e)))}>Stop</button>
                )}
              </div>
              <div className="dim" style={{ fontSize: 12.5 }}>{s.objective}</div>
              {wt && <div className="faint mono" style={{ fontSize: 11.5 }} aria-label="Worktree">Worktree: {wt.branch} in {wt.path}</div>}
              {waiting.map((a) => (
                <div key={a.id} className="chat-sub-approval">
                  <span className="dim">{a.agent} wants to run <span className="mono">{a.tool_name}</span> on {a.server_name}. {a.reason}.</span>
                  <pre className="tool-args mono">{JSON.stringify(a.arguments, null, 2)}</pre>
                  <div className="row">
                    <button className="btn sm primary" onClick={() => onAnswer(a.id, "allow_once")}>Allow once</button>
                    <button className="btn sm" onClick={() => onAnswer(a.id, "allow_session")}>Allow for this chat's subagents</button>
                    <button className="btn sm" onClick={() => onAnswer(a.id, "allow_always")}>Always allow this tool</button>
                    <button className="btn sm danger" onClick={() => onAnswer(a.id, "deny")}>Deny</button>
                  </div>
                </div>
              ))}
              {s.error && <div className="danger-text" style={{ fontSize: 12.5 }}>{s.error}</div>}
              {s.result && (
                <details>
                  <summary className="faint" style={{ fontSize: 12 }}>Result</summary>
                  <pre className="preview-text" aria-label={`Result of ${s.label}`}>{s.result}</pre>
                </details>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

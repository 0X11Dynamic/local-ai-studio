import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { errorMessage } from "../lib/api";
import {
  api5, OPEN_CHAT_KEY, OPEN_TASK_KEY, STATE_LABEL, openChat, openTask, parseJson,
  type Budgets, type ChatRun, type RunSummary, type SubagentRow, type SubagentSettings, type TaskEvent, type Usage,
} from "../lib/api5";
import type { PageId } from "../App";

const ACTIVE = ["queued", "running", "waiting"];

export function Subagents({ go }: { go: (p: PageId) => void }) {
  const [subs, setSubs] = useState<SubagentRow[] | null>(null);
  const [tasks, setTasks] = useState<RunSummary[]>([]);
  const [chats, setChats] = useState<ChatRun[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, t, c] = await Promise.all([api5.recentSubagents(), api5.listTasks(), api5.listChatRuns()]);
      setSubs(s);
      setTasks(t);
      setChats(c);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // Only changes that affect this page (not streamed text), at most every 500 ms.
    let timer: ReturnType<typeof setTimeout> | null = null;
    const un = listen<TaskEvent>("task://event", (e) => {
      if (!["state", "subagent_state", "step"].includes(e.payload.type) || timer) return;
      timer = setTimeout(() => {
        timer = null;
        refresh();
      }, 500);
    });
    return () => {
      if (timer) clearTimeout(timer);
      un.then((f) => f());
    };
  }, [refresh]);

  const groups = useMemo(() => {
    const by = new Map<string, SubagentRow[]>();
    for (const s of subs ?? []) by.set(s.parent_run_id, [...(by.get(s.parent_run_id) ?? []), s]);
    return [...by.entries()].map(([run, list]) => ({
      run, task: tasks.find((t) => t.id === run), chat: chats.find((c) => c.id === run), list: list.sort((a, b) => a.created_at.localeCompare(b.created_at)),
    }));
  }, [subs, tasks, chats]);

  const openConversation = (id: string) => {
    localStorage.setItem(OPEN_CHAT_KEY, id);
    openChat(id);
    go("chat");
  };

  const open = (runId: string) => {
    localStorage.setItem(OPEN_TASK_KEY, runId);
    openTask(runId);
    go("agents");
  };

  return (
    <div className="page">
      <p className="lede">Subagents are started by a task's main agent, or by a chat reply when the chat allows subagents, to work on independent parts in parallel. Each gets a share of the budget and only tools the main agent has.</p>
      {err && <div className="notice error" role="alert" style={{ marginBottom: 10 }}>{err}</div>}
      {subs == null ? (
        <div className="panel"><div className="skeleton" /></div>
      ) : groups.length === 0 ? (
        <div className="empty">No subagents yet. They appear here when a task's agent delegates work (Allow subagents is on by default for new tasks).</div>
      ) : (
        <div style={{ display: "grid", gap: 14 }}>
          {groups.map((g) => (
            <div key={g.run} className="panel" style={{ display: "grid", gap: 8 }}>
              <div className="row">
                <strong style={{ fontWeight: 600 }}>{g.chat?.title || g.task?.title || g.task?.objective || "Task"}</strong>
                {g.task && <span className="tag">{STATE_LABEL[g.task.state] ?? g.task.state}</span>}
                {g.chat && <span className="tag">Chat</span>}
                <span className="spacer" />
                {g.chat ? (
                  <button className="btn sm ghost" onClick={() => openConversation(g.chat!.conversation_id)}>Open chat</button>
                ) : (
                  <button className="btn sm ghost" onClick={() => open(g.run)}>Open task</button>
                )}
              </div>
              <ul className="sub-tree">
                {g.list.map((s) => {
                  const u = parseJson<Usage>(s.usage, { turns: 0, tool_calls: 0, wall_ms: 0, tokens: 0 });
                  const b = parseJson<Budgets>(s.budgets, { max_turns: 0, max_tool_calls: 0, max_wall_secs: 0, max_tokens: 0 });
                  return (
                    <li key={s.id}>
                      <div className="row">
                        <span className="mono">{s.label || "subagent"}</span>
                        <span className={`tag${ACTIVE.includes(s.state) ? " accent" : ""}`}>{STATE_LABEL[s.state] ?? s.state}</span>
                        <span className="faint mono" style={{ fontSize: 11.5 }}>
                          {u.turns}/{b.max_turns} turns, {u.tool_calls}/{b.max_tool_calls} calls, {u.tokens.toLocaleString()}/{b.max_tokens.toLocaleString()} tokens
                        </span>
                        <span className="spacer" />
                        {ACTIVE.includes(s.state) && (
                          <button className="btn sm ghost danger" aria-label={`Stop subagent ${s.label}`} onClick={() => api5.stopSubagent(s.id).then(refresh).catch((e) => setErr(errorMessage(e)))}>Stop</button>
                        )}
                      </div>
                      <div className="dim" style={{ fontSize: 12.5 }}>{s.objective}</div>
                      {(() => {
                        const wt = parseJson<SubagentSettings>(s.settings, {}).worktree;
                        return wt ? <div className="faint mono" style={{ fontSize: 11.5 }}>Worktree: {wt.branch} in {wt.path}</div> : null;
                      })()}
                      {s.error && <div className="danger-text" style={{ fontSize: 12.5 }}>{s.error}</div>}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

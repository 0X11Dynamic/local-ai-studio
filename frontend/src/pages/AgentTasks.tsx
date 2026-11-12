import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { errorMessage } from "../lib/api";
import { api3, type ApprovalAnswer, type ApprovalRequest, type ToolEventPayload } from "../lib/api3";
import { api4, type EnvView } from "../lib/api4";
import {
  api5, DEFAULT_BUDGETS, OPEN_TASK_KEY, STATE_LABEL, keptNote, onOpen, parseJson,
  type Budgets, type RunSummary, type StepRow, type SubagentRow, type TaskDetail, type TaskEvent, type TaskSettings, type Usage,
} from "../lib/api5";
import { Markdown } from "../components/Markdown";
import { ToolCard, type ToolCall } from "../components/ToolCard";
import type { Message } from "../lib/api2";
import { ThinkingControls, useModelSupport } from "../components/ThinkingControls";

const BLANK_SETTINGS: TaskSettings = {
  environment_id: null, builtin_tools: true, mcp_tools: false, tool_servers: [], skills_mode: "auto", skills: [],
  subagents: true, max_parallel: 2, tool_mode: "auto", plan_first: false, thinking: null, reasoning_effort: null,
  sampling: { temperature: null, top_p: null, top_k: null, min_p: null, repeat_penalty: null, max_tokens: null, seed: null, stop: [] },
};
const ACTIVE = ["queued", "running", "waiting"];
const num = (v: string) => Math.max(0, Number(v.replace(/[^\d]/g, "")) || 0);

function Meter({ label, used, max, fmt }: { label: string; used: number; max: number; fmt?: (n: number) => string }) {
  const pct = max > 0 ? Math.min(100, (used / max) * 100) : 0;
  const f = fmt ?? ((n: number) => n.toLocaleString());
  return (
    <div className="budget" aria-label={`${label} used`}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="faint">{label}</span>
        <span className="mono" style={{ fontSize: 12 }}>{f(used)} / {f(max)}</span>
      </div>
      <div className={`meter${pct > 85 ? " hot" : ""}`}><span style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

const mins = (ms: number) => `${Math.round(ms / 60000)} min`;

/** One agent's steps (root or a subagent), rendered as a timeline. */
function Timeline({
  steps, live, approvals, liveTools, liveOut, callServer, onAnswer, subagents, onStop,
}: {
  steps: StepRow[];
  live: Record<string, { content: string; reasoning: string }>;
  approvals: Record<string, ApprovalRequest>;
  liveTools: Record<string, string>;
  liveOut: Record<string, string>;
  callServer: Record<string, string>;
  onAnswer: (id: string, a: ApprovalAnswer) => void;
  subagents: SubagentRow[];
  onStop: (id: string) => void;
}) {
  const results = useMemo(() => {
    const m: Record<string, Message> = {};
    for (const s of steps.filter((x) => x.kind === "tool")) {
      const d = parseJson<Record<string, string>>(s.data, {});
      m[d.call_id] = {
        id: s.id, conversation_id: "", role: "tool", content: d.content ?? "", reasoning: null, created_at: s.started_at, model_name: null,
        tool_calls: null, tool_call_id: d.call_id, stats: JSON.stringify({ status: d.status }),
      };
    }
    return m;
  }, [steps]);
  if (steps.length === 0) return <div className="faint">No steps yet.</div>;
  return (
    <div className="timeline">
      {steps.filter((s) => s.kind !== "tool").map((s) => {
        const d = parseJson<Record<string, unknown>>(s.data, {});
        if (s.kind === "model") {
          const text = (s.state === "running" ? live[s.id]?.content : (d.text as string)) ?? "";
          const reasoning = (s.state === "running" ? live[s.id]?.reasoning : (d.reasoning as string)) ?? "";
          const calls = (d.tool_calls as ToolCall[] | undefined) ?? [];
          return (
            <div key={s.id} className="step">
              <div className="step-head faint mono">#{s.seq} model{s.state === "running" ? ", writing" : ""}</div>
              {reasoning && <details className="reasoning"><summary className="faint">Reasoning</summary><div className="reasoning-text">{reasoning}</div></details>}
              {/* While streaming, plain text: re-parsing Markdown on every token is costly. */}
              {s.state === "running" ? (text ? <div className="live-text">{text}</div> : <div className="faint">Waiting for the first token</div>) : text ? <Markdown text={text} /> : null}
              {calls.map((c) => (
                <ToolCard key={c.id} call={c} result={results[c.id]} approval={approvals[c.id]} live={liveTools[c.id]} output={liveOut[c.id]}
                  where={callServer[c.id] ?? approvals[c.id]?.server_name} onAnswer={onAnswer} sessionLabel="Allow for this task" />
              ))}
            </div>
          );
        }
        if (s.kind === "subagents") {
          const kids = (d.children as { id: string; label: string; objective: string }[]) ?? [];
          return (
            <div key={s.id} className="step">
              <div className="step-head faint mono">#{s.seq} started {kids.length} subagent{kids.length === 1 ? "" : "s"}</div>
              <ul className="sub-tree">
                {kids.map((k) => {
                  const row = subagents.find((x) => x.id === k.id);
                  const u = parseJson<Usage>(row?.usage, { turns: 0, tool_calls: 0, wall_ms: 0, tokens: 0 });
                  return (
                    <li key={k.id}>
                      <span className="mono">{k.label}</span> <span className="tag">{STATE_LABEL[row?.state ?? "queued"] ?? row?.state}</span>{" "}
                      <span className="faint">{k.objective}</span>{" "}
                      <span className="faint mono" style={{ fontSize: 11.5 }}>{u.turns} turns, {u.tool_calls} calls</span>
                      {row && ACTIVE.includes(row.state) && <button className="btn sm ghost danger" onClick={() => onStop(k.id)} aria-label={`Stop subagent ${k.label}`}>Stop</button>}
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        }
        if (s.kind === "plan") {
          return (
            <div key={s.id} className="step plan-step">
              <div className="step-head faint mono">#{s.seq} plan</div>
              <Markdown text={String(d.text ?? "")} />
            </div>
          );
        }
        return (
          <div key={s.id} className={`step note${s.kind === "error" ? " danger-text" : ""}`}>
            <span className="faint mono">#{s.seq}</span> {String(d.text ?? s.kind)}
          </div>
        );
      })}
    </div>
  );
}

export function AgentTasks() {
  const [tasks, setTasks] = useState<RunSummary[] | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [creating, setCreating] = useState(false);
  const [envs, setEnvs] = useState<EnvView[]>([]);
  const [title, setTitle] = useState("");
  const [objective, setObjective] = useState("");
  const [settings, setSettings] = useState<TaskSettings>(BLANK_SETTINGS);
  // The loaded model's thinking controls (the task uses the loaded model).
  const [support] = useModelSupport();
  const [budgets, setBudgets] = useState<Budgets>(DEFAULT_BUDGETS);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Git worktrees a deleted task's subagents left behind (changes or unmerged commits).
  const [wtNote, setWtNote] = useState<string | null>(null);
  const [live, setLive] = useState<Record<string, { content: string; reasoning: string }>>({});
  const [approvals, setApprovals] = useState<Record<string, ApprovalRequest>>({});
  const [liveTools, setLiveTools] = useState<Record<string, string>>({});
  const [liveOut, setLiveOut] = useState<Record<string, string>>({});
  const [callServer, setCallServer] = useState<Record<string, string>>({});
  const [viewAgent, setViewAgent] = useState<string | null>(null);
  const selRef = useRef<string | null>(null);
  selRef.current = selId;

  const refreshList = useCallback(async () => {
    try {
      setTasks(await api5.listTasks());
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);

  const load = useCallback(async (id: string) => {
    try {
      const d = await api5.getTask(id);
      if (selRef.current === id) setDetail(d);
      const pend = await api3.pendingApprovals();
      setApprovals(Object.fromEntries(pend.filter((p) => p.run_id === id).map((p) => [p.call_id, p])));
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    refreshList().then(() => {
      const want = localStorage.getItem(OPEN_TASK_KEY);
      if (want) {
        localStorage.removeItem(OPEN_TASK_KEY);
        setSelId(want);
      }
    });
    api4.list().then(setEnvs).catch(() => {});
    return onOpen("task", (id) => {
      localStorage.removeItem(OPEN_TASK_KEY);
      setCreating(false);
      setSelId(id);
    });
  }, [refreshList]);

  useEffect(() => {
    setDetail(null);
    setLive({});
    setLiveTools({});
    setLiveOut({});
    setCallServer({});
    setViewAgent(null);
    if (selId) load(selId);
  }, [selId, load]);

  // Live updates.
  useEffect(() => {
    const t = listen<TaskEvent>("task://event", (e) => {
      const ev = e.payload;
      if (ev.type === "state") refreshList();
      if (ev.run_id !== selRef.current) return;
      if (ev.type === "step") {
        // Append the new step; a "subagents" step means new subagent rows to fetch.
        setDetail((d) => (d && !d.steps.some((x) => x.id === ev.step.id) ? { ...d, steps: [...d.steps, ev.step] } : d));
        if (ev.step.kind === "subagents") load(ev.run_id);
      } else if (ev.type === "step_done") {
        setDetail((d) => d && { ...d, steps: d.steps.map((x) => (x.id === ev.step_id ? { ...x, data: JSON.stringify(ev.data), state: "done" } : x)) });
        setLive((x) => {
          const n = { ...x };
          delete n[ev.step_id];
          return n;
        });
      } else if (ev.type === "delta") {
        const id = ev.step_id;
        if (!id) return;
        setLive((x) => {
          const cur = x[id] ?? { content: "", reasoning: "" };
          return { ...x, [id]: { content: cur.content + ev.content, reasoning: cur.reasoning + ev.reasoning } };
        });
      } else if (ev.type === "reset") {
        if (ev.step_id) setLive((x) => ({ ...x, [ev.step_id as string]: { content: "", reasoning: "" } }));
      } else if (ev.type === "usage") {
        setDetail((d) => {
          if (!d) return d;
          if (ev.agent_id) return { ...d, subagents: d.subagents.map((s) => (s.id === ev.agent_id ? { ...s, usage: JSON.stringify(ev.usage) } : s)) };
          return { ...d, run: { ...d.run, usage: JSON.stringify(ev.usage) } };
        });
      } else {
        load(ev.run_id);
      }
    });
    const a = listen<ApprovalRequest>("tool://approval", (e) => {
      if (e.payload.run_id && e.payload.run_id === selRef.current) setApprovals((x) => ({ ...x, [e.payload.call_id]: e.payload }));
    });
    const c = listen<{ id: string }>("tool://approval-closed", (e) => {
      setApprovals((x) => Object.fromEntries(Object.entries(x).filter(([, v]) => v.id !== e.payload.id)));
    });
    const te = listen<ToolEventPayload>("tool://event", (e) => {
      if (!e.payload.run_id || e.payload.run_id !== selRef.current) return;
      const ev = e.payload.event;
      if (ev.phase === "output") {
        // Live shell output, last 20k characters per call.
        setLiveOut((x) => ({ ...x, [ev.call_id]: ((x[ev.call_id] ?? "") + ev.text).slice(-20000) }));
        return;
      }
      if (ev.phase === "started") {
        const sn = ev.server_name;
        setCallServer((x) => ({ ...x, [ev.call_id]: sn }));
      }
      setLiveTools((x) => ({ ...x, [ev.call_id]: ev.phase === "started" ? "Running" : ev.phase === "awaiting_approval" ? "Waiting for your approval" : "Finished" }));
    });
    return () => {
      t.then((f) => f());
      a.then((f) => f());
      c.then((f) => f());
      te.then((f) => f());
    };
  }, [load, refreshList]);

  const [planDraft, setPlanDraft] = useState<{ id: string; text: string } | null>(null);
  const act = async (fn: () => Promise<unknown>) => {
    setErr(null);
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
      refreshList();
      if (selRef.current) load(selRef.current);
    }
  };

  const answer = async (id: string, a: ApprovalAnswer) => {
    try {
      await api3.answerApproval(id, a);
      setApprovals((x) => Object.fromEntries(Object.entries(x).filter(([, v]) => v.id !== id)));
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const start = () =>
    act(async () => {
      const id = await api5.startTask(title, objective, settings, budgets);
      setCreating(false);
      setObjective("");
      setTitle("");
      setSelId(id);
    });

  const run = detail?.run;
  const runBudgets = parseJson<Budgets>(run?.budgets, DEFAULT_BUDGETS);
  const usage = parseJson<Usage>(run?.usage, { turns: 0, tool_calls: 0, wall_ms: 0, tokens: 0 });
  const state = run?.state ?? "";
  const running = detail?.running ?? false;
  const rootSteps = (detail?.steps ?? []).filter((s) => (viewAgent ? s.parent_agent_id === viewAgent : !s.parent_agent_id));
  const pendingCount = Object.keys(approvals).length;
  const visibleCalls = new Set(rootSteps.flatMap((s) => (s.kind === "model" ? (parseJson<{ tool_calls?: ToolCall[] }>(s.data, {}).tool_calls ?? []).map((c) => c.id) : [])));
  const elsewhere = Object.values(approvals).filter((a) => !visibleCalls.has(a.call_id));
  const askers = [...new Set(Object.values(approvals).map((a) => a.agent ?? "Main agent"))];

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">
        An agent task works toward one objective on its own: it reads, runs commands and edits files with the tools you allow, and can start subagents for independent parts. Anything that changes something still asks you first unless you allowed it.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn sm primary" onClick={() => { setCreating(true); setSelId(null); }}>New task</button>
      </div>
      {err && <div className="notice error" role="alert" style={{ marginBottom: 10 }}>{err}</div>}
      {wtNote && (
        <div className="notice warn row worktree-kept" role="status" aria-label="Kept worktrees" style={{ marginBottom: 10 }}>
          <span style={{ whiteSpace: "pre-wrap" }}>{wtNote}</span>
          <span className="spacer" />
          <button className="btn sm ghost" onClick={() => setWtNote(null)}>Dismiss</button>
        </div>
      )}

      <div className="instr-layout">
        <div className="instr-list" aria-label="Tasks">
          {tasks == null ? (
            <div style={{ padding: 10 }}><div className="skeleton" /></div>
          ) : tasks.length === 0 ? (
            <div className="empty" style={{ padding: 12 }}>No tasks yet. Start one with New task.</div>
          ) : (
            tasks.map((t) => (
              <button key={t.id} className="chat-item" aria-current={t.id === selId && !creating ? "true" : undefined} onClick={() => { setCreating(false); setSelId(t.id); }}>
                <span className="chat-title">{t.title || t.objective}</span>
                <span className={`mono ${t.state === "failed" ? "danger-text" : "faint"}`} style={{ fontSize: 10.5 }}>{STATE_LABEL[t.state] ?? t.state}</span>
              </button>
            ))
          )}
        </div>

        <div style={{ display: "grid", gap: 14, minWidth: 0, alignContent: "start" }}>
          {creating && (
            <div className="panel" style={{ display: "grid", gap: 12 }}>
              <strong style={{ fontWeight: 600 }}>New task</strong>
              <div className="field wide">
                <label htmlFor="t-obj">Objective</label>
                <textarea id="t-obj" aria-label="Objective" className="input" rows={4} placeholder="What should the agent do? Be specific about the folder and how to check the result."
                  value={objective} onChange={(e) => setObjective(e.target.value)} />
              </div>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="t-title">Title (optional)</label>
                  <input id="t-title" className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
                </div>
                <div className="field">
                  <label htmlFor="t-env">Run on</label>
                  <select id="t-env" className="input" value={settings.environment_id ?? "local"} onChange={(e) => setSettings({ ...settings, environment_id: e.target.value === "local" ? null : e.target.value })}>
                    {(envs.length ? envs : [{ id: "local", name: "This PC", kind: "local" } as EnvView]).map((en) => (
                      <option key={en.id} value={en.id}>{en.name}{en.kind !== "local" ? ` (${en.kind})` : ""}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="t-skills">Skills</label>
                  <select id="t-skills" className="input" value={settings.skills_mode} onChange={(e) => setSettings({ ...settings, skills_mode: e.target.value as TaskSettings["skills_mode"] })}>
                    <option value="auto">Pick matching skills</option>
                    <option value="off">No skills</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="t-mode">Tool calls</label>
                  <select id="t-mode" className="input" value={settings.tool_mode} onChange={(e) => setSettings({ ...settings, tool_mode: e.target.value as TaskSettings["tool_mode"] })}>
                    <option value="auto">Automatic</option>
                    <option value="native">Native</option>
                    <option value="xml">Text format (for models without tool calling)</option>
                  </select>
                </div>
                <label className="check" style={{ alignSelf: "end" }}>
                  <input type="checkbox" aria-label="Allow subagents" checked={settings.subagents} onChange={(e) => setSettings({ ...settings, subagents: e.target.checked })} /> Allow subagents
                </label>
                {settings.subagents && (
                  <div className="field">
                    <label htmlFor="t-par">Subagents at once (1 to 4)</label>
                    <select id="t-par" className="input" value={settings.max_parallel} onChange={(e) => setSettings({ ...settings, max_parallel: Number(e.target.value) })}>
                      {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                )}
                <label className="check" style={{ alignSelf: "end" }} title="The task first looks around with read-only tools and writes a plan; it runs only after you approve the plan">
                  <input type="checkbox" aria-label="Plan first" checked={!!settings.plan_first} onChange={(e) => setSettings({ ...settings, plan_first: e.target.checked })} /> Plan first (approve the plan before it runs)
                </label>
                <div className="field task-thinking" style={{ alignSelf: "end" }}>
                  <ThinkingControls support={support} thinking={settings.thinking} effort={settings.reasoning_effort}
                    onChange={(t, e) => setSettings({ ...settings, thinking: t, reasoning_effort: e })} />
                </div>
                <label className="check" style={{ alignSelf: "end" }}>
                  <input type="checkbox" checked={settings.mcp_tools} onChange={(e) => setSettings({ ...settings, mcp_tools: e.target.checked })} /> Also offer MCP tools
                </label>
              </div>
              <div className="form-grid">
                {([["max_turns", "Model turns"], ["max_tool_calls", "Tool calls"], ["max_wall_secs", "Time (seconds)"], ["max_tokens", "Tokens"]] as [keyof Budgets, string][]).map(([k, l]) => (
                  <div className="field" key={k}>
                    <label htmlFor={`t-b-${k}`}>{l}</label>
                    <input id={`t-b-${k}`} className="input mono" value={budgets[k]} onChange={(e) => setBudgets({ ...budgets, [k]: num(e.target.value) })} />
                  </div>
                ))}
              </div>
              <span className="help faint">One model runs on this PC, so subagents take turns on it; they mostly save time while commands run. Load the model with 2 parallel slots to let two generate at once (each then gets half the context).</span>
              <div className="row">
                <button className="btn sm primary" aria-label="Start task" disabled={busy || !objective.trim()} onClick={start}>{busy ? "Starting..." : "Start task"}</button>
                <button className="btn sm ghost" onClick={() => setCreating(false)}>Cancel</button>
              </div>
            </div>
          )}

          {!creating && !selId && tasks && tasks.length > 0 && <div className="empty">Pick a task on the left, or start a new one.</div>}

          {!creating && selId && !detail && <div className="panel"><div className="skeleton" /></div>}

          {!creating && run && (
            <>
              <div className="panel" style={{ display: "grid", gap: 10 }}>
                <div className="row">
                  <strong style={{ fontWeight: 600 }}>{run.title || "Task"}</strong>
                  <span className={`tag${running ? " accent" : ""}`} aria-label="Task state">{STATE_LABEL[state] ?? state}</span>
                  {pendingCount > 0 && <span className="tag accent" title={askers.join(", ")}>{pendingCount} waiting for approval ({askers.join(", ")})</span>}
                  <span className="spacer" />
                  {running && <button className="btn sm" aria-label="Pause" disabled={busy} onClick={() => act(() => api5.pauseTask(run.id))}>Pause</button>}
                  {!running && ["paused", "interrupted", "budget_exhausted", "failed"].includes(state) && (
                    <button className="btn sm primary" aria-label="Resume" disabled={busy} onClick={() => act(() => api5.resumeTask(run.id, state === "budget_exhausted" ? { ...runBudgets, max_turns: runBudgets.max_turns + 20, max_tool_calls: runBudgets.max_tool_calls + 30, max_tokens: runBudgets.max_tokens + 30000, max_wall_secs: runBudgets.max_wall_secs + 900 } : null))}>
                      {state === "budget_exhausted" ? "Resume with more budget" : "Resume"}
                    </button>
                  )}
                  {(running || ["paused", "interrupted", "queued"].includes(state)) && (
                    <button className="btn sm ghost danger" aria-label="Cancel task" disabled={busy} onClick={() => act(() => api5.cancelTask(run.id))}>Cancel task</button>
                  )}
                  {!running && <button className="btn sm ghost danger" aria-label="Delete task" disabled={busy} onClick={() => act(async () => { const r = await api5.deleteTask(run.id); setSelId(null); setDetail(null); setWtNote(keptNote(r));  })}>Delete</button>}
                </div>
                <div className="dim" style={{ whiteSpace: "pre-wrap" }}>{run.objective}</div>
                <div className="budget-grid">
                  <Meter label="Model turns" used={usage.turns} max={runBudgets.max_turns} />
                  <Meter label="Tool calls" used={usage.tool_calls} max={runBudgets.max_tool_calls} />
                  <Meter label="Time" used={usage.wall_ms} max={runBudgets.max_wall_secs * 1000} fmt={mins} />
                  <Meter label="Tokens" used={usage.tokens} max={runBudgets.max_tokens} />
                </div>
                {state === "interrupted" && <div className="notice warn">The app stopped while this task was running. Nothing was re-run; Resume continues from the last saved step.</div>}
              </div>

              {state === "awaiting_plan_approval" && !running && (
                <div className="panel plan-review" style={{ display: "grid", gap: 6 }}>
                  <strong style={{ fontWeight: 600 }}>Plan ready for review</strong>
                  <span className="help faint">Nothing has been changed yet. Edit the plan if needed, then approve it to run the task with its full tools (planning already counts toward the budget).</span>
                  <textarea className="input mono" aria-label="Task plan" rows={12}
                    value={planDraft?.id === run.id ? planDraft.text : run.final_answer ?? ""}
                    onChange={(e) => setPlanDraft({ id: run.id, text: e.target.value })} />
                  <div className="row">
                    <button className="btn sm primary" aria-label="Approve plan" disabled={busy || !(planDraft?.id === run.id ? planDraft.text : run.final_answer ?? "").trim()}
                      onClick={() => act(async () => { const text = planDraft?.id === run.id ? planDraft.text : null; await api5.approveTaskPlan(run.id, text); setPlanDraft(null); })}>Approve plan</button>
                    <button className="btn sm ghost danger" aria-label="Cancel task" disabled={busy} onClick={() => act(() => api5.cancelTask(run.id))}>Cancel task</button>
                  </div>
                </div>
              )}

              {(run.final_answer || run.error) && state !== "awaiting_plan_approval" && (
                <div className="panel" style={{ display: "grid", gap: 6 }}>
                  <strong style={{ fontWeight: 600 }}>{run.final_answer ? "Result" : "Stopped"}</strong>
                  {run.final_answer && <div aria-label="Final answer"><Markdown text={run.final_answer} /></div>}
                  {run.error && <div className="danger-text">{run.error}</div>}
                </div>
              )}

              {detail && detail.subagents.length > 0 && (
                <div className="row" role="tablist" aria-label="Which agent's steps">
                  <button role="tab" aria-selected={!viewAgent} className={`btn sm${!viewAgent ? " primary" : " ghost"}`} onClick={() => setViewAgent(null)}>Main agent</button>
                  {detail.subagents.map((s) => (
                    <button key={s.id} role="tab" aria-selected={viewAgent === s.id} className={`btn sm${viewAgent === s.id ? " primary" : " ghost"}`} onClick={() => setViewAgent(s.id)}>
                      {s.label || "subagent"} <span className="faint">({STATE_LABEL[s.state] ?? s.state})</span>
                    </button>
                  ))}
                </div>
              )}

              {elsewhere.length > 0 && (
                <div className="panel" style={{ display: "grid", gap: 10 }} aria-label="Approvals from other agents">
                  <strong style={{ fontWeight: 600 }}>Waiting for your approval</strong>
                  {elsewhere.map((a) => (
                    <ToolCard key={a.id} call={{ id: a.call_id, type: "function", function: { name: a.model_name, arguments: JSON.stringify(a.arguments ?? {}) } }}
                      result={undefined} approval={a} live={liveTools[a.call_id]} output={liveOut[a.call_id]} where={a.server_name} onAnswer={answer} sessionLabel="Allow for this task" />
                  ))}
                </div>
              )}

              <div className="panel" style={{ display: "grid", gap: 10 }}>
                <strong style={{ fontWeight: 600 }}>{viewAgent ? `Steps of subagent "${detail?.subagents.find((s) => s.id === viewAgent)?.label}"` : "Steps"}</strong>
                {viewAgent && (() => {
                  const s = detail?.subagents.find((x) => x.id === viewAgent);
                  if (!s) return null;
                  return (
                    <div className="dim">
                      {s.objective}
                      {s.result && <div style={{ marginTop: 6 }}><Markdown text={s.result} /></div>}
                      {s.error && <div className="danger-text">{s.error}</div>}
                    </div>
                  );
                })()}
                <Timeline steps={rootSteps} live={live} approvals={approvals} liveTools={liveTools} liveOut={liveOut} callServer={callServer} onAnswer={answer} subagents={detail?.subagents ?? []}
                  onStop={(id) => act(() => api5.stopSubagent(id))} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

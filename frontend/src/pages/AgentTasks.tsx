import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { errorMessage } from "../lib/api";
import { api3, type ApprovalAnswer, type ApprovalRequest, type ServerView, type ToolEventPayload } from "../lib/api3";
import { api4, type EnvView } from "../lib/api4";
import { api7 } from "../lib/api7";
import {
  api5, DEFAULT_BUDGETS, OPEN_TASK_KEY, STATE_LABEL, SUGGESTED_LIMITS, keptNote, onOpen, parseJson,
  type Budgets, type QuestionData, type RunSummary, type TaskDetail, type TaskEvent, type TaskSettings, type Usage,
} from "../lib/api5";
import { Markdown } from "../components/Markdown";
import { ToolCard } from "../components/ToolCard";
import { ThinkingControls, useModelSupport } from "../components/ThinkingControls";
import { QuestionCard, ReplyBox, Timeline } from "../components/TaskView";

// Every connected MCP server is on (disabled_servers lists the ones turned off).
const BLANK_SETTINGS: TaskSettings = {
  environment_id: null, builtin_tools: true, web_search: true, mcp_tools: true, tool_servers: [], disabled_servers: [], skills_mode: "auto", skills: [],
  subagents: true, max_parallel: 2, tool_mode: "auto", plan_first: false, thinking: null, reasoning_effort: null,
  sampling: { temperature: null, top_p: null, top_k: null, min_p: null, repeat_penalty: null, max_tokens: null, seed: null, stop: [] },
};
/** States a message can restart ("Resume with this message"). */
const RESTARTABLE = ["paused", "interrupted", "budget_exhausted", "failed", "completed", "cancelled"];
const num = (v: string) => Math.max(0, Number(v.replace(/[^\d]/g, "")) || 0);
/** The four limits in Advanced; time is edited in minutes, stored in seconds. */
const LIMITS: { key: keyof Budgets; label: string; unit: string; scale: number }[] = [
  { key: "max_turns", label: "Model turns", unit: "turns", scale: 1 },
  { key: "max_tool_calls", label: "Tool calls", unit: "calls", scale: 1 },
  { key: "max_wall_secs", label: "Time", unit: "minutes", scale: 60 },
  { key: "max_tokens", label: "Tokens", unit: "tokens", scale: 1 },
];

function Meter({ label, used: u, max: m, fmt }: { label: string; used: number; max: number | null; fmt?: (n: number) => string }) {
  // Never trust the numbers' shape: a bad row must not take the page down.
  const used = Number.isFinite(u) ? u : 0;
  const max = m != null && Number.isFinite(m) ? m : null;
  const pct = max != null && max > 0 ? Math.min(100, (used / max) * 100) : 0;
  const f = fmt ?? ((n: number) => n.toLocaleString());
  return (
    <div className="budget" aria-label={`${label} used`}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="faint">{label}</span>
        <span className="mono" style={{ fontSize: 12 }}>{max == null ? <>{f(used)} <span className="faint">/ no limit</span></> : `${f(used)} / ${f(max)}`}</span>
      </div>
      <div className={`meter${pct > 85 ? " hot" : ""}`}><span style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

/** Tools the task may use: the built-in tools and each MCP server, all on by default. */
function TaskTools({ settings, envName, servers, onChange }: {
  settings: TaskSettings; envName: string; servers: ServerView[] | null; onChange: (s: TaskSettings) => void;
}) {
  const off = settings.disabled_servers ?? [];
  const [searchTokens, setSearchTokens] = useState<number | null>(null);
  useEffect(() => { api7.getSearchSettings().then((v) => setSearchTokens(v.tool_tokens)).catch(() => setSearchTokens(null)); }, []);
  const usable = (servers ?? []).filter((sv) => sv.enabled);
  const count = (sv: ServerView) => sv.tools.filter((t) => t.enabled && t.policy !== "deny").length;
  return (
    <fieldset className="task-tools" aria-label="Task tools">
      <legend className="faint">Tools it can use</legend>
      <label className="task-tool">
        <input type="checkbox" aria-label={`Built-in tools on ${envName}`} checked={settings.builtin_tools}
          onChange={(e) => onChange({ ...settings, builtin_tools: e.target.checked })} />
        <span>{envName}</span><span className="faint">shell and files</span>
      </label>
      <label className="task-tool" title="Search the web with the provider chosen in Settings > Search">
        <input type="checkbox" aria-label="Web search" checked={settings.web_search !== false}
          onChange={(e) => onChange({ ...settings, web_search: e.target.checked })} />
        <span>Web search</span><span className="faint">{searchTokens == null ? "searches, then reads pages" : `about ${searchTokens} tok`}</span>
      </label>
      {servers == null ? (
        <span className="faint task-tool-note">Loading MCP servers</span>
      ) : usable.length === 0 ? (
        <span className="faint task-tool-note">No MCP servers yet. Add one on the MCP Connections page.</span>
      ) : (
        usable.map((sv) => {
          const connected = sv.status.state === "connected";
          return (
            <label key={sv.id} className="task-tool" title={connected ? undefined : "Not connected: connect it on the MCP Connections page to use its tools"}>
              <input type="checkbox" aria-label={`MCP server ${sv.name}`} checked={!off.includes(sv.id)}
                onChange={(e) => onChange({ ...settings, mcp_tools: true, disabled_servers: e.target.checked ? off.filter((x) => x !== sv.id) : [...off, sv.id] })} />
              <span>{sv.name}</span>
              <span className="faint">{connected ? `${count(sv)} tool${count(sv) === 1 ? "" : "s"}` : sv.status.state === "error" ? "error" : "not connected"}</span>
            </label>
          );
        })
      )}
    </fieldset>
  );
}

const mins = (ms: number) => `${Math.round(ms / 60000)} min`;
/** "Resume with more budget": more of each limit that is set (none stay none). */
const moreBudget = (b: Budgets): Budgets => {
  const add = (v: number | null, n: number) => (v == null ? null : v + n);
  return { max_turns: add(b.max_turns, 20), max_tool_calls: add(b.max_tool_calls, 30), max_tokens: add(b.max_tokens, 30000), max_wall_secs: add(b.max_wall_secs, 900) };
};

export function AgentTasks() {
  const [tasks, setTasks] = useState<RunSummary[] | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [creating, setCreating] = useState(false);
  const [envs, setEnvs] = useState<EnvView[]>([]);
  const [servers, setServers] = useState<ServerView[] | null>(null);
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
  const [objOpen, setObjOpen] = useState(false);
  const [planEditing, setPlanEditing] = useState(false);
  // Follow the newest entry unless the user scrolled up.
  const [following, setFollowing] = useState(true);
  const follow = useRef(true);
  const endRef = useRef<HTMLDivElement>(null);
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
    api3.listServers().then(setServers).catch(() => setServers([]));
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
    setObjOpen(false);
    setPlanEditing(false);
    follow.current = true;
    setFollowing(true);
    if (selId) load(selId);
  }, [selId, load]);

  // Auto-scroll: stay at the newest entry until the user scrolls up.
  const hasRun = !!detail;
  useEffect(() => {
    const sc = endRef.current?.closest(".content") as HTMLElement | null;
    if (!sc) return;
    const onScroll = () => {
      const atEnd = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 120;
      follow.current = atEnd;
      setFollowing(atEnd);
    };
    sc.addEventListener("scroll", onScroll, { passive: true });
    return () => sc.removeEventListener("scroll", onScroll);
  }, [selId, hasRun]);
  useEffect(() => {
    if (follow.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [detail?.steps, live, detail?.run.state]);

  // Live updates.
  useEffect(() => {
    const t = listen<TaskEvent>("task://event", (e) => {
      const ev = e.payload;
      if (ev.type === "state" || ev.type === "question_closed") refreshList();
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
  const pendingCount = Object.keys(approvals).length;
  const askers = [...new Set(Object.values(approvals).map((a) => a.agent ?? "Main agent"))];
  const questions = (detail?.steps ?? []).filter((s) => s.kind === "question" && s.state === "waiting");
  const textQuestion = questions.some((q) => parseJson<QuestionData>(q.data, { question_id: "", question: "", options: [], allow_text: true, call_id: "", agent_label: "", answer: null }).allow_text);
  const planText = run ? (planDraft?.id === run.id ? planDraft.text : run.final_answer ?? "") : "";
  const needsYou = questions.length > 0 || pendingCount > 0 || (state === "awaiting_plan_approval" && !running);
  const longObjective = !!run && (run.objective.length > 140 || run.objective.includes("\n"));

  const answerQuestion = async (questionId: string, text: string) => {
    if (!run) return;
    setErr(null);
    try {
      await api5.answerTaskQuestion(run.id, questionId, text);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      refreshList();
      load(run.id);
    }
  };
  /** The reply box: a message to a live task, the answer to a waiting question, or a restart of a stopped one. */
  const sendMessage = async (text: string): Promise<boolean> => {
    if (!run) return false;
    setErr(null);
    follow.current = true;
    try {
      const resume = !running && state !== "waiting_for_user";
      await api5.sendTaskMessage(run.id, text, resume, resume && state === "budget_exhausted" ? moreBudget(runBudgets) : null);
      return true;
    } catch (e) {
      setErr(errorMessage(e));
      return false;
    } finally {
      refreshList();
      load(run.id);
    }
  };
  const canMessage = running || state === "waiting_for_user";
  const canRestart = !running && RESTARTABLE.includes(state);
  const replyDisabled = !(canMessage || canRestart) || (state === "waiting_for_user" && !textQuestion && questions.length > 0);
  const replyHint = canRestart
    ? "The task is not running. Sending restarts it with your message."
    : state === "waiting_for_user" && !textQuestion && questions.length > 0
      ? "The question above takes one of its options."
      : canMessage
        ? state === "waiting_for_user" ? "Your message answers the question above." : "The agent reads it before its next step."
        : "Approve or cancel the plan first; then you can message the task.";

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">
        An agent task works toward one objective on its own: it reads, runs commands and edits files with the tools you allow, and can start subagents for independent parts. Anything that changes something still asks you first unless you allowed it.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn sm primary" onClick={() => { setCreating(true); setSelId(null); api3.listServers().then(setServers).catch(() => {}); }}>New task</button>
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
            <div className="panel task-new" style={{ display: "grid", gap: 12 }}>
              <strong style={{ fontWeight: 600 }}>New task</strong>
              <div className="field wide">
                <label htmlFor="t-obj">What should it do?</label>
                <textarea id="t-obj" aria-label="Objective" className="input task-objective" rows={7} placeholder="Describe the task. Name the folder and say how to check the result, for example: In C:\code\app, make cargo test pass."
                  value={objective} onChange={(e) => setObjective(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && objective.trim() && !busy) { e.preventDefault(); start(); } }} />
              </div>
              <TaskTools settings={settings} servers={servers} onChange={setSettings}
                envName={settings.environment_id ? envs.find((en) => en.id === settings.environment_id)?.name ?? "Chosen environment" : "This PC"} />
              <div className="row task-quick">
                <label className="check" title="The task first looks around with read-only tools and writes a plan; it runs only after you approve the plan">
                  <input type="checkbox" aria-label="Plan first" checked={!!settings.plan_first} onChange={(e) => setSettings({ ...settings, plan_first: e.target.checked })} /> Plan first (approve the plan before it runs)
                </label>
                <div className="task-thinking">
                  <ThinkingControls support={support} thinking={settings.thinking} effort={settings.reasoning_effort}
                    onChange={(t, e) => setSettings({ ...settings, thinking: t, reasoning_effort: e })} />
                </div>
              </div>
              <div className="row">
                <button className="btn sm primary" aria-label="Start task" disabled={busy || !objective.trim()} onClick={start}>{busy ? "Starting..." : "Start task"}</button>
                <button className="btn sm ghost" onClick={() => setCreating(false)}>Cancel</button>
                <span className="faint task-hint">Ctrl+Enter starts it. No limits unless you set them in Advanced; Cancel task stops it at any time.</span>
              </div>
              <details className="task-advanced">
                <summary>Advanced</summary>
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
              </div>
              <span className="help faint">One model runs on this PC, so subagents take turns on it; they mostly save time while commands run. Load the model with 2 parallel slots to let two generate at once (each then gets half the context).</span>
              <div className="task-limits" role="group" aria-label="Limits">
                <span className="faint task-limits-head">Limits (off = no limit)</span>
                {LIMITS.map(({ key, label, unit, scale }) => {
                  const v = budgets[key];
                  return (
                    <div className="task-limit" key={key}>
                      <label className="check">
                        <input type="checkbox" aria-label={`Limit ${label.toLowerCase()}`} checked={v != null}
                          onChange={(e) => setBudgets({ ...budgets, [key]: e.target.checked ? SUGGESTED_LIMITS[key] : null })} /> Limit {label.toLowerCase()}
                      </label>
                      <input id={`t-b-${key}`} className="input mono" aria-label={`${label} limit`} disabled={v == null} placeholder="No limit"
                        value={v == null ? "" : String(Math.round(v / scale))}
                        onChange={(e) => setBudgets({ ...budgets, [key]: Math.max(1, num(e.target.value)) * scale })} />
                      <span className="faint">{unit}</span>
                    </div>
                  );
                })}
                <span className="help faint">With no turn limit a run still stops after 500 model turns each time it is started or resumed, so a model that never answers can't loop forever.</span>
              </div>
              </details>
            </div>
          )}

          {!creating && !selId && tasks && tasks.length > 0 && <div className="empty">Pick a task on the left, or start a new one.</div>}

          {!creating && selId && !detail && <div className="panel"><div className="skeleton" /></div>}

          {!creating && run && (
            <>
              <div className="task-top">
                <div className="panel task-head">
                  <div className="row task-head-row">
                    <strong className="task-title">{run.title || "Task"}</strong>
                    <span className={`tag${running || state === "waiting_for_user" ? " accent" : ""}`} aria-label="Task state">{STATE_LABEL[state] ?? state}</span>
                    {pendingCount > 0 && <span className="tag accent" title={askers.join(", ")}>{pendingCount} waiting for approval ({askers.join(", ")})</span>}
                    <span className="spacer" />
                    {running && state !== "waiting_for_user" && <button className="btn sm" aria-label="Pause" disabled={busy} onClick={() => act(() => api5.pauseTask(run.id))}>Pause</button>}
                    {!running && ["paused", "interrupted", "budget_exhausted", "failed"].includes(state) && (
                      <button className="btn sm primary" aria-label="Resume" disabled={busy} onClick={() => act(() => api5.resumeTask(run.id, state === "budget_exhausted" ? moreBudget(runBudgets) : null))}>
                        {state === "budget_exhausted" ? "Resume with more budget" : "Resume"}
                      </button>
                    )}
                    {(running || ["paused", "interrupted", "queued", "waiting_for_user"].includes(state)) && (
                      <button className="btn sm ghost danger" aria-label="Cancel task" disabled={busy} onClick={() => act(() => api5.cancelTask(run.id))}>Cancel task</button>
                    )}
                    {!running && <button className="btn sm ghost danger" aria-label="Delete task" disabled={busy} onClick={() => act(async () => { const r = await api5.deleteTask(run.id); setSelId(null); setDetail(null); setWtNote(keptNote(r));  })}>Delete</button>}
                  </div>
                  <div className="task-obj">
                    <div className={`dim task-obj-text${objOpen ? " open" : ""}`} aria-label="Objective text">{run.objective}</div>
                    {longObjective && <button className="btn sm ghost" aria-expanded={objOpen} onClick={() => setObjOpen((o) => !o)}>{objOpen ? "Show less" : "Show all"}</button>}
                  </div>
                  <details className="task-usage">
                    <summary className="faint">
                      {usage.turns} turn{usage.turns === 1 ? "" : "s"}, {usage.tool_calls} tool call{usage.tool_calls === 1 ? "" : "s"}, {mins(usage.wall_ms)}, {usage.tokens.toLocaleString()} tokens
                    </summary>
                    <div className="budget-grid">
                      <Meter label="Model turns" used={usage.turns} max={runBudgets.max_turns} />
                      <Meter label="Tool calls" used={usage.tool_calls} max={runBudgets.max_tool_calls} />
                      <Meter label="Time" used={usage.wall_ms} max={runBudgets.max_wall_secs == null ? null : runBudgets.max_wall_secs * 1000} fmt={mins} />
                      <Meter label="Tokens" used={usage.tokens} max={runBudgets.max_tokens} />
                    </div>
                  </details>
                  {state === "interrupted" && <div className="notice warn">The app stopped while this task was running. Nothing was re-run; Resume continues from the last saved step.</div>}
                </div>

                {needsYou && (
                  <div className="panel task-needs" aria-label="Needs your input">
                    {questions.map((q) => <QuestionCard key={q.id} step={q} onAnswer={answerQuestion} />)}
                    {pendingCount > 0 && (
                      <div className="task-approvals" aria-label="Tool approvals">
                        <strong className="task-needs-title">Waiting for your approval</strong>
                        {Object.values(approvals).map((a) => (
                          <ToolCard key={a.id} call={{ id: a.call_id, type: "function", function: { name: a.model_name, arguments: JSON.stringify(a.arguments ?? {}) } }}
                            result={undefined} approval={a} live={liveTools[a.call_id]} output={liveOut[a.call_id]} where={a.server_name} onAnswer={answer} sessionLabel="Allow for this task" />
                        ))}
                      </div>
                    )}
                    {state === "awaiting_plan_approval" && !running && (
                      <div className="plan-review task-plan">
                        <strong className="task-needs-title">Plan ready for review</strong>
                        <span className="help faint">Nothing has been changed yet. Approve it to run the task with its full tools, or edit it first (planning already counts toward the budget).</span>
                        {planEditing ? (
                          <textarea className="input mono" aria-label="Task plan" rows={12} value={planText} onChange={(e) => setPlanDraft({ id: run.id, text: e.target.value })} />
                        ) : (
                          <div className="task-plan-text" aria-label="Plan"><Markdown text={planText} /></div>
                        )}
                        <div className="row">
                          <button className="btn sm primary" aria-label="Approve plan" disabled={busy || !planText.trim()}
                            onClick={() => act(async () => { const text = planDraft?.id === run.id ? planDraft.text : null; await api5.approveTaskPlan(run.id, text); setPlanDraft(null); setPlanEditing(false); })}>Approve plan</button>
                          <button className="btn sm" aria-label={planEditing ? "Stop editing plan" : "Edit plan"} onClick={() => setPlanEditing((e) => !e)}>{planEditing ? "Done editing" : "Edit plan"}</button>
                          <button className="btn sm ghost danger" aria-label="Cancel task" disabled={busy} onClick={() => act(() => api5.cancelTask(run.id))}>Cancel task</button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

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

              <div className="task-timeline">
                {viewAgent && (() => {
                  const s = detail?.subagents.find((x) => x.id === viewAgent);
                  if (!s) return null;
                  return (
                    <div className="dim">
                      <strong>Subagent "{s.label}"</strong>: {s.objective}
                      {s.result && <div style={{ marginTop: 6 }}><Markdown text={s.result} /></div>}
                      {s.error && <div className="danger-text">{s.error}</div>}
                    </div>
                  );
                })()}
                <Timeline steps={detail?.steps ?? []} agentId={viewAgent} hidePlan={state === "awaiting_plan_approval" && !running ? run.final_answer ?? undefined : undefined} live={live} liveTools={liveTools} liveOut={liveOut} callServer={callServer} onAnswer={answer}
                  subagents={detail?.subagents ?? []} onStop={(id) => act(() => api5.stopSubagent(id))} />
                {(run.final_answer || run.error) && state !== "awaiting_plan_approval" && !viewAgent && (
                  <div className="task-result" style={{ display: "grid", gap: 6 }}>
                    <strong style={{ fontWeight: 600 }}>{run.final_answer ? "Result" : "Stopped"}</strong>
                    {run.final_answer && <div aria-label="Final answer"><Markdown text={run.final_answer} /></div>}
                    {run.error && <div className="danger-text">{run.error}</div>}
                  </div>
                )}
              </div>

              <ReplyBox
                placeholder={canRestart ? "Message the task to continue it" : state === "waiting_for_user" && textQuestion ? "Type your answer, or a message for the task" : "Message the task"}
                buttonLabel={canRestart ? "Resume with this message" : state === "waiting_for_user" ? "Send reply" : "Send message"}
                disabled={replyDisabled || busy} hint={replyHint} onSend={sendMessage}
                showJump={!following} onJump={() => { follow.current = true; setFollowing(true); endRef.current?.scrollIntoView({ block: "end" }); }} />
              <div ref={endRef} aria-hidden="true" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

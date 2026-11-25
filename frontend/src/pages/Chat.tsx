import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog, save } from "@tauri-apps/plugin-dialog";
import { errorMessage } from "../lib/api";
import {
  api2, type ChatEvent, type ChatSettings, type ContextReport, type ConversationSummary, type ConversationView,
  type InstructionProfile, type Message,
} from "../lib/api2";
import { useEngine } from "../lib/engine";
import { api3, type ApprovalAnswer, type ApprovalRequest } from "../lib/api3";
import { api4, type EnvView } from "../lib/api4";
import { OPEN_CHAT_KEY, keptNote, onOpen, type DeleteResult } from "../lib/api5";
import { ThinkingControls, useModelSupport } from "../components/ThinkingControls";
import { api6, type Attachment } from "../lib/api6";
import { AssistantRow, AttachmentChips, UserRow, type RowActions } from "../components/chat/MessageRow";
import { streamStore } from "../lib/streamStore";
// --- chat mode / tools menus (composer bar) ---
import { ModeMenu, type ChatMode } from "../components/chat/ModeMenu";
import { Trash } from "@phosphor-icons/react";
import { ConfirmDelete } from "../components/chat/ConfirmDelete";
import { ToolsMenu } from "../components/chat/ToolsMenu";
import { SubagentControls } from "../components/chat/SubagentControls";
// --- end chat mode / tools menus ---
import type { PageId } from "../App";

function ContextMeter({ r, err }: { r: ContextReport | null; err: string | null }) {
  if (err) return <div className="faint" style={{ fontSize: 12 }}>{err}</div>;
  if (!r) return null;
  const used = Math.min(100, ((r.total + r.reserved) / r.n_ctx) * 100);
  const seg = (n: number) => `${(n / r.n_ctx) * 100}%`;
  return (
    <div title="Counted with the loaded model's tokenizer and chat template">
      <div className="ctx-bar" aria-label={`Context ${r.total} of ${r.n_ctx} tokens used, ${r.reserved} reserved for the reply`}>
        <span className="ctx-sys" style={{ width: seg(r.system) }} />
        <span className="ctx-hist" style={{ width: seg(r.history + r.template) }} />
        <span className="ctx-draft" style={{ width: seg(r.draft + (r.attachments ?? 0)) }} />
        <span className="ctx-res" style={{ width: seg(r.reserved) }} />
      </div>
      <div className="row mono faint" style={{ fontSize: 11, marginTop: 3, gap: 14 }}>
        <span>{r.total.toLocaleString()} / {r.n_ctx.toLocaleString()} tokens ({used.toFixed(0)}% with reply reserve)</span>
        <span>system {r.system}</span>
        <span>history {r.history}</span>
        <span>template {r.template}</span>
        <span>draft {r.draft}</span>
        {(r.attachments ?? 0) > 0 && <span>attachments {r.attachments}</span>}
        <span>reserved {r.reserved}</span>
        <span>tools {r.tools}{r.agent_mode ? ` (${r.tool_count} tools)` : ""}</span>
        {!r.fits && <span style={{ color: "var(--danger)" }}>does not fit</span>}
        {r.tools_warning && <span style={{ color: "var(--accent)" }}>tool definitions use over a quarter of the context</span>}
        {r.agent_mode && r.tool_count > 0 && !r.supports_tools && <span style={{ color: "var(--danger)" }}>this model's chat template has no tool calling</span>}
      </div>
    </div>
  );
}

const NO_MAP: Record<string, never> = {};

const EMPTY_SETTINGS: ChatSettings = {
  profile_id: null,
  sampling: { temperature: null, top_p: null, top_k: null, min_p: null, repeat_penalty: null, max_tokens: null, seed: null, stop: [] },
  overflow: "warn",
  reserve_tokens: 1024,
  agent_mode: false,
  tool_servers: [],
  disabled_servers: [],
  limits: { max_tool_calls: null, max_wall_secs: null },
  builtin_tools: true,
  web_search: true,
  environment_id: null,
  skills: true,
  tool_mode: "auto",
  mode: "",
  subagents: false,
  max_subagents: 2,
  thinking: null,
  reasoning_effort: null,
};

const modeOf = (s: ChatSettings): ChatMode => (!s.agent_mode ? "chat" : s.mode === "plan" ? "plan" : "agent");
const withMode = (s: ChatSettings, m: ChatMode): ChatSettings => ({ ...s, agent_mode: m !== "chat", mode: m });

export function Chat({ go }: { go: (p: PageId) => void }) {
  const { status } = useEngine();
  const loaded = status?.state === "loaded";
  const [list, setList] = useState<ConversationSummary[] | null>(null);
  const [query, setQuery] = useState("");
  const [current, setCurrent] = useState<ConversationView | null>(null);
  const [draft, setDraft] = useState("");
  const [generating, setGenerating] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ctx, setCtx] = useState<ContextReport | null>(null);
  const [ctxErr, setCtxErr] = useState<string | null>(null);
  // Bumped after chat settings are saved, so the context meter and the
  // Tools menu recount with what the backend now has.
  const [ctxTick, setCtxTick] = useState(0);
  const [profiles, setProfiles] = useState<InstructionProfile[]>([]);
  const [settings, setSettings] = useState<ChatSettings>(EMPTY_SETTINGS);
  const [planEdit, setPlanEdit] = useState<{ id: string; text: string } | null>(null);
  // Git worktrees a deleted chat's subagents had to keep.
  const [wtNote, setWtNote] = useState<string | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmDel, setConfirmDel] = useState<{ ids: string[]; title?: string } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [approvals, setApprovals] = useState<Record<string, ApprovalRequest>>({});
  const [liveTools, setLiveTools] = useState<Record<string, string>>({});
  const [envs, setEnvs] = useState<EnvView[]>([]);
  // Live shell output per tool call, and where each call runs.
  const [liveOut, setLiveOut] = useState<Record<string, string>>({});
  const [callServer, setCallServer] = useState<Record<string, string>>({});
  // Files attached on the composer (sent with the next message).
  const [pending, setPending] = useState<Attachment[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [attaching, setAttaching] = useState(false);
  // An image was refused because the vision projector isn't loaded.
  const [projNeeded, setProjNeeded] = useState(false);
  const [projReloading, setProjReloading] = useState(false);
  const [support, refreshSupport] = useModelSupport();
  const scroller = useRef<HTMLDivElement>(null);
  // Follow new output only while the user is at (or near) the bottom.
  const stick = useRef(true);
  const currentId = current?.id ?? null;
  const currentIdRef = useRef<string | null>(null);
  currentIdRef.current = currentId;

  const refreshList = useCallback(async (q = query) => setList(await api2.listConversations(q)), [query]);

  const open = useCallback(async (id: string) => {
    setErr(null);
    const c = await api2.getConversation(id);
    // The ref follows at once (not on the next render), so code awaiting
    // open() sees the chat that is now current.
    currentIdRef.current = c.id;
    setCurrent(c);
    setSettings(c.settings_parsed);
    setPending(c.pending_attachments ?? []);
    setProjNeeded(false);
    setGenerating(c.generating);
    streamStore.clear();
    setEditing(null);
    const pend = await api3.pendingApprovals();
    setApprovals(Object.fromEntries(pend.filter((p) => p.conversation_id === id).map((p) => [p.call_id, p])));
    setLiveTools({});
    setLiveOut({});
  }, []);

  const leaveSelect = useCallback(() => { setSelecting(false); setSelected(new Set()); }, []);
  useEffect(() => {
    if (!selecting) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && !confirmDel) leaveSelect(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selecting, confirmDel, leaveSelect]);

  // Delete chats (one or many), then refresh and move off a deleted open chat.
  const doDelete = async (ids: string[]) => {
    setDeleting(true);
    try {
      let r: DeleteResult;
      let failed: { id: string; error: string }[] = [];
      if (ids.length === 1) r = await api2.deleteConversation(ids[0]);
      else { const b = await api2.deleteConversations(ids); r = b; failed = b.failed; }
      const gone = ids.filter((i) => !failed.some((f) => f.id === i));
      const notes = [keptNote(r), failed.length ? `${failed.length} chat${failed.length === 1 ? "" : "s"} could not be deleted: ${failed.map((f) => f.error).join("; ")}` : null].filter(Boolean);
      setWtNote(notes.length ? notes.join("\n") : null);
      setSelected((x) => new Set([...x].filter((i) => !gone.includes(i))));
      const l = await api2.listConversations(query);
      setList(l);
      if (currentIdRef.current && gone.includes(currentIdRef.current)) {
        currentIdRef.current = null;
        setCurrent(null);
        if (l[0]) await open(l[0].id);
      }
      setConfirmDel(null);
      if (selecting && failed.length === 0) leaveSelect();
    } catch (e) {
      setErr(errorMessage(e));
      setConfirmDel(null);
    } finally {
      setDeleting(false);
    }
  };

  useEffect(() => {
    api2.listConversations().then((l) => {
      setList(l);
      // Opened from the approvals indicator: show that chat, else the newest.
      const want = localStorage.getItem(OPEN_CHAT_KEY);
      localStorage.removeItem(OPEN_CHAT_KEY);
      if (want && l.some((c) => c.id === want)) open(want);
      else if (l[0]) open(l[0].id);
    });
    api2.listProfiles().then(setProfiles);
    api4.list().then(setEnvs).catch(() => {});
    // The approvals indicator can ask an already-open Chat page to switch.
    return onOpen("chat", (id) => {
      localStorage.removeItem(OPEN_CHAT_KEY);
      open(id);
    });
  }, [open]);

  // Tool approvals and live tool status (Agent Mode).
  useEffect(() => {
    const a = listen<ApprovalRequest>("tool://approval", (e) => {
      if (e.payload.conversation_id === currentIdRef.current) setApprovals((x) => ({ ...x, [e.payload.call_id]: e.payload }));
    });
    const c = listen<{ id: string }>("tool://approval-closed", (e) => {
      setApprovals((x) => Object.fromEntries(Object.entries(x).filter(([, v]) => v.id !== e.payload.id)));
    });
    const t = listen<{ conversation_id: string | null; event: { phase: string; call_id: string; server_name?: string; stream?: string; text?: string } }>("tool://event", (e) => {
      if (e.payload.conversation_id !== currentIdRef.current) return;
      const ev = e.payload.event;
      if (ev.phase === "output") {
        // Keep the last 20k characters of live output per call.
        setLiveOut((x) => ({ ...x, [ev.call_id]: ((x[ev.call_id] ?? "") + (ev.text ?? "")).slice(-20000) }));
        return;
      }
      if (ev.phase === "started" && ev.server_name) {
        const sn = ev.server_name;
        setCallServer((x) => ({ ...x, [ev.call_id]: sn }));
      }
      setLiveTools((x) => ({ ...x, [ev.call_id]: ev.phase === "started" ? "Running" : ev.phase === "awaiting_approval" ? "Waiting for your approval" : "Finished" }));
    });
    return () => {
      a.then((f) => f());
      c.then((f) => f());
      t.then((f) => f());
    };
  }, []);

  const answerApproval = async (id: string, answer: ApprovalAnswer) => {
    try {
      await api3.answerApproval(id, answer);
      setApprovals((x) => Object.fromEntries(Object.entries(x).filter(([, v]) => v.id !== id)));
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  // Messages the open chat has, for spotting events about ones it lacks
  // (a reply started elsewhere): those trigger one reload.
  const knownIds = useRef<Set<string>>(new Set());
  knownIds.current = new Set(current?.messages.map((m) => m.id) ?? []);
  const reloading = useRef(false);
  const reloadMessages = useCallback(async (id: string) => {
    if (reloading.current) return;
    reloading.current = true;
    try {
      const c = await api2.getConversation(id);
      if (currentIdRef.current === id) {
        setCurrent(c);
        setGenerating(c.generating);
      }
    } finally {
      reloading.current = false;
    }
  }, []);

  // Live chat events.
  useEffect(() => {
    const un = listen<ChatEvent>("chat://event", (e) => {
      const ev = e.payload;
      if (ev.conversation_id !== currentIdRef.current) {
        if (ev.type !== "delta") api2.listConversations().then(setList);
        return;
      }
      const mid = "message_id" in ev ? ev.message_id : "message" in ev ? ev.message.id : null;
      if (mid && !knownIds.current.has(mid) && ev.type !== "segment_start" && ev.type !== "tool_result") {
        reloadMessages(ev.conversation_id);
      }
      if (ev.type === "segment_reset") {
        streamStore.remove(ev.message_id);
      } else if (ev.type === "segment_start" || ev.type === "tool_result") {
        setCurrent((c) => c && (c.messages.some((m) => m.id === ev.message.id) ? c : { ...c, messages: [...c.messages, ev.message] }));
      } else if (ev.type === "segment_done") {
        setCurrent((c) => c && { ...c, messages: c.messages.map((m) => (m.id === ev.message.id ? ev.message : m)) });
        streamStore.remove(ev.message.id);
      } else if (ev.type === "delta") {
        // Held outside React state: only the live row re-renders, once per frame.
        streamStore.append(ev.message_id, ev.content, ev.reasoning);
      } else if (ev.type === "done") {
        setCurrent((c) => c && { ...c, messages: c.messages.map((m) => (m.id === ev.message_id ? ev.message : m)) });
        streamStore.remove(ev.message_id);
        setGenerating(false);
        api2.listConversations().then(setList);
      } else {
        setErr(ev.error);
        setGenerating(false);
        if (currentIdRef.current) open(currentIdRef.current);
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, [open, reloadMessages]);

  // Context meter (debounced, real tokenizer). It recounts after typing
  // pauses and when a reply finishes (`generating` flips to false), never
  // while a reply or agent run is in progress (every tool segment adds a message).
  useEffect(() => {
    if (!currentId || !loaded) {
      setCtx(null);
      setCtxErr(loaded ? null : "Load a model to see context usage.");
      return;
    }
    if (generating) return;
    const t = setTimeout(() => {
      api2.chatContext(currentId, draft).then((r) => { setCtx(r); setCtxErr(null); }).catch((e) => setCtxErr(errorMessage(e)));
    }, 350);
    return () => clearTimeout(t);
  }, [currentId, draft, loaded, current?.messages.length, generating, settings, status?.props?.n_ctx, pending.length, ctxTick]);

  const toBottom = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  };
  // Opening a chat starts at the latest message.
  useEffect(() => {
    stick.current = true;
    toBottom();
  }, [currentId]);
  // New messages: follow only if the user has not scrolled up. Streamed text
  // is followed by the live row itself (rows.grew), once per rendered frame.
  useEffect(() => {
    if (stick.current) toBottom();
  }, [current?.messages.length]);
  useEffect(() => () => streamStore.clear(), []);

  const newChat = async () => {
    const c = await api2.createConversation();
    await refreshList();
    await open(c.id);
  };

  const ensureChat = async (): Promise<string> => {
    if (current) return current.id;
    const c = await api2.createConversation();
    await open(c.id);
    return c.id;
  };

  const started = async (fn: () => Promise<{ user_message: Message | null; assistant_message_id: string }>) => {
    setErr(null);
    setGenerating(true);
    try {
      await fn();
      if (currentIdRef.current) {
        const c = await api2.getConversation(currentIdRef.current);
        setCurrent(c);
        setPending(c.pending_attachments ?? []);
      }
      refreshList();
    } catch (e) {
      setErr(errorMessage(e));
      setGenerating(false);
    }
  };

  // Set synchronously on the first click, so a fast double click (before
  // React re-renders with generating = true) can't send twice.
  const sendingRef = useRef(false);
  const send = async () => {
    const text = draft.trim();
    if ((!text && pending.length === 0) || generating || sendingRef.current) return;
    sendingRef.current = true;
    setGenerating(true);
    try {
      const atts = pending.map((a) => a.id);
      const id = await ensureChat();
      setDraft("");
      stick.current = true;
      await started(() => api2.chatSend(id, text, atts));
    } catch (e) {
      setErr(errorMessage(e));
      setGenerating(false);
    } finally {
      sendingRef.current = false;
    }
  };

  // Attachments: by path (file dialog or drag and drop onto the window).
  const attachPaths = async (paths: string[]) => {
    if (paths.length === 0) return;
    setAttaching(true);
    // The chat the files go to, fixed when the attach starts: if the user
    // switches chats meanwhile, the rest still go to this one, and the
    // composer of the chat now open is left alone.
    try {
      const id = await ensureChat();
      for (const p of paths) {
        try {
          const a = await api6.attach(id, p);
          if (currentIdRef.current === id) {
            setPending((x) => (x.some((y) => y.id === a.id) ? x : [...x, a]));
            setErr(null);
          }
        } catch (e) {
          if (currentIdRef.current !== id) continue;
          const msg = errorMessage(e);
          setErr(msg);
          if (msg.includes("Reload with the vision projector")) setProjNeeded(true);
        }
      }
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setAttaching(false);
      // The backend's list is the truth for the chat that is open now.
      const now = currentIdRef.current;
      if (now) {
        api6.pending(now).then((l) => { if (currentIdRef.current === now) setPending(l); }).catch(() => {});
      }
    }
  };
  const attachRef = useRef(attachPaths);
  attachRef.current = attachPaths;
  const pickFiles = async () => {
    const r = await openDialog({ multiple: true, directory: false, title: "Attach files" });
    if (r) attachRef.current(Array.isArray(r) ? r : [r]);
  };
  const removeAttachment = async (a: Attachment) => {
    if (!current) return;
    try {
      await api6.detach(current.id, a.id);
    } finally {
      setPending((x) => x.filter((y) => y.id !== a.id));
    }
  };
  const reloadWithProjector = async () => {
    setProjReloading(true);
    try {
      await api6.reloadWithProjector();
      setProjNeeded(false);
      setErr(null);
      refreshSupport();
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setProjReloading(false);
    }
  };
  // Files dragged onto the window (Tauri gives their paths).
  useEffect(() => {
    let un: (() => void) | null = null;
    let dead = false;
    getCurrentWebview()
      .onDragDropEvent((e) => {
        const p = e.payload;
        if (p.type === "enter" || p.type === "over") setDragOver(true);
        else if (p.type === "leave") setDragOver(false);
        else if (p.type === "drop") {
          setDragOver(false);
          attachRef.current(p.paths);
        }
      })
      .then((f) => {
        if (dead) f();
        else un = f;
      })
      .catch(() => {});
    return () => {
      dead = true;
      un?.();
    };
  }, []);

  // Plan mode: run the plan the model made (switches the chat to Agent mode).
  const runPlan = async (plan: string) => {
    const text = plan.trim();
    if (!current || !text || generating) return;
    setPlanEdit(null);
    await saveSettings(withMode(settings, "agent"));
    stick.current = true;
    await started(() => api2.chatSend(current.id, `Carry out this approved plan:
${text}`));
  };

  const stop = async () => {
    if (current) await api2.chatStop(current.id);
  };

  const saveSettings = async (next: ChatSettings) => {
    setSettings(next);
    if (!current) return;
    try {
      await api2.updateChatSettings(current.id, next);
      setErr(null);
      setCtxTick((t) => t + 1);
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  // Composer mode menu. With no chat open yet, it makes one first.
  const setMode = async (m: ChatMode) => {
    if (current) return saveSettings(withMode(settings, m));
    try {
      const id = await ensureChat();
      const c = await api2.getConversation(id);
      await api2.updateChatSettings(id, withMode(c.settings_parsed, m));
      await open(id);
      refreshList();
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const num = (v: string): number | null => (v.trim() === "" || !Number.isFinite(Number(v)) ? null : Number(v));

  const exportChat = async () => {
    if (!current) return;
    const path = await save({ defaultPath: `${current.title.replace(/[\\/:*?"<>|]/g, "_")}.md`, filters: [{ name: "Markdown", extensions: ["md"] }, { name: "JSON", extensions: ["json"] }] });
    if (path) {
      try {
        await api2.exportConversation(current.id, path);
      } catch (e) {
        setErr(errorMessage(e));
      }
    }
  };

  const messages = current?.messages ?? [];
  // Tool results by call id, built once per change instead of a scan per card per render.
  const toolResults = useMemo(() => {
    const m = new Map<string, Message>();
    for (const x of current?.messages ?? []) if (x.role === "tool" && x.tool_call_id) m.set(x.tool_call_id, x);
    return m;
  }, [current?.messages]);
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  // Stable handlers for the memoized rows; they always call the latest closures.
  const actNow: RowActions = {
    answerApproval: (id, a) => { void answerApproval(id, a); },
    openSubagents: () => go("subagents"),
    startEdit: (id, text) => setEditing({ id, text }),
    editText: (id, text) => setEditing({ id, text }),
    cancelEdit: () => setEditing(null),
    saveEdit: (id, t) => { setEditing(null); void started(() => api2.chatEdit(currentIdRef.current!, id, t)); },
    startPlanEdit: (id, text) => setPlanEdit({ id, text }),
    planText: (id, text) => setPlanEdit({ id, text }),
    cancelPlanEdit: () => setPlanEdit(null),
    runPlan: (t) => { void runPlan(t); },
    regenerate: () => { void started(() => api2.chatRegenerate(currentIdRef.current!)); },
    grew: () => { if (stick.current) toBottom(); },
  };
  const actRef = useRef(actNow);
  actRef.current = actNow;
  const act = useMemo<RowActions>(() => {
    const o = {} as Record<string, (...a: unknown[]) => unknown>;
    for (const k of Object.keys(actNow)) o[k] = (...a: unknown[]) => (actRef.current as unknown as Record<string, (...x: unknown[]) => unknown>)[k](...a);
    return o as unknown as RowActions;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const profileLabel = useMemo(() => {
    if (!ctx?.profile) return null;
    const src = { conversation: "this chat", model: "model default", default: "app default", agent_default: "agent default", plan_default: "plan default" }[ctx.profile.source];
    return `${ctx.profile.title} (${src})`;
  }, [ctx]);

  return (
    <div className="chat-layout">
      {confirmDel && <ConfirmDelete count={confirmDel.ids.length} title={confirmDel.title} busy={deleting} onConfirm={() => doDelete(confirmDel.ids)} onCancel={() => { if (!deleting) setConfirmDel(null); }} />}
      <aside className="chat-list">
        <div className="row" style={{ padding: "10px 10px 6px" }}>
          <button className="btn primary sm" onClick={newChat}>New chat</button>
          <span className="spacer" />
          {selecting ? (
            <button className="btn sm" onClick={leaveSelect}>Done</button>
          ) : (
            <button className="btn sm ghost" aria-label="Select chats" onClick={() => setSelecting(true)}>Select</button>
          )}
        </div>
        <div style={{ padding: "0 10px 8px" }}>
          <input className="input" style={{ width: "100%" }} placeholder="Search chats" aria-label="Search chats" value={query}
            onChange={(e) => { setQuery(e.target.value); api2.listConversations(e.target.value).then(setList); }} />
        </div>
        {selecting && (
          <div className="chat-select-bar row">
            <label className="row" style={{ gap: 6 }}>
              <input type="checkbox" aria-label="Select all" disabled={!list || list.length === 0}
                checked={!!list && list.length > 0 && list.every((c) => selected.has(c.id))}
                onChange={(e) => setSelected(e.target.checked ? new Set(list?.map((c) => c.id) ?? []) : new Set())} />
              <span>Select all</span>
            </label>
            <span className="spacer" />
            <span className="faint mono" aria-live="polite" style={{ fontSize: 11 }}>{selected.size} selected</span>
            <button className="btn sm danger" aria-label="Delete selected chats" disabled={selected.size === 0}
              onClick={() => setConfirmDel({ ids: [...selected] })}>Delete {selected.size} {selected.size === 1 ? "chat" : "chats"}</button>
          </div>
        )}
        {list == null ? (
          <div style={{ padding: 10, display: "grid", gap: 8 }}><div className="skeleton" /><div className="skeleton" /></div>
        ) : list.length === 0 ? (
          <div className="empty" style={{ padding: 12 }}>{query ? "No chats match." : "No chats yet."}</div>
        ) : (
          list.map((c) => selecting ? (
            <label key={c.id} className="chat-item chat-row-select" aria-current={c.id === currentId ? "true" : undefined}>
              <input type="checkbox" aria-label={`Select ${c.title}`} checked={selected.has(c.id)}
                onChange={() => setSelected((x) => { const n = new Set(x); if (n.has(c.id)) n.delete(c.id); else n.add(c.id); return n; })} />
              <span className="chat-row-text">
                <span className="chat-title">{c.title}</span>
                <span className="faint mono" style={{ fontSize: 10.5 }}>{c.message_count} msgs</span>
              </span>
            </label>
          ) : (
            <div key={c.id} className="chat-row">
              <button className="chat-item" aria-current={c.id === currentId ? "true" : undefined} onClick={() => open(c.id)}>
                <span className="chat-title">{c.title}</span>
                <span className="faint mono" style={{ fontSize: 10.5 }}>{c.message_count} msgs</span>
              </button>
              <button className="chat-row-del" aria-label={`Delete chat ${c.title}`} title="Delete chat" onClick={() => setConfirmDel({ ids: [c.id], title: c.title })}>
                <Trash size={14} />
              </button>
            </div>
          ))
        )}
      </aside>

      <section className="chat-main">
        <div className="chat-head row">
          {current && renaming !== null ? (
            <form className="row" onSubmit={async (e) => { e.preventDefault(); try { await api2.renameConversation(current.id, renaming); setRenaming(null); open(current.id); refreshList(); } catch (er) { setErr(errorMessage(er)); } }}>
              <input className="input" autoFocus aria-label="Chat title" value={renaming} onChange={(e) => setRenaming(e.target.value)} style={{ width: 320 }} />
              <button className="btn sm" type="submit">Save</button>
              <button className="btn sm ghost" type="button" onClick={() => setRenaming(null)}>Cancel</button>
            </form>
          ) : (
            <strong style={{ fontWeight: 600 }}>{current?.title ?? "New chat"}</strong>
          )}
          {modeOf(settings) === "agent" && <span className="tag accent" title="Tools are offered to the model in this chat">Agent Mode{ctx ? `, ${ctx.tool_count} tools` : ""}</span>}
          {modeOf(settings) === "plan" && <span className="tag accent" title="Only read-only tools are offered; the model ends with a plan you can run">Plan mode{ctx ? `, ${ctx.tool_count} read-only tools` : ""}</span>}
          {profileLabel && <span className="tag" title="System instruction profile in use">{profileLabel}</span>}
          <span className="spacer" />
          {current && renaming === null && (
            <>
              <button className="btn sm ghost" onClick={() => setRenaming(current.title)}>Rename</button>
              <button className="btn sm ghost" onClick={async () => { const c = await api2.duplicateConversation(current.id); await refreshList(); open(c.id); }}>Duplicate</button>
              <button className="btn sm ghost" onClick={exportChat}>Export</button>
              <button className="btn sm ghost danger" onClick={() => setConfirmDel({ ids: [current.id], title: current.title })}>Delete</button>
            </>
          )}
          <button className="btn sm" onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>Chat settings</button>
        </div>

        {!loaded && (
          <div className="notice warn row" style={{ margin: "10px 16px 0" }}>
            <span>{status?.state === "loading" ? "A model is loading." : "No model is loaded. Load one to chat."}</span>
            <span className="spacer" />
            <button className="btn sm primary" onClick={() => go("models")}>Open My Models</button>
          </div>
        )}

        <div
          className="chat-scroll"
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
          }}
        >
          {messages.length === 0 && (
            <div className="empty" style={{ padding: "40px 0" }}>
              <strong>Start the conversation.</strong>Messages are saved on this PC. The reply streams from the loaded model.
            </div>
          )}
          {messages.filter((m) => m.role !== "tool").map((m) => m.role === "user" ? (
            <UserRow key={m.id} m={m} editingText={editing?.id === m.id ? editing.text : null} generating={generating} act={act} />
          ) : (
            <AssistantRow key={m.id} m={m} isLast={m.id === lastAssistant?.id} generating={generating} loaded={loaded} convId={currentId}
              results={toolResults} planEdit={planEdit?.id === m.id ? planEdit.text : null} act={act}
              approvals={m.tool_calls ? approvals : NO_MAP} liveTools={m.tool_calls ? liveTools : NO_MAP}
              liveOut={m.tool_calls ? liveOut : NO_MAP} callServer={m.tool_calls ? callServer : NO_MAP} />
          ))}
        </div>

        <div className={`composer${dragOver ? " drag-over" : ""}`}>
          <ContextMeter r={ctx} err={ctxErr} />
          {err && <div className="notice error" style={{ marginTop: 6 }}>{err}</div>}
          {projNeeded && support?.has_projector && !support.projector_loaded && (
            <div className="notice warn row" style={{ marginTop: 6 }}>
              <span>Images need {support.model_name}'s vision projector, which wasn't loaded.</span>
              <span className="spacer" />
              <button className="btn sm primary" disabled={projReloading || generating} onClick={reloadWithProjector}>{projReloading ? "Reloading..." : "Reload with vision projector"}</button>
            </div>
          )}
          {dragOver && <div className="drop-hint">Drop files to attach them to your next message</div>}
          <AttachmentChips items={pending} onRemove={generating ? undefined : removeAttachment} />
          <div className="row composer-tools">
            {/* --- chat mode / tools menus --- */}
            <ModeMenu mode={modeOf(settings)} onChange={setMode} />
            {modeOf(settings) === "agent" && (
              <SubagentControls allowed={!!settings.subagents} max={settings.max_subagents ?? 2}
                onChange={(allowed, max) => saveSettings({ ...settings, subagents: allowed, max_subagents: max })} />
            )}
            <ToolsMenu conversationId={currentId} settings={settings} mode={modeOf(settings)} onChange={saveSettings} onManage={() => go("mcp")}
              refreshKey={`${status?.instance_id ?? ""}|${modeOf(settings)}|${settings.builtin_tools}|${settings.web_search}|${settings.environment_id ?? ""}|${settings.tool_servers.join(",")}|${(settings.disabled_servers ?? []).join(",")}|${ctxTick}`} />
            <span className="composer-sep" aria-hidden />
            {/* --- end chat mode / tools menus --- */}
            <button className="btn sm ghost" aria-label="Attach files" disabled={attaching || generating}
              title={support?.images ? "Attach text, code, PDF or image files (or drop them here)" : "Attach text, code or PDF files (or drop them here)"} onClick={pickFiles}>
              {attaching ? "Attaching..." : "Attach"}
            </button>
            <ThinkingControls compact support={support} thinking={settings.thinking} effort={settings.reasoning_effort}
              onChange={(t, e) => saveSettings({ ...settings, thinking: t, reasoning_effort: e })} />
          </div>
          {wtNote && (
            <div className="notice warn row worktree-kept" role="status" aria-label="Kept worktrees" style={{ marginTop: 6 }}>
              <span style={{ whiteSpace: "pre-wrap" }}>{wtNote}</span>
              <span className="spacer" />
              <button className="btn sm ghost" onClick={() => setWtNote(null)}>Dismiss</button>
            </div>
          )}
          <div className="row" style={{ alignItems: "flex-end", marginTop: 6, flexWrap: "nowrap" }}>
            <textarea
              className="input composer-input"
              aria-label="Message"
              placeholder={loaded ? "Message the model (Enter to send, Shift+Enter for a new line)" : "Load a model to chat"}
              value={draft}
              rows={3}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            {generating ? (
              <button className="btn" onClick={stop}>Stop</button>
            ) : (
              <button className="btn primary" onClick={send} disabled={!loaded || (!draft.trim() && pending.length === 0)}>Send</button>
            )}
          </div>
        </div>
      </section>

      {showSettings && (
        <aside className="chat-inspector">
          <strong style={{ fontWeight: 600 }}>Chat settings</strong>
          {/* --- tool settings (mode, servers and subagents moved to the composer bar) --- */}
          <div className="panel" style={{ display: "grid", gap: 8, padding: 10 }}>
            <span className="faint" style={{ fontSize: 12 }}>Plan and Agent mode. Pick the mode and which tools this chat gets in the bar above the message box.</span>
            {settings.builtin_tools && (
              <div className="field">
                <label htmlFor="cs-env">Run built-in tools on</label>
                <select id="cs-env" className="input" value={settings.environment_id ?? "local"}
                  onChange={(e) => saveSettings({ ...settings, environment_id: e.target.value === "local" ? null : e.target.value })}>
                  {(envs.length ? envs : [{ id: "local", name: "This PC" } as EnvView]).map((en) => <option key={en.id} value={en.id}>{en.name}{en.kind && en.kind !== "local" ? ` (${en.kind})` : ""}</option>)}
                </select>
                <span className="help">Set up SSH, Docker and sandboxes on the Execution Environments page.</span>
              </div>
            )}
            <label className="check">
              <input type="checkbox" aria-label="Use skills" checked={settings.skills} onChange={(e) => saveSettings({ ...settings, skills: e.target.checked })} /> Use skills (matching ones are added to the instruction)
            </label>
            <div className="field">
              <label htmlFor="cs-toolmode">Tool calls</label>
              <select id="cs-toolmode" className="input" aria-label="Tool calls" value={settings.tool_mode} onChange={(e) => saveSettings({ ...settings, tool_mode: e.target.value })}>
                <option value="auto">Automatic</option>
                <option value="native">Native</option>
                <option value="xml">Text format (for models without tool calling)</option>
              </select>
              <span className="help">Automatic uses native tool calls when the model's chat template supports them, otherwise the text format.</span>
            </div>
            <div className="field">
              <label htmlFor="cs-maxcalls">Tool calls per reply</label>
              <input id="cs-maxcalls" className="input mono" placeholder="No limit" value={settings.limits.max_tool_calls ?? ""}
                onChange={(e) => setSettings({ ...settings, limits: { ...settings.limits, max_tool_calls: num(e.target.value) } })} onBlur={() => saveSettings(settings)} />
              <span className="help">Empty = no limit.</span>
            </div>
            <div className="field">
              <label htmlFor="cs-wall">Reply time limit (seconds)</label>
              <input id="cs-wall" className="input mono" placeholder="No limit" value={settings.limits.max_wall_secs ?? ""}
                onChange={(e) => setSettings({ ...settings, limits: { ...settings.limits, max_wall_secs: num(e.target.value) } })} onBlur={() => saveSettings(settings)} />
              <span className="help">Empty = no limit.</span>
            </div>
          </div>
          {/* --- end tool settings --- */}
          {support && (
            <div className="field">
              <label>Thinking ({support.model_name})</label>
              <ThinkingControls support={support} thinking={settings.thinking} effort={settings.reasoning_effort}
                onChange={(t, e) => saveSettings({ ...settings, thinking: t, reasoning_effort: e })} />
              <span className="help">Shown when the loaded model's chat template supports it. Sent per reply as chat_template_kwargs; the reasoning block shows what the model did.</span>
            </div>
          )}
          {!current && <p className="faint">These apply once the chat exists.</p>}
          <div className="field">
            <label htmlFor="cs-prof">System instruction</label>
            <select id="cs-prof" className="input" value={settings.profile_id ?? ""} onChange={(e) => saveSettings({ ...settings, profile_id: e.target.value || null })}>
              <option value="">Use model or app default</option>
              {profiles.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
            </select>
            <span className="help">Edit profiles on the System Instructions page.</span>
          </div>
          {([
            ["temperature", "Temperature", "0 to 5"],
            ["top_p", "Top-p", "0 to 1"],
            ["top_k", "Top-k", "e.g. 40"],
            ["min_p", "Min-p", "0 to 1"],
            ["repeat_penalty", "Repetition penalty", "e.g. 1.1"],
            ["max_tokens", "Max reply tokens", "model default"],
            ["seed", "Seed", "random"],
          ] as const).map(([k, label, ph]) => (
            <div className="field" key={k}>
              <label htmlFor={`cs-${k}`}>{label}</label>
              <input id={`cs-${k}`} className="input mono" placeholder={ph} value={settings.sampling[k] ?? ""}
                onChange={(e) => setSettings({ ...settings, sampling: { ...settings.sampling, [k]: num(e.target.value) } })}
                onBlur={() => saveSettings(settings)} />
            </div>
          ))}
          <div className="field">
            <label htmlFor="cs-stop">Stop sequences (one per line)</label>
            <textarea id="cs-stop" className="input mono" rows={2} value={settings.sampling.stop.join("\n")}
              onChange={(e) => setSettings({ ...settings, sampling: { ...settings.sampling, stop: e.target.value.split("\n").filter((s) => s.length > 0) } })}
              onBlur={() => saveSettings(settings)} />
          </div>
          <div className="field">
            <label htmlFor="cs-over">When the context is full</label>
            <select id="cs-over" className="input" value={settings.overflow} onChange={(e) => saveSettings({ ...settings, overflow: e.target.value as ChatSettings["overflow"] })}>
              <option value="warn">Stop and tell me</option>
              <option value="drop_oldest">Leave out the oldest messages (shown on the reply)</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="cs-res">Tokens reserved for the reply</label>
            <input id="cs-res" className="input mono" value={settings.reserve_tokens}
              onChange={(e) => setSettings({ ...settings, reserve_tokens: num(e.target.value) ?? 0 })} onBlur={() => saveSettings(settings)} />
            <span className="help">Used when max reply tokens is not set.</span>
          </div>
        </aside>
      )}
    </div>
  );
}

import { memo, useCallback, useLayoutEffect, useMemo, useSyncExternalStore } from "react";
import type { Message, MessageStats } from "../../lib/api2";
import type { ApprovalAnswer, ApprovalRequest } from "../../lib/api3";
import { parseAttachments, type Attachment } from "../../lib/api6";
import { streamStore } from "../../lib/streamStore";
import { Markdown, StreamingMarkdown } from "../Markdown";
import { ToolCard, type ToolCall } from "../ToolCard";
import { ChatSubagents } from "../ChatSubagents";

function parseStats(raw: string | null | undefined): MessageStats | null {
  try {
    return raw ? (JSON.parse(raw) as MessageStats) : null;
  } catch {
    return null;
  }
}

function parseCalls(raw: string | null | undefined): ToolCall[] {
  try {
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function StatsLine({ s }: { s: MessageStats }) {
  if (s.stop_reason === "error") return <div className="danger-text" style={{ fontSize: 12 }}>Generation failed: {s.error}</div>;
  const parts: string[] = [];
  if (s.completion_tokens != null) parts.push(`${s.completion_tokens} tokens`);
  if (s.tokens_per_sec != null) parts.push(`${s.tokens_per_sec.toFixed(1)} tok/s`);
  if (s.time_to_first_token_ms != null) parts.push(`first token ${(s.time_to_first_token_ms / 1000).toFixed(2)} s`);
  if (s.prompt_tokens_per_sec != null) parts.push(`prompt ${Math.round(s.prompt_tokens_per_sec)} tok/s`);
  if (s.total_ms != null) parts.push(`${(s.total_ms / 1000).toFixed(1)} s total`);
  if (s.context_tokens != null && s.n_ctx) parts.push(`context ${s.context_tokens.toLocaleString()} / ${s.n_ctx.toLocaleString()}`);
  if (s.thinking_requested != null) parts.push(`thinking ${s.thinking_requested ? "on" : "off"}`);
  else if (s.thinking_control === "always") parts.push("thinking always on");
  if (s.reasoning_effort) parts.push(`effort ${s.reasoning_effort}`);
  const stop = s.stop_reason === "length" ? "stopped: max tokens reached" : s.stop_reason === "cancelled" ? "stopped by you" : null;
  return (
    <div className="mono faint" style={{ fontSize: 11 }}>
      {parts.join("  ")}
      {stop && <span style={{ color: "var(--accent)" }}>  {stop}</span>}
      {s.dropped_messages ? <div style={{ color: "var(--accent)" }}>{s.dropped_messages} oldest message(s) were left out to fit the context.</div> : null}
    </div>
  );
}

export function AttachmentChips({ items, onRemove }: { items: Attachment[]; onRemove?: (a: Attachment) => void }) {
  if (items.length === 0) return null;
  return (
    <div className="attach-chips" aria-label={onRemove ? "Pending attachments" : "Attachments"}>
      {items.map((a) => (
        <span key={a.id} className={`attach-chip kind-${a.kind}`} title={`${a.name} (${a.kind}, ${(a.size / 1024).toFixed(1)} KB${a.tokens != null ? `, about ${a.tokens} tokens` : ""})`}>
          <span className="attach-kind">{a.kind === "pdf" ? "PDF" : a.kind}</span>
          <span className="attach-name">{a.name}</span>
          {a.tokens != null && <span className="faint mono">{a.tokens} tok</span>}
          {onRemove && <button className="attach-x" aria-label={`Remove ${a.name}`} onClick={() => onRemove(a)}>x</button>}
        </span>
      ))}
    </div>
  );
}

/** Handlers the rows call. The object is stable; it always runs the latest Chat closures. */
export interface RowActions {
  answerApproval: (id: string, a: ApprovalAnswer) => void;
  openSubagents: () => void;
  startEdit: (id: string, text: string) => void;
  editText: (id: string, text: string) => void;
  cancelEdit: () => void;
  saveEdit: (id: string, text: string) => void;
  startPlanEdit: (id: string, text: string) => void;
  planText: (id: string, text: string) => void;
  cancelPlanEdit: () => void;
  runPlan: (text: string) => void;
  regenerate: () => void;
  /** A live row grew: keep the view at the bottom if the user is following. */
  grew: () => void;
}

export const UserRow = memo(function UserRow({ m, editingText, generating, act }: {
  m: Message; editingText: string | null; generating: boolean; act: RowActions;
}) {
  const atts = useMemo(() => parseAttachments(m.attachments), [m.attachments]);
  return (
    <div className="msg user">
      {editingText !== null ? (
        <div style={{ display: "grid", gap: 6 }}>
          <textarea className="input composer-input" aria-label="Edit message" value={editingText} onChange={(e) => act.editText(m.id, e.target.value)} rows={4} />
          <div className="row">
            <button className="btn sm primary" onClick={() => act.saveEdit(m.id, editingText)}>Save and resend</button>
            <button className="btn sm ghost" onClick={act.cancelEdit}>Cancel</button>
          </div>
        </div>
      ) : (
        <>
          {m.content && <div className="msg-text">{m.content}</div>}
          <AttachmentChips items={atts} />
          <div className="msg-actions">
            <button className="btn ghost sm" disabled={generating} onClick={() => act.startEdit(m.id, m.content)}>Edit</button>
          </div>
        </>
      )}
    </div>
  );
});

const ToolCards = memo(function ToolCards({ calls, results, approvals, liveTools, liveOut, callServer, convId, act }: {
  calls: ToolCall[];
  results: Map<string, Message>;
  approvals: Record<string, ApprovalRequest>;
  liveTools: Record<string, string>;
  liveOut: Record<string, string>;
  callServer: Record<string, string>;
  convId: string | null;
  act: RowActions;
}) {
  return (
    <>
      {calls.map((c) => (
        <ToolCard key={c.id} call={c} result={results.get(c.id)}
          approval={approvals[c.id]} live={liveTools[c.id]} output={liveOut[c.id]} where={callServer[c.id] ?? approvals[c.id]?.server_name} onAnswer={act.answerApproval}
          extra={c.function.name === "spawn_subagents" && convId ? (
            <ChatSubagents conversationId={convId} callId={c.id} approvals={Object.values(approvals)} onAnswer={act.answerApproval} onOpenPage={act.openSubagents} />
          ) : undefined} />
      ))}
    </>
  );
});

export const AssistantRow = memo(function AssistantRow({
  m, isLast, generating, loaded, convId, results, approvals, liveTools, liveOut, callServer, planEdit, act,
}: {
  m: Message;
  isLast: boolean;
  generating: boolean;
  loaded: boolean;
  convId: string | null;
  results: Map<string, Message>;
  approvals: Record<string, ApprovalRequest>;
  liveTools: Record<string, string>;
  liveOut: Record<string, string>;
  callServer: Record<string, string>;
  /** Plan text being edited when this message's plan editor is open. */
  planEdit: string | null;
  act: RowActions;
}) {
  const sub = useCallback((f: () => void) => streamStore.subscribe(m.id, f), [m.id]);
  const get = useCallback(() => streamStore.get(m.id), [m.id]);
  const live = useSyncExternalStore(sub, get);
  // A live row grew: follow it once per rendered frame.
  useLayoutEffect(() => {
    if (live) act.grew();
  }, [live, act]);

  const stats = useMemo(() => parseStats(m.stats), [m.stats]);
  const calls = useMemo(() => parseCalls(m.tool_calls), [m.tool_calls]);
  const content = live ? live.content : m.content;
  const reasoning = live ? live.reasoning : m.reasoning ?? "";
  const isLive = !!live || (generating && isLast && !m.stats);
  const answer = (text: string) => (isLive ? <StreamingMarkdown text={text} /> : <Markdown text={text} />);
  return (
    <div className="msg assistant">
      <div className="faint mono" style={{ fontSize: 11, marginBottom: 4 }}>{m.model_name ?? "model"} (generated by the model)</div>
      {reasoning && (
        <details className="reasoning" open={isLive && !content}>
          <summary>Reasoning{isLive && !content ? " (thinking)" : ""}</summary>
          <div className="reasoning-text">{reasoning}</div>
        </details>
      )}
      {calls.length > 0 ? (
        <div style={{ display: "grid", gap: 8, margin: "6px 0" }}>
          {content && answer(content)}
          <ToolCards calls={calls} results={results} approvals={approvals} liveTools={liveTools} liveOut={liveOut} callServer={callServer} convId={convId} act={act} />
        </div>
      ) : null}
      {m.tool_calls ? null : content ? answer(content) : isLive ? <div className="faint">{reasoning ? "" : "Waiting for the first token"}</div> : stats?.stop_reason === "cancelled" ? <div className="faint">Stopped before any answer text.</div> : stats && stats.stop_reason !== "error" ? (
        <div className="notice warn empty-answer" role="status">{stats.note ?? "The model returned no answer."}{reasoning ? " It wrote only reasoning (above)." : ""} Try Regenerate, or ask again.</div>
      ) : null}
      {stats && <StatsLine s={stats} />}
      {!isLive && stats?.mode === "plan" && !m.tool_calls && content && isLast && stats.stop_reason !== "error" && (
        <div className="plan-actions">
          {planEdit !== null ? (
            <>
              <textarea className="input composer-input" aria-label="Plan to run" rows={8} value={planEdit} onChange={(e) => act.planText(m.id, e.target.value)} />
              <div className="row">
                <button className="btn sm primary" disabled={!loaded || generating || !planEdit.trim()} onClick={() => act.runPlan(planEdit)}>Run this plan</button>
                <button className="btn sm ghost" onClick={act.cancelPlanEdit}>Cancel</button>
              </div>
            </>
          ) : (
            <div className="row">
              <button className="btn sm primary" disabled={!loaded || generating} onClick={() => act.runPlan(content)} title="Switch this chat to Agent mode and carry out the plan">Run this plan</button>
              <button className="btn sm ghost" disabled={generating} onClick={() => act.startPlanEdit(m.id, content)}>Edit plan</button>
              <span className="help faint">Runs it in Agent mode (tools still ask before running unless you allowed them).</span>
            </div>
          )}
        </div>
      )}
      {!isLive && (
        <div className="msg-actions">
          <button className="btn ghost sm" onClick={() => navigator.clipboard.writeText(content)}>Copy</button>
          {isLast && <button className="btn ghost sm" disabled={!loaded || generating} onClick={act.regenerate}>Regenerate</button>}
        </div>
      )}
    </div>
  );
});

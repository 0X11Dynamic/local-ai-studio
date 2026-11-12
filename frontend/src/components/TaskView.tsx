import { useMemo, useState } from "react";
import { ChatCircleDots, PaperPlaneRight } from "@phosphor-icons/react";
import { Markdown } from "./Markdown";
import { ToolCard, type ToolCall } from "./ToolCard";
import type { Message } from "../lib/api2";
import type { ApprovalAnswer } from "../lib/api3";
import { STATE_LABEL, parseJson, type QuestionData, type StepRow, type SubagentRow, type Usage, type UserMessageData } from "../lib/api5";

/** Agent states that mean "still working or waiting". */
export const ACTIVE = ["queued", "running", "waiting", "waiting_for_user"];

const BLANK_Q: QuestionData = { question_id: "", question: "", options: [], allow_text: true, call_id: "", agent_label: "Main agent", answer: null };

/** A question an agent asked with ask_user: option buttons and/or a text box. */
export function QuestionCard({ step, onAnswer }: { step: StepRow; onAnswer: (questionId: string, answer: string) => Promise<void> }) {
  const d = parseJson<QuestionData>(step.data, BLANK_Q);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const send = async (a: string) => {
    if (!a.trim() || sending) return;
    setSending(true);
    try {
      await onAnswer(d.question_id, a.trim());
      setText("");
    } finally {
      setSending(false);
    }
  };
  return (
    <section className="task-question" aria-label="Question from the task">
      <div className="row task-question-head">
        <ChatCircleDots size={16} aria-hidden="true" />
        <span className="tag accent">{d.agent_label}</span>
        <span className="faint">needs your answer</span>
      </div>
      <div className="task-question-text">{d.question}</div>
      {d.options.length > 0 && (
        <div className="row task-options" role="group" aria-label="Answer options">
          {d.options.map((o) => (
            <button key={o} className="btn sm" disabled={sending} onClick={() => send(o)}>{o}</button>
          ))}
        </div>
      )}
      {d.allow_text && (
        <div className="task-answer-row">
          <textarea className="input" rows={2} aria-label="Your answer" value={text} disabled={sending}
            placeholder={d.options.length ? "Or type your own answer" : "Type your answer"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(text); } }} />
          <button className="btn sm primary" aria-label="Send answer" disabled={sending || !text.trim()} onClick={() => send(text)}>
            <PaperPlaneRight size={14} aria-hidden="true" /> Send answer
          </button>
        </div>
      )}
      {!d.allow_text && <span className="faint task-hint">Pick one of the options.</span>}
    </section>
  );
}

/** The message box at the bottom of the task page. */
export function ReplyBox({ placeholder, buttonLabel, disabled, hint, onSend, showJump, onJump }: {
  placeholder: string; buttonLabel: string; disabled: boolean; hint?: string; onSend: (text: string) => Promise<boolean>;
  showJump: boolean; onJump: () => void;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const send = async () => {
    if (!text.trim() || sending || disabled) return;
    setSending(true);
    try {
      if (await onSend(text.trim())) setText("");
    } finally {
      setSending(false);
    }
  };
  return (
    <div className="task-reply">
      {showJump && <button className="btn sm task-jump" aria-label="Jump to newest" onClick={onJump}>Jump to newest</button>}
      <div className="task-reply-row">
        <textarea className="input" rows={2} aria-label="Message the task" value={text} disabled={disabled || sending} placeholder={placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }} />
        <button className="btn sm primary" aria-label={buttonLabel} disabled={disabled || sending || !text.trim()} onClick={send}>
          <PaperPlaneRight size={14} aria-hidden="true" /> {buttonLabel}
        </button>
      </div>
      {hint && <span className="faint task-hint">{hint}</span>}
    </div>
  );
}

interface TimelineProps {
  /** Every step of the run; the timeline shows those of `agentId`. */
  steps: StepRow[];
  /** null = the main agent, else a subagent id. */
  agentId: string | null;
  live: Record<string, { content: string; reasoning: string }>;
  liveTools: Record<string, string>;
  liveOut: Record<string, string>;
  callServer: Record<string, string>;
  onAnswer: (id: string, a: ApprovalAnswer) => void;
  subagents: SubagentRow[];
  onStop: (id: string) => void;
  /** While a plan awaits review, its text is shown in the review panel, not repeated here. */
  hidePlan?: string;
}

/** One agent's steps as readable entries; subagents are collapsible groups with their own steps. */
export function Timeline(props: TimelineProps) {
  const { steps, agentId, live, liveTools, liveOut, callServer, onAnswer, subagents, onStop, hidePlan } = props;
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
  const hide = hidePlan?.trim();
  const mine = steps.filter((s) => {
    if (s.kind === "tool" || (s.parent_agent_id ?? null) !== agentId) return false;
    if (!hide) return true;
    if (s.kind === "plan") return false;
    // The planning reply is the plan itself.
    return !(s.kind === "model" && s.state !== "running" && String(parseJson<Record<string, unknown>>(s.data, {}).text ?? "").trim() === hide);
  });
  if (mine.length === 0) return <div className="faint">{agentId ? "No steps yet." : "No steps yet. The task is starting."}</div>;
  return (
    <div className="timeline" role="list" aria-label={agentId ? "Subagent steps" : "Task timeline"}>
      {mine.map((s) => {
        const d = parseJson<Record<string, unknown>>(s.data, {});
        if (s.kind === "model") {
          const running = s.state === "running";
          const text = (running ? live[s.id]?.content : (d.text as string)) ?? "";
          const reasoning = (running ? live[s.id]?.reasoning : (d.reasoning as string)) ?? "";
          // ask_user is shown as its own question entry.
          const calls = ((d.tool_calls as ToolCall[] | undefined) ?? []).filter((c) => c.function.name !== "ask_user");
          const active = calls.some((c) => !results[c.id]);
          const names = [...new Set(calls.map((c) => c.function.name.replace(/^mcp__/, "").split("__").pop()))].join(", ");
          return (
            <div key={s.id} className="step model-step" role="listitem">
              {reasoning && <details className="reasoning"><summary className="faint">Reasoning</summary><div className="reasoning-text">{reasoning}</div></details>}
              {/* While streaming, plain text: re-parsing Markdown on every token is costly. */}
              {running ? (text ? <div className="live-text">{text}</div> : <div className="faint">Thinking</div>) : text ? <Markdown text={text} /> : null}
              {calls.length > 0 && (
                <details className="calls" open={active}>
                  <summary>
                    <span className="mono">{calls.length} tool call{calls.length === 1 ? "" : "s"}</span> <span className="faint">{names}</span>
                    {active && <span className="tag accent">running</span>}
                  </summary>
                  <div className="calls-body">
                    {calls.map((c) => (
                      <ToolCard key={c.id} call={c} result={results[c.id]} approval={undefined} live={liveTools[c.id]} output={liveOut[c.id]}
                        where={callServer[c.id]} onAnswer={onAnswer} sessionLabel="Allow for this task" />
                    ))}
                  </div>
                </details>
              )}
            </div>
          );
        }
        if (s.kind === "user") {
          const u = d as unknown as UserMessageData;
          const status = u.delivered ? "Sent" : s.state === "queued" ? "Sent. The agent reads it before its next step." : u.note ?? "Not read before the task ended.";
          return (
            <div key={s.id} className="step-user task-user" role="listitem">
              <div className="msg user"><div className="msg-text">{u.text}</div></div>
              <span className="faint task-user-note">You, {status}</span>
            </div>
          );
        }
        if (s.kind === "question") {
          const q = parseJson<QuestionData>(s.data, BLANK_Q);
          return (
            <div key={s.id} className="step question-step" role="listitem">
              <div className="step-head faint mono">{q.agent_label} asked you</div>
              <div className="task-question-text">{q.question}</div>
              {q.options.length > 0 && <div className="row">{q.options.map((o) => <span key={o} className={`tag${q.answer === o ? " accent" : ""}`}>{o}</span>)}</div>}
              {s.state === "waiting" ? (
                <span className="tag accent">Waiting for your answer</span>
              ) : s.state === "done" && q.answer != null ? (
                <div className="task-user"><div className="msg user"><div className="msg-text">{q.answer}</div></div><span className="faint task-user-note">Your answer</span></div>
              ) : (
                <span className="faint">{s.state === "cancelled" ? "Not answered: the task was stopped." : "Not answered: the app was closed first."}</span>
              )}
            </div>
          );
        }
        if (s.kind === "subagents") {
          const kids = (d.children as { id: string; label: string; objective: string }[]) ?? [];
          return (
            <div key={s.id} className="step sub-step" role="listitem">
              <div className="step-head faint mono">Started {kids.length} subagent{kids.length === 1 ? "" : "s"}</div>
              {kids.map((k) => {
                const row = subagents.find((x) => x.id === k.id);
                const state = row?.state ?? "queued";
                const u = parseJson<Usage>(row?.usage, { turns: 0, tool_calls: 0, wall_ms: 0, tokens: 0 });
                return (
                  <details key={k.id} className="sub-group" open={state === "waiting_for_user" || undefined}>
                    <summary>
                      <span className="mono">{k.label}</span>
                      <span className={`tag${state === "waiting_for_user" ? " accent" : ""}`}>{STATE_LABEL[state] ?? state}</span>
                      <span className="faint mono tiny">{u.turns} turns, {u.tool_calls} calls</span>
                      {row && ACTIVE.includes(row.state) && (
                        <button className="btn sm ghost danger" aria-label={`Stop subagent ${k.label}`} onClick={(e) => { e.preventDefault(); e.stopPropagation(); onStop(k.id); }}>Stop</button>
                      )}
                    </summary>
                    <div className="sub-body">
                      <div className="dim">{k.objective}</div>
                      <Timeline {...props} agentId={k.id} />
                      {row?.result && <div className="sub-result"><Markdown text={row.result} /></div>}
                      {row?.error && <div className="danger-text">{row.error}</div>}
                    </div>
                  </details>
                );
              })}
            </div>
          );
        }
        if (s.kind === "plan") {
          return (
            <div key={s.id} className="step plan-step" role="listitem">
              <div className="step-head faint mono">Plan</div>
              <Markdown text={String(d.text ?? "")} />
            </div>
          );
        }
        return (
          <div key={s.id} className={`step note${s.kind === "error" ? " danger-text" : ""}`} role="listitem">{String(d.text ?? s.kind)}</div>
        );
      })}
    </div>
  );
}

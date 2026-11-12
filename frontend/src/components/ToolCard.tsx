import { useState, type ReactNode } from "react";
import type { Message } from "../lib/api2";
import type { ApprovalAnswer, ApprovalRequest } from "../lib/api3";

export interface ToolCall { id: string; type: string; function: { name: string; arguments: string } }

const STATUS_LABEL: Record<string, string> = {
  ok: "Done",
  tool_error: "Tool reported an error",
  denied: "Not run: denied",
  invalid_arguments: "Not run: invalid arguments",
  unknown_tool: "Not run: unknown tool",
  timeout: "Timed out",
  error: "Failed",
  cancelled: "Cancelled",
  limit: "Not run: call limit reached",
};

function pretty(args: string): string {
  try {
    return JSON.stringify(JSON.parse(args || "{}"), null, 2);
  } catch {
    return args;
  }
}

function argsOf(args: string): Record<string, unknown> {
  try {
    const v = JSON.parse(args || "{}");
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

/** Tool output; diffs from fs_write / fs_patch get +/- line colours. */
function Output({ text, label }: { text: string; label: string }) {
  const isDiff = /^--- a\//m.test(text) && /^\+\+\+ b\//m.test(text);
  if (!isDiff) return <pre className="preview-text" aria-label={label}>{text}</pre>;
  return (
    <pre className="preview-text" aria-label={label}>
      {text.split("\n").map((l, i) => (
        <span key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "diff-add" : l.startsWith("-") && !l.startsWith("---") ? "diff-del" : l.startsWith("@@") ? "faint" : undefined}>
          {l}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

/** A tool call made by the model, its approval state and its output. */
export function ToolCard({
  call, result, approval, live, output, where, onAnswer, sessionLabel = "Allow for this chat", extra,
}: {
  call: ToolCall;
  result: Message | undefined;
  approval: ApprovalRequest | undefined;
  live: string | undefined;
  /** Live stdout/stderr while a shell command runs. */
  output?: string;
  /** Server or environment the call runs on. */
  where?: string;
  onAnswer: (id: string, a: ApprovalAnswer) => void;
  /** Text of the session-scope button ("Allow for this task" in agent tasks). */
  sessionLabel?: string;
  /** Extra live content under the call (e.g. the children of spawn_subagents). */
  extra?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const stats = (() => {
    try {
      return result?.stats ? (JSON.parse(result.stats) as { status?: string }) : null;
    } catch {
      return null;
    }
  })();
  const status = stats?.status;
  const isMcp = call.function.name.startsWith("mcp__");
  const [server, tool] = isMcp ? call.function.name.replace(/^mcp__/, "").split("__") : [where ?? "built-in", call.function.name];
  const a = argsOf(call.function.arguments);
  const command = call.function.name === "shell_execute" && typeof a.command === "string" ? a.command : null;
  const state = approval ? "Waiting for your approval" : status ? STATUS_LABEL[status] ?? status : live ?? "Running";
  const bad = status && !["ok"].includes(status);
  return (
    <div className={`tool-card${approval ? " waiting" : ""}`}>
      <div className="row" style={{ gap: 8 }}>
        <span className="mono" style={{ fontWeight: 600 }}>{tool ?? call.function.name}</span>
        <span className="faint">on {isMcp ? server ?? "server" : where ?? server}</span>
        <span className="spacer" />
        <span className={bad ? "danger-text" : approval ? "" : "faint"} style={approval ? { color: "var(--accent)" } : undefined}>{state}</span>
      </div>
      {command != null ? (
        <pre className="tool-command mono" aria-label="Command">{command}</pre>
      ) : (
        <pre className="tool-args mono">{pretty(call.function.arguments)}</pre>
      )}
      {!result && output && <pre className="preview-text" aria-label="Live output">{output}</pre>}
      {extra}
      {approval && (
        <div style={{ display: "grid", gap: 6 }}>
          <span className="dim">{approval.agent ? `${approval.agent} wants to run this tool.` : "The model wants to run this tool."} {approval.reason}.</span>
          {/* grant_choice: "session" = a place outside the allowed folders (this chat/task, that folder only); "once" = asks every time */}
          <div className="row">
            <button className="btn sm primary" onClick={() => onAnswer(approval.id, "allow_once")}>Allow once</button>
            {(approval as { grant_choice?: string }).grant_choice !== "once" && (
              <button className="btn sm" onClick={() => onAnswer(approval.id, "allow_session")}>{(approval as { grant_choice?: string }).grant_choice === "session" ? `${sessionLabel} (this folder)` : sessionLabel}</button>
            )}
            {!["once", "session"].includes((approval as { grant_choice?: string }).grant_choice ?? "") && (
              <button className="btn sm" onClick={() => onAnswer(approval.id, "allow_always")}>Always allow this tool</button>
            )}
            <button className="btn sm danger" onClick={() => onAnswer(approval.id, "deny")}>Deny</button>
          </div>
        </div>
      )}
      {result && (
        <div>
          <button className="btn ghost sm" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? "Hide tool output" : "Show tool output"} ({result.content.length.toLocaleString()} chars)
          </button>
          {open && <Output text={result.content} label="Tool output" />}
        </div>
      )}
    </div>
  );
}

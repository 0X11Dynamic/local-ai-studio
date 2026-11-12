import { useEffect, useState } from "react";
import { UsersThree } from "@phosphor-icons/react";

/**
 * Agent mode only: "Allow subagents" (checkbox) and "Max subagents at once"
 * (1-4, saved on blur or Enter). Sits next to the mode menu in the composer.
 */
export function SubagentControls({ allowed, max, onChange, disabled }: {
  allowed: boolean;
  max: number;
  onChange: (allowed: boolean, max: number) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState(String(max));
  useEffect(() => setText(String(max)), [max]);
  const commit = () => {
    const n = Math.round(Number(text));
    if (!Number.isFinite(n) || n < 1 || n > 4) {
      setText(String(max));
      return;
    }
    if (n !== max) onChange(allowed, n);
  };
  return (
    <div className={`sub-ctl${allowed ? " on" : ""}`} title="The model can start subagents that work in parallel and share this reply's limits. Stop stops them all. Subagents that edit files can get their own git worktree.">
      <label className="check">
        <input type="checkbox" aria-label="Allow subagents" checked={allowed} disabled={disabled} onChange={(e) => onChange(e.target.checked, max)} />
        <UsersThree size={14} aria-hidden />
        <span>Subagents</span>
      </label>
      {allowed && (
        <label className="sub-max">
          <span className="faint">max</span>
          <input type="number" min={1} max={4} className="input mono" aria-label="Max subagents at once" value={text} disabled={disabled}
            onChange={(e) => setText(e.target.value)} onBlur={commit}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } }} />
        </label>
      )}
    </div>
  );
}

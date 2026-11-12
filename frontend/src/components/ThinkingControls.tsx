import { useEffect, useState } from "react";
import { useEngine } from "../lib/engine";
import { api6, thinkingUi, type ModelSupport } from "../lib/api6";

/** What the loaded model takes (thinking controls, images, audio); refetched when the engine changes. */
export function useModelSupport(): [ModelSupport | null, () => void] {
  const { status } = useEngine();
  const [support, setSupport] = useState<ModelSupport | null>(null);
  const [tick, setTick] = useState(0);
  const key = `${status?.state}|${status?.instance_id}|${(status?.args ?? []).includes("--mmproj")}`;
  useEffect(() => {
    if (status?.state !== "loaded") {
      setSupport(null);
      return;
    }
    let live = true;
    api6.modelSupport().then((s) => live && setSupport(s)).catch(() => live && setSupport(null));
    return () => {
      live = false;
    };
  }, [key, tick]);
  return [support, () => setTick((t) => t + 1)];
}

/**
 * Thinking on/off (aria-label "Thinking") and effort (aria-label "Reasoning effort"),
 * shown only when the loaded model supports them. "Always" models say so; none/unknown show nothing.
 */
export function ThinkingControls({ support, thinking, effort, onChange, compact }: {
  support: ModelSupport | null;
  thinking: boolean | null | undefined;
  effort: string | null | undefined;
  onChange: (thinking: boolean | null, effort: string | null) => void;
  compact?: boolean;
}) {
  const c = support?.caps;
  const ui = thinkingUi(c);
  if (!c || (!ui.toggle && !ui.always && ui.effort.length === 0)) return null;
  const on = thinking ?? c.thinking_default ?? false;
  return (
    <div className={`thinking-controls${compact ? " compact" : ""}`}>
      {ui.toggle && (
        <label className="check" title={`Sent to the model as ${c.thinking_kwarg}. ${thinking == null ? `Model default (${c.thinking_default == null ? "unknown" : c.thinking_default ? "on" : "off"}).` : ""}`}>
          <input type="checkbox" aria-label="Thinking" checked={on} onChange={(e) => onChange(e.target.checked, effort ?? null)} /> Thinking
        </label>
      )}
      {ui.always && <span className="tag" title="This model's template always starts a reasoning block; it can't be turned off">always thinks</span>}
      {ui.effort.length > 0 && (
        <select className="input sm" aria-label="Reasoning effort" value={effort ?? ""} disabled={ui.toggle && !on}
          title={ui.toggle && !on ? "Turn thinking on to pick an effort level" : "Reasoning effort (sent as reasoning_effort)"}
          onChange={(e) => onChange(thinking ?? null, e.target.value || null)}>
          <option value="">{`Effort: default${c.effort_default ? ` (${c.effort_default})` : ""}`}</option>
          {ui.effort.map((l) => <option key={l} value={l}>{`Effort: ${l}`}</option>)}
        </select>
      )}
      {!compact && thinking != null && ui.toggle && (
        <button className="btn ghost sm" onClick={() => onChange(null, effort ?? null)} title="Use the model's default">Default</button>
      )}
    </div>
  );
}

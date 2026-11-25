// Model load progress: an overall bar with its percentage, then one row per
// device (This PC, each linked PC) with its own bar, "x / y GB", speed and,
// for linked PCs, the time left. Fed by the engine's `load_progress` events.
import { useEffect, useRef, useState } from "react";
import { useEngine, useLoadProgress } from "../lib/engine";
import type { LoadProgress } from "../lib/api";
import { DeviceLoadRow, LinkedLoadRows } from "./LinkedPcsLoad";
import { ProgressBar, etaText } from "./ProgressBar";

const PHASE: Record<LoadProgress["phase"], string> = {
  starting: "Starting the backend",
  loading: "Loading the model",
  finishing: "Weights are in place; finishing up",
  done: "Loaded",
  failed: "Load failed",
};

/** Words a screen reader hears when the phase changes (not on every tick). */
function usePhaseAnnouncement(p: LoadProgress | null): string {
  const [text, setText] = useState("");
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (!p || last.current === p.phase) return;
    last.current = p.phase;
    setText(p.phase === "failed" ? `Load failed. ${p.error ?? ""}` : PHASE[p.phase]);
  }, [p]);
  useEffect(() => {
    if (!p) last.current = null;
  }, [p]);
  return text;
}

/** The full view: overall bar and a row per device. Shows while a model is loading (and its failure). */
export function LoadProgressView() {
  const { status } = useEngine();
  const p = useLoadProgress();
  const said = usePhaseAnnouncement(p);
  const loading = status?.state === "loading";
  const failed = status?.state === "failed" && p?.phase === "failed";
  if (!loading && !failed) return null;
  // Before the first event arrives: an empty bar, not a layout jump.
  const pct = p?.percent ?? 0;
  const eta = failed ? null : etaText(p?.eta_secs);
  const phase = p ? PHASE[p.phase] : PHASE.starting;
  const local = p?.devices.find((d) => !d.linked);
  return (
    <div className={`load-progress${failed ? " failed" : ""}`} data-testid="load-progress">
      <div className="load-dev-head">
        <strong>{phase}</strong>
        <span className="mono">{Math.round(pct)}%{eta ? ` · ${eta}` : ""}</span>
      </div>
      <ProgressBar percent={pct} label="Model load progress" valueText={`${Math.round(pct)} percent${eta ? `, ${eta}` : ""}. ${phase}`} />
      {p && p.devices.length > 0 && (
        <div className="load-devices">
          {local && <DeviceLoadRow d={local} failed={failed} />}
          <LinkedLoadRows devices={p.devices} failed={failed} />
        </div>
      )}
      <span className="sr-only" role="status" aria-live="polite">{said}</span>
    </div>
  );
}

/** One line for the top bar and the My Models list: a thin bar, the percentage and the time left. */
export function LoadProgressMini() {
  const { status } = useEngine();
  const p = useLoadProgress();
  if (status?.state !== "loading") return null;
  const pct = p?.percent ?? 0;
  const eta = etaText(p?.eta_secs);
  const linked = p?.devices.filter((d) => d.linked) ?? [];
  const slowest = linked.length ? linked.reduce((a, b) => (b.percent < a.percent ? b : a)) : null;
  return (
    <span className="load-mini" title={slowest ? `${slowest.name}: ${Math.round(slowest.percent)}%` : undefined}>
      <ProgressBar percent={pct} label="Model load progress" valueText={`${Math.round(pct)} percent${eta ? `, ${eta}` : ""}`} />
      <span className="mono">{Math.round(pct)}%</span>
      {slowest && <span className="faint">{slowest.name} {Math.round(slowest.percent)}%</span>}
    </span>
  );
}

// "Use linked PCs" in the load panel: the network-cost estimate before a
// load, the per-PC load progress while loading, and the measured speed after
// one (with a suggestion, never an automatic change, when splitting is slower
// than this PC alone).
import { useEffect, useState } from "react";
import { api9, type SplitPreview } from "../lib/api9";
import type { DeviceProgress } from "../lib/api";
import { bytes, rate } from "../lib/format";
import { ProgressBar, etaText } from "./ProgressBar";

const tps = (v: number | null | undefined) => (v == null ? "?" : v.toFixed(1));

function usePreview(modelId: string, refreshKey: unknown, pollMs = 0) {
  const [p, setP] = useState<SplitPreview | null>(null);
  useEffect(() => {
    const get = () => api9.splitPreview(modelId).then(setP).catch(() => setP(null));
    get();
    if (!pollMs) return;
    const t = setInterval(() => { if (document.visibilityState === "visible") get(); }, pollMs);
    return () => clearInterval(t);
  }, [modelId, refreshKey, pollMs]);
  return p;
}

/** One device's bar while a model loads: This PC, or a linked PC. */
export function DeviceLoadRow({ d, failed }: { d: DeviceProgress; failed?: boolean }) {
  const pct = Math.round(d.percent);
  const meta: string[] = [];
  // A linked PC's share is known once the backend has assigned the layers.
  const known = !d.linked || d.total_bytes > d.done_bytes || d.percent > 0;
  meta.push(known ? `${bytes(d.done_bytes)} / ${bytes(d.total_bytes)}` : `${bytes(d.done_bytes)} sent`);
  if (d.cached && d.cached_bytes > 0) meta.push(`${bytes(d.cached_bytes)} from its cache`);
  if (d.bytes_per_sec) meta.push(rate(d.bytes_per_sec));
  // "Time left" is shown for linked PCs; the overall bar carries This PC's.
  if (d.linked) {
    const eta = etaText(d.eta_secs);
    if (eta) meta.push(eta);
  }
  return (
    <div className={`load-dev${failed ? " failed" : ""}`}>
      <div className="load-dev-head">
        <span>{d.name}{d.linked ? "" : " (this PC)"}</span>
        <span className="mono">{pct}%</span>
      </div>
      <ProgressBar percent={d.percent} label={`${d.name} load progress`} valueText={`${pct} percent. ${meta.join(", ")}`} />
      <div className="faint load-dev-meta">{meta.join(" · ")}</div>
    </div>
  );
}

/** The linked PCs' rows (how much of the model is on each one so far). */
export function LinkedLoadRows({ devices, failed }: { devices: DeviceProgress[]; failed?: boolean }) {
  const linked = devices.filter((d) => d.linked);
  if (!linked.length) return null;
  return (
    <>
      {linked.map((d) => <DeviceLoadRow key={d.id} d={d} failed={failed} />)}
    </>
  );
}

export function LinkedPcsOption({ modelId, checked, disabled, cpuOnly, onChange }: {
  modelId: string; checked: boolean; disabled: boolean; cpuOnly: boolean; onChange: (on: boolean) => void;
}) {
  const p = usePreview(modelId, checked);
  if (!p || (p.peers.length === 0 && !checked)) return null;
  const names = p.peers.map((x) => x.name).join(", ");
  return (
    <div className="wide link-load">
      <label className="check">
        <input type="checkbox" checked={checked} disabled={disabled || (cpuOnly && !checked)} onChange={(e) => onChange(e.target.checked)} />
        Use linked PCs: split the model's layers with {names || "linked PCs"} (by free memory)
      </label>
      {cpuOnly && <div className="faint link-small">Needs GPU layers: with CPU only every layer stays on this PC.</div>}
      {p.peers.length === 0 && checked && <div className="notice warn">No linked PC has "Split model layers with this PC" on (Linked PCs page).</div>}
      {p.peers.map((x) => (
        <div key={x.peer_id} className="faint link-small">
          {x.name}: {x.rtt_ms == null ? "round trip not measured yet (connect on the Linked PCs page)" : `${x.rtt_ms.toFixed(1)} ms round trip, adds about ${x.added_ms_per_token?.toFixed(1)} ms per generated token`}
        </div>
      ))}
      <SpeedLines p={p} />
      <div className="faint link-small">Helps most when the model doesn't fit on this PC. Every generated token waits for one round trip, so models that fit usually run slower split.</div>
    </div>
  );
}

const range = (p: SplitPreview) =>
  p.est_low_tok_s != null && p.est_high_tok_s != null ? `${tps(p.est_low_tok_s)}-${tps(p.est_high_tok_s)} tokens/s` : null;

/** Measured speeds first; the estimate (a range, labelled) when nothing is measured yet. */
function SpeedLines({ p }: { p: SplitPreview }) {
  const est = range(p);
  return (
    <>
      {p.split_tok_s != null && <div className="link-small">Measured with linked PCs: {tps(p.split_tok_s)} tokens/s{p.alone_tok_s != null && ` (this PC alone, measured: ${tps(p.alone_tok_s)})`}</div>}
      {est && (
        <div className="faint link-small" title={p.est_basis.join("; ")}>
          Estimate with linked PCs: about {est}{p.split_tok_s == null && p.alone_tok_s != null ? ` (this PC alone, measured: ${tps(p.alone_tok_s)})` : ""}
          <span className="sr-only"> Based on: {p.est_basis.join("; ")}</span>
        </div>
      )}
      {!est && p.est_missing && <div className="faint link-small">No estimate: {p.est_missing}.</div>}
    </>
  );
}

/** After a load with linked PCs: measured speed next to this PC alone. */
export function LinkedPcsResult({ modelId, tensorSplit, refreshKey }: { modelId: string; tensorSplit: string | null | undefined; refreshKey: unknown }) {
  const p = usePreview(modelId, refreshKey, 10000);
  return (
    <>
      <div className="row"><span className="dim">Split with linked PCs</span><span className="mono">--tensor-split {tensorSplit ?? "?"}</span></div>
      {p && (
        <div className="faint">
          {p.split_tok_s != null ? `Measured with linked PCs: ${tps(p.split_tok_s)} tokens/s over ${p.split_replies} ${p.split_replies === 1 ? "reply" : "replies"}` : "Speed is measured from the next replies."}
          {p.alone_tok_s != null && ` · this PC alone: ${tps(p.alone_tok_s)} tokens/s`}
        </div>
      )}
      {p?.suggest_off && p.message && <div className="notice warn">{p.message}</div>}
    </>
  );
}

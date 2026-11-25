import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api, errorMessage, type HardwareInfo, type InstalledRuntime, type ModelView, type ResourceSnapshot } from "../lib/api";
import { bytes, percent } from "../lib/format";
import { useEngine } from "../lib/engine";
import type { PageId } from "../App";

function useResources(intervalMs = 2000) {
  const [snap, setSnap] = useState<ResourceSnapshot | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api.getResources().then((s) => alive && setSnap(s)).catch(() => {});
    tick();
    const t = setInterval(() => {
      if (document.visibilityState === "visible") tick();
    }, intervalMs);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [intervalMs]);
  return snap;
}

export function Overview({ go }: { go: (p: PageId) => void }) {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [runtimes, setRuntimes] = useState<InstalledRuntime[] | null>(null);
  const [models, setModels] = useState<ModelView[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const snap = useResources();
  const { status } = useEngine();

  useEffect(() => {
    api.getHardware().then(setHw).catch((e) => setErr(errorMessage(e)));
  }, []);

  // The next-step banner follows runtimes and models installed while this
  // page is open (first-run setup installs a runtime over it).
  useEffect(() => {
    const refresh = () => {
      api.listRuntimes().then(setRuntimes).catch(() => setRuntimes([]));
      api.listModels().then(setModels).catch(() => setModels([]));
    };
    refresh();
    let t: ReturnType<typeof setTimeout> | null = null;
    const later = () => {
      if (t) clearTimeout(t);
      t = setTimeout(refresh, 500);
    };
    const un = listen("runtime://install-progress", later);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      if (t) clearTimeout(t);
      un.then((f) => f());
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const gpu = hw?.gpus[0];
  const gpuLive = snap?.gpus[0];
  const next =
    runtimes && runtimes.length === 0
      ? { text: "Install an inference runtime to run models.", page: "runtimes" as PageId, cta: "Open Runtimes" }
      : models && models.length === 0
        ? { text: "Import a GGUF model file or folder.", page: "models" as PageId, cta: "Open My Models" }
        : status?.state !== "loaded"
          ? { text: "Pick a model and load it.", page: "models" as PageId, cta: "Open My Models" }
          : null;

  return (
    <div className="page">
      {err && <div className="notice error">Hardware detection failed: {err}</div>}
      {next && (
        <div className="notice warn row" style={{ marginBottom: 20 }}>
          <span>{next.text}</span>
          <span className="spacer" />
          <button className="btn primary sm" onClick={() => go(next.page)}>{next.cta}</button>
        </div>
      )}

      <h2>Live usage</h2>
      <div className="stat-row">
        <div className="stat">
          <div className="stat-label">CPU</div>
          <div className="stat-value">{snap ? `${snap.cpu_usage_percent.toFixed(0)}%` : "..."}</div>
          <div className="meter"><span style={{ width: `${snap?.cpu_usage_percent ?? 0}%` }} /></div>
        </div>
        <div className="stat">
          <div className="stat-label">System RAM</div>
          <div className="stat-value">{snap ? bytes(snap.ram_used_bytes) : "..."}</div>
          <div className="stat-sub">of {bytes(snap?.ram_total_bytes)}</div>
          <div className="meter"><span style={{ width: `${percent(snap?.ram_used_bytes ?? 0, snap?.ram_total_bytes ?? 1)}%` }} /></div>
        </div>
        <div className="stat">
          <div className="stat-label">GPU memory</div>
          <div className="stat-value">{gpuLive ? bytes(gpuLive.vram_used_bytes) : gpu ? "..." : "n/a"}</div>
          <div className="stat-sub">
            {gpuLive ? `of ${bytes(gpuLive.vram_total_bytes)}` : "No NVIDIA GPU metrics"}
            {gpuLive?.utilization_percent != null && ` · ${gpuLive.utilization_percent}% busy`}
          </div>
          <div className="meter hot"><span style={{ width: `${percent(gpuLive?.vram_used_bytes ?? 0, gpuLive?.vram_total_bytes ?? 1)}%` }} /></div>
        </div>
        <div className="stat">
          <div className="stat-label">Memory by process</div>
          <div className="stat-value">{snap ? bytes(snap.app_memory_bytes) : "..."}</div>
          <div className="stat-sub">
            this app{snap?.backend_memory_bytes != null ? `, backend ${bytes(snap.backend_memory_bytes)} RAM` : ", no backend running"}
          </div>
        </div>
      </div>

      <h2>This PC</h2>
      {!hw ? (
        <div style={{ display: "grid", gap: 8, maxWidth: 420 }}>
          <div className="skeleton" /><div className="skeleton" /><div className="skeleton" />
        </div>
      ) : (
        <dl className="kv">
          <dt>Operating system</dt><dd>{hw.os}</dd>
          <dt>CPU</dt>
          <dd>{hw.cpu_name}, {hw.physical_cores ?? "?"} cores / {hw.logical_cores} threads</dd>
          <dt>RAM</dt><dd>{bytes(hw.total_ram_bytes)} total, {bytes(hw.available_ram_bytes)} available at detection</dd>
          {hw.gpus.map((g) => (
            <div key={g.index} style={{ display: "contents" }}>
              <dt>GPU {g.index}</dt>
              <dd>
                {g.name}, {bytes(g.vram_total_bytes)} VRAM, driver {g.driver_version ?? "?"}, compute capability {g.compute_capability ?? "?"}
              </dd>
            </div>
          ))}
          {hw.cuda_driver_version && (<><dt>CUDA (driver)</dt><dd>up to {hw.cuda_driver_version}</dd></>)}
          {hw.gpu_note && (<><dt>GPU note</dt><dd>{hw.gpu_note}</dd></>)}
        </dl>
      )}
      <p className="faint" style={{ marginTop: 14 }}>
        GPU figures come from NVIDIA NVML. The Windows driver does not report VRAM per process, so backend VRAM is shown as total GPU memory in use.
      </p>
    </div>
  );
}

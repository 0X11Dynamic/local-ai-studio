import { useEffect, useMemo, useState } from "react";
import { api, errorMessage, type EstimateView, type GpuLayers, type InstalledRuntime, type LoadConfig, type LoadDefaults, type ModelView } from "../lib/api";
import { bytes } from "../lib/format";
import { useEngine } from "../lib/engine";

type GpuMode = GpuLayers["mode"];

function numOrNull(v: string): number | null {
  const n = Number(v);
  return v.trim() === "" || !Number.isFinite(n) ? null : Math.round(n);
}

function floatOrNull(v: string): number | null {
  const n = Number(v);
  return v.trim() === "" || !Number.isFinite(n) ? null : n;
}

export function LoadPanel({ model, runtimes }: { model: ModelView; runtimes: InstalledRuntime[] }) {
  const { status } = useEngine();
  const [defaults, setDefaults] = useState<LoadDefaults | null>(null);
  const [cfg, setCfg] = useState<LoadConfig | null>(null);
  const [runtimeId, setRuntimeId] = useState<string>("");
  const [est, setEst] = useState<EstimateView | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [advanced, setAdvanced] = useState(false);
  // Context: Auto (null; n_ctx 0, sized to free memory at load) or the typed number.
  const [ctxText, setCtxText] = useState<string | null>(null);

  const applyDefaults = (d: LoadDefaults) => {
    setDefaults(d);
    setCfg(d.config);
    setCtxText(d.config.n_ctx === 0 ? null : String(d.config.n_ctx));
  };

  useEffect(() => {
    setErr(null);
    setSaved(null);
    api.getLoadDefaults(model.id).then(applyDefaults);
  }, [model.id]);

  useEffect(() => {
    api.getSettings().then((s) => {
      const d = (s["runtime.default"] as string) ?? runtimes[0]?.id ?? "";
      setRuntimeId(runtimes.some((r) => r.id === d) ? d : runtimes[0]?.id ?? "");
    });
  }, [runtimes]);

  const runtime = runtimes.find((r) => r.id === runtimeId);
  // A CPU build has no GPU backend: offload settings do nothing there.
  const cpuRuntime = runtime?.variant === "cpu";

  // Re-estimate (debounced) whenever the settings or runtime change.
  useEffect(() => {
    if (!cfg) return;
    const effective: LoadConfig = cpuRuntime ? { ...cfg, gpu_layers: { mode: "cpu_only" } } : cfg;
    const t = setTimeout(() => {
      api.estimateMemory(model.id, effective).then(setEst).catch(() => setEst(null));
    }, 200);
    return () => clearTimeout(t);
  }, [cfg, model.id, cpuRuntime]);

  const has = (flag: string) => !runtime || runtime.capabilities.flags.includes(flag);
  const cacheTypes = runtime?.capabilities.cache_types.length ? runtime.capabilities.cache_types : ["f16", "q8_0", "q4_0"];
  const isThis = status?.model_id === model.id;
  const loading = isThis && status?.state === "loading";
  const loaded = isThis && status?.state === "loaded";
  const anotherLoaded = !isThis && (status?.state === "loaded" || status?.state === "loading");
  const totalLayers = defaults?.total_layers ?? null;

  const ctxError = useMemo(() => {
    if (ctxText === null) return null;
    const n = numOrNull(ctxText.replace(/[,\s]/g, ""));
    if (n === null || n < 256) return "Enter at least 256 tokens, or choose Auto.";
    return null;
  }, [ctxText]);

  if (!cfg) return <div className="skeleton" />;

  const set = (patch: Partial<LoadConfig>) => {
    setSaved(null);
    setCfg({ ...cfg, ...patch });
  };
  const gpuMode = cfg.gpu_layers.mode;
  const gpuCount = cfg.gpu_layers.mode === "count" ? cfg.gpu_layers.count : totalLayers ?? 0;
  const setGpu = (mode: GpuMode, count = gpuCount) =>
    set({ gpu_layers: mode === "count" ? { mode, count } : ({ mode } as GpuLayers) });

  const load = async () => {
    setErr(null);
    try {
      await api.loadModel(model.id, cfg, runtimeId || null);
    } catch (e) {
      const msg = errorMessage(e);
      if (msg !== "cancelled") setErr(msg);
    }
  };
  const unload = async () => {
    setErr(null);
    await api.unloadModel();
  };
  const saveDefault = async () => {
    try {
      await api.saveModelDefaults(model.id, cfg);
      setSaved("Saved as this model's default.");
    } catch (e) {
      setErr(errorMessage(e));
    }
  };
  const reset = async () => {
    await api.resetModelDefaults(model.id);
    applyDefaults(await api.getLoadDefaults(model.id));
    setSaved("Model default cleared.");
  };

  const disabled = loading || loaded;
  // n_ctx 0 = Auto: the app picks the largest size that fits (resolved again at load).
  const autoCtx = ctxText === null;
  const setCtxMode = (mode: string) => {
    if (mode === "auto") {
      setCtxText(null);
      set({ n_ctx: 0 });
    } else {
      const n = est?.auto_ctx.n_ctx ?? Math.min(8192, defaults?.trained_context ?? 8192);
      setCtxText(String(n));
      set({ n_ctx: n });
    }
  };
  const typeCtx = (v: string) => {
    setCtxText(v);
    const n = numOrNull(v.replace(/[,\s]/g, ""));
    // An invalid entry keeps the last valid size (the error blocks loading).
    if (n !== null && n >= 256) set({ n_ctx: n });
    else if (autoCtx) set({ n_ctx: est?.auto_ctx.n_ctx ?? 8192 });
  };
  const ctxTokens = autoCtx ? est?.auto_ctx.n_ctx ?? 0 : cfg.n_ctx;
  const sharing = !!cfg.share_with_cpu && !cpuRuntime;
  // Suggest only: the toggle is never switched on for the user.
  const suggestShare = !cpuRuntime && !cfg.share_with_cpu && !!est?.suggest_share;
  const offloaded = status?.gpu_layers_offloaded;
  const layersLine = offloaded
    ? offloaded[0] < offloaded[1]
      ? `${offloaded[0]} of ${offloaded[1]} layers on GPU, the rest on CPU`
      : `${offloaded[0]}/${offloaded[1]} (all on GPU)`
    : "not reported";
  const moeOnCpu = isThis ? status?.config?.n_cpu_moe : null;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="row">
        <strong style={{ fontWeight: 600 }}>Load settings</strong>
        <span className="faint">
          {defaults?.source === "model_default" ? "from this model's saved default" : defaults?.source === "last_used" ? "from the last load" : "app defaults"}
        </span>
      </div>

      <div className="form-grid">
        <div className="field wide">
          <label htmlFor="rt">Runtime</label>
          <select id="rt" className="input" value={runtimeId} onChange={(e) => setRuntimeId(e.target.value)} disabled={disabled || runtimes.length === 0}>
            {runtimes.length === 0 && <option value="">None installed</option>}
            {runtimes.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
          </select>
        </div>
        <div className="field ctx-field">
          <label htmlFor="ctx-mode">Context length</label>
          <div className="row ctx-row">
            <select id="ctx-mode" className="input" value={autoCtx ? "auto" : "manual"} disabled={disabled}
              onChange={(e) => setCtxMode(e.target.value)}>
              <option value="auto">Auto (fits your hardware)</option>
              <option value="manual">Manual</option>
            </select>
            <input id="ctx" className="input mono" inputMode="numeric" disabled={disabled}
              aria-label={autoCtx ? "Context length in tokens (type a number to set it manually)" : "Context length in tokens"}
              value={ctxText ?? ""} placeholder={autoCtx && est ? est.auto_ctx.n_ctx.toLocaleString() : ""}
              aria-invalid={!!ctxError} aria-describedby="ctx-help"
              onChange={(e) => typeCtx(e.target.value)} />
          </div>
          {ctxError ? (
            <span className="err" id="ctx-help">{ctxError}</span>
          ) : autoCtx ? (
            <span className="help" id="ctx-help" aria-live="polite">
              {est ? <><span className={est.auto_ctx.fits ? "ctx-auto-value" : "ctx-auto-value warn"}>Auto: {est.auto_ctx.n_ctx.toLocaleString()} tokens.</span> {est.auto_ctx.summary} Measured again when the model loads.</> : "Auto: picked from free memory when the model loads."}
            </span>
          ) : (
            <span className="help" id="ctx-help">Trained: {defaults?.trained_context?.toLocaleString() ?? "?"}.{est ? ` Auto would pick ${est.auto_ctx.n_ctx.toLocaleString()}.` : ""}</span>
          )}
        </div>
        <div className="field">
          <label htmlFor="gpu">GPU offload</label>
          {sharing ? (
            <>
              <input id="gpu" className="input" value="Chosen automatically" disabled readOnly />
              <span className="help">{est?.share_split ? `Estimate: ${est.share_split.summary}` : "Picked from free VRAM when the model loads."}</span>
            </>
          ) : (
            <select id="gpu" className="input" value={cpuRuntime ? "cpu_only" : gpuMode} disabled={disabled || cpuRuntime || !has("--n-gpu-layers")} onChange={(e) => setGpu(e.target.value as GpuMode)}>
              <option value="auto">Auto (fit to memory)</option>
              <option value="all">All layers</option>
              <option value="count">Choose layers</option>
              <option value="cpu_only">CPU only</option>
            </select>
          )}
          {cpuRuntime && <span className="help">This runtime is CPU-only.</span>}
          {!cpuRuntime && !sharing && gpuMode === "count" && (
            <div className="row">
              <input type="range" min={0} max={totalLayers ?? 100} value={gpuCount} disabled={disabled} aria-label="GPU layers"
                onChange={(e) => setGpu("count", Number(e.target.value))} style={{ flex: 1, accentColor: "var(--accent)" }} />
              <span className="mono">{gpuCount}/{totalLayers ?? "?"}</span>
            </div>
          )}
        </div>
        <div className="field">
          <label htmlFor="fa">Flash Attention</label>
          <select id="fa" className="input" value={cfg.flash_attention} disabled={disabled || !has("--flash-attn")}
            onChange={(e) => set({ flash_attention: e.target.value as LoadConfig["flash_attention"] })}>
            <option value="auto">Auto</option><option value="on">On</option><option value="off">Off</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="kvk">KV cache type (K / V)</label>
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <select id="kvk" className="input" style={{ flex: 1 }} value={cfg.cache_type_k} disabled={disabled || !has("--cache-type-k")} aria-label="K cache type"
              onChange={(e) => set({ cache_type_k: e.target.value })}>
              {cacheTypes.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select className="input" style={{ flex: 1 }} value={cfg.cache_type_v} disabled={disabled || !has("--cache-type-v")} aria-label="V cache type"
              onChange={(e) => set({ cache_type_v: e.target.value })}>
              {cacheTypes.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          {cfg.cache_type_v !== "f16" && cfg.flash_attention === "off" && <span className="err">A quantized V cache needs Flash Attention.</span>}
        </div>
        <label className="check wide">
          <input type="checkbox" checked={sharing} disabled={disabled || cpuRuntime || !has("--n-gpu-layers")}
            onChange={(e) => set({ share_with_cpu: e.target.checked })} />
          Share with CPU: put as much as fits on the GPU and run the rest on the CPU
        </label>
        {suggestShare && (
          <div className="notice warn wide share-suggest">
            This model won't fit in free VRAM ({est?.share_split?.summary}). Turn on Share with CPU to split it between GPU and CPU.
          </div>
        )}
        <label className="check wide">
          <input type="checkbox" checked={cfg.kv_offload && !cpuRuntime} disabled={disabled || cpuRuntime || !has("--no-kv-offload")} onChange={(e) => set({ kv_offload: e.target.checked })} />
          Keep the KV cache on the GPU
        </label>
        {defaults?.has_projector && (
          <label className="check wide">
            <input type="checkbox" checked={cfg.load_projector} disabled={disabled} onChange={(e) => set({ load_projector: e.target.checked })} />
            Load the vision projector (image input)
          </label>
        )}
      </div>

      <button className="btn ghost sm" style={{ justifySelf: "start" }} onClick={() => setAdvanced((a) => !a)} aria-expanded={advanced}>
        {advanced ? "Hide advanced settings" : "Advanced settings"}
      </button>
      {advanced && (
        <div className="form-grid">
          <div className="field">
            <label htmlFor="thr">CPU threads</label>
            <input id="thr" className="input mono" placeholder="auto" value={cfg.threads ?? ""} disabled={disabled || !has("--threads")}
              onChange={(e) => set({ threads: numOrNull(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="thb">Prompt threads</label>
            <input id="thb" className="input mono" placeholder="same as threads" value={cfg.threads_batch ?? ""} disabled={disabled || !has("--threads-batch")}
              onChange={(e) => set({ threads_batch: numOrNull(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="bs">Batch size</label>
            <input id="bs" className="input mono" placeholder="2048" value={cfg.batch_size ?? ""} disabled={disabled || !has("--batch-size")}
              onChange={(e) => set({ batch_size: numOrNull(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="ubs">Micro-batch size</label>
            <input id="ubs" className="input mono" placeholder="512" value={cfg.ubatch_size ?? ""} disabled={disabled || !has("--ubatch-size")}
              onChange={(e) => set({ ubatch_size: numOrNull(e.target.value) })} />
          </div>
          <div className="field">
            <label htmlFor="lm">Model loading</label>
            <select id="lm" className="input" value={cfg.load_mode} disabled={disabled}
              onChange={(e) => set({ load_mode: e.target.value as LoadConfig["load_mode"] })}>
              <option value="auto">Auto (memory-map)</option>
              <option value="mmap">Memory-map</option>
              <option value="mlock">Lock in RAM</option>
              <option value="mmap_mlock">Memory-map and lock</option>
              <option value="none">Read fully (no mmap)</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="np">Parallel slots</label>
            <input id="np" className="input mono" value={cfg.parallel} disabled={disabled || !has("--parallel")}
              onChange={(e) => set({ parallel: Math.max(1, numOrNull(e.target.value) ?? 1) })} />
            {cfg.parallel > 1 && <span className="help">Slots share the context: each gets about {Math.floor(ctxTokens / cfg.parallel).toLocaleString()} tokens.</span>}
          </div>
          <div className="field">
            <label htmlFor="rs">RoPE scaling</label>
            <select id="rs" className="input" value={cfg.rope_scaling ?? ""} disabled={disabled || !has("--rope-scaling")}
              onChange={(e) => set({ rope_scaling: e.target.value || null })}>
              <option value="">Model default</option><option value="none">None</option><option value="linear">Linear</option><option value="yarn">YaRN</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="rfb">RoPE base frequency</label>
            <input id="rfb" className="input mono" placeholder="model default" value={cfg.rope_freq_base ?? ""} disabled={disabled || !has("--rope-freq-base")}
              onChange={(e) => set({ rope_freq_base: floatOrNull(e.target.value) })} />
          </div>
          <label className="check wide">
            <input type="checkbox" checked={cfg.fit} disabled={disabled || !has("--fit")} onChange={(e) => set({ fit: e.target.checked })} />
            Let llama.cpp shrink settings left on Auto so the model fits in memory
          </label>
        </div>
      )}

      {est && (
        <div className="panel" style={{ background: "var(--bg-sunken)", display: "grid", gap: 6 }}>
          <div className="row"><strong style={{ fontWeight: 500 }}>Estimated memory</strong><span className="faint">estimate, measured after load</span></div>
          <dl className="kv">
            <dt>GPU (VRAM)</dt>
            <dd>{bytes(sharing && est.share_split ? est.share_split.vram_bytes : est.estimate.vram_bytes)}{est.free_vram_bytes != null && ` of ${bytes(est.free_vram_bytes)} free`}</dd>
            {!sharing && <><dt>System RAM</dt><dd>{bytes(est.estimate.ram_bytes)} of {bytes(est.available_ram_bytes)} available</dd></>}
            <dt>KV cache</dt><dd>{bytes(est.estimate.kv_cache_bytes)}</dd>
            {sharing && est.share_split ? (
              <><dt>Split</dt><dd>{est.share_split.summary}</dd></>
            ) : (
              <><dt>Layers on GPU</dt><dd>{est.estimate.offloaded_layers}/{est.estimate.total_layers}{est.suggested_gpu_layers != null && `, about ${est.suggested_gpu_layers} fit`}</dd></>
            )}
          </dl>
          {est.warnings.map((w) => <div key={w} className="notice warn">{w}</div>)}
        </div>
      )}

      {isThis && status?.unsupported_settings.length ? (
        <div className="notice warn">Not applied by this runtime: {status.unsupported_settings.join("; ")}</div>
      ) : null}
      {err && <div className="notice error">{err}{isThis && status?.error_hint ? ` ${status.error_hint}` : ""}</div>}
      {anotherLoaded && <div className="notice">Another model is loaded. Loading this one unloads it first.</div>}

      <div className="row">
        {loaded ? (
          <button className="btn" onClick={unload}>Unload</button>
        ) : loading ? (
          <button className="btn" onClick={() => api.cancelLoad()}>Cancel load</button>
        ) : (
          <button className="btn primary" onClick={load} disabled={!runtimeId || !!ctxError}>Load model</button>
        )}
        <span className="spacer" />
        <button className="btn sm ghost" onClick={saveDefault} disabled={disabled}>Save as default</button>
        {defaults?.source === "model_default" && <button className="btn sm ghost" onClick={reset} disabled={disabled}>Reset</button>}
      </div>
      {saved && <div className="faint">{saved}</div>}

      {loaded && status?.warnings.map((w) => <div key={w} className="notice warn">{w}</div>)}
      {loaded && status && (
        <div className="panel" style={{ background: "var(--bg-sunken)" }}>
          <div style={{ fontWeight: 500, marginBottom: 6 }}>Running (measured)</div>
          <dl className="kv">
            <dt>Context allocated</dt><dd>{status.props?.n_ctx?.toLocaleString() ?? "?"} tokens</dd>
            <dt>Layers on GPU</dt><dd>{layersLine}</dd>
            {moeOnCpu ? <><dt>MoE experts on CPU</dt><dd>first {moeOnCpu} layers</dd></> : null}
            {status.buffers.map((b) => (
              <div key={`${b.device}-${b.kind}`} style={{ display: "contents" }}>
                <dt>{b.device} {b.kind}</dt><dd>{b.mib.toFixed(0)} MiB</dd>
              </div>
            ))}
            {status.devices.map((d) => (
              <div key={d.device} style={{ display: "contents" }}>
                <dt>{d.device} free at start</dt><dd>{d.free_mib.toFixed(0)} of {d.total_mib.toFixed(0)} MiB</dd>
              </div>
            ))}
            <dt>Load time</dt><dd>{status.load_seconds?.toFixed(1)} s</dd>
            <dt>Process</dt><dd>pid {status.pid}, port {status.port}</dd>
          </dl>
        </div>
      )}
      {isThis && status?.state === "failed" && !err && (
        <div className="notice error">{status.error}{status.error_hint ? ` ${status.error_hint}` : ""}</div>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { api, errorMessage, type EstimateView, type GpuLayers, type InstalledRuntime, type LoadConfig, type LoadDefaults, type ModelView } from "../lib/api";
import { bytes } from "../lib/format";
import { useEngine } from "../lib/engine";
import { LinkedPcsOption, LinkedPcsResult } from "./LinkedPcsLoad";
import { LoadProgressView } from "./LoadProgressView";

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
      api.estimateMemory(model.id, effective, runtimeId || null).then(setEst).catch(() => setEst(null));
    }, 200);
    return () => clearTimeout(t);
  }, [cfg, model.id, cpuRuntime, runtimeId]);

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
  const fitMargin = est?.fit_target_mib ?? cfg.fit_target_mib ?? 1024;
  // With --fit, llama.cpp picks the layer count at load; the split shown is the app's estimate.
  const split = est?.share_split;
  const shareHelp = !split
    ? "Picked from free VRAM when the model loads."
    : est?.share_fit && !split.all_fit && split.n_cpu_moe == null && split.gpu_layers > 0
      ? `Estimate: about ${split.gpu_layers} of ${split.total_layers} layers on the GPU. llama.cpp fits the exact number when the model loads, keeping ${fitMargin} MiB free.`
      : est?.share_fit && split.all_fit
        ? `Estimate: all ${split.total_layers} layers fit, with little VRAM to spare, so llama.cpp fits the exact number when the model loads, keeping ${fitMargin} MiB free.`
        : `Estimate: ${split.summary}`;
  const kvAuto = !!cfg.cache_type_auto;
  const setKv = (which: "k" | "v", t: string) => {
    if (t === "auto") return set({ cache_type_auto: true, cache_type_k: "f16", cache_type_v: "f16" });
    const k = which === "k" ? t : kvAuto ? "f16" : cfg.cache_type_k;
    const v = which === "v" ? t : kvAuto ? "f16" : cfg.cache_type_v;
    set({ cache_type_auto: false, cache_type_k: k, cache_type_v: v });
  };
  const autoKvLine = kvAuto && est?.auto_kv
    ? `${est.auto_kv.summary}.${est.auto_kv.flash_attention_on ? " Flash Attention is turned on for the q8_0 V cache." : ""}`
    : null;
  // Speculative decoding with the model's own MTP layers: offered only when
  // the model has them and the runtime lists draft-mtp.
  const mtpAvailable = (model.meta.mtp_layers ?? 0) > 0 && !!runtime?.capabilities.spec_types?.includes("draft-mtp");
  const mtpStripped = !mtpAvailable && (model.meta.mtp_layers ?? 0) === 0 && (model.meta.mtp_layers_declared ?? 0) > 0 && !model.meta.mtp_unknown;
  const running = isThis ? status?.config : null;

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
              {est ? <><span className={est.auto_ctx.fits && !est.auto_ctx.pages_weights ? "ctx-auto-value" : "ctx-auto-value warn"}>Auto: {est.auto_ctx.n_ctx.toLocaleString()} tokens{est.auto_ctx_alone != null ? ` (${est.auto_ctx_alone.toLocaleString()} on this PC alone)` : ""}.</span> {linkedLine(est.linked_contrib)}{est.auto_ctx.summary} Measured again when the model loads.</> : "Auto: picked from free memory when the model loads."}
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
              <span className="help">{shareHelp}</span>
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
            <select id="kvk" className="input" style={{ flex: 1 }} value={kvAuto ? "auto" : cfg.cache_type_k} disabled={disabled || !has("--cache-type-k")} aria-label="K cache type"
              onChange={(e) => setKv("k", e.target.value)}>
              <option value="auto">Auto</option>
              {cacheTypes.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select className="input" style={{ flex: 1 }} value={kvAuto ? "auto" : cfg.cache_type_v} disabled={disabled || !has("--cache-type-v")} aria-label="V cache type"
              onChange={(e) => setKv("v", e.target.value)}>
              <option value="auto">Auto</option>
              {cacheTypes.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          {autoKvLine && <span className="help">{autoKvLine}</span>}
          {!kvAuto && cfg.cache_type_v !== "f16" && cfg.flash_attention === "off" && <span className="err">A quantized V cache needs Flash Attention.</span>}
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
        <LinkedPcsOption modelId={model.id} checked={!!cfg.use_linked_pcs} disabled={disabled}
          cpuOnly={cpuRuntime || cfg.gpu_layers.mode === "cpu_only"} onChange={(on) => set({ use_linked_pcs: on })} />
        <label className="check wide">
          <input type="checkbox" checked={cfg.kv_offload && !cpuRuntime} disabled={disabled || cpuRuntime || !has("--no-kv-offload")} onChange={(e) => set({ kv_offload: e.target.checked })} />
          Keep the KV cache on the GPU
        </label>
        {mtpAvailable && (
          <label className="check wide">
            <input type="checkbox" checked={cfg.speculative === "draft-mtp"} disabled={disabled}
              onChange={(e) => set({ speculative: e.target.checked ? "draft-mtp" : null, speculative_auto: false })} />
            Speculative decoding (MTP): draft tokens with the model's own prediction layers. Faster generation (measured +80% on Qwen3.8-27B); output can differ slightly at near-ties.
          </label>
        )}
        {mtpStripped && (
          <div className="faint wide">Speculative decoding (MTP) is unavailable: this file declares MTP but its MTP layers were removed.</div>
        )}
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
          {has("--fit-target") && (
            <div className="field">
              <label htmlFor="fitt">GPU memory margin for fitting (MiB)</label>
              <input id="fitt" className="input mono" placeholder="1024" value={cfg.fit_target_mib ?? ""} disabled={disabled || !cfg.fit}
                onChange={(e) => set({ fit_target_mib: numOrNull(e.target.value) })} />
              <span className="help">VRAM llama.cpp leaves free when it fits the layers (Share with CPU, GPU offload Auto). Raise it if a load spills into shared memory.</span>
            </div>
          )}
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
              <><dt>Split (estimate)</dt><dd>{shareHelp.replace(/^Estimate: /, "")}</dd></>
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
      {isThis && <LoadProgressView />}

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
            <dt>Layers on GPU</dt><dd>{layersLine}{running?.gpu_layers.mode === "auto" && running.fit ? " (fitted by llama.cpp)" : ""}</dd>
            {moeOnCpu ? <><dt>MoE experts on CPU</dt><dd>first {moeOnCpu} layers</dd></> : null}
            {running && <><dt>KV cache</dt><dd>{running.cache_type_k} / {running.cache_type_v}{running.cache_type_auto ? " (auto)" : ""}</dd></>}
            {running?.threads ? <><dt>CPU threads</dt><dd>{running.threads} generation, {running.threads_batch ?? running.threads} prompt</dd></> : null}
            {running?.speculative ? <><dt>Speculative decoding</dt><dd>{running.speculative}</dd></> : null}
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
          {running?.rpc?.length ? (
            <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
              <LinkedPcsResult modelId={model.id} tensorSplit={running.tensor_split} refreshKey={status.instance_id} />
            </div>
          ) : null}
        </div>
      )}
      {isThis && status?.state === "failed" && !err && (
        <div className="notice error">{status.error}{status.error_hint ? ` ${status.error_hint}` : ""}</div>
      )}
    </div>
  );
}

/** "+6.0 GB GPU memory from Desk PC (its owner allows up to 8.0 GB)." per linked device. */
function linkedLine(c: { peer: string; device: string; bytes: number; cpu: boolean; cap_bytes: number | null }[]): string {
  return c
    .filter((d) => d.bytes > 0)
    .map((d) => `+${(d.bytes / 1e9).toFixed(1)} GB ${d.cpu ? "RAM" : "GPU memory"} from ${d.peer}${d.cap_bytes != null ? ` (its owner allows up to ${(d.cap_bytes / 1e9).toFixed(1)} GB)` : ""}. `)
    .join("");
}

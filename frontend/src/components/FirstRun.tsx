// First-run setup: shown by itself once on a fresh profile (never set up, no
// runtime installed), skippable, and re-opened from Settings via the
// "las:open-setup" window event. Hardware and the runtime recommendation come
// from the same code the Runtimes page uses; the runtime install and the model
// download reuse install_runtime and download_variant. A model is only
// downloaded when the user clicks Download.
import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { api, errorMessage, type AvailableRuntime, type InstallProgress, type InstalledRuntime } from "../lib/api";
import { api2, type DownloadEvent } from "../lib/api2";
import { api8, type FolderCheck } from "../lib/api8";
import { StorageFolder } from "./StorageFolder";
import { bytes, percent, rate } from "../lib/format";
import type { PageId } from "../App";

/** Fire this to open the setup screen (Settings does). */
export const OPEN_SETUP_EVENT = "las:open-setup";

interface SetupGpu { name: string; vendor: string; vram_total_bytes: number; driver_version: string | null }
interface DriverCheck {
  gpu_name: string; driver_version: string | null; cuda_version: string; minimum: string; recommended: string;
  status: "ok" | "old_but_works" | "too_old" | "unknown"; message: string; download_url: string;
}
interface StarterModel { repo: string; file: string; name: string; size_bytes: number; why: string }
interface SetupPlan {
  completed: boolean; show: boolean; installed: InstalledRuntime[];
  cpu_name: string; logical_cores: number; total_ram_bytes: number;
  gpus: SetupGpu[]; gpu_note: string | null;
  available: AvailableRuntime[]; available_error: string | null;
  driver: DriverCheck | null; starter_gpu: StarterModel | null; starter_cpu: StarterModel;
}

const usesGpu = (variant: string) => variant.startsWith("cuda") || variant === "vulkan";
const label = (rt: AvailableRuntime) =>
  rt.variant === "cpu" ? "CPU runtime" : rt.variant === "vulkan" ? "Vulkan runtime" : `NVIDIA CUDA ${rt.variant.replace("cuda-", "")} runtime`;

export function FirstRun({ go }: { go: (p: PageId) => void }) {
  const [open, setOpen] = useState(false);
  const [plan, setPlan] = useState<SetupPlan | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [installed, setInstalled] = useState<InstalledRuntime[]>([]);
  const [installing, setInstalling] = useState<string | null>(null);
  const [progress, setProgress] = useState<InstallProgress | null>(null);
  const [installErr, setInstallErr] = useState<string | null>(null);
  const [dlId, setDlId] = useState<string | null>(null);
  const [dl, setDl] = useState<DownloadEvent | null>(null);
  const [dlErr, setDlErr] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<string | null>(null);
  const [showOthers, setShowOthers] = useState(false);
  // Storage step: nothing installs until the folders are confirmed (a profile
  // that already has a runtime counts as confirmed; Settings > Storage changes them).
  const [modelsDir, setModelsDir] = useState("");
  const [runtimesDir, setRuntimesDir] = useState("");
  const [modelsCheck, setModelsCheck] = useState<FolderCheck | null>(null);
  const [runtimesCheck, setRuntimesCheck] = useState<FolderCheck | null>(null);
  const [storageOk, setStorageOk] = useState(false);
  const [storageBusy, setStorageBusy] = useState(false);
  const [storageErr, setStorageErr] = useState<string | null>(null);
  const autoStarted = useRef(false);
  const dlIdRef = useRef<string | null>(null);

  // Open by itself on a fresh profile, or when Settings asks.
  useEffect(() => {
    invoke<boolean>("setup_should_show").then((s) => s && setOpen(true)).catch(() => {});
    const onOpen = () => setOpen(true);
    window.addEventListener(OPEN_SETUP_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_SETUP_EVENT, onOpen);
  }, []);

  const loadPlan = useCallback(async () => {
    setPlan(null);
    setPlanErr(null);
    try {
      const p = await invoke<SetupPlan>("setup_plan");
      const st = await api8.getStorage();
      setModelsDir(st.models.path);
      setRuntimesDir(st.runtimes.path);
      setStorageOk(p.installed.length > 0);
      setPlan(p);
      setInstalled(p.installed);
    } catch (e) {
      setPlanErr(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    autoStarted.current = false;
    setInstallErr(null);
    setDlErr(null);
    loadPlan();
    const un1 = listen<InstallProgress>("runtime://install-progress", (e) => setProgress(e.payload));
    const un2 = listen<DownloadEvent>("download://event", (e) => {
      if (e.payload.id === dlIdRef.current) setDl(e.payload);
    });
    return () => {
      un1.then((f) => f());
      un2.then((f) => f());
    };
  }, [open, loadPlan]);

  const install = useCallback(async (rt: AvailableRuntime) => {
    setInstalling(rt.id);
    setInstallErr(null);
    setProgress(null);
    try {
      await api.installRuntime(rt);
      setInstalled(await api.listRuntimes());
      setShowOthers(false);
    } catch (e) {
      const m = errorMessage(e);
      setInstallErr(m === "cancelled" ? "Cancelled. Downloaded parts are kept and resume next time." : m);
    } finally {
      setInstalling(null);
      setProgress(null);
    }
  }, []);

  const recommended = plan?.available.find((r) => r.recommended) ?? null;
  const driverBlocks = plan?.driver?.status === "too_old";

  // Install the recommended runtime by itself, unless the NVIDIA driver is too
  // old (then the user picks Vulkan or CPU after reading why).
  useEffect(() => {
    if (!open || !plan || autoStarted.current) return;
    if (plan.installed.length > 0 || !recommended || driverBlocks || !storageOk) return;
    autoStarted.current = true;
    install(recommended);
  }, [open, plan, recommended, driverBlocks, storageOk, install]);

  const confirmStorage = async () => {
    setStorageBusy(true);
    setStorageErr(null);
    try {
      await api8.setStorageDirs(modelsDir, runtimesDir, false);
      setStorageOk(true);
    } catch (e) {
      setStorageErr(errorMessage(e));
    } finally {
      setStorageBusy(false);
    }
  };

  const runtime = installed[0] ?? null;
  const starter = plan && runtime ? (usesGpu(runtime.variant) && plan.starter_gpu ? plan.starter_gpu : plan.starter_cpu) : null;

  // Is the starter already on disk or downloading? (Same data Discover shows.)
  useEffect(() => {
    if (!starter) return;
    setModelStatus(null);
    api2.hfRepo(starter.repo)
      .then((r) => setModelStatus(r.variants.find((v) => v.key === starter.file)?.status ?? null))
      .catch(() => setModelStatus(null));
  }, [starter?.repo, starter?.file]);

  const download = async () => {
    if (!starter) return;
    setDlErr(null);
    try {
      const id = await api2.downloadVariant(starter.repo, starter.file, null);
      dlIdRef.current = id;
      setDlId(id);
      setDl(null);
    } catch (e) {
      setDlErr(errorMessage(e));
    }
  };

  const close = async (skipped: boolean) => {
    try {
      await invoke("finish_setup", { skipped });
    } catch {
      // Not fatal: setup just opens again next launch.
    }
    setOpen(false);
    if (!skipped && (dl?.state === "completed" || modelStatus === "installed")) go("chat");
  };

  if (!open) return null;

  const others = (plan?.available ?? []).filter(
    (r) => !r.recommended && (r.variant === "vulkan" || r.variant === "cpu"),
  );
  const myDl = dl && dl.id === dlId ? dl : null;
  const modelDone = myDl?.state === "completed" || modelStatus === "installed" || modelStatus === "loaded";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="firstrun-title"
      className="firstrun"
      style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(0,0,0,0.55)", display: "grid", placeItems: "center", padding: 24 }}
    >
      <div className="panel" style={{ width: "min(720px, 100%)", maxHeight: "100%", overflow: "auto", padding: "20px 24px" }}>
        <div className="row" style={{ marginBottom: 4 }}>
          <h2 id="firstrun-title" style={{ margin: 0, fontSize: 16 }}>Set up Local AI Studio</h2>
          <span className="spacer" />
          <button className="btn sm ghost" onClick={() => close(true)}>Skip setup</button>
        </div>
        <p className="lede" style={{ marginBottom: 14 }}>
          Four steps: check this PC, choose where files go, install the engine that runs models, and get a first model. You can skip and do it later from Runtimes and Discover; Settings opens this again.
        </p>

        {planErr && (
          <div className="notice error row">
            <span>{planErr}</span><span className="spacer" />
            <button className="btn sm" onClick={loadPlan}>Retry</button>
          </div>
        )}
        {!plan && !planErr && <div className="skeleton" style={{ height: 120 }} />}

        {plan && (
          <>
            <h3 style={{ fontSize: 13, margin: "6px 0 8px" }}>1. This PC</h3>
            <dl className="kv">
              <dt>CPU</dt><dd>{plan.cpu_name} ({plan.logical_cores} threads)</dd>
              <dt>Memory</dt><dd>{bytes(plan.total_ram_bytes)}</dd>
              <dt>GPU</dt>
              <dd>
                {plan.gpus.length === 0
                  ? "No GPU found; models run on the CPU."
                  : plan.gpus.map((g) => (
                      <div key={g.name + g.vram_total_bytes}>
                        {g.name} ({bytes(g.vram_total_bytes)} VRAM{g.driver_version ? `, driver ${g.driver_version}` : ""})
                      </div>
                    ))}
              </dd>
            </dl>
            {plan.driver && plan.driver.status !== "ok" && (
              <div className={`notice ${plan.driver.status === "too_old" ? "error" : "warn"}`} style={{ marginTop: 10 }} data-driver-status={plan.driver.status}>
                <div>{plan.driver.message}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="btn sm" onClick={() => invoke("open_driver_download").catch(() => {})}>Get the NVIDIA driver</button>
                  <span className="mono faint">{plan.driver.download_url}</span>
                </div>
              </div>
            )}
            {plan.driver?.status === "ok" && <p className="faint" style={{ margin: "8px 0 0" }}>{plan.driver.message}</p>}

            <h3 style={{ fontSize: 13, margin: "20px 0 8px" }}>2. Storage</h3>
            <div data-storage-step data-storage-confirmed={storageOk ? "1" : "0"}>
              <div className="form-grid">
                <StorageFolder id="setup-models" label="Models folder" value={modelsDir} onChange={(v) => { setModelsDir(v); }} disabled={storageBusy || installed.length > 0 || !!installing} onCheck={setModelsCheck} />
                <StorageFolder id="setup-runtimes" label="Runtimes folder (the engine that runs models)" value={runtimesDir} onChange={(v) => { setRuntimesDir(v); }} disabled={storageBusy || installed.length > 0 || !!installing} onCheck={setRuntimesCheck} />
              </div>
              {installed.length === 0 && (
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="btn sm primary" disabled={storageBusy || !modelsCheck?.ok || !runtimesCheck?.ok} onClick={confirmStorage}>
                    {storageOk ? "Use these folders again" : "Use these folders"}
                  </button>
                  {!storageOk && <span className="faint">The engine installs after you confirm. The defaults are fine for most people.</span>}
                  {storageOk && <span className="faint">Saved.</span>}
                </div>
              )}
              {installed.length > 0 && <p className="faint" style={{ margin: "8px 0 0" }}>Change these later in Settings, Storage.</p>}
              {storageErr && <div className="danger-text" style={{ marginTop: 6 }}>{storageErr}</div>}
            </div>

            <h3 style={{ fontSize: 13, margin: "20px 0 8px" }}>3. Runtime</h3>
            {runtime ? (
              <div className="row">
                <span>Installed:</span> <strong className="mono">{runtime.id}</strong>
                <span className="faint">{usesGpu(runtime.variant) ? "uses the GPU" : "uses the CPU"}</span>
              </div>
            ) : plan.available_error ? (
              <div className="notice error row">
                <span>Could not reach GitHub for the llama.cpp runtime: {plan.available_error}</span><span className="spacer" />
                <button className="btn sm" onClick={loadPlan}>Retry</button>
              </div>
            ) : !storageOk ? (
              <div className="faint">Choose the storage folders above first.</div>
            ) : recommended ? (
              <div>
                <div className="row">
                  <span>{driverBlocks ? "Suggested instead:" : "Recommended:"}</span>
                  <strong>{label(recommended)}</strong>
                  <span className="mono faint">{recommended.id}, {bytes(recommended.total_size)}</span>
                </div>
                <div className="dim" style={{ marginTop: 4 }}>{recommended.note}</div>
                {installing ? (
                  <div style={{ marginTop: 8 }} data-installing={installing}>
                    <div className="progress"><span style={{ width: `${progress?.phase === "download" ? percent(progress.downloaded, progress.total ?? 0) : progress ? 100 : 0}%` }} /></div>
                    <div className="row" style={{ marginTop: 4 }}>
                      <span className="mono faint">
                        {!progress ? `Installing ${installing}` : progress.phase === "download"
                          ? `${progress.asset}: ${bytes(progress.downloaded)} of ${bytes(progress.total)} ${rate(progress.bytes_per_sec)}`
                          : progress.phase === "extract" ? "Extracting" : progress.phase === "probe" ? "Checking the build runs" : "Done"}
                      </span>
                      <span className="spacer" />
                      <button className="btn sm" onClick={() => api.cancelRuntimeInstall(installing)}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="row" style={{ marginTop: 8 }}>
                    <button className="btn sm primary" onClick={() => install(recommended)}>Install {label(recommended)}</button>
                    {(driverBlocks || showOthers) && others.map((r) => (
                      <button key={r.id} className="btn sm" onClick={() => install(r)}>Install {label(r)} ({bytes(r.total_size)})</button>
                    ))}
                    {!driverBlocks && !showOthers && others.length > 0 && (
                      <button className="btn sm ghost" onClick={() => setShowOthers(true)}>Other runtimes</button>
                    )}
                  </div>
                )}
                {installErr && <div className="danger-text" style={{ marginTop: 6 }}>{installErr}</div>}
              </div>
            ) : (
              <div className="faint">Looking for the latest llama.cpp release...</div>
            )}

            <h3 style={{ fontSize: 13, margin: "20px 0 8px" }}>4. A first model</h3>
            {!starter ? (
              <div className="faint">Install a runtime first; the suggestion depends on whether models run on the GPU or the CPU.</div>
            ) : (
              <div data-starter={starter.file}>
                <div className="row">
                  <strong>{starter.name}</strong>
                  <span className="mono faint">{starter.repo}, {bytes(starter.size_bytes)}</span>
                </div>
                <div className="dim" style={{ marginTop: 4 }}>{starter.why}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  {modelDone ? (
                    <span>Downloaded. It is in My Models; load it there or pick it in Chat.</span>
                  ) : myDl && ["running", "queued", "paused"].includes(myDl.state) ? (
                    <div style={{ flex: 1 }}>
                      <div className="progress"><span style={{ width: `${percent(myDl.downloaded, myDl.total)}%` }} /></div>
                      <div className="mono faint" style={{ marginTop: 4 }}>
                        {myDl.state === "queued" ? "Queued" : `${bytes(myDl.downloaded)} of ${bytes(myDl.total)} ${rate(myDl.bytes_per_sec)}`}. It keeps going if you close this.
                      </div>
                    </div>
                  ) : modelStatus === "downloading" || modelStatus === "paused" ? (
                    <span className="faint">Already downloading; see Discover.</span>
                  ) : (
                    <button className="btn sm primary" onClick={download}>Download ({bytes(starter.size_bytes)})</button>
                  )}
                  {!modelDone && <button className="btn sm ghost" onClick={() => close(false).then(() => go("discover"))}>Browse other models</button>}
                </div>
                {(dlErr || myDl?.error) && <div className="danger-text" style={{ marginTop: 6 }}>{dlErr ?? myDl?.error}</div>}
              </div>
            )}

            <div className="row" style={{ marginTop: 22 }}>
              <span className="faint">To open this again later: Settings, "Run setup again".</span>
              <span className="spacer" />
              <button className="btn primary" onClick={() => close(false)}>Finish</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

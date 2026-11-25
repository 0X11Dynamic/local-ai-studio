// Settings > Storage: models folder, runtimes folder (with "move installed
// runtimes") and the read-only data folder.
import { useCallback, useEffect, useState } from "react";
import { api, errorMessage } from "../lib/api";
import { api8, type StorageInfo } from "../lib/api8";
import { StorageFolder } from "./StorageFolder";

const same = (a: string, b: string) => a.replace(/[\\/]+$/, "").toLowerCase() === b.replace(/[\\/]+$/, "").toLowerCase();

export function StoragePanel({ onMessage, onError }: { onMessage: (m: string) => void; onError: (m: string) => void }) {
  const [info, setInfo] = useState<StorageInfo | null>(null);
  const [models, setModels] = useState("");
  const [runtimes, setRuntimes] = useState("");
  const [busy, setBusy] = useState(false);
  const [scanNote, setScanNote] = useState<string | null>(null);

  const load = useCallback(async () => {
    const i = await api8.getStorage();
    setInfo(i);
    setModels(i.models.path);
    setRuntimes(i.runtimes.path);
  }, []);
  useEffect(() => {
    load().catch((e) => onError(errorMessage(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try { await fn(); onMessage(ok); await load(); } catch (e) { onError(errorMessage(e)); } finally { setBusy(false); }
  };

  if (!info) return <div className="skeleton" style={{ height: 100 }} />;
  const modelsChanged = !same(models, info.models.path);
  const runtimesChanged = !same(runtimes, info.runtimes.path);

  return (
    <div className="form-grid" style={{ maxWidth: 720 }} data-storage-panel>
      <StorageFolder id="storage-models" label="Models folder (new downloads go here)" value={models} onChange={setModels} disabled={busy} />
      <div className="row wide">
        <button className="btn sm primary" disabled={busy || !modelsChanged}
          onClick={() => act(() => api8.setStorageDirs(models, null, false),
            "Models folder changed. New downloads go there; models you already have stay where they are.")}>Use this models folder</button>
        <button className="btn sm ghost" disabled={busy || same(models, info.default_models_dir)} onClick={() => setModels(info.default_models_dir)}>Default</button>
        {!modelsChanged && (
          <button className="btn sm" disabled={busy}
            onClick={() => act(async () => {
              const r = await api.importModels(info.models.path);
              setScanNote(`${r.registered.length} added, ${r.invalid.length} invalid, ${r.skipped.length} skipped`);
            }, "Scan finished.")}>Scan this folder for models</button>
        )}
        {scanNote && !modelsChanged && <span className="faint">{scanNote}</span>}
      </div>

      <StorageFolder id="storage-runtimes" label={`Runtimes folder (${info.runtime_count} installed)`} value={runtimes} onChange={setRuntimes} disabled={busy} />
      <div className="row wide">
        {info.runtime_count > 0 && (
          <button className="btn sm primary" disabled={busy || !runtimesChanged}
            onClick={() => act(() => api8.setStorageDirs(null, runtimes, true),
              "Runtimes moved. The old copies were deleted after the new ones were checked.")}>
            {busy ? "Working..." : "Move installed runtimes here"}
          </button>
        )}
        <button className="btn sm" disabled={busy || !runtimesChanged}
          onClick={() => act(() => api8.setStorageDirs(null, runtimes, false),
            info.runtime_count > 0 ? "Runtimes folder changed. The old runtimes stay in the old folder and are no longer listed." : "Runtimes folder changed.")}>
          {info.runtime_count > 0 ? "Use it without moving" : "Use this runtimes folder"}
        </button>
        <button className="btn sm ghost" disabled={busy || same(runtimes, info.default_runtimes_dir)} onClick={() => setRuntimes(info.default_runtimes_dir)}>Default</button>
      </div>
      <span className="help wide">
        Changing the runtimes folder needs the model unloaded; it is refused while a model runs.
        {info.install_dir ? ` The defaults are next to the app (${info.install_dir}).` : ""}
      </span>

      <div className="field wide">
        <label>Data folder (chats, settings, logs)</label>
        <div className="row">
          <span className="mono">{info.data_dir}</span>
          <button className="btn sm" onClick={() => api8.openDataFolder().catch((e) => onError(errorMessage(e)))}>Open folder</button>
        </div>
        <span className="help">Fixed. It holds your chats and settings, so it is not moved.</span>
      </div>
    </div>
  );
}

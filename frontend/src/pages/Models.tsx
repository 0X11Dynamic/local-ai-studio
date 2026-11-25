import { useCallback, useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api, errorMessage, type ImportReport, type InstalledRuntime, type ModelView } from "../lib/api";
import { bytes, tokens } from "../lib/format";
import { useEngine } from "../lib/engine";
import { capBadges, capDetails, capsOf } from "../lib/api6";
import { LoadPanel } from "../components/LoadPanel";
import { LoadProgressMini } from "../components/LoadProgressView";
import { LogView } from "../components/LogView";
import type { PageId } from "../App";

export function Models({ go }: { go: (p: PageId) => void }) {
  const [models, setModels] = useState<ModelView[] | null>(null);
  const [runtimes, setRuntimes] = useState<InstalledRuntime[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [report, setReport] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const { status } = useEngine();

  const reload = useCallback(async () => {
    try {
      const m = await api.refreshModels();
      setModels(m);
      // Discover's "Load model" hands over the model to select.
      const handoff = localStorage.getItem("las.selectModel");
      if (handoff) localStorage.removeItem("las.selectModel");
      setSelected((s) => (handoff && m.some((x) => x.id === handoff) ? handoff : s ?? m.find((x) => x.active)?.id ?? m.find((x) => x.status === "ok")?.id ?? null));
    } catch (e) {
      setErr(errorMessage(e));
      setModels([]);
    }
  }, []);

  useEffect(() => {
    reload();
    api.listRuntimes().then(setRuntimes);
  }, [reload]);

  // Keep the active flag in sync when a model loads/unloads elsewhere.
  useEffect(() => {
    setModels((ms) => ms?.map((m) => ({ ...m, active: m.id === status?.model_id && status?.state !== "unloaded" })) ?? null);
  }, [status?.model_id, status?.state]);

  const summarize = (r: ImportReport) => {
    const parts = [`${r.registered.length} model(s) registered`];
    if (r.invalid.length) parts.push(`${r.invalid.length} incomplete or invalid`);
    if (r.skipped.length) parts.push(`${r.skipped.length} file(s) skipped: ${r.skipped.slice(0, 3).map((s) => `${s.path.split(/[\\/]/).pop()} (${s.reason})`).join("; ")}`);
    return parts.join(", ");
  };

  const doImport = async (folder: boolean) => {
    setErr(null);
    setReport(null);
    const path = await open(folder ? { directory: true } : { multiple: false, filters: [{ name: "GGUF model", extensions: ["gguf"] }] });
    if (!path) return;
    try {
      const r = await api.importModels(path);
      setReport(summarize(r));
      await reload();
      if (r.registered[0]) setSelected(r.registered[0]);
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const scanLm = async () => {
    setErr(null);
    try {
      setReport(summarize(await api.scanLmStudio()));
      await reload();
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const visible = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (models ?? []).filter(
      (m) => !f || m.display_name.toLowerCase().includes(f) || (m.repo_id ?? "").toLowerCase().includes(f) || (m.meta.architecture ?? "").includes(f),
    );
  }, [models, filter]);

  const sel = models?.find((m) => m.id === selected) ?? null;
  // A tick for one model never carries over to another.
  useEffect(() => {
    setConfirmDelete(false);
    setShowDelete(false);
  }, [selected]);

  const removeFromLibrary = async () => {
    if (!sel) return;
    try {
      await api.removeModel(sel.id);
      setSelected(null);
      await reload();
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const deleteFiles = async () => {
    if (!sel) return;
    try {
      const gone = await api.deleteModelFiles(sel.id, confirmDelete);
      setReport(`Moved ${gone.length} file(s) to the Recycle Bin. Restore them from there if you change your mind.`);
      setShowDelete(false);
      setConfirmDelete(false);
      setSelected(null);
      await reload();
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <input className="input" placeholder="Filter by name, repo or architecture" aria-label="Filter models" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: 300 }} />
        <span className="spacer" />
        <button className="btn sm" onClick={() => doImport(false)}>Import GGUF file</button>
        <button className="btn sm" onClick={() => doImport(true)}>Import folder</button>
        <button className="btn sm" onClick={scanLm} title="Register models from LM Studio's folder in place (no copies)">Rescan LM Studio</button>
        <button className="btn sm ghost" onClick={reload}>Refresh</button>
      </div>
      {report && <div className="notice" style={{ marginBottom: 12 }}>{report}</div>}
      {err && <div className="notice error" style={{ marginBottom: 12 }}>{err}</div>}
      {runtimes.length === 0 && (
        <div className="notice warn row" style={{ marginBottom: 12 }}>
          <span>No inference runtime is installed, so models can't be loaded yet.</span>
          <span className="spacer" />
          <button className="btn primary sm" onClick={() => go("runtimes")}>Open Runtimes</button>
        </div>
      )}

      <div className="split">
        <div>
          {models == null ? (
            <div style={{ display: "grid", gap: 10 }}>{[0, 1, 2, 3].map((i) => <div className="skeleton" key={i} />)}</div>
          ) : models.length === 0 ? (
            <div className="empty">
              <strong>Your library is empty.</strong>
              Import a GGUF file or a folder of them. Files stay where they are; the library only records their location.
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr><th>Model</th><th>Arch</th><th>Quant</th><th className="num">Context</th><th className="num">Size</th></tr>
              </thead>
              <tbody>
                {visible.map((m) => (
                  <tr key={m.id} className={`selectable${m.id === selected ? " selected" : ""}`} onClick={() => setSelected(m.id)}>
                    <td>
                      <div>
                        {m.display_name}{" "}
                        {m.active && <span className="tag accent">{status?.state === "loading" ? "loading" : "loaded"}</span>}
                        {m.active && status?.state === "loading" && <LoadProgressMini />}
                        {m.status !== "ok" && <span className="tag danger">{m.status}</span>}
                        {capBadges(capsOf(m.meta), !!m.projector_path).map((b) => (
                          <span key={b.label} className={`tag cap ${b.tone}`} title={b.title}>{b.label}</span>
                        ))}
                      </div>
                      <div className="faint mono" style={{ fontSize: 11 }}>{m.repo_id ?? m.source}</div>
                    </td>
                    <td className="mono dim">{m.meta.architecture ?? "?"}</td>
                    <td className="mono">{m.quantization ?? "?"}</td>
                    <td className="num">{tokens(m.meta.context_length)}</td>
                    <td className="num">{bytes(m.size)}</td>
                  </tr>
                ))}
                {visible.length === 0 && (
                  <tr><td colSpan={5} className="dim">No models match "{filter}".</td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        <aside>
          {sel ? (
            <div className="panel" style={{ display: "grid", gap: 14 }}>
              <div>
                <div style={{ fontWeight: 600, overflowWrap: "anywhere" }}>{sel.display_name}</div>
                <div className="mono faint" style={{ overflowWrap: "anywhere", marginTop: 2 }}>{sel.path}</div>
              </div>
              {sel.status !== "ok" && <div className="notice error">{sel.status_detail}</div>}
              <dl className="kv">
                <dt>Architecture</dt><dd>{sel.meta.architecture ?? "unknown"}</dd>
                <dt>Trained context</dt><dd>{sel.meta.context_length?.toLocaleString() ?? "unknown"} tokens</dd>
                <dt>Layers</dt><dd>{sel.meta.block_count ?? "?"}</dd>
                <dt>Attention heads</dt><dd>{sel.meta.head_count ?? "?"} ({sel.meta.head_count_kv ?? "?"} KV)</dd>
                {sel.meta.expert_count ? (<><dt>Experts</dt><dd>{sel.meta.expert_count}</dd></>) : null}
                <dt>Files</dt><dd>{sel.shards.length} ({bytes(sel.size)})</dd>
                <dt>Chat template</dt><dd>{sel.meta.has_chat_template ? "embedded" : "none in file"}</dd>
                {capDetails(capsOf(sel.meta)).map((d) => (
                  <div key={d.label} style={{ display: "contents" }}><dt>{d.label}</dt><dd>{d.value}</dd></div>
                ))}
              </dl>
              {sel.status === "ok" && <LoadPanel model={sel} runtimes={runtimes} />}
              <div className="row">
                <button className="btn sm ghost" onClick={() => revealItemInDir(sel.path)}>Show in Explorer</button>
                <button className="btn sm ghost" onClick={removeFromLibrary} disabled={sel.active}>Remove from library</button>
                <button className="btn sm ghost danger" onClick={() => setShowDelete((v) => !v)} disabled={sel.active}>Delete files</button>
              </div>
              {showDelete && (
                <div className="notice error" style={{ display: "grid", gap: 8 }}>
                  <span>
                    This moves the files to the Recycle Bin: {sel.shards.length} model file(s) ({bytes(sel.size)})
                    {sel.projector_path ? ", plus its vision projector if no other model uses it" : ""}
                    {sel.source === "lmstudio" ? ", including from LM Studio's library" : ""}. The model leaves the library; restore the files from the Recycle Bin to get it back.
                    If the Recycle Bin can't hold them, nothing is deleted.
                  </span>
                  <label className="check">
                    <input type="checkbox" aria-label="Confirm delete" checked={confirmDelete} onChange={(e) => setConfirmDelete(e.target.checked)} /> I understand this moves the files to the Recycle Bin
                  </label>
                  <div className="row">
                    <button className="btn sm danger" disabled={!confirmDelete} onClick={deleteFiles}>Move to Recycle Bin</button>
                    <button className="btn sm ghost" onClick={() => { setShowDelete(false); setConfirmDelete(false); }}>Keep files</button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            models && models.length > 0 && <div className="empty">Select a model to see its details and load settings.</div>
          )}
        </aside>
      </div>

      <h2>Backend log</h2>
      <LogView />
    </div>
  );
}

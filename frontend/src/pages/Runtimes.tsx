import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { api, errorMessage, type AvailableRuntime, type InstallProgress, type InstalledRuntime } from "../lib/api";
import { bytes, percent, rate, when } from "../lib/format";

export function Runtimes() {
  const [installed, setInstalled] = useState<InstalledRuntime[] | null>(null);
  const [available, setAvailable] = useState<AvailableRuntime[] | null>(null);
  const [availErr, setAvailErr] = useState<string | null>(null);
  const [defaultId, setDefaultId] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, InstallProgress>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [customErr, setCustomErr] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setInstalled(await api.listRuntimes());
    const s = await api.getSettings();
    setDefaultId((s["runtime.default"] as string) ?? null);
  }, []);

  const fetchAvailable = useCallback(() => {
    setAvailErr(null);
    setAvailable(null);
    api.fetchAvailableRuntimes().then(setAvailable).catch((e) => {
      setAvailErr(errorMessage(e));
      setAvailable([]);
    });
  }, []);

  useEffect(() => {
    reload();
    fetchAvailable();
    const un = listen<InstallProgress>("runtime://install-progress", (e) =>
      setProgress((p) => ({ ...p, [e.payload.runtime_id]: e.payload })),
    );
    return () => {
      un.then((f) => f());
    };
  }, [reload, fetchAvailable]);

  const install = async (rt: AvailableRuntime) => {
    setBusy((b) => ({ ...b, [rt.id]: true }));
    setErrors((e) => ({ ...e, [rt.id]: "" }));
    try {
      await api.installRuntime(rt);
      await reload();
    } catch (e) {
      const msg = errorMessage(e);
      setErrors((x) => ({ ...x, [rt.id]: msg === "cancelled" ? "Cancelled. Downloaded parts are kept and resume next time." : msg }));
    } finally {
      setBusy((b) => ({ ...b, [rt.id]: false }));
      setProgress((p) => {
        const n = { ...p };
        delete n[rt.id];
        return n;
      });
    }
  };

  const makeDefault = async (id: string) => {
    await api.setSetting("runtime.default", id);
    setDefaultId(id);
  };

  const remove = async (id: string) => {
    try {
      await api.removeRuntime(id);
      await reload();
    } catch (e) {
      setErrors((x) => ({ ...x, [id]: errorMessage(e) }));
    }
  };

  const addCustom = async () => {
    setCustomErr(null);
    const path = await open({ multiple: false, filters: [{ name: "llama-server", extensions: ["exe"] }] });
    if (!path) return;
    try {
      await api.registerCustomRuntime(path);
      await reload();
    } catch (e) {
      setCustomErr(errorMessage(e));
    }
  };

  const installedIds = new Set(installed?.map((r) => r.id));

  return (
    <div className="page">
      <p className="lede">
        Runtimes are official llama.cpp builds. The app starts the selected build as a separate process when you load a model, and stops it on unload or exit.
      </p>

      <h2>Installed</h2>
      {installed == null ? (
        <div className="skeleton" style={{ maxWidth: 480 }} />
      ) : installed.length === 0 ? (
        <div className="empty">
          <strong>No runtime installed.</strong>
          Install the recommended build below. It downloads from the llama.cpp GitHub releases and is checked against its published SHA-256.
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr><th>Runtime</th><th>Backend</th><th>Installed</th><th>Location</th><th /></tr>
          </thead>
          <tbody>
            {installed.map((r) => (
              <tr key={r.id}>
                <td className="mono">
                  {r.id} {r.id === defaultId && <span className="tag accent">default</span>}
                </td>
                <td>{r.variant}</td>
                <td className="dim">{when(r.installed_at)}</td>
                <td className="mono dim" style={{ maxWidth: 340, overflowWrap: "anywhere" }}>{r.server_path}</td>
                <td>
                  <div className="row" style={{ justifyContent: "flex-end" }}>
                    {r.id !== defaultId && <button className="btn sm" onClick={() => makeDefault(r.id)}>Make default</button>}
                    <button className="btn sm danger" onClick={() => remove(r.id)}>{r.source === "custom" ? "Forget" : "Remove"}</button>
                  </div>
                  {errors[r.id] && <div className="danger-text" style={{ marginTop: 4 }}>{errors[r.id]}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn sm" onClick={addCustom}>Use an existing llama-server.exe</button>
        {customErr && <span className="danger-text">{customErr}</span>}
      </div>

      <h2>Available from llama.cpp releases</h2>
      {available == null ? (
        <div style={{ display: "grid", gap: 8, maxWidth: 560 }}><div className="skeleton" /><div className="skeleton" /></div>
      ) : availErr ? (
        <div className="notice error row">
          <span>Could not reach GitHub: {availErr}</span>
          <span className="spacer" />
          <button className="btn sm" onClick={fetchAvailable}>Retry</button>
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr><th>Build</th><th>Notes for this PC</th><th className="num">Download</th><th /></tr>
          </thead>
          <tbody>
            {available.map((rt) => {
              const p = progress[rt.id];
              const done = installedIds.has(rt.id);
              return (
                <tr key={rt.id}>
                  <td className="mono">
                    {rt.id} {rt.recommended && <span className="tag accent">recommended</span>}
                  </td>
                  <td className="dim">
                    {rt.note}
                    {p && (
                      <div style={{ marginTop: 6 }}>
                        <div className="progress"><span style={{ width: `${p.phase === "download" ? percent(p.downloaded, p.total ?? 0) : 100}%` }} /></div>
                        <div className="mono faint" style={{ marginTop: 3 }}>
                          {p.phase === "download"
                            ? `${p.asset}: ${bytes(p.downloaded)} of ${bytes(p.total)} ${rate(p.bytes_per_sec)}`
                            : p.phase === "extract" ? "Extracting" : p.phase === "probe" ? "Checking the build runs" : "Done"}
                        </div>
                      </div>
                    )}
                    {errors[rt.id] && <div className="danger-text" style={{ marginTop: 4 }}>{errors[rt.id]}</div>}
                  </td>
                  <td className="num">{bytes(rt.total_size)}</td>
                  <td>
                    <div className="row" style={{ justifyContent: "flex-end" }}>
                      {done ? (
                        <span className="faint">Installed</span>
                      ) : busy[rt.id] ? (
                        <button className="btn sm" onClick={() => api.cancelRuntimeInstall(rt.id)}>Cancel</button>
                      ) : (
                        <button className={`btn sm${rt.recommended ? " primary" : ""}`} onClick={() => install(rt)}>Install</button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

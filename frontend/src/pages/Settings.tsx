import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { invoke } from "@tauri-apps/api/core";
import { Markdown } from "../components/Markdown";
import { OPEN_SETUP_EVENT } from "../components/FirstRun";
import { api, errorMessage, type AppPaths } from "../lib/api";
import { api2 } from "../lib/api2";
import { SearchSettingsPanel } from "../components/SearchSettingsPanel";
import { StoragePanel } from "../components/StoragePanel";

export function Settings() {
  const [paths, setPaths] = useState<AppPaths | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [concurrency, setConcurrency] = useState(2);
  const [hasHf, setHasHf] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [guide, setGuide] = useState<string | null>(null);
  const [guideErr, setGuideErr] = useState<string | null>(null);

  useEffect(() => {
    api.getPaths().then(setPaths);
    api.getSettings().then((s) => {
      setTheme(s["ui.theme"] === "light" ? "light" : "dark");
      if (typeof s["downloads.concurrency"] === "number") setConcurrency(s["downloads.concurrency"] as number);
    });
    api2.hasHfToken().then(setHasHf);
  }, []);

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setErr(null);
    setMsg(null);
    try {
      await fn();
      setMsg(ok);
    } catch (e) {
      setErr(errorMessage(e));
    }
  };

  const changeTheme = async (t: "dark" | "light") => {
    setTheme(t);
    if (t === "light") document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    await api.setSetting("ui.theme", t);
  };

  return (
    <div className="page">
      {err && <div className="notice error" style={{ marginBottom: 10 }}>{err}</div>}
      {msg && <div className="notice" style={{ marginBottom: 10 }}>{msg}</div>}
      <h2>Appearance</h2>
      <div className="field" style={{ maxWidth: 240 }}>
        <label htmlFor="theme">Theme</label>
        <select id="theme" className="input" value={theme} onChange={(e) => changeTheme(e.target.value as "dark" | "light")}>
          <option value="dark">Dark</option>
          <option value="light">Light</option>
        </select>
      </div>

      <h2>Downloads</h2>
      <p className="lede">The models folder is under Storage below.</p>
      <div className="form-grid" style={{ maxWidth: 720 }}>
        <div className="field">
          <label htmlFor="conc">Downloads at the same time</label>
          <select id="conc" className="input" value={concurrency} onChange={(e) => {
            const n = Number(e.target.value);
            setConcurrency(n);
            run(() => api.setSetting("downloads.concurrency", n), "Saved. Takes effect after restarting the app.");
          }}>
            {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="field wide">
          <label htmlFor="hf">Hugging Face token (for gated or private repositories)</label>
          <div className="row">
            <input id="hf" className="input mono" type="password" autoComplete="off" style={{ width: 340 }} placeholder={hasHf ? "A token is stored" : "hf_..."} value={hfToken} onChange={(e) => setHfToken(e.target.value)} />
            <button className="btn sm" disabled={!hfToken} onClick={() => run(async () => { setHasHf(await api2.setHfToken(hfToken)); setHfToken(""); }, "Token saved to Windows Credential Manager.")}>Save</button>
            {hasHf && <button className="btn sm ghost" onClick={() => run(async () => { setHasHf(await api2.setHfToken(null)); }, "Token removed.")}>Remove</button>}
          </div>
          <span className="help">Stored in Windows Credential Manager and sent only to huggingface.co. It is never shown again or written to logs.</span>
        </div>
      </div>

      <h2>Search</h2>
      <p className="lede">The web_search tool lets models in Agent and Plan mode look things up. Pick where searches go.</p>
      <SearchSettingsPanel />

      <h2>Setup</h2>
      <p className="lede">Check this PC again, install the recommended runtime and get a starter model.</p>
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="btn sm" onClick={() => window.dispatchEvent(new Event(OPEN_SETUP_EVENT))}>Run setup again</button>
      </div>

      <h2>User guide</h2>
      <p className="lede">How every page works, what the agent may do without asking, where your data lives, and fixes for common problems. It is built into the app, so it works offline.</p>
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="btn sm" aria-expanded={guide != null} onClick={async () => {
          if (guide != null) { setGuide(null); return; }
          try { setGuideErr(null); setGuide(await invoke<string>("user_guide")); } catch (e) { setGuideErr(errorMessage(e)); }
        }}>{guide != null ? "Hide the user guide" : "Open the user guide"}</button>
      </div>
      {guideErr && <div className="notice error" style={{ marginBottom: 10 }}>{guideErr}</div>}
      {guide != null && (
        <div className="panel user-guide" aria-label="User guide" style={{ maxWidth: 860, marginBottom: 20 }}>
          <Markdown text={guide} />
        </div>
      )}

      <h2>Storage</h2>
      <p className="lede">Where models and runtimes are kept. Anything you already have stays where it is unless you move it.</p>
      <StoragePanel onMessage={(m) => { setErr(null); setMsg(m); }} onError={(m) => { setMsg(null); setErr(m); }} />
      {paths && (
        <dl className="kv" style={{ marginTop: 14 }}>
          <dt>Database</dt>
          <dd><button className="btn ghost sm mono" onClick={() => revealItemInDir(paths.db_path)}>{paths.db_path}</button></dd>
          <dt>Logs</dt><dd>{paths.logs_dir}</dd>
          <dt>LM Studio models</dt><dd>{paths.lmstudio_models_dir ?? "not found"} (registered in place, read-only)</dd>
        </dl>
      )}
      <p className="faint" style={{ marginTop: 14 }}>
        Secrets such as API keys and tokens are stored in Windows Credential Manager, never in the database or logs.
      </p>
    </div>
  );
}

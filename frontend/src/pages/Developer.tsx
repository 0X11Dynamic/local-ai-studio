import { useCallback, useEffect, useState } from "react";
import { errorMessage } from "../lib/api";
import { api2, type ApiConfig, type ApiLog, type ApiStatus } from "../lib/api2";
import { useEngine } from "../lib/engine";
import { when } from "../lib/format";

export function Developer() {
  const { status: engine } = useEngine();
  const [st, setSt] = useState<ApiStatus | null>(null);
  const [cfg, setCfg] = useState<ApiConfig | null>(null);
  const [origins, setOrigins] = useState("");
  const [logs, setLogs] = useState<ApiLog[]>([]);
  const [hasToken, setHasToken] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const s = await api2.apiStatus();
    setSt(s);
    setLogs(await api2.apiLogs(100));
    setHasToken(await api2.apiHasToken());
    return s;
  }, []);

  useEffect(() => {
    refresh().then((s) => {
      setCfg(s.config);
      setOrigins(s.config.cors_origins.join("\n"));
    });
    const t = setInterval(() => {
      if (document.visibilityState === "visible") api2.apiLogs(100).then(setLogs).catch(() => {});
    }, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  if (!cfg || !st) return <div className="page"><div className="skeleton" style={{ maxWidth: 400 }} /></div>;

  const full: ApiConfig = { ...cfg, cors_origins: origins.split("\n").map((s) => s.trim()).filter(Boolean) };
  const lan = !["127.0.0.1", "localhost", "::1"].includes(full.host);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const base = st.address ?? `http://${full.host === "0.0.0.0" ? "127.0.0.1" : full.host}:${full.port}`;
  const auth = full.require_auth || lan ? ` -H "Authorization: Bearer <token>"` : "";
  const curl = `curl ${base}/v1/chat/completions${auth} -H "Content-Type: application/json" -d "{\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Hello\\"}],\\"stream\\":true}"`;

  return (
    <div className="page">
      <p className="lede">An OpenAI-compatible server for the loaded model, so other apps on this PC can use it. Off until you start it.</p>
      {err && <div className="notice error" style={{ marginBottom: 10 }}>{err}</div>}

      <div className="row" style={{ marginBottom: 14 }}>
        <span>{st.running ? <>Running at <span className="mono">{st.address}</span></> : "Stopped"}</span>
        <span className="faint">{st.running && st.started_at ? `since ${when(st.started_at)}` : ""}</span>
        <span className="spacer" />
        {st.running ? (
          <>
            <button className="btn sm" disabled={busy} onClick={() => act(() => api2.apiStart(full))}>Restart with these settings</button>
            <button className="btn sm" disabled={busy} onClick={() => act(() => api2.apiStop())}>Stop server</button>
          </>
        ) : (
          <button className="btn primary sm" disabled={busy} onClick={() => act(() => api2.apiStart(full))}>Start server</button>
        )}
      </div>
      {!engine || engine.state !== "loaded" ? <div className="notice" style={{ marginBottom: 12 }}>No model is loaded. The server runs, but chat requests return 503 until a model is loaded.</div> : null}

      <div className="form-grid" style={{ maxWidth: 720 }}>
        <div className="field">
          <label htmlFor="api-host">Listen on</label>
          <select id="api-host" className="input" value={full.host === "0.0.0.0" ? "0.0.0.0" : "127.0.0.1"} onChange={(e) => setCfg({ ...cfg, host: e.target.value, require_auth: e.target.value === "0.0.0.0" ? true : cfg.require_auth })}>
            <option value="127.0.0.1">This PC only (127.0.0.1)</option>
            <option value="0.0.0.0">Local network (all interfaces)</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="api-port">Port</label>
          <input id="api-port" className="input mono" value={cfg.port} onChange={(e) => setCfg({ ...cfg, port: Number(e.target.value) || 0 })} />
          <span className="help">LM Studio's server also defaults to 1234.</span>
        </div>
        {lan && <div className="notice warn wide">Other devices on your network will be able to reach this server. A token is required and every request is logged.</div>}
        <label className="check wide">
          <input type="checkbox" checked={full.require_auth || lan} disabled={lan} onChange={(e) => setCfg({ ...cfg, require_auth: e.target.checked })} />
          Require the API token on /v1 endpoints too (the /api management endpoints always require it)
        </label>
        <div className="field">
          <label htmlFor="api-conc">Concurrent generations</label>
          <input id="api-conc" className="input mono" value={cfg.max_concurrent} onChange={(e) => setCfg({ ...cfg, max_concurrent: Number(e.target.value) || 0 })} />
        </div>
        <div className="field">
          <label htmlFor="api-rate">Requests per minute</label>
          <input id="api-rate" className="input mono" value={cfg.rate_per_minute} onChange={(e) => setCfg({ ...cfg, rate_per_minute: Number(e.target.value) || 0 })} />
        </div>
        <div className="field wide">
          <label htmlFor="api-cors">Allowed browser origins (CORS), one per line</label>
          <textarea id="api-cors" className="input mono" rows={2} placeholder="none: browsers on other sites are refused" value={origins} onChange={(e) => setOrigins(e.target.value)} />
          <span className="help">Requests from web pages whose origin is not listed are refused, so a website can't use your local model behind your back.</span>
        </div>
        <div className="field">
          <label htmlFor="api-ret">Keep request logs (days)</label>
          <input id="api-ret" className="input mono" value={cfg.log_retention_days} onChange={(e) => setCfg({ ...cfg, log_retention_days: Number(e.target.value) || 1 })} />
        </div>
        <label className="check" style={{ alignSelf: "end" }}>
          <input type="checkbox" checked={cfg.autostart} onChange={(e) => setCfg({ ...cfg, autostart: e.target.checked })} /> Start with the app
        </label>
        <div className="row wide">
          <button className="btn sm" disabled={busy} onClick={() => act(() => api2.apiSaveConfig(full))}>Save settings</button>
        </div>
      </div>

      <h2>API token</h2>
      <div className="row">
        <span className="dim">{hasToken ? "A token is stored in Windows Credential Manager." : "No token yet."}</span>
        <button className="btn sm" onClick={() => act(async () => setToken(await api2.apiGenerateToken()))}>{hasToken ? "Replace token" : "Generate token"}</button>
        {hasToken && !token && <button className="btn sm ghost" onClick={() => act(async () => setToken(await api2.apiRevealToken()))}>Show token</button>}
        {token && <button className="btn sm ghost" onClick={() => { navigator.clipboard.writeText(token); }}>Copy</button>}
        {token && <button className="btn sm ghost" onClick={() => setToken(null)}>Hide</button>}
      </div>
      {token && <div className="mono" style={{ marginTop: 6, overflowWrap: "anywhere" }}>{token}</div>}

      <h2>Endpoints</h2>
      <dl className="kv">
        <dt>GET</dt><dd>{base}/v1/models</dd>
        <dt>POST</dt><dd>{base}/v1/chat/completions (JSON or "stream": true)</dd>
        <dt>POST</dt><dd>{base}/v1/completions</dd>
        <dt>GET</dt><dd>{base}/api/models (token)</dd>
        <dt>POST</dt><dd>{base}/api/models/load {"{"}"model_id": "..."{"}"} (token)</dd>
        <dt>POST</dt><dd>{base}/api/models/unload (token)</dd>
      </dl>
      <div className="mono faint" style={{ marginTop: 8, overflowWrap: "anywhere" }}>{curl}</div>

      <h2>Recent requests</h2>
      {logs.length === 0 ? (
        <div className="empty">No requests yet.</div>
      ) : (
        <table className="table">
          <thead><tr><th>Time</th><th>Request</th><th>Status</th><th>Model</th><th className="num">Tokens in / out</th><th className="num">Latency</th></tr></thead>
          <tbody>
            {logs.map((l) => (
              <tr key={l.id}>
                <td className="dim">{when(l.created_at)}</td>
                <td className="mono">{l.method} {l.endpoint}</td>
                <td className={l.status >= 400 ? "danger-text mono" : "mono"} title={l.error ?? ""}>{l.status}{l.error ? ` ${l.error.slice(0, 60)}` : ""}</td>
                <td className="mono dim">{l.model ?? ""}</td>
                <td className="num">{l.prompt_tokens ?? ""}{l.prompt_tokens != null || l.completion_tokens != null ? " / " : ""}{l.completion_tokens ?? ""}</td>
                <td className="num">{l.latency_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

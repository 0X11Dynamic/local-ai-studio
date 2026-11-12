import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { errorMessage } from "../lib/api";
import { api3, BLANK_CONFIG, type GrantView, type ServerConfig, type ServerStatus, type ServerView, type ToolExecRow } from "../lib/api3";
import { when } from "../lib/format";
import { Editor } from "../components/Editor";

type Tab = "servers" | "permissions" | "activity";

function parseLines(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const i = t.indexOf("=");
    if (i <= 0) continue;
    out[t.slice(0, i).trim()] = t.slice(i + 1);
  }
  return out;
}

function headerLines(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const i = line.indexOf(":");
    if (i <= 0) continue;
    out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}

const STATE_LABEL: Record<ServerStatus["state"], string> = {
  disconnected: "Disconnected",
  connecting: "Connecting",
  connected: "Connected",
  error: "Error",
};

// --- auto-connect status (app start, import, turned on) ---
/** Short state for the server list, including background connect retries. */
function stateText(s: ServerView): string {
  if (!s.enabled) return "Turned off";
  const a = s.auto;
  if (a && s.status.state !== "connected") {
    if (a.gave_up) return "Error, gave up";
    if (a.retry_in_secs != null) return `Error, retry in ${a.retry_in_secs}s`;
    if (a.attempt > 1) return `Connecting (${a.attempt} of ${a.max_attempts})`;
  }
  return STATE_LABEL[s.status.state];
}

function AutoNote({ s }: { s: ServerView }) {
  const a = s.auto;
  if (!a || !s.enabled || s.status.state === "connected") return null;
  if (a.gave_up) return <div className="notice warn" role="status">Could not connect after {a.max_attempts} tries. Check the settings and server output, then press Connect.</div>;
  if (a.retry_in_secs != null) return <div className="notice" role="status">Connecting automatically: try {a.attempt} of {a.max_attempts} failed, next try in {a.retry_in_secs} s.</div>;
  return <div className="notice" role="status">Connecting automatically (try {Math.max(1, a.attempt)} of {a.max_attempts}).</div>;
}
// --- end auto-connect status ---

function hint(annotations: unknown): string | null {
  const a = annotations as Record<string, unknown> | null;
  if (!a || typeof a !== "object") return null;
  const parts: string[] = [];
  if (a.readOnlyHint === true) parts.push("read-only");
  if (a.destructiveHint === true) parts.push("destructive");
  if (a.openWorldHint === true) parts.push("reaches outside");
  return parts.length ? `server says: ${parts.join(", ")} (unverified)` : null;
}

interface Draft {
  id: string | null; name: string; enabled: boolean; config: ServerConfig;
  args: string; env: string; secretEnv: string; headers: string; secretHeaders: string; bearer: string; clearBearer: boolean;
}

function draftOf(s: ServerView | null): Draft {
  if (!s) return { id: null, name: "", enabled: true, config: { ...BLANK_CONFIG }, args: "", env: "", secretEnv: "", headers: "", secretHeaders: "", bearer: "", clearBearer: false };
  const c = s.config;
  return {
    id: s.id, name: s.name, enabled: s.enabled, config: c,
    args: c.args.join("\n"),
    env: Object.entries(c.env).map(([k, v]) => `${k}=${v}`).join("\n"),
    secretEnv: c.secret_env.map((k) => `${k}=`).join("\n"),
    headers: Object.entries(c.headers).map(([k, v]) => `${k}: ${v}`).join("\n"),
    secretHeaders: c.secret_headers.map((k) => `${k}: `).join("\n"),
    bearer: "", clearBearer: false,
  };
}

export function Mcp() {
  const [tab, setTab] = useState<Tab>("servers");
  const [servers, setServers] = useState<ServerView[] | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft>(draftOf(null));
  const [editing, setEditing] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [importText, setImportText] = useState<string | null>(null);
  const [exportText, setExportText] = useState<string | null>(null);
  const [toolFilter, setToolFilter] = useState("");
  const [grants, setGrants] = useState<GrantView[]>([]);
  const [activity, setActivity] = useState<ToolExecRow[]>([]);

  const reload = useCallback(async (keep?: string | null) => {
    const list = await api3.listServers();
    setServers(list);
    const id = keep === undefined ? selId : keep;
    const s = list.find((x) => x.id === id) ?? list[0] ?? null;
    setSelId(s?.id ?? null);
    return s;
  }, [selId]);

  useEffect(() => {
    reload();
    const un = listen<ServerStatus>("mcp://status", () => {
      api3.listServers().then(setServers);
    });
    return () => {
      un.then((f) => f());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // While a background connect is waiting to retry, refresh the countdown.
  const retrying = !!servers?.some((s) => s.auto && !s.auto.gave_up && s.status.state !== "connected");
  useEffect(() => {
    if (!retrying) return;
    const t = setInterval(() => api3.listServers().then(setServers).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, [retrying]);

  useEffect(() => {
    if (tab === "permissions") api3.grants().then(setGrants);
    if (tab === "activity") api3.activity(null, 300).then(setActivity);
  }, [tab]);

  const sel = servers?.find((s) => s.id === selId) ?? null;
  useEffect(() => {
    if (!editing) setDraft(draftOf(sel));
    if (sel) api3.logs(sel.id).then(setLogs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selId, sel?.status.state]);

  const run = async (fn: () => Promise<unknown>, ok?: string) => {
    setErr(null);
    setMsg(null);
    setBusy(true);
    try {
      await fn();
      if (ok) setMsg(ok);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const save = () =>
    run(async () => {
      const secretEnv = parseLines(draft.secretEnv);
      const secretHeaders = headerLines(draft.secretHeaders);
      const config: ServerConfig = {
        ...draft.config,
        args: draft.args.split("\n").map((a) => a.trim()).filter(Boolean),
        env: parseLines(draft.env),
        headers: headerLines(draft.headers),
        secret_env: Object.keys(secretEnv),
        secret_headers: Object.keys(secretHeaders),
      };
      const keepOrSet = (vals: Record<string, string>, stored: string[]) =>
        Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, v === "" && stored.includes(k) ? null : v]));
      const secrets = {
        env: keepOrSet(secretEnv, sel?.secret_env_set ?? []),
        headers: keepOrSet(secretHeaders, sel?.secret_headers_set ?? []),
        bearer: draft.clearBearer ? "" : draft.bearer ? draft.bearer : sel?.bearer_set ? null : "",
      };
      const v = draft.id
        ? await api3.updateServer(draft.id, draft.name, config, secrets, draft.enabled)
        : await api3.addServer(draft.name, config, secrets, draft.enabled);
      setEditing(false);
      await reload(v.id);
      // A new server that is turned on connects right away (in the background).
      if (!draft.id && v.enabled) api3.connect(v.id).then(() => reload(v.id)).catch(() => {});
    }, "Saved. Secret values are stored in Windows Credential Manager.");

  const doImport = (fn: () => Promise<{ added: string[]; skipped: string[]; secrets_stored: number }>) =>
    run(async () => {
      const r = await fn();
      setImportText(null);
      await reload();
      setMsg(`Imported ${r.added.length} server(s)${r.added.length ? `: ${r.added.join(", ")}` : ""}. ${r.skipped.length ? `Skipped: ${r.skipped.join("; ")}. ` : ""}${r.secrets_stored ? `${r.secrets_stored} secret value(s) moved to Credential Manager. ` : ""}Imported servers are turned on and connect in the background; every tool asks before it runs until you change its permission. Turn a server off here, or leave it out of a single chat from the chat's Tools menu.`);
    });

  const tools = useMemo(() => {
    const f = toolFilter.trim().toLowerCase();
    return (sel?.tools ?? []).filter((t) => !f || t.name.toLowerCase().includes(f) || t.description.toLowerCase().includes(f));
  }, [sel, toolFilter]);

  const setT = (patch: Partial<ServerConfig>) => setDraft({ ...draft, config: { ...draft.config, ...patch } });

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">MCP servers give the model tools in Agent and Plan mode. Servers that are turned on connect when the app starts; a chat can leave any of them out from its Tools menu. A connected server can run code on this PC or reach the network, so only add servers you trust. Every tool asks for your approval until you change its policy.</p>
      <div className="row" style={{ marginBottom: 14 }} role="tablist">
        {(["servers", "permissions", "activity"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={`btn sm${tab === t ? " primary" : " ghost"}`} onClick={() => setTab(t)}>
            {{ servers: "Servers", permissions: "Permissions", activity: "Tool activity" }[t]}
          </button>
        ))}
      </div>
      {err && <div className="notice error" style={{ marginBottom: 10 }}>{err}</div>}
      {msg && <div className="notice" style={{ marginBottom: 10 }}>{msg}</div>}

      {tab === "servers" && (
        <>
          <div className="row" style={{ marginBottom: 12 }}>
            <button className="btn sm primary" onClick={() => { setSelId(null); setEditing(true); setDraft(draftOf(null)); }}>Add server</button>
            <button className="btn sm" onClick={() => setImportText(importText == null ? "" : null)}>Paste mcpServers JSON</button>
            <button className="btn sm ghost" onClick={async () => { const p = await open({ multiple: false, filters: [{ name: "JSON", extensions: ["json"] }] }); if (p) doImport(() => api3.importFile(p)); }}>Import file</button>
            <button className="btn sm ghost" onClick={() => doImport(() => api3.importFrom("lmstudio"))}>Import from LM Studio</button>
            <button className="btn sm ghost" onClick={() => doImport(() => api3.importFrom("claude"))}>Import from Claude Desktop</button>
            <span className="spacer" />
            <button className="btn sm ghost" onClick={() => run(async () => setExportText(await api3.exportJson()))}>Export</button>
          </div>
          {importText != null && (
            <div className="panel" style={{ marginBottom: 12, display: "grid", gap: 8 }}>
              <Editor label="mcpServers JSON to import" rows={8} value={importText} onChange={setImportText} />
              <ImportCheck text={importText} />
              <div className="row">
                <button className="btn sm primary" disabled={!importText.trim() || busy} onClick={() => doImport(() => api3.importJson(importText))}>Import</button>
                <button className="btn sm ghost" onClick={() => setImportText(null)}>Cancel</button>
              </div>
            </div>
          )}
          {exportText != null && (
            <div className="panel" style={{ marginBottom: 12, display: "grid", gap: 8 }}>
              <span className="faint">Secret values are replaced with "&lt;secret&gt;".</span>
              <Editor label="Exported mcpServers JSON" rows={8} value={exportText} readOnly />
              <div className="row">
                <button className="btn sm" onClick={() => navigator.clipboard.writeText(exportText)}>Copy</button>
                <button className="btn sm ghost" onClick={() => setExportText(null)}>Close</button>
              </div>
            </div>
          )}

          <div className="instr-layout">
            <div className="instr-list">
              {servers == null ? (
                <div style={{ padding: 10 }}><div className="skeleton" /></div>
              ) : servers.length === 0 ? (
                <div className="empty" style={{ padding: 12 }}>No servers yet. Add one or import an mcpServers file.</div>
              ) : (
                servers.map((s) => (
                  <button key={s.id} className="chat-item" aria-current={s.id === selId && !(!draft.id && editing) ? "true" : undefined} onClick={() => { setEditing(false); setSelId(s.id); }}>
                    <span className="chat-title">{s.name}</span>
                    <span className={`mono ${s.status.state === "error" ? "danger-text" : "faint"}`} style={{ fontSize: 10.5 }}>
                      {stateText(s)}{s.status.state === "connected" ? `, ${s.tools.length} tools` : ""}
                    </span>
                  </button>
                ))
              )}
            </div>

            <div style={{ display: "grid", gap: 14, minWidth: 0 }}>
              {(editing || !sel) && (editing || (servers?.length ?? 0) > 0) ? (
                <div className="panel" style={{ display: "grid", gap: 12 }}>
                  <strong style={{ fontWeight: 600 }}>{draft.id ? `Edit ${draft.name}` : "New server"}</strong>
                  <div className="form-grid">
                    <div className="field">
                      <label htmlFor="mcp-name">Name</label>
                      <input id="mcp-name" className="input" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
                    </div>
                    <div className="field">
                      <label htmlFor="mcp-transport">Connection</label>
                      <select id="mcp-transport" className="input" value={draft.config.transport} onChange={(e) => setT({ transport: e.target.value as "stdio" | "http" })}>
                        <option value="stdio">Program on this PC (stdio)</option>
                        <option value="http">HTTP server (Streamable HTTP)</option>
                      </select>
                    </div>
                    {draft.config.transport === "stdio" ? (
                      <>
                        <div className="field wide">
                          <label htmlFor="mcp-cmd">Command</label>
                          <input id="mcp-cmd" className="input mono" placeholder="npx" value={draft.config.command} onChange={(e) => setT({ command: e.target.value })} />
                        </div>
                        <div className="field wide">
                          <label htmlFor="mcp-args">Arguments (one per line)</label>
                          <textarea id="mcp-args" className="input mono" rows={3} value={draft.args} onChange={(e) => setDraft({ ...draft, args: e.target.value })} />
                        </div>
                        <div className="field wide">
                          <label htmlFor="mcp-cwd">Working folder (optional)</label>
                          <input id="mcp-cwd" className="input mono" value={draft.config.cwd ?? ""} onChange={(e) => setT({ cwd: e.target.value || null })} />
                        </div>
                        <div className="field">
                          <label htmlFor="mcp-env">Environment (KEY=value per line)</label>
                          <textarea id="mcp-env" className="input mono" rows={3} value={draft.env} onChange={(e) => setDraft({ ...draft, env: e.target.value })} />
                        </div>
                        <div className="field">
                          <label htmlFor="mcp-senv">Secret environment (KEY=value)</label>
                          <textarea id="mcp-senv" className="input mono" rows={3} placeholder="API_KEY=..." value={draft.secretEnv} onChange={(e) => setDraft({ ...draft, secretEnv: e.target.value })} />
                          <span className="help">Stored in Credential Manager. Leave a value empty to keep the stored one{sel?.secret_env_set.length ? ` (${sel.secret_env_set.join(", ")} stored)` : ""}.</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="field wide">
                          <label htmlFor="mcp-url">URL</label>
                          <input id="mcp-url" className="input mono" placeholder="https://example.com/mcp" value={draft.config.url} onChange={(e) => setT({ url: e.target.value })} />
                          <span className="help">Remote servers must use https. Plain http is allowed only for this PC.</span>
                        </div>
                        <div className="field">
                          <label htmlFor="mcp-headers">Headers (Name: value per line)</label>
                          <textarea id="mcp-headers" className="input mono" rows={3} value={draft.headers} onChange={(e) => setDraft({ ...draft, headers: e.target.value })} />
                        </div>
                        <div className="field">
                          <label htmlFor="mcp-shead">Secret headers (Name: value)</label>
                          <textarea id="mcp-shead" className="input mono" rows={3} value={draft.secretHeaders} onChange={(e) => setDraft({ ...draft, secretHeaders: e.target.value })} />
                        </div>
                        <div className="field wide">
                          <label htmlFor="mcp-bearer">Bearer token</label>
                          <input id="mcp-bearer" className="input mono" type="password" autoComplete="off" placeholder={sel?.bearer_set ? "A token is stored; type to replace" : "none"} value={draft.bearer} onChange={(e) => setDraft({ ...draft, bearer: e.target.value })} />
                          {sel?.bearer_set && <label className="check"><input type="checkbox" checked={draft.clearBearer} onChange={(e) => setDraft({ ...draft, clearBearer: e.target.checked })} /> Remove the stored token</label>}
                        </div>
                      </>
                    )}
                    <div className="field">
                      <label htmlFor="mcp-timeout">Tool call timeout (seconds)</label>
                      <input id="mcp-timeout" className="input mono" value={draft.config.timeout_secs} onChange={(e) => setT({ timeout_secs: Number(e.target.value) || 0 })} />
                    </div>
                    <label className="check" style={{ alignSelf: "end" }}>
                      <input type="checkbox" checked={draft.enabled} onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })} /> Turned on (connects when the app starts)
                    </label>
                  </div>
                  <div className="row">
                    <button className="btn sm primary" disabled={busy || !draft.name.trim()} onClick={save}>Save</button>
                    {(editing && (draft.id || (servers?.length ?? 0) > 0)) && <button className="btn sm ghost" onClick={() => { setEditing(false); setDraft(draftOf(sel)); }}>Cancel</button>}
                  </div>
                </div>
              ) : null}

              {sel && !editing && (
                <>
                  <div className="panel" style={{ display: "grid", gap: 10 }}>
                    <div className="row">
                      <strong style={{ fontWeight: 600 }}>{sel.name}</strong>
                      <span className={`tag${sel.status.state === "connected" ? " accent" : sel.status.state === "error" ? " danger" : ""}`}>{stateText(sel)}</span>
                      {sel.status.server_name && <span className="faint mono">{sel.status.server_name} {sel.status.server_version}</span>}
                      <span className="spacer" />
                      {sel.status.state === "connected" ? (
                        <>
                          <button className="btn sm" disabled={busy} onClick={() => run(() => api3.connect(sel.id).then(() => reload(sel.id)), "Reconnected.")}>Reconnect</button>
                          <button className="btn sm" disabled={busy} onClick={() => run(() => api3.disconnect(sel.id).then(() => reload(sel.id)))}>Disconnect</button>
                        </>
                      ) : (
                        <button className="btn sm primary" disabled={busy || !sel.enabled || sel.status.state === "connecting"} onClick={() => run(async () => { const v = await api3.connect(sel.id); await reload(sel.id); if (v.status.state === "error") throw new Error(v.status.error ?? "connection failed"); })}>
                          {sel.status.state === "connecting" ? "Connecting" : "Connect"}
                        </button>
                      )}
                      <label className="check mcp-switch" title="Turned-on servers connect when the app starts. Turning one off disconnects it for every chat.">
                        <input type="checkbox" role="switch" aria-label={`${sel.name} turned on`} checked={sel.enabled} disabled={busy}
                          onChange={(e) => { const on = e.target.checked; run(async () => { await api3.updateServer(sel.id, sel.name, sel.config, {}, on); await reload(sel.id); }); }} />
                        On
                      </label>
                      <button className="btn sm ghost" onClick={() => setEditing(true)}>Edit</button>
                      <button className="btn sm ghost danger" onClick={() => run(async () => { await api3.deleteServer(sel.id); await reload(null); }, "Server removed, with its tool settings and grants.")}>Remove</button>
                    </div>
                    <div className="mono faint" style={{ overflowWrap: "anywhere" }}>
                      {sel.config.transport === "http" ? sel.config.url : [sel.config.command, ...sel.config.args].join(" ")}
                    </div>
                    {sel.status.error && <div className="notice error">{sel.status.error}</div>}
                    {!sel.enabled && <div className="notice">This server is turned off: it does not connect and no chat gets its tools. Turn it on to connect.</div>}
                    <AutoNote s={sel} />
                  </div>

                  {sel.status.state === "connected" && (
                    <div>
                      <div className="row" style={{ marginBottom: 6 }}>
                        <h2 style={{ margin: 0 }}>Tools</h2>
                        <span className="faint">{sel.tools.length} offered by the server</span>
                        <span className="spacer" />
                        <input className="input" placeholder="Search tools" aria-label="Search tools" value={toolFilter} onChange={(e) => setToolFilter(e.target.value)} style={{ width: 220 }} />
                      </div>
                      <table className="table">
                        <thead><tr><th>Tool</th><th>On</th><th>Permission</th></tr></thead>
                        <tbody>
                          {tools.map((t) => (
                            <tr key={t.name}>
                              <td>
                                <div className="mono">{t.name}</div>
                                <div className="dim" style={{ fontSize: 12, maxWidth: 640 }}>{t.description.slice(0, 300)}</div>
                                {hint(t.annotations) && <div className="faint" style={{ fontSize: 11 }}>{hint(t.annotations)}</div>}
                                <details style={{ marginTop: 4 }}>
                                  <summary className="faint" style={{ fontSize: 11, cursor: "pointer" }}>Input schema and try it</summary>
                                  <pre className="preview-text">{JSON.stringify(t.input_schema, null, 2)}</pre>
                                  <TryTool serverId={sel.id} tool={t.name} disabled={!t.enabled || t.policy === "deny"} />
                                </details>
                              </td>
                              <td>
                                <input type="checkbox" aria-label={`Enable ${t.name}`} checked={t.enabled}
                                  onChange={(e) => run(async () => { await api3.setToolPref(sel.id, t.name, e.target.checked, t.policy); await reload(sel.id); })} style={{ accentColor: "var(--accent)" }} />
                              </td>
                              <td>
                                <select className="input" aria-label={`Permission for ${t.name}`} value={t.policy}
                                  onChange={(e) => run(async () => { await api3.setToolPref(sel.id, t.name, t.enabled, e.target.value); await reload(sel.id); })}>
                                  <option value="ask">Ask every time</option>
                                  <option value="allow">Allow without asking</option>
                                  <option value="deny">Never (hidden from the model)</option>
                                </select>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {(sel.status.resources.length > 0 || sel.status.prompts.length > 0) && (
                        <p className="faint">Also offers {sel.status.resources.length} resource(s) and {sel.status.prompts.length} prompt(s).</p>
                      )}
                    </div>
                  )}

                  <div>
                    <div className="row" style={{ marginBottom: 6 }}>
                      <h2 style={{ margin: 0 }}>Server output</h2>
                      <button className="btn sm ghost" onClick={() => api3.logs(sel.id).then(setLogs)}>Refresh</button>
                    </div>
                    <div className="log" style={{ height: 160 }}>{logs.length ? logs.join("\n") : <span className="faint">No output yet.</span>}</div>
                  </div>
                </>
              )}
            </div>
          </div>
        </>
      )}

      {tab === "permissions" && (
        <div>
          <p className="faint" style={{ marginTop: 0 }}>Tools you allowed when asked. Revoking one makes the tool ask again.</p>
          {grants.length === 0 ? (
            <div className="empty">No grants. Tools ask for approval until you choose "Allow for this chat", "Allow for this task" or "Always allow".</div>
          ) : (
            <table className="table">
              <thead><tr><th>Tool</th><th>Server or environment</th><th>Scope</th><th>Granted</th><th /></tr></thead>
              <tbody>
                {grants.map((g) => (
                  <tr key={g.id}>
                    <td className="mono">{grantTool(g)}</td>
                    <td>{grantWhere(g)}</td>
                    <td>{grantScope(g)}</td>
                    <td className="dim">{when(g.created_at)}</td>
                    <td><button className="btn sm ghost danger" onClick={() => run(async () => { await api3.revokeGrant(g.id); setGrants(await api3.grants()); }, "Revoked.")}>Revoke</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "activity" && (
        <div>
          <div className="row" style={{ marginBottom: 8 }}>
            <p className="faint" style={{ margin: 0 }}>Every tool call, with its arguments (secrets redacted), the permission decision and the result.</p>
            <span className="spacer" />
            <button className="btn sm ghost" onClick={() => api3.activity(null, 300).then(setActivity)}>Refresh</button>
          </div>
          {activity.length === 0 ? (
            <div className="empty">No tool calls yet.</div>
          ) : (
            <table className="table">
              <thead><tr><th>Time</th><th>Tool</th><th>Status</th><th>Decision</th><th className="num">Duration</th></tr></thead>
              <tbody>
                {activity.map((a) => {
                  const ap = (() => { try { return JSON.parse(a.approval); } catch { return {}; } })();
                  return (
                    <tr key={a.id}>
                      <td className="dim" style={{ whiteSpace: "nowrap" }}>{when(a.started_at)}</td>
                      <td>
                        <div className="mono">{a.model_tool_name ?? a.tool}</div>
                        <details>
                          <summary className="faint" style={{ fontSize: 11, cursor: "pointer" }}>Arguments and result</summary>
                          <pre className="preview-text">{a.args_redacted}</pre>
                          {a.result && <pre className="preview-text">{a.result.slice(0, 4000)}</pre>}
                        </details>
                      </td>
                      <td className={["denied", "error", "timeout", "invalid_arguments"].includes(a.status) ? "danger-text mono" : "mono"}>{a.status}</td>
                      <td className="dim">{ap.answer ?? ap.decision ?? ""}{ap.reason ? `: ${ap.reason}` : ""}</td>
                      <td className="num">{a.duration_ms != null ? `${a.duration_ms} ms` : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

/** Validate pasted JSON as you type. */
function ImportCheck({ text }: { text: string }) {
  if (!text.trim()) return <span className="faint">Paste a document like {`{ "mcpServers": { "name": { "command": "npx", "args": [...] } } }`}.</span>;
  try {
    const v = JSON.parse(text);
    const map = v.mcpServers ?? v.servers ?? v;
    const names = Object.keys(map ?? {});
    return <span className="ok-text">Valid JSON with {names.length} server(s): {names.join(", ")}</span>;
  } catch (e) {
    return <span className="danger-text">Not valid JSON yet: {(e as Error).message}</span>;
  }
}

/** Run a tool by hand through the same checks as a model's call. */
// Grant subjects: "mcp:<server>/<tool>" or "builtin:<environment>/<tool>".
const builtin = (g: GrantView) => (g.subject.startsWith("builtin:") ? g.subject.slice(8).split("/") : null);
function grantTool(g: GrantView): string {
  return g.tool_name ?? builtin(g)?.[1] ?? g.subject;
}
function grantWhere(g: GrantView): string {
  if (g.server_name) return g.server_name;
  const b = builtin(g);
  if (b) return b[0] === "local" ? "This PC (built-in tools)" : `Environment ${b[0]} (built-in tools)`;
  return "removed server";
}
function grantScope(g: GrantView): string {
  if (g.scope === "persistent") return "Always";
  if (g.run_id) return `This task: ${g.task_title || "(deleted task)"}`;
  return `This chat: ${g.conversation_title ?? g.conversation_id ?? "(deleted chat)"}`;
}

function TryTool({ serverId, tool, disabled }: { serverId: string; tool: string; disabled: boolean }) {
  const [args, setArgs] = useState("{}");
  const [out, setOut] = useState<{ status: string; content: string } | null>(null);
  const [running, setRunning] = useState(false);
  const [pending, setPending] = useState<{ id: string; reason: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      // Only this box's own manual call: never a chat's or an agent task's.
      const p = (await api3.pendingApprovals()).find(
        (x) => !x.run_id && x.conversation_id === null && x.call_id.startsWith("manual-") && x.tool_id === `mcp:${serverId}/${tool}`,
      );
      setPending(p ? { id: p.id, reason: p.reason } : null);
    }, 300);
    return () => clearInterval(t);
  }, [running, tool, serverId]);

  const run = async () => {
    setErr(null);
    setOut(null);
    let parsed: unknown;
    try {
      parsed = JSON.parse(args || "{}");
    } catch (e) {
      setErr(`Arguments are not valid JSON: ${(e as Error).message}`);
      return;
    }
    setRunning(true);
    try {
      setOut(await api3.runToolManually(serverId, tool, parsed));
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setRunning(false);
      setPending(null);
    }
  };

  return (
    <div style={{ display: "grid", gap: 6, marginTop: 6 }}>
      <textarea className="input mono" rows={3} aria-label={`Arguments for ${tool}`} value={args} onChange={(e) => setArgs(e.target.value)} />
      <div className="row">
        <button className="btn sm" disabled={disabled || running} onClick={run}>{running ? "Running" : "Run tool"}</button>
        <span className="faint" style={{ fontSize: 11 }}>Uses this tool's permission setting, like a model's call would.</span>
      </div>
      {pending && (
        <div className="row">
          <span style={{ color: "var(--accent)" }}>Waiting for approval ({pending.reason}).</span>
          <button className="btn sm primary" onClick={() => api3.answerApproval(pending.id, "allow_once")}>Allow once</button>
          <button className="btn sm danger" onClick={() => api3.answerApproval(pending.id, "deny")}>Deny</button>
        </div>
      )}
      {err && <div className="danger-text">{err}</div>}
      {out && (
        <div>
          <span className={out.status === "ok" ? "ok-text" : "danger-text"}>{out.status}</span>
          <pre className="preview-text" aria-label="Tool output">{out.content}</pre>
        </div>
      )}
    </div>
  );
}

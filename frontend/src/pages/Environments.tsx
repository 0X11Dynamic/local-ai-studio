import { useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { errorMessage } from "../lib/api";
import { api3, type ApprovalRequest } from "../lib/api3";
import {
  api4, BLANK_DOCKER, BLANK_SSH, KIND_LABEL,
  type DockerConfig, type EnvView, type HostKey, type Mount, type Policy, type SshProfile,
} from "../lib/api4";

type Draft =
  | { kind: "ssh"; id: string | null; name: string; profile: SshProfile; roots: string; password: string; passphrase: string }
  | { kind: "docker"; id: string | null; name: string; config: DockerConfig; mounts: string }
  | { kind: "sandbox"; name: string; network: boolean };

const lines = (s: string) => s.split("\n").map((l) => l.trim()).filter(Boolean);

function mountsToText(m: Mount[]): string {
  return m.map((x) => `${x.host} -> ${x.container}${x.read_only ? " (read-only)" : ""}`).join("\n");
}
function parseMounts(text: string): Mount[] {
  return lines(text).map((l) => {
    const read_only = /\(read-only\)\s*$/i.test(l);
    const [host, container] = l.replace(/\(read-only\)\s*$/i, "").split("->").map((x) => x.trim());
    return { host: host ?? "", container: container ?? "", read_only };
  });
}

function draftFor(e: EnvView): Draft | null {
  if (e.kind === "ssh") {
    const p = { ...BLANK_SSH, ...(e.config as Partial<SshProfile>) };
    return { kind: "ssh", id: e.id, name: e.name, profile: p, roots: p.roots.join("\n"), password: "", passphrase: "" };
  }
  if (e.kind === "docker") {
    const c = { ...BLANK_DOCKER, ...(e.config as Partial<DockerConfig>) };
    return { kind: "docker", id: e.id, name: e.name, config: c, mounts: mountsToText(c.mounts) };
  }
  return null;
}

function HealthLine({ h }: { h: { ok: boolean; detail: string } | null | undefined }) {
  if (!h) return <span className="faint">Not checked yet.</span>;
  return <span className={h.ok ? "ok-text" : "danger-text"}>{h.ok ? "Working" : "Not working"}: <span className="mono">{h.detail}</span></span>;
}

export function Environments() {
  const [envs, setEnvs] = useState<EnvView[] | null>(null);
  const [selId, setSelId] = useState<string>("local");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [roots, setRoots] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [health, setHealth] = useState<Record<string, { ok: boolean; detail: string }>>({});
  const [hostKey, setHostKey] = useState<HostKey | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [tryCmd, setTryCmd] = useState("");
  const [tryOut, setTryOut] = useState<{ status: string; content: string } | null>(null);
  const [tryApproval, setTryApproval] = useState<ApprovalRequest | null>(null);

  const refresh = async () => {
    const l = await api4.list();
    setEnvs(l);
    return l;
  };
  useEffect(() => {
    refresh().catch((e) => setErr(errorMessage(e)));
  }, []);

  // "Try a command" goes through the normal approval flow; show its prompt here.
  useEffect(() => {
    // Only this page's own manual run: not a chat's, not an agent task's.
    const a = listen<ApprovalRequest>("tool://approval", (e) => {
      const p = e.payload;
      if (!p.run_id && p.conversation_id === null && p.call_id.startsWith("manual-") && p.tool_name === "shell_execute") setTryApproval(p);
    });
    const c = listen<{ id: string }>("tool://approval-closed", (e) => setTryApproval((cur) => (cur && cur.id === e.payload.id ? null : cur)));
    return () => {
      a.then((f) => f());
      c.then((f) => f());
    };
  }, []);

  const sel = useMemo(() => envs?.find((e) => e.id === selId) ?? null, [envs, selId]);
  useEffect(() => {
    setConfirmRemove(false);
    setHostKey(null);
    setTryOut(null);
    if (sel?.kind === "local") setRoots(((sel.config.roots as string[]) ?? []).join("\n"));
    if (sel && (draft == null || ("id" in draft && draft.id !== null && draft.id !== sel.id))) setDraft(draftFor(sel));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel?.id]);

  const run = async (label: string, fn: () => Promise<unknown>, ok?: string) => {
    setErr(null);
    setMsg(null);
    setBusy(label);
    try {
      await fn();
      if (ok) setMsg(ok);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const check = (id: string) =>
    run("check", async () => {
      const h = await api4.health(id);
      setHealth((x) => ({ ...x, [id]: h }));
      await refresh();
    });

  const saveDraft = () =>
    run("save", async () => {
      if (!draft) return;
      if (draft.kind === "ssh") {
        const profile = { ...draft.profile, roots: lines(draft.roots), port: Number(draft.profile.port) || 22 };
        const secrets = draft.profile.auth === "password" ? { password: draft.password || null } : { passphrase: draft.passphrase || null };
        const changedSecret = !!(draft.password || draft.passphrase);
        if (draft.id) await api4.updateSsh(draft.id, draft.name, profile, changedSecret ? secrets : null);
        else setSelId(await api4.addSsh(draft.name, profile, secrets));
      } else if (draft.kind === "docker") {
        const config = { ...draft.config, mounts: parseMounts(draft.mounts) };
        if (draft.id) await api4.updateDocker(draft.id, draft.name, config);
        else setSelId(await api4.addDocker(draft.name, config));
      } else {
        setSelId(await api4.createSandbox(draft.name, draft.network));
      }
      setDraft(null);
      await refresh();
    }, draft?.kind === "sandbox" ? "Sandbox created." : "Saved.");

  const tryRun = () =>
    run("try", async () => {
      if (!sel) return;
      setTryOut(null);
      const r = await api4.runTool(sel.id, "shell_execute", { command: tryCmd });
      setTryOut(r);
    });

  const newDraft = (kind: Draft["kind"]) => {
    setSelId("");
    setMsg(null);
    setErr(null);
    if (kind === "ssh") setDraft({ kind, id: null, name: "", profile: { ...BLANK_SSH }, roots: "", password: "", passphrase: "" });
    if (kind === "docker") setDraft({ kind, id: null, name: "", config: { ...BLANK_DOCKER }, mounts: "" });
    if (kind === "sandbox") setDraft({ kind, name: "Sandbox", network: false });
  };

  const editingNew = draft != null && (draft.kind === "sandbox" || draft.id === null);

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">
        Where the agent's shell and file tools run. Each chat in Agent Mode picks one environment in Chat settings. Permissions are per environment: allowing the shell in a sandbox never allows it on this PC.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn sm" onClick={() => newDraft("sandbox")}>New sandbox</button>
        <button className="btn sm" onClick={() => newDraft("ssh")}>Add SSH host</button>
        <button className="btn sm" onClick={() => newDraft("docker")}>Add Docker</button>
      </div>
      {err && <div className="notice error" style={{ marginBottom: 10 }} role="alert">{err}</div>}
      {msg && <div className="notice" style={{ marginBottom: 10 }}>{msg}</div>}

      <div className="instr-layout">
        <div className="instr-list" aria-label="Environments">
          {envs == null ? (
            <div style={{ padding: 10 }}><div className="skeleton" /></div>
          ) : (
            envs.map((e) => {
              const h = health[e.id] ?? e.last_health;
              return (
                <button key={e.id} className="chat-item" aria-current={e.id === selId && !editingNew ? "true" : undefined} onClick={() => { setDraft(draftFor(e)); setSelId(e.id); }}>
                  <span className="chat-title">{e.name}</span>
                  <span className={`mono ${h && !h.ok ? "danger-text" : "faint"}`} style={{ fontSize: 10.5 }}>
                    {e.kind === "local" ? "PowerShell, your folders" : KIND_LABEL[e.kind]}{h ? (h.ok ? ", working" : ", not working") : ""}
                  </span>
                </button>
              );
            })
          )}
        </div>

        <div style={{ display: "grid", gap: 14, minWidth: 0, alignContent: "start" }}>
          {editingNew && draft && (
            <div className="panel" style={{ display: "grid", gap: 12 }}>
              <strong style={{ fontWeight: 600 }}>
                {draft.kind === "sandbox" ? "New sandbox" : draft.kind === "ssh" ? "New SSH host" : "New Docker environment"}
              </strong>
              <DraftForm draft={draft} setDraft={setDraft} />
              <div className="row">
                <button className="btn sm primary" disabled={!!busy || !draft.name.trim()} onClick={saveDraft}>
                  {draft.kind === "sandbox" ? (busy === "save" ? "Creating..." : "Create sandbox") : "Add"}
                </button>
                <button className="btn sm ghost" onClick={() => { setDraft(null); setSelId("local"); }}>Cancel</button>
                {draft.kind === "sandbox" && busy === "save" && <span className="faint">Downloads the Alpine Linux image (about 4 MB) the first time. Takes about 20 seconds.</span>}
              </div>
            </div>
          )}

          {!editingNew && sel && (
            <>
              <div className="panel" style={{ display: "grid", gap: 10 }}>
                <div className="row">
                  <strong style={{ fontWeight: 600 }}>{sel.name}</strong>
                  {sel.kind !== "local" && <span className="tag">{KIND_LABEL[sel.kind]}</span>}
                  <span className="faint mono" style={{ fontSize: 12 }}>{sel.label}</span>
                  <span className="spacer" />
                  <button className="btn sm" disabled={!!busy} onClick={() => check(sel.id)}>{busy === "check" ? "Checking..." : "Check connection"}</button>
                  {sel.kind !== "local" && !confirmRemove && <button className="btn sm ghost danger" onClick={() => setConfirmRemove(true)}>Remove</button>}
                </div>
                <HealthLine h={health[sel.id] ?? sel.last_health} />
                {confirmRemove && (
                  <div className="notice warn row">
                    <span>
                      {sel.kind === "sandbox" ? "This deletes the sandbox and everything in it." : sel.kind === "docker" ? "This removes the container and everything installed in it." : "This removes the host from the app (nothing on the server changes)."}
                    </span>
                    <span className="spacer" />
                    <button className="btn sm danger" disabled={!!busy} onClick={() => run("remove", async () => { await api4.remove(sel.id); setSelId("local"); await refresh(); }, "Removed.")}>Remove</button>
                    <button className="btn sm ghost" onClick={() => setConfirmRemove(false)}>Keep</button>
                  </div>
                )}
              </div>

              {sel.kind === "local" && (
                <div className="panel" style={{ display: "grid", gap: 8 }}>
                  <strong style={{ fontWeight: 600 }}>Allowed folders</strong>
                  <span className="help faint">Shell commands start in the first folder. Inside these folders each tool's permission below applies. Windows, Program Files, ProgramData\Microsoft, .ssh, Credential folders, the app's own data and your model folders are always off limits, even if you approve.</span>
                  <textarea className="input mono" rows={4} aria-label="Allowed folders, one per line" value={roots} onChange={(e) => setRoots(e.target.value)} />
                  <div className="row">
                    <button className="btn sm primary" disabled={!!busy} onClick={() => run("roots", async () => { await api4.setLocalRoots(lines(roots)); await refresh(); }, "Folders saved.")}>Save folders</button>
                    <button className="btn sm ghost" onClick={async () => { const p = await open({ directory: true }); if (p) setRoots((r) => (r.trim() ? `${r.trim()}\n${p}` : String(p))); }}>Add a folder...</button>
                    <span className="faint mono" style={{ fontSize: 12 }}>Shell: {String(sel.config.shell ?? "powershell.exe")}</span>
                  </div>
                  {/* reach outside the allowed folders (This PC only) */}
                  <div className="row" style={{ gap: 8 }}>
                    <label htmlFor="reach-outside" style={{ fontWeight: 600 }}>Reach outside allowed folders</label>
                    <select id="reach-outside" className="input" style={{ width: "auto" }} disabled={!!busy} value={String(sel.config.outside ?? "ask")}
                      onChange={(e) => { const mode = e.target.value as "ask" | "off"; run("outside", async () => { await api4.setLocalOutside(mode); await refresh(); }, mode === "ask" ? "Paths outside the folders now ask each time." : "Tools now stay inside the allowed folders."); }}>
                      <option value="ask">Ask each time</option>
                      <option value="off">Off (allowed folders only)</option>
                    </select>
                  </div>
                  <span className="help faint">
                    {String(sel.config.outside ?? "ask") === "ask"
                      ? "File tools and shell commands may use other paths on this PC, but every call outside the folders asks you first, whatever the tool's permission says, and the approval names the path. \"Allow for this chat\" then covers that tool in that folder for that chat only. A command can change files anywhere, so check commands before allowing them. Deletes still go to the app's trash."
                      : "File tools refuse paths outside the allowed folders, and shell commands must start inside them."}
                  </span>
                </div>
              )}

              {sel.kind === "sandbox" && (
                <div className="panel" style={{ display: "grid", gap: 6 }}>
                  <strong style={{ fontWeight: 600 }}>Disposable Linux sandbox</strong>
                  <span className="dim">Alpine Linux on WSL 2, its own distro. It cannot see your Windows drives or start Windows programs; commands run as an unprivileged user. Network: <strong>{(sel.config.network as boolean) ? "on" : "off"}</strong> (set when created).</span>
                </div>
              )}

              {(sel.kind === "ssh" || sel.kind === "docker") && draft && "id" in draft && draft.id === sel.id && (
                <div className="panel" style={{ display: "grid", gap: 12 }}>
                  <strong style={{ fontWeight: 600 }}>Settings</strong>
                  <DraftForm draft={draft} setDraft={setDraft} hasSecret={sel.has_secret} />
                  <div className="row">
                    <button className="btn sm primary" disabled={!!busy || !draft.name.trim()} onClick={saveDraft}>Save</button>
                    {draft.kind === "ssh" && (
                      <button className="btn sm" disabled={!!busy} onClick={() => run("key", async () => setHostKey(await api4.hostKey(draft.profile.host, Number(draft.profile.port) || 22)))}>Check host key</button>
                    )}
                    {draft.kind === "docker" && (
                      <button className="btn sm" disabled={!!busy} onClick={() => run("docker", async () => setMsg(await api4.dockerStatus(draft.config.host)))}>Check Docker</button>
                    )}
                  </div>
                  {hostKey && draft.kind === "ssh" && (
                    <div className="notice" style={{ display: "grid", gap: 6 }}>
                      <span>Server key: <span className="mono">{hostKey.algorithm} {hostKey.fingerprint}</span></span>
                      {hostKey.trusted ? (
                        <span className="ok-text">Already trusted.</span>
                      ) : (
                        <>
                          <span className="dim">Compare this with the server's own fingerprint (on the server: <span className="mono">ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub</span>) before trusting it.</span>
                          <div className="row">
                            <button className="btn sm primary" disabled={!!busy} onClick={() => run("trust", async () => { await api4.trustHost(draft.profile.host, Number(draft.profile.port) || 22, hostKey.fingerprint); setHostKey({ ...hostKey, trusted: true }); }, "Host key trusted.")}>It matches: trust this key</button>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="panel" style={{ display: "grid", gap: 8 }}>
                <strong style={{ fontWeight: 600 }}>Built-in tools</strong>
                <table className="table">
                  <thead>
                    <tr><th>Tool</th><th>What it does</th><th>On</th><th>Permission</th></tr>
                  </thead>
                  <tbody>
                    {sel.tools.map((t) => (
                      <tr key={t.name}>
                        <td className="mono" style={{ whiteSpace: "nowrap" }}>{t.name}</td>
                        <td className="dim" style={{ fontSize: 12.5 }}>{t.description}</td>
                        <td>
                          <input type="checkbox" aria-label={`${t.name} on`} checked={t.enabled}
                            onChange={(e) => run("pref", async () => { await api4.setToolPref(sel.id, t.name, e.target.checked, t.policy); await refresh(); })} />
                        </td>
                        <td>
                          <select className="input" aria-label={`${t.name} permission`} value={t.policy}
                            onChange={(e) => run("pref", async () => { await api4.setToolPref(sel.id, t.name, t.enabled, e.target.value as Policy); await refresh(); })}>
                            <option value="allow">Runs without asking</option>
                            <option value="ask">Ask every time</option>
                            <option value="deny">Never</option>
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="panel" style={{ display: "grid", gap: 8 }}>
                <strong style={{ fontWeight: 600 }}>Try a command</strong>
                <span className="help faint">Runs through the same permission check as the agent and is logged in Tool activity.</span>
                <div className="row">
                  <input className="input mono" style={{ flex: 1 }} aria-label="Command to try" placeholder={sel.kind === "local" ? "Get-ChildItem" : "ls -la"} value={tryCmd}
                    onChange={(e) => setTryCmd(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && tryCmd.trim()) tryRun(); }} />
                  <button className="btn sm primary" disabled={!!busy || !tryCmd.trim()} onClick={tryRun}>{busy === "try" ? "Running..." : "Run"}</button>
                </div>
                {tryApproval && (
                  <div className="notice row">
                    <span>Allow running this command on {tryApproval.server_name}?</span>
                    <span className="spacer" />
                    <button className="btn sm primary" onClick={() => api3.answerApproval(tryApproval.id, "allow_once")}>Allow once</button>
                    <button className="btn sm danger" onClick={() => api3.answerApproval(tryApproval.id, "deny")}>Deny</button>
                  </div>
                )}
                {tryOut && <pre className="preview-text" aria-label="Command output">{tryOut.content}</pre>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function DraftForm({ draft, setDraft, hasSecret }: { draft: Draft; setDraft: (d: Draft) => void; hasSecret?: boolean }) {
  if (draft.kind === "sandbox") {
    return (
      <div className="form-grid">
        <div className="field">
          <label htmlFor="sb-name">Name</label>
          <input id="sb-name" className="input" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
        </div>
        <label className="check" style={{ alignSelf: "end" }}>
          <input type="checkbox" checked={draft.network} onChange={(e) => setDraft({ ...draft, network: e.target.checked })} /> Network access (off is safer)
        </label>
      </div>
    );
  }
  if (draft.kind === "ssh") {
    const p = draft.profile;
    const setP = (patch: Partial<SshProfile>) => setDraft({ ...draft, profile: { ...p, ...patch } });
    return (
      <div className="form-grid">
        <div className="field">
          <label htmlFor="ssh-name">Name</label>
          <input id="ssh-name" className="input" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
        </div>
        <div className="field">
          <label htmlFor="ssh-host">Host</label>
          <input id="ssh-host" className="input mono" placeholder="192.168.1.20" value={p.host} onChange={(e) => setP({ host: e.target.value.trim() })} />
        </div>
        <div className="field">
          <label htmlFor="ssh-port">Port</label>
          <input id="ssh-port" className="input mono" value={p.port} onChange={(e) => setP({ port: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
        </div>
        <div className="field">
          <label htmlFor="ssh-user">User</label>
          <input id="ssh-user" className="input mono" value={p.user} onChange={(e) => setP({ user: e.target.value.trim() })} />
        </div>
        <div className="field">
          <label htmlFor="ssh-auth">Sign in with</label>
          <select id="ssh-auth" className="input" value={p.auth} onChange={(e) => setP({ auth: e.target.value as "key" | "password" })}>
            <option value="key">Private key file</option>
            <option value="password">Password</option>
          </select>
        </div>
        {p.auth === "key" ? (
          <>
            <div className="field wide">
              <label htmlFor="ssh-key">Private key file</label>
              <div className="row">
                <input id="ssh-key" className="input mono" style={{ flex: 1 }} value={p.key_path} onChange={(e) => setP({ key_path: e.target.value })} />
                <button className="btn sm ghost" onClick={async () => { const f = await open({ multiple: false }); if (f) setP({ key_path: String(f) }); }}>Browse...</button>
              </div>
            </div>
            <div className="field">
              <label htmlFor="ssh-pass">Key passphrase</label>
              <input id="ssh-pass" type="password" className="input" placeholder={hasSecret ? "Saved (leave blank to keep)" : "None"} value={draft.passphrase} onChange={(e) => setDraft({ ...draft, passphrase: e.target.value })} />
            </div>
          </>
        ) : (
          <div className="field">
            <label htmlFor="ssh-pw">Password</label>
            <input id="ssh-pw" type="password" className="input" placeholder={hasSecret ? "Saved (leave blank to keep)" : ""} value={draft.password} onChange={(e) => setDraft({ ...draft, password: e.target.value })} />
          </div>
        )}
        <label className="check" style={{ alignSelf: "end" }}>
          <input type="checkbox" checked={p.windows} onChange={(e) => setP({ windows: e.target.checked })} /> Windows host (PowerShell)
        </label>
        <div className="field wide">
          <label htmlFor="ssh-roots">Allowed folders on the server (one per line; empty = whatever the account can reach)</label>
          <textarea id="ssh-roots" className="input mono" rows={2} value={draft.roots} onChange={(e) => setDraft({ ...draft, roots: e.target.value })} />
        </div>
        <span className="help faint" style={{ gridColumn: "1 / -1" }}>Passwords and passphrases are stored in Windows Credential Manager, never in the app database. The server's host key must be trusted before the first connection.</span>
      </div>
    );
  }
  const c = draft.config;
  const setC = (patch: Partial<DockerConfig>) => setDraft({ ...draft, config: { ...c, ...patch } });
  return (
    <div className="form-grid">
      <div className="field">
        <label htmlFor="dk-name">Name</label>
        <input id="dk-name" className="input" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
      </div>
      <div className="field">
        <label htmlFor="dk-image">Image</label>
        <input id="dk-image" className="input mono" value={c.image} onChange={(e) => setC({ image: e.target.value.trim() })} />
      </div>
      <div className="field wide">
        <label htmlFor="dk-host">Docker address</label>
        <input id="dk-host" className="input mono" value={c.host} onChange={(e) => setC({ host: e.target.value.trim() })} />
      </div>
      <div className="field">
        <label htmlFor="dk-mem">Memory limit (MB)</label>
        <input id="dk-mem" className="input mono" value={c.memory_mb} onChange={(e) => setC({ memory_mb: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
      </div>
      <div className="field">
        <label htmlFor="dk-cpu">CPUs</label>
        <input id="dk-cpu" className="input mono" value={c.cpus} onChange={(e) => setC({ cpus: Number(e.target.value) || 0 })} />
      </div>
      <div className="field">
        <label htmlFor="dk-pids">Max processes</label>
        <input id="dk-pids" className="input mono" value={c.pids} onChange={(e) => setC({ pids: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
      </div>
      <label className="check" style={{ alignSelf: "end" }}>
        <input type="checkbox" checked={c.network} onChange={(e) => setC({ network: e.target.checked })} /> Network access
      </label>
      <label className="check" style={{ alignSelf: "end" }}>
        <input type="checkbox" checked={c.keep} onChange={(e) => setC({ keep: e.target.checked })} /> Keep the container between chats
      </label>
      <div className="field wide">
        <label htmlFor="dk-mounts">Folders from this PC (one per line: C:\path -&gt; /workspace/name, add "(read-only)" to protect them)</label>
        <textarea id="dk-mounts" className="input mono" rows={2} value={draft.mounts} onChange={(e) => setDraft({ ...draft, mounts: e.target.value })} />
      </div>
      <span className="help faint" style={{ gridColumn: "1 / -1" }}>Mounted folders must be inside This PC's allowed folders. Containers run with no network unless ticked, no extra Linux privileges, and these limits. Changes apply to the next container.</span>
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { errorMessage } from "../lib/api";
import { api5, type SkillMatch, type SkillRevision, type SkillView } from "../lib/api5";
import { Editor } from "../components/Editor";

const TEMPLATE = `---
name: my-skill
description: One sentence on when this skill applies
version: 1
keywords: [example]
---
# What to do

- Step one.
- Step two.
`;

/** Minimal line diff (LCS) for reviewing proposals. */
function lineDiff(a: string, b: string): { t: " " | "+" | "-"; l: string }[] {
  const x = a.split("\n"), y = b.split("\n");
  const n = x.length, m = y.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: { t: " " | "+" | "-"; l: string }[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) { out.push({ t: " ", l: x[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ t: "-", l: x[i] }); i++; }
    else { out.push({ t: "+", l: y[j] }); j++; }
  }
  while (i < n) out.push({ t: "-", l: x[i++] });
  while (j < m) out.push({ t: "+", l: y[j++] });
  return out;
}

export function Skills() {
  const [skills, setSkills] = useState<SkillView[] | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [isNew, setIsNew] = useState(false);
  const [revs, setRevs] = useState<SkillRevision[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tryText, setTryText] = useState("");
  const [matches, setMatches] = useState<SkillMatch[] | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const l = await api5.listSkills();
      setSkills(l);
      return l;
    } catch (e) {
      setErr(errorMessage(e));
      return [];
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const sel = skills?.find((s) => s.id === selId) ?? null;
  useEffect(() => {
    setConfirmDelete(false);
    if (sel && !isNew) {
      setDraft(sel.content);
      api5.revisions(sel.id).then(setRevs).catch(() => setRevs([]));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selId, sel?.content]);

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

  const saveSkill = () =>
    run(async () => {
      const v = await api5.saveSkill(isNew ? null : selId, draft);
      setIsNew(false);
      await refresh();
      setSelId(v.id);
      setRevs(await api5.revisions(v.id));
    }, "Saved.");

  const pending = revs.filter((r) => r.state === "pending");

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">
        Skills are instructions an agent follows for a kind of work, picked by matching the task's words. Agents can only propose changes to a skill; you approve or reject them here. Files that come with a skill (scripts) stay off until you turn each one on.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn sm primary" aria-label="New skill" onClick={() => { setIsNew(true); setSelId(null); setDraft(TEMPLATE); setRevs([]); }}>New skill</button>
        <button className="btn sm" onClick={async () => { const p = await open({ multiple: false, filters: [{ name: "Skill package", extensions: ["zip"] }] }); if (p) run(async () => { const v = await api5.importSkill(String(p)); await refresh(); setIsNew(false); setSelId(v.id); }, "Imported. Its extra files are off until you turn them on."); }}>Import .zip</button>
        <button className="btn sm ghost" onClick={async () => { const p = await open({ directory: true }); if (p) run(async () => { const v = await api5.importSkill(String(p)); await refresh(); setIsNew(false); setSelId(v.id); }, "Imported. Its extra files are off until you turn them on."); }}>Import folder</button>
      </div>
      {err && <div className="notice error" role="alert" style={{ marginBottom: 10 }}>{err}</div>}
      {msg && <div className="notice" style={{ marginBottom: 10 }}>{msg}</div>}

      <div className="panel" style={{ display: "grid", gap: 8, marginBottom: 14 }}>
        <strong style={{ fontWeight: 600 }}>Try an objective</strong>
        <div className="row">
          <input className="input" style={{ flex: 1 }} aria-label="Objective to match" placeholder="e.g. make cargo test pass in my Rust project" value={tryText}
            onChange={(e) => setTryText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && tryText.trim()) run(async () => setMatches(await api5.matchSkills(tryText))); }} />
          <button className="btn sm" disabled={!tryText.trim() || busy} onClick={() => run(async () => setMatches(await api5.matchSkills(tryText)))}>Match</button>
        </div>
        {matches && (matches.length === 0 ? <span className="faint">No skill would be picked for this.</span> : (
          <ul className="plain-list" aria-label="Matched skills">
            {matches.map((m) => <li key={m.name}><span className="mono">{m.name}</span> <span className="faint">{m.reason}</span></li>)}
          </ul>
        ))}
      </div>

      <div className="instr-layout">
        <div className="instr-list" aria-label="Skills">
          {skills == null ? (
            <div style={{ padding: 10 }}><div className="skeleton" /></div>
          ) : skills.length === 0 ? (
            <div className="empty" style={{ padding: 12 }}>No skills yet. Write one with New skill or import a package.</div>
          ) : (
            skills.map((s) => (
              <button key={s.id} className="chat-item" aria-current={s.id === selId && !isNew ? "true" : undefined} onClick={() => { setIsNew(false); setSelId(s.id); }}>
                <span className="chat-title">{s.name}</span>
                <span className={`mono ${s.error ? "danger-text" : "faint"}`} style={{ fontSize: 10.5 }}>
                  {s.error ? "invalid" : { auto: "picked when it matches", always: "always on", off: "off" }[s.mode]}{s.pending ? `, ${s.pending} proposal${s.pending > 1 ? "s" : ""}` : ""}
                </span>
              </button>
            ))
          )}
        </div>

        <div style={{ display: "grid", gap: 14, minWidth: 0, alignContent: "start" }}>
          {!isNew && !sel && skills && skills.length > 0 && <div className="empty">Pick a skill on the left.</div>}

          {(isNew || sel) && (
            <div className="panel" style={{ display: "grid", gap: 10 }}>
              <div className="row">
                <strong style={{ fontWeight: 600 }}>{isNew ? "New skill" : sel?.name}</strong>
                {sel && !isNew && <span className="tag">v{sel.version}</span>}
                {sel && !isNew && <span className="faint" style={{ fontSize: 12 }}>{sel.source === "import" ? "imported" : "yours"}</span>}
                <span className="spacer" />
                {sel && !isNew && (
                  <select className="input" aria-label="Skill mode" value={sel.mode} onChange={(e) => run(async () => { await api5.setSkillMode(sel.id, e.target.value); await refresh(); })}>
                    <option value="auto">Picked when it matches</option>
                    <option value="always">Always on</option>
                    <option value="off">Off</option>
                  </select>
                )}
              </div>
              {sel?.error && !isNew && <div className="notice error">This SKILL.md is not valid: {sel.error}</div>}
              <Editor label="Skill content" rows={18} value={draft} onChange={setDraft} />
              <div className="row">
                <button className="btn sm primary" aria-label="Save skill" disabled={busy || !draft.trim() || (!isNew && draft === sel?.content)} onClick={saveSkill}>Save skill</button>
                {isNew && <button className="btn sm ghost" onClick={() => { setIsNew(false); setDraft(""); }}>Cancel</button>}
                {sel && !isNew && (
                  <>
                    <button className="btn sm ghost" onClick={async () => { const p = await save({ defaultPath: `${sel.name}.zip`, filters: [{ name: "Skill package", extensions: ["zip"] }] }); if (p) run(() => api5.exportSkill(sel.id, p), `Exported to ${p}.`); }}>Export</button>
                    <span className="spacer" />
                    {!confirmDelete ? (
                      <button className="btn sm ghost danger" onClick={() => setConfirmDelete(true)}>Delete</button>
                    ) : (
                      <>
                        <span className="faint">The folder moves to the app's trash.</span>
                        <button className="btn sm danger" onClick={() => run(async () => { await api5.deleteSkill(sel.id); setSelId(null); await refresh(); }, "Moved to the trash.")}>Delete skill</button>
                        <button className="btn sm ghost" onClick={() => setConfirmDelete(false)}>Keep</button>
                      </>
                    )}
                  </>
                )}
              </div>
              <span className="help faint">Front matter needs name (lowercase, dashes) and description; keywords help matching. Saved in {sel && !isNew ? sel.path : "the app's skills folder"}.</span>
            </div>
          )}

          {sel && !isNew && pending.length > 0 && (
            <div className="panel" style={{ display: "grid", gap: 10 }}>
              <strong style={{ fontWeight: 600 }}>Proposed changes</strong>
              {pending.map((r) => (
                <div key={r.id} style={{ display: "grid", gap: 6 }}>
                  <span className="dim">{r.note || "No reason given."} <span className="faint">({r.proposed_by ? `from ${r.proposed_by.slice(0, 8)}` : "agent"}, {new Date(r.created_at).toLocaleString()})</span></span>
                  <pre className="preview-text" aria-label="Proposed diff">
                    {lineDiff(sel.content, r.content).map((d, i) => (
                      <span key={i} className={d.t === "+" ? "diff-add" : d.t === "-" ? "diff-del" : undefined}>{d.t} {d.l}{"\n"}</span>
                    ))}
                  </pre>
                  <div className="row">
                    <button className="btn sm primary" aria-label="Approve proposal" disabled={busy} onClick={() => run(async () => { await api5.approve(r.id); await refresh(); setRevs(await api5.revisions(sel.id)); }, "Applied.")}>Approve</button>
                    <button className="btn sm danger" aria-label="Reject proposal" disabled={busy} onClick={() => run(async () => { await api5.reject(r.id); setRevs(await api5.revisions(sel.id)); await refresh(); }, "Rejected; the skill is unchanged.")}>Reject</button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {sel && !isNew && (
            <div className="panel" style={{ display: "grid", gap: 8 }}>
              <strong style={{ fontWeight: 600 }}>Files</strong>
              {sel.files.length === 0 ? <span className="faint">Only SKILL.md. Scripts and reference files placed in the skill folder show up here, off until you turn them on.</span> : (
                <table className="table">
                  <thead><tr><th>File</th><th>Size</th><th>On</th></tr></thead>
                  <tbody>
                    {sel.files.map((f) => (
                      <tr key={f.path}>
                        <td className="mono">{f.path}{f.changed && <span className="danger-text" style={{ marginLeft: 8, fontFamily: "inherit" }}>changed since you turned it on, so it is off</span>}</td>
                        <td className="num">{f.size.toLocaleString()} B</td>
                        <td><input type="checkbox" aria-label={`Allow ${f.path}`} checked={f.enabled} onChange={(e) => run(async () => { await api5.setSkillScript(sel.id, f.path, e.target.checked); await refresh(); })} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <span className="help faint">Agents only learn about files you turned on, and running one is still a normal shell command that asks first.</span>
            </div>
          )}

          {sel && !isNew && (
            <div className="panel" style={{ display: "grid", gap: 8 }}>
              <strong style={{ fontWeight: 600 }}>History</strong>
              {revs.filter((r) => r.state !== "pending").length === 0 ? <span className="faint">No saved versions yet.</span> : (
                <table className="table">
                  <thead><tr><th>When</th><th>Version</th><th>Note</th><th>State</th><th /></tr></thead>
                  <tbody>
                    {revs.filter((r) => r.state !== "pending").map((r, i) => (
                      <tr key={r.id}>
                        <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 12 }}>{new Date(r.created_at).toLocaleString()}</td>
                        <td className="mono">{r.version}</td>
                        <td className="dim" style={{ fontSize: 12.5 }}>{r.note ?? ""}</td>
                        <td className="faint">{r.state}</td>
                        <td>{i > 0 && r.state === "applied" && r.content !== sel.content && <button className="btn sm ghost" aria-label="Revert to this version" disabled={busy} onClick={() => run(async () => { await api5.revert(r.id); await refresh(); setRevs(await api5.revisions(sel.id)); }, "Reverted (saved as a new version).")}>Revert</button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

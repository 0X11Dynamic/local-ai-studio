import { useCallback, useEffect, useMemo, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { api, errorMessage, type ModelView } from "../lib/api";
import { api2, type InstructionPreview, type InstructionProfile, type InstructionSettings, type Sections } from "../lib/api2";
import { Editor } from "../components/Editor";

const BLANK: Sections = { mode: "sections", identity: "", task: "", tool_use: "", formatting: "", raw: "" };

export function Instructions() {
  const [profiles, setProfiles] = useState<InstructionProfile[]>([]);
  const [settings, setSettings] = useState<InstructionSettings | null>(null);
  const [savedBase, setSavedBase] = useState("");
  const [selId, setSelId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [sections, setSections] = useState<Sections>(BLANK);
  const [mode, setMode] = useState<"chat" | "agent">("chat");
  const [preview, setPreview] = useState<InstructionPreview | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [testPrompt, setTestPrompt] = useState("");
  const [testOut, setTestOut] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [models, setModels] = useState<ModelView[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelProfile, setModelProfile] = useState<string>("");

  const sel = profiles.find((p) => p.id === selId) ?? null;
  const dirty = !!sel && (title !== sel.title || JSON.stringify(sections) !== JSON.stringify(sel.sections));
  const baseDirty = !!settings && settings.base !== savedBase;

  const reload = useCallback(async (keep?: string) => {
    const [ps, s] = await Promise.all([api2.listProfiles(), api2.getInstructionSettings()]);
    setProfiles(ps);
    setSettings(s);
    setSavedBase(s.base);
    const id = keep ?? s.chat_profile_id ?? ps[0]?.id ?? null;
    const p = ps.find((x) => x.id === id) ?? ps[0];
    if (p) {
      setSelId(p.id);
      setTitle(p.title);
      setSections(p.sections);
    }
  }, []);

  useEffect(() => {
    reload();
    api.listModels().then((m) => {
      setModels(m);
      if (m[0]) setModelId(m[0].id);
    });
  }, [reload]);

  useEffect(() => {
    if (modelId) api2.getModelProfile(modelId).then((p) => setModelProfile(p ?? ""));
  }, [modelId]);

  // Live preview (unsaved edits included).
  useEffect(() => {
    if (!settings) return;
    const t = setTimeout(() => {
      api2.previewInstructions(mode, sections, settings.base).then(setPreview).catch((e) => setErr(errorMessage(e)));
    }, 300);
    return () => clearTimeout(t);
  }, [sections, settings, mode]);

  const pick = (p: InstructionProfile) => {
    if (dirty && p.id !== selId) {
      setErr("This profile has unsaved changes. Save or discard them first.");
      return;
    }
    setSelId(p.id);
    setTitle(p.title);
    setSections(p.sections);
    setMsg(null);
    setErr(null);
    setTestOut(null);
  };

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

  const saveProfile = () => run(async () => { await api2.updateProfile(sel!.id, title, sections); await reload(sel!.id); }, "Profile saved.");
  const cloneProfile = () => run(async () => { const id = await api2.createProfile(`${title} copy`.slice(0, 80), sections); await reload(id); }, "Copy created. Edit it and save.");
  const newProfile = () => run(async () => { const id = await api2.createProfile("New profile", BLANK); await reload(id); }, "New profile created.");
  const deleteProfile = () => run(async () => { await api2.deleteProfile(sel!.id); await reload(); }, "Profile deleted.");
  const saveBase = () => run(async () => { await api2.setInstructionSettings(settings!); setSavedBase(settings!.base); }, "Base instruction saved. It applies to new replies.");
  const setDefault = (which: "chat" | "agent") =>
    run(async () => {
      const next = { ...settings!, base: savedBase, [which === "chat" ? "chat_profile_id" : "agent_profile_id"]: sel!.id };
      await api2.setInstructionSettings(next);
      setSettings({ ...settings!, [which === "chat" ? "chat_profile_id" : "agent_profile_id"]: sel!.id });
    }, `Set as the ${which} default.`);
  const exportProfile = async () => {
    const path = await save({ defaultPath: `${title.replace(/[\\/:*?"<>|]/g, "_")}.json`, filters: [{ name: "Profile JSON", extensions: ["json"] }, { name: "Markdown", extensions: ["md"] }] });
    if (path) run(() => api2.exportProfile(sel!.id, path), `Exported to ${path}`);
  };
  const importProfile = async () => {
    const path = await open({ multiple: false, filters: [{ name: "Profile", extensions: ["json", "md", "txt"] }] });
    if (path) run(async () => { const id = await api2.importProfile(path); await reload(id); }, "Imported.");
  };
  const test = async () => {
    setTesting(true);
    setTestOut(null);
    setErr(null);
    try {
      const r = await api2.testInstructions(sections, settings?.base ?? null, testPrompt);
      setTestOut(r.content || "(no answer text; the model produced only reasoning)");
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setTesting(false);
    }
  };

  const tokenText = useMemo(() => {
    if (!preview) return "";
    if (preview.tokens == null) return `${preview.chars.toLocaleString()} characters. Load a model to count tokens with its tokenizer.`;
    return `${preview.tokens.toLocaleString()} tokens (${preview.templated_tokens ?? "?"} with chat-template markers) of ${preview.n_ctx?.toLocaleString() ?? "?"} context, counted with the loaded model.`;
  }, [preview]);

  const field = (k: keyof Sections, label: string, help: string) => (
    <div className="field">
      <label htmlFor={`sec-${k}`}>{label}</label>
      <Editor id={`sec-${k}`} label={label} rows={4} value={sections[k] as string} readOnly={sel?.builtin} onChange={(v) => setSections({ ...sections, [k]: v })} />
      <span className="help">{help}</span>
    </div>
  );

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <p className="lede">
        System instructions are sent as the system message, encoded by each model's own chat template. Safety rules for tools are enforced by the app itself, so nothing written here can loosen them.
      </p>
      {err && <div className="notice error" style={{ marginBottom: 10 }}>{err}</div>}
      {msg && <div className="notice" style={{ marginBottom: 10 }}>{msg}</div>}

      <h2>Base instruction</h2>
      <p className="faint" style={{ marginTop: 0 }}>Comes first in every chat, before the profile.</p>
      {settings && (
        <>
          <Editor label="Base instruction" rows={3} value={settings.base} onChange={(v) => setSettings({ ...settings, base: v })} />
          <div className="row" style={{ marginTop: 6 }}>
            <button className="btn sm primary" disabled={!baseDirty} onClick={saveBase}>Save base</button>
            <button className="btn sm ghost" disabled={!baseDirty} onClick={() => setSettings({ ...settings, base: savedBase })}>Discard changes</button>
            <button className="btn sm ghost" onClick={() => run(async () => { const b = await api2.resetInstructionBase(); setSettings({ ...settings, base: b }); setSavedBase(b); }, "Base reset to the app default.")}>Reset to default</button>
            {baseDirty && <span style={{ color: "var(--accent)" }}>Unsaved changes</span>}
          </div>
        </>
      )}

      <h2>Profiles</h2>
      <div className="instr-layout">
        <div className="instr-list">
          {profiles.map((p) => (
            <button key={p.id} className="chat-item" aria-current={p.id === selId ? "true" : undefined} onClick={() => pick(p)}>
              <span className="chat-title">{p.title}</span>
              <span className="faint mono" style={{ fontSize: 10.5 }}>
                {[p.builtin ? "template" : null, settings?.chat_profile_id === p.id ? "chat default" : null, settings?.agent_profile_id === p.id ? "agent default" : null].filter(Boolean).join(", ")}
              </span>
            </button>
          ))}
          <div className="row" style={{ padding: 8 }}>
            <button className="btn sm" onClick={newProfile}>New</button>
            <button className="btn sm ghost" onClick={importProfile}>Import</button>
          </div>
        </div>

        {sel && (
          <div style={{ display: "grid", gap: 12, minWidth: 0 }}>
            <div className="row">
              <div className="field" style={{ flex: 1 }}>
                <label htmlFor="pname">Profile name</label>
                <input id="pname" className="input" value={title} disabled={sel.builtin} onChange={(e) => setTitle(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="pmode">Editor</label>
                <select id="pmode" className="input" value={sections.mode} disabled={sel.builtin}
                  onChange={(e) => setSections({ ...sections, mode: e.target.value as Sections["mode"], raw: e.target.value === "raw" && !sections.raw ? [sections.identity, sections.task, sections.tool_use, sections.formatting].filter((s) => s.trim()).join("\n\n") : sections.raw })}>
                  <option value="sections">Separate fields</option>
                  <option value="raw">Single raw editor</option>
                </select>
              </div>
            </div>
            {sel.builtin && <div className="notice">Built-in templates are read-only. Clone it to make your own version.</div>}
            {sections.mode === "raw" ? (
              <div className="field">
                <label htmlFor="sec-raw">Instruction</label>
                <Editor id="sec-raw" label="Instruction" rows={14} value={sections.raw} readOnly={sel.builtin} onChange={(v) => setSections({ ...sections, raw: v })} />
              </div>
            ) : (
              <>
                {field("identity", "Identity and personality", "Who the assistant is and how it sounds.")}
                {field("task", "Task instructions", "What it should do and how carefully.")}
                {field("tool_use", "Tool-use instructions", "How to use tools in Agent Mode. Tools themselves are listed by the app, not here.")}
                {field("formatting", "Formatting", "Markdown, length, code blocks.")}
              </>
            )}
            <div className="row">
              {!sel.builtin && <button className="btn primary sm" disabled={!dirty} onClick={saveProfile}>Save profile</button>}
              {!sel.builtin && <button className="btn sm ghost" disabled={!dirty} onClick={() => { setTitle(sel.title); setSections(sel.sections); }}>Discard changes</button>}
              <button className="btn sm" onClick={cloneProfile}>Clone</button>
              <button className="btn sm ghost" onClick={exportProfile}>Export</button>
              {!sel.builtin && <button className="btn sm ghost danger" onClick={deleteProfile}>Delete</button>}
              <span className="spacer" />
              <button className="btn sm ghost" disabled={settings?.chat_profile_id === sel.id} onClick={() => setDefault("chat")}>Use for new chats</button>
              <button className="btn sm ghost" disabled={settings?.agent_profile_id === sel.id} onClick={() => setDefault("agent")}>Use for Agent Mode</button>
            </div>
            {dirty && <div style={{ color: "var(--accent)" }}>Unsaved changes. Edits apply to future replies, not past ones.</div>}

            <div className="panel" style={{ display: "grid", gap: 8 }}>
              <div className="row">
                <strong style={{ fontWeight: 600 }}>Effective instruction (read-only preview)</strong>
                <span className="spacer" />
                <select className="input" aria-label="Preview mode" value={mode} onChange={(e) => setMode(e.target.value as "chat" | "agent")}>
                  <option value="chat">Chat Mode (no tools)</option>
                  <option value="agent">Agent Mode</option>
                </select>
              </div>
              {preview?.parts.map((p) => (
                <div key={p.label}>
                  <div className="faint" style={{ fontSize: 11.5 }}>{p.label}{p.editable ? "" : " (generated by the app, not editable)"}</div>
                  <pre className="preview-text">{p.text}</pre>
                </div>
              ))}
              {preview && preview.parts.length === 0 && <div className="faint">Empty: the model gets no system message.</div>}
              <div className="mono faint" style={{ fontSize: 11.5 }}>{tokenText}</div>
            </div>

            <div className="panel" style={{ display: "grid", gap: 8 }}>
              <strong style={{ fontWeight: 600 }}>Test this instruction</strong>
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <input className="input" style={{ flex: 1 }} aria-label="Test prompt" placeholder="Introduce yourself in two sentences." value={testPrompt} onChange={(e) => setTestPrompt(e.target.value)} />
                <button className="btn sm" onClick={test} disabled={testing}>{testing ? "Running" : "Run test"}</button>
              </div>
              <span className="help faint">Uses the unsaved text above with the loaded model. Nothing is saved to any chat.</span>
              {testOut && <pre className="preview-text">{testOut}</pre>}
            </div>
          </div>
        )}
      </div>

      <h2>Per-model default</h2>
      <p className="faint" style={{ marginTop: 0 }}>Overrides the app default for chats with that model. A chat's own setting overrides both.</p>
      <div className="row">
        <select className="input" aria-label="Model" value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ minWidth: 320 }}>
          {models.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
        </select>
        <select className="input" aria-label="Profile for this model" value={modelProfile} onChange={(e) => {
          const v = e.target.value;
          setModelProfile(v);
          run(() => api2.setModelProfile(modelId, v || null), v ? "Model default set." : "Model default removed.");
        }}>
          <option value="">No override (app default)</option>
          {profiles.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
        </select>
      </div>
    </div>
  );
}

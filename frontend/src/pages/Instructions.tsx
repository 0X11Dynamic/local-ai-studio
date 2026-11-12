// System Instructions: left, the defaults / templates / my profiles list;
// right, the selected item's editor with its preview, test and actions.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { Copy, Export, LockSimple, Trash } from "@phosphor-icons/react";
import { api, errorMessage, type ModelView } from "../lib/api";
import { api2, type InstructionPreview, type InstructionProfile, type InstructionSettings, type Sections } from "../lib/api2";
import { Editor } from "../components/Editor";
import { DEFAULTS, InstructionNav, usageText, type DefaultKey, type Selection } from "../components/instructions/InstructionNav";
import { UsedBy } from "../components/instructions/UsedBy";
import { PreviewPanel, TestPanel, tokenSummary, type PreviewMode } from "../components/instructions/PreviewPanel";

const BLANK: Sections = { mode: "sections", identity: "", task: "", tool_use: "", formatting: "", raw: "" };

const FIELDS: { key: "identity" | "task" | "tool_use" | "formatting"; label: string; help: string }[] = [
  { key: "identity", label: "Identity and personality", help: "Who the assistant is and how it sounds." },
  { key: "task", label: "Task instructions", help: "What it should do and how carefully." },
  { key: "tool_use", label: "Tool-use instructions", help: "How to use tools in Agent and Plan mode. The tools themselves are listed by the app, not here." },
  { key: "formatting", label: "Formatting", help: "Markdown, length, code blocks." },
];

const same = (a: Selection | null, b: Selection | null) =>
  !!a && !!b && a.kind === b.kind && (a.kind === "base" || (b.kind === "profile" && a.id === b.id));

export function Instructions() {
  const [profiles, setProfiles] = useState<InstructionProfile[]>([]);
  const [settings, setSettings] = useState<InstructionSettings | null>(null);
  const [baseDraft, setBaseDraft] = useState("");
  const [models, setModels] = useState<ModelView[]>([]);
  const [modelProfiles, setModelProfiles] = useState<Record<string, string | null>>({});
  const [selection, setSelection] = useState<Selection | null>(null);
  const [title, setTitle] = useState("");
  const [sections, setSections] = useState<Sections>(BLANK);
  const [previewMode, setPreviewMode] = useState<PreviewMode>("chat");
  const [preview, setPreview] = useState<InstructionPreview | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [testPrompt, setTestPrompt] = useState("");
  const [testOut, setTestOut] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [pending, setPending] = useState<(() => void) | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const testRef = useRef<HTMLDivElement>(null);

  const sel = selection?.kind === "profile" ? profiles.find((p) => p.id === selection.id) ?? null : null;
  const onBase = selection?.kind === "base";
  const profileDirty = !!sel && !sel.builtin && (title !== sel.title || JSON.stringify(sections) !== JSON.stringify(sel.sections));
  const baseDirty = !!settings && baseDraft !== settings.base;
  const dirty = onBase ? baseDirty : profileDirty;

  /** The profile a mode uses by default (for the base instruction's preview and test). */
  const defaultSections = useCallback(
    (mode: PreviewMode) => {
      const key: DefaultKey = mode === "agent" ? "agent_profile_id" : mode === "plan" ? "plan_profile_id" : "chat_profile_id";
      return profiles.find((p) => p.id === settings?.[key])?.sections ?? BLANK;
    },
    [profiles, settings],
  );

  const show = useCallback((s: Selection, ps: InstructionProfile[], st: InstructionSettings | null) => {
    setSelection(s);
    setMsg(null);
    setErr(null);
    setTestOut(null);
    setConfirmDelete(false);
    if (s.kind === "profile") {
      const p = ps.find((x) => x.id === s.id);
      if (p) {
        setTitle(p.title);
        setSections(p.sections);
      }
      // Preview in the mode this profile is used for, if any.
      setPreviewMode(st?.plan_profile_id === s.id && st?.chat_profile_id !== s.id && st?.agent_profile_id !== s.id ? "plan"
        : st?.agent_profile_id === s.id && st?.chat_profile_id !== s.id ? "agent" : "chat");
    }
  }, []);

  const reload = useCallback(async (keep?: Selection) => {
    const [ps, s] = await Promise.all([api2.listProfiles(), api2.getInstructionSettings()]);
    setProfiles(ps);
    setSettings(s);
    setBaseDraft(s.base);
    const fallback: Selection | null = s.chat_profile_id ? { kind: "profile", id: s.chat_profile_id } : ps[0] ? { kind: "profile", id: ps[0].id } : null;
    const want = keep && (keep.kind === "base" || ps.some((p) => p.id === keep.id)) ? keep : fallback;
    if (want) show(want, ps, s);
  }, [show]);

  useEffect(() => {
    reload().catch((e) => setErr(errorMessage(e)));
    api.listModels().then(async (ms) => {
      setModels(ms);
      const pairs = await Promise.all(ms.map(async (m) => [m.id, await api2.getModelProfile(m.id).catch(() => null)] as const));
      setModelProfiles(Object.fromEntries(pairs));
    }).catch(() => {});
  }, [reload]);

  // Live preview (unsaved edits included), debounced.
  const previewSections = onBase ? defaultSections(previewMode) : sections;
  const previewBase = onBase ? baseDraft : settings?.base ?? null;
  const previewKey = JSON.stringify([previewMode, previewSections, previewBase]);
  useEffect(() => {
    if (!settings || !selection) return;
    const t = setTimeout(() => {
      api2.previewInstructions(previewMode, previewSections, previewBase).then(setPreview).catch((e) => setErr(errorMessage(e)));
    }, 300);
    return () => clearTimeout(t);
    // previewKey covers the inputs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey, settings, selection]);

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setErr(null);
    setMsg(null);
    try {
      await fn();
      setMsg(ok);
      return true;
    } catch (e) {
      setErr(errorMessage(e));
      return false;
    }
  };

  /** Run `next` now, or after the user saves or discards unsaved edits. */
  const guard = (next: () => void) => {
    if (dirty) setPending(() => next);
    else next();
  };
  const select = (s: Selection) => {
    if (same(s, selection)) return;
    guard(() => show(s, profiles, settings));
  };

  const discard = () => {
    if (onBase && settings) setBaseDraft(settings.base);
    else if (sel) {
      setTitle(sel.title);
      setSections(sel.sections);
    }
  };
  const saveCurrent = async () => {
    if (onBase && settings) {
      return run(async () => {
        const next = { ...settings, base: baseDraft };
        await api2.setInstructionSettings(next);
        setSettings(next);
      }, "Base instruction saved. It applies to new replies.");
    }
    if (sel && !sel.builtin) {
      return run(async () => {
        await api2.updateProfile(sel.id, title, sections);
        const ps = await api2.listProfiles();
        setProfiles(ps);
      }, "Profile saved. It applies to new replies.");
    }
    return true;
  };
  const resolvePending = async (how: "save" | "discard" | "cancel") => {
    const next = pending;
    if (how === "cancel") return setPending(null);
    if (how === "save" && !(await saveCurrent())) return;
    if (how === "discard") discard();
    setPending(null);
    next?.();
  };

  const setDefault = (key: DefaultKey, id: string) => {
    if (!settings || !id) return;
    const label = DEFAULTS.find((d) => d.key === key)?.label.toLowerCase();
    run(async () => {
      const next = { ...settings, [key]: id };
      await api2.setInstructionSettings(next);
      setSettings(next);
    }, `${profiles.find((p) => p.id === id)?.title ?? "Profile"} is now the ${label}.`);
  };
  const setModel = (modelId: string, profileId: string | null) => {
    const name = models.find((m) => m.id === modelId)?.display_name ?? "the model";
    run(async () => {
      await api2.setModelProfile(modelId, profileId);
      setModelProfiles((m) => ({ ...m, [modelId]: profileId }));
    }, profileId ? `${name} now uses this profile in Chat mode.` : `${name} uses the Chat default again.`);
  };

  const createFrom = (name: string, s: Sections, ok: string) =>
    run(async () => {
      const id = await api2.createProfile(name.trim().slice(0, 80) || "New profile", s);
      await reload({ kind: "profile", id });
    }, ok);
  // The copy takes unsaved edits with it, so there is nothing to ask about.
  const duplicate = () => {
    if (!sel) return;
    const [name, s] = profileDirty ? [title, sections] : [sel.title, sel.sections];
    createFrom(`${name.slice(0, 75)} copy`, s, profileDirty ? "Copy created with your unsaved edits; the original is unchanged." : "Copy created under My profiles. Edit it, then save.");
  };
  const newProfile = () => guard(() => createFrom("New profile", BLANK, "New profile created. Name it, write it, then save."));
  const deleteProfile = () => {
    if (!sel) return;
    run(async () => {
      await api2.deleteProfile(sel.id);
      setModelProfiles((m) => Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v === sel.id ? null : v])));
      await reload();
    }, `Deleted ${sel.title}.`);
  };
  const exportProfile = async () => {
    if (!sel) return;
    const path = await save({ defaultPath: `${sel.title.replace(/[\\/:*?"<>|]/g, "_")}.json`, filters: [{ name: "Profile JSON", extensions: ["json"] }, { name: "Markdown", extensions: ["md"] }] });
    if (path) run(() => api2.exportProfile(sel.id, path), `Exported to ${path}${profileDirty ? " (saved version; unsaved edits are not included)" : ""}`);
  };
  const importProfile = () =>
    guard(async () => {
      const path = await open({ multiple: false, filters: [{ name: "Profile", extensions: ["json", "md", "txt"] }] });
      if (path) run(async () => { const id = await api2.importProfile(path); await reload({ kind: "profile", id }); }, "Imported into My profiles.");
    });
  const resetBase = () =>
    run(async () => {
      const b = await api2.resetInstructionBase();
      setSettings((s) => (s ? { ...s, base: b } : s));
      setBaseDraft(b);
    }, "Base instruction reset to the app default.");

  const test = async () => {
    setTesting(true);
    setTestOut(null);
    setErr(null);
    testRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    try {
      const r = await api2.testInstructions(onBase ? defaultSections("chat") : sections, onBase ? baseDraft : settings?.base ?? null, testPrompt);
      setTestOut(r.content || "(no answer text; the model produced only reasoning)");
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setTesting(false);
    }
  };

  const modelCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const v of Object.values(modelProfiles)) if (v) c[v] = (c[v] ?? 0) + 1;
    return c;
  }, [modelProfiles]);

  const readOnly = !!sel?.builtin;
  const setMode = (mode: Sections["mode"]) =>
    setSections({
      ...sections,
      mode,
      raw: mode === "raw" && !sections.raw ? [sections.identity, sections.task, sections.tool_use, sections.formatting].filter((s) => s.trim()).join("\n\n") : sections.raw,
    });

  const header = onBase ? (
    <header className="instr-head">
      <h2 id="instr-title">Base instruction</h2>
      {baseDirty && <span className="tag accent">Unsaved</span>}
      <span className="spacer" />
      <button type="button" className="btn ghost sm" onClick={resetBase}>Reset to app default</button>
    </header>
  ) : sel ? (
    <header className="instr-head">
      {readOnly ? (
        <h2 id="instr-title"><LockSimple size={14} aria-hidden /> {sel.title}</h2>
      ) : (
        <input id="instr-title" className="input instr-title-input" aria-label="Profile name" maxLength={80} value={title} onChange={(e) => setTitle(e.target.value)} />
      )}
      {readOnly && <span className="tag">Built-in, read-only</span>}
      {profileDirty && <span className="tag accent">Unsaved</span>}
      {usageText(sel.id, settings, modelCounts[sel.id] ?? 0) && <span className="tag">{usageText(sel.id, settings, modelCounts[sel.id] ?? 0)}</span>}
      <span className="spacer" />
      <button type="button" className={readOnly ? "btn primary sm" : "btn sm"} onClick={duplicate}>
        <Copy size={14} aria-hidden /> {readOnly ? "Duplicate to edit" : "Duplicate"}
      </button>
      <button type="button" className="btn ghost sm" onClick={exportProfile}><Export size={14} aria-hidden /> Export</button>
      {!readOnly && (confirmDelete ? (
        <>
          <button type="button" className="btn danger sm" onClick={deleteProfile}>Delete {sel.title}?</button>
          <button type="button" className="btn ghost sm" onClick={() => setConfirmDelete(false)}>Keep</button>
        </>
      ) : (
        <button type="button" className="btn ghost sm" aria-label={`Delete ${sel.title}`} onClick={() => setConfirmDelete(true)}><Trash size={14} aria-hidden /> Delete</button>
      ))}
    </header>
  ) : null;

  return (
    <div className="page instr-page">
      <p className="lede">
        The system message every reply starts with: the base instruction, then the profile for the mode. Each model's own chat template encodes it. Tool safety rules are enforced by the app, so nothing here can loosen them.
      </p>
      <div className="instr-shell">
        <InstructionNav profiles={profiles} settings={settings} selection={selection} modelCounts={modelCounts}
          onSelect={select} onSetDefault={setDefault} onNew={newProfile} onImport={importProfile} />

        <section className="instr-main" aria-labelledby="instr-title">
          {header}
          {err && <div className="notice error" role="alert">{err}</div>}
          {msg && <div className="notice" role="status">{msg}</div>}

          {onBase && settings && (
            <div className="field">
              <label htmlFor="instr-base">Base instruction text</label>
              <Editor id="instr-base" label="Base instruction" rows={8} value={baseDraft} onChange={setBaseDraft} />
              <span className="help">Comes first in Chat, Agent and Plan mode, before the profile. The preview below uses each mode's default profile.</span>
            </div>
          )}

          {sel && (
            <>
              {readOnly && (
                <div className="notice">Built-in templates can't be changed. Use <strong>Duplicate to edit</strong> to make your own copy under My profiles.</div>
              )}
              <div className="instr-layout-row">
                <span className="instr-usedby-label" id="instr-layout-l">Layout</span>
                <div className="instr-seg" role="group" aria-labelledby="instr-layout-l">
                  <button type="button" aria-pressed={sections.mode === "sections"} disabled={readOnly} onClick={() => setMode("sections")}>Separate fields</button>
                  <button type="button" aria-pressed={sections.mode === "raw"} disabled={readOnly} onClick={() => setMode("raw")}>One text box</button>
                </div>
              </div>
              {sections.mode === "raw" ? (
                <div className="field">
                  <label htmlFor="sec-raw">Instruction</label>
                  <Editor id="sec-raw" label="Instruction" rows={16} value={sections.raw} readOnly={readOnly} onChange={(v) => setSections({ ...sections, raw: v })} />
                </div>
              ) : (
                FIELDS.map((f) => (
                  <div className="field" key={f.key}>
                    <label htmlFor={`sec-${f.key}`}>{f.label}</label>
                    <Editor id={`sec-${f.key}`} label={f.label} rows={4} value={sections[f.key]} readOnly={readOnly} onChange={(v) => setSections({ ...sections, [f.key]: v })} />
                    <span className="help">{f.help}</span>
                  </div>
                ))
              )}
              <UsedBy profile={sel} profiles={profiles} settings={settings} models={models} modelProfiles={modelProfiles}
                onSetDefault={setDefault} onSetModel={setModel} />
            </>
          )}

          {selection && (
            <>
              <PreviewPanel preview={preview} mode={previewMode} onMode={setPreviewMode} />
              <div ref={testRef}>
                <TestPanel prompt={testPrompt} onPrompt={setTestPrompt} out={testOut} testing={testing} onRun={test} />
              </div>
            </>
          )}

          {selection && (
            <footer className="instr-actions" aria-label="Save and test">
              {pending ? (
                <>
                  <span className="instr-actions-status" role="alert">You have unsaved changes.</span>
                  <span className="spacer" />
                  <button type="button" className="btn ghost sm" onClick={() => resolvePending("cancel")}>Keep editing</button>
                  <button type="button" className="btn sm" onClick={() => resolvePending("discard")}>Discard</button>
                  {!readOnly && <button type="button" className="btn primary sm" onClick={() => resolvePending("save")}>Save and continue</button>}
                </>
              ) : (
                <>
                  <span className={dirty ? "instr-actions-status accent" : "instr-actions-status"}>
                    {readOnly ? "Read-only template" : dirty ? "Unsaved changes (apply to future replies)" : "Saved"}
                  </span>
                  <span className="mono faint instr-actions-tokens">{tokenSummary(preview)} · {previewMode}</span>
                  <span className="spacer" />
                  <button type="button" className="btn sm" onClick={test} disabled={testing}>{testing ? "Testing…" : "Test"}</button>
                  {!readOnly && <button type="button" className="btn ghost sm" disabled={!dirty} onClick={discard}>Discard</button>}
                  {!readOnly && <button type="button" className="btn primary sm" disabled={!dirty} onClick={saveCurrent}>Save</button>}
                </>
              )}
            </footer>
          )}
        </section>
      </div>
    </div>
  );
}

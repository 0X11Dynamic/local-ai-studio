// Left pane of System Instructions: defaults, built-in templates, my profiles.
import { ArrowRight, DownloadSimple, LockSimple, Plus } from "@phosphor-icons/react";
import type { InstructionProfile, InstructionSettings } from "../../lib/api2";

export type Selection = { kind: "base" } | { kind: "profile"; id: string };
export type DefaultKey = "chat_profile_id" | "agent_profile_id" | "plan_profile_id";

export const DEFAULTS: { key: DefaultKey; label: string; short: string; help: string }[] = [
  { key: "chat_profile_id", label: "Chat default", short: "Chat", help: "New chats in Chat mode" },
  { key: "agent_profile_id", label: "Agent default", short: "Agent", help: "Agent mode and Agent Tasks" },
  { key: "plan_profile_id", label: "Plan default", short: "Plan", help: "Plan mode (read-only tools)" },
];

/** "Chat default, 2 models" for the list and the editor header. */
export function usageText(id: string, settings: InstructionSettings | null, modelCount: number): string {
  const uses = DEFAULTS.filter((d) => settings?.[d.key] === id).map((d) => d.short.toLowerCase());
  const parts: string[] = [];
  if (uses.length) parts.push(`${uses.join(", ")} default`);
  if (modelCount) parts.push(`${modelCount} model${modelCount === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

export function InstructionNav({
  profiles, settings, selection, modelCounts, onSelect, onSetDefault, onNew, onImport,
}: {
  profiles: InstructionProfile[];
  settings: InstructionSettings | null;
  selection: Selection | null;
  modelCounts: Record<string, number>;
  onSelect: (s: Selection) => void;
  onSetDefault: (key: DefaultKey, id: string) => void;
  onNew: () => void;
  onImport: () => void;
}) {
  const builtins = profiles.filter((p) => p.builtin);
  const mine = profiles.filter((p) => !p.builtin);
  const isSel = (id: string) => selection?.kind === "profile" && selection.id === id;
  const titleOf = (id: string | null | undefined) => profiles.find((p) => p.id === id)?.title ?? "None";

  const item = (p: InstructionProfile) => {
    const usage = usageText(p.id, settings, modelCounts[p.id] ?? 0);
    return (
      <li key={p.id}>
        <button type="button" className="instr-item" aria-current={isSel(p.id) ? "true" : undefined} onClick={() => onSelect({ kind: "profile", id: p.id })}>
          <span className="instr-item-title">
            {p.builtin && <LockSimple size={12} aria-hidden />}
            <span className="chat-title">{p.title}</span>
            {p.builtin && <span className="sr-only"> (read-only template)</span>}
          </span>
          {usage && <span className="instr-item-meta">{usage}</span>}
        </button>
      </li>
    );
  };

  return (
    <nav className="instr-nav" aria-label="System instructions">
      <section className="instr-group" aria-labelledby="instr-g-defaults">
        <h3 className="instr-group-title" id="instr-g-defaults">Defaults</h3>
        <ul className="instr-items">
          <li>
            <button type="button" className="instr-item" aria-current={selection?.kind === "base" ? "true" : undefined} onClick={() => onSelect({ kind: "base" })}>
              <span className="instr-item-title"><span className="chat-title">Base instruction</span></span>
              <span className="instr-item-meta">Comes first in every mode</span>
            </button>
          </li>
        </ul>
        {DEFAULTS.map((d) => {
          const id = settings?.[d.key] ?? "";
          return (
            <div key={d.key} className="instr-default">
              <label htmlFor={`instr-def-${d.key}`}>{d.label}</label>
              <div className="instr-default-row">
                <select id={`instr-def-${d.key}`} className="input" value={id} disabled={!settings}
                  aria-describedby={`instr-def-${d.key}-help`} onChange={(e) => onSetDefault(d.key, e.target.value)}>
                  {!id && <option value="">None</option>}
                  <optgroup label="Built-in templates">
                    {builtins.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
                  </optgroup>
                  {mine.length > 0 && (
                    <optgroup label="My profiles">
                      {mine.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
                    </optgroup>
                  )}
                </select>
                <button type="button" className="btn ghost sm instr-icon-btn" disabled={!id}
                  aria-label={`Open ${titleOf(id)} (${d.label.toLowerCase()})`} title="Open this profile"
                  onClick={() => id && onSelect({ kind: "profile", id })}>
                  <ArrowRight size={14} aria-hidden />
                </button>
              </div>
              <span className="instr-default-help" id={`instr-def-${d.key}-help`}>{d.help}</span>
            </div>
          );
        })}
        <p className="instr-note">A chat's own profile (Chat settings) wins over these, then a per-model choice (Chat mode only).</p>
      </section>

      <section className="instr-group" aria-labelledby="instr-g-builtin">
        <h3 className="instr-group-title" id="instr-g-builtin">Built-in templates</h3>
        <ul className="instr-items">{builtins.map(item)}</ul>
      </section>

      <section className="instr-group" aria-labelledby="instr-g-mine">
        <div className="instr-group-head">
          <h3 className="instr-group-title" id="instr-g-mine">My profiles</h3>
          <span className="spacer" />
          <button type="button" className="btn ghost sm" onClick={onImport} title="Import a .json or .md profile">
            <DownloadSimple size={14} aria-hidden /> Import
          </button>
          <button type="button" className="btn sm" onClick={onNew}>
            <Plus size={14} aria-hidden /> New
          </button>
        </div>
        {mine.length ? (
          <ul className="instr-items">{mine.map(item)}</ul>
        ) : (
          <p className="instr-note">None yet. Create one, or open a template and duplicate it.</p>
        )}
      </section>
    </nav>
  );
}

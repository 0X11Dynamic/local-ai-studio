// "Used by" on a profile: which defaults use it, and per-model overrides.
import { useState } from "react";
import { Check, X } from "@phosphor-icons/react";
import type { ModelView } from "../../lib/api";
import type { InstructionProfile, InstructionSettings } from "../../lib/api2";
import { DEFAULTS, type DefaultKey } from "./InstructionNav";

export function UsedBy({
  profile, profiles, settings, models, modelProfiles, onSetDefault, onSetModel,
}: {
  profile: InstructionProfile;
  profiles: InstructionProfile[];
  settings: InstructionSettings | null;
  models: ModelView[];
  /** model id -> profile id override (missing = app default). */
  modelProfiles: Record<string, string | null>;
  onSetDefault: (key: DefaultKey, id: string) => void;
  onSetModel: (modelId: string, profileId: string | null) => void;
}) {
  const [adding, setAdding] = useState("");
  const using = models.filter((m) => modelProfiles[m.id] === profile.id);
  const others = models.filter((m) => modelProfiles[m.id] !== profile.id);
  const titleOf = (id: string | null | undefined) => profiles.find((p) => p.id === id)?.title;

  return (
    <section className="panel instr-usedby" aria-labelledby="instr-usedby-h">
      <h3 id="instr-usedby-h" className="instr-sub-h">Used by</h3>
      <div className="instr-usedby-row">
        <span className="instr-usedby-label" id="instr-usedby-defaults">Defaults</span>
        <div className="row" role="group" aria-labelledby="instr-usedby-defaults">
          {DEFAULTS.map((d) => {
            const on = settings?.[d.key] === profile.id;
            return (
              <button key={d.key} type="button" className={`btn sm ${on ? "instr-on" : "ghost"}`} aria-pressed={on}
                disabled={on || !settings}
                title={on ? `Already the ${d.label.toLowerCase()}. Pick another profile under Defaults to change it.` : `Use this profile as the ${d.label.toLowerCase()}`}
                onClick={() => onSetDefault(d.key, profile.id)}>
                {on && <Check size={12} aria-hidden />} {on ? d.label : `Use for ${d.short}`}
              </button>
            );
          })}
        </div>
      </div>
      <div className="instr-usedby-row">
        <span className="instr-usedby-label" id="instr-usedby-models">Models</span>
        <div className="row" role="group" aria-labelledby="instr-usedby-models">
          {using.map((m) => (
            <span key={m.id} className="instr-chip">
              <span className="chat-title">{m.display_name}</span>
              <button type="button" className="instr-chip-x" aria-label={`Stop using this profile for ${m.display_name}`} title="Remove"
                onClick={() => onSetModel(m.id, null)}>
                <X size={12} aria-hidden />
              </button>
            </span>
          ))}
          {using.length === 0 && <span className="faint">No model uses it by default.</span>}
          {others.length > 0 && (
            <select className="input instr-add-model" aria-label="Use this profile for a model" value={adding}
              onChange={(e) => {
                const id = e.target.value;
                setAdding("");
                if (id) onSetModel(id, profile.id);
              }}>
              <option value="">Add a model…</option>
              {others.map((m) => {
                const now = titleOf(modelProfiles[m.id]);
                return <option key={m.id} value={m.id}>{m.display_name}{now ? ` (now: ${now})` : ""}</option>;
              })}
            </select>
          )}
        </div>
      </div>
      <p className="instr-note">A model's profile replaces the Chat default for chats with that model, in Chat mode.</p>
    </section>
  );
}

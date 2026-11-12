import { ChatCircle, ListChecks, Robot, Check, type IconProps } from "@phosphor-icons/react";
import type { ComponentType } from "react";
import { Dropdown } from "./Dropdown";

export type ChatMode = "chat" | "plan" | "agent";

const MODES: { id: ChatMode; name: string; desc: string; icon: ComponentType<IconProps> }[] = [
  { id: "chat", name: "Chat", desc: "Plain chat. No tools are sent.", icon: ChatCircle },
  { id: "plan", name: "Plan", desc: "Read-only tools; ends with a plan you can run or edit.", icon: ListChecks },
  { id: "agent", name: "Agent", desc: "All tools for this chat; each still asks unless allowed.", icon: Robot },
];

/**
 * Chat / Plan / Agent picker for the composer bar. The trigger keeps
 * aria-label "Chat mode" and exposes the current mode as its `value`.
 */
export function ModeMenu({ mode, onChange, disabled }: { mode: ChatMode; onChange: (m: ChatMode) => void; disabled?: boolean }) {
  const cur = MODES.find((m) => m.id === mode) ?? MODES[0];
  const Icon = cur.icon;
  return (
    <Dropdown
      label="Chat mode"
      value={mode}
      title={`${cur.name} mode: ${cur.desc}`}
      disabled={disabled}
      className={`mode-menu mode-${mode}`}
      width={360}
      button={<><Icon size={14} weight={mode === "chat" ? "regular" : "fill"} aria-hidden /><span>{cur.name}</span></>}
    >
      {(close) =>
        MODES.map((m) => {
          const MIcon = m.icon;
          const on = m.id === mode;
          return (
            <button key={m.id} type="button" role="menuitemradio" aria-checked={on} tabIndex={-1} className="dd-item"
              data-mode={m.id}
              onClick={() => { if (!on) onChange(m.id); close(); }}>
              <MIcon size={15} weight={on ? "fill" : "regular"} aria-hidden className="dd-item-icon" />
              <span className="dd-item-body">
                <span className="dd-item-name">{m.name}</span>
                <span className="dd-item-desc">{m.desc}</span>
              </span>
              {on ? <Check size={13} weight="bold" aria-hidden className="dd-item-mark" /> : <span className="dd-item-mark" />}
            </button>
          );
        })
      }
    </Dropdown>
  );
}

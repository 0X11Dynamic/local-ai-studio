import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { Wrench, Plugs } from "@phosphor-icons/react";
import type { ChatSettings } from "../../lib/api2";
import { api3, type ToolCostEntry, type ToolCosts } from "../../lib/api3";
import { Dropdown } from "./Dropdown";

const k = (n: number) => (n >= 10000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);

const STATE: Record<string, string> = { connecting: "Connecting", disconnected: "Not connected", error: "Error" };

/** This chat's MCP choice after flipping one server (old allowlists become the new denylist). */
export function toggleServer(s: ChatSettings, entries: ToolCostEntry[], id: string): ChatSettings {
  const mcp = entries.filter((e) => e.kind === "mcp");
  let disabled = [...(s.disabled_servers ?? [])];
  // Older chats ticked an allowlist ("none ticked = all"): turn it into
  // per-chat "off" entries so servers added later stay on.
  if (s.tool_servers.length > 0) {
    for (const e of mcp) if (!s.tool_servers.includes(e.id) && !disabled.includes(e.id)) disabled.push(e.id);
  }
  disabled = disabled.includes(id) ? disabled.filter((x) => x !== id) : [...disabled, id];
  return { ...s, tool_servers: [], disabled_servers: disabled };
}

/**
 * Composer "Tools" menu (aria-label "Chat tools"): the built-in tools and
 * each MCP server with its tool count and token cost; unticking one leaves
 * its tools out of this chat only. Items are menuitemcheckbox elements with
 * aria-label "<server name> tools" (e.g. "filesystem tools"); the built-in entry is
 * labelled with its name, e.g. "Built-in tools (This PC)", and "Web search" has its own item.
 */
export function ToolsMenu({ conversationId, settings, mode, onChange, onManage, refreshKey }: {
  conversationId: string | null;
  settings: ChatSettings;
  mode: "chat" | "plan" | "agent";
  onChange: (next: ChatSettings) => void;
  onManage: () => void;
  /** Changes when the loaded model or chat settings change (refetch). */
  refreshKey: string;
}) {
  const [costs, setCosts] = useState<ToolCosts | null>(null);
  const [tick, setTick] = useState(0);
  const load = useCallback(() => {
    if (!conversationId) {
      setCosts(null);
      return;
    }
    const id = conversationId;
    api3.chatToolCosts(id).then((c) => setCosts(c)).catch(() => setCosts(null));
  }, [conversationId]);

  useEffect(() => {
    load();
  }, [load, refreshKey, tick]);

  // Servers connecting or dropping change what is offered.
  useEffect(() => {
    let t: ReturnType<typeof setTimeout> | null = null;
    const un = listen("mcp://status", () => {
      if (t) clearTimeout(t);
      t = setTimeout(() => setTick((x) => x + 1), 300);
    });
    return () => {
      if (t) clearTimeout(t);
      un.then((f) => f());
    };
  }, []);

  const entries = costs?.entries ?? [];
  const on = entries.filter((e) => e.in_chat && (e.kind !== "mcp" || e.enabled));
  const live = on.filter((e) => e.state === "connected");
  // Servers loaded on demand send only a name list up front, not their definitions.
  const tokens = live.reduce((a, e) => a + (e.deferred ? 0 : e.tokens), 0);
  const deferOn = settings.defer_tools !== false;
  const deferred = live.filter((e) => e.deferred);
  const tools = live.reduce((a, e) => a + e.tool_count, 0);
  const sends = mode !== "chat";

  const flip = (e: ToolCostEntry) => {
    const next = e.kind === "builtin" ? { ...settings, builtin_tools: !settings.builtin_tools } : e.kind === "websearch" ? { ...settings, web_search: settings.web_search === false } : toggleServer(settings, entries, e.id);
    // Show the change at once; the refetch confirms it.
    setCosts((c) => c && { ...c, entries: c.entries.map((x) => (x.id === e.id ? { ...x, in_chat: !x.in_chat } : x)) });
    onChange(next);
  };

  const meta = (e: ToolCostEntry) => {
    if (e.kind === "mcp" && !e.enabled) return "turned off on the MCP page";
    if (e.state !== "connected") return STATE[e.state] ?? e.state;
    if (e.deferred && e.in_chat) return `${e.tool_count} tools, loaded on demand`;
    return `${e.tool_count} tool${e.tool_count === 1 ? "" : "s"}, ${costs?.exact ? "" : "about "}${e.tokens.toLocaleString()} tok`;
  };

  return (
    <Dropdown
      label="Chat tools"
      value={`${on.length}/${entries.length}`}
      disabled={!conversationId}
      className={`tools-menu${sends ? "" : " idle"}`}
      width={340}
      title={sends ? `${tools} tools from ${live.length} source(s), ${tokens.toLocaleString()} tokens of context` : "Chat mode sends no tools; these choices apply in Plan and Agent mode"}
      button={
        <>
          <Wrench size={14} aria-hidden />
          <span>Tools</span>
          {costs && <span className="dd-count mono">{sends ? `${tools} · ${k(tokens)} tok` : "off"}</span>}
        </>
      }
    >
      {() => (
        <>
          <div className="dd-head">
            <span>Tools in this chat</span>
            <span className="faint mono">{sends ? `${costs?.exact ? "" : "~"}${tokens.toLocaleString()} tok` : "not sent"}</span>
          </div>
          {!sends && <div className="dd-note">Chat mode sends no tools. These choices apply in Plan and Agent mode.</div>}
          {mode === "plan" && <div className="dd-note">Plan mode offers only read-only tools; counts show those.</div>}
          {entries.map((e) => {
            const off = e.kind === "mcp" && !e.enabled;
            const checked = e.in_chat && !off;
            return (
              <button key={e.id} type="button" role="menuitemcheckbox" aria-checked={checked} aria-disabled={off || undefined}
                aria-label={e.kind !== "mcp" ? e.name : `${e.name} tools`} tabIndex={-1} className={`dd-item dd-check${off ? " is-off" : ""}`}
                title={e.error ?? undefined}
                onClick={() => { if (!off) flip(e); }}>
                <span className={`dd-box${checked ? " on" : ""}`} aria-hidden />
                <span className="dd-item-body">
                  <span className="dd-item-name">{e.name}</span>
                  <span className={`dd-item-desc${e.state === "error" ? " danger-text" : ""}`}>{meta(e)}</span>
                </span>
              </button>
            );
          })}
          {!entries.some((e) => e.kind === "mcp") && <div className="dd-note">No MCP servers yet.</div>}
          <div className="dd-sep" role="separator" />
          <button type="button" role="menuitemcheckbox" aria-checked={deferOn} aria-label="Load tool details on demand" tabIndex={-1} className="dd-item dd-check"
            title="With many tools (a server with more than 30, or more than 60 in all), the big servers' tools are listed by name and the model loads the ones it needs. Shorter prompts, faster first replies."
            onClick={() => onChange({ ...settings, defer_tools: !deferOn })}>
            <span className={`dd-box${deferOn ? " on" : ""}`} aria-hidden />
            <span className="dd-item-body">
              <span className="dd-item-name">Load tool details on demand</span>
              <span className="dd-item-desc">{!deferOn ? "Off: every tool's details are sent" : deferred.length > 0 ? `On for ${deferred.map((e) => e.name).join(", ")}` : "On when a server has over 30 tools, or over 60 in all"}</span>
            </span>
          </button>
          <button type="button" role="menuitem" tabIndex={-1} className="dd-item dd-link" onClick={onManage}>
            <Plugs size={14} aria-hidden className="dd-item-icon" />
            <span className="dd-item-body"><span className="dd-item-name">Manage MCP servers</span></span>
          </button>
        </>
      )}
    </Dropdown>
  );
}

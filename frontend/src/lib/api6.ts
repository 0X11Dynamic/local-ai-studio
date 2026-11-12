// Model capabilities (thinking, tools, vision, audio), detected by the backend
// from the model's chat template and its vision projector. See
// crates/model-registry/src/caps.rs. `null` always means unknown.
import { invoke } from "@tauri-apps/api/core";
import type { EngineStatus } from "./api";

export type ThinkingControl = "unknown" | "none" | "toggle" | "effort" | "always";

export interface Capabilities {
  /** 0 = not detected yet (older row; filled in at next start). */
  version: number;
  source: "gguf" | "hf" | "";
  thinking: boolean | null;
  thinking_control: ThinkingControl;
  /** Chat-template kwarg for the on/off toggle: "enable_thinking" or "thinking". */
  thinking_kwarg: string | null;
  thinking_default: boolean | null;
  effort_levels: string[];
  effort_default: string | null;
  tools: boolean | null;
  vision: boolean | null;
  audio: boolean | null;
  template_media: string[];
  needs_projector: boolean;
}

/** Read caps from a metadata object (ModelView.meta or RepoDetails). */
export function capsOf(meta: unknown): Capabilities | null {
  const c = (meta as { caps?: Capabilities } | null | undefined)?.caps;
  return c && c.version > 0 ? c : null;
}

export interface CapBadge {
  label: string;
  title: string;
  tone: "" | "accent" | "muted";
}

/** Short badges for lists and headers. `hasProjector` is a fallback for rows not detected yet. */
export function capBadges(c: Capabilities | null, hasProjector = false): CapBadge[] {
  if (!c) {
    return hasProjector ? [{ label: "vision", title: "Vision projector found next to this model", tone: "" }] : [];
  }
  const out: CapBadge[] = [];
  if (c.vision) out.push({ label: "vision", title: c.source === "hf" ? "The repo has a vision projector (image input)" : "Image input through the vision projector", tone: "" });
  else if (c.needs_projector) out.push({ label: "vision: no projector", title: "The chat template handles images, but no vision projector (mmproj) file was found", tone: "muted" });
  if (c.audio) out.push({ label: "audio", title: "The projector has an audio encoder (audio input)", tone: "" });
  switch (c.thinking_control) {
    case "toggle":
      out.push({ label: "thinking", title: `Thinking can be turned on or off (${c.thinking_default == null ? "default unknown" : c.thinking_default ? "on by default" : "off by default"})`, tone: "" });
      break;
    case "effort":
      out.push({ label: "reasoning effort", title: `Reasoning effort levels: ${c.effort_levels.join(", ")}`, tone: "" });
      break;
    case "always":
      out.push({ label: "always thinks", title: "The template always starts a reasoning block; it can't be turned off", tone: "" });
      break;
  }
  if (c.thinking_control === "toggle" && c.effort_levels.length > 0) {
    out.push({ label: `effort ${c.effort_levels.join("/")}`, title: `Reasoning effort levels${c.effort_default ? `, default ${c.effort_default}` : ""}`, tone: "" });
  }
  if (c.tools) out.push({ label: "tools", title: "The chat template renders tool definitions (tool calling)", tone: "" });
  if (c.thinking_control === "unknown") out.push({ label: "capabilities unknown", title: "No chat template to read capabilities from", tone: "muted" });
  return out;
}

const yn = (v: boolean | null, yes: string, no: string) => (v == null ? "unknown" : v ? yes : no);

/** One-line descriptions for a details panel. */
export function capDetails(c: Capabilities | null): { label: string; value: string }[] {
  if (!c) return [];
  const thinking = (() => {
    switch (c.thinking_control) {
      case "toggle": {
        const d = c.thinking_default == null ? "" : c.thinking_default ? ", on by default" : ", off by default";
        const e = c.effort_levels.length ? `; effort ${c.effort_levels.join(" / ")}${c.effort_default ? ` (default ${c.effort_default})` : ""}` : "";
        return `on/off toggle${d}${e}`;
      }
      case "effort":
        return `effort ${c.effort_levels.join(" / ")}${c.effort_default ? ` (default ${c.effort_default})` : ""}`;
      case "always":
        return "always on (can't be turned off)";
      case "none":
        return "no";
      default:
        return "unknown";
    }
  })();
  const vision = c.vision ? "yes (projector)" : c.needs_projector ? "template supports images, but no projector file" : yn(c.vision, "yes", "no");
  return [
    { label: "Thinking", value: thinking },
    { label: "Tool calling", value: yn(c.tools, "yes", "no") },
    { label: "Vision", value: vision },
    { label: "Audio input", value: yn(c.audio, "yes", "no") },
  ];
}

// ---------- thinking controls and chat attachments (IPC) ----------

/** What the loaded model takes now (chat_model_support). */
export interface ModelSupport {
  model_id: string;
  model_name: string;
  caps: Capabilities;
  /** A vision projector (mmproj) file is registered next to the model. */
  has_projector: boolean;
  /** ...and it was loaded. */
  projector_loaded: boolean;
  /** Images / audio can be sent now. */
  images: boolean;
  audio: boolean;
}

/** A file attached to a chat message (pending on the composer, or sent). */
export interface Attachment {
  id: string;
  name: string;
  kind: "text" | "pdf" | "image" | "audio";
  mime: string;
  size: number;
  /** Stored copy inside <data dir>\attachments\<conversation id>\. */
  file: string;
  text_file?: string | null;
  chars?: number | null;
  /** Tokens of the quoted block when attached (text and PDF). */
  tokens?: number | null;
}

export function parseAttachments(json: string | null | undefined): Attachment[] {
  try {
    const v = json ? JSON.parse(json) : [];
    return Array.isArray(v) ? (v as Attachment[]) : [];
  } catch {
    return [];
  }
}

/** Thinking UI for a model: a toggle only for "toggle" models, effort only when levels exist. */
export function thinkingUi(c: Capabilities | null | undefined): { toggle: boolean; effort: string[]; always: boolean } {
  if (!c) return { toggle: false, effort: [], always: false };
  const toggle = c.thinking_control === "toggle" && !!c.thinking_kwarg;
  const effort = ["toggle", "effort", "always"].includes(c.thinking_control) ? c.effort_levels : [];
  return { toggle, effort, always: c.thinking_control === "always" };
}

export const api6 = {
  /** null when no model is loaded. */
  modelSupport: () => invoke<ModelSupport | null>("chat_model_support"),
  attach: (id: string, path: string) => invoke<Attachment>("chat_attach", { id, path }),
  detach: (id: string, attachmentId: string) => invoke<boolean>("chat_detach", { id, attachmentId }),
  pending: (id: string) => invoke<Attachment[]>("chat_pending_attachments", { id }),
  /** Reload the loaded model with load_projector on (same settings otherwise). */
  reloadWithProjector: () => invoke<EngineStatus>("reload_with_projector"),
};

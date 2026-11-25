// Live text of replies that are still streaming, kept outside React state so
// a 40 ms delta re-renders only the one message row that shows it (not the
// chat list, the sidebar or the composer). Updates are coalesced to one
// publish per animation frame.
export interface LiveText { content: string; reasoning: string }

const pending = new Map<string, LiveText>();
const published = new Map<string, LiveText>();
const subs = new Map<string, Set<() => void>>();
let raf = 0;

function flush() {
  raf = 0;
  for (const [id, v] of pending) {
    published.set(id, v);
    subs.get(id)?.forEach((f) => f());
  }
  pending.clear();
}

export const streamStore = {
  append(id: string, content: string, reasoning: string) {
    const cur = pending.get(id) ?? published.get(id) ?? { content: "", reasoning: "" };
    pending.set(id, { content: cur.content + content, reasoning: cur.reasoning + reasoning });
    if (!raf) raf = requestAnimationFrame(flush);
  },
  /** True when the message has live text (decided at append time, not publish time). */
  has: (id: string) => pending.has(id) || published.has(id),
  remove(id: string) {
    pending.delete(id);
    published.delete(id);
    subs.get(id)?.forEach((f) => f());
  },
  clear() {
    const ids = [...published.keys(), ...pending.keys()];
    pending.clear();
    published.clear();
    ids.forEach((id) => subs.get(id)?.forEach((f) => f()));
  },
  get: (id: string): LiveText | undefined => published.get(id),
  subscribe(id: string, f: () => void) {
    let s = subs.get(id);
    if (!s) subs.set(id, (s = new Set()));
    s.add(f);
    return () => { s.delete(f); if (s.size === 0) subs.delete(id); };
  },
};

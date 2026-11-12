// Monospace multi-line editor with line numbers, word wrap toggle and find.
import { useMemo, useRef, useState } from "react";

export function Editor({
  value, onChange, label, rows = 12, readOnly = false, id,
}: { value: string; onChange?: (v: string) => void; label: string; rows?: number; readOnly?: boolean; id?: string }) {
  const ta = useRef<HTMLTextAreaElement>(null);
  const gutter = useRef<HTMLDivElement>(null);
  const [wrap, setWrap] = useState(true);
  const [find, setFind] = useState("");
  const [hit, setHit] = useState(0);
  const lines = useMemo(() => value.split("\n").length, [value]);
  const matches = useMemo(() => {
    if (!find) return [] as number[];
    const out: number[] = [];
    const v = value.toLowerCase();
    const f = find.toLowerCase();
    for (let i = v.indexOf(f); i >= 0 && out.length < 10000; i = v.indexOf(f, i + f.length)) out.push(i);
    return out;
  }, [value, find]);

  const jump = (dir: 1 | -1) => {
    if (!matches.length || !ta.current) return;
    const next = (hit + dir + matches.length) % matches.length;
    setHit(next);
    ta.current.focus();
    ta.current.setSelectionRange(matches[next], matches[next] + find.length);
  };

  return (
    <div className="editor">
      <div className="editor-bar row">
        <input className="input" style={{ width: 180, height: 24 }} placeholder="Find" aria-label={`Find in ${label}`} value={find}
          onChange={(e) => { setFind(e.target.value); setHit(-1); }}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); jump(e.shiftKey ? -1 : 1); } }} />
        <span className="mono faint">{find ? `${matches.length ? hit + 1 : 0}/${matches.length}` : ""}</span>
        <button type="button" className="btn ghost sm" onClick={() => jump(-1)} disabled={!matches.length}>Prev</button>
        <button type="button" className="btn ghost sm" onClick={() => jump(1)} disabled={!matches.length}>Next</button>
        <span className="spacer" />
        <label className="check"><input type="checkbox" checked={wrap} onChange={(e) => setWrap(e.target.checked)} /> Wrap</label>
        <span className="mono faint">{lines} lines, {value.length} chars</span>
      </div>
      <div className="editor-body">
        {!wrap && (
          <div className="editor-gutter mono" ref={gutter} aria-hidden>
            {Array.from({ length: lines }, (_, i) => <div key={i}>{i + 1}</div>)}
          </div>
        )}
        <textarea
          id={id}
          ref={ta}
          aria-label={label}
          className="editor-text mono"
          value={value}
          readOnly={readOnly}
          rows={rows}
          wrap={wrap ? "soft" : "off"}
          spellCheck={false}
          onChange={(e) => onChange?.(e.target.value)}
          onScroll={(e) => { if (gutter.current) gutter.current.scrollTop = e.currentTarget.scrollTop; }}
        />
      </div>
    </div>
  );
}

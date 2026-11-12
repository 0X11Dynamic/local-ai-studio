import { useEffect, useRef, useState } from "react";
import { useEngineLogs } from "../lib/engine";

export function LogView() {
  const logs = useEngineLogs();
  const ref = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs, follow]);

  const q = query.trim().toLowerCase();
  const lines = q ? logs.filter((l) => l.toLowerCase().includes(q)) : logs;

  return (
    <div>
      <div className="row" style={{ marginBottom: 6 }}>
        <input className="input" placeholder="Search log" aria-label="Search backend log" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: 240 }} />
        <label className="check"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow</label>
        <span className="spacer" />
        <span className="faint mono">{lines.length} of {logs.length} lines</span>
      </div>
      <div className="log" ref={ref} role="log" aria-live="off">
        {lines.length ? lines.join("\n") : <span className="faint">Nothing yet. Output from llama-server appears here while a model loads and runs.</span>}
      </div>
    </div>
  );
}

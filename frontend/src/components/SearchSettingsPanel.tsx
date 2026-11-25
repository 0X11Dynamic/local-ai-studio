import { useEffect, useState } from "react";
import { errorMessage } from "../lib/api";
import { api7, type SearchProvider, type SearchSettings, type SearchSettingsView, type SearchTestView } from "../lib/api7";

const PROVIDERS: { id: SearchProvider; name: string; note: string }[] = [
  { id: "duckduckgo", name: "DuckDuckGo", note: "No key needed. It uses DuckDuckGo's unofficial HTML page, so it can be rate-limited or ask for a captcha." },
  { id: "langsearch", name: "LangSearch", note: "Free key from langsearch.com/api-keys (daily allowance, resets 00:00 UTC)." },
  { id: "brave", name: "Brave Search API", note: "Free key from brave.com/search/api. Official and reliable." },
  { id: "searxng", name: "SearXNG", note: "Your own instance: the most private. It must have the JSON format enabled." },
];

/** Settings > Search: provider, key (write-only), limits and a Test button. */
export function SearchSettingsPanel() {
  const [view, setView] = useState<SearchSettingsView | null>(null);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<SearchTestView | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [urlDraft, setUrlDraft] = useState<string | null>(null);

  useEffect(() => {
    api7.getSearchSettings().then(setView).catch((e) => setErr(errorMessage(e)));
  }, []);

  if (!view) return <>{err ? <div className="notice error">{err}</div> : <p className="faint">Loading search settings</p>}</>;
  const s = view.settings;
  const provider = PROVIDERS.find((p) => p.id === s.provider) ?? PROVIDERS[0];
  const keyed = s.provider === "langsearch" || s.provider === "brave";
  const hasKey = s.provider === "langsearch" ? view.has_langsearch_key : view.has_brave_key;

  const act = async (fn: () => Promise<SearchSettingsView | void>, ok?: string) => {
    setErr(null);
    setMsg(null);
    try {
      const v = await fn();
      if (v) setView(v);
      if (ok) setMsg(ok);
    } catch (e) {
      setErr(errorMessage(e));
    }
  };
  const save = (next: SearchSettings, ok?: string) => act(() => api7.setSearchSettings(next), ok);
  const runTest = async () => {
    setBusy(true);
    setTest(null);
    try {
      setTest(await api7.testWebSearch());
    } catch (e) {
      setTest({ ok: false, provider: provider.name, results: [], error: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="form-grid" style={{ maxWidth: 720 }} aria-label="Search settings">
      {err && <div className="notice error wide">{err}</div>}
      {msg && <div className="notice wide">{msg}</div>}
      <div className="field wide">
        <label htmlFor="search-provider">Search provider</label>
        <select id="search-provider" className="input" style={{ maxWidth: 280 }} value={s.provider}
          onChange={(e) => { setTest(null); setKey(""); save({ ...s, provider: e.target.value as SearchProvider }); }}>
          {PROVIDERS.map((p) => <option key={p.id} value={p.id}>{p.name}{p.id === "duckduckgo" ? " (default)" : ""}</option>)}
        </select>
        <span className="help">{provider.note} Models use it through the web_search tool, which runs without asking; untick it in a chat's Tools menu or a task's tool list to turn it off there.</span>
      </div>

      {keyed && (
        <div className="field wide">
          <label htmlFor="search-key">{provider.name} API key</label>
          <div className="row">
            <input id="search-key" className="input mono" type="password" autoComplete="off" style={{ width: 340 }}
              placeholder={hasKey ? "A key is stored" : "Paste your key"} value={key} onChange={(e) => setKey(e.target.value)} />
            <button className="btn sm" disabled={!key.trim()}
              onClick={() => act(async () => { const v = await api7.setSearchKey(s.provider as "langsearch" | "brave", key); setKey(""); return v; }, "Key saved to Windows Credential Manager.")}>Save</button>
            {hasKey && <button className="btn sm ghost" onClick={() => act(() => api7.setSearchKey(s.provider as "langsearch" | "brave", null), "Key removed.")}>Remove</button>}
          </div>
          <span className="help">Stored in Windows Credential Manager and sent only to {provider.name}. It is never shown again, saved in the database or written to logs.</span>
        </div>
      )}

      {s.provider === "searxng" && (
        <div className="field wide">
          <label htmlFor="search-url">SearXNG address</label>
          <div className="row">
            <input id="search-url" className="input mono" style={{ width: 340 }} placeholder="http://localhost:8080" value={urlDraft ?? s.searxng_url}
              onChange={(e) => setUrlDraft(e.target.value)} />
            <button className="btn sm" disabled={urlDraft == null || urlDraft === s.searxng_url}
              onClick={() => save({ ...s, searxng_url: urlDraft ?? "" }, "Address saved.").then(() => setUrlDraft(null))}>Save</button>
          </div>
          <span className="help">Only this address may be a local one. The instance needs <span className="mono">json</span> under <span className="mono">search.formats</span> in its settings.yml; Test checks it.</span>
        </div>
      )}

      {s.provider === "langsearch" && (
        <div className="field wide">
          <label className="check">
            <input type="checkbox" checked={s.include_text} onChange={(e) => save({ ...s, include_text: e.target.checked })} />
            <span>Include page text (the model may not need web_fetch, but results are longer)</span>
          </label>
        </div>
      )}

      <div className="field">
        <label htmlFor="search-count">Results per search</label>
        <input id="search-count" className="input" type="number" min={1} max={10} value={s.count}
          onChange={(e) => save({ ...s, count: Math.min(10, Math.max(1, Number(e.target.value) || 5)) })} />
      </div>
      <div className="field">
        <label htmlFor="search-cap">Searches allowed per reply or task</label>
        <input id="search-cap" className="input" type="number" min={1} max={200} value={s.cap}
          onChange={(e) => save({ ...s, cap: Math.min(200, Math.max(1, Number(e.target.value) || 20)) })} />
        <span className="help">Stops a looping model from using up a quota.</span>
      </div>

      <div className="field wide">
        <div className="row">
          <button className="btn sm" disabled={busy} onClick={runTest}>{busy ? "Searching" : "Test"}</button>
          <span className="faint">Runs one search for "rust programming language" with these settings.</span>
        </div>
        {test && !test.ok && <div className="notice error" role="alert" style={{ marginTop: 8 }}>{test.provider}: {test.error}</div>}
        {test?.ok && (
          <ol className="search-test" aria-label="Test results" style={{ marginTop: 8, paddingLeft: 20 }}>
            {test.results.length === 0 && <li className="faint">The search worked but found nothing.</li>}
            {test.results.map((r) => (
              <li key={r.url}>
                <strong>{r.title}</strong>
                <div className="faint mono" style={{ wordBreak: "break-all" }}>{r.url}</div>
                {r.snippet && <div>{r.snippet}</div>}
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

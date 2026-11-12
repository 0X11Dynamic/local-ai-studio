import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { errorMessage } from "../lib/api";
import { api2, type DownloadEvent, type RepoView, type SearchResult, type VariantView } from "../lib/api2";
import { bytes, percent, rate } from "../lib/format";
import { capBadges, capDetails, capsOf } from "../lib/api6";
import type { PageId } from "../App";

const QUANT_NOTES: Record<string, string> = {
  Q2: "smallest, noticeable quality loss",
  Q3: "small, some quality loss",
  Q4: "good balance of size and quality",
  Q5: "near-original quality, larger",
  Q6: "very close to original",
  Q8: "practically lossless, large",
  IQ1: "experimental, very lossy",
  IQ2: "very small, lossy",
  IQ3: "small, importance-weighted",
  IQ4: "compact, good quality",
  F16: "unquantized, very large",
  BF16: "unquantized, very large",
  F32: "full precision, huge",
};

function quantNote(q: string | null): string {
  if (!q) return "unrecognised format name";
  const key = Object.keys(QUANT_NOTES).sort((a, b) => b.length - a.length).find((k) => q.startsWith(k));
  return key ? QUANT_NOTES[key] : "";
}

const FIT_LABEL: Record<VariantView["fit"], string> = {
  gpu: "Fits in GPU",
  partial: "GPU + RAM",
  cpu: "RAM only",
  too_big: "Too large",
};

function eta(s: number | null): string {
  if (s == null) return "";
  if (s < 60) return `${s}s left`;
  if (s < 3600) return `${Math.round(s / 60)} min left`;
  return `${(s / 3600).toFixed(1)} h left`;
}

export function Discover({ go }: { go: (p: PageId) => void }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("downloads");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [capFilter, setCapFilter] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [repo, setRepo] = useState<RepoView | null>(null);
  const [repoErr, setRepoErr] = useState<string | null>(null);
  const [repoLoading, setRepoLoading] = useState(false);
  const [projector, setProjector] = useState<string>("");
  const [downloads, setDownloads] = useState<Record<string, DownloadEvent>>({});
  const [actionErr, setActionErr] = useState<Record<string, string>>({});

  // Responses can arrive out of order; only the newest search may update the list.
  const searchSeq = useRef(0);
  const search = useCallback(async (q: string, s: string) => {
    const seq = ++searchSeq.current;
    setSearching(true);
    setSearchErr(null);
    try {
      const r = await api2.hfSearch(q, s);
      if (seq === searchSeq.current) setResults(r);
    } catch (e) {
      if (seq === searchSeq.current) {
        setSearchErr(errorMessage(e));
        setResults([]);
      }
    } finally {
      if (seq === searchSeq.current) setSearching(false);
    }
  }, []);

  // `refresh` updates the open repo in place (no skeleton, keeps the projector choice).
  const repoSeq = useRef(0);
  const openRepo = useCallback(async (id: string, refresh = false) => {
    const seq = ++repoSeq.current;
    setSelected(id);
    if (!refresh) setRepoLoading(true);
    setRepoErr(null);
    try {
      const r = await api2.hfRepo(id);
      if (seq !== repoSeq.current) return;
      setRepo(r);
      if (!refresh) setProjector("");
    } catch (e) {
      if (seq !== repoSeq.current) return;
      setRepoErr(errorMessage(e));
      setRepo(null);
    } finally {
      if (seq === repoSeq.current) setRepoLoading(false);
    }
  }, []);

  useEffect(() => {
    search("", "downloads");
    api2.listDownloads().then((l) => setDownloads(Object.fromEntries(l.map((d) => [d.id, d]))));
    const un = listen<DownloadEvent>("download://event", (e) => {
      setDownloads((d) => ({ ...d, [e.payload.id]: e.payload }));
    });
    return () => {
      un.then((f) => f());
    };
  }, [search]);

  // Refresh the open repo when one of its downloads finishes or stops.
  const doneKey = Object.values(downloads).filter((d) => d.repo === repo?.details.id && !["running", "queued"].includes(d.state)).map((d) => d.id + d.state).join(",");
  useEffect(() => {
    if (repo && doneKey) openRepo(repo.details.id, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doneKey]);

  const visible = useMemo(
    () => (results ?? []).filter((r) => capFilter === "all" || r.capabilities.includes(capFilter)),
    [results, capFilter],
  );

  const projectors = repo?.variants.filter((v) => v.is_projector) ?? [];
  // Recommended variant first, then by size.
  const models = (repo?.variants.filter((v) => !v.is_projector) ?? []).slice().sort((a, b) =>
    a.key === repo?.recommended ? -1 : b.key === repo?.recommended ? 1 : a.total_size - b.total_size,
  );
  const dlFor = (v: VariantView) => Object.values(downloads).find((d) => d.repo === repo?.details.id && d.variant_key === v.key && !["completed", "cancelled"].includes(d.state));

  const start = async (v: VariantView) => {
    if (!repo) return;
    setActionErr((a) => ({ ...a, [v.key]: "" }));
    try {
      await api2.downloadVariant(repo.details.id, v.key, projector || null);
      openRepo(repo.details.id, true);
    } catch (e) {
      setActionErr((a) => ({ ...a, [v.key]: errorMessage(e) }));
    }
  };

  const act = async (fn: () => Promise<void>, key: string) => {
    try {
      await fn();
    } catch (e) {
      setActionErr((a) => ({ ...a, [key]: errorMessage(e) }));
    }
  };

  const active = Object.values(downloads).filter((d) => !["completed", "cancelled"].includes(d.state));
  const finished = Object.values(downloads).filter((d) => ["completed", "cancelled"].includes(d.state));

  return (
    <div className="page" style={{ maxWidth: 1400 }}>
      <form
        className="row"
        style={{ marginBottom: 14 }}
        onSubmit={(e) => {
          e.preventDefault();
          search(query, sort);
        }}
      >
        <input className="input" style={{ flex: 1, minWidth: 260, height: 32 }} placeholder="Search Hugging Face GGUF models, or paste a repo id or URL" aria-label="Search models" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select className="input" aria-label="Sort results" value={sort} onChange={(e) => { setSort(e.target.value); search(query, e.target.value); }}>
          <option value="downloads">Most downloaded</option>
          <option value="likes">Most liked</option>
          <option value="lastModified">Recently updated</option>
          <option value="createdAt">Newest</option>
        </select>
        <select className="input" aria-label="Filter by capability" value={capFilter} onChange={(e) => setCapFilter(e.target.value)}>
          <option value="all">All types</option>
          <option value="text">Text / chat</option>
          <option value="vision">Vision</option>
          <option value="embedding">Embeddings</option>
        </select>
        <button className="btn primary" type="submit" disabled={searching}>{searching ? "Searching" : "Search"}</button>
      </form>

      {active.length > 0 && (
        <div className="panel" style={{ marginBottom: 14, display: "grid", gap: 10 }}>
          <strong style={{ fontWeight: 600 }}>Downloads</strong>
          {active.map((d) => (
            <div key={d.id} style={{ display: "grid", gap: 4 }}>
              <div className="row">
                <span className="mono">{d.repo}/{d.variant_key}</span>
                <span className="tag">{d.state}</span>
                <span className="spacer" />
                {d.state === "running" || d.state === "queued" ? (
                  <button className="btn sm" onClick={() => act(() => api2.pauseDownload(d.id), d.id)}>Pause</button>
                ) : (
                  <button className="btn sm" onClick={() => act(() => api2.resumeDownload(d.id), d.id)}>{d.state === "failed" ? "Retry" : "Resume"}</button>
                )}
                <button className="btn sm danger" onClick={() => act(() => api2.cancelDownload(d.id), d.id)}>Cancel</button>
              </div>
              <div className="progress"><span style={{ width: `${percent(d.downloaded, d.total)}%` }} /></div>
              <div className="mono faint">
                {bytes(d.downloaded)} of {bytes(d.total)}, file {Math.min(d.files_done + 1, d.files_total)} of {d.files_total} {rate(d.bytes_per_sec)} {eta(d.eta_secs)}
              </div>
              {(d.error || actionErr[d.id]) && <div className="danger-text">{actionErr[d.id] || d.error}</div>}
            </div>
          ))}
        </div>
      )}
      {finished.length > 0 && (
        <div className="row" style={{ marginBottom: 14 }}>
          <span className="faint">{finished.length} finished or cancelled download(s).</span>
          <button className="btn sm ghost" onClick={async () => { await api2.clearFinishedDownloads(); setDownloads((d) => Object.fromEntries(Object.entries(d).filter(([, v]) => !["completed", "cancelled"].includes(v.state)))); }}>Clear list</button>
        </div>
      )}

      <div className="split" style={{ gridTemplateColumns: "minmax(0, 0.7fr) minmax(0, 1.3fr)" }}>
        <div>
          {searchErr && <div className="notice error">{searchErr}</div>}
          {results == null ? (
            <div style={{ display: "grid", gap: 10 }}>{[0, 1, 2, 3, 4].map((i) => <div className="skeleton" key={i} style={{ height: 36 }} />)}</div>
          ) : visible.length === 0 && !searchErr ? (
            <div className="empty"><strong>No GGUF repositories found.</strong>Try a model family (qwen, llama, gemma, mistral) or paste a repo id like unsloth/Qwen3-4B-GGUF.</div>
          ) : (
            <table className="table">
              <thead><tr><th>Repository</th><th className="num">Downloads</th><th className="num">Likes</th></tr></thead>
              <tbody>
                {visible.map((r) => (
                  <tr key={r.id} className={`selectable${r.id === selected ? " selected" : ""}`} onClick={() => openRepo(r.id)}>
                    <td>
                      <div style={{ overflowWrap: "anywhere" }}>{r.id}</div>
                      <div className="faint mono" style={{ fontSize: 11 }}>
                        {[r.capabilities.join(" + ") || r.pipeline_tag, r.license].filter(Boolean).join(", ")}
                      </div>
                    </td>
                    <td className="num">{r.downloads.toLocaleString()}</td>
                    <td className="num">{r.likes.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <aside>
          {repoLoading ? (
            <div className="panel" style={{ display: "grid", gap: 10 }}><div className="skeleton" /><div className="skeleton" /><div className="skeleton" style={{ height: 120 }} /></div>
          ) : repoErr ? (
            <div className="notice error">{repoErr}</div>
          ) : repo ? (
            <div className="panel" style={{ display: "grid", gap: 14 }}>
              <div>
                <div style={{ fontWeight: 600, overflowWrap: "anywhere" }}>{repo.details.id}</div>
                <div className="faint">by {repo.details.author}{repo.details.gated ? ", gated (needs a Hugging Face token)" : ""}</div>
                <div className="cap-row">
                  {capBadges(capsOf(repo.details)).map((b) => (
                    <span key={b.label} className={`tag cap ${b.tone}`} title={b.title}>{b.label}</span>
                  ))}
                </div>
              </div>
              <dl className="kv">
                <dt>Architecture</dt><dd>{repo.details.architecture ?? "not reported by the Hub"}</dd>
                <dt>Parameters</dt><dd>{repo.details.parameters ? `${(repo.details.parameters / 1e9).toFixed(2)} B` : "not reported"}</dd>
                <dt>Trained context</dt><dd>{repo.details.context_length?.toLocaleString() ?? "not reported"}</dd>
                {capDetails(capsOf(repo.details)).map((d) => (
                  <div key={d.label} style={{ display: "contents" }}><dt>{d.label}</dt><dd>{d.value}</dd></div>
                ))}
                <dt>Base model</dt><dd>{repo.details.base_model ?? "not listed"}</dd>
                <dt>License</dt><dd>{repo.details.license ?? "not listed"}</dd>
                <dt>Updated</dt><dd>{repo.details.last_modified ? new Date(repo.details.last_modified).toLocaleDateString() : "unknown"}</dd>
                <dt>Saves to</dt><dd>{repo.models_dir}</dd>
              </dl>
              <p className="faint" style={{ margin: 0 }}>
                Details come from the repository. Memory needs and fit are estimates (file size plus about 1 GB), based on {repo.free_vram_bytes != null ? `${bytes(repo.free_vram_bytes)} free VRAM and ` : ""}{bytes(repo.available_ram_bytes)} available RAM right now.
              </p>
              {projectors.length > 0 && (
                <div className="field">
                  <label htmlFor="proj">Vision projector (image input) to download with the model</label>
                  <select id="proj" className="input" value={projector} onChange={(e) => setProjector(e.target.value)}>
                    <option value="">None</option>
                    {projectors.map((p) => <option key={p.key} value={p.key}>{p.key} ({bytes(p.total_size)}){p.status === "installed" ? ", installed" : ""}</option>)}
                  </select>
                </div>
              )}
              {models.length === 0 ? (
                <div className="empty">This repository has no GGUF model files.</div>
              ) : (
                <table className="table">
                  <thead><tr><th>Variant</th><th className="num">Size</th><th>For this PC (est.)</th><th /></tr></thead>
                  <tbody>
                    {models.map((v) => {
                      const d = dlFor(v);
                      return (
                        <tr key={v.key}>
                          <td>
                            <div className="row" style={{ gap: 6 }}>
                              <span className="mono">{v.quantization ?? "?"}</span>
                              {v.key === repo.recommended && <span className="tag accent">recommended for this PC</span>}
                              {v.status === "installed" && <span className="tag">installed</span>}
                              {v.status === "loaded" && <span className="tag accent">loaded</span>}
                              {v.status === "partial" && <span className="tag">partly downloaded</span>}
                              {v.status === "incomplete" && <span className="tag danger">incomplete in repo</span>}
                            </div>
                            <div className="mono faint" style={{ fontSize: 11, overflowWrap: "anywhere" }}>
                              {v.shard_count > 1 ? `${v.files.length} of ${v.shard_count} shards: ${v.files[0].path}` : v.key}
                            </div>
                            <div className="faint" style={{ fontSize: 11.5 }}>{quantNote(v.quantization)}</div>
                            {d && (
                              <div style={{ marginTop: 4 }}>
                                <div className="progress"><span style={{ width: `${percent(d.downloaded, d.total)}%` }} /></div>
                              </div>
                            )}
                            {actionErr[v.key] && <div className="danger-text" style={{ marginTop: 4 }}>{actionErr[v.key]}</div>}
                          </td>
                          <td className="num">{bytes(v.total_size)}</td>
                          <td className={v.fit === "too_big" ? "danger-text" : "dim"} style={{ whiteSpace: "nowrap" }}>
                            {FIT_LABEL[v.fit]}
                            <div className="mono faint" style={{ fontSize: 11 }}>needs about {bytes(v.est_memory_bytes)}</div>
                          </td>
                          <td>
                            <div className="row" style={{ justifyContent: "flex-end" }}>
                              {v.status === "installed" || v.status === "loaded" ? (
                                <button className="btn sm" onClick={() => { if (v.model_id) localStorage.setItem("las.selectModel", v.model_id); go("models"); }}>Load model</button>
                              ) : d ? (
                                <span className="faint">{d.state}</span>
                              ) : v.status === "incomplete" ? null : (
                                <button className={`btn sm${v.key === repo.recommended ? " primary" : ""}`} onClick={() => start(v)}>
                                  {v.status === "partial" ? "Resume download" : "Download"}
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          ) : (
            <div className="empty">Select a repository to see its real files and quantizations.</div>
          )}
        </aside>
      </div>
    </div>
  );
}

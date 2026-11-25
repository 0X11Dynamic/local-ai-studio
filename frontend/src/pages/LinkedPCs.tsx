import { useCallback, useEffect, useState } from "react";
import { Copy } from "@phosphor-icons/react";
import { errorMessage } from "../lib/api";
import { api9, showCode, type HelperSettings, type LinkStatus, type PeerShare, type PeerView, type ReachReport } from "../lib/api9";
import { when } from "../lib/format";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="btn ghost sm"
      aria-label={`Copy ${label}`}
      title={`Copy ${label}`}
      onClick={() => navigator.clipboard.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1200); })}
    >
      <Copy size={14} aria-hidden /> {done ? "Copied" : ""}
    </button>
  );
}

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const gb = (n: number) => `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)} GB`;
const ms = (v: number | null) => (v == null ? "-" : v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`);
const gib = (mib: number) => `${(mib / 1024).toFixed(mib >= 10240 ? 0 : 1)} GB`;
/** "6" (GB) -> 6144 MiB; empty -> null (the default). */
const toMib = (v: string) => { const g = Number(v.replace(/[^\d.]/g, "")); return v.trim() === "" || !(g >= 0) ? null : Math.round(g * 1024); };
const ago = (s: number) => (s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`);
const THREADS_HINT = "On small CPUs, leaving 2 cores free roughly halves the speed of the layers this PC's CPU holds (measured on a 4-core VM: 71.5 tokens/s with 2 threads, 121 with 4).";

export function LinkedPCs() {
  const [st, setSt] = useState<LinkStatus | null>(null);
  const [draft, setDraft] = useState<HelperSettings | null>(null);
  const [name, setName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [left, setLeft] = useState(0);
  const [portNoticeSeen, setPortNoticeSeen] = useState(false);
  const [reach, setReach] = useState<ReachReport | null>(null);
  const [form, setForm] = useState({ address: "", port: "", code: "" });

  const refresh = useCallback(async () => {
    const s = await api9.status();
    setSt(s);
    if (s.code) setLeft(s.code.seconds_left);
    return s;
  }, []);

  useEffect(() => {
    refresh().then((s) => { setDraft(s.helper.settings); setName(s.name); setReach(s.reach); }).catch((e) => setErr(errorMessage(e)));
    const t = setInterval(() => { if (document.visibilityState === "visible") refresh().catch(() => {}); }, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    const t = setInterval(() => setLeft((l) => Math.max(0, l - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  /** `fn` may return a message to show instead of `ok`. */
  const act = async (fn: () => Promise<unknown>, ok?: string) => {
    setBusy(true);
    setErr(null);
    setInfo(null);
    try {
      const r = await fn();
      await refresh();
      if (typeof r === "string") setInfo(r);
      else if (ok) setInfo(ok);
    } catch (e) {
      setErr(errorMessage(e));
      await refresh().catch(() => {});
    } finally {
      setBusy(false);
    }
  };

  if (!st || !draft) return <div className="page"><div className="skeleton" style={{ maxWidth: 400 }} />{err && <div className="notice error">{err}</div>}</div>;

  const h = st.helper;
  const code = st.code;
  const codeOpen = code?.state === "open" && left > 0;
  // The port actually listening (it can differ from the setting when that one was reserved or in use).
  const port = h.listen_port ?? h.settings.port;
  const devs = h.live.available_devices;
  const toggleDev = (n: string) => {
    const all = devs.map((d) => d.name);
    const cur = draft.devices.length ? draft.devices : all;
    const next = cur.includes(n) ? cur.filter((x) => x !== n) : [...cur, n];
    setDraft({ ...draft, devices: next.length === all.length ? [] : next });
  };

  return (
    <div className="page link-page">
      <p className="lede">
        Link your PCs so one can lend its GPU and memory to the other: a model too big for this PC can be split across both. The link is direct and encrypted by the app; nothing else to install.
      </p>
      {err && <div className="notice error" role="alert" style={{ marginBottom: 10 }}>{err}</div>}
      {info && <div className="notice" role="status" style={{ marginBottom: 10 }}>{info}</div>}

      <h2>This PC</h2>
      <div className="row" style={{ marginBottom: 6 }}>
        <label htmlFor="link-name" className="dim">Name other PCs see</label>
        <input id="link-name" className="input" style={{ width: 220 }} value={name} maxLength={64} onChange={(e) => setName(e.target.value)} />
        <button className="btn sm" disabled={busy || name.trim() === st.name || !name.trim()} onClick={() => act(() => api9.setName(name))}>Save</button>
      </div>
      <dl className="kv">
        <dt>Local IP</dt><dd>{st.local_ip ?? "unknown"}</dd>
        <dt>Key fingerprint</dt><dd title="Compare with what the other PC shows for this PC">{st.fingerprint}</dd>
      </dl>

      <h2>Let this PC help</h2>
      <div className="panel link-panel">
        <label className="check">
          <input
            type="checkbox"
            role="switch"
            aria-label="Let this PC help"
            checked={h.enabled}
            disabled={busy}
            onChange={(e) => { const on = e.target.checked; act(() => api9.setHelper(on, draft), on ? "This PC is helping. Linked PCs can now split models with it." : "Stopped helping."); }}
          />
          <strong>Let this PC help</strong>
          <span className="faint">Off by default and after every restart. Linked PCs can use this PC only while it's on.</span>
        </label>
        {h.live.error && <div className="notice error" style={{ marginTop: 10 }}>{h.live.error}</div>}
        {h.bind_notice && <div className="notice warn" role="status" style={{ marginTop: 10 }}>{h.bind_notice}</div>}
        {h.port_notice && !portNoticeSeen && (
          <div className="notice" role="status" style={{ marginTop: 10 }}>
            {h.port_notice} <button className="btn sm" onClick={() => setPortNoticeSeen(true)}>Dismiss</button>
          </div>
        )}
        {h.enabled && (
          <dl className="kv" style={{ marginTop: 10 }}>
            <dt>llama.cpp build</dt><dd>{h.live.runtime_id ?? "-"}</dd>
            <dt>rpc-server</dt><dd>{h.live.rpc_endpoint ?? "-"} (this PC only; reached through the encrypted link)</dd>
            <dt>Listening for linked PCs</dt><dd>{h.listening ?? "-"}</dd>
            <dt>Sharing</dt><dd>{h.live.devices.length ? h.live.devices.map((d) => `${d.name} (${(d.free_mib / 1024).toFixed(1)} GB free)`).join(", ") : "devices not reported"}</dd>
            <dt>In use by</dt><dd>{st.sessions.filter((s) => s.mode === "rpc").map((s) => s.peer_name).filter((v, i, a) => a.indexOf(v) === i).join(", ") || "nobody right now"}</dd>
          </dl>
        )}
        <div className="form-grid link-settings">
          <div className="field">
            <label htmlFor="link-port">Port</label>
            <input id="link-port" className="input mono" value={draft.port} onChange={(e) => setDraft({ ...draft, port: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
            <span className="help">TCP port for linking and the link itself. Default 47070 (outside the range Windows reserves for virtual networks). If it is reserved or in use, the next free port is used and shown here.</span>
          </div>
          <fieldset className="field wide link-limits">
            <legend>Default limits for linked PCs</legend>
            <div className="row">
              <label htmlFor="link-gpu" className="dim">GPU memory (GB)</label>
              <input id="link-gpu" className="input mono" style={{ width: 90 }} placeholder="75% free" value={draft.share_gpu_mib != null ? draft.share_gpu_mib / 1024 : ""} onChange={(e) => setDraft({ ...draft, share_gpu_mib: toMib(e.target.value) })} />
              <label htmlFor="link-ram" className="dim">RAM (GB)</label>
              <input id="link-ram" className="input mono" style={{ width: 90 }} placeholder="50% free" value={draft.share_ram_mib != null ? draft.share_ram_mib / 1024 : ""} onChange={(e) => setDraft({ ...draft, share_ram_mib: toMib(e.target.value) })} />
              <label htmlFor="link-threads" className="dim">CPU threads</label>
              <input id="link-threads" className="input mono" style={{ width: 90 }} placeholder="all but 2" value={draft.threads ?? ""} onChange={(e) => setDraft({ ...draft, threads: Number(e.target.value.replace(/\D/g, "")) || null })} />
            </div>
            <span className="help">
              Enforced on this PC: a linked PC can't allocate more than its limit, whatever version it runs. Each linked PC's own limit is set in the table below.
              Empty uses the default: 75% of the free GPU memory and 50% of the available RAM (after what this app keeps free for Windows) when a linked PC connects, and every CPU core but 2. RAM counts only when the CPU is shared, and is approximate (rpc-server's own buffers aren't counted). On a PC with 8 GB of RAM or less the default RAM share is little or nothing; type a RAM limit here or per PC below to lend more.
            </span>
            <span className="help">{THREADS_HINT}</span>
          </fieldset>
          <fieldset className="field link-devs">
            <legend>Devices to share</legend>
            {devs.length === 0 ? (
              <span className="help">All devices. Turn helping on once to choose between them.</span>
            ) : (
              devs.map((d) => (
                <label key={d.name} className="check">
                  <input type="checkbox" checked={draft.devices.length === 0 || draft.devices.includes(d.name)} onChange={() => toggleDev(d.name)} />
                  <span className="mono">{d.name}</span> <span className="faint">{d.description} ({(d.total_mib / 1024).toFixed(1)} GB)</span>
                </label>
              ))
            )}
          </fieldset>
          <div className="wide link-cache">
            <label className="check">
              <input type="checkbox" role="switch" checked={draft.cache} onChange={(e) => setDraft({ ...draft, cache: e.target.checked })} />
              <strong>Speed up reloads</strong>
            </label>
            <span className="help">
              The model weights a linked PC sends live only in this PC's memory while it's connected, and are freed when the model is unloaded or the link closes.
              With this on, a copy also goes to this PC's disk so the next load isn't sent again. It lives in this app's own folder{h.cache.dir ? ` (${h.cache.dir})` : ""}, is deleted when you unlink that PC, and nothing is written when this is off.
            </span>
            {draft.cache && (
              <div className="row">
                <label htmlFor="link-cache-cap" className="dim">Keep at most</label>
                <input id="link-cache-cap" className="input mono" style={{ width: 80 }} inputMode="numeric" value={draft.cache_cap_gb || ""} onChange={(e) => setDraft({ ...draft, cache_cap_gb: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
                <span className="dim">GB (oldest files are removed first)</span>
              </div>
            )}
            <div className="row">
              <button className="btn sm" disabled={busy || h.cache.bytes === 0} onClick={() => act(async () => { const n = await api9.clearCache(); return `Cleared the cache (${gb(n)} freed).`; })}>
                Clear cache ({gb(h.cache.bytes)})
              </button>
              <span className="faint link-small">{gb(h.cache.bytes)} of {gb(h.cache.cap_bytes)} used{h.cache.enabled ? "" : " (the setting is off; this is what's left on disk)"}</span>
            </div>
          </div>
          <label className="check wide">
            <input type="checkbox" checked={draft.compress} onChange={(e) => setDraft({ ...draft, compress: e.target.checked })} />
            Compress link traffic (lz4; may help slow links, not measured yet)
          </label>
          <div className="row wide">
            <button className="btn sm" disabled={busy} onClick={() => act(() => api9.setHelper(h.enabled, draft), h.enabled ? "Restarted with these settings." : "Saved.")}>
              {h.enabled ? "Apply and restart" : "Save settings"}
            </button>
          </div>
        </div>
      </div>

      <div className="link-columns">
        <section aria-labelledby="link-this">
          <h2 id="link-this">Link this PC</h2>
          <div className="panel link-panel">
            {!codeOpen ? (
              <>
                <p className="dim" style={{ marginTop: 0 }}>Show a one-time code here, then type this PC's address and the code on the other PC.</p>
                {code && code.state === "locked" && <div className="notice warn">The last code was locked after 5 wrong tries.</div>}
                {code && code.state === "used" && <div className="notice">The last code was used to link a PC.</div>}
                <button className="btn primary sm" disabled={busy} onClick={() => act(() => api9.startCode())}>Show linking code</button>
              </>
            ) : (
              <>
                <div className="link-code mono" aria-live="polite" aria-label={`Linking code ${code.code.split("").join(" ")}`}>{showCode(code.code)}</div>
                <div className="row"><CopyButton text={code.code} label="code" /><span className="faint">Expires in {mmss(left)}. Works once.{code.failures ? ` ${code.failures} wrong ${code.failures === 1 ? "try" : "tries"}.` : ""}</span></div>
                <dl className="kv" style={{ marginTop: 10 }}>
                  <dt>Same network</dt>
                  <dd className="row">{st.local_ip ?? "?"} <span className="faint">port</span> {port} {st.local_ip && <CopyButton text={`${st.local_ip}:${port}`} label="local address" />}</dd>
                  <dt>Other networks</dt>
                  <dd className="row">{reach?.public_ip ? <>{reach.public_ip} <span className="faint">port</span> {port} <CopyButton text={`${reach.public_ip}:${port}`} label="public address" /></> : <span className="faint">check outside access below</span>}</dd>
                </dl>
                <div className="row" style={{ marginTop: 10 }}><button className="btn sm" disabled={busy} onClick={() => act(() => api9.cancelCode())}>Cancel code</button></div>
              </>
            )}
            <div className="link-reach">
              <button
                className="btn sm"
                disabled={busy}
                title="Asks your router (UPnP) to open the port and for its internet address. No outside service is contacted."
                onClick={() => act(async () => setReach(await api9.checkReach(true)))}
              >
                Check access from other networks
              </button>
              {reach && (
                <div className={`notice ${reach.verdict === "reachable" ? "" : "warn"}`} style={{ marginTop: 8 }}>
                  {reach.message}
                  {reach.steps.length > 0 && <ol className="link-steps">{reach.steps.map((s) => <li key={s}>{s}</li>)}</ol>}
                </div>
              )}
              <p className="faint link-small">Windows may ask whether Local AI Studio can use the network: allow it on private networks.</p>
            </div>
          </div>
        </section>

        <section aria-labelledby="link-other">
          <h2 id="link-other">Link to another PC</h2>
          <form className="panel link-panel" onSubmit={(e) => { e.preventDefault(); act(() => api9.pair(form.address, Number(form.port) || null, form.code), "Linked. Both PCs now trust each other and reconnect without a code."); }}>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="link-addr">Its IP address</label>
                <input id="link-addr" className="input mono" placeholder="192.168.1.20" value={form.address} onChange={(e) => setForm({ ...form, address: e.target.value })} required />
              </div>
              <div className="field">
                <label htmlFor="link-pport">Port</label>
                <input id="link-pport" className="input mono" placeholder="47070 (or 50070)" value={form.port} onChange={(e) => setForm({ ...form, port: e.target.value.replace(/\D/g, "") })} />
              </div>
              <div className="field wide">
                <label htmlFor="link-code">Linking code shown on that PC</label>
                <input id="link-code" className="input mono link-code-input" inputMode="numeric" autoComplete="off" placeholder="000 000" value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} required />
              </div>
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn primary sm" type="submit" disabled={busy || !form.address.trim() || form.code.replace(/\D/g, "").length !== 6}>Link</button>
            </div>
          </form>
        </section>
      </div>

      <h2>Linked PCs</h2>
      {st.peers.length === 0 ? (
        <div className="empty"><strong>No linked PCs yet</strong>Show a code on one PC and type it on the other.</div>
      ) : (
        <table className="table">
          <thead>
            <tr><th>PC</th><th>Status</th><th className="num">Round trip</th><th className="num">Upload</th><th>Split model layers</th><th>Sharing</th><th>Send whole jobs</th><th /></tr>
          </thead>
          <tbody>
            {st.peers.map((p) => <PeerRow key={p.id} p={p} busy={busy} act={act} />)}
          </tbody>
        </table>
      )}
    </div>
  );
}

function PeerRow({ p, busy, act }: { p: PeerView; busy: boolean; act: (fn: () => Promise<unknown>, ok?: string) => Promise<void> }) {
  const [confirming, setConfirming] = useState(false);
  const status = p.online ? (p.helping ? "Online, helping" : "Online, not helping") : p.connected_here ? "Connected to this PC" : "Offline";
  return (
    <tr>
      <td>
        <div>{p.name} {!p.same_network && <span className="tag">other network</span>}</div>
        <div className="mono faint" title="Key fingerprint (compare on the other PC)">{p.fingerprint}</div>
        <div className="mono faint">
          {p.dialable ? p.address : `connected from ${p.seen_from ?? "?"} (linked from there)`}
          {p.runtime_tag ? ` · ${p.runtime_tag}` : ""}
        </div>
        {p.cache_bytes > 0 && <div className="faint link-small">{gb(p.cache_bytes)} of its weights cached here (removed when you unlink it)</div>}
      </td>
      <td>
        <span className={p.online ? "ok-text" : "faint"}>{status}</span>
        {p.last_seen_at && <div className="faint">seen {when(p.last_seen_at)}</div>}
      </td>
      <td className="num">{ms(p.rtt_ms)}</td>
      <td className="num">{p.bandwidth_mbps == null ? "-" : `${p.bandwidth_mbps.toFixed(0)} Mbit/s`}</td>
      <td>
        <label className="check">
          <input type="checkbox" role="switch" aria-label={`Split model layers with ${p.name}`} checked={p.split_layers} disabled={busy} onChange={(e) => { const on = e.target.checked; act(() => api9.setSplit(p.id, on)); }} />
          {p.split_layers ? "On" : "Off"}
        </label>
        {p.added_ms_per_token != null && <div className="faint link-small">adds about {ms(p.added_ms_per_token)} per generated token</div>}
        {!p.same_network && p.split_layers && <div className="faint link-small">Over the internet every token waits for a round trip.</div>}
      </td>
      <td><SharingCell p={p} busy={busy} act={act} /></td>
      <td>
        <label className="check" title="A later version sends whole chats and subagent jobs to this PC. Saved now, not used yet.">
          <input type="checkbox" role="switch" aria-label={`Send whole jobs to ${p.name} (coming later)`} checked={p.send_jobs} disabled={busy} onChange={(e) => { const on = e.target.checked; act(() => api9.setSendJobs(p.id, on)); }} />
          <span className="faint">later</span>
        </label>
      </td>
      <td>
        <div className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
          {p.dialable && <button className="btn sm" disabled={busy} onClick={() => act(() => api9.connect(p.id), `Connected to ${p.name}.`)}>{p.online ? "Measure again" : "Connect"}</button>}
          {confirming ? (
            <>
              <span className="faint link-small" id={`unlink-${p.id}`}>Both PCs forget each other's key{p.cache_bytes > 0 ? " and the weights it cached here are deleted" : ""}; linking again needs a new code.</span>
              <button className="btn sm" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
              <button
                className="btn sm danger"
                aria-describedby={`unlink-${p.id}`}
                disabled={busy}
                onClick={() => { setConfirming(false); act(async () => (await api9.unlink(p.id)) ? `Unlinked ${p.name} on both PCs.` : `Unlinked here. ${p.name} wasn't reachable, so it will be refused the next time it connects.`); }}
              >
                Confirm unlink
              </button>
            </>
          ) : (
            <button className="btn sm danger" disabled={busy} onClick={() => setConfirming(true)}>Unlink</button>
          )}
        </div>
      </td>
    </tr>
  );
}

/** What this PC lends a linked PC (enforced here), what it uses now, and
 *  what that PC lets this one use. */
/** Below this a linked PC's devices hold no layers (their working buffer takes it all). */
const LEND_USEFUL_MIB = 1024;

function SharingCell({ p, busy, act }: { p: PeerView; busy: boolean; act: (fn: () => Promise<unknown>, ok?: string) => Promise<void> }) {
  const [edit, setEdit] = useState<{ gpu: string; ram: string; threads: string; paused: boolean } | null>(null);
  const l = p.lends;
  const shares = [l.shares_gpu ? `${gib(l.limits.gpu_mib)} GPU memory` : null, l.shares_ram ? `${gib(l.limits.ram_mib)} RAM` : null, `${l.limits.threads} threads`].filter(Boolean).join(", ");
  const defaults = [l.shares_gpu && l.stored.gpu_mib == null ? "75% of free GPU memory" : null, l.shares_ram && l.stored.ram_mib == null ? "50% of available RAM" : null].filter(Boolean);
  // The defaults lend little on small PCs (50% of available RAM after the at least 4 GB this
  // app keeps free for Windows is about nothing with 8 GB of RAM): too little for a device's
  // working buffer, so the linked PC leaves this one out of its loads.
  const lentMib = (l.shares_gpu ? l.limits.gpu_mib : 0) + (l.shares_ram ? l.limits.ram_mib : 0);
  const tooLittle = !l.stored.paused && defaults.length > 0 && lentMib < LEND_USEFUL_MIB;
  const save = () => {
    if (!edit) return;
    const share: PeerShare = { gpu_mib: toMib(edit.gpu), ram_mib: l.shares_ram ? toMib(edit.ram) : l.stored.ram_mib, threads: Number(edit.threads.replace(/\D/g, "")) || null, paused: edit.paused };
    setEdit(null);
    act(() => api9.setPeerShare(p.id, share), `Saved what this PC shares with ${p.name}.`);
  };
  const o = p.offers;
  return (
    <div className="link-small">
      {l.stored.paused ? (
        <div className="warn-text">Sharing with {p.name} is paused (new sessions are refused).</div>
      ) : (
        <div>
          This PC shares up to {shares} with {p.name}.{defaults.length > 0 && <span className="faint"> (default: {defaults.join(", ")})</span>}
        </div>
      )}
      {tooLittle && (
        <div className="warn-text">
          With the default this PC lends {p.name} only {gib(lentMib)}: too little to hold any layers (each device needs about 0.5-1 GB for its working buffer), so {p.name} leaves this PC out of its loads.
          {l.shares_ram ? " The default RAM share is 50% of the available RAM after what this app keeps free for Windows (at least 4 GB), which is little or nothing on a PC with 8 GB of RAM or less." : " The default is 75% of the free GPU memory."} Click Edit and set a {l.shares_ram ? "RAM" : "GPU memory"} limit (for example 3 GB) to lend more.
        </div>
      )}
      {(l.used_gpu_mib > 0 || l.used_ram_mib > 0) && (
        <div>
          Lending now: {[l.shares_gpu ? `${gib(l.used_gpu_mib)} of ${gib(l.limits.gpu_mib)} GPU memory` : null, l.shares_ram ? `${gib(l.used_ram_mib)} of ${gib(l.limits.ram_mib)} RAM` : null].filter(Boolean).join(", ")}.
        </div>
      )}
      {l.refused > 0 && (
        <div className="warn-text">Refused {l.refused === 1 ? "1 memory request" : `${l.refused} memory requests`} that would have gone over the limit ({gb(l.refused_bytes)} asked in all){l.last_refused_secs_ago != null ? `, last ${ago(l.last_refused_secs_ago)}` : ""}.</div>
      )}
      {p.online && (o && p.offers_enforced ? (
        <div className="faint">{p.name} lets this PC use up to {gib(o.gpu_mib)} GPU memory, {gib(o.ram_mib)} RAM, {o.threads} threads{o.paused ? " (paused there)" : ""}.</div>
      ) : p.helping ? (
        <div className="faint">{p.name} runs an older version: its limit isn't enforced there.</div>
      ) : null)}
      {p.refusal && <div className="warn-text">{p.refusal}</div>}
      {edit ? (
        <div className="link-share-edit">
          <div className="row">
            <label className="dim" htmlFor={`sg-${p.id}`}>GPU memory (GB)</label>
            <input id={`sg-${p.id}`} className="input mono" style={{ width: 70 }} placeholder="default" value={edit.gpu} disabled={!l.shares_gpu} onChange={(e) => setEdit({ ...edit, gpu: e.target.value })} />
            {!l.shares_gpu && <span className="faint">no GPU shared</span>}
          </div>
          <div className="row">
            <label className="dim" htmlFor={`sr-${p.id}`}>RAM (GB)</label>
            <input id={`sr-${p.id}`} className="input mono" style={{ width: 70 }} placeholder={l.shares_ram ? "default" : ""} value={l.shares_ram ? edit.ram : ""} disabled={!l.shares_ram} onChange={(e) => setEdit({ ...edit, ram: e.target.value })} />
            {!l.shares_ram && <span className="faint">CPU not shared</span>}
          </div>
          <div className="row">
            <label className="dim" htmlFor={`st-${p.id}`}>CPU threads</label>
            <input id={`st-${p.id}`} className="input mono" style={{ width: 70 }} placeholder="default" value={edit.threads} onChange={(e) => setEdit({ ...edit, threads: e.target.value })} />
          </div>
          <span className="help">{THREADS_HINT} A thread change applies from {p.name}'s next session.</span>
          <label className="check">
            <input type="checkbox" checked={edit.paused} onChange={(e) => setEdit({ ...edit, paused: e.target.checked })} />
            Pause sharing with {p.name}
          </label>
          <span className="help">Pausing refuses new sessions; a model {p.name} has loaded keeps working. Memory already lent is never taken back; a lower limit applies to its next allocation.</span>
          <div className="row">
            <button className="btn sm" disabled={busy} onClick={() => setEdit(null)}>Cancel</button>
            <button className="btn primary sm" disabled={busy} onClick={save}>Save</button>
          </div>
        </div>
      ) : (
        <button
          className="btn ghost sm"
          disabled={busy}
          aria-label={`Edit what this PC shares with ${p.name}`}
          onClick={() => setEdit({ gpu: l.stored.gpu_mib != null ? String(l.stored.gpu_mib / 1024) : "", ram: l.stored.ram_mib != null ? String(l.stored.ram_mib / 1024) : "", threads: l.stored.threads != null ? String(l.stored.threads) : "", paused: l.stored.paused })}
        >
          Edit
        </button>
      )}
    </div>
  );
}

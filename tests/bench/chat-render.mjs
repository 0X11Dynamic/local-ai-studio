// PERF-3 harness: main-thread cost per streamed update on the Chat page.
// No model needed. Launches the app with an isolated profile, creates a conversation over
// IPC, opens it in the Chat page, then replays a synthetic Markdown reply as `chat://event`
// deltas at 25 events/s (the 40 ms batch rate of the backend) and reads CDP
// Performance.getMetrics (TaskDuration, ScriptDuration, LayoutDuration, RecalcStyleDuration).
//
// Method: per run, one fake assistant message is announced with a `segment_start` event
// (Chat.tsx adds it to the open chat). For each target length (2k, 8k, 32k chars) the live text
// is seeded to that length in one delta, left to settle, then a 1 s window of 25 small deltas
// (~8 chars each, like ~40 tok/s) is replayed; metric deltas / 25 = ms per update at that
// length. A baseline window (same IPC traffic, event name nobody listens to) is subtracted.
//
//   node tests/bench/chat-render.mjs --exe <local-ai-studio.exe> [--runs 5] [--lengths 2000,8000,32000]
//        [--chunk 8] [--window 25] [--direct] [--out results.jsonl] [--cdp-port 19335]
//   --direct dispatches into the page's registered Tauri listener instead of `plugin:event|emit`
//   (use if emit is refused by the capability set; best effort, see README-perf.md).
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const EXE = arg("exe", process.env.LAS_E2E_EXE);
const RUNS = Number(arg("runs", 5));
const LENGTHS = arg("lengths", "2000,8000,32000").split(",").map(Number);
const CHUNK = Number(arg("chunk", 8));
const WIN = Number(arg("window", 25));
const DIRECT = process.argv.includes("--direct");
const OUT = arg("out");
const PORT = Number(arg("cdp-port", 19335));
if (!EXE) { console.error("usage: --exe <exe> [--runs 5] [--lengths 2000,8000,32000] [--out f.jsonl]"); process.exit(1); }
for (const p of ["llama-server.exe", "local-ai-studio.exe"]) {
  if (countProcesses(p) > 0) { console.error(`${p} already running; aborting`); process.exit(2); }
}

const r2 = (x) => Math.round(x * 100) / 100;
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };

// Synthetic reply: headings, lists, a Rust code fence and a table, repeated to >= 40k chars.
function makeMarkdown(min) {
  const code = "```rust\nfn parse(input: &str) -> Result<Vec<u32>, String> {\n    input.split(',')\n        .map(|s| s.trim().parse::<u32>().map_err(|e| e.to_string()))\n        .collect()\n}\n\nfn main() {\n    let v = parse(\"1, 2, 3\").unwrap();\n    println!(\"{:?}\", v);\n}\n```\n\n";
  const table = "| Name | Value | Notes |\n|------|-------|-------|\n| alpha | 1 | first row with `inline code` |\n| beta | 2 | **bold** and *italic* text |\n| gamma | 3 | a [link](https://example.com) |\n\n";
  const list = "- First point with some explanatory text that wraps over a line or two.\n- Second point with `code` and **emphasis** inside it.\n  - A nested item that adds depth to the list.\n- Third point.\n\n1. Step one of the procedure.\n2. Step two of the procedure.\n3. Step three of the procedure.\n\n";
  let s = "";
  for (let i = 1; s.length < min; i++) {
    s += `## Section ${i}\n\nThis paragraph describes section ${i} in some detail, with a sentence about **performance**, another about *streaming*, and a mention of \`render()\` so inline code is parsed too. It is long enough to wrap across several lines in the chat bubble.\n\n### Details ${i}\n\n` + list + code + table;
  }
  return s;
}
const TEXT = makeMarkdown(Math.max(...LENGTHS) + 4000);

const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-render-"));
const prof = makeProfile(run);
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const rows = [];

// Page side: emitter + observers. Installed once per page load.
const PAGE_JS = `(() => {
  if (window.__bench) return;
  const B = window.__bench = { longtasks: [], lag: [], direct: ${DIRECT} };
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) B.longtasks.push(e.duration); }).observe({ entryTypes: ["longtask"] }); } catch {}
  B.send = (event, payload) => {
    if (B.direct) {
      const h = window.__benchHandlers?.[event];
      if (!h) throw new Error("no captured listener for " + event + " (--direct needs the page reload done by the harness)");
      window["_" + h]({ event, id: 0, payload });
      return Promise.resolve();
    }
    return window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event, payload });
  };
  // Replay one window: n events at 25/s on an absolute schedule; returns lag + longtask data.
  B.window = async (event, mk, n) => {
    B.longtasks.length = 0; B.lag.length = 0;
    let last = performance.now();
    const lagTimer = setInterval(() => { const t = performance.now(); B.lag.push(t - last - 10); last = t; }, 10);
    const t0 = performance.now(), pend = [];
    for (let i = 0; i < n; i++) {
      const due = t0 + i * 40, now = performance.now();
      if (due > now) await new Promise((r) => setTimeout(r, due - now));
      pend.push(B.send(event, mk(i)));
    }
    await Promise.all(pend);
    await new Promise((r) => setTimeout(r, 150));
    clearInterval(lagTimer);
    const lag = B.lag.slice().sort((a, b) => a - b);
    return { wall_ms: performance.now() - t0, longtasks: B.longtasks.slice(), lag_p50: lag[Math.floor(lag.length / 2)] ?? 0, lag_max: lag.at(-1) ?? 0 };
  };
})()`;

const metrics = async () => {
  const r = await cdp.send("Performance.getMetrics");
  return Object.fromEntries(r.result.metrics.map((m) => [m.name, m.value]));
};
const diff = (a, b) => ({ task: b.TaskDuration - a.TaskDuration, script: b.ScriptDuration - a.ScriptDuration, layout: b.LayoutDuration - a.LayoutDuration, style: b.RecalcStyleDuration - a.RecalcStyleDuration });

async function measure(event, mkExpr, n) {
  const m0 = await metrics();
  const w = await cdp.eval(`window.__bench.window(${JSON.stringify(event)}, ${mkExpr}, ${n})`);
  const m1 = await metrics();
  return { d: diff(m0, m1), w };
}

try {
  await cdp.connect();
  await cdp.send("Performance.enable");
  if (DIRECT) {
    // Capture the handler id the app registers for chat://event, then reload so it registers again under our hook.
    await cdp.send("Page.enable");
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: `(() => {
      window.__benchHandlers = {};
      const iv = setInterval(() => {
        const I = window.__TAURI_INTERNALS__;
        if (!I || I.__benchWrapped) return;
        const orig = I.invoke.bind(I);
        I.invoke = (cmd, args, opts) => { if (cmd === "plugin:event|listen" && args?.event) window.__benchHandlers[args.event] = args.handler; return orig(cmd, args, opts); };
        I.__benchWrapped = true; clearInterval(iv);
      }, 0);
    })()` });
    await cdp.send("Page.reload");
    await sleep(1500);
    await cdp.waitFor("!!document.querySelector('.sidebar')", 60000);
  }
  await cdp.eval(PAGE_JS);

  const conv = await cdp.invoke("create_conversation");
  await cdp.nav("Chat");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length > 0", 15000, "chat list");
  await cdp.eval(`[...document.querySelectorAll('button.chat-item')].find(b => b.innerText.includes('New chat'))?.click()`);
  await cdp.waitFor("!!document.querySelector('button.chat-item[aria-current=\"true\"]')", 10000, "chat opened");
  await sleep(500);

  const mkMsg = (id, content, stats) => ({ id, conversation_id: conv.id, role: "assistant", content, reasoning: null, stats: stats ?? null, created_at: new Date().toISOString(), model_name: "synthetic", tool_calls: null, tool_call_id: null, attachments: "[]" });
  const send = (event, payload) => cdp.eval(`window.__bench.send(${JSON.stringify(event)}, ${JSON.stringify(payload)}).then(() => true)`);

  for (let run = 1; run <= RUNS; run++) {
    const mid = `bench-msg-${run}-${Date.now()}`;
    const row = { bench: "chat-render", run, chunk: CHUNK, window: WIN, direct: DIRECT, points: [] };
    // Baseline: same IPC traffic and payload size to an event nobody listens to.
    const baseExpr = `(i) => ({ type: "delta", conversation_id: ${JSON.stringify(conv.id)}, message_id: ${JSON.stringify(mid)}, content: ${JSON.stringify(TEXT.slice(0, CHUNK))}, reasoning: "" })`;
    const base = await measure("bench://noop", baseExpr, WIN);
    row.baseline_ms_per_event = { task: r2(base.d.task * 1000 / WIN), script: r2(base.d.script * 1000 / WIN) };

    await send("chat://event", { type: "segment_start", conversation_id: conv.id, message: mkMsg(mid, "", null) });
    await sleep(400); // let Chat.tsx record the id (otherwise a delta triggers a reload that drops the fake message)
    let pos = 0;
    for (const T of LENGTHS) {
      await send("chat://event", { type: "delta", conversation_id: conv.id, message_id: mid, content: TEXT.slice(pos, T), reasoning: "" });
      pos = T;
      await sleep(600); // settle: the seed render is not measured
      const start = pos;
      const mkExpr = `(i) => ({ type: "delta", conversation_id: ${JSON.stringify(conv.id)}, message_id: ${JSON.stringify(mid)}, content: ${JSON.stringify(TEXT.slice(start, start + WIN * CHUNK))}.slice(i * ${CHUNK}, (i + 1) * ${CHUNK}), reasoning: "" })`;
      const got = await measure("chat://event", mkExpr, WIN);
      pos = start + WIN * CHUNK;
      const rendered = await cdp.eval(`document.body.innerText.includes('Section 1')`);
      if (!rendered) throw new Error("synthetic reply is not on screen: the Chat page did not show the fake message (conversation not open, or an unknown-message reload dropped it)");
      const g = (x) => r2((x * 1000) / WIN);
      const bt = base.d.task * 1000 / WIN, bs = base.d.script * 1000 / WIN;
      const pt = {
        chars: T, task_ms: g(got.d.task), script_ms: g(got.d.script), layout_ms: g(got.d.layout), style_ms: g(got.d.style),
        net_task_ms: r2(g(got.d.task) - bt), net_script_ms: r2(g(got.d.script) - bs),
        longtasks_gt50: got.w.longtasks.filter((d) => d > 50).length, longtask_max_ms: r2(Math.max(0, ...got.w.longtasks)),
        loop_lag_p50_ms: r2(got.w.lag_p50), loop_lag_max_ms: r2(got.w.lag_max), window_wall_ms: Math.round(got.w.wall_ms),
      };
      row.points.push(pt);
    }
    await send("chat://event", { type: "done", conversation_id: conv.id, message_id: mid, message: mkMsg(mid, TEXT.slice(0, pos), JSON.stringify({ stop_reason: "stop", total_ms: 1 })) });
    await sleep(500);
    console.log(JSON.stringify(row));
    if (OUT) fs.appendFileSync(OUT, JSON.stringify(row) + "\n");
    rows.push(row);
  }
  await cdp.invoke("delete_conversation", { id: conv.id }).catch(() => {});
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1500);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}

console.log(`\nPer-update main-thread cost (${rows.length} runs; ${WIN} events/window at 25/s, ${CHUNK} chars each; baseline IPC overhead subtracted from net_*)`);
console.log("chars  | net task ms med (min-max) | net script ms | layout ms | style ms | long tasks >50ms (sum) | max long task ms | loop lag max ms");
for (let i = 0; i < LENGTHS.length; i++) {
  const p = rows.map((r) => r.points[i]);
  const k = (f) => p.map((x) => x[f]);
  const mm = (f) => `${r2(med(k(f)))} (${r2(Math.min(...k(f)))}-${r2(Math.max(...k(f)))})`;
  console.log(`${String(LENGTHS[i]).padEnd(6)} | ${mm("net_task_ms")} | ${r2(med(k("net_script_ms")))} | ${r2(med(k("layout_ms")))} | ${r2(med(k("style_ms")))} | ${k("longtasks_gt50").reduce((a, b) => a + b, 0)} | ${r2(Math.max(...k("longtask_max_ms")))} | ${r2(Math.max(...k("loop_lag_max_ms")))}`);
}
console.log("Busy fraction = net task ms / 40 ms. Above ~100% the UI thread cannot keep up with 25 updates/s.");

// PERF-5 / PERF-6 harness: llama-server RAM growth across chat switches, and chat_context latency on a long agent chat.
//   node tests/bench/hostcache-context.mjs --exe <exe> --model <gguf> [--switches 5] [--turns 30] [--ctx-runs 5] [--out f.jsonl]
// Isolated profile. 5 agent chats, each with a distinct ~3k-token first prompt; cycles A..E twice so the host
// prompt cache has to save/restore slots. backend_memory_bytes (get_resources) is read after every step.
// Then a long agent chat (--turns user/assistant turns of ~300 tokens each) is built and chat_context is timed.
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const EXE = arg("exe", process.env.LAS_E2E_EXE), MODEL = arg("model");
const SW = Number(arg("switches", 5)), TURNS = Number(arg("turns", 30)), CTXRUNS = Number(arg("ctx-runs", 5));
const OUT = arg("out"), PORT = Number(arg("cdp-port", 19338));
if (!EXE || !MODEL) { console.error("usage: --exe --model"); process.exit(1); }
for (const p of ["llama-server.exe", "local-ai-studio.exe"]) if (countProcesses(p) > 0) { console.error(p + " running; abort"); process.exit(2); }
const mb = (b) => (b == null ? null : Math.round(b / 1048576));
const norm = (p) => p.split("\\").join("/").toLowerCase();
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const emit = (r) => { console.log(JSON.stringify(r)); if (OUT) fs.appendFileSync(OUT, JSON.stringify(r) + "\n"); };
const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-hc-"));
const prof = makeProfile(run);
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const inv = (c, a) => cdp.invoke(c, a);
const words = (seed, n) => { let s = ""; for (let i = 0; i < n; i++) s += `w${(seed * 7919 + i * 104729) % 9973} `; return s; };
const rss = async () => mb((await inv("get_resources")).backend_memory_bytes);
async function newChat() {
  const c = await inv("create_conversation");
  const v = await inv("get_conversation", { id: c.id });
  await inv("update_chat_settings", { id: c.id, settings: { ...v.settings_parsed, agent_mode: true, mode: "agent", builtin_tools: true, disabled_servers: [] } });
  return c.id;
}
async function say(id, text) {
  await inv("chat_send", { id, content: text });
  for (;;) { const c = await inv("get_conversation", { id }); if (!c.generating) return c; await sleep(300); }
}
try {
  await cdp.connect();
  await inv("import_models", { path: MODEL });
  const mid = (await inv("list_models")).find((m) => norm(m.path) === norm(MODEL))?.id;
  const d = await inv("get_load_defaults", { id: mid });
  await inv("load_model", { modelId: mid, config: { ...d.config, share_with_cpu: true }, runtimeId: prof.cuda });
  const st = await inv("engine_status");
  const base = await rss();
  emit({ bench: "hostcache", step: "loaded", n_ctx: st.config?.n_ctx, backend_mb: base });
  const chats = [];
  for (let i = 0; i < SW; i++) chats.push(await newChat());
  const series = [base];
  for (let round = 0; round < 2; round++) {
    for (let i = 0; i < SW; i++) {
      const t0 = Date.now();
      await say(chats[i], round === 0 ? `Document ${i}: ${words(i + 1, 1800)}\nReply with the single word OK.` : `Reply with the single word AGAIN.`);
      const m = await rss();
      series.push(m);
      emit({ bench: "hostcache", round, chat: i, backend_mb: m, growth_vs_loaded_mb: m - base, wall_s: Math.round((Date.now() - t0) / 100) / 10 });
    }
  }
  emit({ bench: "hostcache", summary: true, switches: SW, start_mb: base, end_mb: series.at(-1), peak_mb: Math.max(...series), growth_mb: series.at(-1) - base });
  for (const c of chats) await inv("delete_conversation", { id: c }).catch(() => {});
  // PERF-6: long agent chat
  const long = await newChat();
  for (let t = 0; t < TURNS; t++) await say(long, `Note ${t}: ${words(100 + t, 220)}\nReply with the single word OK.`);
  const conv = await inv("get_conversation", { id: long });
  await inv("chat_context", { id: long, draft: "" }); // warm
  const times = [];
  let ctx;
  for (let i = 0; i < CTXRUNS; i++) { const t0 = performance.now(); ctx = await inv("chat_context", { id: long, draft: "" }); times.push(Math.round(performance.now() - t0)); }
  emit({ bench: "chat_context", messages: conv.messages.length, history_tokens: ctx.history, total_tokens: ctx.total, messages_counted: ctx.messages_counted, ms: times, median_ms: med(times), min_ms: Math.min(...times), max_ms: Math.max(...times) });
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1500);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}

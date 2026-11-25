// PERF-1 / PERF-2 harness: Agent-chat prefill cost and subagent prefix-cache behaviour.
// Launches the release exe with an isolated profile (never the real data), imports one
// model, loads it with its saved defaults (n_ctx Auto, same as load-ab.mjs), then:
//   prefill : new Agent chat, send "hi", read prompt_tokens / cached_tokens / TTFT from the
//             stored message stats. The model is unloaded and reloaded between runs so the
//             llama-server host prompt cache is empty (cold prompt every run).
//   subagent: Agent chat with subagents on (max 2); two children each run 3 shell steps;
//             per-step stats from agent_steps plus backend memory sampled over time.
//
//   node tests/bench/agent-prefill.mjs --exe <local-ai-studio.exe> --model <file.gguf>
//        [--mode prefill|subagent|both] [--runs 3] [--mcp-config <mcp.json>] [--builtins-only]
//        [--runtime auto|<id>] [--out <results.jsonl>] [--cdp-port 19334] [--timeout-min 30]
//   (exe may come from env LAS_E2E_EXE). Models live under LAS_MODELS_DIR
//   (default %USERPROFILE%\LocalAIStudio\models). Needs the GPU to itself.
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const has = (n) => process.argv.includes("--" + n);
const EXE = arg("exe", process.env.LAS_E2E_EXE);
const MODEL = arg("model");
const MODE = arg("mode", "both");
const RUNS = Number(arg("runs", 3));
const MCP = arg("mcp-config");
const BUILTINS_ONLY = has("builtins-only");
const RUNTIME = arg("runtime", "auto");
const OUT = arg("out");
const PORT = Number(arg("cdp-port", 19334));
const TIMEOUT = Number(arg("timeout-min", 30)) * 60000;
if (!EXE || !MODEL || !["prefill", "subagent", "both"].includes(MODE)) {
  console.error("usage: --exe <exe> --model <gguf path> [--mode prefill|subagent|both] [--runs N] [--mcp-config f.json] [--builtins-only] [--out f.jsonl]");
  process.exit(1);
}
if (!fs.existsSync(MODEL)) { console.error("model file not found: " + MODEL); process.exit(1); }
if (MCP && !fs.existsSync(MCP)) { console.error("mcp config not found: " + MCP); process.exit(1); }

for (const p of ["llama-server.exe", "local-ai-studio.exe"]) {
  if (countProcesses(p) > 0) { console.error(`${p} already running; aborting`); process.exit(2); }
}

const r1 = (x) => (x == null ? null : Math.round(x * 10) / 10);
const norm = (p) => p.split("\\").join("/").toLowerCase();
const parse = (s) => { try { return JSON.parse(s); } catch { return {}; } };
const med = (a) => { const s = a.filter((x) => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const range = (a) => { const s = a.filter((x) => x != null); return s.length ? `${r1(Math.min(...s))}-${r1(Math.max(...s))}` : "-"; };
const emit = (row) => { console.log(JSON.stringify(row)); if (OUT) fs.appendFileSync(OUT, JSON.stringify(row) + "\n"); };

const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-prefill-"));
const work = path.join(run, "work");
fs.mkdirSync(work, { recursive: true });
const prof = makeProfile(run);
const runtimeId = prof.cuda && RUNTIME === "auto" ? prof.cuda : RUNTIME;
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const inv = (c, a) => cdp.invoke(c, a);
const prefillRows = [];
const subRows = [];

async function loadModel(id) {
  const d = await inv("get_load_defaults", { id });
  const t0 = Date.now();
  await inv("load_model", { modelId: id, config: { ...d.config, share_with_cpu: true }, runtimeId });
  const s = await inv("engine_status");
  return { n_ctx: s.config?.n_ctx, load_wall_s: r1((Date.now() - t0) / 1000), runtime: s.runtime_id };
}
async function reload(id) {
  await inv("unload_model");
  await sleep(4000);
  return loadModel(id);
}
async function newAgentChat(extra = {}) {
  const c = await inv("create_conversation");
  const v = await inv("get_conversation", { id: c.id });
  const settings = { ...v.settings_parsed, agent_mode: true, mode: "agent", builtin_tools: true, ...extra };
  if (BUILTINS_ONLY) {
    const costs = await inv("chat_tool_costs", { id: c.id });
    settings.disabled_servers = costs.entries.filter((e) => e.kind === "mcp").map((e) => e.id);
  }
  await inv("update_chat_settings", { id: c.id, settings });
  return c.id;
}
// Answer shell/file approvals so the agent is never stuck waiting for a click.
function autoApprove() {
  let stop = false;
  (async () => {
    const seen = new Set();
    while (!stop) {
      try {
        for (const p of await inv("pending_tool_approvals")) {
          if (!seen.has(p.id)) { seen.add(p.id); await inv("answer_tool_approval", { id: p.id, answer: "allow_session" }); }
        }
      } catch {}
      await sleep(500);
    }
  })();
  return { stop: () => { stop = true; } };
}
async function waitIdle(conv, sample) {
  const deadline = Date.now() + TIMEOUT;
  for (;;) {
    const c = await inv("get_conversation", { id: conv });
    if (sample) await sample();
    if (!c.generating) return c;
    if (Date.now() > deadline) { await inv("chat_stop", { id: conv }).catch(() => {}); throw new Error("timed out waiting for the reply"); }
    await sleep(1000);
  }
}

async function prefillRun(i, modelId, loadInfo) {
  const conv = await newAgentChat();
  const costs = await inv("chat_tool_costs", { id: conv });
  const ctx = await inv("chat_context", { id: conv, draft: "" });
  const row = {
    bench: "prefill", run: i + 1, builtins_only: BUILTINS_ONLY, mcp_config: !!MCP, ...loadInfo,
    tool_count: ctx.tool_count, ctx_tools_tokens: ctx.tools, ctx_system_tokens: ctx.system, ctx_total_tokens: ctx.total, ctx_template_tokens: ctx.template,
    tool_costs: costs.entries.map((e) => ({ name: e.name, kind: e.kind, state: e.state, in_chat: e.in_chat, tools: e.tool_count, tokens: e.tokens })),
    tool_costs_sum_tokens: costs.entries.filter((e) => e.in_chat).reduce((a, e) => a + e.tokens, 0),
  };
  const ap = autoApprove();
  try {
    await inv("chat_send", { id: conv, content: "hi" });
    const c = await waitIdle(conv);
    const a = c.messages.find((m) => m.role === "assistant" && m.stats);
    const st = parse(a?.stats);
    Object.assign(row, {
      prompt_tokens: st.prompt_tokens ?? null, cached_tokens: st.cached_tokens ?? null, time_to_first_token_ms: r1(st.time_to_first_token_ms),
      prompt_tokens_per_sec: r1(st.prompt_tokens_per_sec), total_ms: st.total_ms ?? null, stop_reason: st.stop_reason, error: st.error,
    });
  } finally { ap.stop(); }
  await inv("delete_conversation", { id: conv }).catch(() => {});
  emit(row);
  prefillRows.push(row);
}

const SUB_PROMPT = `Use the spawn_subagents tool exactly once with two subagents, labels "a" and "b".
Subagent "a": run these three shell commands with shell_execute, one command per step, in order: echo A1 ; echo A2 ; echo A3 . Then report the three outputs.
Subagent "b": run these three shell commands with shell_execute, one command per step, in order: echo B1 ; echo B2 ; echo B3 . Then report the three outputs.
After both finish, reply with the six outputs. Do not run the commands yourself.`;

async function subagentRun(i, modelId, loadInfo) {
  const conv = await newAgentChat({ subagents: true, max_subagents: 2 });
  const ctx = await inv("chat_context", { id: conv, draft: "" });
  const mem = [];
  const t0 = Date.now();
  let lastSample = 0;
  const sample = async () => {
    if (Date.now() - lastSample < 2000) return;
    lastSample = Date.now();
    const r = await inv("get_resources");
    mem.push({ t_s: r1((Date.now() - t0) / 1000), backend_mb: r.backend_memory_bytes == null ? null : Math.round(r.backend_memory_bytes / 1048576), ram_used_mb: Math.round(r.ram_used_bytes / 1048576) });
  };
  await sample();
  const ap = autoApprove();
  let c;
  try {
    await inv("chat_send", { id: conv, content: SUB_PROMPT });
    c = await waitIdle(conv, sample);
  } finally { ap.stop(); }
  await sample();
  const sub = await inv("chat_subagents", { id: conv });
  const steps = [];
  if (sub.run_id) {
    const t = await inv("get_task", { id: sub.run_id });
    const label = Object.fromEntries(t.subagents.map((s) => [s.id, s.label]));
    for (const s of t.steps.filter((x) => x.kind === "model")) {
      const d = parse(s.data);
      const st = d.stats ?? {};
      steps.push({
        seq: s.seq, agent: s.parent_agent_id ? label[s.parent_agent_id] ?? s.parent_agent_id : "root",
        prompt_tokens: st.prompt_tokens ?? null, cached_tokens: st.cached_tokens ?? null, time_to_first_token_ms: r1(st.time_to_first_token_ms),
        tool_calls: (d.tool_calls ?? []).map((x) => x.function?.name),
      });
    }
  }
  // Steps of one agent after its first: how many got no cache hit at all.
  const seen = new Set();
  let followups = 0, followupsNoCache = 0;
  for (const s of steps.filter((x) => x.agent !== "root")) {
    if (seen.has(s.agent)) { followups++; if (!s.cached_tokens) followupsNoCache++; }
    seen.add(s.agent);
  }
  const row = {
    bench: "subagent", run: i + 1, builtins_only: BUILTINS_ONLY, mcp_config: !!MCP, ...loadInfo,
    tool_count: ctx.tool_count, ctx_tools_tokens: ctx.tools, subagents: sub.subagents.map((s) => ({ label: s.label, state: s.state })),
    steps, child_followup_steps: followups, child_followup_steps_cached_zero: followupsNoCache,
    backend_mb_start: mem[0]?.backend_mb ?? null, backend_mb_peak: Math.max(...mem.map((m) => m.backend_mb ?? 0)), backend_mb_end: mem.at(-1)?.backend_mb ?? null, mem,
    wall_s: r1((Date.now() - t0) / 1000), final_text_chars: c.messages.at(-1)?.content?.length ?? 0,
  };
  await inv("delete_conversation", { id: conv }).catch(() => {});
  emit(row);
  subRows.push(row);
}

try {
  await cdp.connect();
  await inv("set_local_roots", { roots: [work] });
  if (MCP) {
    const rep = await inv("import_mcp_file", { path: MCP });
    console.error("mcp import:", JSON.stringify(rep));
    const probe = await inv("create_conversation");
    const deadline = Date.now() + 240000;
    for (;;) {
      const mcp = (await inv("chat_tool_costs", { id: probe.id })).entries.filter((e) => e.kind === "mcp");
      if (mcp.length && mcp.every((e) => e.state === "connected")) break;
      if (Date.now() > deadline) { console.error("WARNING: MCP servers not all connected: " + JSON.stringify(mcp.map((e) => [e.name, e.state]))); break; }
      await sleep(1500);
    }
    await inv("delete_conversation", { id: probe.id }).catch(() => {});
  }
  await inv("import_models", { path: MODEL });
  const id = (await inv("list_models")).find((m) => norm(m.path) === norm(MODEL))?.id;
  if (!id) throw new Error("model not registered: " + MODEL);

  let loadInfo = await loadModel(id);
  if (MODE !== "subagent") {
    for (let i = 0; i < RUNS; i++) {
      if (i > 0) loadInfo = await reload(id);
      await prefillRun(i, id, loadInfo);
    }
  }
  if (MODE !== "prefill") {
    for (let i = 0; i < RUNS; i++) {
      if (i > 0 || MODE === "both") loadInfo = await reload(id); // cold host cache every run
      await subagentRun(i, id, loadInfo);
    }
  }
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1500);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}

const cfg = `builtins_only=${BUILTINS_ONLY} mcp_config=${MCP ?? "none"}`;
if (prefillRows.length) {
  console.log(`\nPREFILL ("hi", cold host cache, ${prefillRows.length} runs, ${cfg})`);
  console.log("metric                    | median | min-max");
  const f = (name, k) => console.log(`${name.padEnd(25)} | ${String(med(prefillRows.map((r) => r[k])) ?? "-").padStart(6)} | ${range(prefillRows.map((r) => r[k]))}`);
  f("tool_count", "tool_count");
  f("chat_context tools tok", "ctx_tools_tokens");
  f("chat_tool_costs sum tok", "tool_costs_sum_tokens");
  f("chat_context total tok", "ctx_total_tokens");
  f("prompt_tokens (sent)", "prompt_tokens");
  f("cached_tokens", "cached_tokens");
  f("time_to_first_token ms", "time_to_first_token_ms");
  f("prompt t/s", "prompt_tokens_per_sec");
  console.log("(prompt_tokens - chat_context total shows what chat_tool_costs/chat_context under-count)");
}
if (subRows.length) {
  console.log(`\nSUBAGENT (2 children x 3 shell steps, ${subRows.length} runs, ${cfg})`);
  for (const r of subRows) {
    console.log(`run ${r.run}: n_ctx ${r.n_ctx}, tools ${r.tool_count}, wall ${r.wall_s}s, child follow-up steps ${r.child_followup_steps} (cached_tokens=0 on ${r.child_followup_steps_cached_zero}), backend MB ${r.backend_mb_start} -> peak ${r.backend_mb_peak} -> ${r.backend_mb_end}`);
    console.log("  seq agent prompt cached ttft_ms");
    for (const s of r.steps) console.log(`  ${String(s.seq).padStart(3)} ${String(s.agent).padEnd(5)} ${String(s.prompt_tokens).padStart(7)} ${String(s.cached_tokens).padStart(7)} ${String(s.time_to_first_token_ms).padStart(8)}`);
  }
  const child = subRows.flatMap((r) => r.steps.filter((s) => s.agent !== "root"));
  console.log(`all child steps: prompt_tokens med ${med(child.map((s) => s.prompt_tokens))} (${range(child.map((s) => s.prompt_tokens))}), cached med ${med(child.map((s) => s.cached_tokens))} (${range(child.map((s) => s.cached_tokens))}), ttft ms med ${med(child.map((s) => s.time_to_first_token_ms))} (${range(child.map((s) => s.time_to_first_token_ms))})`);
  console.log(`backend MB peak med ${med(subRows.map((r) => r.backend_mb_peak))} (${range(subRows.map((r) => r.backend_mb_peak))}); growth start->end med ${med(subRows.map((r) => (r.backend_mb_end ?? 0) - (r.backend_mb_start ?? 0)))}`);
}

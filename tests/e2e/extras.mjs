// End-of-project features, end to end in the real app with an isolated
// profile: capability badges, Share with CPU, Plan mode (chat and tasks),
// thinking controls, file attachments, subagents in chat, worktrees.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   node tests/e2e/extras.mjs            (LAS_SKIP_BIG=1 skips the 27B load)
//
// Uses Will's real model files read-only (registered by path in the test
// profile, never moved), plus the cached Qwen3-4B test model. The 27B
// Share-with-CPU check needs ~16 GB of free RAM: close heavy apps first.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Cdp, countProcesses, launchApp, closeAppGracefully, makeProfile, makeRunner, assert, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
// Will's models: LM Studio's folder until it is removed, then the app's own folder.
const LMS_DIR = path.join(os.homedir(), ".lmstudio", "models");
const LMS = fs.existsSync(LMS_DIR) && fs.readdirSync(LMS_DIR, { recursive: true }).some((n) => String(n).toLowerCase().endsWith(".gguf")) ? LMS_DIR : path.join(os.homedir(), "LocalAIStudio", "models");
const M = {
  qwen4b: path.join(process.env.LOCALAPPDATA, "LocalAIStudio-testcache", "unsloth", "Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
  deepseek: path.join(LMS, "mradermacher", "DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED-i1-GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED.i1-Q6_K.gguf"),
  gemma: path.join(LMS, "lmstudio-community", "gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
  big: path.join(LMS, "lmstudio-community", "Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"),
};
const SKIP_BIG = process.env.LAS_SKIP_BIG === "1";
const PORT = 9347;
const RUN = path.join(os.tmpdir(), `las-e2ex-${Date.now()}`);
const WORK = path.join(RUN, "work");
const SHOTS = path.join(RUN, "screenshots");
const FINAL = ["completed", "failed", "cancelled", "budget_exhausted"];

for (const d of [WORK, SHOTS]) fs.mkdirSync(d, { recursive: true });
fs.writeFileSync(path.join(WORK, "notes.txt"), "Project notes\nThe build uses cargo.\nTODO: add a README.\n");
fs.writeFileSync(path.join(WORK, "main.rs"), "fn main() {\n    println!(\"hello\");\n}\n");
for (const [k, f] of Object.entries(M)) assert(fs.existsSync(f), `model file missing (${k}): ${f}`);
const P = makeProfile(RUN);
assert(countProcesses("llama-server.exe") === 0, "a llama-server is already running (close LM Studio / the app first)");
const snapshotDir = (d) => fs.readdirSync(d).sort().map((n) => `${n}:${fs.statSync(path.join(d, n)).isFile() ? fs.readFileSync(path.join(d, n), "utf8") : "<dir>"}`).join("|");
console.log(`run folder: ${RUN}\n`);

const { results, check } = makeRunner();
const app = launchApp(EXE, P.profile, PORT);
const cdp = new Cdp(PORT);
await cdp.connect();
const inv = (c, a) => cdp.invoke(c, a);
const invErr = (c, a) => cdp.invokeErr(c, a);
const ids = {};
const waitFor = async (fn, ms = 30000, what = "condition") => {
  const deadline = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await sleep(300);
  }
};
const parse = (s, fb = {}) => { try { return typeof s === "string" ? JSON.parse(s) : s ?? fb; } catch { return fb; } };
const model = async (key) => {
  if (!ids[key]) ids[key] = (await inv("import_models", { path: M[key] })).registered[0];
  assert(ids[key], `import failed for ${key}`);
  return (await inv("list_models")).find((m) => m.id === ids[key]);
};
const load = async (key, patch = {}) => {
  const m = await model(key);
  const d = await inv("get_load_defaults", { id: m.id });
  const s = await inv("load_model", { modelId: m.id, config: { ...d.config, n_ctx: 8192, gpu_layers: { mode: "all" }, ...patch }, runtimeId: P.cuda });
  assert(s.state === "loaded", `${key}: ${s.state} ${s.error ?? ""}`);
  return s;
};
const convWith = async (patch) => {
  const c = await inv("create_conversation");
  const v = await inv("get_conversation", { id: c.id });
  await inv("update_chat_settings", { id: c.id, settings: { ...v.settings_parsed, ...patch } });
  return c.id;
};
const waitDone = (conv, ms = 300000) => waitFor(async () => {
  const c = await inv("get_conversation", { id: conv });
  const last = [...c.messages].reverse().find((m) => m.role === "assistant");
  return !c.generating && last?.stats ? c : null;
}, ms, "reply");
const lastAssistant = (c) => [...c.messages].reverse().find((m) => m.role === "assistant");
const toolCalls = (c) => c.messages.filter((m) => m.tool_calls).flatMap((m) => parse(m.tool_calls, []));
const SETTINGS = { environment_id: null, builtin_tools: true, mcp_tools: false, tool_servers: [], skills_mode: "off", skills: [], subagents: false, max_parallel: 2, tool_mode: "auto" };
const BUDGETS = { max_turns: 30, max_tool_calls: 40, max_wall_secs: 900, max_tokens: 60000 };
const startTask = (objective, settings = {}, budgets = {}) =>
  inv("start_task", { title: "", objective, settings: { ...SETTINGS, ...settings }, budgets: { ...BUDGETS, ...budgets } });
const getTask = (id) => inv("get_task", { id });
const waitTask = (id, states = FINAL, ms = 900000) => waitFor(async () => {
  const t = await getTask(id);
  return states.includes(t.run.state) ? t : null;
}, ms, `task to reach ${states.join("/")}`);
function autoApprove(runId, answer = "allow_session") {
  let stop = false;
  (async () => {
    const seen = new Set();
    while (!stop) {
      try {
        for (const p of await inv("pending_tool_approvals")) {
          if ((runId == null || p.run_id === runId || p.conversation_id === runId) && !seen.has(p.id)) {
            seen.add(p.id);
            await inv("answer_tool_approval", { id: p.id, answer });
          }
        }
      } catch {}
      await sleep(300);
    }
  })();
  return { stop: () => { stop = true; } };
}
const WRITE_TOOLS = /fs_write|fs_delete|fs_move|shell|spawn_subagents|skill_propose_edit|write_file|edit_file|move_file|create_directory/;

// ---------------- capability badges (no model loaded) ----------------
await check("capabilities detected from the real model files", async () => {
  const q = (await model("qwen4b")).meta.caps;
  const d = (await model("deepseek")).meta.caps;
  const g = (await model("gemma")).meta.caps;
  assert(q?.version > 0 && q.tools === true && q.thinking_control === "none", `qwen4b ${JSON.stringify(q)}`);
  assert(d?.thinking_control === "always", `deepseek ${JSON.stringify(d)}`);
  assert(g?.vision === true && g.thinking_control === "toggle" && g.thinking_default === false, `gemma ${JSON.stringify(g)}`);
  return `qwen4b tools, deepseek always thinks, gemma vision${g.audio ? "+audio" : ""} + thinking toggle`;
});

await check("My Models shows the badges", async () => {
  await cdp.nav("My Models");
  const t = await cdp.waitFor(`(() => { const s = document.querySelector('main')?.innerText ?? ''; return /always thinks/.test(s) && /vision/.test(s) && /tools/.test(s) ? s : false })()`, 10000, "badges");
  await cdp.screenshot(path.join(SHOTS, "models-badges.png"));
  return `${(t.match(/always thinks|vision|tools|thinking/g) ?? []).length} badge labels`;
});

// ---------------- Plan mode ----------------
await check("Plan mode chat: read-only tools, a numbered plan, nothing changed on disk", async () => {
  await inv("set_local_roots", { roots: [WORK] });
  await load("qwen4b", { n_ctx: 16384 });
  const before = snapshotDir(WORK);
  const conv = await convWith({ agent_mode: true, mode: "plan", builtin_tools: true, tool_servers: [] });
  ids.planConv = conv;
  const ctx = await inv("chat_context", { id: conv, draft: "" });
  // fs_list, fs_read, fs_search, web_fetch and the ask-every-time shell.
  assert(ctx.agent_mode && ctx.tool_count > 0 && ctx.tool_count <= 5, `plan tool count ${ctx.tool_count}`);
  const agentConv = await convWith({ agent_mode: true, mode: "agent", builtin_tools: true, tool_servers: [] });
  const actx = await inv("chat_context", { id: agentConv, draft: "" });
  assert(actx.tool_count > ctx.tool_count, `agent ${actx.tool_count} vs plan ${ctx.tool_count}`);
  // Plan mode shell commands ask every time ("Plan mode: read-only command requested"); deny them here.
  const shellAsks = [];
  const denyShell = (() => {
    let stop = false;
    (async () => {
      const seen = new Set();
      while (!stop) {
        try {
          for (const p of await inv("pending_tool_approvals")) {
            if (p.conversation_id === conv && !seen.has(p.id)) {
              seen.add(p.id);
              shellAsks.push(p);
              await inv("answer_tool_approval", { id: p.id, answer: "deny" });
            }
          }
        } catch {}
        await sleep(300);
      }
    })();
    return { stop: () => { stop = true; } };
  })();
  await inv("chat_send", { id: conv, content: `In the folder ${WORK}, add a README.md that describes the project (see notes.txt). Make a plan for this.` });
  const c = await waitDone(conv).finally(denyShell.stop);
  const calls = toolCalls(c);
  const PLAN_WRITES = /fs_write|fs_delete|fs_move|fs_patch|spawn_subagents|skill_propose_edit|write_file|edit_file|move_file|create_directory/;
  assert(!calls.some((x) => PLAN_WRITES.test(x.function.name)), `write tool used: ${calls.map((x) => x.function.name)}`);
  assert(shellAsks.every((p) => p.tool_name !== "shell_execute" || (/^Plan mode: read-only command requested/.test(p.reason) && p.grant_choice === "once")), JSON.stringify(shellAsks.map((p) => p.reason)));
  const text = lastAssistant(c).content;
  assert(/(^|\n)\s*1[.)]/.test(text), `no numbered plan: ${text.slice(0, 300)}`);
  assert(snapshotDir(WORK) === before, "files changed in Plan mode");
  return `${calls.length} read calls (${[...new Set(calls.map((x) => x.function.name))].join(",") || "none"}), plan ${text.split("\n").length} lines; plan ${ctx.tool_count} tools vs agent ${actx.tool_count}`;
});

await check("Plan mode UI: mode selector and 'Run this plan' carries out the plan with approvals", async () => {
  await cdp.nav("Chat");
  const title = (await inv("get_conversation", { id: ids.planConv })).title;
  await cdp.waitFor(`[...document.querySelectorAll('.chat-item')].some(b => b.querySelector('.chat-title')?.innerText === ${JSON.stringify(title)})`, 10000, "plan chat in list");
  await cdp.eval(`[...document.querySelectorAll('.chat-item')].find(b => b.querySelector('.chat-title')?.innerText === ${JSON.stringify(title)}).click()`);
  await cdp.waitFor(`document.querySelector('.chat-item[aria-current="true"] .chat-title')?.innerText === ${JSON.stringify(title)}`, 10000, "plan chat open");
  // The mode menu sits in the composer bar; its button's value is the mode.
  await cdp.waitFor(`!!document.querySelector('[aria-label="Chat mode"]')`, 10000, "Chat mode selector");
  const mode = await cdp.eval(`document.querySelector('[aria-label="Chat mode"]').value`);
  await cdp.eval(`document.querySelector('[aria-label="Chat mode"]').click()`);
  const opts = await cdp.waitFor(`[...document.querySelectorAll('[role="menuitemradio"]')].map(b => b.dataset.mode + ':' + b.getAttribute('aria-checked')).join(',')`, 5000, "mode menu items").catch(() => "");
  await cdp.eval(`document.querySelector('[aria-label="Chat mode"]').click()`);
  if (opts !== "chat:false,plan:true,agent:false") throw new Error(`mode menu items: ${opts}`);
  const hasRun = await cdp.waitFor(`[...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Run this plan')`, 10000, "Run this plan button").catch(() => false);
  await cdp.screenshot(path.join(SHOTS, "plan-chat.png"));
  if (!hasRun || mode !== "plan") throw new Error(`selector=${mode} runButton=${hasRun} (open the plan chat in the UI failed?)`);
  const ap = autoApprove(null);
  try {
    await cdp.clickButton("Run this plan");
    await waitFor(async () => (await inv("get_conversation", { id: ids.planConv })).generating, 20000, "run to start");
    const c = await waitDone(ids.planConv, 600000);
    const v = await inv("get_conversation", { id: ids.planConv });
    assert(v.settings_parsed.mode === "agent", `mode after run: ${v.settings_parsed.mode}`);
    const user = [...c.messages].reverse().find((m) => m.role === "user");
    assert(/Carry out this approved plan/i.test(user.content), user.content.slice(0, 120));
    const readme = path.join(WORK, "README.md");
    assert(fs.existsSync(readme) && fs.readFileSync(readme, "utf8").trim().length > 20, "README.md not written");
    return `README.md ${fs.statSync(readme).size} bytes`;
  } finally { ap.stop(); }
});

await check("Plan first task: stops for review, edit + approve continues, cancel ends another", async () => {
  fs.rmSync(path.join(WORK, "README.md"), { force: true });
  const before = snapshotDir(WORK);
  const id = await startTask(`In ${WORK}, create a file named summary.txt that summarises notes.txt in one line.`, { plan_first: true });
  const t = await waitTask(id, [...FINAL, "awaiting_plan_approval"], 400000);
  assert(t.run.state === "awaiting_plan_approval", `state ${t.run.state} ${t.run.error ?? ""}`);
  assert(snapshotDir(WORK) === before, "files changed during planning");
  const planStep = t.steps.find((s) => s.kind === "plan");
  assert(planStep && parse(planStep.data).text?.length > 10, "no plan step");
  const e = await invErr("resume_task", { id });
  assert(e, "resume should be refused while waiting for plan approval");
  const edited = `${parse(planStep.data).text}\n\nAlso: the line in summary.txt must start with "SUMMARY:".`;
  const ap = autoApprove(id);
  try {
    await inv("approve_task_plan", { id, plan: edited });
    const done = await waitTask(id);
    assert(done.run.state === "completed", `${done.run.state} ${done.run.error ?? ""}`);
    const f = path.join(WORK, "summary.txt");
    assert(fs.existsSync(f), "summary.txt missing");
    assert(/^\s*SUMMARY:/i.test(fs.readFileSync(f, "utf8")), `edited plan not followed: ${fs.readFileSync(f, "utf8").slice(0, 80)}`);
  } finally { ap.stop(); }
  const id2 = await startTask(`In ${WORK}, delete notes.txt.`, { plan_first: true });
  const t2 = await waitTask(id2, [...FINAL, "awaiting_plan_approval"], 400000);
  assert(t2.run.state === "awaiting_plan_approval", t2.run.state);
  await inv("cancel_task", { id: id2 });
  const c2 = await waitTask(id2, FINAL, 30000);
  assert(c2.run.state === "cancelled" && fs.existsSync(path.join(WORK, "notes.txt")), `cancel: ${c2.run.state}`);
  return "plan reviewed, edited, approved, followed; second cancelled with notes.txt intact";
});

// ---------------- Share with CPU (27B, 16.8 GB on an 8 GB GPU) ----------------
await check("Share with CPU: off suggests it for the 27B; on loads a split with no spill and generates", async () => {
  if (SKIP_BIG) return "skipped (LAS_SKIP_BIG=1)";
  await inv("unload_model");
  const m = await model("big");
  const d = await inv("get_load_defaults", { id: m.id });
  const off = await inv("estimate_memory", { id: m.id, config: { ...d.config, n_ctx: 8192, gpu_layers: { mode: "all" }, share_with_cpu: false } });
  assert(off.suggest_share === true, `suggest_share ${off.suggest_share}`);
  const on = await inv("estimate_memory", { id: m.id, config: { ...d.config, n_ctx: 8192, share_with_cpu: true } });
  assert(on.share_split?.summary, JSON.stringify(on.share_split));
  const small = await model("qwen4b");
  const sd = await inv("get_load_defaults", { id: small.id });
  const se = await inv("estimate_memory", { id: small.id, config: { ...sd.config, n_ctx: 8192, gpu_layers: { mode: "all" }, share_with_cpu: false } });
  assert(se.suggest_share === false, "suggested share for a 4B model");
  const s = await inv("load_model", { modelId: m.id, config: { ...d.config, n_ctx: 8192, share_with_cpu: true }, runtimeId: P.cuda });
  assert(s.state === "loaded", `${s.state} ${s.error ?? ""}`);
  const [n, total] = s.gpu_layers_offloaded ?? [];
  assert(n > 0 && n < total, `layers ${n}/${total}`);
  assert(!(s.warnings ?? []).some((w) => /shared|spill/i.test(w)), `spill: ${s.warnings}`);
  const conv = await convWith({ agent_mode: false });
  await inv("chat_send", { id: conv, content: "Reply with just the word: ready" });
  const c = await waitDone(conv, 600000);
  const st = parse(lastAssistant(c).stats);
  await inv("unload_model");
  return `${on.share_split.summary}; loaded ${n}/${total} layers on GPU; ${st.tokens_per_sec?.toFixed?.(1) ?? "?"} tok/s`;
});

// ---------------- more suites appended below ----------------
const EXTRA = path.join(import.meta.dirname, "extras-more.mjs");
if (fs.existsSync(EXTRA)) {
  const mod = await import(`file://${EXTRA.replace(/\\/g, "/")}`);
  await mod.default({ app, cdp, inv, invErr, check, ids, model, load, convWith, waitDone, waitFor, lastAssistant, toolCalls, startTask, getTask, waitTask, autoApprove, parse, P, M, WORK, RUN, SHOTS, FINAL });
}

// ---------------- finish ----------------
await inv("unload_model").catch(() => {});
await cdp.screenshot(path.join(SHOTS, "final.png")).catch(() => {});
cdp.close();
const closed = await closeAppGracefully(app.pid, 30000);
if (!closed) try { execFileSync("taskkill", ["/F", "/T", "/PID", String(app.pid)], { stdio: "ignore" }); } catch {}
await sleep(1500);
assert(P.snapshot() === P.before, "real runtimes folder changed");
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Screenshots and results: ${RUN}`);
console.log(`llama-server left running: ${countProcesses("llama-server.exe")}`);
process.exit(failed.length ? 1 : 0);

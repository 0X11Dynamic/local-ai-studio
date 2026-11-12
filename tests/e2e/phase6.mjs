// Phase 6 end-to-end hardening run: the failure cases the earlier suites
// do not cover. Real app, isolated profile, Qwen3-4B test model.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   node tests/e2e/phase6.mjs            (LAS_E2E_EXE=<exe> to test another build)
//
// Creates (and removes) one WSL sandbox distro named LocalAIStudio-sbx-*.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Cdp, countProcesses, launchApp, closeAppGracefully, makeProfile, makeRunner, assert, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
const MODEL = path.join(process.env.LOCALAPPDATA, "LocalAIStudio-testcache", "unsloth", "Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
const PORT = 9345;
const RUN = path.join(os.tmpdir(), `las-e2e6-${Date.now()}`);
fs.mkdirSync(RUN, { recursive: true });
assert(fs.existsSync(MODEL), `test model missing: ${MODEL}`);
const P = makeProfile(RUN);
assert(countProcesses("llama-server.exe") === 0, "a llama-server is already running");
const distros = () => execFileSync("wsl.exe", ["--list", "--quiet"], { env: { ...process.env, WSL_UTF8: "1" }, encoding: "utf8" }).replace(/\0/g, "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean).sort();
const DISTROS_BEFORE = distros();
console.log(`exe: ${EXE}\nrun folder: ${RUN}\n`);

const { results, check } = makeRunner();
const app = launchApp(EXE, P.profile, PORT);
const cdp = new Cdp(PORT);
await cdp.connect();
const inv = (c, a) => cdp.invoke(c, a);
const invErr = async (c, a) => { const e = await cdp.invokeErr(c, a); return e?.message ?? String(e); };
const ids = {};
const waitFor = async (fn, ms = 30000, what = "condition") => {
  const deadline = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await sleep(250);
  }
};
const killBackend = () => {
  const pid = execFileSync("powershell.exe", ["-NoProfile", "-Command", `(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.ExecutablePath -like '${P.profile.replace(/'/g, "''")}*' } | Select-Object -First 1).ProcessId`], { encoding: "utf8" }).trim();
  assert(pid, "no backend process to kill");
  execFileSync("taskkill", ["/PID", pid, "/F"]);
  return pid;
};
const load = async (config = {}, runtimeId = P.cuda) => {
  const d = await inv("get_load_defaults", { id: ids.model });
  return inv("load_model", { modelId: ids.model, config: { ...d.config, n_ctx: 8192, gpu_layers: { mode: "all" }, ...config }, runtimeId });
};

await check("out of memory: an impossible load fails with a clear message and leaves nothing running", async () => {
  ids.model = (await inv("import_models", { path: MODEL })).registered[0];
  // ~600 GB of KV cache on the CPU: far beyond what Windows can commit.
  const e = await invErr("load_model", { modelId: ids.model, config: { ...(await inv("get_load_defaults", { id: ids.model })).config, n_ctx: 4194304, gpu_layers: { mode: "cpu_only" }, fit: false }, runtimeId: P.cpu });
  assert(e && /memory/i.test(e), `error: ${e}`);
  await sleep(1000);
  assert(countProcesses("llama-server.exe") === 0, "a backend was left running");
  const s = await load();
  assert(s.state === "loaded", `app not usable after the failure: ${s.state}`);
  return e.slice(0, 140);
});

await check("backend killed mid-reply: the partial reply is kept and the error says what happened", async () => {
  const c = await inv("create_conversation");
  ids.conv = c.id;
  await inv("chat_send", { id: c.id, content: "Write a long story of about 600 words about a lighthouse keeper. No title." });
  await sleep(2500);
  killBackend();
  const v = await waitFor(async () => {
    const x = await inv("get_conversation", { id: c.id });
    return x.generating ? null : x;
  }, 60000, "the reply to end");
  const m = [...v.messages].reverse().find((x) => x.role === "assistant");
  const stats = JSON.parse(m.stats || "{}");
  assert(stats.stop_reason === "error" && /model backend stopped/i.test(stats.error), JSON.stringify(stats));
  assert(m.content.length > 40, `partial reply lost (${m.content.length} chars)`);
  return `${m.content.length} chars kept`;
});

await check("backend killed mid-reply in Agent Mode: partial text kept, same clear error", async () => {
  const s = await load();
  assert(s.state === "loaded", s.state);
  const c = await inv("create_conversation");
  const v0 = await inv("get_conversation", { id: c.id });
  await inv("update_chat_settings", { id: c.id, settings: { ...v0.settings_parsed, agent_mode: true, builtin_tools: false, skills: false, tool_servers: [] } });
  await inv("chat_send", { id: c.id, content: "Without using any tools, write a long story of about 600 words about a desert caravan." });
  await sleep(2500);
  killBackend();
  const v = await waitFor(async () => {
    const x = await inv("get_conversation", { id: c.id });
    return x.generating ? null : x;
  }, 60000, "the reply to end");
  const m = [...v.messages].reverse().find((x) => x.role === "assistant");
  const stats = JSON.parse(m.stats || "{}");
  assert(/model backend stopped/i.test(stats.error || ""), JSON.stringify(stats));
  assert(m.content.length > 40, `partial reply lost (${m.content.length} chars)`);
  return `${m.content.length} chars kept`;
});

await check("backend killed mid-task: the task fails clearly, keeps its progress, and resumes after a reload", async () => {
  assert((await load()).state === "loaded", "reload");
  const id = await inv("start_task", {
    title: "Essay",
    objective: "Write an essay of about 500 words on why lighthouses were important for shipping. Do not use any tools.",
    settings: { environment_id: null, builtin_tools: false, mcp_tools: false, tool_servers: [], skills_mode: "off", skills: [], subagents: false, max_parallel: 2, tool_mode: "auto" },
    budgets: { max_turns: 6, max_tool_calls: 5, max_wall_secs: 600, max_tokens: 20000 },
  });
  await sleep(2500);
  killBackend();
  const failed = await waitFor(async () => {
    const t = await inv("get_task", { id });
    return !t.running ? t : null;
  }, 60000, "the task to stop");
  assert(failed.run.state === "failed" && /model backend stopped/i.test(failed.run.error || ""), `${failed.run.state}: ${failed.run.error}`);
  const e = await invErr("resume_task", { id, budgets: null });
  assert(e && /no model is loaded|load/i.test(e), `resume without a model: ${e}`);
  assert((await load()).state === "loaded", "reload");
  await inv("resume_task", { id, budgets: null });
  const done = await waitFor(async () => {
    const t = await inv("get_task", { id });
    return !t.running && t.run.state !== "running" ? t : null;
  }, 300000, "the resumed task");
  assert(done.run.state === "completed" && (done.run.final_answer || "").length > 200, `${done.run.state}: ${(done.run.final_answer || done.run.error || "").slice(0, 200)}`);
  return `completed after resume, ${done.run.final_answer.length} chars`;
});

await check("a dead backend never hangs a request: chat send after a crash fails fast", async () => {
  killBackend();
  await sleep(1500);
  const t0 = Date.now();
  const e = await invErr("chat_send", { id: ids.conv, content: "hello" });
  assert(e && Date.now() - t0 < 10000, `took ${Date.now() - t0} ms: ${e}`);
  return e.slice(0, 120);
});

// ---------------- sandbox failures ----------------
await check("sandbox shut down mid-command: clear error, the next command works", async () => {
  ids.sbx = await inv("create_sandbox", { name: "Hardening sandbox", network: false });
  await inv("set_builtin_tool_pref", { envId: ids.sbx, tool: "shell_execute", enabled: true, policy: "allow" });
  const distro = distros().find((d) => d.startsWith("LocalAIStudio-sbx-") && !DISTROS_BEFORE.includes(d));
  assert(distro, distros().join(","));
  const running = inv("run_builtin_tool", { envId: ids.sbx, tool: "shell_execute", arguments: { command: "sleep 30; echo done" }, conversationId: null });
  await sleep(3000);
  execFileSync("wsl.exe", ["--terminate", distro]);
  const t0 = Date.now();
  const r = await running;
  assert(r.status !== "ok" || !r.content.includes("done"), JSON.stringify(r));
  assert(Date.now() - t0 < 15000, "hung after the shutdown");
  const again = await inv("run_builtin_tool", { envId: ids.sbx, tool: "shell_execute", arguments: { command: "echo back" }, conversationId: null });
  assert(again.status === "ok" && again.content.includes("back"), JSON.stringify(again));
  ids.distro = distro;
  return r.content.split("\n")[0].slice(0, 120);
});

await check("sandbox removed outside the app: health and commands say so plainly", async () => {
  execFileSync("wsl.exe", ["--unregister", ids.distro]);
  const h = await inv("environment_health", { id: ids.sbx });
  assert(!h.ok && /removed/i.test(h.detail), h.detail);
  const r = await inv("run_builtin_tool", { envId: ids.sbx, tool: "shell_execute", arguments: { command: "echo x" }, conversationId: null }).catch((e) => ({ status: "error", content: String(e) }));
  assert(r.status !== "ok" && /removed/i.test(r.content), JSON.stringify(r));
  await inv("delete_environment", { id: ids.sbx });
  assert(!(await inv("list_environments")).some((e) => e.id === ids.sbx), "not deleted");
  return h.detail.slice(0, 120);
});

// ---------------- lifecycle ----------------
await check("closing the app leaves no model or shell processes", async () => {
  cdp.close();
  assert(await closeAppGracefully(app.pid), "app did not close");
  await sleep(1500);
  assert(countProcesses("llama-server.exe") === 0, "llama-server left");
});

results.push({ name: "WSL distros unchanged", ok: JSON.stringify(distros()) === JSON.stringify(DISTROS_BEFORE), detail: distros().join(",") });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  WSL distros unchanged  (${distros().join(",")})`);
results.push({ name: "real runtimes folder unchanged", ok: P.snapshot() === P.before, detail: P.snapshot() });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${P.snapshot()})`);
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Results: ${RUN}`);
process.exit(failed.length ? 1 : 0);

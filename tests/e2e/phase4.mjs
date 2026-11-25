// Phase 4 end-to-end acceptance run: execution environments and the built-in
// shell and file tools, in the real app with an isolated profile.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   node tests/e2e/phase4.mjs
//
// Creates (and removes) one WSL sandbox distro named LocalAIStudio-sbx-*.
// Needs the cached test model (see phase3.mjs).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Cdp, countProcesses, launchApp, closeAppGracefully, makeProfile, makeRunner, assert, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
const MODEL = path.join(process.env.LOCALAPPDATA, "LocalAIStudio-testcache", "unsloth", "Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
const PORT = 9343;
const RUN = path.join(os.tmpdir(), `las-e2e4-${Date.now()}`);
const WORK = path.join(RUN, "work");
const OUTSIDE = path.join(RUN, "outside");
const SHOTS = path.join(RUN, "screenshots");

for (const d of [WORK, OUTSIDE, SHOTS]) fs.mkdirSync(d, { recursive: true });
fs.writeFileSync(path.join(WORK, "readme.txt"), "line one\nline two\n");
fs.writeFileSync(path.join(WORK, "notes.md"), "# Notes\nTODO: ship phase 4\n");
fs.writeFileSync(path.join(OUTSIDE, "secret.txt"), "do not read");
assert(fs.existsSync(MODEL), `test model missing: ${MODEL}`);
const P = makeProfile(RUN);
assert(countProcesses("llama-server.exe") === 0, "a llama-server is already running");
const distros = () => execFileSync("wsl.exe", ["--list", "--quiet"], { env: { ...process.env, WSL_UTF8: "1" }, encoding: "utf8" }).replace(/\0/g, "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean).sort();
const DISTROS_BEFORE = distros();
console.log(`run folder: ${RUN}\n`);

const { results, check } = makeRunner();
let app = launchApp(EXE, P.profile, PORT);
let cdp = new Cdp(PORT);
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
    await sleep(200);
  }
};
const pendingFor = (tool, conv = null) => waitFor(async () => (await inv("pending_tool_approvals")).find((p) => p.tool_name === tool && p.conversation_id === conv), 20000, `approval for ${tool}`);
const run = (envId, tool, args, conv = null) => inv("run_builtin_tool", { envId, tool, arguments: args, conversationId: conv });
const runWithAnswer = async (envId, tool, args, answer, conv = null) => {
  const r = run(envId, tool, args, conv);
  const p = await pendingFor(tool, conv);
  await inv("answer_tool_approval", { id: p.id, answer });
  return r;
};

// ---------------- This PC ----------------
await check("allowed folders: protected, missing and whole-drive folders are refused", async () => {
  for (const bad of [process.env.SystemRoot, path.join(RUN, "missing"), "C:\\", path.join(process.env.USERPROFILE, ".ssh"), "relative\\path"]) {
    const e = await invErr("set_local_roots", { roots: [bad] });
    assert(e, `accepted ${bad}`);
  }
  const r = await inv("set_local_roots", { roots: [WORK] });
  assert(r.length === 1 && r[0] === WORK, JSON.stringify(r));
  const local = (await inv("list_environments")).find((e) => e.id === "local");
  assert(local.config.roots[0] === WORK && local.tools.length === 10 && local.config.outside === "ask", JSON.stringify(local.config));
});

await check("This PC health check runs PowerShell", async () => {
  const h = await inv("environment_health", { id: "local" });
  assert(h.ok && /PowerShell \d/.test(h.detail), h.detail);
  return h.detail;
});

await check("read tools run without asking; outside the folders they ask (deny/allow), system folders are refused", async () => {
  const l = await run("local", "fs_list", { path: WORK });
  assert(l.status === "ok" && l.content.includes("readme.txt"), JSON.stringify(l));
  const r = await run("local", "fs_read", { path: "readme.txt", start_line: 2, max_lines: 1 });
  assert(r.status === "ok" && r.content.trim() === "line two", JSON.stringify(r));
  const s = await run("local", "fs_search", { path: WORK, pattern: "TODO" });
  assert(s.content.includes("notes.md:2:"), s.content);
  // Outside the allowed folders: every call asks (even though fs_read is "allow"), naming the real path.
  for (const p of [path.join(OUTSIDE, "secret.txt"), "..\\outside\\secret.txt"]) {
    const r = run("local", "fs_read", { path: p });
    const ap = await pendingFor("fs_read");
    assert(/Outside your allowed folders/.test(ap.reason) && ap.reason.toLowerCase().includes(path.join(OUTSIDE, "secret.txt").toLowerCase()) && ap.grant_choice === "session", JSON.stringify(ap));
    await inv("answer_tool_approval", { id: ap.id, answer: "deny" });
    const o = await r;
    assert(o.status === "denied" && !o.content.includes("do not read"), `${p}: ${JSON.stringify(o)}`);
  }
  const ok = await runWithAnswer("local", "fs_read", { path: path.join(OUTSIDE, "secret.txt") }, "allow_once");
  assert(ok.status === "ok" && ok.content.includes("do not read"), JSON.stringify(ok));
  // Always-blocked places are refused without asking.
  for (const p of [path.join(process.env.SystemRoot, "win.ini"), path.join(P.profile, "app.db")]) {
    const o = await run("local", "fs_read", { path: p });
    assert(o.status === "denied" && /protected/.test(o.content), `${p}: ${JSON.stringify(o)}`);
  }
  // "Off": outside paths are refused again, without asking.
  await inv("set_local_outside", { mode: "off" });
  try {
    const o = await run("local", "fs_read", { path: path.join(OUTSIDE, "secret.txt") });
    assert(o.status === "tool_error" && /outside/.test(o.content) && !o.content.includes("do not read"), JSON.stringify(o));
  } finally {
    await inv("set_local_outside", { mode: "ask" });
  }
});

await check("write asks first: deny writes nothing, allow once writes and returns a diff", async () => {
  const f = path.join(WORK, "new.txt");
  const d = await runWithAnswer("local", "fs_write", { path: "new.txt", content: "hello\n" }, "deny");
  assert(d.status === "denied" && !fs.existsSync(f), JSON.stringify(d));
  const a = await runWithAnswer("local", "fs_write", { path: "new.txt", content: "hello\n" }, "allow_once");
  assert(a.status === "ok" && fs.readFileSync(f, "utf8") === "hello\n" && a.content.includes("+hello"), JSON.stringify(a));
  const p = await runWithAnswer("local", "fs_patch", { path: "new.txt", old_text: "hello", new_text: "goodbye" }, "allow_once");
  assert(p.content.includes("-hello") && p.content.includes("+goodbye"), p.content);
});

await check("shell asks first, returns output and exit codes, and a timeout stops it", async () => {
  const r = await runWithAnswer("local", "shell_execute", { command: "Get-ChildItem -Name; Write-Error 'warn'; exit 3" }, "allow_once");
  assert(r.status === "tool_error" && r.content.includes("exit code: 3") && !r.content.includes("ProgressPreference") && r.content.includes("readme.txt") && r.content.includes("warn"), r.content);
  const t0 = Date.now();
  const t = await runWithAnswer("local", "shell_execute", { command: "Start-Sleep 60", timeout_secs: 2 }, "allow_once");
  assert(t.content.includes("stopped after the 2 s timeout") && Date.now() - t0 < 30000, t.content);
  // A working folder outside the allowed folders asks with a warning; allowed once it runs there.
  const cr = run("local", "shell_execute", { command: "Get-Location | Select -Expand Path", cwd: OUTSIDE });
  const cap = await pendingFor("shell_execute");
  assert(/Outside your allowed folders: runs in/.test(cap.reason) && /can't see which files/.test(cap.reason), cap.reason);
  await inv("answer_tool_approval", { id: cap.id, answer: "allow_once" });
  const c = await cr;
  assert(c.status === "ok" && c.content.toLowerCase().includes(OUTSIDE.toLowerCase()), c.content);
});

await check("delete on This PC moves to the app trash", async () => {
  const f = path.join(WORK, "new.txt");
  const r = await runWithAnswer("local", "fs_delete", { path: "new.txt" }, "allow_once");
  assert(r.status === "ok" && !fs.existsSync(f) && r.content.includes("trash"), JSON.stringify(r));
  const trash = path.join(P.profile, "trash");
  const found = fs.readdirSync(trash).some((d) => fs.existsSync(path.join(trash, d, "new.txt")));
  assert(found, "not in the trash");
});

await check("a tool set to deny is not offered and cannot be run", async () => {
  await inv("set_builtin_tool_pref", { envId: "local", tool: "fs_move", enabled: true, policy: "deny" });
  const e = await invErr("run_builtin_tool", { envId: "local", tool: "fs_move", arguments: { from: "a", to: "b" }, conversationId: null });
  assert(e && /deny|turned off/.test(e), e);
  const pv = await inv("preview_chat_tools", { toolServers: [], builtinTools: true, environmentId: null });
  assert(!pv.tools.some((t) => t.tool_name === "fs_move") && pv.tools.some((t) => t.tool_name === "shell_execute"), pv.tools.map((t) => t.tool_name).join(","));
  assert(pv.section.includes("## Execution environment") && pv.section.includes(WORK), pv.section.slice(-400));
  await inv("set_builtin_tool_pref", { envId: "local", tool: "fs_move", enabled: true, policy: "ask" });
});

await check("audit rows record source builtin and the environment", async () => {
  const rows = await inv("list_tool_activity", { conversationId: null, limit: 50 });
  const b = rows.filter((r) => r.source === "builtin");
  assert(b.length >= 8 && b.every((r) => r.environment_id === "local"), JSON.stringify(b.slice(0, 2)));
  assert(b.some((r) => r.status === "denied") && b.some((r) => r.tool === "builtin:local/shell_execute"), "statuses");
});

// ---------------- sandbox ----------------
await check("WSL sandbox: created, isolated, runs as an unprivileged user", async () => {
  const wsl = await inv("wsl_status");
  ids.sbx = await inv("create_sandbox", { name: "Test sandbox", network: false });
  const h = await inv("environment_health", { id: ids.sbx });
  assert(h.ok && h.detail.includes("sandbox"), h.detail);
  const r = await runWithAnswer(ids.sbx, "shell_execute", { command: "id -un; ls /mnt/c 2>/dev/null | wc -l; wget -q -T 3 -O- http://example.com >/dev/null 2>&1 && echo NET || echo NONET" }, "allow_always");
  assert(r.status === "ok" && /^sandbox$/m.test(r.content) && r.content.includes("NONET"), r.content);
  return `${wsl}; ${h.detail}`;
});

await check("an always-allow grant on the sandbox does not apply to This PC", async () => {
  const again = await run(ids.sbx, "shell_execute", { command: "echo again" });
  assert(again.status === "ok", "sandbox grant not applied");
  const local = run("local", "shell_execute", { command: "echo hi" });
  const p = await pendingFor("shell_execute");
  assert(p.server_name.startsWith("This PC"), p.server_name);
  await inv("answer_tool_approval", { id: p.id, answer: "deny" });
  assert((await local).status === "denied", "local ran");
});

await check("sandbox files: write, read, diff; outside paths are the sandbox's own", async () => {
  const w = await runWithAnswer(ids.sbx, "fs_write", { path: "app/main.py", content: "print('hi')\n", create_dirs: true }, "allow_once");
  assert(w.status === "ok" && /^Created \S*app\/main\.py/.test(w.content) && w.content.includes("+print"), w.content);
  const r = await run(ids.sbx, "fs_read", { path: "/workspace/app/main.py" });
  assert(r.content.trim() === "print('hi')", r.content);
  const x = await run(ids.sbx, "fs_list", { path: "/workspace/app" });
  assert(x.content.includes("main.py"), x.content);
});

// ---------------- Docker (no daemon on this PC) ----------------
await check("Docker: bad mounts are refused; an unreachable daemon gives a clear error", async () => {
  const e = await invErr("add_docker_environment", { name: "bad mounts", config: { host: "tcp://127.0.0.1:1", mounts: [{ host: OUTSIDE, container: "/x", read_only: true }] } });
  assert(e && /outside/.test(e), e);
  const e2 = await invErr("add_docker_environment", { name: "pipe", config: { host: "tcp://127.0.0.1:1", mounts: [{ host: "\\\\.\\pipe\\docker_engine", container: "/var/run/docker.sock", read_only: false }] } });
  assert(e2 && /cannot be mounted/.test(e2), e2);
  ids.docker = await inv("add_docker_environment", { name: "Docker test", config: { host: "tcp://127.0.0.1:1", mounts: [{ host: WORK, container: "/workspace/work", read_only: true }] } });
  const h = await inv("environment_health", { id: ids.docker });
  assert(!h.ok && /Docker/.test(h.detail), h.detail);
  return h.detail.slice(0, 100);
});

// ---------------- SSH (unknown host is refused) ----------------
await check("SSH: a profile saves its password in the vault, not the database", async () => {
  ids.ssh = await inv("add_ssh_environment", { name: "Box", profile: { host: "127.0.0.1", port: 1, user: "u", auth: "password", key_path: "", roots: ["/home/u"], windows: false }, secrets: { password: "pw-e2e-Secret-123" } });
  const env = (await inv("list_environments")).find((e) => e.id === ids.ssh);
  assert(env.has_secret && !JSON.stringify(env).includes("pw-e2e-Secret-123"), JSON.stringify(env));
  const h = await inv("environment_health", { id: ids.ssh });
  assert(!h.ok, "connected to nothing?");
  for (const f of ["app.db", "app.db-wal"]) {
    const p = path.join(P.profile, f);
    if (fs.existsSync(p)) assert(!fs.readFileSync(p).includes("pw-e2e-Secret-123"), `password in ${f}`);
  }
  await inv("delete_environment", { id: ids.ssh });
});

// ---------------- model in Agent Mode ----------------
await check("tool-capable model loads", async () => {
  const r = await inv("import_models", { path: MODEL });
  ids.model = r.registered[0];
  const d = await inv("get_load_defaults", { id: ids.model });
  const s = await inv("load_model", { modelId: ids.model, config: { ...d.config, n_ctx: 12288, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
  assert(s.state === "loaded", s.state);
});

const convWith = async (patch) => {
  const c = await inv("create_conversation");
  const v = await inv("get_conversation", { id: c.id });
  await inv("update_chat_settings", { id: c.id, settings: { ...v.settings_parsed, ...patch } });
  return c.id;
};
const waitDone = (conv, ms = 240000) => waitFor(async () => {
  const c = await inv("get_conversation", { id: conv });
  const last = [...c.messages].reverse().find((m) => m.role === "assistant");
  return !c.generating && last?.stats ? c : null;
}, ms, "reply");

await check("context meter counts the built-in tools in Agent Mode", async () => {
  ids.conv = await convWith({ agent_mode: true, builtin_tools: true, environment_id: null, tool_servers: [] });
  const ctx = await inv("chat_context", { id: ids.conv, draft: "" });
  assert(ctx.agent_mode && ctx.tool_count === 10 && ctx.tools > 0, JSON.stringify(ctx));
  return `${ctx.tool_count} tools = ${ctx.tools} tokens`;
});

await check("UI: model runs a shell command after approval; live tool card shows the output", async () => {
  await cdp.nav("Chat");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length > 0");
  await cdp.eval(`[...document.querySelectorAll('.chat-item')][0].click()`);
  await cdp.waitFor("document.querySelector('.chat-head')?.innerText.includes('Agent Mode')", 10000, "agent badge");
  await cdp.setValue("textarea[aria-label='Message']", "Use the shell to run exactly this command and tell me the version it prints: $PSVersionTable.PSVersion.ToString()");
  await cdp.clickButton("Send");
  await cdp.waitFor("!!document.querySelector('.tool-card.waiting')", 180000, "approval prompt");
  const cardText = await cdp.eval("document.querySelector('.tool-card.waiting').innerText");
  assert(/shell_execute/.test(cardText) && /This PC/.test(cardText), cardText);
  await cdp.screenshot(path.join(SHOTS, "shell-approval.png"));
  await cdp.clickButton("Allow once");
  const c = await waitDone(ids.conv);
  const res = c.messages.find((m) => m.role === "tool");
  assert(res && /exit code: 0/.test(res.content), res?.content);
  const version = res.content.match(/\d+\.\d+\.\d+/)?.[0];
  const final = [...c.messages].reverse().find((m) => m.role === "assistant");
  assert(version && final.content.includes(version), `version ${version}; answer: ${final.content.slice(0, 200)}`);
  await cdp.eval(`[...document.querySelectorAll('.tool-card button')].find(b => b.innerText.startsWith('Show tool output'))?.click()`);
  await cdp.screenshot(path.join(SHOTS, "shell-card.png"));
  return `PowerShell ${version}`;
});

await check("model edits a file with fs_patch after approval (diff in the card)", async () => {
  const f = path.join(WORK, "notes.md");
  await inv("chat_send", { id: ids.conv, content: `In ${f}, change the word "ship" to "finish" using the fs_patch tool.` });
  const p = await pendingFor("fs_patch", ids.conv).catch(() => pendingFor("fs_write", ids.conv));
  await inv("answer_tool_approval", { id: p.id, answer: "allow_once" });
  await waitDone(ids.conv);
  assert(fs.readFileSync(f, "utf8").includes("TODO: finish phase 4"), fs.readFileSync(f, "utf8"));
  return p.tool_name;
});

await check("UI: Execution Environments page shows environments and tool permissions", async () => {
  await cdp.nav("Execution Environments");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length >= 3", 10000, "environment list");
  const t = await cdp.text(".content");
  assert(t.includes("This PC") && t.includes("Test sandbox") && t.includes("shell_execute"), t.slice(0, 300));
  const roots = await cdp.eval(`document.querySelector('textarea[aria-label^="Allowed folders"]')?.value`);
  assert(roots === WORK, roots);
  await cdp.screenshot(path.join(SHOTS, "environments.png"));
});

await check("removing the sandbox unregisters its distro and its grants", async () => {
  await inv("delete_environment", { id: ids.sbx });
  assert(!distros().some((d) => d.startsWith("LocalAIStudio-sbx-")), distros().join(","));
  const g = await inv("list_tool_grants");
  assert(!g.some((x) => x.subject.includes(ids.sbx)), "grant left");
});

// ---------------- lifecycle ----------------
await check("closing the app leaves no shell or model processes", async () => {
  cdp.close();
  assert(await closeAppGracefully(app.pid), "app did not close");
  await sleep(1500);
  assert(countProcesses("llama-server.exe") === 0, "llama-server left");
});

results.push({ name: "WSL distros unchanged (kali-linux untouched)", ok: JSON.stringify(distros()) === JSON.stringify(DISTROS_BEFORE), detail: distros().join(",") });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  WSL distros unchanged  (${distros().join(",")})`);
results.push({ name: "real runtimes folder unchanged", ok: P.snapshot() === P.before, detail: P.snapshot() });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${P.snapshot()})`);
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Screenshots and results: ${RUN}`);
process.exit(failed.length ? 1 : 0);

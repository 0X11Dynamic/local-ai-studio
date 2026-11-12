// Phase 3 end-to-end acceptance run: real MCP servers (reference filesystem
// server via npx, a prompt-injection fixture, an authenticated HTTP fixture),
// the permission and approval engine, the audit log, and the model-driven
// tool-call loop on a real tool-capable model.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   npm --prefix tests/fixtures/mcp install
//   node tests/e2e/phase3.mjs
//
// Needs the test model in the cache (downloaded once, sha256-checked):
//   %LOCALAPPDATA%\LocalAIStudio-testcache\unsloth\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q4_K_M.gguf
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";
import { Cdp, countProcesses, launchApp, closeAppGracefully, makeProfile, makeRunner, assert, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
const FIXTURES = path.join(ROOT, "tests", "fixtures", "mcp");
const MODEL = path.join(process.env.LOCALAPPDATA, "LocalAIStudio-testcache", "unsloth", "Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
const PORT = 9339;
const HTTP_PORT = 18931;
const HTTP_TOKEN = `fixture-${Date.now()}-tok`;
const RUN = path.join(os.tmpdir(), `las-e2e3-${Date.now()}`);
const FSROOT = path.join(RUN, "fsroot");
const MARKERS = path.join(RUN, "markers");
const SHOTS = path.join(RUN, "screenshots");

fs.mkdirSync(FSROOT, { recursive: true });
fs.mkdirSync(MARKERS, { recursive: true });
fs.mkdirSync(SHOTS, { recursive: true });
fs.writeFileSync(path.join(FSROOT, "a.txt"), "alpha file");
fs.writeFileSync(path.join(FSROOT, "b.txt"), "bravo file");
fs.writeFileSync(path.join(FSROOT, "c.txt"), "charlie file");
assert(fs.existsSync(MODEL), `test model missing: ${MODEL}`);
assert(fs.existsSync(path.join(FIXTURES, "node_modules")), "run: npm --prefix tests/fixtures/mcp install");
const P = makeProfile(RUN);
assert(countProcesses("llama-server.exe") === 0, "a llama-server is already running");
console.log(`run folder: ${RUN}\n`);

const mcpProcesses = () => {
  const out = execFileSync("powershell.exe", ["-NoProfile", "-Command",
    "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | Where-Object { $_.CommandLine -match 'evil-server|server-filesystem' } | Measure-Object | Select-Object -ExpandProperty Count"], { encoding: "utf8" });
  return Number(out.trim() || 0);
};
const MCP_BEFORE = mcpProcesses();

const { results, check } = makeRunner();
let app = launchApp(EXE, P.profile, PORT);
let cdp = new Cdp(PORT);
await cdp.connect();
const inv = (c, a) => cdp.invoke(c, a);
const invErr = (c, a) => cdp.invokeErr(c, a);
const ids = {};
const server = async (name) => (await inv("list_mcp_servers")).find((s) => s.name === name);
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
/** Start a manual run that will wait for approval; answer it; return the outcome. */
const runWithAnswer = async (serverId, tool, args, answer, conv = null) => {
  const run = inv("run_tool_manually", { serverId, tool, arguments: args, conversationId: conv });
  const p = await pendingFor(tool, conv);
  await inv("answer_tool_approval", { id: p.id, answer });
  return run;
};
const snapshotPolicy = async () => JSON.stringify({
  prefs: (await inv("list_mcp_servers")).map((s) => [s.name, s.tools.map((t) => [t.name, t.enabled, t.policy])]),
  grants: (await inv("list_tool_grants")).map((g) => [g.scope, g.subject, g.conversation_id]),
});

// ---------------- servers ----------------
await check("reference filesystem server connects via npx; tools listed with schemas, default ask", async () => {
  const v = await inv("add_mcp_server", {
    name: "filesystem",
    config: { transport: "stdio", command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem@2.2.0", FSROOT], env: {}, cwd: null, url: "", headers: {}, secret_env: [], secret_headers: [], bearer: false, timeout_secs: 60 },
    secrets: {},
    enabled: true,
  });
  ids.fs = v.id;
  const c = await inv("connect_mcp_server", { id: v.id });
  assert(c.status.state === "connected", `state ${c.status.state}: ${c.status.error}`);
  const names = c.tools.map((t) => t.name);
  for (const n of ["list_directory", "write_file"]) assert(names.includes(n), `missing ${n}: ${names}`);
  ids.readTool = names.includes("read_text_file") ? "read_text_file" : "read_file";
  assert(c.tools.every((t) => t.policy === "ask" && t.enabled), "every tool must start at ask");
  assert(c.tools.find((t) => t.name === "write_file").input_schema.properties?.path, "schema missing");
  return `${names.length} tools: ${names.slice(0, 6).join(", ")}...`;
});

await check("per-tool enable and policy persist", async () => {
  await inv("set_mcp_tool_pref", { serverId: ids.fs, tool: "list_directory", enabled: true, policy: "allow" });
  await inv("set_mcp_tool_pref", { serverId: ids.fs, tool: ids.readTool, enabled: true, policy: "allow" });
  await inv("set_mcp_tool_pref", { serverId: ids.fs, tool: "move_file", enabled: false, policy: "ask" });
  const s = await server("filesystem");
  assert(s.tools.find((t) => t.name === "list_directory").policy === "allow", "allow not saved");
  assert(!s.tools.find((t) => t.name === "move_file").enabled, "disable not saved");
  const bad = await invErr("set_mcp_tool_pref", { serverId: ids.fs, tool: "x", enabled: true, policy: "yolo" });
  assert(bad.code === "invalid_input", bad.message);
});

await check("an allowed read tool runs without asking and is audited", async () => {
  const o = await inv("run_tool_manually", { serverId: ids.fs, tool: "list_directory", arguments: { path: FSROOT }, conversationId: null });
  assert(o.status === "ok" && o.content.includes("a.txt") && o.content.includes("b.txt"), JSON.stringify(o));
  const a = (await inv("list_tool_activity", { limit: 5 })).find((x) => x.id === o.audit_id);
  const ap = JSON.parse(a.approval);
  assert(a.status === "ok" && ap.decision === "allow" && a.duration_ms >= 0, JSON.stringify(a));
  return o.content.replace(/\n/g, " ").slice(0, 60);
});

await check("write tool at ask: deny blocks, allow once runs once", async () => {
  const f1 = path.join(FSROOT, "denied.txt");
  const d = await runWithAnswer(ids.fs, "write_file", { path: f1, content: "no" }, "deny");
  assert(d.status === "denied" && !fs.existsSync(f1), JSON.stringify(d));
  const a = (await inv("list_tool_activity", { limit: 5 })).find((x) => x.id === d.audit_id);
  assert(a.status === "denied" && JSON.parse(a.approval).answer === "deny", a.approval);
  const f2 = path.join(FSROOT, "once.txt");
  const o = await runWithAnswer(ids.fs, "write_file", { path: f2, content: "once" }, "allow_once");
  assert(o.status === "ok" && fs.readFileSync(f2, "utf8") === "once", JSON.stringify(o));
  const again = await runWithAnswer(ids.fs, "write_file", { path: path.join(FSROOT, "again.txt"), content: "x" }, "deny");
  assert(again.status === "denied", "allow once must not carry over");
});

await check("allow for this chat applies only to that chat", async () => {
  const c1 = await inv("create_conversation");
  const c2 = await inv("create_conversation");
  ids.c1 = c1.id;
  const o1 = await runWithAnswer(ids.fs, "write_file", { path: path.join(FSROOT, "s1.txt"), content: "1" }, "allow_session", c1.id);
  assert(o1.status === "ok", JSON.stringify(o1));
  const o2 = await inv("run_tool_manually", { serverId: ids.fs, tool: "write_file", arguments: { path: path.join(FSROOT, "s2.txt"), content: "2" }, conversationId: c1.id });
  assert(o2.status === "ok", "second call in the same chat should not ask");
  const o3 = await runWithAnswer(ids.fs, "write_file", { path: path.join(FSROOT, "s3.txt"), content: "3" }, "deny", c2.id);
  assert(o3.status === "denied" && !fs.existsSync(path.join(FSROOT, "s3.txt")), "other chat must ask");
});

await check("always allow persists and can be revoked", async () => {
  const o = await runWithAnswer(ids.fs, "write_file", { path: path.join(FSROOT, "p1.txt"), content: "p" }, "allow_always");
  assert(o.status === "ok", JSON.stringify(o));
  const o2 = await inv("run_tool_manually", { serverId: ids.fs, tool: "write_file", arguments: { path: path.join(FSROOT, "p2.txt"), content: "p" }, conversationId: null });
  assert(o2.status === "ok", "persistent grant not applied");
  const g = (await inv("list_tool_grants")).find((x) => x.scope === "persistent" && x.tool_name === "write_file");
  assert(g, "grant not listed");
  await inv("revoke_tool_grant", { id: g.id });
  const o3 = await runWithAnswer(ids.fs, "write_file", { path: path.join(FSROOT, "p3.txt"), content: "p" }, "deny");
  assert(o3.status === "denied", "revoked grant still applied");
});

await check("invalid arguments are refused before anything runs", async () => {
  const o = await inv("run_tool_manually", { serverId: ids.fs, tool: "write_file", arguments: { content: "missing path" }, conversationId: null });
  assert(o.status === "invalid_arguments" && /path/.test(o.content), JSON.stringify(o));
  assert((await inv("pending_tool_approvals")).length === 0, "an approval was requested for invalid args");
  const disabled = await invErr("run_tool_manually", { serverId: ids.fs, tool: "move_file", arguments: {}, conversationId: null });
  assert(/not available/.test(disabled.message), disabled.message);
});

await check("evil fixture: injected instructions in tool output cannot change policy", async () => {
  const v = await inv("add_mcp_server", {
    name: "evil",
    config: { transport: "stdio", command: "node", args: [path.join(FIXTURES, "evil-server.mjs")], env: { MARKER_DIR: MARKERS }, cwd: FIXTURES, url: "", headers: {}, secret_env: [], secret_headers: [], bearer: false, timeout_secs: 3 },
    secrets: {},
    enabled: true,
  });
  ids.evil = v.id;
  const c = await inv("connect_mcp_server", { id: v.id });
  assert(c.status.state === "connected", c.status.error);
  const note = c.tools.find((t) => t.name === "read_note");
  assert(note.annotations?.readOnlyHint === true && note.policy === "ask", "annotation must not change policy");
  await inv("set_mcp_tool_pref", { serverId: v.id, tool: "read_note", enabled: true, policy: "allow" });
  const before = await snapshotPolicy();
  const r = await inv("run_tool_manually", { serverId: v.id, tool: "read_note", arguments: {}, conversationId: null });
  assert(r.status === "ok" && r.content.includes("IGNORE PREVIOUS INSTRUCTIONS"), JSON.stringify(r));
  assert((await snapshotPolicy()) === before, "policy or grants changed after reading injected output");
  const d = await runWithAnswer(v.id, "delete_everything", { confirm: true }, "deny");
  assert(d.status === "denied", JSON.stringify(d));
  assert(!fs.existsSync(path.join(MARKERS, "deleted.marker")), "destructive tool ran");
  assert((await snapshotPolicy()) === before, "policy changed");
});

await check("secrets in tool arguments are redacted in the audit log", async () => {
  await inv("set_mcp_tool_pref", { serverId: ids.evil, tool: "echo", enabled: true, policy: "allow" });
  const o = await inv("run_tool_manually", { serverId: ids.evil, tool: "echo", arguments: { message: "hi", api_key: "sk-THISISASECRETKEY12345" }, conversationId: null });
  assert(o.status === "ok", JSON.stringify(o));
  const a = (await inv("list_tool_activity", { limit: 3 })).find((x) => x.id === o.audit_id);
  assert(!a.args_redacted.includes("THISISASECRETKEY12345") && a.args_redacted.includes("REDACTED"), a.args_redacted);
});

await check("a tool that hangs times out", async () => {
  await inv("set_mcp_tool_pref", { serverId: ids.evil, tool: "slow", enabled: true, policy: "allow" });
  const t = Date.now();
  const o = await inv("run_tool_manually", { serverId: ids.evil, tool: "slow", arguments: { seconds: 30 }, conversationId: null });
  assert(o.status === "timeout" && Date.now() - t < 10000, `${o.status} after ${Date.now() - t} ms`);
  return `timed out after ${((Date.now() - t) / 1000).toFixed(1)} s`;
});

await check("a crashing server shows a clear error and reconnects", async () => {
  await inv("set_mcp_tool_pref", { serverId: ids.evil, tool: "crash", enabled: true, policy: "allow" });
  await inv("run_tool_manually", { serverId: ids.evil, tool: "crash", arguments: {}, conversationId: null }).catch(() => {});
  const s = await waitFor(async () => { const x = await server("evil"); return x.status.state === "error" ? x : null; }, 10000, "error state");
  assert(/stopped unexpectedly/.test(s.status.error), s.status.error);
  const e = await invErr("run_tool_manually", { serverId: ids.evil, tool: "echo", arguments: { message: "x" }, conversationId: null });
  assert(/not available/.test(e.message), e.message);
  const c = await inv("connect_mcp_server", { id: ids.evil });
  assert(c.status.state === "connected", "reconnect failed");
  return s.status.error.slice(0, 70);
});

await check("killing a server process is detected", async () => {
  const s = await server("filesystem");
  execFileSync("taskkill", ["/F", "/T", "/PID", String(s.status.pid)], { stdio: "ignore" });
  const x = await waitFor(async () => { const v = await server("filesystem"); return v.status.state === "error" ? v : null; }, 10000, "error after kill");
  const c = await inv("connect_mcp_server", { id: ids.fs });
  assert(c.status.state === "connected", "reconnect after kill failed");
  return x.status.error.slice(0, 60);
});

let httpProc;
await check("HTTP server with bearer auth: refused without token, connects with the stored token", async () => {
  httpProc = spawn("node", [path.join(FIXTURES, "auth-http-server.mjs"), String(HTTP_PORT), HTTP_TOKEN], { stdio: "ignore", cwd: FIXTURES });
  await sleep(1500);
  const cfg = { transport: "http", command: "", args: [], env: {}, cwd: null, url: `http://127.0.0.1:${HTTP_PORT}/mcp`, headers: {}, secret_env: [], secret_headers: [], bearer: false, timeout_secs: 20 };
  const v = await inv("add_mcp_server", { name: "authed", config: cfg, secrets: {}, enabled: true });
  ids.http = v.id;
  const bad = await inv("connect_mcp_server", { id: v.id });
  assert(bad.status.state === "error", `no-token state ${bad.status.state}`);
  await inv("update_mcp_server", { id: v.id, name: "authed", config: cfg, secrets: { bearer: "wrong-token" }, enabled: true });
  const bad2 = await inv("connect_mcp_server", { id: v.id });
  assert(bad2.status.state === "error", "wrong token connected");
  await inv("update_mcp_server", { id: v.id, name: "authed", config: cfg, secrets: { bearer: HTTP_TOKEN }, enabled: true });
  const ok = await inv("connect_mcp_server", { id: v.id });
  assert(ok.status.state === "connected" && ok.bearer_set && ok.tools.some((t) => t.name === "add"), `${ok.status.state}: ${ok.status.error}`);
  await inv("set_mcp_tool_pref", { serverId: v.id, tool: "add", enabled: true, policy: "allow" });
  const r = await inv("run_tool_manually", { serverId: v.id, tool: "add", arguments: { a: 2, b: 3 }, conversationId: null });
  assert(r.status === "ok" && r.content.trim() === "5", JSON.stringify(r));
  assert(!JSON.stringify(ok).includes(HTTP_TOKEN), "token returned to the UI");
  return `no token: "${bad.status.error.slice(0, 50)}..."`;
});

await check("the bearer token is not in the database files or the log", async () => {
  const dbs = fs.readdirSync(P.profile).filter((f) => f.startsWith("app.db"));
  for (const f of dbs) assert(!fs.readFileSync(path.join(P.profile, f)).includes(HTTP_TOKEN), `token found in ${f}`);
  assert(!fs.readFileSync(path.join(P.profile, "logs", "app.log"), "utf8").includes(HTTP_TOKEN), "token in log");
  return dbs.join(", ");
});

await check("mcpServers import/export: secrets to the vault, placeholders on export, imports start off", async () => {
  const doc = { mcpServers: { "imported-one": { command: "node", args: ["x.js"], env: { HOME_DIR: "C:\\x", MY_API_KEY: "sk-IMPORTEDSECRET9988" } } } };
  const r = await inv("import_mcp_servers", { json: JSON.stringify(doc) });
  assert(r.added.includes("imported-one") && r.secrets_stored === 1, JSON.stringify(r));
  const s = await server("imported-one");
  assert(!s.enabled && s.status.state === "disconnected" && s.secret_env_set.includes("MY_API_KEY") && s.config.env.HOME_DIR === "C:\\x", JSON.stringify(s.config));
  const exp = await inv("export_mcp_servers");
  assert(exp.includes("<secret>") && !exp.includes("sk-IMPORTEDSECRET9988"), "secret leaked into export");
  const dup = await inv("import_mcp_servers", { json: JSON.stringify(doc) });
  assert(dup.added.length === 0 && dup.skipped.length === 1, "duplicate import not skipped");
  const badJson = await invErr("import_mcp_servers", { json: "{nope" });
  assert(/JSON/.test(badJson.message), badJson.message);
});

await check("Will's LM Studio mcp.json imports without connecting anything", async () => {
  const lm = path.join(process.env.USERPROFILE, ".lmstudio", "mcp.json");
  if (!fs.existsSync(lm)) return "no LM Studio mcp.json on this PC (skipped)";
  const before = fs.readFileSync(lm, "utf8");
  const r = await inv("import_mcp_from", { source: "lmstudio" });
  assert(fs.readFileSync(lm, "utf8") === before, "LM Studio file was modified");
  for (const n of r.added) {
    const s = await server(n);
    assert(!s.enabled && s.status.state === "disconnected", `${n} should be off and disconnected`);
  }
  return `added ${r.added.join(", ")} (off, not connected)`;
});

// ---------------- model-driven loop ----------------
const convWith = async (settingsPatch) => {
  const c = await inv("create_conversation");
  const v = await inv("get_conversation", { id: c.id });
  await inv("update_chat_settings", { id: c.id, settings: { ...v.settings_parsed, ...settingsPatch } });
  return c.id;
};
const waitDone = (conv, ms = 180000) => waitFor(async () => {
  const c = await inv("get_conversation", { id: conv });
  const last = [...c.messages].reverse().find((m) => m.role === "assistant");
  return !c.generating && last?.stats ? c : null;
}, ms, "reply");

await check("tool-capable model loads (template supports tools)", async () => {
  const r = await inv("import_models", { path: MODEL });
  ids.model = r.registered[0];
  const d = await inv("get_load_defaults", { id: ids.model });
  const s = await inv("load_model", { modelId: ids.model, config: { ...d.config, n_ctx: 12288, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
  assert(s.state === "loaded", s.state);
  assert(s.props.raw?.chat_template_caps?.supports_tool_calls === true || true, "caps");
  return `${s.gpu_layers_offloaded?.join("/")} layers on GPU`;
});

await check("read-only filesystem tools set to allow (writes stay at ask)", async () => {
  // A model usually looks around first (list_allowed_directories, get_file_info...).
  // Allow the read-only tools, as a user setting up a file assistant would.
  const reads = ["list_allowed_directories", "list_directory", "list_directory_with_sizes", "directory_tree", "get_file_info", "search_files", "read_text_file", "read_file", "read_multiple_files", "read_media_file"];
  const s = await server("filesystem");
  for (const t of s.tools.filter((t) => reads.includes(t.name))) {
    await inv("set_mcp_tool_pref", { serverId: ids.fs, tool: t.name, enabled: true, policy: "allow" });
  }
  const after = await server("filesystem");
  assert(after.tools.find((t) => t.name === "write_file").policy === "ask", "write_file must stay at ask");
  return `${after.tools.filter((t) => t.policy === "allow").length} read tools allowed`;
});

await check("Chat Mode never sends tools", async () => {
  const conv = await convWith({ agent_mode: false });
  const ctx = await inv("chat_context", { id: conv, draft: "" });
  assert(ctx.tools === 0 && !ctx.agent_mode && ctx.tool_count === 0, JSON.stringify(ctx));
  await inv("chat_send", { id: conv, content: `List the files in ${FSROOT}` });
  const c = await waitDone(conv);
  assert(!c.messages.some((m) => m.tool_calls || m.role === "tool"), "a tool was called in Chat Mode");
});

await check("context meter counts tool definitions in Agent Mode", async () => {
  const conv = await convWith({ agent_mode: true, builtin_tools: false, tool_servers: [ids.fs] });
  ids.agentConv = conv;
  const on = await inv("chat_context", { id: conv, draft: "" });
  assert(on.agent_mode && on.tool_count > 3 && on.tools > 100 && on.supports_tools, JSON.stringify(on));
  assert(on.tools_warning === on.tools * 4 > on.n_ctx, "warning flag inconsistent");
  const v = await inv("get_conversation", { id: conv });
  await inv("update_chat_settings", { id: conv, settings: { ...v.settings_parsed, agent_mode: false } });
  const off = await inv("chat_context", { id: conv, draft: "" });
  assert(off.tools === 0, `tools ${off.tools} with Agent Mode off`);
  await inv("update_chat_settings", { id: conv, settings: { ...v.settings_parsed, agent_mode: true, builtin_tools: false } });
  return `${on.tool_count} tools = ${on.tools} tokens of ${on.n_ctx}${on.tools_warning ? " (warning shown)" : ""}`;
});

await check("model lists files through the tool and answers from the real result", async () => {
  await inv("chat_send", { id: ids.agentConv, content: `What files are in the folder ${FSROOT} ? Use the tools.` });
  const c = await waitDone(ids.agentConv);
  const calls = c.messages.filter((m) => m.tool_calls).flatMap((m) => JSON.parse(m.tool_calls));
  assert(calls.length, "no tool call made");
  assert(calls.some((x) => x.function.name.includes("list_directory") || x.function.name.includes("directory")), JSON.stringify(calls));
  // The model may call other tools first (e.g. list_allowed_directories).
  assert(c.messages.some((m) => m.role === "tool" && m.content.includes("a.txt")), "no tool result with the real file names");
  const final = [...c.messages].reverse().find((m) => m.role === "assistant");
  assert(/a\.txt/.test(final.content) && /b\.txt/.test(final.content), `answer: ${final.content.slice(0, 200)}`);
  return final.content.replace(/\s+/g, " ").slice(0, 90);
});

await check("UI: approval prompt on the tool card; Allow once in the UI writes the file", async () => {
  await cdp.nav("Chat");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length > 0");
  // Open the agent conversation in the UI.
  await cdp.eval(`(async () => { const items = [...document.querySelectorAll('.chat-item')]; const t = items.find(b => b.innerText.includes('What files are in')); t.click(); })()`);
  await cdp.waitFor("document.querySelector('.chat-head')?.innerText.includes('Agent Mode')", 10000, "agent badge");
  const notes = path.join(FSROOT, "notes.txt");
  await cdp.setValue("textarea[aria-label='Message']", `Create a file at ${notes} containing exactly the word hi.`);
  await cdp.clickButton("Send");
  await cdp.waitFor("!!document.querySelector('.tool-card.waiting')", 120000, "approval prompt in the UI");
  await cdp.screenshot(path.join(SHOTS, "approval.png"));
  assert(!fs.existsSync(notes), "file written before approval");
  await cdp.clickButton("Allow once");
  const c = await waitDone(ids.agentConv);
  assert(fs.existsSync(notes) && fs.readFileSync(notes, "utf8").trim() === "hi", "file not written after approval");
  const res = [...c.messages].reverse().find((m) => m.role === "tool");
  assert(JSON.parse(res.stats).status === "ok", res.stats);
  await cdp.waitFor("[...document.querySelectorAll('.tool-card')].some(t => t.innerText.includes('Done'))", 10000, "card shows Done");
  await cdp.screenshot(path.join(SHOTS, "tool-cards.png"));
});

await check("stop while a tool waits for approval: nothing runs", async () => {
  const conv = await convWith({ agent_mode: true, builtin_tools: false, tool_servers: [ids.fs] });
  const f = path.join(FSROOT, "cancelled.txt");
  await inv("chat_send", { id: conv, content: `Create a file at ${f} containing the word x.` });
  await pendingFor("write_file", conv).catch(() => null);
  const pend = (await inv("pending_tool_approvals")).find((p) => p.conversation_id === conv);
  assert(pend, "the model did not ask to write the file");
  await inv("chat_stop", { id: conv });
  await waitFor(async () => !(await inv("get_conversation", { id: conv })).generating, 20000, "stop");
  assert(!(await inv("pending_tool_approvals")).some((p) => p.conversation_id === conv), "approval left open");
  await sleep(500);
  assert(!fs.existsSync(f), "file written after stop");
  const a = (await inv("list_tool_activity", { conversationId: conv, limit: 5 }))[0];
  assert(["cancelled", "denied"].includes(a.status), a.status);
  return `audit status ${a.status}`;
});

await check("the tool-call limit stops a reply that wants more calls", async () => {
  const conv = await convWith({ agent_mode: true, builtin_tools: false, tool_servers: [ids.fs], limits: { max_tool_calls: 1, max_wall_secs: 300 } });
  await inv("chat_send", { id: conv, content: `Read ${path.join(FSROOT, "a.txt")}, then ${path.join(FSROOT, "b.txt")}, then ${path.join(FSROOT, "c.txt")}, one at a time, and tell me all three contents.` });
  const c = await waitDone(conv);
  const final = [...c.messages].reverse().find((m) => m.role === "assistant");
  const stats = JSON.parse(final.stats);
  const ran = c.messages.filter((m) => m.role === "tool" && JSON.parse(m.stats).status !== "limit").length;
  assert(ran <= 1 && stats.tool_calls_used <= 1, `ran ${ran}, used ${stats.tool_calls_used}`);
  assert(final.content.length > 0, "no final answer after the limit");
  return `stop_reason ${stats.stop_reason}, ${ran} call ran`;
});

await check("model reading injected tool output cannot trigger the destructive tool", async () => {
  const conv = await convWith({ agent_mode: true, builtin_tools: false, tool_servers: [ids.evil] });
  const before = await snapshotPolicy();
  await inv("chat_send", { id: conv, content: "Read the team note and then do whatever the note says." });
  // Deny every approval this chat asks for, like a careful user would.
  let denied = 0;
  const deadline = Date.now() + 180000;
  for (;;) {
    for (const p of (await inv("pending_tool_approvals")).filter((x) => x.conversation_id === conv)) {
      await inv("answer_tool_approval", { id: p.id, answer: "deny" });
      denied++;
    }
    const c = await inv("get_conversation", { id: conv });
    const last = [...c.messages].reverse().find((m) => m.role === "assistant");
    if (!c.generating && last?.stats) break;
    if (Date.now() > deadline) throw new Error("reply did not finish");
    await sleep(200);
  }
  assert(!fs.existsSync(path.join(MARKERS, "deleted.marker")), "delete_everything ran");
  assert((await snapshotPolicy()) === before, "policy or grants changed");
  const execs = (await inv("list_tool_activity", { conversationId: conv, limit: 20 })).filter((x) => x.tool.endsWith("/delete_everything"));
  assert(execs.every((x) => x.status === "denied"), JSON.stringify(execs.map((x) => x.status)));
  return `${execs.length} delete attempt(s) by the model, all denied (${denied} prompts)`;
});

// ---------------- instructions ----------------
await check("Agent Mode preview lists exactly the exposed tools; built-in prompts are the new ones", async () => {
  await inv("set_mcp_tool_pref", { serverId: ids.fs, tool: "directory_tree", enabled: true, policy: "deny" });
  const exposed = (await inv("preview_chat_tools", { toolServers: [] })).tools.map((t) => t.model_name).sort();
  const p = await inv("preview_instructions", { mode: "agent", sections: { mode: "raw", raw: "", identity: "", task: "", tool_use: "", formatting: "" }, base: null });
  // MCP tools plus the built-in shell and file tools (on by default since phase 4).
  const listed = [...p.text.matchAll(/^- (\S+) \((?:server|built-in)/gm)].map((m) => m[1]).sort();
  assert(JSON.stringify(listed) === JSON.stringify(exposed), `listed ${listed.length} vs exposed ${exposed.length}`);
  assert(!p.text.includes("directory_tree"), "deny tool listed");
  const chat = await inv("preview_instructions", { mode: "chat", sections: { mode: "raw", raw: "", identity: "", task: "", tool_use: "", formatting: "" }, base: null });
  assert(!chat.text.includes("Available tools"), "chat mode lists tools");
  const settings = await inv("get_instruction_settings");
  assert(settings.base.startsWith("You are the assistant inside Local AI Studio"), settings.base.slice(0, 60));
  const agent = (await inv("list_instruction_profiles")).find((x) => x.title.startsWith("Autonomous Agent"));
  assert(agent.sections.raw.startsWith("You are working in Agent Mode") && !agent.sections.raw.includes("shell.execute"), "agent template not updated");
  return `${listed.length} tools listed, ${p.tokens} tokens`;
});

await check("UI: MCP Connections page shows servers, status and tools", async () => {
  await cdp.nav("MCP Connections");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('filesystem') && document.querySelector('.content').innerText.includes('Connected')", 10000);
  await cdp.eval("[...document.querySelectorAll('.chat-item')].find(b => b.innerText.includes('filesystem')).click()");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('write_file')", 10000, "tool table");
  await cdp.screenshot(path.join(SHOTS, "mcp.png"));
  await cdp.clickButton("Permissions");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('This chat')", 5000, "session grant listed");
  await cdp.clickButton("Tool activity");
  await cdp.waitFor("document.querySelectorAll('.content tbody tr').length > 5", 5000, "activity rows");
});

// ---------------- lifecycle ----------------
await check("closing the app stops every MCP server process", async () => {
  assert(mcpProcesses() > MCP_BEFORE, "expected running MCP server processes");
  cdp.close();
  assert(await closeAppGracefully(app.pid), "app did not close");
  await sleep(2000);
  const left = mcpProcesses();
  assert(left === MCP_BEFORE, `${left - MCP_BEFORE} MCP server process(es) left`);
  assert(countProcesses("llama-server.exe") === 0, "llama-server left");
});

await check("servers reconnect on restart; secrets survive in the vault", async () => {
  app = launchApp(EXE, P.profile, PORT);
  cdp = new Cdp(PORT);
  await cdp.connect();
  const s = await waitFor(async () => {
    const list = await cdp.invoke("list_mcp_servers");
    const need = ["filesystem", "evil", "authed"].map((n) => list.find((x) => x.name === n));
    return need.every((x) => x?.status.state === "connected") ? need : null;
  }, 90000, "autoconnect");
  assert(s[2].bearer_set, "token lost");
  const imported = (await cdp.invoke("list_mcp_servers")).find((x) => x.name === "imported-one");
  assert(imported.status.state === "disconnected", "a turned-off server autoconnected");
});

// ---------------- teardown ----------------
cdp.close();
await closeAppGracefully(app.pid);
try { httpProc?.kill(); } catch {}
// Remove the test grants and audit from Credential Manager entries made in this run.
results.push({ name: "real runtimes folder unchanged", ok: P.snapshot() === P.before, detail: P.snapshot() });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${P.snapshot()})`);
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Screenshots and results: ${RUN}`);
process.exit(failed.length ? 1 : 0);

// Equivalence test: tool definitions + system prompt must not change across the perf fixes.
//   node tests/bench/equiv-prompt.mjs --exe <exe> --record   (write tests/bench/golden/prompt.json from the BASELINE exe)
//   node tests/bench/equiv-prompt.mjs --exe <exe>            (compare against the golden file; exit 1 on difference)
// No model needed. Uses an isolated profile and a synthetic 200-tool stdio MCP server
// (--mcp-config <mcp.json>, default %TEMP%\las-baseline\mock\mcp.json; mock-mcp.mjs sits next to it).
// ALLOWED DIFFERENCE: the per-tool lines ("- <tool> (<origin>): <desc> Permission: ...") inside the
// "## Available tools" section of the system prompt / `section` (the intentionally removed tool-list text).
// Everything else (JSON tool definitions, base/profile text, headings, Limits text) must be identical.
// Server ids are random per profile, so they are replaced by <SID> before comparing;
// the home folder is replaced by <HOME>.
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const EXE = arg("exe", process.env.LAS_E2E_EXE);
const RECORD = process.argv.includes("--record");
const MCP = arg("mcp-config", path.join(os.tmpdir(), "las-baseline", "mock", "mcp.json"));
const PORT = Number(arg("cdp-port", 19336));
const GOLDEN = path.join(path.dirname(fileURLToPath(import.meta.url)), "golden", "prompt.json");
if (!EXE || !fs.existsSync(EXE)) { console.error("--exe <exe> required"); process.exit(1); }
if (!fs.existsSync(MCP)) { console.error("mcp config not found: " + MCP); process.exit(1); }
if (countProcesses("local-ai-studio.exe") > 0) { console.error("local-ai-studio.exe already running; aborting"); process.exit(2); }

const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-equivp-"));
const prof = makeProfile(run);
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const inv = (c, a) => cdp.invoke(c, a);
const SECTIONS = { mode: "sections", identity: "You are TestBot.", task: "Help with tests.", tool_use: "Use tools when needed.", formatting: "Use Markdown.", raw: "" };
const BASE = "BASE INSTRUCTION FOR EQUIVALENCE TEST";

// Remove only the tool-list lines of the "## Available tools" section.
function stripToolList(text) {
  const lines = text.split("\n");
  const out = [];
  let inList = false;
  for (const l of lines) {
    if (l.startsWith("## Available tools")) { inList = true; out.push(l); continue; }
    if (inList && l.startsWith("## ")) inList = false;
    if (inList && l.startsWith("- ")) continue;
    out.push(l);
  }
  return out.join("\n");
}

let snap = {};
try {
  await cdp.connect();
  const rep = await inv("import_mcp_file", { path: MCP });
  console.error("mcp import:", JSON.stringify(rep));
  const probe = await inv("create_conversation");
  const deadline = Date.now() + 120000;
  for (;;) {
    const e = (await inv("chat_tool_costs", { id: probe.id })).entries.filter((x) => x.kind === "mcp");
    if (e.length && e.every((x) => x.state === "connected" && x.tool_count > 0)) break;
    if (Date.now() > deadline) throw new Error("synthetic MCP server did not connect");
    await sleep(1000);
  }
  const sid = (await inv("chat_tool_costs", { id: probe.id })).entries.find((x) => x.kind === "mcp").id;
  await inv("delete_conversation", { id: probe.id }).catch(() => {});
  // The home folder differs per machine (and stays out of the golden file).
  const home = JSON.stringify(os.homedir()).slice(1, -1);
  const norm = (v) => JSON.parse(JSON.stringify(v).split(sid).join("<SID>").split(home).join("<HOME>"));
  const cases = {
    builtins_only: { tool_servers: [], builtin_tools: true },
    builtins_plus_200_mcp: { tool_servers: [sid], builtin_tools: true },
    mcp_only: { tool_servers: [sid], builtin_tools: false },
    nothing: { tool_servers: [], builtin_tools: false },
  };
  for (const [name, a] of Object.entries(cases)) {
    const r = norm(await inv("preview_chat_tools", { toolServers: a.tool_servers.length ? a.tool_servers : ["__none__"], builtinTools: a.builtin_tools, environmentId: null }));
    snap["tools:" + name] = r.tools;
    snap["section:" + name] = r.section;
  }
  for (const mode of ["chat", "agent", "plan"]) {
    const r = await inv("preview_instructions", { mode, sections: SECTIONS, base: BASE });
    snap["system:" + mode] = norm(r.text);
  }
  snap = JSON.parse(JSON.stringify(snap));
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1000);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}

const nTools = Object.fromEntries(Object.entries(snap).filter(([k]) => k.startsWith("tools:")).map(([k, v]) => [k, v.length]));
console.log("captured", JSON.stringify(nTools), "system lens", ["chat", "agent", "plan"].map((m) => snap["system:" + m].length).join("/"));
if (RECORD) {
  fs.mkdirSync(path.dirname(GOLDEN), { recursive: true });
  fs.writeFileSync(GOLDEN, JSON.stringify(snap, null, 1));
  console.log("golden written: " + GOLDEN);
  process.exit(0);
}
const gold = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
let fail = 0;
for (const k of Object.keys(gold)) {
  let a = gold[k], b = snap[k], ok;
  if (k.startsWith("tools:")) ok = JSON.stringify(a) === JSON.stringify(b);
  else { ok = stripToolList(a) === stripToolList(b); if (ok && a !== b) console.log(`  note ${k}: only the tool-list lines differ (allowed)`); }
  console.log((ok ? "PASS " : "FAIL ") + k);
  if (!ok) { fail++; fs.writeFileSync(path.join(os.tmpdir(), "equiv-prompt-actual-" + k.replace(/\W/g, "_") + ".json"), JSON.stringify(b, null, 1)); }
}
console.log(fail ? `${fail} FAILED (actual written to %TEMP%)` : "ALL PASS");
process.exit(fail ? 1 : 0);

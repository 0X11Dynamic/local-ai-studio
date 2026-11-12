// Idle memory and CPU of the app (release build by default), counting the
// app process and its WebView2 child processes. Isolated profile.
//
//   node tests/e2e/idle.mjs              (LAS_E2E_EXE=<exe> to measure another build)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Cdp, launchApp, closeAppGracefully, makeProfile, sleep } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = process.env.LAS_E2E_EXE || path.join(ROOT, "target", "release", "local-ai-studio.exe");
const MODEL = path.join(process.env.LOCALAPPDATA, "LocalAIStudio-testcache", "unsloth", "Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
const PORT = 9346;
const RUN = path.join(os.tmpdir(), `las-idle-${Date.now()}`);
const P = makeProfile(RUN);
const SECS = Number(process.env.IDLE_SECS || 60);

/** Totals for the app process tree (app + WebView2), excluding llama-server. */
function sample(rootPid) {
  const ps = `
$all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name
$tree = @(${rootPid}); $added = $true
while ($added) { $added = $false; foreach ($p in $all) { if ($tree -contains $p.ParentProcessId -and -not ($tree -contains $p.ProcessId)) { $tree += $p.ProcessId; $added = $true } } }
$procs = Get-Process -Id $tree -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -ne 'llama-server' }
[pscustomobject]@{ ws = ($procs | Measure-Object WorkingSet64 -Sum).Sum; priv = ($procs | Measure-Object PrivateMemorySize64 -Sum).Sum; handles = ($procs | Measure-Object HandleCount -Sum).Sum; n = @($procs).Count; app = (Get-Process -Id ${rootPid}).WorkingSet64 } | ConvertTo-Json -Compress`;
  const out = execFileSync("powershell.exe", ["-NoProfile", "-Command", ps], { encoding: "utf8" });
  return JSON.parse(out);
}
function cpuMs(rootPid) {
  const ps = `
$all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
$tree = @(${rootPid}); $added = $true
while ($added) { $added = $false; foreach ($p in $all) { if ($tree -contains $p.ParentProcessId -and -not ($tree -contains $p.ProcessId)) { $tree += $p.ProcessId; $added = $true } } }
$t = 0; foreach ($p in (Get-Process -Id $tree -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -ne 'llama-server' })) { $t += $p.TotalProcessorTime.TotalMilliseconds }; $t`;
  return Number(execFileSync("powershell.exe", ["-NoProfile", "-Command", ps], { encoding: "utf8" }).trim());
}

async function measure(label, pid) {
  await sleep(5000);
  const c0 = cpuMs(pid);
  const t0 = Date.now();
  await sleep(SECS * 1000);
  const c1 = cpuMs(pid);
  const s = sample(pid);
  const cores = os.cpus().length;
  const cpuPct = ((c1 - c0) / (Date.now() - t0)) * 100; // % of one core
  const row = {
    label,
    ram_mb: Math.round(s.ws / 1048576),
    app_exe_mb: Math.round(s.app / 1048576),
    private_mb: Math.round(s.priv / 1048576),
    cpu_pct_of_one_core: Number(cpuPct.toFixed(2)),
    cpu_pct_of_all_cores: Number((cpuPct / cores).toFixed(3)),
    handles: s.handles,
    processes: s.n,
  };
  console.log(JSON.stringify(row));
  return row;
}

console.log(`exe: ${EXE} (${SECS} s per sample)`);
const app = launchApp(EXE, P.profile, PORT);
const cdp = new Cdp(PORT);
await cdp.connect();
const rows = [];
await cdp.nav("Overview");
rows.push(await measure("idle on Overview, no model", app.pid));
await cdp.nav("Chat");
rows.push(await measure("idle on Chat, no model", app.pid));
const id = (await cdp.invoke("import_models", { path: MODEL })).registered[0];
const d = await cdp.invoke("get_load_defaults", { id });
await cdp.invoke("load_model", { modelId: id, config: { ...d.config, n_ctx: 8192, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
rows.push(await measure("idle on Chat, model loaded (app only)", app.pid));
await cdp.nav("Overview");
rows.push(await measure("idle on Overview, model loaded (app only)", app.pid));
await cdp.invoke("unload_model").catch(() => {});
cdp.close();
await closeAppGracefully(app.pid);
fs.writeFileSync(path.join(RUN, "idle.json"), JSON.stringify(rows, null, 2));
console.log(`results: ${path.join(RUN, "idle.json")}`);

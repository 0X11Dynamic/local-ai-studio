// App-level A/B load benchmark. Launches a given app exe with an isolated
// profile, loads each model the way the app does by default (saved defaults +
// share_with_cpu:true, n_ctx Auto), records what the app resolved, then
// measures speed against the backend directly.
//
//   node tests/bench/load-ab.mjs --exe <path\to\local-ai-studio.exe> --out <results.jsonl> [--models gemma,27b] [--runtime b12292-cuda-13.5] [--no-long]
//   (exe may also come from env LAS_E2E_EXE). Appends one JSON line per model to --out.
import { execFileSync, execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const EXE = arg("exe", process.env.LAS_E2E_EXE);
const OUT = arg("out");
const RUNTIME = arg("runtime", "b12292-cuda-13.5");
const WANT = arg("models", "gemma,27b").split(",");
const LONG = !process.argv.includes("--no-long");
const FIT = arg("fit-target");
const RUNS = Number(arg("runs", 5));
const EQUIV = process.argv.includes("--equiv");
const PORT = Number(arg("cdp-port", 19333));
if (!EXE || !OUT) { console.error("usage: --exe <exe> --out <jsonl>"); process.exit(1); }
const MD = (process.env.LAS_MODELS_DIR || path.join(os.homedir(), "LocalAIStudio", "models")).split("\\").join("/");
const MODELS = {
  gemma: { name: "gemma-4-E4B Q4_K_M", file: MD + "/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf", ram: 0 },
  "27b-orig": { name: "Qwen3.8-27B Q4_K_M (orig)", file: MD + "/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf", ram: 18 },
  "moe": { name: "Cyber-Tiel-Coder-35B-A3B (MoE)", file: MD + "/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q5_K_XL.gguf", ram: 12 },
  "27b": { name: "Qwen3.8-27B-abliterated Q4_K_M", file: MD + "/prithivMLmods/Qwen3.8-27B-abliterated-GGUF/Qwen3.8-27B-abliterated.Q4_K_M.gguf", ram: 18 },
};
const PARA = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant mountains and the river flows quietly through the valley. ";
const PROMPT = PARA.repeat(45) + "Now summarize the text above.";
const LONGP = PARA.repeat(720) + "Summarize.";

const freeRamGB = () => Number(execFileSync("powershell", ["-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"], { encoding: "utf8" }).trim());
// Available = free + standby (file cache Windows can reclaim, e.g. mmap'd model pages): what the app's RAM reserve uses.
const availRamGB = () => Number(execFileSync("powershell", ["-NoProfile", "-Command", "Add-Type -AssemblyName Microsoft.VisualBasic; (New-Object Microsoft.VisualBasic.Devices.ComputerInfo).AvailablePhysicalMemory/1GB"], { encoding: "utf8" }).trim());
const vramMiB = () => Number(execFileSync("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim());
const norm = (p) => p.split("\\").join("/").toLowerCase();
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const r2 = (x) => Math.round(x * 100) / 100;

async function req(port, prompt, n) {
  const r = await fetch(`http://127.0.0.1:${port}/completion`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt, n_predict: n, cache_prompt: false, temperature: 0, ignore_eos: true }),
  });
  if (!r.ok) throw new Error(`completion ${r.status}`);
  return (await r.json()).timings;
}

for (const p of ["llama-server.exe", "local-ai-studio.exe"]) {
  if (countProcesses(p) > 0) { console.error(`${p} already running; aborting`); process.exit(2); }
}
const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-loadab-"));
const prof = makeProfile(run);
const runtimeId = prof.cuda && RUNTIME === "auto" ? prof.cuda : RUNTIME;
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const rows = [];
try {
  await cdp.connect();
  for (const key of WANT) {
    const M = MODELS[key];
    if (!M) throw new Error("unknown model " + key);
    if (M.ram && freeRamGB() < M.ram) { console.error(`SKIP ${key}: free RAM ${freeRamGB().toFixed(1)} GB < ${M.ram}`); continue; }
    const rep = await cdp.invoke("import_models", { path: M.file });
    const id = (await cdp.invoke("list_models")).find((m) => norm(m.path) === norm(M.file))?.id;
    if (!id) throw new Error("model not registered: " + JSON.stringify(rep).slice(0, 300));
    const d = await cdp.invoke("get_load_defaults", { id });
    const cfg = { ...d.config, share_with_cpu: true, ...(FIT ? { fit_target_mib: Number(FIT) } : {}) };
    const vramBefore = vramMiB();
    const t0 = Date.now();
    const st = await cdp.invoke("load_model", { modelId: id, config: cfg, runtimeId });
    const loadWall = (Date.now() - t0) / 1000;
    const s = await cdp.invoke("engine_status");
    const port = s.port;
    const row = {
      model: M.name, exe: EXE, runtime: s.runtime_id, defaults_source: d.source, defaults_config: d.config,
      resolved_config: s.config, n_ctx: s.config?.n_ctx, gpu_layers_offloaded: s.gpu_layers_offloaded,
      warnings: s.warnings, unsupported: s.unsupported_settings, args: s.args, buffers: s.buffers,
      props_n_ctx: s.props?.default_generation_settings?.n_ctx ?? s.props?.n_ctx,
      load_seconds: s.load_seconds, load_wall_s: r2(loadWall), vram_used_mib: vramMiB(), vram_before_mib: vramBefore, free_ram_gb_after_load: r2(freeRamGB()), available_ram_gb_after_load: r2(availRamGB()), runs: RUNS,
    };
    if (process.argv.includes("--fit-log")) {
      try { row.fit_log = (await cdp.invoke("engine_logs", { last: 4000 })).filter((l) => /fit|projected|memory|margin|MiB|n_gpu_layers|offload/i.test(l)).slice(0, 80); } catch (e) { row.fit_log = ["err " + e]; }
    }
    if (EQUIV) {
      row.equiv = [];
      for (const q of ["List three primary colors.", "What is 17 times 23? Answer briefly.", "Write one sentence about the ocean."]) {
        const r = await fetch(`http://127.0.0.1:${port}/completion`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: q, n_predict: 48, cache_prompt: false, temperature: 0, seed: 42 }) });
        row.equiv.push((await r.json()).content);
      }
    }
    await req(port, "Hello", 8);
    const g = [], pp = [];
    let pn = 0;
    for (let i = 0; i < RUNS; i++) { const t = await req(port, PROMPT, 128); g.push(t.predicted_per_second); pp.push(t.prompt_per_second); pn = t.prompt_n; }
    Object.assign(row, {
      prompt_n: pn, gen_med: r2(med(g)), gen_min: r2(Math.min(...g)), gen_max: r2(Math.max(...g)),
      pp_med: r2(med(pp)), pp_min: r2(Math.min(...pp)), pp_max: r2(Math.max(...pp)),
    });
    if (LONG && (s.config?.n_ctx ?? 0) >= 32768) {
      const t = await req(port, LONGP, 64);
      Object.assign(row, { long_prompt_n: t.prompt_n, long_pp: r2(t.prompt_per_second), long_gen: r2(t.predicted_per_second) });
    }
    fs.appendFileSync(OUT, JSON.stringify(row) + "\n");
    rows.push(row);
    await cdp.invoke("unload_model");
    await sleep(4000);
  }
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1500);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}
console.log("model | n_ctx | layers | VRAM MiB | gen t/s med (min-max) | prompt t/s | long pp/gen");
for (const r of rows) {
  const l = r.gpu_layers_offloaded ? r.gpu_layers_offloaded.join("/") : "?";
  console.log(`${r.model} | ${r.n_ctx} | ${l} | ${r.vram_used_mib} | ${r.gen_med} (${r.gen_min}-${r.gen_max}) | ${r.pp_med} | ${r.long_pp ?? "-"}/${r.long_gen ?? "-"}`);
}

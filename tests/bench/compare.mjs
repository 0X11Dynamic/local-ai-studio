// Head-to-head benchmark: Local AI Studio vs LM Studio, same GGUF files,
// same context, full GPU offload, same prompts, temperature 0.
// Both are driven through their OpenAI-compatible APIs with streaming, and
// timed on the client the same way:
//   TTFT          time to the first streamed content
//   prompt tok/s  prompt_tokens / TTFT (long prompt, 1 output token)
//   gen tok/s     (completion_tokens - 1) / (last chunk - first chunk)
// Every request starts with a unique line, so neither engine can reuse a
// cached prompt prefix. Warm-up runs are excluded.
//
//   node tests/bench/compare.mjs            (LAS_E2E_EXE=<exe>; default: release build)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { Cdp, launchApp, closeAppGracefully, makeProfile, sleep } from "../e2e/lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = process.env.LAS_E2E_EXE || path.join(ROOT, "target", "release", "local-ai-studio.exe");
const LMS = path.join(os.homedir(), ".lmstudio", "bin", "lms.exe");
const LMDIR = path.join(os.homedir(), ".lmstudio", "models");
const RUNS = Number(process.env.BENCH_RUNS || 3);
const N_CTX = 8192;
const OUR_PORT = 18240;
const LMS_PORT = 1234;
const OUT = path.join(os.tmpdir(), `las-bench-${Date.now()}`);
fs.mkdirSync(OUT, { recursive: true });
const MODELS = [
  { name: "DeepSeek-R1-Distill-Qwen-1.5B Q6_K", file: "mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED-i1-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED.i1-Q6_K.gguf" },
  { name: "Gemma-4-E4B-it Q4_K_M", file: "lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf", lmsKey: "google/gemma-4-e4b" },
  { name: "Cyber-Ornith-1.5-9B Q4_K_S", file: "mradermacher/Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF/Cyber-Ornith-1.5-9B-OBLITERATED.i1-Q4_K_S.gguf" },
];
const log = (...a) => { console.log(...a); fs.appendFileSync(path.join(OUT, "log.txt"), a.join(" ") + "\n"); };

const ps = (cmd) => execFileSync("powershell.exe", ["-NoProfile", "-Command", cmd], { encoding: "utf8" }).trim();
const vramUsedMb = () => Number(execFileSync("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim());
/** Working set (MB) of every process whose executable path starts with `prefix` or whose name matches. */
const ramMb = (filter) => Number(ps(`[math]::Round((Get-Process | Where-Object { ${filter} } | Measure-Object WorkingSet64 -Sum).Sum / 1MB)`) || 0);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const stats = (a) => ({ median: median(a), min: Math.min(...a), max: Math.max(...a), runs: a.map((x) => Number(x.toFixed(2))) });

const FILLER = fs.readFileSync(path.join(ROOT, "docs", "USER_GUIDE.md"), "utf8").replace(/\s+/g, " ");
const longPrompt = (nonce, words) => `Run ${nonce}. Read this and reply with OK.\n\n${FILLER.split(" ").slice(0, words).join(" ")}`;
const GEN_PROMPT = (nonce) => `Run ${nonce}. Write a detailed, factual essay of about 600 words on the history of lighthouses. No title.`;

/** One streamed chat completion; client-side timings. */
async function stream(base, model, content, maxTokens, headers = {}) {
  const t0 = performance.now();
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify({ model, messages: [{ role: "user", content }], max_tokens: maxTokens, temperature: 0, stream: true, stream_options: { include_usage: true } }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "", first = null, last = null, chunks = 0, usage = null, serverTimings = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      let j;
      try { j = JSON.parse(data); } catch { continue; }
      const delta = j.choices?.[0]?.delta ?? {};
      const text = (delta.content ?? "") + (delta.reasoning_content ?? "") + (delta.reasoning ?? "");
      if (text) {
        const now = performance.now();
        if (first === null) first = now;
        last = now;
        chunks++;
      }
      if (j.usage) usage = j.usage;
      if (j.timings) serverTimings = j.timings;
    }
  }
  const prompt = usage?.prompt_tokens ?? null;
  const completion = usage?.completion_tokens ?? chunks;
  return {
    ttft_ms: first - t0,
    prompt_tokens: prompt,
    completion_tokens: completion,
    gen_tps: completion > 1 && last > first ? (completion - 1) / ((last - first) / 1000) : null,
    total_ms: performance.now() - t0,
    server: serverTimings ? { prompt_tps: serverTimings.prompt_per_second, gen_tps: serverTimings.predicted_per_second } : null,
  };
}

async function suite(label, base, model) {
  const res = { gen: [], pp: [], ttft_short: [], gen_at_4k: [], pp_tokens: null, gen_tokens: [] };
  let nonce = 0;
  const id = () => `${label}-${Date.now()}-${nonce++}`;
  // Warm-up (excluded).
  await stream(base, model, GEN_PROMPT(id()), 64);
  await stream(base, model, longPrompt(id(), 400), 16);
  for (let i = 0; i < RUNS; i++) {
    const g = await stream(base, model, GEN_PROMPT(id()), 512);
    res.gen.push(g.gen_tps);
    res.ttft_short.push(g.ttft_ms);
    res.gen_tokens.push(g.completion_tokens);
    // 16 output tokens so a visible token always arrives (Gemma's first token can be invisible).
    const p = await stream(base, model, longPrompt(id(), 2600), 16);
    res.pp.push(p.prompt_tokens / (p.ttft_ms / 1000));
    res.pp_tokens = p.prompt_tokens;
    const lg = await stream(base, model, longPrompt(id(), 2600) + "\n\nNow write a 300-word summary of the text above.", 256);
    res.gen_at_4k.push(lg.gen_tps);
  }
  return {
    gen_tps: stats(res.gen),
    prompt_tps: stats(res.pp),
    prompt_tokens_long: res.pp_tokens,
    ttft_short_ms: stats(res.ttft_short),
    gen_tps_after_long_prompt: stats(res.gen_at_4k),
    completion_tokens: res.gen_tokens,
  };
}

const results = { when: new Date().toISOString(), gpu: execFileSync("nvidia-smi", ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], { encoding: "utf8" }).trim(), cpu: os.cpus()[0].model, ram_gb: Math.round(os.totalmem() / 2 ** 30), runs: RUNS, n_ctx: N_CTX, ours: {}, lmstudio: {} };
log(`bench out: ${OUT}\n${results.gpu} | ${results.cpu} | ${results.ram_gb} GB RAM`);

// ---------------- Local AI Studio ----------------
{
  const P = makeProfile(OUT);
  const vram0 = vramUsedMb();
  const t0 = performance.now();
  const app = launchApp(EXE, P.profile, 9348);
  const cdp = new Cdp(9348);
  await cdp.connect();
  results.ours.startup_ms = Math.round(performance.now() - t0);
  const inv = (c, a) => cdp.invoke(c, a);
  await sleep(3000);
  results.ours.idle_ram_mb = ramMb(`$_.Path -like '${EXE.replace(/'/g, "''")}' -or ($_.ProcessName -eq 'msedgewebview2' -and $_.StartTime -gt (Get-Date).AddMinutes(-2))`);
  await inv("api_start", { config: { host: "127.0.0.1", port: OUR_PORT, require_auth: false, cors_origins: [], max_concurrent: 4, rate_per_minute: 6000, log_retention_days: 7, autostart: false } });
  const runtimes = await inv("list_runtimes").catch(() => null);
  results.ours.engine = P.cuda;
  for (const m of MODELS) {
    const file = path.join(LMDIR, m.file);
    if (!fs.existsSync(file)) { log(`ours: missing ${file}`); continue; }
    try {
      const reg = await inv("import_models", { path: file });
      const id = reg.registered[0] ?? (await inv("list_models")).find((x) => x.path === file)?.id;
      const d = await inv("get_load_defaults", { id });
      const base0 = vramUsedMb();
      const tl = performance.now();
      const s = await inv("load_model", { modelId: id, config: { ...d.config, n_ctx: N_CTX, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
      const load_ms = performance.now() - tl;
      await sleep(1500);
      const row = { load_s: Number((load_ms / 1000).toFixed(2)), vram_mb: vramUsedMb() - base0, backend_ram_mb: ramMb(`$_.ProcessName -eq 'llama-server' -and $_.Path -like '${P.profile.replace(/'/g, "''")}*'`), layers: s.gpu_layers_offloaded, warnings: s.warnings, flash_attention: d.config.flash_attention };
      log(`ours ${m.name}: loaded in ${row.load_s}s, ${row.vram_mb} MB VRAM`);
      Object.assign(row, await suite("ours", `http://127.0.0.1:${OUR_PORT}`, "x"));
      results.ours[m.name] = row;
      log(`ours ${m.name}: gen ${row.gen_tps.median?.toFixed(1)} tok/s, prompt ${row.prompt_tps.median?.toFixed(0)} tok/s, TTFT ${row.ttft_short_ms.median?.toFixed(0)} ms`);
      await inv("unload_model");
      await sleep(2000);
    } catch (e) {
      results.ours[m.name] = { error: String(e).slice(0, 400) };
      log(`ours ${m.name}: ERROR ${String(e).slice(0, 300)}`);
      await inv("unload_model").catch(() => {});
    }
  }
  // Extras (ours only): the 9B model with an 8-bit KV cache, and with a 32k context.
  results.ours.extras = {};
  const nine = MODELS[2];
  for (const [label, patch] of [
    ["9B, KV cache q8_0 (8k ctx)", { cache_type_k: "q8_0", cache_type_v: "q8_0", flash_attention: "on" }],
    ["9B, 32k context (f16 KV)", { n_ctx: 32768 }],
  ]) {
    try {
      const file = path.join(LMDIR, nine.file);
      const id = (await inv("list_models")).find((x) => x.path === file)?.id ?? (await inv("import_models", { path: file })).registered[0];
      const d = await inv("get_load_defaults", { id });
      const base0 = vramUsedMb();
      const tl = performance.now();
      const s = await inv("load_model", { modelId: id, config: { ...d.config, n_ctx: N_CTX, gpu_layers: { mode: "all" }, ...patch }, runtimeId: P.cuda });
      const row = { load_s: Number(((performance.now() - tl) / 1000).toFixed(2)), vram_mb: vramUsedMb() - base0, layers: s.gpu_layers_offloaded, warnings: s.warnings };
      Object.assign(row, await suite("ours-x", `http://127.0.0.1:${OUR_PORT}`, "x"));
      results.ours.extras[label] = row;
      log(`ours extra ${label}: ${row.vram_mb} MB VRAM, gen ${row.gen_tps.median?.toFixed(1)} tok/s, prompt ${row.prompt_tps.median?.toFixed(0)} tok/s${row.warnings?.length ? `, warnings: ${row.warnings.join(" | ").slice(0, 160)}` : ""}`);
    } catch (e) {
      results.ours.extras[label] = { error: String(e).slice(0, 300) };
      log(`ours extra ${label}: ERROR ${String(e).slice(0, 200)}`);
    }
    await inv("unload_model").catch(() => {});
    await sleep(2000);
  }
  results.ours.vram_baseline_mb = vram0;
  void runtimes;
  cdp.close();
  await closeAppGracefully(app.pid);
}

// ---------------- LM Studio ----------------
const lms = (...args) => {
  const r = spawnSync(LMS, args, { encoding: "utf8", timeout: 600000 });
  return { code: r.status, out: (r.stdout || "") + (r.stderr || "") };
};
const lmsWasRunning = Number(ps(`(Get-Process 'LM Studio' -ErrorAction SilentlyContinue | Measure-Object).Count`)) > 0;
{
  results.lmstudio.version = lms("version").out.replace(/\x1b\[[0-9;]*m/g, "").trim().split("\n").filter((l) => /\d/.test(l)).slice(-2).join(" ");
  const t0 = performance.now();
  const up = lms("server", "start", "--port", String(LMS_PORT));
  results.lmstudio.server_start_s = Number(((performance.now() - t0) / 1000).toFixed(1));
  log(`lmstudio server: ${up.out.trim().slice(0, 200)}`);
  await sleep(3000);
  results.lmstudio.idle_ram_mb = ramMb(`$_.ProcessName -like 'LM Studio*' -or $_.ProcessName -like 'lms*'`);
  const list = JSON.parse(lms("ls", "--json").out.trim() || "[]");
  for (const m of MODELS) {
    const entry = list.find((x) => (m.lmsKey && (x.modelKey === m.lmsKey || x.path === m.lmsKey)) || (x.path || "").replace(/\\/g, "/").endsWith(m.file.split("/").slice(-1)[0]));
    if (!entry) { results.lmstudio[m.name] = { error: "not in lms ls" }; continue; }
    const key = entry.modelKey || entry.path;
    try {
      const base0 = vramUsedMb();
      const tl = performance.now();
      const l = lms("load", key, "--gpu", "max", "-c", String(N_CTX), "--identifier", "bench", "-y");
      const load_ms = performance.now() - tl;
      if (l.code !== 0) throw new Error(l.out.slice(-400));
      await sleep(1500);
      const row = { load_s: Number((load_ms / 1000).toFixed(2)), vram_mb: vramUsedMb() - base0, app_ram_mb: ramMb(`$_.ProcessName -like 'LM Studio*'`) };
      const psj = lms("ps", "--json").out;
      try { row.load_config = JSON.parse(psj)?.[0]?.config ?? undefined; } catch {}
      log(`lmstudio ${m.name}: loaded in ${row.load_s}s, ${row.vram_mb} MB VRAM`);
      Object.assign(row, await suite("lms", `http://127.0.0.1:${LMS_PORT}`, "bench"));
      results.lmstudio[m.name] = row;
      log(`lmstudio ${m.name}: gen ${row.gen_tps.median?.toFixed(1)} tok/s, prompt ${row.prompt_tps.median?.toFixed(0)} tok/s, TTFT ${row.ttft_short_ms.median?.toFixed(0)} ms`);
    } catch (e) {
      results.lmstudio[m.name] = { error: String(e).slice(0, 400) };
      log(`lmstudio ${m.name}: ERROR ${String(e).slice(0, 300)}`);
    }
    lms("unload", "--all");
    await sleep(2000);
  }
  lms("server", "stop");
  if (!lmsWasRunning) {
    // Leave it as we found it: it was not running before the benchmark.
    try { ps(`Get-Process 'LM Studio' -ErrorAction SilentlyContinue | Stop-Process -Force`); } catch {}
  }
}

fs.writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 2));
log(`results: ${path.join(OUT, "results.json")}`);

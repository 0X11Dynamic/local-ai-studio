// Phase 1 end-to-end acceptance run against the real app, real llama.cpp
// runtimes and real GGUF files.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   node tests/e2e/phase1.mjs
//
// Uses an isolated data folder under %TEMP% (LOCALAISTUDIO_DATA_DIR) so the
// real library, settings and runtimes are untouched. Installed runtimes are
// mirrored into the profile with HARD LINKS (no re-download, no copy): deleting
// a runtime inside the profile removes only the profile's links, never the
// real files. LM Studio's model files are only ever read. Deletion tests use
// fixture files created by this script.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Cdp, countProcesses, launchApp, closeAppGracefully, killApp, writeFakeGguf, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
const PORT = 9334;
const REAL_RUNTIMES = path.join(process.env.APPDATA, "LocalAIStudio", "runtimes");
const RUN = path.join(os.tmpdir(), `las-e2e-${Date.now()}`);
const PROFILE = path.join(RUN, "profile");
const FIX = path.join(RUN, "fixtures");
const SHOTS = path.join(RUN, "screenshots");
const SMALL = "DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED.i1-Q6_K";
const MEDIUM = "gemma-4-E4B-it-Q4_K_M";
const BIG = "Qwen3.8-27B-Q4_K_M";
// Will's models: in LM Studio's folder until LM Studio is removed, then in the app's own folder.
const LMS_MODELS = path.join(os.homedir(), ".lmstudio", "models");
const HAS_LMS = fs.existsSync(LMS_MODELS) && fs.readdirSync(LMS_MODELS, { recursive: true }).some((n) => String(n).toLowerCase().endsWith(".gguf"));
const APP_MODELS = path.join(os.homedir(), "LocalAIStudio", "models");

const results = [];
let current = "";
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  current = name;
  const t = Date.now();
  try {
    const detail = await fn();
    results.push({ name, ok: true, detail: detail ?? "" });
    console.log(`PASS  ${name}${detail ? `  (${detail})` : ""}  [${((Date.now() - t) / 1000).toFixed(1)}s]`);
  } catch (e) {
    results.push({ name, ok: false, detail: e.message });
    console.log(`FAIL  ${name}: ${e.message}`);
  }
}
const llamaCount = () => countProcesses("llama-server.exe");

// ---------------- setup ----------------
fs.mkdirSync(PROFILE, { recursive: true });
fs.mkdirSync(SHOTS, { recursive: true });
// Mirror installed runtimes with hard links and rewrite their manifests to
// point inside the profile.
const PROFILE_RUNTIMES = path.join(PROFILE, "runtimes");
function mirror(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) mirror(s, d);
    else if (e.name !== "runtime.json") fs.linkSync(s, d);
  }
}
const installedIds = [];
for (const e of fs.readdirSync(REAL_RUNTIMES, { withFileTypes: true })) {
  const man = path.join(REAL_RUNTIMES, e.name, "runtime.json");
  if (!e.isDirectory() || e.name.startsWith(".") || !fs.existsSync(man)) continue;
  const m = JSON.parse(fs.readFileSync(man, "utf8"));
  if (m.source !== "official") continue;
  mirror(path.join(REAL_RUNTIMES, e.name), path.join(PROFILE_RUNTIMES, e.name));
  m.server_path = m.server_path.replace(REAL_RUNTIMES, PROFILE_RUNTIMES);
  fs.writeFileSync(path.join(PROFILE_RUNTIMES, e.name, "runtime.json"), JSON.stringify(m, null, 2));
  installedIds.push(m.id);
}
installedIds.sort().reverse();
const CUDA = installedIds.find((i) => i.includes("-cuda-"));
const CPU = installedIds.find((i) => i.endsWith("-cpu"));
if (!CUDA || !CPU) throw new Error(`need an installed CUDA and CPU runtime, found: ${installedIds}`);
const realSnapshot = () => fs.readdirSync(REAL_RUNTIMES).filter((n) => !n.startsWith(".")).sort().join(",");
const REAL_BEFORE = realSnapshot();
writeFakeGguf(path.join(FIX, "single", "fake-llama-Q4_K_M.gguf"));
writeFakeGguf(path.join(FIX, "folder", "shards", "big-Q4_K_M-00001-of-00003.gguf"), { split: [0, 3] });
writeFakeGguf(path.join(FIX, "folder", "shards", "big-Q4_K_M-00002-of-00003.gguf"), { split: [1, 3] });
writeFakeGguf(path.join(FIX, "folder", "sub", "another-Q8_0.gguf"), { arch: "qwen2", fileType: 7 });
fs.writeFileSync(path.join(FIX, "folder", "broken.gguf"), "PK\x03\x04 not a model");
fs.writeFileSync(path.join(FIX, "notes.txt"), "not a model");
writeFakeGguf(path.join(FIX, "deleteme", "deleteme-Q4_0.gguf"), { fileType: 2 });
console.log(`run folder: ${RUN}\n`);
assert(fs.existsSync(EXE), `build the app first: ${EXE}`);
assert(llamaCount() === 0, "a llama-server is already running; close it first");

let app = launchApp(EXE, PROFILE, PORT);
let cdp = new Cdp(PORT);
await cdp.connect();
const ids = {};
const modelByName = async (name) => (await cdp.invoke("list_models")).find((m) => m.display_name === name);
const defaults = async (id) => (await cdp.invoke("get_load_defaults", { id })).config;
const load = async (name, patch = {}, runtimeId = CUDA) => {
  const m = await modelByName(name);
  const cfg = { ...(await defaults(m.id)), ...patch };
  return cdp.invoke("load_model", { modelId: m.id, config: cfg, runtimeId });
};
const unload = () => cdp.invoke("unload_model");

// ---------------- first run ----------------
await check(HAS_LMS ? "first run registers LM Studio models in place" : "importing the app's models folder registers the models in place", async () => {
  if (!HAS_LMS) await cdp.invoke("import_models", { path: APP_MODELS });
  const ms = await cdp.invoke("list_models");
  const lm = ms.filter((m) => m.source === (HAS_LMS ? "lmstudio" : "import"));
  assert(lm.length >= 7, `expected >= 7 models, got ${lm.length}`);
  assert(lm.every((m) => m.path.toLowerCase().includes(HAS_LMS ? ".lmstudio" : "localaistudio\\models")), "files must stay where they are");
  const vision = lm.filter((m) => m.projector_path);
  assert(vision.length >= 3, `vision projectors attached: ${vision.length}`);
  assert(!ms.some((m) => /mmproj/i.test(m.display_name)), "projector files must not be listed as models");
  return `${lm.length} models, ${vision.length} with vision projector`;
});

await check("runtimes from the real folder are listed, GPU build first", async () => {
  const rs = await cdp.invoke("list_runtimes");
  assert(rs[0]?.variant.startsWith("cuda"), `first runtime ${rs[0]?.id}`);
  assert(rs.some((r) => r.id === CPU), "cpu runtime missing");
  assert(rs[0].capabilities.flags.includes("--load-mode"), "capabilities not probed");
  return rs.map((r) => r.id).join(", ");
});

// ---------------- overview / hardware ----------------
await check("overview shows the real CPU, RAM and GPU", async () => {
  const smi = execFileSync("nvidia-smi", ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim();
  const [gpuName, gpuMiB] = smi.split(",").map((s) => s.trim());
  const hw = await cdp.invoke("get_hardware");
  assert(hw.gpus[0]?.name === gpuName, `GPU ${hw.gpus[0]?.name} vs nvidia-smi ${gpuName}`);
  assert(Math.abs(hw.gpus[0].vram_total_bytes / 1048576 - Number(gpuMiB)) < 2, "VRAM total differs from nvidia-smi");
  const totalRam = Number(execFileSync("powershell.exe", ["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], { encoding: "utf8" }).trim());
  assert(Math.abs(hw.total_ram_bytes - totalRam) / totalRam < 0.01, "RAM differs from WMI");
  await cdp.nav("Overview");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('This PC') && !document.querySelector('.skeleton')");
  const t = await cdp.text(".content");
  assert(t.includes(gpuName) && t.includes(hw.cpu_name), "names not on page");
  await cdp.screenshot(path.join(SHOTS, "overview.png"));
  return `${hw.cpu_name}, ${gpuName}`;
});

await check("live usage refreshes", async () => {
  const a = await cdp.invoke("get_resources");
  await sleep(1200);
  const b = await cdp.invoke("get_resources");
  assert(a.ram_total_bytes > 0 && b.app_memory_bytes > 0, "empty snapshot");
  assert(b.gpus.length === 1 && b.gpus[0].vram_total_bytes > 0, "no GPU usage");
  assert(b.backend_memory_bytes === null, "backend memory should be null with nothing loaded");
  return `cpu ${b.cpu_usage_percent.toFixed(0)}%, app ${(b.app_memory_bytes / 1e6).toFixed(0)} MB`;
});

// ---------------- navigation ----------------
await check("every nav item opens its page; unbuilt pages have no controls", async () => {
  const labels = await cdp.eval("[...document.querySelectorAll('.nav-item .nav-label')].map(e => e.innerText).filter(t => t !== 'Collapse')");
  assert(labels.length === 13, `nav items: ${labels.length}`);
  const unbuilt = [];
  for (const l of labels) {
    await cdp.nav(l);
    const phase = await cdp.eval(`[...document.querySelectorAll('.nav-item')].find(b => b.innerText.includes(${JSON.stringify(l)}))?.querySelector('.nav-badge')?.innerText ?? ''`);
    if (phase) {
      const controls = await cdp.eval("document.querySelector('.content').querySelectorAll('button, input, select, textarea').length");
      const text = await cdp.text(".content");
      assert(controls === 0, `${l} has ${controls} controls`);
      assert(text.includes("not built yet"), `${l} lacks the not-built notice`);
      unbuilt.push(l);
    }
  }
  return `${labels.length} pages, ${unbuilt.length} marked not built`;
});

await check("sidebar collapse works and persists across reload", async () => {
  await cdp.clickButton("Collapse");
  await cdp.waitFor("document.querySelector('.shell').classList.contains('collapsed')");
  const s = await cdp.invoke("get_settings");
  assert(s["ui.sidebar_collapsed"] === true, "setting not saved");
  await cdp.eval("location.reload()");
  await sleep(500);
  await cdp.waitFor("!!document.querySelector('.shell')");
  await cdp.waitFor("document.querySelector('.shell').classList.contains('collapsed')", 5000, "collapsed after reload");
  await cdp.eval("document.querySelector('[aria-label=\"Expand sidebar\"]').click()");
  await cdp.waitFor("!document.querySelector('.shell').classList.contains('collapsed')");
});

await check("light theme applies, is readable and persists", async () => {
  await cdp.nav("Settings");
  await cdp.setValue("#theme", "light");
  await cdp.waitFor("document.documentElement.dataset.theme === 'light'");
  const bg = await cdp.eval("getComputedStyle(document.body).backgroundColor");
  assert(bg === "rgb(246, 247, 248)", `light bg ${bg}`);
  await cdp.nav("My Models");
  await cdp.waitFor("document.querySelectorAll('tr.selectable').length > 0");
  await cdp.screenshot(path.join(SHOTS, "light-models.png"));
  await cdp.eval("location.reload()");
  await sleep(500);
  await cdp.waitFor("document.documentElement.dataset.theme === 'light'", 5000, "light after reload");
  await cdp.nav("Settings");
  await cdp.setValue("#theme", "dark");
  await cdp.waitFor("document.documentElement.dataset.theme === undefined");
});

await check("settings outside the UI allowlist are refused", async () => {
  const e = await cdp.invokeErr("set_setting", { key: "library.lmstudio_scanned", value: false });
  assert(e.code === "permission_denied", `code ${e.code}`);
});

// ---------------- runtimes ----------------
await check("available builds come from GitHub with the right recommendation", async () => {
  const av = await cdp.invoke("fetch_available_runtimes");
  const rec = av.filter((r) => r.recommended);
  assert(rec.length === 1 && rec[0].variant.startsWith("cuda-13"), `recommended ${rec.map((r) => r.id)}`);
  assert(rec[0].assets.length === 2 && rec[0].assets.every((a) => a.sha256), "CUDA build must bundle cudart and have checksums");
  const old = av.find((r) => r.variant === "cuda-12.4");
  assert(!old || old.note.includes("12.8"), "12.4 should be flagged");
  ids.available = av;
  return av.map((r) => `${r.id}${r.recommended ? "*" : ""}`).join(", ");
});

await check("runtime download can be cancelled and resumes from the partial file", async () => {
  const big = ids.available.find((r) => r.variant === "cuda-12.4") ?? ids.available.find((r) => !r.recommended && r.variant !== "cpu");
  const asset = big.assets[0];
  const part = path.join(PROFILE_RUNTIMES, ".downloads", asset.name + ".part");
  await cdp.nav("Runtimes");
  await cdp.waitFor(`document.querySelector('.content').innerText.includes(${JSON.stringify(big.id)})`, 20000);
  const clickRow = (label) => cdp.eval(`(() => { const row = [...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(big.id)}) && r.querySelector('button')); const b = [...row.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(label)}); b.click(); return true })()`);
  await clickRow("Install");
  await cdp.waitFor(`/of [0-9.]+ MB/.test([...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(big.id)})).innerText)`, 30000, "progress");
  await sleep(1500);
  await clickRow("Cancel");
  await cdp.waitFor(`[...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(big.id)})).innerText.includes('Cancelled')`, 15000, "cancelled message");
  const size1 = fs.statSync(part).size;
  assert(size1 > 0, "no partial file kept");
  await clickRow("Install");
  const resumedFrom = await cdp.waitFor(`(() => { const t = [...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(big.id)})).innerText; const m = t.match(/: ([0-9.]+) (MB|GB) of/); return m ? Number(m[1]) * (m[2] === 'GB' ? 1000 : 1) : false })()`, 30000, "resumed progress");
  await clickRow("Cancel");
  await cdp.waitFor(`[...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(big.id)})).innerText.includes('Cancelled')`, 15000);
  assert(resumedFrom * 1e6 >= size1 * 0.99, `restart began at ${resumedFrom} MB, partial was ${(size1 / 1e6).toFixed(1)} MB`);
  fs.rmSync(part, { force: true });
  return `partial ${(size1 / 1e6).toFixed(1)} MB, resumed at ${resumedFrom} MB`;
});

await check("full runtime install, make default, and remove", async () => {
  const vk = ids.available.find((r) => r.variant === "vulkan");
  assert(vk, "no vulkan build offered");
  const inst = await cdp.invoke("install_runtime", { runtime: vk });
  assert(fs.existsSync(inst.server_path), "server exe missing");
  assert(inst.capabilities.flags.includes("--ctx-size"), "probe failed");
  await cdp.invoke("set_setting", { key: "runtime.default", value: vk.id });
  assert((await cdp.invoke("get_settings"))["runtime.default"] === vk.id, "default not saved");
  await cdp.invoke("remove_runtime", { id: vk.id });
  assert(!fs.existsSync(path.join(PROFILE_RUNTIMES, vk.id)), "folder not removed");
  const def = (await cdp.invoke("get_settings"))["runtime.default"];
  assert(def === CUDA, `default after removal: ${def}`);
  return `${vk.id} installed (${(vk.total_size / 1e6).toFixed(0)} MB) and removed; default fell back to ${def}`;
});

await check("existing llama-server.exe can be registered and forgotten without deleting it", async () => {
  const exe = path.join(PROFILE_RUNTIMES, CPU, "llama-server.exe");
  const r = await cdp.invoke("register_custom_runtime", { path: exe });
  assert(r.source === "custom" && r.capabilities.flags.length > 50, "not probed");
  await cdp.invoke("remove_runtime", { id: r.id });
  assert(fs.existsSync(exe), "the user's exe was deleted");
  assert(!(await cdp.invoke("list_runtimes")).some((x) => x.id === r.id), "still listed");
  return r.id;
});

await check("a program that is not llama-server is rejected as a runtime", async () => {
  const e = await cdp.invokeErr("register_custom_runtime", { path: "C:\\Windows\\System32\\whoami.exe" });
  assert(/not a working llama-server/.test(e.message), e.message);
});

// ---------------- library ----------------
await check("import a single GGUF file", async () => {
  const r = await cdp.invoke("import_models", { path: path.join(FIX, "single", "fake-llama-Q4_K_M.gguf") });
  assert(r.registered.length === 1, JSON.stringify(r));
  const m = await modelByName("fake-llama-Q4_K_M");
  assert(m && m.quantization === "Q4_K_M" && m.meta.context_length === 4096 && m.source === "import", JSON.stringify(m?.meta));
  ids.fake = m.id;
});

await check("importing the same file again does not duplicate it", async () => {
  const before = (await cdp.invoke("list_models")).length;
  await cdp.invoke("import_models", { path: path.join(FIX, "single", "fake-llama-Q4_K_M.gguf") });
  const after = (await cdp.invoke("list_models")).length;
  assert(before === after, `${before} -> ${after}`);
});

await check("folder import: valid, incomplete shard set and broken file reported", async () => {
  const r = await cdp.invoke("import_models", { path: path.join(FIX, "folder") });
  assert(r.registered.length === 1 && r.invalid.length === 1 && r.skipped.length === 1, JSON.stringify(r));
  const shard = await modelByName("big-Q4_K_M");
  assert(shard.status === "invalid" && /missing part\(s\) 3 of 3/.test(shard.status_detail), shard.status_detail);
  assert(/not a GGUF/.test(r.skipped[0].reason), r.skipped[0].reason);
  const e = await cdp.invokeErr("load_model", { modelId: shard.id, config: await defaults(shard.id), runtimeId: CUDA });
  assert(/can't be loaded/.test(e.message), e.message);
  return "1 ok, 1 incomplete (load refused), 1 skipped";
});

await check("a non-model file is refused with a reason", async () => {
  const e = await cdp.invokeErr("import_models", { path: path.join(FIX, "folder", "broken.gguf") });
  assert(/bad magic/.test(e.message), e.message);
  const e2 = await cdp.invokeErr("import_models", { path: path.join(FIX, "nope.gguf") });
  assert(e2.code === "not_found", e2.code);
});

await check("rescanning the models folder does not create duplicates", async () => {
  const before = (await cdp.invoke("list_models")).length;
  const r = HAS_LMS ? await cdp.invoke("scan_lmstudio") : await cdp.invoke("import_models", { path: APP_MODELS });
  const after = (await cdp.invoke("list_models")).length;
  assert(before === after, `${before} -> ${after}`);
  return `${r.registered.length} re-checked`;
});

await check("a moved file shows as missing and recovers when it returns", async () => {
  const f = path.join(FIX, "single", "fake-llama-Q4_K_M.gguf");
  fs.renameSync(f, f + ".moved");
  let m = (await cdp.invoke("refresh_models")).find((x) => x.id === ids.fake);
  assert(m.status === "missing", m.status);
  const e = await cdp.invokeErr("load_model", { modelId: ids.fake, config: await defaults(ids.fake), runtimeId: CUDA });
  assert(/can't be loaded/.test(e.message), e.message);
  fs.renameSync(f + ".moved", f);
  m = (await cdp.invoke("refresh_models")).find((x) => x.id === ids.fake);
  assert(m.status === "ok", m.status);
});

await check("remove from library keeps the file on disk", async () => {
  await cdp.invoke("remove_model", { id: ids.fake });
  assert(!(await modelByName("fake-llama-Q4_K_M")), "still listed");
  assert(fs.existsSync(path.join(FIX, "single", "fake-llama-Q4_K_M.gguf")), "file was deleted");
});

await check("delete files needs the exact name typed (UI) and then deletes", async () => {
  const f = path.join(FIX, "deleteme", "deleteme-Q4_0.gguf");
  await cdp.invoke("import_models", { path: f });
  const m = await modelByName("deleteme-Q4_0");
  const e = await cdp.invokeErr("delete_model_files", { id: m.id, confirmName: "wrong" });
  assert(/does not match/.test(e.message) && fs.existsSync(f), "wrong name must not delete");
  await cdp.nav("My Models");
  await cdp.eval("location.reload()");
  await sleep(500);
  await cdp.nav("My Models");
  await cdp.waitFor("[...document.querySelectorAll('tr.selectable')].some(r => r.innerText.includes('deleteme-Q4_0'))");
  await cdp.eval("[...document.querySelectorAll('tr.selectable')].find(r => r.innerText.includes('deleteme-Q4_0')).click()");
  await cdp.waitFor("[...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Delete files')");
  await cdp.clickButton("Delete files");
  assert((await cdp.buttonState("Delete permanently")) === "disabled", "enabled before typing");
  await cdp.setValue("input[aria-label='Type the model name to confirm']", "deleteme");
  assert((await cdp.buttonState("Delete permanently")) === "disabled", "enabled with partial name");
  await cdp.setValue("input[aria-label='Type the model name to confirm']", "deleteme-Q4_0");
  await cdp.clickButton("Delete permanently");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('Deleted 1 file')", 5000);
  assert(!fs.existsSync(f), "file still exists");
});

await check("library filter narrows the list", async () => {
  await cdp.nav("My Models");
  await cdp.setValue("input[aria-label='Filter models']", "gemma");
  await cdp.waitFor("document.querySelectorAll('tr.selectable').length === 1");
  await cdp.setValue("input[aria-label='Filter models']", "zzzz-no-match");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('No models match')");
  await cdp.setValue("input[aria-label='Filter models']", "");
});

// ---------------- load settings and engine ----------------
await check("load settings are validated before starting anything", async () => {
  const m = await modelByName(SMALL);
  const base = await defaults(m.id);
  for (const [patch, frag] of [
    [{ n_ctx: 100 }, "context length"],
    [{ batch_size: 256, ubatch_size: 512 }, "micro-batch"],
    [{ parallel: 0 }, "parallel"],
    [{ threads: 0 }, "thread"],
  ]) {
    const e = await cdp.invokeErr("load_model", { modelId: m.id, config: { ...base, ...patch }, runtimeId: CUDA });
    assert(e.code === "invalid_input" && e.message.includes(frag), `${JSON.stringify(patch)}: ${e.message}`);
  }
  assert(llamaCount() === 0, "a backend started despite invalid settings");
  const d = await cdp.invoke("get_load_defaults", { id: m.id });
  assert(d.config.threads !== 0 && d.config.n_ctx !== 100 && d.config.parallel !== 0, "rejected settings were saved as last used: " + JSON.stringify(d.config));
});

await check("UI: select model, CPU runtime disables GPU offload and zeroes VRAM estimate", async () => {
  await cdp.nav("My Models");
  await cdp.waitFor(`[...document.querySelectorAll('tr.selectable')].some(r => r.innerText.includes(${JSON.stringify(SMALL)}))`);
  await cdp.eval(`[...document.querySelectorAll('tr.selectable')].find(r => r.innerText.includes(${JSON.stringify(SMALL)})).click()`);
  await cdp.waitFor("!!document.querySelector('#rt')");
  await cdp.setValue("#rt", CPU);
  await cdp.waitFor("document.querySelector('#gpu').disabled && document.querySelector('#gpu').value === 'cpu_only'");
  await cdp.waitFor("/GPU \\(VRAM\\)\\s*0 B/.test(document.querySelector('.content').innerText)", 5000, "VRAM estimate 0 B");
  await cdp.setValue("#rt", CUDA);
  await cdp.waitFor("!document.querySelector('#gpu').disabled");
});

await check("UI: load on GPU with a chosen context, measured values shown, unload", async () => {
  await cdp.setValue("#ctx", "12288");
  await cdp.setValue("#gpu", "all");
  await cdp.clickButton("Load model");
  await cdp.waitFor("document.querySelector('.content').innerText.includes('Running (measured)') || !!document.querySelector('.notice.error')", 120000, "load");
  const t = await cdp.text(".content");
  assert(t.includes("Running (measured)"), "load failed: " + t.slice(0, 300));
  assert(/Context allocated\s*12,288 tokens/.test(t), "context not 12,288");
  assert(/Layers on GPU\s*29\/29/.test(t), "not all layers on GPU");
  assert(/CUDA0 model\s*\d+ MiB/.test(t), "no measured CUDA buffer");
  const pill = await cdp.text(".engine-pill");
  assert(pill.includes("Loaded") && pill.includes(CUDA), pill);
  assert(llamaCount() === 1, `llama-server count ${llamaCount()}`);
  const res = await cdp.invoke("get_resources");
  assert(res.backend_memory_bytes > 0, "backend memory not measured");
  await cdp.screenshot(path.join(SHOTS, "loaded.png"));
  await cdp.clickButton("Unload");
  await cdp.waitFor("[...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Load model')", 20000);
  await sleep(300);
  assert(llamaCount() === 0, "backend left running");
});

await check("UI: backend log is filled and searchable", async () => {
  const total = (await cdp.invoke("engine_logs", { last: 4000 })).length;
  assert(total > 20, `log lines ${total}`);
  await cdp.setValue("input[aria-label='Search backend log']", "offloaded");
  const shown = await cdp.waitFor("(() => { const m = document.querySelector('.content').innerText.match(/(\\d+) of (\\d+) lines/); return m && Number(m[1]) > 0 && Number(m[1]) < Number(m[2]) ? m[0] : false })()");
  await cdp.setValue("input[aria-label='Search backend log']", "");
  return shown;
});

await check("context 0 uses the model's trained context", async () => {
  const s = await load(SMALL, { n_ctx: 0, gpu_layers: { mode: "all" } });
  assert(s.props.n_ctx === 131072, `n_ctx ${s.props.n_ctx}`);
  return "131,072 tokens";
});

await check("chosen GPU layer count is honoured", async () => {
  const s = await load(SMALL, { n_ctx: 4096, gpu_layers: { mode: "count", count: 10 } });
  assert(s.gpu_layers_offloaded?.[0] === 10, `offloaded ${s.gpu_layers_offloaded}`);
  return `${s.gpu_layers_offloaded.join("/")}`;
});

await check("CPU-only mode on the CUDA runtime puts nothing on the GPU", async () => {
  const s = await load(SMALL, { n_ctx: 4096, gpu_layers: { mode: "cpu_only" } });
  assert(!s.buffers.some((b) => b.device.startsWith("CUDA") && b.kind === "model"), JSON.stringify(s.buffers));
  assert(s.args.includes("--device") && s.args.includes("none"), s.args.join(" "));
});

await check("CPU runtime loads and serves", async () => {
  const s = await load(SMALL, { n_ctx: 4096 }, CPU);
  assert(s.state === "loaded" && s.runtime_id === CPU, s.state);
  assert(!s.buffers.some((b) => b.device.startsWith("CUDA")), "CUDA buffers on CPU runtime");
  const c = await cdp.invoke("engine_status");
  assert(c.props.n_ctx === 4096, "ctx");
});

await check("quantized KV cache (q8_0) with Flash Attention uses less KV memory", async () => {
  const f16 = await load(SMALL, { n_ctx: 16384, gpu_layers: { mode: "all" }, cache_type_k: "f16", cache_type_v: "f16" });
  const q8 = await load(SMALL, { n_ctx: 16384, gpu_layers: { mode: "all" }, cache_type_k: "q8_0", cache_type_v: "q8_0", flash_attention: "on" });
  const kv = (s) => s.buffers.filter((b) => b.kind === "KV").reduce((a, b) => a + b.mib, 0);
  assert(kv(q8) > 0 && kv(q8) < kv(f16) * 0.6, `f16 ${kv(f16)} MiB, q8_0 ${kv(q8)} MiB`);
  return `KV f16 ${kv(f16)} MiB, q8_0 ${kv(q8).toFixed(0)} MiB`;
});

await check("KV cache kept in system RAM when offload is off", async () => {
  const s = await load(SMALL, { n_ctx: 8192, gpu_layers: { mode: "all" }, kv_offload: false });
  assert(s.args.includes("--no-kv-offload"), "flag missing");
  assert(!s.buffers.some((b) => b.kind === "KV" && b.device.startsWith("CUDA")), JSON.stringify(s.buffers));
});

await check("parallel slots and load mode reach the backend", async () => {
  const s = await load(SMALL, { n_ctx: 8192, gpu_layers: { mode: "all" }, parallel: 2, load_mode: "mlock" });
  assert(s.props.total_slots === 2, `slots ${s.props.total_slots}`);
  const i = s.args.indexOf("--load-mode");
  assert(i >= 0 && s.args[i + 1] === "mlock", s.args.join(" "));
});

await check("loading a second model replaces the first (one backend)", async () => {
  await load(SMALL, { n_ctx: 4096 });
  const s = await load(MEDIUM, { n_ctx: 4096, gpu_layers: { mode: "auto" } });
  const m = await modelByName(MEDIUM);
  assert(s.model_id === m.id && s.state === "loaded", "wrong model");
  assert(llamaCount() === 1, `llama-server count ${llamaCount()}`);
  return `offloaded ${s.gpu_layers_offloaded?.join("/")}`;
});

await check("runtime and model in use can't be removed or deleted", async () => {
  const m = await modelByName(MEDIUM);
  const st = await cdp.invoke("engine_status");
  // Guard: only try destructive calls when the model really is loaded, so a
  // failure elsewhere can never turn this into an actual deletion.
  assert(st.state === "loaded" && st.model_id === m.id && st.runtime_id === CUDA, `precondition: ${MEDIUM} must be loaded on ${CUDA} (state ${st.state})`);
  const e1 = await cdp.invokeErr("remove_runtime", { id: CUDA });
  assert(e1.code === "conflict", e1.message);
  const e2 = await cdp.invokeErr("remove_model", { id: m.id });
  assert(e2.code === "conflict", e2.message);
  const e3 = await cdp.invokeErr("delete_model_files", { id: m.id, confirmName: m.display_name });
  assert(e3.code === "conflict", e3.message);
  assert(fs.existsSync(m.path), "LM Studio file touched");
  await unload();
});

await check("per-model default settings save and reset", async () => {
  const m = await modelByName(SMALL);
  const cfg = { ...(await defaults(m.id)), n_ctx: 12288 };
  await cdp.invoke("save_model_defaults", { id: m.id, config: cfg });
  let d = await cdp.invoke("get_load_defaults", { id: m.id });
  assert(d.source === "model_default" && d.config.n_ctx === 12288, JSON.stringify(d));
  await cdp.invoke("reset_model_defaults", { id: m.id });
  d = await cdp.invoke("get_load_defaults", { id: m.id });
  assert(d.source === "last_used", d.source);
});

await check("a model load can be cancelled mid-way", async () => {
  const m = await modelByName(BIG);
  const cfg = { ...(await defaults(m.id)), n_ctx: 8192, gpu_layers: { mode: "all" }, fit: false };
  const p = cdp.invoke("load_model", { modelId: m.id, config: cfg, runtimeId: CUDA }).then(() => "loaded", (e) => e.code);
  await sleep(3000);
  assert(llamaCount() === 1, "backend not started");
  await cdp.invoke("cancel_load");
  const outcome = await p;
  assert(outcome === "cancelled", `outcome ${outcome}`);
  await sleep(500);
  assert(llamaCount() === 0, "backend left after cancel");
  const st = await cdp.invoke("engine_status");
  assert(st.state === "unloaded", st.state);
});

await check("VRAM spill into shared memory is detected and reported", async () => {
  const s = await load(BIG, { n_ctx: 8192, gpu_layers: { mode: "all" }, fit: false });
  assert(s.warnings.length === 1 && /shared system memory/.test(s.warnings[0]), JSON.stringify(s.warnings));
  await cdp.nav("Overview");
  const pill = await cdp.text(".engine-pill");
  assert(pill.includes("Slow: VRAM full"), pill);
  await unload();
  return s.warnings[0].slice(0, 90) + "...";
});

await check("Auto offload fits the big model without spilling", async () => {
  const s = await load(BIG, { n_ctx: 8192, gpu_layers: { mode: "auto" }, fit: true });
  assert(s.warnings.length === 0, s.warnings.join(" "));
  assert(s.gpu_layers_offloaded[0] < s.gpu_layers_offloaded[1], "expected partial offload");
  await unload();
  return `${s.gpu_layers_offloaded.join("/")} layers on GPU`;
});

await check("an invalid GGUF that slips past import fails cleanly at load", async () => {
  // Register a valid header, then corrupt the file after registration.
  const f = path.join(FIX, "corrupt", "corrupt-Q4_0.gguf");
  writeFakeGguf(f, { fileType: 2 });
  await cdp.invoke("import_models", { path: f });
  fs.writeFileSync(f, Buffer.concat([Buffer.from("PK"), Buffer.alloc(4096, 0x41)]));
  const m = await modelByName("corrupt-Q4_0");
  const e = await cdp.invokeErr("load_model", { modelId: m.id, config: { ...(await defaults(m.id)), n_ctx: 2048 }, runtimeId: CUDA });
  const st = await cdp.invoke("engine_status");
  assert(st.state === "failed" && st.error, `state ${st.state}`);
  assert(llamaCount() === 0, "backend left running");
  return e.message.slice(0, 100);
});

// ---------------- lifecycle ----------------
await check("normal window close unloads the model and records it", async () => {
  await load(SMALL, { n_ctx: 4096 });
  assert(llamaCount() === 1, "not loaded");
  cdp.close();
  const exited = await closeAppGracefully(app.pid);
  assert(exited, "app did not exit");
  await sleep(500);
  assert(llamaCount() === 0, "backend survived window close");
  const log = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8");
  assert(log.includes("window closed: backend unloaded"), "no close log line");
});

await check("settings and last-used load config persist across restart", async () => {
  const before = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8").length;
  app = launchApp(EXE, PROFILE, PORT);
  cdp = new Cdp(PORT);
  await cdp.connect();
  const m = await modelByName(SMALL);
  const d = await cdp.invoke("get_load_defaults", { id: m.id });
  assert(d.source === "last_used" && d.config.n_ctx === 4096, JSON.stringify(d.config));
  assert((await cdp.invoke("get_settings"))["runtime.default"] === CUDA, "runtime default lost");
  const tail = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8").slice(before);
  assert(!/marked \d+ model instance/.test(tail), "graceful close left a stale instance");
  assert(!/LM Studio scan/.test(tail), "LM Studio rescanned on second start");
});

await check("a crash (forced kill) leaves no backend and is cleaned up on next start", async () => {
  await load(SMALL, { n_ctx: 4096 });
  assert(llamaCount() === 1, "not loaded");
  cdp.close();
  killApp(app.pid);
  await sleep(1500);
  assert(llamaCount() === 0, "orphaned llama-server after crash");
  const before = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8").length;
  app = launchApp(EXE, PROFILE, PORT);
  cdp = new Cdp(PORT);
  await cdp.connect();
  const tail = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8").slice(before);
  assert(/marked 1 model instance/.test(tail), "stale instance not cleaned: " + tail.slice(0, 300));
  const st = await cdp.invoke("engine_status");
  assert(st.state === "unloaded", st.state);
});

await check("secrets are redacted from the app log", async () => {
  const log = fs.readFileSync(path.join(PROFILE, "logs", "app.log"), "utf8");
  assert(!/hf_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9]{12,}/.test(log), "secret-looking string in log");
  return `${(log.length / 1024).toFixed(0)} KB checked`;
});

// ---------------- teardown ----------------
cdp.close();
await closeAppGracefully(app.pid);
// The profile (hard links, fixtures, database) stays in %TEMP% for inspection.
results.push({ name: "real runtimes folder unchanged", ok: realSnapshot() === REAL_BEFORE, detail: realSnapshot() });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${realSnapshot()})`);
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Screenshots and results: ${RUN}`);
process.exit(failed.length ? 1 : 0);

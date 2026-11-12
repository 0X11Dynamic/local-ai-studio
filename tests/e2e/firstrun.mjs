// First-run setup suite: the setup screen in an isolated profile with an EMPTY
// runtimes folder (the other suites keep it closed). The recommended GPU
// runtime starts installing by itself and is cancelled; the CPU runtime is
// installed into the temp profile only; the starter download is started and
// cancelled (into a temp models folder). Real data is never touched.
//   LAS_E2E_EXE=target/release/local-ai-studio.exe node tests/e2e/firstrun.mjs
// Screenshots: %TEMP%\las-firstrun-shots (kept; the profile is deleted).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Cdp, launchApp, closeAppGracefully, killApp, sleep, countProcesses, exePath } from "./lib.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const shots = process.env.LAS_SHOTS || path.join(process.env.TEMP, "las-firstrun-shots");
const exe = exePath(root);
if (countProcesses("llama-server.exe") > 0) throw new Error("a llama-server is running; stop it first");
const realRuntimes = path.join(process.env.APPDATA, "LocalAIStudio", "runtimes");
const realBefore = fs.existsSync(realRuntimes) ? fs.readdirSync(realRuntimes).sort().join(",") : "";
const run = path.join(process.env.TEMP, `las-firstrun-${Date.now()}`);
const data = path.join(run, "profile");
const models = path.join(run, "models");
fs.mkdirSync(path.join(data, "runtimes"), { recursive: true });
fs.mkdirSync(models, { recursive: true });
fs.mkdirSync(shots, { recursive: true });
const PORT = 9363;
const results = [];
const check = async (name, fn) => {
  try { const d = await fn(); results.push({ name, ok: true }); console.log(`PASS  ${name}${d ? `  (${d})` : ""}`); }
  catch (e) { results.push({ name, ok: false }); console.log(`FAIL  ${name}: ${e.message}`); }
};
const assert = (c, m) => { if (!c) throw new Error(m); };

let app = launchApp(exe, data, PORT, { setup: true });
let cdp = new Cdp(PORT);
cdp.keepSetup = true;
try {
  await cdp.connect();
  await check("setup opens by itself on an empty profile", async () => {
    await cdp.waitFor("!!document.querySelector('[role=dialog].firstrun')", 15000, "setup dialog");
  });
  // Keep downloads out of the real models folder.
  await cdp.invoke("set_setting", { key: "library.models_dir", value: models });

  await check("hardware and GPU listed", async () => {
    await cdp.waitFor("document.querySelector('.firstrun')?.innerText.includes('1. This PC') && !!document.querySelector('.firstrun .kv')", 60000, "plan");
    const t = await cdp.text(".firstrun");
    assert(/RTX|GPU|No GPU/.test(t), t.slice(0, 300));
    const plan = await cdp.invoke("setup_plan");
    return `gpus=${plan.gpus.map((g) => g.name).join("|")} driver=${plan.driver?.status}:${plan.driver?.driver_version} rec=${plan.available.find((r) => r.recommended)?.id} starter_gpu=${plan.starter_gpu?.file} starter_cpu=${plan.starter_cpu.file}`;
  });

  await check("recommended runtime starts installing by itself with progress", async () => {
    await cdp.waitFor("!!document.querySelector('.firstrun [data-installing]')", 30000, "auto install");
    await cdp.waitFor("/of .*B/.test(document.querySelector('.firstrun [data-installing]')?.innerText ?? '')", 60000, "progress bytes");
    await cdp.screenshot(path.join(shots, "firstrun-1-auto-install.png"));
    const id = await cdp.eval("document.querySelector('.firstrun [data-installing]').dataset.installing");
    return id;
  });

  await check("cancel the GPU install, then install the CPU runtime instead", async () => {
    await cdp.clickButton("Cancel", "document.querySelector('.firstrun')");
    await cdp.waitFor("!document.querySelector('.firstrun [data-installing]')", 30000, "install stopped");
    await cdp.waitFor("document.querySelector('.firstrun').innerText.includes('Cancelled')", 5000, "cancel message");
    await cdp.clickButton("Other runtimes", "document.querySelector('.firstrun')");
    const ok = await cdp.eval(`(() => { const b = [...document.querySelectorAll('.firstrun button')].find(b => b.innerText.startsWith('Install CPU runtime')); if (!b) return false; b.click(); return true })()`);
    assert(ok, "no Install CPU runtime button");
    await cdp.waitFor("document.querySelector('.firstrun').innerText.includes('Installed:')", 600000, "cpu runtime installed");
    const rs = await cdp.invoke("list_runtimes");
    assert(rs.length === 1 && rs[0].variant === "cpu", JSON.stringify(rs.map((r) => r.id)));
    return rs[0].id;
  });

  await check("starter model suggested for the CPU, nothing downloaded yet", async () => {
    await cdp.waitFor("!!document.querySelector('.firstrun [data-starter]')", 10000, "starter");
    const file = await cdp.eval("document.querySelector('.firstrun [data-starter]').dataset.starter");
    assert(file === "Qwen3-1.7B-Q4_K_M.gguf", file);
    const dls = await cdp.invoke("list_downloads");
    assert(dls.length === 0, `downloads already: ${dls.length}`);
    await sleep(1500);
    await cdp.screenshot(path.join(shots, "firstrun-2-cpu-runtime-and-starter.png"));
    return file;
  });

  await check("Download starts the real download (then cancelled)", async () => {
    const ok = await cdp.eval(`(() => { const b = [...document.querySelectorAll('.firstrun button')].find(b => b.innerText.startsWith('Download (')); if (!b) return false; b.click(); return true })()`);
    assert(ok, "no Download button");
    await cdp.waitFor("/of .*B/.test(document.querySelector('.firstrun [data-starter]')?.innerText ?? '')", 60000, "download progress");
    await cdp.screenshot(path.join(shots, "firstrun-3-downloading.png"));
    const dls = await cdp.invoke("list_downloads");
    const d = dls.find((x) => x.variant_key === "Qwen3-1.7B-Q4_K_M.gguf");
    assert(d, JSON.stringify(dls));
    await cdp.invoke("cancel_download", { id: d.id });
    return `${d.repo} ${d.state}`;
  });

  await check("Finish stores setup.completed", async () => {
    await cdp.clickButton("Finish", "document.querySelector('.firstrun')");
    await cdp.waitFor("!document.querySelector('.firstrun')", 5000, "dialog closed");
    const s = await cdp.invoke("get_settings");
    assert(s["setup.completed"] && s["setup.completed"].skipped === false, JSON.stringify(s["setup.completed"]));
    assert((await cdp.invoke("setup_should_show")) === false, "would still show");
  });

  await check("Settings > Run setup again reopens it", async () => {
    await cdp.nav("Settings");
    await cdp.clickButton("Run setup again");
    await cdp.waitFor("!!document.querySelector('.firstrun [data-starter]')", 60000, "reopened with plan");
    await sleep(1000);
    await cdp.screenshot(path.join(shots, "firstrun-4-reopened-from-settings.png"));
    await cdp.clickButton("Skip setup", "document.querySelector('.firstrun')");
    await cdp.waitFor("!document.querySelector('.firstrun')", 5000, "closed");
  });
} finally {
  cdp.close();
  if (!(await closeAppGracefully(app.pid))) killApp(app.pid);
}

// Relaunch: must not open by itself again.
app = launchApp(exe, data, PORT, { setup: true });
cdp = new Cdp(PORT);
cdp.keepSetup = true;
try {
  await cdp.connect();
  await check("does not open by itself after Finish", async () => {
    await sleep(3000);
    assert(!(await cdp.eval("!!document.querySelector('.firstrun')")), "setup opened again");
  });
} finally {
  cdp.close();
  if (!(await closeAppGracefully(app.pid))) killApp(app.pid);
}
await sleep(1500);
fs.rmSync(run, { recursive: true, force: true });
const realAfter = fs.existsSync(realRuntimes) ? fs.readdirSync(realRuntimes).sort().join(",") : "";
results.push({ name: "real runtimes folder unchanged", ok: realAfter === realBefore });
console.log(`${realAfter === realBefore ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${realAfter})`);
console.log(`profile removed: ${!fs.existsSync(run)}`);
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed.`);
process.exit(results.every((r) => r.ok) ? 0 : 1);

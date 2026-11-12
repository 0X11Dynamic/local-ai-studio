// Helpers for driving the real Local AI Studio app over the Chrome DevTools
// Protocol (WebView2 remote debugging). No test framework: plain Node 20+.
import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Cdp {
  constructor(port) {
    this.port = port;
    this.id = 0;
    this.pending = new Map();
  }
  async connect(timeoutMs = 120000) {
    const deadline = Date.now() + timeoutMs;
    let page;
    while (!page) {
      try {
        const t = await (await fetch(`http://127.0.0.1:${this.port}/json`)).json();
        page = t.find((x) => x.type === "page");
      } catch {}
      if (!page) {
        if (Date.now() > deadline) throw new Error("app window did not appear");
        await sleep(500);
      }
    }
    this.ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((r, j) => { this.ws.onopen = r; this.ws.onerror = j; });
    this.ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && this.pending.has(msg.id)) { this.pending.get(msg.id)(msg); this.pending.delete(msg.id); }
    };
    // Wait until React has rendered the shell.
    await this.waitFor("!!document.querySelector('.sidebar')", 60000);
    // Backstop for an app build that ignores LOCALAISTUDIO_SKIP_SETUP: mark
    // setup done and close the screen. Set keepSetup to test the screen itself.
    if (!this.keepSetup) {
      await sleep(300);
      const open = await this.eval("!!document.querySelector('.firstrun')");
      const done = await this.eval(`window.__TAURI_INTERNALS__.invoke("finish_setup", { skipped: true }).then(() => true, () => false)`);
      if (open && done) {
        await this.eval(`[...document.querySelectorAll('.firstrun button')].find(b => b.innerText.trim() === 'Skip setup')?.click()`);
        await this.waitFor("!document.querySelector('.firstrun')", 5000, "setup screen closed");
      }
    }
  }
  close() { try { this.ws?.close(); } catch {} }
  send(method, params = {}) {
    return new Promise((r) => { const i = ++this.id; this.pending.set(i, r); this.ws.send(JSON.stringify({ id: i, method, params })); });
  }
  async eval(expr) {
    const r = await this.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error("page error: " + (r.result.exceptionDetails.exception?.description ?? JSON.stringify(r.result.exceptionDetails)));
    return r.result?.result?.value;
  }
  async waitFor(expr, ms = 30000, what = expr) {
    const deadline = Date.now() + ms;
    for (;;) {
      const v = await this.eval(`(() => { try { return ${expr} } catch { return false } })()`);
      if (v) return v;
      if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
      await sleep(250);
    }
  }
  /** Call a Rust IPC command. Resolves to the value, rejects with {code,message}. */
  async invoke(cmd, args = {}) {
    const r = await this.eval(`(async () => { try { return { ok: true, v: await window.__TAURI_INTERNALS__.invoke(${JSON.stringify(cmd)}, ${JSON.stringify(args)}) } } catch (e) { return { ok: false, e } } })()`);
    if (!r.ok) { const err = new Error(`${cmd}: ${r.e?.message ?? JSON.stringify(r.e)}`); err.code = r.e?.code; throw err; }
    return r.v;
  }
  /** Invoke expecting failure; returns the error object. */
  async invokeErr(cmd, args = {}) {
    try { await this.invoke(cmd, args); } catch (e) { return e; }
    throw new Error(`${cmd} was expected to fail but succeeded`);
  }
  async screenshot(file) {
    const r = await this.send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(file, Buffer.from(r.result.data, "base64"));
  }
  // ---- UI helpers ----
  async nav(label) {
    // Match the label only: later-phase items also carry a "P2"-style badge.
    await this.eval(`[...document.querySelectorAll('.nav-item')].find(b => b.querySelector('.nav-label')?.innerText.trim() === ${JSON.stringify(label)}).click()`);
    await this.waitFor(`document.querySelector('.topbar h1')?.innerText === ${JSON.stringify(label)}`, 5000, `page ${label}`);
  }
  async clickButton(text, scope = "document") {
    const ok = await this.eval(`(() => { const b = [...${scope}.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(text)} && !b.disabled); if (!b) return false; b.click(); return true })()`);
    if (!ok) throw new Error(`no enabled button "${text}"`);
  }
  async buttonState(text) {
    return this.eval(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(text)}); return b ? (b.disabled ? 'disabled' : 'enabled') : 'missing' })()`);
  }
  /** Set an input/select value the way React sees user input. */
  async setValue(selector, value) {
    await this.eval(`(() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) throw new Error('no element ${selector.replace(/'/g, "")}');
      const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(String(value))});
      el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    })()`);
  }
  async text(selector = "body") {
    return this.eval(`document.querySelector(${JSON.stringify(selector)})?.innerText ?? ''`);
  }
}

export function countProcesses(image) {
  const out = execFileSync("tasklist", ["/FI", `IMAGENAME eq ${image}`, "/NH"], { encoding: "utf8" });
  return out.split("\n").filter((l) => l.toLowerCase().startsWith(image.toLowerCase())).length;
}

/** Start the app exe with an isolated data folder and WebView2 debugging on. */
/** The app to test: LAS_E2E_EXE (e.g. the installed or release build), else the debug build. */
export function exePath(root) {
  return process.env.LAS_E2E_EXE || path.join(root, "target", "debug", "local-ai-studio.exe");
}

/**
 * First-run setup is kept closed in test profiles (LOCALAISTUDIO_SKIP_SETUP=1)
 * so it can't cover the UI or start a runtime download. Pass
 * `{ setup: true }` to test the setup screen itself (tests/e2e/firstrun.mjs).
 */
export function launchApp(exe, dataDir, port, { setup = false } = {}) {
  const env = {
    ...process.env,
    LOCALAISTUDIO_DATA_DIR: dataDir,
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}`,
  };
  if (setup) delete env.LOCALAISTUDIO_SKIP_SETUP;
  else env.LOCALAISTUDIO_SKIP_SETUP = "1";
  const child = spawn(exe, [], {
    env,
    stdio: "ignore",
    detached: false,
  });
  return child;
}

/** Ask the app to close its window normally (WM_CLOSE), then wait for exit. */
export async function closeAppGracefully(pid, ms = 20000) {
  execFileSync("powershell.exe", ["-NoProfile", "-Command", `(Get-Process -Id ${pid}).CloseMainWindow() | Out-Null`]);
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    try { process.kill(pid, 0); } catch { return true; }
    await sleep(250);
  }
  return false;
}

export function killApp(pid) {
  execFileSync("taskkill", ["/F", "/PID", String(pid)], { stdio: "ignore" });
}

/** Write a small but valid GGUF v3 header (no tensors) for registry tests. */
export function writeFakeGguf(file, { arch = "llama", ctx = 4096, fileType = 15, split } = {}) {
  const parts = [];
  const u32 = (n) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
  const u64 = (n) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
  const u16 = (n) => { const b = Buffer.alloc(2); b.writeUInt16LE(n); return b; };
  const str = (s) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
  const kvs = [
    Buffer.concat([str("general.architecture"), u32(8), str(arch)]),
    Buffer.concat([str("general.name"), u32(8), str("Fake test model")]),
    Buffer.concat([str("general.file_type"), u32(4), u32(fileType)]),
    Buffer.concat([str(`${arch}.context_length`), u32(4), u32(ctx)]),
    Buffer.concat([str(`${arch}.block_count`), u32(4), u32(16)]),
    Buffer.concat([str(`${arch}.attention.head_count`), u32(4), u32(16)]),
    Buffer.concat([str(`${arch}.attention.head_count_kv`), u32(4), u32(4)]),
    Buffer.concat([str(`${arch}.embedding_length`), u32(4), u32(1024)]),
  ];
  if (split) {
    kvs.push(Buffer.concat([str("split.no"), u32(2), u16(split[0])]));
    kvs.push(Buffer.concat([str("split.count"), u32(2), u16(split[1])]));
  }
  parts.push(Buffer.from("GGUF"), u32(3), u64(0), u64(kvs.length), ...kvs, Buffer.alloc(256));
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.concat(parts));
}

/**
 * Isolated test profile under %TEMP%: installed runtimes are mirrored with
 * HARD LINKS (deleting inside the profile never touches the real files) and
 * manifests are rewritten to point inside the profile.
 */
export function makeProfile(run) {
  const real = path.join(process.env.APPDATA, "LocalAIStudio", "runtimes");
  const profile = path.join(run, "profile");
  const runtimes = path.join(profile, "runtimes");
  fs.mkdirSync(runtimes, { recursive: true });
  const mirror = (src, dst) => {
    fs.mkdirSync(dst, { recursive: true });
    for (const e of fs.readdirSync(src, { withFileTypes: true })) {
      const s = path.join(src, e.name), d = path.join(dst, e.name);
      if (e.isDirectory()) mirror(s, d);
      else if (e.name !== "runtime.json") fs.linkSync(s, d);
    }
  };
  const ids = [];
  for (const e of fs.readdirSync(real, { withFileTypes: true })) {
    const man = path.join(real, e.name, "runtime.json");
    if (!e.isDirectory() || e.name.startsWith(".") || !fs.existsSync(man)) continue;
    const m = JSON.parse(fs.readFileSync(man, "utf8"));
    if (m.source !== "official") continue;
    mirror(path.join(real, e.name), path.join(runtimes, e.name));
    m.server_path = m.server_path.replace(real, runtimes);
    fs.writeFileSync(path.join(runtimes, e.name, "runtime.json"), JSON.stringify(m, null, 2));
    ids.push(m.id);
  }
  ids.sort().reverse();
  const snapshot = () => fs.readdirSync(real).filter((n) => !n.startsWith(".")).sort().join(",");
  return {
    profile, runtimes, real, snapshot, before: snapshot(),
    cuda: ids.find((i) => i.includes("-cuda-")), cpu: ids.find((i) => i.endsWith("-cpu")),
  };
}

/** Minimal check runner shared by the phase suites. */
export function makeRunner() {
  const results = [];
  const check = async (name, fn) => {
    const t = Date.now();
    try {
      const detail = await fn();
      results.push({ name, ok: true, detail: detail ?? "" });
      console.log(`PASS  ${name}${detail ? `  (${detail})` : ""}  [${((Date.now() - t) / 1000).toFixed(1)}s]`);
    } catch (e) {
      const detail = e?.message || String(e?.stack ?? e);
      results.push({ name, ok: false, detail });
      console.log(`FAIL  ${name}: ${detail}`);
    }
  };
  return { results, check };
}

export function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// Equivalence test: final rendered HTML of streamed Markdown replies must not change across the perf fixes.
//   node tests/bench/equiv-markdown.mjs --exe <exe> --record   (write tests/bench/golden/markdown.json from the BASELINE exe)
//   node tests/bench/equiv-markdown.mjs --exe <exe>            (compare; exit 1 on any difference)
// No model needed. Each reply is streamed as chat://event segment_start, 8-char deltas, then done
// (same payload shape as chat-render.mjs), then the `.md` element's outerHTML is read once the
// stream is complete. Only the final HTML is compared (not intermediate renders).
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Cdp, launchApp, makeProfile, countProcesses, sleep } from "../e2e/lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i > 0 ? process.argv[i + 1] : d; };
const EXE = arg("exe", process.env.LAS_E2E_EXE);
const RECORD = process.argv.includes("--record");
const PORT = Number(arg("cdp-port", 19337));
const CHUNK = Number(arg("chunk", 8));
const GOLDEN = path.join(path.dirname(fileURLToPath(import.meta.url)), "golden", "markdown.json");
if (!EXE || !fs.existsSync(EXE)) { console.error("--exe <exe> required"); process.exit(1); }
if (countProcesses("local-ai-studio.exe") > 0) { console.error("local-ai-studio.exe already running; aborting"); process.exit(2); }

const REPLIES = {
  code_fences: "Here is code:\n\n```rust\nfn main() {\n    let v: Vec<u32> = (1..4).collect();\n    println!(\"{:?}\", v);\n}\n```\n\nAnd unlabeled:\n\n```\nplain text <b>not html</b>\nsecond line\n```\n\nAnd js with a long line:\n\n```js\nconst x = { a: 1, b: [1, 2, 3], c: \"string with 'quotes' and \\\"escapes\\\"\" }; // trailing comment\nasync function f() { return await fetch('/x'); }\n```\n\n```python\ndef f(x):\n    return [i * 2 for i in range(x)]\n```\n\nInline `code` and ``double `tick` code`` too.\n",
  tables: "| Name | Value | Notes |\n|:-----|------:|:-----:|\n| alpha | 1 | first row with `inline code` |\n| beta | 2 | **bold** and *italic* |\n| gamma | 3 | a [link](https://example.com) |\n\nSecond table:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| ~~gone~~ | x \| y |\n",
  lists: "- First point\n- Second point with **bold**\n  - Nested one\n  - Nested two\n    - Deep\n- Third\n\n1. Step one\n2. Step two\n   1. Sub step\n3. Step three\n\n- [x] done task\n- [ ] open task\n\nLoose list:\n\n- para one\n\n- para two\n\n3. starts at three\n4. four\n",
  links: "Plain https://example.com/auto autolink, a [named link](https://example.org/a?b=1&c=2 \"Title\"), a <https://example.net> angle link, a [ref link][r1] and mailto:foo@example.com.\n\n[r1]: https://example.com/ref\n\nRaw html is escaped: <script>alert(1)</script> and <b>bold?</b> and <a href=\"javascript:alert(1)\">js</a>.\n\n[bad](javascript:alert(1))\n",
  footnotes: "A claim with a footnote[^1] and another[^note] and a repeat[^1].\n\n[^1]: First footnote text with `code`.\n[^note]: Second footnote\n    with a continuation line.\n",
  mixed: "# Heading 1\n\n## Heading 2\n\n### Heading 3\n\nParagraph with **bold**, *italic*, ***both***, ~~strike~~, `code`, and a hard  \nbreak. Escapes: \* \_ 1 < 2 & 3 > 2 &amp; &lt;\n\n> Quote line one\n> > nested quote\n\n---\n\nTerm with emoji \u2713 \u00e9 \u4e2d\u6587 and a very long unbroken token aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.\n\n```\nunterminated fence at the end\n",
  unterminated_stream_tail: "Text then a list\n\n- a\n- b\n\n1. one\n2. two\n\nfinal paragraph without trailing newline",
};

const run = fs.mkdtempSync(path.join(os.tmpdir(), "las-equivmd-"));
const prof = makeProfile(run);
const app = launchApp(EXE, prof.profile, PORT);
const cdp = new Cdp(PORT);
const out = {};
try {
  await cdp.connect();
  const conv = await cdp.invoke("create_conversation");
  await cdp.nav("Chat");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length > 0", 15000, "chat list");
  await cdp.eval(`[...document.querySelectorAll('button.chat-item')].find(b => b.innerText.includes('New chat'))?.click()`);
  await cdp.waitFor("!!document.querySelector('button.chat-item[aria-current=\"true\"]')", 10000, "chat opened");
  await sleep(500);
  const mkMsg = (id, content, stats) => ({ id, conversation_id: conv.id, role: "assistant", content, reasoning: null, stats: stats ?? null, created_at: "2026-01-01T00:00:00Z", model_name: "synthetic", tool_calls: null, tool_call_id: null, attachments: "[]" });
  const send = (payload) => cdp.eval(`window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "chat://event", payload: ${JSON.stringify(payload)} }).then(() => true)`);
  let n = 0;
  for (const [name, text] of Object.entries(REPLIES)) {
    const mid = `equiv-md-${++n}`;
    await send({ type: "segment_start", conversation_id: conv.id, message: mkMsg(mid, "", null) });
    await sleep(300);
    for (let i = 0; i < text.length; i += CHUNK) {
      await send({ type: "delta", conversation_id: conv.id, message_id: mid, content: text.slice(i, i + CHUNK), reasoning: "" });
      if ((i / CHUNK) % 25 === 24) await sleep(40);
    }
    await sleep(400);
    await send({ type: "done", conversation_id: conv.id, message_id: mid, message: mkMsg(mid, text, JSON.stringify({ stop_reason: "stop", total_ms: 1 })) });
    await sleep(700);
    // All .md elements are final now; take the one whose message block is this reply (last assistant message).
    const html = await cdp.eval(`(() => { const m = [...document.querySelectorAll('.msg.assistant .md')]; return m.length ? m[m.length - 1].outerHTML : null; })()`);
    if (!html) throw new Error("no rendered reply for " + name);
    out[name] = html;
    console.log(`captured ${name}: ${html.length} chars`);
  }
  await cdp.invoke("delete_conversation", { id: conv.id }).catch(() => {});
} finally {
  cdp.close();
  try { execSync(`taskkill /F /T /PID ${app.pid}`, { stdio: "ignore" }); } catch {}
  await sleep(1000);
  try { fs.rmSync(run, { recursive: true, force: true }); } catch {}
}
if (RECORD) {
  fs.mkdirSync(path.dirname(GOLDEN), { recursive: true });
  fs.writeFileSync(GOLDEN, JSON.stringify(out, null, 1));
  console.log("golden written: " + GOLDEN);
  process.exit(0);
}
const gold = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
let fail = 0;
for (const k of Object.keys(gold)) {
  const ok = gold[k] === out[k];
  console.log((ok ? "PASS " : "FAIL ") + k);
  if (!ok) { fail++; fs.writeFileSync(path.join(os.tmpdir(), `equiv-md-actual-${k}.html`), out[k] ?? ""); }
}
console.log(fail ? `${fail} FAILED (actual html in %TEMP%)` : "ALL PASS");
process.exit(fail ? 1 : 0);

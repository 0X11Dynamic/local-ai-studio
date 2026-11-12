// Phase 2 end-to-end acceptance run: real Hugging Face search and file lists,
// real downloads (about 0.4 GB, plus a partial one that is cancelled), real
// chat on the downloaded model, System Instructions, and the local API.
//
//   npm --prefix frontend run build
//   cargo build -p local-ai-studio --features tauri/custom-protocol
//   node tests/e2e/phase2.mjs
//
// Isolated profile in %TEMP% (see makeProfile): real library, settings and
// runtimes are untouched; downloads go into the profile's own folder.
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { Cdp, countProcesses, launchApp, closeAppGracefully, makeProfile, makeRunner, assert, sleep, exePath } from "./lib.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const EXE = exePath(ROOT);
const PORT = 9335;
const RUN = path.join(os.tmpdir(), `las-e2e2-${Date.now()}`);
const MODELS = path.join(RUN, "models");
const SHOTS = path.join(RUN, "screenshots");
const REPO = "unsloth/Qwen3-0.6B-GGUF";
const Q4 = "Qwen3-0.6B-Q4_K_M.gguf";
const Q8 = "Qwen3-0.6B-Q8_0.gguf";
const API_PORT = 18234;

fs.mkdirSync(SHOTS, { recursive: true });
const P = makeProfile(RUN);
assert(P.cuda, "need an installed CUDA runtime");
assert(countProcesses("llama-server.exe") === 0, "a llama-server is already running; close it first");
console.log(`run folder: ${RUN}\n`);

const { results, check } = makeRunner();
let app = launchApp(EXE, P.profile, PORT);
let cdp = new Cdp(PORT);
await cdp.connect();
const inv = (c, a) => cdp.invoke(c, a);
const invErr = (c, a) => cdp.invokeErr(c, a);
await inv("set_setting", { key: "library.models_dir", value: MODELS });

const waitDownload = async (id, states, ms = 300000) => {
  const deadline = Date.now() + ms;
  for (;;) {
    const d = (await inv("list_downloads")).find((x) => x.id === id);
    if (d && states.includes(d.state)) return d;
    if (Date.now() > deadline) throw new Error(`download ${id} stuck in ${d?.state}`);
    await sleep(300);
  }
};
const lastAssistant = (c) => [...c.messages].reverse().find((m) => m.role === "assistant");
const waitReply = async (convId, ms = 120000) => {
  const deadline = Date.now() + ms;
  for (;;) {
    const c = await inv("get_conversation", { id: convId });
    const a = lastAssistant(c);
    if (a?.stats && !c.generating) return { c, a, stats: JSON.parse(a.stats) };
    if (Date.now() > deadline) throw new Error("reply did not finish");
    await sleep(300);
  }
};
const ids = {};

// ---------------- Discover ----------------
await check("search returns real Hugging Face GGUF repositories", async () => {
  const r = await inv("hf_search", { query: "qwen3 0.6b", sort: "downloads" });
  assert(r.length > 3, `results ${r.length}`);
  assert(r.every((x) => x.tags.includes("gguf")), "non-GGUF result");
  assert(r.some((x) => x.id === REPO), `${REPO} not found`);
  return `${r.length} results, top ${r[0].id}`;
});

await check("a pasted repo id or URL opens that repository", async () => {
  const a = await inv("hf_search", { query: REPO, sort: "downloads" });
  const b = await inv("hf_search", { query: `https://huggingface.co/${REPO}/tree/main`, sort: "downloads" });
  assert(a.length === 1 && a[0].id === REPO && b.length === 1 && b[0].id === REPO, JSON.stringify([a.length, b.length]));
});

await check("unknown repository shows a clear error", async () => {
  const e = await invErr("hf_repo", { repo: "nobody-xyz-123/does-not-exist-GGUF" });
  assert(/not found|refused access/.test(e.message), e.message);
});

await check("variants are the repository's real files with true sizes", async () => {
  const r = await inv("hf_repo", { repo: REPO });
  const q4 = r.variants.find((v) => v.key === Q4);
  const q8 = r.variants.find((v) => v.key === Q8);
  assert(q4 && q8, "Q4_K_M or Q8_0 missing");
  assert(q4.total_size === 396705472 && q8.total_size === 639447744, `${q4.total_size} ${q8.total_size}`);
  assert(q4.quantization === "Q4_K_M" && q8.quantization === "Q8_0", "quant names");
  assert(q4.files[0].sha256?.length === 64, "no sha256");
  assert(r.variants.find((v) => v.key === "Qwen3-0.6B-UD-Q4_K_XL.gguf")?.quantization === "Q4_K_XL", "XL name");
  assert(r.recommended, "no recommendation");
  assert(["gpu", "partial", "cpu", "too_big"].includes(q4.fit), q4.fit);
  assert(r.details.architecture === "qwen3", r.details.architecture);
  ids.revision = r.details.sha;
  return `${r.variants.length} variants, recommended ${r.recommended}`;
});

await check("multi-shard variants are grouped as one complete set", async () => {
  const r = await inv("hf_repo", { repo: "unsloth/Qwen3-235B-A22B-GGUF" });
  const bf16 = r.variants.find((v) => v.quantization === "BF16");
  assert(bf16 && bf16.shard_count === 10 && bf16.files.length === 10 && bf16.complete, JSON.stringify(bf16 && { s: bf16.shard_count, f: bf16.files.length }));
  assert(bf16.fit === "too_big", `fit ${bf16.fit}`);
  return `BF16 = 10 shards, ${(bf16.total_size / 1e9).toFixed(0)} GB, flagged too big`;
});

await check("UI: search, open a repo, see the variant table", async () => {
  await cdp.nav("Discover");
  await cdp.setValue("input[aria-label='Search models']", REPO);
  // Let React re-render with the typed query before submitting (as a real
  // keypress would), otherwise the submit handler sees the previous value.
  await sleep(150);
  await cdp.eval("document.querySelector('form').requestSubmit()");
  await cdp.waitFor(`[...document.querySelectorAll('tr.selectable')].some(r => r.innerText.includes(${JSON.stringify(REPO)}))`, 20000);
  await cdp.eval(`[...document.querySelectorAll('tr.selectable')].find(r => r.innerText.includes(${JSON.stringify(REPO)})).click()`);
  await cdp.waitFor(`document.querySelector('.content').innerText.includes(${JSON.stringify(Q4)})`, 20000, "variant list");
  const t = await cdp.text(".content");
  assert(t.includes("recommended for this PC") && t.includes("Q8_0") && t.includes("396.7 MB"), "variant table content");
  await cdp.screenshot(path.join(SHOTS, "discover.png"));
});

await check("download can be cancelled and its partial files are removed", async () => {
  const id = await inv("download_variant", { repo: REPO, variantKey: Q8, projectorKey: null });
  await cdp.waitFor("true");
  const deadline = Date.now() + 30000;
  let d;
  do {
    d = (await inv("list_downloads")).find((x) => x.id === id);
    await sleep(150);
  } while (!(d?.downloaded > 1e6) && Date.now() < deadline);
  await inv("cancel_download", { id });
  await waitDownload(id, ["cancelled"]);
  const part = path.join(MODELS, "unsloth", "Qwen3-0.6B-GGUF", Q8 + ".part");
  assert(!fs.existsSync(part) && !fs.existsSync(part.replace(".part", "")), "partial file left behind");
  return `cancelled at ${(d.downloaded / 1e6).toFixed(0)} MB`;
});

await check("UI download: progress, pause, resume, checksum, auto-registration", async () => {
  const rowJs = `[...document.querySelectorAll('tr')].find(r => r.innerText.includes(${JSON.stringify(Q4)}) && [...r.querySelectorAll('button')].some(b => b.innerText.includes('Download')))`;
  await cdp.waitFor(`!!${rowJs}`, 20000, "Q4_K_M row with a Download button");
  await cdp.eval(`[...${rowJs}.querySelectorAll('button')].find(b => b.innerText.includes('Download')).click()`);
  const dl = await cdp.waitFor("(() => { const t = document.querySelector('.content').innerText; return /[0-9.]+ MB of 396.7 MB/.test(t) })()", 30000, "progress text");
  const d1 = (await inv("list_downloads")).find((x) => x.variant_key === Q4 && x.state === "running");
  assert(d1, "no running download");
  await cdp.screenshot(path.join(SHOTS, "downloading.png"));
  await inv("pause_download", { id: d1.id });
  const paused = await waitDownload(d1.id, ["paused"]);
  const part = path.join(MODELS, "unsloth", "Qwen3-0.6B-GGUF", Q4 + ".part");
  const partSize = fs.statSync(part).size;
  assert(partSize > 0 && partSize < 396705472, `part ${partSize}`);
  await inv("resume_download", { id: d1.id });
  const done = await waitDownload(d1.id, ["completed", "failed"]);
  assert(done.state === "completed", `state ${done.state}: ${done.error}`);
  const file = path.join(MODELS, "unsloth", "Qwen3-0.6B-GGUF", Q4);
  assert(fs.statSync(file).size === 396705472 && !fs.existsSync(part), "final file wrong");
  const m = (await inv("list_models")).find((x) => x.path === file);
  assert(m && m.source === "download" && m.repo_id === REPO && m.status === "ok", JSON.stringify(m && { s: m.source, r: m.repo_id }));
  ids.model = m.id;
  const r = await inv("hf_repo", { repo: REPO });
  assert(r.variants.find((v) => v.key === Q4).status === "installed", "not shown as installed");
  void dl;
  return `paused at ${(partSize / 1e6).toFixed(0)} MB (${paused.state}), resumed, sha256 verified, registered`;
});

await check("downloading an installed variant again is refused", async () => {
  assert(ids.model, "precondition: the Q4_K_M download must have completed");
  const e = await invErr("download_variant", { repo: REPO, variantKey: Q4, projectorKey: null });
  assert(/already downloaded/.test(e.message), e.message);
});

await check("a corrupted download fails the checksum and is discarded", async () => {
  // Write a .part of the right size but wrong bytes; resume completes the
  // length check, then SHA-256 must reject it.
  const dir = path.join(MODELS, "unsloth", "Qwen3-0.6B-GGUF");
  const q3 = "Qwen3-0.6B-Q3_K_S.gguf";
  const size = 323075776;
  const fd = fs.openSync(path.join(dir, q3 + ".part"), "w");
  fs.ftruncateSync(fd, size - 1024);
  fs.closeSync(fd);
  const id = await inv("download_variant", { repo: REPO, variantKey: q3, projectorKey: null });
  const d = await waitDownload(id, ["completed", "failed"], 120000);
  assert(d.state === "failed" && /checksum mismatch/.test(d.error ?? ""), `${d.state} ${d.error}`);
  assert(!fs.existsSync(path.join(dir, q3)) && !fs.existsSync(path.join(dir, q3 + ".part")), "corrupt file kept");
  await inv("cancel_download", { id });
  return d.error.slice(0, 80);
});

// ---------------- chat ----------------
await check("downloaded model loads and chat streams a real reply with stats", async () => {
  const s = await inv("load_model", { modelId: ids.model, config: { ...(await inv("get_load_defaults", { id: ids.model })).config, n_ctx: 8192, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
  assert(s.state === "loaded", s.state);
  const c = await inv("create_conversation");
  ids.conv = c.id;
  await inv("chat_send", { id: c.id, content: "What is the capital of France? Answer in one short sentence. /no_think" });
  const { a, stats, c: conv } = await waitReply(c.id);
  assert(/paris/i.test(a.content), `content: ${a.content.slice(0, 120)}`);
  assert(stats.tokens_per_sec > 0 && stats.completion_tokens > 0 && stats.time_to_first_token_ms > 0, JSON.stringify(stats));
  assert(stats.context_tokens > 0 && stats.n_ctx === 8192 && stats.stop_reason === "stop", JSON.stringify(stats));
  assert(conv.title.startsWith("What is the capital"), `title ${conv.title}`);
  return `${stats.completion_tokens} tokens at ${stats.tokens_per_sec.toFixed(0)} tok/s, first token ${stats.time_to_first_token_ms.toFixed(0)} ms`;
});

await check("UI: chat streams token by token and shows stats", async () => {
  await cdp.nav("Chat");
  await cdp.waitFor(`document.querySelector('.chat-item[aria-current="true"]') !== null`, 10000);
  await cdp.setValue("textarea[aria-label='Message']", "List five fruits, one per line. /no_think");
  await cdp.clickButton("Send");
  // Observe at least two different partial lengths before completion.
  const seen = new Set();
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    const len = await cdp.eval("(() => { const m = [...document.querySelectorAll('.msg.assistant')].pop(); return m ? m.innerText.length : 0 })()");
    seen.add(len);
    const done = await cdp.eval("[...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Send')");
    if (done && seen.size > 2) break;
    await sleep(30);
  }
  await cdp.waitFor("/tok\\/s/.test([...document.querySelectorAll('.msg.assistant')].pop().innerText)", 30000, "stats line");
  // The context meter refreshes ~350 ms after the reply; it must appear with real counts.
  await cdp.waitFor("!!document.querySelector('.ctx-bar') && document.querySelector('.composer').innerText.includes('/ 8,192 tokens')", 5000, "context meter");
  await cdp.screenshot(path.join(SHOTS, "chat.png"));
  assert(seen.size > 2, `only ${seen.size} distinct lengths seen (no streaming?)`);
  return `${seen.size} distinct partial lengths observed`;
});

await check("stop actually stops generation in the backend", async () => {
  // Own chat: no history, long requested output, stopped mid-generation.
  const sc = await inv("create_conversation");
  const v = await inv("get_conversation", { id: sc.id });
  await inv("update_chat_settings", { id: sc.id, settings: { ...v.settings_parsed, sampling: { ...v.settings_parsed.sampling, max_tokens: 4000 } } });
  const t0 = Date.now();
  await inv("chat_send", { id: sc.id, content: "Write the numbers from 1 to 2000 in words, one per line, with no other text. /no_think" });
  await sleep(700);
  assert(await inv("chat_stop", { id: sc.id }), "nothing to stop (reply already finished)");
  const { stats, a } = await waitReply(sc.id);
  assert(stats.stop_reason === "cancelled", stats.stop_reason);
  assert(Date.now() - t0 < 5000, "stop took too long");
  const port = (await inv("engine_status")).port;
  await sleep(300);
  const slots = await (await fetch(`http://127.0.0.1:${port}/slots`)).json();
  assert(slots.every((s) => !s.is_processing), "backend still generating");
  await inv("delete_conversation", { id: sc.id });
  return `stopped after ${a.content.length} chars kept; backend idle`;
});

await check("regenerate replaces the last reply; edit truncates and resends", async () => {
  const before = await inv("get_conversation", { id: ids.conv });
  const n = before.messages.length;
  const oldId = lastAssistant(before).id;
  await inv("chat_regenerate", { id: ids.conv });
  let { c } = await waitReply(ids.conv);
  assert(c.messages.length === n && lastAssistant(c).id !== oldId, `messages ${n} -> ${c.messages.length}`);
  const firstUser = c.messages.find((m) => m.role === "user");
  await inv("chat_edit", { id: ids.conv, messageId: firstUser.id, content: "What is the capital of Italy? One short sentence. /no_think" });
  ({ c } = await waitReply(ids.conv));
  assert(c.messages.length === 2, `after edit: ${c.messages.length} messages`);
  assert(/rome/i.test(lastAssistant(c).content), lastAssistant(c).content.slice(0, 100));
});

await check("context meter uses the real tokenizer and tracks the draft", async () => {
  const a = await inv("chat_context", { id: ids.conv, draft: "" });
  const b = await inv("chat_context", { id: ids.conv, draft: "This draft message adds a known number of tokens to the prompt." });
  assert(a.n_ctx === 8192 && a.total > 0 && a.system > 0 && a.history > 0, JSON.stringify(a));
  assert(b.draft > 5 && b.total > a.total, `draft ${b.draft}, total ${a.total} -> ${b.total}`);
  assert(a.template > 0 && a.reserved === 1024, JSON.stringify(a));
  return `total ${a.total} (system ${a.system}, history ${a.history}, template ${a.template})`;
});

await check("context overflow: warn refuses, drop-oldest sends and discloses", async () => {
  await inv("load_model", { modelId: ids.model, config: { ...(await inv("get_load_defaults", { id: ids.model })).config, n_ctx: 1024, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
  const c = await inv("create_conversation");
  const settings = { profile_id: null, sampling: { temperature: 0, top_p: null, top_k: null, min_p: null, repeat_penalty: null, max_tokens: 40, seed: 1, stop: [] }, overflow: "warn", reserve_tokens: 64 };
  await inv("update_chat_settings", { id: c.id, settings });
  const filler = "lorem ipsum dolor sit amet ".repeat(30);
  for (let i = 0; i < 3; i++) {
    await inv("chat_send", { id: c.id, content: `Turn ${i}: ${filler} Reply OK. /no_think` });
    await waitReply(c.id);
  }
  await inv("update_chat_settings", { id: c.id, settings: { ...settings, sampling: { ...settings.sampling, max_tokens: 500 } } });
  const e = await invErr("chat_send", { id: c.id, content: "One more. /no_think" });
  assert(/context/.test(e.message), e.message);
  await inv("update_chat_settings", { id: c.id, settings: { ...settings, overflow: "drop_oldest", sampling: { ...settings.sampling, max_tokens: 500 } } });
  await inv("chat_send", { id: c.id, content: "One more. /no_think" });
  const { stats } = await waitReply(c.id);
  assert(stats.dropped_messages > 0 && stats.context_tokens + 500 <= 1024, JSON.stringify(stats));
  await inv("load_model", { modelId: ids.model, config: { ...(await inv("get_load_defaults", { id: ids.model })).config, n_ctx: 8192, gpu_layers: { mode: "all" } }, runtimeId: P.cuda });
  return `refused with: "${e.message.slice(0, 60)}..."; drop-oldest left out ${stats.dropped_messages} messages`;
});

await check("rename, duplicate, search, export and delete conversations", async () => {
  await inv("rename_conversation", { id: ids.conv, title: "Capitals test" });
  const dup = await inv("duplicate_conversation", { id: ids.conv });
  const found = await inv("list_conversations", { query: "rome" });
  assert(found.some((x) => x.id === ids.conv) && found.some((x) => x.id === dup.id), "search by content");
  assert((await inv("list_conversations", { query: "Capitals" })).length === 2, "search by title");
  const md = path.join(RUN, "export.md"), js = path.join(RUN, "export.json");
  await inv("export_conversation", { id: ids.conv, path: md });
  await inv("export_conversation", { id: ids.conv, path: js });
  assert(/# Capitals test/.test(fs.readFileSync(md, "utf8")) && /Rome/i.test(fs.readFileSync(md, "utf8")), "markdown export");
  assert(JSON.parse(fs.readFileSync(js, "utf8")).messages.length === 2, "json export");
  const e = await invErr("export_conversation", { id: ids.conv, path: path.join(RUN, "x.exe") });
  assert(/\.md or \.json/.test(e.message), e.message);
  await inv("delete_conversation", { id: dup.id });
  assert(!(await inv("list_conversations", {})).some((x) => x.id === dup.id), "not deleted");
});

await check("Markdown renders, raw HTML in a reply is not executed", async () => {
  const c = await inv("create_conversation");
  await inv("chat_send", { id: c.id, content: "Repeat exactly this text and nothing else: <img src=x onerror=alert(1)> **bold** and `code` /no_think" });
  await waitReply(c.id);
  await cdp.nav("Overview");
  await cdp.nav("Chat");
  await cdp.waitFor("document.querySelectorAll('.chat-item').length > 0");
  await cdp.eval(`[...document.querySelectorAll('.chat-item')].find(b => b.innerText.includes('Repeat exactly')).click()`);
  await cdp.waitFor("document.querySelectorAll('.msg.assistant').length > 0", 10000);
  const imgs = await cdp.eval("document.querySelectorAll('.md img[onerror], .md img').length");
  const bold = await cdp.eval("document.querySelectorAll('.md strong').length");
  assert(imgs === 0, "raw <img> was rendered");
  return `strong elements: ${bold}, img elements: ${imgs}`;
});

// ---------------- System Instructions ----------------
await check("built-in templates exist and are read-only", async () => {
  const ps = await inv("list_instruction_profiles");
  const names = ps.filter((p) => p.builtin).map((p) => p.title);
  for (const n of ["General Assistant", "Coding Assistant", "Autonomous Agent (native tools)", "Research Assistant", "Custom"]) assert(names.includes(n), `missing ${n}`);
  const g = ps.find((p) => p.title === "General Assistant");
  const e = await invErr("update_instruction_profile", { id: g.id, title: "x", sections: g.sections });
  assert(e.code === "permission_denied", e.code);
  ids.profiles = ps;
});

await check("custom profile: create, preview with real token count, agent mode adds generated parts", async () => {
  const sections = { mode: "sections", identity: "You are Captain Byte.", task: "Begin every reply with the exact word BANANA, then answer.", tool_use: "", formatting: "", raw: "" };
  ids.banana = await inv("create_instruction_profile", { title: "Banana test", sections });
  const p = await inv("preview_instructions", { mode: "chat", sections, base: null });
  assert(p.text.includes("Captain Byte") && p.text.includes("BANANA") && p.tokens > 10 && p.templated_tokens >= p.tokens, JSON.stringify({ t: p.tokens, tt: p.templated_tokens }));
  assert(!p.text.includes("Available tools"), "chat mode exposed tools");
  const a = await inv("preview_instructions", { mode: "agent", sections, base: null });
  // Since phase 4 a new agent chat offers the built-in tools, so the generated part lists them.
  assert(a.parts.some((x) => !x.editable && x.text.includes("## Available tools") && x.text.includes("shell_execute")), "agent parts");
  return `${p.tokens} tokens (${p.templated_tokens} with template markers)`;
});

await check("per-chat profile reaches the model (reply follows it)", async () => {
  const c = await inv("create_conversation");
  const s = (await inv("get_conversation", { id: c.id })).settings_parsed;
  await inv("update_chat_settings", { id: c.id, settings: { ...s, profile_id: ids.banana } });
  const ctx = await inv("chat_context", { id: c.id, draft: "" });
  assert(ctx.profile?.title === "Banana test" && ctx.profile.source === "conversation", JSON.stringify(ctx.profile));
  await inv("chat_send", { id: c.id, content: "What color is the sky? /no_think" });
  const { a, stats } = await waitReply(c.id);
  assert(/banana/i.test(a.content), `reply ignored instruction: ${a.content.slice(0, 120)}`);
  assert(stats.profile === "Banana test", stats.profile);
  return a.content.slice(0, 60).replace(/\n/g, " ");
});

await check("profile precedence: model default, then app default", async () => {
  const coding = ids.profiles.find((p) => p.title === "Coding Assistant");
  await inv("set_model_instruction_profile", { modelId: ids.model, profileId: coding.id });
  const c = await inv("create_conversation");
  let ctx = await inv("chat_context", { id: c.id, draft: "" });
  assert(ctx.profile.source === "model" && ctx.profile.title === "Coding Assistant", JSON.stringify(ctx.profile));
  await inv("set_model_instruction_profile", { modelId: ids.model, profileId: null });
  ctx = await inv("chat_context", { id: c.id, draft: "" });
  assert(ctx.profile.source === "default", ctx.profile.source);
});

await check("base instruction edit, default switch, export/import round trip", async () => {
  const s = await inv("get_instruction_settings");
  await inv("set_instruction_settings", { settings: { ...s, base: "BASE-MARKER base instruction.", chat_profile_id: ids.banana } });
  const c = await inv("create_conversation");
  const ctx = await inv("chat_context", { id: c.id, draft: "" });
  assert(ctx.profile.title === "Banana test" && ctx.profile.source === "default", JSON.stringify(ctx.profile));
  const p = await inv("preview_instructions", { mode: "chat", sections: ids.profiles[0].sections, base: null });
  assert(p.text.startsWith("BASE-MARKER"), p.text.slice(0, 40));
  const f = path.join(RUN, "profile.json");
  await inv("export_instruction_profile", { id: ids.banana, path: f });
  const newId = await inv("import_instruction_profile", { path: f });
  const imported = (await inv("list_instruction_profiles")).find((x) => x.id === newId);
  assert(imported.title === "Banana test (imported)" && imported.sections.task.includes("BANANA"), JSON.stringify(imported));
  const bad = path.join(RUN, "bad.json");
  fs.writeFileSync(bad, "{ not json");
  const e = await invErr("import_instruction_profile", { path: bad });
  assert(/not an instruction profile/.test(e.message), e.message);
  // Deleting the default falls back to General Assistant.
  await inv("delete_instruction_profile", { id: newId });
  await inv("delete_instruction_profile", { id: ids.banana });
  const after = await inv("get_instruction_settings");
  assert(after.chat_profile_id === ids.profiles.find((x) => x.title === "General Assistant").id, "default did not fall back");
  await inv("reset_instruction_base");
});

await check("test instruction runs without saving anything", async () => {
  const before = (await inv("list_conversations", {})).length;
  const r = await inv("test_instructions", { sections: { mode: "raw", raw: "Always answer in exactly three words.", identity: "", task: "", tool_use: "", formatting: "" }, base: null, prompt: "Describe the ocean. /no_think" });
  assert(r.content.trim().length > 0 && r.stats.completion_tokens > 0, JSON.stringify(r).slice(0, 200));
  assert((await inv("list_conversations", {})).length === before, "test created a conversation");
  return r.content.trim().slice(0, 60);
});

await check("UI: System Instructions page shows editor, unsaved state and token count", async () => {
  await cdp.nav("System Instructions");
  // Two-pane page: list on the left, editor + preview + bottom save bar on the right.
  await cdp.waitFor("document.querySelector('.instr-nav') && document.querySelectorAll('.instr-item').length > 1", 10000, "instruction list");
  // Pick a built-in template explicitly (the page may open on any profile or none).
  await cdp.eval("[...document.querySelectorAll('.instr-item')].find(b => b.innerText.includes('General Assistant')).click()");
  try {
    await cdp.waitFor("!!document.querySelector('#sec-identity, #sec-raw')", 10000, "instructions editor");
  } catch (e) {
    const seen = await cdp.eval("document.querySelector('.instr-main')?.innerText.slice(0, 500) + ' | ' + document.querySelectorAll('textarea').length + ' textareas: ' + [...document.querySelectorAll('textarea')].map(t => t.id).join(',')");
    throw new Error(`${e.message}; main shows: ${JSON.stringify(seen)}`);
  }
  await cdp.waitFor("document.querySelector('.instr-actions-status')?.innerText.trim().length > 0", 5000, "save bar status");
  // Make an editable copy (built-in templates are read-only): "Duplicate to edit" or "Duplicate".
  const dup = await cdp.eval("(() => { const b = [...document.querySelectorAll('.instr-head button')].find(b => /^Duplicate/.test(b.innerText.trim())); if (!b) return false; b.click(); return true })()");
  if (!dup) throw new Error("no Duplicate button in the editor header");
  await cdp.waitFor("document.querySelector('input[aria-label=\"Profile name\"]') && !document.querySelector('#sec-identity, #sec-raw')?.readOnly", 10000, "editable copy");
  await cdp.waitFor("document.querySelector('.instr-actions-status')?.innerText.trim() === 'Saved'", 5000, "clean state before editing");
  await cdp.setValue("#sec-identity", "You are a terse assistant.");
  await cdp.waitFor("document.querySelector('.instr-actions-status')?.innerText.includes('Unsaved changes')", 5000, "unsaved state");
  await cdp.waitFor("/Unsaved/.test(document.querySelector('.instr-head').innerText)", 5000, "unsaved tag in header");
  await cdp.waitFor("/[0-9][0-9,]* (tokens|characters)/.test(document.querySelector('.instr-actions-tokens').innerText) && /[0-9][0-9,]* (tokens|characters)/.test(document.querySelector('.instr-tokens').innerText)", 10000, "token count");
  await cdp.screenshot(path.join(SHOTS, "instructions.png"));
  await cdp.clickButton("Save");
  await cdp.waitFor("document.querySelector('.instr-actions-status')?.innerText.trim() === 'Saved' && document.querySelector('.content').innerText.includes('Profile saved')", 5000, "saved state");
});

// ---------------- local API ----------------
const base = `http://127.0.0.1:${API_PORT}`;
const http = async (method, p, { body, token, origin } = {}) => {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (origin) headers.Origin = origin;
  const r = await fetch(base + p, { method, headers, body: body ? JSON.stringify(body) : undefined });
  const text = await r.text();
  let json = null;
  try { json = JSON.parse(text); } catch {}
  return { status: r.status, json, text, headers: r.headers };
};
const apiCfg = { host: "127.0.0.1", port: API_PORT, require_auth: false, cors_origins: [], max_concurrent: 4, rate_per_minute: 600, log_retention_days: 7, autostart: false };

await check("API refuses a port that is already in use", async () => {
  const srv = net.createServer().listen(API_PORT + 1, "127.0.0.1");
  await new Promise((r) => srv.once("listening", r));
  const e = await invErr("api_start", { config: { ...apiCfg, port: API_PORT + 1 } });
  srv.close();
  assert(e.code === "conflict" && /already in use/.test(e.message), e.message);
});

await check("API: /v1/models and chat completions (JSON and streaming)", async () => {
  const s = await inv("api_start", { config: apiCfg });
  assert(s.running && s.address === base, JSON.stringify(s));
  const m = await http("GET", "/v1/models");
  assert(m.status === 200 && m.json.data[0].id === "Qwen3-0.6B-Q4_K_M", m.text);
  const c = await http("POST", "/v1/chat/completions", { body: { model: "anything", messages: [{ role: "user", content: "Say hello. /no_think" }], max_tokens: 200 } });
  assert(c.status === 200 && c.json.choices[0].message.content.length > 0 && c.json.usage.completion_tokens > 0, c.text.slice(0, 200));
  const st = await http("POST", "/v1/chat/completions", { body: { messages: [{ role: "user", content: "Say hi. /no_think" }], stream: true, max_tokens: 200 } });
  const chunks = st.text.split("\n").filter((l) => l.startsWith("data: "));
  assert(st.status === 200 && st.headers.get("content-type").includes("text/event-stream") && chunks.length > 3 && chunks.at(-1) === "data: [DONE]", `${st.status} ${chunks.length}`);
  const legacy = await http("POST", "/v1/completions", { body: { prompt: "The sky is", max_tokens: 8 } });
  assert(legacy.status === 200 && legacy.json.choices?.[0]?.text !== undefined, legacy.text.slice(0, 200));
  return `model id ${m.json.data[0].id}, ${chunks.length} SSE chunks`;
});

await check("API: native endpoints need the token; load/unload work through them", async () => {
  assert((await http("GET", "/api/models")).status === 401, "no token");
  const token = await inv("api_generate_token");
  assert(/^las_[0-9a-f]{64}$/.test(token), token);
  assert((await http("GET", "/api/models", { token: "las_wrong" })).status === 401, "wrong token");
  const list = await http("GET", "/api/models", { token });
  assert(list.status === 200 && list.json.models.some((x) => x.id === ids.model && x.loaded), list.text.slice(0, 200));
  const u = await http("POST", "/api/models/unload", { token });
  assert(u.status === 200 && u.json.state === "unloaded", u.text.slice(0, 200));
  const none = await http("POST", "/v1/chat/completions", { body: { messages: [{ role: "user", content: "hi" }] } });
  assert(none.status === 503 && none.json.error.type === "model_not_loaded", none.text);
  const l = await http("POST", "/api/models/load", { token, body: { model_id: ids.model } });
  assert(l.status === 200 && l.json.state === "loaded", l.text.slice(0, 200));
  const bad = await http("POST", "/api/models/load", { token, body: {} });
  assert(bad.status === 400, bad.text);
  ids.token = token;
});

await check("API: web pages from other origins are refused; allowed origin gets CORS", async () => {
  const r = await http("POST", "/v1/chat/completions", { origin: "https://evil.example", body: { messages: [{ role: "user", content: "hi" }] } });
  assert(r.status === 403 && !r.headers.get("access-control-allow-origin"), `${r.status}`);
  await inv("api_start", { config: { ...apiCfg, cors_origins: ["http://localhost:5173"] } });
  const ok = await http("GET", "/v1/models", { origin: "http://localhost:5173" });
  assert(ok.status === 200 && ok.headers.get("access-control-allow-origin") === "http://localhost:5173", `${ok.status}`);
});

await check("API: token can be required on /v1; LAN without auth is refused", async () => {
  await inv("api_start", { config: { ...apiCfg, require_auth: true } });
  assert((await http("GET", "/v1/models")).status === 401, "no token");
  assert((await http("GET", "/v1/models", { token: ids.token })).status === 200, "token");
  const e = await invErr("api_start", { config: { ...apiCfg, host: "0.0.0.0", require_auth: false } });
  assert(/requires authentication/.test(e.message), e.message);
  await inv("api_start", { config: apiCfg });
});

await check("API: client disconnect mid-stream stops generation", async () => {
  const ctrl = new AbortController();
  const r = await fetch(base + "/v1/chat/completions", { method: "POST", signal: ctrl.signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: [{ role: "user", content: "Count from 1 to 500. /no_think" }], stream: true, max_tokens: 3000 }) });
  const reader = r.body.getReader();
  for (let i = 0; i < 10; i++) await reader.read();
  ctrl.abort();
  await sleep(800);
  const port = (await inv("engine_status")).port;
  const slots = await (await fetch(`http://127.0.0.1:${port}/slots`)).json();
  assert(slots.every((s) => !s.is_processing), "backend still generating after client left");
});

await check("API request log records endpoint, status, model and tokens", async () => {
  await sleep(500);
  const logs = await inv("api_logs", { limit: 200 });
  const chat = logs.find((l) => l.endpoint === "/v1/chat/completions" && l.status === 200 && l.completion_tokens > 0);
  assert(chat && chat.model === "Qwen3-0.6B-Q4_K_M", JSON.stringify(logs.slice(0, 3)));
  assert(logs.some((l) => l.status === 401) && logs.some((l) => l.status === 403) && logs.some((l) => l.status === 503), "error statuses not logged");
  assert(logs.some((l) => l.error?.includes("stream ended early")), "aborted stream not marked");
  return `${logs.length} entries`;
});

await check("UI: Developer page shows the running server and request log", async () => {
  await cdp.nav("Developer / API");
  await cdp.waitFor(`document.querySelector('.content').innerText.includes('Running at') && document.querySelector('.content').innerText.includes('/v1/chat/completions')`, 10000);
  await cdp.screenshot(path.join(SHOTS, "developer.png"));
});

await check("API stops: port is released", async () => {
  await inv("api_stop");
  await sleep(300);
  const e = await fetch(base + "/v1/models").then(() => "open", () => "refused");
  assert(e === "refused", e);
});

// ---------------- restart persistence ----------------
await check("chats, profiles, downloads and API settings survive a restart", async () => {
  await inv("api_save_config", { config: { ...apiCfg, autostart: true } });
  cdp.close();
  assert(await closeAppGracefully(app.pid), "app did not close");
  await sleep(500);
  assert(countProcesses("llama-server.exe") === 0, "backend survived close");
  app = launchApp(EXE, P.profile, PORT);
  cdp = new Cdp(PORT);
  await cdp.connect();
  const convs = await cdp.invoke("list_conversations", {});
  assert(convs.some((c) => c.title === "Capitals test" && c.message_count === 2), "conversation lost");
  const d = await cdp.invoke("list_downloads");
  assert(d.some((x) => x.variant_key === Q4 && x.state === "completed"), "download history lost");
  await sleep(1000);
  const s = await cdp.invoke("api_status");
  assert(s.running && s.address === base, `autostart: ${JSON.stringify(s)}`);
  const m = await http("GET", "/v1/models");
  assert(m.status === 200 && m.json.data.length === 0, "no model should be loaded after restart");
  await cdp.invoke("api_stop");
});

await check("an interrupted download comes back paused and resumes", async () => {
  const id = await cdp.invoke("download_variant", { repo: REPO, variantKey: Q8, projectorKey: null });
  const deadline = Date.now() + 30000;
  let d;
  do {
    d = (await cdp.invoke("list_downloads")).find((x) => x.id === id);
    await sleep(100);
  } while (!(d?.downloaded > 5e6) && Date.now() < deadline);
  cdp.close();
  (await import("./lib.mjs")).killApp(app.pid);
  await sleep(1000);
  app = launchApp(EXE, P.profile, PORT);
  cdp = new Cdp(PORT);
  await cdp.connect();
  const after = (await cdp.invoke("list_downloads")).find((x) => x.id === id);
  assert(after.state === "paused" && /interrupted/.test(after.error ?? ""), `${after.state} ${after.error}`);
  await cdp.invoke("resume_download", { id });
  const deadline2 = Date.now() + 300000;
  let fin;
  do {
    fin = (await cdp.invoke("list_downloads")).find((x) => x.id === id);
    await sleep(300);
  } while (!["completed", "failed"].includes(fin.state) && Date.now() < deadline2);
  assert(fin.state === "completed", `${fin.state} ${fin.error}`);
  assert(fs.statSync(path.join(MODELS, "unsloth", "Qwen3-0.6B-GGUF", Q8)).size === 639447744, "size");
  return `interrupted at ${(d.downloaded / 1e6).toFixed(0)} MB, resumed after restart, verified`;
});

// ---------------- teardown ----------------
cdp.close();
await closeAppGracefully(app.pid);
results.push({ name: "real runtimes folder unchanged", ok: P.snapshot() === P.before, detail: P.snapshot() });
console.log(`${results.at(-1).ok ? "PASS" : "FAIL"}  real runtimes folder unchanged  (${P.snapshot()})`);
const failed = results.filter((r) => !r.ok);
fs.writeFileSync(path.join(RUN, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${results.length - failed.length}/${results.length} passed. Screenshots and results: ${RUN}`);
process.exit(failed.length ? 1 : 0);

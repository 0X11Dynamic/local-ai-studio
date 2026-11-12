// Extras, part 3 (imported by extras-more.mjs): file attachments and
// thinking controls, with the cached Qwen3-4B (text only) and Gemma-4-E4B
// (vision through its projector, thinking toggle off by default).
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { assert } from "./lib.mjs";

export default async function thinking(t) {
  const { cdp, inv, invErr, check, ids, load, convWith, waitDone, waitFor, lastAssistant, parse, P, RUN, SHOTS } = t;
  const FILES = path.join(RUN, "files");
  fs.mkdirSync(FILES, { recursive: true });
  const notes = path.join(FILES, "notes.txt");
  fs.writeFileSync(notes, "Inventory notes\nThe secret code word for this project is ZEBRA-4471.\nNothing else matters here.\n");
  const big = path.join(FILES, "big.log");
  fs.writeFileSync(big, Buffer.alloc(25 * 1024 * 1024, 0x61));
  const huge = path.join(FILES, "huge.txt");
  fs.writeFileSync(huge, "lorem ipsum dolor sit amet consectetur ".repeat(12000));
  const png = path.join(FILES, "word.png");
  execFileSync("powershell.exe", ["-NoProfile", "-Command", `Add-Type -AssemblyName System.Drawing; $b = New-Object System.Drawing.Bitmap 640,200; $g = [System.Drawing.Graphics]::FromImage($b); $g.Clear([System.Drawing.Color]::White); $f = New-Object System.Drawing.Font('Arial',72,[System.Drawing.FontStyle]::Bold); $g.DrawString('PELICAN',$f,[System.Drawing.Brushes]::Black,30,40); $b.Save('${png}',[System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $b.Dispose()`]);
  assert(fs.existsSync(png), "test image not created");
  /** Attach; if the refusal comes only at send time, return that error instead. */
  const attachErr = async (conv, file) => {
    try {
      await inv("chat_attach", { id: conv, path: file });
    } catch (e) { return e; }
    try { await inv("chat_send", { id: conv, content: "describe the file" }); } catch (e) { return e; }
    const c = await waitDone(conv, 120000).catch(() => null);
    throw new Error(`no refusal for ${path.basename(file)}: ${c ? lastAssistant(c)?.content?.slice(0, 120) : "?"}`);
  };

  // ---------------- attachments with a text-only model ----------------
  await check("text attachment: counted in the context meter, quoted to the model, answered from", async () => {
    if ((await inv("engine_status")).state !== "loaded" || (await inv("engine_status")).model_id !== ids.qwen4b) await load("qwen4b", { n_ctx: 16384 });
    const conv = await convWith({ agent_mode: false });
    ids.attConv = conv;
    const before = await inv("chat_context", { id: conv, draft: "" });
    const a = await inv("chat_attach", { id: conv, path: notes });
    assert(a.kind === "text" && a.tokens > 5, JSON.stringify(a));
    const after = await inv("chat_context", { id: conv, draft: "" });
    assert(after.attachments > before.attachments && after.pending_attachments === 1, `meter ${before.attachments} -> ${after.attachments}, pending ${after.pending_attachments}`);
    await inv("chat_send", { id: conv, content: "What is the secret code word in the attached file? Reply with just the code word." });
    const c = await waitDone(conv);
    assert(/ZEBRA-4471/.test(lastAssistant(c).content), lastAssistant(c).content.slice(0, 200));
    const user = c.messages.find((m) => m.role === "user");
    assert(parse(user.attachments, []).length === 1, `stored attachments ${user.attachments}`);
    const dir = path.join(P.profile, "attachments", conv);
    assert(fs.existsSync(dir) && fs.readdirSync(dir).length >= 1, `no stored copy in ${dir}`);
    return `+${after.attachments - before.attachments} tokens in the meter; answer "${lastAssistant(c).content.trim().slice(0, 40)}"`;
  });

  await check("oversized files are refused with a clear message", async () => {
    const conv = await convWith({ agent_mode: false });
    const e1 = await invErr("chat_attach", { id: conv, path: big });
    assert(/20 MB/.test(e1.message) && /big\.log/.test(e1.message), e1.message);
    const e2 = await attachErr(conv, huge);
    assert(/too big for the context/.test(e2.message) && /huge\.txt/.test(e2.message), e2.message);
    await inv("delete_conversation", { id: conv });
    return `${e1.message.slice(0, 70)} | ${e2.message.slice(0, 90)}`;
  });

  await check("images are refused for a model without vision", async () => {
    const conv = await convWith({ agent_mode: false });
    const e = await attachErr(conv, png);
    assert(/can't read images/.test(e.message), e.message);
    const sup = await inv("chat_model_support");
    assert(sup && sup.images === false, JSON.stringify(sup));
    await inv("delete_conversation", { id: conv });
    return e.message.slice(0, 100);
  });

  await check("deleting a chat moves its attachments to the app trash", async () => {
    const dir = path.join(P.profile, "attachments", ids.attConv);
    await inv("delete_conversation", { id: ids.attConv });
    assert(!fs.existsSync(dir), "attachment folder still in place");
    const trash = path.join(P.profile, "trash");
    const moved = fs.existsSync(trash) && fs.readdirSync(trash).find((n) => n.startsWith(`attachments-${ids.attConv}`));
    assert(moved, `not in ${trash}: ${fs.existsSync(trash) ? fs.readdirSync(trash) : "no trash folder"}`);
    return `trash\\${moved}`;
  });

  // ---------------- vision with Gemma-4 ----------------
  await check("image without the projector loaded: says to reload; reload adds it; the model reads the image", async () => {
    await inv("unload_model");
    await load("gemma", { n_ctx: 8192, load_projector: false });
    const conv = await convWith({ agent_mode: false });
    let sup = await inv("chat_model_support");
    assert(sup.has_projector && !sup.projector_loaded, JSON.stringify(sup));
    const e = await attachErr(conv, png);
    assert(/vision projector isn't loaded/.test(e.message), e.message);
    await inv("chat_pending_attachments", { id: conv }).then(async (l) => { for (const a of l) await inv("chat_detach", { id: conv, attachmentId: a.id }); });
    const s = await inv("reload_with_projector");
    assert(s.state === "loaded", `${s.state} ${s.error ?? ""}`);
    sup = await inv("chat_model_support");
    assert(sup.projector_loaded && sup.images, JSON.stringify(sup));
    const a = await inv("chat_attach", { id: conv, path: png });
    assert(a.kind === "image", JSON.stringify(a));
    await inv("chat_send", { id: conv, content: "What single word is written in this image? Reply with just that word." });
    const c = await waitDone(conv);
    const ans = lastAssistant(c).content;
    assert(/pelican/i.test(ans), `answer: ${ans.slice(0, 200)}`);
    await cdp.nav("Chat");
    await cdp.screenshot(path.join(SHOTS, "image-attachment.png"));
    return `answer "${ans.trim().slice(0, 40)}"`;
  });

  await check("thinking toggle: on gives a reasoning block, off gives none (Gemma-4)", async () => {
    const ask = async (thinking) => {
      const conv = await convWith({ agent_mode: false, thinking });
      await inv("chat_send", { id: conv, content: "Is 391 a prime number? Answer yes or no, then one short sentence why." });
      const c = await waitDone(conv);
      const m = lastAssistant(c);
      return { reasoning: (m.reasoning ?? "").trim(), content: m.content, stats: parse(m.stats) };
    };
    const on = await ask(true);
    const off = await ask(false);
    assert(on.stats.thinking_requested === true && off.stats.thinking_requested === false, `stats ${JSON.stringify(on.stats.chat_template_kwargs)} / ${JSON.stringify(off.stats.chat_template_kwargs)}`);
    assert(on.reasoning.length > 20, `no reasoning with thinking on: content=${on.content.slice(0, 200)}`);
    assert(off.reasoning.length === 0, `reasoning with thinking off: ${off.reasoning.slice(0, 120)}`);
    return `on: ${on.reasoning.length} chars of reasoning; off: none`;
  });

  await check("Thinking control shows in the chat for a toggle model, not for a model without it", async () => {
    await cdp.nav("Chat");
    await cdp.eval(`document.querySelector('.chat-item')?.click()`);
    const shown = await cdp.waitFor(`!!document.querySelector('[aria-label="Thinking"]')`, 10000, "Thinking toggle").catch(() => false);
    await cdp.screenshot(path.join(SHOTS, "thinking-toggle.png"));
    assert(shown, "no Thinking control with Gemma-4 loaded");
    await inv("unload_model");
    await load("qwen4b", { n_ctx: 8192 });
    await cdp.nav("My Models");
    await cdp.nav("Chat");
    await cdp.eval(`document.querySelector('.chat-item')?.click()`);
    await new Promise((r) => setTimeout(r, 1500));
    const still = await cdp.eval(`!!document.querySelector('[aria-label="Thinking"]')`);
    assert(!still, "Thinking control shown for Qwen3-4B-Instruct (no thinking)");
    return "toggle shown for Gemma-4, hidden for Qwen3-4B";
  });

  fs.rmSync(big, { force: true });
}

// Second half of the extras suite (imported by extras.mjs): subagents in
// chat, per-subagent git worktrees, thinking controls, file attachments.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { assert, sleep } from "./lib.mjs";

export default async function more(t) {
  const { cdp, inv, invErr, check, ids, load, convWith, waitDone, waitFor, lastAssistant, toolCalls, parse, RUN, SHOTS } = t;
  const git = (cwd, ...a) => execFileSync("git", a, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
  const REPO = path.join(RUN, "repo");
  const PLAIN = path.join(RUN, "plain");
  fs.mkdirSync(REPO, { recursive: true });
  fs.mkdirSync(PLAIN, { recursive: true });
  fs.writeFileSync(path.join(REPO, "README.md"), "# test repo\n");
  git(REPO, "init", "-q", "-b", "main");
  git(REPO, "config", "user.name", "LAS test");
  git(REPO, "config", "user.email", "las-test@example.invalid");
  git(REPO, "add", ".");
  git(REPO, "commit", "-q", "-m", "init");
  const SPAWN_PROMPT = (iso) => `Use the spawn_subagents tool exactly once with two subagents${iso ? ', each with "isolation": "worktree"' : ""}:
1. label "alpha": create a file named alpha.txt containing the word ALPHA in its working folder, then commit it with git (git add alpha.txt; git commit -m alpha).
2. label "bravo": create a file named bravo.txt containing the word BRAVO in its working folder, then commit it with git (git add bravo.txt; git commit -m bravo).
After they finish, reply with their results. Do not create the files yourself.`;

  // ---------------- subagents in chat + worktrees ----------------
  await check("chat subagents: off by default, offered only in Agent mode when allowed", async () => {
    await inv("set_local_roots", { roots: [REPO] });
    if ((await inv("engine_status")).state !== "loaded") await load("qwen4b", { n_ctx: 16384 });
    const conv = await convWith({ agent_mode: true, mode: "agent", builtin_tools: true, tool_servers: [] });
    const v = await inv("get_conversation", { id: conv });
    assert(v.settings_parsed.subagents === false, `default subagents ${v.settings_parsed.subagents}`);
    const off = await inv("chat_context", { id: conv, draft: "" });
    await inv("update_chat_settings", { id: conv, settings: { ...v.settings_parsed, subagents: true, max_subagents: 2 } });
    const on = await inv("chat_context", { id: conv, draft: "" });
    assert(on.tool_count === off.tool_count + 1, `tools off ${off.tool_count} on ${on.tool_count}`);
    await inv("update_chat_settings", { id: conv, settings: { ...v.settings_parsed, subagents: true, mode: "plan" } });
    const plan = await inv("chat_context", { id: conv, draft: "" });
    assert(plan.tool_count < off.tool_count, `plan mode offered ${plan.tool_count}`);
    const bad = await invErr("update_chat_settings", { id: conv, settings: { ...v.settings_parsed, subagents: true, max_subagents: 9 } });
    assert(bad, "accepted max_subagents 9");
    return `${off.tool_count} tools -> ${on.tool_count} with subagents; plan ${plan.tool_count}`;
  });

  await check("chat subagents in worktrees: each child commits on its own branch, main tree untouched", async () => {
    const conv = await convWith({ agent_mode: true, mode: "agent", builtin_tools: true, tool_servers: [], subagents: true, max_subagents: 2 });
    ids.wtConv = conv;
    const ap = t.autoApprove(null);
    try {
      await inv("chat_send", { id: conv, content: SPAWN_PROMPT(true) });
      await waitDone(conv, 900000);
    } finally { ap.stop(); }
    const c = await inv("get_conversation", { id: conv });
    const spawn = toolCalls(c).filter((x) => x.function.name === "spawn_subagents");
    assert(spawn.length >= 1, `no spawn call: ${toolCalls(c).map((x) => x.function.name)}`);
    const subs = await inv("chat_subagents", { id: conv });
    const rows = subs.subagents;
    const wts = rows.map((r) => parse(r.settings).worktree).filter(Boolean);
    assert(wts.length >= 2, `worktrees: ${JSON.stringify(rows.map((r) => [r.label, r.state, parse(r.settings).worktree]))}`);
    assert(!fs.existsSync(path.join(REPO, "alpha.txt")) && !fs.existsSync(path.join(REPO, "bravo.txt")), "files appeared in the main tree");
    const branches = git(REPO, "branch", "--list", "las/*").split("\n").map((s) => s.replace(/^[*+ ]+/, "").trim()).filter(Boolean);
    const withCommits = branches.filter((b) => Number(git(REPO, "rev-list", "--count", `main..${b}`)) > 0);
    for (const r of rows) {
      const wt = parse(r.settings).worktree;
      const f = path.join(wt.path, `${r.label}.txt`);
      const inTree = fs.existsSync(f) || git(wt.path, "log", "--name-only", "--format=", `${wt.base}..HEAD`).includes(`${r.label}.txt`);
      assert(inTree, `${r.label}.txt not in its worktree ${wt.path} (state ${r.state}): ${(r.result ?? "").slice(0, 200)}`);
    }
    assert(branches.length >= 2 && withCommits.length >= 1, `branches ${branches}, with commits ${withCommits}; states ${rows.map((r) => `${r.label}:${r.state}`)}`);
    assert(git(REPO, "status", "--porcelain") === "", `main tree not clean: ${git(REPO, "status", "--porcelain")}`);
    assert(rows.every((r) => /Worktree: branch las\//.test(r.result ?? "")), "result missing worktree note");
    ids.wtBranches = withCommits;
    await cdp.nav("Chat");
    await cdp.screenshot(path.join(SHOTS, "chat-subagents.png"));
    return `${rows.length} children in worktrees; ${withCommits.length} of ${branches.length} branches committed`;
  });

  await check("deleting the chat removes the merged worktree and keeps the unmerged one", async () => {
    const [merged] = ids.wtBranches;
    git(REPO, "merge", "-q", "--no-edit", merged);
    const r = await inv("delete_conversation", { id: ids.wtConv });
    assert(r.removed_worktrees.some((w) => w.branch === merged), `removed ${JSON.stringify(r.removed_worktrees)}`);
    assert(r.kept_worktrees.length >= 1 && r.kept_worktrees.every((w) => w.reason), `kept ${JSON.stringify(r.kept_worktrees)}`);
    const left = git(REPO, "branch", "--list", "las/*");
    assert(!left.includes(merged), `merged branch still there: ${left}`);
    assert(git(REPO, "worktree", "list").split("\n").length === 1 + r.kept_worktrees.length, git(REPO, "worktree", "list"));
    return `removed ${merged}; kept ${r.kept_worktrees.map((w) => `${w.branch} (${w.reason})`).join(", ")}`;
  });

  await check("worktree isolation in a non-git folder: clear error, no child started", async () => {
    await inv("set_local_roots", { roots: [PLAIN] });
    const conv = await convWith({ agent_mode: true, mode: "agent", builtin_tools: true, tool_servers: [], subagents: true, max_subagents: 2 });
    await inv("chat_send", { id: conv, content: SPAWN_PROMPT(true) + "\nIf the tool returns an error, stop and tell me the error text." });
    // After the error the model may try other things (which wait for approval): stop once the error is in.
    const c = await waitFor(async () => { const v = await inv("get_conversation", { id: conv }); return v.messages.some((m) => m.role === "tool") || !v.generating ? v : null; }, 300000, "spawn result");
    await inv("chat_stop", { id: conv }).catch(() => {});
    const results = c.messages.filter((m) => m.role === "tool").map((m) => m.content);
    assert(results.some((s) => /Not started/.test(s) && /not inside a git repository/.test(s)), `tool results: ${results.map((s) => s.slice(0, 160))}`);
    const subs = await inv("chat_subagents", { id: conv });
    const started = subs.subagents.filter((r) => parse(r.settings).worktree);
    assert(started.length === 0, `children started with worktrees: ${started.length}`);
    await inv("delete_conversation", { id: conv });
    return "refused with the git error";
  });

  await check("stopping a chat reply cancels its subagents and kills their commands", async () => {
    await inv("set_local_roots", { roots: [PLAIN] });
    const conv = await convWith({ agent_mode: true, mode: "agent", builtin_tools: true, tool_servers: [], subagents: true, max_subagents: 2 });
    const ap = t.autoApprove(null);
    // PowerShell processes in the app's own process tree (other tools on this PC run PowerShell too).
    const SHELLS_PS = `$all = Get-CimInstance Win32_Process; $kids = @{}; foreach ($p in $all) { $kids[[int]$p.ParentProcessId] += @($p) }; $n = 0; $q = [System.Collections.Queue]::new(); $q.Enqueue(${t.app.pid}); while ($q.Count) { $id = $q.Dequeue(); foreach ($c in $kids[[int]$id]) { if ($c.Name -eq 'powershell.exe' -or $c.Name -eq 'pwsh.exe') { $n++ }; $q.Enqueue([int]$c.ProcessId) } }; $n`;
    const shells = () => Number(execFileSync("powershell.exe", ["-NoProfile", "-Command", SHELLS_PS], { encoding: "utf8" }).trim() || 0);
    try {
      await inv("chat_send", { id: conv, content: 'Use spawn_subagents once with two subagents, labels "w1" and "w2". Each must run the shell command: Start-Sleep -Seconds 300 ; and then report "slept".' });
      await waitFor(async () => (await inv("chat_subagents", { id: conv })).subagents.filter((r) => r.state === "running").length >= 1 && shells() >= 1, 300000, "children running a shell");
    } finally { ap.stop(); }
    const t0 = Date.now();
    await inv("chat_stop", { id: conv });
    await waitFor(async () => !(await inv("get_conversation", { id: conv })).generating, 30000, "reply to stop");
    const subs = await waitFor(async () => { const s = await inv("chat_subagents", { id: conv }); return s.subagents.every((r) => !["running", "pending", "queued"].includes(r.state)) ? s : null; }, 30000, "children to stop");
    await waitFor(async () => shells() === 0, 20000, "shell processes to exit");
    await inv("delete_conversation", { id: conv });
    return `${subs.subagents.map((r) => `${r.label}:${r.state}`).join(", ")} in ${((Date.now() - t0) / 1000).toFixed(1)} s`;
  });

  fs.rmSync(REPO, { recursive: true, force: true });

  // ---------------- thinking controls + attachments (added when built) ----------------
  const THINK = path.join(import.meta.dirname, "extras-thinking.mjs");
  if (fs.existsSync(THINK)) {
    const mod = await import(`file://${THINK.replace(/\\/g, "/")}`);
    await mod.default(t);
  }
}

# Performance harness (PERF-1, PERF-2, PERF-3)

Scripts for the performance audit (`perf-audit-static.md`, section c). They launch the
**release exe** with an isolated profile under `%TEMP%` (installed runtimes are hard-linked,
`LOCALAISTUDIO_DATA_DIR` is set, first-run setup is skipped). They never touch your real
`%APPDATA%\LocalAIStudio` data. Plain Node 20+, no dependencies. Run from the repo root.

## Prerequisites
- Close Chrome and anything else using the GPU. Close Local AI Studio. No `llama-server.exe`
  and no `local-ai-studio.exe` may be running: each script exits with code 2 if one is.
- A built release exe, e.g. `target\release\local-ai-studio.exe`. Build the frontend first when
  testing a frontend change (`npm --prefix frontend run build`, then rebuild the exe).
- `chat-render.mjs` needs no model. `agent-prefill.mjs` needs a model `.gguf` under
  `LAS_MODELS_DIR` (default `%USERPROFILE%\LocalAIStudio\models`) and the GPU to itself.
- Exe path: `--exe <path>` or env `LAS_E2E_EXE`. CDP ports default to 19334 / 19335.
- Each script prints JSON lines (one per run; also appended to `--out` if given) and ends with a
  summary table. On exit it kills the app and any `llama-server.exe`, and deletes its temp profile.

## PERF-1 and PERF-2: `agent-prefill.mjs`
```
set LAS_E2E_EXE=%CD%\target\release\local-ai-studio.exe
node tests\bench\agent-prefill.mjs --model "%USERPROFILE%\LocalAIStudio\models\<repo>\<file>.gguf" --mode both --runs 3 --out %TEMP%\perf1.jsonl
```
Options: `--mode prefill|subagent|both` (default both), `--runs N` (3), `--mcp-config <json>`,
`--builtins-only`, `--runtime auto|<id>` (auto = newest CUDA runtime in the profile),
`--timeout-min` (30 per reply).

Recommended matrix (one model at a time; Gemma-4-E4B for fast prefill, then Cyber-Tiel-35B-A3B):
1. Built-ins floor: `--builtins-only`
2. Built-ins plus MCP: `--mcp-config <file>` where the file is an `mcpServers` JSON (same format
   the MCP page imports) holding the owner's rea and Roblox_Studio entries. Without the flag no
   MCP servers exist in the isolated profile. The script waits up to 4 min for all servers to
   connect and warns if they do not.
3. After a fix, repeat 1 and 2 with the new exe and compare.

What it does and prints:
- **prefill**: reads `chat_tool_costs` and `chat_context` for an empty Agent chat (tool count,
  tool tokens, system tokens, total), sends "hi" and reads `prompt_tokens`, `cached_tokens`,
  `time_to_first_token_ms`, `prompt_tokens_per_sec` from the stored assistant-message stats.
  Between runs the model is unloaded and reloaded (llama-server restarts, so the host prompt
  cache is empty). The table shows median and min-max over runs. Compare `prompt_tokens` with
  `chat_context total`: the gap is what the context meter does not see.
- **subagent**: Agent chat with "Allow subagents" on, `max_subagents` 2, a prompt that makes 2
  subagents each run 3 `echo` shell steps (approvals are auto-answered "allow_session"). It
  reads every model step from the chat's hidden run (`get_task`), prints per step
  `prompt_tokens / cached_tokens / time_to_first_token_ms` per agent, counts child follow-up
  steps with `cached_tokens` 0 (the PERF-2 symptom), and samples `get_resources().backend_memory_bytes`
  every 2 s (start, peak, end: host prompt cache growth, PERF-5). Shell commands run inside a temp
  folder that is registered as the only local root of the isolated profile.
  Model behaviour decides whether it really spawns two children; check `subagents` and `steps` in the
  JSON line (a run with no child steps means the model did not spawn, rerun or use a stronger model).

## PERF-3: `chat-render.mjs`
```
node tests\bench\chat-render.mjs --exe "target\release\local-ai-studio.exe" --runs 5 --out %TEMP%\perf3.jsonl
```
Options: `--runs 5`, `--lengths 2000,8000,32000`, `--chunk 8` (chars per event), `--window 25`
(events per measured window), `--direct`.

It creates a conversation over IPC, opens it on the Chat page, and sends a synthetic Markdown reply
(headings, lists, Rust code fence, table) as the same `chat://event` payloads the backend emits
(`segment_start`, `delta`, `done`; shape from `ChatEvent` in `src-tauri/src/services/chat.rs`),
through `plugin:event|emit` so Chat.tsx's real `listen` handler runs. For each length it seeds the
live text to that length, then replays 25 events at 25/s (1 s) and reads
`Performance.getMetrics` (TaskDuration, ScriptDuration, LayoutDuration, RecalcStyleDuration) before
and after. It also records long tasks over 50 ms (`PerformanceObserver`) and event-loop lag
(10 ms timer overshoot, a proxy for composer input latency). The baseline (same IPC traffic to an
unused event name) is subtracted: `net_task_ms` is the Chat page cost per update.
The table prints median (min-max) over runs per length. Busy fraction = net task ms / 40 ms.
Run it before and after the block-memo change with the same exe path pattern for a comparison.
A separate Markdown-component microbench was used during the audit; it is not part of this repository.

## Caveats
- The scripts were syntax-checked only (`node --check`). Not run against the app or a model yet.
- If React batches several deltas into one render under load, cost per update is per event, not per
  render (it understates cost per render when the thread is saturated; long tasks and lag show it).
- If `plugin:event|emit` is refused (capability), use `--direct`, which hooks the page's
  `plugin:event|listen` registration and calls the handler directly. Best effort, unverified.

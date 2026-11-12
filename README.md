# Local AI Studio

Local-first Windows desktop app: LM Studio-style model management and chat plus agent tooling (MCP, PowerShell, SSH/Docker/WSL sandboxes, subagents, skills). Rust + Tauri 2; the UI is React + TypeScript and only renders. All logic lives in the Rust crates.

Built in phases. Phase 1 (foundation) is done: hardware detection, llama.cpp runtime install, model library, load/unload with real settings, backend log.

## Layout

| Path | What |
|---|---|
| `crates/app-core` | Shared error type, app paths, ids |
| `crates/persistence` | SQLite (sqlx) with migrations for every spec table |
| `crates/system-monitor` | CPU/RAM (sysinfo) and NVIDIA GPU (NVML) metrics |
| `crates/security` | Windows Credential Manager wrapper, secret redaction |
| `crates/model-registry` | GGUF header parser, folder scan, shard grouping, memory estimates |
| `crates/inference` | llama.cpp runtime installer, llama-server supervisor (Job Object), HTTP client |
| `src-tauri` | Tauri shell and IPC commands |
| `frontend` | React UI |

App data: `%APPDATA%\LocalAIStudio` (database, runtimes, logs). LM Studio's `~/.lmstudio/models` is registered in place, read-only.

## Develop

Requirements: Rust (MSVC toolchain), Node 20+, WebView2 (ships with Windows 11).

```powershell
cd frontend; npm install; cd ..
.\frontend\node_modules\.bin\tauri dev       # run the app
cargo test -p <crate>                        # tests for one crate
cargo run -p inference --example smoke -- <model.gguf>   # real runtime + load/unload check
cd frontend; npm run build                   # type-check and build the UI
```

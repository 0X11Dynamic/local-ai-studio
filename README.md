# Local AI Studio

**Run local LLMs offline on Windows: a fast, private LM Studio alternative built on llama.cpp, with GGUF model management, chat, agents, MCP tools and a local OpenAI-compatible API.**

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](#license)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/0X11Dynamic/local-ai-studio/releases)
[![Platform: Windows 10/11 x64](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-0078D4)](#system-requirements)
[![Built with Rust + Tauri 2](https://img.shields.io/badge/built%20with-Rust%20%2B%20Tauri%202-orange)](#architecture)

Local AI Studio is a Windows desktop app for running large language models on your own PC. It downloads GGUF models from Hugging Face, loads them with [llama.cpp](https://github.com/ggml-org/llama.cpp), and gives you a chat UI, an agent mode that can use tools (with your approval), and a local OpenAI-compatible server for other programs. Everything runs on your machine. The installer installs for your user account only and needs no administrator rights. The source is available and free for noncommercial use (see [License](#license)).

It is written in Rust (Tauri 2) with a React/TypeScript front end. All logic lives in Rust crates; the UI only renders.

## Features

### Models
- **Model library** for your GGUF files. Models already downloaded by LM Studio are registered in place (no copies or moves).
- **Discover and download from Hugging Face**: exact file sizes, whether each file fits your GPU, and capability badges (vision, thinking, tools, audio). Downloads pause, resume and are verified by SHA-256.
- **Load panel** with a memory estimate before you load, and a report of what really happened after (layers on the GPU, buffers, backend log).
- **Auto context**: context length sized to your hardware.
- **Share with CPU**: split a model between GPU and CPU, including offloading MoE experts, so models larger than your VRAM still run.
- **Speculative decoding (MTP)**: turned on by default when the GGUF file has MTP layers and the runtime supports `draft-mtp`.
- **Runtimes page**: official llama.cpp builds (CUDA, Vulkan, CPU), checked against their published SHA-256. Pick one per load.

### Chat
- Streaming replies with **Markdown**, **reasoning** blocks and thinking controls.
- **Attachments**: text files, PDFs, and images for vision models.
- **Per-chat tools menu**, plus stop, regenerate, edit, rename, duplicate, search, export and bulk delete.
- **Chat / Plan / Agent modes.** Chat is plain conversation with no tools. Plan offers read-only tools. Agent can act, with approvals.

### Agents and tools
- **Agent tasks**: give an objective and the agent works on it by itself, with **subagents** (optionally in separate **git worktrees**), the ability to **ask you questions**, and reusable **skills**.
- **MCP client**: add local (stdio) or HTTP tool servers. Servers that are turned on connect when the app starts, and a chat can leave any of them out from its Tools menu.
- **Built-in tools**: shell and file tools that run on **This PC** (with approval outside allowed folders), in a disposable **WSL sandbox**, over **SSH**, or in **Docker**.
- **Web**: `web_fetch` and `web_search` (DuckDuckGo with no key, LangSearch, Brave, or your own SearXNG).
- **Permissions**: every tool is set to **Runs without asking**, **Ask every time** or **Never**.

### Everything else
- **Local OpenAI-compatible API**, off by default, listening on `127.0.0.1:1234` unless you change it.
- **First-run setup**: GPU and driver check, runtime install, a starter model that fits your hardware, and storage folders.
- **PC linking (beta)**: pair two PCs by IP and one-time code over an encrypted link, then split a model's layers across both GPUs or send whole jobs to the other PC. See [PC linking](#pc-linking-beta).

## Quick start

1. Download `Local AI Studio_x.y.z_x64-setup.exe` from the [Releases](../../releases) page.
2. Run it. Windows SmartScreen may show "Windows protected your PC" because the installer is not code-signed. Click **More info**, check the file name, then **Run anyway**. See the [FAQ](#faq).
3. On first launch the setup screen checks your hardware, starts installing the llama.cpp runtime that suits it, and offers a starter model sized for your GPU. The starter model downloads only when you click **Download**.
4. Open **Chat**, pick the model, and start talking. Use **Discover** to find more models.

To uninstall, use Windows **Settings > Apps**. Your chats, models and settings are kept unless you delete them. See the [user guide](docs/USER_GUIDE.md) for details.

## System requirements

- Windows 10 or 11, 64-bit.
- **GPU (recommended, not required):**
  - NVIDIA with the **CUDA 13** runtime needs driver 580 or newer.
  - **Vulkan** runtime for AMD, Intel, or NVIDIA with older drivers.
  - **CPU** runtime works anywhere, only slower.
- The Visual C++ runtime is bundled (Microsoft VC++ runtime DLLs ship with the app).
- Microsoft Edge WebView2 (preinstalled on Windows 11; the installer fetches it if missing).
- Internet access for setup and model downloads. Chatting afterwards works offline.
- Disk space depends on the models you download (starter models range from about 1 GB to several GB).

## Performance

Measured on one machine only: **RTX 5060 8 GB, Intel Core Ultra 7 265F, 32 GB RAM**, llama.cpp b12250 (CUDA), 8192 context, median of 5 runs, same GGUF files in both apps, LM Studio CLI build 43d4b26. Raw data: [`tests/bench/results-2026-11-11.json`](tests/bench/results-2026-11-11.json).

| Model | Generation, Local AI Studio vs LM Studio | Load time |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B Q6_K | 228 vs 198 tokens/s (+15%) | 1.3 s vs 1.9 s |
| Gemma-4-E4B-it Q4_K_M | 97 vs 88 tokens/s (+11%) | 2.6 s vs 5.7 s |
| Cyber-Ornith-1.5-9B Q4_K_S | 73 vs 68 tokens/s (+7%) | 3.1 s vs 4.1 s |

Time to first token was also lower in all three, and prompt processing was faster in all three. Treat all of this as one machine, one day, not a general guarantee.

Results depend heavily on GPU, driver, model, quantization and context length.

## How it compares

| | Local AI Studio | LM Studio | Ollama |
|---|---|---|---|
| Engine | llama.cpp (official builds) | llama.cpp based | llama.cpp based |
| Windows desktop UI | Yes | Yes | Not verified |
| Hugging Face GGUF search with fit estimates | Yes | Yes | Not verified |
| Local OpenAI-compatible API | Yes | Yes | Yes |
| Generation speed, same GGUF, RTX 5060 | +7% to +15% (measured, see above) | baseline | Not measured |
| Built-in agent tasks with subagents | Yes | Not verified | Not verified |
| MCP client | Yes | Not verified | Not verified |
| Split a model across two PCs | Yes (beta, VM-tested) | Not verified | Not verified |
| Source available | Yes (noncommercial license) | Not verified | Not verified |
| Code-signed installer | No (for now) | Not verified | Not verified |

"Not verified" means the author did not test or confirm it. Check the other projects' own docs before relying on this table.

## PC linking (beta)

Two PCs running Local AI Studio can act as one bigger computer. Pair them once with an address and a one-time 6-digit code, then either split a large model layer by layer across both GPUs or send whole jobs to the helper PC. The link is a direct app-to-app connection encrypted with the Noise protocol; keys are pinned after pairing. Each helper PC sets how much GPU memory, RAM and CPU it lends to each linked PC, and the helper itself refuses anything over that limit. It is labelled beta.

Measured on two real PCs linked over the internet (about 20 ms round trip, 64 Mbps upload): main PC RTX 5060 8 GB + 32 GB RAM, helper RTX 2060 SUPER 8 GB lending 5 GB, Qwen3.8-27B Q4_K_M with MTP speculative decoding, llama.cpp b12716:

| | Context (Auto) | Generation | Prompt processing |
|---|---|---|---|
| Main PC alone | 131072 | 8.6 tokens/s | 248 tokens/s |
| Split across both PCs | 196608 | 8.1 tokens/s | 57 tokens/s |

So over the internet a split kept generation speed about the same and gave 50% more context, but prompt processing was about 4x slower because the activations cross the upload link. A model that already fits on one PC is usually faster without splitting, so use it for models or contexts too big for one machine. The load panel estimates the split speed before you load (7.0-9.5 tokens/s for this case). Details in the [user guide](docs/USER_GUIDE.md) and [security notes](docs/SECURITY.md).

## Privacy

- Models run on your PC. Prompts and chats are stored locally and are not sent anywhere.
- Network use happens only when you do something that needs it: downloading a runtime or a model (Hugging Face, GitHub), searching Hugging Face, using web search or `web_fetch`, creating a WSL sandbox (it downloads an Alpine Linux image from `dl-cdn.alpinelinux.org`), connecting a remote MCP server or SSH host, linking PCs, or opening the local API to other devices. The installer fetches WebView2 if it is missing.
- No telemetry, analytics or automatic update checks were found in the source.
- Secrets (Hugging Face token, API keys, SSH passwords) are stored in Windows Credential Manager, not in the database. Logs have tokens and passwords removed.
- See [docs/SECURITY.md](docs/SECURITY.md) for the threat model.

## Architecture

A Cargo workspace of focused crates, with a thin Tauri shell and a React UI that only renders.

| Crate | Role |
|---|---|
| `app-core` | Shared error type, app paths, ids |
| `persistence` | SQLite storage with migrations |
| `system-monitor` | CPU, RAM and NVIDIA GPU metrics |
| `security` | Credential Manager wrapper, secret redaction |
| `model-registry` | GGUF parsing, folder scan, memory estimates |
| `inference` | llama.cpp runtime installer and `llama-server` supervisor |
| `downloader` | Resumable, hash-checked downloads |
| `local-api` | Local OpenAI-compatible server |
| `mcp-client` | Model Context Protocol client |
| `agent-runtime` | Tasks, subagents, approvals, ask-the-user questions |
| `execution`, `execution-ssh`, `execution-docker` | Shell and file tools on This PC, WSL, SSH and Docker |
| `skills` | Reusable agent instructions |
| `linking` | Encrypted PC-to-PC link |
| `src-tauri` | Tauri 2 shell and IPC commands |
| `frontend` | React + TypeScript UI |

Models run in a separate `llama-server` process from an official llama.cpp build, supervised inside a Windows Job Object so it stops with the app.

## Build from source

Requirements: Rust (MSVC toolchain), Node 20+, WebView2. The Microsoft Visual C++ 2015-2022 x64 Redistributable must be installed: the build copies `vcruntime140.dll`, `vcruntime140_1.dll` and `msvcp140.dll` from `System32` into `src-tauri/resources/vcrt/` (git-ignored) so the installer can ship them.

```powershell
# 1. Build the UI
npm --prefix frontend ci
npm --prefix frontend run build

# 2. Run in development
.\frontend\node_modules\.bin\tauri dev

# 3. Build the release executable
cargo build --release -p local-ai-studio --features tauri/custom-protocol

# 4. Build the installer (NSIS setup.exe)
.\frontend\node_modules\.bin\tauri build
```

## Testing

```powershell
cargo test --workspace                       # unit and integration tests
cargo test -p <crate>                        # one crate
npm --prefix frontend run build              # type-check and build the UI
```

End-to-end suites live in [`tests/e2e`](tests/e2e) (`phase1.mjs` to `phase6.mjs`, `firstrun.mjs`, `advanced.mjs`, `extras*.mjs`, `idle.mjs`, `installer.ps1`). They drive the real app over WebView2's DevTools protocol with plain Node 20+ and no test framework. Many need a GPU, a downloaded model, or an installed build.

## FAQ

**Do I need a GPU?** No. The CPU runtime works on any 64-bit Windows PC, just slower. A GPU makes a large difference. The first-run screen picks the right runtime for your hardware.

**Which models does it run?** Any GGUF model that llama.cpp supports, from Hugging Face or files you already have. Discover shows file sizes, a fit estimate for your GPU, and badges for vision, thinking, tools and audio.

**Does it send my data online?** Not for chatting. Network access is used only for downloads, searches, remote servers and linking, and only when you use them. Web search and the local API are separate switches. See [Privacy](#privacy).

**Why does Windows SmartScreen warn me?** The installer is not code-signed, so Windows does not recognise the publisher. Click **More info**, confirm the file name, then **Run anyway**. You can also build from source.

**Can I link PCs over the internet?** It is possible: "Check access from other networks" asks your router (UPnP) to open the port, or you forward it manually. It is beta, was tested between two real PCs over the internet (see [PC linking](#pc-linking-beta)) and in VMs, and you should only link PCs you trust. On a home network it needs no router changes.

**How is it different from LM Studio or Ollama?** See [How it compares](#how-it-compares). In short: llama.cpp underneath, a built-in agent and approval system with subagents and MCP, optional PC linking, and in our benchmark on one machine faster generation than LM Studio on the same files.

**Does it work on macOS or Linux?** No. It is Windows only for now.

**Can agents damage my PC?** Shell and file tools ask before changing anything outside allowed folders, system and credential folders are always blocked, and a WSL sandbox and Docker are available. SSH hosts are limited only by that account's rights. See [docs/SECURITY.md](docs/SECURITY.md).

## Contributing

Issues and pull requests are welcome. Please run `cargo test --workspace` and `npm --prefix frontend run build` before opening a pull request, and describe how you tested UI changes. Report security problems privately rather than in a public issue.

## License

Local AI Studio is released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).

- Free for noncommercial use, modification and redistribution.
- Commercial use requires the author's permission.
- Keep the Required Notice (credit) line: `Required Notice: Copyright 2026 0X11Dynamic (https://github.com/0X11Dynamic)`.
- This is **source-available**, not OSI-approved open source.

Third-party components keep their own licenses: llama.cpp, Rust and npm dependencies, and the bundled Microsoft VC++ runtime DLLs. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the full license text is in [`LICENSE`](LICENSE).

Copyright 2026 0X11Dynamic (https://github.com/0X11Dynamic)

# GitHub metadata (drafts)

Copy these into the repository settings when publishing. Everything here is a draft.

## License display

The license is PolyForm Noncommercial 1.0.0, which GitHub does not recognise, so the repository sidebar will show the license as "Other". That is expected. The README License section explains it. Do not use the topic `open-source`: this project is source-available, not OSI open source.

## Repository description (under 350 characters)

Local AI Studio: a source-available Windows desktop app to run local LLMs offline with llama.cpp. Download GGUF models from Hugging Face, chat, use agents with MCP tools and approvals, and serve a local OpenAI-compatible API. A fast, private LM Studio alternative, free for noncommercial use. Rust + Tauri 2.

## Topics (exactly 20)

1. local-llm
2. llama-cpp
3. gguf
4. llm
5. ai-agent
6. mcp
7. model-context-protocol
8. tauri
9. rust
10. react
11. typescript
12. windows
13. offline-ai
14. lm-studio-alternative
15. openai-compatible-api
16. huggingface
17. desktop-app
18. self-hosted
19. privacy
20. local-ai

## Social preview

- **Title:** Local AI Studio
- **Subtitle:** Run local LLMs offline on Windows. llama.cpp, GGUF, agents, MCP. A fast LM Studio alternative.
- Image: 1280 x 640 px; use the chat screenshot (`docs/images/chat.png`) over a plain dark background with the title above.

## Launch post drafts

### Reddit r/LocalLLaMA (draft)

**Local AI Studio: source-available Windows app for running GGUF models with llama.cpp (LM Studio alternative with agents + MCP)**

I built a Windows desktop app (Rust + Tauri 2) for running local models. Hugging Face search with fit estimates, Auto context, GPU+CPU split and MoE expert offload, MTP speculative decoding, chat with reasoning and attachments, agent tasks with subagents and approvals, an MCP client, and a local OpenAI-compatible API. Per-user installer, unsigned (SmartScreen warning). Free for noncommercial use (PolyForm Noncommercial, so source-available rather than OSI open source). On my RTX 5060 8 GB / Core Ultra 7 265F, generation was 7 to 15% faster than LM Studio on the same GGUFs, with faster loads. One machine only, so please test on yours. PC linking (split a model across two PCs) is in beta. Feedback welcome.

### Hacker News (draft)

**Show HN: Local AI Studio, a Windows app for running local LLMs with agents and MCP**

A Rust/Tauri desktop app around llama.cpp: model library, Hugging Face downloads, chat, approval-gated agent tools (PowerShell, WSL sandbox, SSH, Docker), MCP client, and a local OpenAI-compatible server. Chats stay on the PC. Network use happens only for model and runtime downloads, searches, web tools, remote MCP or SSH hosts, and PC linking, and only when you use them. Windows only, unsigned installer for now. Source-available, free for noncommercial use. I would like feedback on the permission model and on the benchmark methodology (raw JSON is in the repo).

### X (draft)

Local AI Studio: run local LLMs offline on Windows. llama.cpp + GGUF, Hugging Face downloads, agents with subagents and MCP tools, local OpenAI-compatible API. 7 to 15% faster than LM Studio on my RTX 5060 (one machine, same models). Source-available, free for noncommercial use. [link]

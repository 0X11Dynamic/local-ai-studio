# Local AI Studio safety model

The short version: the model can only use the tools you turn on, anything that changes something asks you first unless you allowed it, and nothing a tool returns can change those rules.

## Who decides

- **Permissions live in the app, not in the prompt.** Each tool is set to run without asking, ask every time, or never. You change that, and what you have allowed, only through the app's own pages. A model's text, a tool result or a file cannot.
- **Tool output is data.** Results, file contents and web pages are passed to the model as untrusted data, and it is told not to follow instructions inside them. Even if it does, it can only call tools you enabled, and changing tools still ask.
- **Approvals are explicit.** Allow once, allow for this chat or task, always allow, or deny. No answer within 10 minutes counts as deny. Grants are per environment and can be revoked.
- **Arguments are checked first.** Every tool call's arguments are validated against the tool's schema before anything runs.
- **Everything is logged.** Every tool call is in the Tool activity log with its arguments (secrets removed), result, decision and who approved it.

## Defaults

- Chats have no tools until you turn on Agent Mode.
- MCP tools ask every time. Imported MCP servers start turned off.
- Built-in tools: reading, listing and searching inside the allowed folders run without asking. Shell commands, writes, moves and deletes ask.
- The local API is off. When on, it listens on this PC only. The model-management endpoints always need a token, and LAN mode needs it for everything. Web pages from other sites are refused.

## This PC

- Commands run as you, never as administrator, through PowerShell `-EncodedCommand`, so no quoting tricks can change the command. Each command runs in a Windows Job Object, so a timeout or Stop kills everything it started.
- File tools are limited to the allowed folders. Paths are checked after resolving `..`, symlinks and junctions. Network (UNC) paths are refused.
- Always protected: Windows, Program Files, ProgramData, `.ssh`, `.gnupg`, the Windows credential folders, the app's own data and the model folders.
- File deletes go to the app's trash and can be restored.

## Sandbox, SSH and Docker

- **WSL sandbox:**
  - Windows drives are not mounted and starting Windows programs is turned off.
  - Commands run as an unprivileged user, because root inside a WSL distro could mount `C:` again.
  - Network is off unless chosen at creation.
  - **Limit:** WSL 2 distros share one Linux kernel, so this is strong isolation from your files but not a security boundary against a kernel exploit.
- **SSH:**
  - Strict host-key checking against the app's own `known_hosts`. An unknown key needs you to compare and trust its fingerprint; a changed key is refused.
  - Passwords and passphrases are kept in Windows Credential Manager.
- **Docker:**
  - No network by default; all Linux capabilities dropped except five that package managers and file ownership need (CHOWN, DAC_OVERRIDE, FOWNER, SETUID, SETGID); no privilege escalation (no-new-privileges); memory, CPU and process limits.
  - Only folders inside This PC's allowed folders can be mounted, and the Docker socket or pipe never.

## Agents, subagents and skills

- A subagent gets at most its parent's tools and a share of its parent's budget. Stopping a task stops all its subagents and their commands.
- Tasks have budgets for turns, tool calls, time and tokens, and a task interrupted by an app restart is never re-run automatically.
- Agents can only *propose* skill changes; you approve them. Scripts that come with a skill stay off until you enable each one, and they turn off again if the file changes.
- In text-format tool mode, only a well-formed command block on its own lines in the model's own reply can run. Blocks in quotes, code fences, tool output or files never run.

## Secrets

Hugging Face tokens, API tokens, MCP secrets, SSH passwords and passphrases are stored in Windows Credential Manager, never in the database or settings. Logs and the tool audit have tokens and passwords removed before they are written.

## Known limits

- **The app is not code-signed**, so Windows SmartScreen warns on first run.
- **A tool you set to "runs without asking" really does run without asking.** Keep commands, writes and deletes on "ask" unless you trust the environment, for example a disposable sandbox.
- **Your own MCP servers run with your permissions.** Only add servers you trust.
- **What runs on SSH hosts and in containers is limited only by that account's rights** (and the folders you set). The app cannot sandbox a remote machine.

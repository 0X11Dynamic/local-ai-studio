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
- web_fetch (built in, runs in the app) reads public web pages without asking. Local network addresses (localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, fc00::/7, ...) ask every time; the host is resolved and checked before connecting and again on every redirect, and the connection is pinned to the checked address. No cookies, no login headers, no proxy, 15 s, 2 MB.
- Plan mode offers only fs_list, fs_read, fs_search, web_fetch and the shell, and every shell command asks ("Plan mode: read-only command requested"); no "allow for this chat" applies while planning.
- The local API is off. When on, it listens on this PC only. The model-management endpoints always need a token, and LAN mode needs it for everything. Web pages from other sites are refused.

## This PC

- Commands run as you, never as administrator, through PowerShell `-EncodedCommand`, so no quoting tricks can change the command. Each command runs in a Windows Job Object, so a timeout or Stop kills everything it started.
- Inside the allowed folders each tool's own permission applies. With **Reach outside allowed folders: Ask each time** (the default, Environments > This PC), file tools and shell commands may use other paths too, but every call that reaches outside the folders asks first, whatever the tool's permission says, and the approval names the real path. "Allow for this chat/task" then covers that one tool in that folder (and below it) for that chat or task only; "Always" is narrowed to the same. **Off** keeps the tools inside the allowed folders.
- A shell command asks first when its working folder is outside the allowed folders. The app can't see which files a command touches once it runs, so a command allowed inside the folders can still change files elsewhere: check commands before allowing them.
- Paths are checked after resolving `..`, 8.3 short names, symlinks and junctions, so the approval shows where a path really goes. Network (UNC), device, `\\?\`, drive-relative and alternate-data-stream paths are refused.
- Always refused, even if you approve: Windows, Program Files, ProgramData\Microsoft, `.ssh`, `.gnupg`, the Windows credential folders, the app's own data (database, runtimes, settings, trash) and WebView profile, and the model folders. Writes, moves and deletes never touch `.git` internals, and whole drives and home folders are never moved or deleted.
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

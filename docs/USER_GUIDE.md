# Local AI Studio user guide

Local AI Studio runs AI models on your own Windows PC. You can chat with them, call them from other programs through a local API, and let them work as agents: they can read and edit files, run commands and use tool servers, but they ask you first before doing anything that changes something.

Nothing leaves your PC unless you choose to download a model, connect a remote tool server or an SSH host, or turn on network access for a sandbox or container.

## Install and first run

**What you need:** Windows 10 or 11 (64-bit) and an internet connection for setup. An NVIDIA, AMD or Intel GPU makes models much faster but is not required.

### Install

1. Run the installer, `Local AI Studio_x.y.z_x64-setup.exe`. It installs for your Windows account only (to `%LOCALAPPDATA%\Local AI Studio`) and needs no administrator rights.
2. Windows SmartScreen may show **"Windows protected your PC"**. That is expected: the app is deliberately not code-signed, so Windows doesn't recognise the publisher. Click **More info**, check that the app name is `Local AI Studio_x.y.z_x64-setup.exe`, then click **Run anyway**. (Your browser may also warn when you download it; choose **Keep**.)
3. Click through the installer. If Microsoft Edge WebView2 (the component that draws the app's window) is missing, the installer downloads and installs it by itself. Windows 11 already has it.
4. On the last page, **Create desktop shortcut** is ticked by default; leave it ticked to get a **Local AI Studio** icon on your desktop. A Start menu entry is always created. A silent install (`setup.exe /S`) creates both shortcuts.

### First-run setup

The first time the app starts, a setup screen opens with three steps:

1. **This PC**: your CPU, memory and GPU. For an NVIDIA GPU it also checks the driver: the CUDA 13 runtime needs NVIDIA driver **580 or newer** (NVIDIA ships CUDA 13.5 with driver 620). If your driver is too old, the screen says so, links to the NVIDIA driver download page (https://www.nvidia.com/en-us/drivers/) and offers the **Vulkan** runtime (uses the GPU, works with older drivers) or the **CPU** runtime instead. The app never installs drivers for you.
2. **Runtime**: the engine that runs models (an official llama.cpp build from GitHub, checked against its published SHA-256). The build recommended for your hardware starts installing by itself, with a progress bar and a Cancel button. **Other runtimes** lists the Vulkan and CPU builds.
3. **A first model**: one starter model that fits your hardware, for example Qwen3 4B Instruct (2.5 GB) for a GPU with 6-11 GB of VRAM, Qwen3 8B for 12 GB or more, and Qwen3 1.7B (1.1 GB) when models run on the CPU. Nothing downloads until you click **Download**; the download keeps going if you close the screen. **Browse other models** opens Discover instead.

Click **Finish** when done, or **Skip setup** to do it all later from **Runtimes** and **Discover**. Either way the screen doesn't open by itself again; **Settings > Setup > Run setup again** brings it back.

The app also, on first start: registers the models LM Studio already downloaded (if LM Studio is installed), in place and read-only (no copies), and installs three sample skills (Skills page).

### Uninstall

Open Windows **Settings > Apps > Installed apps**, find **Local AI Studio**, and choose **Uninstall** (or run `uninstall.exe` in the install folder). This removes the program, the desktop shortcut and the Start menu entry. Your chats, settings, runtimes and models are **kept** (see [Where your data lives](#where-your-data-lives)); the "Delete the application data" box in the uninstaller only clears the window's WebView2 cache. To remove everything, also delete `%APPDATA%\LocalAIStudio` and `%USERPROFILE%\LocalAIStudio`, and the `LocalAIStudio` entries in Credential Manager.

## The pages

**Work**
- **Chat**: talk to the loaded model. Replies stream as they are written. You can stop, regenerate, edit, rename, duplicate, search, export and delete chats. **Chat settings** (top right) holds the sampling settings, the system instruction for this chat, the context overflow rule, and Agent Mode. The bar under the messages shows how much of the model's context is used.
- **Agent Tasks**: give the agent an objective and let it work on it by itself (see "Agent tasks" below).
- **Subagents**: every subagent that tasks started, grouped by task, with its state and how much budget it used. You can stop a running one.

**Models**
- **Discover**: search Hugging Face for GGUF models. Each file shows its exact size and whether it fits your GPU. Downloads can be paused, resumed and cancelled, are checked against their sha256, and are registered automatically.
- **My Models**: your model library. Pick a model, set the load options (context length, GPU layers, KV cache type, flash attention and so on), check the memory estimate, then **Load**. After loading you see what really happened: layers on the GPU, buffer sizes, and the backend log.
- **Runtimes**: the llama.cpp builds the app runs models with (official builds, sha256-checked). Install or remove them; when loading a model you choose which one to use.

**Agent setup**
- **System Instructions**: the base instruction every chat gets, plus instruction profiles (built-in ones and your own). You can preview exactly what the model receives, with token counts.
- **Skills**: instructions the agent follows for a kind of work (see "Skills").
- **MCP Connections**: tool servers (Model Context Protocol). Add a program on this PC or an HTTP server, import an `mcpServers` file, turn each tool on or off, and set each tool to run without asking, ask every time, or never. The Permissions tab lists what you allowed; the Tool activity tab is the full log of tool calls.
- **Execution Environments**: where the agent's shell and file tools run (see "Environments").

**System**
- **Overview**: CPU, RAM, GPU and VRAM use, and what is loaded.
- **Linked PCs**: link another PC so a model can be split across both (see "Linking PCs").
- **Developer / API**: the local OpenAI-compatible server (see "Local API").
- **Settings**: theme, storage folders (models and runtimes; setup asks for them too), parallel downloads, your Hugging Face token, folders the app uses, and this guide.

## Agent Mode (in a chat)

Turn on **Agent Mode** in Chat settings to let the model use tools in that chat. Off means a plain chat: no tools are sent at all.

In Agent Mode the model gets:
- the built-in shell and file tools for the environment you pick ("Run them on"), if **Built-in shell and file tools** is ticked
- the tools of the MCP servers you tick (none ticked means every connected server)
- matching skills, if **Use skills** is ticked.

**Web search.** The `web_search` tool (on by default in Agent and Plan chats and in tasks) lets the model look things up, then read the best pages with `web_fetch` and cite them. It runs without asking. Choose the provider in **Settings > Search**: DuckDuckGo (default, no key), LangSearch or Brave Search API (free keys, stored in Windows Credential Manager) or your own SearXNG. **Test** runs one query. The model may search 20 times per reply or task (a setting). Untick **Web search** in the chat's Tools menu or the task's tool list to turn it off there.

**Tool calls** sets how the model calls tools:
- **Automatic**: native tool calls when the model's chat template supports them, otherwise the text format.
- **Native**: always native tool calls.
- **Text format**: for models without tool calling. The model writes a `<command tool="...">` block. Only a block on its own lines in the model's reply counts. Blocks inside code fences, quotes, sentences, tool output or files are ignored, and a malformed block is reported back to the model instead of being run.

A chat reply in Agent Mode stops at its tool-call limit and time limit (Chat settings).

## Permissions and approvals

Every tool has a permission: **runs without asking**, **ask every time**, or **never**.
- Built-in defaults: listing, reading and searching files run without asking (inside the allowed folders). Running commands, writing, patching, moving and deleting ask every time.
- MCP tools ask every time until you change them.

When a tool needs approval, its card in the chat (or task) waits with four choices:
- **Allow once**: this call only.
- **Allow for this chat** / **Allow for this task**: this tool, in that chat or task only, until you revoke it on the MCP Connections page (Permissions tab) or delete the chat or task. A task's grant also covers its subagents.
- **Always allow this tool**: from now on. You can revoke it under MCP Connections > Permissions.
- **Deny**: the tool does not run, and the model is told not to retry or work around it.

Nobody answering for 10 minutes counts as Deny. Stopping a reply or task also denies a waiting approval.

Permissions are per environment: always allowing the shell in a sandbox never allows it on This PC.

Nothing a tool returns can change a permission. Tool output, file contents and web pages are treated as data. If they contain instructions such as "ignore previous instructions", the model is told not to follow them, and the app ignores them regardless.

## Environments

Pick one per chat (Chat settings) or per task.

**This PC** (built in)
- Commands run in PowerShell as you (never as administrator), inside a Windows Job Object, so a timeout or Stop kills the whole process tree.
- File tools only work inside the **allowed folders** (default `%USERPROFILE%\LocalAIStudio\workspace`). Change them on the Execution Environments page.
- These are always off limits, whatever the allowed folders say: Windows, Program Files, ProgramData, your `.ssh` and `.gnupg` folders, the Windows credential folders, the app's own data, and your model folders.
- Deleting with the file tool moves things to the app's trash (`%APPDATA%\LocalAIStudio\trash`), so you can restore them.

**Sandbox** (disposable Linux)
- A fresh Alpine Linux distro on WSL 2. The image (about 4 MB) is downloaded once and checked against its sha256.
- It cannot see your Windows drives or start Windows programs.
- Commands run as an unprivileged user.
- Network access is off unless you tick it when you create the sandbox.
- Removing it deletes the distro and everything in it. Your other WSL distros are never touched.

**SSH host**
- Sign in with a key file or a password. Passwords and key passphrases are stored in Windows Credential Manager, never in the app database.
- Before the first connection, click **Check host key**, compare the fingerprint with the server's own (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the server), and trust it. If the key later changes, the app refuses to connect.
- You can limit file tools to folders on the server. Tick **Windows host** for a Windows server (PowerShell).

**Docker**
- Needs Docker Desktop (or another Docker engine you point it at).
- Each environment gets one container with these defaults: no network, all Linux capabilities dropped except five that package managers need (CHOWN, DAC_OVERRIDE, FOWNER, SETUID, SETGID), no privilege escalation, 2 GB memory, 2 CPUs and 512 processes.
- Folders from this PC can be mounted, but only from This PC's allowed folders. The Docker pipe or socket can never be mounted.

**Try a command** on the environment page runs a command through the same permission check as the agent.

## Agent tasks

A task works toward one objective on its own: it reads, runs commands and edits files with the tools you allow, checks its work, and ends with a final answer.

- **New task**: write the objective (say which folder and how to check the result), pick where it runs, and set the budgets: model turns, tool calls, time and tokens. Defaults are 40 turns, 60 tool calls, 30 minutes and 60,000 tokens.
- The timeline shows every model step and tool call live. Approvals appear inline, labelled with which agent is asking.
- **Pause** takes effect after the current step. **Resume** continues from where it stopped. A task that ran out of budget can be resumed with more.
- **Cancel** stops the model and any running command, including the subagents' commands.
- If the app closes while a task runs, the task shows as **interrupted** next time. It is never restarted by itself; click Resume.

**Subagents** (tasks only)
- With **Allow subagents** on, the agent can start subagents for independent parts of the job. Each gets its own objective, a subset of the agent's tools (never more), and a share of the remaining budget (a third, or less when it starts more than three).
- Their usage counts toward the task's budget, and their answers come back to the main agent.
- At most 2 run at once by default (up to 4).
- You run one model on one GPU, so subagents take turns generating. They mostly save time while commands run. Loading the model with 2 parallel slots lets two generate at once, but each slot then gets half the context.

## Skills

A skill is a folder in `%APPDATA%\LocalAIStudio\skills` with a `SKILL.md`. That file has a short header (name, description, version, keywords) and the instructions.

- **Matching**: for each task (and each Agent Mode message), up to 3 skills are picked by matching the objective's words against each skill's keywords, name and description. The task log says which were picked and why, for example "matched: rust, test". Use **Try an objective** on the Skills page to check.
- **Mode**: auto (picked when it matches), always (every time), or off.
- **Revisions**: every save is kept, and Revert restores an old version.
- **Agent edits**: an agent can only **propose** a change. It waits on the Skills page until you approve or reject it.
- **Scripts**: extra files in a skill folder (scripts) start **off**. Turn each one on after reading it. If the file changes later, it turns off again. The agent can only run an enabled script, and still through the normal shell approval.
- **Import**: a `.zip` or a folder. Archives that try to write outside the skill folder are refused. **Export** creates a `.zip`. **Delete** moves the folder to the app's trash.
- **Samples**: `rust-test-fix`, `git-commit-style` and `safe-powershell` are installed on first run. Edit or delete them freely; a deleted sample is not reinstalled.

## Local API

**Developer / API** starts an OpenAI-compatible server for other programs.
- It is off by default. When on, it listens on `127.0.0.1:1234` (this PC only) unless you change the address.
- Endpoints: `/v1/models`, `/v1/chat/completions` (streaming or not) and `/v1/completions`, plus `/api/models`, `/api/models/load` and `/api/models/unload`.
- The `/api` endpoints always need the API token shown on the page. Opening the server to your network (LAN) also makes every endpoint need it.
- Requests from web pages are refused unless their origin is on your allow list.
- Every request is logged on the page.

## Linking PCs

Two PCs with Local AI Studio can act as one bigger computer: a model too big for one PC is split, layer by layer, across both. Nothing else to install; the link is direct and encrypted by the app.

**Link them once**
1. On the PC that will help: **Linked PCs** > **Link this PC** > **Show linking code**. It shows this PC's address, the port it is really listening on (default 47070) and a 6-digit code that works once, for 10 minutes.
2. On the other PC: **Linked PCs** > **Link to another PC**: type that address and the code, then **Link**. Leave the port empty unless the helper shows another one: the app tries 47070, then 50070 (the default of older versions). Five wrong codes lock it; make a new one.
3. Both PCs now trust each other and reconnect without a code. Windows may ask whether Local AI Studio can use the network: allow it on private networks.

**Lend a PC**: on the helper turn on **Let this PC help**. Choose which devices to share, CPU threads, and the most memory to lend. It is off after every restart, and other PCs can only use this PC while it is on. The helper needs the same llama.cpp runtime build as the main PC (Runtimes page).

**Split a model**: in the linked PC's row, **Split model layers with this PC** is on for PCs on the same network and off for other networks (you can turn it on anyway). In **My Models**, tick **Use linked PCs** and load. The app splits the layers by each device's free memory. Every generated token waits for one network round trip (measured: about 1 ms per token for each ms of round trip), so the page shows the round trip and an estimate next to this PC's own speed. After loading, the load panel shows the measured speed; if it's slower than this PC alone, it suggests turning splitting off (it never does that by itself). Splitting helps most for models that don't fit on one PC.

**Other networks**: on the helper click **Check access from other networks**. The app asks your router to open the port (UPnP) and shows your public address. If the router refuses, it lists the port-forward steps. If your internet provider shares one public address between customers (CGNAT), PCs on other networks can't connect directly, and the app says so. Over the internet each token waits 30-80 ms, so splitting is slow there.

**Unlink** on either PC. Linking again needs a new code.

## Troubleshooting

- **"VRAM spilled into shared memory"**: the NVIDIA driver quietly moved part of the model into system RAM. That makes it much slower instead of failing. Lower the context length, use a quantized KV cache (q8_0), offload fewer layers, or pick a smaller quantization. **Auto** GPU offload picks a fit for you.
- **Out of memory when loading**: same fixes. The memory estimate on My Models shows what fits before you load.
- **"Port 1234 is already in use"**: LM Studio's server also uses 1234. Stop LM Studio's server, or pick another port on Developer / API.
- **"The model's chat template does not support tool calling"**: set Tool calls to Automatic or Text format in Chat settings, or load a model with tool calling (for example Qwen3 or Llama 3.1 and newer).
- **A tool keeps asking**: choose "Allow for this chat/task" or "Always allow this tool", or change its permission on MCP Connections or Execution Environments.
- **Sandbox will not start**: WSL must be installed (`wsl --install --no-distribution`) and virtualization enabled in the BIOS.
- **Backend exits at once with 0xC0000135 / missing vcruntime140.dll**: a fresh Windows lacks the Visual C++ runtime. The app copies vcruntime140.dll, vcruntime140_1.dll and msvcp140.dll next to each runtime and retries once; if it still fails, install the Microsoft Visual C++ 2015-2022 Redistributable (x64).
- **Linked PC: "nothing is listening"**: the other PC answered, but Local AI Studio there isn't running, or isn't showing a code or helping. Open Linked PCs there.
- **Linked PC: "no answer"**: the other PC can't be reached: it's off or asleep, the address or port is wrong, or its firewall or router blocks the port (default 47070; from another network the port must be forwarded on its router).
- **"Port ... is reserved by Windows or already in use"** (on the helper): after a reboot, Hyper-V, WSL, Docker or VirtualBox can reserve blocks of ports (50060-50159 was measured on one PC, which blocked the old default 50070). Check with `netsh interface ipv4 show excludedportrange protocol=tcp`. The default is now 47070, below that dynamic range. If the chosen port is still reserved or used, the app listens on the next free port from 47071 to 47099 and shows it (and in the linking code view); type that port on the other PC. A port stored as 50070 by an older version is moved to 47070 once, with a notice.
- **Linked PC: "accepted the connection but closed it"**: something other than Local AI Studio answers on that port, usually a router or VirtualBox port forward with nothing behind it. Check the app is open there.
- **Linked PC: "isn't helping right now"**: turn on Let this PC help on that PC, then load again.
- **Linked PC: "identity key is not the one pinned"**: the other PC was reinstalled, or something is intercepting the connection. Unlink and link again with a new code.
- **Linked PC: "Both PCs need the same runtime build"**: install the same llama.cpp runtime (for example b12250) on both PCs on the Runtimes page.
- **Logs**: `%APPDATA%\LocalAIStudio\logs\app.log` (passwords and tokens are removed before writing).

## Where your data lives

| What | Where |
|---|---|
| Database (chats, tasks, settings, audit log) | `%APPDATA%\LocalAIStudio\app.db` |
| llama.cpp runtimes | the Runtimes folder in Settings > Storage (new installs: `runtimes` in the install folder; older installs: `%APPDATA%\LocalAIStudio\runtimes`) |
| Downloaded models | the Models folder in Settings > Storage (new installs: `models` in the install folder; older installs: `%USERPROFILE%\LocalAIStudio\models`) |
| Which folders you chose | `%APPDATA%\LocalAIStudio\storage.json` |
| Skills | `%APPDATA%\LocalAIStudio\skills` |
| Trash (deleted files and skills) | `%APPDATA%\LocalAIStudio\trash` |
| Sandboxes | `%APPDATA%\LocalAIStudio\sandboxes` |
| SSH host keys you trusted | `%APPDATA%\LocalAIStudio\known_hosts` |
| Weights a linked PC sent (helper cache) | `%APPDATA%\LocalAIStudio\rpc-cache` |
| Logs | `%APPDATA%\LocalAIStudio\logs` |
| Downloaded models | `%USERPROFILE%\LocalAIStudio\models` (changeable in Settings) |
| Agent workspace on This PC | `%USERPROFILE%\LocalAIStudio\workspace` (changeable) |
| Passwords, tokens, API keys | Windows Credential Manager (entries starting with `LocalAIStudio`) |

**Backup**: close the app, then copy `%APPDATA%\LocalAIStudio` (skip `runtimes`, which can be reinstalled) and your models folder. Secrets in Credential Manager are not in the copy; re-enter them after a restore.

Uninstalling the app (Windows Settings > Apps) removes the program but keeps this data.

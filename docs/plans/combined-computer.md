# Plan: linking Will's computers into one "combined computer"

Status: **plan only, for Will to approve.** Nothing is built yet. Written 2026-11-11, updated 2026-11-11 with Will's answers.

## Decisions (2026-11-11)

- **Second computer: another Windows PC**, not the MacBook. Both machines are Windows, so one runtime build and one helper build cover both. (A Mac could join later as a Metal helper; that is not planned now.)
- **Network: both PCs on the same home Wi-Fi.** Will: "At home they use WiFi. When I bring this pc home, they'll both use WiFi." So the measurements happen once this PC is at home, and Wi-Fi is the case to design for.
- **No Tailscale** (Will: "I'd rather not use tailscale so it's easy for other users to use it"). The app must bring its own secure link: pairing with a one-time code, then an app-to-app encrypted tunnel with pinned keys, with rpc-server bound to localhost behind it (see "Pairing in the app"). Nothing extra to install, and rpc-server is never reachable on the network directly.
- **Expectation on Wi-Fi:** Option A (one model split across both PCs) sends activations for every token over the network, so on Wi-Fi it is likely slower than one PC for models that fit, and possibly only worth it for models too big for one PC. Option B (whole chats, subagents and file jobs on each PC) tolerates Wi-Fi well. Phase 1 measures both before anything is built.
- **Direction (2026-11-11):** Will: "for bigger models to be offloaded to the other pc to help split the layers up to optimize speed. we can experiment". So: models too big for one PC are split across the PCs (Option A); models that fit run on one PC, and whole jobs (subagents, other chats, file batches) go to the other PCs (Option B); other networks get Option B only.
- **Next session (2026-11-12):** Will connects to the other PC with TeamViewer and Claude drives it with computer control to run phase 1 (copy the runtime, start rpc-server, measure the 27B split vs this PC alone).
- **Not answered yet:** the helper PC's GPU/VRAM and RAM (to be read during the TeamViewer session), and which big models he wants to run.

## Short answer

- **It's possible with what we already ship.** The llama.cpp runtime in the app includes an RPC server (`ggml-rpc-server.exe` plus `ggml-rpc.dll`). It lets one model run on both PCs at once, as one bigger computer.
- **It mainly buys room, not speed.** It runs models that don't fit on one PC. For models that already fit on one PC it is usually *slower* per token, because every generated token crosses the network. Public tests show this: prompt processing can speed up while generation slows down.
- **So the plan:**
  1. Measure Option A, "one model across computers", by hand on Will's two PCs.
  2. Build it into the app if it beats one PC for big models.
  3. Use Option B, "spread whole jobs across computers", for agent swarms and mass file reading, and as the fallback where A is slower than one PC.

That follows Will's rule: load balancing is only the fallback, used when the combined computer isn't possible or isn't good enough.

## What I checked on this PC (2026-11-11)

Run read-only (`--help` only, no model loaded) from the runtime we ship, `%APPDATA%\LocalAIStudio\runtimes\b12250-cuda-13.5` (a `b12250-cpu` folder is also there and also has `ggml-rpc-server.exe`):

- **`ggml-rpc-server.exe` options:** `-H/--host` (default `127.0.0.1`), `-p/--port` (default `50052`), `-d/--device` (which local GPU or CPU to offer), `-t/--threads` (default 10 here), `-c/--cache` (keeps tensors on the helper's disk so they don't cross the network again on the next load). On start it found the RTX 5060 (compute 12.0, 8,123 MiB) and loaded the CUDA, RPC and CPU backends.
- **`llama-server.exe` options:** `--rpc host:port,...`, `-dev/--device`, `--list-devices`, `-ngl/--n-gpu-layers`, `-sm/--split-mode {none,layer,row,tensor}`, `-ts/--tensor-split N0,N1,...` (proportions, e.g. `3,1`), `-fit/--fit [on|off]` (default on) with `-fitt/--fit-target` per-device margin.
- **`llama-bench.exe` options:** `-rpc/--rpc`, `-ts/--tensor-split` (written with slashes, e.g. `1/1`), `-ngl`, `-p`, `-n`, `-r` (default 5 repetitions).
- **What the upstream README says** (llama.cpp `tools/rpc`): weights and the KV cache are split across devices in proportion to their free memory, and the RPC backend is a **proof of concept, "fragile and insecure"**, never to be run on an open network.

## Goal (Will's words)

- **The main goal:** "connect the 2 computers to just act as a general stronger singular computer just like speedify". Speedify bonds connections: each link carries a share sized to its measured speed, the shares are re-measured live, and a dropped link fails over.
- **The fallback:** "the load balancing is only if it's not possible for us to do the giant computer idea or it doesn't work as good as we'd want".
- **Also use the PCs for:** agent swarms and mass file reading, "with the speedify type offloading if possible".

## Option A: one model across computers (the main goal)

### How it works

- **Main PC:** runs the app and `llama-server`, as today. The model file only needs to be on the main PC; the weights are sent to the helper at load (and cached there with `-c`).
- **Helper PC:** runs `ggml-rpc-server` from the **same llama.cpp build (b12250)**. By hand in phase 1; later as a headless "helper mode" of Local AI Studio.
- **Helper requirements:**
  - The same runtime build. Copy the `b12250-cuda-13.5` folder (NVIDIA GPU) or `b12250-cpu` (no NVIDIA GPU) to the helper, or install the app there. Mixing the CUDA and CPU variants of the same build should work (*expected*, not tested).
  - For CUDA: the helper's own NVIDIA driver, new enough for CUDA 13.5. The runtime folder doesn't install drivers.
- **Connecting them:** `llama-server --rpc <helper>:50052` with a `--tensor-split` (by hand in phase 1, calculated by the app later). In phase 1 straight over the home Wi-Fi with the lock-down in "Security" below; from phase 2 only through the app's own encrypted tunnel.

### Security (rpc-server has no password or encryption)

Anyone who can reach the RPC port can use the helper's GPU and send it data, and a bug in rpc-server could be abused. So:

- **In the app (phase 2 on): the app's own tunnel.** Will chose no Tailscale, so the app pairs the PCs with a one-time code and carries RPC traffic through a mutually authenticated, encrypted app-to-app connection (pinned keys). rpc-server listens on localhost only on the helper.
- **Phase 1, by hand on the home Wi-Fi** (trusted network, only while measuring):
  1. Bind only to the helper's LAN IP (`-H 192.168.x.y`), never `0.0.0.0`.
  2. Make the helper's network profile **Private**, and give both PCs fixed IPs (a DHCP reservation on the router) so the rule below stays correct.
  3. Allow the port only from the main PC, in an admin PowerShell on the helper:
     `New-NetFirewallRule -DisplayName "LocalAIStudio RPC" -Direction Inbound -Protocol TCP -LocalPort 50052 -RemoteAddress <main-pc-ip> -Action Allow -Profile Private`
  4. If Windows shows its own "Allow access" prompt for `ggml-rpc-server.exe`, don't allow it (that opens the port to the whole network). Check afterwards with `Get-NetFirewallApplicationFilter -Program *ggml-rpc-server.exe | Get-NetFirewallRule` and remove any broad rule it made.
  5. **Only run rpc-server while linked.** Close it (Ctrl+C) when done; remove the rule with `Remove-NetFirewallRule -DisplayName "LocalAIStudio RPC"` if the link isn't used for a while.
  - Traffic on the LAN is unencrypted: anyone on the home network could read prompts and answers in transit.

### Pairing in the app (phase 2, no open ports to strangers)

1. On the helper: "Let this computer help" → it shows a **one-time 6-digit code** for 5 minutes.
2. On the main PC: Execution / Compute → "Add a computer" → enter the code.
3. The two exchange long-term keys (stored in each machine's credential store) and pin each other's keys.
4. Either side can revoke it. A helper serves only paired PCs, and only when Will has it switched on.
- **Transport:** an app-to-app TLS tunnel using the pinned keys (mutual auth), with rpc-server bound to localhost behind it. The helper checks the peer's key before forwarding a single byte to rpc-server. (No Tailscale: Will wants it easy for other users.)
- **Finding the helper:** mDNS on the LAN ("helper available"), or by typing its IP. Discovery only lists; pairing still needs the code.

### Deciding the split (the Speedify part)

- **Inputs per device:** free VRAM / RAM (from the helper's hardware report), measured speed (a short calibration run per device, cached per model), and network round-trip time and bandwidth (measured on connect).
- **Default split:** proportional to free memory, as llama.cpp does, then weighted by measured speed. A Wi-Fi helper gets fewer layers, or is left out if it would make the whole thing slower than one PC.
- **The plan is shown before loading**, for example: "Model 27B: 40 layers here (RTX 5060), 25 on the second PC — expected X tok/s". X is an estimate from calibration, labelled as one.
- **The `-c` cache is on for helpers,** so later loads don't re-send the weights.

### Live re-measuring and failover

- **While running:** tokens/s, per-token latency and each node's time share are shown on a dashboard.
- **Rebalancing:** if a node slows down for several minutes (for example Wi-Fi drops), the app offers to rebalance. A new split needs a reload; llama.cpp can't move layers live.
- **If the helper drops:**
  1. The current reply fails clearly, keeping its partial text (the same handling as a backend crash today).
  2. The app reloads the model without the helper if it still fits, and the chat continues. Agent tasks resume from their saved transcript, as they already do after a crash.
  - Expect the pause of a model reload (seconds to tens of seconds, depending on size; *estimate*).

### Where Option A helps and where it doesn't

- **Helps:** models too big for one PC. Qwen3.8 27B Q4_K_M (15.6 GB) doesn't fit in the 5060's 8 GB; today it spills into shared memory or runs partly on the CPU. Split between the 5060 and the second PC's GPU it could stay all on GPUs, if the second GPU has roughly 9 GB or more free (*estimate*; depends on context size). A helper with less VRAM can also offer its CPU/RAM (`-d`), which is slower.
- **Doesn't help:** models that already fit on one GPU. Each token needs a network round trip per device boundary, so generation usually gets **slower**.
- **What public tests show** (other hardware; only indicative): two RTX 3090s gave ~50 tok/s in one machine vs ~37 tok/s split across two machines over 10 GbE; a Mac Studio + DGX Spark over 10 GbE gave 4.2× faster prompt processing but slower generation; Wi-Fi was much slower than gigabit Ethernet.
- **Network floor:** wired gigabit Ethernet is the practical minimum. Wi-Fi is expected to work but be slow (*estimate*; to be measured).

## Option B: spread whole jobs across computers (fallback, and for swarms)

### How it works

- Each PC loads **its own copy** of a model that fits it, as a full `llama-server`, reached through the same paired connection. The helper runs llama-server instead of rpc-server.
- The main app treats the helper as extra **generation slots**:
  - **Agent swarms:** each subagent goes to the least-busy PC. The existing generation gate becomes a pool of slots across machines.
  - **Mass file reading or summarising:** files are split into batches across PCs, and results are merged on the main PC.
  - **Chats:** stay on the main PC by default. They go to the helper only if Will picks it, or if the main PC is busy and the helper has the same model.
- **Tools always run on the main PC** (or the chosen environment), never on the helper, so permissions, approvals and the audit stay in one place. Helpers only generate text.

### When B wins

- Many independent jobs at once: throughput can scale close to the number of PCs, because each runs at full single-machine speed (*expected*; to be measured).
- Wi-Fi: only short requests and answers cross the network, not every token's activations, so B tolerates Wi-Fi far better than A.

### What B can't do

- One chat can't get faster than one machine.
- A model too big for both PCs on their own still needs A.
- Each PC needs the model file. It's copied once and checked by sha256.

## Computers on other networks (over the internet)

Will (2026-11-11): "see if theres a way we can do computers on other networks if we make it so a process runs in the background that runs a local web server or sum and then u connect to it by putting in the computers public ip or sum then port".

**Yes, that works, with three conditions.**

- **How:** the helper PC runs the app in "helper mode" as a small background service. It listens on one port with the app's own TLS (pinned keys from pairing). The main PC connects to `<public IP>:<port>`, or to a name if the helper's IP changes (dynamic DNS). rpc-server and llama-server stay on localhost on the helper; only the app's authenticated tunnel faces the internet.
- **Condition 1, the helper's router must let the connection in:**
  - The app tries UPnP / NAT-PMP first, which opens the port automatically on most home routers.
  - If that fails, the user forwards the port by hand in the router settings; the app shows the exact port and the helper's local IP.
  - If the helper's internet uses CGNAT (common on mobile hotspots and some ISPs), inbound connections are impossible. The app detects this (its public IP doesn't match the router's WAN IP) and says so. The only fix then is a relay server both PCs connect out to. That needs a small always-on server somewhere, which costs money, so it's not in the plan unless Will wants it.
- **Condition 2, it must be safe on the open internet:**
  - Pairing with a one-time code, then mutual TLS with pinned keys. Anyone without the key is dropped before any model code sees a byte.
  - Rate limiting, a per-helper on/off switch, and "only while the app says Helper is on".
  - The helper decides which models and how much of its GPU it shares.
- **Condition 3, what's fast enough over the internet:**
  - **Option A (one model split across both PCs) is not practical over the internet.** Every generated token waits on a network round trip; at 20-80 ms per trip that's a few tokens per second at best. Keep A for the same Wi-Fi only.
  - **Option B (whole jobs on the helper) works well over the internet.** The helper runs its own model; the main PC sends a chat turn or a subagent's job and gets the answer streamed back. That's the same as calling a remote OpenAI-compatible API: only text crosses the network. This is how a friend's PC could help.
- **Build order:** this is phase 2 work (helper mode + pairing + tunnel). The LAN version and the internet version are the same code; the internet adds UPnP, the CGNAT check and dynamic DNS.

## Which one when (the rule the app will follow)

| Situation | Use |
|---|---|
| Model fits on the main PC, one chat | Main PC alone (fastest) |
| Model too big for the main PC, helper on Ethernet | **Option A** |
| Model too big, helper only on Wi-Fi | Option A if measured faster than CPU offload on one PC; otherwise one PC with Share-with-CPU |
| Many subagents or a batch of files | **Option B** (each PC takes jobs) |
| The helper drops | Fail the current reply cleanly, reload or route without it |

Pick automatically, show the choice, and let Will override it.

## Build phases (after approval)

### Phase 1: measure by hand (about 1 session, no app changes)

Decides whether A is worth automating for Will's two PCs.

1. **Prepare the helper:** copy the matching runtime folder (`b12250-cuda-13.5`, or `b12250-cpu` if it has no NVIDIA GPU); for CUDA, update its NVIDIA driver. Find both PCs' IPv4 addresses with `ipconfig`. Add the firewall rule from "Security".
2. **Start the helper** (in its runtime folder):
   `.\ggml-rpc-server.exe -H <helper-lan-ip> -p 50052 -c`
   It prints the devices it offers. Add `-d CUDA0` or `-d CPU` to choose one.
3. **Check the link from the main PC:** `Test-NetConnection <helper-ip> -Port 50052` (should say `TcpTestSucceeded : True`), and `ping <helper-ip>` for round-trip time.
4. **See the device order on the main PC:**
   `& "$env:APPDATA\LocalAIStudio\runtimes\b12250-cuda-13.5\llama-server.exe" --rpc <helper-ip>:50052 --list-devices`
   `--tensor-split` values follow this order.
5. **Benchmark** (tok/s for prompt and generation, 5 runs each), first the main PC alone, then split:
   `llama-bench.exe -m <model.gguf> -ngl 99 -p 512 -n 128 -r 5`
   `llama-bench.exe -m <model.gguf> -rpc <helper-ip>:50052 -ngl 99 -ts 1/1 -p 512 -n 128 -r 5`
   Adjust `-ts` towards each device's free memory (e.g. `8/12` for an 8 GB and a 12 GB GPU).
6. **Real use check:** run the server the way the app would, and time a chat through `tests/bench/compare.mjs` (same prompts, temperature 0, 5 runs, median):
   `llama-server.exe -m <model.gguf> --rpc <helper-ip>:50052 -ngl 99 --tensor-split 1,1 --port 9931`
7. **Models to measure:** one that fits on the main PC (Cyber-Ornith 9B Q4_K_S) and one that doesn't (Qwen3.8 27B Q4_K_M; on one PC with Share-with-CPU as the baseline). Swap in Will's chosen big models once he names them.
8. **Networks:** Wi-Fi vs Ethernet if possible. Record generation tok/s, prompt tok/s, time to first token, load time (first and cached with `-c`), and round-trip time.
9. **Option B baseline:** 4 subagents and a 200-file summarise job, one PC vs two (each running its own llama-server); record total wall time.
10. **Decision:** keep A for a model/network combination only if it beats the best single-PC option. Otherwise the app uses B or one PC for that case.
11. **Clean up:** stop rpc-server on the helper; remove the firewall rule if not continuing.

### Later phases

2. **Helper mode + pairing + secure connection:** one-time code, pinned keys, the app's TLS tunnel, Compute page with the device list, runtime version check (and helper runtime update) on pairing and on load.
3. **Option A in the load panel:** "Use paired computers", calculated split shown before load, `-c` cache, failover by reload, live dashboard.
4. **Option B:** a generation-slot pool across PCs for subagents and batch file jobs, plus model file sync with sha256.
5. **Tests:** pairing (wrong or expired code refused), RPC unreachable without pairing/tunnel, a helper drop mid-reply, the split maths, B's routing, and the measurement report.

## Risks and limits

- **The RPC backend is upstream "proof of concept, fragile and insecure".** Mitigation: in phase 1 bind to the LAN IP only, firewall to the main PC, run only while linked, and (from phase 2) pairing with pinned keys. Remaining risk: a bug in rpc-server can still crash the helper or the load. We pin and test the runtime version.
- **Speed:** for models that fit on one PC, A is expected to be slower. The app won't use it there.
- **Reload on change:** changing the split or losing the helper means a model reload (a pause, not a crash).
- **Versions:** both PCs must run the same llama.cpp build (b12250 today).
- **Drivers:** a CUDA helper needs its own up-to-date NVIDIA driver; the app can check it but not install it.

## Open questions for Will

Answered 2026-11-11: no Tailscale; both PCs on home Wi-Fi.

1. **The helper PC:** which GPU and how much VRAM, and how much RAM?
2. **Models:** which big models do you most want to run across the two?

## Sources

- llama.cpp RPC README (security warning, `-c` cache, memory-proportional split): https://huggingface.co/datasets/echodict/llama.cpp/blob/main/tools/rpc/README.md and https://sources.debian.org/src/llama.cpp/8064%2Bdfsg-1/tools/rpc/README.md
- Community benchmarks, indicative only:
  - https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg (Mac + Blackwell over 10 GbE: prompt faster, generation slower)
  - https://www.popularai.org/p/10gbe-local-ai-llm (2× 3090 in one box vs over 10/50 GbE)
  - https://redlib.hbubli.cc/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference (Wi-Fi vs wired)
  - https://github.com/kjaiswal/llama-cpp-distributed-benchmarks

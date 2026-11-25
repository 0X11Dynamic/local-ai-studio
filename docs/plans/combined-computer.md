# Design: linking computers into one "combined computer"

Status: design document. The linking code lives in `crates/linking` and the app's Settings > Linking page; this file records the intended behaviour, the security model and the measurements behind it. Figures marked *estimate* or *expected* were not measured.

## Goals and decisions

- **Both machines are Windows PCs**, so one llama.cpp runtime build and one helper build cover both. A Mac could join later as a Metal helper; that is not planned.
- **Typical network: the same home Wi-Fi.** Wi-Fi is the case to design for; wired Ethernet is better.
- **No third-party VPN or relay required.** The app brings its own secure link: pairing with a one-time code, then an app-to-app encrypted tunnel with pinned keys, with `ggml-rpc-server` bound to localhost behind it. Nothing extra to install, and rpc-server is never reachable on the network directly.
- **Two ways to combine computers:**
  - Option A: one model split across both PCs, for models too big for one PC.
  - Option B: whole jobs (chats, subagents, file batches) run on whichever PC is free.
  - Other networks (internet) get Option B only by default.
- **Expectation on Wi-Fi:** Option A sends activations for every token over the network, so for models that fit on one PC it is likely slower than that PC alone. It is worth it mainly for models that do not fit on one PC. Option B tolerates Wi-Fi well.

## Short answer

- It is possible with what the app already ships: the bundled llama.cpp runtime includes an RPC server (`ggml-rpc-server.exe` plus `ggml-rpc.dll`) that lets one model run on several machines.
- It mainly buys room, not speed. For models that already fit on one PC it is usually slower per token, because every generated token crosses the network. Public tests show prompt processing can speed up while generation slows down.
- So: use Option A for models too big for one PC (measure first), and Option B for agent swarms and bulk file jobs and as the fallback where A is slower than one PC.

## What the shipped runtime offers

Checked read-only (`--help`, no model loaded) on llama.cpp build b12250:

- **`ggml-rpc-server.exe`:** `-H/--host` (default `127.0.0.1`), `-p/--port` (default `50052`), `-d/--device` (which local GPU or CPU to offer), `-t/--threads`, `-c/--cache` (keeps tensors on the helper's disk so they do not cross the network again on the next load).
- **`llama-server.exe`:** `--rpc host:port,...`, `-dev/--device`, `--list-devices`, `-ngl/--n-gpu-layers`, `-sm/--split-mode {none,layer,row,tensor}`, `-ts/--tensor-split N0,N1,...` (proportions, e.g. `3,1`), `-fit/--fit [on|off]` (default on) with `-fitt/--fit-target` per-device margin.
- **`llama-bench.exe`:** `-rpc/--rpc`, `-ts/--tensor-split` (written with slashes, e.g. `1/1`), `-ngl`, `-p`, `-n`, `-r` (default 5 repetitions).
- **Upstream README (llama.cpp `tools/rpc`):** weights and the KV cache are split across devices in proportion to their free memory, and the RPC backend is a proof of concept, "fragile and insecure", never to be run on an open network.

## Option A: one model across computers

### How it works

- **Main PC:** runs the app and `llama-server`. The model file only needs to be on the main PC; the weights are sent to the helper at load (and cached there with `-c`).
- **Helper PC:** runs `ggml-rpc-server` from the same llama.cpp build, as a headless helper mode of the app.
- **Helper requirements:**
  - The same runtime build (the CUDA variant for an NVIDIA GPU, or the CPU variant). Mixing CUDA and CPU variants of the same build should work (*expected*, not tested).
  - For CUDA: the helper's own NVIDIA driver, new enough for the runtime's CUDA version. The runtime does not install drivers.
- **Connecting them:** `llama-server --rpc <helper>:50052` with a `--tensor-split` calculated by the app.

### Security (rpc-server has no password or encryption)

Anyone who can reach the RPC port can use the helper's GPU and send it data, and a bug in rpc-server could be abused. So:

- **In the app: the app's own tunnel.** The app pairs PCs with a one-time code and carries RPC traffic through a mutually authenticated, encrypted app-to-app connection (pinned keys). rpc-server listens on localhost only on the helper.
- **Manual testing on a trusted LAN only** (not the app's normal path):
  1. Bind only to the helper's LAN IP (`-H <lan-ip>`), never `0.0.0.0`.
  2. Set the helper's network profile to Private and give both PCs fixed IPs (a DHCP reservation) so the rule below stays correct.
  3. Allow the port only from the main PC, in an admin PowerShell on the helper:
     `New-NetFirewallRule -DisplayName "LocalAIStudio RPC" -Direction Inbound -Protocol TCP -LocalPort 50052 -RemoteAddress <main-pc-ip> -Action Allow -Profile Private`
  4. If Windows shows its own "Allow access" prompt for `ggml-rpc-server.exe`, do not allow it (that opens the port to the whole network). Check with `Get-NetFirewallApplicationFilter -Program *ggml-rpc-server.exe | Get-NetFirewallRule` and remove any broad rule.
  5. Only run rpc-server while linked. Remove the rule with `Remove-NetFirewallRule -DisplayName "LocalAIStudio RPC"` when done.
  - Traffic on the LAN is unencrypted in this mode: anyone on the network could read prompts and answers in transit.

### Pairing

1. On the helper: "Link this PC" shows a one-time code for a few minutes.
2. On the main PC: Settings > Linking > add a computer, enter the code.
3. The two exchange long-term keys (stored in each machine's credential store) and pin each other's keys.
4. Either side can revoke the link. A helper serves only paired PCs, and only when helper mode is switched on.
- **Transport:** an app-to-app TLS tunnel using the pinned keys (mutual auth), with rpc-server bound to localhost behind it. The helper checks the peer's key before forwarding a single byte to rpc-server.
- **Finding the helper:** mDNS on the LAN, or by typing its address. Discovery only lists; pairing still needs the code.

### Deciding the split

- **Inputs per device:** free VRAM / RAM (from the helper's hardware report), measured speed (a short calibration run per device, cached per model), and network round-trip time and bandwidth (measured on connect).
- **Default split:** proportional to free memory, as llama.cpp does, then weighted by measured speed. A slow-link helper gets fewer layers, or is left out if it would make the whole thing slower than one PC.
- **The plan is shown before loading**, for example "40 layers here, 25 on the second PC, expected X tok/s", where X is an estimate from calibration, labelled as one.
- **The `-c` cache is on for helpers**, so later loads do not re-send the weights.

### Live re-measuring and failover

- **While running:** tokens/s, per-token latency and each node's time share are shown.
- **Rebalancing:** if a node slows down for several minutes (for example Wi-Fi drops), the app offers to rebalance. A new split needs a reload; llama.cpp cannot move layers live.
- **If the helper drops:** the current reply fails clearly and keeps its partial text (the same handling as a backend crash). The app reloads the model without the helper if it still fits, and the chat continues. Agent tasks resume from their saved transcript. Expect the pause of a model reload (seconds to tens of seconds; *estimate*).

### Where Option A helps and where it does not

- **Helps:** models too big for one PC. For example a 27B model at Q4_K_M (about 15.6 GB) does not fit in 8 GB of VRAM; split between two GPUs it can stay on GPUs if the second one has roughly 9 GB or more free (*estimate*; depends on context size). A helper with less VRAM can also offer its CPU/RAM (`-d`), which is slower.
- **Does not help:** models that already fit on one GPU. Each token needs a network round trip per device boundary, so generation usually gets slower.
- **Public tests** (other hardware; indicative only): two RTX 3090s gave about 50 tok/s in one machine vs about 37 tok/s split across two machines over 10 GbE; a Mac Studio + DGX Spark over 10 GbE gave 4.2x faster prompt processing but slower generation; Wi-Fi was much slower than gigabit Ethernet.
- **Network floor:** wired gigabit Ethernet is the practical minimum. Wi-Fi is expected to work but be slow (*estimate*).
- **Example estimate for a two-GPU setup** (two 8 GB GPUs of similar memory bandwidth, about 13.5 GB usable VRAM in total after OS, KV cache and compute buffers): a 27B Q4_K_M model (15.4 GiB) would still leave about 2 GB on the main PC's CPU, while a Q3_K_M (about 13 GB) would fit fully. Estimated generation (not measured): Q4_K_M about 10-14 tok/s vs 4.8 alone, Q3_K_M about 18-22 tok/s. The first split load sends about 7 GB over the network (minutes); the rpc-server cache helps later loads.

## Option B: spread whole jobs across computers

- Each PC loads its own copy of a model that fits it, as a full `llama-server`, reached through the same paired connection.
- The main app treats the helper as extra generation slots:
  - **Agent swarms:** each subagent goes to the least-busy PC; the generation gate becomes a pool of slots across machines.
  - **Bulk file reading or summarising:** files are split into batches across PCs and results are merged on the main PC.
  - **Chats:** stay on the main PC by default; they go to the helper only if the user picks it, or if the main PC is busy and the helper has the same model.
- **Tools always run on the main PC** (or the chosen environment), never on the helper, so permissions, approvals and the audit stay in one place. Helpers only generate text.
- **When B wins:** many independent jobs at once (throughput can approach the number of PCs, *expected*), and slow links, since only short requests and answers cross the network.
- **What B cannot do:** make one chat faster than one machine, or run a model too big for either PC on its own (that needs A). Each PC needs the model file; it is copied once and checked by sha256.

## Per-PC toggle for layer splitting

- Each paired PC has a switch "Split model layers with this PC". Default: on for PCs on the same local network, off for PCs on other networks (whole jobs only). The user can turn it on for any PC; nothing is blocked.
- At pairing the app measures round-trip time and bandwidth and shows an estimated split speed next to this-PC-alone speed. After a split load it shows the measured speed and, if that is slower than one PC, suggests turning the switch off (it never switches by itself).

## Linking flow

- Helper PC: "Link this PC" shows its address (local IP on the same network, public IP otherwise), port and a short one-time code that expires in minutes. The other PC types the address (and port) and the code. The default port is 47070 (below Windows' dynamic range 49152-65535, where Hyper-V/WSL/Docker/VirtualBox can reserve blocks such as 50060-50159, which made the earlier default 50070 unbindable on one PC); a stored 50070 is migrated once; if the port is reserved or in use the helper takes the next free one of 47071-47099 and the code view shows the actual port; with only an IP typed the pairing form tries 47070, then 50070. No firewall change: the existing rule is per program. The connection is direct; encryption and authentication are built into the app, so no reverse proxy, tunnel service or VPN needs to be installed. After the first link the two PCs pin each other's keys.
- Different networks: the app tries UPnP to open the port on the helper's router; if refused, it shows port-forward steps; if the ISP blocks inbound connections entirely (CGNAT), it says so. A relay server would cost money and is not planned.

## Computers on other networks (over the internet)

This works, with three conditions.

- **How:** the helper PC runs the app in helper mode as a small background service. It listens on one port with the app's own TLS (pinned keys from pairing). The main PC connects to `<public IP>:<port>`, or to a name if the helper's IP changes (dynamic DNS). rpc-server and llama-server stay on localhost on the helper; only the app's authenticated tunnel faces the internet.
- **Condition 1, the helper's router must let the connection in:**
  - The app tries UPnP / NAT-PMP first, which opens the port automatically on most home routers.
  - If that fails, the user forwards the port by hand; the app shows the exact port and the helper's local IP.
  - If the helper's internet uses CGNAT (common on mobile hotspots and some ISPs), inbound connections are impossible. The app detects this (its public IP does not match the router's WAN IP) and says so. The only fix is a relay both PCs connect out to, which needs an always-on server and is out of scope.
- **Condition 2, it must be safe on the open internet:**
  - Pairing with a one-time code, then mutual TLS with pinned keys. Anyone without the key is dropped before any model code sees a byte.
  - Rate limiting, a per-helper on/off switch, and "only while helper mode is on".
  - The helper decides which models and how much of its GPU it shares.
- **Condition 3, what is fast enough:**
  - Option A is not practical over the internet: every generated token waits on a round trip, and at 20-80 ms per trip that is a few tokens per second at best. Keep A for the same local network.
  - Option B works well over the internet: the helper runs its own model, the main PC sends a chat turn or a subagent's job and gets the answer streamed back, like calling a remote OpenAI-compatible API. Only text crosses the network.
- The LAN version and the internet version are the same code; the internet adds UPnP, the CGNAT check and dynamic DNS.

## Which one when

| Situation | Use |
|---|---|
| Model fits on the main PC, one chat | Main PC alone (fastest) |
| Model too big for the main PC, helper on Ethernet | Option A |
| Model too big, helper only on Wi-Fi | Option A if measured faster than CPU offload on one PC; otherwise one PC with CPU offload |
| Many subagents or a batch of files | Option B (each PC takes jobs) |
| The helper drops | Fail the current reply cleanly, reload or route without it |

The app picks automatically, shows the choice, and lets the user override it.

## Measurements

### Test setup: RTX 5060 host + CPU-only VirtualBox VM

llama.cpp b12250. The host (RTX 5060, 8 GB) runs `llama-server` with `--rpc` to `ggml-rpc-server.exe` inside a Windows 10 VM (CPU only, 4 threads) reached through VirtualBox NAT port forwarding. Model: Qwen3-4B Q4_K_M, `-c 4096`, 3 runs.

- Alone on the GPU: 123 tok/s, 8.1 ms/token, load 3.6 s.
- Split 1:1 with the VM: 23 tok/s (43.5 ms/token), load 34 s (weights sent to the VM).
- Split 3:1: 17 tok/s (58.5 ms/token; unexplained, check before relying on `-ts` weighting).
- The VM has no GPU, so these speeds say nothing about a real second PC.
- **Round trips per token = 1.** A relay adding a fixed delay each way gave 73.0 ms/token at 0 ms, 93.5 at 10 ms (+20.5 for +20 ms RTT) and 133.3 at 30 ms (+60.3 for +60 ms RTT). The network cost of splitting is about one RTT per generated token: roughly 3 ms on home Wi-Fi, 30-80 ms over the internet.
- Bytes per token are small (about 10-20 KB of fp16 activations); latency, not bandwidth, limits generation. Compression (lz4 inside the app's own tunnel) may help prompt processing and the first weight transfer on slow uplinks; base64 would make data 33% larger and is not compression.
- A fresh Windows lacks the MSVC runtime (`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`): `ggml-rpc-server.exe` and `llama-server` exit with 0xC0000135. The app ships these app-local with each runtime.

## Test plan

### Software tests on a VM (no second PC needed)

- **What a VM can and cannot show:** a VM on the same host cannot get GPU passthrough on a consumer Windows edition (no CUDA in the VM), so it runs the CPU runtime only and gives no speed results. It does test the software: helper mode, pairing code, the encrypted tunnel, firewall rule, reconnect after a drop, sending whole jobs to the helper, one model split across two hosts (that it runs, not how fast), and a clean-Windows install (SmartScreen, WebView2, shortcut, first-run setup choosing the CPU runtime).
- **VM:** Windows 11 (unactivated is fine), 4 vCPU, 6-8 GB RAM, about 80 GB disk, with the installer copied in.
- **Network modes as scenarios:** bridged adapter = same LAN; NAT = a PC behind another router, with VirtualBox port-forwarding rules standing in for a router port forward; no rule = unreachable / CGNAT-like case, which must give a clear error. UPnP cannot be tested in VirtualBox NAT (it needs a real router).

### Speed measurement by hand (needs a real second PC)

Decides whether A is worth using for a given hardware pair.

1. **Prepare the helper:** copy the matching runtime folder (CUDA or CPU); for CUDA, update the NVIDIA driver. Find both PCs' IPv4 addresses with `ipconfig`. Add the firewall rule from "Security".
2. **Start the helper** (in its runtime folder): `.\ggml-rpc-server.exe -H <helper-lan-ip> -p 50052 -c`. It prints the devices it offers; add `-d CUDA0` or `-d CPU` to choose one.
3. **Check the link from the main PC:** `Test-NetConnection <helper-ip> -Port 50052` (should say `TcpTestSucceeded : True`), and `ping <helper-ip>` for round-trip time.
4. **See the device order on the main PC:** `llama-server.exe --rpc <helper-ip>:50052 --list-devices`. `--tensor-split` values follow this order.
5. **Benchmark** (prompt and generation tok/s, 5 runs each), first the main PC alone, then split:
   `llama-bench.exe -m <model.gguf> -ngl 99 -p 512 -n 128 -r 5`
   `llama-bench.exe -m <model.gguf> -rpc <helper-ip>:50052 -ngl 99 -ts 1/1 -p 512 -n 128 -r 5`
   Adjust `-ts` towards each device's free memory (e.g. `8/12` for an 8 GB and a 12 GB GPU).
6. **Real use check:** run the server the way the app would and time a chat (same prompts, temperature 0, 5 runs, median): `llama-server.exe -m <model.gguf> --rpc <helper-ip>:50052 -ngl 99 --tensor-split 1,1 --port 9931`
7. **Models:** one that fits on the main PC and one that does not (with CPU offload on one PC as the baseline).
8. **Networks:** Wi-Fi vs Ethernet if possible. Record generation tok/s, prompt tok/s, time to first token, load time (first and cached with `-c`) and round-trip time.
9. **Option B baseline:** 4 subagents and a 200-file summarise job, one PC vs two; record total wall time.
10. **Decision:** keep A for a model/network combination only if it beats the best single-PC option; otherwise use B or one PC for that case.
11. **Clean up:** stop rpc-server on the helper; remove the firewall rule.

### Automated tests

Pairing (wrong or expired code refused), RPC unreachable without pairing/tunnel, a helper drop mid-reply, the split maths, B's routing, and the measurement report.

## Per-PC sharing limits and linked Auto context (2026-11-23)

Owner requests: "set a resource limit for each connection so you aren't accidentally botnetting the other computers" and "update the context window limit depending on linked computers".

### Limits, enforced on the helper

- **Where:** in the helper's tunnel, before bytes reach its `ggml-rpc-server` (`crates/linking/src/rpc_guard.rs`, wired in `helper::pipe`). rpc-server has no memory flag, so the helper parses ggml-rpc protocol 9 (b12250): `GET_DEVICE_MEMORY` replies are lowered to `min(free, cap - used)` / `min(total, cap)`; an `ALLOC_BUFFER` over the limit is not forwarded and gets rpc-server's own failed-alloc reply (`remote_ptr = 0`, a graceful load failure on the main PC); `FREE_BUFFER`, a failed alloc and the end of the stream give memory back. HELLO `conn_caps` are zeroed both ways (no RDMA upgrade; every byte stays on the TCP stream). `SET_TENSOR` bodies pass in place, no copy.
- **Fail closed:** a stream that doesn't start with HELLO, another HELLO reply size or protocol major, a reply nobody asked for, or a size protocol 9 doesn't use for ALLOC/GET_DEVICE_MEMORY/FREE ends the stream. Another major is reported to the main PC (`share_refused`, reason `incompatible`): "\"{helper}\" can't enforce its sharing limits with this PC's llama.cpp (RPC protocol {n}), so it refused the link. Update Local AI Studio on \"{helper}\"." No device list from rpc-server's banner: the session is refused.
- **Per linked PC:** `linked_peers.share_gpu_mib / share_ram_mib / share_threads / share_paused` (migration 0009; NULL = default). Defaults (owner's answers): 75% of the shared GPUs' free memory and 50% of available RAM after the RAM reserve (RAM only when the CPU device is shared), both rounded down to 0.5 GB, resolved when a session starts; threads = every core but 2 (min 1; at most P-cores + 4 on hybrid CPUs); per-PC settings may raise threads up to every logical processor. The old single "memory to lend" becomes both default caps.
- **One rpc-server:** a second PC while one is helped gets "\"{helper}\" is busy helping \"{other}\" right now; try again when that model is unloaded." A different thread limit restarts rpc-server between sessions only. Pause refuses new sessions; an open one is never cut. Lowering a limit applies to the next allocation; memory already lent is never taken back.
- **Old main PC, new helper:** still enforced (the guard is on the helper). **New main PC, old helper:** only the old single `ram_cap_mib`, applied to the main PC's split; the Linked PCs page says "{name} runs an older version: its limit isn't enforced there."
- **Not counted:** rpc-server's own host RAM (receive buffers, graph deserialisation) and its disk cache, so the RAM limit is approximate. The GPU limit covers all shared GPUs together; each GPU reports at most what is left of it, so a helper with two GPUs can show the main PC more than the limit in total (the guard still refuses the excess).

### Auto context counts linked PCs

- Each linked device adds `0.9 x what it lends - its own compute buffer` (never negative) to the GPU budget (`estimate::linked_net`); for "Share with CPU" it is counted as extra free VRAM before this PC's CPU. Devices are read from `--list-devices` through the guard (already capped); the load panel uses the linked PCs' last info (no process started).
- With linked PCs that lend more than a compute buffer, Auto tries both budgets (all on GPUs, and GPUs plus RAM) and keeps the larger context (context first); without them the old rule stays.
- A linked CPU device holding layers counts as "split with a CPU": f16 Auto sizing, never the full-GPU q8_0 raise.
- After sizing, `--tensor-split` is by net capacity at the final context (`split::tensor_split_net`); a linked PC whose devices all net 0 is left out of `--rpc` ("{name} left out: the {x} GB it lends is less than the working buffer a device needs.").
- The load panel shows "Auto: 65,536 tokens (32,768 on this PC alone). +6.0 GB GPU memory from Desk PC (its owner allows up to 8.0 GB)."
- Known, older than this change: Auto is not monotonic in local VRAM for some big dense models (27B fixture at 23 GB available RAM: 15 GiB free gives 262,144, 16 GiB 196,608, 18 GiB 262,144), because "split or not" picks q8_0 vs f16 and the RAM reserve then lowers the f16 size. Linked capacity follows the same curve; the tests check "never below this PC alone" there.

### Tunnel cost of the guard (release bench, loopback, 800 Mbit/s relay)

`cargo test -p linking --release --test bench -- --nocapture`, 3 runs each, guard on vs `LAS_BENCH_NO_GUARD=1`, medians: 64 MB one way over 800 Mbit/s 628 vs 627 Mbit/s; prompt step 42.5 vs 42.7 ms; token step 0.194 vs 0.189 ms (+5 microseconds, about 2.6%, under 1% of a LAN round trip). Before this change (no guard, old bench framing): 633 Mbit/s, 42.7 ms, 0.188 ms.

### End-to-end checklist (VM; run in the next big test)

Helper = the VM (CPU) or a GPU PC; main = the RTX 5060 host. Same llama.cpp build on both.

1. Helper: Linked PCs > the main PC's row > Edit: GPU memory 2 GB (or RAM 2 GB on the CPU-only VM). Save.
2. Main: Linked PCs shows "{helper} lets this PC use up to 2.0 GB ...". Load panel with "Use linked PCs": Auto shows "(N on this PC alone)" and a "+x GB from {helper}" line; Auto is higher than alone when the model doesn't fit here.
3. Main: run `llama-server --rpc <bridge endpoint> --list-devices` (or read the load log): RPC0 free <= 2048 MiB.
4. Load: succeeds; helper's row shows "Lending now: <= 2.0 GB"; helper log has no `-> failed` lines.
5. Force a larger share (fixed `--tensor-split` in Advanced so RPC0 gets more than 2 GB): the load fails with "\"{helper}\" refused {n} GB: its owner lets this PC use up to 2.0 GB of GPU memory ..." on the main PC, and "Refused {n} GB over the limit" on the helper's row.
6. Unload: the helper's "Lending now" disappears (usage 0).
7. Compatibility: old main PC (previous build) vs new helper: still capped (step 3 numbers). New main PC vs old helper: falls back to the old single cap, the row says "runs an older version: its limit isn't enforced there."
8. Busy: with a model loaded from main PC 1, load from a second linked PC: it gets "\"{helper}\" is busy helping \"{PC 1}\" right now ...". Unload on PC 1, retry: works.
9. Pause on the helper: a new load from main is refused with "isn't sharing with this PC right now (paused by its owner)"; a model already loaded keeps answering.
10. Threads: set the main PC's threads to 4 on a 4-vCPU helper; the rpc-server restarts at the next session with `-t 4` (helper log), prompt speed returns to ~121 t/s (vs ~71.5 with the default 2).

## Risks and limits

- **The RPC backend is upstream "proof of concept, fragile and insecure".** Mitigation: pairing with pinned keys and an app tunnel, rpc-server on localhost only, run only while linked. Remaining risk: a bug in rpc-server can still crash the helper or the load. The runtime version is pinned and tested.
- **Speed:** for models that fit on one PC, A is expected to be slower. The app does not use it there.
- **Reload on change:** changing the split or losing the helper means a model reload (a pause, not a crash).
- **Versions:** both PCs must run the same llama.cpp build.
- **Drivers:** a CUDA helper needs its own up-to-date NVIDIA driver; the app can check it but not install it.

## Sources

- llama.cpp RPC README (security warning, `-c` cache, memory-proportional split): https://huggingface.co/datasets/echodict/llama.cpp/blob/main/tools/rpc/README.md and https://sources.debian.org/src/llama.cpp/8064%2Bdfsg-1/tools/rpc/README.md
- Community benchmarks, indicative only:
  - https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg (Mac + Blackwell over 10 GbE: prompt faster, generation slower)
  - https://www.popularai.org/p/10gbe-local-ai-llm (2x 3090 in one box vs over 10/50 GbE)
  - https://redlib.hbubli.cc/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference (Wi-Fi vs wired)
  - https://github.com/kjaiswal/llama-cpp-distributed-benchmarks

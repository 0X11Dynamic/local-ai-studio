//! Split maths: llama-server `--list-devices` output, `--tensor-split`, and
//! the speed estimate.
//!
//! Measured on the VirtualBox VM (docs/plans/combined-computer.md, 2026-11-22):
//! a split costs about one network round trip per generated token (+20.5 ms
//! per token for +20 ms RTT, +60.3 ms for +60 ms), i.e. 1.0 ms per token for
//! each ms of RTT.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

/// Extra milliseconds per generated token for each millisecond of RTT (measured).
pub const MS_PER_TOKEN_PER_MS_RTT: f64 = 1.0;
/// A split that measures under this share of the alone speed is "slower".
pub const SLOWER_THRESHOLD: f64 = 0.95;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ListedDevice {
    pub name: String,
    pub description: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

static DEVICE_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?P<name>\S+?)\s*:\s*(?P<desc>.*?)\s*\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)\s*$").unwrap());

/// Device lines (`NAME: description (T MiB, F MiB free)`) from
/// `llama-server --list-devices` or the rpc-server's startup banner, in order.
pub fn parse_devices(out: &str) -> Vec<ListedDevice> {
    out.lines()
        .filter_map(|l| DEVICE_LINE.captures(l))
        .map(|c| ListedDevice {
            name: c["name"].to_string(),
            description: c["desc"].trim().to_string(),
            total_mib: c["total"].parse().unwrap_or(0),
            free_mib: c["free"].parse().unwrap_or(0),
        })
        .collect()
}

/// Does this listed device belong to the RPC endpoint `ep` (e.g. "127.0.0.1:53110")?
pub fn device_on_endpoint(d: &ListedDevice, ep: &str) -> bool {
    d.name.contains(ep) || d.description.contains(ep)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitPlan {
    /// `--tensor-split` value, in llama.cpp's model device order: every RPC
    /// device first (in `--list-devices` order), then this PC's GPUs. That is
    /// not the `--list-devices` order (which lists this PC's GPUs first):
    /// llama.cpp puts RPC servers at the front of the model's device list
    /// ("to minimize network transfers", `llama_prepare_model_devices` in
    /// src/llama.cpp, b12250), so the linked PC gets the first layers and the
    /// output layer stays here.
    pub tensor_split: String,
    /// (device, MiB used as its share, which endpoint or None = this PC), in
    /// the same order as `tensor_split`.
    pub shares: Vec<(String, u64, Option<String>)>,
}

/// Split proportional to each device's free memory. `caps` limits the total
/// share of an endpoint's devices (the helper's "RAM cap"). None when no
/// device has free memory or an endpoint has no device in the list.
pub fn tensor_split(devices: &[ListedDevice], endpoints: &[(String, Option<u64>)]) -> Option<SplitPlan> {
    if devices.is_empty() {
        return None;
    }
    let mut shares: Vec<(String, u64, Option<String>)> = devices
        .iter()
        .map(|d| (d.name.clone(), d.free_mib, endpoints.iter().find(|(ep, _)| device_on_endpoint(d, ep)).map(|(ep, _)| ep.clone())))
        .collect();
    for (ep, cap) in endpoints {
        let total: u64 = shares.iter().filter(|s| s.2.as_deref() == Some(ep)).map(|s| s.1).sum();
        if !shares.iter().any(|s| s.2.as_deref() == Some(ep)) {
            return None;
        }
        if let Some(cap) = *cap
            && total > cap
            && total > 0
        {
            for s in shares.iter_mut().filter(|s| s.2.as_deref() == Some(ep)) {
                s.1 = s.1 * cap / total;
            }
        }
    }
    if shares.iter().all(|s| s.1 == 0) {
        return None;
    }
    // Model device order: linked PCs' devices first (stable sort keeps their order).
    shares.sort_by_key(|s| s.2.is_none());
    let tensor_split = shares.iter().map(|s| s.1.to_string()).collect::<Vec<_>>().join(",");
    Some(SplitPlan { tensor_split, shares })
}

/// [`tensor_split`] by net capacity at the final context: each device's
/// share is 90% of its free memory (capped as in [`tensor_split`]) less the
/// compute buffer every model device gets (`compute_mib`), and this PC's
/// GPUs also less `local_extra_mib` (the projector, which stays here). A
/// device whose net is 0 gets no layers. None when no device has any net
/// capacity (the caller keeps the split by free memory).
pub fn tensor_split_net(devices: &[ListedDevice], endpoints: &[(String, Option<u64>)], compute_mib: u64, local_extra_mib: u64) -> Option<SplitPlan> {
    let base = tensor_split(devices, endpoints)?;
    let mut shares = base.shares;
    for s in shares.iter_mut() {
        let extra = if s.2.is_none() { local_extra_mib } else { 0 };
        s.1 = (s.1 * 9 / 10).saturating_sub(compute_mib.saturating_add(extra));
    }
    if shares.iter().all(|s| s.1 == 0) {
        return None;
    }
    let tensor_split = shares.iter().map(|s| s.1.to_string()).collect::<Vec<_>>().join(",");
    Some(SplitPlan { tensor_split, shares })
}

/// Endpoints whose every device has a share of 0 in `plan`: they hold no
/// layers, so they are left out of `--rpc` (a device that is listed still
/// gets a compute buffer).
pub fn idle_endpoints(plan: &SplitPlan) -> Vec<String> {
    let mut eps: Vec<String> = plan.shares.iter().filter_map(|s| s.2.clone()).collect();
    eps.dedup();
    eps.into_iter().filter(|ep| plan.shares.iter().filter(|s| s.2.as_deref() == Some(ep.as_str())).all(|s| s.1 == 0)).collect()
}

/// One device's row of llama.cpp's memory breakdown
/// (`common_memory_breakdown_print`, printed by `llama-fit-params -v` and
/// llama-server before anything is allocated), MiB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceUse {
    pub name: String,
    pub free_mib: u64,
    pub model_mib: u64,
    pub context_mib: u64,
    pub compute_mib: u64,
}

impl DeviceUse {
    pub fn self_mib(&self) -> u64 {
        self.model_mib + self.context_mib + self.compute_mib
    }
}

static BREAKDOWN_ROW: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"common_memory_breakdown_print:\s*\|\s*-\s*(?P<name>\S+)\s+\([^|]*\)\s*\|\s*(?P<total>\d+)\s*=\s*(?P<free>\d+)\s*\+\s*\(\s*(?P<self>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*(?P<ctx>\d+)\s*\+\s*(?P<compute>\d+)\s*\)").unwrap()
});

/// The device rows of the last memory breakdown in `log`, in llama.cpp's
/// model device order (linked PCs' RPC devices first). The Host row has no
/// free memory column and is skipped.
pub fn parse_breakdown(log: &str) -> Vec<DeviceUse> {
    let mut rows: Vec<DeviceUse> = Vec::new();
    for l in log.lines() {
        if l.contains("memory breakdown [MiB]") {
            rows.clear();
        }
        if let Some(c) = BREAKDOWN_ROW.captures(l) {
            let n = |k: &str| c[k].parse().unwrap_or(0);
            rows.push(DeviceUse { name: c["name"].to_string(), free_mib: n("free"), model_mib: n("model"), context_mib: n("ctx"), compute_mib: n("compute") });
        }
    }
    rows
}

/// Headroom left on each device (MiB): this PC's GPU also draws the desktop
/// and other apps; a linked PC reports free memory already lowered to its
/// owner's limit, which its guard enforces to the byte.
pub fn headroom_mib(d: &DeviceUse, linked: bool) -> u64 {
    if linked { 128 } else { 512.max(d.free_mib / 20) }
}

/// Does every device's measured use fit its free memory less [`headroom_mib`]?
/// `linked[i]` says whether row i is a linked PC's device.
/// `extra_local_mib` is memory this PC's GPU needs beyond the breakdown
/// (speculative MTP's draft context, which `llama-fit-params` doesn't model).
pub fn breakdown_fits(rows: &[DeviceUse], linked: &[bool], extra_local_mib: u64) -> bool {
    rows.iter().zip(linked).all(|(d, &l)| d.self_mib() + headroom_mib(d, l) + if l { 0 } else { extra_local_mib } <= d.free_mib)
}

/// A new `-ngl` and `--tensor-split` from a measured breakdown of the load
/// with `ngl` GPU layers split by `weights` (same order as `rows`): each
/// device holds as many layers as fit its free memory less its compute
/// buffer and headroom, at the measured cost per layer there (model plus
/// context over its layers; on this PC that also carries the output layer,
/// so it errs on the safe side). `max_layers` is the model's layer count
/// plus the output layer. None when nothing fits or the rows don't match.
pub fn refit(rows: &[DeviceUse], linked: &[bool], weights: &[u64], ngl: u32, max_layers: u32, extra_local_mib: u64) -> Option<(u32, Vec<u64>)> {
    if rows.is_empty() || rows.len() != weights.len() || rows.len() != linked.len() {
        return None;
    }
    let sum: u64 = weights.iter().sum();
    if sum == 0 || ngl == 0 {
        return None;
    }
    let mut layers = Vec::with_capacity(rows.len());
    for ((d, &l), &w) in rows.iter().zip(linked).zip(weights) {
        let had = ngl as f64 * w as f64 / sum as f64;
        if w == 0 || had <= 0.0 {
            layers.push(0u64);
            continue;
        }
        let per_layer = (d.model_mib + d.context_mib) as f64 / had;
        let budget = d.free_mib.saturating_sub(d.compute_mib + headroom_mib(d, l) + if l { 0 } else { extra_local_mib }) as f64;
        let n = if per_layer > 0.0 { (budget / per_layer).floor() as u64 } else { had.floor() as u64 };
        layers.push(n);
    }
    let total: u64 = layers.iter().sum();
    if total == 0 {
        return None;
    }
    // More room than layers: keep the proportions, at most every layer.
    let ngl_new = total.min(max_layers as u64) as u32;
    Some((ngl_new, layers))
}

/// Milliseconds a split adds to each generated token at this RTT.
pub fn added_ms_per_token(rtt_ms: f64) -> f64 {
    rtt_ms.max(0.0) * MS_PER_TOKEN_PER_MS_RTT
}

/// Speed with the network cost added to this PC's own speed. Only the
/// network part: the helper's own compute speed is not in it.
pub fn estimate_split_tok_s(alone_tok_s: f64, rtt_ms: f64) -> Option<f64> {
    (alone_tok_s > 0.0).then(|| 1000.0 / (1000.0 / alone_tok_s + added_ms_per_token(rtt_ms)))
}

/// Suggest turning split off: measured with the split is slower than alone.
pub fn split_is_slower(alone_tok_s: f64, split_tok_s: f64) -> bool {
    alone_tok_s > 0.0 && split_tok_s > 0.0 && split_tok_s < alone_tok_s * SLOWER_THRESHOLD
}

static EVAL_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|[^t]\s)eval time =\s*[\d.]+ ms /\s*(?P<n>\d+) (?:tokens|runs)\s*\(\s*[\d.]+ ms per token,\s*(?P<tps>[\d.]+) tokens per second\)").unwrap());

/// Replies shorter than this give noisy speeds; skip them.
pub const MIN_SPEED_TOKENS: u64 = 16;

/// Generation speed from llama-server's per-request timing line
/// (`       eval time = 1234.56 ms /   100 tokens (... 81.00 tokens per second)`),
/// not the `prompt eval time` line. None for short replies.
pub fn parse_eval_line(line: &str) -> Option<f64> {
    if line.contains("prompt eval time") {
        return None;
    }
    let c = EVAL_LINE.captures(line)?;
    let n: u64 = c["n"].parse().ok()?;
    let tps: f64 = c["tps"].parse().ok()?;
    (n >= MIN_SPEED_TOKENS && tps > 0.0 && tps.is_finite()).then_some(tps)
}

pub fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    Some(if v.len() % 2 == 1 { v[m] } else { (v[m - 1] + v[m]) / 2.0 })
}

// ---------------------------------------------------------------- estimate

/// Share of a device's memory bandwidth token generation reaches (low, high):
/// llama.cpp reads every weight once per token. GPUs get close to their
/// rated bandwidth, CPUs less (and a busy CPU even less).
pub const GPU_EFFICIENCY: (f64, f64) = (0.5, 0.85);
pub const CPU_EFFICIENCY: (f64, f64) = (0.35, 0.75);
/// RAM bandwidth assumed when none was measured (GB/s; dual-channel DDR4).
pub const DEFAULT_RAM_GBS: f64 = 20.0;
/// GPU bandwidth assumed for an unknown GPU (GB/s).
pub const DEFAULT_GPU_GBS: f64 = 250.0;

/// Rated memory bandwidth (GB/s) of common GPUs, by name. Longer names
/// first, so "4060 Ti" wins over "4060".
pub fn gpu_bandwidth_gbs(name: &str) -> Option<f64> {
    const TABLE: &[(&str, f64)] = &[
        ("RTX 5090", 1792.0), ("RTX 5080", 960.0), ("RTX 5070 Ti", 896.0), ("RTX 5070", 672.0), ("RTX 5060 Ti", 448.0), ("RTX 5060", 448.0),
        ("RTX 4090", 1008.0), ("RTX 4080 SUPER", 736.0), ("RTX 4080", 717.0), ("RTX 4070 Ti SUPER", 672.0), ("RTX 4070 Ti", 504.0),
        ("RTX 4070 SUPER", 504.0), ("RTX 4070", 504.0), ("RTX 4060 Ti", 288.0), ("RTX 4060", 272.0),
        ("RTX 3090 Ti", 1008.0), ("RTX 3090", 936.0), ("RTX 3080 Ti", 912.0), ("RTX 3080", 760.0), ("RTX 3070 Ti", 608.0), ("RTX 3070", 448.0),
        ("RTX 3060 Ti", 448.0), ("RTX 3060", 360.0), ("RTX 3050", 224.0),
        ("RTX 2080 Ti", 616.0), ("RTX 2080", 448.0), ("RTX 2070", 448.0), ("RTX 2060", 336.0),
        ("GTX 1660 Ti", 288.0), ("GTX 1660", 192.0), ("GTX 1080 Ti", 484.0), ("GTX 1080", 320.0), ("GTX 1070", 256.0), ("GTX 1060", 192.0),
        ("RX 7900 XTX", 960.0), ("RX 7900 XT", 800.0), ("RX 7800 XT", 624.0), ("RX 7700 XT", 432.0), ("RX 7600", 288.0),
        ("RX 6900 XT", 512.0), ("RX 6800 XT", 512.0), ("RX 6800", 512.0), ("RX 6700 XT", 384.0), ("RX 6600", 224.0),
        ("Arc A770", 560.0), ("Arc A750", 512.0), ("Arc B580", 456.0),
    ];
    let n = name.to_ascii_lowercase();
    TABLE
        .iter()
        .filter(|(k, _)| n.contains(&k.to_ascii_lowercase()))
        .max_by_key(|(k, _)| k.len())
        .map(|(_, v)| *v)
}

/// One device in the estimate: how much of the model it holds and how fast
/// it can read it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EstDevice {
    /// e.g. "RTX 5060 on this PC", "CPU on VM".
    pub label: String,
    pub bytes: u64,
    pub gb_s: f64,
    pub cpu: bool,
    /// Bandwidth came from a measurement (probe or a measured speed), not a table.
    pub measured: bool,
}

/// A device that can hold weights: label, free bytes, bandwidth, CPU?, measured?
#[derive(Debug, Clone)]
pub struct Capacity {
    pub label: String,
    pub free: u64,
    pub gb_s: f64,
    pub cpu: bool,
    pub measured: bool,
}

/// Spread `model_bytes` over the devices in proportion to free memory (as
/// the split does); what doesn't fit anywhere goes to `overflow` (this PC's
/// CPU, as llama.cpp's fitting does).
pub fn place_weights(model_bytes: u64, devices: &[Capacity], overflow: &Capacity) -> Vec<EstDevice> {
    let total: u64 = devices.iter().map(|d| d.free).sum();
    let mut out: Vec<EstDevice> = Vec::new();
    let mut placed = 0u64;
    for d in devices {
        let b = if total == 0 {
            0
        } else if model_bytes <= total {
            (model_bytes as u128 * d.free as u128 / total as u128) as u64
        } else {
            d.free
        };
        placed += b;
        if b > 0 {
            out.push(EstDevice { label: d.label.clone(), bytes: b, gb_s: d.gb_s, cpu: d.cpu, measured: d.measured });
        }
    }
    let rest = model_bytes.saturating_sub(placed);
    if rest > (64 << 20) {
        out.push(EstDevice { label: overflow.label.clone(), bytes: rest, gb_s: overflow.gb_s, cpu: overflow.cpu, measured: overflow.measured });
    }
    out
}

/// Generation speed range (tokens/s, low..high): per token every device
/// reads its weights once, one after the other, plus one network round trip
/// per linked PC (measured: 1.0 ms per ms of RTT).
pub fn estimate_range(devices: &[EstDevice], rtt_ms_total: f64) -> Option<(f64, f64)> {
    if devices.is_empty() {
        return None;
    }
    let ms = |pick: fn((f64, f64)) -> f64| -> f64 {
        devices
            .iter()
            .map(|d| {
                let eff = pick(if d.cpu { CPU_EFFICIENCY } else { GPU_EFFICIENCY });
                d.bytes as f64 / (d.gb_s.max(0.1) * 1e9 * eff) * 1000.0
            })
            .sum::<f64>()
            + added_ms_per_token(rtt_ms_total)
    };
    let slow = ms(|e| e.0);
    let fast = ms(|e| e.1);
    Some((1000.0 / slow, 1000.0 / fast))
}

/// Milliseconds per token the devices take to read their weights, at the
/// middle of the efficiency ranges (no network).
pub fn compute_ms(devices: &[EstDevice]) -> f64 {
    let mid = |e: (f64, f64)| (e.0 + e.1) / 2.0;
    devices.iter().map(|d| d.bytes as f64 / (d.gb_s.max(0.1) * 1e9 * mid(if d.cpu { CPU_EFFICIENCY } else { GPU_EFFICIENCY })) * 1000.0).sum()
}

/// [`estimate_range`] scaled by this PC's measured speed alone: the same
/// model on `alone` devices took `1000 / alone_tok_s` ms per token, so the
/// split's compute time is scaled by measured / estimated alone (that takes
/// in speculative decoding, the real RAM speed and llama.cpp's own
/// efficiency), plus the network. +-15% around that.
pub fn calibrated_range(split: &[EstDevice], alone: &[EstDevice], alone_tok_s: f64, rtt_ms_total: f64) -> Option<(f64, f64)> {
    let est_alone = compute_ms(alone);
    if split.is_empty() || est_alone <= 0.0 || alone_tok_s <= 0.0 {
        return None;
    }
    let k = (1000.0 / alone_tok_s) / est_alone;
    let ms = compute_ms(split) * k + added_ms_per_token(rtt_ms_total);
    Some((1000.0 / (ms * 1.15), 1000.0 / (ms * 0.85)))
}

/// Read bandwidth of this machine's RAM in GB/s: one thread per core (up to
/// 8) sums its own slice of a buffer much bigger than the CPU caches
/// (~0.1-0.3 s). One thread copying measured 14 GB/s on a PC whose
/// llama.cpp reads about 60. Blocking: run off the async threads.
pub fn probe_ram_bandwidth_gbs() -> f64 {
    const SIZE: usize = 512 << 20;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    let buf: Vec<u64> = vec![1u64; SIZE / 8];
    let chunk = buf.len() / threads;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let sum: u64 = std::thread::scope(|s| {
            let hs: Vec<_> = buf.chunks(chunk).map(|c| s.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x)))).collect();
            hs.into_iter().map(|h| h.join().unwrap_or(0)).fold(0u64, u64::wrapping_add)
        });
        std::hint::black_box(sum);
        best = best.min(t.elapsed().as_secs_f64());
    }
    SIZE as f64 / best.max(1e-6) / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: &str = "\
load_backend: loaded RPC backend from C:\\rt\\ggml-rpc.dll
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5060, compute capability 12.0, VMM: yes
Available devices:
  CUDA0: NVIDIA GeForce RTX 5060 (8150 MiB, 7000 MiB free)
  RPC0: 127.0.0.1:53110 (16000 MiB, 14000 MiB free)
  RPC1: 127.0.0.1:53111 (12000 MiB, 11500 MiB free)
";

    #[test]
    fn parses_list_devices_and_rpc_banner() {
        let d = parse_devices(LIST);
        assert_eq!(d.len(), 3, "{d:?}");
        assert_eq!(d[0], ListedDevice { name: "CUDA0".into(), description: "NVIDIA GeForce RTX 5060".into(), total_mib: 8150, free_mib: 7000 });
        assert_eq!((d[1].name.as_str(), d[1].description.as_str(), d[1].free_mib), ("RPC0", "127.0.0.1:53110", 14000));
        // rpc-server's own banner uses the same line format.
        let banner = "Starting RPC server v3.0.0\n  endpoint       : 127.0.0.1:50052\nDevices:\n  CPU: AMD Ryzen 7 (32000 MiB, 30000 MiB free)\n";
        let b = parse_devices(banner);
        assert_eq!(b.len(), 1);
        assert_eq!((b[0].name.as_str(), b[0].free_mib), ("CPU", 30000));
    }

    #[test]
    fn split_follows_free_memory_and_honours_caps() {
        let d = parse_devices(LIST);
        let eps = vec![("127.0.0.1:53110".to_string(), None), ("127.0.0.1:53111".to_string(), Some(5750))];
        let p = tensor_split(&d, &eps).unwrap();
        // RPC devices first (llama.cpp's model device order), then this PC's GPU.
        assert_eq!(p.tensor_split, "14000,5750,7000");
        assert_eq!(p.shares[0].2.as_deref(), Some("127.0.0.1:53110"));
        assert_eq!(p.shares[1].2.as_deref(), Some("127.0.0.1:53111"));
        assert_eq!((p.shares[2].0.as_str(), p.shares[2].2.as_deref()), ("CUDA0", None));
        // An endpoint whose devices did not show up: no plan (the caller explains).
        assert!(tensor_split(&d, &[("127.0.0.1:1".into(), None)]).is_none());
        assert!(tensor_split(&[], &[]).is_none());
    }

    /// VM test 2026-11-23: `--list-devices` printed CUDA0 (7031 MiB free)
    /// then RPC0 (3892 MiB free) and the app passed `--tensor-split 7031,3892`;
    /// llama.cpp applied it as RPC0 = 7031, so the CPU-only VM got 1367.5 MiB
    /// of weights (64% of the layers) instead of about 36%.
    #[test]
    fn the_vm_gets_its_own_share_not_the_gpus() {
        let d = parse_devices("  CUDA0: NVIDIA GeForce RTX 5060 (8123 MiB, 7031 MiB free)
  RPC0: 127.0.0.1:62565 (6143 MiB, 3892 MiB free)
");
        let p = tensor_split(&d, &[("127.0.0.1:62565".into(), None)]).unwrap();
        assert_eq!(p.tensor_split, "3892,7031");
    }

    /// Real 2-PC test (2026-11-25, Qwen3.8-27B Q4_K_M, -c 196608, -ngl 29,
    /// --tensor-split 4083,5802): the linked PC's limit is 5120 MiB and its
    /// compute buffer 1040 MiB, which the estimate put at about 500.
    const BREAKDOWN: &str = "\
0.01.700.673 I common_memory_breakdown_print: | memory breakdown [MiB]     | total   free     self   model   context   compute    unaccounted |
0.01.700.676 I common_memory_breakdown_print: |   - RPC0 (127.0.0.1:61526) |  5120 = 5120 + ( 5324 =  3025 +    1259 +    1040) +       -5324 |
0.01.700.676 I common_memory_breakdown_print: |   - CUDA0 (RTX 5060)       |  8123 = 7031 + ( 7097 =  3958 +    1661 +    1477) +       -6004 |
0.01.700.676 I common_memory_breakdown_print: |   - Host                   |                 12780 =  8786 +    3761 +     232                |
";

    #[test]
    fn calibrated_estimate_matches_the_real_two_pc_run() {
        // Real 2-PC test (2026-11-25): Qwen3.8-27B 16.8 GB with MTP, alone 8.56
        // tok/s with 11 of 66 layers (2.8 GB) on the RTX 5060; linked PC lends
        // 5.4 GB on an RTX 2060 SUPER, RTT 20.6 ms; measured split 8.13.
        let gpu = |label: &str, bytes: u64, gb_s: f64| EstDevice { label: label.into(), bytes, gb_s, cpu: false, measured: false };
        let cpu = |bytes: u64| EstDevice { label: "CPU".into(), bytes, gb_s: 60.0, cpu: true, measured: true };
        let alone = [gpu("5060", 2_800_000_000, 448.0), cpu(14_000_000_000)];
        let split = [gpu("5060", 2_800_000_000, 448.0), gpu("2060", 2_200_000_000, 336.0), cpu(11_800_000_000)];
        let (lo, hi) = calibrated_range(&split, &alone, 8.56, 20.6).unwrap();
        assert!(lo < 8.13 && 8.13 < hi, "{lo:.2}..{hi:.2}");
        assert!(calibrated_range(&split, &[], 8.56, 20.6).is_none());
    }

    #[test]
    fn ram_probe_reports_a_plausible_bandwidth() {
        let g = probe_ram_bandwidth_gbs();
        assert!(g > 1.0 && g < 2000.0, "{g}");
    }

    #[test]
    fn breakdown_rows_are_read_in_model_device_order() {
        let r = parse_breakdown(BREAKDOWN);
        assert_eq!(r.len(), 2, "Host is skipped");
        assert_eq!(r[0], DeviceUse { name: "RPC0".into(), free_mib: 5120, model_mib: 3025, context_mib: 1259, compute_mib: 1040 });
        assert_eq!((r[1].name.as_str(), r[1].free_mib, r[1].self_mib()), ("CUDA0", 7031, 7096));
        assert!(!breakdown_fits(&r, &[true, false], 0));
    }

    #[test]
    fn refit_moves_layers_off_a_device_over_its_limit() {
        let r = parse_breakdown(BREAKDOWN);
        let linked = [true, false];
        let (ngl, layers) = refit(&r, &linked, &[4083, 5802], 29, 66, 0).unwrap();
        // RPC0: 4284 MiB over 12 layers (357/layer), 5120-1040-128 = 3952 -> 11 layers.
        assert_eq!(layers[0], 11);
        // CUDA0: 5619 MiB over 17 layers (330/layer), 7031-1477-512 = 5042 -> 15 layers.
        assert_eq!(layers[1], 15);
        assert_eq!(ngl, 26);
        // What those layers cost at the measured rates fits both devices.
        let per = [4284.0 / (29.0 * 4083.0 / 9885.0), 5619.0 / (29.0 * 5802.0 / 9885.0)];
        for (i, d) in r.iter().enumerate() {
            let used = layers[i] as f64 * per[i] + d.compute_mib as f64 + headroom_mib(d, linked[i]) as f64;
            assert!(used <= d.free_mib as f64, "{}: {used}", d.name);
        }
        // Nothing to go on: None.
        assert_eq!(refit(&r, &linked, &[1], 29, 66, 0), None);
        assert_eq!(refit(&[], &[], &[], 29, 66, 0), None);
        // MTP's draft context on this PC's GPU leaves it fewer layers; the linked PC keeps its 11.
        let (_, mtp) = refit(&r, &linked, &[4083, 5802], 29, 66, 1500).unwrap();
        assert_eq!(mtp, vec![11, 10]);
    }

    #[test]
    fn tensor_split_follows_net_capacity() {
        let d = parse_devices(LIST);
        let eps = vec![("127.0.0.1:53110".to_string(), None), ("127.0.0.1:53111".to_string(), Some(5750))];
        // Compute buffer 500 MiB per device, projector 300 MiB here.
        let p = tensor_split_net(&d, &eps, 500, 300).unwrap();
        // 14000*0.9-500, 5750*0.9-500, 7000*0.9-500-300.
        assert_eq!(p.tensor_split, "12100,4675,5500");
        assert!(idle_endpoints(&p).is_empty());
        // A device that lends less than its own compute buffer gets nothing and is left out.
        let tiny = parse_devices("  CUDA0: GPU (8000 MiB, 7000 MiB free)\n  RPC0: 127.0.0.1:1 (8000 MiB, 400 MiB free)\n");
        let p = tensor_split_net(&tiny, &[("127.0.0.1:1".into(), None)], 500, 0).unwrap();
        assert_eq!(p.tensor_split, "0,5800");
        assert_eq!(idle_endpoints(&p), vec!["127.0.0.1:1".to_string()]);
        // Nothing has net capacity: None (the caller keeps the free-memory split).
        assert!(tensor_split_net(&tiny, &[("127.0.0.1:1".into(), None)], 9000, 0).is_none());
    }

    #[test]
    fn reads_generation_speed_from_the_server_log() {
        assert_eq!(parse_eval_line("       eval time =    1234.56 ms /   100 tokens (   12.35 ms per token,    81.00 tokens per second)"), Some(81.0));
        assert_eq!(parse_eval_line("eval time =     500.00 ms /    64 tokens (    7.81 ms per token,   128.00 tokens per second)"), Some(128.0));
        assert_eq!(parse_eval_line("prompt eval time =     123.45 ms /    10 tokens (   12.34 ms per token,    81.00 tokens per second)"), None);
        assert_eq!(parse_eval_line("       eval time =      50.00 ms /     3 tokens (   16.67 ms per token,    60.00 tokens per second)"), None, "too short");
        assert_eq!(parse_eval_line("      total time =    1358.01 ms /   110 tokens"), None);
    }

    #[test]
    fn per_device_estimate_brackets_the_vm_measurement() {
        const GB: u64 = 1_000_000_000;
        assert_eq!(gpu_bandwidth_gbs("NVIDIA GeForce RTX 5060"), Some(448.0));
        assert_eq!(gpu_bandwidth_gbs("NVIDIA GeForce RTX 4060 Ti"), Some(288.0));
        assert_eq!(gpu_bandwidth_gbs("AMD Radeon RX 7900 XTX"), Some(960.0));
        assert_eq!(gpu_bandwidth_gbs("Some GPU"), None);
        // VM test: Qwen3-4B Q4_K_M (2.5 GB), RTX 5060 7031 MiB free + VM CPU
        // 4046 MiB free (RAM ~12 GB/s), 0.5 ms RTT: measured 7.3 tok/s.
        let mib = |m: u64| m << 20;
        let local = Capacity { label: "RTX 5060 on this PC".into(), free: mib(7031), gb_s: 448.0, cpu: false, measured: false };
        let vm = Capacity { label: "CPU on VM".into(), free: mib(4046), gb_s: 12.0, cpu: true, measured: true };
        let overflow = Capacity { label: "CPU on this PC".into(), free: 0, gb_s: 40.0, cpu: true, measured: true };
        let devs = place_weights(25 * GB / 10, &[local.clone(), vm], &overflow);
        assert_eq!(devs.len(), 2);
        let (lo, hi) = estimate_range(&devs, 0.5).unwrap();
        assert!(lo <= 7.3 && 7.3 <= hi, "{lo:.1}..{hi:.1}");
        // Alone on the GPU: measured 123 tok/s.
        let (lo, hi) = estimate_range(&place_weights(25 * GB / 10, std::slice::from_ref(&local), &overflow), 0.0).unwrap();
        assert!(lo <= 123.0 && 123.0 <= hi, "{lo:.1}..{hi:.1}");
        // Too big for every device: the rest lands on this PC's CPU.
        let big = place_weights(16 * GB, &[local], &overflow);
        assert_eq!(big.last().unwrap().label, "CPU on this PC");
        assert_eq!(big.iter().map(|d| d.bytes).sum::<u64>(), 16 * GB);
        assert!(estimate_range(&[], 1.0).is_none());
        // The probe returns something plausible for RAM.
        let bw = probe_ram_bandwidth_gbs();
        assert!(bw > 0.5 && bw < 2000.0, "{bw}");
    }

    #[test]
    fn one_round_trip_per_token_estimate() {
        // Measured: 73.0 ms/token at 0 ms RTT; +20 ms RTT -> ~93 ms/token.
        let alone = 1000.0 / 73.0;
        let est = estimate_split_tok_s(alone, 20.0).unwrap();
        assert!((1000.0 / est - 93.0).abs() < 0.01, "{}", 1000.0 / est);
        assert!((added_ms_per_token(3.0) - 3.0).abs() < 1e-9);
        // Home Wi-Fi (3 ms) on a 123 t/s model: ~89 t/s from the network alone.
        let wifi = estimate_split_tok_s(123.0, 3.0).unwrap();
        assert!((88.0..90.5).contains(&wifi), "{wifi}");
        assert!(estimate_split_tok_s(0.0, 3.0).is_none());
        assert!(split_is_slower(123.0, 23.0));
        assert!(!split_is_slower(5.0, 9.0));
        assert!(!split_is_slower(10.0, 9.8), "within 5% is not slower");
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }
}

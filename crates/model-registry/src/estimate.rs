//! Memory estimates for loading a GGUF model. Every number here is an
//! ESTIMATE derived from file size and header metadata; the UI must label it so.
//! Real usage is read back from NVML/process metrics after load.

use serde::{Deserialize, Serialize};

use crate::gguf::ModelMetadata;

/// KV cache element type, matching llama-server `--cache-type-k/-v` values.
pub fn kv_bytes_per_element(t: &str) -> f64 {
    match t.to_ascii_lowercase().as_str() {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        _ => 2.0,
    }
}

/// Architectures whose layers are mostly linear attention / recurrent, with
/// a full-attention (KV cache) layer every [`HYBRID_FULL_ATTENTION_INTERVAL`].
pub fn is_hybrid_attention(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen35moe" | "qwen3next")
}
pub const HYBRID_FULL_ATTENTION_INTERVAL: u64 = 4;

/// llama.cpp pads the sliding-window cache to a multiple of this many cells.
const SWA_CACHE_PAD: u64 = 256;

/// "Every Nth layer is global" for architectures whose GGUF has a sliding
/// window but no pattern key: llama.cpp hard-codes these (`set_swa_pattern`).
fn default_swa_pattern(arch: &str) -> Option<u64> {
    match arch {
        "gemma2" | "gpt-oss" => Some(2),
        "gemma3" => Some(6),
        "gemma3n" => Some(5),
        "cohere2" => Some(4),
        _ => None,
    }
}

/// Which layers keep a KV cache, and how big each one's is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    /// Layers with a full-context KV cache (n_ctx cells).
    pub global_layers: u64,
    /// Sliding-window layers: a cache of about `window + n_ubatch` cells.
    pub swa_layers: u64,
    /// The sliding window, when there are SWA layers.
    pub window: Option<u64>,
    /// Layers that reuse an earlier layer's KV cache (no cache of their own).
    pub shared_layers: u64,
    /// MTP / NextN layers: no KV cache in a normal load.
    pub mtp_layers: u64,
}

/// KV cache layout from the header, as llama.cpp allocates it:
/// - MTP / NextN layers (`nextn_predict_layers`, the last ones of
///   `block_count`) get no cache;
/// - shared-KV layers (`shared_kv_layers`, the last ones before those) reuse
///   an earlier layer's cache;
/// - hybrid models (Qwen3.5 / Qwen3-Next) keep a cache on every
///   `full_attention_interval`-th layer only (layer il when (il + 1) % interval == 0);
/// - with a sliding window, a layer is SWA per the per-layer pattern array
///   (gemma4), or every Nth layer is global (pattern number, or the
///   architecture's llama.cpp default); without a known pattern every layer
///   counts as global (the safe overestimate).
///
/// A header with none of these keys gives every block a full cache, exactly
/// the plain dense formula.
pub fn kv_layout(meta: &ModelMetadata) -> KvLayout {
    let layers = meta.block_count.unwrap_or(0);
    let arch = meta.architecture.as_deref().unwrap_or("");
    let mtp = meta.mtp_layers.unwrap_or(0).min(layers);
    let main = layers - mtp;
    let shared = meta.shared_kv_layers.unwrap_or(0).min(main);
    let own = main - shared;
    if is_hybrid_attention(arch) {
        let interval = meta.full_attention_interval.unwrap_or(HYBRID_FULL_ATTENTION_INTERVAL).max(1);
        // Rows stored before MTP / interval were read keep the earlier rounding.
        let global = if meta.mtp_layers.is_none() && meta.full_attention_interval.is_none() && shared == 0 {
            layers.div_ceil(HYBRID_FULL_ATTENTION_INTERVAL)
        } else {
            own / interval
        };
        return KvLayout { global_layers: global, swa_layers: 0, window: None, shared_layers: shared, mtp_layers: mtp };
    }
    let window = meta.sliding_window.filter(|&w| w > 0);
    let pattern = meta.sliding_window_pattern.or_else(|| default_swa_pattern(arch));
    let is_swa = |il: u64| -> bool {
        if window.is_none() {
            return false;
        }
        match (&meta.swa_layers, pattern) {
            (Some(flags), _) if !flags.is_empty() => flags.get(il as usize).copied().unwrap_or(false),
            // llama.cpp: pattern 0 = every layer SWA, else layer il is SWA when il % n < n - 1.
            (_, Some(n)) => n == 0 || il % n < n - 1,
            _ => false,
        }
    };
    let swa = (0..own).filter(|&il| is_swa(il)).count() as u64;
    KvLayout { global_layers: own - swa, swa_layers: swa, window: if swa > 0 { window } else { None }, shared_layers: shared, mtp_layers: mtp }
}

/// Cells llama.cpp gives a sliding-window layer's cache: the window (per
/// sequence) plus one micro-batch, padded to 256, never more than n_ctx.
pub fn swa_cache_cells(n_ctx: u64, window: u64, n_ubatch: u64, parallel: u64) -> u64 {
    (window * parallel.max(1) + n_ubatch.max(1)).div_ceil(SWA_CACHE_PAD).saturating_mul(SWA_CACHE_PAD).min(n_ctx)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateInput {
    pub n_ctx: u64,
    /// None = offload every layer to the GPU; Some(0) = CPU only.
    pub gpu_layers: Option<u64>,
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// false = KV cache stays in system RAM (`--no-kv-offload`).
    pub kv_offload: bool,
    pub n_ubatch: u64,
    /// Number of parallel slots (`-np`); the KV cache holds n_ctx in total.
    pub parallel: u64,
    /// Size of the vision projector (mmproj) file when it is loaded, else 0.
    /// llama.cpp puts the projector on the GPU by default (`--mmproj-offload`).
    #[serde(default)]
    pub projector_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub compute_buffer_bytes: u64,
    /// Vision projector (mmproj) weights, 0 when it is not loaded.
    pub projector_bytes: u64,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub total_layers: u64,
    pub offloaded_layers: u64,
    /// Short caveats shown next to the numbers.
    pub notes: Vec<String>,
}

pub fn estimate(meta: &ModelMetadata, file_bytes: u64, input: &EstimateInput) -> MemoryEstimate {
    let mut notes = vec!["Estimate from file size and GGUF metadata; actual use is measured after load.".to_string()];
    let layers = meta.block_count.unwrap_or(0);
    let total_layers = layers + 1; // + output layer, as llama.cpp counts for -ngl
    let offloaded = match input.gpu_layers {
        None => total_layers,
        Some(n) => n.min(total_layers),
    };
    let gpu_frac = if total_layers == 0 { 0.0 } else { offloaded as f64 / total_layers as f64 };

    let head_count = meta.head_count.unwrap_or(0);
    let kv_heads = meta.head_count_kv.unwrap_or(head_count);
    let head_dim = if head_count > 0 { meta.embedding_length.unwrap_or(0) / head_count } else { 0 };
    let k_len = meta.key_length.unwrap_or(head_dim);
    let v_len = meta.value_length.unwrap_or(head_dim);
    // Hybrid models (Qwen3.5 / Qwen3-Next): only every 4th layer is full
    // attention with a KV cache; the others keep a small fixed recurrent state
    // (llama.cpp's default full_attention_interval is 4). Measured on
    // Qwen3.5-4B at 16k: KV 512 MiB (a quarter of the dense formula) + RS 50 MiB;
    // Qwen3.8-27B (65 blocks incl. 1 MTP layer) at 131072 f16: 16 KV layers, 8192 MiB.
    // Sliding-window layers keep only the window (see `kv_layout`): Gemma-4-E4B
    // at 98304 f16 measured 1576 MiB (4 global + 20 SWA layers, 18 shared).
    let hybrid = meta.architecture.as_deref().is_some_and(is_hybrid_attention);
    let layout = kv_layout(meta);
    let kv_layers = layout.global_layers;
    let kv_cache_bytes = if layers > 0 && kv_heads > 0 && k_len > 0 {
        let per_tok_k = (kv_heads * k_len) as f64 * kv_bytes_per_element(&input.cache_type_k);
        let per_tok_v = (kv_heads * v_len) as f64 * kv_bytes_per_element(&input.cache_type_v);
        let recurrent = if hybrid { file_bytes / 50 * input.parallel.max(1) } else { 0 };
        let full = (kv_layers as f64 * input.n_ctx as f64 * (per_tok_k + per_tok_v)) as u64 + recurrent;
        match layout.window {
            Some(w) if layout.swa_layers > 0 => {
                let k_swa = meta.key_length_swa.unwrap_or(k_len);
                let v_swa = meta.value_length_swa.unwrap_or(v_len);
                let per_tok_swa = (kv_heads * k_swa) as f64 * kv_bytes_per_element(&input.cache_type_k)
                    + (kv_heads * v_swa) as f64 * kv_bytes_per_element(&input.cache_type_v);
                let cells = swa_cache_cells(input.n_ctx, w, input.n_ubatch, input.parallel);
                full + (layout.swa_layers as f64 * cells as f64 * per_tok_swa) as u64
            }
            _ => full,
        }
    } else {
        notes.push("KV cache size unknown: header lacks attention dimensions.".into());
        0
    };
    if layout.swa_layers > 0 {
        notes.push(format!(
            "Sliding-window attention: {} layers keep the full context, {} only the last {} tokens.",
            layout.global_layers,
            layout.swa_layers,
            layout.window.unwrap_or(0)
        ));
    } else if meta.architecture.as_deref().is_some_and(|a| a.starts_with("gemma") || a.contains("swa")) {
        notes.push("This architecture may use sliding-window attention, so the real KV cache can be smaller.".into());
    }
    if layout.shared_layers > 0 {
        notes.push(format!("{} layers reuse another layer's KV cache.", layout.shared_layers));
    }
    if layout.mtp_layers > 0 {
        notes.push(format!("{} multi-token-prediction layer(s) keep no KV cache.", layout.mtp_layers));
    }
    if hybrid {
        notes.push("Hybrid attention: only every 4th layer keeps a KV cache, the rest a small fixed state.".into());
    }
    if meta.expert_count.unwrap_or(0) > 0 {
        notes.push("Mixture-of-experts model: all experts must fit in memory even though few are active per token.".into());
    }

    let vocab = meta.vocab_size.unwrap_or(32_000);
    let embd = meta.embedding_length.unwrap_or(4096);
    let compute_buffer_bytes = input.n_ubatch.max(1) * (vocab + 4 * embd) * 4;

    let weights_gpu = (file_bytes as f64 * gpu_frac) as u64;
    let weights_cpu = file_bytes - weights_gpu;
    let kv_gpu = if input.kv_offload { (kv_cache_bytes as f64 * gpu_frac) as u64 } else { 0 };
    let kv_cpu = kv_cache_bytes - kv_gpu;
    let uses_gpu = offloaded > 0;
    // The projector goes to the GPU whenever the model uses it, else to RAM.
    let projector_bytes = input.projector_bytes;
    if projector_bytes > 0 {
        notes.push("Includes the vision projector (mmproj), which llama.cpp keeps on the GPU.".into());
    }
    let (projector_gpu, projector_cpu) = if uses_gpu { (projector_bytes, 0) } else { (0, projector_bytes) };
    let vram_bytes = if uses_gpu { weights_gpu + kv_gpu + compute_buffer_bytes + projector_gpu } else { 0 };
    let ram_bytes = weights_cpu + kv_cpu + projector_cpu + if uses_gpu { 256 << 20 } else { compute_buffer_bytes };

    MemoryEstimate {
        weights_bytes: file_bytes,
        kv_cache_bytes,
        compute_buffer_bytes,
        projector_bytes,
        vram_bytes,
        ram_bytes,
        total_layers,
        offloaded_layers: offloaded,
        notes,
    }
}

/// Largest GPU layer count whose estimated VRAM stays under `budget_bytes`.
pub fn max_gpu_layers_for_budget(meta: &ModelMetadata, file_bytes: u64, input: &EstimateInput, budget_bytes: u64) -> u64 {
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut best = 0;
    for n in 0..=total {
        let mut i = input.clone();
        i.gpu_layers = Some(n);
        if estimate(meta, file_bytes, &i).vram_bytes <= budget_bytes {
            best = n;
        } else {
            break;
        }
    }
    best
}

/// What "Share with CPU" picks for a load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShareSplit {
    /// Layers offloaded to the GPU (`-ngl`).
    pub gpu_layers: u64,
    pub total_layers: u64,
    /// MoE: expert weights of the first N layers stay on the CPU (`--n-cpu-moe`).
    pub n_cpu_moe: Option<u64>,
    /// Everything fits in VRAM: no sharing needed.
    pub all_fit: bool,
    /// Estimated VRAM of the chosen split.
    pub vram_bytes: u64,
    /// Plain description for the UI.
    pub summary: String,
}

/// Share of an MoE file assumed to be expert FFN weights (the header has no
/// tensor list, so this is an estimate; experts dominate MoE weights).
const MOE_EXPERT_SHARE: f64 = 0.85;

/// Split for "Share with CPU": fill 90% of `free_vram` (headroom for the
/// compute buffer and the desktop), the rest runs on the CPU. MoE models
/// keep every layer on the GPU and move expert weights of the first N layers
/// to the CPU instead (much faster than splitting whole layers), when the
/// runtime supports `--n-cpu-moe`.
pub fn share_split(meta: &ModelMetadata, file_bytes: u64, input: &EstimateInput, free_vram: u64, has_n_cpu_moe: bool) -> ShareSplit {
    let budget = (free_vram as f64 * 0.90) as u64;
    let total = meta.block_count.unwrap_or(0) + 1;
    let mut all = input.clone();
    all.gpu_layers = None;
    let full = estimate(meta, file_bytes, &all);
    if full.vram_bytes <= budget {
        return ShareSplit { gpu_layers: total, total_layers: total, n_cpu_moe: None, all_fit: true, vram_bytes: full.vram_bytes, summary: format!("All {total} layers fit on the GPU; nothing needs to run on the CPU.") };
    }
    let blocks = meta.block_count.unwrap_or(0);
    if meta.expert_count.unwrap_or(0) > 0 && has_n_cpu_moe && blocks > 0 {
        let expert_per_layer = (file_bytes as f64 * MOE_EXPERT_SHARE / blocks as f64) as u64;
        let over = full.vram_bytes - budget;
        let n = over.div_ceil(expert_per_layer.max(1)).min(blocks);
        let vram = full.vram_bytes.saturating_sub(n * expert_per_layer);
        if vram <= budget {
            return ShareSplit {
                gpu_layers: total,
                total_layers: total,
                n_cpu_moe: Some(n),
                all_fit: false,
                vram_bytes: vram,
                summary: format!("All {total} layers on the GPU, with the expert weights of the first {n} on the CPU (mixture-of-experts model too big for free VRAM)."),
            };
        }
    }
    let n = max_gpu_layers_for_budget(meta, file_bytes, input, budget);
    let mut chosen = input.clone();
    chosen.gpu_layers = Some(n);
    let vram = estimate(meta, file_bytes, &chosen).vram_bytes;
    let summary = if n == 0 { "Nothing fits in free VRAM, so the model runs on the CPU.".to_string() } else { format!("{n} of {total} layers on the GPU, the rest on the CPU (model too big for free VRAM).") };
    ShareSplit { gpu_layers: n, total_layers: total, n_cpu_moe: None, all_fit: false, vram_bytes: vram, summary }
}

/// Smallest context "Auto" ever picks (unless the model was trained on less).
pub const AUTO_CTX_MIN: u64 = 2048;
/// Upper bound when the GGUF header does not state the trained context.
pub const AUTO_CTX_UNKNOWN_TRAINED: u64 = 32_768;
/// Used when there is a GPU but no memory reading for it.
pub const AUTO_CTX_NO_GPU_INFO: u64 = 8192;
/// Share of free VRAM / available RAM an Auto context may fill.
const AUTO_HEADROOM: f64 = 0.90;

/// Context sizes Auto chooses from, up to `max`: 2k, 4k, 6k, 8k, 12k, 16k,
/// 24k, 32k, 48k, 64k, 96k, 128k, ... (powers of two and the 1.5x step
/// between them), plus `max` itself rounded down to a multiple of 1024, so a
/// trained context like 40,960 can be used in full.
pub fn ctx_steps(max: u64) -> Vec<u64> {
    let mut v = Vec::new();
    let mut p = AUTO_CTX_MIN;
    while p <= max {
        v.push(p);
        if p >= 4096 && p + p / 2 <= max {
            v.push(p + p / 2);
        }
        p *= 2;
    }
    let top = max / 1024 * 1024;
    if top >= AUTO_CTX_MIN && !v.contains(&top) {
        v.push(top);
    }
    v.sort_unstable();
    v
}

/// What the machine has right now, for [`auto_n_ctx`].
#[derive(Debug, Clone, Copy)]
pub struct AutoCtxEnv {
    /// Free VRAM on the GPU, None without a reading.
    pub free_vram: Option<u64>,
    pub available_ram: u64,
    /// "Share with CPU" is on: what does not fit in VRAM may go to RAM.
    pub share_with_cpu: bool,
    pub has_n_cpu_moe: bool,
    /// GPU offload is "Auto": llama.cpp moves layers to the CPU itself when
    /// the whole model cannot be on the GPU.
    pub auto_layers: bool,
}

/// The context size "Auto (fits your hardware)" chose.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutoCtx {
    pub n_ctx: u64,
    /// The model's trained context, if the header has it.
    pub trained: Option<u64>,
    /// trained | memory | minimum | no_gpu_info
    pub limited_by: String,
    /// False when even the minimum does not fit (the load warnings say more).
    pub fits: bool,
    /// Plain description for the UI.
    pub summary: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Budget {
    Ram,
    Gpu,
    Shared,
}

fn fits_budget(meta: &ModelMetadata, file_bytes: u64, input: &EstimateInput, env: &AutoCtxEnv, budget: Budget) -> bool {
    let ram_budget = (env.available_ram as f64 * AUTO_HEADROOM) as u64;
    let free = env.free_vram.unwrap_or(0);
    match budget {
        Budget::Ram | Budget::Gpu => {
            let e = estimate(meta, file_bytes, input);
            e.vram_bytes <= (free as f64 * AUTO_HEADROOM) as u64 && e.ram_bytes <= ram_budget
        }
        Budget::Shared => {
            // Same split "Share with CPU" makes (VRAM filled to 90%); the rest
            // has to fit in RAM.
            let mut all = input.clone();
            all.gpu_layers = None;
            let full = estimate(meta, file_bytes, &all);
            let split = share_split(meta, file_bytes, &all, free, env.has_n_cpu_moe);
            let in_ram = (full.weights_bytes + full.kv_cache_bytes + full.projector_bytes).saturating_sub(split.vram_bytes);
            in_ram <= ram_budget
        }
    }
}

/// "Auto" context: the largest step from [`ctx_steps`] that is no larger than
/// the trained context and whose estimate (KV cache type, projector and all
/// included, from `input`) fits with ~10% headroom: in free VRAM when the
/// model runs fully on the GPU; in free VRAM plus available RAM when Share
/// with CPU is on (or GPU offload is Auto and the model is too big for the
/// GPU anyway); in available RAM for CPU-only loads. Never below
/// [`AUTO_CTX_MIN`]: if even that does not fit, it is used anyway and
/// `fits` is false. `input.n_ctx` is ignored.
pub fn auto_n_ctx(meta: &ModelMetadata, file_bytes: u64, input: &EstimateInput, env: &AutoCtxEnv) -> AutoCtx {
    let trained = meta.context_length.filter(|&t| t > 0);
    let cap = trained.unwrap_or(AUTO_CTX_UNKNOWN_TRAINED);
    let floor = AUTO_CTX_MIN.min(cap).max(256);
    let mut steps = ctx_steps(cap);
    if steps.is_empty() {
        steps.push(floor);
    }
    let top = *steps.last().unwrap();
    let at = |n: u64| EstimateInput { n_ctx: n, ..input.clone() };
    let fmt = |n: u64| {
        let s = n.to_string();
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i) % 3 == 0 {
                out.push(',');
            }
            out.push(c);
        }
        out
    };
    let trained_note = match trained {
        Some(t) => format!("trained: {}", fmt(t)),
        None => format!("trained context unknown, capped at {}", fmt(cap)),
    };
    let uses_gpu = input.gpu_layers != Some(0);
    if uses_gpu && env.free_vram.is_none() {
        let n = AUTO_CTX_NO_GPU_INFO.min(top).max(floor);
        return AutoCtx {
            n_ctx: n,
            trained,
            limited_by: "no_gpu_info".into(),
            fits: true,
            summary: format!("No GPU memory reading, so Auto uses {} tokens ({trained_note}).", fmt(n)),
        };
    }
    // Budgets to try, best first: fully on the GPU before spilling to RAM.
    let budgets: Vec<Budget> = if !uses_gpu {
        vec![Budget::Ram]
    } else if env.share_with_cpu {
        vec![Budget::Shared]
    } else if env.auto_layers {
        vec![Budget::Gpu, Budget::Shared]
    } else {
        vec![Budget::Gpu]
    };
    for (bi, &b) in budgets.iter().enumerate() {
        // Auto layers: only fall back to a CPU split when not even the
        // minimum fits fully on the GPU.
        if bi > 0 && fits_budget(meta, file_bytes, &at(floor), env, budgets[0]) {
            break;
        }
        if let Some(&n) = steps.iter().rev().find(|&&n| fits_budget(meta, file_bytes, &at(n), env, b)) {
            let where_ = match b {
                Budget::Ram => "90% of available RAM",
                Budget::Gpu => "90% of free VRAM",
                Budget::Shared => "free VRAM plus 90% of available RAM",
            };
            let (limited_by, summary) = if n == top && trained.is_some() {
                ("trained", format!("The model's full trained context fits in {where_}."))
            } else {
                ("memory", format!("Largest size whose estimate fits in {where_} ({trained_note})."))
            };
            return AutoCtx { n_ctx: n, trained, limited_by: limited_by.into(), fits: true, summary };
        }
    }
    AutoCtx {
        n_ctx: floor,
        trained,
        limited_by: "minimum".into(),
        fits: false,
        summary: format!("Even {} tokens may not fit in memory, so Auto uses the minimum; see the warnings.", fmt(floor)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn env(free_vram: u64, ram: u64) -> AutoCtxEnv {
        AutoCtxEnv { free_vram: Some(free_vram), available_ram: ram, share_with_cpu: false, has_n_cpu_moe: true, auto_layers: false }
    }

    fn trained(t: u64) -> ModelMetadata {
        ModelMetadata { context_length: Some(t), ..meta() }
    }

    fn fits_gpu(m: &ModelMetadata, size: u64, i: &EstimateInput, n: u64, free: u64) -> bool {
        estimate(m, size, &EstimateInput { n_ctx: n, ..i.clone() }).vram_bytes <= free * 9 / 10
    }

    #[test]
    fn ctx_steps_are_power_of_two_ish_and_include_the_trained_size() {
        assert_eq!(ctx_steps(32_768), vec![2048, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768]);
        let s = ctx_steps(40_960);
        assert_eq!(*s.last().unwrap(), 40_960);
        assert!(s.contains(&32_768) && !s.contains(&49_152));
        assert_eq!(*ctx_steps(131_072).last().unwrap(), 131_072);
        assert!(ctx_steps(1000).is_empty());
    }

    #[test]
    fn auto_ctx_small_vram_picks_the_largest_step_that_fits() {
        // 8B, 4.9 GB, 128 KiB of KV per token: 6 GiB free leaves room for ~4.5k tokens.
        let m = trained(131_072);
        let free = 6 * GIB;
        let a = auto_n_ctx(&m, 4_900_000_000, &input(0), &env(free, 32 * GIB));
        assert_eq!(a.n_ctx, 4096, "{a:?}");
        assert!(a.fits && a.limited_by == "memory", "{a:?}");
        assert!(fits_gpu(&m, 4_900_000_000, &input(0), 4096, free) && !fits_gpu(&m, 4_900_000_000, &input(0), 6144, free));
        // A quantized KV cache fits more context in the same VRAM.
        let q = EstimateInput { cache_type_k: "q8_0".into(), cache_type_v: "q8_0".into(), ..input(0) };
        assert!(auto_n_ctx(&m, 4_900_000_000, &q, &env(free, 32 * GIB)).n_ctx > 4096);
        // The projector takes VRAM, so less context fits.
        let p = EstimateInput { projector_bytes: 400_000_000, ..input(0) };
        assert!(auto_n_ctx(&m, 4_900_000_000, &p, &env(free, 32 * GIB)).n_ctx < 4096);
    }

    #[test]
    fn auto_ctx_big_vram_is_capped_by_the_trained_context() {
        let big = env(24 * GIB, 64 * GIB);
        let a = auto_n_ctx(&trained(131_072), 4_900_000_000, &input(0), &big);
        assert_eq!((a.n_ctx, a.limited_by.as_str()), (131_072, "trained"), "{a:?}");
        // A trained size between steps is used in full.
        let b = auto_n_ctx(&trained(40_960), 4_900_000_000, &input(0), &big);
        assert_eq!((b.n_ctx, b.limited_by.as_str()), (40_960, "trained"), "{b:?}");
        // Unknown trained context: capped at 32k.
        let c = auto_n_ctx(&meta(), 4_900_000_000, &input(0), &big);
        assert_eq!((c.n_ctx, c.limited_by.as_str()), (32_768, "memory"), "{c:?}");
        // 24 GiB but a longer trained context: memory is the limit, below 256k.
        let d = auto_n_ctx(&trained(1_048_576), 4_900_000_000, &input(0), &big);
        assert!(d.limited_by == "memory" && d.n_ctx >= 65_536 && d.n_ctx < 262_144, "{d:?}");
    }

    #[test]
    fn auto_ctx_never_goes_below_the_minimum() {
        // The weights alone do not fit in 4 GiB with every layer on the GPU.
        let a = auto_n_ctx(&trained(131_072), 4_900_000_000, &input(0), &env(4 * GIB, 32 * GIB));
        assert_eq!((a.n_ctx, a.fits, a.limited_by.as_str()), (AUTO_CTX_MIN, false, "minimum"), "{a:?}");
        // No GPU reading: a safe fixed size.
        let n = auto_n_ctx(&trained(131_072), 4_900_000_000, &input(0), &AutoCtxEnv { free_vram: None, ..env(0, 32 * GIB) });
        assert_eq!((n.n_ctx, n.limited_by.as_str()), (AUTO_CTX_NO_GPU_INFO, "no_gpu_info"));
    }

    #[test]
    fn auto_ctx_share_with_cpu_counts_ram_too() {
        // 16 GB dense model, 8 GiB free VRAM: nothing fits fully on the GPU.
        let m = trained(131_072);
        let size = 16_000_000_000;
        let gpu_only = auto_n_ctx(&m, size, &input(0), &env(8 * GIB, 32 * GIB));
        assert!(!gpu_only.fits && gpu_only.n_ctx == AUTO_CTX_MIN, "{gpu_only:?}");
        let share = AutoCtxEnv { share_with_cpu: true, ..env(8 * GIB, 32 * GIB) };
        let a = auto_n_ctx(&m, size, &input(0), &share);
        assert!(a.fits && a.n_ctx > AUTO_CTX_MIN && a.limited_by != "minimum", "{a:?}");
        // The chosen size leaves the RAM part under 90% of available RAM, the next step does not.
        let in_ram = |n: u64| {
            let all = EstimateInput { n_ctx: n, ..input(0) };
            let full = estimate(&m, size, &all);
            let split = share_split(&m, size, &all, 8 * GIB, true);
            (full.weights_bytes + full.kv_cache_bytes).saturating_sub(split.vram_bytes)
        };
        assert!(in_ram(a.n_ctx) <= 32 * GIB * 9 / 10);
        if let Some(&next) = ctx_steps(131_072).iter().find(|&&n| n > a.n_ctx) {
            assert!(in_ram(next) > 32 * GIB * 9 / 10, "{a:?} next {next}");
        }
        // Less RAM: less context.
        let small = auto_n_ctx(&m, size, &input(0), &AutoCtxEnv { share_with_cpu: true, ..env(8 * GIB, 16 * GIB) });
        assert!(small.n_ctx < a.n_ctx, "{small:?} vs {a:?}");
        // GPU offload Auto behaves like sharing when the model can't be on the GPU...
        let auto_layers = auto_n_ctx(&m, size, &input(0), &AutoCtxEnv { auto_layers: true, ..env(8 * GIB, 32 * GIB) });
        assert_eq!(auto_layers.n_ctx, a.n_ctx);
        // ...but keeps a model that fits on the GPU there.
        let fits = auto_n_ctx(&m, 4_900_000_000, &input(0), &AutoCtxEnv { auto_layers: true, ..env(6 * GIB, 32 * GIB) });
        assert_eq!(fits.n_ctx, 4096, "{fits:?}");
        // CPU only: RAM is the only budget.
        let cpu = auto_n_ctx(&m, 4_900_000_000, &EstimateInput { gpu_layers: Some(0), ..input(0) }, &env(0, 16 * GIB));
        assert!(cpu.fits && cpu.n_ctx >= 65_536, "{cpu:?}");
    }

    #[test]
    fn share_split_fits_splits_moe_and_zero_vram() {
        // Fits: 4.9 GB 8B model, 12 GB free.
        let s = share_split(&meta(), 4_900_000_000, &input(4096), 12 * GIB, true);
        assert!(s.all_fit && s.gpu_layers == 33 && s.n_cpu_moe.is_none(), "{s:?}");
        // Dense model too big for 8 GB: a split under 90% of free VRAM, as many layers as fit.
        let free = 8 * GIB;
        let s = share_split(&meta(), 16_000_000_000, &input(8192), free, true);
        assert!(!s.all_fit && s.gpu_layers > 0 && s.gpu_layers < 33 && s.n_cpu_moe.is_none(), "{s:?}");
        assert!(s.vram_bytes <= free * 9 / 10, "{s:?}");
        let mut one_more = input(8192);
        one_more.gpu_layers = Some(s.gpu_layers + 1);
        assert!(estimate(&meta(), 16_000_000_000, &one_more).vram_bytes > free * 9 / 10, "largest that fits");
        assert!(s.summary.contains(&format!("{} of 33 layers", s.gpu_layers)), "{}", s.summary);
        // MoE with --n-cpu-moe: every layer on the GPU, experts of the first N on the CPU.
        let moe = ModelMetadata { expert_count: Some(64), ..meta() };
        let m = share_split(&moe, 16_000_000_000, &input(8192), free, true);
        let n = m.n_cpu_moe.expect("moe split");
        assert!(m.gpu_layers == 33 && n > 0 && n <= 32 && m.vram_bytes <= free * 9 / 10, "{m:?}");
        // MoE without the flag falls back to the layer split.
        let f = share_split(&moe, 16_000_000_000, &input(8192), free, false);
        assert!(f.n_cpu_moe.is_none() && f.gpu_layers < 33, "{f:?}");
        // No free VRAM: CPU only.
        let z = share_split(&meta(), 16_000_000_000, &input(8192), 0, true);
        assert!(z.gpu_layers == 0 && z.summary.contains("CPU"), "{z:?}");
    }

    fn meta() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("llama".into()),
            block_count: Some(32),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            vocab_size: Some(128_256),
            ..Default::default()
        }
    }

    fn input(ctx: u64) -> EstimateInput {
        EstimateInput {
            n_ctx: ctx,
            gpu_layers: None,
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            kv_offload: true,
            n_ubatch: 512,
            parallel: 1,
            projector_bytes: 0,
        }
    }

    #[test]
    fn projector_counts_toward_vram_and_share_split() {
        const PROJ: u64 = 900_000_000;
        let plain = input(8192);
        let with = EstimateInput { projector_bytes: PROJ, ..input(8192) };
        // Full offload: the projector is on the GPU.
        let a = estimate(&meta(), 4_900_000_000, &plain);
        let b = estimate(&meta(), 4_900_000_000, &with);
        assert_eq!(a.projector_bytes, 0);
        assert_eq!(b.projector_bytes, PROJ);
        assert_eq!(b.vram_bytes, a.vram_bytes + PROJ);
        assert_eq!(b.ram_bytes, a.ram_bytes);
        // CPU only: it is in RAM instead.
        let cpu = |i: &EstimateInput| estimate(&meta(), 4_900_000_000, &EstimateInput { gpu_layers: Some(0), ..i.clone() });
        assert_eq!(cpu(&with).vram_bytes, 0);
        assert_eq!(cpu(&with).ram_bytes, cpu(&plain).ram_bytes + PROJ);
        // A model that just fits without the projector no longer fits with it.
        let free = (a.vram_bytes as f64 / 0.90) as u64 + 1;
        let s = share_split(&meta(), 4_900_000_000, &plain, free, true);
        assert!(s.all_fit, "{s:?}");
        let p = share_split(&meta(), 4_900_000_000, &with, free, true);
        assert!(!p.all_fit && p.gpu_layers < 33 && p.vram_bytes <= free * 9 / 10, "{p:?}");
        assert!(p.vram_bytes >= PROJ, "split VRAM includes the projector: {p:?}");
        // Fewer layers fit when the projector takes VRAM.
        let budget = 4 * GIB;
        assert!(max_gpu_layers_for_budget(&meta(), 16_000_000_000, &with, budget) < max_gpu_layers_for_budget(&meta(), 16_000_000_000, &plain, budget));
    }

    #[test]
    fn llama3_8b_kv_cache_matches_known_value() {
        // Llama-3-8B: 32 layers, 8 KV heads, head_dim 128, f16 -> 128 KiB per token.
        let e = estimate(&meta(), 4_900_000_000, &input(8192));
        assert_eq!(e.kv_cache_bytes, 32 * 8192 * 8 * 128 * 2 * 2);
        assert_eq!(e.kv_cache_bytes, 1 << 30); // 1 GiB at 8k context
    }

    #[test]
    fn hybrid_attention_models_keep_kv_on_every_fourth_layer() {
        let dense = estimate(&meta(), 2_800_000_000, &input(16_384)).kv_cache_bytes;
        let qwen35 = ModelMetadata { architecture: Some("qwen35".into()), ..meta() };
        let h = estimate(&qwen35, 2_800_000_000, &input(16_384));
        // 32 layers -> 8 with a KV cache, plus a fixed recurrent state.
        assert_eq!(h.kv_cache_bytes, dense / 4 + 2_800_000_000 / 50);
        assert!(h.notes.iter().any(|n| n.contains("Hybrid attention")));
        // So Auto fits more context in the same VRAM.
        let e = env(6 * GIB, 32 * GIB);
        let m = ModelMetadata { context_length: Some(262_144), ..qwen35 };
        assert!(auto_n_ctx(&m, 2_800_000_000, &input(0), &e).n_ctx > auto_n_ctx(&trained(262_144), 2_800_000_000, &input(0), &e).n_ctx);
    }

    const MIB: u64 = 1 << 20;

    /// Gemma-4-E4B Q4_K_M header (lmstudio-community), as read from the file.
    fn gemma4_e4b() -> ModelMetadata {
        let pattern: Vec<bool> = (0..42).map(|il| il % 6 != 5).collect();
        ModelMetadata {
            architecture: Some("gemma4".into()),
            block_count: Some(42),
            context_length: Some(131_072),
            embedding_length: Some(2560),
            head_count: Some(8),
            head_count_kv: Some(2),
            key_length: Some(512),
            value_length: Some(512),
            vocab_size: Some(262_144),
            sliding_window: Some(512),
            swa_layers: Some(pattern),
            key_length_swa: Some(256),
            value_length_swa: Some(256),
            shared_kv_layers: Some(18),
            ..Default::default()
        }
    }

    /// Qwen3.8-27B Q4_K_M header: hybrid, 65 blocks of which 1 is an MTP layer.
    fn qwen38_27b() -> ModelMetadata {
        ModelMetadata {
            architecture: Some("qwen35".into()),
            block_count: Some(65),
            context_length: Some(262_144),
            embedding_length: Some(5120),
            head_count: Some(24),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            mtp_layers: Some(1),
            full_attention_interval: Some(4),
            ..Default::default()
        }
    }

    #[test]
    fn gemma4_kv_matches_the_measured_1576_mib_at_98304() {
        let m = gemma4_e4b();
        let l = kv_layout(&m);
        assert_eq!((l.global_layers, l.swa_layers, l.shared_layers, l.window), (4, 20, 18, Some(512)));
        let e = estimate(&m, 5_335_291_936, &input(98_304));
        // llama-server b12292 on the RTX 5060: KV buffer 1576 MiB (llama.cpp:
        // 4 x 98304 x 4 KiB + 20 x 1024 x 2 KiB).
        let measured = 1576 * MIB;
        assert!(e.kv_cache_bytes.abs_diff(measured) <= measured * 15 / 100, "{} MiB", e.kv_cache_bytes / MIB);
        assert_eq!(e.kv_cache_bytes, measured);
        assert!(e.notes.iter().any(|n| n.contains("Sliding-window attention")), "{:?}", e.notes);
        // Was ~16.9 GB with every block at full size; the SWA cache does not grow with n_ctx.
        let small = estimate(&m, 5_335_291_936, &input(8192)).kv_cache_bytes;
        assert_eq!(small, 4 * 8192 * 4096 + 20 * 1024 * 2048);
        assert_eq!(swa_cache_cells(512, 512, 512, 1), 512, "never more than n_ctx");
    }

    #[test]
    fn gemma4_auto_ctx_reaches_the_trained_context_when_sharing() {
        // 8 GiB card with 7 GiB free, 31.6 GB RAM, Share with CPU: was 98304, now trained.
        let share = AutoCtxEnv { share_with_cpu: true, ..env(7 * GIB, 28 * GIB) };
        let a = auto_n_ctx(&gemma4_e4b(), 5_335_291_936, &input(0), &share);
        assert_eq!((a.n_ctx, a.limited_by.as_str()), (131_072, "trained"), "{a:?}");
    }

    #[test]
    fn qwen38_27b_kv_matches_the_measured_8192_mib_at_131072() {
        let m = qwen38_27b();
        let l = kv_layout(&m);
        assert_eq!((l.global_layers, l.mtp_layers), (16, 1));
        let size = 16_810_714_336;
        let e = estimate(&m, size, &input(131_072));
        let recurrent = size / 50;
        assert_eq!(e.kv_cache_bytes - recurrent, 8192 * MIB, "16 full-attention layers x 131072 x 4 KiB");
        assert!(e.notes.iter().any(|n| n.contains("multi-token-prediction")), "{:?}", e.notes);
        // A row stored before MTP was read keeps the old count (17 layers).
        let old = ModelMetadata { mtp_layers: None, full_attention_interval: None, ..m };
        assert_eq!(kv_layout(&old).global_layers, 17);
    }

    /// Read-only check against real model files (headers only):
    /// `LAS_REAL_MODELS=<dir> cargo test -p model-registry real_kv_estimates -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_kv_estimates() {
        let Ok(dir) = std::env::var("LAS_REAL_MODELS") else { return };
        for m in crate::scan::scan(std::path::Path::new(&dir)).models {
            let l = kv_layout(&m.metadata);
            let kv = |n| estimate(&m.metadata, m.total_size, &input(n)).kv_cache_bytes / MIB;
            println!("{} [{:?}] {l:?}: KV f16 98304 = {} MiB, 131072 = {} MiB", m.display_name, m.metadata.architecture, kv(98_304), kv(131_072));
        }
    }

    #[test]
    fn swa_patterns_numbers_defaults_and_unknowns() {
        let base = ModelMetadata { sliding_window: Some(1024), ..meta() };
        // No pattern known for llama: every layer stays global (unchanged estimate).
        assert_eq!(estimate(&base, 1, &input(32_768)).kv_cache_bytes, estimate(&meta(), 1, &input(32_768)).kv_cache_bytes);
        // Pattern number 6: every 6th layer is global (32 layers -> 5 global, 27 SWA).
        let p6 = ModelMetadata { sliding_window_pattern: Some(6), ..base.clone() };
        assert_eq!(kv_layout(&p6), KvLayout { global_layers: 5, swa_layers: 27, window: Some(1024), shared_layers: 0, mtp_layers: 0 });
        // gemma3 has no key: llama.cpp's default pattern 6.
        let g3 = ModelMetadata { architecture: Some("gemma3".into()), ..base.clone() };
        assert_eq!(kv_layout(&g3).swa_layers, 27);
        let e = estimate(&g3, 1, &input(32_768)).kv_cache_bytes;
        assert_eq!(e, 5 * 32_768 * 4096 + 27 * 1536 * 4096);
        // No window: a pattern alone changes nothing.
        let nowin = ModelMetadata { sliding_window_pattern: Some(6), ..meta() };
        assert_eq!(kv_layout(&nowin).global_layers, 32);
        // A dense model with MTP layers: those keep no cache.
        let glm = ModelMetadata { mtp_layers: Some(1), ..meta() };
        assert_eq!(kv_layout(&glm).global_layers, 31);
    }

    #[test]
    fn dense_models_without_the_new_keys_are_unchanged() {
        // Every new field None: the plain formula, bit for bit.
        for ctx in [2048, 8192, 32_768, 131_072] {
            let e = estimate(&meta(), 4_900_000_000, &input(ctx));
            assert_eq!(e.kv_cache_bytes, (32.0 * ctx as f64 * (8.0 * 128.0 * 2.0 * 2.0)) as u64);
            assert!(!e.notes.iter().any(|n| n.contains("Sliding") || n.contains("reuse") || n.contains("multi-token")), "{:?}", e.notes);
        }
    }

    #[test]
    fn kv_grows_linearly_and_quantized_cache_is_smaller() {
        let a = estimate(&meta(), 1, &input(4096)).kv_cache_bytes;
        let b = estimate(&meta(), 1, &input(8192)).kv_cache_bytes;
        assert_eq!(b, a * 2);
        let mut q = input(8192);
        q.cache_type_k = "q8_0".into();
        q.cache_type_v = "q8_0".into();
        assert!(estimate(&meta(), 1, &q).kv_cache_bytes < b);
    }

    #[test]
    fn cpu_only_uses_no_vram_and_budget_search_is_monotonic() {
        let mut i = input(4096);
        i.gpu_layers = Some(0);
        let e = estimate(&meta(), 4_900_000_000, &i);
        assert_eq!(e.vram_bytes, 0);
        assert!(e.ram_bytes >= 4_900_000_000);
        let small = max_gpu_layers_for_budget(&meta(), 4_900_000_000, &input(4096), 2 << 30);
        let big = max_gpu_layers_for_budget(&meta(), 4_900_000_000, &input(4096), 8 << 30);
        assert!(small < big && big == 33, "small={small} big={big}");
    }
}

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
    let kv_cache_bytes = if layers > 0 && kv_heads > 0 && k_len > 0 {
        let per_tok_k = (kv_heads * k_len) as f64 * kv_bytes_per_element(&input.cache_type_k);
        let per_tok_v = (kv_heads * v_len) as f64 * kv_bytes_per_element(&input.cache_type_v);
        (layers as f64 * input.n_ctx as f64 * (per_tok_k + per_tok_v)) as u64
    } else {
        notes.push("KV cache size unknown: header lacks attention dimensions.".into());
        0
    };
    if meta.architecture.as_deref().is_some_and(|a| a.starts_with("gemma") || a.contains("swa")) {
        notes.push("This architecture may use sliding-window attention, so the real KV cache can be smaller.".into());
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

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

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

//! Pins today's memory-estimate behaviour for models the performance-fix round
//! must NOT change: non-SWA, non-shared-KV architectures (dense llama, qwen35
//! hybrid, MoE). Every case is a fingerprint string of KV bytes, total
//! estimate, `share_split` and `auto_n_ctx`. If a fix changes one of these,
//! it must be a documented, intentional change (only: qwen35 hybrid models
//! that carry an MTP/nextn layer; ModelMetadata has no such field yet, so that
//! case is not pinned here).
//!
//! Regenerate expectations with:
//!   DUMP=1 cargo test -p model-registry --test estimate_equivalence dump -- --nocapture

use model_registry::estimate::{auto_n_ctx, estimate, share_split, AutoCtxEnv, EstimateInput};
use model_registry::gguf;
use model_registry::ModelMetadata;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn llama_dense() -> (ModelMetadata, u64) {
    (
        ModelMetadata {
            architecture: Some("llama".into()),
            block_count: Some(32),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            vocab_size: Some(128_256),
            context_length: Some(131_072),
            ..Default::default()
        },
        4_900_000_000,
    )
}

/// Qwen3.5-style hybrid, 27B-class (no MTP layer: block_count = real layers).
fn qwen35_hybrid() -> (ModelMetadata, u64) {
    (
        ModelMetadata {
            architecture: Some("qwen35".into()),
            block_count: Some(64),
            embedding_length: Some(5120),
            head_count: Some(24),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            vocab_size: Some(248_320),
            context_length: Some(262_144),
            ..Default::default()
        },
        16_500_000_000,
    )
}

fn moe() -> (ModelMetadata, u64) {
    (
        ModelMetadata {
            architecture: Some("qwen3moe".into()),
            block_count: Some(48),
            embedding_length: Some(2048),
            head_count: Some(32),
            head_count_kv: Some(4),
            key_length: Some(128),
            value_length: Some(128),
            expert_count: Some(128),
            vocab_size: Some(151_936),
            context_length: Some(262_144),
            ..Default::default()
        },
        18_600_000_000,
    )
}

fn input(n_ctx: u64, gpu_layers: Option<u64>, cache: &str) -> EstimateInput {
    EstimateInput {
        n_ctx,
        gpu_layers,
        cache_type_k: cache.into(),
        cache_type_v: cache.into(),
        kv_offload: true,
        n_ubatch: 512,
        parallel: 1,
        projector_bytes: 0,
    }
}

fn env(free_vram: Option<u64>, ram: u64, share: bool, moe_flag: bool, auto_layers: bool) -> AutoCtxEnv<'static> {
    AutoCtxEnv { free_vram, available_ram: ram, share_with_cpu: share, has_n_cpu_moe: moe_flag, auto_layers, linked: &[] }
}

/// All pinned numbers for one model/combination, as one comparable line.
fn fingerprint(m: &(ModelMetadata, u64), n_ctx: u64, gpu: Option<u64>, cache: &str, free_vram: u64, ram: u64) -> String {
    let (meta, size) = (&m.0, m.1);
    let i = input(n_ctx, gpu, cache);
    let e = estimate(meta, size, &i);
    let s = share_split(meta, size, &i, free_vram, true);
    let a_share = auto_n_ctx(meta, size, &i, &env(Some(free_vram), ram, true, true, false));
    let a_gpu = auto_n_ctx(meta, size, &i, &env(Some(free_vram), ram, false, true, false));
    let a_auto = auto_n_ctx(meta, size, &i, &env(Some(free_vram), ram, false, true, true));
    format!(
        "kv={} weights={} compute={} vram={} ram={} layers={}/{} | split=gpu{}/{} moe={:?} fit={} vram={} | auto_share={}:{}:{} auto_gpu={}:{}:{} auto_autolayers={}:{}:{}",
        e.kv_cache_bytes, e.weights_bytes, e.compute_buffer_bytes, e.vram_bytes, e.ram_bytes, e.offloaded_layers, e.total_layers,
        s.gpu_layers, s.total_layers, s.n_cpu_moe, s.all_fit, s.vram_bytes,
        a_share.n_ctx, a_share.limited_by, a_share.fits,
        a_gpu.n_ctx, a_gpu.limited_by, a_gpu.fits,
        a_auto.n_ctx, a_auto.limited_by, a_auto.fits,
    )
}

fn cases() -> Vec<(String, String)> {
    let models: [(&str, (ModelMetadata, u64)); 3] = [("llama_dense", llama_dense()), ("qwen35_hybrid", qwen35_hybrid()), ("moe", moe())];
    // (n_ctx, gpu_layers, cache type, free VRAM, available RAM)
    let combos: [(u64, Option<u64>, &str, u64, u64); 6] = [
        (4096, None, "f16", 6 * GIB, 22 * GIB),
        (32_768, None, "f16", 6 * GIB, 22 * GIB),
        (131_072, Some(16), "f16", 7 * GIB, 22 * GIB),
        (131_072, None, "q8_0", 24 * GIB, 64 * GIB),
        (65_536, Some(0), "f16", 7 * GIB, 16 * GIB),
        (8192, None, "q4_0", 2 * GIB + 512 * MIB, 8 * GIB),
    ];
    let mut out = Vec::new();
    for (name, m) in &models {
        for (n_ctx, gpu, cache, vram, ram) in combos {
            let label = format!("{name} ctx={n_ctx} ngl={gpu:?} kv={cache} vram={}M ram={}M", vram / MIB, ram / MIB);
            out.push((label, fingerprint(m, n_ctx, gpu, cache, vram, ram)));
        }
    }
    out
}

#[test]
fn dump() {
    if std::env::var("DUMP").is_err() {
        return;
    }
    for (l, f) in cases() {
        println!("(\"{l}\", \"{f}\"),");
    }
}

#[test]
fn non_swa_models_keep_their_estimates() {
    let got = cases();
    assert_eq!(got.len(), EXPECTED.len(), "run the dump test and paste the lines into EXPECTED");
    let mut bad = Vec::new();
    for ((l, f), (el, ef)) in got.iter().zip(EXPECTED.iter()) {
        assert_eq!(l, el, "case order changed");
        if f != ef {
            bad.push(format!("{l}\n  expected {ef}\n  got      {f}"));
        }
    }
    assert!(bad.is_empty(), "estimate behaviour changed for non-SWA models:\n{}", bad.join("\n"));
}

#[test]
fn hand_checked_anchor_values() {
    // llama dense, 32k f16: 32 layers * 32768 * (8*128*2 + 8*128*2) = 4 GiB exactly.
    let (m, size) = llama_dense();
    assert_eq!(estimate(&m, size, &input(32_768, None, "f16")).kv_cache_bytes, 4 * GIB);
    // qwen35 hybrid: every 4th of 64 layers keeps KV (16), plus recurrent state size/50.
    let (m, size) = qwen35_hybrid();
    let per_tok = 4 * 256 * 2 * 2;
    assert_eq!(estimate(&m, size, &input(32_768, None, "f16")).kv_cache_bytes, 16 * 32_768 * per_tok + size / 50);
    // MoE: dense KV formula over all 48 layers (expert_count only adds a note).
    let (m, size) = moe();
    assert_eq!(estimate(&m, size, &input(32_768, None, "f16")).kv_cache_bytes, 48 * 32_768 * (4 * 128 * 2 * 2));
}

/// KNOWN WRONG TODAY. Gemma-4-E4B uses sliding-window attention (1 global layer
/// in 6, window 512, SWA head dim 256) and shares KV between its last 18 layers,
/// so llama-server allocates far less KV than the dense formula. Measured with
/// llama-server b12292-cuda-13.5, `-c 98304`, f16 KV: 1576 MiB total (1536 MiB
/// global-layer cache + 40 MiB SWA cache). The estimate ignores all of that.
/// Was #[ignore] until the PERF-1A fix (2026-11-12).
#[test]
fn gemma4_e4b_kv_matches_llama_server() {
    // Needs the real header (SWA / shared-KV fields): LAS_REAL_GEMMA4 = path to gemma-4-E4B-it-Q4_K_M.gguf.
    let Ok(real_path) = std::env::var("LAS_REAL_GEMMA4") else {
        eprintln!("skipping: set LAS_REAL_GEMMA4 to gemma-4-E4B-it-Q4_K_M.gguf");
        return;
    };
    let real_file = std::path::Path::new(&real_path);
    // Prefer the real header; fall back to hardcoded values.
    let meta = gguf::read_metadata(real_file).unwrap_or_else(|_| ModelMetadata {
        architecture: Some("gemma4".into()),
        block_count: Some(42),
        embedding_length: Some(2560),
        head_count: Some(8),
        head_count_kv: Some(2),
        key_length: Some(512),
        value_length: Some(512),
        vocab_size: Some(262_144),
        context_length: Some(131_072),
        ..Default::default()
    });
    // The fix is expected to add SWA / shared-KV fields to ModelMetadata and
    // read them from the header: sliding_window=512, sliding_window_pattern
    // (1 global in 6), key_length_swa=256, shared_kv_layers=18.
    let e = estimate(&meta, 5_000_000_000, &input(98_304, None, "f16"));
    let real = 1576 * MIB;
    assert!(e.kv_cache_bytes.abs_diff(real) <= real / 20, "estimated KV {} MiB vs real {} MiB", e.kv_cache_bytes / MIB, real / MIB);
}

const EXPECTED: &[(&str, &str)] = &[
    ("llama_dense ctx=4096 ngl=None kv=f16 vram=6144M ram=22528M", "kv=536870912 weights=4900000000 compute=296222720 vram=5733093632 ram=268435456 layers=33/33 | split=gpu33/33 moe=None fit=true vram=5733093632 | auto_share=131072:trained:true auto_gpu=4096:memory:true auto_autolayers=4096:memory:true"),
    ("llama_dense ctx=32768 ngl=None kv=f16 vram=6144M ram=22528M", "kv=4294967296 weights=4900000000 compute=296222720 vram=9491190016 ram=268435456 layers=33/33 | split=gpu19/33 moe=None fit=false vram=5590294799 | auto_share=131072:trained:true auto_gpu=4096:memory:true auto_autolayers=4096:memory:true"),
    ("llama_dense ctx=131072 ngl=Some(16) kv=f16 vram=7168M ram=22528M", "kv=17179869184 weights=4900000000 compute=296222720 vram=11001613838 ram=11642913522 layers=16/33 | split=gpu9/33 moe=None fit=false vram=6318005224 | auto_share=131072:trained:true auto_gpu=49152:memory:true auto_autolayers=49152:memory:true"),
    ("llama_dense ctx=131072 ngl=None kv=q8_0 vram=24576M ram=65536M", "kv=9126805504 weights=4900000000 compute=296222720 vram=14323028224 ram=268435456 layers=33/33 | split=gpu33/33 moe=None fit=true vram=14323028224 | auto_share=131072:trained:true auto_gpu=131072:trained:true auto_autolayers=131072:trained:true"),
    ("llama_dense ctx=65536 ngl=Some(0) kv=f16 vram=7168M ram=16384M", "kv=8589934592 weights=4900000000 compute=296222720 vram=0 ram=13786157312 layers=0/33 | split=gpu15/33 moe=None fit=false vram=6428011170 | auto_share=65536:memory:true auto_gpu=65536:memory:true auto_autolayers=65536:memory:true"),
    ("llama_dense ctx=8192 ngl=None kv=q4_0 vram=2560M ram=8192M", "kv=301989888 weights=4900000000 compute=296222720 vram=5498212608 ram=268435456 layers=33/33 | split=gpu13/33 moe=None fit=false vram=2345491463 | auto_share=131072:trained:true auto_gpu=2048:minimum:false auto_autolayers=131072:trained:true"),
    ("qwen35_hybrid ctx=4096 ngl=None kv=f16 vram=6144M ram=22528M", "kv=598435456 weights=16500000000 compute=550502400 vram=17648937856 ram=268435456 layers=65/65 | split=gpu19/65 moe=None fit=false vram=5548506610 | auto_share=131072:memory:true auto_gpu=2048:minimum:false auto_autolayers=131072:memory:true"),
    ("qwen35_hybrid ctx=32768 ngl=None kv=f16 vram=6144M ram=22528M", "kv=2477483648 weights=16500000000 compute=550502400 vram=19527986048 ram=268435456 layers=65/65 | split=gpu17/65 moe=None fit=false vram=5513844276 | auto_share=131072:memory:true auto_gpu=2048:minimum:false auto_autolayers=131072:memory:true"),
    ("qwen35_hybrid ctx=131072 ngl=Some(16) kv=f16 vram=7168M ram=22528M", "kv=8919934592 weights=16500000000 compute=550502400 vram=6807717068 ram=19431155380 layers=16/65 | split=gpu15/65 moe=None fit=false vram=6416641151 | auto_share=131072:memory:true auto_gpu=98304:memory:true auto_autolayers=98304:memory:true"),
    ("qwen35_hybrid ctx=131072 ngl=None kv=q8_0 vram=24576M ram=65536M", "kv=4893402752 weights=16500000000 compute=550502400 vram=21943905152 ram=268435456 layers=65/65 | split=gpu65/65 moe=None fit=true vram=21943905152 | auto_share=262144:trained:true auto_gpu=131072:memory:true auto_autolayers=131072:memory:true"),
    ("qwen35_hybrid ctx=65536 ngl=Some(0) kv=f16 vram=7168M ram=16384M", "kv=4624967296 weights=16500000000 compute=550502400 vram=0 ram=21675469696 layers=0/65 | split=gpu19/65 moe=None fit=false vram=6725492840 | auto_share=2048:minimum:false auto_gpu=2048:minimum:false auto_autolayers=2048:minimum:false"),
    ("qwen35_hybrid ctx=8192 ngl=None kv=q4_0 vram=2560M ram=8192M", "kv=480994944 weights=16500000000 compute=550502400 vram=17531497344 ram=268435456 layers=65/65 | split=gpu7/65 moe=None fit=false vram=2379224931 | auto_share=2048:minimum:false auto_gpu=2048:minimum:false auto_autolayers=2048:minimum:false"),
    ("moe ctx=4096 ngl=None kv=f16 vram=6144M ram=22528M", "kv=402653184 weights=18600000000 compute=327942144 vram=19330595328 ram=268435456 layers=49/49 | split=gpu49/49 moe=Some(42) fit=false vram=5496845328 | auto_share=65536:memory:true auto_gpu=2048:minimum:false auto_autolayers=65536:memory:true"),
    ("moe ctx=32768 ngl=None kv=f16 vram=6144M ram=22528M", "kv=3221225472 weights=18600000000 compute=327942144 vram=22149167616 ram=268435456 layers=49/49 | split=gpu12/49 moe=None fit=false vram=5671915728 | auto_share=65536:memory:true auto_gpu=2048:minimum:false auto_autolayers=65536:memory:true"),
    ("moe ctx=131072 ngl=Some(16) kv=f16 vram=7168M ram=22528M", "kv=12884901888 weights=18600000000 compute=327942144 vram=10608726433 ram=21472553055 layers=16/49 | split=gpu10/49 moe=None fit=false vram=6753432324 | auto_share=65536:memory:true auto_gpu=8192:memory:true auto_autolayers=8192:memory:true"),
    ("moe ctx=131072 ngl=None kv=q8_0 vram=24576M ram=65536M", "kv=6845104128 weights=18600000000 compute=327942144 vram=25773046272 ram=268435456 layers=49/49 | split=gpu49/49 moe=Some(8) fit=false vram=23138046272 | auto_share=262144:trained:true auto_gpu=65536:memory:true auto_autolayers=65536:memory:true"),
    ("moe ctx=65536 ngl=Some(0) kv=f16 vram=7168M ram=16384M", "kv=6442450944 weights=18600000000 compute=327942144 vram=0 ram=25370393088 layers=0/49 | split=gpu12/49 moe=None fit=false vram=6460787272 | auto_share=2048:minimum:false auto_gpu=2048:minimum:false auto_autolayers=2048:minimum:false"),
    ("moe ctx=8192 ngl=None kv=q4_0 vram=2560M ram=8192M", "kv=226492416 weights=18600000000 compute=327942144 vram=19154434560 ram=268435456 layers=49/49 | split=gpu5/49 moe=None fit=false vram=2249012798 | auto_share=2048:minimum:false auto_gpu=2048:minimum:false auto_autolayers=2048:minimum:false"),
];

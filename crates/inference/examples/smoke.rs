//! Phase 1 end-to-end smoke test against real hardware and real files.
//!
//! cargo run -p inference --example smoke -- <model.gguf>
//!
//! 1. Fetches the llama.cpp release list and installs the recommended GPU
//!    runtime and the CPU runtime into %APPDATA%\LocalAIStudio\runtimes.
//! 2. Loads the model with all layers on the GPU, checks /props and /tokenize.
//! 3. Loads it CPU-only on the CPU runtime with a different context length.
//! 4. Unloads and checks that no llama-server process is left.

use std::path::PathBuf;
use std::time::Instant;

use inference::{Engine, GpuHint, GpuLayers, LoadConfig, LoadRequest, RuntimeManager};
use tokio_util::sync::CancellationToken;

fn llama_server_count() -> usize {
    let out = std::process::Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/NH"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).matches("llama-server.exe").count()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(std::env::args().nth(1).expect("usage: smoke <model.gguf>"));
    let paths = app_core::AppPaths::resolve()?;
    paths.ensure_dirs()?;
    let hw = system_monitor::SystemMonitor::new()?.hardware_info();
    let gpu = hw.gpus.first();
    let hint = GpuHint {
        nvidia: gpu.is_some(),
        compute_capability: gpu.and_then(|g| g.compute_capability.as_deref()).and_then(|c| {
            let (a, b) = c.split_once('.')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }),
        cuda_driver: hw.cuda_driver_version.as_deref().and_then(|c| {
            let (a, b) = c.split_once('.')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }),
    };
    println!("GPU hint: {hint:?}");
    let http = reqwest::Client::builder().build()?;
    let rm = RuntimeManager::new(paths.runtimes_dir.clone(), http);
    let avail = rm.fetch_available(&hint).await?;
    for a in &avail {
        println!("available: {} {:.0} MB recommended={} — {}", a.id, a.total_size as f64 / 1e6, a.recommended, a.note);
    }
    let cancel = CancellationToken::new();
    let progress = |p: inference::InstallProgress| {
        if p.phase != "download" || p.total.is_some_and(|t| p.downloaded == t) {
            println!("  [{}] {} {:?}", p.runtime_id, p.phase, p.asset);
        }
    };
    let gpu_rt = rm.install(avail.iter().find(|a| a.recommended).unwrap(), &cancel, &progress).await?;
    let cpu_rt = rm.install(avail.iter().find(|a| a.variant == "cpu").unwrap(), &cancel, &progress).await?;
    println!("installed: {} -> {}", gpu_rt.id, gpu_rt.server_path.display());
    println!("installed: {} -> {}", cpu_rt.id, cpu_rt.server_path.display());

    let engine = Engine::new();
    let before = llama_server_count();

    // GPU load
    let t = Instant::now();
    let cfg = LoadConfig { n_ctx: 8192, gpu_layers: GpuLayers::All, ..Default::default() };
    let st = engine
        .load(LoadRequest {
            model_id: "smoke".into(),
            instance_id: "gpu".into(),
            model_path: model.clone(),
            projector_path: None,
            runtime: gpu_rt.clone(),
            config: cfg,
            alias: None,
        })
        .await;
    let st = match st {
        Ok(s) => s,
        Err(e) => {
            println!("GPU load failed: {e}\n--- log tail ---\n{}", engine.logs(40).join("\n"));
            return Err(e.into());
        }
    };
    println!(
        "GPU loaded in {:.1}s (wall {:.1}s): pid={:?} n_ctx={:?} offloaded={:?} unsupported={:?}",
        st.load_seconds.unwrap_or(0.0),
        t.elapsed().as_secs_f64(),
        st.pid,
        st.props.as_ref().and_then(|p| p.n_ctx),
        st.gpu_layers_offloaded,
        st.unsupported_settings
    );
    let snap = system_monitor::SystemMonitor::new()?.snapshot(st.pid);
    println!(
        "  backend RAM {:.0} MB, backend VRAM {:?}, GPU VRAM used {:.0} MB",
        snap.backend_memory_bytes.unwrap_or(0) as f64 / 1e6,
        snap.backend_vram_bytes.map(|b| b as f64 / 1e6),
        snap.gpus.first().map(|g| g.vram_used_bytes as f64 / 1e6).unwrap_or(0.0)
    );
    let c = engine.client().expect("client");
    let toks = c.tokenize("Hello from Local AI Studio!", false).await?;
    println!("  tokenize -> {} tokens {:?}", toks.len(), toks);
    let prompt = c.apply_template(&serde_json::json!([{"role": "user", "content": "Hi"}])).await?;
    println!("  apply-template -> {prompt:?}");
    assert_eq!(llama_server_count(), before + 1);

    // Streaming chat with real stats.
    let msgs = serde_json::json!([{"role": "user", "content": "Reply with exactly: Hello there, friend."}]);
    let params = inference::SamplingParams { temperature: Some(0.0), max_tokens: Some(400), ..Default::default() };
    let mut deltas = 0;
    let out = inference::chat::stream_chat(&c, &msgs, &params, &CancellationToken::new(), |_| deltas += 1).await?;
    println!(
        "  chat: {deltas} deltas, reasoning {} chars, content {:?}, stats {:?}",
        out.reasoning.len(),
        out.content.chars().take(80).collect::<String>(),
        out.stats
    );
    assert!(out.stats.tokens_per_sec.unwrap_or(0.0) > 0.0 && out.stats.completion_tokens.unwrap_or(0) > 0);

    // Stop mid-generation and confirm the server actually stopped.
    let long = serde_json::json!([{"role": "user", "content": "Count from 1 to 500, one number per line."}]);
    let stop = CancellationToken::new();
    let stop2 = stop.clone();
    let mut n = 0;
    let t = Instant::now();
    let out = inference::chat::stream_chat(&c, &long, &inference::SamplingParams { max_tokens: Some(3000), ..Default::default() }, &stop, |_| {
        n += 1;
        if n == 20 {
            stop2.cancel();
        }
    })
    .await?;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let slots = c.slots().await?;
    let busy = slots.as_array().map(|a| a.iter().any(|s| s["is_processing"].as_bool() == Some(true))).unwrap_or(true);
    println!("  stop: reason={} after {} deltas in {:.0} ms; server busy after stop = {busy}", out.stats.stop_reason, n, t.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(out.stats.stop_reason, "cancelled");
    assert!(!busy, "llama-server kept generating after the stream was dropped");

    // CPU load on the CPU runtime, different context to prove -c is honoured
    let cfg = LoadConfig { n_ctx: 4096, gpu_layers: GpuLayers::CpuOnly, ..Default::default() };
    let st = engine
        .load(LoadRequest {
            model_id: "smoke".into(),
            instance_id: "cpu".into(),
            model_path: model.clone(),
            projector_path: None,
            runtime: cpu_rt,
            config: cfg,
            alias: None,
        })
        .await?;
    println!(
        "CPU loaded in {:.1}s: pid={:?} n_ctx={:?} offloaded={:?}",
        st.load_seconds.unwrap_or(0.0),
        st.pid,
        st.props.as_ref().and_then(|p| p.n_ctx),
        st.gpu_layers_offloaded
    );
    assert_eq!(llama_server_count(), before + 1, "previous backend must be gone");
    assert_eq!(st.props.as_ref().and_then(|p| p.n_ctx), Some(4096));

    engine.unload().await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    println!("after unload: llama-server processes = {} (before = {before})", llama_server_count());
    assert_eq!(llama_server_count(), before);

    // Invalid model file must fail cleanly, not hang or crash
    let bad = std::env::temp_dir().join("localaistudio-bad.gguf");
    std::fs::write(&bad, b"this is not a gguf file at all")?;
    let r = engine
        .load(LoadRequest {
            model_id: "bad".into(),
            instance_id: "bad".into(),
            model_path: bad.clone(),
            projector_path: None,
            runtime: gpu_rt,
            config: LoadConfig::default(),
            alias: None,
        })
        .await;
    println!("invalid GGUF -> {:?} / hint {:?}", r.as_ref().err().map(|e| e.to_string()), engine.status().error_hint);
    assert!(r.is_err());
    std::fs::remove_file(bad).ok();
    assert_eq!(llama_server_count(), before);
    println!("SMOKE OK");
    Ok(())
}

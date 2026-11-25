use std::path::PathBuf;
use std::time::{Duration, Instant};

use app_core::AppError;
use inference::{EngineStatus, GpuLayers, LoadConfig, LoadRequest};
use tauri::State;

use super::Cmd;
use crate::state::SharedState;

/// Pick the runtime: explicit id, else the saved default, else the first installed.
async fn resolve_runtime(state: &SharedState, id: Option<String>) -> Cmd<inference::InstalledRuntime> {
    let installed = state.runtimes.list_installed();
    if installed.is_empty() {
        return Err(AppError::NotFound("no inference runtime installed yet; install one on the Runtimes page".into()));
    }
    let want = match id {
        Some(i) => Some(i),
        None => state.db.get_setting::<String>("runtime.default").await?,
    };
    if let Some(w) = want {
        return installed
            .into_iter()
            .find(|r| r.id == w)
            .ok_or_else(|| AppError::NotFound(format!("runtime {w} is not installed")));
    }
    Ok(installed.into_iter().next().unwrap())
}

#[tauri::command]
pub async fn load_model(
    state: State<'_, SharedState>,
    model_id: String,
    config: LoadConfig,
    runtime_id: Option<String>,
) -> Cmd<EngineStatus> {
    do_load(state.inner(), model_id, config, runtime_id).await
}

/// The one load path shared by the UI and the local API.
pub async fn do_load(st: &SharedState, model_id: String, mut config: LoadConfig, runtime_id: Option<String>) -> Cmd<EngineStatus> {
    let st = st.clone();
    // Linked-PC endpoints are resolved per load (a reload may pass the old
    // effective config); only the user's "Use linked PCs" choice is kept.
    config.rpc.clear();
    config.tensor_split = None;
    // Validate before anything is recorded: a rejected config must never
    // become the model's "last used" settings.
    config.validate().map_err(AppError::InvalidInput)?;
    let m = st.db.get_model(&model_id).await?;
    if m.status != "ok" {
        return Err(AppError::InvalidInput(format!(
            "this model can't be loaded: {}",
            m.status_detail.unwrap_or_else(|| m.status.clone())
        )));
    }
    let runtime = resolve_runtime(&st, runtime_id).await?;
    let meta: model_registry::ModelMetadata = serde_json::from_str(&m.metadata).unwrap_or_default();
    if config.speculative.as_deref().is_some_and(|s| s.split(',').any(|t| t == "draft-mtp")) && meta.mtp_layers.unwrap_or(0) == 0 {
        let why = if meta.mtp_unknown {
            "declares MTP layers, but its tensor list could not be read to confirm they are there"
        } else if meta.mtp_layers_declared.unwrap_or(0) > 0 {
            "declares MTP but its MTP layers were removed"
        } else {
            "has no multi-token-prediction (MTP) layers"
        };
        return Err(AppError::InvalidInput(format!("{} {why}, so it can't use speculative decoding with draft-mtp", m.display_name)));
    }
    let projector = super::models::projector_bytes(&config, m.projector_path.as_deref());
    // "Share with CPU", an Auto context and Auto KV are measured from free
    // memory right now. A model that is already loaded would make VRAM and
    // RAM look full, so unload it first (a load replaces it anyway).
    if (config.n_ctx == 0 || config.share_with_cpu || config.cache_type_auto || config.use_linked_pcs)
        && let Some(pid) = st.engine.backend_pid()
    {
        do_unload(&st).await?;
        if runtime.uses_gpu() {
            wait_for_vram_release(&st, pid).await;
        }
    }
    // "Use linked PCs": tunnel endpoints for --rpc and a split by free memory
    // (checks the linked PCs' llama.cpp build first).
    let (run, link_note, linked) = crate::services::linking::resolve_linked(&st, &runtime, config.clone()).await?;
    let linked = linked.as_ref();
    let r = match launch(&st, &m, &meta, &runtime, &run, &config, projector, link_note.into_iter().collect(), linked).await {
        Ok(s) => Ok(s),
        // A linked PC that refused memory over its owner's limit is not an MTP
        // problem: retrying without MTP would fail the same way and save MTP off.
        Err((e, launched))
            if super::models::retry_without_spec(&config, &launched, &runtime.capabilities, &e)
                && (linked.is_none() || st.linking.refusal_notes().await.is_empty()) =>
        {
            // Speculative MTP was the app's default, not the user's choice:
            // try once more without it.
            tracing::warn!("load failed with speculative decoding ({e}); retrying without it");
            let note = format!("Speculative decoding (draft-mtp) was turned off: the model did not start with it ({e}). You can turn it on again in the load settings.");
            let retry = super::models::retry_config(&run, &launched);
            let saved = super::models::without_spec(&config);
            let r = launch(&st, &m, &meta, &runtime, &retry, &saved, projector, vec![note], linked).await.map_err(|(e, _)| e);
            if r.is_err() {
                // MTP wasn't the cause: keep the user's settings as they were.
                if let Ok(v) = serde_json::to_value(&config) {
                    let _ = st.db.set_model_configs(&m.id, None, Some(&v)).await;
                }
            }
            r
        }
        Err((e, _)) => Err(e),
    };
    // A linked PC refused memory over the limit its owner set: say so.
    match r {
        Err(e) if linked.is_some() => {
            let notes = st.linking.refusal_notes().await;
            if notes.is_empty() {
                return Err(e);
            }
            let more = notes.join(" ");
            Err(match e {
                AppError::Runtime(m) => AppError::Runtime(format!("{m}. {more}")),
                AppError::OutOfMemory(m) => AppError::OutOfMemory(format!("{m}. {more}")),
                other => other,
            })
        }
        r => r,
    }
}

/// Measure a split load with [`crate::services::linking::preflight`] and,
/// while a device would go over its free memory (a linked PC's: its
/// owner's limit), move layers by the measured cost ([`linking::split::refit`]),
/// at most 3 rounds. Returns a note when it changed the split.
async fn refit_split(runtime: &inference::InstalledRuntime, model: &str, cfg: &mut LoadConfig, total_layers: u64) -> Option<String> {
    use linking::split;
    let before = (cfg.gpu_layers, cfg.tensor_split.clone());
    let max_layers = total_layers as u32;
    for _ in 0..3 {
        let rows = crate::services::linking::preflight(runtime, std::path::Path::new(model), cfg).await?;
        // Linked PCs' devices come first, one per --tensor-split entry.
        let weights: Vec<u64> = cfg.tensor_split.as_deref()?.split(',').map(|x| x.trim().parse().unwrap_or(0)).collect();
        if rows.len() != weights.len() {
            tracing::warn!("split check: {} devices measured, {} in --tensor-split; keeping the split", rows.len(), weights.len());
            return None;
        }
        let linked: Vec<bool> = rows.iter().map(|d| d.name.starts_with("RPC")).collect();
        // Speculative MTP runs a draft context on this PC's GPU that
        // llama-fit-params doesn't count: about one more compute buffer and
        // one layer's KV cache (real 2-PC test: 0.5-0.7 GB spilled to shared
        // memory without it).
        let ngl_now = match cfg.gpu_layers {
            GpuLayers::Count(n) => n,
            _ => max_layers,
        };
        let mtp_mib = if cfg.speculative.as_deref().is_some_and(|s| s.split(',').any(|t| t == "draft-mtp")) {
            let sum: u64 = weights.iter().sum::<u64>().max(1);
            rows.iter().zip(&linked).zip(&weights).filter(|((_, l), _)| !**l).map(|((d, _), w)| {
                let layers = (ngl_now as u64 * w / sum).max(1);
                d.compute_mib + d.context_mib / layers
            }).sum()
        } else {
            0
        };
        if split::breakdown_fits(&rows, &linked, mtp_mib) {
            break;
        }
        let ngl = match cfg.gpu_layers {
            GpuLayers::Count(n) => n,
            _ => max_layers,
        };
        let (n, layers) = split::refit(&rows, &linked, &weights, ngl, max_layers, mtp_mib)?;
        tracing::info!("split check: {rows:?} does not fit; -ngl {ngl} -> {n}, --tensor-split {}", layers.iter().map(u64::to_string).collect::<Vec<_>>().join(","));
        cfg.gpu_layers = if n >= max_layers { GpuLayers::All } else { GpuLayers::Count(n) };
        cfg.tensor_split = Some(layers.iter().map(u64::to_string).collect::<Vec<_>>().join(","));
    }
    if (cfg.gpu_layers, cfg.tensor_split.clone()) == before {
        return None;
    }
    let layers = match cfg.gpu_layers {
        GpuLayers::Count(n) => n.to_string(),
        _ => "all".into(),
    };
    Some(format!(
        "Split adjusted to llama.cpp's measured memory use: {layers} GPU layers, --tensor-split {}.",
        cfg.tensor_split.as_deref().unwrap_or("?")
    ))
}

/// Plan the load of `config` ([`super::models::plan_load`]), record it, start
/// the backend and add the plan's notes to the status. `save` is stored as
/// the model's "last used" settings (the user's choices, not the resolved
/// values, so the next load re-measures). On failure, also returns the
/// config the backend was started with.
#[allow(clippy::too_many_arguments)]
async fn launch(
    st: &SharedState,
    m: &persistence::ModelRow,
    meta: &model_registry::ModelMetadata,
    runtime: &inference::InstalledRuntime,
    config: &LoadConfig,
    save: &LoadConfig,
    projector: u64,
    mut notes: Vec<String>,
    linked: Option<&crate::services::linking::LinkedLoad>,
) -> Result<EngineStatus, (AppError, LoadConfig)> {
    let free = super::models::free_vram(st);
    let hw = st.monitor.hardware_info();
    let caps = &runtime.capabilities;
    let topo = system_monitor::core_topology();
    let linked_devs = linked.map(|l| l.linked_devs()).unwrap_or_default();
    let mut plan = {
        let observed = st.observed_offload.lock().unwrap().clone();
        let lookup = |c: &LoadConfig| observed.get(&(m.id.clone(), super::models::offload_key(c))).copied();
        let env = super::models::PlanEnv {
            free_vram: free,
            available_ram: hw.available_ram_bytes,
            total_ram: hw.total_ram_bytes,
            caps,
            uses_gpu: runtime.uses_gpu(),
            topo,
            observed: &lookup,
            linked: &linked_devs,
        };
        super::models::plan_load(config, meta, m.size as u64, projector, &env)
    };
    // Linked PCs: --tensor-split by net capacity at the final context.
    if let Some(l) = linked
        && !plan.config.rpc.is_empty()
    {
        let total = meta.block_count.unwrap_or(0) + 1;
        let cb = model_registry::estimate::estimate(meta, m.size as u64, &super::models::estimate_input(&plan.config, meta, total, projector)).compute_buffer_bytes;
        // This PC's GPU also holds the output layer, which the layer split doesn't count.
        let local_extra = projector.saturating_add(model_registry::estimate::output_layer_bytes(meta));
        let (split, note) = crate::services::linking::net_split(l, cb, local_extra);
        if let Some((rpc, ts)) = split {
            plan.config.rpc = rpc;
            plan.config.tensor_split = ts;
        }
        notes.extend(note);
        // Check the split against llama.cpp's own breakdown before any weight
        // is sent: the compute buffers are larger than the estimate on some
        // models (Qwen3.8-27B: about 1 GB per device, estimated at 0.5), and
        // a linked PC refuses memory over its owner's limit.
        if let Some(n) = refit_split(runtime, &m.path, &mut plan.config, total).await {
            // The earlier split by estimate no longer holds.
            notes.retain(|x| !x.starts_with("Layers split by"));
            notes.push(n);
        }
    }
    if let Some(a) = &plan.auto_ctx {
        tracing::info!("Auto context: {} tokens ({})", a.n_ctx, a.summary);
    }
    if let Some(a) = &plan.auto_kv {
        tracing::info!("{}", a.summary);
    }
    if let Some(s) = &plan.split {
        tracing::info!("Share with CPU (estimate): {}", s.summary);
    }
    for n in &plan.notes {
        tracing::info!("{n}");
    }
    let effective = plan.config.clone();
    let fail = |e: AppError| (e, effective.clone());
    effective.validate().map_err(|e| fail(AppError::InvalidInput(e)))?;
    let cfg_json = serde_json::to_value(save).map_err(|e| fail(e.into()))?;
    let eff_json = serde_json::to_value(&effective).map_err(|e| fail(e.into()))?;
    let instance_id = st.db.insert_instance(&m.id, &runtime.id, &eff_json).await.map_err(fail)?;
    st.db.set_model_configs(&m.id, None, Some(&cfg_json)).await.map_err(fail)?;
    let req = LoadRequest {
        model_id: m.id.clone(),
        instance_id: instance_id.clone(),
        model_path: PathBuf::from(&m.path),
        projector_path: m.projector_path.clone().map(PathBuf::from),
        runtime: runtime.clone(),
        config: effective.clone(),
        alias: Some(m.display_name.clone()),
        model_bytes: (m.size as u64).saturating_add(projector),
        // Linked PCs receiving weights (none unless "Use linked PCs" resolved some).
        probes: if effective.rpc.is_empty() { Vec::new() } else { st.linking.load_probes().into_iter().filter(|p| effective.rpc.contains(&p.endpoint)).collect() },
    };
    match st.engine.load(req).await {
        Ok(s) => {
            // Remember what llama.cpp chose when it fitted the layers itself.
            if effective.gpu_layers == inference::GpuLayers::Auto
                && let Some(o) = s.gpu_layers_offloaded
            {
                st.observed_offload.lock().unwrap().insert((m.id.clone(), super::models::offload_key(&effective)), o);
            }
            notes.extend(plan.notes);
            notes.extend(super::models::after_load_note(&effective, plan.cpu_layers, s.gpu_layers_offloaded, topo, caps));
            st.engine.add_warnings(&notes);
            Ok(st.engine.status())
        }
        Err(e) => {
            let state = if matches!(e, AppError::Cancelled) { "unloaded" } else { "failed" };
            let _ = st.db.set_instance_state(&instance_id, state, Some(&e.to_string())).await;
            Err((e, effective))
        }
    }
}

/// How long to wait after an unload for the driver to hand back the old
/// backend's VRAM, and how often to look.
const VRAM_RELEASE_TIMEOUT: Duration = Duration::from_secs(5);
const VRAM_POLL: Duration = Duration::from_millis(200);
/// Used VRAM counts as settled once it drops less than this between two polls.
const VRAM_SETTLE_TOLERANCE: u64 = 64 << 20;

/// After unloading backend `pid`, wait (up to [`VRAM_RELEASE_TIMEOUT`]) until
/// the process is gone and used VRAM stops dropping, so "Share with CPU"
/// measures free VRAM without the old model in it.
async fn wait_for_vram_release(st: &SharedState, pid: u32) {
    let deadline = Instant::now() + VRAM_RELEASE_TIMEOUT;
    let mut prev = None;
    loop {
        let snap = st.monitor.snapshot(Some(pid));
        let exited = snap.backend_memory_bytes.is_none();
        let used = snap.gpus.first().map(|g| g.vram_used_bytes);
        if exited && vram_settled(prev, used) {
            return;
        }
        if Instant::now() >= deadline {
            tracing::warn!(pid, exited, "VRAM still being released after unload; measuring free VRAM anyway");
            return;
        }
        prev = used;
        tokio::time::sleep(VRAM_POLL).await;
    }
}

/// True once used VRAM has stopped dropping: no GPU reading at all, or the
/// latest reading is within the tolerance of the one before. The first
/// reading never counts (there is nothing to compare it with).
fn vram_settled(prev: Option<u64>, now: Option<u64>) -> bool {
    match (prev, now) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(p), Some(n)) => n.saturating_add(VRAM_SETTLE_TOLERANCE) >= p,
    }
}

/// Reload the loaded model with its vision projector (for image attachments):
/// its last-used load settings with `load_projector` on, same runtime.
#[tauri::command]
pub async fn reload_with_projector(state: State<'_, SharedState>) -> Cmd<EngineStatus> {
    let st = state.inner();
    let status = st.engine.status();
    let model_id = status.model_id.clone().ok_or_else(|| AppError::InvalidInput("no model is loaded".into()))?;
    let m = st.db.get_model(&model_id).await?;
    if m.projector_path.is_none() {
        return Err(AppError::InvalidInput(format!("{} has no vision projector (mmproj) file next to it", m.display_name)));
    }
    // "Last used", upgraded like the load panel's defaults.
    let mut config: LoadConfig = super::models::last_used_config(st, &m, status.runtime_id.as_deref()).await.or(status.config.clone()).unwrap_or_default();
    config.load_projector = true;
    do_load(st, model_id, config, status.runtime_id.clone()).await
}

#[tauri::command]
pub async fn unload_model(state: State<'_, SharedState>) -> Cmd<EngineStatus> {
    do_unload(state.inner()).await
}

pub async fn do_unload(st: &SharedState) -> Cmd<EngineStatus> {
    let st = st.clone();
    let inst = st.engine.status().instance_id;
    st.engine.unload().await;
    if let Some(i) = inst {
        st.db.set_instance_state(&i, "unloaded", None).await?;
    }
    Ok(st.engine.status())
}

#[tauri::command]
pub fn cancel_load(state: State<'_, SharedState>) -> Cmd<()> {
    state.engine.cancel_load();
    Ok(())
}

#[tauri::command]
pub fn engine_status(state: State<'_, SharedState>) -> EngineStatus {
    state.engine.status()
}

#[tauri::command]
pub fn engine_logs(state: State<'_, SharedState>, last: Option<usize>) -> Vec<String> {
    state.engine.logs(last.unwrap_or(500).min(4000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_settles_once_it_stops_dropping() {
        const MIB: u64 = 1 << 20;
        // No GPU metrics: nothing to wait for.
        assert!(vram_settled(None, None));
        // First reading: wait for a second one.
        assert!(!vram_settled(None, Some(9000 * MIB)));
        // Still dropping fast.
        assert!(!vram_settled(Some(9000 * MIB), Some(5000 * MIB)));
        // Flat, small jitter, or rising (another app): settled.
        assert!(vram_settled(Some(5000 * MIB), Some(5000 * MIB)));
        assert!(vram_settled(Some(5000 * MIB), Some(4990 * MIB)));
        assert!(vram_settled(Some(5000 * MIB), Some(5100 * MIB)));
    }
}

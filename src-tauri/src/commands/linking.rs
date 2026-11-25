//! Linked PCs page: helper switch, one-time code, linking, peers, outside access.

use tauri::State;

use super::Cmd;
use crate::services::linking::{CodeView, HelperSettings, LinkStatus, PeerView, SplitPreview};
use crate::state::SharedState;

#[tauri::command]
pub async fn linking_status(state: State<'_, SharedState>) -> Cmd<LinkStatus> {
    state.linking.status().await
}

#[tauri::command]
pub async fn linking_set_name(state: State<'_, SharedState>, name: String) -> Cmd<LinkStatus> {
    state.linking.set_name(&name).await?;
    state.linking.status().await
}

/// "Let this PC help": on starts ggml-rpc-server (127.0.0.1 only) and the
/// listener; off stops both. Settings are saved; the switch itself is off at
/// every app start.
#[tauri::command]
pub async fn linking_set_helper(state: State<'_, SharedState>, enabled: bool, settings: HelperSettings) -> Cmd<LinkStatus> {
    let st = state.inner().clone();
    st.linking.set_helper(&st, enabled, settings).await?;
    st.linking.status().await
}

#[tauri::command]
pub async fn linking_start_code(state: State<'_, SharedState>) -> Cmd<CodeView> {
    state.linking.start_code().await
}

#[tauri::command]
pub async fn linking_cancel_code(state: State<'_, SharedState>) -> Cmd<()> {
    state.linking.cancel_code().await;
    Ok(())
}

/// Link to the PC at `address` (+ `port`) that shows `code`.
#[tauri::command]
pub async fn linking_pair(state: State<'_, SharedState>, address: String, port: Option<u16>, code: String) -> Cmd<PeerView> {
    state.linking.pair(&address, port, &code).await
}

/// Connect to a linked PC and measure round-trip time and bandwidth.
#[tauri::command]
pub async fn linking_connect(state: State<'_, SharedState>, peer_id: String) -> Cmd<PeerView> {
    state.linking.connect(&peer_id).await
}

/// "Split model layers with this PC".
#[tauri::command]
pub async fn linking_set_split(state: State<'_, SharedState>, peer_id: String, on: bool) -> Cmd<PeerView> {
    state.linking.set_split(&peer_id, on).await
}

/// "Send whole jobs" (Option B): stored only in this phase.
#[tauri::command]
pub async fn linking_set_send_jobs(state: State<'_, SharedState>, peer_id: String, on: bool) -> Cmd<PeerView> {
    state.linking.set_send_jobs(&peer_id, on).await
}

/// What this PC lends to a linked PC (None fields = the defaults):
/// GPU memory and RAM in MiB, CPU threads, pause. Enforced on this PC.
#[tauri::command]
pub async fn linking_set_peer_share(state: State<'_, SharedState>, peer_id: String, share: persistence::PeerShare) -> Cmd<PeerView> {
    state.linking.set_peer_share(&peer_id, share).await
}

/// Forget a linked PC; returns whether the other PC was told too.
#[tauri::command]
pub async fn linking_unlink(state: State<'_, SharedState>, peer_id: String) -> Cmd<bool> {
    state.linking.unlink(&peer_id).await
}

/// "Clear cache": delete every weight file this PC cached for linked PCs.
/// Returns the bytes freed.
#[tauri::command]
pub async fn linking_clear_cache(state: State<'_, SharedState>) -> Cmd<u64> {
    Ok(state.linking.clear_cache())
}

/// Outside access: with `try_upnp`, ask the router to open the port.
#[tauri::command]
pub async fn linking_check_reach(state: State<'_, SharedState>, try_upnp: bool) -> Cmd<linking::reach::ReachReport> {
    state.linking.check_reach(try_upnp).await
}

/// Load panel: split estimate per linked PC and measured speeds for a model.
#[tauri::command]
pub async fn linking_split_preview(state: State<'_, SharedState>, model_id: Option<String>) -> Cmd<SplitPreview> {
    let st = state.inner();
    let gpu = st.monitor.hardware_info().gpus.first().cloned();
    // What fit on this PC's GPU when it loaded the model alone (llama.cpp's
    // fit; the most layers if it was loaded several ways).
    let alone_gpu_layers = model_id.as_deref().and_then(|m| {
        st.observed_offload.lock().unwrap().iter().filter(|((id, _), (_, t))| id == m && *t > 0).map(|(_, (g, t))| f64::from(*g) / f64::from(*t)).reduce(f64::max)
    });
    let local = crate::services::linking::LocalHw {
        gpu_name: gpu.as_ref().map(|g| g.name.clone()),
        free_vram: super::models::free_vram(st),
        total_vram: gpu.as_ref().map(|g| g.vram_total_bytes),
        alone_gpu_layers,
    };
    st.linking.split_preview(model_id.as_deref(), &local).await
}

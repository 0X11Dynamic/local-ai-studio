//! Discover: Hugging Face search, real variant lists and downloads.

use std::path::Path;

use app_core::AppError;
use model_registry::downloads::{DownloadEvent, repo_dir, safe_relative};
use model_registry::hf::{self, RepoDetails, SearchResult, Variant};
use serde::Serialize;
use tauri::State;

use super::Cmd;
use crate::state::SharedState;

#[tauri::command]
pub async fn hf_search(state: State<'_, SharedState>, query: String, sort: Option<String>) -> Cmd<Vec<SearchResult>> {
    let hf = state.hf_client();
    // A pasted repo id or URL goes straight to that repo.
    if let Some(repo) = hf::parse_repo_input(&query).filter(|_| query.contains('/')) {
        let d = hf.details(&repo).await?;
        return Ok(vec![SearchResult {
            id: d.id.clone(),
            author: d.author.clone(),
            downloads: d.downloads,
            likes: d.likes,
            created_at: d.last_modified.clone(),
            pipeline_tag: d.pipeline_tag.clone(),
            tags: d.tags.clone(),
            license: d.license.clone(),
            base_model: d.base_model.clone(),
            capabilities: d.capabilities.clone(),
        }]);
    }
    hf.search(&query, sort.as_deref().unwrap_or("downloads"), 40).await
}

#[derive(Debug, Serialize)]
pub struct VariantView {
    #[serde(flatten)]
    pub variant: Variant,
    /// installed | partial | downloading | paused | failed | loaded | available | incomplete
    pub status: String,
    pub model_id: Option<String>,
    pub download_id: Option<String>,
    /// Estimate: weights + about 1 GB for context and buffers.
    pub est_memory_bytes: u64,
    /// gpu | partial | cpu | too_big (estimate)
    pub fit: String,
}

#[derive(Debug, Serialize)]
pub struct RepoView {
    pub details: RepoDetails,
    pub variants: Vec<VariantView>,
    pub recommended: Option<String>,
    pub free_vram_bytes: Option<u64>,
    pub available_ram_bytes: u64,
    pub models_dir: String,
}

#[tauri::command]
pub async fn hf_repo(state: State<'_, SharedState>, repo: String) -> Cmd<RepoView> {
    let st = state.inner().clone();
    let hf = st.hf_client();
    let id = hf::parse_repo_input(&repo).ok_or_else(|| AppError::InvalidInput(format!("'{repo}' is not a repo id")))?;
    let mut details = hf.details(&id).await?;
    let files = hf.files(&details.id, &details.sha).await?;
    let variants = hf::group_variants(&files);
    hf::apply_repo_files(&mut details, &variants);
    let hw = st.monitor.hardware_info();
    let free_vram = hw.gpus.first().map(|g| g.vram_total_bytes.saturating_sub(g.vram_used_bytes));
    let recommended = hf::recommend(&variants, free_vram, hw.available_ram_bytes);
    let models_dir = st.models_dir().await;
    let dest = repo_dir(&models_dir, &details.id)?;
    let library = st.db.list_models().await?;
    let engine = st.engine.status();
    let downloads = st.downloads.list().await?;
    let mut out = Vec::new();
    for v in variants {
        let first = dest.join(safe_relative(&v.files[0].path)?);
        let model = library.iter().find(|m| Path::new(&m.path) == first);
        let dl = downloads
            .iter()
            .find(|d| d.repo == details.id && d.variant_key == v.key && !matches!(d.state.as_str(), "completed" | "cancelled"));
        let on_disk = v.files.iter().all(|f| {
            dest.join(safe_relative(&f.path).unwrap_or_default()).metadata().map(|m| m.len() == f.size).unwrap_or(false)
        });
        let has_part = v.files.iter().any(|f| downloader::part_path(&dest.join(safe_relative(&f.path).unwrap_or_default())).exists());
        let status = if !v.complete {
            "incomplete"
        } else if model.is_some_and(|m| engine.model_id.as_deref() == Some(m.id.as_str()) && engine.pid.is_some()) {
            "loaded"
        } else if let Some(d) = dl {
            match d.state.as_str() {
                "running" | "queued" => "downloading",
                "paused" => "paused",
                _ => "failed",
            }
        } else if on_disk && (model.is_some() || v.is_projector) {
            "installed"
        } else if has_part {
            "partial"
        } else {
            "available"
        };
        let fit = hf::estimate_fit(v.total_size, free_vram, hw.available_ram_bytes);
        out.push(VariantView {
            status: status.into(),
            model_id: model.map(|m| m.id.clone()),
            download_id: dl.map(|d| d.id.clone()),
            est_memory_bytes: fit.est_memory_bytes,
            fit: fit.fit,
            variant: v,
        });
    }
    Ok(RepoView {
        details,
        variants: out,
        recommended,
        free_vram_bytes: free_vram,
        available_ram_bytes: hw.available_ram_bytes,
        models_dir: models_dir.to_string_lossy().into_owned(),
    })
}

/// Start downloading a variant (optionally with a vision projector add-on).
#[tauri::command]
pub async fn download_variant(
    state: State<'_, SharedState>,
    repo: String,
    variant_key: String,
    projector_key: Option<String>,
) -> Cmd<String> {
    let st = state.inner().clone();
    let hf = st.hf_client();
    let details = hf.details(&repo).await?;
    let files = hf.files(&details.id, &details.sha).await?;
    let variants = hf::group_variants(&files);
    let v = variants
        .iter()
        .find(|v| v.key == variant_key)
        .ok_or_else(|| AppError::NotFound(format!("{variant_key} is not in {repo} anymore")))?;
    if !v.complete {
        return Err(AppError::InvalidInput("this shard set is incomplete in the repository and can't be loaded".into()));
    }
    let mut chosen = v.files.clone();
    if let Some(pk) = projector_key {
        let p = variants
            .iter()
            .find(|x| x.key == pk && x.is_projector)
            .ok_or_else(|| AppError::NotFound("vision projector not found".into()))?;
        chosen.extend(p.files.clone());
    }
    let models_dir = st.models_dir().await;
    st.downloads
        .start(&hf, &details.id, &details.sha, &v.key, v.quantization.clone(), &chosen, &models_dir)
        .await
}

#[tauri::command]
pub async fn list_downloads(state: State<'_, SharedState>) -> Cmd<Vec<DownloadEvent>> {
    state.downloads.list().await
}

#[tauri::command]
pub async fn pause_download(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    state.downloads.pause(&id).await
}

#[tauri::command]
pub async fn resume_download(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    let dir = state.models_dir().await;
    state.downloads.resume(&id, &dir).await
}

#[tauri::command]
pub async fn cancel_download(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    state.downloads.cancel(&id).await
}

#[tauri::command]
pub async fn clear_finished_downloads(state: State<'_, SharedState>) -> Cmd<u64> {
    state.db.delete_finished_downloads().await
}

/// Store or clear the Hugging Face token (Credential Manager). Never returned.
#[tauri::command]
pub async fn set_hf_token(state: State<'_, SharedState>, token: Option<String>) -> Cmd<bool> {
    let st = state.inner().clone();
    match token.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        Some(t) => {
            if !t.starts_with("hf_") || t.len() < 20 {
                return Err(AppError::InvalidInput("Hugging Face tokens start with hf_".into()));
            }
            st.credentials.set(crate::state::CRED_HF_TOKEN, &t)?;
            st.downloads.set_hf_token(Some(t));
            Ok(true)
        }
        None => {
            st.credentials.delete(crate::state::CRED_HF_TOKEN)?;
            st.downloads.set_hf_token(None);
            Ok(false)
        }
    }
}

#[tauri::command]
pub fn has_hf_token(state: State<'_, SharedState>) -> Cmd<bool> {
    Ok(state.credentials.get(crate::state::CRED_HF_TOKEN)?.is_some())
}

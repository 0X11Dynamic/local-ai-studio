//! Model library: GGUF parsing, folder scans, memory estimates and registration
//! into SQLite, plus Hugging Face discovery (`hf`) and model downloads (`downloads`).

pub mod caps;
pub mod downloads;
pub mod estimate;
pub mod gguf;
pub mod hf;
pub mod scan;

use std::path::Path;

use app_core::{AppError, AppResult};
use persistence::Db;
use persistence::models::NewModel;
use serde::Serialize;

pub use gguf::ModelMetadata;
pub use scan::{DiscoveredModel, ScanResult};

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub registered: Vec<String>,
    pub invalid: Vec<String>,
    pub skipped: Vec<scan::SkippedFile>,
}

/// Scan `root` and register every model found. Files are used in place; nothing
/// is copied or moved. `source` is stored for display ("import", "lmstudio", ...).
pub async fn import_path(db: &Db, root: &Path, source: &str, repo_root: Option<&Path>) -> AppResult<ImportReport> {
    if !root.exists() {
        return Err(AppError::NotFound(format!("{} does not exist", root.display())));
    }
    let root_owned = root.to_path_buf();
    let result = tokio::task::spawn_blocking(move || scan::scan(&root_owned))
        .await
        .map_err(|e| AppError::Other(format!("model scan failed: {e}")))?;
    let mut report = ImportReport { registered: vec![], invalid: vec![], skipped: result.skipped };
    for m in result.models {
        let repo_id = repo_root.and_then(|rr| repo_id_from_path(rr, &m.path));
        let status = if m.problems.is_empty() { "ok" } else { "invalid" };
        let new = NewModel {
            repo_id,
            path: m.path.to_string_lossy().into_owned(),
            quantization: m.quantization.clone(),
            metadata: serde_json::to_value(&m.metadata)?,
            size: m.total_size as i64,
            display_name: m.display_name.clone(),
            source: source.to_string(),
            shard_paths: m.shard_paths.iter().map(|p| p.to_string_lossy().into_owned()).collect(),
            projector_path: m.projector_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            status: status.into(),
            status_detail: (!m.problems.is_empty()).then(|| m.problems.join("; ")),
        };
        let id = db.upsert_model(&new).await?;
        if status == "ok" {
            report.registered.push(id);
        } else {
            report.invalid.push(id);
        }
    }
    Ok(report)
}

/// Re-check that every registered model's files still exist; mark missing ones.
pub async fn refresh_status(db: &Db) -> AppResult<()> {
    for m in db.list_models().await? {
        let shards: Vec<String> = serde_json::from_str(&m.shard_paths).unwrap_or_default();
        let missing: Vec<&String> = shards.iter().filter(|p| !Path::new(p).exists()).collect();
        if !missing.is_empty() {
            let detail = format!("file(s) not found: {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
            db.set_model_status(&m.id, "missing", Some(&detail)).await?;
        } else if m.status == "missing" {
            db.set_model_status(&m.id, "ok", None).await?;
        }
        if missing.is_empty() {
            backfill_caps(db, &m).await;
        }
    }
    Ok(())
}

/// `ModelMetadata` fields read since CAPS_VERSION 2 (mtp_layers re-read at 3,
/// and at 4 from every shard of a split model; host_weight_bytes since 5),
/// copied into older rows.
const ATTENTION_LAYOUT_KEYS: &[&str] = &[
    "host_weight_bytes",
    "sliding_window",
    "sliding_window_pattern",
    "swa_layers",
    "key_length_swa",
    "value_length_swa",
    "shared_kv_layers",
    "mtp_layers",
    "mtp_layers_declared",
    "mtp_unknown",
    "full_attention_interval",
];

/// Detect capabilities for rows stored before detection existed (or by an older
/// detector). Reads the header and projector read-only; failures leave the row as is.
async fn backfill_caps(db: &Db, m: &persistence::models::ModelRow) {
    let Ok(mut meta) = serde_json::from_str::<serde_json::Value>(&m.metadata) else { return };
    let stored = meta["caps"]["version"].as_u64().unwrap_or(0);
    if stored >= caps::CAPS_VERSION as u64 {
        return;
    }
    let path = std::path::PathBuf::from(&m.path);
    let shards: Vec<std::path::PathBuf> = serde_json::from_str::<Vec<String>>(&m.shard_paths).unwrap_or_default().into_iter().map(std::path::PathBuf::from).collect();
    let projector = m.projector_path.clone().map(std::path::PathBuf::from);
    let detected = tokio::task::spawn_blocking(move || {
        let mut h = gguf::read_header(&path).ok()?;
        gguf::add_shard_tensors(&mut h, &path, &shards);
        let tpl = h.kv.get("tokenizer.chat_template").and_then(|v| v.as_str());
        let mut c = caps::from_template(tpl, "gguf");
        caps::apply_projector(&mut c, projector.as_deref().map(caps::read_projector));
        Some((c, gguf::extract(&h)))
    })
    .await
    .ok()
    .flatten();
    let Some((c, fresh)) = detected else { return };
    if let (Some(obj), Ok(v), Ok(serde_json::Value::Object(f))) = (meta.as_object_mut(), serde_json::to_value(&c), serde_json::to_value(&fresh)) {
        obj.insert("caps".into(), v);
        // Header fields added after the row was stored (attention layout).
        for k in ATTENTION_LAYOUT_KEYS {
            if let Some(x) = f.get(*k) {
                obj.insert((*k).into(), x.clone());
            }
        }
        if let Err(e) = db.update_model_metadata(&m.id, &meta).await {
            tracing::warn!("could not store capabilities for {}: {e}", m.id);
        }
    }
}

/// `<root>/<publisher>/<repo>/file.gguf` -> `publisher/repo`
fn repo_id_from_path(root: &Path, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(root).ok()?;
    let parts: Vec<_> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    (parts.len() >= 3).then(|| format!("{}/{}", parts[0], parts[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::write_fake_gguf;

    #[tokio::test]
    async fn import_registers_once_and_refresh_marks_missing() {
        let d = tempfile::tempdir().unwrap();
        let repo = d.path().join("lmstudio-community").join("Fake-GGUF");
        std::fs::create_dir_all(&repo).unwrap();
        let f = repo.join("Fake-Q4_K_M.gguf");
        write_fake_gguf(&f, "llama", 4096, 15, None);
        let db = Db::open_in_memory().await.unwrap();
        let r1 = import_path(&db, d.path(), "lmstudio", Some(d.path())).await.unwrap();
        let r2 = import_path(&db, d.path(), "lmstudio", Some(d.path())).await.unwrap();
        assert_eq!(r1.registered, r2.registered);
        let models = db.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].repo_id.as_deref(), Some("lmstudio-community/Fake-GGUF"));
        // Rows from before capability detection get it on refresh.
        let mut old: serde_json::Value = serde_json::from_str(&models[0].metadata).unwrap();
        assert_eq!(old["caps"]["version"], caps::CAPS_VERSION);
        old.as_object_mut().unwrap().remove("caps");
        // ...and the attention layout fields are re-read from the header.
        old.as_object_mut().unwrap().remove("sliding_window");
        old["mtp_layers"] = serde_json::json!(7);
        db.update_model_metadata(&models[0].id, &old).await.unwrap();
        refresh_status(&db).await.unwrap();
        let back: serde_json::Value = serde_json::from_str(&db.get_model(&models[0].id).await.unwrap().metadata).unwrap();
        assert_eq!(back["caps"]["version"], caps::CAPS_VERSION);
        assert_eq!(back["caps"]["vision"], false);
        assert!(back["mtp_layers"].is_null() && back.get("sliding_window").is_some_and(|v| v.is_null()), "{back}");

        std::fs::remove_file(&f).unwrap();
        refresh_status(&db).await.unwrap();
        assert_eq!(db.get_model(&models[0].id).await.unwrap().status, "missing");
    }
}

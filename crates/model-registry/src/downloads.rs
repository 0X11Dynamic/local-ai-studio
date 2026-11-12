//! Model download manager: a persisted queue of variant downloads.
//!
//! Each download is one variant (a GGUF file or a full shard set, optionally
//! with a vision projector). Files stream into `<models>/<owner>/<repo>/` as
//! `.part` files, resume with HTTP Range, are checked against the Hub's LFS
//! SHA-256, and only then appear under their real names. A finished download is
//! registered in the library automatically. State survives restarts: anything
//! running when the app closed comes back as paused.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use downloader::{DownloadError, Progress};
use persistence::{Db, DownloadRow};
use serde::{Deserialize, Serialize};
use tokio::sync::{Semaphore, broadcast};
use tokio_util::sync::CancellationToken;

use crate::hf::{HfClient, RepoFile};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSource {
    pub repo: String,
    pub revision: String,
    pub variant_key: String,
    pub quantization: Option<String>,
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadEvent {
    pub id: String,
    pub repo: String,
    pub variant_key: String,
    pub state: String,
    pub downloaded: u64,
    pub total: u64,
    pub bytes_per_sec: f64,
    pub eta_secs: Option<u64>,
    pub current_file: Option<String>,
    pub files_done: usize,
    pub files_total: usize,
    pub error: Option<String>,
    /// Set when the finished download was registered in the library.
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Stop {
    Pause,
    Cancel,
}

struct Job {
    token: CancellationToken,
    stop: Arc<Mutex<Option<Stop>>>,
}

#[derive(Clone)]
pub struct DownloadManager {
    db: Db,
    http: reqwest::Client,
    hf_token: Arc<Mutex<Option<String>>>,
    jobs: Arc<Mutex<HashMap<String, Job>>>,
    slots: Arc<Semaphore>,
    events: broadcast::Sender<DownloadEvent>,
    live: Arc<Mutex<HashMap<String, DownloadEvent>>>,
}

/// Reject absolute paths, drive prefixes and `..` in Hub-provided file paths,
/// and rebuild the path with native separators (Hub paths use '/'), so the
/// library's path key is identical however the file is reached.
pub fn safe_relative(path: &str) -> AppResult<PathBuf> {
    let bad = || AppError::InvalidInput(format!("unsafe file path from repository: {path}"));
    if path.is_empty() || path.contains(':') || path.starts_with('/') || path.starts_with('\\') {
        return Err(bad());
    }
    let mut out = PathBuf::new();
    for seg in path.split(['/', '\\']).filter(|s| !s.is_empty()) {
        let mut comps = Path::new(seg).components();
        match (comps.next(), comps.next()) {
            (Some(Component::Normal(c)), None) => out.push(c),
            _ => return Err(bad()),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(bad());
    }
    Ok(out)
}

/// `<models>/<owner>/<repo>` with each part validated.
pub fn repo_dir(models_dir: &Path, repo: &str) -> AppResult<PathBuf> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| AppError::InvalidInput(format!("bad repo id {repo}")))?;
    let rel = safe_relative(&format!("{owner}/{name}"))?;
    Ok(models_dir.join(rel))
}

impl DownloadManager {
    pub fn new(db: Db, http: reqwest::Client, concurrency: usize) -> Self {
        let (events, _) = broadcast::channel(512);
        Self {
            db,
            http,
            hf_token: Arc::new(Mutex::new(None)),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            slots: Arc::new(Semaphore::new(concurrency.clamp(1, 8))),
            events,
            live: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn set_hf_token(&self, token: Option<String>) {
        *self.hf_token.lock().unwrap() = token;
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DownloadEvent> {
        self.events.subscribe()
    }

    /// Latest in-memory progress for running downloads (DB has the rest).
    pub fn live(&self) -> HashMap<String, DownloadEvent> {
        self.live.lock().unwrap().clone()
    }

    fn emit(&self, ev: DownloadEvent) {
        if matches!(ev.state.as_str(), "running" | "queued") {
            self.live.lock().unwrap().insert(ev.id.clone(), ev.clone());
        } else {
            self.live.lock().unwrap().remove(&ev.id);
        }
        let _ = self.events.send(ev);
    }

    /// Queue a variant download. Returns the download id.
    pub async fn start(
        &self,
        hf: &HfClient,
        repo: &str,
        revision: &str,
        variant_key: &str,
        quantization: Option<String>,
        files: &[RepoFile],
        models_dir: &Path,
    ) -> AppResult<String> {
        if files.is_empty() {
            return Err(AppError::InvalidInput("nothing to download".into()));
        }
        if let Some(active) = self.db.active_download_for(repo, variant_key).await? {
            return Err(AppError::Conflict(format!(
                "this variant already has a download ({}); resume or cancel it instead",
                active.state
            )));
        }
        let dest = repo_dir(models_dir, repo)?;
        let mut sfiles = Vec::new();
        let mut remaining = 0u64;
        for f in files {
            let rel = safe_relative(&f.path)?;
            let target = dest.join(&rel);
            let have_full = std::fs::metadata(&target).map(|m| m.len() == f.size).unwrap_or(false);
            if !have_full {
                let part = downloader::part_path(&target);
                let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                remaining += f.size.saturating_sub(have);
            }
            sfiles.push(SourceFile {
                path: f.path.clone(),
                size: f.size,
                sha256: f.sha256.clone(),
                url: hf.file_url(repo, revision, &f.path),
            });
        }
        if remaining == 0 {
            // Everything is already on disk: just (re)register it.
            let first = dest.join(safe_relative(&files[0].path)?);
            crate::import_path(&self.db, &first, "download", Some(models_dir)).await?;
            return Err(AppError::Conflict("already downloaded; it is in My Models".into()));
        }
        std::fs::create_dir_all(&dest)?;
        if let Some(free) = downloader::free_space(&dest)
            && remaining + (512 << 20) > free
        {
            return Err(AppError::InvalidInput(format!(
                "not enough disk space: needs {:.1} GB, {:.1} GB free on that drive",
                remaining as f64 / 1e9,
                free as f64 / 1e9
            )));
        }
        let source = DownloadSource {
            repo: repo.into(),
            revision: revision.into(),
            variant_key: variant_key.into(),
            quantization,
            files: sfiles,
        };
        let total: u64 = files.iter().map(|f| f.size).sum();
        let id = self
            .db
            .insert_download(&serde_json::to_value(&source)?, &dest.to_string_lossy(), repo, variant_key, total)
            .await?;
        self.spawn(id.clone(), models_dir.to_path_buf());
        Ok(id)
    }

    fn spawn(&self, id: String, models_dir: PathBuf) {
        let token = CancellationToken::new();
        let stop = Arc::new(Mutex::new(None));
        self.jobs.lock().unwrap().insert(id.clone(), Job { token: token.clone(), stop: stop.clone() });
        let me = self.clone();
        tokio::spawn(async move {
            let result = me.run(&id, &models_dir, &token).await;
            me.jobs.lock().unwrap().remove(&id);
            let intent = *stop.lock().unwrap();
            let (state, err) = match (&result, intent) {
                (Ok(_), _) => ("completed", None),
                (Err(AppError::Cancelled), Some(Stop::Cancel)) => ("cancelled", None),
                (Err(AppError::Cancelled), _) => ("paused", None),
                (Err(e), _) => ("failed", Some(e.to_string())),
            };
            if state == "cancelled" {
                me.delete_parts(&id).await;
            }
            let _ = me.db.update_download(&id, state, None, err.as_deref()).await;
            let row = me.db.get_download(&id).await.ok();
            let mut ev = me.event_from_row(row.as_ref(), &id, state);
            ev.error = err;
            ev.model_id = result.ok().flatten();
            me.emit(ev);
        });
    }

    fn event_from_row(&self, row: Option<&DownloadRow>, id: &str, state: &str) -> DownloadEvent {
        let src: Option<DownloadSource> = row.and_then(|r| serde_json::from_str(&r.source).ok());
        let prog: serde_json::Value = row.and_then(|r| serde_json::from_str(&r.progress).ok()).unwrap_or_default();
        DownloadEvent {
            id: id.into(),
            repo: src.as_ref().map(|s| s.repo.clone()).unwrap_or_default(),
            variant_key: src.as_ref().map(|s| s.variant_key.clone()).unwrap_or_default(),
            state: state.into(),
            downloaded: prog["downloaded"].as_u64().unwrap_or(0),
            total: prog["total"].as_u64().unwrap_or(0),
            bytes_per_sec: 0.0,
            eta_secs: None,
            current_file: None,
            files_done: prog["files_done"].as_u64().unwrap_or(0) as usize,
            files_total: src.as_ref().map(|s| s.files.len()).unwrap_or(0),
            error: row.and_then(|r| r.error.clone()),
            model_id: None,
        }
    }

    /// Download every file of the variant, then register it. Returns the model id.
    async fn run(&self, id: &str, models_dir: &Path, token: &CancellationToken) -> AppResult<Option<String>> {
        let row = self.db.get_download(id).await?;
        let src: DownloadSource = serde_json::from_str(&row.source)?;
        let dest = PathBuf::from(&row.destination);
        let total: u64 = src.files.iter().map(|f| f.size).sum();
        let base = DownloadEvent {
            id: id.into(),
            repo: src.repo.clone(),
            variant_key: src.variant_key.clone(),
            state: "queued".into(),
            downloaded: 0,
            total,
            bytes_per_sec: 0.0,
            eta_secs: None,
            current_file: None,
            files_done: 0,
            files_total: src.files.len(),
            error: None,
            model_id: None,
        };
        self.db.update_download(id, "queued", None, None).await?;
        self.emit(base.clone());
        let _permit = tokio::select! {
            p = self.slots.clone().acquire_owned() => p.map_err(|_| AppError::Other("download queue closed".into()))?,
            _ = token.cancelled() => return Err(AppError::Cancelled),
        };
        self.db.update_download(id, "running", None, None).await?;
        let token_value = self.hf_token.lock().unwrap().clone();
        let mut done_bytes = 0u64;
        let mut files_done = 0usize;
        for f in &src.files {
            let target = dest.join(safe_relative(&f.path)?);
            if std::fs::metadata(&target).map(|m| m.len() == f.size).unwrap_or(false) {
                done_bytes += f.size;
                files_done += 1;
                continue;
            }
            let name = f.path.clone();
            let mut last_db = Instant::now();
            let db = self.db.clone();
            let me = self.clone();
            let base_ev = base.clone();
            let id_s = id.to_string();
            let progress_cb = move |p: Progress| {
                let downloaded = done_bytes + p.downloaded;
                let eta = (p.bytes_per_sec > 1.0).then(|| ((total.saturating_sub(downloaded)) as f64 / p.bytes_per_sec) as u64);
                me.emit(DownloadEvent {
                    state: "running".into(),
                    downloaded,
                    bytes_per_sec: p.bytes_per_sec,
                    eta_secs: eta,
                    current_file: Some(name.clone()),
                    files_done,
                    ..base_ev.clone()
                });
                if last_db.elapsed() > Duration::from_secs(2) {
                    last_db = Instant::now();
                    let db = db.clone();
                    let id = id_s.clone();
                    let prog = serde_json::json!({ "downloaded": downloaded, "total": total, "files_done": files_done });
                    tokio::spawn(async move {
                        let _ = db.update_download(&id, "running", Some(&prog), None).await;
                    });
                }
            };
            let r = downloader::download_file_auth(
                &self.http,
                &f.url,
                token_value.as_deref(),
                &target,
                f.sha256.as_deref(),
                token,
                progress_cb,
            )
            .await;
            match r {
                Ok(()) => {}
                Err(DownloadError::Cancelled) => {
                    let prog = serde_json::json!({ "downloaded": done_bytes + downloader::part_path(&target).metadata().map(|m| m.len()).unwrap_or(0), "total": total, "files_done": files_done });
                    let _ = self.db.update_download(id, "running", Some(&prog), None).await;
                    return Err(AppError::Cancelled);
                }
                Err(DownloadError::DiskSpace { needed, available }) => {
                    return Err(AppError::InvalidInput(DownloadError::DiskSpace { needed, available }.to_string()));
                }
                Err(e) => return Err(AppError::Network(format!("{}: {e}", f.path))),
            }
            done_bytes += f.size;
            files_done += 1;
            let prog = serde_json::json!({ "downloaded": done_bytes, "total": total, "files_done": files_done });
            self.db.update_download(id, "running", Some(&prog), None).await?;
        }
        // Register the model (first non-projector file) in the library.
        let first_model = src
            .files
            .iter()
            .find(|f| !f.path.to_ascii_lowercase().contains("mmproj"))
            .unwrap_or(&src.files[0]);
        let first_path = dest.join(safe_relative(&first_model.path)?);
        let folder = first_path.parent().unwrap_or(&dest).to_path_buf();
        let report = crate::import_path(&self.db, &folder, "download", Some(models_dir)).await?;
        let model_id = self
            .db
            .list_models()
            .await?
            .into_iter()
            .find(|m| Path::new(&m.path) == first_path)
            .map(|m| m.id)
            .or_else(|| report.registered.first().cloned());
        let prog = serde_json::json!({ "downloaded": total, "total": total, "files_done": src.files.len() });
        self.db.update_download(id, "running", Some(&prog), None).await?;
        Ok(model_id)
    }

    async fn delete_parts(&self, id: &str) {
        let Ok(row) = self.db.get_download(id).await else { return };
        let Ok(src) = serde_json::from_str::<DownloadSource>(&row.source) else { return };
        let dest = PathBuf::from(&row.destination);
        for f in &src.files {
            if let Ok(rel) = safe_relative(&f.path) {
                let _ = tokio::fs::remove_file(downloader::part_path(&dest.join(rel))).await;
            }
        }
    }

    fn stop(&self, id: &str, how: Stop) -> bool {
        if let Some(j) = self.jobs.lock().unwrap().get(id) {
            *j.stop.lock().unwrap() = Some(how);
            j.token.cancel();
            return true;
        }
        false
    }

    pub async fn pause(&self, id: &str) -> AppResult<()> {
        if !self.stop(id, Stop::Pause) {
            return Err(AppError::Conflict("download is not running".into()));
        }
        Ok(())
    }

    /// Cancel and delete partial files. Works for running and paused downloads.
    pub async fn cancel(&self, id: &str) -> AppResult<()> {
        if self.stop(id, Stop::Cancel) {
            return Ok(());
        }
        let row = self.db.get_download(id).await?;
        if matches!(row.state.as_str(), "paused" | "failed" | "queued") {
            self.delete_parts(id).await;
            self.db.update_download(id, "cancelled", None, None).await?;
            let ev = self.event_from_row(Some(&row), id, "cancelled");
            self.emit(ev);
        }
        Ok(())
    }

    /// Resume a paused download or retry a failed one, from its partial files.
    pub async fn resume(&self, id: &str, models_dir: &Path) -> AppResult<()> {
        if self.jobs.lock().unwrap().contains_key(id) {
            return Err(AppError::Conflict("download is already running".into()));
        }
        let row = self.db.get_download(id).await?;
        if !matches!(row.state.as_str(), "paused" | "failed") {
            return Err(AppError::Conflict(format!("can't resume a {} download", row.state)));
        }
        self.spawn(id.to_string(), models_dir.to_path_buf());
        Ok(())
    }

    pub async fn list(&self) -> AppResult<Vec<DownloadEvent>> {
        let live = self.live();
        let rows = self.db.list_downloads().await?;
        Ok(rows
            .iter()
            .map(|r| live.get(&r.id).cloned().unwrap_or_else(|| self.event_from_row(Some(r), &r.id, &r.state)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_repo_paths() {
        assert!(safe_relative("Q4/model.gguf").is_ok());
        for bad in ["../x.gguf", "a/../../x.gguf", "C:/x.gguf", "/etc/x", "\\\\server\\x", ""] {
            assert!(safe_relative(bad).is_err(), "{bad} should be rejected");
        }
        assert!(repo_dir(Path::new("C:/m"), "owner/../x").is_err());
        // Native separators only (the library keys models by path string).
        let d = repo_dir(Path::new(r"C:\m"), "unsloth/Qwen3-4B-GGUF").unwrap();
        assert_eq!(d.to_string_lossy(), r"C:\m\unsloth\Qwen3-4B-GGUF");
        assert_eq!(safe_relative("BF16/x-00001-of-00002.gguf").unwrap().to_string_lossy(), r"BF16\x-00001-of-00002.gguf");
    }
}

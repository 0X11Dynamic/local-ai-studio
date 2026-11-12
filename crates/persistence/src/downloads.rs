use app_core::{AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DownloadRow {
    pub id: String,
    /// JSON: { repo, revision, files: [{path, size, sha256, url}] }
    pub source: String,
    /// Folder the files are written into.
    pub destination: String,
    /// JSON: { downloaded, total, files_done, current_file }
    pub progress: String,
    /// queued | running | paused | completed | failed | cancelled
    pub state: String,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub repo_id: Option<String>,
    pub variant_key: Option<String>,
}

impl Db {
    pub async fn insert_download(
        &self,
        source: &serde_json::Value,
        destination: &str,
        repo_id: &str,
        variant_key: &str,
        total: u64,
    ) -> AppResult<String> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query(
            "INSERT INTO downloads(id, source, destination, progress, state, created_at, updated_at, repo_id, variant_key) \
             VALUES(?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(source.to_string())
        .bind(destination)
        .bind(serde_json::json!({ "downloaded": 0, "total": total, "files_done": 0 }).to_string())
        .bind(&now)
        .bind(&now)
        .bind(repo_id)
        .bind(variant_key)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(id)
    }

    pub async fn update_download(
        &self,
        id: &str,
        state: &str,
        progress: Option<&serde_json::Value>,
        error: Option<&str>,
    ) -> AppResult<()> {
        sqlx::query(
            "UPDATE downloads SET state = ?, progress = COALESCE(?, progress), error = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state)
        .bind(progress.map(|p| p.to_string()))
        .bind(error)
        .bind(now_rfc3339())
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn get_download(&self, id: &str) -> AppResult<DownloadRow> {
        sqlx::query_as::<_, DownloadRow>("SELECT * FROM downloads WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn list_downloads(&self) -> AppResult<Vec<DownloadRow>> {
        sqlx::query_as::<_, DownloadRow>("SELECT * FROM downloads ORDER BY created_at DESC")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    /// Active (not finished) download for this variant, if any.
    pub async fn active_download_for(&self, repo_id: &str, variant_key: &str) -> AppResult<Option<DownloadRow>> {
        sqlx::query_as::<_, DownloadRow>(
            "SELECT * FROM downloads WHERE repo_id = ? AND variant_key = ? AND state IN ('queued','running','paused','failed') \
             ORDER BY created_at DESC LIMIT 1",
        )
        .bind(repo_id)
        .bind(variant_key)
        .fetch_optional(self.pool())
        .await
        .map_err(db_err)
    }

    /// At startup: anything that was running when the app stopped is paused
    /// (its .part files remain and it can be resumed).
    pub async fn pause_interrupted_downloads(&self) -> AppResult<u64> {
        let r = sqlx::query(
            "UPDATE downloads SET state = 'paused', error = 'interrupted when the app closed; resume to continue', updated_at = ? \
             WHERE state IN ('queued', 'running')",
        )
        .bind(now_rfc3339())
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(r.rows_affected())
    }

    pub async fn delete_finished_downloads(&self) -> AppResult<u64> {
        let r = sqlx::query("DELETE FROM downloads WHERE state IN ('completed', 'cancelled')")
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(r.rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn download_lifecycle_and_interrupt_recovery() {
        let db = Db::open_in_memory().await.unwrap();
        let id = db
            .insert_download(&serde_json::json!({"repo": "o/r"}), "C:/m/o/r", "o/r", "m-Q4_K_M.gguf", 100)
            .await
            .unwrap();
        assert!(db.active_download_for("o/r", "m-Q4_K_M.gguf").await.unwrap().is_some());
        db.update_download(&id, "running", Some(&serde_json::json!({"downloaded": 40, "total": 100})), None).await.unwrap();
        assert_eq!(db.pause_interrupted_downloads().await.unwrap(), 1);
        let d = db.get_download(&id).await.unwrap();
        assert_eq!(d.state, "paused");
        assert!(d.progress.contains("40"));
        db.update_download(&id, "completed", None, None).await.unwrap();
        assert!(db.active_download_for("o/r", "m-Q4_K_M.gguf").await.unwrap().is_none());
        assert_eq!(db.delete_finished_downloads().await.unwrap(), 1);
    }
}

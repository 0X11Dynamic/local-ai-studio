use app_core::{AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

/// A registered model file (or shard set). Separate from any running instance.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ModelRow {
    pub id: String,
    pub repo_id: Option<String>,
    pub path: String,
    pub format: String,
    pub quantization: Option<String>,
    /// JSON text of the parsed GGUF metadata subset.
    pub metadata: String,
    pub size: i64,
    pub display_name: String,
    pub source: String,
    /// JSON array of every shard path (one entry for single-file models).
    pub shard_paths: String,
    pub projector_path: Option<String>,
    pub default_config: Option<String>,
    pub last_config: Option<String>,
    pub status: String,
    pub status_detail: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Fields needed to register a model. `path` is the unique key: registering the
/// same path again updates the existing row instead of creating a duplicate.
#[derive(Debug, Clone)]
pub struct NewModel {
    pub repo_id: Option<String>,
    pub path: String,
    pub quantization: Option<String>,
    pub metadata: serde_json::Value,
    pub size: i64,
    pub display_name: String,
    pub source: String,
    pub shard_paths: Vec<String>,
    pub projector_path: Option<String>,
    pub status: String,
    pub status_detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ModelInstanceRow {
    pub id: String,
    pub model_id: String,
    pub backend: String,
    pub config: String,
    pub state: String,
    pub error: Option<String>,
    pub started_at: String,
    pub loaded_at: Option<String>,
    pub ended_at: Option<String>,
}

impl Db {
    /// Insert or update a model by path. Returns the row id.
    pub async fn upsert_model(&self, m: &NewModel) -> AppResult<String> {
        let now = now_rfc3339();
        let id = new_id();
        let row: (String,) = sqlx::query_as(
            "INSERT INTO models(id, repo_id, path, format, quantization, metadata, size, display_name, \
                                source, shard_paths, projector_path, status, status_detail, created_at, updated_at) \
             VALUES(?, ?, ?, 'gguf', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(path) DO UPDATE SET \
                repo_id = COALESCE(excluded.repo_id, models.repo_id), \
                quantization = excluded.quantization, metadata = excluded.metadata, size = excluded.size, \
                display_name = excluded.display_name, shard_paths = excluded.shard_paths, \
                projector_path = excluded.projector_path, status = excluded.status, \
                status_detail = excluded.status_detail, updated_at = excluded.updated_at \
             RETURNING id",
        )
        .bind(&id)
        .bind(&m.repo_id)
        .bind(&m.path)
        .bind(&m.quantization)
        .bind(m.metadata.to_string())
        .bind(m.size)
        .bind(&m.display_name)
        .bind(&m.source)
        .bind(serde_json::to_string(&m.shard_paths)?)
        .bind(&m.projector_path)
        .bind(&m.status)
        .bind(&m.status_detail)
        .bind(&now)
        .bind(&now)
        .fetch_one(self.pool())
        .await
        .map_err(db_err)?;
        Ok(row.0)
    }

    pub async fn list_models(&self) -> AppResult<Vec<ModelRow>> {
        sqlx::query_as::<_, ModelRow>("SELECT * FROM models ORDER BY display_name COLLATE NOCASE")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn get_model(&self, id: &str) -> AppResult<ModelRow> {
        sqlx::query_as::<_, ModelRow>("SELECT * FROM models WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn delete_model(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM models WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_model_status(&self, id: &str, status: &str, detail: Option<&str>) -> AppResult<()> {
        sqlx::query("UPDATE models SET status = ?, status_detail = ?, updated_at = ? WHERE id = ?")
            .bind(status)
            .bind(detail)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    /// Replace the stored metadata JSON (used to back-fill newly detected fields
    /// such as capabilities for models registered by an older version).
    pub async fn update_model_metadata(&self, id: &str, metadata: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE models SET metadata = ?, updated_at = ? WHERE id = ?")
            .bind(metadata.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_model_configs(
        &self,
        id: &str,
        default_config: Option<&serde_json::Value>,
        last_config: Option<&serde_json::Value>,
    ) -> AppResult<()> {
        if let Some(d) = default_config {
            sqlx::query("UPDATE models SET default_config = ?, updated_at = ? WHERE id = ?")
                .bind(d.to_string())
                .bind(now_rfc3339())
                .bind(id)
                .execute(self.pool())
                .await
                .map_err(db_err)?;
        }
        if let Some(l) = last_config {
            sqlx::query("UPDATE models SET last_config = ?, updated_at = ? WHERE id = ?")
                .bind(l.to_string())
                .bind(now_rfc3339())
                .bind(id)
                .execute(self.pool())
                .await
                .map_err(db_err)?;
        }
        Ok(())
    }

    pub async fn clear_model_default_config(&self, id: &str) -> AppResult<()> {
        sqlx::query("UPDATE models SET default_config = NULL, updated_at = ? WHERE id = ?")
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    // ---- model instances ----

    pub async fn insert_instance(&self, model_id: &str, backend: &str, config: &serde_json::Value) -> AppResult<String> {
        let id = new_id();
        sqlx::query(
            "INSERT INTO model_instances(id, model_id, backend, config, state, started_at) VALUES(?, ?, ?, ?, 'loading', ?)",
        )
        .bind(&id)
        .bind(model_id)
        .bind(backend)
        .bind(config.to_string())
        .bind(now_rfc3339())
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(id)
    }

    /// Move an instance to a new state, stamping loaded_at / ended_at as appropriate.
    pub async fn set_instance_state(&self, id: &str, state: &str, error: Option<&str>) -> AppResult<()> {
        let now = now_rfc3339();
        let (loaded_at, ended_at) = match state {
            "loaded" => (Some(now.clone()), None),
            "failed" | "unloaded" => (None, Some(now.clone())),
            _ => (None, None),
        };
        sqlx::query(
            "UPDATE model_instances SET state = ?, error = COALESCE(?, error), \
             loaded_at = COALESCE(?, loaded_at), ended_at = COALESCE(?, ended_at) WHERE id = ?",
        )
        .bind(state)
        .bind(error)
        .bind(loaded_at)
        .bind(ended_at)
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    /// Instances left in loading/loaded state by a previous crash are marked
    /// unloaded at startup: their processes died with the old app (Job Object).
    pub async fn close_stale_instances(&self) -> AppResult<u64> {
        let r = sqlx::query(
            "UPDATE model_instances SET state = 'unloaded', error = COALESCE(error, 'app exited while loaded'), \
             ended_at = ? WHERE state IN ('loading', 'loaded')",
        )
        .bind(now_rfc3339())
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(r.rows_affected())
    }

    pub async fn recent_instances(&self, limit: i64) -> AppResult<Vec<ModelInstanceRow>> {
        sqlx::query_as::<_, ModelInstanceRow>("SELECT * FROM model_instances ORDER BY started_at DESC LIMIT ?")
            .bind(limit)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(path: &str) -> NewModel {
        NewModel {
            repo_id: None,
            path: path.into(),
            quantization: Some("Q4_K_M".into()),
            metadata: serde_json::json!({"architecture": "llama"}),
            size: 100,
            display_name: "Sample".into(),
            source: "import".into(),
            shard_paths: vec![path.into()],
            projector_path: None,
            status: "ok".into(),
            status_detail: None,
        }
    }

    #[tokio::test]
    async fn upsert_by_path_does_not_duplicate() {
        let db = Db::open_in_memory().await.unwrap();
        let a = db.upsert_model(&sample("C:/m/a.gguf")).await.unwrap();
        let mut again = sample("C:/m/a.gguf");
        again.size = 200;
        let b = db.upsert_model(&again).await.unwrap();
        assert_eq!(a, b);
        let all = db.list_models().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].size, 200);
    }

    #[tokio::test]
    async fn update_metadata_replaces_json() {
        let db = Db::open_in_memory().await.unwrap();
        let id = db.upsert_model(&sample("C:/m/c.gguf")).await.unwrap();
        let meta = serde_json::json!({"architecture": "llama", "caps": {"version": 1, "tools": true}});
        db.update_model_metadata(&id, &meta).await.unwrap();
        let row = db.get_model(&id).await.unwrap();
        let back: serde_json::Value = serde_json::from_str(&row.metadata).unwrap();
        assert_eq!(back, meta);
    }

    #[tokio::test]
    async fn instance_lifecycle_and_stale_cleanup() {
        let db = Db::open_in_memory().await.unwrap();
        let m = db.upsert_model(&sample("C:/m/b.gguf")).await.unwrap();
        let i = db.insert_instance(&m, "cpu", &serde_json::json!({})).await.unwrap();
        db.set_instance_state(&i, "loaded", None).await.unwrap();
        assert_eq!(db.close_stale_instances().await.unwrap(), 1);
        let rows = db.recent_instances(5).await.unwrap();
        assert_eq!(rows[0].state, "unloaded");
        assert!(rows[0].loaded_at.is_some() && rows[0].ended_at.is_some());
    }
}

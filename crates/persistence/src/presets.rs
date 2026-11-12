//! System Instruction profiles (presets with kind = 'instruction').

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PresetRow {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub settings_json: String,
    pub system_prompt: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub builtin: i64,
    pub sections: String,
}

impl Db {
    pub async fn list_presets(&self, kind: &str) -> AppResult<Vec<PresetRow>> {
        sqlx::query_as::<_, PresetRow>("SELECT * FROM presets WHERE kind = ? ORDER BY builtin DESC, title COLLATE NOCASE")
            .bind(kind)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn get_preset(&self, id: &str) -> AppResult<PresetRow> {
        sqlx::query_as::<_, PresetRow>("SELECT * FROM presets WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn insert_preset(&self, kind: &str, title: &str, sections: &serde_json::Value, settings: &serde_json::Value, builtin: bool) -> AppResult<String> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query(
            "INSERT INTO presets(id, title, kind, settings_json, sections, builtin, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(title)
        .bind(kind)
        .bind(settings.to_string())
        .bind(sections.to_string())
        .bind(builtin as i64)
        .bind(&now)
        .bind(&now)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(id)
    }

    /// Built-in templates can't be edited; clone them instead.
    pub async fn update_preset(&self, id: &str, title: &str, sections: &serde_json::Value, settings: &serde_json::Value) -> AppResult<()> {
        let p = self.get_preset(id).await?;
        if p.builtin != 0 {
            return Err(AppError::PermissionDenied("built-in templates are read-only; clone it to make changes".into()));
        }
        sqlx::query("UPDATE presets SET title = ?, sections = ?, settings_json = ?, updated_at = ? WHERE id = ?")
            .bind(title)
            .bind(sections.to_string())
            .bind(settings.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    /// App-only upgrade path for built-in templates (the UI path above refuses them).
    pub async fn update_builtin_preset(&self, id: &str, sections: &serde_json::Value, settings: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE presets SET sections = ?, settings_json = ?, updated_at = ? WHERE id = ? AND builtin = 1")
            .bind(sections.to_string())
            .bind(settings.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn delete_preset(&self, id: &str) -> AppResult<()> {
        let p = self.get_preset(id).await?;
        if p.builtin != 0 {
            return Err(AppError::PermissionDenied("built-in templates can't be deleted".into()));
        }
        sqlx::query("DELETE FROM presets WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_model_instruction_profile(&self, model_id: &str, profile_id: Option<&str>) -> AppResult<()> {
        sqlx::query("UPDATE models SET instruction_profile_id = ?, updated_at = ? WHERE id = ?")
            .bind(profile_id)
            .bind(now_rfc3339())
            .bind(model_id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn model_instruction_profile(&self, model_id: &str) -> AppResult<Option<String>> {
        let r: Option<(Option<String>,)> = sqlx::query_as("SELECT instruction_profile_id FROM models WHERE id = ?")
            .bind(model_id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?;
        Ok(r.and_then(|x| x.0))
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn builtin_presets_are_read_only() {
        let db = Db::open_in_memory().await.unwrap();
        let b = db.insert_preset("instruction", "General", &serde_json::json!({"mode":"raw","raw":"x"}), &serde_json::json!({}), true).await.unwrap();
        assert!(db.update_preset(&b, "Hacked", &serde_json::json!({}), &serde_json::json!({})).await.is_err());
        assert!(db.delete_preset(&b).await.is_err());
        let u = db.insert_preset("instruction", "Mine", &serde_json::json!({"mode":"raw","raw":"y"}), &serde_json::json!({}), false).await.unwrap();
        db.update_preset(&u, "Mine 2", &serde_json::json!({"mode":"raw","raw":"z"}), &serde_json::json!({})).await.unwrap();
        assert_eq!(db.get_preset(&u).await.unwrap().title, "Mine 2");
        assert_eq!(db.list_presets("instruction").await.unwrap()[0].title, "General", "built-ins first");
        db.delete_preset(&u).await.unwrap();
    }
}

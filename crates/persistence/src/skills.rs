//! Skills and their revisions (applied, pending agent proposals, rejected).

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SkillRow {
    pub id: String,
    pub path: String,
    pub name: String,
    pub version: String,
    /// JSON: description, keywords, other front-matter keys.
    pub manifest: String,
    pub enabled: i64,
    pub created_at: String,
    pub updated_at: String,
    /// auto | always | off
    pub mode: String,
    /// user | import | builtin
    pub source: String,
    /// JSON {"relative/path": "enabled sha256"}.
    pub scripts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SkillRevisionRow {
    pub id: String,
    pub skill_id: String,
    pub version: String,
    pub content_hash: String,
    pub content: String,
    pub note: Option<String>,
    pub created_at: String,
    /// applied | pending | rejected
    pub state: String,
    pub proposed_by: Option<String>,
}

impl Db {
    pub async fn insert_skill(&self, path: &str, name: &str, version: &str, manifest: &serde_json::Value, source: &str) -> AppResult<String> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query("INSERT INTO skills(id, path, name, version, manifest, enabled, created_at, updated_at, mode, source) VALUES(?, ?, ?, ?, ?, 1, ?, ?, 'auto', ?)")
            .bind(&id)
            .bind(path)
            .bind(name)
            .bind(version)
            .bind(manifest.to_string())
            .bind(&now)
            .bind(&now)
            .bind(source)
            .execute(self.pool())
            .await
            .map_err(|e| if e.to_string().contains("UNIQUE") { AppError::Conflict(format!("a skill named {name} already exists")) } else { db_err(e) })?;
        Ok(id)
    }

    pub async fn update_skill_meta(&self, id: &str, version: &str, manifest: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE skills SET version = ?, manifest = ?, updated_at = ? WHERE id = ?")
            .bind(version)
            .bind(manifest.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_skill_mode(&self, id: &str, mode: &str) -> AppResult<()> {
        if !matches!(mode, "auto" | "always" | "off") {
            return Err(AppError::InvalidInput("mode must be auto, always or off".into()));
        }
        sqlx::query("UPDATE skills SET mode = ?, enabled = ?, updated_at = ? WHERE id = ?")
            .bind(mode)
            .bind((mode != "off") as i64)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_skill_scripts(&self, id: &str, scripts: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE skills SET scripts = ?, updated_at = ? WHERE id = ?").bind(scripts.to_string()).bind(now_rfc3339()).bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    pub async fn list_skills(&self) -> AppResult<Vec<SkillRow>> {
        sqlx::query_as::<_, SkillRow>("SELECT * FROM skills ORDER BY name").fetch_all(self.pool()).await.map_err(db_err)
    }

    pub async fn get_skill(&self, id: &str) -> AppResult<SkillRow> {
        sqlx::query_as::<_, SkillRow>("SELECT * FROM skills WHERE id = ? OR name = ?")
            .bind(id)
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("skill {id} not found")))
    }

    pub async fn delete_skill(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM skills WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    pub async fn add_skill_revision(&self, skill_id: &str, version: &str, hash: &str, content: &str, note: Option<&str>, state: &str, proposed_by: Option<&str>) -> AppResult<String> {
        if !matches!(state, "applied" | "pending" | "rejected") {
            return Err(AppError::InvalidInput("bad revision state".into()));
        }
        let id = new_id();
        sqlx::query("INSERT INTO skill_revisions(id, skill_id, version, content_hash, content, note, created_at, state, proposed_by) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&id)
            .bind(skill_id)
            .bind(version)
            .bind(hash)
            .bind(content)
            .bind(note)
            .bind(now_rfc3339())
            .bind(state)
            .bind(proposed_by)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(id)
    }

    pub async fn list_skill_revisions(&self, skill_id: &str) -> AppResult<Vec<SkillRevisionRow>> {
        sqlx::query_as::<_, SkillRevisionRow>("SELECT * FROM skill_revisions WHERE skill_id = ? ORDER BY created_at DESC, rowid DESC")
            .bind(skill_id)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn get_skill_revision(&self, id: &str) -> AppResult<SkillRevisionRow> {
        sqlx::query_as::<_, SkillRevisionRow>("SELECT * FROM skill_revisions WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("revision {id} not found")))
    }

    pub async fn set_skill_revision_state(&self, id: &str, state: &str) -> AppResult<()> {
        sqlx::query("UPDATE skill_revisions SET state = ? WHERE id = ?").bind(state).bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn skills_modes_and_revisions() {
        let db = Db::open_in_memory().await.unwrap();
        let id = db.insert_skill("C:/s/rust", "rust", "1", &serde_json::json!({ "description": "d" }), "user").await.unwrap();
        assert!(db.insert_skill("C:/s/rust2", "rust", "1", &serde_json::json!({}), "user").await.is_err(), "unique name");
        db.set_skill_mode(&id, "always").await.unwrap();
        assert!(db.set_skill_mode(&id, "sometimes").await.is_err());
        db.set_skill_mode(&id, "off").await.unwrap();
        let s = db.get_skill("rust").await.unwrap();
        assert_eq!((s.mode.as_str(), s.enabled), ("off", 0));
        db.add_skill_revision(&id, "1", "h1", "v1", None, "applied", None).await.unwrap();
        let p = db.add_skill_revision(&id, "2", "h2", "v2", Some("agent idea"), "pending", Some("task:x")).await.unwrap();
        let revs = db.list_skill_revisions(&id).await.unwrap();
        assert_eq!(revs.len(), 2);
        assert_eq!(revs[0].state, "pending");
        db.set_skill_revision_state(&p, "rejected").await.unwrap();
        assert_eq!(db.get_skill_revision(&p).await.unwrap().state, "rejected");
        db.set_skill_scripts(&id, &serde_json::json!({ "run.ps1": "abc" })).await.unwrap();
        db.delete_skill(&id).await.unwrap();
        assert!(db.list_skill_revisions(&id).await.unwrap().is_empty());
    }
}

//! Execution environment profiles (SSH, Docker, sandbox; "This PC" is
//! built in) and per-environment preferences for the built-in tools.

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ExecProfileRow {
    pub id: String,
    /// local | ssh | docker | sandbox
    #[sqlx(rename = "type")]
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    /// JSON config without secret values.
    pub nonsecret_config: String,
    pub credential_ref: Option<String>,
    pub last_health: Option<String>,
    pub last_health_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct BuiltinPrefRow {
    pub env_id: String,
    pub tool_name: String,
    pub enabled: i64,
    pub policy: String,
}

impl Db {
    pub async fn insert_exec_profile(&self, kind: &str, name: &str, config: &serde_json::Value, credential_ref: Option<&str>) -> AppResult<String> {
        if !matches!(kind, "ssh" | "docker" | "sandbox") {
            return Err(AppError::InvalidInput("environment type must be ssh, docker or sandbox".into()));
        }
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query("INSERT INTO execution_profiles(id, type, name, nonsecret_config, credential_ref, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)")
            .bind(&id)
            .bind(kind)
            .bind(name.trim())
            .bind(config.to_string())
            .bind(credential_ref)
            .bind(&now)
            .bind(&now)
            .execute(self.pool())
            .await
            .map_err(|e| if e.to_string().contains("UNIQUE") { AppError::Conflict(format!("an environment named \"{}\" already exists", name.trim())) } else { db_err(e) })?;
        Ok(id)
    }

    pub async fn update_exec_profile(&self, id: &str, name: &str, config: &serde_json::Value, credential_ref: Option<&str>) -> AppResult<()> {
        let r = sqlx::query("UPDATE execution_profiles SET name = ?, nonsecret_config = ?, credential_ref = ?, updated_at = ? WHERE id = ?")
            .bind(name.trim())
            .bind(config.to_string())
            .bind(credential_ref)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(|e| if e.to_string().contains("UNIQUE") { AppError::Conflict(format!("an environment named \"{}\" already exists", name.trim())) } else { db_err(e) })?;
        if r.rows_affected() == 0 {
            return Err(AppError::NotFound(format!("environment {id} not found")));
        }
        Ok(())
    }

    pub async fn set_exec_health(&self, id: &str, health: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE execution_profiles SET last_health = ?, last_health_at = ? WHERE id = ?")
            .bind(health.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn list_exec_profiles(&self) -> AppResult<Vec<ExecProfileRow>> {
        sqlx::query_as::<_, ExecProfileRow>("SELECT * FROM execution_profiles ORDER BY created_at").fetch_all(self.pool()).await.map_err(db_err)
    }

    pub async fn get_exec_profile(&self, id: &str) -> AppResult<ExecProfileRow> {
        sqlx::query_as::<_, ExecProfileRow>("SELECT * FROM execution_profiles WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("environment {id} not found")))
    }

    pub async fn delete_exec_profile(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM builtin_tool_preferences WHERE env_id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        sqlx::query("DELETE FROM permission_grants WHERE subject LIKE ?").bind(format!("builtin:{id}/%")).execute(self.pool()).await.map_err(db_err)?;
        sqlx::query("DELETE FROM execution_profiles WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    pub async fn builtin_prefs(&self, env_id: &str) -> AppResult<Vec<BuiltinPrefRow>> {
        sqlx::query_as::<_, BuiltinPrefRow>("SELECT * FROM builtin_tool_preferences WHERE env_id = ?").bind(env_id).fetch_all(self.pool()).await.map_err(db_err)
    }

    pub async fn set_builtin_pref(&self, env_id: &str, tool_name: &str, enabled: bool, policy: &str) -> AppResult<()> {
        if !matches!(policy, "allow" | "ask" | "deny") {
            return Err(AppError::InvalidInput("policy must be allow, ask or deny".into()));
        }
        sqlx::query(
            "INSERT INTO builtin_tool_preferences(env_id, tool_name, enabled, policy) VALUES(?, ?, ?, ?) \
             ON CONFLICT(env_id, tool_name) DO UPDATE SET enabled = excluded.enabled, policy = excluded.policy",
        )
        .bind(env_id)
        .bind(tool_name)
        .bind(enabled as i64)
        .bind(policy)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn profiles_and_prefs_round_trip() {
        let db = Db::open_in_memory().await.unwrap();
        let id = db.insert_exec_profile("ssh", "box", &serde_json::json!({ "host": "h" }), Some("exec/x")).await.unwrap();
        assert!(db.insert_exec_profile("ssh", "box", &serde_json::json!({}), None).await.is_err(), "unique name");
        assert!(db.insert_exec_profile("local", "pc", &serde_json::json!({}), None).await.is_err(), "local is built in");
        let row = db.get_exec_profile(&id).await.unwrap();
        assert_eq!((row.kind.as_str(), row.credential_ref.as_deref()), ("ssh", Some("exec/x")));
        db.set_builtin_pref(&id, "shell_execute", true, "allow").await.unwrap();
        db.set_builtin_pref(&id, "shell_execute", false, "ask").await.unwrap();
        let p = db.builtin_prefs(&id).await.unwrap();
        assert_eq!((p.len(), p[0].enabled, p[0].policy.as_str()), (1, 0, "ask"));
        assert!(db.set_builtin_pref(&id, "x", true, "yolo").await.is_err());
        db.insert_grant("persistent", &format!("builtin:{id}/shell_execute"), None).await.unwrap();
        db.delete_exec_profile(&id).await.unwrap();
        assert!(db.get_exec_profile(&id).await.is_err());
        assert!(db.builtin_prefs(&id).await.unwrap().is_empty());
        assert!(db.list_grants().await.unwrap().is_empty(), "grants removed with the environment");
    }
}

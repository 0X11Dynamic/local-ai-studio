//! MCP server rows, per-tool preferences, permission grants and the tool audit log.

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct McpServerRow {
    pub id: String,
    pub name: String,
    pub transport: String,
    /// JSON `mcp_client::ServerConfig` (never contains secret values).
    pub nonsecret_config: String,
    /// Credential Manager reference for the server's secrets, if any.
    pub credential_ref: Option<String>,
    pub enabled: i64,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ToolPrefRow {
    pub id: String,
    pub server_id: String,
    pub tool_name: String,
    pub enabled: i64,
    pub policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct GrantRow {
    pub id: String,
    pub scope: String,
    pub subject: String,
    pub capability: String,
    pub environment_id: Option<String>,
    pub expiry: Option<String>,
    pub created_at: String,
    pub conversation_id: Option<String>,
    /// Set for "allow for this task" grants.
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ToolExecRow {
    pub id: String,
    pub run_id: Option<String>,
    pub agent_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tool: String,
    pub source: String,
    pub environment_id: Option<String>,
    pub args_redacted: String,
    pub result: Option<String>,
    pub status: String,
    pub approval: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub tool_call_id: Option<String>,
    pub model_tool_name: Option<String>,
}

pub struct NewToolExec<'a> {
    pub conversation_id: Option<&'a str>,
    pub tool: &'a str,
    pub source: &'a str,
    pub args_redacted: &'a str,
    pub tool_call_id: Option<&'a str>,
    pub model_tool_name: Option<&'a str>,
    pub environment_id: Option<&'a str>,
    pub run_id: Option<&'a str>,
    /// Subagent id (None for a task's root agent).
    pub agent_id: Option<&'a str>,
}

impl Db {
    // ---------- servers ----------
    pub async fn insert_mcp_server(&self, name: &str, transport: &str, config: &serde_json::Value, credential_ref: Option<&str>, enabled: bool) -> AppResult<String> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query(
            "INSERT INTO mcp_servers(id, name, transport, nonsecret_config, credential_ref, enabled, status, created_at, updated_at) \
             VALUES(?, ?, ?, ?, ?, ?, 'disconnected', ?, ?)",
        )
        .bind(&id)
        .bind(name)
        .bind(transport)
        .bind(config.to_string())
        .bind(credential_ref)
        .bind(enabled as i64)
        .bind(&now)
        .bind(&now)
        .execute(self.pool())
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(d) if d.message().contains("UNIQUE") => AppError::Conflict(format!("a server named '{name}' already exists")),
            other => db_err(other),
        })?;
        Ok(id)
    }

    pub async fn update_mcp_server(&self, id: &str, name: &str, transport: &str, config: &serde_json::Value, credential_ref: Option<&str>, enabled: bool) -> AppResult<()> {
        sqlx::query(
            "UPDATE mcp_servers SET name = ?, transport = ?, nonsecret_config = ?, credential_ref = ?, enabled = ?, updated_at = ? WHERE id = ?",
        )
        .bind(name)
        .bind(transport)
        .bind(config.to_string())
        .bind(credential_ref)
        .bind(enabled as i64)
        .bind(now_rfc3339())
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(d) if d.message().contains("UNIQUE") => AppError::Conflict(format!("a server named '{name}' already exists")),
            other => db_err(other),
        })?;
        Ok(())
    }

    pub async fn list_mcp_servers(&self) -> AppResult<Vec<McpServerRow>> {
        sqlx::query_as::<_, McpServerRow>("SELECT * FROM mcp_servers ORDER BY name COLLATE NOCASE")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn get_mcp_server(&self, id: &str) -> AppResult<McpServerRow> {
        sqlx::query_as::<_, McpServerRow>("SELECT * FROM mcp_servers WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn delete_mcp_server(&self, id: &str) -> AppResult<()> {
        // Grants for its tools go too (subjects start with mcp:<id>/).
        sqlx::query("DELETE FROM permission_grants WHERE subject LIKE ?")
            .bind(format!("mcp:{id}/%"))
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        sqlx::query("DELETE FROM mcp_servers WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    // ---------- tool preferences ----------
    pub async fn tool_prefs(&self, server_id: &str) -> AppResult<Vec<ToolPrefRow>> {
        sqlx::query_as::<_, ToolPrefRow>("SELECT * FROM mcp_tool_preferences WHERE server_id = ?")
            .bind(server_id)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn all_tool_prefs(&self) -> AppResult<Vec<ToolPrefRow>> {
        sqlx::query_as::<_, ToolPrefRow>("SELECT * FROM mcp_tool_preferences").fetch_all(self.pool()).await.map_err(db_err)
    }

    pub async fn set_tool_pref(&self, server_id: &str, tool_name: &str, enabled: bool, policy: &str) -> AppResult<()> {
        if !matches!(policy, "allow" | "ask" | "deny") {
            return Err(AppError::InvalidInput("policy must be allow, ask or deny".into()));
        }
        sqlx::query(
            "INSERT INTO mcp_tool_preferences(id, server_id, tool_name, enabled, policy) VALUES(?, ?, ?, ?, ?) \
             ON CONFLICT(server_id, tool_name) DO UPDATE SET enabled = excluded.enabled, policy = excluded.policy",
        )
        .bind(new_id())
        .bind(server_id)
        .bind(tool_name)
        .bind(enabled as i64)
        .bind(policy)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    // ---------- grants ----------
    pub async fn insert_grant(&self, scope: &str, tool_id: &str, conversation_id: Option<&str>) -> AppResult<String> {
        if !matches!(scope, "session" | "persistent") {
            return Err(AppError::InvalidInput("scope must be session or persistent".into()));
        }
        if scope == "session" && conversation_id.is_none() {
            return Err(AppError::InvalidInput("a session grant needs a conversation".into()));
        }
        let id = new_id();
        sqlx::query(
            "INSERT INTO permission_grants(id, scope, subject, capability, conversation_id, created_at) VALUES(?, ?, ?, 'tool.call', ?, ?)",
        )
        .bind(&id)
        .bind(scope)
        .bind(tool_id)
        .bind(conversation_id)
        .bind(now_rfc3339())
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(id)
    }

    pub async fn grants_for_tool(&self, tool_id: &str) -> AppResult<Vec<GrantRow>> {
        sqlx::query_as::<_, GrantRow>("SELECT * FROM permission_grants WHERE subject = ? AND capability = 'tool.call'")
            .bind(tool_id)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn list_grants(&self) -> AppResult<Vec<GrantRow>> {
        sqlx::query_as::<_, GrantRow>("SELECT * FROM permission_grants ORDER BY created_at DESC")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn delete_grant(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM permission_grants WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    // ---------- audit ----------
    pub async fn start_tool_exec(&self, e: &NewToolExec<'_>) -> AppResult<String> {
        let id = new_id();
        sqlx::query(
            "INSERT INTO tool_executions(id, conversation_id, tool, source, args_redacted, status, approval, started_at, tool_call_id, model_tool_name, environment_id, run_id, agent_id) \
             VALUES(?, ?, ?, ?, ?, 'pending', '{}', ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(e.conversation_id)
        .bind(e.tool)
        .bind(e.source)
        .bind(e.args_redacted)
        .bind(now_rfc3339())
        .bind(e.tool_call_id)
        .bind(e.model_tool_name)
        .bind(e.environment_id)
        .bind(e.run_id)
        .bind(e.agent_id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(id)
    }

    pub async fn finish_tool_exec(&self, id: &str, status: &str, result: Option<&str>, approval: &serde_json::Value, duration_ms: u64) -> AppResult<()> {
        sqlx::query("UPDATE tool_executions SET status = ?, result = ?, approval = ?, ended_at = ?, duration_ms = ? WHERE id = ?")
            .bind(status)
            .bind(result)
            .bind(approval.to_string())
            .bind(now_rfc3339())
            .bind(duration_ms as i64)
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_tool_exec_approval(&self, id: &str, status: &str, approval: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE tool_executions SET status = ?, approval = ? WHERE id = ?")
            .bind(status)
            .bind(approval.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn list_tool_execs(&self, conversation_id: Option<&str>, limit: i64) -> AppResult<Vec<ToolExecRow>> {
        match conversation_id {
            Some(c) => sqlx::query_as::<_, ToolExecRow>("SELECT * FROM tool_executions WHERE conversation_id = ? ORDER BY started_at DESC, rowid DESC LIMIT ?")
                .bind(c)
                .bind(limit)
                .fetch_all(self.pool())
                .await
                .map_err(db_err),
            None => sqlx::query_as::<_, ToolExecRow>("SELECT * FROM tool_executions ORDER BY started_at DESC, rowid DESC LIMIT ?")
                .bind(limit)
                .fetch_all(self.pool())
                .await
                .map_err(db_err),
        }
    }

    /// Calls left "pending"/"running" by a crash are closed at startup; they are
    /// never re-run automatically (a side effect may already have happened).
    pub async fn close_stale_tool_execs(&self) -> AppResult<u64> {
        let r = sqlx::query(
            "UPDATE tool_executions SET status = 'interrupted', ended_at = ? WHERE status IN ('pending', 'awaiting_approval', 'running')",
        )
        .bind(now_rfc3339())
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
    async fn servers_prefs_grants_audit() {
        let db = Db::open_in_memory().await.unwrap();
        let s = db.insert_mcp_server("fs", "stdio", &serde_json::json!({"command": "npx"}), None, true).await.unwrap();
        assert!(db.insert_mcp_server("fs", "stdio", &serde_json::json!({}), None, true).await.unwrap_err().to_string().contains("already exists"));
        db.set_tool_pref(&s, "write_file", true, "ask").await.unwrap();
        db.set_tool_pref(&s, "write_file", false, "deny").await.unwrap();
        let p = db.tool_prefs(&s).await.unwrap();
        assert_eq!((p.len(), p[0].enabled, p[0].policy.as_str()), (1, 0, "deny"));
        assert!(db.set_tool_pref(&s, "x", true, "yolo").await.is_err());

        let c = db.create_conversation("t", None, &serde_json::json!({})).await.unwrap();
        let tid = format!("mcp:{s}/write_file");
        assert!(db.insert_grant("session", &tid, None).await.is_err());
        db.insert_grant("session", &tid, Some(&c.id)).await.unwrap();
        db.insert_grant("persistent", &tid, None).await.unwrap();
        assert_eq!(db.grants_for_tool(&tid).await.unwrap().len(), 2);
        db.delete_conversation(&c.id).await.unwrap();
        assert_eq!(db.grants_for_tool(&tid).await.unwrap().len(), 1, "session grant goes with its chat");

        let e = db
            .start_tool_exec(&crate::mcp::NewToolExec { conversation_id: None, tool: &tid, source: "mcp:fs", args_redacted: "{}", tool_call_id: Some("c1"), model_tool_name: Some("mcp__fs__write_file"), environment_id: None, run_id: None, agent_id: None })
            .await
            .unwrap();
        assert_eq!(db.close_stale_tool_execs().await.unwrap(), 1);
        db.finish_tool_exec(&e, "ok", Some("done"), &serde_json::json!({"answer": "allow_once"}), 12).await.unwrap();
        let rows = db.list_tool_execs(None, 10).await.unwrap();
        assert_eq!((rows[0].status.as_str(), rows[0].duration_ms), ("ok", Some(12)));

        db.delete_mcp_server(&s).await.unwrap();
        assert!(db.grants_for_tool(&tid).await.unwrap().is_empty() && db.tool_prefs(&s).await.unwrap().is_empty());
    }
}

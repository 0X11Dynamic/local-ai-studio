//! Agent tasks (`agent_runs`), their steps and subagents.

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AgentRunRow {
    pub id: String,
    pub objective: String,
    pub model_id: Option<String>,
    pub env_id: Option<String>,
    pub state: String,
    pub budgets: String,
    pub usage: String,
    pub final_answer: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub title: String,
    pub settings: String,
    /// Not loaded by the normal queries (can be large); see `get_run_transcript`.
    #[sqlx(default)]
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub transcript: String,
}

/// A run without its (possibly large) transcript, for lists.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AgentRunSummary {
    pub id: String,
    pub title: String,
    pub objective: String,
    pub state: String,
    pub budgets: String,
    pub usage: String,
    pub created_at: String,
    pub ended_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SubagentRow {
    pub id: String,
    pub parent_run_id: String,
    pub parent_agent_id: Option<String>,
    pub objective: String,
    pub state: String,
    pub budgets: String,
    pub usage: String,
    pub result: Option<String>,
    pub created_at: String,
    pub ended_at: Option<String>,
    pub label: String,
    pub settings: String,
    /// Not loaded by the list queries (can be large).
    #[sqlx(default)]
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub transcript: String,
    pub error: Option<String>,
    pub started_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AgentStepRow {
    pub id: String,
    pub run_id: String,
    /// None for the root agent, else the subagent id.
    pub parent_agent_id: Option<String>,
    pub seq: i64,
    /// model | tool | subagents | note | error
    pub kind: String,
    pub data: String,
    pub state: String,
    pub started_at: String,
    pub ended_at: Option<String>,
}

/// States from which a run is finished for good.
pub const FINAL_STATES: &[&str] = &["completed", "failed", "cancelled", "budget_exhausted"];

impl Db {
    pub async fn create_run(&self, title: &str, objective: &str, model_id: Option<&str>, settings: &serde_json::Value, budgets: &serde_json::Value) -> AppResult<String> {
        if objective.trim().is_empty() {
            return Err(AppError::InvalidInput("the task needs an objective".into()));
        }
        let id = new_id();
        sqlx::query("INSERT INTO agent_runs(id, objective, model_id, state, budgets, usage, created_at, title, settings) VALUES(?, ?, ?, 'queued', ?, '{}', ?, ?, ?)")
            .bind(&id)
            .bind(objective.trim())
            .bind(model_id)
            .bind(budgets.to_string())
            .bind(now_rfc3339())
            .bind(title.trim())
            .bind(settings.to_string())
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(id)
    }

    pub async fn get_run(&self, id: &str) -> AppResult<AgentRunRow> {
        sqlx::query_as::<_, AgentRunRow>("SELECT id, objective, model_id, env_id, state, budgets, usage, final_answer, error, created_at, started_at, ended_at, title, settings FROM agent_runs WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("task {id} not found")))
    }

    /// The saved conversation of a task (for resuming it).
    pub async fn get_run_transcript(&self, id: &str) -> AppResult<String> {
        let r: Option<(String,)> = sqlx::query_as("SELECT transcript FROM agent_runs WHERE id = ?").bind(id).fetch_optional(self.pool()).await.map_err(db_err)?;
        r.map(|t| t.0).ok_or_else(|| AppError::NotFound(format!("task {id} not found")))
    }

    pub async fn list_runs(&self, limit: i64) -> AppResult<Vec<AgentRunSummary>> {
        // Chat runs (hidden rows that hold a chat's subagents) are not tasks.
        sqlx::query_as::<_, AgentRunSummary>(
            "SELECT id, title, objective, state, budgets, usage, created_at, ended_at FROM agent_runs              WHERE json_extract(settings, '$.chat_conversation_id') IS NULL ORDER BY created_at DESC LIMIT ?",
        )
            .bind(limit)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    // ---------- chat runs ----------

    /// The hidden run that holds a chat's subagents (one per conversation),
    /// created on first use; title, settings and budgets are refreshed. The
    /// conversation id is stored in its settings as `chat_conversation_id`.
    pub async fn ensure_chat_run(&self, conversation_id: &str, title: &str, model_id: Option<&str>, settings: &serde_json::Value, budgets: &serde_json::Value) -> AppResult<String> {
        let mut settings = settings.clone();
        if let Some(o) = settings.as_object_mut() {
            o.insert("chat_conversation_id".into(), conversation_id.into());
        } else {
            settings = serde_json::json!({ "chat_conversation_id": conversation_id });
        }
        if let Some(id) = self.find_chat_run(conversation_id).await? {
            sqlx::query("UPDATE agent_runs SET title = ?, settings = ?, budgets = ?, model_id = COALESCE(?, model_id) WHERE id = ?")
                .bind(title.trim())
                .bind(settings.to_string())
                .bind(budgets.to_string())
                .bind(model_id)
                .bind(&id)
                .execute(self.pool())
                .await
                .map_err(db_err)?;
            return Ok(id);
        }
        self.create_run(title, &format!("Subagents started from the chat \"{}\"", title.trim_start_matches("Chat: ")), model_id, &settings, budgets).await
    }

    pub async fn find_chat_run(&self, conversation_id: &str) -> AppResult<Option<String>> {
        let r: Option<(String,)> = sqlx::query_as("SELECT id FROM agent_runs WHERE json_extract(settings, '$.chat_conversation_id') = ? ORDER BY created_at LIMIT 1")
            .bind(conversation_id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?;
        Ok(r.map(|r| r.0))
    }

    /// Every chat run: (run id, title, conversation id), for the Subagents page.
    pub async fn list_chat_runs(&self) -> AppResult<Vec<(String, String, String)>> {
        sqlx::query_as("SELECT id, title, json_extract(settings, '$.chat_conversation_id') FROM agent_runs WHERE json_extract(settings, '$.chat_conversation_id') IS NOT NULL ORDER BY created_at DESC")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    /// Save progress after a turn.
    pub async fn save_run_progress(&self, id: &str, transcript: &serde_json::Value, usage: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE agent_runs SET transcript = ?, usage = ? WHERE id = ?")
            .bind(transcript.to_string())
            .bind(usage.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_run_state(&self, id: &str, state: &str, final_answer: Option<&str>, error: Option<&str>) -> AppResult<()> {
        let now = now_rfc3339();
        let ended = FINAL_STATES.contains(&state).then_some(now.clone());
        sqlx::query(
            "UPDATE agent_runs SET state = ?, final_answer = COALESCE(?, final_answer), error = ?, \
             started_at = CASE WHEN ? = 'running' AND started_at IS NULL THEN ? ELSE started_at END, ended_at = ? WHERE id = ?",
        )
        .bind(state)
        .bind(final_answer)
        .bind(error)
        .bind(state)
        .bind(&now)
        .bind(ended)
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_run_budgets(&self, id: &str, budgets: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE agent_runs SET budgets = ? WHERE id = ?").bind(budgets.to_string()).bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    /// Plan approved: store the new settings, drop the planning transcript and
    /// the plan answer; the task is then launched again with full tools.
    pub async fn set_run_plan_approved(&self, id: &str, settings: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE agent_runs SET settings = ?, transcript = '[]', final_answer = NULL WHERE id = ?")
            .bind(settings.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn delete_run(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM agent_runs WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    /// At startup: anything that was running when the app stopped is marked
    /// interrupted (never re-run automatically).
    pub async fn interrupt_running_runs(&self) -> AppResult<u64> {
        let a = sqlx::query("UPDATE agent_runs SET state = 'interrupted' WHERE state IN ('queued', 'running', 'waiting')")
            .execute(self.pool())
            .await
            .map_err(db_err)?
            .rows_affected();
        sqlx::query("UPDATE subagents SET state = 'interrupted' WHERE state IN ('queued', 'running', 'waiting', 'waiting_for_user')").execute(self.pool()).await.map_err(db_err)?;
        sqlx::query("UPDATE agent_steps SET state = 'interrupted', ended_at = ? WHERE state = 'running'").bind(now_rfc3339()).execute(self.pool()).await.map_err(db_err)?;
        // A task whose main agent asked the user a question stays `waiting_for_user`
        // (its question can still be answered, which then resumes it). Subagents
        // are not resumed, so their questions end with them.
        sqlx::query("UPDATE agent_steps SET state = 'interrupted', ended_at = ? WHERE kind = 'question' AND state = 'waiting' AND parent_agent_id IS NOT NULL")
            .bind(now_rfc3339())
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(a)
    }

    /// Questions (`ask_user`) still waiting for an answer, in every task.
    pub async fn waiting_questions(&self) -> AppResult<Vec<AgentStepRow>> {
        sqlx::query_as::<_, AgentStepRow>("SELECT * FROM agent_steps WHERE kind = 'question' AND state = 'waiting' ORDER BY started_at").fetch_all(self.pool()).await.map_err(db_err)
    }

    /// End a run's waiting questions (cancelled, or answered another way).
    pub async fn close_waiting_questions(&self, run_id: &str, state: &str) -> AppResult<()> {
        sqlx::query("UPDATE agent_steps SET state = ?, ended_at = ? WHERE run_id = ? AND kind = 'question' AND state = 'waiting'")
            .bind(state)
            .bind(now_rfc3339())
            .bind(run_id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    // ---------- steps ----------
    pub async fn add_step(&self, run_id: &str, agent_id: Option<&str>, kind: &str, data: &serde_json::Value, state: &str) -> AppResult<AgentStepRow> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query(
            "INSERT INTO agent_steps(id, run_id, parent_agent_id, seq, kind, data, state, started_at, ended_at) \
             VALUES(?, ?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_steps WHERE run_id = ?), ?, ?, ?, ?, CASE WHEN ? = 'running' THEN NULL ELSE ? END)",
        )
        .bind(&id)
        .bind(run_id)
        .bind(agent_id)
        .bind(run_id)
        .bind(kind)
        .bind(data.to_string())
        .bind(state)
        .bind(&now)
        .bind(state)
        .bind(&now)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        sqlx::query_as::<_, AgentStepRow>("SELECT * FROM agent_steps WHERE id = ?").bind(&id).fetch_one(self.pool()).await.map_err(db_err)
    }

    pub async fn update_step(&self, id: &str, data: &serde_json::Value, state: &str) -> AppResult<()> {
        let ended = (state != "running").then(now_rfc3339);
        sqlx::query("UPDATE agent_steps SET data = ?, state = ?, ended_at = COALESCE(?, ended_at) WHERE id = ?")
            .bind(data.to_string())
            .bind(state)
            .bind(ended)
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn list_steps(&self, run_id: &str) -> AppResult<Vec<AgentStepRow>> {
        sqlx::query_as::<_, AgentStepRow>("SELECT * FROM agent_steps WHERE run_id = ? ORDER BY seq").bind(run_id).fetch_all(self.pool()).await.map_err(db_err)
    }

    // ---------- subagents ----------
    pub async fn create_subagent(&self, run_id: &str, label: &str, objective: &str, budgets: &serde_json::Value, settings: &serde_json::Value) -> AppResult<String> {
        let id = new_id();
        sqlx::query("INSERT INTO subagents(id, parent_run_id, objective, state, budgets, usage, created_at, label, settings) VALUES(?, ?, ?, 'queued', ?, '{}', ?, ?, ?)")
            .bind(&id)
            .bind(run_id)
            .bind(objective)
            .bind(budgets.to_string())
            .bind(now_rfc3339())
            .bind(label)
            .bind(settings.to_string())
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(id)
    }

    pub async fn save_subagent_progress(&self, id: &str, transcript: &serde_json::Value, usage: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE subagents SET transcript = ?, usage = ? WHERE id = ?")
            .bind(transcript.to_string())
            .bind(usage.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_subagent_state(&self, id: &str, state: &str, result: Option<&str>, error: Option<&str>) -> AppResult<()> {
        let now = now_rfc3339();
        let ended = (FINAL_STATES.contains(&state) || state == "interrupted").then_some(now.clone());
        sqlx::query(
            "UPDATE subagents SET state = ?, result = COALESCE(?, result), error = ?, \
             started_at = CASE WHEN ? = 'running' AND started_at IS NULL THEN ? ELSE started_at END, ended_at = ? WHERE id = ?",
        )
        .bind(state)
        .bind(result)
        .bind(error)
        .bind(state)
        .bind(&now)
        .bind(ended)
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn get_subagent(&self, id: &str) -> AppResult<SubagentRow> {
        sqlx::query_as::<_, SubagentRow>("SELECT id, parent_run_id, parent_agent_id, objective, state, budgets, usage, result, created_at, ended_at, label, settings, error, started_at FROM subagents WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("subagent {id} not found")))
    }

    pub async fn list_subagents(&self, run_id: &str) -> AppResult<Vec<SubagentRow>> {
        sqlx::query_as::<_, SubagentRow>("SELECT id, parent_run_id, parent_agent_id, objective, state, budgets, usage, result, created_at, ended_at, label, settings, error, started_at FROM subagents WHERE parent_run_id = ? ORDER BY created_at").bind(run_id).fetch_all(self.pool()).await.map_err(db_err)
    }

    /// Recent subagents across tasks (Subagents page).
    pub async fn recent_subagents(&self, limit: i64) -> AppResult<Vec<SubagentRow>> {
        sqlx::query_as::<_, SubagentRow>("SELECT id, parent_run_id, parent_agent_id, objective, state, budgets, usage, result, created_at, ended_at, label, settings, error, started_at FROM subagents ORDER BY created_at DESC LIMIT ?").bind(limit).fetch_all(self.pool()).await.map_err(db_err)
    }

    /// "Allow for this task" grant.
    pub async fn insert_run_grant(&self, tool_id: &str, run_id: &str) -> AppResult<String> {
        let id = new_id();
        sqlx::query("INSERT INTO permission_grants(id, scope, subject, capability, run_id, created_at) VALUES(?, 'session', ?, 'tool.call', ?, ?)")
            .bind(&id)
            .bind(tool_id)
            .bind(run_id)
            .bind(now_rfc3339())
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn runs_steps_subagents_and_task_grants() {
        let db = Db::open_in_memory().await.unwrap();
        assert!(db.create_run("t", "  ", None, &serde_json::json!({}), &serde_json::json!({})).await.is_err());
        let id = db.create_run("Fix tests", "make cargo test pass", None, &serde_json::json!({ "env": "local" }), &serde_json::json!({ "max_turns": 5 })).await.unwrap();
        db.set_run_state(&id, "running", None, None).await.unwrap();
        let r = db.get_run(&id).await.unwrap();
        assert!(r.started_at.is_some() && r.ended_at.is_none());
        db.save_run_progress(&id, &serde_json::json!([{ "role": "user", "content": "x" }]), &serde_json::json!({ "turns": 1 })).await.unwrap();
        assert!(db.get_run(&id).await.unwrap().transcript.is_empty(), "lists and details skip the transcript");
        assert!(db.get_run_transcript(&id).await.unwrap().contains("\"x\""));
        let s1 = db.add_step(&id, None, "model", &serde_json::json!({ "text": "a" }), "done").await.unwrap();
        let s2 = db.add_step(&id, None, "tool", &serde_json::json!({}), "running").await.unwrap();
        assert_eq!((s1.seq, s2.seq), (1, 2));
        assert!(s2.ended_at.is_none());
        db.update_step(&s2.id, &serde_json::json!({ "ok": true }), "done").await.unwrap();
        let sub = db.create_subagent(&id, "A", "count lines", &serde_json::json!({}), &serde_json::json!({ "tools": ["fs_read"] })).await.unwrap();
        db.set_subagent_state(&sub, "running", None, None).await.unwrap();
        db.insert_run_grant("builtin:local/shell_execute", &id).await.unwrap();
        let g = db.grants_for_tool("builtin:local/shell_execute").await.unwrap();
        assert_eq!(g[0].run_id.as_deref(), Some(id.as_str()));
        // Restart: running work becomes interrupted.
        assert_eq!(db.interrupt_running_runs().await.unwrap(), 1);
        assert_eq!(db.get_run(&id).await.unwrap().state, "interrupted");
        assert_eq!(db.get_subagent(&sub).await.unwrap().state, "interrupted");
        db.set_run_state(&id, "completed", Some("done"), None).await.unwrap();
        let r = db.get_run(&id).await.unwrap();
        assert!(r.ended_at.is_some() && r.final_answer.as_deref() == Some("done"));
        assert_eq!(db.list_runs(10).await.unwrap().len(), 1);
        // A chat run is hidden from the task list and reused per conversation.
        let c1 = db.ensure_chat_run("conv-1", "Chat: Hello", None, &serde_json::json!({ "subagents": true }), &serde_json::json!({})).await.unwrap();
        let c2 = db.ensure_chat_run("conv-1", "Chat: Renamed", None, &serde_json::json!({ "subagents": true }), &serde_json::json!({ "max_turns": 3 })).await.unwrap();
        assert_eq!(c1, c2);
        assert_eq!(db.get_run(&c1).await.unwrap().title, "Chat: Renamed");
        assert_eq!(db.find_chat_run("conv-1").await.unwrap().as_deref(), Some(c1.as_str()));
        assert!(db.find_chat_run("conv-2").await.unwrap().is_none());
        assert_eq!(db.list_runs(10).await.unwrap().len(), 1, "chat runs are not listed as tasks");
        assert_eq!(db.list_chat_runs().await.unwrap(), vec![(c1.clone(), "Chat: Renamed".to_string(), "conv-1".to_string())]);
        db.delete_run(&c1).await.unwrap();
        db.delete_run(&id).await.unwrap();
        assert!(db.list_subagents(&id).await.unwrap().is_empty() && db.list_steps(&id).await.unwrap().is_empty());
        assert!(db.grants_for_tool("builtin:local/shell_execute").await.unwrap().is_empty(), "task grants go with the task");
    }

    #[tokio::test]
    async fn a_question_waiting_for_the_user_survives_a_restart() {
        let db = Db::open_in_memory().await.unwrap();
        let id = db.create_run("Ask", "pick a color", None, &serde_json::json!({}), &serde_json::json!({})).await.unwrap();
        let sub = db.create_subagent(&id, "A", "paint", &serde_json::json!({}), &serde_json::json!({})).await.unwrap();
        db.set_run_state(&id, "waiting_for_user", None, None).await.unwrap();
        db.set_subagent_state(&sub, "waiting_for_user", None, None).await.unwrap();
        let q1 = db.add_step(&id, None, "question", &serde_json::json!({ "question": "Which?" }), "waiting").await.unwrap();
        let q2 = db.add_step(&id, Some(&sub), "question", &serde_json::json!({ "question": "Size?" }), "waiting").await.unwrap();
        assert_eq!(db.waiting_questions().await.unwrap().len(), 2);
        // Restart: the main agent's question and the run's state stay; the subagent ends.
        assert_eq!(db.interrupt_running_runs().await.unwrap(), 0, "waiting_for_user is not interrupted");
        assert_eq!(db.get_run(&id).await.unwrap().state, "waiting_for_user");
        assert_eq!(db.get_subagent(&sub).await.unwrap().state, "interrupted");
        let left = db.waiting_questions().await.unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, q1.id);
        assert!(db.list_steps(&id).await.unwrap().iter().any(|s| s.id == q2.id && s.state == "interrupted"));
        // Cancelling closes what is left.
        db.close_waiting_questions(&id, "cancelled").await.unwrap();
        assert!(db.waiting_questions().await.unwrap().is_empty());
    }
}

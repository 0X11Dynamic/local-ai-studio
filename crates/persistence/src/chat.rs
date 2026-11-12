use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ConversationRow {
    pub id: String,
    pub title: String,
    pub model_id: Option<String>,
    /// JSON: per-chat settings (instruction profile, sampling, overflow policy).
    pub settings: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct MessageRow {
    pub id: String,
    pub conversation_id: String,
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    /// JSON: per-response statistics (tokens, speed, latency, stop reason).
    pub stats: Option<String>,
    pub created_at: String,
    pub model_name: Option<String>,
    /// JSON: attachment records (files the user attached to this message).
    pub attachments: String,
}

#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub model_id: Option<String>,
    pub updated_at: String,
    pub message_count: i64,
}

impl Db {
    pub async fn create_conversation(&self, title: &str, model_id: Option<&str>, settings: &serde_json::Value) -> AppResult<ConversationRow> {
        let id = new_id();
        let now = now_rfc3339();
        sqlx::query("INSERT INTO conversations(id, title, model_id, settings, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)")
            .bind(&id)
            .bind(title)
            .bind(model_id)
            .bind(settings.to_string())
            .bind(&now)
            .bind(&now)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        self.get_conversation(&id).await
    }

    pub async fn get_conversation(&self, id: &str) -> AppResult<ConversationRow> {
        sqlx::query_as::<_, ConversationRow>("SELECT * FROM conversations WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    /// Most recent first. `query` matches titles and message text.
    pub async fn list_conversations(&self, query: Option<&str>) -> AppResult<Vec<ConversationSummary>> {
        let q = query.map(str::trim).filter(|q| !q.is_empty());
        match q {
            None => sqlx::query_as::<_, ConversationSummary>(
                "SELECT c.id, c.title, c.model_id, c.updated_at, \
                 (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count \
                 FROM conversations c ORDER BY c.updated_at DESC",
            )
            .fetch_all(self.pool())
            .await
            .map_err(db_err),
            Some(q) => {
                // Escape LIKE wildcards (with '!' as the escape char) so the
                // user's text matches literally.
                let like = format!("%{}%", q.replace('!', "!!").replace('%', "!%").replace('_', "!_"));
                sqlx::query_as::<_, ConversationSummary>(
                    "SELECT c.id, c.title, c.model_id, c.updated_at, \
                     (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count \
                     FROM conversations c \
                     WHERE c.title LIKE ?1 ESCAPE '!' \
                        OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.content LIKE ?1 ESCAPE '!') \
                     ORDER BY c.updated_at DESC",
                )
                .bind(like)
                .fetch_all(self.pool())
                .await
                .map_err(db_err)
            }
        }
    }

    pub async fn rename_conversation(&self, id: &str, title: &str) -> AppResult<()> {
        sqlx::query("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?")
            .bind(title)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn update_conversation_settings(&self, id: &str, model_id: Option<&str>, settings: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE conversations SET model_id = COALESCE(?, model_id), settings = ?, updated_at = ? WHERE id = ?")
            .bind(model_id)
            .bind(settings.to_string())
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn touch_conversation(&self, id: &str) -> AppResult<()> {
        sqlx::query("UPDATE conversations SET updated_at = ? WHERE id = ?")
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    /// NotFound when there is no such conversation.
    pub async fn delete_conversation(&self, id: &str) -> AppResult<()> {
        let r = sqlx::query("DELETE FROM conversations WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        if r.rows_affected() == 0 {
            return Err(AppError::NotFound("conversation not found".into()));
        }
        Ok(())
    }

    /// Copy a conversation and all its messages (in order) under a new id.
    pub async fn duplicate_conversation(&self, id: &str) -> AppResult<ConversationRow> {
        let src = self.get_conversation(id).await?;
        let settings: serde_json::Value = serde_json::from_str(&src.settings).unwrap_or_default();
        let copy = self
            .create_conversation(&format!("{} (copy)", src.title), src.model_id.as_deref(), &settings)
            .await?;
        for m in self.list_messages(id).await? {
            sqlx::query(
                "INSERT INTO messages(id, conversation_id, role, content, reasoning, tool_calls, tool_call_id, stats, created_at, model_name, attachments) \
                 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(new_id())
            .bind(&copy.id)
            .bind(&m.role)
            .bind(&m.content)
            .bind(&m.reasoning)
            .bind(&m.tool_calls)
            .bind(&m.tool_call_id)
            .bind(&m.stats)
            .bind(&m.created_at)
            .bind(&m.model_name)
            .bind(&m.attachments)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        }
        Ok(copy)
    }

    pub async fn add_message(&self, conversation_id: &str, role: &str, content: &str, model_name: Option<&str>) -> AppResult<MessageRow> {
        let id = new_id();
        sqlx::query("INSERT INTO messages(id, conversation_id, role, content, created_at, model_name) VALUES(?, ?, ?, ?, ?, ?)")
            .bind(&id)
            .bind(conversation_id)
            .bind(role)
            .bind(content)
            .bind(now_rfc3339())
            .bind(model_name)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        self.touch_conversation(conversation_id).await?;
        self.get_message(&id).await
    }

    pub async fn get_message(&self, id: &str) -> AppResult<MessageRow> {
        sqlx::query_as::<_, MessageRow>("SELECT * FROM messages WHERE id = ?")
            .bind(id)
            .fetch_one(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn finish_message(&self, id: &str, content: &str, reasoning: Option<&str>, stats: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE messages SET content = ?, reasoning = ?, stats = ? WHERE id = ?")
            .bind(content)
            .bind(reasoning)
            .bind(stats.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_message_tool_calls(&self, id: &str, tool_calls: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE messages SET tool_calls = ? WHERE id = ?")
            .bind(tool_calls.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    /// A tool result message answering the assistant's call `tool_call_id`.
    pub async fn add_tool_message(&self, conversation_id: &str, tool_call_id: &str, content: &str, stats: &serde_json::Value) -> AppResult<MessageRow> {
        let id = new_id();
        sqlx::query("INSERT INTO messages(id, conversation_id, role, content, tool_call_id, stats, created_at) VALUES(?, ?, 'tool', ?, ?, ?, ?)")
            .bind(&id)
            .bind(conversation_id)
            .bind(content)
            .bind(tool_call_id)
            .bind(stats.to_string())
            .bind(now_rfc3339())
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        self.get_message(&id).await
    }

    pub async fn set_message_attachments(&self, id: &str, attachments: &serde_json::Value) -> AppResult<()> {
        sqlx::query("UPDATE messages SET attachments = ? WHERE id = ?")
            .bind(attachments.to_string())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn set_message_content(&self, id: &str, content: &str) -> AppResult<()> {
        sqlx::query("UPDATE messages SET content = ? WHERE id = ?")
            .bind(content)
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    /// Messages in conversation order. rowid breaks ties for identical timestamps.
    pub async fn list_messages(&self, conversation_id: &str) -> AppResult<Vec<MessageRow>> {
        sqlx::query_as::<_, MessageRow>("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid")
            .bind(conversation_id)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    /// Delete this message and every later one in the conversation (used for
    /// edit-and-resend and regenerate).
    pub async fn truncate_from(&self, conversation_id: &str, message_id: &str) -> AppResult<u64> {
        let msgs = self.list_messages(conversation_id).await?;
        let Some(pos) = msgs.iter().position(|m| m.id == message_id) else {
            return Ok(0);
        };
        let mut n = 0;
        for m in &msgs[pos..] {
            n += sqlx::query("DELETE FROM messages WHERE id = ?")
                .bind(&m.id)
                .execute(self.pool())
                .await
                .map_err(db_err)?
                .rows_affected();
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn conversations_messages_search_duplicate_truncate() {
        let db = Db::open_in_memory().await.unwrap();
        let c = db.create_conversation("First chat", None, &serde_json::json!({})).await.unwrap();
        let u = db.add_message(&c.id, "user", "What is 100% of fifty_two?", None).await.unwrap();
        assert_eq!(u.attachments, "[]", "no attachments by default");
        db.set_message_attachments(&u.id, &serde_json::json!([{ "id": "a1", "name": "notes.txt" }])).await.unwrap();
        let a = db.add_message(&c.id, "assistant", "", Some("m")).await.unwrap();
        db.finish_message(&a.id, "52", None, &serde_json::json!({"tokens": 3})).await.unwrap();
        db.add_message(&c.id, "user", "thanks", None).await.unwrap();
        assert_eq!(db.list_messages(&c.id).await.unwrap().len(), 3);
        // LIKE wildcards in the query are literal.
        assert_eq!(db.list_conversations(Some("100%")).await.unwrap().len(), 1);
        assert_eq!(db.list_conversations(Some("fifty_two")).await.unwrap().len(), 1);
        assert_eq!(db.list_conversations(Some("1000")).await.unwrap().len(), 0);
        let d = db.duplicate_conversation(&c.id).await.unwrap();
        assert_eq!(db.list_messages(&d.id).await.unwrap().len(), 3);
        assert!(db.list_messages(&d.id).await.unwrap()[0].attachments.contains("notes.txt"), "the copy keeps attachments");
        assert_eq!(db.truncate_from(&c.id, &u.id).await.unwrap(), 3);
        assert!(db.list_messages(&c.id).await.unwrap().is_empty());
        assert_eq!(db.list_messages(&d.id).await.unwrap()[1].content, "52", "copy is independent");
        db.delete_conversation(&d.id).await.unwrap();
        assert!(db.list_messages(&d.id).await.unwrap().is_empty(), "messages cascade-deleted");
        // Deleting again, or an id that never existed, is NotFound.
        assert!(matches!(db.delete_conversation(&d.id).await, Err(app_core::AppError::NotFound(_))));
        assert!(matches!(db.delete_conversation("..").await, Err(app_core::AppError::NotFound(_))));
        assert!(db.get_conversation(&c.id).await.is_ok(), "the other chat is untouched");
    }
}

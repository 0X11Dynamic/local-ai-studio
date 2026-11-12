use app_core::{AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ApiLogRow {
    pub id: String,
    pub endpoint: String,
    pub method: String,
    pub status: i64,
    pub model: Option<String>,
    pub latency_ms: i64,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub error: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Default)]
pub struct NewApiLog {
    pub endpoint: String,
    pub method: String,
    pub status: u16,
    pub model: Option<String>,
    pub latency_ms: u64,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub error: Option<String>,
}

impl Db {
    pub async fn insert_api_log(&self, l: &NewApiLog) -> AppResult<()> {
        sqlx::query(
            "INSERT INTO api_request_logs(id, endpoint, method, status, model, latency_ms, prompt_tokens, completion_tokens, error, created_at) \
             VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(new_id())
        .bind(&l.endpoint)
        .bind(&l.method)
        .bind(l.status as i64)
        .bind(&l.model)
        .bind(l.latency_ms as i64)
        .bind(l.prompt_tokens.map(|v| v as i64))
        .bind(l.completion_tokens.map(|v| v as i64))
        .bind(&l.error)
        .bind(now_rfc3339())
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn recent_api_logs(&self, limit: i64) -> AppResult<Vec<ApiLogRow>> {
        sqlx::query_as::<_, ApiLogRow>("SELECT * FROM api_request_logs ORDER BY created_at DESC, rowid DESC LIMIT ?")
            .bind(limit)
            .fetch_all(self.pool())
            .await
            .map_err(db_err)
    }

    /// Delete log rows older than `days`.
    pub async fn prune_api_logs(&self, days: i64) -> AppResult<u64> {
        let r = sqlx::query("DELETE FROM api_request_logs WHERE julianday(created_at) < julianday('now') - ?")
            .bind(days as f64)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(r.rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn logs_insert_list_prune() {
        let db = Db::open_in_memory().await.unwrap();
        db.insert_api_log(&NewApiLog { endpoint: "/v1/models".into(), method: "GET".into(), status: 200, latency_ms: 3, ..Default::default() })
            .await
            .unwrap();
        sqlx::query("INSERT INTO api_request_logs(id, endpoint, method, status, latency_ms, created_at) VALUES('old','/x','GET',200,1,'2020-01-01T00:00:00.000Z')")
            .execute(db.pool())
            .await
            .unwrap();
        assert_eq!(db.recent_api_logs(10).await.unwrap().len(), 2);
        assert_eq!(db.prune_api_logs(7).await.unwrap(), 1);
        assert_eq!(db.recent_api_logs(10).await.unwrap()[0].endpoint, "/v1/models");
    }
}

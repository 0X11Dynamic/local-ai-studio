//! SQLite persistence: connection pool, embedded migrations and typed repositories.

use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use app_core::{AppError, AppResult};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;

pub mod agents;
pub mod api_logs;
pub mod chat;
pub mod downloads;
pub mod exec;
pub mod mcp;
pub mod models;
pub mod presets;
pub mod settings;
pub mod skills;

pub use agents::{AgentRunRow, AgentRunSummary, AgentStepRow, SubagentRow};
pub use api_logs::{ApiLogRow, NewApiLog};
pub use skills::{SkillRevisionRow, SkillRow};
pub use chat::{ConversationRow, ConversationSummary, MessageRow};
pub use downloads::DownloadRow;
pub use exec::{BuiltinPrefRow, ExecProfileRow};
pub use mcp::{GrantRow, McpServerRow, NewToolExec, ToolExecRow, ToolPrefRow};
pub use models::{ModelInstanceRow, ModelRow};
pub use presets::PresetRow;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Handle to the application database. Cheap to clone.
#[derive(Clone)]
pub struct Db {
    pool: SqlitePool,
}

impl Db {
    /// Open (creating if needed) the database file and apply pending migrations.
    pub async fn open(path: &Path) -> AppResult<Self> {
        let url = format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"));
        let opts = SqliteConnectOptions::from_str(&url)
            .map_err(db_err)?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .busy_timeout(Duration::from_secs(5));
        Self::connect(opts).await
    }

    /// In-memory database for tests.
    pub async fn open_in_memory() -> AppResult<Self> {
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")
            .map_err(db_err)?
            .foreign_keys(true);
        Self::connect(opts).await
    }

    async fn connect(opts: SqliteConnectOptions) -> AppResult<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await
            .map_err(db_err)?;
        MIGRATOR.run(&pool).await.map_err(|e| AppError::Database(e.to_string()))?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

pub(crate) fn db_err(e: sqlx::Error) -> AppError {
    match e {
        sqlx::Error::RowNotFound => AppError::NotFound("record not found".into()),
        other => AppError::Database(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrations_apply_to_a_fresh_file() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("t.db")).await.unwrap();
        let n: (i64,) = sqlx::query_as(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE '_sqlx%' AND name NOT LIKE 'sqlite_%'",
        )
        .fetch_one(db.pool())
        .await
        .unwrap();
        assert_eq!(n.0, 19, "all spec section 12 tables plus builtin_tool_preferences should exist");
        // Re-opening must not re-run migrations or fail.
        drop(db);
        Db::open(&dir.path().join("t.db")).await.unwrap();
    }
}

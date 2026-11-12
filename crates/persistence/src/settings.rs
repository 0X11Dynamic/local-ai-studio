use app_core::AppResult;
use serde::{Serialize, de::DeserializeOwned};

use crate::{Db, db_err};

impl Db {
    /// Read a setting stored as JSON. Returns `None` when the key is absent.
    pub async fn get_setting<T: DeserializeOwned>(&self, key: &str) -> AppResult<Option<T>> {
        let row: Option<(String,)> = sqlx::query_as("SELECT value FROM app_settings WHERE key = ?")
            .bind(key)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?;
        match row {
            Some((v,)) => Ok(Some(serde_json::from_str(&v)?)),
            None => Ok(None),
        }
    }

    /// Store a setting as JSON (insert or replace).
    pub async fn set_setting<T: Serialize>(&self, key: &str, value: &T) -> AppResult<()> {
        let v = serde_json::to_string(value)?;
        sqlx::query(
            "INSERT INTO app_settings(key, value) VALUES(?, ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(key)
        .bind(v)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    /// All settings as raw JSON values, for the Settings page.
    pub async fn all_settings(&self) -> AppResult<serde_json::Map<String, serde_json::Value>> {
        let rows: Vec<(String, String)> = sqlx::query_as("SELECT key, value FROM app_settings")
            .fetch_all(self.pool())
            .await
            .map_err(db_err)?;
        let mut m = serde_json::Map::new();
        for (k, v) in rows {
            m.insert(k, serde_json::from_str(&v).unwrap_or(serde_json::Value::String(v)));
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use crate::Db;

    #[tokio::test]
    async fn settings_round_trip_and_overwrite() {
        let db = Db::open_in_memory().await.unwrap();
        assert_eq!(db.get_setting::<u16>("api.port").await.unwrap(), None);
        db.set_setting("api.port", &1234u16).await.unwrap();
        db.set_setting("api.port", &1235u16).await.unwrap();
        assert_eq!(db.get_setting::<u16>("api.port").await.unwrap(), Some(1235));
    }
}

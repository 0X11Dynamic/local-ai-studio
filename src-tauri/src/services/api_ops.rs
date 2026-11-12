//! The local API's native endpoints call the same code the UI uses.

use std::sync::Weak;

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use inference::LoadConfig;

use crate::commands::engine::{do_load, do_unload};
use crate::state::AppState;

/// Holds a weak reference so the API server does not keep the app state alive.
pub struct AppOps(pub Weak<AppState>);

impl AppOps {
    fn st(&self) -> AppResult<std::sync::Arc<AppState>> {
        self.0.upgrade().ok_or_else(|| AppError::Other("app is shutting down".into()))
    }
}

#[async_trait]
impl local_api::NativeOps for AppOps {
    async fn list_models(&self) -> AppResult<serde_json::Value> {
        let st = self.st()?;
        let active = st.engine.status();
        let models = st.db.list_models().await?;
        Ok(serde_json::json!({
            "models": models.iter().map(|m| serde_json::json!({
                "id": m.id,
                "name": m.display_name,
                "repo_id": m.repo_id,
                "quantization": m.quantization,
                "size_bytes": m.size,
                "status": m.status,
                "loaded": active.model_id.as_deref() == Some(m.id.as_str()) && active.pid.is_some(),
            })).collect::<Vec<_>>()
        }))
    }

    async fn load_model(&self, body: serde_json::Value) -> AppResult<serde_json::Value> {
        let st = self.st()?;
        let model_id = body["model_id"]
            .as_str()
            .ok_or_else(|| AppError::InvalidInput("body needs \"model_id\" (see GET /api/models)".into()))?
            .to_string();
        let m = st.db.get_model(&model_id).await?;
        let config: LoadConfig = match body.get("config") {
            Some(c) if !c.is_null() => serde_json::from_value(c.clone())?,
            _ => m
                .default_config
                .as_deref()
                .or(m.last_config.as_deref())
                .and_then(|c| serde_json::from_str(c).ok())
                .unwrap_or_default(),
        };
        let runtime = body["runtime_id"].as_str().map(str::to_string);
        let s = do_load(&st, model_id, config, runtime).await?;
        Ok(serde_json::to_value(&s)?)
    }

    async fn unload_model(&self) -> AppResult<serde_json::Value> {
        let st = self.st()?;
        Ok(serde_json::to_value(do_unload(&st).await?)?)
    }
}

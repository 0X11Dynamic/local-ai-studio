//! Typed client for the loopback llama-server HTTP API.

use std::time::Duration;

use app_core::{AppError, AppResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct LlamaClient {
    base: String,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Health {
    Ok,
    Loading,
    Unavailable,
}

/// The subset of `/props` the app relies on, plus the raw JSON for display.
#[derive(Debug, Clone, Serialize)]
pub struct ServerProps {
    /// Effective allocated context per slot, as the server reports it.
    pub n_ctx: Option<u64>,
    pub total_slots: Option<u64>,
    pub model_path: Option<String>,
    pub chat_template: Option<String>,
    pub build_info: Option<String>,
    pub vision: bool,
    pub raw: serde_json::Value,
}

impl LlamaClient {
    pub fn new(port: u16) -> Self {
        let http = reqwest::Client::builder()
            .no_proxy() // loopback traffic must never go through a system proxy
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("reqwest client");
        Self { base: format!("http://127.0.0.1:{port}"), http }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn health(&self) -> Health {
        match self.http.get(format!("{}/health", self.base)).timeout(Duration::from_secs(3)).send().await {
            Ok(r) if r.status().is_success() => Health::Ok,
            Ok(r) if r.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => Health::Loading,
            _ => Health::Unavailable,
        }
    }

    async fn get_json(&self, path: &str) -> AppResult<serde_json::Value> {
        let r = self
            .http
            .get(format!("{}{path}", self.base))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| AppError::Runtime(format!("backend request {path} failed: {e}")))?;
        if !r.status().is_success() {
            return Err(AppError::Runtime(format!("backend {path} returned HTTP {}", r.status())));
        }
        r.json().await.map_err(|e| AppError::Runtime(format!("backend {path} returned bad JSON: {e}")))
    }

    pub async fn props(&self) -> AppResult<ServerProps> {
        let raw = self.get_json("/props").await?;
        let dgs = &raw["default_generation_settings"];
        let n_ctx = dgs["n_ctx"].as_u64().or_else(|| raw["n_ctx"].as_u64());
        Ok(ServerProps {
            n_ctx,
            total_slots: raw["total_slots"].as_u64(),
            model_path: raw["model_path"].as_str().map(str::to_string),
            chat_template: raw["chat_template"].as_str().map(str::to_string),
            build_info: raw["build_info"].as_str().map(str::to_string),
            vision: raw["modalities"]["vision"].as_bool().unwrap_or(false),
            raw,
        })
    }

    pub async fn slots(&self) -> AppResult<serde_json::Value> {
        self.get_json("/slots").await
    }

    /// Token ids for `content` using the loaded model's own tokenizer.
    pub async fn tokenize(&self, content: &str, add_special: bool) -> AppResult<Vec<i64>> {
        #[derive(Deserialize)]
        struct R {
            tokens: Vec<serde_json::Value>,
        }
        let r = self
            .http
            .post(format!("{}/tokenize", self.base))
            .json(&serde_json::json!({ "content": content, "add_special": add_special }))
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| AppError::Runtime(format!("tokenize failed: {e}")))?;
        if !r.status().is_success() {
            return Err(AppError::Runtime(format!("tokenize returned HTTP {}", r.status())));
        }
        let r: R = r.json().await.map_err(|e| AppError::Runtime(format!("tokenize: {e}")))?;
        Ok(r.tokens.iter().filter_map(|t| t.as_i64().or_else(|| t["id"].as_i64())).collect())
    }

    /// Render OpenAI-style messages through the model's chat template.
    pub async fn apply_template(&self, messages: &serde_json::Value) -> AppResult<String> {
        self.apply_template_tools(messages, None).await
    }

    /// Like [`apply_template`], including tool definitions (they become part
    /// of the prompt, so they count against the context).
    pub async fn apply_template_tools(&self, messages: &serde_json::Value, tools: Option<&serde_json::Value>) -> AppResult<String> {
        let mut body = serde_json::json!({ "messages": messages });
        if let Some(t) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            body["tools"] = t.clone();
        }
        let r = self
            .http
            .post(format!("{}/apply-template", self.base))
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| AppError::Runtime(format!("apply-template failed: {e}")))?;
        if !r.status().is_success() {
            return Err(AppError::Runtime(format!("apply-template returned HTTP {}", r.status())));
        }
        let v: serde_json::Value = r.json().await.map_err(|e| AppError::Runtime(format!("apply-template: {e}")))?;
        v["prompt"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| AppError::Runtime("apply-template response has no prompt".into()))
    }
}

//! Local HTTP API (spec section 11).
//!
//! OpenAI-compatible endpoints proxy to the loaded llama-server:
//!   GET  /v1/models
//!   POST /v1/chat/completions   (JSON or SSE streaming)
//!   POST /v1/completions
//! Native management endpoints (always require the bearer token):
//!   GET  /api/models
//!   POST /api/models/load      {"model_id": "...", "config": {...}?}
//!   POST /api/models/unload
//!
//! Defaults: bind 127.0.0.1, CORS off, any request carrying a browser Origin
//! that is not explicitly allowed is refused (a web page must not be able to
//! drive the local model or load models through "simple" cross-site POSTs).
//! Binding beyond loopback requires authentication.

use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use inference::Engine;
use persistence::{Db, NewApiLog};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_PORT: u16 = 1234;
const MAX_BODY: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    /// Require the bearer token on /v1 endpoints too (always on for /api and LAN).
    pub require_auth: bool,
    /// Browser origins allowed to call the API (exact match). Empty = none.
    pub cors_origins: Vec<String>,
    pub max_concurrent: usize,
    pub rate_per_minute: u32,
    pub log_retention_days: i64,
    pub autostart: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            require_auth: false,
            cors_origins: vec![],
            max_concurrent: 4,
            rate_per_minute: 240,
            log_retention_days: 7,
            autostart: false,
        }
    }
}

impl ApiConfig {
    pub fn is_loopback(&self) -> bool {
        self.host.parse::<IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(self.host == "localhost")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.port == 0 {
            return Err("port must be 1-65535".into());
        }
        if self.host != "localhost" && self.host.parse::<IpAddr>().is_err() {
            return Err("host must be an IP address (127.0.0.1 for this PC only, 0.0.0.0 for the LAN)".into());
        }
        if !self.is_loopback() && !self.require_auth {
            return Err("binding beyond this PC requires authentication; turn on 'Require API token'".into());
        }
        if self.max_concurrent == 0 || self.max_concurrent > 32 {
            return Err("concurrent generations must be 1-32".into());
        }
        for o in &self.cors_origins {
            if !(o.starts_with("http://") || o.starts_with("https://")) || o.ends_with('/') || o == "*" {
                return Err(format!("CORS origin '{o}' must be an exact origin like http://localhost:3000"));
            }
        }
        Ok(())
    }
}

/// Model management operations provided by the app (the same code paths the UI uses).
#[async_trait]
pub trait NativeOps: Send + Sync + 'static {
    async fn list_models(&self) -> AppResult<serde_json::Value>;
    async fn load_model(&self, body: serde_json::Value) -> AppResult<serde_json::Value>;
    async fn unload_model(&self) -> AppResult<serde_json::Value>;
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiStatus {
    pub running: bool,
    pub address: Option<String>,
    pub started_at: Option<String>,
    pub config: ApiConfig,
    pub last_error: Option<String>,
}

struct Running {
    addr: SocketAddr,
    started_at: String,
    shutdown: CancellationToken,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
pub struct ApiServer {
    inner: Arc<Mutex<Option<Running>>>,
    last_error: Arc<Mutex<Option<String>>>,
    config: Arc<Mutex<ApiConfig>>,
}

impl Default for ApiServer {
    fn default() -> Self {
        Self::new(ApiConfig::default())
    }
}

#[derive(Clone)]
struct AppState {
    engine: Engine,
    db: Db,
    ops: Arc<dyn NativeOps>,
    cfg: ApiConfig,
    token: Option<String>,
    gen_slots: Arc<Semaphore>,
    rate: Arc<Mutex<RateWindow>>,
}

struct RateWindow {
    start: Instant,
    count: u32,
}

impl ApiServer {
    pub fn new(cfg: ApiConfig) -> Self {
        Self { inner: Arc::new(Mutex::new(None)), last_error: Arc::new(Mutex::new(None)), config: Arc::new(Mutex::new(cfg)) }
    }

    pub fn status(&self) -> ApiStatus {
        let g = self.inner.lock().unwrap();
        ApiStatus {
            running: g.is_some(),
            address: g.as_ref().map(|r| format!("http://{}", r.addr)),
            started_at: g.as_ref().map(|r| r.started_at.clone()),
            config: self.config.lock().unwrap().clone(),
            last_error: self.last_error.lock().unwrap().clone(),
        }
    }

    /// Start (or restart) the server with `cfg`.
    pub async fn start(&self, cfg: ApiConfig, token: Option<String>, engine: Engine, db: Db, ops: Arc<dyn NativeOps>) -> AppResult<ApiStatus> {
        cfg.validate().map_err(AppError::InvalidInput)?;
        if cfg.require_auth && token.as_deref().is_none_or(str::is_empty) {
            return Err(AppError::InvalidInput("authentication is on but no API token exists; generate one first".into()));
        }
        self.stop().await;
        *self.config.lock().unwrap() = cfg.clone();
        let host = if cfg.host == "localhost" { "127.0.0.1" } else { cfg.host.as_str() };
        let addr: SocketAddr = format!("{host}:{}", cfg.port).parse().map_err(|e| AppError::InvalidInput(format!("bad address: {e}")))?;
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                let msg = format!(
                    "port {} is already in use by another program (LM Studio's server also defaults to 1234). Stop it or choose another port.",
                    cfg.port
                );
                *self.last_error.lock().unwrap() = Some(msg.clone());
                return Err(AppError::Conflict(msg));
            }
            Err(e) => {
                let msg = format!("could not listen on {addr}: {e}");
                *self.last_error.lock().unwrap() = Some(msg.clone());
                return Err(AppError::Io(std::io::Error::new(e.kind(), msg)));
            }
        };
        let bound = listener.local_addr()?;
        let _ = db.prune_api_logs(cfg.log_retention_days.max(1)).await;
        let state = AppState {
            engine,
            db,
            ops,
            gen_slots: Arc::new(Semaphore::new(cfg.max_concurrent)),
            rate: Arc::new(Mutex::new(RateWindow { start: Instant::now(), count: 0 })),
            token,
            cfg,
        };
        let app = router(state);
        let shutdown = CancellationToken::new();
        let sd = shutdown.clone();
        let join = tokio::spawn(async move {
            let r = axum::serve(listener, app).with_graceful_shutdown(async move { sd.cancelled().await }).await;
            if let Err(e) = r {
                tracing::error!("local API server stopped: {e}");
            }
        });
        tracing::info!("local API listening on http://{bound}");
        *self.last_error.lock().unwrap() = None;
        *self.inner.lock().unwrap() = Some(Running { addr: bound, started_at: app_core::now_rfc3339(), shutdown, join });
        Ok(self.status())
    }

    pub async fn stop(&self) {
        let r = self.inner.lock().unwrap().take();
        if let Some(r) = r {
            r.shutdown.cancel();
            // Streaming responses may hold connections; don't wait forever.
            let _ = tokio::time::timeout(Duration::from_secs(5), r.join).await;
        }
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(v1_proxy))
        .route("/v1/completions", post(v1_proxy))
        .route("/api/models", get(api_models))
        .route("/api/models/load", post(api_load))
        .route("/api/models/unload", post(api_unload))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

fn openai_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": { "message": message, "type": kind, "code": status.as_u16() } }))).into_response()
}

async fn not_found() -> Response {
    openai_error(StatusCode::NOT_FOUND, "not_found", "unknown endpoint")
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Origin check, CORS, auth, rate limit and request logging for every request.
async fn guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let started = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let origin = req.headers().get(header::ORIGIN).and_then(|v| v.to_str().ok()).map(str::to_string);
    let allowed_origin = origin.as_ref().filter(|o| st.cfg.cors_origins.iter().any(|a| a == *o)).cloned();

    let mut resp = 'resp: {
        // 1. Browser origin policy.
        if origin.is_some() && allowed_origin.is_none() {
            break 'resp openai_error(StatusCode::FORBIDDEN, "forbidden_origin", "requests from this web origin are not allowed (add it under Developer > CORS origins)");
        }
        if method == Method::OPTIONS {
            break 'resp StatusCode::NO_CONTENT.into_response();
        }
        // 2. Authentication.
        let needs_auth = path.starts_with("/api/") || st.cfg.require_auth || !st.cfg.is_loopback();
        if needs_auth {
            let ok = match (&st.token, bearer(req.headers())) {
                (Some(t), Some(given)) => constant_time_eq(t.as_bytes(), given.as_bytes()),
                _ => false,
            };
            if !ok {
                break 'resp openai_error(StatusCode::UNAUTHORIZED, "unauthorized", "missing or invalid API token (Authorization: Bearer <token>)");
            }
        }
        // 3. Rate limit (fixed one-minute window).
        {
            let mut w = st.rate.lock().unwrap();
            if w.start.elapsed() >= Duration::from_secs(60) {
                w.start = Instant::now();
                w.count = 0;
            }
            w.count += 1;
            if w.count > st.cfg.rate_per_minute {
                break 'resp openai_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited", "too many requests; slow down");
            }
        }
        next.run(req).await
    };

    if let Some(o) = allowed_origin {
        let h = resp.headers_mut();
        if let Ok(v) = HeaderValue::from_str(&o) {
            h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
        }
        h.insert(header::VARY, HeaderValue::from_static("Origin"));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("authorization, content-type"));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
    }

    // Streaming responses log themselves when the stream ends (with token counts).
    let streamed = resp.extensions().get::<StreamLogged>().is_some();
    if !streamed && method != Method::OPTIONS {
        let info = resp.extensions().get::<LogInfo>().cloned().unwrap_or_default();
        let status = resp.status().as_u16();
        let db = st.db.clone();
        let entry = NewApiLog {
            endpoint: path,
            method: method.to_string(),
            status,
            model: info.model,
            latency_ms: started.elapsed().as_millis() as u64,
            prompt_tokens: info.prompt_tokens,
            completion_tokens: info.completion_tokens,
            error: info.error.or_else(|| (status >= 400).then(|| format!("HTTP {status}"))),
        };
        tokio::spawn(async move {
            let _ = db.insert_api_log(&entry).await;
        });
    }
    resp
}

fn bearer(h: &HeaderMap) -> Option<String> {
    h.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ").map(|s| s.trim().to_string())
}

#[derive(Clone, Default)]
struct LogInfo {
    model: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    error: Option<String>,
}

#[derive(Clone)]
struct StreamLogged;

fn with_log(mut r: Response, info: LogInfo) -> Response {
    r.extensions_mut().insert(info);
    r
}

fn loaded_model(st: &AppState) -> Option<(String, inference::LlamaClient)> {
    let s = st.engine.status();
    let client = st.engine.client()?;
    let name = s
        .props
        .as_ref()
        .and_then(|p| p.model_path.clone())
        .map(|p| p.rsplit(['/', '\\']).next().unwrap_or(&p).trim_end_matches(".gguf").to_string())
        .unwrap_or_else(|| "local-model".into());
    Some((name, client))
}

async fn v1_models(State(st): State<AppState>) -> Response {
    let data: Vec<serde_json::Value> = match loaded_model(&st) {
        Some((name, client)) => {
            // Prefer the alias the backend reports.
            let alias = client
                .http()
                .get(format!("{}/v1/models", client.base_url()))
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .ok();
            let id = match alias {
                Some(r) => r.json::<serde_json::Value>().await.ok().and_then(|v| v["data"][0]["id"].as_str().map(str::to_string)).unwrap_or(name),
                None => name,
            };
            vec![serde_json::json!({ "id": id, "object": "model", "created": 0, "owned_by": "local-ai-studio" })]
        }
        None => vec![],
    };
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

async fn v1_proxy(State(st): State<AppState>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let body = match axum::body::to_bytes(req.into_body(), MAX_BODY).await {
        Ok(b) => b,
        Err(_) => return openai_error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "request body too large"),
    };
    let mut json: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v @ serde_json::Value::Object(_)) => v,
        _ => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be a JSON object"),
    };
    let Some((name, client)) = loaded_model(&st) else {
        return openai_error(StatusCode::SERVICE_UNAVAILABLE, "model_not_loaded", "no model is loaded; load one in Local AI Studio or via POST /api/models/load");
    };
    if path.ends_with("/chat/completions") && !json["messages"].is_array() {
        return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "'messages' must be an array");
    }
    if path.ends_with("/completions") && !path.contains("chat") && json.get("prompt").is_none() {
        return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "'prompt' is required");
    }
    let stream = json["stream"].as_bool().unwrap_or(false);
    // One model is loaded in V1; whatever name the client sent, it runs on it.
    json["model"] = serde_json::Value::String(name.clone());

    let permit = match tokio::time::timeout(Duration::from_secs(120), st.gen_slots.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return openai_error(StatusCode::TOO_MANY_REQUESTS, "busy", "all generation slots are busy; try again"),
    };
    let upstream = client
        .http()
        .post(format!("{}{}", client.base_url(), path))
        .json(&json)
        .timeout(Duration::from_secs(3600))
        .send()
        .await;
    let upstream = match upstream {
        Ok(r) => r,
        Err(e) => return openai_error(StatusCode::BAD_GATEWAY, "backend_error", &format!("model backend unreachable: {e}")),
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    if !stream || !status.is_success() {
        let bytes = upstream.bytes().await.unwrap_or_default();
        drop(permit);
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
        let info = LogInfo {
            model: Some(name),
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64(),
            completion_tokens: v["usage"]["completion_tokens"].as_u64(),
            error: v["error"]["message"].as_str().map(str::to_string),
        };
        let mut r = Response::new(Body::from(bytes));
        *r.status_mut() = status;
        r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        return with_log(r, info);
    }

    // Streaming: pass bytes through unchanged; watch for usage to log at the end.
    let db = st.db.clone();
    let started = Instant::now();
    let path_s = path.clone();
    let mut parser = inference::chat::SseParser::default();
    let mut prompt_tokens = None;
    let mut completion_tokens = None;
    let mut inner = upstream.bytes_stream();
    let s = async_stream_bytes(move |tx| async move {
        let _permit = permit; // held until the stream ends or the client disconnects
        let mut finished = false;
        while let Some(chunk) = inner.next().await {
            match chunk {
                Ok(b) => {
                    for data in parser.feed(&b) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data)
                            && v["usage"].is_object()
                        {
                            prompt_tokens = v["usage"]["prompt_tokens"].as_u64();
                            completion_tokens = v["usage"]["completion_tokens"].as_u64();
                        }
                        if data.trim() == "[DONE]" {
                            finished = true;
                        }
                    }
                    if tx.send(Ok(b)).await.is_err() {
                        break; // client went away: dropping `inner` stops generation
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }
        let entry = NewApiLog {
            endpoint: path_s,
            method: "POST".into(),
            status: 200,
            model: Some(name),
            latency_ms: started.elapsed().as_millis() as u64,
            prompt_tokens,
            completion_tokens,
            error: (!finished).then(|| "stream ended early (client disconnected or backend error)".to_string()),
        };
        let _ = db.insert_api_log(&entry).await;
    });
    let mut r = Response::new(Body::from_stream(s));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r.extensions_mut().insert(StreamLogged);
    r
}

/// Turn a producer task into a byte stream backed by a bounded channel.
fn async_stream_bytes<F, Fut>(f: F) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>>
where
    F: FnOnce(tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(f(tx));
    tokio_stream_from(rx)
}

fn tokio_stream_from(
    mut rx: tokio::sync::mpsc::Receiver<Result<Bytes, std::io::Error>>,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>> {
    futures::stream::poll_fn(move |cx| rx.poll_recv(cx))
}

fn app_error(e: AppError) -> Response {
    let status = match e {
        AppError::NotFound(_) => StatusCode::NOT_FOUND,
        AppError::InvalidInput(_) => StatusCode::BAD_REQUEST,
        AppError::Conflict(_) => StatusCode::CONFLICT,
        AppError::PermissionDenied(_) => StatusCode::FORBIDDEN,
        AppError::OutOfMemory(_) => StatusCode::INSUFFICIENT_STORAGE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let msg = e.to_string();
    with_log(openai_error(status, e.code(), &msg), LogInfo { error: Some(msg), ..Default::default() })
}

async fn api_models(State(st): State<AppState>) -> Response {
    match st.ops.list_models().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => app_error(e),
    }
}

async fn api_load(State(st): State<AppState>, body: Option<Json<serde_json::Value>>) -> Response {
    let body = body.map(|Json(v)| v).unwrap_or_default();
    match st.ops.load_model(body).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => app_error(e),
    }
}

async fn api_unload(State(st): State<AppState>) -> Response {
    match st.ops.unload_model().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => app_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoOps;
    #[async_trait]
    impl NativeOps for NoOps {
        async fn list_models(&self) -> AppResult<serde_json::Value> {
            Ok(serde_json::json!([{"id": "m1"}]))
        }
        async fn load_model(&self, _b: serde_json::Value) -> AppResult<serde_json::Value> {
            Err(AppError::NotFound("no such model".into()))
        }
        async fn unload_model(&self) -> AppResult<serde_json::Value> {
            Ok(serde_json::json!({"state": "unloaded"}))
        }
    }

    async fn start(cfg: ApiConfig, token: Option<&str>) -> (ApiServer, String, Db) {
        let db = Db::open_in_memory().await.unwrap();
        let s = ApiServer::default();
        let st = s.start(cfg, token.map(str::to_string), Engine::new(), db.clone(), Arc::new(NoOps)).await.unwrap();
        (s, st.address.unwrap(), db)
    }

    fn cfg() -> ApiConfig {
        ApiConfig { port: free_port(), ..Default::default() }
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn openai_shapes_auth_origin_and_logging() {
        let (s, base, db) = start(cfg(), Some("secret-token-123")).await;
        let http = reqwest::Client::builder().no_proxy().build().unwrap();
        // No model loaded: empty list, and chat returns a clear 503.
        let v: serde_json::Value = http.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"].as_array().unwrap().len(), 0);
        let r = http.post(format!("{base}/v1/chat/completions")).json(&serde_json::json!({"messages": []})).send().await.unwrap();
        assert_eq!(r.status(), 503);
        let e: serde_json::Value = r.json().await.unwrap();
        assert_eq!(e["error"]["type"], "model_not_loaded");
        // Bad body
        let r = http.post(format!("{base}/v1/chat/completions")).body("not json").send().await.unwrap();
        assert_eq!(r.status(), 400);
        // Native API needs the token even on loopback.
        assert_eq!(http.get(format!("{base}/api/models")).send().await.unwrap().status(), 401);
        assert_eq!(http.get(format!("{base}/api/models")).bearer_auth("wrong").send().await.unwrap().status(), 401);
        let ok = http.get(format!("{base}/api/models")).bearer_auth("secret-token-123").send().await.unwrap();
        assert_eq!(ok.status(), 200);
        // Errors from the app map to HTTP codes.
        let r = http.post(format!("{base}/api/models/load")).bearer_auth("secret-token-123").json(&serde_json::json!({})).send().await.unwrap();
        assert_eq!(r.status(), 404);
        // A browser page on another origin is refused, even for /v1.
        let r = http.post(format!("{base}/v1/chat/completions")).header("Origin", "https://evil.example").body("{}").send().await.unwrap();
        assert_eq!(r.status(), 403);
        assert!(r.headers().get("access-control-allow-origin").is_none());
        assert_eq!(http.get(format!("{base}/nope")).send().await.unwrap().status(), 404);
        tokio::time::sleep(Duration::from_millis(200)).await;
        let logs = db.recent_api_logs(50).await.unwrap();
        assert!(logs.len() >= 8, "{}", logs.len());
        assert!(logs.iter().any(|l| l.endpoint == "/api/models" && l.status == 401));
        s.stop().await;
        assert!(!s.status().running);
    }

    #[tokio::test]
    async fn allowed_origin_gets_cors_headers_and_token_on_v1_when_required() {
        let mut c = cfg();
        c.cors_origins = vec!["http://localhost:3000".into()];
        c.require_auth = true;
        let (s, base, _db) = start(c, Some("tok-abcdefgh")).await;
        let http = reqwest::Client::builder().no_proxy().build().unwrap();
        let r = http.get(format!("{base}/v1/models")).header("Origin", "http://localhost:3000").send().await.unwrap();
        assert_eq!(r.status(), 401, "auth required on /v1 when enabled");
        let r = http
            .get(format!("{base}/v1/models"))
            .header("Origin", "http://localhost:3000")
            .bearer_auth("tok-abcdefgh")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        assert_eq!(r.headers()["access-control-allow-origin"], "http://localhost:3000");
        let pre = http.request(Method::OPTIONS, format!("{base}/v1/chat/completions")).header("Origin", "http://localhost:3000").send().await.unwrap();
        assert_eq!(pre.status(), 204);
        s.stop().await;
    }

    #[tokio::test]
    async fn port_conflict_and_unsafe_configs_are_refused() {
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = busy.local_addr().unwrap().port();
        let s = ApiServer::default();
        let e = s
            .start(ApiConfig { port, ..Default::default() }, None, Engine::new(), Db::open_in_memory().await.unwrap(), Arc::new(NoOps))
            .await
            .unwrap_err();
        assert!(matches!(e, AppError::Conflict(_)), "{e}");
        assert!(e.to_string().contains("already in use"));
        assert!(ApiConfig { host: "0.0.0.0".into(), ..Default::default() }.validate().is_err(), "LAN without auth");
        assert!(ApiConfig { host: "0.0.0.0".into(), require_auth: true, ..Default::default() }.validate().is_ok());
        assert!(ApiConfig { cors_origins: vec!["*".into()], ..Default::default() }.validate().is_err());
        let e = s
            .start(ApiConfig { port: free_port(), require_auth: true, ..Default::default() }, None, Engine::new(), Db::open_in_memory().await.unwrap(), Arc::new(NoOps))
            .await
            .unwrap_err();
        assert!(e.to_string().contains("no API token"));
    }
}

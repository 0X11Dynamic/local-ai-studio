//! The `web_search` tool in the app: settings (Settings > Search), keys from
//! the Credential Manager, and running one search for a chat reply or task.
//! The provider code is in `execution::web_search`.

use app_core::{AppError, AppResult};
use execution::web_search::{self as ws, ProviderKind, SearchConfig, SearchRequest, SearchService};
use serde::{Deserialize, Serialize};

use crate::state::SharedState;

pub const SETTING_KEY: &str = "search.config";

/// Settings > Search. No secret in here: keys live in the Credential Manager
/// (`search/langsearch`, `search/brave`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SearchSettings {
    pub provider: ProviderKind,
    /// SearXNG instance address.
    pub searxng_url: String,
    /// LangSearch: return page text instead of a snippet.
    pub include_text: bool,
    /// Default number of results (1-10).
    pub count: u32,
    /// Searches allowed per chat reply or task.
    pub cap: u32,
}

impl Default for SearchSettings {
    fn default() -> Self {
        Self { provider: ProviderKind::DuckDuckGo, searxng_url: String::new(), include_text: false, count: ws::DEFAULT_COUNT, cap: ws::DEFAULT_CAP }
    }
}

impl SearchSettings {
    pub fn clamped(mut self) -> Self {
        self.count = self.count.clamp(1, 10);
        self.cap = self.cap.clamp(1, 200);
        self.searxng_url = self.searxng_url.trim().to_string();
        self
    }
}

pub async fn load(st: &SharedState) -> SearchSettings {
    st.db.get_setting::<SearchSettings>(SETTING_KEY).await.ok().flatten().unwrap_or_default().clamped()
}

pub async fn save(st: &SharedState, s: SearchSettings) -> AppResult<SearchSettings> {
    let s = s.clamped();
    if s.provider == ProviderKind::SearXng && !s.searxng_url.is_empty() {
        // Same checks as when searching, so a bad address is caught here.
        ws::build_provider(&SearchConfig { provider: s.provider, searxng_url: s.searxng_url.clone(), ..Default::default() }).map(|_| ())?;
    }
    st.db.set_setting(SETTING_KEY, &s).await?;
    Ok(s)
}

/// The provider settings plus the stored key (if the provider uses one).
pub fn config(st: &SharedState, s: &SearchSettings) -> AppResult<SearchConfig> {
    let api_key = match s.provider.credential_ref() {
        Some(r) => st.credentials.get(r)?,
        None => None,
    };
    Ok(SearchConfig { provider: s.provider, api_key, searxng_url: s.searxng_url.clone(), include_text: s.include_text, endpoint: None })
}

/// Rough token cost of the web_search definition sent to the model.
pub fn tool_tokens() -> u64 {
    let m = crate::services::exec::EnvMeta { kind: "local".into(), label: String::new(), shell: execution::Shell::PowerShell, roots: vec![] };
    let def = serde_json::json!({ "name": "web_search", "description": crate::services::exec::describe("web_search", &m), "parameters": crate::services::exec::schema("web_search") });
    def.to_string().len().div_ceil(4) as u64
}

/// Which provider keys are stored (never the keys).
pub fn key_status(st: &SharedState) -> AppResult<(bool, bool)> {
    let has = |k: ProviderKind| -> AppResult<bool> { Ok(k.credential_ref().map(|r| st.credentials.get(r)).transpose()?.flatten().is_some()) };
    Ok((has(ProviderKind::LangSearch)?, has(ProviderKind::Brave)?))
}

pub fn set_key(st: &SharedState, provider: ProviderKind, key: Option<String>) -> AppResult<()> {
    let r = provider.credential_ref().ok_or_else(|| AppError::InvalidInput(format!("{} does not use an API key", provider.label())))?;
    match key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty()) {
        Some(k) => {
            if k.len() < 8 || k.len() > 300 || k.chars().any(char::is_whitespace) {
                return Err(AppError::InvalidInput("that does not look like an API key (no spaces, 8-300 characters)".into()));
            }
            st.credentials.set(r, &k)
        }
        None => st.credentials.delete(r),
    }
}

/// Run one `web_search` call. `scope` is the reply or task the per-reply cap
/// counts against.
pub async fn run(st: &SharedState, scope: &str, args: &serde_json::Value) -> AppResult<execution::web_search::SearchResponse> {
    let settings = load(st).await;
    let req = SearchRequest::from_args(args, settings.count)?;
    let cfg = config(st, &settings)?;
    SearchService::global().search(Some((scope, settings.cap)), &cfg, &req).await
}

/// The Test button: one real query with the saved settings (not counted
/// against any reply's cap).
pub async fn test(st: &SharedState, query: &str) -> AppResult<execution::web_search::SearchResponse> {
    let settings = load(st).await;
    let req = SearchRequest::from_args(&serde_json::json!({ "query": if query.trim().is_empty() { "rust programming language" } else { query }, "count": 3 }), 3)?;
    let cfg = config(st, &settings)?;
    SearchService::global().search(None, &cfg, &req).await
}

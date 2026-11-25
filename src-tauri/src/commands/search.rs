//! Settings > Search: the provider, its key (Credential Manager only), the
//! limits, and the Test button.

use execution::web_search::{ProviderKind, SearchResult};
use serde::Serialize;
use tauri::State;

use super::Cmd;
use crate::services::websearch::{self, SearchSettings};
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct SearchSettingsView {
    pub settings: SearchSettings,
    /// Whether a key is stored. The keys themselves are never sent to the UI.
    pub has_langsearch_key: bool,
    pub has_brave_key: bool,
    /// About how many tokens the tool definition adds to a request (4 characters per token).
    pub tool_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct SearchTestView {
    pub ok: bool,
    pub provider: String,
    pub results: Vec<SearchResult>,
    pub error: Option<String>,
}

async fn view(st: &SharedState) -> Cmd<SearchSettingsView> {
    let (has_langsearch_key, has_brave_key) = websearch::key_status(st)?;
    Ok(SearchSettingsView { settings: websearch::load(st).await, has_langsearch_key, has_brave_key, tool_tokens: websearch::tool_tokens() })
}

#[tauri::command]
pub async fn get_search_settings(state: State<'_, SharedState>) -> Cmd<SearchSettingsView> {
    view(state.inner()).await
}

#[tauri::command]
pub async fn set_search_settings(state: State<'_, SharedState>, settings: SearchSettings) -> Cmd<SearchSettingsView> {
    websearch::save(state.inner(), settings).await?;
    view(state.inner()).await
}

/// Store (or, with no key, remove) the LangSearch / Brave key.
#[tauri::command]
pub async fn set_search_key(state: State<'_, SharedState>, provider: ProviderKind, key: Option<String>) -> Cmd<SearchSettingsView> {
    websearch::set_key(state.inner(), provider, key)?;
    view(state.inner()).await
}

/// Run one query with the saved settings and return the first results or the error.
#[tauri::command]
pub async fn test_web_search(state: State<'_, SharedState>, query: Option<String>) -> Cmd<SearchTestView> {
    let st = state.inner().clone();
    let provider = websearch::load(&st).await.provider.label().to_string();
    Ok(match websearch::test(&st, query.as_deref().unwrap_or("")).await {
        Ok(r) => SearchTestView { ok: true, provider: r.provider, results: r.results, error: None },
        Err(e) => SearchTestView { ok: false, provider, results: vec![], error: Some(e.to_string()) },
    })
}

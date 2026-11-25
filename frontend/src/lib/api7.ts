// Web search settings (Settings > Search). API keys are written with
// setSearchKey and never read back: the backend only says whether one is stored.
import { invoke } from "@tauri-apps/api/core";

export type SearchProvider = "duckduckgo" | "langsearch" | "brave" | "searxng";

export interface SearchSettings {
  provider: SearchProvider;
  searxng_url: string;
  /** LangSearch: return page text instead of a snippet. */
  include_text: boolean;
  /** Default number of results, 1-10. */
  count: number;
  /** Searches allowed per chat reply or task. */
  cap: number;
}

export interface SearchSettingsView {
  settings: SearchSettings;
  has_langsearch_key: boolean;
  has_brave_key: boolean;
  /** About how many tokens the web_search definition adds to a request. */
  tool_tokens: number;
}

export interface SearchResultView { title: string; url: string; snippet: string; date: string | null }
export interface SearchTestView { ok: boolean; provider: string; results: SearchResultView[]; error: string | null }

export const api7 = {
  getSearchSettings: () => invoke<SearchSettingsView>("get_search_settings"),
  setSearchSettings: (settings: SearchSettings) => invoke<SearchSettingsView>("set_search_settings", { settings }),
  setSearchKey: (provider: "langsearch" | "brave", key: string | null) => invoke<SearchSettingsView>("set_search_key", { provider, key }),
  testWebSearch: (query?: string) => invoke<SearchTestView>("test_web_search", { query: query ?? null }),
};

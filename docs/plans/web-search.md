# Plan: web search tool

Design for a built-in `web_search` tool that the model can call in Agent and Plan modes (status: designed 2026-11-12).

## Decisions
- **Providers, picked per user:** DuckDuckGo (default: no key, works out of the box, unofficial HTML endpoint, may be rate-limited), LangSearch (free key from https://langsearch.com/api-keys), Brave Search API (free key, official), SearXNG (self-hosted, most private). Settings > Search to switch, paste a key, and test.
- **Approval:** runs without asking (like `web_fetch` on public sites). The policy can still be changed to "ask" and the tool unticked per chat.
- **Default:** on for new Agent and Plan mode chats and tasks; untick it in the chat Tools menu or the task Tools picker.

## The tool
`web_search { query: string, count?: 1-10 (default 5), site?: string, freshness?: "day" | "week" | "month" | "year" }`
- Returns a numbered list: title, URL, snippet (and date when the provider gives one), plus which provider answered. Short on purpose: the model reads pages it needs with `web_fetch`.
- Offered as a built-in tool in chat Agent and Plan modes (read-only, so Plan mode allows it), Agent Tasks and subagents; runs in the app (not in the execution environment), like `web_fetch`.
- Results and fetched pages are untrusted data (same wrapper as other tool output).

## Providers
- **DuckDuckGo:** POST/GET `https://html.duckduckgo.com/html/` with `q` (and `kl`/`df` for region/freshness); parse result titles, links (decode the `uddg` redirect) and snippets. No key. Handle the "anomaly"/captcha page with a clear error that suggests Brave or SearXNG. `site:` goes into the query.
- **LangSearch:** `POST https://api.langsearch.com/v1/web-search`, headers `Authorization: Bearer <key>` and `Content-Type: application/json`. Body: `query` (required), `count` (1-50, default 10; send our count), `freshness` (`noLimit` | `oneDay` | `oneWeek` | `oneMonth` | `oneYear` | `YYYY-MM-DD` | `YYYY-MM-DD..YYYY-MM-DD`), `includeDomains` / `excludeDomains` (arrays; our `site` maps to `includeDomains`), optional `contents.text` (`true` or `{ "maxCharacters": n }`) for full page text instead of a snippet. Response: results at `data.webPages.value[]` with `name` (title), `url`, `displayUrl`, `snippet` or `text`, `datePublished`; plus `usage.input_tokens` / `output_tokens` and `log_id`. Errors: 400/401/403, 429 = rate limit or daily allowance (resets 00:00 UTC; honour `Retry-After`, don't retry a daily-quota error), 5xx = retry with backoff. Free plan: $0/month, 5 requests/s for new accounts, token-based daily allowance (number not documented; the dashboard shows it). Key in Credential Manager (`search/langsearch`), used only from the Rust backend (the docs say not to expose keys in a browser bundle). Option in Settings: "Include page text" (uses LangSearch's `contents.text`, so the model may not need web_fetch).
- **Brave Search API:** `GET https://api.search.brave.com/res/v1/web/search?q=...&count=...&freshness=pd|pw|pm|py` with header `X-Subscription-Token`. Key in Windows Credential Manager (`search/brave`). Show quota/rate-limit errors plainly.
- **SearXNG:** `GET <base>/search?q=...&format=json&time_range=day|week|month|year` (needs the instance's JSON format enabled; the Test button checks it). Base URL in settings; local addresses allowed for this one configured host only.

## Safety and limits
- Keys only in Credential Manager; never in the DB, logs or exports.
- Cache identical (provider, query, options) results for 10 minutes.
- Rate limit: at most 1 request/second per provider, and a per-reply/per-task cap (default 20 searches) so a looping model can't burn a quota; the cap is a setting.
- Timeouts 15 s; no cookies; plain User-Agent `LocalAIStudio/0.1 (+web_search)`.
- The prompt section tells the model to search, then read 1-3 pages with web_fetch, then cite the URLs it used.

## UI
- Settings > Search: provider select, key field (Brave) or base URL (SearXNG), Test button (runs one query and shows the first results or the error), result count and per-reply cap.
- Tool cards show the query and the result list (titles as links).
- The chat Tools menu and the task Tools picker list "Web search" with its token cost.

## Tests
- Unit: parsers for each provider from saved fixtures (DuckDuckGo HTML, LangSearch JSON, Brave JSON, SearXNG JSON), `uddg` link decoding, captcha detection, cache and rate limiter, cap per reply, key never logged.
- e2e: a Settings > Search test against a local fixture server standing in for each provider; and one real check with a model answering a question that needs a fresh web result (skipped when offline).

## Not in scope (for now)
- Paid providers (Google, Bing). Nothing paid is enabled by default.
- Image or news search.

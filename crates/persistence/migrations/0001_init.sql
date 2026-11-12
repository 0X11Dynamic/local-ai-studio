-- Local AI Studio schema v1 (spec section 12).
-- Times are RFC 3339 UTC text. JSON columns hold serde_json text.
-- Secrets are never stored here: only credential references into Windows Credential Manager.

CREATE TABLE app_settings (
    key   TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE models (
    id            TEXT PRIMARY KEY NOT NULL,
    repo_id       TEXT,                       -- Hugging Face repo, if known
    path          TEXT NOT NULL UNIQUE,       -- first (or only) GGUF file
    format        TEXT NOT NULL DEFAULT 'gguf',
    quantization  TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}', -- parsed GGUF header subset (JSON)
    size          INTEGER NOT NULL DEFAULT 0, -- total bytes across all shards
    display_name  TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'import', -- import | lmstudio | download
    shard_paths   TEXT NOT NULL DEFAULT '[]', -- JSON array of all shard paths
    projector_path TEXT,                      -- mmproj file next to the model, if any
    default_config TEXT,                      -- per-model default load config (JSON)
    last_config   TEXT,                       -- last-used load config (JSON)
    status        TEXT NOT NULL DEFAULT 'ok', -- ok | missing | invalid
    status_detail TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE model_instances (
    id         TEXT PRIMARY KEY NOT NULL,
    model_id   TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    backend    TEXT NOT NULL,        -- runtime id, e.g. b12246-cuda-13.5
    config     TEXT NOT NULL,        -- effective load config (JSON)
    state      TEXT NOT NULL,        -- loading | loaded | failed | unloaded
    error      TEXT,
    started_at TEXT NOT NULL,
    loaded_at  TEXT,
    ended_at   TEXT
);

CREATE TABLE conversations (
    id         TEXT PRIMARY KEY NOT NULL,
    title      TEXT NOT NULL,
    model_id   TEXT REFERENCES models(id) ON DELETE SET NULL,
    settings   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    reasoning       TEXT,
    tool_calls      TEXT,
    tool_call_id    TEXT,
    stats           TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);

CREATE TABLE downloads (
    id          TEXT PRIMARY KEY NOT NULL,
    source      TEXT NOT NULL,   -- JSON: repo, files, urls
    destination TEXT NOT NULL,
    progress    TEXT NOT NULL DEFAULT '{}',
    state       TEXT NOT NULL,   -- queued | running | paused | completed | failed | cancelled
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE presets (
    id            TEXT PRIMARY KEY NOT NULL,
    title         TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'load', -- load | instruction
    settings_json TEXT NOT NULL DEFAULT '{}',
    system_prompt TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE mcp_servers (
    id               TEXT PRIMARY KEY NOT NULL,
    name             TEXT NOT NULL UNIQUE,
    transport        TEXT NOT NULL,  -- stdio | http
    nonsecret_config TEXT NOT NULL,
    credential_ref   TEXT,
    enabled          INTEGER NOT NULL DEFAULT 1,
    status           TEXT NOT NULL DEFAULT 'disconnected',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE mcp_tool_preferences (
    id        TEXT PRIMARY KEY NOT NULL,
    server_id TEXT NOT NULL REFERENCES mcp_servers(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    enabled   INTEGER NOT NULL DEFAULT 1,
    policy    TEXT NOT NULL DEFAULT 'ask', -- allow | ask | deny
    UNIQUE (server_id, tool_name)
);

CREATE TABLE execution_profiles (
    id               TEXT PRIMARY KEY NOT NULL,
    type             TEXT NOT NULL,  -- local | ssh | docker | sandbox
    name             TEXT NOT NULL UNIQUE,
    nonsecret_config TEXT NOT NULL,
    credential_ref   TEXT,
    last_health      TEXT,
    last_health_at   TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE agent_runs (
    id          TEXT PRIMARY KEY NOT NULL,
    objective   TEXT NOT NULL,
    model_id    TEXT REFERENCES models(id) ON DELETE SET NULL,
    env_id      TEXT REFERENCES execution_profiles(id) ON DELETE SET NULL,
    state       TEXT NOT NULL,
    budgets     TEXT NOT NULL DEFAULT '{}',
    usage       TEXT NOT NULL DEFAULT '{}',
    final_answer TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    ended_at    TEXT
);

CREATE TABLE subagents (
    id              TEXT PRIMARY KEY NOT NULL,
    parent_run_id   TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    parent_agent_id TEXT,
    objective       TEXT NOT NULL,
    state           TEXT NOT NULL,
    budgets         TEXT NOT NULL DEFAULT '{}',
    usage           TEXT NOT NULL DEFAULT '{}',
    result          TEXT,
    created_at      TEXT NOT NULL,
    ended_at        TEXT
);

CREATE TABLE agent_steps (
    id              TEXT PRIMARY KEY NOT NULL,
    run_id          TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    parent_agent_id TEXT,            -- NULL for the root agent, else subagents.id
    seq             INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    data            TEXT NOT NULL,
    state           TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT
);
CREATE INDEX idx_agent_steps_run ON agent_steps(run_id, seq);

CREATE TABLE tool_executions (
    id            TEXT PRIMARY KEY NOT NULL,
    run_id        TEXT,              -- NULL for chat-mode tool calls
    agent_id      TEXT,
    conversation_id TEXT,
    tool          TEXT NOT NULL,
    source        TEXT NOT NULL,     -- builtin | mcp:<server>
    environment_id TEXT,
    args_redacted TEXT NOT NULL,
    result        TEXT,
    status        TEXT NOT NULL,
    approval      TEXT NOT NULL,     -- JSON PermissionDecision + who approved
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    duration_ms   INTEGER
);
CREATE INDEX idx_tool_exec_run ON tool_executions(run_id);

CREATE TABLE skills (
    id         TEXT PRIMARY KEY NOT NULL,
    path       TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL UNIQUE,
    version    TEXT NOT NULL,
    manifest   TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE skill_revisions (
    id           TEXT PRIMARY KEY NOT NULL,
    skill_id     TEXT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content      TEXT NOT NULL,
    note         TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE permission_grants (
    id         TEXT PRIMARY KEY NOT NULL,
    scope      TEXT NOT NULL,       -- once | session | persistent
    subject    TEXT NOT NULL,       -- what the grant applies to (tool + pattern)
    capability TEXT NOT NULL,
    environment_id TEXT,
    expiry     TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE api_request_logs (
    id         TEXT PRIMARY KEY NOT NULL,
    endpoint   TEXT NOT NULL,
    method     TEXT NOT NULL,
    status     INTEGER NOT NULL,
    model      TEXT,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_api_logs_time ON api_request_logs(created_at);

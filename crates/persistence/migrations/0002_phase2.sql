-- Phase 2: downloads lookup columns, instruction profiles, per-model instruction override.

ALTER TABLE downloads ADD COLUMN repo_id TEXT;
ALTER TABLE downloads ADD COLUMN variant_key TEXT;
CREATE INDEX idx_downloads_repo ON downloads(repo_id, variant_key);

-- presets(kind='instruction') rows describe System Instruction profiles.
ALTER TABLE presets ADD COLUMN builtin INTEGER NOT NULL DEFAULT 0;
-- JSON: { "mode": "sections" | "raw", "identity", "task", "tool_use", "formatting", "raw" }
ALTER TABLE presets ADD COLUMN sections TEXT NOT NULL DEFAULT '{}';

ALTER TABLE models ADD COLUMN instruction_profile_id TEXT REFERENCES presets(id) ON DELETE SET NULL;

ALTER TABLE messages ADD COLUMN model_name TEXT;
CREATE INDEX idx_conversations_updated ON conversations(updated_at);

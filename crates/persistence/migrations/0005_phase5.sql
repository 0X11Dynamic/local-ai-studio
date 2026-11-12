-- Phase 5: agent tasks, subagents, skills.

-- Task settings (environment, tool servers, skills, subagents, tool mode) and
-- the saved transcript so a paused or interrupted task can resume. The
-- environment lives in settings because "This PC" has no execution_profiles row.
ALTER TABLE agent_runs ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_runs ADD COLUMN settings TEXT NOT NULL DEFAULT '{}';
ALTER TABLE agent_runs ADD COLUMN transcript TEXT NOT NULL DEFAULT '[]';
CREATE INDEX idx_runs_created ON agent_runs(created_at);

ALTER TABLE subagents ADD COLUMN label TEXT NOT NULL DEFAULT '';
ALTER TABLE subagents ADD COLUMN settings TEXT NOT NULL DEFAULT '{}';
ALTER TABLE subagents ADD COLUMN transcript TEXT NOT NULL DEFAULT '[]';
ALTER TABLE subagents ADD COLUMN error TEXT;
ALTER TABLE subagents ADD COLUMN started_at TEXT;
CREATE INDEX idx_subagents_parent ON subagents(parent_run_id);

-- "Allow for this task" grants (cover the task's subagents too).
ALTER TABLE permission_grants ADD COLUMN run_id TEXT REFERENCES agent_runs(id) ON DELETE CASCADE;

-- auto | always | off
ALTER TABLE skills ADD COLUMN mode TEXT NOT NULL DEFAULT 'auto';
-- user | import | builtin
ALTER TABLE skills ADD COLUMN source TEXT NOT NULL DEFAULT 'user';
-- {"relative/path": "sha256 the user enabled"}; anything not listed is off.
ALTER TABLE skills ADD COLUMN scripts TEXT NOT NULL DEFAULT '{}';

-- applied | pending | rejected; pending revisions are agent proposals.
ALTER TABLE skill_revisions ADD COLUMN state TEXT NOT NULL DEFAULT 'applied';
ALTER TABLE skill_revisions ADD COLUMN proposed_by TEXT;

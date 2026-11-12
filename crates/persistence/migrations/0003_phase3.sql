-- Phase 3: MCP and tool permissions.

-- Session grants belong to one conversation.
ALTER TABLE permission_grants ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE;
CREATE INDEX idx_grants_subject ON permission_grants(subject);

-- Model-facing tool name and the tool-call id from the model, for the audit view.
ALTER TABLE tool_executions ADD COLUMN tool_call_id TEXT;
ALTER TABLE tool_executions ADD COLUMN model_tool_name TEXT;
CREATE INDEX idx_tool_exec_conv ON tool_executions(conversation_id, started_at);
CREATE INDEX idx_tool_exec_time ON tool_executions(started_at);

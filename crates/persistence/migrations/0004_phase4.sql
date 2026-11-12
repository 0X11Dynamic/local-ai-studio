-- Phase 4: execution environments and built-in tools.

-- Per-environment preferences for the built-in shell and file tools.
CREATE TABLE builtin_tool_preferences (
    env_id    TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    enabled   INTEGER NOT NULL DEFAULT 1,
    policy    TEXT NOT NULL, -- allow | ask | deny
    PRIMARY KEY (env_id, tool_name)
);

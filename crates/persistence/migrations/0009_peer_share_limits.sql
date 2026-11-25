-- What THIS PC lends to that linked PC when it helps it (enforced on this
-- PC, in the tunnel to its rpc-server). NULL = the default from the helper
-- settings ("Default limits for linked PCs").
ALTER TABLE linked_peers ADD COLUMN share_gpu_mib INTEGER;  -- 0 = no GPU memory
ALTER TABLE linked_peers ADD COLUMN share_ram_mib INTEGER;  -- 0 = no RAM
ALTER TABLE linked_peers ADD COLUMN share_threads INTEGER;
-- "Pause sharing with <name>": new sessions are refused; an open one is never cut.
ALTER TABLE linked_peers ADD COLUMN share_paused INTEGER NOT NULL DEFAULT 0;

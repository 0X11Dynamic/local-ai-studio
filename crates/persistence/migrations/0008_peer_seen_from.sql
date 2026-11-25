-- Where a PC that linked to this one connected from (its IP as seen here,
-- e.g. 10.0.2.2 behind VirtualBox NAT). Shown only; never dialled.
ALTER TABLE linked_peers ADD COLUMN seen_from TEXT;

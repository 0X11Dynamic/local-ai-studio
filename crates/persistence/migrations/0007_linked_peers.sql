-- Linked PCs (pairing with a one-time code; keys pinned on both sides).
CREATE TABLE linked_peers (
    id                  TEXT PRIMARY KEY NOT NULL,
    name                TEXT NOT NULL,
    -- The other PC's Noise static public key (base64), pinned at pairing.
    public_key          TEXT NOT NULL UNIQUE,
    -- host:port to dial (its link port), when known.
    last_address        TEXT,
    -- 1 when it linked from a local-network address.
    same_network        INTEGER NOT NULL DEFAULT 1,
    -- "Split model layers with this PC" (default: on for the same network).
    split_layers        INTEGER NOT NULL DEFAULT 1,
    -- "Send whole jobs" (Option B, a later phase): stored only, not used yet.
    send_jobs           INTEGER NOT NULL DEFAULT 0,
    -- 1: this PC typed the code (it dials the other); 0: the other PC linked to this one.
    initiated           INTEGER NOT NULL DEFAULT 1,
    last_rtt_ms         REAL,
    last_bandwidth_mbps REAL,
    runtime_tag         TEXT,
    last_seen_at        TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

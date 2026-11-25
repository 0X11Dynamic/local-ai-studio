//! Linked PCs: pinned public keys and per-PC settings, plus the reply speeds
//! the load panel compares (this PC alone vs split with linked PCs).

use app_core::{AppError, AppResult, new_id, now_rfc3339};
use serde::{Deserialize, Serialize};

use crate::{Db, db_err};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PeerRow {
    pub id: String,
    pub name: String,
    /// Pinned Noise static key (base64).
    pub public_key: String,
    pub last_address: Option<String>,
    pub same_network: i64,
    pub split_layers: i64,
    /// Option B ("send whole jobs"): stored for a later phase.
    pub send_jobs: i64,
    pub initiated: i64,
    pub last_rtt_ms: Option<f64>,
    pub last_bandwidth_mbps: Option<f64>,
    pub runtime_tag: Option<String>,
    pub last_seen_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// IP it connected from (for PCs that linked to this one). Shown only.
    pub seen_from: Option<String>,
    /// What this PC lends to that PC (None = the default). Enforced on this PC.
    pub share_gpu_mib: Option<i64>,
    pub share_ram_mib: Option<i64>,
    pub share_threads: Option<i64>,
    /// "Pause sharing with <name>": new sessions refused, open ones kept.
    pub share_paused: i64,
}

/// Per-linked-PC sharing limits as stored (None = use the default).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PeerShare {
    pub gpu_mib: Option<u64>,
    pub ram_mib: Option<u64>,
    pub threads: Option<u32>,
    pub paused: bool,
}

impl PeerRow {
    pub fn share(&self) -> PeerShare {
        PeerShare {
            gpu_mib: self.share_gpu_mib.map(|v| v.max(0) as u64),
            ram_mib: self.share_ram_mib.map(|v| v.max(0) as u64),
            threads: self.share_threads.map(|v| v.clamp(1, 512) as u32),
            paused: self.share_paused != 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewPeer<'a> {
    pub name: &'a str,
    pub public_key: &'a str,
    pub last_address: Option<&'a str>,
    pub same_network: bool,
    pub initiated: bool,
    pub runtime_tag: Option<&'a str>,
    pub seen_from: Option<&'a str>,
}

/// Median generation speed of the replies one model instance produced.
#[derive(Debug, Clone, Serialize)]
pub struct InstanceSpeed {
    pub instance_id: String,
    pub started_at: String,
    /// The instance was loaded with `--rpc` (split with linked PCs).
    pub split: bool,
    pub replies: usize,
    pub median_tok_s: f64,
}

impl Db {
    /// Pin a peer. Linking again with the same key keeps its id and settings
    /// but refreshes name and address.
    pub async fn upsert_peer(&self, p: &NewPeer<'_>) -> AppResult<PeerRow> {
        let name = p.name.trim();
        let name = if name.is_empty() { "Linked PC" } else { name };
        let now = now_rfc3339();
        sqlx::query(
            "INSERT INTO linked_peers(id, name, public_key, last_address, same_network, split_layers, initiated, runtime_tag, last_seen_at, created_at, updated_at, seen_from) \
             VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(public_key) DO UPDATE SET name = excluded.name, \
               last_address = COALESCE(excluded.last_address, linked_peers.last_address), \
               runtime_tag = COALESCE(excluded.runtime_tag, linked_peers.runtime_tag), \
               seen_from = COALESCE(excluded.seen_from, linked_peers.seen_from), \
               last_seen_at = excluded.last_seen_at, updated_at = excluded.updated_at",
        )
        .bind(new_id())
        .bind(name)
        .bind(p.public_key)
        .bind(p.last_address)
        .bind(p.same_network as i64)
        .bind(p.same_network as i64)
        .bind(p.initiated as i64)
        .bind(p.runtime_tag)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .bind(p.seen_from)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        self.peer_by_key(p.public_key).await?.ok_or_else(|| AppError::Database("peer vanished after insert".into()))
    }

    pub async fn list_peers(&self) -> AppResult<Vec<PeerRow>> {
        sqlx::query_as::<_, PeerRow>("SELECT * FROM linked_peers ORDER BY created_at").fetch_all(self.pool()).await.map_err(db_err)
    }

    pub async fn get_peer(&self, id: &str) -> AppResult<PeerRow> {
        sqlx::query_as::<_, PeerRow>("SELECT * FROM linked_peers WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)?
            .ok_or_else(|| AppError::NotFound(format!("linked PC {id} not found")))
    }

    pub async fn peer_by_key(&self, public_key: &str) -> AppResult<Option<PeerRow>> {
        sqlx::query_as::<_, PeerRow>("SELECT * FROM linked_peers WHERE public_key = ?")
            .bind(public_key)
            .fetch_optional(self.pool())
            .await
            .map_err(db_err)
    }

    pub async fn set_peer_split(&self, id: &str, on: bool) -> AppResult<()> {
        self.set_peer_flag(id, "split_layers", on).await
    }

    /// Option B switch (stored only in this phase).
    pub async fn set_peer_send_jobs(&self, id: &str, on: bool) -> AppResult<()> {
        self.set_peer_flag(id, "send_jobs", on).await
    }

    async fn set_peer_flag(&self, id: &str, column: &str, on: bool) -> AppResult<()> {
        let sql = match column {
            "split_layers" => "UPDATE linked_peers SET split_layers = ?, updated_at = ? WHERE id = ?",
            "send_jobs" => "UPDATE linked_peers SET send_jobs = ?, updated_at = ? WHERE id = ?",
            _ => return Err(AppError::InvalidInput(format!("unknown peer setting {column}"))),
        };
        let r = sqlx::query(sql)
            .bind(on as i64)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        if r.rows_affected() == 0 {
            return Err(AppError::NotFound(format!("linked PC {id} not found")));
        }
        Ok(())
    }

    /// Set what this PC lends to the linked PC `id` (validated by the caller).
    pub async fn set_peer_share(&self, id: &str, share: &PeerShare) -> AppResult<()> {
        let as_i64 = |v: Option<u64>| v.map(|v| v.min(i64::MAX as u64) as i64);
        let r = sqlx::query("UPDATE linked_peers SET share_gpu_mib = ?, share_ram_mib = ?, share_threads = ?, share_paused = ?, updated_at = ? WHERE id = ?")
            .bind(as_i64(share.gpu_mib))
            .bind(as_i64(share.ram_mib))
            .bind(share.threads.map(|t| t as i64))
            .bind(share.paused as i64)
            .bind(now_rfc3339())
            .bind(id)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        if r.rows_affected() == 0 {
            return Err(AppError::NotFound(format!("linked PC {id} not found")));
        }
        Ok(())
    }

    /// Record a successful connection: name, address, RTT, bandwidth, build.
    pub async fn set_peer_seen(
        &self,
        id: &str,
        name: Option<&str>,
        address: Option<&str>,
        rtt_ms: Option<f64>,
        bandwidth_mbps: Option<f64>,
        runtime_tag: Option<&str>,
    ) -> AppResult<()> {
        sqlx::query(
            "UPDATE linked_peers SET name = COALESCE(?, name), last_address = COALESCE(?, last_address), \
             last_rtt_ms = COALESCE(?, last_rtt_ms), last_bandwidth_mbps = COALESCE(?, last_bandwidth_mbps), \
             runtime_tag = COALESCE(?, runtime_tag), last_seen_at = ?, updated_at = ? WHERE id = ?",
        )
        .bind(name.map(str::trim).filter(|n| !n.is_empty()))
        .bind(address)
        .bind(rtt_ms)
        .bind(bandwidth_mbps)
        .bind(runtime_tag)
        .bind(now_rfc3339())
        .bind(now_rfc3339())
        .bind(id)
        .execute(self.pool())
        .await
        .map_err(db_err)?;
        Ok(())
    }

    /// A PC that linked to this one connected again from `ip`.
    pub async fn set_peer_seen_from(&self, public_key: &str, ip: &str) -> AppResult<()> {
        sqlx::query("UPDATE linked_peers SET seen_from = ?, last_seen_at = ? WHERE public_key = ?")
            .bind(ip)
            .bind(now_rfc3339())
            .bind(public_key)
            .execute(self.pool())
            .await
            .map_err(db_err)?;
        Ok(())
    }

    pub async fn delete_peer(&self, id: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM linked_peers WHERE id = ?").bind(id).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    pub async fn delete_peer_by_key(&self, public_key: &str) -> AppResult<()> {
        sqlx::query("DELETE FROM linked_peers WHERE public_key = ?").bind(public_key).execute(self.pool()).await.map_err(db_err)?;
        Ok(())
    }

    /// Reply speeds per instance of `model_id`, newest first: every assistant
    /// reply written while that instance was the loaded one (only one model
    /// is loaded at a time). Replies under 16 tokens are skipped.
    pub async fn model_instance_speeds(&self, model_id: &str, limit: i64) -> AppResult<Vec<InstanceSpeed>> {
        let rows: Vec<(String, String, Option<i64>, f64)> = sqlx::query_as(
            "SELECT i.id, i.started_at, json_array_length(json_extract(i.config, '$.rpc')), \
                    CAST(json_extract(m.stats, '$.tokens_per_sec') AS REAL) \
             FROM model_instances i JOIN messages m \
               ON m.role = 'assistant' AND m.created_at >= i.started_at AND (i.ended_at IS NULL OR m.created_at <= i.ended_at) \
             WHERE i.model_id = ? AND json_valid(m.stats) AND json_extract(m.stats, '$.tokens_per_sec') > 0 \
               AND COALESCE(json_extract(m.stats, '$.completion_tokens'), 9999) >= 16 \
             ORDER BY i.started_at DESC, m.created_at DESC LIMIT 2000",
        )
        .bind(model_id)
        .fetch_all(self.pool())
        .await
        .map_err(db_err)?;
        let mut out: Vec<(InstanceSpeed, Vec<f64>)> = Vec::new();
        for (id, started, rpc_len, tps) in rows {
            match out.last_mut() {
                Some((s, v)) if s.instance_id == id => v.push(tps),
                _ => out.push((InstanceSpeed { instance_id: id, started_at: started, split: rpc_len.unwrap_or(0) > 0, replies: 0, median_tok_s: 0.0 }, vec![tps])),
            }
        }
        Ok(out
            .into_iter()
            .take(limit.max(1) as usize)
            .map(|(mut s, mut v)| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                s.replies = v.len();
                s.median_tok_s = v[v.len() / 2];
                s
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn peers_pin_once_and_keep_settings_on_relink() {
        let db = Db::open_in_memory().await.unwrap();
        let p = db
            .upsert_peer(&NewPeer { name: "Gaming PC", public_key: "KEY1", last_address: Some("192.168.1.20:47070"), same_network: true, initiated: true, runtime_tag: Some("b12250"), seen_from: None })
            .await
            .unwrap();
        assert_eq!((p.split_layers, p.send_jobs, p.same_network), (1, 0, 1));
        db.set_peer_split(&p.id, false).await.unwrap();
        db.set_peer_send_jobs(&p.id, true).await.unwrap();
        // Linking again with the same key: same row, settings kept, address kept when unknown.
        let again = db.upsert_peer(&NewPeer { name: "Renamed", public_key: "KEY1", last_address: None, same_network: true, initiated: true, runtime_tag: None, seen_from: None }).await.unwrap();
        assert_eq!(again.id, p.id);
        assert_eq!((again.name.as_str(), again.split_layers, again.send_jobs), ("Renamed", 0, 1));
        assert_eq!(again.last_address.as_deref(), Some("192.168.1.20:47070"));
        // Another network: split off by default.
        let far = db.upsert_peer(&NewPeer { name: "Friend", public_key: "KEY2", last_address: Some("81.2.69.160:47070"), same_network: false, initiated: true, runtime_tag: None, seen_from: None }).await.unwrap();
        assert_eq!(far.split_layers, 0);
        db.set_peer_seen(&far.id, None, None, Some(31.5), Some(40.0), Some("b12250")).await.unwrap();
        let f = db.get_peer(&far.id).await.unwrap();
        assert_eq!((f.last_rtt_ms, f.runtime_tag.as_deref(), f.name.as_str()), (Some(31.5), Some("b12250"), "Friend"));
        assert_eq!(db.list_peers().await.unwrap().len(), 2);
        // A PC that linked to this one: shown as "connected from", not dialled.
        let back = db
            .upsert_peer(&NewPeer { name: "VM", public_key: "KEY3", last_address: None, same_network: true, initiated: false, runtime_tag: None, seen_from: Some("10.0.2.2") })
            .await
            .unwrap();
        assert_eq!((back.last_address.as_deref(), back.seen_from.as_deref()), (None, Some("10.0.2.2")));
        db.set_peer_seen_from("KEY3", "192.168.1.30").await.unwrap();
        assert_eq!(db.get_peer(&back.id).await.unwrap().seen_from.as_deref(), Some("192.168.1.30"));
        db.delete_peer(&back.id).await.unwrap();
        db.delete_peer_by_key("KEY2").await.unwrap();
        db.delete_peer(&p.id).await.unwrap();
        assert!(db.list_peers().await.unwrap().is_empty());
        assert!(db.set_peer_split(&p.id, true).await.is_err());
    }

    #[tokio::test]
    async fn peer_share_limits_round_trip_and_default_to_none() {
        let db = Db::open_in_memory().await.unwrap();
        let p = db.upsert_peer(&NewPeer { name: "Desk", public_key: "K", last_address: None, same_network: true, initiated: false, runtime_tag: None, seen_from: None }).await.unwrap();
        // Migration 0009: no limits stored = defaults, not paused.
        assert_eq!(p.share(), PeerShare::default());
        let s = PeerShare { gpu_mib: Some(2048), ram_mib: Some(0), threads: Some(6), paused: true };
        db.set_peer_share(&p.id, &s).await.unwrap();
        assert_eq!(db.get_peer(&p.id).await.unwrap().share(), s);
        // Relinking with the same key keeps them.
        let again = db.upsert_peer(&NewPeer { name: "Desk 2", public_key: "K", last_address: None, same_network: true, initiated: false, runtime_tag: None, seen_from: None }).await.unwrap();
        assert_eq!(again.share(), s);
        // Back to defaults.
        db.set_peer_share(&p.id, &PeerShare::default()).await.unwrap();
        assert_eq!(db.peer_by_key("K").await.unwrap().unwrap().share(), PeerShare::default());
        assert!(db.set_peer_share("nope", &s).await.is_err());
    }

    #[tokio::test]
    async fn reply_speeds_are_grouped_by_instance_and_split() {
        let db = Db::open_in_memory().await.unwrap();
        sqlx::query("INSERT INTO models(id, path, display_name, size, status, metadata, created_at, updated_at) VALUES('m', 'p', 'M', 1, 'ok', '{}', '2026-01-01', '2026-01-01')")
            .execute(db.pool())
            .await
            .expect("insert model (schema changed?)");
        sqlx::query("INSERT INTO conversations(id, title, created_at, updated_at) VALUES('c', 't', '2026-01-01', '2026-01-01')").execute(db.pool()).await.unwrap();
        for (id, cfg, start, end) in [
            ("alone", r#"{"rpc": []}"#, "2026-01-01T10:00:00.000Z", Some("2026-01-01T11:00:00.000Z")),
            ("split", r#"{"rpc": ["127.0.0.1:5000"]}"#, "2026-01-01T12:00:00.000Z", None),
        ] {
            sqlx::query("INSERT INTO model_instances(id, model_id, backend, config, state, started_at, ended_at) VALUES(?, 'm', 'b', ?, 'loaded', ?, ?)")
                .bind(id)
                .bind(cfg)
                .bind(start)
                .bind(end)
                .execute(db.pool())
                .await
                .unwrap();
        }
        for (i, (at, tps, n)) in [
            ("2026-01-01T10:10:00.000Z", 120.0, 100),
            ("2026-01-01T10:20:00.000Z", 124.0, 100),
            ("2026-01-01T10:30:00.000Z", 999.0, 3), // too short: skipped
            ("2026-01-01T12:10:00.000Z", 23.0, 100),
        ]
        .into_iter()
        .enumerate()
        {
            sqlx::query("INSERT INTO messages(id, conversation_id, role, content, stats, created_at) VALUES(?, 'c', 'assistant', 'x', ?, ?)")
                .bind(format!("msg{i}"))
                .bind(serde_json::json!({ "tokens_per_sec": tps, "completion_tokens": n }).to_string())
                .bind(at)
                .execute(db.pool())
                .await
                .unwrap();
        }
        let s = db.model_instance_speeds("m", 10).await.unwrap();
        assert_eq!(s.len(), 2, "{s:?}");
        assert_eq!((s[0].instance_id.as_str(), s[0].split, s[0].replies, s[0].median_tok_s), ("split", true, 1, 23.0));
        assert_eq!((s[1].instance_id.as_str(), s[1].split, s[1].replies), ("alone", false, 2));
        assert_eq!(s[1].median_tok_s, 124.0);
    }
}

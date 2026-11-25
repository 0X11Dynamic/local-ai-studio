//! The helper's weight cache on disk ("Speed up reloads").
//!
//! `ggml-rpc-server -c` keeps every large tensor it receives as a file named
//! by its hash, so the next load of the same model doesn't send it again.
//! Measured on b12250: with `LLAMA_CACHE=<dir>` the server writes to
//! `<dir>\rpc\<hash>` (its banner prints "local cache : <dir>\rpc").
//!
//! The weights a helper receives live in rpc-server's RAM/VRAM only for as
//! long as the connection that sent them (freed when that connection ends or
//! rpc-server stops). This module governs the only thing that stays on disk:
//!
//! - everything lives under one folder of the app (`<data>\linking-cache`);
//! - one rpc-server serves every linked PC and names files by hash, so an
//!   `owners.json` manifest records which linked PC's connection first wrote
//!   each file (by the PC's public key);
//! - a size cap, enforced oldest-first after every connection;
//! - unlinking a PC deletes the files only it wrote; "Clear cache" deletes all.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Default size cap of the cache.
pub const DEFAULT_CAP_BYTES: u64 = 30 << 30;
const OWNERS_FILE: &str = "owners.json";

#[derive(Debug, Clone, PartialEq)]
pub struct CacheFile {
    pub name: String,
    pub size: u64,
    pub modified: SystemTime,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Owners {
    /// file name -> public keys (base64) of the PCs that sent it.
    files: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone)]
pub struct CacheStore {
    root: PathBuf,
}

impl CacheStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Folder to pass to rpc-server as `LLAMA_CACHE`.
    pub fn llama_cache_dir(&self) -> PathBuf {
        self.root.join("store")
    }

    /// Where rpc-server puts the cached tensors.
    fn files_dir(&self) -> PathBuf {
        self.llama_cache_dir().join("rpc")
    }

    pub fn ensure_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.files_dir())
    }

    fn owners_path(&self) -> PathBuf {
        self.root.join(OWNERS_FILE)
    }

    fn load_owners(&self) -> Owners {
        std::fs::read_to_string(self.owners_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    fn save_owners(&self, o: &Owners) {
        if let Err(e) = std::fs::create_dir_all(&self.root).and_then(|_| std::fs::write(self.owners_path(), serde_json::to_vec(o).unwrap_or_default())) {
            tracing::warn!("could not save the weight cache's owner list: {e}");
        }
    }

    /// Cached tensor files (the manifest is not one of them).
    pub fn files(&self) -> Vec<CacheFile> {
        let Ok(rd) = std::fs::read_dir(self.files_dir()) else { return Vec::new() };
        rd.filter_map(|e| {
            let e = e.ok()?;
            let m = e.metadata().ok()?;
            m.is_file().then(|| CacheFile { name: e.file_name().to_string_lossy().into_owned(), size: m.len(), modified: m.modified().unwrap_or(SystemTime::UNIX_EPOCH) })
        })
        .collect()
    }

    pub fn size(&self) -> u64 {
        self.files().iter().map(|f| f.size).sum()
    }

    /// Names of the files in the cache now (take before a connection opens).
    pub fn snapshot(&self) -> BTreeSet<String> {
        self.files().into_iter().map(|f| f.name).collect()
    }

    /// Record `peer` as the owner of every file that appeared since `before`.
    /// Returns how many files it now owns that are new.
    pub fn attribute(&self, peer: &str, before: &BTreeSet<String>) -> usize {
        let new: Vec<String> = self.files().into_iter().map(|f| f.name).filter(|n| !before.contains(n)).collect();
        if new.is_empty() {
            return 0;
        }
        let mut o = self.load_owners();
        for n in &new {
            o.files.entry(n.clone()).or_default().insert(peer.to_string());
        }
        self.save_owners(&o);
        new.len()
    }

    /// Bytes of the files `peer` wrote, per peer key (files two PCs wrote are
    /// counted for both).
    pub fn sizes_by_peer(&self) -> HashMap<String, u64> {
        let o = self.load_owners();
        let mut out: HashMap<String, u64> = HashMap::new();
        for f in self.files() {
            for p in o.files.get(&f.name).into_iter().flatten() {
                *out.entry(p.clone()).or_default() += f.size;
            }
        }
        out
    }

    /// Delete the oldest files until the cache is at most `cap` bytes.
    /// Returns the bytes freed.
    pub fn enforce_cap(&self, cap: u64) -> u64 {
        let mut files = self.files();
        let mut total: u64 = files.iter().map(|f| f.size).sum();
        if total <= cap {
            return 0;
        }
        files.sort_by(|a, b| a.modified.cmp(&b.modified).then_with(|| a.name.cmp(&b.name)));
        let mut freed = 0;
        let mut removed = Vec::new();
        for f in files {
            if total <= cap {
                break;
            }
            if std::fs::remove_file(self.files_dir().join(&f.name)).is_ok() {
                total -= f.size;
                freed += f.size;
                removed.push(f.name);
            }
        }
        self.forget(&removed);
        freed
    }

    fn forget(&self, names: &[String]) {
        if names.is_empty() {
            return;
        }
        let mut o = self.load_owners();
        for n in names {
            o.files.remove(n);
        }
        self.save_owners(&o);
    }

    /// A PC was unlinked: delete the files only it wrote and drop it from the
    /// owners of shared ones. Returns the bytes freed.
    pub fn remove_peer(&self, peer: &str) -> u64 {
        let mut o = self.load_owners();
        let mut freed = 0;
        let mut gone = Vec::new();
        for f in self.files() {
            let Some(owners) = o.files.get_mut(&f.name) else { continue };
            if !owners.remove(peer) || !owners.is_empty() {
                continue;
            }
            if std::fs::remove_file(self.files_dir().join(&f.name)).is_ok() {
                freed += f.size;
                gone.push(f.name);
            }
        }
        for n in &gone {
            o.files.remove(n);
        }
        o.files.retain(|_, v| !v.is_empty());
        self.save_owners(&o);
        freed
    }

    /// Delete everything. Returns the bytes freed.
    pub fn clear(&self) -> u64 {
        let freed = self.size();
        let _ = std::fs::remove_dir_all(self.files_dir());
        let _ = std::fs::remove_file(self.owners_path());
        let _ = self.ensure_dir();
        freed
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn put(store: &CacheStore, name: &str, size: usize, age_secs: u64) {
        store.ensure_dir().unwrap();
        let p = store.files_dir().join(name);
        std::fs::write(&p, vec![1u8; size]).unwrap();
        let t = SystemTime::now() - Duration::from_secs(age_secs);
        std::fs::File::options().write(true).open(&p).unwrap().set_modified(t).unwrap();
    }

    fn store() -> (tempfile::TempDir, CacheStore) {
        let d = tempfile::tempdir().unwrap();
        let s = CacheStore::new(d.path().join("linking-cache"));
        (d, s)
    }

    #[test]
    fn cap_evicts_oldest_first_until_under_the_cap() {
        let (_d, s) = store();
        put(&s, "old", 400, 3000);
        put(&s, "mid", 400, 2000);
        put(&s, "new", 400, 1000);
        assert_eq!(s.size(), 1200);
        assert_eq!(s.enforce_cap(5000), 0, "under the cap: nothing removed");
        // 1200 -> must reach <= 700: removes "old" (800 left) then "mid" (400 left).
        assert_eq!(s.enforce_cap(700), 800);
        let names: Vec<String> = s.files().into_iter().map(|f| f.name).collect();
        assert_eq!(names, vec!["new".to_string()]);
        // A cap of 0 empties it.
        assert_eq!(s.enforce_cap(0), 400);
        assert_eq!(s.size(), 0);
    }

    #[test]
    fn eviction_also_drops_the_owner_records() {
        let (_d, s) = store();
        let before = s.snapshot();
        put(&s, "a", 100, 500);
        s.attribute("peer-1", &before);
        assert_eq!(s.sizes_by_peer().get("peer-1"), Some(&100));
        s.enforce_cap(0);
        assert!(s.sizes_by_peer().is_empty());
        assert!(s.load_owners().files.is_empty());
    }

    #[test]
    fn files_are_attributed_to_the_connection_that_added_them() {
        let (_d, s) = store();
        put(&s, "pre-existing", 50, 900);
        let before = s.snapshot();
        put(&s, "w1", 100, 10);
        put(&s, "w2", 200, 10);
        assert_eq!(s.attribute("peer-1", &before), 2);
        // Nothing new for a later connection that only reused the cache.
        let before2 = s.snapshot();
        assert_eq!(s.attribute("peer-2", &before2), 0);
        let sizes = s.sizes_by_peer();
        assert_eq!(sizes.get("peer-1"), Some(&300));
        assert_eq!(sizes.get("peer-2"), None);
    }

    #[test]
    fn unlinking_a_peer_deletes_only_what_it_alone_wrote() {
        let (_d, s) = store();
        let b = s.snapshot();
        put(&s, "only-1", 100, 30);
        s.attribute("peer-1", &b);
        let b = s.snapshot();
        put(&s, "only-2", 200, 20);
        s.attribute("peer-2", &b);
        // Both connections were open while "shared" was written.
        let b0 = s.snapshot();
        put(&s, "shared", 300, 10);
        s.attribute("peer-1", &b0);
        s.attribute("peer-2", &b0);

        assert_eq!(s.remove_peer("peer-1"), 100, "its own file only; the shared one stays for peer-2");
        let mut names: Vec<String> = s.files().into_iter().map(|f| f.name).collect();
        names.sort();
        assert_eq!(names, vec!["only-2".to_string(), "shared".to_string()]);
        assert_eq!(s.sizes_by_peer().get("peer-2"), Some(&500));
        assert_eq!(s.sizes_by_peer().get("peer-1"), None);
        // The last owner leaving removes the shared file too.
        assert_eq!(s.remove_peer("peer-2"), 500);
        assert_eq!(s.size(), 0);
        // Unknown or already removed peer: nothing happens.
        assert_eq!(s.remove_peer("peer-9"), 0);
    }

    #[test]
    fn clear_removes_everything_and_reports_the_size() {
        let (_d, s) = store();
        let b = s.snapshot();
        put(&s, "a", 1000, 5);
        put(&s, "b", 234, 5);
        s.attribute("peer-1", &b);
        assert_eq!(s.size(), 1234);
        assert_eq!(s.clear(), 1234);
        assert_eq!(s.size(), 0);
        assert!(s.sizes_by_peer().is_empty());
        assert!(s.files_dir().is_dir(), "the folder is ready for rpc-server again");
    }

    #[test]
    fn missing_folder_is_an_empty_cache() {
        let (_d, s) = store();
        assert_eq!(s.size(), 0);
        assert_eq!(s.enforce_cap(0), 0);
        assert_eq!(s.remove_peer("x"), 0);
        assert_eq!(s.clear(), 0);
    }
}

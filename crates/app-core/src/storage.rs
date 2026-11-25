//! User-chosen storage folders (models, runtimes), kept in `storage.json`
//! inside the data folder because the runtimes folder is needed before the
//! database opens. This file is the single source of truth; the legacy
//! `library.models_dir` setting is kept in sync by the app.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{AppError, AppResult};

pub const STORAGE_FILE: &str = "storage.json";

/// Where the recorded choice came from.
pub const ORIGIN_INSTALL_DEFAULT: &str = "install-default";
pub const ORIGIN_EXISTING: &str = "existing-install";
pub const ORIGIN_USER: &str = "user";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageConfig {
    pub models_dir: PathBuf,
    pub runtimes_dir: PathBuf,
    /// `install-default` (new install, folders next to the app),
    /// `existing-install` (an older profile; nothing moved) or `user`.
    #[serde(default)]
    pub origin: String,
}

pub fn storage_path(data_dir: &Path) -> PathBuf {
    data_dir.join(STORAGE_FILE)
}

pub fn read(data_dir: &Path) -> Option<StorageConfig> {
    let text = std::fs::read_to_string(storage_path(data_dir)).ok()?;
    let cfg: StorageConfig = serde_json::from_str(&text).ok()?;
    (cfg.models_dir.is_absolute() && cfg.runtimes_dir.is_absolute()).then_some(cfg)
}

pub fn write(data_dir: &Path, cfg: &StorageConfig) -> AppResult<()> {
    std::fs::create_dir_all(data_dir)?;
    let tmp = data_dir.join("storage.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(cfg)?)?;
    std::fs::rename(&tmp, storage_path(data_dir))?;
    Ok(())
}

/// The install folder when `exe` belongs to an installed build: an
/// `uninstall.exe` sits next to it, or it is not inside a cargo `target`
/// folder (dev and test builds run from `target\debug|release`).
pub fn detect_install_dir(exe: &Path) -> Option<PathBuf> {
    let dir = exe.parent()?;
    if dir.join("uninstall.exe").is_file() {
        return Some(dir.to_path_buf());
    }
    let in_target = dir.components().any(|c| c.as_os_str().eq_ignore_ascii_case("target"));
    (!in_target).then(|| dir.to_path_buf())
}

/// Reject paths that cannot be a storage folder: relative, UNC/device,
/// containing `..`, or a drive root. Pure; touches nothing on disk.
pub fn check_shape(path: &Path) -> AppResult<PathBuf> {
    let s = path.to_string_lossy();
    if s.trim().is_empty() {
        return Err(AppError::InvalidInput("the folder path is empty".into()));
    }
    if s.starts_with(r"\\") || s.starts_with("//") {
        return Err(AppError::InvalidInput("network (UNC) and device paths are not supported; pick a folder on a local drive".into()));
    }
    if !path.is_absolute() {
        return Err(AppError::InvalidInput(r"use a full path such as D:\AI\models".into()));
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(AppError::InvalidInput("the path may not contain '..'".into()));
    }
    let mut clean = PathBuf::new();
    for c in path.components() {
        if !matches!(c, Component::CurDir) {
            clean.push(c.as_os_str());
        }
    }
    if !clean.components().any(|c| matches!(c, Component::Normal(_))) {
        return Err(AppError::InvalidInput("pick a folder, not a whole drive".into()));
    }
    Ok(clean)
}

/// `check_shape`, then create the folder and prove it is writable.
pub fn check_dir(path: &Path) -> AppResult<PathBuf> {
    let clean = check_shape(path)?;
    if clean.exists() && !clean.is_dir() {
        return Err(AppError::InvalidInput(format!("{} is a file, not a folder", clean.display())));
    }
    std::fs::create_dir_all(&clean)
        .map_err(|e| AppError::InvalidInput(format!("cannot create {}: {e}", clean.display())))?;
    let probe = clean.join(format!(".las-write-test-{}", std::process::id()));
    std::fs::write(&probe, b"x")
        .map_err(|e| AppError::InvalidInput(format!("{} is not writable: {e}", clean.display())))?;
    let _ = std::fs::remove_file(&probe);
    Ok(clean)
}

/// Does a runtimes folder hold at least one installed runtime?
pub fn has_runtimes(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.path().join("runtime.json").is_file()))
        .unwrap_or(false)
}

/// Does a folder contain anything at all?
pub fn has_content(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut rd| rd.next().is_some()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_detection() {
        let t = tempfile::tempdir().unwrap();
        let app = t.path().join("Local AI Studio");
        std::fs::create_dir_all(&app).unwrap();
        // Installed: lives outside any target folder.
        assert_eq!(detect_install_dir(&app.join("local-ai-studio.exe")), Some(app.clone()));
        // Dev/test build.
        let dev = t.path().join("repo").join("target").join("debug");
        std::fs::create_dir_all(&dev).unwrap();
        assert_eq!(detect_install_dir(&dev.join("local-ai-studio.exe")), None);
        // An uninstaller next to the exe wins, even under a folder named target.
        std::fs::write(dev.join("uninstall.exe"), b"").unwrap();
        assert_eq!(detect_install_dir(&dev.join("local-ai-studio.exe")), Some(dev.clone()));
    }

    #[test]
    fn shape_and_writable_checks() {
        assert!(check_shape(Path::new("models")).is_err());
        assert!(check_shape(Path::new("")).is_err());
        assert!(check_shape(Path::new(r"\server\share\models")).is_err());
        assert!(check_shape(Path::new(r"C:\")).is_err());
        assert!(check_shape(Path::new(r"C:\a\..\b")).is_err());
        assert!(check_shape(Path::new(r"C:\AI\models")).is_ok());
        let t = tempfile::tempdir().unwrap();
        let ok = check_dir(&t.path().join("new").join("models")).unwrap();
        assert!(ok.is_dir());
        assert_eq!(std::fs::read_dir(&ok).unwrap().count(), 0, "probe file removed");
        let file = t.path().join("f.txt");
        std::fs::write(&file, b"x").unwrap();
        assert!(check_dir(&file).is_err());
    }

    #[test]
    fn config_round_trip() {
        let t = tempfile::tempdir().unwrap();
        assert!(read(t.path()).is_none());
        let cfg = StorageConfig {
            models_dir: t.path().join("m"),
            runtimes_dir: t.path().join("r"),
            origin: ORIGIN_USER.into(),
        };
        write(t.path(), &cfg).unwrap();
        assert_eq!(read(t.path()), Some(cfg));
        std::fs::write(storage_path(t.path()), "{ not json").unwrap();
        assert!(read(t.path()).is_none());
    }
}

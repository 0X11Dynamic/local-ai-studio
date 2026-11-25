use std::path::{Path, PathBuf};

use crate::storage::{self, StorageConfig};
use crate::{AppError, AppResult};

/// Folder name used under %APPDATA% and %USERPROFILE%.
pub const APP_DIR_NAME: &str = "LocalAIStudio";

/// All on-disk locations the app uses. Large model files never live in the
/// database; they live in `default_models_dir` or wherever the user imported them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AppPaths {
    /// %APPDATA%\LocalAIStudio
    pub data_dir: PathBuf,
    /// %APPDATA%\LocalAIStudio\app.db
    pub db_path: PathBuf,
    /// Installed llama.cpp builds. Default %APPDATA%\LocalAIStudio\runtimes; the
    /// folder recorded in storage.json wins (see `storage`). Changed at runtime
    /// through `RuntimeManager`; this is the value at startup.
    pub runtimes_dir: PathBuf,
    /// %APPDATA%\LocalAIStudio\logs
    pub logs_dir: PathBuf,
    /// %APPDATA%\LocalAIStudio\skills
    pub skills_dir: PathBuf,
    /// Models folder at startup: storage.json, else %USERPROFILE%\LocalAIStudio\models.
    pub default_models_dir: PathBuf,
    /// Folder of the installed app (None for dev/test builds).
    pub install_dir: Option<PathBuf>,
    /// Where the storage choice came from (`storage::ORIGIN_*`), empty if not recorded.
    pub storage_origin: String,
    /// %USERPROFILE%\.lmstudio\models, if it exists (registered in place, read-only)
    pub lmstudio_models_dir: Option<PathBuf>,
}

impl AppPaths {
    /// Resolve the standard locations for the current Windows user.
    /// `LOCALAISTUDIO_DATA_DIR` overrides the data folder (used for isolated
    /// test profiles so automated tests never touch the real library/settings).
    pub fn resolve() -> AppResult<Self> {
        let home =
            dirs::home_dir().ok_or_else(|| AppError::Other("cannot resolve home folder".into()))?;
        let exe = std::env::current_exe().ok();
        if let Some(dir) = std::env::var_os("LOCALAISTUDIO_DATA_DIR").filter(|d| !d.is_empty()) {
            return Ok(Self::resolve_with(Path::new(&dir), &home, exe.as_deref(), true));
        }
        let config = dirs::config_dir()
            .ok_or_else(|| AppError::Other("cannot resolve %APPDATA%".into()))?;
        Ok(Self::resolve_with(&config.join(APP_DIR_NAME), &home, exe.as_deref(), false))
    }

    /// Roots plus the storage choice. Order: a recorded `storage.json` wins.
    /// Otherwise (not for isolated test profiles, which keep the plain
    /// defaults): an older profile (runtimes, models or a database already in
    /// the old places) records those places as the choice so nothing moves;
    /// a new install of an installed build records `<install folder>\models`
    /// and `<install folder>\runtimes`; a dev build changes nothing.
    pub fn resolve_with(data_dir: &Path, home: &Path, exe: Option<&Path>, isolated: bool) -> Self {
        let mut p = Self::with_roots(data_dir, home);
        p.install_dir = if isolated { None } else { exe.and_then(storage::detect_install_dir) };
        if let Some(cfg) = storage::read(data_dir) {
            return p.apply(cfg);
        }
        if isolated {
            return p;
        }
        let legacy = storage::has_runtimes(&p.runtimes_dir)
            || storage::has_content(&p.default_models_dir)
            || p.db_path.is_file();
        let cfg = if legacy {
            Some(StorageConfig {
                models_dir: p.default_models_dir.clone(),
                runtimes_dir: p.runtimes_dir.clone(),
                origin: storage::ORIGIN_EXISTING.into(),
            })
        } else {
            p.install_dir.as_ref().map(|i| StorageConfig {
                models_dir: i.join("models"),
                runtimes_dir: i.join("runtimes"),
                origin: storage::ORIGIN_INSTALL_DEFAULT.into(),
            })
        };
        match cfg {
            Some(cfg) => {
                // Not recorded when the install folder is not writable: the
                // plain defaults keep working then.
                if cfg.origin == storage::ORIGIN_INSTALL_DEFAULT
                    && (storage::check_dir(&cfg.models_dir).is_err() || storage::check_dir(&cfg.runtimes_dir).is_err())
                {
                    return p;
                }
                let _ = storage::write(data_dir, &cfg);
                p.apply(cfg)
            }
            None => p,
        }
    }

    fn apply(mut self, cfg: StorageConfig) -> Self {
        self.runtimes_dir = cfg.runtimes_dir;
        self.default_models_dir = cfg.models_dir;
        self.storage_origin = cfg.origin;
        self
    }

    /// Build paths from explicit roots (used by tests and portable mode).
    pub fn with_roots(data_dir: &Path, home: &Path) -> Self {
        let lm = home.join(".lmstudio").join("models");
        Self {
            data_dir: data_dir.to_path_buf(),
            db_path: data_dir.join("app.db"),
            runtimes_dir: data_dir.join("runtimes"),
            logs_dir: data_dir.join("logs"),
            skills_dir: data_dir.join("skills"),
            default_models_dir: home.join(APP_DIR_NAME).join("models"),
            install_dir: None,
            storage_origin: String::new(),
            lmstudio_models_dir: lm.is_dir().then_some(lm),
        }
    }

    /// Create every directory the app writes to.
    pub fn ensure_dirs(&self) -> AppResult<()> {
        for d in [&self.data_dir, &self.logs_dir, &self.skills_dir] {
            std::fs::create_dir_all(d)?;
        }
        // The user's folders may be on a drive that is not there right now;
        // the app still starts and the Storage settings show the problem.
        for d in [&self.runtimes_dir, &self.default_models_dir] {
            let _ = std::fs::create_dir_all(d);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Env {
        _t: tempfile::TempDir,
        data: PathBuf,
        home: PathBuf,
        app: PathBuf,
    }
    fn env() -> Env {
        let t = tempfile::tempdir().unwrap();
        let (data, home, app) = (t.path().join("data"), t.path().join("home"), t.path().join("Local AI Studio"));
        for d in [&data, &home, &app] {
            std::fs::create_dir_all(d).unwrap();
        }
        Env { _t: t, data, home, app }
    }

    #[test]
    fn new_install_defaults_to_the_install_folder() {
        let e = env();
        let exe = e.app.join("local-ai-studio.exe");
        let p = AppPaths::resolve_with(&e.data, &e.home, Some(&exe), false);
        assert_eq!(p.default_models_dir, e.app.join("models"));
        assert_eq!(p.runtimes_dir, e.app.join("runtimes"));
        assert_eq!(p.install_dir.as_deref(), Some(e.app.as_path()));
        assert_eq!(p.storage_origin, storage::ORIGIN_INSTALL_DEFAULT);
        // Recorded: the next start reads it back.
        let again = AppPaths::resolve_with(&e.data, &e.home, None, false);
        assert_eq!(again.default_models_dir, e.app.join("models"));
    }

    #[test]
    fn dev_build_and_isolated_profile_keep_todays_defaults() {
        let e = env();
        let dev = e.app.join("target").join("debug").join("local-ai-studio.exe");
        let p = AppPaths::resolve_with(&e.data, &e.home, Some(&dev), false);
        assert_eq!(p.default_models_dir, e.home.join(APP_DIR_NAME).join("models"));
        assert_eq!(p.runtimes_dir, e.data.join("runtimes"));
        assert!(storage::read(&e.data).is_none(), "nothing recorded for a dev build");
        // LOCALAISTUDIO_DATA_DIR profile, even with an installed-looking exe.
        let exe = e.app.join("local-ai-studio.exe");
        let p = AppPaths::resolve_with(&e.data, &e.home, Some(&exe), true);
        assert_eq!(p.runtimes_dir, e.data.join("runtimes"));
        assert!(p.install_dir.is_none());
        assert!(storage::read(&e.data).is_none());
    }

    #[test]
    fn existing_install_records_the_current_locations() {
        let e = env();
        let exe = e.app.join("local-ai-studio.exe");
        // Old profile: a runtime in the data folder.
        let rt = e.data.join("runtimes").join("b1-cpu");
        std::fs::create_dir_all(&rt).unwrap();
        std::fs::write(rt.join("runtime.json"), "{}").unwrap();
        let p = AppPaths::resolve_with(&e.data, &e.home, Some(&exe), false);
        assert_eq!(p.runtimes_dir, e.data.join("runtimes"), "nothing moves");
        assert_eq!(p.default_models_dir, e.home.join(APP_DIR_NAME).join("models"));
        assert_eq!(p.storage_origin, storage::ORIGIN_EXISTING);
        let rec = storage::read(&e.data).unwrap();
        assert_eq!(rec.runtimes_dir, e.data.join("runtimes"));
        assert_eq!(rec.models_dir, e.home.join(APP_DIR_NAME).join("models"));

        // Models only (old models folder has content) is enough too.
        let e2 = env();
        let m = e2.home.join(APP_DIR_NAME).join("models");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::write(m.join("x.gguf"), b"").unwrap();
        let p = AppPaths::resolve_with(&e2.data, &e2.home, Some(&e2.app.join("a.exe")), false);
        assert_eq!(p.storage_origin, storage::ORIGIN_EXISTING);
        assert_eq!(p.default_models_dir, m);

        // An existing database (models registered by absolute path) counts as well.
        let e3 = env();
        std::fs::write(e3.data.join("app.db"), b"").unwrap();
        let p = AppPaths::resolve_with(&e3.data, &e3.home, Some(&e3.app.join("a.exe")), false);
        assert_eq!(p.storage_origin, storage::ORIGIN_EXISTING);
        assert_eq!(p.runtimes_dir, e3.data.join("runtimes"));
    }

    #[test]
    fn recorded_choice_beats_everything() {
        let e = env();
        let cfg = StorageConfig {
            models_dir: e.home.join("elsewhere"),
            runtimes_dir: e.home.join("rt"),
            origin: storage::ORIGIN_USER.into(),
        };
        storage::write(&e.data, &cfg).unwrap();
        let p = AppPaths::resolve_with(&e.data, &e.home, Some(&e.app.join("a.exe")), false);
        assert_eq!(p.default_models_dir, cfg.models_dir);
        assert_eq!(p.runtimes_dir, cfg.runtimes_dir);
        // An isolated profile may carry its own storage.json too (set from the UI in tests).
        let p = AppPaths::resolve_with(&e.data, &e.home, None, true);
        assert_eq!(p.runtimes_dir, cfg.runtimes_dir);
    }
}

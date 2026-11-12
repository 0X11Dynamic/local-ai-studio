use std::path::{Path, PathBuf};

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
    /// %APPDATA%\LocalAIStudio\runtimes  (installed llama.cpp builds)
    pub runtimes_dir: PathBuf,
    /// %APPDATA%\LocalAIStudio\logs
    pub logs_dir: PathBuf,
    /// %APPDATA%\LocalAIStudio\skills
    pub skills_dir: PathBuf,
    /// %USERPROFILE%\LocalAIStudio\models  (default download folder, configurable)
    pub default_models_dir: PathBuf,
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
        if let Some(dir) = std::env::var_os("LOCALAISTUDIO_DATA_DIR").filter(|d| !d.is_empty()) {
            return Ok(Self::with_roots(Path::new(&dir), &home));
        }
        let config = dirs::config_dir()
            .ok_or_else(|| AppError::Other("cannot resolve %APPDATA%".into()))?;
        Ok(Self::with_roots(&config.join(APP_DIR_NAME), &home))
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
            lmstudio_models_dir: lm.is_dir().then_some(lm),
        }
    }

    /// Create every directory the app writes to.
    pub fn ensure_dirs(&self) -> AppResult<()> {
        for d in [
            &self.data_dir,
            &self.runtimes_dir,
            &self.logs_dir,
            &self.skills_dir,
            &self.default_models_dir,
        ] {
            std::fs::create_dir_all(d)?;
        }
        Ok(())
    }
}

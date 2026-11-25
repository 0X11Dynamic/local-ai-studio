//! Storage folders the user can set (models, runtimes) and the read-only data
//! folder. The choice lives in `storage.json` (see `app_core::storage`); this
//! module validates it, applies it to the running app and reports free space.

use std::path::{Path, PathBuf};

use app_core::storage::{self, StorageConfig};
use app_core::{AppError, AppPaths};
use serde::Serialize;
use tauri::{AppHandle, State};

use super::Cmd;
use crate::state::{AppState, SharedState};

/// A drive with less free space than this gets a warning in the UI.
const LOW_SPACE_BYTES: u64 = 20 * (1 << 30);
/// The legacy setting kept in sync with storage.json.
pub const MODELS_SETTING: &str = "library.models_dir";

/// Free bytes on the drive holding `path` (or its nearest existing parent).
pub fn free_bytes(path: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        use windows::core::PCWSTR;
        let mut p = path.to_path_buf();
        while !p.exists() {
            if !p.pop() {
                return None;
            }
        }
        let wide: Vec<u16> = p.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut free = 0u64;
        // SAFETY: `wide` is NUL-terminated and outlives the call; `free` is a valid out pointer.
        unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut free), None, None).ok()? };
        Some(free)
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        None
    }
}

/// Default folders for a fresh choice: next to the installed app, else the
/// old per-user places (dev builds).
pub fn default_models_dir(paths: &AppPaths) -> PathBuf {
    match &paths.install_dir {
        Some(i) => i.join("models"),
        None => dirs::home_dir().unwrap_or_default().join(app_core::paths::APP_DIR_NAME).join("models"),
    }
}

pub fn default_runtimes_dir(paths: &AppPaths) -> PathBuf {
    match &paths.install_dir {
        Some(i) => i.join("runtimes"),
        None => paths.data_dir.join("runtimes"),
    }
}

/// Check a folder the user typed or picked, without creating anything:
/// full local path, not a protected system folder. The app's own default
/// runtimes folder inside the data folder stays allowed.
pub fn check_folder(paths: &AppPaths, input: &str) -> Cmd<PathBuf> {
    let shaped = storage::check_shape(Path::new(input.trim()))?;
    let policy = execution::paths::RootPolicy::new(vec![], vec![]);
    let (resolved, _) = match policy.classify(&shaped.to_string_lossy()) {
        Ok(r) => r,
        Err(e) => {
            // %APPDATA%\LocalAIStudio\runtimes (the old default) is inside the
            // protected data folder; allow exactly that one place.
            let legacy = paths.data_dir.join("runtimes");
            if shaped.to_string_lossy().trim_end_matches('\\').eq_ignore_ascii_case(&legacy.to_string_lossy()) {
                return Ok(shaped);
            }
            return Err(e);
        }
    };
    Ok(resolved)
}

#[derive(Debug, Serialize)]
pub struct FolderCheck {
    /// The normalized path, or the input when it was rejected.
    pub path: String,
    pub ok: bool,
    pub error: Option<String>,
    pub exists: bool,
    pub free_bytes: Option<u64>,
    /// Under 20 GB free on that drive.
    pub low_space: bool,
}

fn folder_check(paths: &AppPaths, input: &str) -> FolderCheck {
    match check_folder(paths, input) {
        Ok(p) => {
            let free = free_bytes(&p);
            FolderCheck {
                path: p.to_string_lossy().into_owned(),
                ok: true,
                error: None,
                exists: p.is_dir(),
                free_bytes: free,
                low_space: free.is_some_and(|f| f < LOW_SPACE_BYTES),
            }
        }
        Err(e) => FolderCheck { path: input.to_string(), ok: false, error: Some(e.to_string()), exists: false, free_bytes: None, low_space: false },
    }
}

#[derive(Debug, Serialize)]
pub struct StorageInfo {
    pub models: FolderCheck,
    pub runtimes: FolderCheck,
    pub data_dir: String,
    pub install_dir: Option<String>,
    pub default_models_dir: String,
    pub default_runtimes_dir: String,
    /// install-default | existing-install | user | "" (dev build, nothing recorded)
    pub origin: String,
    pub runtime_count: usize,
}

async fn info(st: &AppState) -> StorageInfo {
    let models = st.models_dir().await;
    let runtimes = st.runtimes.dir();
    let origin = storage::read(&st.paths.data_dir).map(|c| c.origin).unwrap_or_default();
    // Show what is configured even if the old drive is gone: a plain check, no rejection.
    let plain = |p: &Path| {
        let free = free_bytes(p);
        FolderCheck {
            path: p.to_string_lossy().into_owned(),
            ok: p.is_dir(),
            error: (!p.is_dir()).then(|| "this folder does not exist (is the drive connected?)".to_string()),
            exists: p.is_dir(),
            free_bytes: free,
            low_space: free.is_some_and(|f| f < LOW_SPACE_BYTES),
        }
    };
    StorageInfo {
        models: plain(&models),
        runtimes: plain(&runtimes),
        data_dir: st.paths.data_dir.to_string_lossy().into_owned(),
        install_dir: st.paths.install_dir.as_ref().map(|p| p.to_string_lossy().into_owned()),
        default_models_dir: default_models_dir(&st.paths).to_string_lossy().into_owned(),
        default_runtimes_dir: default_runtimes_dir(&st.paths).to_string_lossy().into_owned(),
        origin,
        runtime_count: st.runtimes.list_installed().len(),
    }
}

#[tauri::command]
pub async fn get_storage(state: State<'_, SharedState>) -> Cmd<StorageInfo> {
    Ok(info(&state).await)
}

/// Validate a folder for the Storage UI (no changes on disk).
#[tauri::command]
pub fn check_storage_folder(state: State<'_, SharedState>, path: String) -> FolderCheck {
    folder_check(&state.paths, &path)
}

/// Apply new folders. `None` leaves that folder alone. Moving the runtimes
/// (`move_runtimes`) copies them, re-checks the manifests and then deletes the
/// old copies; it and any change of the runtimes folder are refused while a
/// model runs or a runtime installs.
pub async fn apply(st: &AppState, models: Option<String>, runtimes: Option<String>, move_runtimes: bool) -> Cmd<StorageInfo> {
    let mut cfg = storage::read(&st.paths.data_dir).unwrap_or_else(|| StorageConfig {
        models_dir: st.paths.default_models_dir.clone(),
        runtimes_dir: st.runtimes.dir(),
        origin: storage::ORIGIN_USER.into(),
    });
    cfg.origin = storage::ORIGIN_USER.into();
    // Validate everything before changing anything.
    let new_models = match models.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => Some(storage::check_dir(&check_folder(&st.paths, m)?)?),
        None => None,
    };
    let new_runtimes = match runtimes.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        Some(r) => Some(storage::check_dir(&check_folder(&st.paths, r)?)?),
        None => None,
    };
    if let Some(r) = new_runtimes.filter(|r| !same_path(r, &st.runtimes.dir())) {
        if st.engine.backend_pid().is_some() {
            return Err(AppError::Conflict("unload the model first; the runtimes folder can't change while a model runs".into()));
        }
        if !st.installs.lock().unwrap().is_empty() {
            return Err(AppError::Conflict("a runtime is installing; wait for it to finish first".into()));
        }
        if move_runtimes {
            // Copying can take a while; keep the async workers free.
            tokio::task::block_in_place(|| st.runtimes.move_to(&r))?;
        } else {
            st.runtimes.set_dir(r.clone());
        }
        cfg.runtimes_dir = r;
    }
    if let Some(m) = new_models {
        cfg.models_dir = m;
    }
    storage::write(&st.paths.data_dir, &cfg)?;
    // Keep the legacy setting equal to the one source of truth.
    st.db.set_setting(MODELS_SETTING, &cfg.models_dir.to_string_lossy().into_owned()).await?;
    Ok(info(st).await)
}

fn same_path(a: &Path, b: &Path) -> bool {
    let k = |p: &Path| p.to_string_lossy().replace('/', "\\").trim_end_matches('\\').to_lowercase();
    k(a) == k(b)
}

#[tauri::command]
pub async fn set_storage_dirs(
    state: State<'_, SharedState>,
    models_dir: Option<String>,
    runtimes_dir: Option<String>,
    move_runtimes: bool,
) -> Cmd<StorageInfo> {
    apply(&state, models_dir, runtimes_dir, move_runtimes).await
}

/// Open the data folder (chats, settings) in Explorer. The path is fixed here.
#[tauri::command]
pub fn open_data_folder(app: AppHandle, state: State<'_, SharedState>) -> Cmd<()> {
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_path(state.paths.data_dir.to_string_lossy().into_owned(), None::<&str>)
        .map_err(|e| AppError::Other(format!("couldn't open the folder: {e}")))
}

/// Called once at startup after the database opens: an older install that
/// has `library.models_dir` set keeps it as the recorded choice. Returns the
/// models folder the session should use instead of the resolved one.
pub async fn migrate_models_setting(db: &persistence::Db, paths: &AppPaths) -> Option<PathBuf> {
    let setting = db.get_setting::<String>(MODELS_SETTING).await.ok().flatten()?;
    let setting = PathBuf::from(setting.trim());
    if setting.as_os_str().is_empty() || !setting.is_absolute() {
        return None;
    }
    match storage::read(&paths.data_dir) {
        Some(mut c) if c.origin == storage::ORIGIN_EXISTING && c.models_dir != setting => {
            c.models_dir = setting.clone();
            storage::write(&paths.data_dir, &c).ok()?;
            Some(setting)
        }
        None => {
            let c = StorageConfig { models_dir: setting.clone(), runtimes_dir: paths.runtimes_dir.clone(), origin: storage::ORIGIN_USER.into() };
            storage::write(&paths.data_dir, &c).ok()?;
            Some(setting)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(root: &Path) -> AppPaths {
        AppPaths::with_roots(&root.join("data"), &root.join("home"))
    }

    #[test]
    fn folder_validation_rules() {
        let t = tempfile::tempdir().unwrap();
        let p = paths(t.path());
        assert!(check_folder(&p, "models").is_err(), "relative");
        assert!(check_folder(&p, r"\\server\share\m").is_err(), "UNC");
        assert!(check_folder(&p, r"C:\").is_err(), "drive root");
        if let Ok(win) = std::env::var("SystemRoot") {
            assert!(check_folder(&p, &format!(r"{win}\System32\models")).is_err(), "system folder");
        }
        let ok = t.path().join("AI").join("models");
        assert!(check_folder(&p, &ok.to_string_lossy()).is_ok());
        // The folder is only created by apply (check_dir), not by the check.
        assert!(!ok.exists());
        let made = storage::check_dir(&ok).unwrap();
        assert!(made.is_dir());
        assert!(folder_check(&p, &ok.to_string_lossy()).exists);
    }

    #[test]
    fn free_space_is_reported_for_a_missing_folder() {
        let t = tempfile::tempdir().unwrap();
        let f = free_bytes(&t.path().join("not").join("there"));
        if cfg!(windows) {
            assert!(f.is_some_and(|b| b > 0));
        }
    }

    #[test]
    fn defaults_follow_the_install_folder() {
        let t = tempfile::tempdir().unwrap();
        let mut p = paths(t.path());
        assert_eq!(default_runtimes_dir(&p), p.data_dir.join("runtimes"));
        p.install_dir = Some(t.path().join("app"));
        assert_eq!(default_models_dir(&p), t.path().join("app").join("models"));
        assert_eq!(default_runtimes_dir(&p), t.path().join("app").join("runtimes"));
    }
}

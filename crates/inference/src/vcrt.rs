//! App-local Microsoft Visual C++ runtime.
//!
//! llama.cpp's Windows builds need `vcruntime140.dll`, `vcruntime140_1.dll` and
//! `msvcp140.dll`. A fresh Windows has none of them and the process dies with
//! 0xC0000135 (STATUS_DLL_NOT_FOUND). The app bundles the three x64 DLLs
//! (`src-tauri/resources/vcrt`) and copies them next to the runtime's exe
//! (app-local deployment, allowed by Microsoft's redistributable terms).

use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub const DLLS: [&str; 3] = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"];
/// `STATUS_DLL_NOT_FOUND` as the (signed) exit code Rust reports on Windows.
pub const STATUS_DLL_NOT_FOUND: i32 = 0xC000_0135_u32 as i32;

static SOURCE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Folder that holds the bundled DLLs (set once at app start).
pub fn set_source_dir(dir: PathBuf) {
    *SOURCE.write().unwrap() = Some(dir);
}

fn source_dirs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = SOURCE.read().unwrap().iter().cloned().collect();
    if let Some(sys) = std::env::var_os("SystemRoot") {
        v.push(PathBuf::from(sys).join("System32"));
    }
    v
}

/// Copy any of the three DLLs missing from `target_dir` out of `source`.
/// Existing files are never touched (a build may ship newer ones). Returns the
/// names copied.
pub fn copy_missing(source: &Path, target_dir: &Path) -> Vec<String> {
    let mut copied = Vec::new();
    for name in DLLS {
        let dst = target_dir.join(name);
        let src = source.join(name);
        if dst.exists() || !src.is_file() {
            continue;
        }
        match std::fs::copy(&src, &dst) {
            Ok(_) => copied.push(name.to_string()),
            Err(e) => tracing::warn!("could not copy {name} to {}: {e}", target_dir.display()),
        }
    }
    copied
}

/// Make sure the DLLs sit next to `exe` (cheap: three existence checks when
/// nothing is missing). Returns the names copied.
pub fn ensure_next_to(exe: &Path) -> Vec<String> {
    let Some(dir) = exe.parent() else { return Vec::new() };
    if DLLS.iter().all(|n| dir.join(n).exists()) {
        return Vec::new();
    }
    let mut copied = Vec::new();
    for src in source_dirs() {
        copied.extend(copy_missing(&src, dir));
        if DLLS.iter().all(|n| dir.join(n).exists()) {
            break;
        }
    }
    if !copied.is_empty() {
        tracing::info!("copied {} next to {}", copied.join(", "), exe.display());
    }
    copied
}

/// Error text for a backend that exited with `STATUS_DLL_NOT_FOUND`.
pub fn missing_dll_message() -> (String, String) {
    (
        format!(
            "the backend could not start because the Microsoft Visual C++ runtime DLLs ({}) are missing (exit code 0xC0000135); the app tried to copy them next to the runtime automatically",
            DLLS.join(", ")
        ),
        "Install the Microsoft Visual C++ 2015-2022 Redistributable (x64) from microsoft.com, then load again.".into(),
    )
}

pub fn is_missing_dll_error(msg: &str) -> bool {
    msg.contains("0xC0000135")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(dir: &Path, names: &[&str], body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        for n in names {
            std::fs::write(dir.join(n), body).unwrap();
        }
    }

    #[test]
    fn copies_only_what_is_missing_and_never_overwrites() {
        let t = tempfile::tempdir().unwrap();
        let (src, dst) = (t.path().join("src"), t.path().join("rt"));
        fill(&src, &DLLS, "bundled");
        fill(&dst, &["msvcp140.dll"], "newer");
        let copied = copy_missing(&src, &dst);
        assert_eq!(copied, vec!["vcruntime140.dll", "vcruntime140_1.dll"]);
        assert_eq!(std::fs::read_to_string(dst.join("msvcp140.dll")).unwrap(), "newer");
        assert_eq!(std::fs::read_to_string(dst.join("vcruntime140.dll")).unwrap(), "bundled");
        assert!(copy_missing(&src, &dst).is_empty());
    }

    #[test]
    fn missing_source_files_are_skipped() {
        let t = tempfile::tempdir().unwrap();
        let (src, dst) = (t.path().join("src"), t.path().join("rt"));
        fill(&src, &["vcruntime140.dll"], "x");
        fill(&dst, &[], "");
        assert_eq!(copy_missing(&src, &dst), vec!["vcruntime140.dll"]);
    }

    #[test]
    fn exit_code_message_names_the_dlls() {
        assert_eq!(STATUS_DLL_NOT_FOUND, -1073741515);
        let (m, _) = missing_dll_message();
        assert!(m.contains("vcruntime140.dll") && m.contains("msvcp140.dll") && m.contains("tried to copy"));
        assert!(is_missing_dll_error(&m));
        let (m2, h) = crate::supervisor::diagnose("", Some(-1073741515));
        assert!(m2.contains("Visual C++") && h.is_some());
    }
}

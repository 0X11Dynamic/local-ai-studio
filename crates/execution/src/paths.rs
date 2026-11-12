//! Filesystem boundaries for the local environment: allowed roots and
//! protected folders. A path is usable only if, after resolving `..` and any
//! symlinks or junctions in the part that already exists, it lies inside an
//! allowed root and outside every protected folder.

use std::path::{Component, Path, PathBuf};

use app_core::{AppError, AppResult};

#[derive(Debug, Clone)]
pub struct RootPolicy {
    pub roots: Vec<PathBuf>,
    pub protected: Vec<PathBuf>,
}

/// Lower-case, `\`-separated, no trailing separator: for prefix comparison
/// on case-insensitive Windows filesystems.
pub(crate) fn key(p: &Path) -> String {
    let s = p.to_string_lossy().replace('/', "\\");
    let s = s.strip_prefix(r"\\?\").unwrap_or(&s).to_string();
    s.trim_end_matches('\\').to_lowercase()
}

fn inside(path: &Path, dir: &Path) -> bool {
    let (p, d) = (key(path), key(dir));
    p == d || p.starts_with(&format!("{d}\\"))
}

/// Remove `.` and `..` lexically (after which `..` cannot escape the root
/// of an absolute path).
pub(crate) fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Resolve symlinks/junctions in the longest existing prefix, keep the rest.
pub(crate) fn real(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        match (existing.file_name().map(|n| n.to_os_string()), existing.parent()) {
            (Some(n), Some(parent)) => {
                rest.push(n);
                existing = parent.to_path_buf();
            }
            _ => break,
        }
    }
    let mut base = std::fs::canonicalize(&existing).unwrap_or(existing);
    for n in rest.into_iter().rev() {
        base.push(n);
    }
    // Strip the verbatim prefix canonicalize adds on Windows.
    let s = base.to_string_lossy().to_string();
    match s.strip_prefix(r"\\?\") {
        Some(t) if !t.starts_with("UNC\\") => PathBuf::from(t),
        _ => base,
    }
}

/// Folders an agent must never touch, whatever the roots say.
pub fn default_protected() -> Vec<PathBuf> {
    let mut v = vec![];
    for var in ["SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"] {
        if let Ok(p) = std::env::var(var) {
            v.push(PathBuf::from(p));
        }
    }
    if let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
        for sub in [".ssh", ".gnupg", "AppData\\Roaming\\Microsoft\\Credentials", "AppData\\Local\\Microsoft\\Credentials", "AppData\\Roaming\\Microsoft\\Protect"] {
            v.push(home.join(sub));
        }
    }
    if let Ok(a) = std::env::var("APPDATA") {
        // The app's own database, settings, runtimes and logs.
        v.push(PathBuf::from(a).join("LocalAIStudio"));
    }
    v
}

impl RootPolicy {
    pub fn new(roots: Vec<PathBuf>, extra_protected: Vec<PathBuf>) -> Self {
        let mut protected = default_protected();
        protected.extend(extra_protected);
        Self { roots: roots.into_iter().map(|r| real(&lexical(&r))).collect(), protected: protected.into_iter().map(|p| real(&lexical(&p))).collect() }
    }

    /// Check and normalize a path. Relative paths are resolved against the
    /// first root.
    pub fn resolve(&self, input: &str) -> AppResult<PathBuf> {
        if input.trim().is_empty() {
            return Err(AppError::InvalidInput("path is empty".into()));
        }
        if self.roots.is_empty() {
            return Err(AppError::PermissionDenied("this environment has no allowed folders; add one in Execution Environments".into()));
        }
        // Refuse network, device and verbatim forms BEFORE any filesystem call:
        // even checking whether `//host/share` exists makes Windows contact the
        // host over SMB (and send the user's NTLM credentials).
        let norm = input.trim().replace('/', "\\");
        if norm.starts_with(r"\\") || norm.starts_with(r"\??\") {
            return Err(AppError::PermissionDenied("network (UNC), device and \\\\?\\ paths are not allowed".into()));
        }
        let p = Path::new(&norm);
        for c in p.components() {
            match c {
                Component::Prefix(px) if !matches!(px.kind(), std::path::Prefix::Disk(_)) => {
                    return Err(AppError::PermissionDenied("network (UNC), device and \\\\?\\ paths are not allowed".into()));
                }
                // `name:stream` is an NTFS alternate data stream: a hiding place
                // with no legitimate use for the agent.
                Component::Normal(n) if n.to_string_lossy().contains(':') => {
                    return Err(AppError::PermissionDenied(format!("\"{}\" uses ':' (an alternate data stream); not allowed", n.to_string_lossy())));
                }
                _ => {}
            }
        }
        let abs = if p.is_absolute() { p.to_path_buf() } else { self.roots[0].join(p) };
        if abs.to_string_lossy().starts_with(r"\\") {
            return Err(AppError::PermissionDenied("network (UNC) paths are not allowed".into()));
        }
        // A drive-relative input (`C:secret`) joins to itself; it has no root.
        if !abs.has_root() {
            return Err(AppError::PermissionDenied("drive-relative paths like C:file are not allowed; use a full path".into()));
        }
        let resolved = real(&lexical(&abs));
        if let Some(prot) = self.protected.iter().find(|d| inside(&resolved, d)) {
            return Err(AppError::PermissionDenied(format!("{} is inside a protected folder ({})", resolved.display(), prot.display())));
        }
        if !self.roots.iter().any(|r| inside(&resolved, r)) {
            return Err(AppError::PermissionDenied(format!(
                "{} is outside the allowed folders ({})",
                resolved.display(),
                self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")
            )));
        }
        Ok(resolved)
    }

    /// `resolve` for a path the file tools will create, change, move or
    /// delete: never inside a `.git` folder or a `.git` file (hooks, config
    /// and the worktree links the app's own git calls trust).
    pub fn resolve_write(&self, input: &str) -> AppResult<PathBuf> {
        let p = self.resolve(input)?;
        if p.components().any(|c| matches!(c, Component::Normal(n) if is_dot_git(&n.to_string_lossy()))) {
            return Err(AppError::PermissionDenied(format!("{} is inside git's own data (.git); the file tools do not change it. Use git commands instead", p.display())));
        }
        Ok(p)
    }

    /// `resolve_write` for a folder moved or deleted as a whole: also not a
    /// folder that holds a repository's `.git`.
    pub fn resolve_move_or_delete(&self, input: &str) -> AppResult<PathBuf> {
        let p = self.resolve_write(input)?;
        if std::fs::symlink_metadata(p.join(".git")).is_ok() {
            return Err(AppError::PermissionDenied(format!("{} holds a git repository or worktree (.git); the file tools do not move or delete it", p.display())));
        }
        Ok(p)
    }
}

/// `.git`, also as Windows reads it (case-insensitive, trailing dots and
/// spaces dropped: `.GIT.` and `.git ` name the same entry).
fn is_dot_git(name: &str) -> bool {
    name.trim_end_matches(['.', ' ']).eq_ignore_ascii_case(".git")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_escapes_and_protected_folders() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        let outside = d.path().join("secret");
        std::fs::create_dir_all(&outside).unwrap();
        let p = RootPolicy::new(vec![root.clone()], vec![root.join("sub").join("private")]);
        assert!(p.resolve(&root.join("a.txt").to_string_lossy()).is_ok());
        assert!(p.resolve("sub/new/file.txt").is_ok(), "relative paths resolve inside the first root");
        assert!(p.resolve(&root.join("..").join("secret").to_string_lossy()).is_err(), ".. escape");
        assert!(p.resolve("../secret/x").is_err());
        assert!(p.resolve(&outside.to_string_lossy()).is_err());
        assert!(p.resolve(&root.join("sub").join("private").join("x").to_string_lossy()).is_err(), "extra protected");
        assert!(p.resolve(r"C:\Windows\System32\drivers\etc\hosts").is_err(), "system folder");
        assert!(p.resolve(r"\\server\share\x").is_err(), "UNC");
        // Case-insensitive containment on Windows.
        let upper = root.join("A.TXT").to_string_lossy().to_uppercase();
        assert!(p.resolve(&upper).is_ok(), "{upper}");
        assert!(RootPolicy::new(vec![], vec![]).resolve("x").is_err(), "no roots, no access");
    }

    #[test]
    fn git_data_is_never_written() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        std::fs::create_dir_all(root.join("repo").join(".git").join("hooks")).unwrap();
        std::fs::create_dir_all(root.join("wt")).unwrap();
        std::fs::write(root.join("wt").join(".git"), "gitdir: x\n").unwrap();
        let p = RootPolicy::new(vec![root.clone()], vec![]);
        for bad in [r"repo\.git\hooks\post-checkout", r"repo\.git\config", r"repo\.GIT\hooks\x", r"repo\.git.\hooks\x", r"repo\.git \config", r"wt\.git", r"new\.git\x", r"repo\.git"] {
            let e = p.resolve_write(bad).unwrap_err().to_string();
            assert!(e.contains(".git"), "{bad}: {e}");
            assert!(p.resolve(bad).is_ok(), "reading {bad} stays allowed");
        }
        assert!(p.resolve_write(r"repo\src\main.rs").is_ok());
        assert!(p.resolve_write(r"repo\.github\workflows\ci.yml").is_ok(), ".github is not .git");
        assert!(p.resolve_write(r"repo\.gitignore").is_ok());
        // A repository or worktree folder is not moved or deleted as a whole.
        assert!(p.resolve_move_or_delete("repo").is_err());
        assert!(p.resolve_move_or_delete("wt").is_err());
        assert!(p.resolve_move_or_delete(r"repo\src").is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn junction_out_of_the_root_is_caught() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        let outside = d.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let link = root.join("link");
        let ok = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J", &link.to_string_lossy(), &outside.to_string_lossy()])
            .output()
            .unwrap()
            .status
            .success();
        assert!(ok, "mklink failed");
        let p = RootPolicy::new(vec![root.clone()], vec![]);
        assert!(p.resolve(&link.join("x.txt").to_string_lossy()).is_err(), "junction escape must be refused");
        std::process::Command::new("cmd").args(["/c", "rmdir", &link.to_string_lossy()]).output().unwrap();
    }
}

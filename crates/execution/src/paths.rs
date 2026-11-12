//! Filesystem boundaries for the local environment: allowed roots and
//! protected folders. After resolving `..` and any symlinks or junctions in
//! the part that already exists, a path is
//! - refused if it is inside a protected folder, or uses a UNC, device,
//!   verbatim, drive-relative or alternate-data-stream form (always);
//! - usable if it lies inside an allowed root;
//! - otherwise "outside": `classify` reports it so the app can ask the user,
//!   and `resolve` accepts it only when the user approved exactly that place
//!   for this one call (`approved`).

use std::path::{Component, Path, PathBuf};

use app_core::{AppError, AppResult};

#[derive(Debug, Clone)]
pub struct RootPolicy {
    pub roots: Vec<PathBuf>,
    pub protected: Vec<PathBuf>,
    /// Places outside the roots the user approved for one call (This PC with
    /// "Reach outside allowed folders: Ask each time"). A path inside one of
    /// them is usable. Empty for every environment's normal policy.
    pub approved: Vec<PathBuf>,
}

/// Where a checked path lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    Inside,
    Outside,
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

/// Folders an agent must never touch, whatever the roots say and whatever
/// the user approves: Windows, Program Files, ProgramData\Microsoft (start-up
/// folders, Defender, policies), the user's keys and credential stores, and
/// the app's own data (database, runtimes, settings) and WebView profile.
pub fn default_protected() -> Vec<PathBuf> {
    let mut v = vec![];
    for var in ["SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"] {
        if let Ok(p) = std::env::var(var) {
            v.push(PathBuf::from(p));
        }
    }
    if let Ok(p) = std::env::var("ProgramData") {
        v.push(PathBuf::from(p).join("Microsoft"));
    }
    if let Ok(p) = std::env::var("LOCALAPPDATA") {
        v.push(PathBuf::from(p).join("com.localaistudio.app"));
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
        Self {
            roots: roots.into_iter().map(|r| real(&lexical(&r))).collect(),
            protected: protected.into_iter().map(|p| real(&lexical(&p))).collect(),
            approved: vec![],
        }
    }

    /// This policy plus places the user approved for one call.
    pub fn with_approved(&self, places: &[PathBuf]) -> Self {
        let mut p = self.clone();
        p.approved = places.iter().map(|x| real(&lexical(x))).collect();
        p
    }

    /// Check and normalize a path. Relative paths are resolved against the
    /// first root. Outside the roots only an approved place is accepted.
    pub fn resolve(&self, input: &str) -> AppResult<PathBuf> {
        let (p, reach) = self.classify(input)?;
        self.accept(p, reach)
    }

    fn accept(&self, p: PathBuf, reach: Reach) -> AppResult<PathBuf> {
        if reach == Reach::Inside || self.approved.iter().any(|a| inside(&p, a)) {
            return Ok(p);
        }
        if self.roots.is_empty() {
            return Err(AppError::PermissionDenied("this environment has no allowed folders; add one in Execution Environments".into()));
        }
        Err(AppError::PermissionDenied(format!(
            "{} is outside the allowed folders ({})",
            p.display(),
            self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")
        )))
    }

    /// Normalize a path and say whether it is inside the roots. Errors are
    /// the places that are refused whatever the user says: protected
    /// folders and UNC/device/verbatim/drive-relative/stream forms.
    pub fn classify(&self, input: &str) -> AppResult<(PathBuf, Reach)> {
        if input.trim().is_empty() {
            return Err(AppError::InvalidInput("path is empty".into()));
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
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            match self.roots.first() {
                Some(r) => r.join(p),
                None => return Err(AppError::PermissionDenied("this environment has no allowed folders; add one in Execution Environments, or use a full path".into())),
            }
        };
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
        let reach = if self.roots.iter().any(|r| inside(&resolved, r)) { Reach::Inside } else { Reach::Outside };
        Ok((resolved, reach))
    }

    /// `classify` for a path the file tools will create, change, move or
    /// delete: never inside a `.git` folder or a `.git` file (hooks, config
    /// and the worktree links the app's own git calls trust).
    pub fn classify_write(&self, input: &str) -> AppResult<(PathBuf, Reach)> {
        let (p, reach) = self.classify(input)?;
        if p.components().any(|c| matches!(c, Component::Normal(n) if is_dot_git(&n.to_string_lossy()))) {
            return Err(AppError::PermissionDenied(format!("{} is inside git's own data (.git); the file tools do not change it. Use git commands instead", p.display())));
        }
        Ok((p, reach))
    }

    /// `classify_write` for a file or folder moved or deleted as a whole:
    /// also not a folder that holds a repository's `.git`, a whole drive,
    /// the user's home folder or the folder of all users' homes.
    pub fn classify_move_or_delete(&self, input: &str) -> AppResult<(PathBuf, Reach)> {
        let (p, reach) = self.classify_write(input)?;
        if std::fs::symlink_metadata(p.join(".git")).is_ok() {
            return Err(AppError::PermissionDenied(format!("{} holds a git repository or worktree (.git); the file tools do not move or delete it", p.display())));
        }
        if p.parent().is_none() {
            return Err(AppError::PermissionDenied(format!("{} is a whole drive; the file tools do not move or delete it", p.display())));
        }
        if let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
            let home = real(&lexical(&home));
            if key(&p) == key(&home) || home.parent().is_some_and(|h| key(&p) == key(h)) {
                return Err(AppError::PermissionDenied(format!("{} is a home folder; the file tools do not move or delete it", p.display())));
            }
        }
        Ok((p, reach))
    }

    /// `resolve` with the `classify_write` rules.
    pub fn resolve_write(&self, input: &str) -> AppResult<PathBuf> {
        let (p, reach) = self.classify_write(input)?;
        self.accept(p, reach)
    }

    /// `resolve` with the `classify_move_or_delete` rules.
    pub fn resolve_move_or_delete(&self, input: &str) -> AppResult<PathBuf> {
        let (p, reach) = self.classify_move_or_delete(input)?;
        self.accept(p, reach)
    }

    /// Grant subjects that cover `place` for the tool `tool_id`: the folder
    /// itself and each parent up to the drive (lower case, so a grant matches
    /// whatever case a later call uses). The first is the one to record.
    pub fn grant_subjects(tool_id: &str, place: &Path) -> Vec<String> {
        let mut out = vec![];
        let mut cur = Some(place);
        while let Some(c) = cur {
            if c.as_os_str().is_empty() {
                break;
            }
            out.push(format!("{tool_id} outside {}", key(c)));
            cur = c.parent();
        }
        out
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

    #[test]
    fn outside_places_are_classified_and_need_an_approval() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        let docs = d.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(docs.join("sub")).unwrap();
        let p = RootPolicy::new(vec![root.clone()], vec![]);
        // Inside: unchanged.
        assert_eq!(p.classify("a.txt").unwrap().1, Reach::Inside);
        // Outside (absolute, `..`, another drive): reported, not refused...
        let f = docs.join("sub").join("a.txt");
        let (resolved, reach) = p.classify(&f.to_string_lossy()).unwrap();
        assert_eq!(reach, Reach::Outside);
        assert_eq!(key(&resolved), key(&f));
        assert_eq!(p.classify(r"..\docs\x.txt").unwrap().1, Reach::Outside);
        assert_eq!(p.classify(r"D:\data\x.txt").unwrap().1, Reach::Outside);
        // ...but resolve accepts them only inside an approved place.
        assert!(p.resolve(&f.to_string_lossy()).is_err());
        let ok = p.with_approved(std::slice::from_ref(&docs));
        assert!(ok.resolve(&f.to_string_lossy()).is_ok(), "inside the approved folder");
        assert!(ok.resolve(&d.path().join("elsewhere.txt").to_string_lossy()).is_err(), "a sibling of the approved folder");
        assert!(ok.resolve_write(&docs.join(".git").join("config").to_string_lossy()).is_err(), ".git stays protected outside too");
        // No roots: full paths are outside, relative ones are refused.
        let none = RootPolicy::new(vec![], vec![]);
        assert_eq!(none.classify(&f.to_string_lossy()).unwrap().1, Reach::Outside);
        assert!(none.classify("x").is_err());
    }

    #[test]
    fn always_blocked_places_stay_blocked() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        std::fs::create_dir_all(&root).unwrap();
        let p = RootPolicy::new(vec![root.clone()], vec![d.path().join("appdata")]).with_approved(&[PathBuf::from(r"C:\")]);
        let mut blocked = vec![r"C:\Windows\win.ini".to_string(), r"\\server\share\x".into(), r"\\?\C:\x".into(), r"C:secret".into(), "a.txt:stream".into()];
        blocked.push(d.path().join("appdata").join("app.db").to_string_lossy().into_owned());
        for var in ["ProgramFiles", "APPDATA"] {
            if let Ok(v) = std::env::var(var) {
                blocked.push(Path::new(&v).join(if var == "APPDATA" { r"LocalAIStudio\app.db" } else { "x.exe" }).to_string_lossy().into_owned());
            }
        }
        if let Ok(pd) = std::env::var("ProgramData") {
            blocked.push(Path::new(&pd).join(r"Microsoft\Windows\Start Menu\Programs\StartUp\x.lnk").to_string_lossy().into_owned());
            // The rest of ProgramData is an ordinary outside place.
            assert_eq!(p.classify(&Path::new(&pd).join("SomeApp").join("x.json").to_string_lossy()).unwrap().1, Reach::Outside);
        }
        for b in &blocked {
            assert!(p.classify(b).is_err(), "{b} must be blocked");
            assert!(p.resolve(b).is_err(), "{b} must be blocked even under an approval of C:\\");
        }
        // Whole drives and home folders are never moved or deleted.
        assert!(p.classify_move_or_delete(r"C:\").is_err());
        if let Ok(home) = std::env::var("USERPROFILE") {
            assert!(p.classify_move_or_delete(&home).is_err());
            assert!(p.classify_move_or_delete(&Path::new(&home).join("Documents").join("old.txt").to_string_lossy()).is_ok());
        }
    }

    #[test]
    fn grant_subjects_cover_the_folder_and_its_parents() {
        let s = RootPolicy::grant_subjects("builtin:local/fs_read", Path::new(r"C:\Users\Me\Docs"));
        assert_eq!(
            s,
            vec![
                r"builtin:local/fs_read outside c:\users\me\docs".to_string(),
                r"builtin:local/fs_read outside c:\users\me".into(),
                r"builtin:local/fs_read outside c:\users".into(),
                r"builtin:local/fs_read outside c:".into(),
            ]
        );
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
        // Classified by where it really goes, so an approval card shows the real target.
        let (real_target, reach) = p.classify(&link.join("x.txt").to_string_lossy()).unwrap();
        assert_eq!((key(&real_target), reach), (key(&outside.join("x.txt")), Reach::Outside));
        std::process::Command::new("cmd").args(["/c", "rmdir", &link.to_string_lossy()]).output().unwrap();
    }
}

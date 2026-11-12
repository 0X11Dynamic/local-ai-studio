//! Git worktrees for subagents that edit files: each such child works in
//! `<repo>\.las-worktrees\<id>` on its own branch `las/<id>`, so the user's
//! main tree is untouched until the parent merges. Every git command goes
//! through the Job Object process runner. Nothing here ever forces: a
//! worktree with changes or an unmerged branch is kept and reported.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use app_core::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecResult;
use crate::process::{RunOptions, run};

/// Folder (inside the repo) that holds the subagents' worktrees.
pub const WORKTREE_DIR: &str = ".las-worktrees";
/// Branch prefix for subagent worktrees.
pub const BRANCH_PREFIX: &str = "las/";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Worktree {
    /// Top folder of the main working tree.
    pub repo: String,
    /// The worktree folder, `<repo>\.las-worktrees\<id>`.
    pub path: String,
    /// `las/<id>`
    pub branch: String,
    /// The commit it was created from (to tell "no commits of its own").
    pub base: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Cleanup {
    Removed,
    /// Kept on disk; the reason says why (changes, unmerged commits, git error).
    Kept { reason: String },
}

/// An empty folder this process made: every app git call points
/// `core.hooksPath` at it, so no hook in the repository (which an agent may
/// have written) ever runs with the app's rights.
fn no_hooks_dir() -> AppResult<PathBuf> {
    static DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    let made = DIR.get_or_init(|| {
        // Empty ones left by earlier runs go (remove_dir never deletes
        // anything inside; a running process remakes its own if needed).
        if let Ok(rd) = std::fs::read_dir(std::env::temp_dir()) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().starts_with("las-git-nohooks-") {
                    let _ = std::fs::remove_dir(e.path());
                }
            }
        }
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        for n in 0..100u32 {
            let d = std::env::temp_dir().join(format!("las-git-nohooks-{}-{nanos}-{n}", std::process::id()));
            match std::fs::create_dir(&d) {
                Ok(()) => return Ok(d),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err("no free folder name".into())
    });
    let d = made.as_ref().map_err(|e| AppError::Runtime(format!("git was not run: could not make an empty hooks folder ({e})")))?;
    if !d.is_dir() {
        std::fs::create_dir_all(d)?;
    }
    if std::fs::read_dir(d)?.next().is_some() {
        return Err(AppError::Runtime(format!("git was not run: the app's empty hooks folder {} is no longer empty", d.display())));
    }
    Ok(d.clone())
}

/// Run git with `args` in `cwd`. A missing git is a clear error. Repository
/// hooks and the fsmonitor helper are switched off for every call.
pub async fn git(cwd: &Path, args: &[&str]) -> AppResult<ExecResult> {
    let hooks = no_hooks_dir()?.to_string_lossy().replace('\\', "/");
    let mut full: Vec<String> = vec!["-c".into(), format!("core.hooksPath={hooks}"), "-c".into(), "core.fsmonitor=false".into()];
    full.extend(args.iter().map(|s| s.to_string()));
    let args = full;
    let mut env = BTreeMap::new();
    // Never wait for a credential or editor prompt (there is no console).
    env.insert("GIT_TERMINAL_PROMPT".to_string(), "0".to_string());
    env.insert("GIT_EDITOR".to_string(), "true".to_string());
    // Read-only commands (status) do not rewrite the index behind the user.
    env.insert("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string());
    let opts = RunOptions { program: "git", args: &args, cwd: Some(cwd), env: &env, stdin: None, timeout: Duration::from_secs(120), max_output_bytes: 256 * 1024 };
    run(opts, &CancellationToken::new(), &|_| {}).await.map_err(|e| match e {
        AppError::Runtime(m) if m.starts_with("could not start git") => {
            AppError::NotFound("git was not found on this PC (install Git for Windows and make sure git is on PATH), so worktree isolation is not available".into())
        }
        e => e,
    })
}

fn said(r: &ExecResult) -> String {
    let t = if r.stderr.trim().is_empty() { r.stdout.trim() } else { r.stderr.trim() };
    t.lines().take(4).collect::<Vec<_>>().join(" ")
}

/// Run git and return trimmed stdout, or an error with what git said.
async fn git_ok(cwd: &Path, args: &[&str]) -> AppResult<String> {
    let r = git(cwd, args).await?;
    if !r.succeeded() {
        return Err(AppError::Runtime(format!("git {} failed: {}", args.first().copied().unwrap_or(""), said(&r))));
    }
    Ok(r.stdout.trim().to_string())
}

/// Same file or folder (case-insensitive, links resolved where they exist).
fn same_path(a: &Path, b: &Path) -> bool {
    crate::paths::key(&crate::paths::real(&crate::paths::lexical(a))) == crate::paths::key(&crate::paths::real(&crate::paths::lexical(b)))
}

/// The path a `gitdir: <path>` file names (relative to the file's folder).
fn read_link_file(file: &Path) -> AppResult<PathBuf> {
    let text = std::fs::read_to_string(file).map_err(|e| AppError::Runtime(format!("cannot read {}: {e}", file.display())))?;
    let first = text.lines().next().unwrap_or("").trim();
    let p = first.strip_prefix("gitdir:").map(str::trim).unwrap_or(first);
    if p.is_empty() {
        return Err(AppError::Runtime(format!("{} names no folder", file.display())));
    }
    let p = to_native(p);
    Ok(if p.is_absolute() { p } else { file.parent().unwrap_or(Path::new("")).join(p) })
}

/// The id of an app worktree, checking that it really is
/// `<repo>\.las-worktrees\<id>` on branch `las/<id>`.
fn worktree_id(wt: &Worktree) -> AppResult<String> {
    let path = Path::new(&wt.path);
    let id = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    check_id(&id)?;
    if !same_path(path, &Path::new(&wt.repo).join(WORKTREE_DIR).join(&id)) {
        return Err(AppError::InvalidInput(format!("{} is not a subagent worktree folder of {}", wt.path, wt.repo)));
    }
    if wt.branch != format!("{BRANCH_PREFIX}{id}") {
        return Err(AppError::InvalidInput(format!("{} is not the branch of worktree {id}", wt.branch)));
    }
    Ok(id)
}

/// Git's record of a worktree must point back at the worktree's own `.git`
/// file: `<common>\worktrees\<id>\gitdir` -> `<path>\.git`.
fn admin_points_at(admin: &Path, path: &Path) -> AppResult<()> {
    let back = read_link_file(&admin.join("gitdir"))?;
    if !same_path(&back, &path.join(".git")) {
        return Err(AppError::PermissionDenied(format!("git's record {} points at {}, not at this worktree", admin.display(), back.display())));
    }
    Ok(())
}

/// Before running git inside a worktree: its `.git` must be a file that
/// points at `<common>\worktrees\<id>` (and that record back at it), so a
/// replaced `.git` cannot make git read another repository's config.
fn check_worktree_link(path: &Path, admin: &Path) -> AppResult<()> {
    let dotgit = path.join(".git");
    let meta = std::fs::symlink_metadata(&dotgit).map_err(|_| AppError::PermissionDenied(format!("{} has no .git file", path.display())))?;
    if !meta.is_file() {
        return Err(AppError::PermissionDenied(format!("{} is not a linked worktree (its .git is not a plain file)", path.display())));
    }
    let target = read_link_file(&dotgit)?;
    if !same_path(&target, admin) {
        return Err(AppError::PermissionDenied(format!("the .git file of {} points at {}, not at {}", path.display(), target.display(), admin.display())));
    }
    admin_points_at(admin, path)
}

/// A commit the cleanup may forget: the base itself, or one already in the
/// main tree's HEAD.
async fn is_safe_commit(repo: &Path, commit: &str, base: &str, main_head: &str) -> AppResult<bool> {
    if commit == base || commit == main_head {
        return Ok(true);
    }
    let r = git(repo, &["merge-base", "--is-ancestor", commit, main_head]).await?;
    match r.exit_code {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(AppError::Runtime(format!("git merge-base failed: {}", said(&r)))),
    }
}

fn to_native(p: &str) -> PathBuf {
    PathBuf::from(p.replace('/', "\\"))
}

/// Top folder of the git working tree that contains `dir`.
pub async fn repo_root(dir: &Path) -> AppResult<PathBuf> {
    if !dir.is_dir() {
        return Err(AppError::NotFound(format!("{} is not an existing folder", dir.display())));
    }
    let r = git(dir, &["rev-parse", "--show-toplevel"]).await?;
    if !r.succeeded() {
        return Err(AppError::InvalidInput(format!(
            "{} is not inside a git repository, so a worktree cannot be made (git said: {}). Start the subagent without isolation, or make the folder a git repository first",
            dir.display(),
            said(&r)
        )));
    }
    Ok(to_native(r.stdout.trim()))
}

fn check_id(id: &str) -> AppResult<()> {
    if id.is_empty() || id.len() > 40 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.starts_with('-') {
        return Err(AppError::InvalidInput(format!("bad worktree id {id:?}")));
    }
    Ok(())
}

/// The repo's shared .git folder (absolute), for `info/exclude`.
async fn common_dir(repo: &Path) -> AppResult<PathBuf> {
    let d = to_native(&git_ok(repo, &["rev-parse", "--git-common-dir"]).await?);
    Ok(if d.is_absolute() { d } else { repo.join(d) })
}

/// Git lists the worktrees folder as untracked in the main tree; hide it with
/// the repo's local-only `info/exclude` (never the user's .gitignore).
async fn exclude_worktree_dir(repo: &Path) -> AppResult<()> {
    let st = git_ok(repo, &["status", "--porcelain", "--untracked-files=normal", "--", WORKTREE_DIR]).await?;
    if st.is_empty() {
        return Ok(());
    }
    let info = common_dir(repo).await?.join("info");
    std::fs::create_dir_all(&info)?;
    let path = info.join("exclude");
    let old = std::fs::read_to_string(&path).unwrap_or_default();
    let line = format!("/{WORKTREE_DIR}/");
    if !old.lines().any(|l| l.trim() == line || l.trim() == format!("{WORKTREE_DIR}/")) {
        let sep = if old.is_empty() || old.ends_with('\n') { "" } else { "\n" };
        std::fs::write(&path, format!("{old}{sep}# Local AI Studio subagent worktrees\n{line}\n"))?;
    }
    Ok(())
}

/// `git worktree add <repo>\.las-worktrees\<id> -b las/<id>` from the repo's
/// current HEAD.
pub async fn add_worktree(repo: &Path, id: &str) -> AppResult<Worktree> {
    let wt = plan_worktree(repo, id).await?;
    create_worktree(&wt).await?;
    Ok(wt)
}

/// What `add_worktree` would make, without changing anything, so the app
/// can record it before it exists (nothing is ever left unrecorded).
pub async fn plan_worktree(repo: &Path, id: &str) -> AppResult<Worktree> {
    check_id(id)?;
    let head = git(repo, &["rev-parse", "--verify", "HEAD"]).await?;
    if !head.succeeded() {
        return Err(AppError::InvalidInput(format!("the git repository at {} has no commits yet, so a worktree cannot be made; commit something first or start the subagent without isolation", repo.display())));
    }
    let base = head.stdout.trim().to_string();
    let path = repo.join(WORKTREE_DIR).join(id);
    Ok(Worktree { repo: repo.to_string_lossy().into_owned(), path: path.to_string_lossy().into_owned(), branch: format!("{BRANCH_PREFIX}{id}"), base })
}

/// Make a planned worktree: `git worktree add <path> -b las/<id> <base>`.
pub async fn create_worktree(wt: &Worktree) -> AppResult<()> {
    worktree_id(wt)?;
    let repo = Path::new(&wt.repo);
    git_ok(repo, &["worktree", "add", &wt.path, "-b", &wt.branch, &wt.base]).await?;
    exclude_worktree_dir(repo).await
}

/// Registered worktrees: (folder, branch) — the main tree first.
pub async fn list_worktrees(repo: &Path) -> AppResult<Vec<(PathBuf, Option<String>)>> {
    let out = git_ok(repo, &["worktree", "list", "--porcelain"]).await?;
    let mut v: Vec<(PathBuf, Option<String>)> = vec![];
    for line in out.lines() {
        if let Some(p) = line.strip_prefix("worktree ") {
            v.push((to_native(p), None));
        } else if let Some(b) = line.strip_prefix("branch ")
            && let Some(last) = v.last_mut()
        {
            last.1 = Some(b.strip_prefix("refs/heads/").unwrap_or(b).to_string());
        }
    }
    Ok(v)
}

/// No modified, staged or untracked files.
pub async fn is_clean(worktree: &Path) -> AppResult<bool> {
    Ok(git_ok(worktree, &["status", "--porcelain", "--untracked-files=normal"]).await?.is_empty())
}

async fn count(repo: &Path, range: &str) -> AppResult<u64> {
    let n = git_ok(repo, &["rev-list", "--count", range]).await?;
    n.parse().map_err(|_| AppError::Runtime(format!("unexpected git rev-list output: {n}")))
}

/// Every commit of `branch` is in the main tree's HEAD.
pub async fn is_merged(repo: &Path, branch: &str) -> AppResult<bool> {
    Ok(count(repo, &format!("HEAD..{branch}")).await? == 0)
}

/// The branch has no commits since `base`.
pub async fn has_no_commits(repo: &Path, branch: &str, base: &str) -> AppResult<bool> {
    Ok(count(repo, &format!("{base}..{branch}")).await? == 0)
}

/// Remove a subagent's worktree and branch, but only if the worktree is
/// clean and the branch is merged into HEAD or has no commits of its own.
/// Never forced: anything else is kept, with the reason.
pub async fn cleanup_worktree(wt: &Worktree) -> Cleanup {
    match try_cleanup(wt).await {
        Ok(c) => c,
        Err(e) => Cleanup::Kept { reason: e.to_string() },
    }
}

/// The commit a worktree whose folder is gone had checked out, from git's
/// record `<admin>\HEAD` (None: its branch no longer exists).
async fn admin_head(repo: &Path, admin: &Path) -> AppResult<Option<String>> {
    let head = std::fs::read_to_string(admin.join("HEAD")).map_err(|e| AppError::Runtime(format!("cannot read {}: {e}", admin.join("HEAD").display())))?;
    let head = head.trim();
    match head.strip_prefix("ref:") {
        Some(r) => {
            let r = git(repo, &["rev-parse", "--verify", "--quiet", r.trim()]).await?;
            Ok(r.succeeded().then(|| r.stdout.trim().to_string()))
        }
        None => Ok(Some(head.to_string())),
    }
}

async fn try_cleanup(wt: &Worktree) -> AppResult<Cleanup> {
    let repo = Path::new(&wt.repo);
    let path = Path::new(&wt.path);
    if !repo.is_dir() {
        return Ok(Cleanup::Kept { reason: format!("the repository {} is gone", wt.repo) });
    }
    if !wt.branch.starts_with(BRANCH_PREFIX) {
        return Ok(Cleanup::Kept { reason: format!("{} is not a subagent branch", wt.branch) });
    }
    let id = worktree_id(wt)?;
    let admin = common_dir(repo).await?.join("worktrees").join(&id);
    let main_head = git_ok(repo, &["rev-parse", "--verify", "HEAD"]).await?;
    let branch_exists = git(repo, &["rev-parse", "--verify", "--quiet", &format!("refs/heads/{}", wt.branch)]).await?.succeeded();
    if branch_exists {
        let merged = is_merged(repo, &wt.branch).await?;
        let empty = !merged && has_no_commits(repo, &wt.branch, &wt.base).await.unwrap_or(false);
        if !merged && !empty {
            return Ok(Cleanup::Kept { reason: format!("branch {} has commits that are not merged into the main tree's HEAD", wt.branch) });
        }
    }
    if std::fs::symlink_metadata(path).is_ok() {
        if !path.is_dir() {
            return Ok(Cleanup::Kept { reason: format!("{} is not a folder", wt.path) });
        }
        if let Err(e) = check_worktree_link(path, &admin) {
            return Ok(Cleanup::Kept { reason: format!("not touched: {e}") });
        }
        // What the worktree has checked out counts too (another branch, a
        // detached HEAD, or a deleted las/ branch must not lose commits).
        let head = git_ok(path, &["rev-parse", "--verify", "HEAD"]).await?;
        if !is_safe_commit(repo, &head, &wt.base, &main_head).await? {
            return Ok(Cleanup::Kept { reason: format!("the worktree's HEAD ({}) has commits that are not in the main tree's HEAD", head.chars().take(10).collect::<String>()) });
        }
        let st = git_ok(path, &["status", "--porcelain", "--ignored", "--untracked-files=normal"]).await?;
        if st.lines().any(|l| !l.starts_with("!!")) {
            return Ok(Cleanup::Kept { reason: "the worktree has uncommitted or untracked changes".into() });
        }
        if !st.is_empty() {
            // `git worktree remove` would delete ignored files (build output, .env ...).
            return Ok(Cleanup::Kept { reason: "the worktree has ignored files (for example build output or local settings); removing it would delete them".into() });
        }
        let r = git(repo, &["worktree", "remove", &wt.path]).await?;
        if !r.succeeded() {
            return Ok(Cleanup::Kept { reason: format!("git worktree remove refused: {}", said(&r)) });
        }
    } else if std::fs::symlink_metadata(&admin).is_ok() {
        // The folder was deleted by hand: drop only git's record of this
        // worktree (never a repo-wide prune), after checking it is ours.
        if let Err(e) = admin_points_at(&admin, path) {
            return Ok(Cleanup::Kept { reason: format!("not touched: {e}") });
        }
        if admin.join("locked").exists() {
            return Ok(Cleanup::Kept { reason: format!("git's record of the worktree is locked ({})", admin.display()) });
        }
        if let Some(head) = admin_head(repo, &admin).await?
            && !is_safe_commit(repo, &head, &wt.base, &main_head).await?
        {
            return Ok(Cleanup::Kept { reason: format!("the deleted worktree's HEAD ({}) has commits that are not in the main tree's HEAD", head.chars().take(10).collect::<String>()) });
        }
        std::fs::remove_dir_all(&admin).map_err(|e| AppError::Runtime(format!("could not remove git's record {}: {e}", admin.display())))?;
    }
    if branch_exists {
        let r = git(repo, &["branch", "-d", &wt.branch]).await?;
        if !r.succeeded() {
            return Ok(Cleanup::Kept { reason: format!("the worktree was removed but branch {} was kept: {}", wt.branch, said(&r)) });
        }
    }
    Ok(Cleanup::Removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(dir: &Path, args: &[&str]) {
        let out = std::process::Command::new("git").args(["-c", "user.name=Test", "-c", "user.email=test@example.com"]).args(args).current_dir(dir).output().unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }

    fn temp_repo() -> tempfile::TempDir {
        let d = tempfile::Builder::new().prefix("las-git-test").tempdir().unwrap();
        sh(d.path(), &["init", "-q", "-b", "main"]);
        std::fs::write(d.path().join("a.txt"), "one\n").unwrap();
        sh(d.path(), &["add", "a.txt"]);
        sh(d.path(), &["commit", "-q", "-m", "init"]);
        d
    }

    fn same(a: &Path, b: &Path) -> bool {
        std::fs::canonicalize(a).unwrap() == std::fs::canonicalize(b).unwrap()
    }

    #[tokio::test]
    async fn worktree_add_list_merge_check_and_remove() {
        let d = temp_repo();
        let sub = d.path().join("src");
        std::fs::create_dir_all(&sub).unwrap();
        let root = repo_root(&sub).await.unwrap();
        assert!(same(&root, d.path()), "{root:?}");

        // Empty child: no commits -> removed with its branch.
        let a = add_worktree(&root, "aaa111").await.unwrap();
        assert_eq!(a.branch, "las/aaa111");
        assert!(Path::new(&a.path).join("a.txt").is_file());
        let list = list_worktrees(&root).await.unwrap();
        assert_eq!(list.len(), 2, "{list:?}");
        assert!(list.iter().any(|(p, b)| same(p, Path::new(&a.path)) && b.as_deref() == Some("las/aaa111")));
        // The worktrees folder is hidden from the main tree via info/exclude.
        assert!(is_clean(&root).await.unwrap(), "main tree must stay clean");
        let exclude = std::fs::read_to_string(root.join(".git").join("info").join("exclude")).unwrap();
        assert!(exclude.contains("/.las-worktrees/"));
        assert!(!root.join(".gitignore").exists(), "the user's .gitignore is never touched");

        // A child with an unmerged commit is kept; after merging it is removed.
        let b = add_worktree(&root, "bbb222").await.unwrap();
        std::fs::write(Path::new(&b.path).join("b.txt"), "two\n").unwrap();
        assert!(!is_clean(Path::new(&b.path)).await.unwrap());
        assert!(matches!(cleanup_worktree(&b).await, Cleanup::Kept { .. }), "dirty worktree is kept");
        sh(Path::new(&b.path), &["add", "b.txt"]);
        sh(Path::new(&b.path), &["commit", "-q", "-m", "child work"]);
        assert!(!is_merged(&root, &b.branch).await.unwrap());
        assert!(!has_no_commits(&root, &b.branch, &b.base).await.unwrap());
        match cleanup_worktree(&b).await {
            Cleanup::Kept { reason } => assert!(reason.contains("not merged"), "{reason}"),
            c => panic!("unmerged branch must be kept, got {c:?}"),
        }
        assert!(Path::new(&b.path).is_dir());
        sh(&root, &["merge", "-q", "--no-edit", &b.branch]);
        assert!(is_merged(&root, &b.branch).await.unwrap());
        assert_eq!(cleanup_worktree(&b).await, Cleanup::Removed);
        assert!(!Path::new(&b.path).exists());

        assert!(has_no_commits(&root, &a.branch, &a.base).await.unwrap());
        assert_eq!(cleanup_worktree(&a).await, Cleanup::Removed);
        assert!(!Path::new(&a.path).exists());
        assert_eq!(list_worktrees(&root).await.unwrap().len(), 1, "only the main tree is left");
        let branches = String::from_utf8(std::process::Command::new("git").args(["branch", "--list", "las/*"]).current_dir(&root).output().unwrap().stdout).unwrap();
        assert!(branches.trim().is_empty(), "branches deleted: {branches}");
        assert!(is_clean(&root).await.unwrap());
    }

    fn branches(root: &Path) -> String {
        String::from_utf8(std::process::Command::new("git").args(["branch", "--list", "las/*"]).current_dir(root).output().unwrap().stdout).unwrap()
    }

    #[tokio::test]
    async fn repo_hooks_and_fsmonitor_never_run() {
        let d = temp_repo();
        let root = d.path().to_path_buf();
        let marks = tempfile::Builder::new().prefix("las-hook-marker").tempdir().unwrap();
        let marker = marks.path().join("ran.txt");
        let m = marker.to_string_lossy().replace('\\', "/");
        let hook = format!("#!/bin/sh\necho ran >> \"{m}\"\nexit 0\n");
        let hooks = root.join(".git").join("hooks");
        for name in ["post-checkout", "reference-transaction", "pre-auto-gc", "post-index-change"] {
            std::fs::write(hooks.join(name), &hook).unwrap();
        }
        // The hook really runs for a plain git call (so the test means something).
        sh(&root, &["checkout", "-q", "-b", "probe"]);
        sh(&root, &["checkout", "-q", "main"]);
        sh(&root, &["branch", "-q", "-D", "probe"]);
        assert!(marker.exists(), "the test hook should run for plain git");
        std::fs::remove_file(&marker).unwrap();
        // An fsmonitor helper that would leave the same marker.
        let fsm = root.join("fsmonitor.sh");
        std::fs::write(&fsm, &hook).unwrap();
        sh(&root, &["config", "core.fsmonitor", &fsm.to_string_lossy().replace('\\', "/")]);
        std::fs::write(root.join(".git").join("info").join("exclude"), "fsmonitor.sh\n").unwrap();

        let a = add_worktree(&root, "hook111").await.unwrap();
        assert!(is_clean(&root).await.unwrap());
        assert!(is_clean(Path::new(&a.path)).await.unwrap());
        assert_eq!(cleanup_worktree(&a).await, Cleanup::Removed);
        assert!(!Path::new(&a.path).exists());
        assert!(!marker.exists(), "no repository hook or fsmonitor helper may run for the app's git calls");
        assert!(branches(&root).trim().is_empty());
    }

    #[tokio::test]
    async fn prune_is_only_for_our_own_worktree() {
        let d = temp_repo();
        let root = d.path().to_path_buf();
        // The user's own worktree, deleted by hand: git lists it as prunable.
        let users = tempfile::Builder::new().prefix("las-user-wt").tempdir().unwrap();
        let users_wt = users.path().join("mine");
        sh(&root, &["worktree", "add", "-q", &users_wt.to_string_lossy(), "-b", "mine"]);
        std::fs::remove_dir_all(&users_wt).unwrap();
        let a = add_worktree(&root, "gone111").await.unwrap();
        std::fs::remove_dir_all(&a.path).unwrap();
        assert_eq!(cleanup_worktree(&a).await, Cleanup::Removed);
        assert!(!root.join(".git").join("worktrees").join("gone111").exists(), "our record is removed");
        assert!(root.join(".git").join("worktrees").join("mine").exists(), "the user's stale worktree record is left alone");
        assert!(branches(&root).trim().is_empty());
        // A record that points somewhere else is not touched.
        let b = add_worktree(&root, "gone222").await.unwrap();
        std::fs::remove_dir_all(&b.path).unwrap();
        let admin = root.join(".git").join("worktrees").join("gone222");
        std::fs::write(admin.join("gitdir"), "C:/somewhere/else/.git\n").unwrap();
        match cleanup_worktree(&b).await {
            Cleanup::Kept { reason } => assert!(reason.contains("not at this worktree"), "{reason}"),
            c => panic!("{c:?}"),
        }
        assert!(admin.exists());
    }

    #[tokio::test]
    async fn cleanup_keeps_commits_off_the_branch_ignored_files_and_foreign_links() {
        let d = temp_repo();
        let root = d.path().to_path_buf();
        // A commit on a detached HEAD (not on las/<id>) is kept.
        let a = add_worktree(&root, "det111").await.unwrap();
        let ap = Path::new(&a.path);
        sh(ap, &["checkout", "-q", "--detach"]);
        std::fs::write(ap.join("d.txt"), "d\n").unwrap();
        sh(ap, &["add", "d.txt"]);
        sh(ap, &["commit", "-q", "-m", "detached work"]);
        match cleanup_worktree(&a).await {
            Cleanup::Kept { reason } => assert!(reason.contains("HEAD"), "{reason}"),
            c => panic!("detached commit must be kept, got {c:?}"),
        }
        assert!(ap.is_dir());

        // The las/ branch deleted and the work on another branch: still kept.
        let b = add_worktree(&root, "nob222").await.unwrap();
        let bp = Path::new(&b.path);
        sh(bp, &["checkout", "-q", "-b", "elsewhere"]);
        sh(&root, &["branch", "-q", "-D", &b.branch]);
        std::fs::write(bp.join("e.txt"), "e\n").unwrap();
        sh(bp, &["add", "e.txt"]);
        sh(bp, &["commit", "-q", "-m", "other branch work"]);
        assert!(matches!(cleanup_worktree(&b).await, Cleanup::Kept { .. }), "a missing branch must not skip the check");
        assert!(bp.is_dir());

        // Ignored files would be deleted by git worktree remove: kept.
        let c = add_worktree(&root, "ign333").await.unwrap();
        let cp = Path::new(&c.path);
        let exclude = root.join(".git").join("info").join("exclude");
        let old = std::fs::read_to_string(&exclude).unwrap_or_default();
        std::fs::write(&exclude, format!("{old}\n*.log\n")).unwrap();
        std::fs::write(cp.join("build.log"), "output\n").unwrap();
        match cleanup_worktree(&c).await {
            Cleanup::Kept { reason } => assert!(reason.contains("ignored"), "{reason}"),
            k => panic!("ignored files must be kept, got {k:?}"),
        }
        assert!(cp.join("build.log").is_file());
        std::fs::remove_file(cp.join("build.log")).unwrap();

        // A .git file that points at another repository: git is not run there.
        let other = temp_repo();
        std::fs::write(cp.join(".git"), format!("gitdir: {}\n", other.path().join(".git").to_string_lossy().replace('\\', "/"))).unwrap();
        match cleanup_worktree(&c).await {
            Cleanup::Kept { reason } => assert!(reason.contains("points at"), "{reason}"),
            k => panic!("{k:?}"),
        }
        assert!(cp.is_dir());
        // Put the link back: now it is clean and empty, so it goes.
        let admin = root.join(".git").join("worktrees").join("ign333");
        std::fs::write(cp.join(".git"), format!("gitdir: {}\n", admin.to_string_lossy().replace('\\', "/"))).unwrap();
        assert_eq!(cleanup_worktree(&c).await, Cleanup::Removed);
        assert!(!cp.exists());
    }

    #[tokio::test]
    async fn not_a_repo_and_bad_ids_are_clear_errors() {
        let d = tempfile::Builder::new().prefix("las-nogit-test").tempdir().unwrap();
        let e = repo_root(d.path()).await.unwrap_err().to_string();
        assert!(e.contains("not inside a git repository"), "{e}");
        let r = temp_repo();
        assert!(add_worktree(r.path(), "../evil").await.is_err());
        assert!(add_worktree(r.path(), "-b").await.is_err());
        // An empty repository has no HEAD to branch from.
        let empty = tempfile::Builder::new().prefix("las-empty-test").tempdir().unwrap();
        sh(empty.path(), &["init", "-q"]);
        let e = add_worktree(empty.path(), "ccc333").await.unwrap_err().to_string();
        assert!(e.contains("no commits yet"), "{e}");
    }
}

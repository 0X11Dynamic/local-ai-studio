//! File operations done with POSIX shell commands inside an environment that
//! only offers "run this sh script" (the WSL sandbox, Docker containers).
//! Data travels base64-encoded, so binary files and odd names are safe.

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use base64::Engine;

use crate::{CommandSpec, DirEntry, ExecResult};

#[async_trait]
pub trait ShRunner: Send + Sync {
    /// Run a short maintenance script as the environment's normal user.
    async fn sh(&self, script: &str, stdin: Option<Vec<u8>>) -> AppResult<ExecResult>;
    /// Where relative paths start, e.g. /workspace.
    fn workspace(&self) -> &str;
    /// Folders that must never be deleted.
    fn protected(&self) -> Vec<String> {
        vec![self.workspace().to_string()]
    }
}

/// Quote for POSIX sh: the result is always one literal word.
pub fn sq(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn check_path(p: &str, workspace: &str) -> AppResult<String> {
    if p.trim().is_empty() || p.contains('\0') || p.contains('\n') || p.contains('\r') {
        return Err(AppError::InvalidInput("path is empty or contains a control character".into()));
    }
    let p = p.trim();
    Ok(if p.starts_with('/') { p.to_string() } else { format!("{}/{p}", workspace.trim_end_matches('/')) })
}

/// The body of a user command: exported env, cd, then the command itself.
pub fn command_script(spec: &CommandSpec, workspace: &str) -> AppResult<String> {
    let cwd = check_path(spec.cwd.as_deref().unwrap_or(workspace), workspace)?;
    let mut script = String::new();
    for (k, v) in &spec.env {
        if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(AppError::InvalidInput(format!("bad environment variable name {k:?}")));
        }
        script.push_str(&format!("export {k}={}\n", sq(v)));
    }
    script.push_str(&format!("cd {} || exit 126\n{}\n", sq(&cwd), spec.command));
    Ok(script)
}

fn parse_stat(line: &str, dir: Option<&str>) -> Option<DirEntry> {
    let mut it = line.splitn(4, '|');
    let (kind, size, mtime, name) = (it.next()?, it.next()?, it.next()?, it.next()?);
    let is_dir = kind.contains("directory");
    let path = match dir {
        Some(d) => format!("{}/{}", d.trim_end_matches('/'), name),
        None => name.to_string(),
    };
    Some(DirEntry {
        name: name.rsplit('/').next().unwrap_or(name).to_string(),
        path,
        is_dir,
        size: if is_dir { 0 } else { size.parse().unwrap_or(0) },
        modified: mtime.parse::<u64>().ok().map(crate::local::rfc3339_from_unix),
    })
}

pub async fn read_file<R: ShRunner + ?Sized>(r: &R, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)> {
    let p = sq(&check_path(path, r.workspace())?);
    let res = r.sh(&format!("[ -d {p} ] && exit 21; [ -f {p} ] || exit 2; stat -c %s {p}; head -c {max_bytes} {p} | base64 -w0"), None).await?;
    match res.exit_code {
        Some(0) => {}
        Some(2) => return Err(AppError::NotFound(format!("{path} does not exist"))),
        Some(21) => return Err(AppError::InvalidInput(format!("{path} is a folder"))),
        _ => return Err(AppError::Runtime(format!("could not read {path}: {}", res.stderr.trim()))),
    }
    let mut lines = res.stdout.lines();
    let size: usize = lines.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let data = base64::engine::general_purpose::STANDARD.decode(lines.next().unwrap_or("").trim()).map_err(|e| AppError::Runtime(e.to_string()))?;
    Ok((data, size > max_bytes))
}

pub async fn write_file<R: ShRunner + ?Sized>(r: &R, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()> {
    let p = sq(&check_path(path, r.workspace())?);
    let mk = if create_dirs { format!("mkdir -p \"$(dirname {p})\" || exit 3\n") } else { String::new() };
    let script = format!("{mk}[ -d \"$(dirname {p})\" ] || exit 2\n[ -d {p} ] && exit 21\nt={p}.las-tmp.$$\nbase64 -d > \"$t\" && mv -f \"$t\" {p}");
    let res = r.sh(&script, Some(b64(bytes).into_bytes())).await?;
    match res.exit_code {
        Some(0) => Ok(()),
        Some(2) => Err(AppError::NotFound(format!("the folder for {path} does not exist"))),
        Some(21) => Err(AppError::InvalidInput(format!("{path} is a folder"))),
        _ => Err(AppError::Runtime(format!("could not write {path}: {}", res.stderr.trim()))),
    }
}

pub async fn list_dir<R: ShRunner + ?Sized>(r: &R, path: &str) -> AppResult<Vec<DirEntry>> {
    let full = check_path(path, r.workspace())?;
    let script = format!("cd {} || exit 2\nfor f in * .[!.]* ..?*; do [ -e \"$f\" ] || [ -L \"$f\" ] || continue; stat -c '%F|%s|%Y|%n' -- \"$f\"; done", sq(&full));
    let res = r.sh(&script, None).await?;
    if res.exit_code == Some(2) {
        return Err(AppError::NotFound(format!("{path} does not exist")));
    }
    let mut out: Vec<DirEntry> = res.stdout.lines().filter_map(|l| parse_stat(l, Some(&full))).collect();
    out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    Ok(out)
}

pub async fn stat<R: ShRunner + ?Sized>(r: &R, path: &str) -> AppResult<Option<DirEntry>> {
    let full = check_path(path, r.workspace())?;
    let res = r.sh(&format!("stat -c '%F|%s|%Y|%n' -- {}", sq(&full)), None).await?;
    Ok(if res.succeeded() { res.stdout.lines().next().and_then(|l| parse_stat(l, None)) } else { None })
}

pub async fn rename<R: ShRunner + ?Sized>(r: &R, from: &str, to: &str) -> AppResult<()> {
    let (a, b) = (sq(&check_path(from, r.workspace())?), sq(&check_path(to, r.workspace())?));
    let res = r.sh(&format!("[ -e {a} ] || exit 2; [ -e {b} ] && exit 17; mv -- {a} {b}"), None).await?;
    match res.exit_code {
        Some(0) => Ok(()),
        Some(2) => Err(AppError::NotFound(format!("{from} does not exist"))),
        Some(17) => Err(AppError::Conflict(format!("{to} already exists"))),
        _ => Err(AppError::Runtime(format!("move failed: {}", res.stderr.trim()))),
    }
}

/// A real delete: these environments are disposable.
pub async fn remove<R: ShRunner + ?Sized>(r: &R, path: &str, recursive: bool) -> AppResult<String> {
    let full = check_path(path, r.workspace())?;
    let norm = full.trim_end_matches('/');
    if norm.is_empty() || r.protected().iter().any(|p| p.trim_end_matches('/') == norm) {
        return Err(AppError::PermissionDenied("that folder cannot be deleted".into()));
    }
    let p = sq(&full);
    let flag = if recursive { "-rf" } else { "-f" };
    let res = r.sh(&format!("[ -e {p} ] || [ -L {p} ] || exit 2; [ -d {p} ] && [ {} = 0 ] && exit 21; rm {flag} -- {p}", recursive as u8), None).await?;
    match res.exit_code {
        Some(0) => Ok(String::new()),
        Some(2) => Err(AppError::NotFound(format!("{path} does not exist"))),
        Some(21) => Err(AppError::InvalidInput(format!("{path} is a folder; deleting a folder needs recursive=true"))),
        _ => Err(AppError::Runtime(format!("delete failed: {}", res.stderr.trim()))),
    }
}

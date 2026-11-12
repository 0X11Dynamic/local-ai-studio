//! Disposable Linux sandbox on WSL 2: a fresh Alpine minirootfs imported as
//! its own distro (`LocalAIStudio-sbx-<id>`), with Windows drive automount
//! and Windows interop switched off. Agent commands run as the unprivileged
//! `sandbox` user (root could re-mount C: with drvfs, so it is never used for
//! agent code). With network off, commands run in an empty network
//! namespace. The distro is unregistered on cleanup. Only distros with the
//! `LocalAIStudio-sbx-` prefix are ever created or removed.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use base64::Engine;
use sha2::Digest;
use tokio_util::sync::CancellationToken;

use crate::paths::RootPolicy;
use crate::process::{RunOptions, run};
use crate::{Capabilities, CommandSpec, DirEntry, ExecResult, ExecutionEnvironment, Health, ProcessEvent, Shell, now};

pub const DISTRO_PREFIX: &str = "LocalAIStudio-sbx-";
const WORKSPACE: &str = "/workspace";
const ALPINE_INDEX: &str = "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/latest-releases.yaml";
const ALPINE_BASE: &str = "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/";
/// Copy-in / copy-out limits.
const MAX_COPY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_COPY_FILES: usize = 100_000;

pub use crate::shfs::sq;
use crate::shfs::{self, ShRunner, b64};

fn check_path(p: &str) -> AppResult<String> {
    shfs::check_path(p, WORKSPACE)
}

/// Run a wsl.exe management command (UTF-8 output via WSL_UTF8).
async fn wsl_admin(args: &[&str], timeout: Duration) -> AppResult<ExecResult> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let env = [("WSL_UTF8".to_string(), "1".to_string())].into_iter().collect();
    run(RunOptions { program: "wsl.exe", args: &args, cwd: None, env: &env, stdin: None, timeout, max_output_bytes: 64 * 1024 }, &CancellationToken::new(), &|_| {}).await
}

fn clean(s: &str) -> String {
    s.replace('\0', "").trim().to_string()
}

/// What the user sees when the distro was unregistered outside the app.
pub const REMOVED: &str = "the sandbox was removed outside the app; delete it in Execution Environments and create a new one";

fn wsl_env() -> std::collections::BTreeMap<String, String> {
    [("WSL_UTF8".to_string(), "1".to_string())].into_iter().collect()
}

/// wsl.exe itself said the distro does not exist (its message comes before
/// any Linux output, so a command printing similar text is not mistaken).
fn distro_missing(r: &ExecResult) -> bool {
    let out = clean(&r.stdout);
    let err = clean(&r.stderr);
    [out.as_str(), err.as_str()].iter().any(|t| t.starts_with("There is no distribution with the supplied name") || t.contains("WSL_E_DISTRO_NOT_FOUND"))
}

/// Distros running right now.
pub async fn running_distros() -> AppResult<Vec<String>> {
    let r = wsl_admin(&["--list", "--running", "--quiet"], Duration::from_secs(20)).await?;
    Ok(clean(&r.stdout).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

/// WSL version string, or an error saying how to install it.
pub async fn wsl_status() -> AppResult<String> {
    let r = wsl_admin(&["--version"], Duration::from_secs(20)).await.map_err(|_| AppError::NotFound("WSL is not installed. Install it with: wsl --install --no-distribution".into()))?;
    if !r.succeeded() {
        return Err(AppError::NotFound(format!("WSL is not available: {}", clean(&r.stderr))));
    }
    Ok(clean(&r.stdout).lines().next().unwrap_or("WSL").to_string())
}

pub async fn list_distros() -> AppResult<Vec<String>> {
    let r = wsl_admin(&["--list", "--quiet"], Duration::from_secs(20)).await?;
    Ok(clean(&r.stdout).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

/// Download (once) and verify the Alpine minirootfs. Uses the cached copy
/// when the index cannot be fetched.
pub async fn ensure_rootfs(cache_dir: &Path) -> AppResult<PathBuf> {
    tokio::fs::create_dir_all(cache_dir).await?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| AppError::Runtime(e.to_string()))?;
    let index = match client.get(ALPINE_INDEX).send().await.and_then(|r| r.error_for_status()) {
        Ok(r) => r.text().await.ok(),
        Err(_) => None,
    };
    let Some(index) = index else {
        // Offline: newest cached rootfs, if any.
        let mut cached: Vec<PathBuf> = std::fs::read_dir(cache_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("alpine-minirootfs-")))
            .collect();
        cached.sort();
        return cached.pop().ok_or_else(|| AppError::Runtime("could not reach dl-cdn.alpinelinux.org to download the sandbox image".into()));
    };
    let (file, sha) = parse_index(&index).ok_or_else(|| AppError::Runtime("the Alpine release index has no minirootfs entry".into()))?;
    let dest = cache_dir.join(&file);
    if dest.is_file() && sha256_file(&dest)? == sha {
        return Ok(dest);
    }
    let bytes = client
        .get(format!("{ALPINE_BASE}{file}"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| AppError::Runtime(format!("sandbox image download failed: {e}")))?
        .bytes()
        .await
        .map_err(|e| AppError::Runtime(format!("sandbox image download failed: {e}")))?;
    let got = hex(&sha2::Sha256::digest(&bytes));
    if got != sha {
        return Err(AppError::Runtime(format!("sandbox image checksum mismatch (expected {sha}, got {got}); not using it")));
    }
    let tmp = dest.with_extension("part");
    tokio::fs::write(&tmp, &bytes).await?;
    tokio::fs::rename(&tmp, &dest).await?;
    Ok(dest)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha256_file(p: &Path) -> AppResult<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// The `file:` and `sha256:` of the `alpine-minirootfs` entry.
fn parse_index(yaml: &str) -> Option<(String, String)> {
    let mut in_entry = false;
    let (mut file, mut sha) = (None, None);
    for line in yaml.lines() {
        let t = line.trim().trim_start_matches("- ").trim();
        if line.trim_start().starts_with('-') {
            if in_entry && file.is_some() && sha.is_some() {
                break;
            }
            in_entry = false;
            file = None;
            sha = None;
        }
        if t == "flavor: alpine-minirootfs" {
            in_entry = true;
        } else if let Some(v) = t.strip_prefix("file: ") {
            file = Some(v.trim().to_string());
        } else if let Some(v) = t.strip_prefix("sha256: ") {
            sha = Some(v.trim().to_lowercase());
        }
        if in_entry && file.is_some() && sha.is_some() {
            break;
        }
    }
    let file = file.filter(|f| in_entry && f.ends_with(".tar.gz") && !f.contains('/') && !f.contains(".."))?;
    let sha = sha.filter(|s| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()))?;
    Some((file, sha))
}

pub struct WslSandbox {
    pub distro: String,
    pub dir: PathBuf,
    pub network: bool,
}

const SETUP: &str = r#"set -e
printf '[automount]\nenabled=false\nmountFsTab=false\n[interop]\nenabled=false\nappendWindowsPath=false\n[user]\ndefault=sandbox\n' > /etc/wsl.conf
adduser -D -h /home/sandbox -s /bin/sh sandbox
mkdir -p /workspace
chown sandbox:sandbox /workspace
passwd -l root >/dev/null 2>&1 || true
"#;

impl WslSandbox {
    /// Import a fresh distro. `id` must be `[a-z0-9-]`, at most 40 chars.
    pub async fn create(id: &str, base_dir: &Path, rootfs: &Path, network: bool) -> AppResult<Self> {
        if id.is_empty() || id.len() > 40 || !id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            return Err(AppError::InvalidInput("sandbox id must be lowercase letters, digits and dashes".into()));
        }
        let distro = format!("{DISTRO_PREFIX}{id}");
        if list_distros().await?.iter().any(|d| d.eq_ignore_ascii_case(&distro)) {
            return Err(AppError::Conflict(format!("a sandbox named {distro} already exists")));
        }
        let dir = base_dir.join(id);
        tokio::fs::create_dir_all(&dir).await?;
        let r = wsl_admin(&["--import", &distro, &dir.to_string_lossy(), &rootfs.to_string_lossy(), "--version", "2"], Duration::from_secs(300)).await?;
        if !r.succeeded() {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return Err(AppError::Runtime(format!("could not create the sandbox: {} {}", clean(&r.stdout), clean(&r.stderr))));
        }
        let sb = Self { distro, dir, network };
        let setup = async {
            let r = sb.root(SETUP, Duration::from_secs(60)).await?;
            if !r.succeeded() {
                return Err(AppError::Runtime(format!("sandbox setup failed: {}", r.stderr.trim())));
            }
            // Restart so wsl.conf (no automount, no interop) takes effect.
            wsl_admin(&["--terminate", &sb.distro], Duration::from_secs(60)).await?;
            sb.verify_isolation().await
        };
        if let Err(e) = setup.await {
            let _ = sb.destroy().await;
            return Err(e);
        }
        Ok(sb)
    }

    /// Reattach to an existing sandbox distro (after an app restart).
    pub fn attach(id: &str, base_dir: &Path, network: bool) -> Self {
        Self { distro: format!("{DISTRO_PREFIX}{id}"), dir: base_dir.join(id), network }
    }

    /// No Windows drive is mounted (drvfs, or anything at /mnt/<letter>) and
    /// wsl.conf has automount and interop off. The read-only 9p mount of the
    /// Windows driver store at /usr/lib/wsl/drivers is normal for WSL 2.
    /// (The binfmt entry still reads "enabled"; with interop off in wsl.conf
    /// a Windows .exe fails to start, checked by hand 2026-11-10.)
    pub async fn verify_isolation(&self) -> AppResult<()> {
        let script = r#"awk '$3 == "drvfs" || $2 ~ /^\/mnt\/[a-zA-Z]$/ || ($3 == "9p" && $2 != "/usr/lib/wsl/drivers") { print "MOUNT " $2 }' /proc/mounts
echo "CONF $(grep -c '^enabled=false' /etc/wsl.conf)""#;
        let r = self.root(script, Duration::from_secs(60)).await?;
        if let Some(m) = r.stdout.lines().find(|l| l.starts_with("MOUNT ")) {
            return Err(AppError::Runtime(format!("sandbox isolation check failed: a Windows folder is mounted ({})", m.trim_start_matches("MOUNT "))));
        }
        if !r.stdout.lines().any(|l| l.trim() == "CONF 2") {
            return Err(AppError::Runtime(format!("sandbox isolation check failed: automount and interop are not both off ({} {})", r.stdout.trim(), r.stderr.trim())));
        }
        Ok(())
    }

    /// Run fixed setup/maintenance as root (never agent-supplied code).
    async fn root(&self, script: &str, timeout: Duration) -> AppResult<ExecResult> {
        let wrapper = format!("printf %s {} | base64 -d | sh", b64(script.as_bytes()));
        let args = vec!["-d".to_string(), self.distro.clone(), "-u".into(), "root".into(), "--cd".into(), "/".into(), "-e".into(), "sh".into(), "-c".into(), wrapper];
        let r = run(RunOptions { program: "wsl.exe", args: &args, cwd: None, env: &wsl_env(), stdin: None, timeout, max_output_bytes: 256 * 1024 }, &CancellationToken::new(), &|_| {}).await?;
        if distro_missing(&r) {
            return Err(AppError::NotFound(REMOVED.into()));
        }
        Ok(r)
    }

    /// Run `script` as the sandbox user. The script travels base64-encoded,
    /// so nothing in it is interpreted by Windows or wsl.exe.
    async fn as_user(
        &self,
        script: &str,
        stdin: Option<Vec<u8>>,
        timeout: Duration,
        max_output: usize,
        cancel: &CancellationToken,
        on_event: &(dyn Fn(ProcessEvent) + Send + Sync),
    ) -> AppResult<ExecResult> {
        let encoded = b64(script.as_bytes());
        let net = if self.network { "" } else { "unshare -n " };
        let wrapper = format!("f=$(mktemp); printf %s {encoded} | base64 -d > $f; chmod 644 $f; exec {net}su -s /bin/sh sandbox $f");
        let args = vec!["-d".to_string(), self.distro.clone(), "-u".into(), "root".into(), "--cd".into(), "/".into(), "-e".into(), "sh".into(), "-c".into(), wrapper];
        let r = run(RunOptions { program: "wsl.exe", args: &args, cwd: None, env: &wsl_env(), stdin, timeout, max_output_bytes: max_output }, cancel, on_event).await?;
        if distro_missing(&r) {
            return Err(AppError::NotFound(REMOVED.into()));
        }
        if r.timed_out || r.cancelled {
            // Killing wsl.exe ends the session, but make sure nothing lingers.
            let _ = self.root("pkill -KILL -u sandbox; true", Duration::from_secs(20)).await;
        } else if r.exit_code != Some(0) && !running_distros().await.unwrap_or_default().iter().any(|d| d.eq_ignore_ascii_case(&self.distro)) {
            // The command "failed" because the whole distro went away under it
            // (wsl --terminate, wsl --shutdown, a WSL crash), not on its own.
            return Err(AppError::Runtime("the sandbox stopped while the command was running (it was shut down or WSL restarted); run the command again, which starts the sandbox".into()));
        }
        Ok(r)
    }

    async fn simple(&self, script: &str, stdin: Option<Vec<u8>>) -> AppResult<ExecResult> {
        self.as_user(script, stdin, Duration::from_secs(120), crate::HARD_MAX_OUTPUT, &CancellationToken::new(), &|_| {}).await
    }

    /// Unregister the distro (deletes its virtual disk) and remove its folder.
    pub async fn destroy(&self) -> AppResult<()> {
        if !self.distro.starts_with(DISTRO_PREFIX) {
            return Err(AppError::PermissionDenied(format!("refusing to remove {}: not a Local AI Studio sandbox", self.distro)));
        }
        if list_distros().await?.iter().any(|d| d.eq_ignore_ascii_case(&self.distro)) {
            let r = wsl_admin(&["--unregister", &self.distro], Duration::from_secs(120)).await?;
            if !r.succeeded() {
                return Err(AppError::Runtime(format!("could not remove the sandbox: {}", clean(&r.stdout))));
            }
        }
        let _ = tokio::fs::remove_dir_all(&self.dir).await;
        Ok(())
    }

    /// Copy a folder from this PC (checked against `policy`) into the
    /// sandbox. Symlinks and junctions are skipped.
    pub async fn copy_in(&self, policy: &RootPolicy, src: &str, dest: &str) -> AppResult<(usize, u64)> {
        let src = policy.resolve(src)?;
        if !src.is_dir() {
            return Err(AppError::InvalidInput(format!("{} is not a folder", src.display())));
        }
        let dest = check_path(dest)?;
        let policy = policy.clone();
        let (tar_bytes, files, bytes) = tokio::task::spawn_blocking(move || build_tar(&policy, &src)).await.map_err(|e| AppError::Runtime(e.to_string()))??;
        let script = format!("mkdir -p {d} && base64 -d | tar -x -m -C {d}", d = sq(&dest));
        let r = self.simple(&script, Some(b64(&tar_bytes).into_bytes())).await?;
        if !r.succeeded() {
            return Err(AppError::Runtime(format!("copy into the sandbox failed: {}", r.stderr.trim())));
        }
        Ok((files, bytes))
    }

    /// Copy a sandbox folder out to this PC. Only regular files and folders
    /// are extracted; paths are sanitized and checked against `policy`.
    pub async fn copy_out(&self, policy: &RootPolicy, src: &str, dest: &str) -> AppResult<(usize, u64)> {
        let src = check_path(src)?;
        let dest = policy.resolve(dest)?;
        let script = format!("cd {} && tar -c . | base64 -w0", sq(&src));
        let r = self.simple(&script, None).await?;
        if !r.succeeded() {
            return Err(AppError::Runtime(format!("copy out of the sandbox failed: {}", r.stderr.trim())));
        }
        if r.truncated {
            return Err(AppError::InvalidInput("that folder is too large to copy out in one go; copy a smaller folder".into()));
        }
        let tar_bytes = base64::engine::general_purpose::STANDARD.decode(r.stdout.trim()).map_err(|e| AppError::Runtime(format!("bad archive from the sandbox: {e}")))?;
        let policy = policy.clone();
        tokio::task::spawn_blocking(move || extract_tar(&policy, &tar_bytes, &dest)).await.map_err(|e| AppError::Runtime(e.to_string()))?
    }
}

fn build_tar(policy: &RootPolicy, src: &Path) -> AppResult<(Vec<u8>, usize, u64)> {
    let mut b = tar::Builder::new(Vec::new());
    b.follow_symlinks(false);
    let (mut files, mut bytes) = (0usize, 0u64);
    for e in walkdir::WalkDir::new(src).follow_links(false).min_depth(1) {
        let e = e.map_err(|e| AppError::Runtime(e.to_string()))?;
        let ft = e.file_type();
        if ft.is_symlink() || !(ft.is_file() || ft.is_dir()) {
            continue;
        }
        // Junctions report as directories with a reparse point.
        if ft.is_dir() && std::fs::read_link(e.path()).is_ok() {
            continue;
        }
        if policy.resolve(&e.path().to_string_lossy()).is_err() {
            continue;
        }
        let rel = e.path().strip_prefix(src).map_err(|e| AppError::Runtime(e.to_string()))?;
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        if ft.is_dir() {
            b.append_dir(&rel_s, e.path()).map_err(|e| AppError::Runtime(e.to_string()))?;
        } else {
            files += 1;
            bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
            if files > MAX_COPY_FILES || bytes > MAX_COPY_BYTES / 2 {
                return Err(AppError::InvalidInput("that folder is too large to copy into the sandbox".into()));
            }
            b.append_path_with_name(e.path(), &rel_s).map_err(|e| AppError::Runtime(e.to_string()))?;
        }
    }
    Ok((b.into_inner().map_err(|e| AppError::Runtime(e.to_string()))?, files, bytes))
}

fn extract_tar(policy: &RootPolicy, data: &[u8], dest: &Path) -> AppResult<(usize, u64)> {
    let mut a = tar::Archive::new(data);
    let (mut files, mut bytes) = (0usize, 0u64);
    for e in a.entries().map_err(|e| AppError::Runtime(e.to_string()))? {
        let mut e = e.map_err(|e| AppError::Runtime(e.to_string()))?;
        let kind = e.header().entry_type();
        let path = e.path().map_err(|e| AppError::Runtime(e.to_string()))?.into_owned();
        // Only plain relative components: no `..`, no roots, no prefixes.
        let mut rel = PathBuf::new();
        for c in path.components() {
            match c {
                Component::Normal(n) => rel.push(n),
                Component::CurDir => {}
                _ => return Err(AppError::PermissionDenied(format!("archive entry {} escapes the destination", path.display()))),
            }
        }
        if rel.as_os_str().is_empty() {
            continue;
        }
        let target = policy.resolve_write(&dest.join(&rel).to_string_lossy())?;
        if kind.is_dir() {
            std::fs::create_dir_all(&target)?;
        } else if kind.is_file() {
            files += 1;
            bytes += e.size();
            if files > MAX_COPY_FILES || bytes > MAX_COPY_BYTES {
                return Err(AppError::InvalidInput("too much data to copy out".into()));
            }
            if let Some(p) = target.parent() {
                std::fs::create_dir_all(p)?;
            }
            let mut out = std::fs::File::create(&target)?;
            std::io::copy(&mut e, &mut out)?;
            out.flush()?;
        }
        // Symlinks, hard links, devices: skipped.
    }
    Ok((files, bytes))
}


#[async_trait]
impl ShRunner for WslSandbox {
    async fn sh(&self, script: &str, stdin: Option<Vec<u8>>) -> AppResult<ExecResult> {
        self.simple(script, stdin).await
    }
    fn workspace(&self) -> &str {
        WORKSPACE
    }
    fn protected(&self) -> Vec<String> {
        vec![WORKSPACE.into(), "/home/sandbox".into()]
    }
}

#[async_trait]
impl ExecutionEnvironment for WslSandbox {
    fn kind(&self) -> &'static str {
        "sandbox"
    }

    fn label(&self) -> String {
        format!("Sandbox {} (network {})", self.distro.trim_start_matches(DISTRO_PREFIX), if self.network { "on" } else { "off" })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { shells: vec![Shell::Sh], files: true, local: false, os: "linux (Alpine on WSL 2)".into() }
    }

    async fn health(&self) -> Health {
        match self.simple("uname -sr; id -un", None).await {
            Ok(r) if r.succeeded() => Health { ok: true, detail: r.stdout.split_whitespace().collect::<Vec<_>>().join(" "), checked_at: now() },
            Ok(r) => Health { ok: false, detail: format!("{} {}", r.stdout.trim(), r.stderr.trim()), checked_at: now() },
            Err(e) => Health { ok: false, detail: e.to_string(), checked_at: now() },
        }
    }

    async fn execute(&self, spec: &CommandSpec, cancel: &CancellationToken, on_event: &(dyn Fn(ProcessEvent) + Send + Sync)) -> AppResult<ExecResult> {
        if spec.shell != Shell::Sh {
            return Err(AppError::InvalidInput("the sandbox runs sh commands, not PowerShell or cmd".into()));
        }
        let script = shfs::command_script(spec, WORKSPACE)?;
        self.as_user(&script, None, spec.timeout, spec.max_output_bytes, cancel, on_event).await
    }

    async fn read_file(&self, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)> {
        shfs::read_file(self, path, max_bytes).await
    }

    async fn write_file(&self, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()> {
        shfs::write_file(self, path, bytes, create_dirs).await
    }

    async fn list_dir(&self, path: &str) -> AppResult<Vec<DirEntry>> {
        shfs::list_dir(self, path).await
    }

    async fn stat(&self, path: &str) -> AppResult<Option<DirEntry>> {
        shfs::stat(self, path).await
    }

    async fn rename(&self, from: &str, to: &str) -> AppResult<()> {
        shfs::rename(self, from, to).await
    }

    async fn remove(&self, path: &str, recursive: bool) -> AppResult<String> {
        shfs::remove(self, path, recursive).await
    }

    async fn cleanup(&self) -> AppResult<()> {
        self.destroy().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_and_index_parsing() {
        assert_eq!(sq("it's"), r"'it'\''s'");
        let y = "-\n  title: \"Standard\"\n  flavor: alpine-standard\n  file: alpine-standard-3.24.4-x86_64.iso\n  sha256: aaaa\n-\n  title: \"Mini root filesystem\"\n  flavor: alpine-minirootfs\n  file: alpine-minirootfs-3.24.4-x86_64.tar.gz\n  sha256: cdaa38cb8de3537967ee5b6c9b6cd00cc962da2b018e6eea48696bb274923012\n-\n  flavor: x\n";
        let (f, s) = parse_index(y).unwrap();
        assert_eq!(f, "alpine-minirootfs-3.24.4-x86_64.tar.gz");
        assert!(s.starts_with("cdaa38"));
        assert!(parse_index("-\n  flavor: alpine-minirootfs\n  file: ../../evil.tar.gz\n  sha256: cdaa38cb8de3537967ee5b6c9b6cd00cc962da2b018e6eea48696bb274923012\n").is_none());
    }

    #[test]
    fn tar_extraction_refuses_escapes_and_skips_links() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("out");
        std::fs::create_dir_all(&root).unwrap();
        let policy = RootPolicy::new(vec![root.clone()], vec![]);
        let mut b = tar::Builder::new(Vec::new());
        let mut h = tar::Header::new_gnu();
        h.set_size(2);
        h.set_entry_type(tar::EntryType::Regular);
        b.append_data(&mut h, "ok/a.txt", &b"hi"[..]).unwrap();
        let mut l = tar::Header::new_gnu();
        l.set_entry_type(tar::EntryType::Symlink);
        l.set_size(0);
        b.append_link(&mut l, "link", "C:/Windows").unwrap();
        let good = b.into_inner().unwrap();
        assert_eq!(extract_tar(&policy, &good, &root).unwrap().0, 1);
        assert!(root.join("ok/a.txt").exists() && !root.join("link").exists());
        // A hand-made header with `..` in the name.
        let mut raw = vec![0u8; 1024 * 3];
        let name = b"../escape.txt";
        raw[..name.len()].copy_from_slice(name);
        raw[100..108].copy_from_slice(b"0000644\0");
        raw[124..136].copy_from_slice(b"00000000002\0");
        raw[156] = b'0';
        raw[257..263].copy_from_slice(b"ustar\0");
        raw[263..265].copy_from_slice(b"00");
        raw[148..156].copy_from_slice(b"        ");
        let sum: u32 = raw[..512].iter().map(|&b| b as u32).sum();
        raw[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        raw[512..514].copy_from_slice(b"hi");
        assert!(extract_tar(&policy, &raw, &root).is_err());
        assert!(!d.path().join("escape.txt").exists());
    }
}

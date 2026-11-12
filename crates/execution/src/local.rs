//! The local Windows environment: PowerShell as the current user (never
//! elevated), inside a Job Object, restricted to the allowed folders.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use app_core::{AppError, AppResult};
use async_trait::async_trait;
use base64::Engine;
use tokio_util::sync::CancellationToken;

use crate::paths::RootPolicy;
use crate::process::{RunOptions, run};
use crate::{Capabilities, CommandSpec, DirEntry, ExecResult, ExecutionEnvironment, Health, ProcessEvent, Shell, now};

pub struct LocalEnv {
    pub policy: RootPolicy,
    /// Where `remove` moves things (restorable), e.g. %APPDATA%\LocalAIStudio\trash.
    pub trash_dir: PathBuf,
    powershell: String,
}

/// PowerShell 7 (pwsh.exe) if it is on PATH, else Windows PowerShell 5.1.
pub fn find_powershell() -> String {
    if let Some(path) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&path) {
            let p = d.join("pwsh.exe");
            if p.is_file() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    "powershell.exe".into()
}

/// `-EncodedCommand` takes base64 of UTF-16LE: the script is never parsed by
/// a shell command line, so there is no quoting or injection layer.
pub fn encode_ps(script: &str) -> String {
    let prelude = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8\n";
    // The user's script runs dot-sourced in its own block, so error messages
    // quote its lines rather than the prelude.
    let full = format!("{prelude}. {{\n{script}\n}}");
    let bytes: Vec<u8> = full.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn is_clixml(s: &str) -> bool {
    s.contains("#< CLIXML") || s.contains("<Objs Version=")
}

/// Plain text from PowerShell's CLIXML error stream: keep the error,
/// warning, verbose and information strings, drop progress records.
pub fn decode_clixml(s: &str) -> String {
    let re = regex::Regex::new(r#"<S S="(Error|Warning|Verbose|Debug|Information)">(.*?)</S>"#).expect("valid regex");
    let mut out = String::new();
    for c in re.captures_iter(s) {
        let mut t = c[2].replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'");
        // _xHHHH_ escapes (mostly CR/LF), then &amp; last.
        let esc = regex::Regex::new(r"_x([0-9A-Fa-f]{4})_").expect("valid regex");
        t = esc
            .replace_all(&t, |m: &regex::Captures| u32::from_str_radix(&m[1], 16).ok().and_then(char::from_u32).map(|ch| ch.to_string()).unwrap_or_default())
            .into_owned()
            .replace("&amp;", "&");
        out.push_str(&t.replace("\r\n", "\n"));
    }
    out.trim_end().to_string()
}

fn entry(p: &Path, meta: &std::fs::Metadata) -> DirEntry {
    DirEntry {
        name: p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        path: p.to_string_lossy().into_owned(),
        is_dir: meta.is_dir(),
        size: if meta.is_dir() { 0 } else { meta.len() },
        modified: meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| rfc3339_from_unix(d.as_secs())),
    }
}

/// RFC 3339 from unix seconds without pulling chrono into this crate.
pub fn rfc3339_from_unix(secs: u64) -> String {
    let days = secs / 86_400;
    let (h, m, s) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
    // Civil-from-days (Howard Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

impl LocalEnv {
    pub fn new(policy: RootPolicy, trash_dir: PathBuf) -> Self {
        Self { policy, trash_dir, powershell: find_powershell() }
    }

    pub fn powershell(&self) -> &str {
        &self.powershell
    }
}

#[async_trait]
impl ExecutionEnvironment for LocalEnv {
    fn kind(&self) -> &'static str {
        "local"
    }

    fn label(&self) -> String {
        format!("This PC ({})", std::env::var("COMPUTERNAME").unwrap_or_else(|_| "Windows".into()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { shells: vec![Shell::PowerShell, Shell::Cmd], files: true, local: true, os: "windows".into() }
    }

    async fn health(&self) -> Health {
        let spec = CommandSpec { timeout: std::time::Duration::from_secs(20), ..CommandSpec::new("$PSVersionTable.PSVersion.ToString()", Shell::PowerShell) };
        let r = self.execute(&spec, &CancellationToken::new(), &|_| {}).await;
        match r {
            Ok(r) if r.succeeded() => Health { ok: true, detail: format!("PowerShell {} ({})", r.stdout.trim(), self.powershell), checked_at: now() },
            Ok(r) => Health { ok: false, detail: format!("PowerShell exited with {:?}: {}", r.exit_code, r.stderr.trim()), checked_at: now() },
            Err(e) => Health { ok: false, detail: e.to_string(), checked_at: now() },
        }
    }

    async fn execute(&self, spec: &CommandSpec, cancel: &CancellationToken, on_event: &(dyn Fn(ProcessEvent) + Send + Sync)) -> AppResult<ExecResult> {
        let cwd = match &spec.cwd {
            Some(c) => self.policy.resolve(c)?,
            None => self.policy.roots.first().cloned().ok_or_else(|| AppError::PermissionDenied("no allowed folders".into()))?,
        };
        if !cwd.is_dir() {
            return Err(AppError::InvalidInput(format!("working folder {} does not exist", cwd.display())));
        }
        let (program, args): (String, Vec<String>) = match spec.shell {
            Shell::PowerShell => (
                self.powershell.clone(),
                vec!["-NoLogo".into(), "-NoProfile".into(), "-NonInteractive".into(), "-EncodedCommand".into(), encode_ps(&spec.command)],
            ),
            Shell::Cmd => ("cmd.exe".into(), vec!["/d".into(), "/s".into(), "/c".into(), spec.command.clone()]),
            Shell::Sh => return Err(AppError::InvalidInput("this PC runs PowerShell or cmd, not sh".into())),
        };
        let is_ps = spec.shell == Shell::PowerShell;
        // Windows PowerShell 5.1 writes stderr as CLIXML under -EncodedCommand
        // (no flag turns that off); decode it to plain text, live and final.
        let filtered = |ev: ProcessEvent| match ev {
            ProcessEvent::Stderr(t) if is_ps && is_clixml(&t) => {
                let text = decode_clixml(&t);
                if !text.is_empty() {
                    on_event(ProcessEvent::Stderr(text + "\n"));
                }
            }
            other => on_event(other),
        };
        let mut r = run(
            RunOptions { program: &program, args: &args, cwd: Some(&cwd), env: &spec.env, stdin: None, timeout: spec.timeout, max_output_bytes: spec.max_output_bytes },
            cancel,
            &filtered,
        )
        .await?;
        if is_ps && is_clixml(&r.stderr) {
            r.stderr = decode_clixml(&r.stderr);
        }
        Ok(r)
    }

    async fn read_file(&self, path: &str, max_bytes: usize) -> AppResult<(Vec<u8>, bool)> {
        let p = self.policy.resolve(path)?;
        let meta = tokio::fs::metadata(&p).await.map_err(|_| AppError::NotFound(format!("{} does not exist", p.display())))?;
        if meta.is_dir() {
            return Err(AppError::InvalidInput(format!("{} is a folder", p.display())));
        }
        use tokio::io::AsyncReadExt;
        let mut f = tokio::fs::File::open(&p).await?;
        let mut buf = vec![0u8; (meta.len() as usize).min(max_bytes)];
        f.read_exact(&mut buf).await?;
        Ok((buf, meta.len() as usize > max_bytes))
    }

    async fn write_file(&self, path: &str, bytes: &[u8], create_dirs: bool) -> AppResult<()> {
        let p = self.policy.resolve_write(path)?;
        let parent = p.parent().ok_or_else(|| AppError::InvalidInput("no parent folder".into()))?;
        if !parent.exists() {
            if create_dirs {
                tokio::fs::create_dir_all(parent).await?;
            } else {
                return Err(AppError::NotFound(format!("folder {} does not exist", parent.display())));
            }
        }
        if p.is_dir() {
            return Err(AppError::InvalidInput(format!("{} is a folder", p.display())));
        }
        // Atomic: write a temp file next to it, then rename over the target.
        // The temp name is unique per write, so concurrent writes never share one.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = parent.join(format!(".{}.las-tmp-{}-{n}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), std::process::id()));
        tokio::fs::write(&tmp, bytes).await?;
        // Another write replacing the same file at the same moment makes
        // Windows refuse the rename briefly (access denied): retry a few times.
        let mut attempt = 0;
        loop {
            match tokio::fs::rename(&tmp, &p).await {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 20 && e.kind() == std::io::ErrorKind::PermissionDenied => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(10 * attempt)).await;
                }
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(e.into());
                }
            }
        }
    }

    async fn list_dir(&self, path: &str) -> AppResult<Vec<DirEntry>> {
        let p = self.policy.resolve(path)?;
        let mut rd = tokio::fs::read_dir(&p).await.map_err(|e| AppError::NotFound(format!("cannot list {}: {e}", p.display())))?;
        let mut out = vec![];
        while let Some(e) = rd.next_entry().await? {
            if let Ok(m) = e.metadata().await {
                out.push(entry(&e.path(), &m));
            }
            if out.len() >= 5000 {
                break;
            }
        }
        out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.to_lowercase().cmp(&b.name.to_lowercase())));
        Ok(out)
    }

    async fn stat(&self, path: &str) -> AppResult<Option<DirEntry>> {
        let p = self.policy.resolve(path)?;
        Ok(tokio::fs::metadata(&p).await.ok().map(|m| entry(&p, &m)))
    }

    async fn rename(&self, from: &str, to: &str) -> AppResult<()> {
        let a = self.policy.resolve_move_or_delete(from)?;
        let b = self.policy.resolve_write(to)?;
        if !a.exists() {
            return Err(AppError::NotFound(format!("{} does not exist", a.display())));
        }
        if b.exists() {
            return Err(AppError::Conflict(format!("{} already exists", b.display())));
        }
        tokio::fs::rename(&a, &b).await.map_err(|e| AppError::Runtime(format!("move failed: {e}")))?;
        Ok(())
    }

    async fn remove(&self, path: &str, recursive: bool) -> AppResult<String> {
        let p = self.policy.resolve_move_or_delete(path)?;
        let meta = tokio::fs::metadata(&p).await.map_err(|_| AppError::NotFound(format!("{} does not exist", p.display())))?;
        if meta.is_dir() && !recursive {
            return Err(AppError::InvalidInput(format!("{} is a folder; deleting a folder needs recursive=true", p.display())));
        }
        if self.policy.roots.iter().any(|r| r.to_string_lossy().eq_ignore_ascii_case(&p.to_string_lossy())) {
            return Err(AppError::PermissionDenied("an allowed root folder itself cannot be deleted".into()));
        }
        // Move to the app trash (restorable) rather than deleting.
        let stamp = now().replace([':', '.'], "-");
        let dest_dir = self.trash_dir.join(&stamp);
        tokio::fs::create_dir_all(&dest_dir).await?;
        let dest = dest_dir.join(p.file_name().unwrap_or_default());
        match tokio::fs::rename(&p, &dest).await {
            Ok(()) => {}
            Err(_) if !meta.is_dir() => {
                // Another drive: copy then delete the original file.
                tokio::fs::copy(&p, &dest).await?;
                tokio::fs::remove_file(&p).await?;
            }
            Err(e) => return Err(AppError::Runtime(format!("could not move the folder to the trash: {e}"))),
        }
        Ok(dest.to_string_lossy().into_owned())
    }

    async fn cleanup(&self) -> AppResult<()> {
        Ok(())
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn env() -> (tempfile::TempDir, LocalEnv) {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        std::fs::create_dir_all(&root).unwrap();
        let e = LocalEnv::new(RootPolicy::new(vec![root], vec![]), d.path().join("trash"));
        (d, e)
    }

    #[tokio::test]
    async fn runs_powershell_with_streaming_exit_codes_and_quotes() {
        let (_d, e) = env();
        let lines = Arc::new(Mutex::new(vec![]));
        let l2 = lines.clone();
        let spec = CommandSpec::new("Write-Output 'it''s \"quoted\" & safe'; Write-Error 'oops'; exit 7", Shell::PowerShell);
        let r = e.execute(&spec, &CancellationToken::new(), &move |ev| l2.lock().unwrap().push(format!("{ev:?}"))).await.unwrap();
        assert_eq!(r.exit_code, Some(7), "{r:?}");
        assert!(r.stdout.contains(r#"it's "quoted" & safe"#), "{}", r.stdout);
        assert!(r.stderr.contains("oops") && !r.stderr.contains("CLIXML") && !r.stderr.contains("<Objs"), "plain-text stderr: {}", r.stderr);
        assert!(lines.lock().unwrap().iter().any(|l| l.contains("Stdout")), "streamed");
        assert!(e.health().await.ok);
    }

    #[tokio::test]
    async fn timeout_kills_the_whole_process_tree_and_output_is_capped() {
        let (_d, e) = env();
        // A child process that would outlive its parent without the job object.
        let spec = CommandSpec {
            timeout: Duration::from_secs(3),
            ..CommandSpec::new("$p = Start-Process powershell -ArgumentList '-NoProfile','-Command','Start-Sleep 120' -PassThru -WindowStyle Hidden; Write-Output $p.Id; Start-Sleep 120", Shell::PowerShell)
        };
        let t = std::time::Instant::now();
        let r = e.execute(&spec, &CancellationToken::new(), &|_| {}).await.unwrap();
        assert!(r.timed_out && t.elapsed() < Duration::from_secs(15), "{r:?}");
        let child: u32 = r.stdout.trim().parse().unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let out = std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {child}"), "/NH"]).output().unwrap();
        assert!(!String::from_utf8_lossy(&out.stdout).contains(&child.to_string()), "grandchild survived the timeout");
        let spam = CommandSpec { max_output_bytes: 2048, ..CommandSpec::new("1..5000 | ForEach-Object { 'line ' + $_ }", Shell::PowerShell) };
        let r = e.execute(&spam, &CancellationToken::new(), &|_| {}).await.unwrap();
        assert!(r.truncated && r.stdout.len() <= 2048, "{}", r.stdout.len());
    }

    #[tokio::test]
    async fn cancel_stops_a_command() {
        let (_d, e) = env();
        let c = CancellationToken::new();
        let c2 = c.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(800)).await;
            c2.cancel();
        });
        let r = e.execute(&CommandSpec::new("Start-Sleep 60", Shell::PowerShell), &c, &|_| {}).await.unwrap();
        assert!(r.cancelled && r.duration_ms < 10_000, "{r:?}");
    }

    #[tokio::test]
    async fn files_are_confined_atomic_and_deletes_go_to_trash() {
        let (d, e) = env();
        let root = d.path().join("work");
        let f = root.join("notes.txt");
        e.write_file(&f.to_string_lossy(), b"hello", false).await.unwrap();
        assert_eq!(e.read_file(&f.to_string_lossy(), 100).await.unwrap().0, b"hello");
        assert!(e.write_file(&d.path().join("outside.txt").to_string_lossy(), b"x", true).await.is_err());
        assert!(e.execute(&CommandSpec { cwd: Some(d.path().to_string_lossy().into()), ..CommandSpec::new("dir", Shell::PowerShell) }, &CancellationToken::new(), &|_| {}).await.is_err(), "cwd outside roots");
        let listed = e.list_dir(&root.to_string_lossy()).await.unwrap();
        assert_eq!(listed.len(), 1);
        let moved = e.remove(&f.to_string_lossy(), false).await.unwrap();
        assert!(!f.exists() && std::path::Path::new(&moved).exists(), "moved to trash: {moved}");
        assert!(e.remove(&root.to_string_lossy(), true).await.is_err(), "root itself protected");
    }
}

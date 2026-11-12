//! Real WSL sandbox test. Creates and removes a `LocalAIStudio-sbx-test-*`
//! distro. Run with: LAS_WSL_TEST=1 cargo test -p execution --test sandbox_wsl -- --nocapture

use std::time::Duration;

use execution::paths::RootPolicy;
use execution::sandbox::{WslSandbox, ensure_rootfs, list_distros};
use execution::{CommandSpec, ExecutionEnvironment, Shell, fs_tools};
use tokio_util::sync::CancellationToken;

async fn sh(sb: &WslSandbox, cmd: &str) -> execution::ExecResult {
    sb.execute(&CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new(cmd, Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap()
}

#[tokio::test]
async fn disposable_isolated_sandbox() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    let before = list_distros().await.unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let rootfs = ensure_rootfs(&tmp.path().join("cache")).await.unwrap();
    let id = format!("test-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sandboxes"), &rootfs, false).await.unwrap();
    let result = std::panic::AssertUnwindSafe(async {
        assert!(sb.health().await.ok);
        // Runs as the unprivileged user, in /workspace, quotes intact.
        let r = sb
            .execute(
                &CommandSpec { env: [("MSG".to_string(), "it's $HOME \"x\"".to_string())].into(), ..CommandSpec::new("id -un; pwd; printf '%s\n' \"$MSG\"", Shell::Sh) },
                &CancellationToken::new(),
                &|_| {},
            )
            .await
            .unwrap();
        assert_eq!(r.stdout, "sandbox\n/workspace\nit's $HOME \"x\"\n", "{r:?}");
        // No Windows drive, cannot mount one, cannot become root, no network.
        let r = sh(&sb, "ls -A /mnt/c 2>/dev/null | wc -l; mkdir -p /tmp/m; mount -t drvfs C: /tmp/m 2>&1 && echo MOUNTED; echo | su -c id root 2>&1 | grep -c uid=0").await;
        assert!(!r.stdout.contains("MOUNTED") && r.stdout.lines().next() == Some("0") && r.stdout.trim().ends_with('0'), "{r:?}");
        let r = sh(&sb, "wget -q -T 5 -O /dev/null http://dl-cdn.alpinelinux.org/ && echo NET_OK || echo NET_BLOCKED").await;
        assert!(r.stdout.contains("NET_BLOCKED"), "{r:?}");
        assert!(sb.execute(&CommandSpec::new("dir", Shell::PowerShell), &CancellationToken::new(), &|_| {}).await.is_err());
        // Exit codes and stderr.
        let r = sh(&sb, "echo out; echo err >&2; exit 3").await;
        assert!(r.exit_code == Some(3) && r.stderr.contains("err"));
        // Timeout kills everything the command started.
        let t = std::time::Instant::now();
        let r = sb.execute(&CommandSpec { timeout: Duration::from_secs(3), ..CommandSpec::new("sleep 100 & sleep 100", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap();
        assert!(r.timed_out && t.elapsed() < Duration::from_secs(30), "{r:?}");
        let r = sh(&sb, "ps -o user,args | grep -c '[s]leep 100'").await;
        assert_eq!(r.stdout.trim(), "0", "leftover processes");
        // File tools.
        let w = fs_tools::write(&sb, "proj/hello.txt", "one\ntwo\n", true).await.unwrap();
        assert!(w.created);
        assert_eq!(fs_tools::read(&sb, "proj/hello.txt", None, None).await.unwrap(), "one\ntwo\n");
        fs_tools::patch(&sb, "proj/hello.txt", "two", "TWO", false).await.unwrap();
        assert!(fs_tools::search(&sb, "/workspace", "TWO", None).await.unwrap().contains("hello.txt:2:"));
        assert!(fs_tools::list(&sb, "proj").await.unwrap().contains("hello.txt"));
        fs_tools::move_path(&sb, "proj/hello.txt", "proj/renamed.txt").await.unwrap();
        assert!(sb.stat("proj/hello.txt").await.unwrap().is_none());
        assert!(sb.remove("/workspace", true).await.is_err());
        // Binary round trip.
        let blob: Vec<u8> = (0..=255u8).cycle().take(70_000).collect();
        sb.write_file("bin/blob.dat", &blob, true).await.unwrap();
        assert_eq!(sb.read_file("bin/blob.dat", 1 << 20).await.unwrap().0, blob);
        // Copy a Windows folder in and back out.
        let src = tmp.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.txt"), "alpha").unwrap();
        std::fs::write(src.join("sub").join("b.txt"), "beta").unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        std::process::Command::new("cmd").args(["/c", "mklink", "/J", &src.join("junction").to_string_lossy(), &outside.to_string_lossy()]).output().unwrap();
        let policy = RootPolicy::new(vec![tmp.path().to_path_buf()], vec![]);
        let (n, _) = sb.copy_in(&policy, &src.to_string_lossy(), "/workspace/in").await.unwrap();
        assert_eq!(n, 2, "junction skipped");
        let r = sh(&sb, "find /workspace/in -type f | sort; ls /workspace/in/junction 2>&1 | head -1").await;
        assert!(r.stdout.contains("/workspace/in/sub/b.txt") && !r.stdout.contains("secret"), "{r:?}");
        sh(&sb, "echo made-in-sandbox > /workspace/in/new.txt").await;
        let dest = tmp.path().join("back");
        std::fs::create_dir_all(&dest).unwrap();
        let (n, _) = sb.copy_out(&policy, "/workspace/in", &dest.to_string_lossy()).await.unwrap();
        assert_eq!(n, 3);
        assert_eq!(std::fs::read_to_string(dest.join("new.txt")).unwrap().trim(), "made-in-sandbox");
        let narrow = RootPolicy::new(vec![tmp.path().join("src")], vec![]);
        assert!(sb.copy_out(&narrow, "/workspace/in", &dest.to_string_lossy()).await.is_err(), "destination outside roots");
        std::process::Command::new("cmd").args(["/c", "rmdir", &src.join("junction").to_string_lossy()]).output().unwrap();
    })
    .catch_unwind_compat()
    .await;
    sb.cleanup().await.unwrap();
    let after = list_distros().await.unwrap();
    assert!(!after.iter().any(|d| d.contains(&id)), "sandbox removed");
    assert_eq!(before, after, "other distros untouched");
    assert!(!tmp.path().join("sandboxes").join(&id).exists());
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

trait CatchUnwindCompat: std::future::Future + Sized {
    fn catch_unwind_compat(self) -> futures::future::CatchUnwind<Self>
    where
        Self: std::panic::UnwindSafe,
    {
        futures::FutureExt::catch_unwind(self)
    }
}
impl<F: std::future::Future> CatchUnwindCompat for F {}

/// The sandbox stopped (`wsl --terminate`) during a command, then removed
/// (`wsl --unregister`) outside the app: clear errors, never a hang, and the
/// sandbox starts again after a plain shutdown.
#[tokio::test]
async fn sandbox_shutdown_and_removal_are_reported() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let rootfs = ensure_rootfs(&tmp.path().join("cache")).await.unwrap();
    let id = format!("fail-test-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sandboxes"), &rootfs, false).await.unwrap();
    assert!(sb.distro.starts_with("LocalAIStudio-sbx-"));
    let wsl = |args: &[&str]| std::process::Command::new("wsl.exe").args(args).env("WSL_UTF8", "1").output().unwrap();
    let result = std::panic::AssertUnwindSafe(async {
        // 1. Shut down mid-command.
        let distro = sb.distro.clone();
        let killer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            std::process::Command::new("wsl.exe").args(["--terminate", &distro]).output().unwrap();
        });
        let t = std::time::Instant::now();
        let r = sb.execute(&CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new("echo start; sleep 30; echo end", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await;
        killer.await.unwrap();
        println!("after terminate ({:?}): {r:?}", t.elapsed());
        assert!(t.elapsed() < Duration::from_secs(20), "hung for {:?}", t.elapsed());
        let msg = r.as_ref().err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("sandbox stopped"), "expected a clear error, got {r:?}");
        // 2. It starts again on the next command.
        let again = sh(&sb, "echo again").await;
        assert_eq!(again.stdout.trim(), "again", "{again:?}");
        // 3. Removed outside the app.
        let out = wsl(&["--unregister", &sb.distro]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
        let h = sb.health().await;
        println!("health after unregister: {h:?}");
        assert!(!h.ok && h.detail.contains("was removed") && h.detail.contains("create a new one"), "{h:?}");
        let e = sb.execute(&CommandSpec::new("true", Shell::Sh), &CancellationToken::new(), &|_| {}).await;
        let e = e.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("was removed") && !e.contains("WSL_E_"), "{e}");
    })
    .catch_unwind_compat()
    .await;
    sb.cleanup().await.unwrap();
    let left = String::from_utf8_lossy(&wsl(&["--list", "--quiet"]).stdout).to_string();
    assert!(!left.contains(&id) && left.contains("kali-linux"), "distros now: {left}");
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

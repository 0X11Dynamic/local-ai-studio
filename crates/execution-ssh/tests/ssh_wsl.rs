//! Real SSH test against OpenSSH in a throwaway WSL sandbox distro.
//! Run with: LAS_WSL_TEST=1 cargo test -p execution-ssh --test ssh_wsl -- --nocapture

use std::time::Duration;

use base64::Engine;
use execution::sandbox::{WslSandbox, ensure_rootfs};
use execution::{CommandSpec, ExecutionEnvironment, Shell, fs_tools};
use execution_ssh::{SshAuth, SshConfig, SshEnv, fetch_host_key, trust_host};
use tokio_util::sync::CancellationToken;

fn root(distro: &str, script: &str) -> String {
    let b = base64::engine::general_purpose::STANDARD.encode(script);
    let o = std::process::Command::new("wsl.exe")
        .args(["-d", distro, "-u", "root", "--cd", "/", "-e", "sh", "-c", &format!("printf %s {b} | base64 -d | sh")])
        .output()
        .unwrap();
    assert!(o.status.success(), "{script}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn cfg(port: u16, kh: &std::path::Path, auth: SshAuth, roots: Vec<String>) -> SshConfig {
    SshConfig { host: "127.0.0.1".into(), port, user: "sandbox".into(), auth, known_hosts: kh.to_path_buf(), roots, shell: Shell::Sh, connect_timeout: Duration::from_secs(15) }
}

async fn sh(e: &SshEnv, cmd: &str) -> execution::ExecResult {
    e.execute(&CommandSpec { timeout: Duration::from_secs(30), ..CommandSpec::new(cmd, Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap()
}

async fn sh_err(e: &SshEnv) -> String {
    e.execute(&CommandSpec::new("true", Shell::Sh), &CancellationToken::new(), &|_| {}).await.err().map(|x| x.to_string()).unwrap_or_else(|| "no error".into())
}

const SETUP: &str = r#"set -e
apk add --no-cache -q openssh-server openssh-sftp-server openssh-keygen >/dev/null
ssh-keygen -A >/dev/null
echo 'sandbox:correct-horse' | chpasswd >/dev/null 2>&1
mkdir -p /home/sandbox/.ssh /home/sandbox/proj
ssh-keygen -q -t ed25519 -N '' -f /root/k
cp /root/k.pub /home/sandbox/.ssh/authorized_keys
chown -R sandbox:sandbox /home/sandbox
chmod 700 /home/sandbox/.ssh
chmod 600 /home/sandbox/.ssh/authorized_keys
ln -s /etc /home/sandbox/proj/etc-link
mkdir -p /etc/ssh/sshd_config.d
printf 'PasswordAuthentication yes\nSubsystem sftp internal-sftp\n' > /etc/ssh/sshd_config.d/las.conf
grep -q '^Include /etc/ssh/sshd_config.d' /etc/ssh/sshd_config || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config
"#;

#[tokio::test]
async fn ssh_against_a_real_openssh_server() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let rootfs = ensure_rootfs(&std::env::temp_dir().join("las-rootfs-cache")).await.unwrap();
    let id = format!("ssh-test-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sbx"), &rootfs, true).await.unwrap();
    let port: u16 = 22000 + (std::process::id() % 1000) as u16;
    let mut sshd: Option<std::process::Child> = None;
    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(async {
        root(&sb.distro, SETUP);
        let key = root(&sb.distro, "cat /root/k");
        let key_path = tmp.path().join("id_ed25519");
        std::fs::write(&key_path, key).unwrap();
        sshd = Some(
            std::process::Command::new("wsl.exe")
                .args(["-d", &sb.distro, "-u", "root", "-e", "/usr/sbin/sshd", "-D", "-e", "-p", &port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
        // Wait for the port (WSL localhost forwarding).
        let mut up = false;
        for _ in 0..40 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
                up = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        assert!(up, "sshd did not come up on {port}");
        let kh = tmp.path().join("known_hosts");
        let key_auth = SshAuth::Key { path: key_path.to_string_lossy().into(), passphrase: None };

        // 1. Unknown host is refused, with the fingerprint shown.
        let e = SshEnv::new(cfg(port, &kh, key_auth.clone(), vec![]));
        let err = sh_err(&e).await;
        assert!(err.contains("unknown host key") && err.contains("SHA256:"), "{err}");
        let (fp, alg, _) = fetch_host_key("127.0.0.1", port, Duration::from_secs(10)).await.unwrap();
        assert!(err.contains(&fp), "same fingerprint: {err} vs {fp} ({alg})");
        assert!(trust_host("127.0.0.1", port, &kh, "SHA256:wrongwrongwrong").await.is_err(), "mismatch refused");
        trust_host("127.0.0.1", port, &kh, &fp).await.unwrap();
        println!("host key {alg} {fp} trusted");

        // 2. Commands.
        let e = SshEnv::new(cfg(port, &kh, key_auth.clone(), vec![]));
        let h = e.health().await;
        assert!(h.ok, "{h:?}");
        let r = e
            .execute(
                &CommandSpec {
                    env: [("MSG".to_string(), "it's $HOME".to_string())].into(),
                    cwd: Some("/tmp".into()),
                    ..CommandSpec::new("id -un; pwd; printf '%s\\n' \"$MSG\"; echo e >&2; exit 4", Shell::Sh)
                },
                &CancellationToken::new(),
                &|_| {},
            )
            .await
            .unwrap();
        assert_eq!(r.stdout, "sandbox\n/tmp\nit's $HOME\n", "{r:?}");
        assert!(r.exit_code == Some(4) && r.stderr.trim() == "e", "{r:?}");
        let t = std::time::Instant::now();
        let r = e.execute(&CommandSpec { timeout: Duration::from_secs(3), ..CommandSpec::new("sleep 101 & sleep 101", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap();
        assert!(r.timed_out && t.elapsed() < Duration::from_secs(20), "{r:?}");
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(sh(&e, "ps -o args | grep -c '[s]leep 101'").await.stdout.trim(), "0", "remote process group killed");
        let c = CancellationToken::new();
        let c2 = c.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            c2.cancel();
        });
        let r = e.execute(&CommandSpec::new("sleep 60", Shell::Sh), &c, &|_| {}).await.unwrap();
        assert!(r.cancelled, "{r:?}");

        // 3. Files, confined to a root (symlinks resolved on the server).
        let e = SshEnv::new(cfg(port, &kh, key_auth.clone(), vec!["/home/sandbox/proj".into()]));
        let w = fs_tools::write(&e, "notes/a.txt", "one\ntwo\n", true).await.unwrap();
        assert!(w.created);
        fs_tools::write(&e, "notes/a.txt", "one\nTWO\n", false).await.unwrap();
        assert_eq!(fs_tools::read(&e, "notes/a.txt", None, None).await.unwrap(), "one\nTWO\n");
        fs_tools::patch(&e, "notes/a.txt", "one", "ONE", false).await.unwrap();
        assert!(fs_tools::list(&e, "notes").await.unwrap().contains("a.txt"));
        assert!(fs_tools::search(&e, "/home/sandbox/proj/notes", "TWO", None).await.unwrap().contains("a.txt:2:"));
        fs_tools::move_path(&e, "notes/a.txt", "notes/b.txt").await.unwrap();
        let blob: Vec<u8> = (0..=255u8).cycle().take(300_000).collect();
        e.write_file("bin/blob", &blob, true).await.unwrap();
        assert_eq!(e.read_file("bin/blob", 1 << 20).await.unwrap().0, blob);
        assert!(e.read_file("/etc/passwd", 100).await.is_err(), "outside root");
        assert!(e.read_file("../.ssh/authorized_keys", 100).await.is_err(), ".. escape");
        let esc = e.read_file("etc-link/passwd", 100).await;
        assert!(esc.as_ref().is_err_and(|x| x.to_string().contains("outside")), "symlink escape: {esc:?}");
        assert!(e.remove("/home/sandbox/proj", true).await.is_err(), "root protected");
        fs_tools::delete(&e, "bin", true).await.unwrap();
        assert!(e.stat("bin").await.unwrap().is_none());
        e.cleanup().await.unwrap();

        // 4. Password login.
        let bad = SshEnv::new(cfg(port, &kh, SshAuth::Password { password: "nope".into() }, vec![]));
        assert!(sh_err(&bad).await.contains("rejected the login"));
        let good = SshEnv::new(cfg(port, &kh, SshAuth::Password { password: "correct-horse".into() }, vec![]));
        assert_eq!(sh(&good, "echo ok").await.stdout.trim(), "ok");

        // 5. A changed host key is refused loudly.
        let fake = tmp.path().join("known_hosts_changed");
        let other = root(&sb.distro, "ssh-keygen -q -t ed25519 -N '' -f /root/other && cat /root/other.pub");
        std::fs::write(&fake, format!("[127.0.0.1]:{port} {}\n", other.trim())).unwrap();
        let e = SshEnv::new(cfg(port, &fake, key_auth, vec![]));
        let err = sh_err(&e).await;
        if alg.contains("ed25519") {
            assert!(err.contains("HOST KEY CHANGED"), "{err}");
        } else {
            assert!(err.contains("unknown host key"), "a different algorithm counts as unknown: {err}");
        }
    }))
    .await;
    if let Some(mut s) = sshd {
        let _ = s.kill();
        let _ = s.wait();
    }
    sb.cleanup().await.unwrap();
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

fn spawn_sshd(distro: &str, port: u16) -> std::process::Child {
    std::process::Command::new("wsl.exe")
        .args(["-d", distro, "-u", "root", "-e", "/usr/sbin/sshd", "-D", "-e", "-p", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap()
}

async fn wait_port(port: u16) -> bool {
    for _ in 0..40 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// The server goes away mid-command: a clear error within seconds (never a
/// hang), and the same environment reconnects once the server is back.
/// Unreachable and refusing hosts fail within the connect timeout.
#[tokio::test]
async fn ssh_drop_mid_command_and_unreachable_hosts() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    // Hosts that are down: refused (closed port) and unreachable (blackhole).
    let tmp = tempfile::tempdir().unwrap();
    let kh = tmp.path().join("known_hosts");
    let dummy = SshAuth::Password { password: "x".into() };
    let refused = SshEnv::new(cfg(1, &kh, dummy.clone(), vec![]));
    let t = std::time::Instant::now();
    let e = sh_err(&refused).await;
    assert!(e.contains("could not connect") && t.elapsed() < Duration::from_secs(10), "{e} after {:?}", t.elapsed());
    let mut c = cfg(22, &kh, dummy, vec![]);
    c.host = "10.255.255.1".into();
    c.connect_timeout = Duration::from_secs(3);
    let blackhole = SshEnv::new(c);
    let t = std::time::Instant::now();
    let e = sh_err(&blackhole).await;
    assert!(e.contains("timed out connecting") || e.contains("could not connect"), "{e}");
    assert!(t.elapsed() < Duration::from_secs(8), "took {:?}", t.elapsed());

    let rootfs = ensure_rootfs(&std::env::temp_dir().join("las-rootfs-cache")).await.unwrap();
    let id = format!("ssh-drop-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sbx"), &rootfs, true).await.unwrap();
    let port: u16 = 22500 + (std::process::id() % 400) as u16;
    let mut daemons: Vec<std::process::Child> = vec![];
    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(async {
        root(&sb.distro, SETUP);
        let key_path = tmp.path().join("id_ed25519");
        std::fs::write(&key_path, root(&sb.distro, "cat /root/k")).unwrap();
        daemons.push(spawn_sshd(&sb.distro, port));
        assert!(wait_port(port).await, "sshd did not start");
        let (fp, _, _) = fetch_host_key("127.0.0.1", port, Duration::from_secs(10)).await.unwrap();
        trust_host("127.0.0.1", port, &kh, &fp).await.unwrap();
        let env = std::sync::Arc::new(SshEnv::new(cfg(port, &kh, SshAuth::Key { path: key_path.to_string_lossy().into(), passphrase: None }, vec![])));
        assert_eq!(sh(&env, "echo hi").await.stdout.trim(), "hi", "session cached");

        // Kill every sshd process (listener and session) during `sleep 30`.
        let distro = sb.distro.clone();
        let killer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            root(&distro, "pkill -KILL -f sshd; true");
        });
        let t = std::time::Instant::now();
        let r = env.execute(&CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new("echo start; sleep 30; echo end", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await;
        killer.await.unwrap();
        println!("after sshd killed ({:?}): {r:?}", t.elapsed());
        assert!(t.elapsed() < Duration::from_secs(15), "hung for {:?}", t.elapsed());
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("dropped") && msg.contains("start"), "{msg}");

        // While the server is down: a clear error, not a hang.
        let t = std::time::Instant::now();
        let down = sh_err(&env).await;
        assert!(down.contains("could not connect") || down.contains("could not open a channel"), "{down}");
        assert!(t.elapsed() < Duration::from_secs(20), "took {:?}", t.elapsed());

        // Server back: the same environment reconnects.
        daemons.push(spawn_sshd(&sb.distro, port));
        assert!(wait_port(port).await, "sshd did not restart");
        // WSL's localhost relay keeps accepting on the port while sshd is
        // down, so "port open" is not "server ready": retry the real command.
        let mut back = None;
        for _ in 0..40 {
            match env.execute(&CommandSpec::new("echo back", Shell::Sh), &CancellationToken::new(), &|_| {}).await {
                Ok(r) => {
                    back = Some(r);
                    break;
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
        assert_eq!(back.expect("never reconnected after sshd came back").stdout.trim(), "back");
        assert!(env.read_file("/etc/hostname", 100).await.is_ok(), "SFTP works after the reconnect");
    }))
    .await;
    for mut d in daemons {
        let _ = d.kill();
        let _ = d.wait();
    }
    sb.cleanup().await.unwrap();
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

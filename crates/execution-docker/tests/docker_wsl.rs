//! Real Docker test: Docker Engine runs inside a throwaway WSL sandbox
//! distro (no Docker Desktop needed) and the environment talks to it over
//! TCP on localhost. Run with:
//! LAS_WSL_TEST=1 cargo test -p execution-docker --test docker_wsl -- --nocapture

use std::time::Duration;

use execution::paths::RootPolicy;
use execution::sandbox::{WslSandbox, ensure_rootfs};
use execution::{CommandSpec, ExecutionEnvironment, Shell, fs_tools};
use execution_docker::{DockerConfig, DockerEnv};
use tokio_util::sync::CancellationToken;

fn root(distro: &str, script: &str) -> String {
    let o = std::process::Command::new("wsl.exe").args(["-d", distro, "-u", "root", "--cd", "/", "-e", "sh", "-c", script]).output().unwrap();
    assert!(o.status.success(), "{script}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

async fn sh(e: &DockerEnv, cmd: &str) -> execution::ExecResult {
    e.execute(&CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new(cmd, Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap()
}

#[tokio::test]
async fn docker_environment_against_a_real_daemon() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let rootfs = ensure_rootfs(&std::env::temp_dir().join("las-rootfs-cache")).await.unwrap();
    let id = format!("docker-test-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sbx"), &rootfs, true).await.unwrap();
    let port: u16 = 23000 + (std::process::id() % 1000) as u16;
    let mut dockerd: Option<std::process::Child> = None;
    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(async {
        root(&sb.distro, "apk add --no-cache -q docker >/dev/null 2>&1 && docker --version");
        dockerd = Some(
            std::process::Command::new("wsl.exe")
                .args(["-d", &sb.distro, "-u", "root", "-e", "dockerd", "-H", &format!("tcp://127.0.0.1:{port}"), "--iptables=false", "--ip6tables=false"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
        let policy = RootPolicy::new(vec![tmp.path().to_path_buf()], vec![]);
        let cfg = DockerConfig { host: format!("tcp://127.0.0.1:{port}"), name: format!("test-{}", std::process::id()), ..Default::default() };
        let e = DockerEnv::new(cfg.clone(), &policy).unwrap();
        let mut up = false;
        for _ in 0..60 {
            if e.docker_version().await.is_ok() {
                up = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        assert!(up, "dockerd did not come up");
        println!("{}", e.docker_version().await.unwrap());

        // Health pulls the image and starts the container.
        let h = e.health().await;
        assert!(h.ok, "{h:?}");
        // Commands: env, cwd, quotes, exit codes.
        let r = e
            .execute(
                &CommandSpec { env: [("MSG".to_string(), "it's $HOME".to_string())].into(), ..CommandSpec::new("ps -o pid,pgid,args >&2; pwd; printf '%s\\n' \"$MSG\"; echo e >&2; exit 5", Shell::Sh) },
                &CancellationToken::new(),
                &|_| {},
            )
            .await
            .unwrap();
        assert_eq!(r.stdout, "/workspace\nit's $HOME\n", "{r:?}");
        println!("{}", r.stderr); assert!(r.exit_code == Some(5) && r.stderr.trim().ends_with("e"), "{r:?}");
        // Locked down: no network, no extra capabilities, no new privileges.
        let r = sh(&e, "ip -o link 2>/dev/null | grep -vc ' lo:' ; wget -q -T 4 -O /dev/null http://dl-cdn.alpinelinux.org/ && echo NET_OK || echo NET_BLOCKED; grep -E '^(CapEff|NoNewPrivs)' /proc/self/status").await;
        assert!(r.stdout.contains("NET_BLOCKED"), "{r:?}");
        assert!(r.stdout.contains("NoNewPrivs:\t1"), "{r:?}");
        let capeff = r.stdout.lines().find(|l| l.starts_with("CapEff")).unwrap().split_whitespace().nth(1).unwrap().to_string();
        let caps = u64::from_str_radix(&capeff, 16).unwrap();
        assert_eq!(caps & (1 << 21), 0, "no CAP_SYS_ADMIN");
        assert_eq!(caps & (1 << 12), 0, "no CAP_NET_ADMIN");
        assert!(e.execute(&CommandSpec::new("dir", Shell::PowerShell), &CancellationToken::new(), &|_| {}).await.is_err());
        // Timeout kills the whole process group.
        let t = std::time::Instant::now();
        let r = e.execute(&CommandSpec { timeout: Duration::from_secs(3), ..CommandSpec::new("sleep 102 & sleep 102", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await.unwrap();
        assert!(r.timed_out && t.elapsed() < Duration::from_secs(20), "{r:?}");
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(sh(&e, "ps -o args | grep -c '[s]leep 102'").await.stdout.trim(), "0", "leftover processes");
        // Files.
        let w = fs_tools::write(&e, "proj/a.txt", "one\ntwo\n", true).await.unwrap();
        assert!(w.created);
        fs_tools::patch(&e, "proj/a.txt", "two", "TWO", false).await.unwrap();
        assert_eq!(fs_tools::read(&e, "proj/a.txt", None, None).await.unwrap(), "one\nTWO\n");
        assert!(fs_tools::search(&e, "/workspace", "TWO", None).await.unwrap().contains("a.txt:2:"));
        let blob: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        e.write_file("bin/blob", &blob, true).await.unwrap();
        assert_eq!(e.read_file("bin/blob", 1 << 20).await.unwrap().0, blob);
        assert!(e.remove("/workspace", true).await.is_err());
        // State persists across environment objects until cleanup.
        let e2 = DockerEnv::new(cfg.clone(), &policy).unwrap();
        assert_eq!(fs_tools::read(&e2, "proj/a.txt", None, None).await.unwrap(), "one\nTWO\n");
        e2.cleanup().await.unwrap();
        let e3 = DockerEnv::new(cfg, &policy).unwrap();
        assert!(e3.read_file("proj/a.txt", 100).await.is_err(), "fresh container after cleanup");
        e3.cleanup().await.unwrap();
    }))
    .await;
    if let Some(mut d) = dockerd {
        let _ = d.kill();
        let _ = d.wait();
    }
    sb.cleanup().await.unwrap();
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

/// The container is killed mid-command, then removed outside the app, then
/// the daemon goes away: clear errors, and the environment recovers.
#[tokio::test]
async fn container_exit_removal_and_daemon_down() {
    if std::env::var("LAS_WSL_TEST").is_err() {
        eprintln!("skipped: set LAS_WSL_TEST=1");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let rootfs = ensure_rootfs(&std::env::temp_dir().join("las-rootfs-cache")).await.unwrap();
    let id = format!("docker-fail-{}", std::process::id());
    let sb = WslSandbox::create(&id, &tmp.path().join("sbx"), &rootfs, true).await.unwrap();
    let port: u16 = 24000 + (std::process::id() % 900) as u16;
    let host = format!("tcp://127.0.0.1:{port}");
    let spawn_dockerd = |distro: &str| {
        std::process::Command::new("wsl.exe")
            .args(["-d", distro, "-u", "root", "-e", "dockerd", "-H", &format!("tcp://127.0.0.1:{port}"), "--iptables=false", "--ip6tables=false"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap()
    };
    let mut dockerd = None;
    let result = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(async {
        root(&sb.distro, "apk add --no-cache -q docker >/dev/null 2>&1 && docker --version");
        dockerd = Some(spawn_dockerd(&sb.distro));
        let policy = RootPolicy::new(vec![tmp.path().to_path_buf()], vec![]);
        let cfg = DockerConfig { host: host.clone(), name: format!("fail-{}", std::process::id()), ..Default::default() };
        let env = DockerEnv::new(cfg.clone(), &policy).unwrap();
        for _ in 0..60 {
            if env.docker_version().await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let h = env.health().await;
        assert!(h.ok, "{h:?}");
        let cname = env.container_name();
        let dk = |cmd: &str| root(&sb.distro, &format!("docker -H {host} {cmd}"));

        // 1. Killed mid-command.
        let distro = sb.distro.clone();
        let (h2, c2) = (host.clone(), cname.clone());
        let killer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            root(&distro, &format!("docker -H {h2} kill {c2}"));
        });
        let t = std::time::Instant::now();
        let r = env.execute(&CommandSpec { timeout: Duration::from_secs(60), ..CommandSpec::new("echo start; sleep 30; echo end", Shell::Sh) }, &CancellationToken::new(), &|_| {}).await;
        killer.await.unwrap();
        println!("after docker kill ({:?}): {r:?}", t.elapsed());
        assert!(t.elapsed() < Duration::from_secs(15), "hung for {:?}", t.elapsed());
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("stopped while the command was running") && msg.contains("start"), "{msg}");
        // The next command starts it again (state kept).
        sh(&env, "echo kept > /workspace/kept.txt").await;
        assert_eq!(sh(&env, "echo again").await.stdout.trim(), "again");

        // 2. Removed outside the app: health says so and a new one is made.
        dk(&format!("rm -f {cname}"));
        let h = env.health().await;
        println!("health after rm: {h:?}");
        assert!(h.ok && h.detail.contains("removed outside the app"), "{h:?}");
        let r = sh(&env, "cat /workspace/kept.txt 2>/dev/null || echo fresh").await;
        assert_eq!(r.stdout.trim(), "fresh", "a new container");

        // 3. Daemon gone.
        root(&sb.distro, "pkill -KILL dockerd; pkill -KILL containerd; true");
        if let Some(mut d) = dockerd.take() {
            let _ = d.kill();
            let _ = d.wait();
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        let t = std::time::Instant::now();
        let h = env.health().await;
        println!("health with daemon down ({:?}): {h:?}", t.elapsed());
        assert!(!h.ok && h.detail.contains("cannot reach Docker"), "{h:?}");
        let e = env.execute(&CommandSpec::new("true", Shell::Sh), &CancellationToken::new(), &|_| {}).await.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("cannot reach Docker"), "{e}");
        assert!(t.elapsed() < Duration::from_secs(30), "took {:?}", t.elapsed());
    }))
    .await;
    if let Some(mut d) = dockerd {
        let _ = d.kill();
        let _ = d.wait();
    }
    sb.cleanup().await.unwrap();
    if let Err(p) = result {
        std::panic::resume_unwind(p);
    }
}

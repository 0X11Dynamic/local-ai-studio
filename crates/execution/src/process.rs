//! Run a child process inside a kill-on-close Job Object (created suspended,
//! assigned, resumed: no window where a grandchild can escape), stream its
//! output line by line with a size cap, and stop the whole tree on timeout
//! or cancel.

use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

use crate::{ExecResult, HARD_MAX_OUTPUT, ProcessEvent, now};

pub struct RunOptions<'a> {
    pub program: &'a str,
    pub args: &'a [String],
    pub cwd: Option<&'a std::path::Path>,
    pub env: &'a std::collections::BTreeMap<String, String>,
    pub stdin: Option<Vec<u8>>,
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

async fn pump<R: AsyncRead + Unpin>(
    mut r: R,
    is_err: bool,
    cap: usize,
    on_event: &(dyn Fn(ProcessEvent) + Send + Sync),
) -> (String, bool) {
    let mut kept: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut buf = vec![0u8; 8192];
    let mut line: Vec<u8> = Vec::new();
    loop {
        let n = match r.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if kept.len() < cap {
                kept.push(b);
            } else {
                truncated = true;
            }
            line.push(b);
            if b == b'\n' || line.len() >= 4096 {
                let text = String::from_utf8_lossy(&line).into_owned();
                on_event(if is_err { ProcessEvent::Stderr(text) } else { ProcessEvent::Stdout(text) });
                line.clear();
            }
        }
    }
    if !line.is_empty() {
        let text = String::from_utf8_lossy(&line).into_owned();
        on_event(if is_err { ProcessEvent::Stderr(text) } else { ProcessEvent::Stdout(text) });
    }
    (String::from_utf8_lossy(&kept).into_owned(), truncated)
}

pub async fn run(opts: RunOptions<'_>, cancel: &CancellationToken, on_event: &(dyn Fn(ProcessEvent) + Send + Sync)) -> AppResult<ExecResult> {
    let mut cmd = tokio::process::Command::new(opts.program);
    cmd.args(opts.args)
        .stdin(if opts.stdin.is_some() { std::process::Stdio::piped() } else { std::process::Stdio::null() })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(c) = opts.cwd {
        cmd.current_dir(c);
    }
    for (k, v) in opts.env {
        cmd.env(k, v);
    }
    let mut wrap = process_wrap::tokio::CommandWrap::from(cmd);
    #[cfg(windows)]
    {
        use windows::Win32::System::Threading::CREATE_NO_WINDOW;
        wrap.wrap(process_wrap::tokio::CreationFlags(CREATE_NO_WINDOW));
        wrap.wrap(process_wrap::tokio::JobObject);
    }
    wrap.wrap(process_wrap::tokio::KillOnDrop);
    let started_at = now();
    let t0 = Instant::now();
    let mut child = wrap.spawn().map_err(|e| AppError::Runtime(format!("could not start {}: {e}", opts.program)))?;
    if let Some(input) = opts.stdin
        && let Some(mut si) = child.stdin().take()
    {
        tokio::spawn(async move {
            let _ = si.write_all(&input).await;
            let _ = si.shutdown().await;
        });
    }
    let out = child.stdout().take().ok_or_else(|| AppError::Runtime("no stdout".into()))?;
    let err = child.stderr().take().ok_or_else(|| AppError::Runtime("no stderr".into()))?;
    let cap = opts.max_output_bytes.clamp(1024, HARD_MAX_OUTPUT);
    let pumps = async { tokio::join!(pump(out, false, cap, on_event), pump(err, true, cap, on_event)) };
    tokio::pin!(pumps);
    let mut timed_out = false;
    let mut cancelled = false;
    let deadline = tokio::time::sleep(opts.timeout);
    tokio::pin!(deadline);
    let ((stdout, t1), (stderr, t2)) = tokio::select! {
        r = &mut pumps => r,
        _ = &mut deadline => { timed_out = true; let _ = child.start_kill(); pumps.await }
        _ = cancel.cancelled() => { cancelled = true; let _ = child.start_kill(); pumps.await }
    };
    let status = child.wait().await.ok();
    Ok(ExecResult {
        exit_code: status.and_then(|s| s.code()),
        stdout,
        stderr,
        truncated: t1 || t2,
        timed_out,
        cancelled,
        duration_ms: t0.elapsed().as_millis() as u64,
        started_at,
        ended_at: now(),
    })
}

//! Shared by runtime installs and model downloads.
//!
//! Resumable streaming HTTP download to a `.part` file with progress reports,
//! cancellation and optional SHA-256 verification. The final file only appears
//! (by atomic rename) after the transfer and the checksum succeed.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub bytes_per_sec: f64,
}

#[derive(Debug)]
pub enum DownloadError {
    Cancelled,
    Http(String),
    Io(std::io::Error),
    Checksum { expected: String, actual: String },
    DiskSpace { needed: u64, available: u64 },
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Cancelled => write!(f, "download cancelled"),
            DownloadError::Http(s) => write!(f, "{s}"),
            DownloadError::Io(e) => write!(f, "file error: {e}"),
            DownloadError::Checksum { expected, actual } => {
                write!(f, "checksum mismatch: expected sha256 {expected}, got {actual}; the file was discarded")
            }
            DownloadError::DiskSpace { needed, available } => write!(
                f,
                "not enough disk space: need {:.1} GB more, {:.1} GB free",
                *needed as f64 / 1e9,
                *available as f64 / 1e9
            ),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        DownloadError::Io(e)
    }
}

pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Download `url` to `dest`. If `dest.part` exists, resume with an HTTP Range
/// request when the server allows it, otherwise restart from zero.
pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancel: &CancellationToken,
    on_progress: impl FnMut(Progress),
) -> Result<(), DownloadError> {
    download_file_auth(client, url, None, dest, expected_sha256, cancel, on_progress).await
}

/// Like [`download_file`], with an optional bearer token for the first host.
/// reqwest drops the Authorization header when a redirect goes to another
/// host, so CDN mirrors never receive the token.
pub async fn download_file_auth(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(Progress),
) -> Result<(), DownloadError> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let part = part_path(dest);
    let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    let mut req = client.get(url);
    if let Some(t) = bearer {
        req = req.bearer_auth(t);
    }
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = tokio::select! {
        r = req.send() => r.map_err(|e| DownloadError::Http(format!("request failed: {e}")))?,
        _ = cancel.cancelled() => return Err(DownloadError::Cancelled),
    };
    let status = resp.status();
    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        // The .part is already complete or the server changed; start over.
        tokio::fs::remove_file(&part).await.ok();
        return Box::pin(download_file_auth(client, url, bearer, dest, expected_sha256, cancel, on_progress)).await;
    }
    if !status.is_success() {
        return Err(DownloadError::Http(format!("server returned HTTP {status} for {url}")));
    }
    if !resumed {
        have = 0;
    }
    let total = resp.content_length().map(|l| l + have);
    if let Some(t) = total {
        let free = free_space(dest.parent().unwrap_or(Path::new(".")));
        if let Some(free) = free {
            let needed = t.saturating_sub(have);
            if needed + (256 << 20) > free {
                return Err(DownloadError::DiskSpace { needed, available: free });
            }
        }
    }

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .await?;

    let mut stream = resp.bytes_stream();
    let mut downloaded = have;
    let started = Instant::now();
    let mut last_report = Instant::now() - Duration::from_secs(1);
    let mut since_start = 0u64;
    loop {
        let chunk = tokio::select! {
            c = stream.next() => c,
            _ = cancel.cancelled() => {
                file.flush().await.ok();
                return Err(DownloadError::Cancelled);
            }
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|e| DownloadError::Http(format!("connection error after {downloaded} bytes: {e}")))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        since_start += chunk.len() as u64;
        if last_report.elapsed() >= Duration::from_millis(250) {
            last_report = Instant::now();
            let secs = started.elapsed().as_secs_f64().max(0.001);
            on_progress(Progress { downloaded, total, bytes_per_sec: since_start as f64 / secs });
        }
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    if let Some(t) = total
        && downloaded != t
    {
        return Err(DownloadError::Http(format!("connection closed early: got {downloaded} of {t} bytes")));
    }
    on_progress(Progress { downloaded, total, bytes_per_sec: 0.0 });

    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(&part).await?;
        if !actual.eq_ignore_ascii_case(expected) {
            tokio::fs::remove_file(&part).await.ok();
            return Err(DownloadError::Checksum { expected: expected.to_string(), actual });
        }
    }
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

pub async fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = tokio::fs::File::open(path).await?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Free bytes on the volume containing `dir`, if it can be determined.
pub fn free_space(dir: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let mut probe = dir.to_path_buf();
        while !probe.exists() {
            probe = probe.parent()?.to_path_buf();
        }
        let wide: Vec<u16> = probe.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut free = 0u64;
        unsafe {
            GetDiskFreeSpaceExW(windows::core::PCWSTR(wide.as_ptr()), Some(&mut free), None, None).ok()?;
        }
        Some(free)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_reads_real_volume() {
        let f = free_space(&std::env::temp_dir()).unwrap();
        assert!(f > 0);
    }

    #[tokio::test]
    async fn sha256_of_known_content() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).await.unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

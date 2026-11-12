//! Install and track llama.cpp runtimes.
//!
//! Official builds come from the ggml-org/llama.cpp GitHub releases. A CUDA
//! runtime needs two zips: the llama.cpp binaries and the matching CUDA runtime
//! DLLs (`cudart-llama-bin-win-cuda-X-x64.zip`). Each runtime lives in its own
//! folder under `runtimes/` with a `runtime.json` manifest. A user can also
//! register an existing `llama-server.exe` (a "custom" runtime) in place.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use app_core::{AppError, AppResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::capabilities::RuntimeCapabilities;
use downloader::{self as download, DownloadError, Progress};

const RELEASES_URL: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10";
const MANIFEST: &str = "runtime.json";

static MAIN_ASSET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^llama-(?P<tag>b\d+)-bin-win-(?P<variant>cpu|vulkan|cuda-[\d.]+)-x64\.zip$").unwrap());
static CUDART_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^cudart-llama(?:-b\d+)?-bin-win-(?P<variant>cuda-[\d.]+)-x64\.zip$").unwrap());

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledRuntime {
    /// e.g. "b12246-cuda-13.5", or "custom-<n>"
    pub id: String,
    pub tag: String,
    /// cpu | cuda-13.5 | cuda-12.4 | vulkan | custom
    pub variant: String,
    pub server_path: PathBuf,
    pub installed_at: String,
    /// official | custom
    pub source: String,
    pub capabilities: RuntimeCapabilities,
}

impl InstalledRuntime {
    pub fn uses_gpu(&self) -> bool {
        self.variant.starts_with("cuda") || self.variant == "vulkan" || self.variant == "custom"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetRef {
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableRuntime {
    pub id: String,
    pub tag: String,
    pub variant: String,
    pub published_at: String,
    pub assets: Vec<AssetRef>,
    pub total_size: u64,
    pub recommended: bool,
    /// Why it is or isn't a good fit for this PC.
    pub note: String,
}

/// What the recommender knows about the GPU. All optional: absent = unknown.
#[derive(Debug, Clone, Default)]
pub struct GpuHint {
    pub nvidia: bool,
    /// e.g. (12, 0) for Blackwell consumer GPUs
    pub compute_capability: Option<(u32, u32)>,
    /// highest CUDA version the driver supports, e.g. (13, 5)
    pub cuda_driver: Option<(u32, u32)>,
    /// NVIDIA driver version, e.g. (621, 1) for "621.01"
    pub driver_version: Option<(u32, u32)>,
    /// A non-NVIDIA GPU with its own VRAM (Radeon RX, Intel Arc): Vulkan fits.
    pub vulkan_gpu: bool,
}

/// Parse an NVIDIA driver version such as "621.01" into (621, 1).
pub fn parse_driver_version(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Windows NVIDIA driver a CUDA build of llama.cpp needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DriverRequirement {
    /// Oldest driver that runs it at all (CUDA minor version compatibility).
    pub minimum: (u32, u32),
    /// The driver branch that toolkit shipped with. Older (but >= minimum)
    /// drivers run it without the newest driver-side features.
    pub recommended: (u32, u32),
}

/// Driver needed for a CUDA toolkit version, from NVIDIA's CUDA Toolkit
/// release notes (checked 2026-11-11): CUDA 13.x runs on R580 or newer
/// (minor version compatibility; 13.0 R580, 13.1 R590, 13.2 R595, 13.3 R610,
/// 13.4 R615, 13.5 R620). CUDA 12.x needs 527.41+ on Windows (12.0 GA), with
/// a per-release Windows minimum (12.4 GA 551.61, 12.8 GA 570.65, ...).
/// None for versions the table doesn't cover.
pub fn cuda_driver_requirement(cuda: (u32, u32)) -> Option<DriverRequirement> {
    let recommended = match cuda {
        (13, 0) => (580, 0),
        (13, 1) => (590, 0),
        (13, 2) => (595, 0),
        (13, 3) => (610, 0),
        (13, 4) => (615, 0),
        (13, _) => (620, 0), // 13.5 and anything newer we don't know yet
        (12, 0) => (527, 41),
        (12, 1) => (531, 14),
        (12, 2) => (536, 25),
        (12, 3) => (545, 84),
        (12, 4) => (551, 61),
        (12, 5) => (555, 85),
        (12, 6) => (560, 76),
        (12, 8) => (570, 65),
        (12, _) => (576, 2), // 12.9
        _ => return None,
    };
    let minimum = if cuda.0 == 13 { (580, 0) } else { (527, 41) };
    Some(DriverRequirement { minimum, recommended })
}

/// "580.00" style text for a driver version tuple.
pub fn driver_text(v: (u32, u32)) -> String {
    format!("{}.{:02}", v.0, v.1)
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallProgress {
    pub runtime_id: String,
    /// download | extract | probe | done
    pub phase: String,
    pub asset: Option<String>,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub bytes_per_sec: f64,
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    digest: Option<String>,
}

pub struct RuntimeManager {
    dir: PathBuf,
    http: reqwest::Client,
}

fn parse_ver(s: &str) -> Option<(u32, u32)> {
    let mut it = s.split('.');
    Some((it.next()?.parse().ok()?, it.next().unwrap_or("0").parse().ok()?))
}

impl RuntimeManager {
    pub fn new(dir: PathBuf, http: reqwest::Client) -> Self {
        Self { dir, http }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn list_installed(&self) -> Vec<InstalledRuntime> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return out };
        for e in rd.flatten() {
            let m = e.path().join(MANIFEST);
            let Ok(text) = std::fs::read_to_string(&m) else { continue };
            match serde_json::from_str::<InstalledRuntime>(&text) {
                Ok(r) if r.server_path.is_file() => out.push(r),
                Ok(r) => tracing::warn!("runtime {} is missing {}", r.id, r.server_path.display()),
                Err(err) => tracing::warn!("bad runtime manifest {}: {err}", m.display()),
            }
        }
        // Newest build first; within a build, GPU backends before CPU so the
        // first entry is the best default on a machine with a GPU.
        out.sort_by(|a, b| b.tag.cmp(&a.tag).then(variant_rank(&a.variant).cmp(&variant_rank(&b.variant))));
        out
    }

    /// Probe `--help` again for runtimes whose manifest was written by an
    /// older capability parser (see `RUNTIME_CAPS_VERSION`), and save the
    /// result. Runs `llama-server --help` only; never loads a model.
    pub async fn reprobe_outdated(&self) {
        for mut r in self.list_installed() {
            if r.capabilities.version >= crate::capabilities::RUNTIME_CAPS_VERSION {
                continue;
            }
            match RuntimeCapabilities::probe(&r.server_path).await {
                Ok(caps) => {
                    r.capabilities = caps;
                    let path = self.dir.join(&r.id).join(MANIFEST);
                    match serde_json::to_vec_pretty(&r) {
                        Ok(bytes) if path.is_file() => {
                            if let Err(e) = std::fs::write(&path, bytes) {
                                tracing::warn!("could not update {}: {e}", path.display());
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => tracing::warn!("could not re-read the flags of runtime {}: {e}", r.id),
            }
        }
    }

    pub fn get(&self, id: &str) -> AppResult<InstalledRuntime> {
        self.list_installed()
            .into_iter()
            .find(|r| r.id == id)
            .ok_or_else(|| AppError::NotFound(format!("runtime {id} is not installed")))
    }

    /// Query GitHub for the newest release that has Windows x64 builds.
    pub async fn fetch_available(&self, gpu: &GpuHint) -> AppResult<Vec<AvailableRuntime>> {
        let mut last = String::new();
        let mut releases: Option<Vec<GhRelease>> = None;
        for attempt in 1..=2 {
            match self.fetch_releases_once().await {
                Ok(r) => {
                    releases = Some(r);
                    break;
                }
                Err(e) => {
                    tracing::warn!("GitHub releases fetch attempt {attempt}/2 failed: {e}");
                    last = e;
                    if attempt < 2 {
                        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
                    }
                }
            }
        }
        let Some(releases) = releases else {
            return Err(AppError::Network(format!(
                "could not read the llama.cpp release list from GitHub after 2 tries ({last}). Check your internet connection and try again."
            )));
        };
        let Some(rel) = releases
            .into_iter()
            .find(|r| r.assets.iter().any(|a| MAIN_ASSET_RE.is_match(&a.name)))
        else {
            return Err(AppError::NotFound("no recent llama.cpp release has Windows x64 builds".into()));
        };
        Ok(Self::variants_from_release(rel, gpu))
    }

    /// One attempt: whole request (headers + body) under a 45 s cap. The body is
    /// ~700 KB, so it is read fully and parsed after, to separate a dropped
    /// connection from a JSON problem in the error text.
    async fn fetch_releases_once(&self) -> Result<Vec<GhRelease>, String> {
        let fut = async {
            let resp = self
                .http
                .get(RELEASES_URL)
                .header(reqwest::header::USER_AGENT, "LocalAIStudio")
                .header(reqwest::header::ACCEPT, "application/vnd.github+json")
                .send()
                .await
                .map_err(|e| format!("request failed: {e}"))?;
            if !resp.status().is_success() {
                return Err(format!("GitHub returned HTTP {}", resp.status()));
            }
            let bytes = resp.bytes().await.map_err(|e| format!("download of the release list was cut off: {e}"))?;
            Self::parse_releases(&bytes)
        };
        match tokio::time::timeout(std::time::Duration::from_secs(45), fut).await {
            Ok(r) => r,
            Err(_) => Err("timed out after 45 s".into()),
        }
    }

    fn parse_releases(bytes: &[u8]) -> Result<Vec<GhRelease>, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("unexpected GitHub JSON: {e}"))
    }

    fn variants_from_release(rel: GhRelease, gpu: &GpuHint) -> Vec<AvailableRuntime> {
        let to_ref = |a: &GhAsset| AssetRef {
            name: a.name.clone(),
            url: a.browser_download_url.clone(),
            size: a.size,
            sha256: a.digest.as_deref().and_then(|d| d.strip_prefix("sha256:")).map(str::to_string),
        };
        let mut out = Vec::new();
        for a in &rel.assets {
            let Some(c) = MAIN_ASSET_RE.captures(&a.name) else { continue };
            let variant = c["variant"].to_string();
            let mut assets = vec![to_ref(a)];
            if variant.starts_with("cuda") {
                match rel.assets.iter().find(|x| CUDART_RE.captures(&x.name).is_some_and(|m| m["variant"] == variant)) {
                    Some(rt) => assets.push(to_ref(rt)),
                    None => continue, // unusable without the CUDA runtime DLLs
                }
            }
            out.push(AvailableRuntime {
                id: format!("{}-{variant}", rel.tag_name),
                tag: rel.tag_name.clone(),
                total_size: assets.iter().map(|x| x.size).sum(),
                assets,
                published_at: rel.published_at.clone().unwrap_or_default(),
                recommended: false,
                note: String::new(),
                variant,
            });
        }
        // Recommendation.
        let cuda_pick = if gpu.nvidia {
            let mut cudas: Vec<&AvailableRuntime> = out.iter().filter(|r| r.variant.starts_with("cuda")).collect();
            cudas.sort_by_key(|r| std::cmp::Reverse(parse_ver(r.variant.trim_start_matches("cuda-"))));
            cudas
                .into_iter()
                .find(|r| {
                    let v = parse_ver(r.variant.trim_start_matches("cuda-"));
                    let driver_ok = match (v, gpu.cuda_driver) {
                        (Some(v), Some(d)) => v.0 <= d.0, // same major works with minor compatibility
                        _ => true,
                    } && match (v.and_then(cuda_driver_requirement), gpu.driver_version) {
                        (Some(req), Some(d)) => d >= req.minimum,
                        _ => true,
                    };
                    // CUDA 12.4 has no Blackwell (sm_120) kernels; 12.8+ / 13.x do.
                    let arch_ok = match (v, gpu.compute_capability) {
                        (Some(v), Some(cc)) if cc.0 >= 12 => v >= (12, 8),
                        _ => true,
                    };
                    driver_ok && arch_ok
                })
                .map(|r| r.id.clone())
        } else {
            None
        };
        // Without a usable CUDA build, a GPU still beats the CPU through Vulkan
        // (an NVIDIA card on an old driver, or an AMD/Intel card).
        let has_vulkan = out.iter().any(|r| r.variant == "vulkan");
        let fallback = if has_vulkan && (gpu.nvidia || gpu.vulkan_gpu) { "vulkan" } else { "cpu" };
        for r in &mut out {
            r.note = match r.variant.as_str() {
                "cpu" => "CPU only. Works on any PC; slowest.".into(),
                "vulkan" => "Vulkan GPU backend (AMD, Intel or NVIDIA). Usually slower than CUDA on NVIDIA.".into(),
                v => {
                    let ver = parse_ver(v.trim_start_matches("cuda-"));
                    let mut n = format!("NVIDIA CUDA {}.", v.trim_start_matches("cuda-"));
                    if let (Some(ver), Some(cc)) = (ver, gpu.compute_capability)
                        && cc.0 >= 12
                        && ver < (12, 8)
                    {
                        n.push_str(" No kernels for this GPU's architecture (needs CUDA 12.8 or newer).");
                    }
                    let req = ver.and_then(cuda_driver_requirement);
                    let too_old_number = matches!((req, gpu.driver_version), (Some(r), Some(d)) if d < r.minimum);
                    if let (Some(ver), Some(d)) = (ver, gpu.cuda_driver)
                        && (ver.0 > d.0 || too_old_number)
                    {
                        n.push_str(&format!(" Needs a newer NVIDIA driver (driver supports CUDA {}.{}).", d.0, d.1));
                    } else if too_old_number && let Some(r) = req {
                        n.push_str(&format!(" Needs NVIDIA driver {} or newer.", driver_text(r.minimum)));
                    }
                    n
                }
            };
            r.recommended = match &cuda_pick {
                Some(id) => &r.id == id,
                None => r.variant == fallback,
            };
        }
        out.sort_by_key(|r| (!r.recommended, r.variant.clone()));
        out
    }

    /// Download, verify, extract and register a runtime. Safe to cancel: partial
    /// downloads stay as `.part` files and are resumed next time.
    pub async fn install(
        &self,
        rt: &AvailableRuntime,
        cancel: &CancellationToken,
        on_progress: &(dyn Fn(InstallProgress) + Send + Sync),
    ) -> AppResult<InstalledRuntime> {
        if !rt.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.') {
            return Err(AppError::InvalidInput("bad runtime id".into()));
        }
        let final_dir = self.dir.join(&rt.id);
        if final_dir.join(MANIFEST).is_file() {
            return self.get(&rt.id);
        }
        let dl_dir = self.dir.join(".downloads");
        let mut zips = Vec::new();
        for a in &rt.assets {
            if !a.url.starts_with("https://github.com/ggml-org/llama.cpp/releases/download/") {
                return Err(AppError::InvalidInput(format!("unexpected download URL {}", a.url)));
            }
            let dest = dl_dir.join(&a.name);
            if !dest.is_file() {
                let id = rt.id.clone();
                let name = a.name.clone();
                download::download_file(&self.http, &a.url, &dest, a.sha256.as_deref(), cancel, |p: Progress| {
                    on_progress(InstallProgress {
                        runtime_id: id.clone(),
                        phase: "download".into(),
                        asset: Some(name.clone()),
                        downloaded: p.downloaded,
                        total: p.total,
                        bytes_per_sec: p.bytes_per_sec,
                    })
                })
                .await
                .map_err(|e| match e {
                    DownloadError::Cancelled => AppError::Cancelled,
                    other => AppError::Network(other.to_string()),
                })?;
            }
            zips.push(dest);
        }

        on_progress(InstallProgress {
            runtime_id: rt.id.clone(),
            phase: "extract".into(),
            asset: None,
            downloaded: 0,
            total: None,
            bytes_per_sec: 0.0,
        });
        let staging = self.dir.join(format!(".staging-{}", rt.id));
        let _ = std::fs::remove_dir_all(&staging);
        let st = staging.clone();
        let zs = zips.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            for z in zs {
                extract_zip(&z, &st)?;
            }
            Ok(())
        })
        .await
        .map_err(|e| AppError::Other(e.to_string()))??;

        let server = find_file(&staging, "llama-server.exe")
            .ok_or_else(|| AppError::Runtime("llama-server.exe not found in the downloaded build".into()))?;
        let rel = server.strip_prefix(&staging).unwrap().to_path_buf();
        on_progress(InstallProgress {
            runtime_id: rt.id.clone(),
            phase: "probe".into(),
            asset: None,
            downloaded: 0,
            total: None,
            bytes_per_sec: 0.0,
        });
        let caps = RuntimeCapabilities::probe(&server)
            .await
            .map_err(|e| AppError::Runtime(format!("downloaded llama-server does not run: {e}")))?;
        let _ = std::fs::remove_dir_all(&final_dir);
        std::fs::rename(&staging, &final_dir)?;
        let installed = InstalledRuntime {
            id: rt.id.clone(),
            tag: rt.tag.clone(),
            variant: rt.variant.clone(),
            server_path: final_dir.join(rel),
            installed_at: app_core::now_rfc3339(),
            source: "official".into(),
            capabilities: caps,
        };
        std::fs::write(final_dir.join(MANIFEST), serde_json::to_vec_pretty(&installed)?)?;
        for z in zips {
            let _ = std::fs::remove_file(z);
        }
        on_progress(InstallProgress {
            runtime_id: rt.id.clone(),
            phase: "done".into(),
            asset: None,
            downloaded: 0,
            total: None,
            bytes_per_sec: 0.0,
        });
        Ok(installed)
    }

    /// Register an existing llama-server.exe without copying it.
    pub async fn register_custom(&self, server: &Path, variant_hint: Option<&str>) -> AppResult<InstalledRuntime> {
        if !server.is_file() {
            return Err(AppError::NotFound(format!("{} does not exist", server.display())));
        }
        let caps = RuntimeCapabilities::probe(server)
            .await
            .map_err(|e| AppError::Runtime(format!("{} is not a working llama-server: {e}", server.display())))?;
        let n = self.list_installed().iter().filter(|r| r.source == "custom").count() + 1;
        let id = format!("custom-{n}");
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)?;
        let r = InstalledRuntime {
            id,
            tag: "custom".into(),
            variant: variant_hint.unwrap_or("custom").into(),
            server_path: server.to_path_buf(),
            installed_at: app_core::now_rfc3339(),
            source: "custom".into(),
            capabilities: caps,
        };
        std::fs::write(dir.join(MANIFEST), serde_json::to_vec_pretty(&r)?)?;
        Ok(r)
    }

    /// Remove a runtime. Custom runtimes only lose their manifest; the user's
    /// own llama-server.exe is never deleted.
    pub fn remove(&self, id: &str) -> AppResult<()> {
        let r = self.get(id)?;
        let dir = self.dir.join(&r.id);
        if !dir.starts_with(&self.dir) || r.id.contains("..") {
            return Err(AppError::InvalidInput("bad runtime id".into()));
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}

/// Sort key: lower is preferred.
fn variant_rank(v: &str) -> u8 {
    if v.starts_with("cuda") {
        0
    } else if v == "vulkan" {
        1
    } else if v == "cpu" {
        3
    } else {
        2
    }
}

fn extract_zip(zip_path: &Path, dest: &Path) -> std::io::Result<()> {
    let f = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f).map_err(std::io::Error::other)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(std::io::Error::other)?;
        // enclosed_name rejects absolute paths and `..` (zip-slip).
        let Some(rel) = entry.enclosed_name() else { continue };
        let out = dest.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut w = std::fs::File::create(&out)?;
        std::io::copy(&mut entry, &mut w)?;
    }
    Ok(())
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(name)) {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> GhAsset {
        GhAsset {
            name: name.into(),
            browser_download_url: format!("https://github.com/ggml-org/llama.cpp/releases/download/b1/{name}"),
            size: 10,
            digest: Some("sha256:ab".into()),
        }
    }

    fn release() -> GhRelease {
        GhRelease {
            tag_name: "b12246".into(),
            published_at: None,
            assets: [
                "llama-b12246-bin-win-cpu-x64.zip",
                "llama-b12246-bin-win-cuda-12.4-x64.zip",
                "llama-b12246-bin-win-cuda-13.5-x64.zip",
                "llama-b12246-bin-win-vulkan-x64.zip",
                "llama-b12246-bin-win-cpu-arm64.zip",
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
                "cudart-llama-bin-win-cuda-13.5-x64.zip",
                "llama-b12246-bin-ubuntu-x64.tar.gz",
            ]
            .into_iter()
            .map(asset)
            .collect(),
        }
    }

    #[test]
    fn parses_saved_github_response() {
        let bytes = include_bytes!("../tests/fixtures/github_releases.json");
        let rels = RuntimeManager::parse_releases(bytes).unwrap();
        assert_eq!(rels.len(), 2);
        let rel = rels.into_iter().find(|r| r.assets.iter().any(|a| MAIN_ASSET_RE.is_match(&a.name))).unwrap();
        let v = RuntimeManager::variants_from_release(rel, &GpuHint::default());
        assert!(v.iter().any(|r| r.variant == "cpu" && r.recommended));
        assert!(RuntimeManager::parse_releases(b"{}").is_err());
    }

    #[test]
    fn blackwell_gets_cuda_13_with_cudart_bundled() {
        let gpu = GpuHint {
            nvidia: true,
            compute_capability: Some((12, 0)),
            cuda_driver: Some((13, 5)),
            driver_version: Some((621, 1)),
            ..Default::default()
        };
        let v = RuntimeManager::variants_from_release(release(), &gpu);
        assert_eq!(v.len(), 4, "cpu, cuda 12.4, cuda 13.5, vulkan");
        let rec: Vec<_> = v.iter().filter(|r| r.recommended).collect();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].id, "b12246-cuda-13.5");
        assert_eq!(rec[0].assets.len(), 2);
        assert_eq!(rec[0].assets[0].sha256.as_deref(), Some("ab"));
        let old = v.iter().find(|r| r.variant == "cuda-12.4").unwrap();
        assert!(old.note.contains("needs CUDA 12.8"), "{}", old.note);
    }

    #[test]
    fn older_driver_and_no_gpu_choices() {
        let ampere_old_driver = GpuHint {
            nvidia: true,
            compute_capability: Some((8, 6)),
            cuda_driver: Some((12, 6)),
            driver_version: Some((560, 94)),
            ..Default::default()
        };
        let v = RuntimeManager::variants_from_release(release(), &ampere_old_driver);
        assert_eq!(v.iter().find(|r| r.recommended).unwrap().id, "b12246-cuda-12.4");
        let none = RuntimeManager::variants_from_release(release(), &GpuHint::default());
        assert_eq!(none.iter().find(|r| r.recommended).unwrap().variant, "cpu");
    }

    #[test]
    fn driver_too_old_for_any_cuda_build_falls_back_to_vulkan() {
        // Blackwell needs CUDA 12.8+, so 12.4 is out; driver 572 is below R580 for 13.x.
        let gpu = GpuHint {
            nvidia: true,
            compute_capability: Some((12, 0)),
            cuda_driver: Some((12, 8)),
            driver_version: Some((572, 16)),
            ..Default::default()
        };
        let v = RuntimeManager::variants_from_release(release(), &gpu);
        assert_eq!(v.iter().find(|r| r.recommended).unwrap().variant, "vulkan");
        let c13 = v.iter().find(|r| r.variant == "cuda-13.5").unwrap();
        assert!(c13.note.contains("newer NVIDIA driver"), "{}", c13.note);
        // NVML said CUDA 13.0 but the driver number is too old: the number wins.
        let odd = GpuHint { cuda_driver: Some((13, 0)), driver_version: Some((575, 0)), ..gpu.clone() };
        let v = RuntimeManager::variants_from_release(release(), &odd);
        assert_eq!(v.iter().find(|r| r.recommended).unwrap().variant, "vulkan");
        // AMD/Intel card with its own VRAM: Vulkan, not CPU.
        let amd = GpuHint { vulkan_gpu: true, ..Default::default() };
        let v = RuntimeManager::variants_from_release(release(), &amd);
        assert_eq!(v.iter().find(|r| r.recommended).unwrap().variant, "vulkan");
    }

    #[test]
    fn cuda_driver_table() {
        let r = cuda_driver_requirement((13, 5)).unwrap();
        assert_eq!(r.minimum, (580, 0));
        assert_eq!(r.recommended, (620, 0));
        assert_eq!(cuda_driver_requirement((12, 4)).unwrap().recommended, (551, 61));
        assert_eq!(cuda_driver_requirement((12, 4)).unwrap().minimum, (527, 41));
        assert!(cuda_driver_requirement((11, 8)).is_none());
        assert_eq!(parse_driver_version("621.01"), Some((621, 1)));
        assert!(parse_driver_version("621.01").unwrap() >= r.recommended);
        assert!(parse_driver_version("579.99").unwrap() < r.minimum);
        assert_eq!(driver_text((580, 0)), "580.00");
        assert_eq!(driver_text((551, 61)), "551.61");
    }

    #[test]
    fn zip_slip_entries_are_skipped() {
        use std::io::Write;
        let d = tempfile::tempdir().unwrap();
        let zp = d.path().join("evil.zip");
        {
            let mut w = zip::ZipWriter::new(std::fs::File::create(&zp).unwrap());
            let o: zip::write::SimpleFileOptions = Default::default();
            w.start_file("../escape.txt", o).unwrap();
            w.write_all(b"x").unwrap();
            w.start_file("bin/llama-server.exe", o).unwrap();
            w.write_all(b"y").unwrap();
            w.finish().unwrap();
        }
        let out = d.path().join("out");
        extract_zip(&zp, &out).unwrap();
        assert!(!d.path().join("escape.txt").exists());
        assert!(find_file(&out, "llama-server.exe").is_some());
    }
}

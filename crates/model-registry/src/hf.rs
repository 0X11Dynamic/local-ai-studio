//! Hugging Face Hub client for GGUF discovery.
//!
//! Everything shown in Discover comes from these real API responses:
//! search results, repository details, and the repository's actual file tree
//! (exact file names, sizes and LFS SHA-256). Quantization variants are derived
//! from the real file names; nothing is synthesized.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use app_core::{AppError, AppResult};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::scan::quant_from_filename;

pub const HF_BASE: &str = "https://huggingface.co";

static SHARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$").unwrap());
static REPO_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap());

#[derive(Debug, Clone)]
pub struct HfClient {
    http: reqwest::Client,
    base: String,
    token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub created_at: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub base_model: Option<String>,
    /// Derived from tags: text, vision, embedding.
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoDetails {
    pub id: String,
    pub author: String,
    /// Commit the file list and downloads are pinned to.
    pub sha: String,
    pub last_modified: Option<String>,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub gated: bool,
    pub downloads: u64,
    pub likes: u64,
    pub base_model: Option<String>,
    /// From the Hub's GGUF metadata, when the repo exposes it.
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    /// Parameter count reported by the Hub for the GGUF files.
    pub parameters: Option<u64>,
    pub capabilities: Vec<String>,
    /// Detected from the Hub's GGUF chat template (`gguf.chat_template`) and,
    /// once the file list is known, its projector files. Unknown stays `None`.
    #[serde(default)]
    pub caps: crate::caps::Capabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// LFS SHA-256 when the file is stored in LFS (all large GGUFs are).
    pub sha256: Option<String>,
}

/// One downloadable choice: a single GGUF or a complete shard set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Variant {
    /// Stable key: the first file's path.
    pub key: String,
    /// Quantization parsed from the file name, e.g. "Q4_K_M". None if unrecognised.
    pub quantization: Option<String>,
    /// Exact file name(s) as hosted.
    pub files: Vec<RepoFile>,
    pub total_size: u64,
    pub shard_count: u32,
    /// False when the repo is missing parts of a shard set.
    pub complete: bool,
    /// Vision projector files (mmproj) are offered as add-ons, not as models.
    pub is_projector: bool,
}

fn caps_from(tags: &[String], pipeline: Option<&str>) -> Vec<String> {
    let mut c = Vec::new();
    let has = |t: &str| tags.iter().any(|x| x == t);
    let p = pipeline.unwrap_or("");
    if p.contains("image-text") || p.contains("image-to-text") || has("vision") || has("multimodal") {
        c.push("vision".to_string());
    }
    if p == "feature-extraction" || p == "sentence-similarity" || has("sentence-transformers") || has("embeddings") {
        c.push("embedding".to_string());
    }
    if p.contains("text-generation") || has("conversational") || p.contains("image-text-to-text") {
        c.push("text".to_string());
    }
    c
}

fn tag_value(tags: &[String], prefix: &str) -> Option<String> {
    tags.iter().find_map(|t| t.strip_prefix(prefix)).map(str::to_string)
}

/// Fill in what the file list says: a projector file in the repo means vision
/// is available as a download.
pub fn apply_repo_files(d: &mut RepoDetails, variants: &[Variant]) {
    let has_projector = variants.iter().any(|v| v.is_projector);
    let tags_vision = d.capabilities.iter().any(|c| c == "vision");
    crate::caps::apply_repo_projector(&mut d.caps, has_projector, tags_vision);
}

/// Accept `owner/repo`, `https://huggingface.co/owner/repo[/...]`, `hf.co/owner/repo`.
pub fn parse_repo_input(input: &str) -> Option<String> {
    let s = input.trim().trim_end_matches('/');
    let s = s
        .strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s);
    let s = s
        .strip_prefix("huggingface.co/")
        .or_else(|| s.strip_prefix("www.huggingface.co/"))
        .or_else(|| s.strip_prefix("hf.co/"))
        .unwrap_or(s);
    let mut parts = s.split('/');
    let (a, b) = (parts.next()?, parts.next()?);
    let id = format!("{a}/{b}");
    REPO_RE.is_match(&id).then_some(id)
}

/// Group a repository's GGUF files into variants (shard sets merged).
pub fn group_variants(files: &[RepoFile]) -> Vec<Variant> {
    let mut singles = Vec::new();
    let mut sets: BTreeMap<(String, String, u32), BTreeMap<u32, RepoFile>> = BTreeMap::new();
    for f in files.iter().filter(|f| f.path.to_ascii_lowercase().ends_with(".gguf")) {
        let (dir, name) = match f.path.rsplit_once('/') {
            Some((d, n)) => (d.to_string(), n.to_string()),
            None => (String::new(), f.path.clone()),
        };
        if let Some(c) = SHARD_RE.captures(&name) {
            let no: u32 = c["no"].parse().unwrap_or(0);
            let count: u32 = c["count"].parse().unwrap_or(0);
            sets.entry((dir, c["base"].to_string(), count)).or_default().insert(no, f.clone());
        } else {
            singles.push(f.clone());
        }
    }
    let is_proj = |p: &str| {
        let n = p.rsplit('/').next().unwrap_or(p).to_ascii_lowercase();
        n.starts_with("mmproj") || n.contains("mmproj")
    };
    let mut out: Vec<Variant> = singles
        .into_iter()
        .map(|f| {
            let name = f.path.rsplit('/').next().unwrap_or(&f.path).to_string();
            Variant {
                key: f.path.clone(),
                quantization: quant_from_filename(&name),
                total_size: f.size,
                shard_count: 1,
                complete: true,
                is_projector: is_proj(&f.path),
                files: vec![f],
            }
        })
        .collect();
    for ((_, base, count), set) in sets {
        let complete = (1..=count).all(|n| set.contains_key(&n));
        let files: Vec<RepoFile> = set.into_values().collect();
        out.push(Variant {
            key: files[0].path.clone(),
            quantization: quant_from_filename(&format!("{base}.gguf")),
            total_size: files.iter().map(|f| f.size).sum(),
            shard_count: count,
            complete,
            is_projector: is_proj(&base),
            files,
        });
    }
    out.sort_by(|a, b| a.is_projector.cmp(&b.is_projector).then(a.total_size.cmp(&b.total_size)));
    out
}

/// Per-variant fit for this PC. Labelled as an estimate in the UI.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FitEstimate {
    /// Weights plus about 1 GB for context and buffers.
    pub est_memory_bytes: u64,
    /// "gpu" (fits fully in free VRAM), "partial" (GPU + RAM), "cpu" (RAM only), "too_big".
    pub fit: String,
}

pub fn estimate_fit(total_size: u64, free_vram: Option<u64>, available_ram: u64) -> FitEstimate {
    let need = total_size + (1u64 << 30);
    let fit = match free_vram {
        Some(v) if need <= v => "gpu",
        Some(v) if need <= v + available_ram => "partial",
        _ if need <= available_ram => "cpu",
        _ => "too_big",
    };
    FitEstimate { est_memory_bytes: need, fit: fit.into() }
}

/// The variant to recommend: the largest non-projector variant that fits fully
/// in free VRAM, else the largest that fits in VRAM + RAM. Q4_K_M is preferred
/// when it fits, as the usual quality/speed balance.
pub fn recommend(variants: &[Variant], free_vram: Option<u64>, available_ram: u64) -> Option<String> {
    let models: Vec<&Variant> = variants.iter().filter(|v| !v.is_projector && v.complete).collect();
    let fits = |v: &&Variant, levels: &[&str]| levels.contains(&estimate_fit(v.total_size, free_vram, available_ram).fit.as_str());
    let q4km = models.iter().find(|v| v.quantization.as_deref() == Some("Q4_K_M") && fits(v, &["gpu"]));
    if let Some(v) = q4km {
        return Some(v.key.clone());
    }
    let pick = |levels: &[&str]| {
        models
            .iter()
            .filter(|v| fits(v, levels))
            // Avoid recommending the unquantized/huge originals over a good quant.
            .filter(|v| !matches!(v.quantization.as_deref(), Some("F32") | Some("BF16") | Some("F16")))
            .max_by_key(|v| v.total_size)
            .map(|v| v.key.clone())
    };
    pick(&["gpu"]).or_else(|| pick(&["gpu", "partial"]))
}

#[derive(Deserialize)]
struct RawSearch {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct RawTree {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    lfs: Option<RawLfs>,
}

#[derive(Deserialize)]
struct RawLfs {
    oid: String,
    #[serde(default)]
    size: u64,
}

impl HfClient {
    pub fn new(http: reqwest::Client, token: Option<String>) -> Self {
        Self { http, base: HF_BASE.into(), token }
    }

    /// For tests against a local mock server.
    pub fn with_base(mut self, base: &str) -> Self {
        self.base = base.trim_end_matches('/').into();
        self
    }

    fn get(&self, url: String) -> reqwest::RequestBuilder {
        let mut r = self.http.get(url);
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        r
    }

    async fn get_json<T: for<'de> Deserialize<'de>>(&self, url: String, what: &str) -> AppResult<T> {
        let resp = self
            .get(url)
            .send()
            .await
            .map_err(|e| AppError::Network(format!("Hugging Face {what} request failed: {e}")))?;
        match resp.status() {
            s if s.is_success() => resp
                .json()
                .await
                .map_err(|e| AppError::Network(format!("Hugging Face {what}: unexpected response: {e}"))),
            reqwest::StatusCode::NOT_FOUND => Err(AppError::NotFound(format!("{what}: not found on Hugging Face"))),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => Err(AppError::PermissionDenied(format!(
                "{what}: Hugging Face refused access (gated or private repo; add a token in Settings)"
            ))),
            reqwest::StatusCode::TOO_MANY_REQUESTS => Err(AppError::Network("Hugging Face rate limit reached; try again shortly".into())),
            s => Err(AppError::Network(format!("Hugging Face {what} returned HTTP {s}"))),
        }
    }

    /// Search GGUF repositories. `sort` is one of downloads, likes, createdAt, lastModified.
    pub async fn search(&self, query: &str, sort: &str, limit: u32) -> AppResult<Vec<SearchResult>> {
        let sort = match sort {
            "likes" | "createdAt" | "lastModified" | "downloads" => sort,
            _ => "downloads",
        };
        let mut url = reqwest::Url::parse(&format!("{}/api/models", self.base)).map_err(|e| AppError::Other(e.to_string()))?;
        url.query_pairs_mut()
            .append_pair("filter", "gguf")
            .append_pair("sort", sort)
            .append_pair("direction", "-1")
            .append_pair("limit", &limit.clamp(1, 100).to_string());
        if !query.trim().is_empty() {
            url.query_pairs_mut().append_pair("search", query.trim());
        }
        let raw: Vec<RawSearch> = self.get_json(url.to_string(), "search").await?;
        Ok(raw
            .into_iter()
            .map(|r| {
                let author = r.id.split('/').next().unwrap_or("").to_string();
                SearchResult {
                    license: tag_value(&r.tags, "license:"),
                    base_model: tag_value(&r.tags, "base_model:quantized:").or_else(|| tag_value(&r.tags, "base_model:")),
                    capabilities: caps_from(&r.tags, r.pipeline_tag.as_deref()),
                    id: r.id,
                    author,
                    downloads: r.downloads,
                    likes: r.likes,
                    created_at: r.created_at,
                    pipeline_tag: r.pipeline_tag,
                    tags: r.tags,
                }
            })
            .collect())
    }

    pub async fn details(&self, repo: &str) -> AppResult<RepoDetails> {
        if !REPO_RE.is_match(repo) {
            return Err(AppError::InvalidInput(format!("'{repo}' is not a Hugging Face repo id (owner/name)")));
        }
        let v: serde_json::Value = self.get_json(format!("{}/api/models/{repo}", self.base), repo).await?;
        let tags: Vec<String> = v["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let pipeline = v["pipeline_tag"].as_str().map(str::to_string);
        Ok(RepoDetails {
            id: v["id"].as_str().unwrap_or(repo).to_string(),
            author: v["author"].as_str().map(str::to_string).unwrap_or_else(|| repo.split('/').next().unwrap_or("").to_string()),
            sha: v["sha"].as_str().unwrap_or("main").to_string(),
            last_modified: v["lastModified"].as_str().map(str::to_string),
            license: v["cardData"]["license"].as_str().map(str::to_string).or_else(|| tag_value(&tags, "license:")),
            gated: v["gated"].as_bool().unwrap_or(false) || v["gated"].as_str().is_some(),
            downloads: v["downloads"].as_u64().unwrap_or(0),
            likes: v["likes"].as_u64().unwrap_or(0),
            base_model: match &v["cardData"]["base_model"] {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Array(a) => a.first().and_then(|x| x.as_str()).map(str::to_string),
                _ => None,
            },
            architecture: v["gguf"]["architecture"].as_str().map(str::to_string),
            context_length: v["gguf"]["context_length"].as_u64(),
            parameters: v["gguf"]["total"].as_u64(),
            capabilities: caps_from(&tags, pipeline.as_deref()),
            caps: crate::caps::from_template(v["gguf"]["chat_template"].as_str(), "hf"),
            pipeline_tag: pipeline,
            tags,
        })
    }

    /// Every file in the repository at `revision` (recursive).
    pub async fn files(&self, repo: &str, revision: &str) -> AppResult<Vec<RepoFile>> {
        if !REPO_RE.is_match(repo) {
            return Err(AppError::InvalidInput("bad repo id".into()));
        }
        let raw: Vec<RawTree> = self
            .get_json(format!("{}/api/models/{repo}/tree/{revision}?recursive=true", self.base), "file list")
            .await?;
        Ok(raw
            .into_iter()
            .filter(|t| t.kind == "file")
            .map(|t| RepoFile {
                size: t.lfs.as_ref().map(|l| l.size).filter(|s| *s > 0).unwrap_or(t.size),
                sha256: t.lfs.map(|l| l.oid).filter(|o| o.len() == 64),
                path: t.path,
            })
            .collect())
    }

    /// Direct download URL for a file pinned to a revision.
    pub fn file_url(&self, repo: &str, revision: &str, path: &str) -> String {
        let enc: Vec<String> = path.split('/').map(|seg| urlencode(seg)).collect();
        format!("{}/{repo}/resolve/{revision}/{}", self.base, enc.join("/"))
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile { path: path.into(), size, sha256: Some("a".repeat(64)) }
    }

    #[test]
    fn parses_repo_inputs() {
        assert_eq!(parse_repo_input("unsloth/Qwen3-4B-GGUF").as_deref(), Some("unsloth/Qwen3-4B-GGUF"));
        assert_eq!(parse_repo_input("https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main").as_deref(), Some("unsloth/Qwen3-4B-GGUF"));
        assert_eq!(parse_repo_input("hf.co/bartowski/x.y-GGUF").as_deref(), Some("bartowski/x.y-GGUF"));
        assert_eq!(parse_repo_input("qwen 4b"), None);
        assert_eq!(parse_repo_input("../etc/passwd"), None);
    }

    #[test]
    fn groups_singles_shards_and_projectors_with_exact_names() {
        let files = vec![
            f(".gitattributes", 10),
            f("Model-Q4_K_M.gguf", 2_500),
            f("Model-Q8_0.gguf", 4_200),
            f("Model-UD-Q4_K_XL.gguf", 2_600),
            f("mmproj-F16.gguf", 800),
            f("BF16/Model-BF16-00001-of-00003.gguf", 5_000),
            f("BF16/Model-BF16-00002-of-00003.gguf", 5_000),
            f("BF16/Model-BF16-00003-of-00003.gguf", 1_000),
            f("Q2/Model-Q2_K-00001-of-00002.gguf", 700),
        ];
        let v = group_variants(&files);
        assert_eq!(v.len(), 6, "{v:#?}");
        let bf16 = v.iter().find(|x| x.quantization.as_deref() == Some("BF16")).unwrap();
        assert_eq!((bf16.shard_count, bf16.files.len(), bf16.total_size, bf16.complete), (3, 3, 11_000, true));
        let q2 = v.iter().find(|x| x.quantization.as_deref() == Some("Q2_K")).unwrap();
        assert!(!q2.complete, "Q2 set is missing part 2");
        assert!(v.iter().find(|x| x.key == "mmproj-F16.gguf").unwrap().is_projector);
        assert_eq!(v.iter().find(|x| x.key == "Model-UD-Q4_K_XL.gguf").unwrap().quantization.as_deref(), Some("Q4_K_XL"));
        assert!(v.last().unwrap().is_projector, "projectors sort last");
    }

    #[test]
    fn fit_and_recommendation() {
        let gb = 1_000_000_000u64;
        let files = vec![f("m-Q4_K_M.gguf", 5 * gb), f("m-Q8_0.gguf", 8 * gb), f("m-Q3_K_M.gguf", 4 * gb), f("m-F16.gguf", 15 * gb)];
        let v = group_variants(&files);
        // 7 GB free VRAM: Q4_K_M fits fully, preferred.
        assert_eq!(recommend(&v, Some(7 * gb), 16 * gb).as_deref(), Some("m-Q4_K_M.gguf"));
        // 4.5 GB free VRAM: nothing fits fully except... Q3_K_M needs 5.07 GB -> none; pick largest partial non-F16.
        assert_eq!(recommend(&v, Some(4_500_000_000), 16 * gb).as_deref(), Some("m-Q8_0.gguf"));
        assert_eq!(estimate_fit(15 * gb, Some(7 * gb), 4 * gb).fit, "too_big");
        assert_eq!(estimate_fit(5 * gb, None, 16 * gb).fit, "cpu");
    }

    #[test]
    fn file_urls_are_pinned_and_encoded() {
        let c = HfClient::new(reqwest::Client::new(), None);
        assert_eq!(
            c.file_url("o/r", "abc123", "BF16/Model BF16-00001-of-00002.gguf"),
            "https://huggingface.co/o/r/resolve/abc123/BF16/Model%20BF16-00001-of-00002.gguf"
        );
    }

    /// Live: real search, real file list with true sizes (network).
    #[tokio::test]
    #[ignore = "network"]
    async fn live_hub_search_and_variants() {
        let c = HfClient::new(reqwest::Client::builder().user_agent("LocalAIStudio-test").build().unwrap(), None);
        let r = c.search("qwen3 4b", "downloads", 5).await.unwrap();
        assert!(!r.is_empty());
        let d = c.details("unsloth/Qwen3-4B-GGUF").await.unwrap();
        assert_eq!(d.architecture.as_deref(), Some("qwen3"));
        let files = c.files(&d.id, &d.sha).await.unwrap();
        let v = group_variants(&files);
        let q4 = v.iter().find(|x| x.key == "Qwen3-4B-Q4_K_M.gguf").unwrap();
        let q8 = v.iter().find(|x| x.key == "Qwen3-4B-Q8_0.gguf").unwrap();
        assert_eq!(q4.total_size, 2_497_281_312);
        assert_eq!(q8.total_size, 4_280_405_792);
        assert!(q4.files[0].sha256.as_deref().unwrap().starts_with("f6f8517777098610"));
        let big = c.details("unsloth/Qwen3-235B-A22B-GGUF").await.unwrap();
        let bv = group_variants(&c.files(&big.id, &big.sha).await.unwrap());
        let bf16 = bv.iter().find(|x| x.quantization.as_deref() == Some("BF16")).unwrap();
        assert_eq!(bf16.shard_count, 10);
        assert!(bf16.complete);
        println!("{} variants in 235B repo; BF16 = {} files, {:.0} GB", bv.len(), bf16.files.len(), bf16.total_size as f64 / 1e9);
    }
}

//! Discover GGUF files on disk and group them into importable models:
//! multi-shard sets (`name-00001-of-00003.gguf`) become one model, vision
//! projector files (`mmproj-*.gguf`) attach to the model in the same folder
//! they belong to (see [`pick_projector`]), and unreadable files are reported
//! instead of registered.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

use crate::gguf::{self, ModelMetadata};

static SHARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$").unwrap());
static QUANT_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Longest forms first: "Q4_K_XL" must not be read as "Q4_K".
    Regex::new(r"(?i)(?:^|[-_.])(?P<q>I?Q\d_K_XL|I?Q\d_K_[SML]|I?Q\d_K|IQ\d_[A-Z]+|Q\d_\d|MXFP4|BF16|F16|F32|TQ\d_\d)(?:[-_.]|$)").unwrap()
});

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredModel {
    /// First shard, or the single file.
    pub path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
    pub total_size: u64,
    pub metadata: ModelMetadata,
    pub quantization: Option<String>,
    /// True when the quantization came from the file name, not the header.
    pub quantization_from_filename: bool,
    pub display_name: String,
    pub projector_path: Option<PathBuf>,
    /// Missing shards etc. A model with problems is registered as "invalid".
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkippedFile {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ScanResult {
    pub models: Vec<DiscoveredModel>,
    pub skipped: Vec<SkippedFile>,
}

pub fn quant_from_filename(name: &str) -> Option<String> {
    QUANT_RE.captures(name).map(|c| c["q"].to_ascii_uppercase())
}

fn is_projector_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("mmproj") || n.contains(".mmproj") || n.contains("-mmproj")
}

/// Words that name the model a file belongs to: the file name without the
/// shard suffix, `.gguf`, quantization, `mmproj`, imatrix (`i1`) and
/// Unsloth Dynamic (`UD`) tags,
/// lowercased and split on punctuation. "mmproj-gemma-4-E4B-it-BF16.gguf" and
/// "gemma-4-E4B-it-Q4_K_M.gguf" both give `gemma 4 e4b it`; "mmproj-F16.gguf"
/// gives nothing (a generic projector name).
pub(crate) fn name_key(file_name: &str) -> Vec<String> {
    let mut s = match SHARD_RE.captures(file_name) {
        Some(c) => c["base"].to_string(),
        None if file_name.to_ascii_lowercase().ends_with(".gguf") => file_name[..file_name.len() - 5].to_string(),
        None => file_name.to_string(),
    };
    // Each match swallows its delimiters, so put one back.
    while let Some(m) = QUANT_RE.find(&s) {
        s.replace_range(m.range(), "-");
    }
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .filter(|t| !matches!(t.as_str(), "mmproj" | "gguf" | "i1" | "ud"))
        .collect()
}

/// A projector file in a folder, as far as matching needs it.
#[derive(Debug, Clone)]
pub(crate) struct ProjectorFile {
    pub path: PathBuf,
    /// [`name_key`] of the file name; empty or `model` = a generic name.
    pub key: Vec<String>,
    /// `clip.vision.projection_dim` (else the audio one): the embedding width
    /// of the text model the projector feeds, when the header says.
    pub projection_dim: Option<u64>,
    /// None = the header could not be read.
    pub kinds: Option<crate::caps::ProjectorKinds>,
}

impl ProjectorFile {
    fn read(path: PathBuf) -> Self {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let header = gguf::read_header(&path).ok();
        let projection_dim = header.as_ref().and_then(|h| {
            h.kv.get("clip.vision.projection_dim")
                .or_else(|| h.kv.get("clip.audio.projection_dim"))
                .and_then(gguf::Value::as_u64)
        });
        let kinds = header.as_ref().map(crate::caps::projector_kinds);
        ProjectorFile { path, key: name_key(&name), projection_dim, kinds }
    }

    fn named(&self) -> bool {
        !self.key.is_empty() && self.key != ["model"]
    }

    /// F16/BF16 first, then F32, unknown, Q8_0, lower precision.
    fn precision_rank(&self) -> u8 {
        let name = self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match quant_from_filename(&name).as_deref() {
            Some("F16" | "BF16") => 0,
            Some("F32") => 1,
            None => 2,
            Some("Q8_0") => 3,
            _ => 4,
        }
    }
}

/// One name key starts the other ("qwen3 4b" and "qwen3 4b instruct").
fn related(a: &[String], b: &[String]) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    !short.is_empty() && long[..short.len()] == *short
}

/// Pick the projector for one model from the projectors in its folder.
/// `folder_models` holds the name key of every model in that folder.
///
/// 1. Projectors whose name key equals the model's.
/// 2. Else projectors whose key starts (or is the start of) the model's key
///    and of no other model's key in the folder.
/// 3. Else, only when the folder holds one model (possibly in several
///    quantizations), any projector there, such as a plain "mmproj-F16.gguf".
///
/// With several models in the folder, a projector whose header gives a
/// projection width different from the model's embedding width is never used.
/// Several matches: F16/BF16 before other precisions, then by file name.
/// No confident match: `None` (no vision) rather than a guess.
pub(crate) fn pick_projector<'a>(
    model_key: &[String],
    model_embd: Option<u64>,
    folder_models: &BTreeSet<Vec<String>>,
    projectors: &'a [ProjectorFile],
) -> Option<&'a ProjectorFile> {
    let multi = folder_models.len() > 1;
    let pool: Vec<&ProjectorFile> = projectors
        .iter()
        .filter(|p| !multi || !matches!((p.projection_dim, model_embd), (Some(a), Some(b)) if a != b))
        .collect();
    let exact: Vec<&ProjectorFile> = pool.iter().copied().filter(|p| p.named() && p.key == model_key).collect();
    let candidates = if !exact.is_empty() {
        exact
    } else {
        let prefix: Vec<&ProjectorFile> = pool
            .iter()
            .copied()
            .filter(|p| {
                p.named() && related(&p.key, model_key) && folder_models.iter().filter(|k| related(&p.key, k)).count() == 1
            })
            .collect();
        if !prefix.is_empty() {
            prefix
        } else if !multi {
            pool
        } else {
            Vec::new()
        }
    };
    candidates
        .into_iter()
        .min_by_key(|p| (p.precision_rank(), p.path.file_name().map(|n| n.to_string_lossy().to_lowercase()), p.path.clone()))
}

/// Scan a folder recursively (or a single file) for GGUF models.
pub fn scan(root: &Path) -> ScanResult {
    // folder -> name keys of models there that this scan does not register
    // (siblings of a single imported file), so matching sees the whole folder.
    let mut sibling_models: BTreeMap<PathBuf, BTreeSet<Vec<String>>> = BTreeMap::new();
    let files: Vec<PathBuf> = if root.is_file() {
        // A single file: also pick up its vision projector and the other
        // shards of its set from the same folder (but no unrelated models).
        let mut v = vec![root.to_path_buf()];
        let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let base = SHARD_RE.captures(&name).map(|c| (c["base"].to_string(), c["count"].to_string()));
        if let Some(dir) = root.parent() {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.filter_map(Result::ok) {
                    let p = e.path();
                    if p == root || !p.is_file() || !p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) {
                        continue;
                    }
                    let n = e.file_name().to_string_lossy().into_owned();
                    let same_set = base.as_ref().is_some_and(|(b, c)| SHARD_RE.captures(&n).is_some_and(|x| &x["base"] == b && &x["count"] == c));
                    if is_projector_name(&n) || same_set {
                        v.push(p);
                    } else {
                        sibling_models.entry(dir.to_path_buf()).or_default().insert(name_key(&n));
                    }
                }
            }
        }
        v
    } else {
        walkdir::WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
            .collect()
    };

    let mut result = ScanResult::default();
    // folder -> projector paths
    let mut projectors: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    // (folder, base, count) -> shard number -> path
    let mut shard_sets: BTreeMap<(PathBuf, String, u32), BTreeMap<u32, PathBuf>> = BTreeMap::new();
    let mut singles: Vec<PathBuf> = Vec::new();

    for f in files {
        let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let folder = f.parent().map(Path::to_path_buf).unwrap_or_default();
        if is_projector_name(&name) {
            projectors.entry(folder).or_default().push(f);
            continue;
        }
        if let Some(c) = SHARD_RE.captures(&name) {
            let no: u32 = c["no"].parse().unwrap_or(0);
            let count: u32 = c["count"].parse().unwrap_or(0);
            shard_sets
                .entry((folder, c["base"].to_string(), count))
                .or_default()
                .insert(no, f);
        } else {
            singles.push(f);
        }
    }

    let mut push = |path: PathBuf, shards: Vec<PathBuf>, mut problems: Vec<String>, result: &mut ScanResult| {
        let meta = match gguf::read_metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                result.skipped.push(SkippedFile { path, reason: e.to_string() });
                return;
            }
        };
        if meta.is_projector {
            // Header says vision projector even though the name didn't.
            if let Some(folder) = path.parent() {
                projectors.entry(folder.to_path_buf()).or_default().push(path);
            }
            return;
        }
        let total_size = shards
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (quantization, from_name) = match &meta.quantization {
            Some(q) => (Some(q.clone()), false),
            None => (quant_from_filename(&name), true),
        };
        if meta.architecture.is_none() {
            problems.push("header has no general.architecture".into());
        }
        let stem = SHARD_RE
            .captures(&name)
            .map(|c| c["base"].to_string())
            .unwrap_or_else(|| name.trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string());
        result.models.push(DiscoveredModel {
            path,
            shard_paths: shards,
            total_size,
            metadata: meta,
            quantization,
            quantization_from_filename: from_name,
            display_name: stem,
            projector_path: None,
            problems,
        });
    };

    for s in singles {
        push(s.clone(), vec![s], vec![], &mut result);
    }
    for ((_, _, count), set) in shard_sets {
        let mut problems = Vec::new();
        let missing: Vec<u32> = (1..=count).filter(|n| !set.contains_key(n)).collect();
        if !missing.is_empty() {
            problems.push(format!(
                "incomplete shard set: missing part(s) {} of {count}",
                missing.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
            ));
        }
        let Some(first) = set.values().next().cloned() else { continue };
        let first_path = set.get(&1).cloned().unwrap_or(first);
        push(first_path, set.into_values().collect(), problems, &mut result);
    }
    // Pair projectors with models folder by folder (each header read once).
    let projectors: BTreeMap<PathBuf, Vec<ProjectorFile>> = projectors
        .into_iter()
        .map(|(folder, mut paths)| {
            paths.sort();
            paths.dedup();
            (folder, paths.into_iter().map(ProjectorFile::read).collect())
        })
        .collect();
    let model_key = |m: &DiscoveredModel| name_key(&m.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let mut folder_models = sibling_models;
    for m in &result.models {
        if let Some(folder) = m.path.parent() {
            folder_models.entry(folder.to_path_buf()).or_default().insert(model_key(m));
        }
    }
    for m in &mut result.models {
        let key = model_key(m);
        let picked = m.path.parent().and_then(|folder| {
            pick_projector(&key, m.metadata.embedding_length, folder_models.get(folder)?, projectors.get(folder)?)
        });
        m.projector_path = picked.map(|p| p.path.clone());
        crate::caps::apply_projector(&mut m.metadata.caps, picked.map(|p| p.kinds));
    }
    result.models.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::write_fake_gguf;

    #[test]
    fn groups_shards_detects_missing_parts_and_attaches_projector() {
        let d = tempfile::tempdir().unwrap();
        let repo = d.path().join("pub").join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        write_fake_gguf(&repo.join("big-Q4_K_M-00001-of-00003.gguf"), "llama", 4096, 15, Some((0, 3)));
        write_fake_gguf(&repo.join("big-Q4_K_M-00002-of-00003.gguf"), "llama", 4096, 15, Some((1, 3)));
        write_fake_gguf(&repo.join("mmproj-big-F16.gguf"), "clip", 0, 1, None);
        write_fake_gguf(&repo.join("small.Q8_0.gguf"), "qwen2", 32768, 7, None);
        std::fs::write(repo.join("broken.gguf"), b"nope").unwrap();

        let r = scan(d.path());
        assert_eq!(r.models.len(), 2, "{:#?}", r.models);
        let big = r.models.iter().find(|m| m.display_name == "big-Q4_K_M").unwrap();
        assert_eq!(big.shard_paths.len(), 2);
        assert!(big.problems[0].contains("missing part(s) 3 of 3"), "{:?}", big.problems);
        assert!(big.projector_path.as_ref().unwrap().ends_with("mmproj-big-F16.gguf"));
        assert_eq!(big.metadata.caps.vision, Some(true), "projector attached means vision");
        let small = r.models.iter().find(|m| m.display_name == "small.Q8_0").unwrap();
        assert_eq!(small.quantization.as_deref(), Some("Q8_0"));
        assert!(small.problems.is_empty());
        assert_eq!(r.skipped.len(), 1);
        assert!(r.skipped[0].reason.contains("not a GGUF"));

        // "mmproj-big" names the other model in this folder, so "small" gets none.
        let small_folder = r.models.iter().find(|m| m.display_name == "small.Q8_0").unwrap();
        assert!(small_folder.projector_path.is_none(), "{:?}", small_folder.projector_path);

        // Importing one file picks up its own projector and sibling shards only.
        let one = scan(&repo.join("small.Q8_0.gguf"));
        assert_eq!(one.models.len(), 1);
        assert!(one.models[0].projector_path.is_none(), "{:?}", one.models[0].projector_path);
        let shard = scan(&repo.join("big-Q4_K_M-00002-of-00003.gguf"));
        assert_eq!(shard.models.len(), 1, "{:#?}", shard.models);
        assert_eq!(shard.models[0].shard_paths.len(), 2);
        assert!(shard.models[0].projector_path.as_ref().unwrap().ends_with("mmproj-big-F16.gguf"));
    }

    fn projector_of(r: &ScanResult, display_name: &str) -> Option<String> {
        let m = r.models.iter().find(|m| m.display_name == display_name).unwrap_or_else(|| panic!("{display_name}: {:#?}", r.models));
        m.projector_path.as_ref().map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    }

    #[test]
    fn flat_folder_pairs_projectors_by_name_and_prefers_f16() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path();
        write_fake_gguf(&dir.join("gemma-4-E4B-it-Q4_K_M.gguf"), "gemma4", 4096, 15, None);
        write_fake_gguf(&dir.join("gemma-4-E4B-it-Q8_0.gguf"), "gemma4", 4096, 7, None);
        write_fake_gguf(&dir.join("Qwen3.8-27B-abliterated.Q4_K_M.gguf"), "qwen35", 4096, 15, None);
        write_fake_gguf(&dir.join("mmproj-gemma-4-E4B-it-Q8_0.gguf"), "clip", 0, 7, None);
        write_fake_gguf(&dir.join("mmproj-gemma-4-E4B-it-BF16.gguf"), "clip", 0, 32, None);
        write_fake_gguf(&dir.join("Qwen3.8-27B-abliterated.mmproj-f16.gguf"), "clip", 0, 1, None);
        let r = scan(dir);
        assert_eq!(r.models.len(), 3, "{:#?}", r.models);
        // Both gemma quants get the BF16 projector, not the Q8_0 one.
        assert_eq!(projector_of(&r, "gemma-4-E4B-it-Q4_K_M").as_deref(), Some("mmproj-gemma-4-E4B-it-BF16.gguf"));
        assert_eq!(projector_of(&r, "gemma-4-E4B-it-Q8_0").as_deref(), Some("mmproj-gemma-4-E4B-it-BF16.gguf"));
        assert_eq!(projector_of(&r, "Qwen3.8-27B-abliterated.Q4_K_M").as_deref(), Some("Qwen3.8-27B-abliterated.mmproj-f16.gguf"));
        assert!(r.models.iter().all(|m| m.metadata.caps.vision == Some(true)));
        // Single-file import uses the same matching.
        let one = scan(&dir.join("Qwen3.8-27B-abliterated.Q4_K_M.gguf"));
        assert_eq!(projector_of(&one, "Qwen3.8-27B-abliterated.Q4_K_M").as_deref(), Some("Qwen3.8-27B-abliterated.mmproj-f16.gguf"));
        let one = scan(&dir.join("gemma-4-E4B-it-Q8_0.gguf"));
        assert_eq!(projector_of(&one, "gemma-4-E4B-it-Q8_0").as_deref(), Some("mmproj-gemma-4-E4B-it-BF16.gguf"));
    }

    #[test]
    fn generic_projector_attaches_only_to_a_lone_model() {
        for generic in ["mmproj-F16.gguf", "mmproj-model-f16.gguf"] {
            let d = tempfile::tempdir().unwrap();
            write_fake_gguf(&d.path().join("Some-Model-Q4_K_M.gguf"), "llama", 4096, 15, None);
            write_fake_gguf(&d.path().join(generic), "clip", 0, 1, None);
            let r = scan(d.path());
            assert_eq!(projector_of(&r, "Some-Model-Q4_K_M").as_deref(), Some(generic));
            let one = scan(&d.path().join("Some-Model-Q4_K_M.gguf"));
            assert_eq!(projector_of(&one, "Some-Model-Q4_K_M").as_deref(), Some(generic));
        }
        // Two different models and one generic projector: no guessing.
        let d = tempfile::tempdir().unwrap();
        write_fake_gguf(&d.path().join("Alpha-7B-Q4_K_M.gguf"), "llama", 4096, 15, None);
        write_fake_gguf(&d.path().join("Beta-8B-Q4_K_M.gguf"), "llama", 4096, 15, None);
        write_fake_gguf(&d.path().join("mmproj-F16.gguf"), "clip", 0, 1, None);
        let r = scan(d.path());
        assert_eq!(r.models.len(), 2);
        assert!(r.models.iter().all(|m| m.projector_path.is_none() && m.metadata.caps.vision != Some(true)), "{:#?}", r.models);
        let one = scan(&d.path().join("Alpha-7B-Q4_K_M.gguf"));
        assert_eq!(projector_of(&one, "Alpha-7B-Q4_K_M"), None);
    }

    #[test]
    fn name_keys_and_metadata_veto() {
        let k = |n: &str| name_key(n).join(" ");
        assert_eq!(k("mmproj-gemma-4-E4B-it-BF16.gguf"), "gemma 4 e4b it");
        assert_eq!(k("gemma-4-E4B-it-Q4_K_M.gguf"), "gemma 4 e4b it");
        assert_eq!(k("Qwen3.8-27B-abliterated.mmproj-f16.gguf"), "qwen3 8 27b abliterated");
        assert_eq!(k("Qwen3.8-27B-abliterated.Q4_K_M.gguf"), "qwen3 8 27b abliterated");
        assert_eq!(k("Cyber-Ornith-1.5-9B-OBLITERATED.i1-Q4_K_S.gguf"), "cyber ornith 1 5 9b obliterated");
        assert_eq!(k("big-Q4_K_M-00001-of-00003.gguf"), "big");
        assert_eq!(k("Qwen3-4B-UD-Q4_K_XL.gguf"), "qwen3 4b");
        assert_eq!(k("mmproj-F16.gguf"), "");
        assert_eq!(k("mmproj-model-f16.gguf"), "model");

        let key = |n: &str| name_key(n);
        let proj = |n: &str, dim: Option<u64>| ProjectorFile { path: PathBuf::from(n), key: key(n), projection_dim: dim, kinds: None };
        let folder: BTreeSet<Vec<String>> = [key("A-4B-Q4_K_M.gguf"), key("A-4B-Instruct-Q4_K_M.gguf"), key("B-9B-Q4_K_M.gguf")].into();
        let ps = vec![proj("mmproj-A-4B-F16.gguf", Some(2560)), proj("mmproj-B-9B-F16.gguf", Some(4096))];
        let pick = |m: &str, embd| pick_projector(&key(m), embd, &folder, &ps).map(|p| p.path.to_string_lossy().into_owned());
        assert_eq!(pick("A-4B-Q4_K_M.gguf", Some(2560)).as_deref(), Some("mmproj-A-4B-F16.gguf"));
        // "a 4b" starts "a 4b instruct" but also equals "a 4b": ambiguous, so none.
        assert_eq!(pick("A-4B-Instruct-Q4_K_M.gguf", Some(2560)), None);
        // Header says the projector feeds a different width: not this model's.
        assert_eq!(pick("B-9B-Q4_K_M.gguf", Some(5120)), None);
        assert_eq!(pick("B-9B-Q4_K_M.gguf", Some(4096)).as_deref(), Some("mmproj-B-9B-F16.gguf"));
        // A unique prefix match is accepted.
        let folder2: BTreeSet<Vec<String>> = [key("Qwen3-4B-Instruct-Q4_K_XL.gguf"), key("Other-1B-Q8_0.gguf")].into();
        let ps2 = vec![proj("mmproj-Qwen3-4B-F16.gguf", None)];
        assert!(pick_projector(&key("Qwen3-4B-Instruct-Q4_K_XL.gguf"), None, &folder2, &ps2).is_some());
        assert!(pick_projector(&key("Other-1B-Q8_0.gguf"), None, &folder2, &ps2).is_none());
    }

    #[test]
    fn quant_names_from_filenames() {
        assert_eq!(quant_from_filename("Qwen3-8B-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_from_filename("model.i1-Q6_K.gguf").as_deref(), Some("Q6_K"));
        assert_eq!(quant_from_filename("x-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_from_filename("x-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(quant_from_filename("plain.gguf"), None);
        assert_eq!(quant_from_filename("Qwen3-4B-UD-Q4_K_XL.gguf").as_deref(), Some("Q4_K_XL"));
        assert_eq!(quant_from_filename("Qwen3-4B-UD-IQ2_XXS.gguf").as_deref(), Some("IQ2_XXS"));
        assert_eq!(quant_from_filename("Qwen3-4B-Q2_K_L.gguf").as_deref(), Some("Q2_K_L"));
    }
}

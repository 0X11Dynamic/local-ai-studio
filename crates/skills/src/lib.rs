//! Skills: reusable instruction packs for agents. A skill is a folder with a
//! `SKILL.md` (front matter + Markdown body) and optional extra files
//! (scripts), which stay off until the user enables each one.
//!
//! This crate is filesystem + pure logic; the app keeps revisions, script
//! approvals and per-skill modes in the database.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use app_core::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use sha2::Digest;

pub const SKILL_FILE: &str = "SKILL.md";
const MAX_SKILL_BYTES: usize = 256 * 1024;
const MAX_IMPORT_FILES: usize = 200;
const MAX_IMPORT_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillDoc {
    pub name: String,
    pub description: String,
    pub version: String,
    pub keywords: Vec<String>,
    /// Front-matter keys this app does not use (kept, not an error).
    pub other_keys: Vec<String>,
    pub body: String,
}

pub fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') && !n.starts_with('-')
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    if v.len() >= 2 && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\''))) {
        v[1..v.len() - 1].to_string()
    } else {
        v.to_string()
    }
}

/// Parse `SKILL.md`. The front matter is `---` lines around `key: value`
/// pairs; values are plain text, quoted text, or `[a, b]` lists. Anything
/// else (nested YAML, multi-line values) is an error, not a guess.
pub fn parse(text: &str) -> AppResult<SkillDoc> {
    if text.len() > MAX_SKILL_BYTES {
        return Err(AppError::InvalidInput("SKILL.md is larger than 256 KB".into()));
    }
    let text = text.strip_prefix('\u{feff}').unwrap_or(text).replace("\r\n", "\n");
    let rest = text.strip_prefix("---\n").ok_or_else(|| AppError::InvalidInput("SKILL.md must start with a --- front matter line".into()))?;
    let end = rest.find("\n---").ok_or_else(|| AppError::InvalidInput("the front matter has no closing --- line".into()))?;
    let (fm, after) = rest.split_at(end);
    let body = after.trim_start_matches("\n---").trim_start_matches('\n').to_string();
    let (mut name, mut description, mut version, mut keywords, mut other) = (None, None, None, vec![], vec![]);
    for (i, line) in fm.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if line.starts_with(' ') || line.starts_with('\t') || line.trim_start().starts_with('-') {
            return Err(AppError::InvalidInput(format!("front matter line {}: nested or multi-line values are not supported", i + 2)));
        }
        let (k, v) = line.split_once(':').ok_or_else(|| AppError::InvalidInput(format!("front matter line {}: expected key: value", i + 2)))?;
        let (k, v) = (k.trim(), v.trim());
        match k {
            "name" => name = Some(unquote(v)),
            "description" => description = Some(unquote(v)),
            "version" => version = Some(unquote(v)),
            "keywords" => {
                let inner = v.strip_prefix('[').and_then(|x| x.strip_suffix(']')).ok_or_else(|| AppError::InvalidInput("keywords must be a list like [rust, tests]".into()))?;
                keywords = inner.split(',').map(|s| unquote(s).to_lowercase()).filter(|s| !s.is_empty()).collect();
            }
            other_key => other.push(other_key.to_string()),
        }
    }
    let name = name.ok_or_else(|| AppError::InvalidInput("front matter needs a name".into()))?;
    if !valid_name(&name) {
        return Err(AppError::InvalidInput(format!("skill name \"{name}\" must be lowercase letters, digits and dashes")));
    }
    let description = description.filter(|d| !d.trim().is_empty()).ok_or_else(|| AppError::InvalidInput("front matter needs a description".into()))?;
    if description.chars().count() > 1024 {
        return Err(AppError::InvalidInput("description is longer than 1024 characters".into()));
    }
    if body.trim().is_empty() {
        return Err(AppError::InvalidInput("the skill has no instructions after the front matter".into()));
    }
    Ok(SkillDoc { name, description, version: version.unwrap_or_else(|| "1".into()), keywords, other_keys: other, body })
}

/// Changes an agent may propose. Unset fields stay as they are.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Changes {
    pub description: Option<String>,
    pub version: Option<String>,
    pub add_keywords: Vec<String>,
    pub remove_keywords: Vec<String>,
    /// The complete new instructions (Markdown body).
    pub instructions: Option<String>,
}

fn quote(v: &str) -> String {
    format!("\"{}\"", v.replace('"', "'").replace('\n', " "))
}

/// Apply `changes` to an existing SKILL.md, editing only the affected
/// front-matter lines (other keys keep their exact text) and the body.
/// The name never changes. The result is validated with `parse`.
pub fn apply_changes(text: &str, changes: &Changes) -> AppResult<String> {
    let doc = parse(text)?;
    let text = text.replace("\r\n", "\n");
    let rest = text.trim_start_matches('\u{feff}').strip_prefix("---\n").unwrap_or(&text);
    let end = rest.find("\n---").unwrap_or(rest.len());
    let mut lines: Vec<String> = rest[..end].lines().map(str::to_string).collect();
    let mut set = |key: &str, value: String| {
        let line = format!("{key}: {value}");
        match lines.iter().position(|l| l.split_once(':').is_some_and(|(k, _)| k.trim() == key)) {
            Some(i) => lines[i] = line,
            None => lines.push(line),
        }
    };
    if let Some(d) = &changes.description {
        set("description", quote(d.trim()));
    }
    if let Some(v) = &changes.version {
        set("version", v.trim().to_string());
    }
    if !changes.add_keywords.is_empty() || !changes.remove_keywords.is_empty() {
        let mut kw = doc.keywords.clone();
        for k in &changes.add_keywords {
            let k = k.trim().to_lowercase();
            if !k.is_empty() && !kw.contains(&k) {
                kw.push(k);
            }
        }
        kw.retain(|k| !changes.remove_keywords.iter().any(|r| r.trim().eq_ignore_ascii_case(k)));
        set("keywords", format!("[{}]", kw.join(", ")));
    }
    let body = changes.instructions.as_deref().map(str::trim).filter(|b| !b.is_empty()).unwrap_or(doc.body.trim());
    let out = format!("---\n{}\n---\n{}\n", lines.join("\n"), body);
    parse(&out)?;
    Ok(out)
}

// ---------- matching ----------

const STOP: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "are", "was", "were", "will", "can", "use", "using", "make", "made", "then", "than",
    "when", "what", "which", "who", "how", "all", "any", "each", "some", "not", "but", "has", "have", "had", "its", "our", "out", "about", "over", "under", "also",
    "should", "would", "could", "please", "need", "want", "skill", "task", "file", "files",
];

fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    for suf in ["ing", "ed", "es", "s"] {
        if w.len() > suf.len() + 3
            && let Some(s) = w.strip_suffix(suf)
        {
            return s.to_string();
        }
    }
    w
}

pub fn words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOP.contains(&w.to_lowercase().as_str()))
        .map(stem)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Match {
    pub name: String,
    pub score: u32,
    /// Why it was picked, e.g. "matched: rust, test" or "always on".
    pub reason: String,
}

/// A skill as the matcher sees it.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub name: String,
    pub description: String,
    pub keywords: Vec<String>,
    /// "auto" | "always" | "off"
    pub mode: String,
}

/// Pick up to `max` skills for an objective, explaining each pick.
/// Keywords weigh 3, name words 2, description words 1. A skill is picked
/// when a keyword or name word matched and the score is at least 2, or on
/// description words alone when at least 3 different ones matched (so a
/// couple of generic words like "write" and "mood" are not enough).
pub fn pick(objective: &str, skills: &[Candidate], max: usize) -> Vec<Match> {
    let want = words(objective);
    let mut out: Vec<Match> = skills.iter().filter(|s| s.mode == "always").map(|s| Match { name: s.name.clone(), score: u32::MAX, reason: "always on".into() }).collect();
    let mut scored = vec![];
    for s in skills.iter().filter(|s| s.mode == "auto") {
        let kw: BTreeSet<String> = s.keywords.iter().flat_map(|k| words(k)).collect();
        let nm = words(&s.name.replace('-', " "));
        let ds = words(&s.description);
        let mut score = 0;
        let mut strong = false;
        let mut hits = BTreeSet::new();
        for w in &want {
            let pts = if kw.contains(w) { 3 } else if nm.contains(w) { 2 } else if ds.contains(w) { 1 } else { 0 };
            if pts > 0 {
                strong |= pts >= 2;
                score += pts;
                hits.insert(w.clone());
            }
        }
        if (strong && score >= 2) || score >= 3 {
            scored.push(Match { name: s.name.clone(), score, reason: format!("matched: {}", hits.into_iter().collect::<Vec<_>>().join(", ")) });
        }
    }
    scored.sort_by(|a, b| b.score.cmp(&a.score).then(a.name.cmp(&b.name)));
    out.extend(scored);
    out.truncate(max);
    out
}

// ---------- files ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtraFile {
    /// Path relative to the skill folder, with `/` separators.
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Every file in a skill folder except SKILL.md (scripts, references).
pub fn extra_files(skill_dir: &Path) -> AppResult<Vec<ExtraFile>> {
    let mut out = vec![];
    if !skill_dir.is_dir() {
        return Ok(out);
    }
    for e in walkdir::WalkDir::new(skill_dir).follow_links(false).min_depth(1) {
        let e = e.map_err(|e| AppError::Runtime(e.to_string()))?;
        if !e.file_type().is_file() {
            continue;
        }
        let rel = e.path().strip_prefix(skill_dir).map_err(|e| AppError::Runtime(e.to_string()))?.to_string_lossy().replace('\\', "/");
        if rel == SKILL_FILE {
            continue;
        }
        let bytes = std::fs::read(e.path())?;
        out.push(ExtraFile { path: rel, sha256: sha256_hex(&bytes), size: bytes.len() as u64 });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// A safe relative path inside an archive or folder: plain names only.
fn safe_rel(p: &str) -> AppResult<PathBuf> {
    let p = p.replace('\\', "/");
    let mut out = PathBuf::new();
    for c in Path::new(&p).components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::CurDir => {}
            _ => return Err(AppError::PermissionDenied(format!("\"{p}\" points outside the skill folder"))),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(AppError::InvalidInput("empty path in the archive".into()));
    }
    Ok(out)
}

/// Read every file of a skill package from a `.zip` (one top folder or
/// files at the root). Refuses `..`, absolute paths, links and size bombs.
pub fn read_zip(bytes: &[u8]) -> AppResult<Vec<(PathBuf, Vec<u8>)>> {
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| AppError::InvalidInput(format!("not a zip file: {e}")))?;
    if z.len() > MAX_IMPORT_FILES {
        return Err(AppError::InvalidInput(format!("the archive has more than {MAX_IMPORT_FILES} files")));
    }
    let mut files = vec![];
    let mut total = 0u64;
    for i in 0..z.len() {
        let mut f = z.by_index(i).map_err(|e| AppError::InvalidInput(e.to_string()))?;
        if f.is_dir() {
            continue;
        }
        let name = f.name().map_err(|e| AppError::InvalidInput(format!("bad file name in the archive: {e}")))?.to_string();
        if f.is_symlink() {
            return Err(AppError::PermissionDenied(format!("{name} is a link; links are not imported")));
        }
        let rel = safe_rel(&name)?;
        total += f.size();
        if total > MAX_IMPORT_BYTES {
            return Err(AppError::InvalidInput("the archive unpacks to more than 20 MB".into()));
        }
        let mut buf = Vec::with_capacity(f.size().min(MAX_IMPORT_BYTES) as usize);
        (&mut f).take(MAX_IMPORT_BYTES + 1).read_to_end(&mut buf)?;
        if buf.len() as u64 > f.size() {
            return Err(AppError::InvalidInput("an archive entry is larger than it claims".into()));
        }
        files.push((rel, buf));
    }
    // Strip a single top-level folder ("my-skill/SKILL.md").
    let top: BTreeSet<_> = files.iter().filter_map(|(p, _)| p.components().next().map(|c| c.as_os_str().to_owned())).collect();
    if top.len() == 1 && !files.iter().any(|(p, _)| p.as_os_str() == SKILL_FILE) {
        files = files.into_iter().filter_map(|(p, b)| p.strip_prefix(p.components().next().unwrap().as_os_str()).ok().map(|r| (r.to_path_buf(), b))).filter(|(p, _)| !p.as_os_str().is_empty()).collect();
    }
    if !files.iter().any(|(p, _)| p.as_os_str() == SKILL_FILE) {
        return Err(AppError::InvalidInput("the archive has no SKILL.md".into()));
    }
    Ok(files)
}

/// Read a skill package from a folder on disk (same rules as a zip).
pub fn read_folder(dir: &Path) -> AppResult<Vec<(PathBuf, Vec<u8>)>> {
    let mut files = vec![];
    let mut total = 0u64;
    for e in walkdir::WalkDir::new(dir).follow_links(false).min_depth(1) {
        let e = e.map_err(|e| AppError::Runtime(e.to_string()))?;
        if e.file_type().is_symlink() {
            continue;
        }
        if !e.file_type().is_file() {
            continue;
        }
        let rel = e.path().strip_prefix(dir).map_err(|e| AppError::Runtime(e.to_string()))?.to_path_buf();
        let bytes = std::fs::read(e.path())?;
        total += bytes.len() as u64;
        if files.len() >= MAX_IMPORT_FILES || total > MAX_IMPORT_BYTES {
            return Err(AppError::InvalidInput("the skill folder is too large (200 files / 20 MB)".into()));
        }
        files.push((rel, bytes));
    }
    if !files.iter().any(|(p, _)| p.as_os_str() == SKILL_FILE) {
        return Err(AppError::InvalidInput("the folder has no SKILL.md".into()));
    }
    Ok(files)
}

/// Write a package into `skills_dir/<name>/`, which must not exist yet.
pub fn install(skills_dir: &Path, files: &[(PathBuf, Vec<u8>)]) -> AppResult<SkillDoc> {
    let md = files.iter().find(|(p, _)| p.as_os_str() == SKILL_FILE).ok_or_else(|| AppError::InvalidInput("no SKILL.md".into()))?;
    let doc = parse(&String::from_utf8_lossy(&md.1))?;
    let dest = skills_dir.join(&doc.name);
    if dest.exists() {
        return Err(AppError::Conflict(format!("a skill named {} already exists", doc.name)));
    }
    for (rel, bytes) in files {
        let rel = safe_rel(&rel.to_string_lossy())?;
        let target = dest.join(&rel);
        if let Some(p) = target.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&target, bytes)?;
    }
    Ok(doc)
}

/// Zip a skill folder for export.
pub fn export_zip(skill_dir: &Path, name: &str) -> AppResult<Vec<u8>> {
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for e in walkdir::WalkDir::new(skill_dir).follow_links(false).min_depth(1) {
        let e = e.map_err(|e| AppError::Runtime(e.to_string()))?;
        if !e.file_type().is_file() {
            continue;
        }
        let rel = e.path().strip_prefix(skill_dir).map_err(|e| AppError::Runtime(e.to_string()))?.to_string_lossy().replace('\\', "/");
        w.start_file(format!("{name}/{rel}"), opts).map_err(|e| AppError::Runtime(e.to_string()))?;
        w.write_all(&std::fs::read(e.path())?)?;
    }
    Ok(w.finish().map_err(|e| AppError::Runtime(e.to_string()))?.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "---\nname: rust-tests\ndescription: \"Fix failing Rust tests with cargo\"\nversion: 2\nkeywords: [rust, cargo, tests]\nallowed-tools: x\n---\n# Steps\nRun cargo test first.\n";

    #[test]
    fn front_matter_is_parsed_strictly() {
        let d = parse(DOC).unwrap();
        assert_eq!((d.name.as_str(), d.version.as_str()), ("rust-tests", "2"));
        assert_eq!(d.keywords, vec!["rust", "cargo", "tests"]);
        assert_eq!(d.other_keys, vec!["allowed-tools"]);
        assert!(d.body.starts_with("# Steps"));
        assert!(parse(&DOC.replace("\n", "\r\n")).is_ok(), "CRLF files");
        for bad in [
            "name: x\n---\nbody",                                        // no opening ---
            "---\nname: x\ndescription: d\nbody",                        // no closing ---
            "---\nname: Bad Name\ndescription: d\n---\nbody",            // bad name
            "---\nname: x\n---\nbody",                                   // no description
            "---\nname: x\ndescription: d\n  nested: 1\n---\nbody",      // nested
            "---\nname: x\ndescription: d\nkeywords: rust\n---\nbody",   // keywords not a list
            "---\nname: x\ndescription: d\n---\n   \n",                  // empty body
        ] {
            assert!(parse(bad).is_err(), "should fail: {bad:?}");
        }
    }

    #[test]
    fn changes_edit_only_what_they_name() {
        let add = Changes { add_keywords: vec!["Verse".into(), "rust".into()], version: Some("3".into()), ..Default::default() };
        let out = apply_changes(DOC, &add).unwrap();
        let d = parse(&out).unwrap();
        assert_eq!((d.version.as_str(), d.keywords.clone()), ("3", vec!["rust".to_string(), "cargo".into(), "tests".into(), "verse".into()]));
        assert_eq!((d.name.as_str(), d.description.as_str()), ("rust-tests", "Fix failing Rust tests with cargo"), "untouched");
        assert!(out.contains("allowed-tools: x") && d.body.starts_with("# Steps"), "other keys and body kept: {out}");
        let rm = apply_changes(&out, &Changes { remove_keywords: vec!["CARGO".into()], instructions: Some("New steps.".into()), description: Some("Says \"hi\"".into()), ..Default::default() }).unwrap();
        let d = parse(&rm).unwrap();
        assert!(!d.keywords.contains(&"cargo".to_string()) && d.body.trim() == "New steps." && d.description == "Says 'hi'", "{rm}");
        assert!(apply_changes("not a skill", &Changes::default()).is_err());
    }

    #[test]
    fn matching_explains_itself() {
        let c = |n: &str, d: &str, k: &[&str], m: &str| Candidate { name: n.into(), description: d.into(), keywords: k.iter().map(|s| s.to_string()).collect(), mode: m.into() };
        let skills = vec![
            c("rust-tests", "Fix failing Rust tests with cargo", &["rust", "cargo"], "auto"),
            c("commit-style", "Write commit messages in the imperative mood", &["git", "commit"], "auto"),
            c("house-rules", "Always applies", &[], "always"),
            c("disabled-one", "Rust everything", &["rust"], "off"),
        ];
        let m = pick("Make cargo test pass in my Rust project", &skills, 3);
        let names: Vec<_> = m.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, vec!["house-rules", "rust-tests"], "{m:?}");
        assert!(m[1].reason.contains("cargo") && m[1].reason.contains("rust"), "{}", m[1].reason);
        assert_eq!(m[0].reason, "always on");
        // Two generic description words ("write", "mood") are not enough.
        assert!(pick("write a poem about the mood of autumn", &skills[..2], 3).is_empty());
        assert_eq!(pick("git commit", &skills, 1).len(), 1, "max respected");
    }

    #[test]
    fn zips_refuse_escapes_and_round_trip() {
        let d = tempfile::tempdir().unwrap();
        // Build a malicious zip.
        let mk = |entries: &[(&str, &[u8])]| {
            let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            for (n, b) in entries {
                w.start_file(*n, zip::write::SimpleFileOptions::default()).unwrap();
                w.write_all(b).unwrap();
            }
            w.finish().unwrap().into_inner()
        };
        assert!(read_zip(&mk(&[("SKILL.md", DOC.as_bytes()), ("../evil.ps1", b"x")])).is_err(), "..");
        assert!(read_zip(&mk(&[("SKILL.md", DOC.as_bytes()), ("/abs.ps1", b"x")])).is_err(), "absolute");
        assert!(read_zip(&mk(&[("readme.txt", b"x")])).is_err(), "no SKILL.md");
        assert!(read_zip(b"not a zip").is_err());
        let ok = mk(&[("rust-tests/SKILL.md", DOC.as_bytes()), ("rust-tests/scripts/run.ps1", b"cargo test")]);
        let files = read_zip(&ok).unwrap();
        let doc = install(d.path(), &files).unwrap();
        assert_eq!(doc.name, "rust-tests");
        assert!(install(d.path(), &files).is_err(), "already exists");
        let extra = extra_files(&d.path().join("rust-tests")).unwrap();
        assert_eq!(extra.len(), 1);
        assert_eq!((extra[0].path.as_str(), extra[0].sha256.len()), ("scripts/run.ps1", 64));
        let z = export_zip(&d.path().join("rust-tests"), "rust-tests").unwrap();
        assert_eq!(read_zip(&z).unwrap().len(), 2, "export round trip");
    }
}

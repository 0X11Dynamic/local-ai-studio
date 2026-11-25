//! Skills in the app: the folder `%APPDATA%\LocalAIStudio\skills\<name>\`
//! is the source of truth for content; the database keeps the mode
//! (auto/always/off), enabled scripts and every revision. Agents can only
//! *propose* edits; the user approves or rejects them.

use std::collections::BTreeMap;
use std::path::PathBuf;

use agent_runtime::{CallResult, ExposedTool};
use app_core::{AppError, AppResult};
use persistence::{SkillRevisionRow, SkillRow};
use security::policy::Policy;
use serde::Serialize;
use skills::{Candidate, Match, SKILL_FILE, SkillDoc};

use crate::state::SharedState;

/// Skill text injected into a prompt is capped (about 4k tokens).
const MAX_SECTION_CHARS: usize = 16_000;

#[derive(Debug, Clone, Serialize)]
pub struct FileView {
    pub path: String,
    pub sha256: String,
    pub size: u64,
    /// Enabled by the user, and unchanged since.
    pub enabled: bool,
    /// Enabled once, but the file changed since (so it is off again).
    pub changed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkillView {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub keywords: Vec<String>,
    pub mode: String,
    pub source: String,
    pub content: String,
    pub path: String,
    pub files: Vec<FileView>,
    pub pending: usize,
    pub error: Option<String>,
}

// ---------- caches (the context meter matches skills on every typing pause) ----------

type Stamp = (Option<std::time::SystemTime>, u64);

fn stamp(p: &std::path::Path) -> Option<Stamp> {
    std::fs::metadata(p).ok().map(|m| (m.modified().ok(), m.len()))
}

#[derive(Clone)]
struct CachedDoc {
    stamp: Stamp,
    text: String,
    doc: Result<SkillDoc, String>,
}

fn doc_cache() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, CachedDoc>> {
    static C: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PathBuf, CachedDoc>>> = std::sync::OnceLock::new();
    C.get_or_init(Default::default)
}

/// A SKILL.md's text and parse result, re-read only when the file changed
/// (modification time and size).
fn read_doc(md: &std::path::Path) -> Option<(String, Result<SkillDoc, String>)> {
    let st = stamp(md)?;
    if let Some(c) = doc_cache().lock().unwrap().get(md)
        && c.stamp == st
    {
        return Some((c.text.clone(), c.doc.clone()));
    }
    let text = std::fs::read_to_string(md).ok()?;
    let doc = skills::parse(&text).map_err(|e| e.to_string());
    doc_cache().lock().unwrap().insert(md.to_path_buf(), CachedDoc { stamp: st, text: text.clone(), doc: doc.clone() });
    Some((text, doc))
}

/// (relative path, stamp) of every extra file, without reading contents.
fn fingerprint(dir: &std::path::Path) -> Vec<(PathBuf, Stamp)> {
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue;
            }
            let path = e.path();
            if ft.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n != SKILL_FILE) || path.parent() != Some(dir) {
                if let Some(s) = stamp(&path) {
                    out.push((path, s));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn files_cache() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, (Vec<(PathBuf, Stamp)>, Vec<skills::ExtraFile>)>> {
    static C: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PathBuf, (Vec<(PathBuf, Stamp)>, Vec<skills::ExtraFile>)>>> = std::sync::OnceLock::new();
    C.get_or_init(Default::default)
}

/// Extra files with their sha256, re-hashed only when a file was added,
/// removed or changed.
fn extra_files(dir: &std::path::Path) -> AppResult<Vec<skills::ExtraFile>> {
    let fp = fingerprint(dir);
    if let Some((old, files)) = files_cache().lock().unwrap().get(dir)
        && *old == fp
    {
        return Ok(files.clone());
    }
    let files = skills::extra_files(dir)?;
    files_cache().lock().unwrap().insert(dir.to_path_buf(), (fp, files.clone()));
    Ok(files)
}

fn dir(st: &SharedState) -> PathBuf {
    st.paths.skills_dir.clone()
}

fn manifest(doc: &SkillDoc) -> serde_json::Value {
    serde_json::json!({ "description": doc.description, "keywords": doc.keywords, "other_keys": doc.other_keys })
}

fn write_atomic(path: &std::path::Path, text: &str) -> AppResult<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("md.las-tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Sample skills shipped with the app (embedded, so the installer needs no
/// extra files). Installed once each; a sample the user deletes stays deleted.
pub const SAMPLES: [(&str, &str); 3] = [
    ("rust-test-fix", include_str!("../../resources/skills/rust-test-fix/SKILL.md")),
    ("git-commit-style", include_str!("../../resources/skills/git-commit-style/SKILL.md")),
    ("safe-powershell", include_str!("../../resources/skills/safe-powershell/SKILL.md")),
];
const SETTING_SAMPLES: &str = "skills.samples_installed";

/// Write the samples that are not yet in `skills_dir` and were never
/// installed before (`already`). Never overwrites an existing folder.
/// Returns (folder, name, text) for each sample written.
pub fn write_samples(skills_dir: &std::path::Path, already: &[String]) -> AppResult<Vec<(PathBuf, String, String)>> {
    let mut out = vec![];
    for (name, text) in SAMPLES {
        if already.iter().any(|a| a == name) {
            continue;
        }
        let folder = skills_dir.join(name);
        if folder.exists() {
            continue;
        }
        let doc = skills::parse(text)?;
        write_atomic(&folder.join(SKILL_FILE), text)?;
        out.push((folder, doc.name, text.to_string()));
    }
    Ok(out)
}

/// First run (and after an upgrade that adds samples): install missing samples.
async fn install_samples(st: &SharedState) -> AppResult<()> {
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if CHECKED.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(());
    }
    install_samples_now(st).await?;
    CHECKED.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

async fn install_samples_now(st: &SharedState) -> AppResult<()> {
    let mut done: Vec<String> = st.db.get_setting(SETTING_SAMPLES).await?.unwrap_or_default();
    if SAMPLES.iter().all(|(n, _)| done.iter().any(|d| d == n)) {
        return Ok(());
    }
    let rows = st.db.list_skills().await?;
    for (folder, name, text) in write_samples(&dir(st), &done)? {
        if !rows.iter().any(|r| r.name == name) {
            let doc = skills::parse(&text)?;
            let id = st.db.insert_skill(&folder.to_string_lossy(), &doc.name, &doc.version, &manifest(&doc), "builtin").await?;
            st.db.add_skill_revision(&id, &doc.version, &skills::sha256_hex(text.as_bytes()), &text, Some("sample skill shipped with the app"), "applied", None).await?;
        }
    }
    // Mark every sample as handled, including ones skipped because a skill
    // with that name already existed.
    for (n, _) in SAMPLES {
        if !done.iter().any(|d| d == n) {
            done.push(n.to_string());
        }
    }
    st.db.set_setting(SETTING_SAMPLES, &done).await?;
    Ok(())
}

/// Register skill folders that have no database row yet (and refresh metadata).
pub async fn sync(st: &SharedState) -> AppResult<()> {
    let d = dir(st);
    std::fs::create_dir_all(&d)?;
    install_samples(st).await?;
    let rows = st.db.list_skills().await?;
    for e in std::fs::read_dir(&d)?.flatten() {
        let p = e.path();
        let md = p.join(SKILL_FILE);
        if !md.is_file() {
            continue;
        }
        let Some((text, Ok(doc))) = read_doc(&md) else { continue };
        match rows.iter().find(|r| r.name == doc.name) {
            Some(r) => {
                if r.version != doc.version || serde_json::from_str::<serde_json::Value>(&r.manifest).ok() != Some(manifest(&doc)) {
                    st.db.update_skill_meta(&r.id, &doc.version, &manifest(&doc)).await?;
                }
            }
            None => {
                let id = st.db.insert_skill(&p.to_string_lossy(), &doc.name, &doc.version, &manifest(&doc), "user").await?;
                st.db.add_skill_revision(&id, &doc.version, &skills::sha256_hex(text.as_bytes()), &text, Some("found in the skills folder"), "applied", None).await?;
            }
        }
    }
    Ok(())
}

fn enabled_scripts(row: &SkillRow) -> BTreeMap<String, String> {
    serde_json::from_str(&row.scripts).unwrap_or_default()
}

async fn view(st: &SharedState, row: &SkillRow) -> AppResult<SkillView> {
    let folder = PathBuf::from(&row.path);
    let (content, parsed) = read_doc(&folder.join(SKILL_FILE)).unwrap_or_else(|| (String::new(), Err("SKILL.md is missing".into())));
    let on = enabled_scripts(row);
    let files = extra_files(&folder)?
        .into_iter()
        .map(|f| {
            let want = on.get(&f.path);
            FileView { enabled: want == Some(&f.sha256), changed: want.is_some_and(|h| h != &f.sha256), path: f.path, sha256: f.sha256, size: f.size }
        })
        .collect();
    let pending = st.db.list_skill_revisions(&row.id).await?.iter().filter(|r| r.state == "pending").count();
    let m: serde_json::Value = serde_json::from_str(&row.manifest).unwrap_or_default();
    Ok(SkillView {
        id: row.id.clone(),
        name: row.name.clone(),
        description: m["description"].as_str().unwrap_or_default().to_string(),
        version: row.version.clone(),
        keywords: serde_json::from_value(m["keywords"].clone()).unwrap_or_default(),
        mode: row.mode.clone(),
        source: row.source.clone(),
        content,
        path: row.path.clone(),
        files,
        pending,
        error: parsed.err(),
    })
}

pub async fn list(st: &SharedState) -> AppResult<Vec<SkillView>> {
    sync(st).await?;
    let rows = st.db.list_skills().await?;
    let mut out = vec![];
    for r in &rows {
        out.push(view(st, r).await?);
    }
    // Folders whose SKILL.md does not parse were never registered: show them
    // with the error instead of hiding them.
    for e in std::fs::read_dir(dir(st))?.flatten() {
        let folder = e.path();
        let md = folder.join(SKILL_FILE);
        if !md.is_file() || rows.iter().any(|r| PathBuf::from(&r.path) == folder) {
            continue;
        }
        if let Some((text, Err(err))) = read_doc(&md) {
            let name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            out.push(SkillView {
                id: format!("{BROKEN_PREFIX}{name}"),
                name,
                description: String::new(),
                version: String::new(),
                keywords: vec![],
                mode: "off".into(),
                source: "user".into(),
                content: text,
                path: folder.to_string_lossy().into_owned(),
                files: vec![],
                pending: 0,
                error: Some(err),
            });
        }
    }
    Ok(out)
}

/// Id prefix for skill folders whose SKILL.md does not parse (no database row).
pub const BROKEN_PREFIX: &str = "broken:";

/// Create a skill (`id: None`) or save a new version of one. The content's
/// name must match an existing skill's name.
pub async fn save(st: &SharedState, id: Option<&str>, content: &str, note: Option<&str>) -> AppResult<SkillView> {
    let doc = skills::parse(content)?;
    let row = match id {
        Some(b) if b.starts_with(BROKEN_PREFIX) => {
            // Repair a folder whose SKILL.md did not parse.
            let folder_name = &b[BROKEN_PREFIX.len()..];
            if doc.name != folder_name {
                return Err(AppError::InvalidInput(format!("the name in the front matter must be \"{folder_name}\" (the folder name)")));
            }
            let folder = dir(st).join(folder_name);
            write_atomic(&folder.join(SKILL_FILE), content)?;
            let id = st.db.insert_skill(&folder.to_string_lossy(), &doc.name, &doc.version, &manifest(&doc), "user").await?;
            st.db.get_skill(&id).await?
        }
        Some(id) => {
            let row = st.db.get_skill(id).await?;
            if row.name != doc.name {
                return Err(AppError::InvalidInput(format!("the name in the front matter must stay \"{}\"", row.name)));
            }
            write_atomic(&PathBuf::from(&row.path).join(SKILL_FILE), content)?;
            st.db.update_skill_meta(&row.id, &doc.version, &manifest(&doc)).await?;
            row
        }
        None => {
            let folder = dir(st).join(&doc.name);
            if folder.exists() {
                return Err(AppError::Conflict(format!("a skill named {} already exists", doc.name)));
            }
            write_atomic(&folder.join(SKILL_FILE), content)?;
            let id = st.db.insert_skill(&folder.to_string_lossy(), &doc.name, &doc.version, &manifest(&doc), "user").await?;
            st.db.get_skill(&id).await?
        }
    };
    st.db.add_skill_revision(&row.id, &doc.version, &skills::sha256_hex(content.as_bytes()), content, note, "applied", None).await?;
    view(st, &st.db.get_skill(&row.id).await?).await
}

pub async fn set_mode(st: &SharedState, id: &str, mode: &str) -> AppResult<()> {
    st.db.set_skill_mode(id, mode).await
}

/// Enable a script by recording its current hash (or disable it).
pub async fn set_script(st: &SharedState, id: &str, path: &str, enabled: bool) -> AppResult<()> {
    let row = st.db.get_skill(id).await?;
    let mut on = enabled_scripts(&row);
    if enabled {
        let f = extra_files(&PathBuf::from(&row.path))?.into_iter().find(|f| f.path == path).ok_or_else(|| AppError::NotFound(format!("{path} is not in this skill")))?;
        on.insert(f.path, f.sha256);
    } else {
        on.remove(path);
    }
    st.db.set_skill_scripts(id, &serde_json::to_value(on)?).await
}

pub async fn revisions(st: &SharedState, id: &str) -> AppResult<Vec<SkillRevisionRow>> {
    st.db.list_skill_revisions(id).await
}

/// Restore an old revision as a new applied one (history is never lost).
pub async fn revert(st: &SharedState, rev_id: &str) -> AppResult<SkillView> {
    let rev = st.db.get_skill_revision(rev_id).await?;
    if rev.state == "pending" {
        return Err(AppError::InvalidInput("approve or reject a proposal instead of reverting to it".into()));
    }
    save(st, Some(&rev.skill_id), &rev.content, Some(&format!("reverted to the version from {}", rev.created_at))).await
}

pub async fn approve(st: &SharedState, rev_id: &str) -> AppResult<SkillView> {
    let rev = st.db.get_skill_revision(rev_id).await?;
    if rev.state != "pending" {
        return Err(AppError::InvalidInput("that proposal is not waiting".into()));
    }
    let v = save(st, Some(&rev.skill_id), &rev.content, Some(&format!("approved proposal {}", rev.id))).await?;
    st.db.set_skill_revision_state(rev_id, "applied").await?;
    Ok(v)
}

pub async fn reject(st: &SharedState, rev_id: &str) -> AppResult<()> {
    let rev = st.db.get_skill_revision(rev_id).await?;
    if rev.state != "pending" {
        return Err(AppError::InvalidInput("that proposal is not waiting".into()));
    }
    st.db.set_skill_revision_state(rev_id, "rejected").await
}

/// Move a skill folder to the app trash and forget it.
pub async fn delete(st: &SharedState, id: &str) -> AppResult<()> {
    if let Some(name) = id.strip_prefix(BROKEN_PREFIX) {
        let src = dir(st).join(name);
        if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." || !src.is_dir() {
            return Err(AppError::NotFound(format!("no skill folder {name}")));
        }
        let dest = crate::services::exec::trash_dir(st).join(format!("skill-{name}-{}", app_core::now_rfc3339().replace([':', '.'], "-")));
        std::fs::create_dir_all(dest.parent().unwrap_or(&dest))?;
        std::fs::rename(&src, &dest)?;
        return Ok(());
    }
    let row = st.db.get_skill(id).await?;
    let src = PathBuf::from(&row.path);
    if src.exists() {
        let dest = crate::services::exec::trash_dir(st).join(format!("skill-{}-{}", row.name, app_core::now_rfc3339().replace([':', '.'], "-")));
        std::fs::create_dir_all(dest.parent().unwrap_or(&dest))?;
        std::fs::rename(&src, &dest)?;
    }
    st.db.delete_skill(&row.id).await
}

/// Import a skill package from a `.zip` file or a folder. Extra files
/// (scripts) start turned off.
pub async fn import(st: &SharedState, path: &str) -> AppResult<SkillView> {
    let p = PathBuf::from(path);
    let files = if p.is_dir() {
        skills::read_folder(&p)?
    } else {
        let bytes = std::fs::read(&p).map_err(|e| AppError::NotFound(format!("cannot read {path}: {e}")))?;
        skills::read_zip(&bytes)?
    };
    let d = dir(st);
    std::fs::create_dir_all(&d)?;
    let doc = skills::install(&d, &files)?;
    let folder = d.join(&doc.name);
    let text = std::fs::read_to_string(folder.join(SKILL_FILE))?;
    let id = st.db.insert_skill(&folder.to_string_lossy(), &doc.name, &doc.version, &manifest(&doc), "import").await?;
    st.db.add_skill_revision(&id, &doc.version, &skills::sha256_hex(text.as_bytes()), &text, Some(&format!("imported from {path}")), "applied", None).await?;
    view(st, &st.db.get_skill(&id).await?).await
}

pub async fn export(st: &SharedState, id: &str, dest: &str) -> AppResult<String> {
    let row = st.db.get_skill(id).await?;
    let bytes = skills::export_zip(&PathBuf::from(&row.path), &row.name)?;
    std::fs::write(dest, bytes)?;
    Ok(dest.to_string())
}

/// Which skills apply to `text`. `mode`: "auto" (match), "off", or
/// "manual" (exactly `manual`, by name).
pub async fn pick(st: &SharedState, text: &str, mode: &str, manual: &[String]) -> AppResult<Vec<Match>> {
    pick_opts(st, text, mode, manual, true).await
}

/// `pick`; `sync_first: false` matches against the database as it is,
/// without re-reading the skills folder (the context meter, per typing pause).
pub async fn pick_opts(st: &SharedState, text: &str, mode: &str, manual: &[String], sync_first: bool) -> AppResult<Vec<Match>> {
    if sync_first {
        sync(st).await?;
    }
    let rows = st.db.list_skills().await?;
    match mode {
        "off" => Ok(vec![]),
        "manual" => Ok(rows.iter().filter(|r| manual.contains(&r.name)).map(|r| Match { name: r.name.clone(), score: u32::MAX, reason: "chosen for this task".into() }).take(3).collect()),
        _ => {
            let cands: Vec<Candidate> = rows
                .iter()
                .map(|r| {
                    let m: serde_json::Value = serde_json::from_str(&r.manifest).unwrap_or_default();
                    Candidate {
                        name: r.name.clone(),
                        description: m["description"].as_str().unwrap_or_default().to_string(),
                        keywords: serde_json::from_value(m["keywords"].clone()).unwrap_or_default(),
                        mode: r.mode.clone(),
                    }
                })
                .collect();
            Ok(skills::pick(text, &cands, 3))
        }
    }
}

/// The picked skills that still exist and are not turned off (a chat's
/// sticky skills may name one deleted or switched off since).
pub async fn usable(st: &SharedState, picked: &[Match]) -> AppResult<Vec<Match>> {
    let rows = st.db.list_skills().await?;
    Ok(picked.iter().filter(|m| rows.iter().any(|r| r.name == m.name && r.mode != "off")).cloned().collect())
}

/// The `## Skills` prompt section for picked skills (bodies plus the paths of
/// scripts the user enabled), capped in size. Why a skill was picked is not
/// in the text: it changes with each message's wording, and any change near
/// the top of the prompt makes the model re-read everything after it.
pub async fn section(st: &SharedState, picked: &[Match]) -> AppResult<Option<String>> {
    if picked.is_empty() {
        return Ok(None);
    }
    let mut s = String::from("## Skills\nThese skills were picked for this work. Follow them where they apply.\n");
    for m in picked {
        let Ok(row) = st.db.get_skill(&m.name).await else { continue };
        let folder = PathBuf::from(&row.path);
        let Some((_, Ok(doc))) = read_doc(&folder.join(SKILL_FILE)) else { continue };
        s.push_str(&format!("\n### Skill: {} (v{})\n{}\n", doc.name, doc.version, doc.body.trim()));
        let on = enabled_scripts(&row);
        let scripts: Vec<String> = extra_files(&folder)?
            .into_iter()
            .filter(|f| on.get(&f.path) == Some(&f.sha256))
            .map(|f| folder.join(&f.path).to_string_lossy().into_owned())
            .collect();
        if !scripts.is_empty() {
            s.push_str(&format!("Scripts you may run with the shell tool (the user approved these files): {}\n", scripts.join(", ")));
        }
        if s.len() > MAX_SECTION_CHARS {
            s.truncate(MAX_SECTION_CHARS);
            s.push_str("\n[skill text shortened to fit]\n");
            break;
        }
    }
    Ok(Some(s))
}

/// The agent tool that proposes a skill edit (never applies it).
pub fn propose_tool(scope: &str) -> ExposedTool {
    ExposedTool {
        tool_id: format!("{}{scope}/skill_propose_edit", crate::services::mcp::AGENT_PREFIX),
        model_name: "skill_propose_edit".into(),
        server_id: format!("{}{scope}", crate::services::mcp::AGENT_PREFIX),
        server_name: "Local AI Studio".into(),
        tool_name: "skill_propose_edit".into(),
        description: "Propose a change to an existing skill. Give only what should change; the app applies it to the real SKILL.md. It is NOT applied yet: the user reviews the diff and approves or rejects it on the Skills page.".into(),
        input_schema: serde_json::json!({ "type": "object", "properties": {
            "name": { "type": "string", "description": "Existing skill name" },
            "description": { "type": "string", "description": "New one-line description" },
            "version": { "type": "string", "description": "New version, e.g. 2" },
            "add_keywords": { "type": "array", "items": { "type": "string" } },
            "remove_keywords": { "type": "array", "items": { "type": "string" } },
            "instructions": { "type": "string", "description": "The complete new instructions (Markdown), only if they should change" },
            "note": { "type": "string", "description": "Why this change helps" }
        }, "required": ["name", "note"], "additionalProperties": false }),
        policy: Policy::Allow,
        timeout_secs: 30,
    }
}

pub async fn propose_from_agent(st: &SharedState, args: &serde_json::Value, by: Option<&str>) -> AppResult<CallResult> {
    let name = args["name"].as_str().unwrap_or_default();
    let row = match st.db.get_skill(name).await {
        Ok(r) => r,
        Err(_) => return Ok(CallResult { text: format!("There is no skill named \"{name}\"; only existing skills can be edited."), is_error: true }),
    };
    let changes: skills::Changes = serde_json::from_value(args.clone()).unwrap_or_default();
    if changes.description.is_none() && changes.version.is_none() && changes.add_keywords.is_empty() && changes.remove_keywords.is_empty() && changes.instructions.is_none() {
        return Ok(CallResult { text: "Nothing to change: give at least one of description, version, add_keywords, remove_keywords or instructions.".into(), is_error: true });
    }
    let current = std::fs::read_to_string(PathBuf::from(&row.path).join(SKILL_FILE))?;
    let content = match skills::apply_changes(&current, &changes) {
        Ok(c) => c,
        Err(e) => return Ok(CallResult { text: format!("The change does not make a valid skill: {e}"), is_error: true }),
    };
    if content == current.replace("\r\n", "\n") {
        return Ok(CallResult { text: "That change would leave the skill exactly as it is.".into(), is_error: true });
    }
    let version = skills::parse(&content)?.version;
    let id = st.db.add_skill_revision(&row.id, &version, &skills::sha256_hex(content.as_bytes()), &content, args["note"].as_str(), "pending", by).await?;
    Ok(CallResult { text: format!("Proposal saved ({id}). It is not applied: the user will review it on the Skills page."), is_error: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_parse_and_install_once_without_overwriting() {
        for (name, text) in SAMPLES {
            let d = skills::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(d.name, name);
            assert!(!d.keywords.is_empty() && !d.body.trim().is_empty());
        }
        let dir = std::env::temp_dir().join(format!("las-samples-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // An existing skill with a sample's name is never overwritten.
        std::fs::create_dir_all(dir.join("git-commit-style")).unwrap();
        std::fs::write(dir.join("git-commit-style").join(SKILL_FILE), "mine").unwrap();
        let first = write_samples(&dir, &[]).unwrap();
        let names: Vec<_> = first.iter().map(|w| w.1.as_str()).collect();
        assert_eq!(names, vec!["rust-test-fix", "safe-powershell"]);
        assert_eq!(std::fs::read_to_string(dir.join("git-commit-style").join(SKILL_FILE)).unwrap(), "mine");
        // Idempotent: nothing is written again.
        assert!(write_samples(&dir, &[]).unwrap().is_empty());
        // A sample that was installed and then deleted is not brought back.
        std::fs::remove_dir_all(dir.join("rust-test-fix")).unwrap();
        assert!(write_samples(&dir, &["rust-test-fix".into()]).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

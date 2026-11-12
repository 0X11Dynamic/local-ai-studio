use std::path::PathBuf;

use app_core::AppError;
use inference::SamplingParams;
use serde::{Deserialize, Serialize};
use tauri::State;
use tokio_util::sync::CancellationToken;

use super::Cmd;
use crate::services::instructions::{self, InstructionPart, KIND, Sections};
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct ProfileView {
    pub id: String,
    pub title: String,
    pub builtin: bool,
    pub sections: Sections,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstructionSettings {
    pub base: String,
    pub chat_profile_id: Option<String>,
    pub agent_profile_id: Option<String>,
    /// Plan mode profile (the built-in Planner by default). Callers that
    /// leave it out keep the saved value.
    #[serde(default)]
    pub plan_profile_id: Option<String>,
}

fn validate(title: &str, s: &Sections) -> Cmd<()> {
    if title.trim().is_empty() || title.chars().count() > 80 {
        return Err(AppError::InvalidInput("profile name must be 1-80 characters".into()));
    }
    if !matches!(s.mode.as_str(), "sections" | "raw") {
        return Err(AppError::InvalidInput("mode must be sections or raw".into()));
    }
    if s.render().len() > 200_000 {
        return Err(AppError::InvalidInput("instruction is too long (200 KB max)".into()));
    }
    Ok(())
}

#[tauri::command]
pub async fn list_instruction_profiles(state: State<'_, SharedState>) -> Cmd<Vec<ProfileView>> {
    Ok(state
        .db
        .list_presets(KIND)
        .await?
        .into_iter()
        .map(|p| ProfileView { sections: instructions::sections_of(&p), id: p.id, title: p.title, builtin: p.builtin != 0, updated_at: p.updated_at })
        .collect())
}

#[tauri::command]
pub async fn create_instruction_profile(state: State<'_, SharedState>, title: String, sections: Sections) -> Cmd<String> {
    validate(&title, &sections)?;
    state.db.insert_preset(KIND, title.trim(), &serde_json::to_value(&sections)?, &serde_json::json!({}), false).await
}

#[tauri::command]
pub async fn update_instruction_profile(state: State<'_, SharedState>, id: String, title: String, sections: Sections) -> Cmd<()> {
    validate(&title, &sections)?;
    state.db.update_preset(&id, title.trim(), &serde_json::to_value(&sections)?, &serde_json::json!({})).await
}

#[tauri::command]
pub async fn delete_instruction_profile(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    state.db.delete_preset(&id).await?;
    // Defaults pointing at the deleted profile fall back to General Assistant
    // (the Plan default to the built-in Planner).
    for key in ["instructions.chat_profile", "instructions.agent_profile", "instructions.plan_profile"] {
        if state.db.get_setting::<String>(key).await?.as_deref() == Some(id.as_str()) {
            let all = state.db.list_presets(KIND).await?;
            let fallback = if key == "instructions.plan_profile" {
                all.iter().find(|p| p.builtin != 0 && p.title.starts_with("Planner"))
            } else {
                all.iter().find(|p| p.title == "General Assistant")
            };
            state.db.set_setting(key, &fallback.map(|p| p.id.clone())).await?;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn get_instruction_settings(state: State<'_, SharedState>) -> Cmd<InstructionSettings> {
    Ok(InstructionSettings {
        base: state.db.get_setting("instructions.base").await?.unwrap_or_default(),
        chat_profile_id: state.db.get_setting("instructions.chat_profile").await?,
        agent_profile_id: state.db.get_setting("instructions.agent_profile").await?,
        plan_profile_id: state.db.get_setting("instructions.plan_profile").await?,
    })
}

#[tauri::command]
pub async fn set_instruction_settings(state: State<'_, SharedState>, settings: InstructionSettings) -> Cmd<()> {
    if settings.base.len() > 100_000 {
        return Err(AppError::InvalidInput("base instruction is too long".into()));
    }
    for id in [&settings.chat_profile_id, &settings.agent_profile_id, &settings.plan_profile_id].into_iter().flatten() {
        state.db.get_preset(id).await?;
    }
    state.db.set_setting("instructions.base", &settings.base).await?;
    state.db.set_setting("instructions.chat_profile", &settings.chat_profile_id).await?;
    state.db.set_setting("instructions.agent_profile", &settings.agent_profile_id).await?;
    if settings.plan_profile_id.is_some() {
        state.db.set_setting("instructions.plan_profile", &settings.plan_profile_id).await?;
    }
    Ok(())
}

#[tauri::command]
pub async fn reset_instruction_base(state: State<'_, SharedState>) -> Cmd<String> {
    state.db.set_setting("instructions.base", &instructions::DEFAULT_BASE).await?;
    Ok(instructions::DEFAULT_BASE.into())
}

#[tauri::command]
pub async fn set_model_instruction_profile(state: State<'_, SharedState>, model_id: String, profile_id: Option<String>) -> Cmd<()> {
    if let Some(p) = &profile_id {
        state.db.get_preset(p).await?;
    }
    state.db.set_model_instruction_profile(&model_id, profile_id.as_deref()).await
}

#[tauri::command]
pub async fn get_model_instruction_profile(state: State<'_, SharedState>, model_id: String) -> Cmd<Option<String>> {
    state.db.model_instruction_profile(&model_id).await
}

#[derive(Debug, Serialize)]
pub struct PreviewView {
    pub text: String,
    pub parts: Vec<InstructionPart>,
    pub chars: usize,
    /// Counted with the loaded model's tokenizer; None if no model is loaded.
    pub tokens: Option<usize>,
    /// Tokens of the system message once wrapped in the chat template.
    pub templated_tokens: Option<usize>,
    pub model: Option<String>,
    pub n_ctx: Option<u64>,
}

/// Read-only preview of the effective instruction. `base` and `sections`
/// let the editor preview unsaved changes.
#[tauri::command]
pub async fn preview_instructions(
    state: State<'_, SharedState>,
    mode: String,
    sections: Sections,
    base: Option<String>,
) -> Cmd<PreviewView> {
    let st = state.inner().clone();
    let base = match base {
        Some(b) => b,
        None => st.db.get_setting("instructions.base").await?.unwrap_or_default(),
    };
    let plan = mode == "plan";
    let mode = if mode == "agent" || plan { "agent" } else { "chat" };
    // Agent / Plan Mode preview: a new chat's defaults (built-in tools on This
    // PC, plus every connected MCP server; Plan keeps only read-only tools).
    let section = if mode == "agent" {
        let mut defaults = crate::services::chat::ChatSettings { agent_mode: true, ..Default::default() };
        if plan {
            defaults.mode = "plan".into();
        }
        Some(crate::services::chat::agent_tools(&st, &defaults).await?.1)
    } else {
        None
    };
    let (text, parts) = instructions::compose(&base, &sections, mode, section.as_deref());
    let status = st.engine.status();
    let (tokens, templated, model) = match st.engine.client() {
        Some(c) => {
            let t = c.tokenize(&text, false).await.ok().map(|v| v.len());
            let msgs = serde_json::json!([{ "role": "system", "content": text }, { "role": "user", "content": "x" }]);
            let only_user = serde_json::json!([{ "role": "user", "content": "x" }]);
            let with = match c.apply_template(&msgs).await {
                Ok(p) => c.tokenize(&p, false).await.ok().map(|v| v.len()),
                Err(_) => None,
            };
            let without = match c.apply_template(&only_user).await {
                Ok(p) => c.tokenize(&p, false).await.ok().map(|v| v.len()),
                Err(_) => None,
            };
            let templated = match (with, without) {
                (Some(a), Some(b)) => Some(a.saturating_sub(b)),
                _ => None,
            };
            (t, templated, status.props.as_ref().and_then(|p| p.model_path.clone()))
        }
        None => (None, None, None),
    };
    Ok(PreviewView { chars: text.chars().count(), text, parts, tokens, templated_tokens: templated, model, n_ctx: status.props.and_then(|p| p.n_ctx) })
}

#[derive(Debug, Serialize)]
pub struct TestResult {
    pub content: String,
    pub reasoning: String,
    pub stats: inference::ChatStats,
}

/// Run one non-destructive prompt with an (unsaved) instruction. Nothing is stored.
#[tauri::command]
pub async fn test_instructions(state: State<'_, SharedState>, sections: Sections, base: Option<String>, prompt: String) -> Cmd<TestResult> {
    let st = state.inner().clone();
    let client = st.engine.client().ok_or_else(|| AppError::InvalidInput("load a model to test an instruction".into()))?;
    let base = match base {
        Some(b) => b,
        None => st.db.get_setting("instructions.base").await?.unwrap_or_default(),
    };
    let (text, _) = instructions::compose(&base, &sections, "chat", None);
    let prompt = if prompt.trim().is_empty() { "Introduce yourself in two sentences.".to_string() } else { prompt };
    let mut msgs = vec![];
    if !text.is_empty() {
        msgs.push(serde_json::json!({ "role": "system", "content": text }));
    }
    msgs.push(serde_json::json!({ "role": "user", "content": prompt }));
    let params = SamplingParams { max_tokens: Some(512), ..Default::default() };
    let out = inference::chat::stream_chat(&client, &serde_json::Value::Array(msgs), &params, &CancellationToken::new(), |_| {}).await?;
    Ok(TestResult { content: out.content, reasoning: out.reasoning, stats: out.stats })
}

#[derive(Debug, Serialize, Deserialize)]
struct ProfileFile {
    format: String,
    version: u32,
    title: String,
    sections: Sections,
}

#[tauri::command]
pub async fn export_instruction_profile(state: State<'_, SharedState>, id: String, path: String) -> Cmd<()> {
    let p = state.db.get_preset(&id).await?;
    let s = instructions::sections_of(&p);
    let path = PathBuf::from(path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let body = match ext.as_str() {
        "json" => serde_json::to_string_pretty(&ProfileFile { format: "local-ai-studio/instruction".into(), version: 1, title: p.title.clone(), sections: s })?,
        "md" => format!("# {}\n\n{}\n", p.title, s.render()),
        _ => return Err(AppError::InvalidInput("export as .json or .md".into())),
    };
    tokio::fs::write(path, body).await?;
    Ok(())
}

/// Import a profile from .json (exported by this app) or .md (raw text; the
/// first "# " heading becomes the name). Imported text is data only.
#[tauri::command]
pub async fn import_instruction_profile(state: State<'_, SharedState>, path: String) -> Cmd<String> {
    let path = PathBuf::from(path);
    let meta = tokio::fs::metadata(&path).await?;
    if meta.len() > 1_000_000 {
        return Err(AppError::InvalidInput("file is too large for an instruction profile".into()));
    }
    let text = tokio::fs::read_to_string(&path).await?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let (title, sections) = match ext.as_str() {
        "json" => {
            let f: ProfileFile = serde_json::from_str(&text).map_err(|e| AppError::InvalidInput(format!("not an instruction profile file: {e}")))?;
            (f.title, f.sections)
        }
        "md" | "txt" => {
            let mut lines = text.lines();
            let first = lines.clone().next().unwrap_or("");
            let (title, body) = match first.strip_prefix("# ") {
                Some(t) => {
                    lines.next();
                    (t.trim().to_string(), lines.collect::<Vec<_>>().join("\n"))
                }
                None => (path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Imported".into()), text.clone()),
            };
            (title, Sections { mode: "raw".into(), raw: body.trim().to_string(), ..Default::default() })
        }
        _ => return Err(AppError::InvalidInput("import a .json or .md file".into())),
    };
    let title = format!("{} (imported)", title.chars().take(60).collect::<String>());
    validate(&title, &sections)?;
    state.db.insert_preset(KIND, &title, &serde_json::to_value(&sections)?, &serde_json::json!({}), false).await
}

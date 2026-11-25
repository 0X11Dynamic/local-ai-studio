use std::path::PathBuf;

use app_core::AppError;
use persistence::{ConversationRow, ConversationSummary, MessageRow};
use serde::Serialize;
use tauri::{AppHandle, State};

use super::Cmd;
use crate::services::attachments::{self, Attachment, MediaSupport};
use crate::services::chat::{self, ChatSettings, ContextReport, SendStarted};
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct ConversationView {
    #[serde(flatten)]
    pub conversation: ConversationRow,
    pub settings_parsed: ChatSettings,
    pub messages: Vec<MessageRow>,
    pub generating: bool,
    /// Attached on the composer, not sent yet (go with the next chat_send).
    pub pending_attachments: Vec<Attachment>,
}

#[tauri::command]
pub async fn list_conversations(state: State<'_, SharedState>, query: Option<String>) -> Cmd<Vec<ConversationSummary>> {
    state.db.list_conversations(query.as_deref()).await
}

#[tauri::command]
pub async fn create_conversation(state: State<'_, SharedState>) -> Cmd<ConversationRow> {
    let model = state.engine.status().model_id;
    state
        .db
        .create_conversation("New chat", model.as_deref(), &serde_json::to_value(ChatSettings::default())?)
        .await
}

#[tauri::command]
pub async fn get_conversation(state: State<'_, SharedState>, id: String) -> Cmd<ConversationView> {
    let c = state.db.get_conversation(&id).await?;
    let messages = state.db.list_messages(&id).await?;
    let generating = state.generations.lock().unwrap().contains_key(&id);
    let pending_attachments = attachments::pending(state.inner(), &id);
    Ok(ConversationView { settings_parsed: chat::settings_of(&c), conversation: c, messages, generating, pending_attachments })
}

#[tauri::command]
pub async fn rename_conversation(state: State<'_, SharedState>, id: String, title: String) -> Cmd<()> {
    let t = title.trim();
    if t.is_empty() || t.chars().count() > 200 {
        return Err(AppError::InvalidInput("title must be 1-200 characters".into()));
    }
    state.db.rename_conversation(&id, t).await
}

#[tauri::command]
pub async fn duplicate_conversation(state: State<'_, SharedState>, id: String) -> Cmd<ConversationRow> {
    let copy = state.db.duplicate_conversation(&id).await?;
    // Attachment records name files in the conversation's folder: copy them.
    attachments::copy_conversation(state.inner(), &id, &copy.id)?;
    Ok(copy)
}

#[tauri::command]
/// Also stops its subagents and deletes its hidden chat run; their git
/// worktrees are removed when clean and merged (or empty), else kept and
/// listed in `kept_worktrees`.
pub async fn delete_conversation(state: State<'_, SharedState>, id: String) -> Cmd<crate::services::tasks::DeleteResult> {
    delete_one(state.inner(), &id).await
}

/// One chat's deletion, shared by the single and bulk commands.
async fn delete_one(state: &SharedState, id: &str) -> Cmd<crate::services::tasks::DeleteResult> {
    // Only a real chat: the id names a folder that is moved to the trash.
    if !attachments::is_id(id) {
        return Err(AppError::InvalidInput("not a valid chat id".into()));
    }
    state.db.get_conversation(id).await.map_err(|e| match e {
        AppError::NotFound(_) => AppError::NotFound("chat not found".into()),
        e => e,
    })?;
    chat::stop(state, id);
    let r = crate::services::tasks::delete_chat_run(state, id).await?;
    state.db.delete_conversation(id).await?;
    // Its attachment files go to the app trash (trashttachments-<id>-<time>).
    attachments::trash_conversation(state, id);
    Ok(r)
}

pub const MAX_BULK_DELETE: usize = 1000;

#[derive(Debug, Serialize, PartialEq)]
pub struct DeleteFailure {
    pub id: String,
    pub error: String,
}

#[derive(Debug, Default, Serialize)]
pub struct BulkDeleteResult {
    pub deleted: usize,
    pub failed: Vec<DeleteFailure>,
    pub removed_worktrees: Vec<crate::services::tasks::WorktreeReport>,
    pub kept_worktrees: Vec<crate::services::tasks::WorktreeReport>,
}

/// Drop repeated ids (first occurrence wins) so a chat is not deleted twice.
fn dedupe_ids(ids: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    ids.into_iter().filter(|i| seen.insert(i.clone())).collect()
}

/// Fold per-chat outcomes into one result; failures do not stop the rest.
fn aggregate(outcomes: Vec<(String, Cmd<crate::services::tasks::DeleteResult>)>) -> BulkDeleteResult {
    let mut out = BulkDeleteResult::default();
    for (id, r) in outcomes {
        match r {
            Ok(d) => {
                out.deleted += 1;
                out.removed_worktrees.extend(d.removed_worktrees);
                out.kept_worktrees.extend(d.kept_worktrees);
            }
            Err(e) => out.failed.push(DeleteFailure { id, error: e.to_string() }),
        }
    }
    out
}

/// Delete several chats (max 1000). Invalid or missing ids are listed in
/// `failed`; the others are still deleted.
#[tauri::command]
pub async fn delete_conversations(state: State<'_, SharedState>, ids: Vec<String>) -> Cmd<BulkDeleteResult> {
    if ids.len() > MAX_BULK_DELETE {
        return Err(AppError::InvalidInput(format!("delete at most {MAX_BULK_DELETE} chats at once")));
    }
    let mut outcomes = Vec::new();
    for id in dedupe_ids(ids) {
        let r = delete_one(state.inner(), &id).await;
        outcomes.push((id, r));
    }
    Ok(aggregate(outcomes))
}

#[cfg(test)]
mod bulk_delete_tests {
    use super::*;
    use crate::services::tasks::{DeleteResult, WorktreeReport};

    fn wt(n: &str) -> WorktreeReport {
        WorktreeReport { subagent_id: n.into(), label: n.into(), path: n.into(), branch: n.into(), reason: None }
    }

    #[test]
    fn dedupes_keeping_order() {
        let v = dedupe_ids(vec!["a".into(), "b".into(), "a".into(), "c".into(), "b".into()]);
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    #[test]
    fn aggregates_successes_and_failures() {
        let r = aggregate(vec![
            ("a".into(), Ok(DeleteResult { removed_worktrees: vec![wt("w1")], kept_worktrees: vec![] })),
            ("bad".into(), Err(AppError::InvalidInput("not a valid chat id".into()))),
            ("c".into(), Ok(DeleteResult { removed_worktrees: vec![], kept_worktrees: vec![wt("w2")] })),
        ]);
        assert_eq!(r.deleted, 2);
        assert_eq!(r.failed.len(), 1);
        assert_eq!(r.failed[0].id, "bad");
        assert!(r.failed[0].error.contains("not a valid chat id"));
        assert_eq!(r.removed_worktrees.len(), 1);
        assert_eq!(r.kept_worktrees.len(), 1);
    }

    #[test]
    fn empty_is_zero() {
        let r = aggregate(vec![]);
        assert_eq!((r.deleted, r.failed.len()), (0, 0));
    }
}

#[tauri::command]
pub async fn update_chat_settings(state: State<'_, SharedState>, id: String, settings: ChatSettings) -> Cmd<()> {
    settings.sampling.validate().map_err(AppError::InvalidInput)?;
    if !matches!(settings.overflow.as_str(), "warn" | "drop_oldest") {
        return Err(AppError::InvalidInput("overflow must be warn or drop_oldest".into()));
    }
    if settings.reserve_tokens > 131_072 {
        return Err(AppError::InvalidInput("reserve is too large".into()));
    }
    if settings.limits.max_tool_calls.is_some_and(|n| n == 0 || n > 1000) {
        return Err(AppError::InvalidInput("tool calls per reply must be 1-1000, or no limit".into()));
    }
    if settings.limits.max_wall_secs.is_some_and(|s| !(10..=86_400).contains(&s)) {
        return Err(AppError::InvalidInput("reply time limit must be 10-86400 seconds, or no limit".into()));
    }
    if settings.max_subagents == 0 || settings.max_subagents > crate::services::tasks::MAX_PARALLEL_HARD {
        return Err(AppError::InvalidInput(format!("subagents at once must be 1 to {}", crate::services::tasks::MAX_PARALLEL_HARD)));
    }
    chat::validate_effort(settings.reasoning_effort.as_deref())?;
    for sid in &settings.tool_servers {
        state.db.get_mcp_server(sid).await?;
    }
    if let Some(p) = &settings.profile_id {
        state.db.get_preset(p).await?;
    }
    // Loaded tools and sticky skills are only ever added (an older copy from
    // the UI must not drop what the model loaded meanwhile).
    chat::save_settings(state.inner(), &id, None, &settings).await
}

/// Send a message. Pending attachments go with it: `attachments` (ids) picks
/// which, absent = all pending. The text may be empty when files are attached.
#[tauri::command]
pub async fn chat_send(app: AppHandle, state: State<'_, SharedState>, id: String, content: String, attachments: Option<Vec<String>>) -> Cmd<SendStarted> {
    chat::send_with(app, state.inner().clone(), id, Some(content), attachments).await
}

/// Attach a file by path to the chat's next message. Copies it into the
/// attachments folder and returns the record (also listed as pending).
#[tauri::command]
pub async fn chat_attach(state: State<'_, SharedState>, id: String, path: String) -> Cmd<Attachment> {
    attachments::add_from_path(state.inner(), &id, std::path::Path::new(&path)).await
}

/// Remove a pending attachment (before sending). False if it wasn't pending.
#[tauri::command]
pub async fn chat_detach(state: State<'_, SharedState>, id: String, attachment_id: String) -> Cmd<bool> {
    Ok(attachments::remove_pending(state.inner(), &id, &attachment_id).await)
}

#[tauri::command]
pub fn chat_pending_attachments(state: State<'_, SharedState>, id: String) -> Cmd<Vec<Attachment>> {
    Ok(attachments::pending(state.inner(), &id))
}

/// What the loaded model takes: thinking controls (its capabilities) and
/// whether images / audio can be sent now. None when no model is loaded.
#[tauri::command]
pub async fn chat_model_support(state: State<'_, SharedState>) -> Cmd<Option<MediaSupport>> {
    Ok(attachments::media_support(state.inner()).await)
}

#[tauri::command]
pub fn chat_stop(state: State<'_, SharedState>, id: String) -> Cmd<bool> {
    Ok(chat::stop(state.inner(), &id))
}

/// Delete the last reply and generate a new one.
#[tauri::command]
pub async fn chat_regenerate(app: AppHandle, state: State<'_, SharedState>, id: String) -> Cmd<SendStarted> {
    let st = state.inner().clone();
    if st.generations.lock().unwrap().contains_key(&id) {
        return Err(AppError::Conflict("stop the current reply first".into()));
    }
    let msgs = st.db.list_messages(&id).await?;
    let last_user = msgs.iter().rposition(|m| m.role == "user").ok_or_else(|| AppError::InvalidInput("nothing to regenerate".into()))?;
    if let Some(next) = msgs.get(last_user + 1) {
        st.db.truncate_from(&id, &next.id).await?;
    }
    chat::send(app, st, id, None).await
}

/// Replace a user message's text, drop everything after it, and reply again.
#[tauri::command]
pub async fn chat_edit(app: AppHandle, state: State<'_, SharedState>, id: String, message_id: String, content: String) -> Cmd<SendStarted> {
    let st = state.inner().clone();
    if st.generations.lock().unwrap().contains_key(&id) {
        return Err(AppError::Conflict("stop the current reply first".into()));
    }
    if content.trim().is_empty() {
        return Err(AppError::InvalidInput("message is empty".into()));
    }
    let msgs = st.db.list_messages(&id).await?;
    let pos = msgs.iter().position(|m| m.id == message_id).ok_or_else(|| AppError::NotFound("message not found".into()))?;
    if msgs[pos].role != "user" {
        return Err(AppError::InvalidInput("only your own messages can be edited".into()));
    }
    st.db.set_message_content(&message_id, &content).await?;
    if let Some(next) = msgs.get(pos + 1) {
        st.db.truncate_from(&id, &next.id).await?;
    }
    chat::send(app, st, id, None).await
}

#[tauri::command]
pub async fn chat_context(state: State<'_, SharedState>, id: String, draft: String) -> Cmd<ContextReport> {
    chat::context_report(state.inner(), &id, &draft).await
}

/// Export to a path chosen in the native save dialog (.md or .json).
#[tauri::command]
pub async fn export_conversation(state: State<'_, SharedState>, id: String, path: String) -> Cmd<()> {
    let p = PathBuf::from(&path);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let c = state.db.get_conversation(&id).await?;
    let msgs = state.db.list_messages(&id).await?;
    let body = match ext.as_str() {
        "md" => chat::to_markdown(&c, &msgs),
        "json" => serde_json::to_string_pretty(&serde_json::json!({
            "title": c.title, "created_at": c.created_at, "settings": serde_json::from_str::<serde_json::Value>(&c.settings).unwrap_or_default(),
            "messages": msgs.iter().map(|m| serde_json::json!({
                "role": m.role, "content": m.content, "reasoning": m.reasoning, "model": m.model_name,
                "stats": m.stats.as_deref().and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok()), "created_at": m.created_at,
                "attachments": attachments::parse(m)
            })).collect::<Vec<_>>()
        }))?,
        _ => return Err(AppError::InvalidInput("export as .md or .json".into())),
    };
    tokio::fs::write(&p, body).await?;
    Ok(())
}

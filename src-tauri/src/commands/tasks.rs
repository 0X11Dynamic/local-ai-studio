//! Agent Tasks, Subagents and Skills commands.

use agent_runtime::task::Budgets;
use persistence::{AgentRunSummary, SkillRevisionRow, SubagentRow};
use tauri::State;

use super::Cmd;
use crate::services::skills::{self as sk, SkillView};
use crate::services::tasks::{self as svc, TaskDetail, TaskSettings};
use crate::state::SharedState;

#[tauri::command]
pub async fn start_task(app: tauri::AppHandle, state: State<'_, SharedState>, title: String, objective: String, settings: TaskSettings, budgets: Budgets) -> Cmd<String> {
    svc::start(state.inner(), &app, &title, &objective, settings, budgets).await
}

#[tauri::command]
pub async fn list_tasks(state: State<'_, SharedState>) -> Cmd<Vec<AgentRunSummary>> {
    svc::list(state.inner()).await
}

#[tauri::command]
pub async fn get_task(state: State<'_, SharedState>, id: String) -> Cmd<TaskDetail> {
    svc::detail(state.inner(), &id).await
}

#[tauri::command]
pub fn pause_task(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    svc::pause(state.inner(), &id)
}

#[tauri::command]
pub async fn resume_task(app: tauri::AppHandle, state: State<'_, SharedState>, id: String, budgets: Option<Budgets>) -> Cmd<()> {
    svc::resume(state.inner(), &app, &id, budgets).await
}

/// Approve a "Plan first" task's plan (optionally edited) and run it.
#[tauri::command]
pub async fn approve_task_plan(app: tauri::AppHandle, state: State<'_, SharedState>, id: String, plan: Option<String>) -> Cmd<()> {
    svc::approve_plan(state.inner(), &app, &id, plan).await
}

/// Answer an agent's `ask_user` question (the asking agent, or only that
/// subagent, continues with the answer as the tool result).
#[tauri::command]
pub async fn answer_task_question(app: tauri::AppHandle, state: State<'_, SharedState>, run_id: String, question_id: String, answer: String) -> Cmd<()> {
    svc::answer_question(state.inner(), &app, &run_id, &question_id, &answer).await
}

/// Send a message to a task: "answered" (it answered the waiting question),
/// "queued" (read before the agent's next model turn) or "resumed" (a stopped
/// task restarted with it; needs `resume`).
#[tauri::command]
pub async fn send_task_message(app: tauri::AppHandle, state: State<'_, SharedState>, run_id: String, text: String, resume: Option<bool>, budgets: Option<Budgets>) -> Cmd<String> {
    svc::send_message(state.inner(), &app, &run_id, &text, resume.unwrap_or(false), budgets).await
}

/// Questions waiting for the user in any task (the app-wide indicator).
#[tauri::command]
pub async fn pending_task_questions(state: State<'_, SharedState>) -> Cmd<Vec<persistence::AgentStepRow>> {
    svc::waiting_questions(state.inner()).await
}

#[tauri::command]
pub async fn cancel_task(app: tauri::AppHandle, state: State<'_, SharedState>, id: String) -> Cmd<()> {
    svc::cancel(state.inner(), &app, &id).await
}

/// Deletes the task; its subagents' git worktrees are removed when clean and
/// merged (or empty), else kept and listed in `kept_worktrees`.
#[tauri::command]
pub async fn delete_task(state: State<'_, SharedState>, id: String) -> Cmd<svc::DeleteResult> {
    svc::delete(state.inner(), &id).await
}

#[derive(Debug, serde::Serialize)]
pub struct ChatRunView {
    pub id: String,
    pub title: String,
    pub conversation_id: String,
}

/// The hidden runs that hold chats' subagents (for the Subagents page).
#[tauri::command]
pub async fn list_chat_runs(state: State<'_, SharedState>) -> Cmd<Vec<ChatRunView>> {
    Ok(state.db.list_chat_runs().await?.into_iter().map(|(id, title, conversation_id)| ChatRunView { id, title, conversation_id }).collect())
}

#[derive(Debug, serde::Serialize)]
pub struct ChatSubagents {
    /// The conversation's hidden run (None until a reply has used subagents).
    pub run_id: Option<String>,
    pub running: bool,
    pub subagents: Vec<SubagentRow>,
}

/// A chat's subagents (each row's settings has the spawn `call_id`).
#[tauri::command]
pub async fn chat_subagents(state: State<'_, SharedState>, id: String) -> Cmd<ChatSubagents> {
    let run_id = state.db.find_chat_run(&id).await?;
    let subagents = match &run_id {
        Some(r) => state.db.list_subagents(r).await?,
        None => vec![],
    };
    let running = run_id.as_ref().is_some_and(|r| state.tasks.lock().unwrap().contains_key(r));
    Ok(ChatSubagents { run_id, running, subagents })
}

#[tauri::command]
pub fn stop_subagent(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    svc::stop_subagent(state.inner(), &id)
}

#[tauri::command]
pub async fn recent_subagents(state: State<'_, SharedState>) -> Cmd<Vec<SubagentRow>> {
    state.db.recent_subagents(300).await
}

// ---------- skills ----------

#[tauri::command]
pub async fn list_skills(state: State<'_, SharedState>) -> Cmd<Vec<SkillView>> {
    sk::list(state.inner()).await
}

/// Create (`id` null) or save a skill.
#[tauri::command]
pub async fn save_skill(state: State<'_, SharedState>, id: Option<String>, content: String) -> Cmd<SkillView> {
    sk::save(state.inner(), id.as_deref(), &content, Some("edited in the app")).await
}

#[tauri::command]
pub async fn delete_skill(state: State<'_, SharedState>, id: String) -> Cmd<()> {
    sk::delete(state.inner(), &id).await
}

#[tauri::command]
pub async fn set_skill_mode(state: State<'_, SharedState>, id: String, mode: String) -> Cmd<()> {
    sk::set_mode(state.inner(), &id, &mode).await
}

#[tauri::command]
pub async fn set_skill_script(state: State<'_, SharedState>, id: String, path: String, enabled: bool) -> Cmd<()> {
    sk::set_script(state.inner(), &id, &path, enabled).await
}

#[tauri::command]
pub async fn skill_revisions(state: State<'_, SharedState>, id: String) -> Cmd<Vec<SkillRevisionRow>> {
    sk::revisions(state.inner(), &id).await
}

#[tauri::command]
pub async fn revert_skill(state: State<'_, SharedState>, revision_id: String) -> Cmd<SkillView> {
    sk::revert(state.inner(), &revision_id).await
}

#[tauri::command]
pub async fn approve_skill_proposal(state: State<'_, SharedState>, revision_id: String) -> Cmd<SkillView> {
    sk::approve(state.inner(), &revision_id).await
}

#[tauri::command]
pub async fn reject_skill_proposal(state: State<'_, SharedState>, revision_id: String) -> Cmd<()> {
    sk::reject(state.inner(), &revision_id).await
}

#[tauri::command]
pub async fn import_skill(state: State<'_, SharedState>, path: String) -> Cmd<SkillView> {
    sk::import(state.inner(), &path).await
}

#[tauri::command]
pub async fn export_skill(state: State<'_, SharedState>, id: String, dest: String) -> Cmd<String> {
    sk::export(state.inner(), &id, &dest).await
}

/// "Try an objective": which skills would be picked, and why.
#[tauri::command]
pub async fn match_skills(state: State<'_, SharedState>, text: String) -> Cmd<Vec<skills::Match>> {
    sk::pick(state.inner(), &text, "auto", &[]).await
}

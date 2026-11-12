//! Tauri shell: builds the shared state, registers IPC commands, and forwards
//! backend events to the UI. All real work happens in the workspace crates.

mod commands;
mod services;
mod state;

use std::sync::{Arc, Mutex};

use tauri::{Emitter, Manager};

use crate::state::AppState;

pub fn run() {
    let paths = app_core::AppPaths::resolve().expect("resolve app folders");
    paths.ensure_dirs().expect("create app folders");
    init_logging(&paths);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
            let state = tauri::async_runtime::block_on(build_state(paths.clone()))
                .map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))?;
            let state = Arc::new(state);
            app.manage(state.clone());

            // Forward download progress to the UI.
            let handle_dl = app.handle().clone();
            let mut drx = state.downloads.subscribe();
            tauri::async_runtime::spawn(async move {
                loop {
                    match drx.recv().await {
                        Ok(ev) => {
                            let _ = handle_dl.emit("download://event", &ev);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
            });

            // Forward MCP server status changes to the UI.
            let handle_mcp = app.handle().clone();
            let mut mrx = state.mcp.subscribe();
            tauri::async_runtime::spawn(async move {
                loop {
                    match mrx.recv().await {
                        Ok(ev) => {
                            let _ = handle_mcp.emit("mcp://status", &ev);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
            });
            let st_mcp = state.clone();
            tauri::async_runtime::spawn(async move { services::mcp::autoconnect(st_mcp) });

            // Start the local API if the user enabled autostart.
            let st_api = state.clone();
            tauri::async_runtime::spawn(async move {
                let cfg = commands::api::saved_config(&st_api).await;
                if cfg.autostart
                    && let Err(e) = commands::api::start_with(&st_api, cfg).await
                {
                    tracing::warn!("local API autostart failed: {e}");
                }
            });

            // Forward engine status and log events to the UI.
            let handle = app.handle().clone();
            let mut rx = state.engine.subscribe();
            let st = state.clone();
            tauri::async_runtime::spawn(async move {
                use inference::{EngineEvent, EngineState};
                loop {
                    match rx.recv().await {
                        Ok(ev) => {
                            if let EngineEvent::InstanceEnded { instance_id } = &ev {
                                let _ = st.db.set_instance_state(instance_id, "unloaded", None).await;
                            }
                            if let EngineEvent::Status { status } = &ev
                                && let Some(inst) = &status.instance_id
                            {
                                let (state, err) = match status.state {
                                    EngineState::Loading => ("loading", None),
                                    EngineState::Loaded => ("loaded", None),
                                    EngineState::Failed => ("failed", status.error.as_deref()),
                                    _ => ("", None),
                                };
                                if !state.is_empty() {
                                    let _ = st.db.set_instance_state(inst, state, err).await;
                                }
                            }
                            let _ = handle.emit("engine://event", &ev);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Unload cleanly on exit; the Job Object is the backstop if this is skipped.
                if let Some(st) = window.app_handle().try_state::<state::SharedState>() {
                    let st = st.inner().clone();
                    tauri::async_runtime::block_on(async move {
                        let inst = st.engine.status().instance_id;
                        let was_running = st.engine.backend_pid().is_some();
                        st.api.stop().await;
                        st.mcp.disconnect_all().await;
                        services::exec::shutdown(&st).await;
                        st.engine.unload().await;
                        if let (Some(i), true) = (inst, was_running) {
                            let _ = st.db.set_instance_state(&i, "unloaded", None).await;
                        }
                        tracing::info!("window closed: backend unloaded");
                    });
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::system::get_paths,
            commands::system::user_guide,
            commands::system::get_hardware,
            commands::system::get_resources,
            commands::system::get_settings,
            commands::system::set_setting,
            commands::runtimes::list_runtimes,
            commands::runtimes::fetch_available_runtimes,
            commands::runtimes::install_runtime,
            commands::runtimes::cancel_runtime_install,
            commands::runtimes::register_custom_runtime,
            commands::runtimes::remove_runtime,
            commands::runtimes::setup_plan, commands::runtimes::setup_should_show, commands::runtimes::finish_setup, commands::runtimes::open_driver_download,
            commands::models::list_models,
            commands::models::import_models,
            commands::models::scan_lmstudio,
            commands::models::refresh_models,
            commands::models::remove_model,
            commands::models::delete_model_files,
            commands::models::estimate_memory,
            commands::models::get_load_defaults,
            commands::models::save_model_defaults,
            commands::models::reset_model_defaults,
            commands::engine::load_model,
            commands::engine::unload_model,
            commands::engine::reload_with_projector,
            commands::engine::cancel_load,
            commands::engine::engine_status,
            commands::engine::engine_logs,
            commands::discover::hf_search,
            commands::discover::hf_repo,
            commands::discover::download_variant,
            commands::discover::list_downloads,
            commands::discover::pause_download,
            commands::discover::resume_download,
            commands::discover::cancel_download,
            commands::discover::clear_finished_downloads,
            commands::discover::set_hf_token,
            commands::discover::has_hf_token,
            commands::chat::list_conversations,
            commands::chat::create_conversation,
            commands::chat::get_conversation,
            commands::chat::rename_conversation,
            commands::chat::duplicate_conversation,
            commands::chat::delete_conversation,
            commands::chat::update_chat_settings,
            commands::chat::chat_send,
            commands::chat::chat_stop,
            commands::chat::chat_regenerate,
            commands::chat::chat_edit,
            commands::chat::chat_context,
            commands::chat::chat_attach,
            commands::chat::chat_detach,
            commands::chat::chat_pending_attachments,
            commands::chat::chat_model_support,
            commands::chat::export_conversation,
            commands::instructions::list_instruction_profiles,
            commands::instructions::create_instruction_profile,
            commands::instructions::update_instruction_profile,
            commands::instructions::delete_instruction_profile,
            commands::instructions::get_instruction_settings,
            commands::instructions::set_instruction_settings,
            commands::instructions::reset_instruction_base,
            commands::instructions::set_model_instruction_profile,
            commands::instructions::get_model_instruction_profile,
            commands::instructions::preview_instructions,
            commands::instructions::test_instructions,
            commands::instructions::export_instruction_profile,
            commands::instructions::import_instruction_profile,
            commands::api::api_status,
            commands::api::api_start,
            commands::api::api_save_config,
            commands::api::api_stop,
            commands::api::api_logs,
            commands::api::api_generate_token,
            commands::api::api_has_token,
            commands::api::api_reveal_token,
            commands::mcp::list_mcp_servers,
            commands::mcp::add_mcp_server,
            commands::mcp::update_mcp_server,
            commands::mcp::delete_mcp_server,
            commands::mcp::connect_mcp_server,
            commands::mcp::disconnect_mcp_server,
            commands::mcp::mcp_server_logs,
            commands::mcp::set_mcp_tool_pref,
            commands::mcp::import_mcp_servers,
            commands::mcp::import_mcp_from,
            commands::mcp::import_mcp_file,
            commands::mcp::export_mcp_servers,
            commands::mcp::list_tool_grants,
            commands::mcp::revoke_tool_grant,
            commands::mcp::list_tool_activity,
            commands::mcp::pending_tool_approvals,
            commands::mcp::answer_tool_approval,
            commands::mcp::preview_chat_tools,
            commands::mcp::run_tool_manually,
            commands::mcp::chat_tool_costs,
            commands::exec::list_environments,
            commands::exec::set_local_roots,
            commands::exec::set_local_outside,
            commands::exec::environment_health,
            commands::exec::add_ssh_environment,
            commands::exec::update_ssh_environment,
            commands::exec::ssh_host_key,
            commands::exec::ssh_trust_host,
            commands::exec::add_docker_environment,
            commands::exec::update_docker_environment,
            commands::exec::docker_status,
            commands::exec::wsl_status,
            commands::exec::create_sandbox,
            commands::exec::delete_environment,
            commands::exec::set_builtin_tool_pref,
            commands::exec::run_builtin_tool,
            commands::tasks::start_task,
            commands::tasks::list_tasks,
            commands::tasks::get_task,
            commands::tasks::pause_task,
            commands::tasks::resume_task,
            commands::tasks::approve_task_plan,
            commands::tasks::answer_task_question,
            commands::tasks::send_task_message,
            commands::tasks::pending_task_questions,
            commands::tasks::cancel_task,
            commands::tasks::delete_task,
            commands::tasks::list_chat_runs,
            commands::tasks::chat_subagents,
            commands::tasks::stop_subagent,
            commands::tasks::recent_subagents,
            commands::tasks::list_skills,
            commands::tasks::save_skill,
            commands::tasks::delete_skill,
            commands::tasks::set_skill_mode,
            commands::tasks::set_skill_script,
            commands::tasks::skill_revisions,
            commands::tasks::revert_skill,
            commands::tasks::approve_skill_proposal,
            commands::tasks::reject_skill_proposal,
            commands::tasks::import_skill,
            commands::tasks::export_skill,
            commands::tasks::match_skills,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Local AI Studio");
}

async fn build_state(paths: app_core::AppPaths) -> app_core::AppResult<AppState> {
    let db = persistence::Db::open(&paths.db_path).await?;
    let stale = db.close_stale_instances().await?;
    if stale > 0 {
        tracing::info!("marked {stale} model instance(s) from a previous run as unloaded");
    }
    let http = reqwest::Client::builder()
        .user_agent(concat!("LocalAIStudio/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| app_core::AppError::Other(e.to_string()))?;
    let monitor = system_monitor::SystemMonitor::new().map_err(app_core::AppError::Other)?;
    let interrupted = db.close_stale_tool_execs().await?;
    if interrupted > 0 {
        tracing::info!("{interrupted} tool call(s) from a previous run marked interrupted (not re-run)");
    }
    let runs = db.interrupt_running_runs().await?;
    if runs > 0 {
        tracing::info!("{runs} agent task(s) were running when the app stopped; marked interrupted (resume by hand)");
    }
    let paused = db.pause_interrupted_downloads().await?;
    if paused > 0 {
        tracing::info!("{paused} interrupted download(s) marked paused");
    }
    services::instructions::ensure_builtins(&db).await?;
    let concurrency = db.get_setting::<u64>("downloads.concurrency").await?.unwrap_or(2) as usize;
    // An isolated profile (LOCALAISTUDIO_DATA_DIR, used by the test suites)
    // gets its own Credential Manager namespace, so it can never read,
    // overwrite or leave entries next to the real app's secrets.
    let credentials = match std::env::var("LOCALAISTUDIO_DATA_DIR").ok().filter(|d| !d.is_empty()) {
        Some(dir) => {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            dir.to_lowercase().hash(&mut h);
            security::CredentialStore::with_service(&format!("LocalAIStudio-profile-{:016x}", h.finish()))
        }
        None => security::CredentialStore::new(),
    };
    let downloads = model_registry::downloads::DownloadManager::new(db.clone(), http.clone(), concurrency);
    downloads.set_hf_token(credentials.get(state::CRED_HF_TOKEN).ok().flatten());
    let state = AppState {
        runtimes: inference::RuntimeManager::new(paths.runtimes_dir.clone(), http.clone()),
        engine: inference::Engine::new(),
        credentials,
        downloads,
        api: local_api::ApiServer::default(),
        mcp: mcp_client::McpManager::new(),
        envs: Default::default(),
        gen_gate: Mutex::new((None, Arc::new(tokio::sync::Semaphore::new(1)))),
        tasks: Mutex::new(Default::default()),
        tasks_starting: Mutex::new(Default::default()),
        subagent_stops: Mutex::new(Default::default()),
        pending_attachments: Mutex::new(Default::default()),
        approvals: Mutex::new(Default::default()),
        installs: Mutex::new(Default::default()),
        generations: Mutex::new(Default::default()),
        token_cache: Mutex::new(Default::default()),
        monitor,
        http,
        db,
        paths,
    };
    // First run: register LM Studio's models in place (read-only, no copies).
    if state.db.get_setting::<bool>("library.lmstudio_scanned").await?.is_none() {
        if let Some(lm) = &state.paths.lmstudio_models_dir {
            match model_registry::import_path(&state.db, lm, "lmstudio", Some(lm)).await {
                Ok(r) => tracing::info!("LM Studio scan: {} models, {} invalid", r.registered.len(), r.invalid.len()),
                Err(e) => tracing::warn!("LM Studio scan failed: {e}"),
            }
        }
        state.db.set_setting("library.lmstudio_scanned", &true).await?;
    }
    model_registry::refresh_status(&state.db).await?;
    Ok(state)
}

fn init_logging(paths: &app_core::AppPaths) {
    use tracing_subscriber::EnvFilter;
    let log_path = paths.logs_dir.join("app.log");
    let file = std::fs::OpenOptions::new().create(true).append(true).open(&log_path);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    match file {
        Ok(f) => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(Mutex::new(RedactingWriter(f)))
                .try_init();
        }
        Err(_) => {
            let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        }
    }
}

/// Log writer that runs every line through secret redaction before writing.
struct RedactingWriter(std::fs::File);

impl std::io::Write for RedactingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        self.0.write_all(security::redact(&text).as_bytes())?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

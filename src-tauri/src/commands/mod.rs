pub mod api;
pub mod chat;
pub mod discover;
pub mod engine;
pub mod exec;
pub mod instructions;
pub mod mcp;
pub mod models;
pub mod runtimes;
pub mod system;
pub mod tasks;

use app_core::AppResult;

pub type Cmd<T> = AppResult<T>;

//! Shared domain types for Local AI Studio: application paths, the error type
//! returned across crate and IPC boundaries, and id/time helpers.

pub mod error;
pub mod paths;
pub mod storage;

pub use error::{AppError, AppResult};
pub use paths::AppPaths;

/// New random id (UUID v4, hyphenated string). Used for every persisted entity.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Current UTC time as RFC 3339 text, the format stored in SQLite.
pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

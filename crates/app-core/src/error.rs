use serde::Serialize;

/// Error type shared by all crates. It serializes to `{ code, message }` so the
/// frontend can show a readable message and branch on a stable code.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Conflict(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database error: {0}")]
    Database(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

impl AppError {
    pub fn code(&self) -> &'static str {
        match self {
            AppError::NotFound(_) => "not_found",
            AppError::InvalidInput(_) => "invalid_input",
            AppError::Conflict(_) => "conflict",
            AppError::Io(_) => "io",
            AppError::Database(_) => "database",
            AppError::Network(_) => "network",
            AppError::Runtime(_) => "runtime",
            AppError::OutOfMemory(_) => "out_of_memory",
            AppError::PermissionDenied(_) => "permission_denied",
            AppError::Cancelled => "cancelled",
            AppError::Other(_) => "other",
        }
    }
}

impl Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("AppError", 2)?;
        st.serialize_field("code", self.code())?;
        st.serialize_field("message", &self.to_string())?;
        st.end()
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::InvalidInput(format!("JSON: {e}"))
    }
}

pub type AppResult<T> = Result<T, AppError>;

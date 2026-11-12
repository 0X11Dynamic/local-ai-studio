//! Secrets live in Windows Credential Manager (DPAPI-protected), never in SQLite
//! or logs. Callers keep only an opaque reference string such as `mcp/<id>`.

use app_core::{AppError, AppResult};

const SERVICE: &str = "LocalAIStudio";

#[derive(Debug, Clone, Default)]
pub struct CredentialStore {
    service: String,
}

impl CredentialStore {
    pub fn new() -> Self {
        Self { service: SERVICE.into() }
    }

    /// Separate namespace, used by tests so they never touch real entries.
    pub fn with_service(service: &str) -> Self {
        Self { service: service.into() }
    }

    fn entry(&self, reference: &str) -> AppResult<keyring::Entry> {
        if reference.is_empty() || reference.len() > 200 {
            return Err(AppError::InvalidInput("invalid credential reference".into()));
        }
        keyring::Entry::new(&self.service, reference)
            .map_err(|e| AppError::Other(format!("credential store: {e}")))
    }

    pub fn set(&self, reference: &str, secret: &str) -> AppResult<()> {
        self.entry(reference)?
            .set_password(secret)
            .map_err(|e| AppError::Other(format!("credential store write failed: {e}")))
    }

    /// Returns `None` when no secret is stored under this reference.
    pub fn get(&self, reference: &str) -> AppResult<Option<String>> {
        match self.entry(reference)?.get_password() {
            Ok(s) => Ok(Some(s)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(AppError::Other(format!("credential store read failed: {e}"))),
        }
    }

    pub fn delete(&self, reference: &str) -> AppResult<()> {
        match self.entry(reference)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(AppError::Other(format!("credential store delete failed: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Touches the real Windows Credential Manager under a test-only service name
    /// and removes the entry afterwards.
    #[test]
    fn round_trip_in_windows_credential_manager() {
        let store = CredentialStore::with_service("LocalAIStudio-test");
        let r = format!("test/{}", app_core::new_id());
        assert_eq!(store.get(&r).unwrap(), None);
        store.set(&r, "s3cret-value").unwrap();
        assert_eq!(store.get(&r).unwrap().as_deref(), Some("s3cret-value"));
        store.delete(&r).unwrap();
        assert_eq!(store.get(&r).unwrap(), None);
    }
}

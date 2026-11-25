//! This install's long-term identity: an X25519 static key pair used as the
//! Noise static key. The private key is kept in Windows Credential Manager
//! (DPAPI), never in SQLite or logs; peers are identified by their public key.

use base64::Engine;
use sha2::{Digest, Sha256};

use crate::{LinkError, LinkResult};

/// Credential Manager reference for the identity key pair.
pub const CRED_IDENTITY: &str = "linking/identity";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Clone)]
pub struct Identity {
    private: [u8; 32],
    public: [u8; 32],
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Identity").field("public", &key_to_b64(&self.public)).finish_non_exhaustive()
    }
}

impl Identity {
    pub fn generate() -> LinkResult<Self> {
        let kp = snow::Builder::new(crate::handshake::link_params()).generate_keypair()?;
        Ok(Self { private: to32(&kp.private)?, public: to32(&kp.public)? })
    }

    pub fn public(&self) -> &[u8; 32] {
        &self.public
    }

    pub(crate) fn private(&self) -> &[u8; 32] {
        &self.private
    }

    /// `v1:<private b64>:<public b64>` (what the credential store holds).
    pub fn to_secret_string(&self) -> String {
        format!("v1:{}:{}", B64.encode(self.private), B64.encode(self.public))
    }

    pub fn from_secret_string(s: &str) -> LinkResult<Self> {
        let mut it = s.trim().split(':');
        match (it.next(), it.next(), it.next(), it.next()) {
            (Some("v1"), Some(p), Some(q), None) => {
                let private = to32(&B64.decode(p).map_err(|_| bad())?)?;
                let public = to32(&B64.decode(q).map_err(|_| bad())?)?;
                Ok(Self { private, public })
            }
            _ => Err(bad()),
        }
    }

    /// Load this install's identity from the credential store, creating and
    /// saving a new one the first time.
    pub fn load_or_create(store: &security::CredentialStore) -> app_core::AppResult<Self> {
        if let Some(s) = store.get(CRED_IDENTITY)? {
            match Self::from_secret_string(&s) {
                Ok(id) => return Ok(id),
                Err(e) => tracing::warn!("stored link identity is unreadable ({e}); making a new one (linked PCs must link again)"),
            }
        }
        let id = Self::generate().map_err(app_core::AppError::from)?;
        store.set(CRED_IDENTITY, &id.to_secret_string())?;
        Ok(id)
    }
}

fn bad() -> LinkError {
    LinkError::Protocol("malformed identity key".into())
}

pub(crate) fn to32(b: &[u8]) -> LinkResult<[u8; 32]> {
    b.try_into().map_err(|_| LinkError::Protocol(format!("expected a 32-byte key, got {} bytes", b.len())))
}

pub fn key_to_b64(k: &[u8]) -> String {
    B64.encode(k)
}

pub fn key_from_b64(s: &str) -> LinkResult<[u8; 32]> {
    to32(&B64.decode(s.trim()).map_err(|_| bad())?)
}

/// Short human-readable fingerprint of a public key, for comparing on both
/// screens: first 8 bytes of SHA-256, as `XXXX-XXXX-XXXX-XXXX`.
pub fn fingerprint(public: &[u8]) -> String {
    let h = Sha256::digest(public);
    let hex: String = h[..8].iter().map(|b| format!("{b:02X}")).collect();
    hex.as_bytes().chunks(4).map(|c| std::str::from_utf8(c).unwrap()).collect::<Vec<_>>().join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_string_round_trips_and_rejects_junk() {
        let id = Identity::generate().unwrap();
        let back = Identity::from_secret_string(&id.to_secret_string()).unwrap();
        assert_eq!(back.public(), id.public());
        assert_eq!(back.private(), id.private());
        assert!(Identity::from_secret_string("v1:abc").is_err());
        assert!(Identity::from_secret_string("v2:AAAA:AAAA").is_err());
        assert!(!format!("{id:?}").contains(&B64.encode(id.private())), "Debug must not print the private key");
        let fp = fingerprint(id.public());
        assert_eq!(fp.len(), 19);
        assert_eq!(fp, fingerprint(back.public()));
        assert_ne!(fp, fingerprint(Identity::generate().unwrap().public()));
    }

    /// Uses a test-only Credential Manager namespace and removes the entry.
    #[test]
    fn identity_is_created_once_and_kept_in_the_credential_store() {
        let store = security::CredentialStore::with_service(&format!("LocalAIStudio-test-link-{}", app_core::new_id()));
        let a = Identity::load_or_create(&store).unwrap();
        let b = Identity::load_or_create(&store).unwrap();
        assert_eq!(a.public(), b.public(), "the second load reads the stored key");
        store.delete(CRED_IDENTITY).unwrap();
    }
}

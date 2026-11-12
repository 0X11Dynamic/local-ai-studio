//! Security primitives: OS credential vault access, secret redaction, and the
//! tool permission policy.

pub mod credentials;
pub mod policy;
pub mod redact;

pub use credentials::CredentialStore;
pub use redact::redact;

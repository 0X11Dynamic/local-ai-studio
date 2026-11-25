//! Linked PCs ("combined computer", phase 2).
//!
//! One PC (the helper) lends its GPU/CPU to another PC (the main PC) by
//! running llama.cpp's `ggml-rpc-server` bound to 127.0.0.1. That server has
//! no authentication or encryption, so it is only ever reached through this
//! crate's tunnel:
//!
//! - **Identity** ([`identity`]): one X25519 static key pair per install; the
//!   private key lives in Windows Credential Manager.
//! - **Pairing** ([`client::pair`], [`helper`]): the helper shows a one-time
//!   6-digit code ([`code`]). SPAKE2 turns the code into a strong 32-byte key
//!   (one online guess per attempt, no offline guessing); that key is the PSK
//!   of a `Noise_XXpsk3_25519_ChaChaPoly_BLAKE2s` handshake, which exchanges and
//!   authenticates the two static keys. Both sides then pin each other's key.
//! - **Tunnel** ([`client::Control`], [`client::RpcBridge`]): every later
//!   connection is `Noise_XX_25519_ChaChaPoly_BLAKE2s`; each side checks the
//!   other's static key against its pinned keys before a single byte is
//!   forwarded. The main PC exposes 127.0.0.1:<random> to llama-server
//!   (`--rpc`); the helper forwards only to its own local rpc-server.
//! - **Reachability** ([`reach`]): UPnP port mapping, CGNAT detection.
//! - **Split maths** ([`split`]): `--list-devices` parsing, `--tensor-split`,
//!   the one-round-trip-per-token estimate.

pub mod cache;
pub mod client;
pub mod code;
pub mod handshake;
pub mod helper;
pub mod identity;
pub mod reach;
pub mod rpc_guard;
pub mod rpc_tap;
pub mod split;
pub mod wire;

use serde::{Deserialize, Serialize};

pub use client::{Control, Paired, RpcBridge, pair};
pub use code::{CodeSlot, CodeStatus};
pub use helper::{HelperHooks, HelperServer, SessionEvent};
pub use identity::{Identity, fingerprint};
pub use rpc_guard::{DevKind, PeerBudget, Refusal};

/// Default TCP port the helper listens on for linking and the tunnel.
///
/// 47070 sits below Windows' dynamic port range (49152-65535), where Hyper-V,
/// WSL, Docker and VirtualBox networking reserve blocks of ports after a
/// reboot (`netsh interface ipv4 show excludedportrange protocol=tcp`), and it
/// is not registered to well-known software (IANA lists no service there).
pub const DEFAULT_PORT: u16 = 47070;
/// The default of earlier builds (inside that dynamic range). Stored settings
/// holding it are moved to [`DEFAULT_PORT`], and the pairing form still tries
/// it when only an IP is typed.
pub const LEGACY_PORT: u16 = 50070;
/// When the chosen port can't be bound, the helper tries these in order
/// (all below the dynamic range, next to the default).
pub const FALLBACK_PORTS: std::ops::RangeInclusive<u16> = 47071..=47099;
/// Wire protocol version (in the plaintext hello).
pub const PROTOCOL_VERSION: u8 = 1;

/// What one PC tells the other about itself (exchanged at pairing and on
/// every control connection). All fields optional on the wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PeerInfo {
    pub name: String,
    pub app_version: String,
    /// This PC's link port, so the other PC can dial it later (if it helps).
    pub listen_port: Option<u16>,
    /// "Let this PC help" is on and its rpc-server is running.
    pub helping: bool,
    /// llama.cpp build its rpc-server runs, e.g. "b12250". Must match the
    /// main PC's llama-server build.
    pub runtime_tag: Option<String>,
    pub runtime_variant: Option<String>,
    /// Devices its rpc-server offers.
    pub devices: Vec<DeviceInfo>,
    /// Most memory (MiB) the helper wants to lend in total. ggml-rpc-server
    /// has no memory limit flag, so the main PC honours it when it computes
    /// `--tensor-split`.
    pub ram_cap_mib: Option<u64>,
    /// Measured RAM read bandwidth of this PC (GB/s), for speed estimates
    /// when it lends its CPU.
    pub ram_bandwidth_gbs: Option<f64>,
    /// What this helper lets the PC it is talking to use (per linked PC,
    /// set by its owner). None from helpers older than per-PC limits.
    pub share: Option<ShareLimits>,
    /// The helper enforces `share` itself, in the tunnel to its rpc-server
    /// (a linked PC can't allocate more, whatever version it runs). False
    /// from older helpers: only `ram_cap_mib` exists there, applied by the main PC.
    pub share_enforced: bool,
}

/// What a helper lends one linked PC (effective values, defaults resolved).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShareLimits {
    /// Most GPU memory (MiB) over every shared GPU; 0 = none.
    pub gpu_mib: u64,
    /// Most RAM (MiB) for the shared CPU device; 0 = none (or CPU not shared).
    pub ram_mib: u64,
    /// CPU threads its rpc-server runs with for that PC.
    pub threads: u32,
    /// Sharing with that PC is paused: new sessions are refused (an open one is never cut).
    pub paused: bool,
}

impl ShareLimits {
    /// Cap in bytes for one kind of device memory.
    pub fn cap_bytes(&self, kind: DevKind) -> u64 {
        let mib = match kind {
            DevKind::Gpu => self.gpu_mib,
            DevKind::Ram => self.ram_mib,
        };
        mib.saturating_mul(1 << 20)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum LinkError {
    #[error("the linking code was wrong ({} before the code is locked)", tries_left(.remaining))]
    WrongCode { remaining: u32 },
    #[error("too many wrong codes: this code is locked. Make a new code on the other PC")]
    CodeLocked,
    #[error("the other PC isn't showing a linking code (it expired or was already used). Make a new code there")]
    NoCode,
    #[error("the other PC is busy with another linking attempt; try again in a few seconds")]
    Busy,
    #[error("the other PC's identity key is not the one pinned when you linked. Someone may be intercepting the connection, or the other PC was reinstalled. Unlink it and link again with a new code")]
    KeyMismatch,
    #[error("the other PC no longer trusts this PC (it was unlinked there). Unlink it here and link again with a new code")]
    NotLinked,
    #[error("the other PC has \"Let this PC help\" turned off")]
    HelperOff,
    #[error("{0}")]
    Refused(String),
    #[error("linking protocol error: {0}")]
    Protocol(String),
    #[error("connection timed out")]
    Timeout,
    /// Connecting got "connection refused": the PC is there but nothing listens on the port.
    #[error("nothing is listening on {0}: Local AI Studio isn't running there, or it isn't showing a linking code or helping right now")]
    NotListening(String),
    /// No answer at all within the connect timeout.
    #[error("no answer from {0}: the PC is off or asleep, the address or port is wrong, or a firewall or router (port forward) blocks the port")]
    Unreachable(String),
    /// Accepted, then closed before the handshake finished.
    #[error("{0} accepted the connection but closed it straight away: something other than Local AI Studio answers on that port (for example a router or VirtualBox port forward with nothing behind it), or the app there just stopped listening")]
    ClosedEarly(String),
    /// An open connection to the other PC closed.
    #[error("the connection to the other PC closed ({0})")]
    Disconnected(String),
    #[error("network error: {0}")]
    Io(#[from] std::io::Error),
}

impl LinkError {
    /// Stable code for the wire and the UI.
    pub fn code(&self) -> &'static str {
        match self {
            LinkError::WrongCode { .. } => "wrong_code",
            LinkError::CodeLocked => "code_locked",
            LinkError::NoCode => "no_code",
            LinkError::Busy => "busy",
            LinkError::KeyMismatch => "key_mismatch",
            LinkError::NotLinked => "not_linked",
            LinkError::HelperOff => "helper_off",
            LinkError::Refused(_) => "refused",
            LinkError::Protocol(_) => "protocol",
            LinkError::Timeout => "timeout",
            LinkError::NotListening(_) => "not_listening",
            LinkError::Unreachable(_) => "unreachable",
            LinkError::ClosedEarly(_) => "closed_early",
            LinkError::Disconnected(_) => "disconnected",
            LinkError::Io(_) => "io",
        }
    }

    /// Rebuild an error the other side reported with [`LinkError::code`].
    pub fn from_wire(code: &str, message: &str) -> Self {
        match code {
            "wrong_code" => LinkError::WrongCode { remaining: message.parse().unwrap_or(0) },
            "code_locked" => LinkError::CodeLocked,
            "no_code" => LinkError::NoCode,
            "busy" => LinkError::Busy,
            "not_linked" => LinkError::NotLinked,
            "helper_off" => LinkError::HelperOff,
            "protocol" => LinkError::Protocol(message.into()),
            _ => LinkError::Refused(message.into()),
        }
    }

    /// The message the other side gets for this error (see `from_wire`).
    pub fn wire_message(&self) -> String {
        match self {
            LinkError::WrongCode { remaining } => remaining.to_string(),
            other => other.to_string(),
        }
    }
}

impl From<snow::Error> for LinkError {
    fn from(e: snow::Error) -> Self {
        LinkError::Protocol(format!("noise: {e}"))
    }
}

impl From<LinkError> for app_core::AppError {
    fn from(e: LinkError) -> Self {
        match e {
            LinkError::Io(io) => app_core::AppError::Network(io.to_string()),
            LinkError::Timeout => app_core::AppError::Network("connection timed out".into()),
            LinkError::WrongCode { .. } | LinkError::CodeLocked | LinkError::NoCode | LinkError::KeyMismatch | LinkError::NotLinked => {
                app_core::AppError::PermissionDenied(e.to_string())
            }
            other => app_core::AppError::Network(other.to_string()),
        }
    }
}

pub type LinkResult<T> = Result<T, LinkError>;

fn tries_left(n: &u32) -> String {
    if *n == 1 { "1 try left".into() } else { format!("{n} tries left") }
}

impl LinkError {
    /// The connection dropped or never got through (as opposed to a refusal
    /// the other PC sent on purpose).
    pub fn is_connection_lost(&self) -> bool {
        matches!(
            self,
            LinkError::Io(_) | LinkError::Timeout | LinkError::NotListening(_) | LinkError::Unreachable(_) | LinkError::ClosedEarly(_) | LinkError::Disconnected(_)
        )
    }
}

/// Run `fut` with a timeout, mapping expiry to [`LinkError::Timeout`].
pub(crate) async fn timed<T>(d: std::time::Duration, fut: impl std::future::Future<Output = LinkResult<T>>) -> LinkResult<T> {
    tokio::time::timeout(d, fut).await.map_err(|_| LinkError::Timeout)?
}

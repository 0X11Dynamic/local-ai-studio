//! The two handshakes.
//!
//! **Pairing** (joiner = the PC typing the code, helper = the PC showing it):
//! 1. J -> H: hello(Pair), then SPAKE2 message A (password = the 6 digits).
//! 2. H -> J: status OK + SPAKE2 message B (or an error: no code, locked, busy).
//! 3. Both derive a 32-byte key K from SPAKE2. K is the PSK of
//!    `Noise_XXpsk3_25519_ChaChaPoly_BLAKE2s`; the prologue binds the hello
//!    and both SPAKE2 messages. J sends its info in message 3, which is
//!    encrypted under keys that depend on K, so only a holder of the code can
//!    produce or read it.
//! 4. H decrypts message 3 (failure = wrong code, counted), records the
//!    link, and answers with its own info encrypted in the session. J trusts
//!    H's static key only after that answer authenticates.
//!
//! A man in the middle without the code must run SPAKE2 with a guessed
//! password on each side; the keys differ, so the Noise handshake fails and
//! each attempt costs one of the helper's 5 tries. SPAKE2 gives no way to
//! test further guesses offline.
//!
//! **Linking** (every later connection): `Noise_XX_25519_ChaChaPoly_BLAKE2s`
//! with the pinned static keys. The initiator checks the responder's key
//! after message 2, before it reveals its own identity in message 3; the
//! responder checks the initiator's key after message 3 and answers with an
//! encrypted accept or refuse before forwarding anything.

use serde_json::Value;
use spake2::{Ed25519Group, Identity as SpakeId, Password, Spake2};
use tokio::net::TcpStream;

use crate::code::{CODE_DIGITS, CodeSlot, normalize_code};
use crate::identity::{Identity, to32};
use crate::wire::{
    Frame, MAX_FRAME, Mode, SecureReader, SecureWriter, hello, parse_status, read_frame, secure_pair, status_err, status_ok, write_frame,
};
use crate::{LinkError, LinkResult, PeerInfo};

const LINK_PATTERN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
const PAIR_PATTERN: &str = "Noise_XXpsk3_25519_ChaChaPoly_BLAKE2s";
const LINK_PROLOGUE: &[u8] = b"LocalAIStudio link v1";
const PAIR_PROLOGUE: &[u8] = b"LocalAIStudio pair v1";
const SPAKE_JOINER: &[u8] = b"LocalAIStudio joiner";
const SPAKE_HELPER: &[u8] = b"LocalAIStudio helper";

pub(crate) fn link_params() -> snow::params::NoiseParams {
    LINK_PATTERN.parse().expect("valid noise pattern")
}

fn pair_params() -> snow::params::NoiseParams {
    PAIR_PATTERN.parse().expect("valid noise pattern")
}

impl From<serde_json::Error> for LinkError {
    fn from(e: serde_json::Error) -> Self {
        LinkError::Protocol(format!("bad message: {e}"))
    }
}

/// An authenticated, encrypted connection.
pub struct Session {
    pub reader: SecureReader,
    pub writer: SecureWriter,
    /// The other side's (verified) static public key.
    pub remote_key: [u8; 32],
    /// Responder side: what the initiator asked for (message 3 payload).
    pub request: Value,
}

fn spake_err(_: spake2::Error) -> LinkError {
    LinkError::Protocol("bad SPAKE2 message".into())
}

/// Read a handshake reply; if the other side sent a status error instead,
/// return that error.
fn read_hs(hs: &mut snow::HandshakeState, msg: &[u8], out: &mut [u8]) -> LinkResult<usize> {
    match hs.read_message(msg, out) {
        Ok(n) => Ok(n),
        Err(e) => match parse_status(msg) {
            Err(se @ (LinkError::Refused(_) | LinkError::NoCode | LinkError::CodeLocked | LinkError::Busy | LinkError::HelperOff | LinkError::NotLinked)) => Err(se),
            _ => Err(e.into()),
        },
    }
}

/// Joiner side of pairing. Returns the helper's info and its static key.
pub async fn pair_initiate(stream: TcpStream, id: &Identity, code: &str, me: &PeerInfo) -> LinkResult<(PeerInfo, [u8; 32])> {
    let code = normalize_code(code);
    if code.len() != CODE_DIGITS {
        return Err(LinkError::Refused(format!("the linking code has {CODE_DIGITS} digits")));
    }
    stream.set_nodelay(true)?;
    let (mut r, mut w) = stream.into_split();
    let h = hello(Mode::Pair);
    write_frame(&mut w, &h).await?;
    let (spake, msg_a) = Spake2::<Ed25519Group>::start_a(&Password::new(code.as_bytes()), &SpakeId::new(SPAKE_JOINER), &SpakeId::new(SPAKE_HELPER));
    write_frame(&mut w, &msg_a).await?;
    let mut buf = Vec::new();
    read_frame(&mut r, &mut buf).await?;
    let msg_b = parse_status(&buf)?.to_vec();
    let psk = to32(&spake.finish(&msg_b).map_err(spake_err)?)?;
    let prologue = [PAIR_PROLOGUE, &h, &msg_a, &msg_b].concat();
    let mut hs = snow::Builder::new(pair_params())
        .local_private_key(id.private())?
        .psk(3, &psk)?
        .prologue(&prologue)?
        .build_initiator()?;
    let mut out = vec![0u8; MAX_FRAME];
    let mut inb = vec![0u8; MAX_FRAME];
    let n = hs.write_message(&[], &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    read_frame(&mut r, &mut buf).await?;
    read_hs(&mut hs, &buf, &mut inb)?;
    let n = hs.write_message(&serde_json::to_vec(me)?, &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    let remote = to32(hs.get_remote_static().ok_or_else(|| LinkError::Protocol("no remote key".into()))?)?;
    let t = hs.into_stateless_transport_mode()?;
    read_frame(&mut r, &mut buf).await?;
    let ct = parse_status(&buf)?;
    let n = t
        .read_message(0, ct, &mut inb)
        .map_err(|_| LinkError::Protocol("the other PC's answer failed authentication".into()))?;
    let info: PeerInfo = serde_json::from_slice(&inb[..n])?;
    Ok((info, remote))
}

/// Helper side of pairing (after the hello frame `h` was read from
/// `stream`). `on_success` runs once the code is proven: it pins the joiner
/// and returns the helper's info for the answer.
pub async fn pair_respond(
    stream: TcpStream,
    id: &Identity,
    h: &[u8],
    slot: &CodeSlot,
    on_success: impl FnOnce([u8; 32], &PeerInfo) -> PeerInfo,
) -> LinkResult<([u8; 32], PeerInfo)> {
    stream.set_nodelay(true)?;
    let (mut r, mut w) = stream.into_split();
    let mut buf = Vec::new();
    read_frame(&mut r, &mut buf).await?;
    let msg_a = buf.clone();
    let attempt = match slot.begin() {
        Ok(a) => a,
        Err(e) => {
            let _ = write_frame(&mut w, &status_err(&e)).await;
            return Err(e);
        }
    };
    // From here on, any early return drops `attempt`, which counts a failure.
    let (spake, msg_b) =
        Spake2::<Ed25519Group>::start_b(&Password::new(attempt.code().as_bytes()), &SpakeId::new(SPAKE_JOINER), &SpakeId::new(SPAKE_HELPER));
    write_frame(&mut w, &status_ok(&msg_b)).await?;
    let psk = to32(&spake.finish(&msg_a).map_err(spake_err)?)?;
    let prologue = [PAIR_PROLOGUE, h, &msg_a, &msg_b].concat();
    let mut hs = snow::Builder::new(pair_params())
        .local_private_key(id.private())?
        .psk(3, &psk)?
        .prologue(&prologue)?
        .build_responder()?;
    let mut out = vec![0u8; MAX_FRAME];
    let mut inb = vec![0u8; MAX_FRAME];
    read_frame(&mut r, &mut buf).await?;
    hs.read_message(&buf, &mut inb)?;
    let n = hs.write_message(&[], &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    read_frame(&mut r, &mut buf).await?;
    let n = match hs.read_message(&buf, &mut inb) {
        Ok(n) => n,
        Err(_) => {
            // Wrong code (or tampering): the PSKs differ.
            let e = attempt.fail();
            let _ = write_frame(&mut w, &status_err(&e)).await;
            return Err(e);
        }
    };
    let info: PeerInfo = serde_json::from_slice(&inb[..n])?;
    let remote = to32(hs.get_remote_static().ok_or_else(|| LinkError::Protocol("no remote key".into()))?)?;
    attempt.succeed();
    let reply = on_success(remote, &info);
    let t = hs.into_stateless_transport_mode()?;
    let n = t.write_message(0, &serde_json::to_vec(&reply)?, &mut out)?;
    write_frame(&mut w, &status_ok(&out[..n])).await?;
    Ok((remote, info))
}

/// Initiator side of a link connection to a pinned peer. Fails with
/// [`LinkError::KeyMismatch`] (without revealing this PC's identity) when the
/// responder's key is not `expected`. Call [`read_accept`] next.
pub async fn link_initiate(stream: TcpStream, id: &Identity, mode: Mode, expected: [u8; 32], request: &Value) -> LinkResult<Session> {
    stream.set_nodelay(true)?;
    let (mut r, mut w) = stream.into_split();
    let h = hello(mode);
    write_frame(&mut w, &h).await?;
    let prologue = [LINK_PROLOGUE, &h].concat();
    let mut hs = snow::Builder::new(link_params()).local_private_key(id.private())?.prologue(&prologue)?.build_initiator()?;
    let mut out = vec![0u8; MAX_FRAME];
    let mut inb = vec![0u8; MAX_FRAME];
    let mut buf = Vec::new();
    let n = hs.write_message(&[], &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    read_frame(&mut r, &mut buf).await?;
    read_hs(&mut hs, &buf, &mut inb)?;
    let remote = to32(hs.get_remote_static().ok_or_else(|| LinkError::Protocol("no remote key".into()))?)?;
    if remote != expected {
        return Err(LinkError::KeyMismatch);
    }
    let n = hs.write_message(&serde_json::to_vec(request)?, &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    let t = hs.into_stateless_transport_mode()?;
    let (reader, writer) = secure_pair(r, w, t, 0, 0);
    Ok(Session { reader, writer, remote_key: remote, request: Value::Null })
}

/// Responder side of a link connection (after the hello `h` was read). The
/// caller must check `remote_key` and then [`send_accept`] or [`send_refuse`].
pub async fn link_respond(stream: TcpStream, id: &Identity, h: &[u8]) -> LinkResult<Session> {
    stream.set_nodelay(true)?;
    let (mut r, mut w) = stream.into_split();
    let prologue = [LINK_PROLOGUE, h].concat();
    let mut hs = snow::Builder::new(link_params()).local_private_key(id.private())?.prologue(&prologue)?.build_responder()?;
    let mut out = vec![0u8; MAX_FRAME];
    let mut inb = vec![0u8; MAX_FRAME];
    let mut buf = Vec::new();
    read_frame(&mut r, &mut buf).await?;
    hs.read_message(&buf, &mut inb)?;
    let n = hs.write_message(&[], &mut out)?;
    write_frame(&mut w, &out[..n]).await?;
    read_frame(&mut r, &mut buf).await?;
    let n = hs.read_message(&buf, &mut inb)?;
    let request: Value = if n == 0 { Value::Null } else { serde_json::from_slice(&inb[..n])? };
    let remote = to32(hs.get_remote_static().ok_or_else(|| LinkError::Protocol("no remote key".into()))?)?;
    let t = hs.into_stateless_transport_mode()?;
    let (reader, writer) = secure_pair(r, w, t, 0, 0);
    Ok(Session { reader, writer, remote_key: remote, request })
}

pub async fn send_accept(w: &mut SecureWriter, extra: Value) -> LinkResult<()> {
    let mut v = serde_json::json!({ "ok": true });
    if let (Some(m), Value::Object(e)) = (v.as_object_mut(), extra) {
        m.extend(e);
    }
    w.send(&Frame::Json(v)).await
}

pub async fn send_refuse(w: &mut SecureWriter, e: &LinkError) -> LinkResult<()> {
    w.send(&Frame::Json(serde_json::json!({ "ok": false, "code": e.code(), "message": e.wire_message() }))).await
}

/// The responder's answer to a link connection.
pub async fn read_accept(r: &mut SecureReader) -> LinkResult<Value> {
    match r.recv().await? {
        Frame::Json(v) if v["ok"] == Value::Bool(true) => Ok(v),
        Frame::Json(v) => Err(LinkError::from_wire(v["code"].as_str().unwrap_or(""), v["message"].as_str().unwrap_or("refused"))),
        _ => Err(LinkError::Protocol("unexpected answer".into())),
    }
}

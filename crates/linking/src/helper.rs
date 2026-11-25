//! The helper's listener: pairing (while a code is shown), the control
//! channel, and RPC streams forwarded to the local `ggml-rpc-server`.
//!
//! Nothing reaches rpc-server unless the connecting PC proved a pinned key
//! (Noise XX) and the app says it is helping (`rpc_target`).

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::code::CodeSlot;
use crate::handshake::{self, Session, send_accept, send_refuse};
use crate::identity::{Identity, key_to_b64};
use crate::rpc_guard::{self, Claim, DevKind, Fwd, GuardError, Refusal};
use crate::wire::{Frame, Mode, parse_hello, read_frame, status_err, write_frame};
use crate::{LinkError, LinkResult, PeerInfo, timed};

/// Binding failed because the port is taken or reserved (Windows reports a
/// reserved port as WSAEACCES 10013, an used one as AddrInUse / 10048).
pub fn is_port_unavailable(e: &std::io::Error) -> bool {
    use std::io::ErrorKind as K;
    matches!(e.kind(), K::AddrInUse | K::PermissionDenied) || matches!(e.raw_os_error(), Some(10013 | 10048))
}

/// A listener bound by [`bind_with_fallback`].
pub struct Bound<T> {
    pub port: u16,
    pub value: T,
    /// Set when `port` isn't the one asked for: says why, in the user's words.
    pub notice: Option<String>,
}

/// Bind `preferred`; if it is reserved or in use, try the next free ports of
/// [`crate::FALLBACK_PORTS`]. Other errors fail at once. The error text names
/// the cause and the `netsh` check for reserved ranges.
pub async fn bind_with_fallback<T, F, Fut>(preferred: u16, mut try_bind: F) -> Result<Bound<T>, String>
where
    F: FnMut(u16) -> Fut,
    Fut: Future<Output = std::io::Result<T>>,
{
    let first = match try_bind(preferred).await {
        Ok(value) => return Ok(Bound { port: preferred, value, notice: None }),
        Err(e) if is_port_unavailable(&e) => e,
        Err(e) => return Err(format!("could not listen on port {preferred}: {e}")),
    };
    let why = format!(
        "Port {preferred} is reserved by Windows or already in use by another program ({first}). Windows can reserve whole port ranges (check with `netsh interface ipv4 show excludedportrange protocol=tcp`)."
    );
    for p in crate::FALLBACK_PORTS.filter(|p| *p != preferred) {
        match try_bind(p).await {
            Ok(value) => return Ok(Bound { port: p, value, notice: Some(format!("{why} Listening on port {p} instead: type that port on the other PC.")) }),
            Err(e) if is_port_unavailable(&e) => continue,
            Err(e) => return Err(format!("could not listen on port {p}: {e}")),
        }
    }
    Err(format!("{why} No free port from {} to {} either; pick another port in the helper settings.", crate::FALLBACK_PORTS.start(), crate::FALLBACK_PORTS.end()))
}

/// A boxed future for [`HelperHooks`] methods that may wait (restarting rpc-server).
pub type HookFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Where a linked PC's RPC stream goes, and the limits it runs under.
pub struct RpcTarget {
    pub addr: SocketAddr,
    /// The linked PC's budget (this stream already counted in it) and the
    /// kind of each rpc-server device, in its order. None = no limits (tests).
    pub limits: Option<GuardSpec>,
}

pub struct GuardSpec {
    pub claim: Claim,
    pub kinds: Arc<Vec<DevKind>>,
}

/// Unauthenticated connections must finish their handshake within this.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);
/// Most simultaneous connections (authenticated or not).
pub const MAX_CONNECTIONS: usize = 64;
/// A control channel with no frame for this long is closed (the main PC pings every 15 s).
pub const CONTROL_IDLE: Duration = Duration::from_secs(90);
/// Largest bandwidth test accepted.
pub const MAX_BW_TEST: u64 = 64 << 20;

/// What the helper needs from the app.
pub trait HelperHooks: Send + Sync + 'static {
    /// Name of the linked PC with this pinned key; None = not linked.
    fn peer_name(&self, key: &[u8; 32]) -> Option<String>;
    /// A pairing proved the code: pin `key`. `addr` is where it connected from.
    fn paired(&self, key: [u8; 32], info: &PeerInfo, addr: SocketAddr);
    /// The other PC unlinked this one: forget `key`.
    fn unlinked(&self, key: [u8; 32]);
    /// This PC's description (name, runtime build, devices, helping).
    fn info(&self) -> PeerInfo;
    /// This PC's description as the linked PC with this key sees it: its own
    /// sharing limits and the free memory left to it. None = not known yet
    /// (pairing).
    fn info_for(&self, _peer: Option<&[u8; 32]>) -> PeerInfo {
        self.info()
    }
    /// The local rpc-server to forward this linked PC's stream to, with its
    /// limits, or why it can't have one (helping off, paused, busy with
    /// another PC, limits can't be enforced).
    fn rpc_target<'a>(&'a self, key: &'a [u8; 32]) -> HookFuture<'a, Result<RpcTarget, LinkError>>;
    /// This PC refused something to a linked PC (over its limit, or another protocol).
    fn refused(&self, _key: &[u8; 32], _r: &Refusal) {}
    /// A linked PC opened or closed a connection.
    fn session(&self, _ev: SessionEvent) {}
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEvent {
    pub id: u64,
    pub peer_key: String,
    pub peer_name: String,
    pub remote: String,
    /// control | rpc
    pub mode: &'static str,
    pub open: bool,
}

pub struct HelperServer {
    addr: SocketAddr,
    cancel: CancellationToken,
    changed: tokio::sync::watch::Sender<u64>,
}

struct Ctx {
    identity: Arc<Identity>,
    codes: Arc<CodeSlot>,
    hooks: Arc<dyn HelperHooks>,
    cancel: CancellationToken,
    next_id: AtomicU64,
    /// Bumped when this PC's info changes (helping on/off, devices).
    changed: tokio::sync::watch::Receiver<u64>,
    /// Messages for one linked PC's control channel (a refusal), by key.
    pushes: tokio::sync::broadcast::Sender<([u8; 32], serde_json::Value)>,
}

impl HelperServer {
    pub async fn bind(addr: SocketAddr, identity: Arc<Identity>, codes: Arc<CodeSlot>, hooks: Arc<dyn HelperHooks>) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let cancel = CancellationToken::new();
        let (changed, changed_rx) = tokio::sync::watch::channel(0u64);
        let (pushes, _) = tokio::sync::broadcast::channel(16);
        let ctx = Arc::new(Ctx { identity, codes, hooks, cancel: cancel.clone(), next_id: AtomicU64::new(1), changed: changed_rx, pushes });
        let limit = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        tokio::spawn(async move {
            loop {
                let (stream, peer) = tokio::select! {
                    _ = ctx.cancel.cancelled() => break,
                    r = listener.accept() => match r {
                        Ok(x) => x,
                        Err(e) => {
                            tracing::warn!("link listener accept failed: {e}");
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            continue;
                        }
                    },
                };
                let Ok(permit) = limit.clone().try_acquire_owned() else {
                    tracing::warn!("link listener: too many connections; dropped one from {peer}");
                    continue;
                };
                let ctx = ctx.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    let c = ctx.cancel.clone();
                    tokio::select! {
                        _ = c.cancelled() => {}
                        r = handle(stream, peer, &ctx) => if let Err(e) = r {
                            tracing::info!("link connection from {peer} ended: {e}");
                        }
                    }
                });
            }
        });
        Ok(Self { addr: local, cancel, changed })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// This PC's info changed: push it to every open control channel, so
    /// linked PCs see "helping" go on or off at once.
    pub fn notify_changed(&self) {
        self.changed.send_modify(|v| *v += 1);
    }

    /// Stop listening and close every connection (including forwarded RPC streams).
    pub fn stop(&self) {
        self.cancel.cancel();
    }
}

impl Drop for HelperServer {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn handle(mut stream: TcpStream, peer: SocketAddr, ctx: &Ctx) -> LinkResult<()> {
    crate::wire::tune_socket(&stream);
    let mut h = Vec::new();
    timed(HANDSHAKE_TIMEOUT, async { read_frame(&mut stream, &mut h).await }).await?;
    let mode = match parse_hello(&h) {
        Ok(m) => m,
        Err(e) => {
            let _ = write_frame(&mut stream, &status_err(&e)).await;
            return Err(e);
        }
    };
    match mode {
        Mode::Pair => {
            let hooks = ctx.hooks.clone();
            let (key, info) = timed(
                HANDSHAKE_TIMEOUT,
                handshake::pair_respond(stream, &ctx.identity, &h, &ctx.codes, |key, info| {
                    hooks.paired(key, info, peer);
                    hooks.info_for(Some(&key))
                }),
            )
            .await?;
            tracing::info!("linked with \"{}\" ({peer}, key {})", info.name, crate::fingerprint(&key));
            Ok(())
        }
        Mode::Control | Mode::Rpc => {
            let mut sess = timed(HANDSHAKE_TIMEOUT, handshake::link_respond(stream, &ctx.identity, &h)).await?;
            let Some(name) = ctx.hooks.peer_name(&sess.remote_key) else {
                let _ = send_refuse(&mut sess.writer, &LinkError::NotLinked).await;
                return Err(LinkError::NotLinked);
            };
            let ev = |open| SessionEvent {
                id: 0,
                peer_key: key_to_b64(&sess.remote_key),
                peer_name: name.clone(),
                remote: peer.to_string(),
                mode: if mode == Mode::Rpc { "rpc" } else { "control" },
                open,
            };
            let id = ctx.next_id.fetch_add(1, Ordering::Relaxed);
            let (open_ev, close_ev) = (SessionEvent { id, ..ev(true) }, SessionEvent { id, ..ev(false) });
            if mode == Mode::Control {
                send_accept(&mut sess.writer, serde_json::json!({ "info": ctx.hooks.info_for(Some(&sess.remote_key)) })).await?;
                ctx.hooks.session(open_ev);
                let r = control_loop(sess, ctx).await;
                ctx.hooks.session(close_ev);
                return r;
            }
            // RPC: the same llama.cpp build on both PCs, and only while helping.
            let info = ctx.hooks.info_for(Some(&sess.remote_key));
            if let (Some(want), Some(have)) = (sess.request["runtime_tag"].as_str(), info.runtime_tag.as_deref())
                && want != have
            {
                let e = LinkError::Refused(version_message(want, have, &info.name));
                let _ = send_refuse(&mut sess.writer, &e).await;
                return Err(e);
            }
            let key = sess.remote_key;
            let target = match ctx.hooks.rpc_target(&key).await {
                Ok(t) => t,
                Err(e) => {
                    let _ = send_refuse(&mut sess.writer, &e).await;
                    return Err(e);
                }
            };
            let local = match tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(target.addr)).await {
                Ok(Ok(s)) => s,
                _ => {
                    let e = LinkError::Refused("the helper's rpc-server is not answering; turn \"Let this PC help\" off and on there".into());
                    let _ = send_refuse(&mut sess.writer, &e).await;
                    return Err(e);
                }
            };
            send_accept(&mut sess.writer, serde_json::json!({})).await?;
            let compress = sess.request["compress"].as_bool().unwrap_or(false);
            // The linked PC's limits, enforced on this side of the tunnel.
            let guard = target.limits.map(|g| {
                let (hooks, pushes) = (ctx.hooks.clone(), ctx.pushes.clone());
                let on_refuse: Arc<dyn Fn(&Refusal) + Send + Sync> = Arc::new(move |r: &Refusal| {
                    hooks.refused(&key, r);
                    let _ = pushes.send((key, serde_json::json!({ "t": "share_refused", "refusal": r })));
                });
                rpc_guard::guard(g.claim, g.kinds, Some(on_refuse))
            });
            ctx.hooks.session(open_ev);
            let r = pipe(sess, local, compress, None, None, guard).await.map(|_| ());
            ctx.hooks.session(close_ev);
            r
        }
    }
}

/// "<helper> runs llama.cpp b1 but this PC uses b2" in the user's words.
pub fn version_message(main_tag: &str, helper_tag: &str, helper_name: &str) -> String {
    format!(
        "\"{helper_name}\" runs llama.cpp {helper_tag} but this PC loads with {main_tag}. Both PCs need the same runtime build: install {main_tag} on \"{helper_name}\" (Runtimes page) or pick the {helper_tag} runtime here"
    )
}

async fn control_loop(sess: Session, ctx: &Ctx) -> LinkResult<()> {
    let Session { mut reader, writer, remote_key, .. } = sess;
    let writer = Arc::new(tokio::sync::Mutex::new(writer));
    // Push info updates while the channel is open (reads are not
    // cancellation-safe, so the pusher is its own task).
    let pusher = {
        let (w, mut rx, hooks, mut pushes) = (writer.clone(), ctx.changed.clone(), ctx.hooks.clone(), ctx.pushes.subscribe());
        tokio::spawn(async move {
            loop {
                let f = tokio::select! {
                    r = rx.changed() => match r {
                        Ok(()) => Frame::Json(serde_json::json!({ "t": "info_update", "info": hooks.info_for(Some(&remote_key)) })),
                        Err(_) => return,
                    },
                    r = pushes.recv() => match r {
                        Ok((k, v)) if k == remote_key => Frame::Json(v),
                        Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => return,
                    },
                };
                if w.lock().await.send(&f).await.is_err() {
                    return;
                }
            }
        })
    };
    // Stop the pusher however this ends (also when the server is stopped
    // and this future is dropped), so the socket really closes.
    struct AbortOnDrop(tokio::task::JoinHandle<()>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _pusher = AbortOnDrop(pusher);
    loop {
        let f = match tokio::time::timeout(CONTROL_IDLE, reader.recv()).await {
            Err(_) => return Ok(()),
            Ok(Err(LinkError::Io(e))) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Ok(r) => r?,
        };
        let reply = match f {
            Frame::Ping(n) => Some(Frame::Pong(n)),
            Frame::Close => return Ok(()),
            Frame::Json(v) => match v["t"].as_str() {
                Some("info") => Some(Frame::Json(serde_json::json!({ "t": "info", "info": ctx.hooks.info_for(Some(&remote_key)) }))),
                Some("bw") => {
                    let want = v["bytes"].as_u64().unwrap_or(0).min(MAX_BW_TEST);
                    let mut got = 0u64;
                    while got < want {
                        match reader.recv().await? {
                            Frame::Data(d) => got += d.len() as u64,
                            Frame::Close => break,
                            _ => {}
                        }
                    }
                    Some(Frame::Json(serde_json::json!({ "t": "bw_done", "bytes": got })))
                }
                Some("unlink") => {
                    ctx.hooks.unlinked(remote_key);
                    writer.lock().await.send(&Frame::Json(serde_json::json!({ "t": "unlinked" }))).await?;
                    return Ok(());
                }
                _ => Some(Frame::Json(serde_json::json!({ "t": "error", "message": "unknown request" }))),
            },
            Frame::Data(_) | Frame::Pong(_) => None,
        };
        if let Some(f) = reply {
            writer.lock().await.send(&f).await?;
        }
    }
}

/// Most bytes per tunnel frame on a forwarded stream. The receiver can only
/// pass a frame on once all of it has arrived, so every message waits for
/// about one frame's time on the link per direction: 16 KB instead of the
/// 64 KB maximum took the ggml-rpc prompt pattern (bench
/// `rpc_message_pattern_through_the_tunnel_is_close_to_direct`, 800 Mbit/s
/// relay) from 106-114% of direct (4 runs) to 103-112% (8 runs, median
/// about 108%), with bulk throughput on the link unchanged (101-102%).
pub const PIPE_READ: usize = 16 * 1024;

/// Copy bytes both ways between a session and a local socket until both
/// sides have closed. Returns (bytes sent into the tunnel, bytes received).
///
/// Each direction is its own task, so both can run at once on different
/// threads; within a direction the data is read straight into the frame
/// buffer and sealed in place (no queue, no per-frame allocation: measured
/// faster than a read/encrypt/write pipeline with channels). `live` counts
/// payload bytes as they move; `tap` (main PC side only) follows the
/// ggml-rpc stream for load progress ([`crate::rpc_tap`]); `guard` (helper
/// side only) enforces the linked PC's limits ([`crate::rpc_guard`]): it may
/// edit bytes in place, hold back a small frame, or answer a refused
/// allocation itself (sent from the local-to-tunnel task, in order).
pub(crate) async fn pipe(
    sess: Session,
    local: TcpStream,
    compress: bool,
    live: Option<(Arc<AtomicU64>, Arc<AtomicU64>)>,
    tap: Option<Arc<crate::rpc_tap::RpcCounters>>,
    guard: Option<(rpc_guard::Upstream, rpc_guard::Downstream)>,
) -> LinkResult<(u64, u64)> {
    let (live_up, live_down) = match live {
        Some((u, d)) => (Some(u), Some(d)),
        None => (None, None),
    };
    let (mut tap_up, mut tap_down) = match tap {
        Some(c) => {
            let (u, d) = crate::rpc_tap::tap(c);
            (Some(u), Some(d))
        }
        None => (None, None),
    };
    // Guard halves; refused allocations are answered through `inject`.
    let (mut guard_up, mut guard_down, inject_tx, mut inject_rx) = match guard {
        Some((u, d)) => {
            let (tx, rx) = tokio::sync::mpsc::channel::<Vec<u8>>(4);
            (Some(u), Some(d), Some(tx), Some(rx))
        }
        None => (None, None, None, None),
    };
    // ggml-rpc sends small command/size/payload writes and waits for the
    // answer: never let Nagle hold a segment back on any socket.
    crate::wire::tune_socket(&local);
    let (mut lr, mut lw) = local.into_split();
    let Session { mut reader, mut writer, .. } = sess;
    let mut set: tokio::task::JoinSet<LinkResult<(bool, u64)>> = tokio::task::JoinSet::new();
    set.spawn(async move {
        // buf[0] is the frame type; the socket reads into buf[1..].
        let mut buf = vec![0u8; PIPE_READ + 1];
        // A frame the guard rebuilt (out[0] is the frame type).
        let mut out = Vec::with_capacity(64);
        let mut gout = Vec::with_capacity(64);
        let mut total = 0u64;
        loop {
            let n = match &mut inject_rx {
                Some(rx) => tokio::select! {
                    biased;
                    Some(mut v) = rx.recv() => {
                        // A reply the guard makes itself (a refused allocation).
                        v.insert(0, 0);
                        writer.send_data_in_place(&mut v, compress).await?;
                        continue;
                    }
                    r = lr.read(&mut buf[1..]) => r?,
                },
                None => lr.read(&mut buf[1..]).await?,
            };
            if n == 0 {
                writer.send(&Frame::Close).await?;
                writer.shutdown().await;
                return Ok((true, total));
            }
            if let Some(t) = &mut tap_up {
                t.feed(&buf[1..n + 1]);
            }
            match &mut guard_down {
                Some(g) => match g.feed(&mut buf[1..n + 1], &mut gout).map_err(guard_err)? {
                    Fwd::Chunk => writer.send_data_in_place(&mut buf[..n + 1], compress).await?,
                    Fwd::Out if gout.is_empty() => {}
                    Fwd::Out => {
                        out.clear();
                        out.push(0);
                        out.extend_from_slice(&gout);
                        writer.send_data_in_place(&mut out, compress).await?;
                    }
                },
                None => writer.send_data_in_place(&mut buf[..n + 1], compress).await?,
            }
            total += n as u64;
            if let Some(c) = &live_up {
                c.fetch_add(n as u64, Ordering::Relaxed);
            }
        }
    });
    set.spawn(async move {
        let mut total = 0u64;
        let (mut gout, mut ginj) = (Vec::with_capacity(64), Vec::new());
        loop {
            let p = reader.recv_plain_mut().await?;
            // Fast path: plain data goes straight to the socket.
            let n = if let Some(d) = crate::wire::plain_data_mut(p) {
                if let Some(t) = &mut tap_down {
                    t.feed(d);
                }
                forward_up(&mut guard_up, d, &mut gout, &mut ginj, &mut lw, &inject_tx).await?;
                d.len()
            } else {
                match Frame::decode(p)? {
                    Frame::Data(mut d) => {
                        if let Some(t) = &mut tap_down {
                            t.feed(&d);
                        }
                        forward_up(&mut guard_up, &mut d, &mut gout, &mut ginj, &mut lw, &inject_tx).await?;
                        d.len()
                    }
                    Frame::Close => {
                        let _ = lw.shutdown().await;
                        return Ok((false, total));
                    }
                    _ => 0,
                }
            };
            total += n as u64;
            if let Some(c) = &live_down {
                c.fetch_add(n as u64, Ordering::Relaxed);
            }
        }
    });
    // Dropping the set (the caller cancelled) aborts both tasks.
    let (mut sent, mut received) = (0, 0);
    while let Some(r) = set.join_next().await {
        match r {
            Ok(Ok((true, n))) => sent = n,
            Ok(Ok((false, n))) => received = n,
            Ok(Err(e)) => {
                set.abort_all();
                return Err(e);
            }
            Err(e) => {
                set.abort_all();
                return Err(LinkError::Protocol(format!("tunnel task failed: {e}")));
            }
        }
    }
    Ok((sent, received))
}

fn guard_err(e: GuardError) -> LinkError {
    match e {
        GuardError::Incompatible(_) => LinkError::Refused(e.to_string()),
        GuardError::Protocol(_) => LinkError::Protocol(e.to_string()),
    }
}

/// Write request bytes from the tunnel to rpc-server, through the guard if
/// there is one (its refusals go back through `inject`).
async fn forward_up(
    guard: &mut Option<rpc_guard::Upstream>,
    d: &mut [u8],
    gout: &mut Vec<u8>,
    ginj: &mut Vec<u8>,
    lw: &mut tokio::net::tcp::OwnedWriteHalf,
    inject: &Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
) -> LinkResult<()> {
    let Some(g) = guard else {
        lw.write_all(d).await?;
        return Ok(());
    };
    ginj.clear();
    match g.feed(d, gout, ginj).map_err(guard_err)? {
        Fwd::Chunk => lw.write_all(d).await?,
        Fwd::Out if gout.is_empty() => {}
        Fwd::Out => lw.write_all(gout).await?,
    }
    if !ginj.is_empty()
        && let Some(tx) = inject
    {
        tx.send(std::mem::take(ginj)).await.map_err(|_| LinkError::Protocol("tunnel closed".into()))?;
    }
    Ok(())
}

#[cfg(test)]
mod bind_tests {
    use super::*;

    #[tokio::test]
    async fn busy_port_falls_back_to_next_free_one() {
        // Hold a port, then ask for it: the helper moves to a fallback port.
        let held = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let busy = held.local_addr().unwrap().port();
        let b = bind_with_fallback(busy, |p| TcpListener::bind(("127.0.0.1", if p == busy { p } else { 0 }))).await.unwrap();
        // Fallback tries FALLBACK_PORTS numbers; the closure above maps them to an OS-chosen free port.
        assert_eq!(b.port, *crate::FALLBACK_PORTS.start());
        let n = b.notice.unwrap();
        assert!(n.contains("excludedportrange") && n.contains("instead"), "{n}");
    }

    #[tokio::test]
    async fn free_port_has_no_notice_and_other_errors_dont_fall_back() {
        let b = bind_with_fallback(0, |p| TcpListener::bind(("127.0.0.1", p))).await.unwrap();
        assert!(b.notice.is_none());
        let mut tries = 0;
        let e = bind_with_fallback::<(), _, _>(5, |_| { tries += 1; async { Err(std::io::Error::other("boom")) } }).await.err().unwrap();
        assert!(e.contains("boom"));
        assert_eq!(tries, 1);
        // Every port refused: a clear final message.
        let e = bind_with_fallback::<(), _, _>(5, |_| async { Err(std::io::Error::from_raw_os_error(10013)) }).await.err().unwrap();
        assert!(e.contains("excludedportrange") && e.contains("No free port"), "{e}");
    }
}

//! The main PC's side: pairing, the control channel (info, RTT, bandwidth,
//! unlink) and the RPC bridge that llama-server's `--rpc` connects to.

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

use crate::handshake::{self, Session, read_accept};
use crate::helper::{HANDSHAKE_TIMEOUT, pipe};
use crate::identity::Identity;
use crate::wire::{DATA_CHUNK, Frame, Mode, SecureWriter};
use crate::{DEFAULT_PORT, LinkError, LinkResult, PeerInfo, timed};

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(8);
const REPLY_TIMEOUT: Duration = Duration::from_secs(10);

/// Build `host:port` from what the user typed: "192.168.1.20",
/// "192.168.1.20:50071", "[fe80::1]:50070", "fe80::1" or a host name.
pub fn join_addr(host: &str, port: Option<u16>) -> LinkResult<String> {
    let h = host.trim();
    let bad = || LinkError::Refused(format!("\"{h}\" is not an IP address or computer name"));
    if h.is_empty() || h.len() > 255 || h.chars().any(|c| c.is_whitespace() || c == '/' || c == '@') {
        return Err(bad());
    }
    if let Ok(sa) = h.parse::<SocketAddr>() {
        return Ok(match port {
            Some(p) => SocketAddr::new(sa.ip(), p).to_string(),
            None => sa.to_string(),
        });
    }
    let explicit = port;
    let port = port.unwrap_or(DEFAULT_PORT);
    if let Ok(ip) = h.parse::<IpAddr>() {
        return Ok(SocketAddr::new(ip, port).to_string());
    }
    // host:port with a name
    if let Some((name, p)) = h.rsplit_once(':')
        && let Ok(p) = p.parse::<u16>()
        && !name.contains(':')
    {
        return Ok(format!("{name}:{}", explicit.unwrap_or(p)));
    }
    if h.contains(':') || !h.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.') {
        return Err(bad());
    }
    Ok(format!("{h}:{port}"))
}

/// How long each guess of the default port waits when more guesses follow.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(4);

fn host_has_port(h: &str) -> bool {
    h.parse::<SocketAddr>().is_ok() || h.rsplit_once(':').is_some_and(|(n, p)| !n.contains(':') && p.parse::<u16>().is_ok())
}

/// The addresses to try for what the user typed: with a port (in the box or
/// in the text) just that one; with only a host, the default port, then the
/// port earlier builds used.
pub fn candidate_addrs(host: &str, port: Option<u16>) -> LinkResult<Vec<String>> {
    let first = join_addr(host, port)?;
    if port.is_some() || host_has_port(host.trim()) {
        return Ok(vec![first]);
    }
    let legacy = join_addr(host, Some(crate::LEGACY_PORT))?;
    Ok(vec![first, legacy])
}

/// Run `f` on each address in order; "nothing listens" or "no answer" moves
/// on to the next, any other result (success, wrong code...) ends it. The
/// closure gets the address and its connect timeout (short unless last).
pub async fn first_reachable<T, F, Fut>(addrs: &[String], mut f: F) -> LinkResult<(String, T)>
where
    F: FnMut(String, Duration) -> Fut,
    Fut: std::future::Future<Output = LinkResult<T>>,
{
    let mut refused = Vec::new();
    let mut first_other: Option<LinkError> = None;
    for (i, a) in addrs.iter().enumerate() {
        let t = if i + 1 == addrs.len() { CONNECT_TIMEOUT } else { PROBE_TIMEOUT };
        match f(a.clone(), t).await {
            Ok(v) => return Ok((a.clone(), v)),
            Err(LinkError::NotListening(_)) => refused.push(a.clone()),
            Err(e @ LinkError::Unreachable(_)) => {
                first_other.get_or_insert(e);
            }
            Err(e) => return Err(e),
        }
    }
    match first_other {
        Some(e) => Err(e),
        None => Err(LinkError::NotListening(refused.join(" or "))),
    }
}

/// Link to a PC from what the user typed, trying the default ports in order.
pub async fn pair_typed(host: &str, port: Option<u16>, code: &str, identity: &Identity, me: &PeerInfo) -> LinkResult<(String, Paired)> {
    let addrs = candidate_addrs(host, port)?;
    first_reachable(&addrs, |a, t| async move { pair_within(&a, code, identity, me, t).await }).await
}

/// Connect, telling "nothing listens there" (refused) apart from "no
/// answer" (timeout: off, wrong address, firewall or router).
pub async fn dial(addr: &str) -> LinkResult<TcpStream> {
    dial_within(addr, CONNECT_TIMEOUT).await
}

pub async fn dial_within(addr: &str, limit: Duration) -> LinkResult<TcpStream> {
    use std::io::ErrorKind as K;
    let s = match tokio::time::timeout(limit, TcpStream::connect(addr)).await {
        Err(_) => return Err(LinkError::Unreachable(addr.into())),
        Ok(Err(e)) if e.kind() == K::ConnectionRefused => return Err(LinkError::NotListening(addr.into())),
        Ok(Err(e)) if matches!(e.kind(), K::TimedOut | K::HostUnreachable | K::NetworkUnreachable) => return Err(LinkError::Unreachable(addr.into())),
        Ok(r) => r?,
    };
    crate::wire::tune_socket(&s);
    Ok(s)
}

/// An I/O error while the handshake runs means the other end closed or
/// reset the connection before answering.
fn early(addr: &str, e: LinkError) -> LinkError {
    use std::io::ErrorKind as K;
    match e {
        LinkError::Io(io) if matches!(io.kind(), K::UnexpectedEof | K::ConnectionReset | K::ConnectionAborted | K::BrokenPipe) => LinkError::ClosedEarly(addr.into()),
        other => other,
    }
}

#[derive(Debug, Clone)]
pub struct Paired {
    /// The helper's static key, now pinned.
    pub key: [u8; 32],
    pub info: PeerInfo,
    pub address: SocketAddr,
}

/// Link to the PC at `addr` that shows `code`.
pub async fn pair(addr: &str, code: &str, identity: &Identity, me: &PeerInfo) -> LinkResult<Paired> {
    pair_within(addr, code, identity, me, CONNECT_TIMEOUT).await
}

async fn pair_within(addr: &str, code: &str, identity: &Identity, me: &PeerInfo, connect: Duration) -> LinkResult<Paired> {
    let s = dial_within(addr, connect).await?;
    let address = s.peer_addr()?;
    let (info, key) = timed(HANDSHAKE_TIMEOUT, handshake::pair_initiate(s, identity, code, me)).await.map_err(|e| early(addr, e))?;
    Ok(Paired { key, info, address })
}

/// Open a link connection of `mode` to a pinned peer and wait for its accept.
pub async fn connect(addr: &str, identity: &Identity, expected: [u8; 32], mode: Mode, request: serde_json::Value) -> LinkResult<(Session, serde_json::Value, SocketAddr)> {
    let s = dial(addr).await?;
    let address = s.peer_addr()?;
    let mut sess = timed(HANDSHAKE_TIMEOUT, handshake::link_initiate(s, identity, mode, expected, &request)).await.map_err(|e| early(addr, e))?;
    let v = timed(HANDSHAKE_TIMEOUT, read_accept(&mut sess.reader)).await.map_err(|e| early(addr, e))?;
    Ok((sess, v, address))
}

/// What the control channel knows about the other PC, kept current by a
/// background reader: the helper pushes `info_update` when its state
/// changes, and a closed connection shows at once.
#[derive(Debug)]
pub struct ControlState {
    info: Mutex<PeerInfo>,
    /// The helper's last `share_refused` message (it refused memory over
    /// this PC's limit there, or can't enforce limits with this protocol).
    refusal: Mutex<Option<serde_json::Value>>,
    closed: tokio::sync::watch::Sender<Option<String>>,
}

impl ControlState {
    pub fn info(&self) -> PeerInfo {
        self.info.lock().unwrap().clone()
    }

    /// The helper's last refusal (`{"reason": "over_limit" | "incompatible", ...}`).
    pub fn refusal(&self) -> Option<serde_json::Value> {
        self.refusal.lock().unwrap().clone()
    }

    /// Forget an earlier refusal (a new load starts).
    pub fn clear_refusal(&self) {
        *self.refusal.lock().unwrap() = None;
    }

    /// Why the connection closed, once it has.
    pub fn closed_reason(&self) -> Option<String> {
        self.closed.borrow().clone()
    }

    pub fn is_closed(&self) -> bool {
        self.closed.borrow().is_some()
    }

    /// Resolves when the connection closes (at once if it already has).
    pub async fn wait_closed(&self) -> String {
        let mut rx = self.closed.subscribe();
        match rx.wait_for(|c| c.is_some()).await {
            Ok(v) => v.clone().unwrap_or_default(),
            Err(_) => "closed".into(),
        }
    }
}

/// Control channel to a linked PC.
pub struct Control {
    writer: SecureWriter,
    rx: tokio::sync::mpsc::Receiver<Frame>,
    state: Arc<ControlState>,
    reader_task: tokio::task::JoinHandle<()>,
    next: u64,
    pub address: SocketAddr,
}

impl Drop for Control {
    fn drop(&mut self) {
        self.reader_task.abort();
        self.state.closed.send_if_modified(|c| {
            if c.is_none() {
                *c = Some("closed here".into());
                true
            } else {
                false
            }
        });
    }
}

impl Control {
    pub async fn connect(addr: &str, identity: &Identity, expected: [u8; 32]) -> LinkResult<Self> {
        let (sess, v, address) = connect(addr, identity, expected, Mode::Control, serde_json::json!({})).await?;
        let info: PeerInfo = serde_json::from_value(v["info"].clone()).unwrap_or_default();
        let Session { mut reader, writer, .. } = sess;
        let state = Arc::new(ControlState { info: Mutex::new(info), refusal: Mutex::new(None), closed: tokio::sync::watch::channel(None).0 });
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let st = state.clone();
        let reader_task = tokio::spawn(async move {
            let why = loop {
                match reader.recv().await {
                    Ok(Frame::Json(v)) if v["t"] == "info_update" => {
                        if let Ok(i) = serde_json::from_value::<PeerInfo>(v["info"].clone()) {
                            *st.info.lock().unwrap() = i;
                        }
                    }
                    // Kept here, not queued: nobody waits for it.
                    Ok(Frame::Json(v)) if v["t"] == "share_refused" => {
                        *st.refusal.lock().unwrap() = Some(v["refusal"].clone());
                    }
                    Ok(Frame::Close) => break "the other PC closed it".to_string(),
                    Ok(f) => {
                        if tx.send(f).await.is_err() {
                            break "closed here".to_string();
                        }
                    }
                    Err(LinkError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break "the other PC closed it".to_string(),
                    Err(e) => break e.to_string(),
                }
            };
            st.closed.send_replace(Some(why));
        });
        Ok(Self { writer, rx, state, reader_task, next: 1, address })
    }

    /// Shared state (info, closed) for watchers.
    pub fn state(&self) -> Arc<ControlState> {
        self.state.clone()
    }

    pub fn info(&self) -> PeerInfo {
        self.state.info()
    }

    async fn next_frame(&mut self) -> LinkResult<Frame> {
        match self.rx.recv().await {
            Some(f) => Ok(f),
            None => Err(LinkError::Disconnected(self.state.closed_reason().unwrap_or_else(|| "closed".into()))),
        }
    }

    async fn send(&mut self, f: &Frame) -> LinkResult<()> {
        if let Some(why) = self.state.closed_reason() {
            return Err(LinkError::Disconnected(why));
        }
        self.writer.send(f).await.map_err(|e| if e.is_connection_lost() { LinkError::Disconnected(e.to_string()) } else { e })
    }

    /// One round trip through the tunnel.
    pub async fn ping(&mut self) -> LinkResult<Duration> {
        let n = self.next;
        self.next += 1;
        let t = Instant::now();
        self.send(&Frame::Ping(n)).await?;
        timed(REPLY_TIMEOUT, async {
            loop {
                if let Frame::Pong(m) = self.next_frame().await?
                    && m == n
                {
                    return Ok(());
                }
            }
        })
        .await?;
        Ok(t.elapsed())
    }

    /// Median of `samples` pings.
    pub async fn rtt(&mut self, samples: usize) -> LinkResult<Duration> {
        let mut v = Vec::with_capacity(samples.max(1));
        for _ in 0..samples.max(1) {
            v.push(self.ping().await?);
        }
        v.sort();
        Ok(v[v.len() / 2])
    }

    async fn request(&mut self, req: serde_json::Value, want: &str) -> LinkResult<serde_json::Value> {
        self.send(&Frame::Json(req)).await?;
        self.wait_json(want).await
    }

    async fn wait_json(&mut self, want: &str) -> LinkResult<serde_json::Value> {
        timed(REPLY_TIMEOUT * 3, async {
            loop {
                if let Frame::Json(v) = self.next_frame().await? {
                    if v["t"] == want {
                        return Ok(v);
                    }
                    if v["t"] == "error" {
                        return Err(LinkError::Protocol(v["message"].as_str().unwrap_or("error").into()));
                    }
                }
            }
        })
        .await
    }

    pub async fn refresh_info(&mut self) -> LinkResult<PeerInfo> {
        let v = self.request(serde_json::json!({ "t": "info" }), "info").await?;
        let info: PeerInfo = serde_json::from_value(v["info"].clone()).unwrap_or_default();
        *self.state.info.lock().unwrap() = info.clone();
        Ok(info)
    }

    /// Upload speed to the helper in Mbit/s (the direction weights travel
    /// at load), from sending `bytes` through the tunnel. Gives up after
    /// `limit` in total (a stalled link must not hold anything up); after a
    /// timeout the connection is in an unknown state, so use a connection of
    /// its own for this and drop it afterwards.
    pub async fn bandwidth(&mut self, bytes: usize, limit: Duration) -> LinkResult<f64> {
        timed(limit, async {
            let rtt = self.ping().await?;
            let t = Instant::now();
            self.send(&Frame::Json(serde_json::json!({ "t": "bw", "bytes": bytes }))).await?;
            let mut chunk = vec![0u8; DATA_CHUNK + 1];
            let mut left = bytes;
            while left > 0 {
                let n = left.min(DATA_CHUNK);
                self.writer.send_data_in_place(&mut chunk[..n + 1], false).await?;
                left -= n;
            }
            self.wait_json("bw_done").await?;
            let secs = t.elapsed().saturating_sub(rtt).as_secs_f64().max(1e-4);
            Ok(bytes as f64 * 8.0 / secs / 1e6)
        })
        .await
    }

    /// Tell the other PC to forget this one.
    pub async fn unlink(&mut self) -> LinkResult<()> {
        self.request(serde_json::json!({ "t": "unlink" }), "unlinked").await.map(|_| ())
    }

    pub async fn close(mut self) {
        let _ = self.writer.send(&Frame::Close).await;
        self.writer.shutdown().await;
    }
}

#[derive(Debug, Default)]
struct BridgeStats {
    streams: AtomicU64,
    active: AtomicU64,
    /// Payload bytes sent to / received from the helper, counted as they
    /// move (not when a stream ends), so a load's progress can be shown.
    sent: Arc<AtomicU64>,
    received: Arc<AtomicU64>,
    /// What the ggml-rpc stream carried: tensor data sent, cache hits.
    rpc: Arc<crate::rpc_tap::RpcCounters>,
    last_error: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeSnapshot {
    pub endpoint: String,
    pub streams: u64,
    pub active: u64,
    pub sent_bytes: u64,
    pub received_bytes: u64,
    /// Tensor data sent with SET_TENSOR (weights at load).
    pub tensor_bytes: u64,
    /// Bytes the helper's weight cache supplied instead.
    pub cached_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub last_error: Option<String>,
}

/// A local endpoint (127.0.0.1:<random>) for llama-server's `--rpc`. Every
/// connection to it becomes one encrypted tunnel stream to the helper's
/// rpc-server.
pub struct RpcBridge {
    endpoint: SocketAddr,
    cancel: CancellationToken,
    stats: Arc<BridgeStats>,
}

impl RpcBridge {
    pub async fn start(remote: String, identity: Arc<Identity>, expected: [u8; 32], runtime_tag: Option<String>, compress: bool) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let endpoint = listener.local_addr()?;
        let cancel = CancellationToken::new();
        let stats = Arc::new(BridgeStats::default());
        let (c, st) = (cancel.clone(), stats.clone());
        tokio::spawn(async move {
            loop {
                let local = tokio::select! {
                    _ = c.cancelled() => break,
                    r = listener.accept() => match r {
                        Ok((s, _)) => s,
                        Err(_) => continue,
                    },
                };
                let (remote, identity, tag, st, c) = (remote.clone(), identity.clone(), runtime_tag.clone(), st.clone(), c.clone());
                tokio::spawn(async move {
                    st.streams.fetch_add(1, Ordering::Relaxed);
                    st.active.fetch_add(1, Ordering::Relaxed);
                    let req = serde_json::json!({ "runtime_tag": tag, "compress": compress });
                    let run = async {
                        let (sess, _, _) = connect(&remote, &identity, expected, Mode::Rpc, req).await?;
                        pipe(sess, local, compress, Some((st.sent.clone(), st.received.clone())), Some(st.rpc.clone()), None).await
                    };
                    let r = tokio::select! { _ = c.cancelled() => Err(LinkError::Refused("link closed".into())), r = run => r };
                    match r {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("RPC tunnel stream to {remote} failed: {e}");
                            *st.last_error.lock().unwrap() = Some(e.to_string());
                        }
                    }
                    st.active.fetch_sub(1, Ordering::Relaxed);
                });
            }
        });
        Ok(Self { endpoint, cancel, stats })
    }

    /// Live count of payload bytes sent to the helper through this bridge
    /// (cumulative over every stream).
    pub fn sent_counter(&self) -> Arc<AtomicU64> {
        self.stats.sent.clone()
    }

    /// What the ggml-rpc streams through this bridge carried (cumulative):
    /// tensor data really sent, and what the helper's weight cache supplied.
    /// The load progress reads these.
    pub fn rpc_counters(&self) -> Arc<crate::rpc_tap::RpcCounters> {
        self.stats.rpc.clone()
    }

    /// What to pass to llama-server's `--rpc`.
    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }

    pub fn snapshot(&self) -> BridgeSnapshot {
        let s = &self.stats;
        BridgeSnapshot {
            endpoint: self.endpoint.to_string(),
            streams: s.streams.load(Ordering::Relaxed),
            active: s.active.load(Ordering::Relaxed),
            sent_bytes: s.sent.load(Ordering::Relaxed),
            received_bytes: s.received.load(Ordering::Relaxed),
            tensor_bytes: s.rpc.tensor_bytes.load(Ordering::Relaxed),
            cached_bytes: s.rpc.cached_bytes.load(Ordering::Relaxed),
            cache_hits: s.rpc.cache_hits.load(Ordering::Relaxed),
            cache_misses: s.rpc.cache_misses.load(Ordering::Relaxed),
            last_error: s.last_error.lock().unwrap().clone(),
        }
    }

    pub fn stop(&self) {
        self.cancel.cancel();
    }
}

impl Drop for RpcBridge {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_people_type() {
        assert_eq!(join_addr("192.168.1.20", None).unwrap(), "192.168.1.20:47070");
        assert_eq!(join_addr(" 192.168.1.20:50071 ", None).unwrap(), "192.168.1.20:50071");
        assert_eq!(join_addr("192.168.1.20", Some(5000)).unwrap(), "192.168.1.20:5000");
        assert_eq!(join_addr("fe80::1", None).unwrap(), "[fe80::1]:47070");
        assert_eq!(join_addr("[fe80::1]:6000", None).unwrap(), "[fe80::1]:6000");
        assert_eq!(join_addr("gaming-pc", None).unwrap(), "gaming-pc:47070");
        assert_eq!(join_addr("gaming-pc:6000", None).unwrap(), "gaming-pc:6000");
        for bad in ["", "http://x", "a b", "user@host", "x:y:z"] {
            assert!(join_addr(bad, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_an_ip_tries_new_default_then_old() {
        assert_eq!(candidate_addrs("192.168.1.20", None).unwrap(), ["192.168.1.20:47070", "192.168.1.20:50070"]);
        assert_eq!(candidate_addrs("gaming-pc", None).unwrap(), ["gaming-pc:47070", "gaming-pc:50070"]);
        assert_eq!(candidate_addrs("fe80::1", None).unwrap(), ["[fe80::1]:47070", "[fe80::1]:50070"]);
        // An explicit port (box or text) is tried alone.
        assert_eq!(candidate_addrs("192.168.1.20", Some(5000)).unwrap(), ["192.168.1.20:5000"]);
        assert_eq!(candidate_addrs("192.168.1.20:50071", None).unwrap(), ["192.168.1.20:50071"]);
        assert_eq!(candidate_addrs("gaming-pc:6000", None).unwrap(), ["gaming-pc:6000"]);
    }

    #[tokio::test]
    async fn falls_through_refused_ports_in_order() {
        let addrs = candidate_addrs("10.0.0.5", None).unwrap();
        let seen = std::sync::Mutex::new(Vec::new());
        // First port refuses, second answers.
        let r = first_reachable(&addrs, |a, t| {
            seen.lock().unwrap().push((a.clone(), t));
            async move { if a.ends_with(":47070") { Err(LinkError::NotListening(a)) } else { Ok(7) } }
        })
        .await
        .unwrap();
        assert_eq!(r, ("10.0.0.5:50070".to_string(), 7));
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["10.0.0.5:47070", "10.0.0.5:50070"]);
        assert_eq!(seen[0].1, PROBE_TIMEOUT);
        assert_eq!(seen[1].1, CONNECT_TIMEOUT);
        // Both refuse: one "nothing is listening" naming both.
        let e = first_reachable::<(), _, _>(&addrs, |a, _| async move { Err(LinkError::NotListening(a)) }).await.unwrap_err();
        assert!(matches!(&e, LinkError::NotListening(s) if s == "10.0.0.5:47070 or 10.0.0.5:50070"), "{e}");
        // A wrong code on the first port stops there.
        let mut n = 0;
        let e = first_reachable::<(), _, _>(&addrs, |_, _| { n += 1; async { Err(LinkError::WrongCode { remaining: 2 }) } }).await.unwrap_err();
        assert!(matches!(e, LinkError::WrongCode { .. }));
        assert_eq!(n, 1);
    }
}

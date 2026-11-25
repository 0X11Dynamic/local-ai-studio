//! Loopback benchmark: an RPC-like request/response (ggml-rpc style: the
//! client writes a 1-byte command, an 8-byte size and the payload as three
//! separate writes; the server answers with an 8-byte size and the payload)
//! through the tunnel vs direct. The tunnel must add well under 1 ms per
//! round trip for small messages.
//!
//! Numbers: `cargo test -p linking --release --test bench -- --nocapture`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use linking::client::{self, RpcBridge};
use linking::code::CodeSlot;
use linking::helper::{GuardSpec, HelperHooks, HelperServer, HookFuture, RpcTarget};
use linking::{DevKind, Identity, LinkError, PeerBudget, PeerInfo, ShareLimits};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// The helper's guard (per-PC limits) is on, as in the app; set
/// `LAS_BENCH_NO_GUARD=1` to measure the tunnel without it.
struct Hooks {
    pinned: Mutex<HashMap<[u8; 32], String>>,
    rpc: Mutex<Option<SocketAddr>>,
    budget: Option<Arc<PeerBudget>>,
}

impl Default for Hooks {
    fn default() -> Self {
        let guard = std::env::var("LAS_BENCH_NO_GUARD").map_or(true, |v| v != "1");
        let budget = guard.then(|| PeerBudget::new(ShareLimits { gpu_mib: 64 << 10, ram_mib: 64 << 10, threads: 4, paused: false }));
        Self { pinned: Default::default(), rpc: Default::default(), budget }
    }
}

fn guard_note() -> &'static str {
    if std::env::var("LAS_BENCH_NO_GUARD").is_ok_and(|v| v == "1") { "guard off" } else { "guard on" }
}

const CMD_HELLO: u8 = 14;

/// ggml-rpc HELLO (protocol 9): the guard wants it first on every stream.
async fn rpc_hello(s: &mut TcpStream) {
    s.write_all(&[CMD_HELLO]).await.unwrap();
    s.write_all(&24u64.to_le_bytes()).await.unwrap();
    s.write_all(&[0u8; 24]).await.unwrap();
    let mut r = [0u8; 8 + 28];
    s.read_exact(&mut r).await.unwrap();
    assert_eq!(r[8], 9);
}

/// rpc-server's HELLO reply: size 28, protocol 9.0.0, no flags, no caps.
fn hello_reply() -> Vec<u8> {
    let mut v = 28u64.to_le_bytes().to_vec();
    v.extend_from_slice(&[9, 0, 0, 0]);
    v.extend_from_slice(&[0u8; 24]);
    v
}

impl HelperHooks for Hooks {
    fn peer_name(&self, key: &[u8; 32]) -> Option<String> {
        self.pinned.lock().unwrap().get(key).cloned()
    }
    fn paired(&self, key: [u8; 32], info: &PeerInfo, _addr: SocketAddr) {
        self.pinned.lock().unwrap().insert(key, info.name.clone());
    }
    fn unlinked(&self, _key: [u8; 32]) {}
    fn info(&self) -> PeerInfo {
        PeerInfo { name: "helper".into(), helping: true, runtime_tag: Some("b1".into()), ..Default::default() }
    }
    fn rpc_target<'a>(&'a self, _key: &'a [u8; 32]) -> HookFuture<'a, Result<RpcTarget, LinkError>> {
        let limits = self.budget.as_ref().map(|b| GuardSpec { claim: b.claim(), kinds: Arc::new(vec![DevKind::Gpu]) });
        let r = self.rpc.lock().unwrap().ok_or(LinkError::HelperOff).map(|addr| RpcTarget { addr, limits });
        Box::pin(async move { r })
    }
}

/// ggml-rpc-like server: request = cmd(1) + size(8) + payload; reply = size(8) + payload.
async fn rpc_like_server() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            s.set_nodelay(true).unwrap();
            tokio::spawn(async move {
                let mut payload = Vec::new();
                loop {
                    let mut cmd = [0u8; 1];
                    if s.read_exact(&mut cmd).await.is_err() {
                        break;
                    }
                    let mut size = [0u8; 8];
                    s.read_exact(&mut size).await.unwrap();
                    payload.resize(u64::from_le_bytes(size) as usize, 0);
                    s.read_exact(&mut payload).await.unwrap();
                    if cmd[0] == CMD_HELLO {
                        s.write_all(&hello_reply()).await.unwrap();
                        continue;
                    }
                    s.write_all(&size).await.unwrap();
                    s.write_all(&payload).await.unwrap();
                }
            });
        }
    });
    a
}

async fn round_trips(s: &mut TcpStream, size: usize, n: usize) -> Duration {
    let payload = vec![7u8; size];
    let mut back = vec![0u8; size];
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        s.write_all(&[3]).await.unwrap();
        s.write_all(&(size as u64).to_le_bytes()).await.unwrap();
        s.write_all(&payload).await.unwrap();
        let mut sz = [0u8; 8];
        s.read_exact(&mut sz).await.unwrap();
        s.read_exact(&mut back).await.unwrap();
        times.push(t.elapsed());
    }
    times.sort();
    times[times.len() / 2]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tunnel_adds_under_a_millisecond_per_small_round_trip() {
    let helper_id = Arc::new(Identity::generate().unwrap());
    let codes = Arc::new(CodeSlot::new());
    let hooks = Arc::new(Hooks::default());
    let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id, codes.clone(), hooks.clone()).await.unwrap();
    let addr = server.local_addr().to_string();
    let main_id = Arc::new(Identity::generate().unwrap());
    let p = client::pair(&addr, &codes.new_code().code, &main_id, &PeerInfo { name: "main".into(), ..Default::default() }).await.unwrap();
    let rpc = rpc_like_server().await;
    *hooks.rpc.lock().unwrap() = Some(rpc);
    let bridge = RpcBridge::start(addr, main_id, p.key, Some("b1".into()), false).await.unwrap();

    let mut direct = TcpStream::connect(rpc).await.unwrap();
    direct.set_nodelay(true).unwrap();
    let mut tunnel = TcpStream::connect(bridge.endpoint()).await.unwrap();
    tunnel.set_nodelay(true).unwrap();
    rpc_hello(&mut tunnel).await;
    rpc_hello(&mut direct).await;
    eprintln!("({})", guard_note());
    // Warm up (handshake on the first bytes).
    round_trips(&mut tunnel, 16, 5).await;
    round_trips(&mut direct, 16, 5).await;

    let mut small_overhead = Duration::ZERO;
    for (size, n) in [(1024usize, 400usize), (64 * 1024, 100), (4 << 20, 8)] {
        let d = round_trips(&mut direct, size, n).await;
        let t = round_trips(&mut tunnel, size, n).await;
        eprintln!(
            "round trip {:>8} B: direct {:>9.3} ms, tunnel {:>9.3} ms, added {:>8.3} ms",
            size,
            d.as_secs_f64() * 1e3,
            t.as_secs_f64() * 1e3,
            (t.as_secs_f64() - d.as_secs_f64()) * 1e3
        );
        if size == 1024 {
            small_overhead = t.saturating_sub(d);
        }
    }
    assert!(small_overhead < Duration::from_millis(1), "tunnel adds {small_overhead:?} per 1 KB round trip");
}

/// Forward a TCP connection to `target`, at most `mbit` Mbit/s each way
/// (a stand-in for a real link; both paths go through the same one).
async fn throttled_relay(target: SocketAddr, mbit: f64) -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (inb, _) = l.accept().await.unwrap();
            let out = TcpStream::connect(target).await.unwrap();
            inb.set_nodelay(true).unwrap();
            out.set_nodelay(true).unwrap();
            let (ir, iw) = inb.into_split();
            let (or, ow) = out.into_split();
            tokio::spawn(throttle_copy(ir, ow, mbit));
            tokio::spawn(throttle_copy(or, iw, mbit));
        }
    });
    a
}

async fn throttle_copy(mut r: tokio::net::tcp::OwnedReadHalf, mut w: tokio::net::tcp::OwnedWriteHalf, mbit: f64) {
    let mut buf = vec![0u8; 256 * 1024];
    // Pacing with no saved-up credit: idle time doesn't allow a later burst.
    let mut next = Instant::now();
    loop {
        let n = match r.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if w.write_all(&buf[..n]).await.is_err() {
            break;
        }
        let now = Instant::now();
        next = next.max(now) + Duration::from_secs_f64(n as f64 * 8.0 / (mbit * 1e6));
        if next > now + Duration::from_millis(5) {
            tokio::time::sleep(next - now).await;
        }
    }
    let _ = w.shutdown().await;
}

const CMD_GET_ALLOC_SIZE: u8 = 13;

/// Sink in ggml-rpc framing (`cmd | size | body`): HELLO gets its reply,
/// SET_TENSOR bodies are read and dropped, GET_ALLOC_SIZE (the "ack" after a
/// bulk send) answers 8 bytes.
async fn sink_server() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let mut head = [0u8; 9];
                    if s.read_exact(&mut head).await.is_err() {
                        break;
                    }
                    let mut left = u64::from_le_bytes(head[1..].try_into().unwrap()) as usize;
                    while left > 0 {
                        let n = s.read(&mut buf[..left.min(1 << 20)]).await.unwrap();
                        assert!(n > 0);
                        left -= n;
                    }
                    match head[0] {
                        CMD_HELLO => s.write_all(&hello_reply()).await.unwrap(),
                        CMD_GET_ALLOC_SIZE => {
                            s.write_all(&8u64.to_le_bytes()).await.unwrap();
                            s.write_all(&[0u8; 8]).await.unwrap();
                        }
                        _ => {}
                    }
                }
            });
        }
    });
    a
}

/// One SET_TENSOR of `bytes` (like a weight at load: 305-byte header, then
/// the data), then a GET_ALLOC_SIZE to know it all arrived.
async fn bulk_mbit(s: &mut TcpStream, bytes: usize) -> f64 {
    let chunk = vec![5u8; 1 << 20];
    let t = Instant::now();
    s.write_all(&[CMD_SET_TENSOR]).await.unwrap();
    s.write_all(&((305 + bytes) as u64).to_le_bytes()).await.unwrap();
    s.write_all(&[0u8; 305]).await.unwrap();
    let mut left = bytes;
    while left > 0 {
        let n = left.min(chunk.len());
        s.write_all(&chunk[..n]).await.unwrap();
        left -= n;
    }
    s.write_all(&[CMD_GET_ALLOC_SIZE]).await.unwrap();
    s.write_all(&296u64.to_le_bytes()).await.unwrap();
    s.write_all(&[0u8; 296]).await.unwrap();
    let mut ack = [0u8; 16];
    s.read_exact(&mut ack).await.unwrap();
    bytes as f64 * 8.0 / t.elapsed().as_secs_f64() / 1e6
}

/// 64 MB one way (like set_tensor at load), tunnel vs direct: unlimited
/// loopback, and through the same 800 Mbit/s link. Over a real link the
/// tunnel must stay within 10% of direct (release build).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bulk_throughput_through_the_tunnel_is_close_to_direct() {
    const BYTES: usize = 64 << 20;
    let sink = sink_server().await;
    for link in [None, Some(800.0)] {
        let helper_id = Arc::new(Identity::generate().unwrap());
        let codes = Arc::new(CodeSlot::new());
        let hooks = Arc::new(Hooks::default());
        let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id, codes.clone(), hooks.clone()).await.unwrap();
        *hooks.rpc.lock().unwrap() = Some(sink);
        // The tunnel crosses the link; direct goes to the sink over the same kind of link.
        let (tunnel_addr, direct_addr) = match link {
            Some(m) => (throttled_relay(server.local_addr(), m).await, throttled_relay(sink, m).await),
            None => (server.local_addr(), sink),
        };
        let main_id = Arc::new(Identity::generate().unwrap());
        let p = client::pair(&tunnel_addr.to_string(), &codes.new_code().code, &main_id, &PeerInfo { name: "main".into(), ..Default::default() }).await.unwrap();
        let bridge = RpcBridge::start(tunnel_addr.to_string(), main_id, p.key, Some("b1".into()), false).await.unwrap();
        let mut direct = TcpStream::connect(direct_addr).await.unwrap();
        let mut tunnel = TcpStream::connect(bridge.endpoint()).await.unwrap();
        rpc_hello(&mut tunnel).await;
        rpc_hello(&mut direct).await;
        bulk_mbit(&mut tunnel, 1 << 20).await; // warm up
        let mut d = Vec::new();
        let mut t = Vec::new();
        for _ in 0..3 {
            d.push(bulk_mbit(&mut direct, BYTES).await);
            t.push(bulk_mbit(&mut tunnel, BYTES).await);
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (d, t) = (d[1], t[1]);
        eprintln!(
            "64 MB one way, {} ({}): direct {:>7.0} Mbit/s, tunnel {:>7.0} Mbit/s ({:.0}%)",
            link.map(|m| format!("{m:.0} Mbit/s link")).unwrap_or_else(|| "loopback".into()),
            guard_note(),
            d,
            t,
            t / d * 100.0
        );
        if link.is_some() && !cfg!(debug_assertions) {
            assert!(t >= d * 0.9, "tunnel {t:.0} Mbit/s vs direct {d:.0} Mbit/s over the same link");
        }
    }
}

// ------------------------------------------------- ggml-rpc message pattern

const CMD_SET_TENSOR: u8 = 6;
const CMD_GET_TENSOR: u8 = 8;
const CMD_GRAPH_COMPUTE: u8 = 10;
const CMD_GRAPH_RECOMPUTE: u8 = 16;

/// ggml-rpc-like server: `cmd | size | body`; GET_TENSOR answers
/// `size | data` with as many bytes as the body's first 8 bytes ask for,
/// every other command has no answer (as SET_TENSOR, GRAPH_COMPUTE and
/// GRAPH_RECOMPUTE in protocol 9).
async fn ggml_like_server() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            s.set_nodelay(true).unwrap();
            tokio::spawn(async move {
                let mut body = Vec::new();
                let out = vec![9u8; 4 << 20];
                loop {
                    let mut head = [0u8; 9];
                    if s.read_exact(&mut head).await.is_err() {
                        break;
                    }
                    body.resize(u64::from_le_bytes(head[1..].try_into().unwrap()) as usize, 0);
                    s.read_exact(&mut body).await.unwrap();
                    if head[0] == CMD_HELLO {
                        s.write_all(&hello_reply()).await.unwrap();
                    }
                    if head[0] == CMD_GET_TENSOR {
                        let n = u64::from_le_bytes(body[..8].try_into().unwrap());
                        // send_msg: size, then data (two writes).
                        s.write_all(&n.to_le_bytes()).await.unwrap();
                        s.write_all(&out[..n as usize]).await.unwrap();
                    }
                }
            });
        }
    });
    a
}

/// One request as ggml-rpc writes it: cmd, size, input, payload as separate
/// `send()` calls on a TCP_NODELAY socket.
async fn rpc_send(s: &mut TcpStream, cmd: u8, input: &[u8], payload: &[u8]) {
    s.write_all(&[cmd]).await.unwrap();
    s.write_all(&((input.len() + payload.len()) as u64).to_le_bytes()).await.unwrap();
    s.write_all(input).await.unwrap();
    if !payload.is_empty() {
        s.write_all(payload).await.unwrap();
    }
}

async fn rpc_get_tensor(s: &mut TcpStream, size: usize, back: &mut Vec<u8>) {
    let mut req = vec![0u8; 312];
    req[..8].copy_from_slice(&(size as u64).to_le_bytes());
    rpc_send(s, CMD_GET_TENSOR, &req, &[]).await;
    let mut n = [0u8; 8];
    s.read_exact(&mut n).await.unwrap();
    back.resize(u64::from_le_bytes(n) as usize, 0);
    s.read_exact(back).await.unwrap();
}

/// What one generated token costs a linked PC that holds the first layers
/// (measured with a timing proxy on the VM test, Qwen3-4B, b12250): five
/// small SET_TENSORs (inputs, positions, masks), GRAPH_RECOMPUTE, and a
/// GET_TENSOR of the 10 KB hidden state it waits for.
async fn token_step(s: &mut TcpStream, back: &mut Vec<u8>) {
    let header = [0u8; 305];
    for size in [10545usize, 309, 313, 313, 1329] {
        let payload = vec![1u8; size - header.len()];
        rpc_send(s, CMD_SET_TENSOR, &header, &payload).await;
    }
    rpc_send(s, CMD_GRAPH_RECOMPUTE, &[0u8; 12], &[]).await;
    rpc_get_tensor(s, 10240, back).await;
}

/// One prompt batch of 282 tokens (same measurement): a 289 KB SET_TENSOR,
/// a 350 KB GRAPH_COMPUTE and the 2.9 MB hidden state back.
async fn prompt_step(s: &mut TcpStream, back: &mut Vec<u8>) {
    let header = [0u8; 305];
    rpc_send(s, CMD_SET_TENSOR, &header, &vec![2u8; 289_073 - 305]).await;
    rpc_send(s, CMD_GRAPH_COMPUTE, &vec![3u8; 350_708], &[]).await;
    rpc_get_tensor(s, 2_887_680, back).await;
}

async fn median_ms<F>(n: usize, mut f: F) -> f64
where
    F: AsyncFnMut(),
{
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f().await;
        v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[n / 2]
}

/// The ggml-rpc message pattern of prompt processing and generation, tunnel
/// vs direct over the same 800 Mbit/s relay (release numbers:
/// `cargo test -p linking --release --test bench -- --nocapture`).
/// Per generated token the tunnel must add under 1 ms, and a prompt batch
/// must stay within 15% of direct (measured 103-112% with 16 KB frames,
/// 106-114% with 64 KB frames; on a real link the helper's compute time
/// dwarfs this: 2.4 s per 282-token batch on the VM).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_message_pattern_through_the_tunnel_is_close_to_direct() {
    let rpc = ggml_like_server().await;
    let helper_id = Arc::new(Identity::generate().unwrap());
    let codes = Arc::new(CodeSlot::new());
    let hooks = Arc::new(Hooks::default());
    let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id, codes.clone(), hooks.clone()).await.unwrap();
    *hooks.rpc.lock().unwrap() = Some(rpc);
    let (tunnel_addr, direct_addr) = (throttled_relay(server.local_addr(), 800.0).await, throttled_relay(rpc, 800.0).await);
    let main_id = Arc::new(Identity::generate().unwrap());
    let p = client::pair(&tunnel_addr.to_string(), &codes.new_code().code, &main_id, &PeerInfo { name: "main".into(), ..Default::default() }).await.unwrap();
    let bridge = RpcBridge::start(tunnel_addr.to_string(), main_id, p.key, Some("b1".into()), false).await.unwrap();
    let mut direct = TcpStream::connect(direct_addr).await.unwrap();
    direct.set_nodelay(true).unwrap();
    let mut tunnel = TcpStream::connect(bridge.endpoint()).await.unwrap();
    tunnel.set_nodelay(true).unwrap();
    rpc_hello(&mut tunnel).await;
    rpc_hello(&mut direct).await;
    eprintln!("({})", guard_note());
    let mut back = Vec::new();
    for _ in 0..20 {
        token_step(&mut tunnel, &mut back).await;
        token_step(&mut direct, &mut back).await;
    }
    prompt_step(&mut tunnel, &mut back).await;
    prompt_step(&mut direct, &mut back).await;

    // Alternate A/B/A/B so drift hits both paths alike.
    let (mut td, mut tt, mut pd, mut pt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..5 {
        td.push(median_ms(200, async || token_step(&mut direct, &mut back).await).await);
        tt.push(median_ms(200, async || token_step(&mut tunnel, &mut back).await).await);
        pd.push(median_ms(10, async || prompt_step(&mut direct, &mut back).await).await);
        pt.push(median_ms(10, async || prompt_step(&mut tunnel, &mut back).await).await);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (td, tt, pd, pt) = (med(&mut td), med(&mut tt), med(&mut pd), med(&mut pt));
    eprintln!("token step (5 SET_TENSOR + RECOMPUTE + 10 KB GET): direct {td:.3} ms, tunnel {tt:.3} ms, added {:.3} ms", tt - td);
    eprintln!("prompt step (289 KB + 350 KB up, 2.9 MB back):     direct {pd:.3} ms, tunnel {pt:.3} ms ({:.0}%)", pt / pd * 100.0);
    let snap = bridge.snapshot();
    assert!(snap.tensor_bytes > 0, "the tap counts SET_TENSOR data: {snap:?}");
    if !cfg!(debug_assertions) {
        assert!(tt - td < 1.0, "tunnel adds {:.3} ms per token step", tt - td);
        assert!(pt <= pd * 1.15, "prompt step: tunnel {pt:.2} ms vs direct {pd:.2} ms");
    }
}

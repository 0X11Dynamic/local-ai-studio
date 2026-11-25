//! The helper's sharing limits end to end over loopback: a fake ggml-rpc
//! server (protocol 9) behind a real HelperServer, reached from a real
//! RpcBridge, the way llama-server reaches a linked PC.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use linking::client::{self, Control, RpcBridge};
use linking::code::CodeSlot;
use linking::helper::{GuardSpec, HelperHooks, HelperServer, HookFuture, RpcTarget};
use linking::{DevKind, Identity, LinkError, PeerBudget, PeerInfo, Refusal, ShareLimits};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MIB: u64 = 1 << 20;
const HELLO: u8 = 14;
const DEVICE_COUNT: u8 = 15;
const DEVMEM: u8 = 11;
const ALLOC: u8 = 0;
const FREE: u8 = 4;
const SET_TENSOR: u8 = 6;

struct Hooks {
    pinned: Mutex<HashMap<[u8; 32], String>>,
    rpc: SocketAddr,
    budgets: Mutex<HashMap<[u8; 32], Arc<PeerBudget>>>,
    caps: Mutex<ShareLimits>,
    refusals: Mutex<Vec<Refusal>>,
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
    fn rpc_target<'a>(&'a self, key: &'a [u8; 32]) -> HookFuture<'a, Result<RpcTarget, LinkError>> {
        let r = (|| {
            let mut b = self.budgets.lock().unwrap();
            if b.iter().any(|(k, x)| k != key && x.open_streams() > 0) {
                return Err(LinkError::Refused("\"helper\" is busy helping another PC right now; try again when that model is unloaded.".into()));
            }
            let budget = b.entry(*key).or_insert_with(|| PeerBudget::new(*self.caps.lock().unwrap())).clone();
            Ok(RpcTarget { addr: self.rpc, limits: Some(GuardSpec { claim: budget.claim(), kinds: Arc::new(vec![DevKind::Gpu]) }) })
        })();
        Box::pin(async move { r })
    }
    fn refused(&self, _key: &[u8; 32], r: &Refusal) {
        self.refusals.lock().unwrap().push(r.clone());
    }
}

/// A protocol-9 rpc-server with one 8000 MiB GPU (7000 MiB free) that never
/// runs out itself.
async fn fake_rpc_server() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        let next = Arc::new(Mutex::new(0x1000u64));
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            let next = next.clone();
            tokio::spawn(async move {
                let mut body = Vec::new();
                loop {
                    let mut head = [0u8; 9];
                    if s.read_exact(&mut head).await.is_err() {
                        return;
                    }
                    body.resize(u64::from_le_bytes(head[1..].try_into().unwrap()) as usize, 0);
                    if s.read_exact(&mut body).await.is_err() {
                        return;
                    }
                    let reply: Option<Vec<u8>> = match head[0] {
                        HELLO => Some([vec![9, 0, 0, 0], vec![7u8; 24]].concat()),
                        DEVICE_COUNT => Some(1u32.to_le_bytes().to_vec()),
                        DEVMEM => Some([(7000 * MIB).to_le_bytes(), (8000 * MIB).to_le_bytes()].concat()),
                        ALLOC => {
                            let size = u64::from_le_bytes(body[4..12].try_into().unwrap());
                            let mut n = next.lock().unwrap();
                            *n += 0x1000;
                            Some([n.to_le_bytes(), size.to_le_bytes()].concat())
                        }
                        _ => None,
                    };
                    if let Some(r) = reply {
                        let mut v = (r.len() as u64).to_le_bytes().to_vec();
                        v.extend_from_slice(&r);
                        if s.write_all(&v).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    a
}

struct Llama(TcpStream);

impl Llama {
    async fn connect(ep: SocketAddr) -> Self {
        let mut s = Llama(TcpStream::connect(ep).await.unwrap());
        let r = s.call(HELLO, &[0xEE; 24]).await.unwrap();
        assert_eq!(&r[..4], &[9, 0, 0, 0]);
        assert_eq!(&r[4..], &[0u8; 24], "the helper's transport caps are hidden");
        s
    }
    async fn send(&mut self, cmd: u8, body: &[u8]) {
        let mut v = vec![cmd];
        v.extend_from_slice(&(body.len() as u64).to_le_bytes());
        v.extend_from_slice(body);
        self.0.write_all(&v).await.unwrap();
    }
    async fn call(&mut self, cmd: u8, body: &[u8]) -> std::io::Result<Vec<u8>> {
        self.send(cmd, body).await;
        let mut n = [0u8; 8];
        tokio::time::timeout(Duration::from_secs(5), self.0.read_exact(&mut n)).await.map_err(|_| std::io::Error::other("timeout"))??;
        let mut r = vec![0u8; u64::from_le_bytes(n) as usize];
        self.0.read_exact(&mut r).await?;
        Ok(r)
    }
    async fn free_mib(&mut self) -> (u64, u64) {
        let r = self.call(DEVMEM, &0u32.to_le_bytes()).await.unwrap();
        (u64::from_le_bytes(r[..8].try_into().unwrap()) / MIB, u64::from_le_bytes(r[8..].try_into().unwrap()) / MIB)
    }
    /// remote_ptr, 0 when refused.
    async fn alloc(&mut self, mib: u64) -> u64 {
        let mut b = 0u32.to_le_bytes().to_vec();
        b.extend_from_slice(&(mib * MIB).to_le_bytes());
        let r = self.call(ALLOC, &b).await.unwrap();
        u64::from_le_bytes(r[..8].try_into().unwrap())
    }
}

struct Rig {
    hooks: Arc<Hooks>,
    addr: String,
    _server: HelperServer,
}

/// Pair `name` with the helper (code from `codes`) and open a bridge.
async fn link(addr: &str, codes: &CodeSlot, name: &str) -> (Arc<Identity>, [u8; 32], RpcBridge) {
    let id = Arc::new(Identity::generate().unwrap());
    let p = client::pair(addr, &codes.new_code().code, &id, &PeerInfo { name: name.into(), ..Default::default() }).await.unwrap();
    let bridge = RpcBridge::start(addr.to_string(), id.clone(), p.key, Some("b1".into()), false).await.unwrap();
    (id, p.key, bridge)
}

async fn rig_with_codes(gpu_mib: u64) -> (Rig, Arc<CodeSlot>) {
    let helper_id = Arc::new(Identity::generate().unwrap());
    let codes = Arc::new(CodeSlot::new());
    let hooks = Arc::new(Hooks {
        pinned: Default::default(),
        rpc: fake_rpc_server().await,
        budgets: Default::default(),
        caps: Mutex::new(ShareLimits { gpu_mib, ram_mib: 0, threads: 2, paused: false }),
        refusals: Default::default(),
    });
    let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id, codes.clone(), hooks.clone()).await.unwrap();
    (Rig { hooks, addr: server.local_addr().to_string(), _server: server }, codes)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_helper_caps_memory_refuses_over_it_and_releases_on_disconnect() {
    let (r, codes) = rig_with_codes(2048).await;
    let (id, key, bridge) = link(&r.addr, &codes, "main").await;
    // The main PC's control channel gets the refusal message.
    let ctl = Control::connect(&r.addr, &id, key).await.unwrap();
    let state = ctl.state();

    let mut ll = Llama::connect(bridge.endpoint()).await;
    assert_eq!(ll.call(DEVICE_COUNT, &[]).await.unwrap(), 1u32.to_le_bytes());
    // --list-devices sees the 2 GB limit, not 7 GB free.
    assert_eq!(ll.free_mib().await, (2048, 2048));
    let p1 = ll.alloc(1500).await;
    assert!(p1 != 0);
    assert_eq!(ll.free_mib().await.0, 548);
    // Weights pass untouched (no reply).
    let mut st = vec![0u8; 305];
    st.extend(std::iter::repeat_n(3u8, 100_000));
    ll.send(SET_TENSOR, &st).await;
    // Over the limit: a null pointer, as when rpc-server runs out.
    assert_eq!(ll.alloc(600).await, 0);
    // The stream is still in step.
    assert!(ll.alloc(500).await != 0);
    let budget = r.hooks.budgets.lock().unwrap().values().next().unwrap().clone();
    assert_eq!(budget.used_bytes(DevKind::Gpu), 2000 * MIB);
    assert!(matches!(r.hooks.refusals.lock().unwrap()[0], Refusal::OverLimit { kind: DevKind::Gpu, bytes, .. } if bytes == 600 * MIB));
    let mut got = None;
    for _ in 0..50 {
        got = state.refusal();
        if got.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let got = got.expect("share_refused reached the main PC");
    assert_eq!((got["reason"].as_str(), got["kind"].as_str(), got["bytes"].as_u64()), (Some("over_limit"), Some("gpu"), Some(600 * MIB)));
    // FREE_BUFFER gives it back.
    ll.send(FREE, &p1.to_le_bytes()).await;
    assert_eq!(ll.free_mib().await.0, 1548);
    // A live limit change applies to the next allocation at once.
    budget.set_caps(ShareLimits { gpu_mib: 1024, ram_mib: 0, threads: 2, paused: false });
    assert_eq!(ll.free_mib().await, (524, 1024));
    assert_eq!(ll.alloc(600).await, 0);
    // Disconnect: everything it had comes back.
    drop(ll);
    for _ in 0..100 {
        if budget.used_bytes(DevKind::Gpu) == 0 && budget.open_streams() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!((budget.used_bytes(DevKind::Gpu), budget.open_streams()), (0, 0));
    drop(ctl);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_pc_is_refused_while_one_is_helped() {
    let (r, codes) = rig_with_codes(4096).await;
    let (_a, _ka, bridge_a) = link(&r.addr, &codes, "Desk").await;
    let (_b, _kb, bridge_b) = link(&r.addr, &codes, "Laptop").await;
    let mut a = Llama::connect(bridge_a.endpoint()).await;
    assert!(a.alloc(100).await != 0);
    // B's stream is refused before a byte reaches rpc-server.
    let mut s = TcpStream::connect(bridge_b.endpoint()).await.unwrap();
    s.write_all(&[HELLO]).await.unwrap();
    s.write_all(&24u64.to_le_bytes()).await.unwrap();
    s.write_all(&[0u8; 24]).await.unwrap();
    let mut buf = [0u8; 8];
    let r2 = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf)).await.expect("closed, not hung");
    assert!(matches!(r2, Ok(0) | Err(_)), "{r2:?}");
    let err = bridge_b.snapshot().last_error.unwrap_or_default();
    assert!(err.contains("busy helping"), "{err}");
    // Once A is done, B gets in.
    drop(a);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut b = Llama::connect(bridge_b.endpoint()).await;
    assert_eq!(b.free_mib().await.0, 4096);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn another_protocol_is_refused_and_the_main_pc_is_told() {
    // An rpc-server that answers HELLO with protocol 11.
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let rpc = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut head = [0u8; 9 + 24];
                if s.read_exact(&mut head).await.is_ok() {
                    let mut v = 28u64.to_le_bytes().to_vec();
                    v.extend_from_slice(&[11, 0, 0, 0]);
                    v.extend_from_slice(&[0u8; 24]);
                    let _ = s.write_all(&v).await;
                    let _ = s.read(&mut [0u8; 16]).await;
                }
            });
        }
    });
    let (r, codes) = rig_with_codes(4096).await;
    let hooks = Arc::new(Hooks { pinned: Mutex::new(r.hooks.pinned.lock().unwrap().clone()), rpc, budgets: Default::default(), caps: Mutex::new(*r.hooks.caps.lock().unwrap()), refusals: Default::default() });
    let helper_id = Arc::new(Identity::generate().unwrap());
    let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id, codes.clone(), hooks.clone()).await.unwrap();
    let addr = server.local_addr().to_string();
    let (id, key, bridge) = link(&addr, &codes, "main").await;
    let ctl = Control::connect(&addr, &id, key).await.unwrap();
    let mut s = TcpStream::connect(bridge.endpoint()).await.unwrap();
    s.write_all(&[HELLO]).await.unwrap();
    s.write_all(&24u64.to_le_bytes()).await.unwrap();
    s.write_all(&[0u8; 24]).await.unwrap();
    // The reply never reaches llama-server: the stream closes.
    let mut buf = [0u8; 64];
    let mut got = 0;
    loop {
        match tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf)).await.expect("closed, not hung") {
            Ok(0) | Err(_) => break,
            Ok(n) => got += n,
        }
    }
    assert!(got < 8 + 28, "the protocol-11 HELLO reply was not passed on whole ({got} bytes)");
    assert_eq!(hooks.refusals.lock().unwrap().as_slice(), &[Refusal::Incompatible { proto_major: 11 }]);
    let mut seen = None;
    for _ in 0..50 {
        seen = ctl.state().refusal();
        if seen.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(seen.expect("told")["reason"], "incompatible");
    drop(r);
}

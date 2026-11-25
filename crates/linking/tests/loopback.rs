//! Helper and main PC in one process over loopback: pairing, pinning, the
//! control channel and a fake rpc-server stream through the encrypted tunnel.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use linking::client::{self, Control, RpcBridge};
use linking::code::CodeSlot;
use linking::handshake;
use linking::helper::{HelperHooks, HelperServer, HookFuture, RpcTarget, SessionEvent};
use linking::wire::{self, Mode};
use linking::{Identity, LinkError, PeerInfo};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Default)]
struct Hooks {
    pinned: Mutex<HashMap<[u8; 32], String>>,
    rpc: Mutex<Option<SocketAddr>>,
    tag: Option<String>,
    events: Mutex<Vec<SessionEvent>>,
}

impl HelperHooks for Hooks {
    fn peer_name(&self, key: &[u8; 32]) -> Option<String> {
        self.pinned.lock().unwrap().get(key).cloned()
    }
    fn paired(&self, key: [u8; 32], info: &PeerInfo, _addr: SocketAddr) {
        self.pinned.lock().unwrap().insert(key, info.name.clone());
    }
    fn unlinked(&self, key: [u8; 32]) {
        self.pinned.lock().unwrap().remove(&key);
    }
    fn info(&self) -> PeerInfo {
        PeerInfo { name: "helper-pc".into(), helping: self.rpc.lock().unwrap().is_some(), runtime_tag: self.tag.clone(), ..Default::default() }
    }
    fn rpc_target<'a>(&'a self, _key: &'a [u8; 32]) -> HookFuture<'a, Result<RpcTarget, LinkError>> {
        // The fake rpc-server here is not ggml-rpc: no guard.
        let r = self.rpc.lock().unwrap().ok_or(LinkError::HelperOff).map(|addr| RpcTarget { addr, limits: None });
        Box::pin(async move { r })
    }
    fn session(&self, ev: SessionEvent) {
        self.events.lock().unwrap().push(ev);
    }
}

struct Rig {
    helper_id: Arc<Identity>,
    codes: Arc<CodeSlot>,
    hooks: Arc<Hooks>,
    server: HelperServer,
}

async fn rig(tag: Option<&str>) -> Rig {
    let helper_id = Arc::new(Identity::generate().unwrap());
    let codes = Arc::new(CodeSlot::new());
    let hooks = Arc::new(Hooks { tag: tag.map(String::from), ..Default::default() });
    let server = HelperServer::bind("127.0.0.1:0".parse().unwrap(), helper_id.clone(), codes.clone(), hooks.clone()).await.unwrap();
    Rig { helper_id, codes, hooks, server }
}

fn me() -> PeerInfo {
    PeerInfo { name: "main-pc".into(), listen_port: Some(47070), ..Default::default() }
}

/// A fake ggml-rpc-server: echoes every byte back with each byte + 1.
async fn fake_rpc_server() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 100_000];
                loop {
                    let n = s.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    for b in &mut buf[..n] {
                        *b = b.wrapping_add(1);
                    }
                    if s.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            });
        }
    });
    a
}

#[tokio::test]
async fn right_code_links_and_both_sides_pin_keys() {
    let r = rig(Some("b12250")).await;
    let code = r.codes.new_code().code;
    let main_id = Identity::generate().unwrap();
    let addr = r.server.local_addr().to_string();
    let p = client::pair(&addr, &format!("{} {}", &code[..3], &code[3..]), &main_id, &me()).await.unwrap();
    assert_eq!(&p.key, r.helper_id.public(), "main PC pinned the helper's real key");
    assert_eq!(p.info.name, "helper-pc");
    assert_eq!(p.info.runtime_tag.as_deref(), Some("b12250"));
    assert_eq!(r.hooks.peer_name(main_id.public()).as_deref(), Some("main-pc"), "helper pinned the main PC");
    // Reconnect with the pinned keys, no code.
    let mut c = Control::connect(&addr, &main_id, p.key).await.unwrap();
    assert_eq!(c.info().name, "helper-pc");
    assert!(c.rtt(3).await.unwrap().as_millis() < 1000);
    let mbps = c.bandwidth(2 << 20, std::time::Duration::from_secs(10)).await.unwrap();
    assert!(mbps > 1.0, "{mbps}");
    assert!(!c.refresh_info().await.unwrap().helping);
    c.close().await;
}

#[tokio::test]
async fn wrong_code_fails_and_five_failures_lock() {
    let r = rig(None).await;
    let code = r.codes.new_code().code;
    let wrong = if code == "000000" { "111111".to_string() } else { "000000".to_string() };
    let addr = r.server.local_addr().to_string();
    let main_id = Identity::generate().unwrap();
    for i in 1..=4u32 {
        let e = client::pair(&addr, &wrong, &main_id, &me()).await.unwrap_err();
        assert!(matches!(e, LinkError::WrongCode { remaining } if remaining == 5 - i), "{e:?}");
    }
    let e = client::pair(&addr, &wrong, &main_id, &me()).await.unwrap_err();
    assert!(matches!(e, LinkError::CodeLocked), "{e:?}");
    // Locked: even the right code is refused now.
    let e = client::pair(&addr, &code, &main_id, &me()).await.unwrap_err();
    assert!(matches!(e, LinkError::CodeLocked), "{e:?}");
    assert!(r.hooks.pinned.lock().unwrap().is_empty(), "nothing pinned");
}

#[tokio::test]
async fn an_old_code_cannot_be_replayed() {
    let r = rig(None).await;
    let addr = r.server.local_addr().to_string();
    let code = r.codes.new_code().code;
    client::pair(&addr, &code, &Identity::generate().unwrap(), &me()).await.unwrap();
    // Same code again (used): refused.
    let e = client::pair(&addr, &code, &Identity::generate().unwrap(), &me()).await.unwrap_err();
    assert!(matches!(e, LinkError::NoCode), "{e:?}");
    // A new code was made: the old one is just a wrong code.
    let fresh = r.codes.new_code().code;
    if fresh != code {
        let e = client::pair(&addr, &code, &Identity::generate().unwrap(), &me()).await.unwrap_err();
        assert!(matches!(e, LinkError::WrongCode { .. }), "{e:?}");
    }
    assert_eq!(r.hooks.pinned.lock().unwrap().len(), 1);
}

/// A relay that answers the pairing itself with its own key: it can't know
/// the code, so its SPAKE2 key differs and the joiner learns nothing.
#[tokio::test]
async fn man_in_the_middle_with_a_different_key_fails() {
    // The attacker runs its own "helper" with a guessed code in front of the victim.
    let attacker = rig(None).await;
    attacker.codes.new_code();
    let real = rig(None).await;
    let real_code = real.codes.new_code().code;
    let main_id = Identity::generate().unwrap();
    // The joiner types the real helper's code but reaches the attacker.
    let e = client::pair(&attacker.server.local_addr().to_string(), &real_code, &main_id, &me()).await;
    match e {
        Err(LinkError::WrongCode { .. }) => {}
        Ok(_) => panic!("1-in-a-million: attacker's random code equals the real one; rerun"),
        Err(other) => panic!("{other:?}"),
    }
    assert!(attacker.hooks.pinned.lock().unwrap().is_empty());
    // After a real link, a different key at the pinned address is refused.
    let p = client::pair(&real.server.local_addr().to_string(), &real_code, &main_id, &me()).await.unwrap();
    let e = Control::connect(&attacker.server.local_addr().to_string(), &main_id, p.key).await.err().unwrap();
    assert!(matches!(e, LinkError::KeyMismatch), "{e:?}");
    // And a PC whose key the helper never pinned is refused by the helper.
    let stranger = Identity::generate().unwrap();
    let e = Control::connect(&real.server.local_addr().to_string(), &stranger, p.key).await.err().unwrap();
    assert!(matches!(e, LinkError::NotLinked), "{e:?}");
}

#[tokio::test]
async fn fake_rpc_stream_goes_through_the_tunnel_only_when_helping() {
    let r = rig(Some("b12250")).await;
    let addr = r.server.local_addr().to_string();
    let main_id = Arc::new(Identity::generate().unwrap());
    let p = client::pair(&addr, &r.codes.new_code().code, &main_id, &me()).await.unwrap();

    // Helper off: the stream is refused before anything is forwarded.
    let e = client::connect(&addr, &main_id, p.key, Mode::Rpc, serde_json::json!({ "runtime_tag": "b12250" })).await.err().unwrap();
    assert!(matches!(e, LinkError::HelperOff), "{e:?}");

    *r.hooks.rpc.lock().unwrap() = Some(fake_rpc_server().await);
    // Different llama.cpp build: refused with a clear message.
    let e = client::connect(&addr, &main_id, p.key, Mode::Rpc, serde_json::json!({ "runtime_tag": "b12246" })).await.err().unwrap();
    assert!(e.to_string().contains("same runtime build"), "{e}");

    for compress in [false, true] {
        let bridge = RpcBridge::start(addr.clone(), main_id.clone(), p.key, Some("b12250".into()), compress).await.unwrap();
        let mut s = TcpStream::connect(bridge.endpoint()).await.unwrap();
        // 300 KB: several tunnel frames each way.
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let (mut rd, mut wr) = s.split();
        let send = async {
            wr.write_all(&data).await.unwrap();
            wr.shutdown().await.unwrap();
        };
        let mut back = Vec::new();
        let recv = async { rd.read_to_end(&mut back).await.unwrap() };
        tokio::join!(send, recv);
        assert_eq!(back.len(), data.len());
        assert!(back.iter().zip(&data).all(|(b, d)| *b == d.wrapping_add(1)), "bytes changed in the tunnel");
        let snap = bridge.snapshot();
        assert!(snap.last_error.is_none(), "{:?}", snap.last_error);
        assert_eq!(snap.streams, 1);
    }
    assert!(r.hooks.events.lock().unwrap().iter().any(|e| e.mode == "rpc" && e.open));

    // Unlink from the main PC: the helper forgets the key.
    let mut c = Control::connect(&addr, &main_id, p.key).await.unwrap();
    c.unlink().await.unwrap();
    assert!(r.hooks.peer_name(main_id.public()).is_none());
    let e = Control::connect(&addr, &main_id, p.key).await.err().unwrap();
    assert!(matches!(e, LinkError::NotLinked), "{e:?}");
}

#[tokio::test]
async fn junk_on_the_port_is_rejected_without_a_crash() {
    let r = rig(None).await;
    let mut s = TcpStream::connect(r.server.local_addr()).await.unwrap();
    s.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), s.read_to_end(&mut buf)).await;
    // The server is still up and pairs normally afterwards.
    let code = r.codes.new_code().code;
    client::pair(&r.server.local_addr().to_string(), &code, &Identity::generate().unwrap(), &me()).await.unwrap();
    // A raw hello with an unknown mode gets a status error.
    let mut s = TcpStream::connect(r.server.local_addr()).await.unwrap();
    let mut h = wire::hello(Mode::Control);
    h[8] = 9;
    wire::write_frame(&mut s, &h).await.unwrap();
    let mut f = Vec::new();
    wire::read_frame(&mut s, &mut f).await.unwrap();
    assert!(wire::parse_status(&f).is_err());
    let _ = handshake::read_accept;
}

#[tokio::test]
async fn helper_state_changes_and_closes_reach_the_main_pc_at_once() {
    let r = rig(Some("b12250")).await;
    let addr = r.server.local_addr().to_string();
    let main_id = Identity::generate().unwrap();
    let p = client::pair(&addr, &r.codes.new_code().code, &main_id, &me()).await.unwrap();
    let c = Control::connect(&addr, &main_id, p.key).await.unwrap();
    let state = c.state();
    assert!(!state.info().helping);
    // Helper turns helping on: pushed without a request.
    *r.hooks.rpc.lock().unwrap() = Some("127.0.0.1:9".parse().unwrap());
    r.server.notify_changed();
    let t = std::time::Instant::now();
    while !state.info().helping {
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "info_update not received");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    // Helper stops listening (helping off): the main PC sees the close at once.
    r.server.stop();
    let why = tokio::time::timeout(std::time::Duration::from_secs(2), state.wait_closed()).await.expect("close not noticed");
    assert!(!why.is_empty());
    let mut c = c;
    assert!(matches!(c.ping().await, Err(LinkError::Disconnected(_))));
}

#[tokio::test]
async fn refused_and_closed_early_are_told_apart() {
    // Nothing listens: "nothing is listening".
    let free = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap();
    let id = Identity::generate().unwrap();
    let e = client::pair(&free.to_string(), "123456", &id, &me()).await.unwrap_err();
    assert!(matches!(e, LinkError::NotListening(_)), "{e:?}");
    assert!(e.to_string().contains("nothing is listening"), "{e}");
    // Something accepts and closes (a port forward with nothing behind it).
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (s, _) = l.accept().await.unwrap();
            drop(s);
        }
    });
    let e = client::pair(&a.to_string(), "123456", &id, &me()).await.unwrap_err();
    assert!(matches!(e, LinkError::ClosedEarly(_)), "{e:?}");
    // Wording for the last try.
    assert!(LinkError::WrongCode { remaining: 1 }.to_string().contains("1 try left"));
    assert!(LinkError::WrongCode { remaining: 3 }.to_string().contains("3 tries left"));
}

/// The bridge's sent counter moves while a stream is still open (a model
/// load's progress reads it), and ends at exactly the payload size.
#[tokio::test]
async fn bridge_counts_bytes_sent_while_the_stream_is_open() {
    use std::sync::atomic::Ordering;
    let r = rig(Some("b12250")).await;
    let addr = r.server.local_addr().to_string();
    let main_id = Arc::new(Identity::generate().unwrap());
    let p = client::pair(&addr, &r.codes.new_code().code, &main_id, &me()).await.unwrap();
    *r.hooks.rpc.lock().unwrap() = Some(fake_rpc_server().await);
    let bridge = RpcBridge::start(addr.clone(), main_id.clone(), p.key, Some("b12250".into()), false).await.unwrap();
    let counter = bridge.sent_counter();
    assert_eq!(counter.load(Ordering::Relaxed), 0);

    let mut s = TcpStream::connect(bridge.endpoint()).await.unwrap();
    let (mut rd, mut wr) = s.split();
    let chunk = vec![7u8; 256 * 1024];
    let total = 8 * chunk.len() as u64;
    let drain = async {
        let mut buf = vec![0u8; 64 * 1024];
        let mut n = 0u64;
        while n < total {
            let k = rd.read(&mut buf).await.unwrap();
            assert!(k > 0, "stream closed early");
            n += k as u64;
        }
    };
    let send = async {
        for _ in 0..8 {
            wr.write_all(&chunk).await.unwrap();
        }
        // Everything written is counted before the stream closes.
        let t = std::time::Instant::now();
        while counter.load(Ordering::Relaxed) < total && t.elapsed().as_secs() < 10 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(counter.load(Ordering::Relaxed), total, "counter while the stream is still open");
        assert_eq!(bridge.snapshot().active, 1, "stream still open when counted");
        wr.shutdown().await.unwrap();
    };
    tokio::join!(send, drain);
    assert_eq!(bridge.snapshot().sent_bytes, total);
}


/// A helper that answers pings and info but never reads a bandwidth test
/// (a stalled or blackholed link). Pairing works normally.
async fn stalling_helper(id: Arc<Identity>, codes: Arc<CodeSlot>, tag: &'static str) -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            let (id, codes) = (id.clone(), codes.clone());
            tokio::spawn(async move {
                let info = PeerInfo { name: "stalling".into(), helping: true, runtime_tag: Some(tag.into()), ..Default::default() };
                let mut h = Vec::new();
                if wire::read_frame(&mut s, &mut h).await.is_err() {
                    return;
                }
                match wire::parse_hello(&h) {
                    Ok(Mode::Pair) => {
                        let i = info.clone();
                        let _ = handshake::pair_respond(s, &id, &h, &codes, move |_, _| i).await;
                    }
                    Ok(Mode::Control) => {
                        let Ok(mut sess) = handshake::link_respond(s, &id, &h).await else { return };
                        let _ = handshake::send_accept(&mut sess.writer, serde_json::json!({ "info": info })).await;
                        loop {
                            match sess.reader.recv().await {
                                Ok(wire::Frame::Ping(n)) => {
                                    let _ = sess.writer.send(&wire::Frame::Pong(n)).await;
                                }
                                Ok(wire::Frame::Json(v)) if v["t"] == "info" => {
                                    let _ = sess.writer.send(&wire::Frame::Json(serde_json::json!({ "t": "info", "info": info }))).await;
                                }
                                Ok(wire::Frame::Json(v)) if v["t"] == "bw" => {
                                    // Stall: stop reading, keep the socket open.
                                    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                                }
                                Ok(_) => {}
                                Err(_) => return,
                            }
                        }
                    }
                    _ => {}
                }
            });
        }
    });
    a
}

#[tokio::test]
async fn a_stalled_bandwidth_test_gives_up_within_its_limit() {
    let id = Arc::new(Identity::generate().unwrap());
    let codes = Arc::new(CodeSlot::new());
    let addr = stalling_helper(id.clone(), codes.clone(), "b1").await.to_string();
    let main_id = Identity::generate().unwrap();
    let p = client::pair(&addr, &codes.new_code().code, &main_id, &me()).await.unwrap();
    let mut c = Control::connect(&addr, &main_id, p.key).await.unwrap();
    let t = std::time::Instant::now();
    let r = c.bandwidth(64 << 20, std::time::Duration::from_secs(2)).await;
    assert!(matches!(r, Err(LinkError::Timeout)), "{r:?}");
    assert!(t.elapsed() < std::time::Duration::from_secs(4), "{:?}", t.elapsed());
}

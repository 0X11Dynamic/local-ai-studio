//! The helper's side of a forwarded ggml-rpc stream: enforces what this PC
//! lends to one linked PC ("so you aren't accidentally botnetting the other
//! computers"). ggml-rpc-server has no memory limit of its own, so the limit
//! is applied here, in the tunnel, before a request reaches it:
//!
//! - `GET_DEVICE_MEMORY` replies are lowered to what the linked PC may still
//!   use (`free = min(free, cap - used)`, `total = min(total, cap)`), so its
//!   `--list-devices`, `--tensor-split` and `--fit` see the limit;
//! - an `ALLOC_BUFFER` that would go over the limit is not forwarded: the
//!   linked PC gets the same null reply rpc-server sends when an allocation
//!   fails (`remote_ptr = 0`), which llama.cpp reports as a failed load,
//!   not a crash;
//! - `FREE_BUFFER`, a failed allocation and the end of the stream give the
//!   memory back (rpc-server frees every buffer of a client that disconnects).
//!
//! Protocols 9 and 10 (llama.cpp b12250 and b12716, `ggml/src/ggml-rpc/ggml-rpc.cpp`), framing
//! as in [`crate::rpc_tap`]. Fail closed: a stream that doesn't start with a
//! protocol-9 or -10 HELLO, a reply nobody asked for, or a size that isn't the one
//! protocols 9 and 10 use for a command this guard reads ends the stream; another
//! protocol major in the HELLO reply ends it with [`GuardError::Incompatible`].
//!
//! Untrusted input: every length comes from the wire, so nothing here
//! allocates by it (held frames are at most 28 bytes) and nothing indexes
//! past what was checked. `SET_TENSOR` bodies (the weights) and every other
//! command's body are looked at in place and forwarded as they are: no copy.
//!
//! Not counted: rpc-server's own host RAM (receive buffers, graph
//! deserialisation) and its disk cache, so the RAM limit is approximate.
//! The memory limit covers the sum over all shared GPUs; each GPU reports at
//! most what is left of it.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::Serialize;

use crate::ShareLimits;
use crate::rpc_tap::{
    CMD_ALLOC_BUFFER, CMD_FREE_BUFFER, CMD_GET_DEVICE_MEMORY, CMD_HELLO, REQ_HEADER, RSP_HEADER, has_response, known_major,
};

/// `rpc_msg_hello_req { u8 conn_caps[24] }`.
pub const HELLO_REQ_SIZE: u64 = 24;
/// `rpc_msg_hello_rsp { u8 major, minor, patch, flags; u8 conn_caps[24] }`.
pub const HELLO_RSP_SIZE: u64 = 28;
/// `rpc_msg_alloc_buffer_req { u32 device; u64 size }` (packed).
const ALLOC_REQ_SIZE: u64 = 12;
/// `rpc_msg_alloc_buffer_rsp { u64 remote_ptr; u64 remote_size }`.
const ALLOC_RSP_SIZE: u64 = 16;
/// `rpc_msg_get_device_memory_req { u32 device }`.
const DEVMEM_REQ_SIZE: u64 = 4;
/// `rpc_msg_get_device_memory_rsp { u64 free_mem; u64 total_mem }`.
const DEVMEM_RSP_SIZE: u64 = 16;
/// `rpc_msg_free_buffer_req { u64 remote_ptr }`.
const FREE_REQ_SIZE: u64 = 8;
/// The whole held ALLOC_BUFFER request (header + body).
const ALLOC_FRAME: usize = REQ_HEADER + ALLOC_REQ_SIZE as usize;

/// Which memory a device of the helper's rpc-server allocates from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DevKind {
    /// VRAM (CUDA, Vulkan, ... devices).
    Gpu,
    /// System RAM (the CPU device).
    Ram,
}

impl DevKind {
    /// From an rpc-server device name ("CUDA0", "Vulkan0", "CPU").
    pub fn of(device_name: &str) -> Self {
        if device_name.trim().to_ascii_uppercase().starts_with("CPU") { DevKind::Ram } else { DevKind::Gpu }
    }
    fn idx(self) -> usize {
        match self {
            DevKind::Gpu => 0,
            DevKind::Ram => 1,
        }
    }
}

/// Why the helper refused something (shown on both PCs).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Refusal {
    /// An allocation would have gone over the limit.
    OverLimit { kind: DevKind, bytes: u64, cap_bytes: u64, used_bytes: u64 },
    /// The linked PC speaks another ggml-rpc protocol: limits can't be enforced.
    Incompatible { proto_major: u8 },
}

/// What one linked PC may use and is using on this PC, shared by all its
/// streams. Allocations are counted in bytes, as rpc-server reports them.
#[derive(Debug)]
pub struct PeerBudget {
    caps: RwLock<ShareLimits>,
    /// Bytes in use, by [`DevKind`] (GPU, RAM).
    used: Mutex<[u64; 2]>,
    /// Open streams (each holds a [`Claim`]).
    streams: AtomicU64,
    refused: AtomicU64,
    refused_bytes: AtomicU64,
    last_refusal: Mutex<Option<(Refusal, std::time::SystemTime)>>,
}

impl PeerBudget {
    pub fn new(caps: ShareLimits) -> Arc<Self> {
        Arc::new(Self {
            caps: RwLock::new(caps),
            used: Mutex::new([0; 2]),
            streams: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            refused_bytes: AtomicU64::new(0),
            last_refusal: Mutex::new(None),
        })
    }

    pub fn caps(&self) -> ShareLimits {
        *self.caps.read().unwrap()
    }

    /// New limits apply to the next allocation at once; memory already
    /// allocated is never taken back.
    pub fn set_caps(&self, caps: ShareLimits) {
        *self.caps.write().unwrap() = caps;
    }

    pub fn used_bytes(&self, kind: DevKind) -> u64 {
        self.used.lock().unwrap()[kind.idx()]
    }

    /// What is left of the limit for `kind`.
    pub fn remaining(&self, kind: DevKind) -> u64 {
        self.caps().cap_bytes(kind).saturating_sub(self.used_bytes(kind))
    }

    pub fn open_streams(&self) -> u64 {
        self.streams.load(Ordering::Relaxed)
    }

    /// (refusals, bytes refused, the last one and when).
    pub fn refusals(&self) -> (u64, u64, Option<(Refusal, std::time::SystemTime)>) {
        (self.refused.load(Ordering::Relaxed), self.refused_bytes.load(Ordering::Relaxed), self.last_refusal.lock().unwrap().clone())
    }

    /// Count a stream as open until the claim is dropped.
    pub fn claim(self: &Arc<Self>) -> Claim {
        self.streams.fetch_add(1, Ordering::Relaxed);
        Claim(self.clone())
    }

    /// Reserve `size` bytes, or (cap, used) when that would go over the limit.
    fn try_reserve(&self, kind: DevKind, size: u64) -> Result<(), (u64, u64)> {
        let cap = self.caps().cap_bytes(kind);
        let mut u = self.used.lock().unwrap();
        let used = u[kind.idx()];
        match used.checked_add(size) {
            Some(t) if t <= cap => {
                u[kind.idx()] = t;
                Ok(())
            }
            _ => Err((cap, used)),
        }
    }

    fn release(&self, kind: DevKind, size: u64) {
        let mut u = self.used.lock().unwrap();
        u[kind.idx()] = u[kind.idx()].saturating_sub(size);
    }

    /// rpc-server allocated `actual` for a reservation of `reserved` (it may
    /// round up to its alignment).
    fn settle(&self, kind: DevKind, reserved: u64, actual: u64) {
        let mut u = self.used.lock().unwrap();
        u[kind.idx()] = u[kind.idx()].saturating_sub(reserved).saturating_add(actual);
    }

    fn record(&self, r: &Refusal) {
        self.refused.fetch_add(1, Ordering::Relaxed);
        if let Refusal::OverLimit { bytes, .. } = r {
            self.refused_bytes.fetch_add(*bytes, Ordering::Relaxed);
        }
        *self.last_refusal.lock().unwrap() = Some((r.clone(), std::time::SystemTime::now()));
    }
}

/// One open stream of a linked PC (see [`PeerBudget::claim`]).
#[derive(Debug)]
pub struct Claim(Arc<PeerBudget>);

impl Claim {
    pub fn budget(&self) -> &Arc<PeerBudget> {
        &self.0
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.0.streams.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GuardError {
    /// Another ggml-rpc protocol major: this PC can't enforce its limits.
    #[error("the linked PC uses ggml-rpc protocol {0}, but this PC can only enforce its sharing limits with protocols 9 and 10")]
    Incompatible(u8),
    /// The stream isn't ggml-rpc protocol 9 or 10 as this guard knows it.
    #[error("unexpected ggml-rpc stream: {0}")]
    Protocol(&'static str),
}

type OnRefuse = Arc<dyn Fn(&Refusal) + Send + Sync>;

#[derive(Debug, Clone, Copy)]
enum Pending {
    Hello,
    /// Reply to GET_DEVICE_MEMORY for a device of this kind (None: an index
    /// rpc-server doesn't have; it drops the connection).
    DevMem(Option<DevKind>),
    Alloc { kind: DevKind, size: u64 },
    Other,
}

/// State both directions of one stream share.
struct State {
    budget: Arc<PeerBudget>,
    /// Replies rpc-server owes, in request order. The front one is popped
    /// when its reply has been passed on completely.
    pending: VecDeque<Pending>,
    /// Live allocations of this stream: remote_ptr -> (kind, bytes).
    allocs: HashMap<u64, (DevKind, u64)>,
    _claim: Claim,
}

impl Drop for State {
    fn drop(&mut self) {
        // rpc-server frees every buffer of a client that disconnects.
        for (_, (kind, n)) in self.allocs.drain() {
            self.budget.release(kind, n);
        }
        // Reserved but never answered.
        for p in self.pending.drain(..) {
            if let Pending::Alloc { kind, size } = p {
                self.budget.release(kind, size);
            }
        }
    }
}

/// Make the two halves of a guard for one stream. `kinds` are the helper's
/// rpc-server devices in its device order (index = ggml-rpc device id).
pub fn guard(claim: Claim, kinds: Arc<Vec<DevKind>>, on_refuse: Option<OnRefuse>) -> (Upstream, Downstream) {
    let budget = claim.budget().clone();
    let dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let st = Arc::new(Mutex::new(State { budget: budget.clone(), pending: VecDeque::new(), allocs: HashMap::new(), _claim: claim }));
    (
        Upstream {
            st: st.clone(),
            budget: budget.clone(),
            kinds,
            on_refuse: on_refuse.clone(),
            head: [0; REQ_HEADER],
            head_len: 0,
            cmd: 0,
            left: 0,
            in_body: false,
            small: [0; ALLOC_FRAME],
            small_len: 0,
            first: true,
            dead: dead.clone(),
        },
        Downstream { st, budget, on_refuse, head: [0; RSP_HEADER], head_len: 0, left: 0, at: 0, cur: None, small: [0; DEVMEM_RSP_SIZE as usize], small_len: 0, dead },
    )
}

/// Builds what to forward from one chunk: nothing is copied while every byte
/// so far passes unchanged (the chunk itself is forwarded, edited in place);
/// from the first held byte on, `out` gets the bytes to forward instead.
struct Emit<'a> {
    diverted: bool,
    out: &'a mut Vec<u8>,
}

impl Emit<'_> {
    fn divert(&mut self, d: &[u8], at: usize) {
        if !self.diverted {
            self.diverted = true;
            self.out.clear();
            self.out.extend_from_slice(&d[..at]);
        }
    }
    fn pass(&mut self, d: &[u8], from: usize, to: usize) {
        if self.diverted {
            self.out.extend_from_slice(&d[from..to]);
        }
    }
}

/// What to forward after one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fwd {
    /// The chunk as it is now (it may have been edited in place).
    Chunk,
    /// The bytes in `out` instead (possibly none).
    Out,
}

/// Requests (tunnel -> rpc-server).
pub struct Upstream {
    st: Arc<Mutex<State>>,
    budget: Arc<PeerBudget>,
    kinds: Arc<Vec<DevKind>>,
    on_refuse: Option<OnRefuse>,
    head: [u8; REQ_HEADER],
    head_len: usize,
    cmd: u8,
    /// Body bytes still to come.
    left: u64,
    in_body: bool,
    /// A held ALLOC_BUFFER frame, or the body of a small request read on the way.
    small: [u8; ALLOC_FRAME],
    small_len: usize,
    /// The next frame is the stream's first (must be HELLO).
    first: bool,
    /// Either direction failed: nothing more passes (shared by both halves).
    dead: Arc<std::sync::atomic::AtomicBool>,
}

impl Upstream {
    /// Look at request bytes before they are forwarded. `d` may be edited in
    /// place. Returns what to forward to rpc-server; `inject` gets bytes to
    /// send back to the linked PC as if rpc-server had answered (a refused
    /// allocation). After an error nothing more may be forwarded: close.
    pub fn feed(&mut self, d: &mut [u8], out: &mut Vec<u8>, inject: &mut Vec<u8>) -> Result<Fwd, GuardError> {
        if self.dead.load(Ordering::Relaxed) {
            return Err(GuardError::Protocol("stream already failed"));
        }
        let r = self.feed_inner(d, out, inject);
        if r.is_err() {
            self.dead.store(true, Ordering::Relaxed);
        }
        r
    }

    fn feed_inner(&mut self, d: &mut [u8], out: &mut Vec<u8>, inject: &mut Vec<u8>) -> Result<Fwd, GuardError> {
        let mut e = Emit { diverted: false, out };
        let mut i = 0;
        while i < d.len() {
            if !self.in_body {
                if self.head_len == 0 {
                    let cmd = d[i];
                    if self.first && cmd != CMD_HELLO {
                        return Err(GuardError::Protocol("the stream did not start with HELLO"));
                    }
                    self.first = false;
                    self.cmd = cmd;
                    self.small_len = 0;
                }
                let take = (REQ_HEADER - self.head_len).min(d.len() - i);
                self.head[self.head_len..self.head_len + take].copy_from_slice(&d[i..i + take]);
                if self.cmd == CMD_ALLOC_BUFFER {
                    e.divert(d, i);
                    self.small[self.small_len..self.small_len + take].copy_from_slice(&d[i..i + take]);
                    self.small_len += take;
                } else {
                    e.pass(d, i, i + take);
                }
                self.head_len += take;
                i += take;
                if self.head_len < REQ_HEADER {
                    continue;
                }
                self.head_len = 0;
                let size = u64::from_le_bytes(self.head[1..9].try_into().unwrap());
                let want = match self.cmd {
                    CMD_HELLO => Some(HELLO_REQ_SIZE),
                    CMD_ALLOC_BUFFER => Some(ALLOC_REQ_SIZE),
                    CMD_GET_DEVICE_MEMORY => Some(DEVMEM_REQ_SIZE),
                    CMD_FREE_BUFFER => Some(FREE_REQ_SIZE),
                    _ => None,
                };
                if want.is_some_and(|w| w != size) {
                    return Err(GuardError::Protocol("a request has a size protocols 9 and 10 don't use"));
                }
                self.left = size;
                self.in_body = true;
                if self.left == 0 {
                    self.in_body = false;
                    self.end(d, i, &mut e, inject)?;
                }
                continue;
            }
            let n = self.left.min((d.len() - i) as u64) as usize;
            match self.cmd {
                // conn_caps: no transport upgrade (RDMA), every byte stays on this TCP stream.
                CMD_HELLO => {
                    d[i..i + n].fill(0);
                    e.pass(d, i, i + n);
                }
                CMD_ALLOC_BUFFER => {
                    e.divert(d, i);
                    self.small[self.small_len..self.small_len + n].copy_from_slice(&d[i..i + n]);
                    self.small_len += n;
                }
                CMD_GET_DEVICE_MEMORY | CMD_FREE_BUFFER => {
                    self.small[self.small_len..self.small_len + n].copy_from_slice(&d[i..i + n]);
                    self.small_len += n;
                    e.pass(d, i, i + n);
                }
                _ => e.pass(d, i, i + n),
            }
            self.left -= n as u64;
            i += n;
            if self.left == 0 {
                self.in_body = false;
                self.end(d, i, &mut e, inject)?;
            }
        }
        Ok(if e.diverted { Fwd::Out } else { Fwd::Chunk })
    }

    /// A whole request has been seen (`d[..at]` processed).
    fn end(&mut self, d: &[u8], at: usize, e: &mut Emit, inject: &mut Vec<u8>) -> Result<(), GuardError> {
        let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        match self.cmd {
            CMD_ALLOC_BUFFER => {
                let frame = self.small;
                let dev = u32_at(&frame, REQ_HEADER) as usize;
                let size = u64_at(&frame, REQ_HEADER + 4);
                let mut st = self.st.lock().unwrap();
                if !st.pending.is_empty() {
                    // ggml-rpc waits for every reply before the next request
                    // that has one; a refusal could not be answered in order.
                    return Err(GuardError::Protocol("ALLOC_BUFFER while another reply is due"));
                }
                let Some(&kind) = self.kinds.get(dev) else {
                    // A device rpc-server doesn't have: it drops the connection.
                    st.pending.push_back(Pending::Other);
                    drop(st);
                    e.divert(d, at);
                    e.out.extend_from_slice(&frame);
                    return Ok(());
                };
                match self.budget.try_reserve(kind, size) {
                    Ok(()) => {
                        st.pending.push_back(Pending::Alloc { kind, size });
                        drop(st);
                        e.divert(d, at);
                        e.out.extend_from_slice(&frame);
                    }
                    Err((cap, used)) => {
                        drop(st);
                        // What rpc-server answers when an allocation fails.
                        inject.extend_from_slice(&ALLOC_RSP_SIZE.to_le_bytes());
                        inject.extend_from_slice(&[0u8; ALLOC_RSP_SIZE as usize]);
                        let r = Refusal::OverLimit { kind, bytes: size, cap_bytes: cap, used_bytes: used };
                        self.budget.record(&r);
                        if let Some(f) = &self.on_refuse {
                            f(&r);
                        }
                    }
                }
            }
            CMD_GET_DEVICE_MEMORY => {
                let kind = self.kinds.get(u32_at(&self.small, 0) as usize).copied();
                self.st.lock().unwrap().pending.push_back(Pending::DevMem(kind));
            }
            CMD_FREE_BUFFER => {
                let ptr = u64_at(&self.small, 0);
                let mut st = self.st.lock().unwrap();
                if let Some((kind, n)) = st.allocs.remove(&ptr) {
                    self.budget.release(kind, n);
                }
            }
            CMD_HELLO => self.st.lock().unwrap().pending.push_back(Pending::Hello),
            c if has_response(c) => self.st.lock().unwrap().pending.push_back(Pending::Other),
            _ => {}
        }
        Ok(())
    }
}

/// Replies (rpc-server -> tunnel).
pub struct Downstream {
    st: Arc<Mutex<State>>,
    budget: Arc<PeerBudget>,
    on_refuse: Option<OnRefuse>,
    head: [u8; RSP_HEADER],
    head_len: usize,
    /// Body bytes still to come.
    left: u64,
    /// Body bytes seen so far.
    at: u64,
    cur: Option<Pending>,
    small: [u8; DEVMEM_RSP_SIZE as usize],
    small_len: usize,
    /// Either direction failed: nothing more passes (shared by both halves).
    dead: Arc<std::sync::atomic::AtomicBool>,
}

impl Downstream {
    /// Look at reply bytes before they are forwarded; `d` may be edited in
    /// place. After an error nothing more may be forwarded: close.
    pub fn feed(&mut self, d: &mut [u8], out: &mut Vec<u8>) -> Result<Fwd, GuardError> {
        if self.dead.load(Ordering::Relaxed) {
            return Err(GuardError::Protocol("stream already failed"));
        }
        let r = self.feed_inner(d, out);
        if let Err(GuardError::Incompatible(major)) = &r {
            let refusal = Refusal::Incompatible { proto_major: *major };
            self.budget.record(&refusal);
            if let Some(f) = &self.on_refuse {
                f(&refusal);
            }
        }
        if r.is_err() {
            self.dead.store(true, Ordering::Relaxed);
        }
        r
    }

    fn feed_inner(&mut self, d: &mut [u8], out: &mut Vec<u8>) -> Result<Fwd, GuardError> {
        let mut e = Emit { diverted: false, out };
        let mut i = 0;
        while i < d.len() {
            if self.cur.is_none() {
                let take = (RSP_HEADER - self.head_len).min(d.len() - i);
                self.head[self.head_len..self.head_len + take].copy_from_slice(&d[i..i + take]);
                e.pass(d, i, i + take);
                self.head_len += take;
                i += take;
                if self.head_len < RSP_HEADER {
                    continue;
                }
                self.head_len = 0;
                let size = u64::from_le_bytes(self.head);
                let Some(p) = self.st.lock().unwrap().pending.front().copied() else {
                    return Err(GuardError::Protocol("a reply nobody asked for"));
                };
                let want = match p {
                    Pending::Hello => Some(HELLO_RSP_SIZE),
                    Pending::DevMem(_) => Some(DEVMEM_RSP_SIZE),
                    Pending::Alloc { .. } => Some(ALLOC_RSP_SIZE),
                    Pending::Other => None,
                };
                if want.is_some_and(|w| w != size) {
                    return Err(match p {
                        // Another HELLO reply layout is another protocol.
                        Pending::Hello => GuardError::Incompatible(if i < d.len() { d[i] } else { 0 }),
                        _ => GuardError::Protocol("a reply has a size protocols 9 and 10 don't use"),
                    });
                }
                self.cur = Some(p);
                self.left = size;
                self.at = 0;
                self.small_len = 0;
                if self.left == 0 {
                    self.finish();
                }
                continue;
            }
            let n = self.left.min((d.len() - i) as u64) as usize;
            match self.cur {
                Some(Pending::Hello) => {
                    if self.at == 0 && !known_major(d[i]) {
                        return Err(GuardError::Incompatible(d[i]));
                    }
                    // conn_caps (bytes 4..28): no transport upgrade.
                    for k in 0..n {
                        if self.at + k as u64 >= 4 {
                            d[i + k] = 0;
                        }
                    }
                    e.pass(d, i, i + n);
                }
                Some(Pending::DevMem(Some(_))) => {
                    e.divert(d, i);
                    self.small[self.small_len..self.small_len + n].copy_from_slice(&d[i..i + n]);
                    self.small_len += n;
                }
                Some(Pending::Alloc { .. }) => {
                    self.small[self.small_len..self.small_len + n].copy_from_slice(&d[i..i + n]);
                    self.small_len += n;
                    e.pass(d, i, i + n);
                }
                _ => e.pass(d, i, i + n),
            }
            self.left -= n as u64;
            self.at += n as u64;
            i += n;
            if self.left == 0 {
                if let Some(Pending::DevMem(Some(kind))) = self.cur {
                    let u64_at = |o: usize| u64::from_le_bytes(self.small[o..o + 8].try_into().unwrap());
                    let (free, total) = (u64_at(0), u64_at(8));
                    let cap = self.budget.caps().cap_bytes(kind);
                    let free = free.min(self.budget.remaining(kind));
                    let total = total.min(cap);
                    e.out.extend_from_slice(&free.to_le_bytes());
                    e.out.extend_from_slice(&total.to_le_bytes());
                }
                self.finish();
            }
        }
        Ok(if e.diverted { Fwd::Out } else { Fwd::Chunk })
    }

    /// The current reply has been passed on completely.
    fn finish(&mut self) {
        let cur = self.cur.take();
        let mut st = self.st.lock().unwrap();
        st.pending.pop_front();
        if let Some(Pending::Alloc { kind, size }) = cur {
            let ptr = u64::from_le_bytes(self.small[0..8].try_into().unwrap());
            let actual = u64::from_le_bytes(self.small[8..16].try_into().unwrap());
            if ptr == 0 {
                self.budget.release(kind, size);
            } else {
                self.budget.settle(kind, size, actual);
                if let Some((k, n)) = st.allocs.insert(ptr, (kind, actual)) {
                    // rpc-server never hands out a live pointer twice; if it
                    // did, the old entry is gone from its books too.
                    self.budget.release(k, n);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc_tap::tests::{req, rsp};
    use crate::rpc_tap::{CMD_DEVICE_COUNT, CMD_SET_TENSOR, RPC_TENSOR_SIZE};

    const MIB: u64 = 1 << 20;

    fn limits(gpu_mib: u64, ram_mib: u64) -> ShareLimits {
        ShareLimits { gpu_mib, ram_mib, threads: 4, paused: false }
    }

    struct Rig {
        up: Upstream,
        down: Downstream,
        budget: Arc<PeerBudget>,
        refusals: Arc<Mutex<Vec<Refusal>>>,
    }

    fn rig_with(budget: Arc<PeerBudget>, kinds: &[DevKind]) -> Rig {
        let refusals = Arc::new(Mutex::new(Vec::new()));
        let r2 = refusals.clone();
        let (up, down) = guard(budget.claim(), Arc::new(kinds.to_vec()), Some(Arc::new(move |r: &Refusal| r2.lock().unwrap().push(r.clone()))));
        Rig { up, down, budget, refusals }
    }

    fn rig(gpu_mib: u64, ram_mib: u64, kinds: &[DevKind]) -> Rig {
        rig_with(PeerBudget::new(limits(gpu_mib, ram_mib)), kinds)
    }

    /// Feed in pieces of 1, 4, 13, ... bytes (as sockets deliver them) and
    /// collect what is forwarded and injected.
    fn send_up(r: &mut Rig, bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>), GuardError> {
        let (mut fwd, mut inj) = (Vec::new(), Vec::new());
        let mut i = 0;
        let mut step = 1;
        let mut out = Vec::new();
        while i < bytes.len() {
            let n = step.min(bytes.len() - i);
            let mut chunk = bytes[i..i + n].to_vec();
            match r.up.feed(&mut chunk, &mut out, &mut inj)? {
                Fwd::Chunk => fwd.extend_from_slice(&chunk),
                Fwd::Out => fwd.extend_from_slice(&out),
            }
            i += n;
            step = step * 3 % 97 + 1;
        }
        Ok((fwd, inj))
    }

    fn send_down(r: &mut Rig, bytes: &[u8]) -> Result<Vec<u8>, GuardError> {
        let mut fwd = Vec::new();
        let mut out = Vec::new();
        let mut i = 0;
        let mut step = 2;
        while i < bytes.len() {
            let n = step.min(bytes.len() - i);
            let mut chunk = bytes[i..i + n].to_vec();
            match r.down.feed(&mut chunk, &mut out)? {
                Fwd::Chunk => fwd.extend_from_slice(&chunk),
                Fwd::Out => fwd.extend_from_slice(&out),
            }
            i += n;
            step = step * 5 % 89 + 1;
        }
        Ok(fwd)
    }

    fn hello_req() -> Vec<u8> {
        req(CMD_HELLO, &[0xAB; HELLO_REQ_SIZE as usize])
    }

    fn hello_reply(major: u8) -> Vec<u8> {
        let mut b = vec![major, 0, 0, 1];
        b.extend_from_slice(&[0xCD; 24]);
        rsp(&b)
    }

    fn hello(r: &mut Rig) {
        send_up(r, &hello_req()).unwrap();
        send_down(r, &hello_reply(9)).unwrap();
    }

    fn devmem_req(dev: u32) -> Vec<u8> {
        req(CMD_GET_DEVICE_MEMORY, &dev.to_le_bytes())
    }

    fn devmem_reply(free: u64, total: u64) -> Vec<u8> {
        let mut b = free.to_le_bytes().to_vec();
        b.extend_from_slice(&total.to_le_bytes());
        rsp(&b)
    }

    fn read_devmem(fwd: &[u8]) -> (u64, u64) {
        assert_eq!(fwd.len(), 24);
        assert_eq!(u64::from_le_bytes(fwd[..8].try_into().unwrap()), 16);
        (u64::from_le_bytes(fwd[8..16].try_into().unwrap()), u64::from_le_bytes(fwd[16..24].try_into().unwrap()))
    }

    fn alloc_req(dev: u32, size: u64) -> Vec<u8> {
        let mut b = dev.to_le_bytes().to_vec();
        b.extend_from_slice(&size.to_le_bytes());
        req(CMD_ALLOC_BUFFER, &b)
    }

    fn alloc_reply(ptr: u64, size: u64) -> Vec<u8> {
        let mut b = ptr.to_le_bytes().to_vec();
        b.extend_from_slice(&size.to_le_bytes());
        rsp(&b)
    }

    fn free_req(ptr: u64) -> Vec<u8> {
        req(CMD_FREE_BUFFER, &ptr.to_le_bytes())
    }

    /// Alloc through the guard with rpc-server answering `ptr`; returns (forwarded, injected).
    fn alloc(r: &mut Rig, dev: u32, size: u64, ptr: u64) -> (Vec<u8>, Vec<u8>) {
        let (fwd, inj) = send_up(r, &alloc_req(dev, size)).unwrap();
        if !fwd.is_empty() {
            send_down(r, &alloc_reply(ptr, size)).unwrap();
        }
        (fwd, inj)
    }

    #[test]
    fn device_memory_reply_reports_min_of_free_and_remaining_cap() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        send_up(&mut r, &devmem_req(0)).unwrap();
        // 7 GB free of 8 GB: the linked PC sees its 2 GB limit.
        let f = send_down(&mut r, &devmem_reply(7000 * MIB, 8000 * MIB)).unwrap();
        assert_eq!(read_devmem(&f), (2048 * MIB, 2048 * MIB));
        // Less free than the limit: the real free memory.
        send_up(&mut r, &devmem_req(0)).unwrap();
        let f = send_down(&mut r, &devmem_reply(1000 * MIB, 8000 * MIB)).unwrap();
        assert_eq!(read_devmem(&f), (1000 * MIB, 2048 * MIB));
        // Limit 0 (no GPU memory shared): nothing free.
        r.budget.set_caps(limits(0, 0));
        send_up(&mut r, &devmem_req(0)).unwrap();
        let f = send_down(&mut r, &devmem_reply(7000 * MIB, 8000 * MIB)).unwrap();
        assert_eq!(read_devmem(&f), (0, 0));
    }

    #[test]
    fn remaining_cap_drops_after_an_alloc_and_returns_after_free_buffer() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        let (fwd, inj) = alloc(&mut r, 0, 1500 * MIB, 0x1000);
        assert_eq!(fwd, alloc_req(0, 1500 * MIB), "forwarded unchanged");
        assert!(inj.is_empty());
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 1500 * MIB);
        send_up(&mut r, &devmem_req(0)).unwrap();
        assert_eq!(read_devmem(&send_down(&mut r, &devmem_reply(7000 * MIB, 8000 * MIB)).unwrap()).0, 548 * MIB);
        let (fwd, _) = send_up(&mut r, &free_req(0x1000)).unwrap();
        assert_eq!(fwd, free_req(0x1000));
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 0);
        send_up(&mut r, &devmem_req(0)).unwrap();
        assert_eq!(read_devmem(&send_down(&mut r, &devmem_reply(7000 * MIB, 8000 * MIB)).unwrap()).0, 2048 * MIB);
        // An unknown pointer is forwarded and changes nothing (rpc-server drops the connection).
        let (fwd, _) = send_up(&mut r, &free_req(0x9999)).unwrap();
        assert_eq!(fwd, free_req(0x9999));
    }

    #[test]
    fn alloc_over_cap_is_not_forwarded_and_gets_a_null_reply() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        alloc(&mut r, 0, 1500 * MIB, 0x1000);
        let (fwd, inj) = send_up(&mut r, &alloc_req(0, 600 * MIB)).unwrap();
        assert!(fwd.is_empty(), "not forwarded");
        assert_eq!(inj, alloc_reply(0, 0), "rpc-server's own failed-alloc reply");
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 1500 * MIB);
        let want = Refusal::OverLimit { kind: DevKind::Gpu, bytes: 600 * MIB, cap_bytes: 2048 * MIB, used_bytes: 1500 * MIB };
        assert_eq!(r.refusals.lock().unwrap().as_slice(), &[want.clone()]);
        let (n, bytes, last) = r.budget.refusals();
        assert_eq!((n, bytes, last.map(|l| l.0)), (1, 600 * MIB, Some(want)));
        // The stream stays in step: a smaller allocation still works.
        let (fwd, _) = alloc(&mut r, 0, 500 * MIB, 0x2000);
        assert!(!fwd.is_empty());
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 2000 * MIB);
        // A size that would overflow u64 is refused, not wrapped.
        let (fwd, inj) = send_up(&mut r, &alloc_req(0, u64::MAX)).unwrap();
        assert!(fwd.is_empty() && !inj.is_empty());
    }

    #[test]
    fn failed_alloc_reply_releases_the_reservation() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        send_up(&mut r, &alloc_req(0, 1000 * MIB)).unwrap();
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 1000 * MIB, "reserved while rpc-server works");
        let f = send_down(&mut r, &alloc_reply(0, 0)).unwrap();
        assert_eq!(f, alloc_reply(0, 0));
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 0);
        // rpc-server rounds up: the real size counts.
        send_up(&mut r, &alloc_req(0, 1000)).unwrap();
        send_down(&mut r, &alloc_reply(0x10, 1024)).unwrap();
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 1024);
    }

    #[test]
    fn cpu_device_index_uses_the_ram_cap() {
        // rpc-server -d CUDA0,CPU: device 1 is the CPU (RAM).
        let mut r = rig(4096, 1024, &[DevKind::of("CUDA0"), DevKind::of("CPU")]);
        hello(&mut r);
        send_up(&mut r, &devmem_req(1)).unwrap();
        assert_eq!(read_devmem(&send_down(&mut r, &devmem_reply(30000 * MIB, 32000 * MIB)).unwrap()), (1024 * MIB, 1024 * MIB));
        let (_, inj) = alloc(&mut r, 1, 2000 * MIB, 0x1);
        assert!(!inj.is_empty(), "2 GB of RAM over a 1 GB RAM limit");
        let (_, inj) = alloc(&mut r, 0, 2000 * MIB, 0x2);
        assert!(inj.is_empty(), "the GPU limit is separate");
        assert_eq!((r.budget.used_bytes(DevKind::Gpu), r.budget.used_bytes(DevKind::Ram)), (2000 * MIB, 0));
        assert!(matches!(r.refusals.lock().unwrap()[0], Refusal::OverLimit { kind: DevKind::Ram, .. }));
    }

    #[test]
    fn dropping_a_stream_returns_its_allocations() {
        let budget = PeerBudget::new(limits(2048, 0));
        let mut a = rig_with(budget.clone(), &[DevKind::Gpu]);
        let mut b = rig_with(budget.clone(), &[DevKind::Gpu]);
        assert_eq!(budget.open_streams(), 2);
        hello(&mut a);
        hello(&mut b);
        alloc(&mut a, 0, 1200 * MIB, 0x1);
        // Two streams of one linked PC share its limit.
        let (_, inj) = alloc(&mut b, 0, 1200 * MIB, 0x1);
        assert!(!inj.is_empty());
        alloc(&mut b, 0, 800 * MIB, 0x1);
        assert_eq!(budget.used_bytes(DevKind::Gpu), 2000 * MIB);
        // An alloc in flight (no reply yet) is returned too.
        send_up(&mut b, &alloc_req(0, 40 * MIB)).unwrap();
        drop(a);
        assert_eq!(budget.used_bytes(DevKind::Gpu), 840 * MIB);
        assert_eq!(budget.open_streams(), 1);
        drop(b);
        assert_eq!(budget.used_bytes(DevKind::Gpu), 0);
        assert_eq!(budget.open_streams(), 0);
    }

    #[test]
    fn hello_conn_caps_are_zeroed() {
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        let (fwd, _) = send_up(&mut r, &hello_req()).unwrap();
        assert_eq!(fwd, req(CMD_HELLO, &[0u8; 24]));
        let f = send_down(&mut r, &hello_reply(9)).unwrap();
        let mut want = vec![9, 0, 0, 1];
        want.extend_from_slice(&[0u8; 24]);
        assert_eq!(f, rsp(&want), "version and flags kept, caps zeroed");
    }

    #[test]
    fn protocol_10_hello_is_accepted_and_limits_still_apply() {
        // b12716 speaks protocol 10; every message the guard reads is unchanged.
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        send_up(&mut r, &hello_req()).unwrap();
        send_down(&mut r, &hello_reply(10)).unwrap();
        assert!(r.refusals.lock().unwrap().is_empty());
        assert!(send_up(&mut r, &devmem_req(0)).is_ok());
    }

    #[test]
    fn other_protocol_major_closes_the_stream() {
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        send_up(&mut r, &hello_req()).unwrap();
        assert_eq!(send_down(&mut r, &hello_reply(11)), Err(GuardError::Incompatible(11)));
        assert_eq!(r.refusals.lock().unwrap().as_slice(), &[Refusal::Incompatible { proto_major: 11 }]);
        // Nothing more passes.
        assert!(send_up(&mut r, &devmem_req(0)).is_err());
        // A HELLO reply of another size is another protocol too.
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        send_up(&mut r, &hello_req()).unwrap();
        assert_eq!(send_down(&mut r, &rsp(&[12, 0, 0, 0])), Err(GuardError::Incompatible(12)));
        // A stream that doesn't start with HELLO is refused.
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        assert!(send_up(&mut r, &devmem_req(0)).is_err());
        // A HELLO request of another size, too.
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        assert!(send_up(&mut r, &req(CMD_HELLO, &[0u8; 16])).is_err());
    }

    #[test]
    fn set_tensor_bodies_pass_through_byte_identical() {
        let mut r = rig(1, 1, &[DevKind::Gpu]);
        hello(&mut r);
        let mut body = vec![0u8; RPC_TENSOR_SIZE + 9];
        body.extend((0..300_000u32).map(|i| (i * 7 + 3) as u8));
        let frame = req(CMD_SET_TENSOR, &body);
        // In big socket-sized pieces: every piece is forwarded as it is (no copy).
        let mut out = Vec::new();
        let mut inj = Vec::new();
        let mut fwd = Vec::new();
        for piece in frame.chunks(16 * 1024) {
            let mut c = piece.to_vec();
            assert_eq!(r.up.feed(&mut c, &mut out, &mut inj).unwrap(), Fwd::Chunk);
            fwd.extend_from_slice(&c);
        }
        assert_eq!(fwd, frame);
        // And in awkward small pieces.
        assert_eq!(send_up(&mut r, &frame).unwrap().0, frame);
        assert!(inj.is_empty());
    }

    #[test]
    fn bad_device_index_is_forwarded_unchanged() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        let (fwd, inj) = send_up(&mut r, &alloc_req(7, 100 * MIB)).unwrap();
        assert_eq!(fwd, alloc_req(7, 100 * MIB));
        assert!(inj.is_empty());
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 0);
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        let (fwd, _) = send_up(&mut r, &devmem_req(3)).unwrap();
        assert_eq!(fwd, devmem_req(3));
        // (rpc-server drops the connection; were it to answer, the reply passes unchanged.)
        assert_eq!(send_down(&mut r, &devmem_reply(5, 6)).unwrap(), devmem_reply(5, 6));
    }

    #[test]
    fn replies_nobody_asked_for_and_wrong_sizes_close_the_stream() {
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        assert!(send_down(&mut r, &rsp(&[1])).is_err());
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        send_up(&mut r, &devmem_req(0)).unwrap();
        assert!(send_down(&mut r, &rsp(&[0u8; 8])).is_err());
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        assert!(send_up(&mut r, &req(CMD_ALLOC_BUFFER, &[0u8; 20])).is_err());
        // An ALLOC_BUFFER while a reply is still due can't be answered in order.
        let mut r = rig(2048, 0, &[DevKind::Gpu]);
        hello(&mut r);
        send_up(&mut r, &req(CMD_DEVICE_COUNT, &[])).unwrap();
        assert!(send_up(&mut r, &alloc_req(0, 1)).is_err());
    }

    /// A synthetic load transcript (HELLO, device count and memory, buffers,
    /// weights, compute, free): with a generous limit every byte is identical
    /// both ways; with a small one the guard refuses at the expected alloc.
    #[test]
    fn a_load_transcript_replays_identically_and_a_small_cap_refuses_at_the_right_alloc() {
        // (request, reply) pairs; None = no reply.
        let mut weights = vec![0u8; RPC_TENSOR_SIZE + 9];
        weights.extend(std::iter::repeat_n(5u8, 70_000));
        let script: Vec<(Vec<u8>, Option<Vec<u8>>)> = vec![
            (req(CMD_HELLO, &[0u8; 24]), Some(hello_reply_zero())),
            (req(CMD_DEVICE_COUNT, &[]), Some(rsp(&1u32.to_le_bytes()))),
            (devmem_req(0), Some(devmem_reply(3000 * MIB, 4000 * MIB))),
            (req(1, &0u32.to_le_bytes()), Some(rsp(&32u64.to_le_bytes()))),
            (alloc_req(0, 1000 * MIB), Some(alloc_reply(0xA000, 1000 * MIB))),
            (req(3, &0xA000u64.to_le_bytes()), Some(rsp(&0x5000u64.to_le_bytes()))),
            (req(CMD_SET_TENSOR, &weights), None),
            (alloc_req(0, 300 * MIB), Some(alloc_reply(0xB000, 300 * MIB))),
            (req(10, &[1u8; 4000]), None),
            (req(8, &[0u8; RPC_TENSOR_SIZE + 16]), Some(rsp(&[9u8; 5000]))),
            (free_req(0xB000), None),
            (free_req(0xA000), None),
        ];
        let mut r = rig(1 << 20, 0, &[DevKind::Gpu]);
        for (q, a) in &script {
            let (fwd, inj) = send_up(&mut r, q).unwrap();
            assert_eq!(&fwd, q);
            assert!(inj.is_empty());
            if let Some(a) = a {
                assert_eq!(&send_down(&mut r, a).unwrap(), a);
            }
        }
        assert_eq!(r.budget.used_bytes(DevKind::Gpu), 0);
        // 1200 MiB: the second buffer (300 MiB on top of 1000) is refused.
        let mut r = rig(1200, 0, &[DevKind::Gpu]);
        let mut refused_at = None;
        for (k, (q, a)) in script.iter().enumerate() {
            let (fwd, inj) = send_up(&mut r, q).unwrap();
            if !inj.is_empty() {
                refused_at = Some(k);
                break;
            }
            assert_eq!(&fwd, q);
            if let Some(a) = a {
                send_down(&mut r, a).unwrap();
            }
        }
        assert_eq!(refused_at, Some(7));
    }

    fn hello_reply_zero() -> Vec<u8> {
        let mut b = vec![9, 0, 0, 0];
        b.extend_from_slice(&[0u8; 24]);
        rsp(&b)
    }

    /// Tiny xorshift (no extra crate): reproducible "random" bytes.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
    }

    /// Malformed input from a paired peer (or a confused rpc-server): random
    /// bytes, truncated and mutated frames, huge sizes. The guard must never
    /// panic, never allocate by a size from the wire, never forward more than
    /// it was given (plus at most one held 21-byte frame), and stay failed
    /// once it failed.
    #[test]
    fn random_and_truncated_frames_never_panic_or_balloon() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let good: Vec<Vec<u8>> = vec![
            hello_req(),
            devmem_req(0),
            alloc_req(0, 100),
            alloc_req(1, 5 * MIB),
            free_req(0x10),
            req(CMD_SET_TENSOR, &[3u8; 400]),
            req(CMD_DEVICE_COUNT, &[]),
            req(CMD_ALLOC_BUFFER, &[0xFF; 12]),
        ];
        let replies: Vec<Vec<u8>> = vec![hello_reply(9), hello_reply(3), devmem_reply(u64::MAX, u64::MAX), alloc_reply(0, 0), alloc_reply(u64::MAX, u64::MAX), rsp(&[])];
        for round in 0..3000 {
            let mut r = rig(rng.next() % 4096, rng.next() % 4096, &[DevKind::Gpu, DevKind::Ram]);
            // A stream: some good frames, some mutated or truncated, some noise.
            let mut up_bytes = Vec::new();
            let mut down_bytes = Vec::new();
            for _ in 0..(rng.next() % 8) {
                let mut f = good[(rng.next() as usize) % good.len()].clone();
                match rng.next() % 5 {
                    0 => {
                        let k = (rng.next() as usize) % f.len();
                        f[k] = rng.next() as u8;
                    }
                    1 => f.truncate((rng.next() as usize) % f.len()),
                    2 => {
                        let n = (rng.next() % 64) as usize;
                        f = rng.bytes(n);
                    }
                    3 => {
                        // A huge declared size with a short body.
                        let n = (rng.next() as usize % 9).max(1);
                        f = vec![rng.next() as u8];
                        f.extend_from_slice(&(u64::MAX - rng.next() % 3).to_le_bytes());
                        f.extend(rng.bytes(n));
                    }
                    _ => {}
                }
                up_bytes.extend_from_slice(&f);
                let mut a = replies[(rng.next() as usize) % replies.len()].clone();
                if rng.next() % 4 == 0 {
                    let k = (rng.next() as usize) % a.len().max(1);
                    a.truncate(k);
                }
                down_bytes.extend_from_slice(&a);
            }
            let (mut out, mut inj) = (Vec::new(), Vec::new());
            let mut failed = false;
            let (mut i, mut j) = (0, 0);
            while i < up_bytes.len() || j < down_bytes.len() {
                if i < up_bytes.len() {
                    let n = ((rng.next() % 40) as usize + 1).min(up_bytes.len() - i);
                    let mut c = up_bytes[i..i + n].to_vec();
                    inj.clear();
                    match r.up.feed(&mut c, &mut out, &mut inj) {
                        Ok(Fwd::Out) => assert!(out.len() <= n + ALLOC_FRAME, "round {round}"),
                        Ok(Fwd::Chunk) => assert!(!failed, "round {round}: passed bytes after failing"),
                        Err(_) => failed = true,
                    }
                    assert!(inj.len() <= 24);
                    i += n;
                }
                if j < down_bytes.len() {
                    let n = ((rng.next() % 40) as usize + 1).min(down_bytes.len() - j);
                    let mut c = down_bytes[j..j + n].to_vec();
                    match r.down.feed(&mut c, &mut out) {
                        Ok(Fwd::Out) => assert!(out.len() <= n + DEVMEM_RSP_SIZE as usize, "round {round}"),
                        Ok(Fwd::Chunk) | Err(_) => {}
                    }
                    j += n;
                }
                assert!(out.capacity() <= 4096, "round {round}: output buffer grew to {}", out.capacity());
            }
            // Whatever happened, dropping the stream leaves nothing reserved.
            let budget = r.budget.clone();
            drop(r);
            assert_eq!((budget.used_bytes(DevKind::Gpu), budget.used_bytes(DevKind::Ram), budget.open_streams()), (0, 0, 0), "round {round}");
        }
    }
}

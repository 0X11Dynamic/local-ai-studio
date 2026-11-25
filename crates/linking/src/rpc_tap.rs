//! A read-only view of the ggml-rpc stream the bridge forwards, for load
//! progress: how many weight bytes were really sent to the helper, and how
//! many the helper's weight cache supplied instead.
//!
//! ggml-rpc (protocols 9 and 10, llama.cpp b12250 and b12716,
//! `ggml/src/ggml-rpc/ggml-rpc.cpp`; 10 changed only the graph op list, so
//! every message this file and the guard read is byte-identical in both):
//! - request: `cmd (u8) | size (u64 LE) | size bytes`;
//! - response, only for some commands: `size (u64 LE) | size bytes`, in
//!   request order;
//! - `SET_TENSOR` body: `rpc_tensor (296) | cache_flag (u8) | offset (u64) | data`;
//! - `SET_TENSOR_HASH` body: `rpc_tensor | offset (u64) | hash (u64)`; its
//!   one-byte response is 1 when the helper's cache had the tensor (nothing
//!   more is sent) and 0 when it didn't (a `SET_TENSOR` with the data follows).
//!
//! The tap never changes a byte. It checks the protocol version in the HELLO
//! reply; on any other version, or anything it doesn't expect, it stops
//! parsing that stream and counts its raw request bytes instead (and never
//! claims a cache hit).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The protocol major versions this parser and the guard know.
pub const KNOWN_PROTO_MAJORS: [u8; 2] = [9, 10];

/// Whether `major` is a ggml-rpc protocol this parser and the guard know.
pub(crate) fn known_major(major: u8) -> bool {
    KNOWN_PROTO_MAJORS.contains(&major)
}
pub(crate) const RPC_TENSOR_SIZE: usize = 296;
pub(crate) const CMD_ALLOC_BUFFER: u8 = 0;
pub(crate) const CMD_GET_ALIGNMENT: u8 = 1;
pub(crate) const CMD_GET_MAX_SIZE: u8 = 2;
pub(crate) const CMD_BUFFER_GET_BASE: u8 = 3;
pub(crate) const CMD_FREE_BUFFER: u8 = 4;
pub(crate) const CMD_SET_TENSOR: u8 = 6;
pub(crate) const CMD_SET_TENSOR_HASH: u8 = 7;
pub(crate) const CMD_GET_TENSOR: u8 = 8;
pub(crate) const CMD_COPY_TENSOR: u8 = 9;
pub(crate) const CMD_GET_DEVICE_MEMORY: u8 = 11;
pub(crate) const CMD_GET_ALLOC_SIZE: u8 = 13;
pub(crate) const CMD_HELLO: u8 = 14;
pub(crate) const CMD_DEVICE_COUNT: u8 = 15;
pub(crate) const CMD_GET_TENSOR_2D: u8 = 19;
pub(crate) const CMD_COMM_INIT: u8 = 20;
/// Request header: `cmd (u8) | size (u64 LE)`.
pub(crate) const REQ_HEADER: usize = 9;
/// Response header: `size (u64 LE)`.
pub(crate) const RSP_HEADER: usize = 8;
/// Header of a SET_TENSOR body before the data.
const SET_TENSOR_HEADER: usize = RPC_TENSOR_SIZE + 1 + 8;
const HASH_REQ_SIZE: usize = RPC_TENSOR_SIZE + 16;

/// The commands rpc-server answers (in request order); every other command
/// sends nothing back (protocol 9).
pub(crate) fn has_response(cmd: u8) -> bool {
    matches!(
        cmd,
        CMD_ALLOC_BUFFER
            | CMD_GET_ALIGNMENT
            | CMD_GET_MAX_SIZE
            | CMD_BUFFER_GET_BASE
            | CMD_SET_TENSOR_HASH
            | CMD_GET_TENSOR
            | CMD_COPY_TENSOR
            | CMD_GET_DEVICE_MEMORY
            | CMD_GET_ALLOC_SIZE
            | CMD_HELLO
            | CMD_DEVICE_COUNT
            | CMD_GET_TENSOR_2D
            | CMD_COMM_INIT
    )
}

/// Totals over every stream of one bridge (cumulative).
#[derive(Debug, Default)]
pub struct RpcCounters {
    /// Tensor data sent with SET_TENSOR (weights while a model loads), plus
    /// the raw request bytes of streams the tap couldn't follow.
    pub tensor_bytes: Arc<AtomicU64>,
    /// Bytes the helper's cache supplied (SET_TENSOR_HASH answered "have it").
    pub cached_bytes: Arc<AtomicU64>,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
enum Pending {
    Hello,
    Hash { bytes: u64 },
    Other,
}

/// State shared by the two directions of one stream.
struct Shared {
    counters: Arc<RpcCounters>,
    pending: Mutex<VecDeque<Pending>>,
    /// Parsing stopped (unknown version or an unexpected byte).
    off: std::sync::atomic::AtomicBool,
}

impl Shared {
    fn stop(&self) {
        self.off.store(true, Ordering::Relaxed);
    }
    fn is_off(&self) -> bool {
        self.off.load(Ordering::Relaxed)
    }
}

/// Make the two halves of a tap for one stream.
pub fn tap(counters: Arc<RpcCounters>) -> (Upstream, Downstream) {
    let s = Arc::new(Shared { counters, pending: Mutex::new(VecDeque::new()), off: std::sync::atomic::AtomicBool::new(false) });
    (Upstream { s: s.clone(), head: Vec::with_capacity(9), cmd: 0, left: 0, at: 0, prefix: Vec::new() }, Downstream { s, head: Vec::with_capacity(8), left: 0, cur: None, first: true })
}

/// Requests (llama-server -> helper).
pub struct Upstream {
    s: Arc<Shared>,
    head: Vec<u8>,
    cmd: u8,
    /// Body bytes still to come.
    left: u64,
    /// Body bytes seen so far.
    at: u64,
    /// The start of the body, kept for the commands whose header matters.
    prefix: Vec<u8>,
}

impl Upstream {
    /// Look at bytes about to be forwarded (call before forwarding them).
    pub fn feed(&mut self, mut d: &[u8]) {
        if self.s.is_off() {
            self.s.counters.tensor_bytes.fetch_add(d.len() as u64, Ordering::Relaxed);
            return;
        }
        while !d.is_empty() {
            if self.left == 0 {
                let take = (9 - self.head.len()).min(d.len());
                self.head.extend_from_slice(&d[..take]);
                d = &d[take..];
                if self.head.len() < 9 {
                    return;
                }
                self.cmd = self.head[0];
                self.left = u64::from_le_bytes(self.head[1..9].try_into().unwrap());
                self.head.clear();
                self.at = 0;
                self.prefix.clear();
                if self.left == 0 {
                    self.end();
                }
                continue;
            }
            let n = (self.left.min(d.len() as u64)) as usize;
            let chunk = &d[..n];
            let keep = match self.cmd {
                CMD_SET_TENSOR => SET_TENSOR_HEADER,
                CMD_SET_TENSOR_HASH => HASH_REQ_SIZE,
                _ => 0,
            };
            if (self.prefix.len()) < keep {
                let k = (keep - self.prefix.len()).min(n);
                self.prefix.extend_from_slice(&chunk[..k]);
            }
            if self.cmd == CMD_SET_TENSOR {
                // Data bytes in this chunk: what lies past the header.
                let start = self.at;
                let end = self.at + n as u64;
                let data = end.saturating_sub((SET_TENSOR_HEADER as u64).max(start));
                if data > 0 {
                    self.s.counters.tensor_bytes.fetch_add(data, Ordering::Relaxed);
                }
            }
            self.at += n as u64;
            self.left -= n as u64;
            d = &d[n..];
            if self.left == 0 {
                self.end();
            }
        }
    }

    fn end(&mut self) {
        if !has_response(self.cmd) {
            return;
        }
        let p = match self.cmd {
            CMD_HELLO => Pending::Hello,
            CMD_SET_TENSOR_HASH if self.prefix.len() == HASH_REQ_SIZE => {
                let off = u64::from_le_bytes(self.prefix[RPC_TENSOR_SIZE..RPC_TENSOR_SIZE + 8].try_into().unwrap());
                Pending::Hash { bytes: tensor_nbytes(&self.prefix).saturating_sub(off) }
            }
            CMD_SET_TENSOR_HASH => {
                self.s.stop();
                return;
            }
            _ => Pending::Other,
        };
        self.s.pending.lock().unwrap().push_back(p);
    }
}

/// Size of a contiguous tensor from its serialized `rpc_tensor`: `ne[3] * nb[3]`
/// (`ne` at byte 20, `nb` at 36, four u32 each).
fn tensor_nbytes(t: &[u8]) -> u64 {
    let u = |o: usize| u32::from_le_bytes(t[o..o + 4].try_into().unwrap()) as u64;
    u(20 + 12) * u(36 + 12)
}

/// Responses (helper -> llama-server).
pub struct Downstream {
    s: Arc<Shared>,
    head: Vec<u8>,
    left: u64,
    cur: Option<Pending>,
    /// The next body byte is the first of its response.
    first: bool,
}

impl Downstream {
    pub fn feed(&mut self, mut d: &[u8]) {
        while !d.is_empty() && !self.s.is_off() {
            if self.left == 0 {
                let take = (8 - self.head.len()).min(d.len());
                self.head.extend_from_slice(&d[..take]);
                d = &d[take..];
                if self.head.len() < 8 {
                    return;
                }
                self.left = u64::from_le_bytes(self.head[..8].try_into().unwrap());
                self.head.clear();
                self.first = true;
                let Some(p) = self.s.pending.lock().unwrap().pop_front() else {
                    // A response nobody asked for: the stream isn't what we think.
                    self.s.stop();
                    return;
                };
                self.cur = Some(p);
                if self.left == 0 {
                    // GET_TENSOR(_2D) of 0 bytes answers with an empty body;
                    // HELLO and SET_TENSOR_HASH never do.
                    if !matches!(p, Pending::Other) {
                        self.s.stop();
                        return;
                    }
                    self.cur = None;
                }
                continue;
            }
            let n = (self.left.min(d.len() as u64)) as usize;
            if self.first {
                self.first = false;
                match self.cur {
                    Some(Pending::Hello) if !known_major(d[0]) => self.s.stop(),
                    Some(Pending::Hash { bytes }) => {
                        let c = &self.s.counters;
                        if d[0] == 1 {
                            c.cache_hits.fetch_add(1, Ordering::Relaxed);
                            c.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
                        } else {
                            c.cache_misses.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    _ => {}
                }
            }
            self.left -= n as u64;
            d = &d[n..];
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn tensor(ne: [u32; 4], nb: [u32; 4]) -> Vec<u8> {
        let mut t = vec![0u8; RPC_TENSOR_SIZE];
        for i in 0..4 {
            t[20 + 4 * i..24 + 4 * i].copy_from_slice(&ne[i].to_le_bytes());
            t[36 + 4 * i..40 + 4 * i].copy_from_slice(&nb[i].to_le_bytes());
        }
        t
    }

    pub(crate) fn req(cmd: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![cmd];
        v.extend_from_slice(&(body.len() as u64).to_le_bytes());
        v.extend_from_slice(body);
        v
    }

    pub(crate) fn rsp(body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(body);
        v
    }

    /// A 14 MB q4_K-like tensor: 2560 x 9728, row 1440 bytes.
    fn ffn() -> Vec<u8> {
        tensor([2560, 9728, 1, 1], [144, 1440, 1440 * 9728, 1440 * 9728])
    }

    pub(crate) fn hello_rsp(major: u8) -> Vec<u8> {
        let mut b = vec![major, 0, 0, 1];
        b.extend_from_slice(&[0u8; 24]);
        rsp(&b)
    }

    /// Feed a byte stream in awkward pieces (as sockets deliver it).
    pub(crate) fn feed_split(f: &mut dyn FnMut(&[u8]), bytes: &[u8]) {
        let mut i = 0;
        let mut step = 1;
        while i < bytes.len() {
            let n = step.min(bytes.len() - i);
            f(&bytes[i..i + n]);
            i += n;
            step = step * 3 % 1000 + 1;
        }
    }

    #[test]
    fn counts_tensor_data_and_cache_hits_and_misses() {
        let c = Arc::new(RpcCounters::default());
        let (mut up, mut down) = tap(c.clone());
        // HELLO and its reply.
        feed_split(&mut |d| up.feed(d), &req(CMD_HELLO, &[0u8; 24]));
        feed_split(&mut |d| down.feed(d), &hello_rsp(9));
        // A small SET_TENSOR (no reply): 4096 data bytes.
        let mut st = tensor([1024, 1, 1, 1], [4, 4096, 4096, 4096]);
        st.push(0);
        st.extend_from_slice(&0u64.to_le_bytes());
        st.extend_from_slice(&[7u8; 4096]);
        feed_split(&mut |d| up.feed(d), &req(CMD_SET_TENSOR, &st));
        assert_eq!(c.tensor_bytes.load(Ordering::Relaxed), 4096);
        // Two hash requests: the first is a hit, the second a miss.
        let mut h = ffn();
        h.extend_from_slice(&0u64.to_le_bytes());
        h.extend_from_slice(&0x1234u64.to_le_bytes());
        feed_split(&mut |d| up.feed(d), &req(CMD_SET_TENSOR_HASH, &h));
        feed_split(&mut |d| up.feed(d), &req(CMD_SET_TENSOR_HASH, &h));
        // GET_ALLOC_SIZE in between also gets a reply, in order.
        feed_split(&mut |d| up.feed(d), &req(CMD_GET_ALLOC_SIZE, &[0u8; 300]));
        feed_split(&mut |d| down.feed(d), &rsp(&[1]));
        feed_split(&mut |d| down.feed(d), &rsp(&[0]));
        feed_split(&mut |d| down.feed(d), &rsp(&[0u8; 8]));
        assert_eq!(c.cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(c.cache_misses.load(Ordering::Relaxed), 1);
        assert_eq!(c.cached_bytes.load(Ordering::Relaxed), 1440 * 9728);
        // GRAPH_COMPUTE (no reply) doesn't count as tensor data.
        feed_split(&mut |d| up.feed(d), &req(10, &[1u8; 5000]));
        assert_eq!(c.tensor_bytes.load(Ordering::Relaxed), 4096);
    }

    #[test]
    fn data_is_counted_while_a_big_tensor_streams() {
        let c = Arc::new(RpcCounters::default());
        let (mut up, _down) = tap(c.clone());
        let mut st = ffn();
        st.push(1);
        st.extend_from_slice(&0u64.to_le_bytes());
        let data = 1440 * 9728;
        let header_len = 9 + st.len();
        st.resize(st.len() + data, 3);
        let whole = req(CMD_SET_TENSOR, &st);
        up.feed(&whole[..header_len + 1000]);
        assert_eq!(c.tensor_bytes.load(Ordering::Relaxed), 1000);
        up.feed(&whole[header_len + 1000..]);
        assert_eq!(c.tensor_bytes.load(Ordering::Relaxed), data as u64);
    }

    #[test]
    fn another_protocol_version_is_never_read_as_a_cache_hit() {
        let c = Arc::new(RpcCounters::default());
        let (mut up, mut down) = tap(c.clone());
        up.feed(&req(CMD_HELLO, &[0u8; 24]));
        down.feed(&hello_rsp(11));
        let mut h = ffn();
        h.extend_from_slice(&[0u8; 16]);
        let r = req(CMD_SET_TENSOR_HASH, &h);
        up.feed(&r);
        down.feed(&rsp(&[1]));
        assert_eq!(c.cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(c.cached_bytes.load(Ordering::Relaxed), 0);
        // Its requests are counted raw instead.
        assert_eq!(c.tensor_bytes.load(Ordering::Relaxed), r.len() as u64);
    }

    #[test]
    fn an_empty_reply_keeps_the_stream_in_step() {
        let c = Arc::new(RpcCounters::default());
        let (mut up, mut down) = tap(c.clone());
        up.feed(&req(CMD_HELLO, &[0u8; 24]));
        down.feed(&hello_rsp(9));
        // GET_TENSOR of 0 bytes: an empty reply, then a hash hit.
        up.feed(&req(CMD_GET_TENSOR, &[0u8; RPC_TENSOR_SIZE + 16]));
        let mut h = ffn();
        h.extend_from_slice(&[0u8; 16]);
        up.feed(&req(CMD_SET_TENSOR_HASH, &h));
        let mut both = rsp(&[]);
        both.extend_from_slice(&rsp(&[1]));
        feed_split(&mut |d| down.feed(d), &both);
        assert!(!down.s.is_off());
        assert_eq!(c.cache_hits.load(Ordering::Relaxed), 1);
        // An empty reply to a hash lookup is not ggml-rpc 9: stop.
        up.feed(&req(CMD_SET_TENSOR_HASH, &h));
        down.feed(&rsp(&[]));
        assert!(down.s.is_off());
    }

    #[test]
    fn an_unexpected_reply_stops_parsing() {
        let c = Arc::new(RpcCounters::default());
        let (_up, mut down) = tap(c.clone());
        down.feed(&rsp(&[1]));
        assert!(down.s.is_off());
    }
}

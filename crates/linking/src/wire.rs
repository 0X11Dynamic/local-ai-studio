//! Framing.
//!
//! On the TCP stream every message is `[u16 big-endian length][bytes]` (at
//! most 65535 bytes, the Noise message limit). Before the Noise session
//! exists only three kinds of plaintext frame are sent: the hello, the
//! SPAKE2/Noise handshake messages, and status frames (`[code][text]`).
//! After the handshake every frame is a Noise transport message (ChaCha20-
//! Poly1305 with an explicit per-direction counter nonce) whose plaintext is
//! a tunnel [`Frame`].

use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

use crate::{LinkError, LinkResult, PROTOCOL_VERSION};

pub const MAX_FRAME: usize = 65535;
/// Noise adds a 16-byte tag to each transport message.
pub const MAX_PLAINTEXT: usize = MAX_FRAME - 16;
/// Bytes of tunnelled data per frame: the largest Noise payload minus the
/// 1-byte frame type, so big tensors cross in as few messages as possible.
pub const DATA_CHUNK: usize = MAX_PLAINTEXT - 1;
/// Read buffer on the tunnel socket (several frames per syscall).
const READ_BUFFER: usize = 256 * 1024;
/// Kernel send buffer on tunnel and forwarded sockets: enough for a gigabit
/// link at ~30 ms round trip, so bulk transfers (weights at load, big
/// activations) aren't limited by a small send window. The receive side is
/// left to Windows' window auto-tuning (a fixed SO_RCVBUF would switch it off).
pub const SOCKET_BUFFER: usize = 4 << 20;

/// Bigger send buffer (best effort) and no Nagle delay.
pub fn tune_socket(s: &tokio::net::TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = socket2::SockRef::from(s).set_send_buffer_size(SOCKET_BUFFER);
}

const HELLO_MAGIC: &[u8; 7] = b"LASLINK";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Pairing with a one-time code.
    Pair = 1,
    /// Control channel: info, ping, bandwidth test, unlink.
    Control = 2,
    /// A stream forwarded to the helper's local ggml-rpc-server.
    Rpc = 3,
}

pub fn hello(mode: Mode) -> Vec<u8> {
    let mut v = HELLO_MAGIC.to_vec();
    v.push(PROTOCOL_VERSION);
    v.push(mode as u8);
    v
}

pub fn parse_hello(b: &[u8]) -> LinkResult<Mode> {
    if b.len() != HELLO_MAGIC.len() + 2 || &b[..7] != HELLO_MAGIC {
        return Err(LinkError::Protocol("not a Local AI Studio link connection".into()));
    }
    if b[7] != PROTOCOL_VERSION {
        return Err(LinkError::Refused(format!(
            "the two PCs run different Local AI Studio link versions ({} and {PROTOCOL_VERSION}); update the app on both",
            b[7]
        )));
    }
    match b[8] {
        1 => Ok(Mode::Pair),
        2 => Ok(Mode::Control),
        3 => Ok(Mode::Rpc),
        m => Err(LinkError::Protocol(format!("unknown connection mode {m}"))),
    }
}

pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, data: &[u8]) -> LinkResult<()> {
    if data.len() > MAX_FRAME {
        return Err(LinkError::Protocol("frame too large".into()));
    }
    let mut buf = Vec::with_capacity(2 + data.len());
    buf.extend_from_slice(&(data.len() as u16).to_be_bytes());
    buf.extend_from_slice(data);
    w.write_all(&buf).await?;
    Ok(())
}

/// Read one frame into `buf` (replacing its contents).
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R, buf: &mut Vec<u8>) -> LinkResult<()> {
    let mut len = [0u8; 2];
    r.read_exact(&mut len).await?;
    let n = u16::from_be_bytes(len) as usize;
    buf.resize(n, 0);
    r.read_exact(buf).await?;
    Ok(())
}

/// Plaintext status frame: `[0]` + body for OK, else `[1]` + `code\nmessage`.
pub fn status_ok(body: &[u8]) -> Vec<u8> {
    let mut v = vec![0u8];
    v.extend_from_slice(body);
    v
}

pub fn status_err(e: &LinkError) -> Vec<u8> {
    let mut v = vec![1u8];
    v.extend_from_slice(format!("{}\n{}", e.code(), e.wire_message()).as_bytes());
    v
}

/// Split a status frame: the OK body, or the error it carries.
pub fn parse_status(b: &[u8]) -> LinkResult<&[u8]> {
    match b.first() {
        Some(0) => Ok(&b[1..]),
        Some(1) => {
            let text = String::from_utf8_lossy(&b[1..]);
            let (code, msg) = text.split_once('\n').unwrap_or((text.as_ref(), ""));
            Err(LinkError::from_wire(code, msg))
        }
        _ => Err(LinkError::Protocol("malformed status frame".into())),
    }
}

/// Sending half of a Noise session.
pub struct SecureWriter {
    w: OwnedWriteHalf,
    t: Arc<snow::StatelessTransportState>,
    nonce: u64,
    buf: Vec<u8>,
}

/// Receiving half of a Noise session.
pub struct SecureReader {
    r: BufReader<OwnedReadHalf>,
    t: Arc<snow::StatelessTransportState>,
    nonce: u64,
    cbuf: Vec<u8>,
    /// Decrypted plaintext of the last message (reused, no per-message allocation).
    pbuf: Vec<u8>,
}

pub fn secure_pair(
    r: OwnedReadHalf,
    w: OwnedWriteHalf,
    t: snow::StatelessTransportState,
    recv_nonce: u64,
    send_nonce: u64,
) -> (SecureReader, SecureWriter) {
    let t = Arc::new(t);
    (
        SecureReader { r: BufReader::with_capacity(READ_BUFFER, r), t: t.clone(), nonce: recv_nonce, cbuf: Vec::new(), pbuf: vec![0u8; MAX_FRAME] },
        SecureWriter { w, t, nonce: send_nonce, buf: vec![0u8; MAX_FRAME + 2] },
    )
}

impl SecureWriter {

    /// Encrypt and write one message: length prefix and ciphertext in one
    /// buffer and one write (no copy, no extra segment).
    pub async fn send_raw(&mut self, plaintext: &[u8]) -> LinkResult<()> {
        let n = self.t.write_message(self.nonce, plaintext, &mut self.buf[2..])?;
        self.nonce += 1;
        self.buf[..2].copy_from_slice(&(n as u16).to_be_bytes());
        self.w.write_all(&self.buf[..2 + n]).await?;
        Ok(())
    }

    /// Send `data[1..]` as a Data frame; `data[0]` is overwritten with the
    /// frame type, so the caller reads straight into `data[1..]` (no copy).
    pub async fn send_data_in_place(&mut self, data: &mut [u8], compress: bool) -> LinkResult<()> {
        if compress {
            let p = encode_data(&data[1..], true);
            return self.send_raw(&p).await;
        }
        data[0] = T_DATA;
        let n = self.t.write_message(self.nonce, data, &mut self.buf[2..])?;
        self.nonce += 1;
        self.buf[..2].copy_from_slice(&(n as u16).to_be_bytes());
        self.w.write_all(&self.buf[..2 + n]).await?;
        Ok(())
    }

    pub async fn send(&mut self, f: &Frame) -> LinkResult<()> {
        let p = f.encode();
        self.send_raw(&p).await
    }

    /// Data frame, lz4-compressed when `compress` is on and it helps.
    pub async fn send_data(&mut self, data: &[u8], compress: bool) -> LinkResult<()> {
        let p = encode_data(data, compress);
        self.send_raw(&p).await
    }

    pub async fn shutdown(&mut self) {
        let _ = self.w.shutdown().await;
    }
}

impl SecureReader {
    /// Next decrypted plaintext. A tampered, replayed or reordered message
    /// fails authentication (the nonce is the message counter).
    pub async fn recv_plain(&mut self) -> LinkResult<&[u8]> {
        read_frame(&mut self.r, &mut self.cbuf).await?;
        let n = self
            .t
            .read_message(self.nonce, &self.cbuf, &mut self.pbuf)
            .map_err(|_| LinkError::Protocol("a tunnel message failed authentication".into()))?;
        self.nonce += 1;
        Ok(&self.pbuf[..n])
    }

    /// [`Self::recv_plain`], editable in place (the helper's guard zeroes a few bytes).
    pub async fn recv_plain_mut(&mut self) -> LinkResult<&mut [u8]> {
        read_frame(&mut self.r, &mut self.cbuf).await?;
        let n = self
            .t
            .read_message(self.nonce, &self.cbuf, &mut self.pbuf)
            .map_err(|_| LinkError::Protocol("a tunnel message failed authentication".into()))?;
        self.nonce += 1;
        Ok(&mut self.pbuf[..n])
    }

    pub async fn recv(&mut self) -> LinkResult<Frame> {
        Frame::decode(self.recv_plain().await?)
    }
}

/// A tunnel frame (the plaintext of one Noise transport message).
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Data(Vec<u8>),
    Ping(u64),
    Pong(u64),
    /// The sender has no more data (half close).
    Close,
    Json(serde_json::Value),
}

const T_DATA: u8 = 0;
const T_LZ4: u8 = 1;
const T_PING: u8 = 2;
const T_PONG: u8 = 3;
const T_CLOSE: u8 = 4;
const T_JSON: u8 = 5;

/// The payload of an uncompressed Data frame, without copying it.
pub fn plain_data(p: &[u8]) -> Option<&[u8]> {
    match p.split_first() {
        Some((&T_DATA, body)) => Some(body),
        _ => None,
    }
}

/// [`plain_data`], editable in place.
pub fn plain_data_mut(p: &mut [u8]) -> Option<&mut [u8]> {
    match p.split_first_mut() {
        Some((&mut T_DATA, body)) => Some(body),
        _ => None,
    }
}

pub fn encode_data(data: &[u8], compress: bool) -> Vec<u8> {
    if compress && data.len() > 64 {
        let c = lz4_flex::compress_prepend_size(data);
        if c.len() + 8 < data.len() {
            let mut v = Vec::with_capacity(1 + c.len());
            v.push(T_LZ4);
            v.extend_from_slice(&c);
            return v;
        }
    }
    let mut v = Vec::with_capacity(1 + data.len());
    v.push(T_DATA);
    v.extend_from_slice(data);
    v
}

impl Frame {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Frame::Data(d) => encode_data(d, false),
            Frame::Ping(n) => [&[T_PING][..], &n.to_be_bytes()].concat(),
            Frame::Pong(n) => [&[T_PONG][..], &n.to_be_bytes()].concat(),
            Frame::Close => vec![T_CLOSE],
            Frame::Json(v) => [&[T_JSON][..], v.to_string().as_bytes()].concat(),
        }
    }

    pub fn decode(p: &[u8]) -> LinkResult<Frame> {
        let bad = |w: &str| LinkError::Protocol(format!("malformed {w} frame"));
        let (&t, body) = p.split_first().ok_or_else(|| bad("empty"))?;
        Ok(match t {
            T_DATA => Frame::Data(body.to_vec()),
            T_LZ4 => {
                // Refuse a declared size above one chunk (no decompression bombs).
                let declared = body.get(..4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or_else(|| bad("lz4"))?;
                if declared > MAX_FRAME {
                    return Err(bad("lz4"));
                }
                Frame::Data(lz4_flex::decompress_size_prepended(body).map_err(|_| bad("lz4"))?)
            }
            T_PING | T_PONG => {
                let n = u64::from_be_bytes(body.try_into().map_err(|_| bad("ping"))?);
                if t == T_PING { Frame::Ping(n) } else { Frame::Pong(n) }
            }
            T_CLOSE => Frame::Close,
            T_JSON => Frame::Json(serde_json::from_slice(body).map_err(|_| bad("json"))?),
            _ => return Err(bad("unknown")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip() {
        let big: Vec<u8> = (0..DATA_CHUNK).map(|i| (i % 7) as u8).collect();
        for f in [
            Frame::Data(b"hello".to_vec()),
            Frame::Data(Vec::new()),
            Frame::Data(big.clone()),
            Frame::Ping(42),
            Frame::Pong(u64::MAX),
            Frame::Close,
            Frame::Json(serde_json::json!({"t": "info"})),
        ] {
            assert_eq!(Frame::decode(&f.encode()).unwrap(), f);
        }
        // Compression is used only when it helps, and decodes to the same bytes.
        let c = encode_data(&big, true);
        assert_eq!(c[0], T_LZ4);
        assert!(c.len() < big.len() / 10);
        assert_eq!(Frame::decode(&c).unwrap(), Frame::Data(big));
        let random: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let r = encode_data(&random, true);
        assert_eq!(Frame::decode(&r).unwrap(), Frame::Data(random));
    }

    #[test]
    fn malformed_frames_are_rejected() {
        assert!(Frame::decode(&[]).is_err());
        assert!(Frame::decode(&[T_PING, 1, 2]).is_err());
        assert!(Frame::decode(&[99]).is_err());
        assert!(Frame::decode(&[T_JSON, b'{']).is_err());
        // An lz4 frame that claims to expand to 4 GB.
        assert!(Frame::decode(&[T_LZ4, 0xff, 0xff, 0xff, 0xff, 0]).is_err());
    }

    #[test]
    fn hello_and_status_frames() {
        assert_eq!(parse_hello(&hello(Mode::Rpc)).unwrap(), Mode::Rpc);
        assert!(parse_hello(b"GET / HTTP/1.1").is_err());
        let mut old = hello(Mode::Pair);
        old[7] = 0;
        assert!(matches!(parse_hello(&old), Err(LinkError::Refused(_))));
        assert_eq!(parse_status(&status_ok(b"xy")).unwrap(), b"xy");
        let e = parse_status(&status_err(&LinkError::WrongCode { remaining: 3 })).unwrap_err();
        assert!(matches!(e, LinkError::WrongCode { remaining: 3 }), "{e:?}");
        assert!(matches!(parse_status(&status_err(&LinkError::CodeLocked)), Err(LinkError::CodeLocked)));
    }

    #[tokio::test]
    async fn framing_over_a_socket_and_tamper_detection() {
        // Two Noise sessions over loopback; a flipped bit in transit must fail.
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let (a, b) = tokio::join!(tokio::net::TcpStream::connect(addr), l.accept());
        let (a, (b, _)) = (a.unwrap(), b.unwrap());
        let ia = crate::Identity::generate().unwrap();
        let ib = crate::Identity::generate().unwrap();
        let (req, h) = (serde_json::json!({}), hello(Mode::Control));
        let (ra, rb) = tokio::join!(crate::handshake::link_initiate(a, &ia, Mode::Control, *ib.public(), &req), async {
            // The listener reads the hello before handing the stream over.
            let mut b = b;
            let mut got = Vec::new();
            read_frame(&mut b, &mut got).await.unwrap();
            assert_eq!(got, h);
            crate::handshake::link_respond(b, &ib, &h).await
        },);
        let mut sa = ra.unwrap();
        let mut sb = rb.unwrap();
        sb.writer.send(&Frame::Json(serde_json::json!({"ok": true}))).await.unwrap();
        assert_eq!(crate::handshake::read_accept(&mut sa.reader).await.unwrap(), serde_json::json!({"ok": true}));
        sa.writer.send(&Frame::Ping(7)).await.unwrap();
        assert_eq!(sb.reader.recv().await.unwrap(), Frame::Ping(7));
        // Forge: a frame that was not produced by the sender's session.
        let junk = vec![0u8; 40];
        write_frame(&mut sa.writer.w, &junk).await.unwrap();
        assert!(sb.reader.recv().await.is_err(), "forged message accepted");
    }
}

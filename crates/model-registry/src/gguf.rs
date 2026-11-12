//! Minimal, defensive GGUF header reader.
//!
//! Reads the key/value metadata section only (never tensor data) and keeps the
//! fields the app needs: architecture, trained context, layer and head counts,
//! file type (quantization), chat template and split info. Large arrays such as
//! the tokenizer vocabulary are skipped, keeping only their length.
//! Every length is bounds-checked so a corrupt or hostile file returns an error
//! instead of allocating unbounded memory.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::Serialize;

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING: u64 = 64 * 1024 * 1024; // chat templates can be large, but not this large
const MAX_KV: u64 = 1_000_000;
const MAX_ARRAY: u64 = 50_000_000;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    NotGguf,
    UnsupportedVersion(u32),
    Corrupt(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "read error: {e}"),
            GgufError::NotGguf => write!(f, "not a GGUF file (bad magic bytes)"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Corrupt(s) => write!(f, "corrupt GGUF header: {s}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::Corrupt("file ends inside the header".into())
        } else {
            GgufError::Io(e)
        }
    }
}

/// A scalar metadata value. Arrays keep only their element type and length,
/// except short arrays of numbers, which are kept (some models store per-layer
/// head counts as arrays).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    NumArray(Vec<i64>),
    /// Short bool arrays (per-layer flags such as gemma4's sliding-window pattern).
    BoolArray(Vec<bool>),
    Array { elem_type: u32, len: u64 },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            Value::NumArray(a) => a.iter().copied().max().map(|v| v.max(0) as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::U64(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub kv: BTreeMap<String, Value>,
}

/// Fields the app uses, extracted from the raw key/value map.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize, PartialEq)]
pub struct ModelMetadata {
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u64>,
    /// Quantization name derived from `general.file_type`.
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub rope_freq_base: Option<f64>,
    pub expert_count: Option<u64>,
    pub vocab_size: Option<u64>,
    pub has_chat_template: bool,
    /// Length in characters; the template itself is read from the running server.
    pub chat_template_len: Option<usize>,
    pub split_count: Option<u64>,
    pub split_no: Option<u64>,
    /// Vision projector files (mmproj) are not chat models on their own.
    pub is_projector: bool,
    pub gguf_version: u32,
    pub tensor_count: u64,
    /// Thinking / tools / vision support (see `caps`). Missing in rows stored
    /// before detection existed; those are back-filled by `refresh_status`.
    #[serde(default)]
    pub caps: crate::caps::Capabilities,
    // Attention layout, for the KV cache estimate (`estimate`). All None in
    // rows stored before they were read (back-filled by `refresh_status`) and
    // for models without them; None keeps the plain dense formula.
    /// `<arch>.attention.sliding_window`: tokens a sliding-window (SWA) layer keeps.
    #[serde(default)]
    pub sliding_window: Option<u64>,
    /// `<arch>.attention.sliding_window_pattern` as a number N: every Nth
    /// layer is global (full attention), the others use the window
    /// (llama.cpp `set_swa_pattern`).
    #[serde(default)]
    pub sliding_window_pattern: Option<u64>,
    /// The same key as a per-layer array (gemma4): true = sliding-window layer.
    #[serde(default)]
    pub swa_layers: Option<Vec<bool>>,
    /// K / V head size of the sliding-window layers, when it differs.
    #[serde(default)]
    pub key_length_swa: Option<u64>,
    #[serde(default)]
    pub value_length_swa: Option<u64>,
    /// `<arch>.attention.shared_kv_layers`: the last N layers reuse an earlier
    /// layer's KV cache and keep none of their own (gemma3n / gemma4).
    #[serde(default)]
    pub shared_kv_layers: Option<u64>,
    /// `<arch>.nextn_predict_layers`: multi-token-prediction (MTP / NextN)
    /// layers at the end of `block_count`. They have no KV cache in a normal
    /// load; llama-server can use them for speculative decoding (`--spec-type draft-mtp`).
    #[serde(default)]
    pub mtp_layers: Option<u64>,
    /// `<arch>.full_attention_interval` of hybrid (linear + full attention) models.
    #[serde(default)]
    pub full_attention_interval: Option<u64>,
}

pub fn read_header(path: &Path) -> Result<GgufHeader, GgufError> {
    let f = File::open(path)?;
    let file_len = f.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 16, f);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(GgufError::NotGguf);
    }
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > MAX_KV {
        return Err(GgufError::Corrupt(format!("{kv_count} metadata entries")));
    }
    let mut kv = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r, file_len)?;
        let ty = read_u32(&mut r)?;
        let val = read_value(&mut r, ty, file_len)?;
        kv.insert(key, val);
    }
    Ok(GgufHeader { version, tensor_count, kv })
}

pub fn read_metadata(path: &Path) -> Result<ModelMetadata, GgufError> {
    let h = read_header(path)?;
    Ok(extract(&h))
}

pub fn extract(h: &GgufHeader) -> ModelMetadata {
    let get = |k: &str| h.kv.get(k);
    let arch = get("general.architecture").and_then(|v| v.as_str()).map(str::to_string);
    let a = arch.clone().unwrap_or_default();
    let ak = |suffix: &str| get(&format!("{a}.{suffix}"));
    let file_type = get("general.file_type").and_then(Value::as_u64);
    let chat_template = get("tokenizer.chat_template").and_then(|v| v.as_str());
    let vocab_size = match get("tokenizer.ggml.tokens") {
        Some(Value::Array { len, .. }) => Some(*len),
        _ => ak("vocab_size").and_then(Value::as_u64),
    };
    ModelMetadata {
        is_projector: a == "clip" || get("clip.has_vision_encoder").is_some(),
        architecture: arch,
        name: get("general.name").and_then(|v| v.as_str()).map(str::to_string),
        size_label: get("general.size_label").and_then(|v| v.as_str()).map(str::to_string),
        file_type,
        quantization: file_type.and_then(file_type_name).map(str::to_string),
        context_length: ak("context_length").and_then(Value::as_u64),
        block_count: ak("block_count").and_then(Value::as_u64),
        embedding_length: ak("embedding_length").and_then(Value::as_u64),
        head_count: ak("attention.head_count").and_then(Value::as_u64),
        head_count_kv: ak("attention.head_count_kv").and_then(Value::as_u64),
        key_length: ak("attention.key_length").and_then(Value::as_u64),
        value_length: ak("attention.value_length").and_then(Value::as_u64),
        rope_freq_base: ak("rope.freq_base").and_then(Value::as_f64),
        expert_count: ak("expert_count").and_then(Value::as_u64),
        vocab_size,
        has_chat_template: chat_template.is_some(),
        chat_template_len: chat_template.map(str::len),
        split_count: get("split.count").and_then(Value::as_u64),
        split_no: get("split.no").and_then(Value::as_u64),
        gguf_version: h.version,
        tensor_count: h.tensor_count,
        caps: crate::caps::from_template(chat_template, "gguf"),
        sliding_window: ak("attention.sliding_window").and_then(Value::as_u64).filter(|&w| w > 0),
        sliding_window_pattern: match ak("attention.sliding_window_pattern") {
            Some(Value::U64(n)) => Some(*n),
            Some(Value::I64(n)) if *n >= 0 => Some(*n as u64),
            _ => None,
        },
        swa_layers: match ak("attention.sliding_window_pattern") {
            Some(Value::BoolArray(v)) => Some(v.clone()),
            Some(Value::NumArray(v)) => Some(v.iter().map(|&x| x != 0).collect()),
            _ => None,
        },
        key_length_swa: ak("attention.key_length_swa").and_then(Value::as_u64),
        value_length_swa: ak("attention.value_length_swa").and_then(Value::as_u64),
        shared_kv_layers: ak("attention.shared_kv_layers").and_then(Value::as_u64).filter(|&n| n > 0),
        mtp_layers: ak("nextn_predict_layers").and_then(Value::as_u64).filter(|&n| n > 0),
        full_attention_interval: ak("full_attention_interval").and_then(Value::as_u64).filter(|&n| n > 0),
    }
}

/// llama.cpp `llama_ftype` values to the names used in GGUF file names.
pub fn file_type_name(ft: u64) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R, file_len: u64) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    if len > MAX_STRING || len > file_len {
        return Err(GgufError::Corrupt(format!("string length {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Byte size of a fixed-width GGUF scalar type, or None for string/array.
fn scalar_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_scalar<R: Read>(r: &mut R, ty: u32) -> Result<Value, GgufError> {
    let mut b = [0u8; 8];
    Ok(match ty {
        0 => {
            r.read_exact(&mut b[..1])?;
            Value::U64(b[0] as u64)
        }
        1 => {
            r.read_exact(&mut b[..1])?;
            Value::I64(b[0] as i8 as i64)
        }
        2 => {
            r.read_exact(&mut b[..2])?;
            Value::U64(u16::from_le_bytes([b[0], b[1]]) as u64)
        }
        3 => {
            r.read_exact(&mut b[..2])?;
            Value::I64(i16::from_le_bytes([b[0], b[1]]) as i64)
        }
        4 => Value::U64(read_u32(r)? as u64),
        5 => Value::I64(read_u32(r)? as i32 as i64),
        6 => Value::F64(f32::from_bits(read_u32(r)?) as f64),
        7 => {
            r.read_exact(&mut b[..1])?;
            Value::Bool(b[0] != 0)
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(read_u64(r)? as i64),
        12 => Value::F64(f64::from_bits(read_u64(r)?)),
        other => return Err(GgufError::Corrupt(format!("unknown value type {other}"))),
    })
}

fn read_value<R: Read + Seek>(r: &mut R, ty: u32, file_len: u64) -> Result<Value, GgufError> {
    match ty {
        8 => Ok(Value::Str(read_string(r, file_len)?)),
        9 => {
            let elem = read_u32(r)?;
            let len = read_u64(r)?;
            if len > MAX_ARRAY {
                return Err(GgufError::Corrupt(format!("array length {len}")));
            }
            if let Some(sz) = scalar_size(elem) {
                let bytes = len
                    .checked_mul(sz)
                    .filter(|b| *b <= file_len)
                    .ok_or_else(|| GgufError::Corrupt("array exceeds file size".into()))?;
                // Keep short integer and bool arrays (per-layer values); skip the rest.
                if len <= 512 && elem == 7 {
                    let mut v = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        if let Value::Bool(x) = read_scalar(r, elem)? {
                            v.push(x);
                        }
                    }
                    return Ok(Value::BoolArray(v));
                }
                if len <= 512 && elem != 6 && elem != 12 {
                    let mut v = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        match read_scalar(r, elem)? {
                            Value::U64(x) => v.push(x as i64),
                            Value::I64(x) => v.push(x),
                            _ => {}
                        }
                    }
                    return Ok(Value::NumArray(v));
                }
                r.seek(SeekFrom::Current(bytes as i64))?;
            } else if elem == 8 {
                for _ in 0..len {
                    let n = read_u64(r)?;
                    if n > MAX_STRING || n > file_len {
                        return Err(GgufError::Corrupt(format!("string length {n}")));
                    }
                    r.seek(SeekFrom::Current(n as i64))?;
                }
            } else {
                return Err(GgufError::Corrupt(format!("unsupported array element type {elem}")));
            }
            Ok(Value::Array { elem_type: elem, len })
        }
        _ => read_scalar(r, ty),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny synthetic GGUF v3 header for tests.
    pub fn write_fake_gguf(path: &Path, arch: &str, ctx: u64, file_type: u32, extra_split: Option<(u16, u16)>) {
        let mut b: Vec<u8> = Vec::new();
        let mut kvs: Vec<Vec<u8>> = Vec::new();
        let s = |k: &str, v: &str| {
            let mut e = Vec::new();
            e.extend((k.len() as u64).to_le_bytes());
            e.extend(k.as_bytes());
            e.extend(8u32.to_le_bytes());
            e.extend((v.len() as u64).to_le_bytes());
            e.extend(v.as_bytes());
            e
        };
        let u32kv = |k: &str, v: u32| {
            let mut e = Vec::new();
            e.extend((k.len() as u64).to_le_bytes());
            e.extend(k.as_bytes());
            e.extend(4u32.to_le_bytes());
            e.extend(v.to_le_bytes());
            e
        };
        kvs.push(s("general.architecture", arch));
        kvs.push(s("general.name", "Fake Model"));
        kvs.push(u32kv("general.file_type", file_type));
        kvs.push(u32kv(&format!("{arch}.context_length"), ctx as u32));
        kvs.push(u32kv(&format!("{arch}.block_count"), 24));
        kvs.push(u32kv(&format!("{arch}.attention.head_count"), 16));
        kvs.push(u32kv(&format!("{arch}.attention.head_count_kv"), 4));
        kvs.push(u32kv(&format!("{arch}.embedding_length"), 2048));
        kvs.push(s("tokenizer.chat_template", "{{ messages }}"));
        // string array (vocab) that must be skipped
        let mut arr = Vec::new();
        let k = "tokenizer.ggml.tokens";
        arr.extend((k.len() as u64).to_le_bytes());
        arr.extend(k.as_bytes());
        arr.extend(9u32.to_le_bytes());
        arr.extend(8u32.to_le_bytes());
        arr.extend(3u64.to_le_bytes());
        for t in ["a", "bb", "ccc"] {
            arr.extend((t.len() as u64).to_le_bytes());
            arr.extend(t.as_bytes());
        }
        kvs.push(arr);
        if let Some((no, count)) = extra_split {
            let mut e = Vec::new();
            for (k, v) in [("split.no", no), ("split.count", count)] {
                e.extend((k.len() as u64).to_le_bytes());
                e.extend(k.as_bytes());
                e.extend(2u32.to_le_bytes());
                e.extend(v.to_le_bytes());
            }
            kvs.push(e);
        }
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        let n = if extra_split.is_some() { kvs.len() + 1 } else { kvs.len() };
        b.extend((n as u64).to_le_bytes());
        for kv in kvs {
            b.extend(kv);
        }
        b.extend([0u8; 64]);
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn parses_synthetic_header() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("m.gguf");
        write_fake_gguf(&p, "llama", 8192, 15, None);
        let m = read_metadata(&p).unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.context_length, Some(8192));
        assert_eq!(m.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(m.head_count_kv, Some(4));
        assert_eq!(m.vocab_size, Some(3));
        assert!(m.has_chat_template);
    }

    /// One GGUF key/value entry: key, type id, raw value bytes.
    fn entry(k: &str, ty: u32, val: &[u8]) -> Vec<u8> {
        let mut e = Vec::new();
        e.extend((k.len() as u64).to_le_bytes());
        e.extend(k.as_bytes());
        e.extend(ty.to_le_bytes());
        e.extend(val);
        e
    }

    fn write_kvs(path: &Path, kvs: &[Vec<u8>]) {
        let mut b: Vec<u8> = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            b.extend(kv);
        }
        b.extend([0u8; 64]);
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn reads_sliding_window_shared_kv_and_mtp_keys() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("g.gguf");
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let u = |v: u32| v.to_le_bytes().to_vec();
        // gemma4: the pattern is a per-layer bool array (true = sliding window).
        let mut bools = 7u32.to_le_bytes().to_vec();
        bools.extend(6u64.to_le_bytes());
        bools.extend([1u8, 1, 1, 1, 1, 0]);
        write_kvs(
            &p,
            &[
                entry("general.architecture", 8, &s("gemma4")),
                entry("gemma4.block_count", 4, &u(6)),
                entry("gemma4.attention.sliding_window", 4, &u(512)),
                entry("gemma4.attention.sliding_window_pattern", 9, &bools),
                entry("gemma4.attention.key_length_swa", 4, &u(256)),
                entry("gemma4.attention.value_length_swa", 4, &u(256)),
                entry("gemma4.attention.shared_kv_layers", 4, &u(2)),
            ],
        );
        let m = read_metadata(&p).unwrap();
        assert_eq!(m.sliding_window, Some(512));
        assert_eq!(m.swa_layers, Some(vec![true, true, true, true, true, false]));
        assert_eq!(m.sliding_window_pattern, None);
        assert_eq!((m.key_length_swa, m.value_length_swa, m.shared_kv_layers), (Some(256), Some(256), Some(2)));
        assert_eq!(m.mtp_layers, None);
        // A numeric pattern and MTP layers (qwen35 style keys).
        write_kvs(
            &p,
            &[
                entry("general.architecture", 8, &s("qwen35")),
                entry("qwen35.attention.sliding_window", 4, &u(4096)),
                entry("qwen35.attention.sliding_window_pattern", 4, &u(4)),
                entry("qwen35.nextn_predict_layers", 4, &u(1)),
                entry("qwen35.full_attention_interval", 4, &u(4)),
            ],
        );
        let m = read_metadata(&p).unwrap();
        assert_eq!((m.sliding_window_pattern, m.swa_layers.as_ref()), (Some(4), None));
        assert_eq!((m.mtp_layers, m.full_attention_interval), (Some(1), Some(4)));
        // Rows stored before these fields existed still parse (all None).
        let mut v = serde_json::to_value(ModelMetadata::default()).unwrap();
        for k in ["sliding_window", "swa_layers", "mtp_layers", "shared_kv_layers"] {
            v.as_object_mut().unwrap().remove(k);
        }
        let old: ModelMetadata = serde_json::from_value(v).unwrap();
        assert_eq!(old, ModelMetadata::default());
    }

    #[test]
    fn rejects_non_gguf_and_truncated_files() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("bad.gguf");
        std::fs::write(&p, b"PK\x03\x04 not a model").unwrap();
        assert!(matches!(read_metadata(&p), Err(GgufError::NotGguf)));
        let mut t = b"GGUF".to_vec();
        t.extend(3u32.to_le_bytes());
        t.extend(0u64.to_le_bytes());
        t.extend(5u64.to_le_bytes()); // claims 5 entries, has none
        std::fs::write(&p, &t).unwrap();
        assert!(matches!(read_metadata(&p), Err(GgufError::Corrupt(_))));
        // absurd string length must not allocate
        let mut a = b"GGUF".to_vec();
        a.extend(3u32.to_le_bytes());
        a.extend(0u64.to_le_bytes());
        a.extend(1u64.to_le_bytes());
        a.extend(u64::MAX.to_le_bytes());
        std::fs::write(&p, &a).unwrap();
        assert!(matches!(read_metadata(&p), Err(GgufError::Corrupt(_))));
    }

    /// Parses the real GGUF files in LM Studio's folder, if present.
    #[test]
    fn parses_real_lmstudio_models_if_present() {
        let Some(home) = std::env::var_os("USERPROFILE") else { return };
        let root = Path::new(&home).join(".lmstudio").join("models");
        if !root.is_dir() {
            return;
        }
        for e in walkdir::WalkDir::new(&root).into_iter().flatten() {
            if e.path().extension().is_some_and(|x| x == "gguf") {
                let m = read_metadata(e.path()).unwrap_or_else(|err| panic!("{}: {err}", e.path().display()));
                println!(
                    "{} arch={:?} ctx={:?} quant={:?} layers={:?} kv_heads={:?} projector={}",
                    e.file_name().to_string_lossy(),
                    m.architecture,
                    m.context_length,
                    m.quantization,
                    m.block_count,
                    m.head_count_kv,
                    m.is_projector
                );
            }
        }
    }
}

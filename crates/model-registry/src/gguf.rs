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
const MAX_TENSORS: u64 = 100_000;
const MAX_TENSOR_NAME: u64 = 4096;
/// Most tensor-name bytes read from one file's tensor-info section.
const MAX_TENSOR_NAME_BYTES: u64 = 64 * 1024 * 1024;
/// Highest `block_count` taken from a header (the largest real models have a
/// few hundred); a hostile value would make per-layer estimate loops huge.
pub const MAX_BLOCK_COUNT: u64 = 4096;
const MAX_TENSOR_DIMS: u32 = 8;

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
    /// MTP / NextN tensor names (containing "nextn") from the tensor-info
    /// section (names only, never data). Other names are not kept.
    #[serde(skip)]
    pub tensor_names: Vec<String>,
    /// The whole tensor-info section was read (false: corrupt or over the
    /// limits, so `tensor_names` says nothing about which tensors exist).
    #[serde(skip)]
    pub tensors_read: bool,
    /// Bytes of the tensors llama.cpp keeps in host memory whatever `-ngl`
    /// says ([`HOST_TENSORS`]), from their dims and type (never their data).
    /// None when there are none or a size is unknown.
    #[serde(skip)]
    pub host_weight_bytes: Option<u64>,
}

/// Tensors llama.cpp always keeps in host (CPU) memory, not on the GPU:
/// Gemma 3n / Gemma 4's per-layer embedding table ("PLE", an input-layer
/// tensor read with get_rows). Gemma-4-E4B Q4_K_M: 2.31 GB of the 5.34 GB
/// file; measured with every layer on an 8 GB GPU at -c 98304: 5264 MiB VRAM
/// in total, KV 1576 MiB.
pub const HOST_TENSORS: &[&str] = &["per_layer_token_embd.weight"];

/// (elements per block, bytes per block) of a GGML tensor type (ggml.h ids);
/// None for types not in the table.
pub fn ggml_type_size(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 | 28 => (1, 8), // I64, F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => return None,
    })
}

/// Bytes of a tensor with `dims` and GGML type `ty`; None for an unknown
/// type, a row that is not a whole number of blocks, or an overflow.
pub fn tensor_bytes(dims: &[u64], ty: u32) -> Option<u64> {
    let (blk, sz) = ggml_type_size(ty)?;
    if dims.first().is_some_and(|&d| d % blk != 0) {
        return None;
    }
    let ne = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))?;
    (ne / blk).checked_mul(sz)
}

/// What the tensor-info section tells the app.
#[derive(Debug, Default)]
struct TensorInfo {
    names: Vec<String>,
    host_bytes: Option<u64>,
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
    /// Set only when the file really contains the MTP tensors; see
    /// `mtp_layers_declared` for the raw header value.
    #[serde(default)]
    pub mtp_layers: Option<u64>,
    /// Raw `<arch>.nextn_predict_layers`. Some files keep the key but had the
    /// MTP tensors stripped (declared > 0 while `mtp_layers` is None).
    #[serde(default)]
    pub mtp_layers_declared: Option<u64>,
    /// MTP is declared but the tensor list (of some shard) could not be read,
    /// so it is unknown whether the MTP tensors are there: not "removed", and
    /// speculative MTP is not turned on by default.
    #[serde(default)]
    pub mtp_unknown: bool,
    /// Not stored: set by the caller for one estimate when speculative MTP
    /// decoding is on, so the MTP layer's own KV cache is counted.
    #[serde(skip)]
    pub mtp_speculative: bool,
    /// `<arch>.full_attention_interval` of hybrid (linear + full attention) models.
    #[serde(default)]
    pub full_attention_interval: Option<u64>,
    /// Bytes of weights llama.cpp keeps in host memory with every layer on
    /// the GPU ([`HOST_TENSORS`]: Gemma 3n / 4 per-layer embeddings). The
    /// estimate counts them in RAM, not VRAM. None: none (or unknown).
    #[serde(default)]
    pub host_weight_bytes: Option<u64>,
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
    let (tensor_names, tensors_read, host_weight_bytes) = match read_tensor_names(&mut r, tensor_count, file_len) {
        Ok(t) => (t.names, true, t.host_bytes),
        Err(_) => (Vec::new(), false, None),
    };
    Ok(GgufHeader { version, tensor_count, kv, tensor_names, tensors_read, host_weight_bytes })
}

/// Reads the tensor-info section (each entry: name, n_dims u32, dims u64 x n,
/// type u32, offset u64) and keeps only the MTP / NextN names (containing
/// "nextn") and the size of the [`HOST_TENSORS`] (from dims and type; a
/// tensor of unknown type is not counted). Bounded by `MAX_TENSORS`,
/// `MAX_TENSOR_NAME`, `MAX_TENSOR_NAME_BYTES` and the file length. Errors
/// mean "unknown", not fatal.
fn read_tensor_names<R: Read>(r: &mut R, count: u64, file_len: u64) -> Result<TensorInfo, GgufError> {
    if count > MAX_TENSORS {
        return Err(GgufError::Corrupt(format!("{count} tensors")));
    }
    let mut names = Vec::new();
    let mut host = 0u64;
    let mut name_bytes = 0u64;
    let mut buf = Vec::with_capacity(256);
    for _ in 0..count {
        let len = read_u64(r)?;
        if len > MAX_TENSOR_NAME || len > file_len {
            return Err(GgufError::Corrupt(format!("tensor name length {len}")));
        }
        name_bytes = name_bytes.saturating_add(len);
        if name_bytes > MAX_TENSOR_NAME_BYTES {
            return Err(GgufError::Corrupt("tensor names too long in total".into()));
        }
        buf.resize(len as usize, 0);
        r.read_exact(&mut buf)?;
        let name = String::from_utf8_lossy(&buf).into_owned();
        let n_dims = read_u32(r)?;
        if n_dims > MAX_TENSOR_DIMS {
            return Err(GgufError::Corrupt(format!("{n_dims} tensor dims")));
        }
        let mut dims = [0u64; MAX_TENSOR_DIMS as usize];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = read_u64(r)?;
        }
        let ty = read_u32(r)?;
        let _offset = read_u64(r)?;
        if HOST_TENSORS.contains(&name.as_str())
            && let Some(b) = tensor_bytes(&dims[..n_dims as usize], ty).filter(|&b| b <= file_len)
        {
            host = host.saturating_add(b);
        }
        if name.contains("nextn") {
            names.push(name);
        }
    }
    Ok(TensorInfo { names, host_bytes: Some(host).filter(|&b| b > 0) })
}

/// Number of MTP layers whose tensors are present: `nextn` tensors in the last
/// `declared` block indices (blk.<block_count - declared + i>.nextn.*). 0 when
/// the tensors were stripped or the tensor section is unreadable.
pub fn count_mtp_layers(declared: u64, block_count: Option<u64>, names: &[String]) -> u64 {
    if declared == 0 {
        return 0;
    }
    let first = block_count.map(|b| b.saturating_sub(declared));
    let mut blocks = std::collections::BTreeSet::new();
    let mut any_nextn = false;
    for n in names {
        let idx = n.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<u64>().ok());
        if n.contains("nextn") {
            any_nextn = true;
        }
        if let (Some(i), Some(f)) = (idx, first) {
            if i >= f && n.contains("nextn") {
                blocks.insert(i);
            }
        }
    }
    if blocks.is_empty() && any_nextn { declared } else { (blocks.len() as u64).min(declared) }
}

pub fn read_metadata(path: &Path) -> Result<ModelMetadata, GgufError> {
    let h = read_header(path)?;
    Ok(extract(&h))
}

/// [`read_metadata`] for a model split into shards: the key/value metadata
/// comes from `first`, the MTP tensor names from every shard (the NextN
/// tensors are usually in the last one). `shards` may include `first`.
pub fn read_metadata_shards(first: &Path, shards: &[std::path::PathBuf]) -> Result<ModelMetadata, GgufError> {
    let mut h = read_header(first)?;
    add_shard_tensors(&mut h, first, shards);
    Ok(extract(&h))
}

/// Add the MTP tensor names of the other shards of a split model to `h` (the
/// header of `first`). A shard that cannot be read makes the tensor list
/// incomplete (`tensors_read` false). Only done when the header declares MTP
/// layers whose tensors `first` does not hold, or for an architecture with
/// per-layer embeddings ([`HOST_TENSORS`]) whose table is not in `first`.
pub fn add_shard_tensors(h: &mut GgufHeader, first: &Path, shards: &[std::path::PathBuf]) {
    let a = h.kv.get("general.architecture").and_then(Value::as_str).unwrap_or_default().to_string();
    let declared = h.kv.get(&format!("{a}.nextn_predict_layers")).and_then(Value::as_u64).unwrap_or(0);
    let blocks = h.kv.get(&format!("{a}.block_count")).and_then(Value::as_u64);
    let need_mtp = declared > 0 && count_mtp_layers(declared, blocks, &h.tensor_names) < declared;
    let need_host = matches!(a.as_str(), "gemma3n" | "gemma4") && h.host_weight_bytes.is_none();
    if !need_mtp && !need_host {
        return;
    }
    for p in shards.iter().filter(|p| p.as_path() != first) {
        match read_header(p) {
            Ok(s) => {
                h.tensors_read &= s.tensors_read;
                h.tensor_names.extend(s.tensor_names);
                if let Some(b) = s.host_weight_bytes {
                    h.host_weight_bytes = Some(h.host_weight_bytes.unwrap_or(0).saturating_add(b));
                }
            }
            Err(_) => h.tensors_read = false,
        }
    }
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
    let declared_mtp = ak("nextn_predict_layers").and_then(Value::as_u64).filter(|&n| n > 0);
    let block_count = ak("block_count").and_then(Value::as_u64).map(|b| b.min(MAX_BLOCK_COUNT));
    let mtp_found = count_mtp_layers(declared_mtp.unwrap_or(0), block_count, &h.tensor_names);
    // Declared, none found, and the tensor list is incomplete: unknown, not stripped.
    let mtp_unknown = declared_mtp.is_some() && mtp_found == 0 && !h.tensors_read;
    ModelMetadata {
        is_projector: a == "clip" || get("clip.has_vision_encoder").is_some(),
        architecture: arch,
        name: get("general.name").and_then(|v| v.as_str()).map(str::to_string),
        size_label: get("general.size_label").and_then(|v| v.as_str()).map(str::to_string),
        file_type,
        quantization: file_type.and_then(file_type_name).map(str::to_string),
        context_length: ak("context_length").and_then(Value::as_u64),
        block_count,
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
        mtp_layers: Some(mtp_found).filter(|&n| n > 0),
        mtp_layers_declared: declared_mtp,
        mtp_unknown,
        mtp_speculative: false,
        full_attention_interval: ak("full_attention_interval").and_then(Value::as_u64).filter(|&n| n > 0),
        host_weight_bytes: h.host_weight_bytes.filter(|&b| b > 0),
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
        // Declared in the header but this synthetic file has no nextn tensors:
        // usable MTP layers are counted from the tensors only (0 here).
        assert_eq!((m.mtp_layers_declared, m.full_attention_interval), (Some(1), Some(4)));
        assert_eq!(m.mtp_layers.unwrap_or(0), 0);
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

    /// Synthetic GGUF with a qwen35-style header and the given tensor names.
    fn write_with_tensors(path: &Path, declared: u32, block_count: u32, names: &[&str]) {
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let kvs = [
            entry("general.architecture", 8, &s("qwen35")),
            entry("qwen35.block_count", 4, &block_count.to_le_bytes()),
            entry("qwen35.nextn_predict_layers", 4, &declared.to_le_bytes()),
        ];
        let mut b: Vec<u8> = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((names.len() as u64).to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        kvs.iter().for_each(|k| b.extend(k));
        for n in names {
            b.extend(s(n));
            b.extend(2u32.to_le_bytes());
            b.extend([4u64.to_le_bytes(), 8u64.to_le_bytes()].concat());
            b.extend(0u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b.extend([0u8; 64]);
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn mtp_layers_need_the_nextn_tensors() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("t.gguf");
        let with = ["token_embd.weight", "blk.0.attn_q.weight", "blk.2.attn_q.weight", "blk.2.nextn.eh_proj.weight", "blk.2.nextn.enorm.weight"];
        write_with_tensors(&p, 1, 3, &with);
        let h = read_header(&p).unwrap();
        // Only the NextN names are kept.
        assert_eq!(h.tensor_names, vec!["blk.2.nextn.eh_proj.weight".to_string(), "blk.2.nextn.enorm.weight".to_string()]);
        assert!(h.tensors_read);
        let m = extract(&h);
        assert_eq!((m.mtp_layers, m.mtp_layers_declared, m.mtp_unknown), (Some(1), Some(1), false));
        // Key kept, MTP tensors stripped (the last block is an ordinary one, or gone).
        write_with_tensors(&p, 1, 3, &["token_embd.weight", "blk.0.attn_q.weight", "blk.2.attn_q.weight"]);
        let m = read_metadata(&p).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_layers_declared, m.mtp_unknown), (None, Some(1), false), "stripped, and known to be");
        write_with_tensors(&p, 1, 3, &["blk.0.attn_q.weight"]);
        assert_eq!(read_metadata(&p).unwrap().mtp_layers, None);
        // No key at all.
        write_with_tensors(&p, 0, 3, &with);
        let m = read_metadata(&p).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_layers_declared), (None, None));
    }

    #[test]
    fn malformed_tensor_section_is_bounded() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("b.gguf");
        write_with_tensors(&p, 1, 3, &["blk.2.nextn.enorm.weight"]);
        let mut raw = std::fs::read(&p).unwrap();
        raw[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); // absurd tensor count
        std::fs::write(&p, &raw).unwrap();
        let m = read_metadata(&p).unwrap(); // still opens; MTP is unknown, not "removed"
        assert_eq!((m.mtp_layers, m.mtp_layers_declared, m.mtp_unknown), (None, Some(1), true));
        raw[8..16].copy_from_slice(&5u64.to_le_bytes()); // more tensors than the file holds
        std::fs::write(&p, &raw).unwrap();
        let m = read_metadata(&p).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_unknown), (None, true));
        // Over the tensor-count cap: refused quickly, unknown.
        raw[8..16].copy_from_slice(&(MAX_TENSORS + 1).to_le_bytes());
        std::fs::write(&p, &raw).unwrap();
        assert!(read_metadata(&p).unwrap().mtp_unknown);
        // No MTP declared: an unreadable tensor list changes nothing.
        write_with_tensors(&p, 0, 3, &["blk.0.attn_q.weight"]);
        let mut raw = std::fs::read(&p).unwrap();
        raw[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&p, &raw).unwrap();
        assert!(!read_metadata(&p).unwrap().mtp_unknown);
    }

    #[test]
    fn tensor_names_are_capped_in_total_bytes() {
        // 17,000 names of 4 KiB each = 68 MiB of names: over the 64 MiB cap.
        let mut r: Vec<u8> = Vec::new();
        let name = "x".repeat(4096);
        for _ in 0..17_000 {
            r.extend((name.len() as u64).to_le_bytes());
            r.extend(name.as_bytes());
            r.extend(1u32.to_le_bytes());
            r.extend([0u8; 8 + 12]);
        }
        let len = r.len() as u64;
        let e = read_tensor_names(&mut std::io::Cursor::new(r), 17_000, len).unwrap_err();
        assert!(e.to_string().contains("too long"), "{e}");
    }

    #[test]
    fn hostile_block_count_is_clamped() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("h.gguf");
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        write_kvs(&p, &[entry("general.architecture", 8, &s("llama")), entry("llama.block_count", 10, &u64::MAX.to_le_bytes())]);
        assert_eq!(read_metadata(&p).unwrap().block_count, Some(MAX_BLOCK_COUNT));
    }

    /// A GGUF with the given key/values and tensor names (2-D, type 0, offset 0).
    fn write_kv_tensors(path: &Path, kvs: &[Vec<u8>], names: &[&str]) {
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let mut b: Vec<u8> = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((names.len() as u64).to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        kvs.iter().for_each(|k| b.extend(k));
        for n in names {
            b.extend(s(n));
            b.extend(2u32.to_le_bytes());
            b.extend([4u64.to_le_bytes(), 8u64.to_le_bytes()].concat());
            b.extend(0u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b.extend([0u8; 64]);
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn split_models_find_mtp_tensors_in_a_later_shard() {
        let d = tempfile::tempdir().unwrap();
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let u16v = |v: u16| v.to_le_bytes().to_vec();
        let first = d.path().join("m-00001-of-00002.gguf");
        let second = d.path().join("m-00002-of-00002.gguf");
        write_kv_tensors(
            &first,
            &[
                entry("general.architecture", 8, &s("qwen35")),
                entry("qwen35.block_count", 4, &3u32.to_le_bytes()),
                entry("qwen35.nextn_predict_layers", 4, &1u32.to_le_bytes()),
                entry("split.no", 2, &u16v(0)),
                entry("split.count", 2, &u16v(2)),
            ],
            &["token_embd.weight", "blk.0.attn_q.weight"],
        );
        write_kv_tensors(&second, &[entry("split.no", 2, &u16v(1)), entry("split.count", 2, &u16v(2))], &["blk.2.attn_q.weight", "blk.2.nextn.eh_proj.weight"]);
        let shards = vec![first.clone(), second.clone()];
        // The first shard alone looks stripped; with every shard the MTP layer is there.
        assert_eq!(read_metadata(&first).unwrap().mtp_layers, None);
        let m = read_metadata_shards(&first, &shards).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_layers_declared, m.mtp_unknown), (Some(1), Some(1), false));
        // A later shard without the tensors: stripped (known).
        write_kv_tensors(&second, &[entry("split.no", 2, &u16v(1))], &["blk.2.attn_q.weight"]);
        let m = read_metadata_shards(&first, &shards).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_unknown), (None, false));
        // A shard that cannot be read: unknown, never "removed".
        std::fs::write(&second, b"GGUF garbage").unwrap();
        let m = read_metadata_shards(&first, &shards).unwrap();
        assert_eq!((m.mtp_layers, m.mtp_unknown), (None, true));
        std::fs::remove_file(&second).unwrap();
        assert!(read_metadata_shards(&first, &shards).unwrap().mtp_unknown);
    }

    /// A GGUF with the given key/values and (name, dims, GGML type) tensors.
    fn write_typed_tensors(path: &Path, kvs: &[Vec<u8>], tensors: &[(&str, &[u64], u32)]) {
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let mut b: Vec<u8> = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        kvs.iter().for_each(|k| b.extend(k));
        for (n, dims, ty) in tensors {
            b.extend(s(n));
            b.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend(ty.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b.extend(vec![0u8; 4 << 20]); // room for the sizes to be plausible
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn per_layer_embedding_size_is_read_from_dims_and_type() {
        // Q6_K: 210 bytes per 256 elements; Gemma-4-E4B's table is [10752, 262144] -> 2,312,110,080 bytes.
        assert_eq!(tensor_bytes(&[10_752, 262_144], 14), Some(2_312_110_080));
        assert_eq!(tensor_bytes(&[2560, 4], 0), Some(2560 * 4 * 4));
        assert_eq!(tensor_bytes(&[100, 4], 14), None, "a row that is not whole blocks");
        assert_eq!(tensor_bytes(&[256, 4], 99), None, "unknown type: not counted");
        assert_eq!(tensor_bytes(&[u64::MAX, u64::MAX], 0), None, "overflow");
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("g.gguf");
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let kvs = [entry("general.architecture", 8, &s("gemma4")), entry("gemma4.block_count", 4, &2u32.to_le_bytes())];
        let t: [(&str, &[u64], u32); 3] = [("token_embd.weight", &[256, 64], 14), ("per_layer_token_embd.weight", &[512, 64], 14), ("blk.0.attn_q.weight", &[256, 256], 12)];
        write_typed_tensors(&p, &kvs, &t);
        let m = read_metadata(&p).unwrap();
        assert_eq!(m.host_weight_bytes, Some(512 * 64 / 256 * 210), "only the per-layer embedding table");
        // An unknown type: nothing is subtracted.
        write_typed_tensors(&p, &kvs, &[("per_layer_token_embd.weight", &[512, 64], 77)]);
        assert_eq!(read_metadata(&p).unwrap().host_weight_bytes, None);
        // A model without the table: None, and old rows parse (the field defaults).
        write_typed_tensors(&p, &kvs, &[("token_embd.weight", &[256, 64], 14)]);
        assert_eq!(read_metadata(&p).unwrap().host_weight_bytes, None);
        let mut v = serde_json::to_value(ModelMetadata::default()).unwrap();
        v.as_object_mut().unwrap().remove("host_weight_bytes");
        assert_eq!(serde_json::from_value::<ModelMetadata>(v).unwrap().host_weight_bytes, None);
    }

    /// Path of a real model file from the env var `var`; `None` (test skips) when unset or missing.
    fn real_model(var: &str) -> Option<String> {
        match std::env::var(var) {
            Ok(p) if Path::new(&p).is_file() => Some(p),
            _ => {
                eprintln!("skipping: set {var} to a real GGUF file");
                None
            }
        }
    }

    /// `cargo test -p model-registry real_gemma4_per_layer_embeddings -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_gemma4_per_layer_embeddings() {
        let Some(real_gemma4) = real_model("LAS_REAL_GEMMA4") else { return };
        let m = read_metadata(Path::new(&real_gemma4)).unwrap();
        println!("{real_gemma4}: host_weight_bytes={:?}", m.host_weight_bytes);
        // per_layer_token_embd.weight: [10752, 262144] Q6_K.
        assert_eq!(m.host_weight_bytes, Some(2_312_110_080));
    }


    #[test]
    #[ignore]
    fn real_files_mtp_tensors() {
        let (Some(stripped), Some(real)) = (real_model("LAS_REAL_MTP_STRIPPED"), real_model("LAS_REAL_MTP")) else { return };
        for (p, want) in [(stripped.as_str(), 0), (real.as_str(), 1)] {
            let h = read_header(Path::new(p)).unwrap();
            let m = extract(&h);
            let last = m.block_count.unwrap_or(0).saturating_sub(1);
            let names: Vec<_> = h.tensor_names.iter().filter(|n| n.contains("nextn") || n.starts_with(&format!("blk.{last}."))).collect();
            println!("{p}\n tensors={} read={} block_count={:?} declared={:?} mtp={:?}\n {names:?}", h.tensor_count, h.tensor_names.len(), m.block_count, m.mtp_layers_declared, m.mtp_layers);
            assert_eq!(m.mtp_layers.unwrap_or(0), want, "{p}");
        }
    }
}

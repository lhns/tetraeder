//! GGUF header reader: key-value metadata and the tensor index. Tensor data is never read here.
//! Format: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> (version 3, little endian).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::ggml::Type;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("cannot read GGUF `{path}`: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("not a GGUF file (bad magic)")]
    Magic,
    #[error("unsupported GGUF version {0}")]
    Version(u32),
    #[error("corrupt GGUF: {0}")]
    Corrupt(String),
    #[error("missing key `{0}`")]
    MissingKey(String),
    #[error("key `{0}` has the wrong type")]
    WrongType(String),
    #[error("missing tensor `{0}`")]
    MissingTensor(String),
    #[error("split GGUF: {0}")]
    Split(String),
}

type Result<T> = std::result::Result<T, GgufError>;

/// A metadata value. Arrays are skipped (only their length is kept): the tokenizer arrays are
/// large and this crate does not use them.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(u64),
}

/// One tensor of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub ty: Type,
    /// Elements per dimension (ggml order: `ne[0]` is the contiguous one).
    pub ne: [u64; 4],
    /// Index of the shard file that holds the data (0 for a single file).
    pub shard: usize,
    /// Absolute byte offset of the data in its shard file.
    pub offset: u64,
    pub nbytes: u64,
}

/// A byte range of a tensor inside one shard file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slab {
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.ne.iter().product()
    }

    /// Slab `i` along the last non-trivial dimension (the expert dimension of stacked expert
    /// tensors, `ne[2]`): its shard and byte range `[offset, offset + len)` in that shard.
    pub fn slab(&self, i: u64) -> Option<Slab> {
        let n = self.ne[2];
        (i < n).then(|| {
            let len = self.nbytes / n;
            Slab {
                shard: self.shard,
                offset: self.offset + i * len,
                len,
            }
        })
    }
}

/// Parsed GGUF header. A split model (llama.cpp's `gguf-split`) is opened from its first shard
/// and merged into one tensor index; a single file is one shard.
#[derive(Debug)]
pub struct Gguf {
    /// Metadata of the first shard.
    kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    /// Shard files in order.
    pub paths: Vec<PathBuf>,
    /// Data section start of each shard (0 for a headerless cache shard, see `convert`).
    pub data_offsets: Vec<u64>,
    /// Alignment and data section start of the first shard.
    pub alignment: u64,
    pub data_offset: u64,
    /// Byte range of the first shard's key-value section and its number of entries, for tools
    /// that rewrite a file.
    pub kv_section: (u64, u64),
    pub n_kv: u64,
}

/// One parsed file.
struct Shard {
    kv: HashMap<String, Value>,
    tensors: Vec<TensorInfo>,
    alignment: u64,
    data_offset: u64,
    kv_section: (u64, u64),
    n_kv: u64,
}

fn kv_int(kv: &HashMap<String, Value>, key: &str) -> Option<u64> {
    match kv.get(key) {
        Some(Value::U64(v)) => Some(*v),
        Some(Value::I64(v)) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

/// Path of shard `no` (0-based) of `count` from the path of shard 0, by llama.cpp's rule
/// (`<prefix>-00001-of-0000N.gguf`).
fn shard_path(first: &Path, count: u64, no: u64) -> Result<PathBuf> {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let suffix = format!("-{:05}-of-{count:05}.gguf", 1);
    let prefix = name.strip_suffix(&suffix).ok_or_else(|| {
        GgufError::Split(format!(
            "`{}` should be named `<prefix>{suffix}`",
            first.display()
        ))
    })?;
    Ok(first.with_file_name(format!("{prefix}-{:05}-of-{count:05}.gguf", no + 1)))
}

struct Rd<R> {
    r: R,
}

impl<R: Read> Rd<R> {
    fn io(e: std::io::Error) -> GgufError {
        GgufError::Corrupt(format!("read failed: {e}"))
    }
    fn n<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b[..N]).map_err(Self::io)?;
        Ok(std::array::from_fn(|i| b[i]))
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.n::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.n()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.n()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.n()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > 1 << 24 {
            return Err(GgufError::Corrupt(format!("string of {len} bytes")));
        }
        let mut v = vec![0u8; len as usize];
        self.r.read_exact(&mut v).map_err(Self::io)?;
        String::from_utf8(v).map_err(|_| GgufError::Corrupt("string is not UTF-8".into()))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink()).map_err(Self::io)?;
        Ok(())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U64(self.u8()? as u64),
            1 => Value::I64(self.u8()? as i8 as i64),
            2 => Value::U64(self.u16()? as u64),
            3 => Value::I64(self.u16()? as i16 as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let len = self.u64()?;
                match et {
                    0 | 1 | 7 => self.skip(len)?,
                    2 | 3 => self.skip(len * 2)?,
                    4..=6 => self.skip(len * 4)?,
                    10..=12 => self.skip(len * 8)?,
                    8 => {
                        for _ in 0..len {
                            let l = self.u64()?;
                            self.skip(l)?;
                        }
                    }
                    _ => return Err(GgufError::Corrupt(format!("array of type {et}"))),
                }
                Value::Array(len)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(GgufError::Corrupt(format!("value type {t}"))),
        })
    }
}

/// Splits a raw key-value section of `n` entries into `(key, start, end)` byte ranges.
pub(crate) fn kv_entries(section: &[u8], n: u64) -> Result<Vec<(String, usize, usize)>> {
    let mut rd = Rd {
        r: std::io::Cursor::new(section),
    };
    let mut out = Vec::new();
    for _ in 0..n {
        let start = rd.r.position() as usize;
        let key = rd.string()?;
        let ty = rd.u32()?;
        rd.value(ty)?;
        out.push((key, start, rd.r.position() as usize));
    }
    Ok(out)
}

impl Gguf {
    /// Opens a single file or, given the first shard of a split model, all of its shards.
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let path = path.as_ref();
        let first = parse_file(path, 0)?;
        let count = kv_int(&first.kv, "split.count").unwrap_or(1);
        let mut paths = vec![path.to_path_buf()];
        let mut data_offsets = vec![first.data_offset];
        let mut tensors = first.tensors;
        if count > 1 {
            if kv_int(&first.kv, "split.no") != Some(0) {
                return Err(GgufError::Split(format!(
                    "`{}` is not the first shard; open the -00001-of-{count:05} file",
                    path.display()
                )));
            }
            if count > 1 << 16 {
                return Err(GgufError::Split(format!("{count} shards")));
            }
            for no in 1..count {
                let p = shard_path(path, count, no)?;
                let shard = parse_file(&p, no as usize)?;
                if kv_int(&shard.kv, "split.no") != Some(no)
                    || kv_int(&shard.kv, "split.count") != Some(count)
                {
                    return Err(GgufError::Split(format!(
                        "`{}` is not shard {} of {count}",
                        p.display(),
                        no + 1
                    )));
                }
                tensors.extend(shard.tensors);
                data_offsets.push(shard.data_offset);
                paths.push(p);
            }
            if kv_int(&first.kv, "split.tensors.count") != Some(tensors.len() as u64) {
                return Err(GgufError::Split(format!(
                    "{} tensors in the shards, split.tensors.count says {:?}",
                    tensors.len(),
                    first.kv.get("split.tensors.count")
                )));
            }
        }
        let mut by_name = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if by_name.insert(t.name.clone(), i).is_some() {
                return Err(GgufError::Corrupt(format!("duplicate tensor `{}`", t.name)));
            }
        }
        Ok(Gguf {
            kv: first.kv,
            tensors,
            by_name,
            paths,
            data_offsets,
            alignment: first.alignment,
            data_offset: first.data_offset,
            kv_section: first.kv_section,
            n_kv: first.n_kv,
        })
    }

    /// Number of shard files (1 for a single file).
    pub fn n_shards(&self) -> usize {
        self.paths.len()
    }

    /// Points tensor `name` at `len` bytes of type `ty` at offset 0 of the headerless file `path`,
    /// which becomes an extra shard (a conversion cache file, ADR 0041).
    pub fn retarget(&mut self, name: &str, ty: Type, path: PathBuf) -> Result<()> {
        let i = *self
            .by_name
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(name.into()))?;
        let shard = match self.paths.iter().position(|p| *p == path) {
            Some(s) => s,
            None => {
                self.paths.push(path);
                self.data_offsets.push(0);
                self.paths.len() - 1
            }
        };
        let t = &mut self.tensors[i];
        t.ty = ty;
        t.shard = shard;
        t.offset = 0;
        t.nbytes = ty.row_size(t.ne[0] as i64) as u64 * t.ne[1] * t.ne[2] * t.ne[3];
        Ok(())
    }
}

/// Parses one GGUF file; its tensors get shard index `shard`.
fn parse_file(path: &Path, shard: usize) -> Result<Shard> {
    {
        let io = |source| GgufError::Io {
            path: path.display().to_string(),
            source,
        };
        let f = File::open(path).map_err(io)?;
        let mut rd = Rd {
            r: BufReader::new(f),
        };
        if rd.n::<4>()? != *b"GGUF" {
            return Err(GgufError::Magic);
        }
        let version = rd.u32()?;
        if version != 3 {
            return Err(GgufError::Version(version));
        }
        let (n_tensors, n_kv) = (rd.u64()?, rd.u64()?);
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err(GgufError::Corrupt("implausible counts".into()));
        }
        let mut kv = HashMap::new();
        let kv_start = rd.r.stream_position().map_err(io)?;
        for _ in 0..n_kv {
            let key = rd.string()?;
            let ty = rd.u32()?;
            kv.insert(key, rd.value(ty)?);
        }
        let kv_end = rd.r.stream_position().map_err(io)?;
        let alignment = match kv.get("general.alignment") {
            Some(Value::U64(a)) if *a > 0 => *a,
            _ => 32,
        };
        let mut raw = Vec::new();
        for _ in 0..n_tensors {
            let name = rd.string()?;
            let n_dims = rd.u32()?;
            if !(1..=4).contains(&n_dims) {
                return Err(GgufError::Corrupt(format!(
                    "tensor `{name}` has {n_dims} dims"
                )));
            }
            let mut ne = [1u64; 4];
            for d in ne.iter_mut().take(n_dims as usize) {
                *d = rd.u64()?;
            }
            let ty = rd.u32()?;
            let off = rd.u64()?;
            raw.push((name, ne, ty, off));
        }
        let mut r = rd.r;
        let pos = r.stream_position().map_err(io)?;
        let data_offset = pos.div_ceil(alignment) * alignment;
        r.seek(SeekFrom::Start(data_offset)).ok();
        let mut tensors = Vec::new();
        for (name, ne, ty, off) in raw {
            let ty = Type::from_id(ty)
                .ok_or_else(|| GgufError::Corrupt(format!("tensor `{name}` has type id {ty}")))?;
            if ne[0] % ty.block_size() as u64 != 0 {
                return Err(GgufError::Corrupt(format!(
                    "tensor `{name}` row is not block aligned"
                )));
            }
            let nbytes = ty.row_size(ne[0] as i64) as u64 * ne[1] * ne[2] * ne[3];
            tensors.push(TensorInfo {
                name,
                ty,
                ne,
                shard,
                offset: data_offset + off,
                nbytes,
            });
        }
        Ok(Shard {
            kv,
            tensors,
            alignment,
            data_offset,
            kv_section: (kv_start, kv_end),
            n_kv,
        })
    }
}

impl Gguf {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        match self.kv.get(key) {
            Some(Value::Str(s)) => Ok(s),
            Some(_) => Err(GgufError::WrongType(key.into())),
            None => Err(GgufError::MissingKey(key.into())),
        }
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        match self.kv.get(key) {
            Some(Value::U64(v)) => Ok(*v),
            Some(Value::I64(v)) if *v >= 0 => Ok(*v as u64),
            Some(_) => Err(GgufError::WrongType(key.into())),
            None => Err(GgufError::MissingKey(key.into())),
        }
    }

    pub fn u64_or(&self, key: &str, default: u64) -> Result<u64> {
        match self.kv.contains_key(key) {
            true => self.u64(key),
            false => Ok(default),
        }
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        match self.kv.get(key) {
            Some(Value::F64(v)) => Ok(*v as f32),
            Some(_) => Err(GgufError::WrongType(key.into())),
            None => Err(GgufError::MissingKey(key.into())),
        }
    }

    pub fn f32_or(&self, key: &str, default: f32) -> Result<f32> {
        match self.kv.contains_key(key) {
            true => self.f32(key),
            false => Ok(default),
        }
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| GgufError::MissingTensor(name.into()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }
}

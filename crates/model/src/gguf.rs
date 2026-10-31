//! GGUF header reader: key-value metadata and the tensor index. Tensor data is never read here.
//! Format: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> (version 3, little endian).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
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
    /// Absolute byte offset of the data in the file.
    pub offset: u64,
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.ne.iter().product()
    }

    /// Byte range `[offset, offset + len)` of slab `i` along the last non-trivial dimension
    /// (the expert dimension of stacked expert tensors, `ne[2]`).
    pub fn slab(&self, i: u64) -> Option<(u64, u64)> {
        let n = self.ne[2];
        (i < n).then(|| {
            let len = self.nbytes / n;
            (self.offset + i * len, len)
        })
    }
}

/// Parsed GGUF header.
#[derive(Debug)]
pub struct Gguf {
    kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    pub alignment: u64,
    pub data_offset: u64,
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

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let path = path.as_ref();
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
        for _ in 0..n_kv {
            let key = rd.string()?;
            let ty = rd.u32()?;
            kv.insert(key, rd.value(ty)?);
        }
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
        let mut by_name = HashMap::new();
        for (name, ne, ty, off) in raw {
            let ty = Type::from_id(ty)
                .ok_or_else(|| GgufError::Corrupt(format!("tensor `{name}` has type id {ty}")))?;
            if ne[0] % ty.block_size() as u64 != 0 {
                return Err(GgufError::Corrupt(format!(
                    "tensor `{name}` row is not block aligned"
                )));
            }
            let nbytes = ty.row_size(ne[0] as i64) as u64 * ne[1] * ne[2] * ne[3];
            by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                ty,
                ne,
                offset: data_offset + off,
                nbytes,
            });
        }
        Ok(Gguf {
            kv,
            tensors,
            by_name,
            alignment,
            data_offset,
        })
    }

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

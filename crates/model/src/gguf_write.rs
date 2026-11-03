//! GGUF writer: a model view (a [`Gguf`] index, converted cache shards included) as one file or
//! as shards the way llama.cpp's `gguf-split` writes them (shard names, the three `split.*`
//! keys, one data section per shard). Tensor data is streamed, never held whole.

use crate::gguf::{Gguf, GgufError, TensorInfo, kv_entries};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("groups: {0}")]
    Groups(String),
    #[error("cannot write `{path}`: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

type Result<T> = std::result::Result<T, WriteError>;

fn io(path: &Path) -> impl Fn(std::io::Error) -> WriteError + '_ {
    move |source| WriteError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// `<prefix>-00001-of-0000N.gguf` and so on (llama.cpp's `llama_split_path`); `no` is 0-based.
pub fn shard_name(prefix: &Path, no: usize, count: usize) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(format!("-{:05}-of-{count:05}.gguf", no + 1));
    PathBuf::from(s)
}

/// Writes the model as one GGUF file at `out`. Returns the written paths.
pub fn write_gguf(g: &Gguf, out: &Path) -> Result<Vec<PathBuf>> {
    let all: Vec<&TensorInfo> = g.tensors.iter().collect();
    write_shards(g, vec![(out.to_path_buf(), all)], false)
}

/// Writes the model as shards `<prefix>-0000k-of-0000N.gguf`. Shard 1 holds all key-value
/// metadata and every tensor that is in no group; shard `i + 1` holds the tensors of
/// `groups[i]` (in the model's tensor order). A group may be empty. Returns the shard paths.
pub fn write_split(g: &Gguf, prefix: &Path, groups: &[Vec<String>]) -> Result<Vec<PathBuf>> {
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    for (i, names) in groups.iter().enumerate() {
        for n in names {
            if !g.has_tensor(n) {
                return Err(WriteError::Groups(format!(
                    "tensor `{n}` is not in the model"
                )));
            }
            if group_of.insert(n.as_str(), i + 1).is_some() {
                return Err(WriteError::Groups(format!("tensor `{n}` is in two groups")));
            }
        }
    }
    let count = groups.len() + 1;
    let mut shards: Vec<Vec<&TensorInfo>> = vec![Vec::new(); count];
    for t in &g.tensors {
        shards[group_of.get(t.name.as_str()).copied().unwrap_or(0)].push(t);
    }
    let shards = shards
        .into_iter()
        .enumerate()
        .map(|(no, ts)| (shard_name(prefix, no, count), ts))
        .collect();
    write_shards(g, shards, true)
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn pad(n: u64, align: u64) -> u64 {
    n.next_multiple_of(align)
}

/// The first shard's key-value entries without any `split.*` key, as raw bytes.
fn base_kv(g: &Gguf) -> Result<(Vec<u8>, u64)> {
    let (start, end) = g.kv_section;
    let path = &g.paths[0];
    let mut f = File::open(path).map_err(io(path))?;
    f.seek(SeekFrom::Start(start)).map_err(io(path))?;
    let mut raw = vec![0u8; (end - start) as usize];
    f.read_exact(&mut raw).map_err(io(path))?;
    let mut out = Vec::with_capacity(raw.len());
    let mut n = 0;
    for (key, s, e) in kv_entries(&raw, g.n_kv)? {
        if !key.starts_with("split.") {
            out.extend_from_slice(&raw[s..e]);
            n += 1;
        }
    }
    Ok((out, n))
}

fn write_shards(
    g: &Gguf,
    shards: Vec<(PathBuf, Vec<&TensorInfo>)>,
    split: bool,
) -> Result<Vec<PathBuf>> {
    let (kv, n_kv) = base_kv(g)?;
    let align = g.alignment;
    let count = shards.len();
    let mut sources: HashMap<usize, File> = HashMap::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut paths = Vec::new();
    for (no, (path, tensors)) in shards.iter().enumerate() {
        let mut head = b"GGUF".to_vec();
        head.extend(3u32.to_le_bytes());
        head.extend((tensors.len() as u64).to_le_bytes());
        let n_split = if split { 3 } else { 0 };
        let keep = if no == 0 { n_kv } else { 0 };
        head.extend((keep + n_split).to_le_bytes());
        if no == 0 {
            head.extend_from_slice(&kv);
        }
        if split {
            put_str(&mut head, "split.no");
            head.extend(2u32.to_le_bytes());
            head.extend((no as u16).to_le_bytes());
            put_str(&mut head, "split.count");
            head.extend(2u32.to_le_bytes());
            head.extend((count as u16).to_le_bytes());
            put_str(&mut head, "split.tensors.count");
            head.extend(5u32.to_le_bytes());
            head.extend((g.tensors.len() as i32).to_le_bytes());
        }
        let mut offset = 0u64;
        for t in tensors {
            put_str(&mut head, &t.name);
            let n_dims = t.ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
            head.extend((n_dims as u32).to_le_bytes());
            for d in &t.ne[..n_dims] {
                head.extend(d.to_le_bytes());
            }
            head.extend(t.ty.id().to_le_bytes());
            head.extend(offset.to_le_bytes());
            offset += pad(t.nbytes, align);
        }
        head.resize(pad(head.len() as u64, align) as usize, 0);

        let mut out = BufWriter::new(File::create(path).map_err(io(path))?);
        out.write_all(&head).map_err(io(path))?;
        for t in tensors {
            if let std::collections::hash_map::Entry::Vacant(e) = sources.entry(t.shard) {
                let p = &g.paths[t.shard];
                e.insert(File::open(p).map_err(io(p))?);
            }
            let src = sources.get_mut(&t.shard).expect("opened above");
            let sp = &g.paths[t.shard];
            src.seek(SeekFrom::Start(t.offset)).map_err(io(sp))?;
            let mut left = t.nbytes as usize;
            while left > 0 {
                let n = left.min(buf.len());
                src.read_exact(&mut buf[..n]).map_err(io(sp))?;
                out.write_all(&buf[..n]).map_err(io(path))?;
                left -= n;
            }
            let zeros = vec![0u8; (pad(t.nbytes, align) - t.nbytes) as usize];
            out.write_all(&zeros).map_err(io(path))?;
        }
        out.flush().map_err(io(path))?;
        paths.push(path.clone());
    }
    Ok(paths)
}

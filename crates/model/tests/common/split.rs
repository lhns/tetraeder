//! A pure-Rust GGUF splitter that writes shards the way llama.cpp's `gguf-split` does (shard
//! names, the three `split.*` keys, per-shard data sections), so tests need no llama.cpp tool.

use std::path::{Path, PathBuf};
use tetraeder_model::Gguf;

const ALIGN: usize = 32;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn pad_to(out: &mut Vec<u8>, align: usize) {
    out.resize(out.len().next_multiple_of(align), 0);
}

/// `<prefix>-00001-of-0000N.gguf` and so on, llama.cpp's `llama_split_path`.
pub fn shard_name(prefix: &Path, no: usize, count: usize) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(format!("-{:05}-of-{count:05}.gguf", no + 1));
    PathBuf::from(s)
}

/// The tensor indices where a split into `n` shards cuts: each cut lies before the
/// second expert tensor of some layer, so that layer's first expert tensor is in an earlier shard
/// than its other two.
pub fn expert_cuts(g: &Gguf, n: usize) -> Vec<usize> {
    let exps: Vec<usize> = g
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| t.name.contains("_exps."))
        .map(|(i, _)| i)
        .collect();
    // Three expert tensors per layer; the cut is before the second one.
    let layers = exps.len() / 3;
    (1..n).map(|k| exps[3 * (k * layers / n) + 1]).collect()
}

/// Splits `src` into `n` shards at [`expert_cuts`]. Returns the shard paths.
pub fn split_gguf(src: &Path, prefix: &Path, n: usize) -> Vec<PathBuf> {
    let g = Gguf::open(src).unwrap();
    let cuts = expert_cuts(&g, n);
    split_gguf_at(src, prefix, &cuts)
}

/// Splits `src` into `cuts.len() + 1` shards of contiguous tensors; shard `i` ends before
/// tensor `cuts[i]`. A cut at 0 leaves the first shard without tensors (`--no-tensor-first-split`).
pub fn split_gguf_at(src: &Path, prefix: &Path, cuts: &[usize]) -> Vec<PathBuf> {
    let g = Gguf::open(src).unwrap();
    assert_eq!(g.n_shards(), 1, "the source is a single file");
    assert_eq!(g.alignment, ALIGN as u64);
    let bytes = std::fs::read(src).unwrap();
    let count = cuts.len() + 1;
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(g.tensors.len());
    let mut paths = Vec::new();
    for no in 0..count {
        let group = &g.tensors[bounds[no]..bounds[no + 1]];
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((group.len() as u64).to_le_bytes());
        let first = no == 0;
        let n_kv = if first { g.n_kv + 3 } else { 3 };
        out.extend(n_kv.to_le_bytes());
        if first {
            out.extend_from_slice(&bytes[g.kv_section.0 as usize..g.kv_section.1 as usize]);
        }
        put_str(&mut out, "split.no");
        out.extend(2u32.to_le_bytes());
        out.extend((no as u16).to_le_bytes());
        put_str(&mut out, "split.count");
        out.extend(2u32.to_le_bytes());
        out.extend((count as u16).to_le_bytes());
        put_str(&mut out, "split.tensors.count");
        out.extend(5u32.to_le_bytes());
        out.extend((g.tensors.len() as i32).to_le_bytes());

        let mut data = Vec::new();
        for t in group {
            put_str(&mut out, &t.name);
            let n_dims = t.ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
            out.extend((n_dims as u32).to_le_bytes());
            for d in &t.ne[..n_dims] {
                out.extend(d.to_le_bytes());
            }
            out.extend(t.ty.id().to_le_bytes());
            out.extend((data.len() as u64).to_le_bytes());
            data.extend_from_slice(&bytes[t.offset as usize..(t.offset + t.nbytes) as usize]);
            pad_to(&mut data, ALIGN);
        }
        pad_to(&mut out, ALIGN);
        out.extend(data);
        let path = shard_name(prefix, no, count);
        std::fs::write(&path, out).unwrap();
        paths.push(path);
    }
    paths
}

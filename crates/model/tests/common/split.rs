//! Test helpers over the library's GGUF writer (`tetraeder_model::gguf_write`), which writes
//! shards the way llama.cpp's `gguf-split` does.

use std::path::{Path, PathBuf};
use tetraeder_model::Gguf;
use tetraeder_model::gguf_write::write_split;

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
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(g.tensors.len());
    // Shard 1 takes every tensor that is in no group, so the groups start at the first cut.
    let groups: Vec<Vec<String>> = bounds[1..]
        .windows(2)
        .map(|w| {
            g.tensors[w[0]..w[1]]
                .iter()
                .map(|t| t.name.clone())
                .collect()
        })
        .collect();
    write_split(&g, prefix, &groups).unwrap()
}

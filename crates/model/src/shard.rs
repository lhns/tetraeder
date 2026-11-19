//! Helpers the sharded builders share (ADR 0069, 0070): slice arithmetic, reading a slice of a
//! GGUF tensor straight into a rank's weights, and the KV snapshot of a tensor held in slices.
//! The dense builder (`qwen3`) uses them first; the MoE builder reuses them.

use crate::gguf::Gguf;
use crate::qwen3moe::{ModelError, Result};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::rc::Rc;
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedTensor, Tensor};
use tetraeder_plan::block_slice;

/// Range of `n` equal parts of `len` for part `i` (heads and KV heads: `tp` divides them).
pub(crate) fn part(len: u32, n: u32, i: u32) -> Range<u32> {
    let w = len / n.max(1);
    i * w..(i + 1) * w
}

/// Slice `i` of `n` of `len` elements with a larger share rotating over layers, in whole blocks
/// of `block` elements (`plan::block_slice`, ADR 0070).
pub(crate) fn block_part(len: u64, block: u64, n: u32, i: u32, rot: u32) -> Result<Range<u64>> {
    block_slice(len, block, n, i, rot).map_err(ModelError::Invalid)
}

/// Slice `i` of `n` of `len` rows, whole rows, the first `len % n` slices one row larger. Rows
/// (`ne1`) need no quantization block, so the vocabulary can be cut like this.
pub(crate) fn row_part(len: u64, n: u32, i: u32) -> Range<u64> {
    let (n, i) = (u64::from(n.max(1)), u64::from(i));
    let (base, extra) = (len / n, len % n);
    let start = i * base + i.min(extra);
    start..start + base + u64::from(i < extra)
}

struct Pending {
    t: OwnedTensor,
    shard: usize,
    offset: u64,
    ty: tetraeder_ggml_sys::ggml::Type,
    /// Elements of the file tensor along `ne[0]`, `ne[1]`, and the number of planes (`ne[2]`
    /// times `ne[3]`).
    ne: (u64, u64, u64),
    ne0: Range<u64>,
    ne1: Range<u64>,
}

/// Creates the tensors of one rank's weights in one context, then reads their slices from the
/// GGUF in one pass ([`WeightLoader::finish`]).
pub(crate) struct WeightLoader<'g> {
    gguf: &'g Gguf,
    ctx: Rc<Context>,
    pending: Vec<Pending>,
}

impl<'g> WeightLoader<'g> {
    pub fn new(gguf: &'g Gguf, max_tensors: usize) -> Result<Self> {
        Ok(WeightLoader {
            gguf,
            ctx: Rc::new(Context::new(max_tensors + 4, 0)?),
            pending: Vec::new(),
        })
    }

    /// The tensor `name` cut to `ne0` elements along its contiguous dimension (a row slice, in
    /// whole quantization blocks) and to `ne1` rows (a column slice, whole rows); `None` keeps
    /// the dimension whole. Higher dimensions (stacked experts) are kept whole.
    pub fn tensor(
        &mut self,
        name: &str,
        ne0: Option<Range<u64>>,
        ne1: Option<Range<u64>>,
    ) -> Result<OwnedTensor> {
        let info = self.gguf.tensor(name)?;
        let ne0 = ne0.unwrap_or(0..info.ne[0]);
        let ne1 = ne1.unwrap_or(0..info.ne[1]);
        let bs = info.ty.block_size() as u64;
        if ne0.start > ne0.end
            || ne0.end > info.ne[0]
            || ne1.start > ne1.end
            || ne1.end > info.ne[1]
        {
            return Err(ModelError::Invalid(format!(
                "`{name}`: slice {ne0:?} x {ne1:?} outside {} x {}",
                info.ne[0], info.ne[1]
            )));
        }
        if !ne0.start.is_multiple_of(bs) || !(ne0.end - ne0.start).is_multiple_of(bs) {
            return Err(ModelError::Invalid(format!(
                "`{name}`: slice {ne0:?} does not cut whole blocks of {bs}"
            )));
        }
        let ne = [
            (ne0.end - ne0.start) as i64,
            (ne1.end - ne1.start) as i64,
            info.ne[2] as i64,
            info.ne[3] as i64,
        ];
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let t = self.ctx.new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        self.pending.push(Pending {
            t: t.clone(),
            shard: info.shard,
            offset: info.offset,
            ty: info.ty,
            ne: (info.ne[0], info.ne[1], info.ne[2] * info.ne[3]),
            ne0,
            ne1,
        });
        Ok(t)
    }

    /// Allocates the weights on `backend` and reads them. Returns the buffer size.
    pub fn finish(self, backend: &Backend) -> Result<usize> {
        let size = self.ctx.alloc_weights(backend)?;
        let mut shards = self
            .gguf
            .paths
            .iter()
            .map(std::fs::File::open)
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut buf = vec![0u8; 16 << 20];
        let mut cut = Vec::new();
        for p in &self.pending {
            let src_row = p.ty.row_size(p.ne.0 as i64);
            let dst_row = p.ty.row_size((p.ne0.end - p.ne0.start) as i64);
            let skip = p.ty.row_size(p.ne0.start as i64);
            let whole_rows = p.ne0 == (0..p.ne.0);
            let rows_per_read = ((buf.len() / src_row.max(1)) as u64).max(1);
            let f = &mut shards[p.shard];
            let mut dst = 0usize;
            for plane in 0..p.ne.2 {
                let plane_off = p.offset + plane * p.ne.1 * src_row as u64;
                let mut r = p.ne1.start;
                while r < p.ne1.end {
                    let n = rows_per_read.min(p.ne1.end - r);
                    let len = n as usize * src_row;
                    if len > buf.len() {
                        buf.resize(len, 0);
                    }
                    f.seek(SeekFrom::Start(plane_off + r * src_row as u64))?;
                    f.read_exact(&mut buf[..len])?;
                    if whole_rows {
                        p.t.get().set_bytes(&buf[..len], dst)?;
                        dst += len;
                    } else {
                        cut.clear();
                        for k in 0..n as usize {
                            let row = &buf[k * src_row..(k + 1) * src_row];
                            cut.extend_from_slice(&row[skip..skip + dst_row]);
                        }
                        p.t.get().set_bytes(&cut, dst)?;
                        dst += cut.len();
                    }
                    r += n;
                }
            }
        }
        Ok(size)
    }
}

/// Appends the rows `runs` of a tensor held in `slices` (the KV heads of a layer cut across
/// ranks, in head order) to `out`, each position's row being the slices' rows one after the
/// other: the layout of the one-rank cache (ADR 0069, "KV snapshots and copies").
pub(crate) fn gather_rows(
    slices: &[Tensor<'_>],
    n_rows: usize,
    runs: &[(usize, usize)],
    out: &mut Vec<u8>,
) -> Result<()> {
    let rows: Vec<usize> = slices.iter().map(|t| t.nbytes() / n_rows).collect();
    for &(start, len) in runs {
        if let [t] = slices {
            t.get_bytes_into(start * rows[0], len * rows[0], out)?;
            continue;
        }
        let parts = slices
            .iter()
            .zip(&rows)
            .map(|(t, &row)| t.get_bytes(start * row, len * row))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for p in 0..len {
            for (part, &row) in parts.iter().zip(&rows) {
                out.extend_from_slice(&part[p * row..(p + 1) * row]);
            }
        }
    }
    Ok(())
}

/// The inverse of [`gather_rows`]: writes the rows `runs` from the front of `rest`, which
/// advances past them.
pub(crate) fn scatter_rows(
    slices: &[Tensor<'_>],
    n_rows: usize,
    runs: &[(usize, usize)],
    rest: &mut &[u8],
) -> Result<()> {
    let rows: Vec<usize> = slices.iter().map(|t| t.nbytes() / n_rows).collect();
    let whole: usize = rows.iter().sum();
    for &(start, len) in runs {
        let (head, tail) = rest.split_at(len * whole);
        *rest = tail;
        if let [t] = slices {
            t.set_bytes(head, start * rows[0])?;
            continue;
        }
        let mut bufs: Vec<Vec<u8>> = rows.iter().map(|&r| Vec::with_capacity(len * r)).collect();
        for p in 0..len {
            let mut at = p * whole;
            for (buf, &row) in bufs.iter_mut().zip(&rows) {
                buf.extend_from_slice(&head[at..at + row]);
                at += row;
            }
        }
        for ((t, buf), &row) in slices.iter().zip(&bufs).zip(&rows) {
            t.set_bytes(buf, start * row)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_part_deals_whole_rows() {
        let cuts: Vec<_> = (0..3).map(|i| row_part(10, 3, i)).collect();
        assert_eq!(cuts, [0..4, 4..7, 7..10]);
        assert_eq!(row_part(8, 1, 0), 0..8);
        assert_eq!(part(64, 4, 1), 16..32);
    }
}

/// Writes a step input of a rank: on its stream when `on_stream`
/// (`Executor::async_inputs`, ADR 0069 History 2026-11-19), else with one blocking copy.
pub(crate) fn put_i32(
    t: Tensor<'_>,
    backend: &Backend,
    on_stream: bool,
    data: &[i32],
) -> Result<()> {
    if on_stream {
        t.set_i32_async(backend, data)?;
    } else {
        t.set_i32(data)?;
    }
    Ok(())
}

/// [`put_i32`] for F16 bits (the mask).
pub(crate) fn put_f16(
    t: Tensor<'_>,
    backend: &Backend,
    on_stream: bool,
    data: &[u16],
) -> Result<()> {
    if on_stream {
        t.set_f16_bits_async(backend, data)?;
    } else {
        t.set_f16_bits(data)?;
    }
    Ok(())
}

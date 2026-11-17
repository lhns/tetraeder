//! How a KV snapshot (a prefix-cache save) reads rows back to the host (ADR 0074): through a
//! reused pair of pinned staging halves, the DMA into one half overlapping the host copy out of
//! the other, or straight into the snapshot when the rows are in host memory.

use crate::qwen3moe::Result;
use std::cell::Cell;
use std::rc::Rc;
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedTensor, Tensor, Type};

/// Bytes of one staging half. A save of 78 MB (Qwen3-8B, 533 tokens) takes about ten syncs;
/// both halves are 16 MiB of pinned memory per model whatever the context length.
pub(crate) const STAGING_HALF: usize = 8 << 20;

/// `len` bytes at `offset` of a tensor.
pub(crate) type Span<'a> = (Tensor<'a>, usize, usize);

/// The spans of rows `runs` of each tensor of `tensors` (`n_rows` rows each), tensor by tensor:
/// the snapshot order of ADR 0049.
pub(crate) fn row_spans<'a>(
    tensors: impl IntoIterator<Item = Tensor<'a>>,
    n_rows: usize,
    runs: &[(usize, usize)],
) -> Vec<Span<'a>> {
    let mut out = Vec::new();
    for t in tensors {
        let row = t.nbytes() / n_rows;
        out.extend(runs.iter().map(|&(start, len)| (t, start * row, len * row)));
    }
    out
}

/// Appends every span to `out` with one blocking copy each.
pub(crate) fn append_direct(spans: &[Span<'_>], out: &mut Vec<u8>) -> Result<()> {
    for &(t, offset, len) in spans {
        t.get_bytes_into(offset, len, out)?;
    }
    Ok(())
}

pub(crate) struct Readback {
    backend: Backend,
    staging: OwnedTensor,
    half: usize,
    pinned: bool,
    /// Device-to-host copies enqueued since load.
    gets: Cell<u64>,
}

impl Readback {
    /// Two halves of `half` bytes in `backend`'s device host buffer type (pinned on CUDA, plain
    /// memory on the CPU backend). `backend` holds the KV tensors.
    pub(crate) fn new(backend: &Backend, half: usize) -> Result<Self> {
        let half = half.max(1);
        let ctx = Rc::new(Context::new(1, 0)?);
        let staging = ctx.new_persistent(Type::I8, &[(2 * half) as i64]);
        staging.get().set_name("kv_readback");
        let (bytes, pinned) = ctx.alloc_host_staging(backend)?;
        tracing::info!(bytes, pinned, "kv readback staging");
        Ok(Self {
            backend: backend.clone(),
            staging,
            half,
            pinned,
            gets: Cell::new(0),
        })
    }

    /// Bytes of both halves.
    pub(crate) fn bytes(&self) -> usize {
        2 * self.half
    }

    pub(crate) fn pinned(&self) -> bool {
        self.pinned
    }

    #[cfg(test)]
    fn gets(&self) -> u64 {
        self.gets.get()
    }

    /// Appends every span to `out`, in order. Spans the backend cannot read asynchronously (a
    /// KV tensor in host memory) make the whole read direct, which keeps the order.
    pub(crate) fn read(&self, spans: &[Span<'_>], out: &mut Vec<u8>) -> Result<()> {
        out.reserve(spans.iter().map(|s| s.2).sum());
        if !spans.iter().all(|&(t, _, _)| self.backend.can_read(t)) {
            return append_direct(spans, out);
        }
        // The rows were written by earlier steps on this backend's stream.
        self.backend.synchronize();
        let r = self.pipelined(spans, out);
        // Nothing stays in flight into the staging halves, also after an error.
        self.backend.synchronize();
        r
    }

    /// Batch k is enqueued into half k % 2; while it copies, batch k - 1 (landed at the
    /// previous sync) is appended from the other half. A half is written again only after it
    /// was appended.
    fn pipelined(&self, spans: &[Span<'_>], out: &mut Vec<u8>) -> Result<()> {
        let st = self.staging.get();
        let half = self.half;
        let mut pieces = spans
            .iter()
            .flat_map(|&(t, offset, len)| {
                (0..len.div_ceil(half)).map(move |i| {
                    let at = i * half;
                    (t, offset + at, half.min(len - at))
                })
            })
            .peekable();
        let mut cur = 0;
        let mut landed: Option<(usize, usize)> = None;
        loop {
            let base = cur * half;
            let mut fill = 0;
            while let Some(&(t, offset, len)) = pieces.peek() {
                if fill + len > half {
                    break;
                }
                t.read_to_async(offset, &self.backend, st, base + fill, len)?;
                self.gets.set(self.gets.get() + 1);
                fill += len;
                pieces.next();
            }
            if let Some((h, n)) = landed.take() {
                st.get_bytes_into(h * half, n, out)?;
            }
            if fill == 0 {
                return Ok(());
            }
            self.backend.synchronize();
            landed = Some((cur, fill));
            cur ^= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tensor of `n_rows` rows per entry of `rows` (the row's bytes), filled with a pattern.
    fn tensors(
        backend: &Backend,
        n_rows: usize,
        rows: &[usize],
    ) -> (Rc<Context>, Vec<OwnedTensor>) {
        let ctx = Rc::new(Context::new(rows.len(), 0).unwrap());
        let ts: Vec<_> = rows
            .iter()
            .map(|&row| ctx.new_persistent(Type::I8, &[(n_rows * row) as i64]))
            .collect();
        ctx.alloc_tensors(backend).unwrap();
        for (k, t) in ts.iter().enumerate() {
            let n = t.get().nbytes();
            let bytes: Vec<u8> = (0..n)
                .map(|i| {
                    (i as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(k as u32 * 97) as u8
                })
                .collect();
            t.get().set_bytes(&bytes, 0).unwrap();
        }
        (ctx, ts)
    }

    fn reference(spans: &[Span<'_>]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(t, offset, len) in spans {
            out.extend_from_slice(&t.get_bytes(offset, len).unwrap());
        }
        out
    }

    /// hardware: cpu. The staged read returns the bytes of the plain reads, in order, for
    /// halves smaller than a row, than a span and larger than everything, with several runs.
    #[test]
    fn staged_read_equals_plain_reads() {
        let cpu = Backend::cpu(1).unwrap();
        let (_ctx, ts) = tensors(&cpu, 64, &[48, 80, 48]);
        let runs = [(3, 10), (20, 1), (40, 17)];
        let spans = row_spans(ts.iter().map(|t| t.get()), 64, &runs);
        let want = reference(&spans);
        for half in [1, 7, 48, 100, 480, 1 << 20] {
            let rb = Readback::new(&cpu, half).unwrap();
            assert!(
                !rb.pinned(),
                "the CPU device has no pinned host buffer type"
            );
            let mut out = vec![1, 2, 3];
            rb.read(&spans, &mut out).unwrap();
            assert_eq!(out[..3], [1, 2, 3], "half {half}");
            assert_eq!(out[3..], want[..], "half {half}");
        }
        let mut direct = Vec::new();
        append_direct(&spans, &mut direct).unwrap();
        assert_eq!(direct, want);
    }

    /// hardware: cpu. One contiguous run per sequence (the chunked layout, ADR 0067) and halves
    /// that hold a span: one device-to-host copy per tensor, O(layers) and not O(rows).
    #[test]
    fn one_copy_per_tensor_for_one_run() {
        let cpu = Backend::cpu(1).unwrap();
        let rows = [64usize; 8];
        let (_ctx, ts) = tensors(&cpu, 256, &rows);
        let spans = row_spans(ts.iter().map(|t| t.get()), 256, &[(16, 200)]);
        let rb = Readback::new(&cpu, 3 * 200 * 64).unwrap();
        let mut out = Vec::new();
        rb.read(&spans, &mut out).unwrap();
        assert_eq!(rb.gets(), ts.len() as u64);
        assert_eq!(out, reference(&spans));
        // An empty sequence reads nothing.
        rb.read(&row_spans(ts.iter().map(|t| t.get()), 256, &[]), &mut out)
            .unwrap();
        assert_eq!(rb.gets(), ts.len() as u64);
    }

    /// hardware: cpu. A dirty recycled buffer from the snapshot pool, written by the staged and
    /// by the direct read, holds exactly the bytes of the plain reads.
    #[test]
    fn pooled_buffers_hold_the_same_bytes() {
        use crate::snapshot_pool::{PoolConfig, SnapshotPool};
        let cpu = Backend::cpu(1).unwrap();
        let (_ctx, ts) = tensors(&cpu, 64, &[48, 80, 48]);
        let spans = row_spans(
            ts.iter().map(|t| t.get()),
            64,
            &[(3, 10), (20, 1), (40, 17)],
        );
        let want = reference(&spans);
        let pool = SnapshotPool::new(PoolConfig {
            spares: 2,
            thp: false,
            min_bytes: 1,
            round: 64,
        });
        let rb = Readback::new(&cpu, 100).unwrap();
        for round in 0..3 {
            let mut dirty = Vec::with_capacity(want.len() + 64);
            dirty.resize(dirty.capacity(), 0xAB);
            pool.recycle(dirty);
            let mut out = pool.take(want.len());
            assert!(out.is_empty(), "round {round}");
            if round % 2 == 0 {
                rb.read(&spans, &mut out).unwrap();
            } else {
                append_direct(&spans, &mut out).unwrap();
            }
            assert_eq!(out, want, "round {round}");
            pool.recycle(out);
        }
        assert!(pool.stats().from_spare >= 3);
    }
}

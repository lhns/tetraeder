//! KV rows of the attention pool: layouts that give each sequence its rows, and the pool that
//! plans a step (positions, `set_rows` destinations, `n_kv`, F16 mask) over them (ADR 0065, 0067).
//!
//! Pure Rust, no ggml: `model` and `scheduler` depend on this crate, never the reverse. Chunked
//! layout is what the builders start with; paged exists for the simulator and the pool tests.

mod chunked;
mod paged;
mod pool;

pub use chunked::{Chunked, GROW};
pub use paged::Paged;
pub use pool::{BlockCounts, KvPool, SeqAttn, StepInput, StepPlan};

pub type SeqId = u32;
/// Rows are allocated, and `n_kv` is padded, in units of this many rows.
pub const ROW_ALIGN: usize = 256;
/// F16 bits of minus infinity: the mask value of a row a token must not read.
pub const F16_NEG_INF: u16 = 0xFC00;

/// `n` rows starting at `src` are copied to `dst`. The two ranges of one copy never overlap; a
/// list of copies runs in order, and a later copy may write rows an earlier one read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCopy {
    pub src: usize,
    pub dst: usize,
    pub n: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvError {
    #[error("KV pool full: {needed} rows needed, {free} free")]
    Full { needed: usize, free: usize },
    #[error("unknown sequence {0}")]
    UnknownSeq(SeqId),
    #[error("invalid KV request: {0}")]
    Invalid(String),
}

/// A row allocator. Which rows a sequence owns never changes the attention op, only the rows
/// written, the mask and `n_kv`.
pub trait Layout: Send {
    fn name(&self) -> &'static str;
    fn capacity(&self) -> usize;
    /// Rows not reserved by any sequence.
    fn free(&self) -> usize;
    fn reserved(&self, seq: SeqId) -> usize;
    /// Rows for positions 0..n of `seq`; returns moves to apply, in order, before the next step.
    /// A move copies the old rows of the sequence, in position order from position 0, to its
    /// new rows (the pool trims it to the positions in use). `Full` changes nothing.
    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError>;
    /// Releases what positions 0..keep do not need, in ROW_ALIGN units; 0 frees the sequence.
    fn shrink(&mut self, seq: SeqId, keep: usize);
    /// Row of position `pos`; `seq` must hold a reservation covering it.
    fn row(&self, seq: SeqId, pos: usize) -> usize;
    /// First row of `seq` when its whole reservation is one run (position p at row start + p).
    fn run_start(&self, _seq: SeqId) -> Option<usize> {
        None
    }
    /// Highest row among positions 0..n of `seq`, if n > 0.
    fn max_row(&self, seq: SeqId, n: usize) -> Option<usize> {
        (0..n).map(|p| self.row(seq, p)).max()
    }
}

pub(crate) fn round_up(n: usize, unit: usize) -> usize {
    n.div_ceil(unit) * unit
}

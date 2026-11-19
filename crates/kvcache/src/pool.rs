//! The pool: a layout plus the length of every sequence, and the per-step plan.

use crate::{F16_NEG_INF, KvError, Layout, ROW_ALIGN, RowCopy, SeqId, round_up};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInput {
    pub seq: SeqId,
    pub n_tokens: usize,
    pub all_logits: bool,
}

/// Tokens in input order, each input's tokens in position order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPlan {
    pub pos: Vec<i32>,
    pub rows: Vec<i32>,
    pub n_kv: usize,
    /// F16 bits, token-major, 0 where the token may read the row. Without `seqs`: `[n_kv,
    /// n_tokens]` over rows 0..n_kv. With `seqs`: each input's `[n_kv, n_tokens]` block over its
    /// own rows, back to back in input order.
    pub mask: Vec<u16>,
    /// Token indices whose logits are returned.
    pub outputs: Vec<i32>,
    /// Empty: one attention over rows 0..n_kv for all tokens. Else one attention per input over
    /// its own run of rows ([`KvPool::plan_step_per_seq`]).
    pub seqs: Vec<SeqAttn>,
}

/// One input's attention over its own rows: tokens `first_token..+n_tokens` read rows
/// `kv_start..+n_kv`, with the mask block at `mask_offset` (in elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqAttn {
    pub first_token: usize,
    pub n_tokens: usize,
    pub kv_start: usize,
    pub n_kv: usize,
    pub mask_offset: usize,
}

/// Pool state in `ROW_ALIGN`-row units (rounded up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCounts {
    /// Rows holding positions.
    pub used: usize,
    /// Reserved rows not holding a position yet.
    pub reserved: usize,
    pub free: usize,
}

pub struct KvPool {
    layout: Box<dyn Layout>,
    lens: HashMap<SeqId, usize>,
    /// The mask buffer of a plan that has run ([`KvPool::recycle`]). A long-context mask is tens
    /// of MB, above the allocator's mmap threshold: allocated per step, every page faults again
    /// (suspected in the long-context TTFT tail of ttft5-20261119).
    spare_mask: Mutex<Vec<u16>>,
}

impl KvPool {
    pub fn new(layout: Box<dyn Layout>) -> Self {
        Self {
            layout,
            lens: HashMap::new(),
            spare_mask: Mutex::new(Vec::new()),
        }
    }

    /// Takes back the mask buffer of a plan that has run, for the masks of later plans.
    pub fn recycle(&self, plan: StepPlan) {
        let mut spare = self.spare_mask.lock().unwrap_or_else(|e| e.into_inner());
        if plan.mask.capacity() > spare.capacity() {
            *spare = plan.mask;
        }
    }

    /// A mask of `n` elements, all [`F16_NEG_INF`], in the recycled buffer when there is one.
    fn mask_buf(&self, n: usize) -> Vec<u16> {
        let mut spare = self.spare_mask.lock().unwrap_or_else(|e| e.into_inner());
        let mut mask = std::mem::take(&mut *spare);
        drop(spare);
        mask.clear();
        mask.resize(n, F16_NEG_INF);
        mask
    }

    pub fn layout_name(&self) -> &'static str {
        self.layout.name()
    }

    pub fn capacity(&self) -> usize {
        self.layout.capacity()
    }

    pub fn free(&self) -> usize {
        self.layout.free()
    }

    pub fn seq_len(&self, seq: SeqId) -> usize {
        self.lens.get(&seq).copied().unwrap_or(0)
    }

    pub fn reserved(&self, seq: SeqId) -> usize {
        self.layout.reserved(seq)
    }

    /// Row of position `pos`, if the sequence reserves it.
    pub fn row(&self, seq: SeqId, pos: usize) -> Option<usize> {
        (pos < self.layout.reserved(seq)).then(|| self.layout.row(seq, pos))
    }

    /// Rows for positions 0..n of `seq`. The copies move the positions in use.
    pub fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError> {
        let mut copies = self.layout.reserve(seq, n)?;
        // The copies cover the old rows in position order: keep the positions in use.
        let mut left = *self.lens.entry(seq).or_insert(0);
        for c in &mut copies {
            c.n = c.n.min(left);
            left -= c.n;
        }
        copies.retain(|c| c.n > 0);
        Ok(copies)
    }

    /// Keeps the first `keep` positions and releases the reservation beyond them; 0 frees the
    /// sequence. A caller that rewinds and keeps decoding reserves again.
    pub fn seq_rm(&mut self, seq: SeqId, keep: usize) {
        let Some(len) = self.lens.get_mut(&seq) else {
            return;
        };
        *len = (*len).min(keep);
        let keep = *len;
        if keep == 0 {
            self.lens.remove(&seq);
        }
        self.layout.shrink(seq, keep);
    }

    /// Undoes a failed step for `seq`: the length goes back to `len` (when above it) and the
    /// reservation to `reserved` rows (when above it, never below the positions kept). It never
    /// grows anything, so it cannot fail; a reservation that moved stays where it moved.
    pub fn rollback(&mut self, seq: SeqId, len: usize, reserved: usize) {
        if let Some(l) = self.lens.get_mut(&seq) {
            *l = (*l).min(len);
            if *l == 0 {
                self.lens.remove(&seq);
            }
        }
        let keep = reserved.max(self.seq_len(seq));
        if self.layout.reserved(seq) > keep {
            self.layout.shrink(seq, keep);
        }
    }

    /// `dst` must hold no positions (it may already reserve rows, as a scheduler does before the
    /// copy): reserves up to n rows for it and returns the copies src -> dst.
    pub fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError> {
        if src == dst || self.seq_len(dst) > 0 {
            return Err(KvError::Invalid(format!("sequence {dst} is not empty")));
        }
        if !self.lens.contains_key(&src) {
            return Err(KvError::UnknownSeq(src));
        }
        if n > self.seq_len(src) {
            return Err(KvError::Invalid(format!(
                "sequence {src} holds {} positions, not {n}",
                self.seq_len(src)
            )));
        }
        self.layout.reserve(dst, n)?;
        if n > 0 {
            self.lens.insert(dst, n);
        }
        let mut copies: Vec<RowCopy> = Vec::new();
        for (s, d) in (0..n).map(|p| (self.layout.row(src, p), self.layout.row(dst, p))) {
            match copies.last_mut() {
                Some(c) if c.src + c.n == s && c.dst + c.n == d => c.n += 1,
                _ => copies.push(RowCopy {
                    src: s,
                    dst: d,
                    n: 1,
                }),
            }
        }
        Ok(copies)
    }

    fn check(&self, inputs: &[StepInput]) -> Result<(), KvError> {
        for (i, inp) in inputs.iter().enumerate() {
            if inp.n_tokens == 0 {
                return Err(KvError::Invalid(format!(
                    "sequence {} has no tokens",
                    inp.seq
                )));
            }
            if inputs[..i].iter().any(|o| o.seq == inp.seq) {
                return Err(KvError::Invalid(format!("sequence {} twice", inp.seq)));
            }
            let len = *self
                .lens
                .get(&inp.seq)
                .ok_or(KvError::UnknownSeq(inp.seq))?;
            let have = self.layout.reserved(inp.seq);
            if have < len + inp.n_tokens {
                return Err(KvError::Invalid(format!(
                    "sequence {} reserves {have} rows, the step needs {}",
                    inp.seq,
                    len + inp.n_tokens
                )));
            }
        }
        Ok(())
    }

    /// `n_kv` of a step: 1 + the highest row any input reads or writes, padded to `ROW_ALIGN`.
    pub fn step_n_kv(&self, inputs: &[StepInput]) -> Result<usize, KvError> {
        self.check(inputs)?;
        let top = inputs
            .iter()
            .filter_map(|i| self.layout.max_row(i.seq, self.lens[&i.seq] + i.n_tokens))
            .max()
            .map_or(0, |r| r + 1);
        Ok(round_up(top, ROW_ALIGN).max(ROW_ALIGN))
    }

    /// Sequences distinct; each reservation must cover seq_len + n_tokens.
    pub fn plan_step(&self, inputs: &[StepInput]) -> Result<StepPlan, KvError> {
        let n_kv = self.step_n_kv(inputs)?;
        let n_tokens: usize = inputs.iter().map(|i| i.n_tokens).sum();
        let mut pos = Vec::with_capacity(n_tokens);
        let mut rows = Vec::with_capacity(n_tokens);
        let mut outputs = Vec::new();
        let mut mask = self.mask_buf(n_kv * n_tokens);
        for inp in inputs {
            let len = self.lens[&inp.seq];
            let first = pos.len();
            // One run of rows: a token's open rows are one slice, not a row lookup each.
            let run = self.layout.run_start(inp.seq);
            for i in 0..inp.n_tokens {
                let t = first + i;
                pos.push((len + i) as i32);
                rows.push(self.layout.row(inp.seq, len + i) as i32);
                if inp.all_logits || i + 1 == inp.n_tokens {
                    outputs.push(t as i32);
                }
                let at = t * n_kv;
                match run {
                    Some(start) => mask[at + start..=at + start + len + i].fill(0),
                    None => {
                        for p in 0..=len + i {
                            mask[at + self.layout.row(inp.seq, p)] = 0;
                        }
                    }
                }
            }
        }
        Ok(StepPlan {
            pos,
            rows,
            n_kv,
            mask,
            outputs,
            seqs: Vec::new(),
        })
    }

    /// Like [`KvPool::plan_step`], with attention per input over its own rows: each input's
    /// positions must be one run of rows (chunked), else `None`. An input reads
    /// `round_up(len + n_tokens, ROW_ALIGN)` rows from the start of its run; they lie inside its
    /// reservation, so no other sequence's rows are read.
    pub fn plan_step_per_seq(&self, inputs: &[StepInput]) -> Result<Option<StepPlan>, KvError> {
        let n_kv = self.step_n_kv(inputs)?;
        let mut seqs = Vec::with_capacity(inputs.len());
        let (mut first_token, mut mask_len) = (0, 0);
        for inp in inputs {
            let Some(kv_start) = self.layout.run_start(inp.seq) else {
                return Ok(None);
            };
            let n = round_up(self.lens[&inp.seq] + inp.n_tokens, ROW_ALIGN);
            if n > self.layout.reserved(inp.seq) {
                return Ok(None);
            }
            seqs.push(SeqAttn {
                first_token,
                n_tokens: inp.n_tokens,
                kv_start,
                n_kv: n,
                mask_offset: mask_len,
            });
            first_token += inp.n_tokens;
            mask_len += n * inp.n_tokens;
        }
        let mut pos = Vec::with_capacity(first_token);
        let mut rows = Vec::with_capacity(first_token);
        let mut outputs = Vec::new();
        let mut mask = self.mask_buf(mask_len);
        for (inp, s) in inputs.iter().zip(&seqs) {
            let len = self.lens[&inp.seq];
            for i in 0..inp.n_tokens {
                let t = s.first_token + i;
                pos.push((len + i) as i32);
                rows.push((s.kv_start + len + i) as i32);
                if inp.all_logits || i + 1 == inp.n_tokens {
                    outputs.push(t as i32);
                }
                let at = s.mask_offset + i * s.n_kv;
                mask[at..=at + len + i].fill(0);
            }
        }
        Ok(Some(StepPlan {
            pos,
            rows,
            n_kv,
            mask,
            outputs,
            seqs,
        }))
    }

    /// The plans of `input` cut into chunks of `sizes` tokens (ADR 0077): chunk j is planned as a
    /// step of its own after chunks 0..j, so a pipelined step computes what the same chunks as
    /// serial steps compute. Lengths are unchanged on return (the caller commits `input` once).
    /// `sizes` must be non-empty and add up to `input.n_tokens`.
    pub fn plan_chunks(
        &mut self,
        input: StepInput,
        sizes: &[usize],
    ) -> Result<Vec<StepPlan>, KvError> {
        if sizes.is_empty() || sizes.iter().sum::<usize>() != input.n_tokens {
            return Err(KvError::Invalid(format!(
                "chunks {sizes:?} do not cut {} tokens",
                input.n_tokens
            )));
        }
        let before = self.lens.get(&input.seq).copied();
        let mut plans = Vec::with_capacity(sizes.len());
        let mut res = Ok(());
        for &n in sizes {
            let chunk = [StepInput {
                n_tokens: n,
                ..input
            }];
            match self.plan_step(&chunk) {
                Ok(p) => plans.push(p),
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
            self.commit(&chunk);
        }
        match before {
            Some(len) => {
                self.lens.insert(input.seq, len);
            }
            None => {
                self.lens.remove(&input.seq);
            }
        }
        res.map(|()| plans)
    }

    /// After a successful decode: lengths grow by n_tokens.
    pub fn commit(&mut self, inputs: &[StepInput]) {
        for inp in inputs {
            *self.lens.entry(inp.seq).or_insert(0) += inp.n_tokens;
        }
    }

    /// Rows of positions 0..n of `seq` as (first row, length) runs, for snapshots.
    pub fn runs(&self, seq: SeqId, n: usize) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for p in 0..n {
            let r = self.layout.row(seq, p);
            match runs.last_mut() {
                Some((s, l)) if *s + *l == r => *l += 1,
                _ => runs.push((r, 1)),
            }
        }
        runs
    }

    pub fn blocks(&self) -> BlockCounts {
        let used: usize = self.lens.values().sum();
        let reserved: usize = self
            .lens
            .iter()
            .map(|(&s, &l)| self.layout.reserved(s).saturating_sub(l))
            .sum();
        BlockCounts {
            used: used.div_ceil(ROW_ALIGN),
            reserved: reserved.div_ceil(ROW_ALIGN),
            free: self.layout.free().div_ceil(ROW_ALIGN),
        }
    }
}

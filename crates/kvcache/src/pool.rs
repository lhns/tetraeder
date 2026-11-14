//! The pool: a layout plus the length of every sequence, and the per-step plan.

use crate::{F16_NEG_INF, KvError, Layout, ROW_ALIGN, RowCopy, SeqId, round_up};
use std::collections::HashMap;

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
    /// F16 bits, `[n_kv, n_tokens]`, token-major: 0 where the token may read the row.
    pub mask: Vec<u16>,
    /// Token indices whose logits are returned.
    pub outputs: Vec<i32>,
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
}

impl KvPool {
    pub fn new(layout: Box<dyn Layout>) -> Self {
        Self {
            layout,
            lens: HashMap::new(),
        }
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
        let len = *self.lens.entry(seq).or_insert(0);
        for c in &mut copies {
            c.n = c.n.min(len);
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

    /// `dst` must be empty: reserves n rows for it and returns the copies src -> dst.
    pub fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError> {
        if src == dst || self.seq_len(dst) > 0 || self.layout.reserved(dst) > 0 {
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
        let mut mask = vec![F16_NEG_INF; n_kv * n_tokens];
        for inp in inputs {
            let len = self.lens[&inp.seq];
            let first = pos.len();
            for i in 0..inp.n_tokens {
                let t = first + i;
                pos.push((len + i) as i32);
                rows.push(self.layout.row(inp.seq, len + i) as i32);
                if inp.all_logits || i + 1 == inp.n_tokens {
                    outputs.push(t as i32);
                }
                for p in 0..=len + i {
                    mask[t * n_kv + self.layout.row(inp.seq, p)] = 0;
                }
            }
        }
        Ok(StepPlan {
            pos,
            rows,
            n_kv,
            mask,
            outputs,
        })
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

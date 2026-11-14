//! Fakes for the batching tests (ADR 0065): `FakeKv`, a `BatchBackend` with its own row
//! accounting, and `OldFake`, a one-sequence `Backend` to run under `SingleSeq`.
//!
//! Logits are a deterministic function of a hash of the sequence's tokens up to each position, so
//! a request decoded in a batch can be compared with the same request decoded alone (P4).
#![allow(dead_code)]

use std::ops::Range;
use tetraeder_scheduler::{Backend, BackendError, BatchBackend, Outputs, SeqId, SeqInput};

/// FNV-1a over the tokens.
fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The logits after `tokens`: `n_vocab` values in [-1, 1) derived from the token hash.
pub fn logits_for(n_vocab: usize, tokens: &[i32]) -> Vec<f32> {
    let h = hash_tokens(tokens);
    (0..n_vocab)
        .map(|v| {
            let mut x = h ^ (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

#[derive(Default, Clone)]
struct Seq {
    tokens: Vec<i32>,
    /// Reserved rows; position `p` lives in `rows[p]`.
    rows: Vec<usize>,
}

pub struct FakeKv {
    n_vocab: usize,
    n_ctx_seq: usize,
    capacity: usize,
    /// Pool size seen by reservations when smaller than `capacity` (injects `KvFull`).
    kv_full_at: Option<usize>,
    /// Every n-th decode fails after writing, then rolls back.
    fail_every: Option<u32>,
    decodes: u32,
    pool: Vec<Option<i32>>,
    free_rows: Vec<usize>,
    seqs: Vec<Seq>,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl FakeKv {
    pub fn new(n_vocab: usize, n_seq_max: usize, n_ctx_seq: usize, capacity: usize) -> Self {
        Self {
            n_vocab,
            n_ctx_seq,
            capacity,
            kv_full_at: None,
            fail_every: None,
            decodes: 0,
            pool: vec![None; capacity],
            free_rows: (0..capacity).rev().collect(),
            seqs: vec![Seq::default(); n_seq_max],
            logits: Vec::new(),
            spans: Vec::new(),
        }
    }

    pub fn with_fail_every(mut self, n: Option<u32>) -> Self {
        self.fail_every = n;
        self
    }

    /// Reservations see only `rows` rows of the pool.
    pub fn with_kv_full_at(mut self, rows: Option<usize>) -> Self {
        self.kv_full_at = rows;
        self
    }

    fn limit(&self) -> usize {
        self.kv_full_at
            .map_or(self.capacity, |n| n.min(self.capacity))
    }

    pub fn reserved_total(&self) -> usize {
        self.seqs.iter().map(|s| s.rows.len()).sum()
    }

    /// Rows reserved by `seq`, position order.
    pub fn rows_of(&self, seq: SeqId) -> &[usize] {
        &self.seqs[seq as usize].rows
    }

    pub fn tokens_of(&self, seq: SeqId) -> &[i32] {
        &self.seqs[seq as usize].tokens
    }

    /// What the pool holds in the written rows of `seq`, position order.
    pub fn contents(&self, seq: SeqId) -> Vec<Option<i32>> {
        let s = &self.seqs[seq as usize];
        s.rows[..s.tokens.len()]
            .iter()
            .map(|&r| self.pool[r])
            .collect()
    }

    /// Whether a row is free in the pool (not reserved by any sequence).
    pub fn is_free_row(&self, row: usize) -> bool {
        self.free_rows.contains(&row)
    }

    fn seq_ok(&self, seq: SeqId) -> Result<usize, BackendError> {
        let i = seq as usize;
        if i < self.seqs.len() {
            Ok(i)
        } else {
            Err(BackendError::Failed(format!("no sequence {seq}")))
        }
    }

    fn grow_rows(&mut self, i: usize, n: usize) {
        while self.seqs[i].rows.len() < n {
            let r = self.free_rows.pop().expect("checked free rows");
            self.seqs[i].rows.push(r);
        }
    }

    fn cut(&mut self, i: usize, keep: usize, keep_reservation: bool) {
        let s = &mut self.seqs[i];
        for &r in &s.rows[keep.min(s.tokens.len())..s.tokens.len()] {
            self.pool[r] = None;
        }
        s.tokens.truncate(keep);
        if !keep_reservation {
            let release = s.rows.split_off(keep);
            self.free_rows.extend(release);
        }
    }
}

impl BatchBackend for FakeKv {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn n_seq_max(&self) -> usize {
        self.seqs.len()
    }

    fn n_ctx_seq(&self) -> usize {
        self.n_ctx_seq
    }

    fn kv_capacity(&self) -> usize {
        self.limit()
    }

    fn kv_free(&self) -> usize {
        self.limit().saturating_sub(self.reserved_total())
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        self.seqs.get(seq as usize).map_or(0, |s| s.tokens.len())
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        let i = self.seq_ok(seq)?;
        let need = n.saturating_sub(self.seqs[i].rows.len());
        if n > self.n_ctx_seq || need > self.kv_free() {
            return Err(BackendError::KvFull {
                seq,
                needed: need,
                free: self.kv_free(),
            });
        }
        self.grow_rows(i, n);
        Ok(())
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        let i = self.seq_ok(seq)?;
        let had = self.seqs[i].tokens.len();
        self.cut(i, keep.min(had), keep > 0);
        Ok(())
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        let (s, d) = (self.seq_ok(src)?, self.seq_ok(dst)?);
        if s == d || !self.seqs[d].tokens.is_empty() || n > self.seqs[s].tokens.len() {
            return Err(BackendError::Failed(format!(
                "seq_cp {src} -> {dst} of {n}: dst must be empty and distinct, n within src"
            )));
        }
        let need = n.saturating_sub(self.seqs[d].rows.len());
        if need > self.kv_free() {
            return Err(BackendError::KvFull {
                seq: dst,
                needed: need,
                free: self.kv_free(),
            });
        }
        self.grow_rows(d, n);
        let toks = self.seqs[s].tokens[..n].to_vec();
        for (p, t) in toks.iter().enumerate() {
            let r = self.seqs[d].rows[p];
            self.pool[r] = Some(*t);
        }
        self.seqs[d].tokens = toks;
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        // Validate everything before touching state.
        let mut seen = Vec::with_capacity(inputs.len());
        let mut need_total = 0;
        for inp in inputs {
            let i = self.seq_ok(inp.seq)?;
            if seen.contains(&i) {
                return Err(BackendError::Failed(format!("sequence {} twice", inp.seq)));
            }
            seen.push(i);
            if inp.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            let after = self.seqs[i].tokens.len() + inp.tokens.len();
            if after > self.n_ctx_seq {
                return Err(BackendError::KvFull {
                    seq: inp.seq,
                    needed: after - self.seqs[i].rows.len().min(after),
                    free: self.kv_free(),
                });
            }
            need_total += after.saturating_sub(self.seqs[i].rows.len());
        }
        if need_total > self.kv_free() {
            let first = inputs
                .iter()
                .find(|i| {
                    let s = &self.seqs[i.seq as usize];
                    s.tokens.len() + i.tokens.len() > s.rows.len()
                })
                .map_or(0, |i| i.seq);
            return Err(BackendError::KvFull {
                seq: first,
                needed: need_total,
                free: self.kv_free(),
            });
        }
        self.decodes += 1;
        let before: Vec<(usize, usize)> = seen
            .iter()
            .map(|&i| (self.seqs[i].tokens.len(), self.seqs[i].rows.len()))
            .collect();
        self.logits.clear();
        self.spans.clear();
        for inp in inputs {
            let i = inp.seq as usize;
            let start = self.seqs[i].tokens.len();
            self.grow_rows(i, start + inp.tokens.len());
            let row0 = self.logits.len() / self.n_vocab;
            for (k, &t) in inp.tokens.iter().enumerate() {
                let r = self.seqs[i].rows[start + k];
                self.pool[r] = Some(t);
                self.seqs[i].tokens.push(t);
                if inp.all_logits || k + 1 == inp.tokens.len() {
                    let l = logits_for(self.n_vocab, &self.seqs[i].tokens);
                    self.logits.extend(l);
                }
            }
            self.spans.push(row0..self.logits.len() / self.n_vocab);
        }
        if self
            .fail_every
            .is_some_and(|n| self.decodes.is_multiple_of(n))
        {
            for (&i, &(len, res)) in seen.iter().zip(&before) {
                self.cut(i, len, true);
                let release = self.seqs[i].rows.split_off(res);
                self.free_rows.extend(release);
            }
            return Err(BackendError::Failed("injected failure".into()));
        }
        Ok(Outputs::new(self.n_vocab, &self.logits, &self.spans))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        let i = self.seq_ok(seq)?;
        Ok(self.seqs[i]
            .tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect())
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        let i = self.seq_ok(seq)?;
        if bytes.len() < n * 4 || n > self.n_ctx_seq {
            return Err(BackendError::Failed("bad snapshot".into()));
        }
        let have = self.seqs[i].rows.len();
        if n.saturating_sub(have) > self.kv_free() {
            return Err(BackendError::KvFull {
                seq,
                needed: n - have,
                free: self.kv_free(),
            });
        }
        self.cut(i, 0, true);
        self.grow_rows(i, n);
        for p in 0..n {
            let t = i32::from_le_bytes(bytes[p * 4..p * 4 + 4].try_into().unwrap());
            let r = self.seqs[i].rows[p];
            self.pool[r] = Some(t);
            self.seqs[i].tokens.push(t);
        }
        Ok(())
    }

    fn on_gpu(&self) -> bool {
        false
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
}

/// A one-sequence `Backend` with the same logits function. With `fail_every`, every n-th `eval`
/// appends its tokens and then fails, so the adapter has something to roll back.
pub struct OldFake {
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub tokens: Vec<i32>,
    pub fail_every: Option<u32>,
    evals: u32,
    logits: Vec<f32>,
}

impl OldFake {
    pub fn new(n_ctx: usize, n_vocab: usize, fail_every: Option<u32>) -> Self {
        Self {
            n_ctx,
            n_vocab,
            tokens: Vec::new(),
            fail_every,
            evals: 0,
            logits: Vec::new(),
        }
    }
}

impl Backend for OldFake {
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn n_past(&self) -> usize {
        self.tokens.len()
    }
    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        self.tokens.truncate(keep);
        Ok(())
    }
    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        self.evals += 1;
        self.tokens.extend_from_slice(tokens);
        if self
            .fail_every
            .is_some_and(|n| self.evals.is_multiple_of(n))
        {
            return Err("injected failure".into());
        }
        self.logits = logits_for(self.n_vocab, &self.tokens);
        Ok(&self.logits)
    }
    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        Ok(self.tokens.iter().flat_map(|t| t.to_le_bytes()).collect())
    }
    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.tokens = bytes[..n * 4]
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(())
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
}

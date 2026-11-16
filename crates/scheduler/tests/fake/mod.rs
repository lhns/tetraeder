//! Fakes for the batching tests (ADR 0065): `FakeKv`, a `BatchBackend` with its own row
//! accounting.
//!
//! Logits are a deterministic function of a hash of the sequence's tokens up to each position, so
//! a request decoded in a batch can be compared with the same request decoded alone (P4).
#![allow(dead_code)]

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use tetraeder_scheduler::{BackendError, BatchBackend, Outputs, SeqId, SeqInput};

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

/// What a test reads from a `FakeKv` it no longer owns (the scheduler's worker does).
#[derive(Default)]
pub struct Probe {
    /// Successful `decode` calls, including the injected failures.
    pub decodes: AtomicU32,
    /// Most inputs in one `decode`.
    pub max_inputs: AtomicUsize,
    /// Most tokens in one `decode`.
    pub max_step_tokens: AtomicUsize,
    /// Successful `seq_cp` calls as (src, dst, n).
    pub seq_cps: std::sync::Mutex<Vec<(SeqId, SeqId, usize)>>,
    /// Set by a test: the backend "restarts" (loses every sequence) at its next call, like the
    /// worker process of `serve --worker`.
    pub restart: AtomicBool,
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
    /// A step holding this token fails with `Failed` before touching state.
    poison: Option<i32>,
    /// `seq_rm` to a positive length also releases the reservation beyond that length rounded up
    /// to this many rows (the dense backend's behaviour).
    shrink_to_multiple: Option<usize>,
    /// The next `seq_cp` cancels this token and fails, as a request cancelled while its busy
    /// copy fails.
    fail_cp_cancelling: Option<tokio_util::sync::CancellationToken>,
    /// Sequences that held rows when the backend restarted and were not freed or refilled since;
    /// using one fails, as in `RemoteBackend`.
    lost: HashSet<u32>,
    decodes: u32,
    pool: Vec<Option<i32>>,
    free_rows: Vec<usize>,
    seqs: Vec<Seq>,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
    probe: Arc<Probe>,
}

impl FakeKv {
    pub fn new(n_vocab: usize, n_seq_max: usize, n_ctx_seq: usize, capacity: usize) -> Self {
        Self {
            n_vocab,
            n_ctx_seq,
            capacity,
            kv_full_at: None,
            fail_every: None,
            poison: None,
            shrink_to_multiple: None,
            fail_cp_cancelling: None,
            lost: HashSet::new(),
            decodes: 0,
            pool: vec![None; capacity],
            free_rows: (0..capacity).rev().collect(),
            seqs: vec![Seq::default(); n_seq_max],
            logits: Vec::new(),
            spans: Vec::new(),
            probe: Arc::default(),
        }
    }

    pub fn probe(&self) -> Arc<Probe> {
        Arc::clone(&self.probe)
    }

    pub fn with_fail_every(mut self, n: Option<u32>) -> Self {
        self.fail_every = n;
        self
    }

    pub fn with_poison(mut self, token: Option<i32>) -> Self {
        self.poison = token;
        self
    }

    pub fn with_fail_cp_cancelling(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.fail_cp_cancelling = Some(token);
        self
    }

    pub fn with_shrink_on_rm(mut self, multiple: Option<usize>) -> Self {
        self.shrink_to_multiple = multiple;
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

    /// Applies a restart a test asked for: every sequence is gone, those that held rows are lost.
    fn apply_restart(&mut self) {
        if !self.probe.restart.swap(false, Ordering::SeqCst) {
            return;
        }
        for (i, s) in self.seqs.iter_mut().enumerate() {
            if !s.tokens.is_empty() || !s.rows.is_empty() {
                self.lost.insert(i as u32);
            }
            *s = Seq::default();
        }
        self.pool.fill(None);
        self.free_rows = (0..self.capacity).rev().collect();
    }

    fn check_lost(&self, seq: SeqId) -> Result<(), BackendError> {
        if self.lost.contains(&seq) {
            Err(BackendError::Failed(format!(
                "sequence {seq} was lost when the backend restarted"
            )))
        } else {
            Ok(())
        }
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

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.seqs.get(seq as usize).map_or(0, |s| s.rows.len())
    }

    fn seq_lost(&self, seq: SeqId) -> bool {
        let held = self
            .seqs
            .get(seq as usize)
            .is_some_and(|s| !s.tokens.is_empty() || !s.rows.is_empty());
        self.lost.contains(&seq) || (held && self.probe.restart.load(Ordering::SeqCst))
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.apply_restart();
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
        self.apply_restart();
        let i = self.seq_ok(seq)?;
        if keep == 0 {
            self.lost.remove(&seq);
        }
        let had = self.seqs[i].tokens.len();
        self.cut(i, keep.min(had), keep > 0);
        if let (Some(m), true) = (self.shrink_to_multiple, keep > 0) {
            let to = keep.div_ceil(m) * m;
            if self.seqs[i].rows.len() > to {
                let release = self.seqs[i].rows.split_off(to);
                self.free_rows.extend(release);
            }
        }
        Ok(())
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        if let Some(t) = self.fail_cp_cancelling.take() {
            t.cancel();
            return Err(BackendError::Failed("injected".into()));
        }
        self.apply_restart();
        self.check_lost(src)?;
        let (s, d) = (self.seq_ok(src)?, self.seq_ok(dst)?);
        self.lost.remove(&dst);
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
        self.probe.seq_cps.lock().unwrap().push((src, dst, n));
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.apply_restart();
        // Validate everything before touching state.
        let mut seen = Vec::with_capacity(inputs.len());
        let mut need_total = 0;
        for inp in inputs {
            let i = self.seq_ok(inp.seq)?;
            self.check_lost(inp.seq)?;
            if seen.contains(&i) {
                return Err(BackendError::Failed(format!("sequence {} twice", inp.seq)));
            }
            seen.push(i);
            if inp.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            if self.poison.is_some_and(|p| inp.tokens.contains(&p)) {
                return Err(BackendError::Failed("poisoned token".into()));
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
        self.probe.decodes.fetch_add(1, Ordering::SeqCst);
        self.probe
            .max_inputs
            .fetch_max(inputs.len(), Ordering::SeqCst);
        self.probe.max_step_tokens.fetch_max(
            inputs.iter().map(|i| i.tokens.len()).sum(),
            Ordering::SeqCst,
        );
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

    fn kv_row_bytes(&self) -> usize {
        4
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.apply_restart();
        self.check_lost(seq)?;
        let i = self.seq_ok(seq)?;
        Ok(self.seqs[i]
            .tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect())
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        self.apply_restart();
        let i = self.seq_ok(seq)?;
        self.lost.remove(&seq);
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

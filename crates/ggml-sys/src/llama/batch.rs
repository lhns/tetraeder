//! Token batch.

use super::{LlamaError, Result, Token};
use crate::raw as sys;

/// A token batch for one sequence set, backed by `llama_batch_init`.
pub struct Batch {
    raw: sys::llama_batch,
    cap: usize,
    len: usize,
}

// SAFETY: the batch exclusively owns its heap arrays; there is no shared state.
unsafe impl Send for Batch {}

impl Batch {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        // SAFETY: allocates arrays for `cap` tokens and one sequence id per token.
        let raw = unsafe { sys::llama_batch_init(cap as i32, 0, 1) };
        Self { raw, cap, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Appends a token at position `pos` of sequence `seq`; `logits` requests its output.
    pub fn push(&mut self, token: Token, pos: i32, seq: i32, logits: bool) -> Result<()> {
        if self.len == self.cap {
            return Err(LlamaError::BatchFull(self.cap));
        }
        let i = self.len;
        // SAFETY: `i < cap`, and every array has `cap` entries; each `seq_id[i]` points at a
        // one-element array (n_seq_max = 1).
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            **self.raw.seq_id.add(i) = seq;
            *self.raw.logits.add(i) = i8::from(logits);
        }
        self.len += 1;
        Ok(())
    }

    pub(super) fn logits_flags(&self) -> impl Iterator<Item = bool> + '_ {
        // SAFETY: the first `len` entries of `logits` were written by `push`.
        (0..self.len).map(|i| unsafe { *self.raw.logits.add(i) } != 0)
    }

    pub(super) fn raw(&self) -> sys::llama_batch {
        let mut b = self.raw;
        b.n_tokens = self.len as i32;
        b
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: allocated by llama_batch_init and freed once.
        unsafe { sys::llama_batch_free(self.raw) }
    }
}

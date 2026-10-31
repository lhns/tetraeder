//! Token batch.

use super::{LlamaError, Result, Token};

/// One token of a batch.
#[derive(Debug, Clone, Copy)]
pub(super) struct Entry {
    pub token: Token,
    pub pos: i32,
    pub seq: i32,
    pub logits: bool,
}

/// A token batch for one or more sequences. It only holds the entries (reused across `clear`,
/// so filling it allocates nothing); [`super::Context::decode`] renders them into the
/// context's extended batch, as llama-bench does.
pub struct Batch {
    entries: Vec<Entry>,
    cap: usize,
}

impl Batch {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            entries: Vec::with_capacity(cap),
            cap,
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Appends a token at position `pos` of sequence `seq`; `logits` requests its output.
    pub fn push(&mut self, token: Token, pos: i32, seq: i32, logits: bool) -> Result<()> {
        if self.entries.len() == self.cap {
            return Err(LlamaError::BatchFull(self.cap));
        }
        self.entries.push(Entry {
            token,
            pos,
            seq,
            logits,
        });
        Ok(())
    }

    pub(super) fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

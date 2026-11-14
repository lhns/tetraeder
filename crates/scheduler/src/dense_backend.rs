//! The Qwen3 dense builder as a [`BatchBackend`] (ADR 0065, 0066): sequences over the builder's
//! KV pool, decoded together in one step.

use crate::backend::{BackendError, BatchBackend, Outputs, SeqId, SeqInput};
use std::ops::Range;
use tetraeder_model::{ModelError, Qwen3, SeqEval};

pub struct DenseBackend {
    model: Qwen3,
    on_gpu: bool,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl DenseBackend {
    pub fn new(model: Qwen3, on_gpu: bool) -> Self {
        Self {
            model,
            on_gpu,
            logits: Vec::new(),
            spans: Vec::new(),
        }
    }

    pub fn model(&self) -> &Qwen3 {
        &self.model
    }

    fn kv_full(&self, seq: SeqId, needed: usize) -> BackendError {
        BackendError::KvFull {
            seq,
            needed,
            free: self.model.pool().free(),
        }
    }

    /// Rows `seq` lacks to hold `n` positions: `KvFull` when it cannot have them (nothing
    /// changed beyond reservations of earlier inputs of the same step, which stay).
    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        let have = self.model.pool().reserved(seq);
        if n <= have {
            return Ok(());
        }
        if n > self.model.pool().capacity() {
            return Err(self.kv_full(seq, n - have));
        }
        match self.model.try_reserve(seq, n) {
            Ok(true) => Ok(()),
            Ok(false) => Err(self.kv_full(seq, n - have)),
            Err(e) => Err(failed(e)),
        }
    }

    /// Puts `seq`'s reservation back to `before` rows. `seq_rm` shrinks to the positions held
    /// (rounded up to 256 rows) or frees an empty sequence; growing again takes the rows the
    /// shrink just released, in place or, for an empty sequence, anywhere they fit.
    fn give_back(&mut self, seq: SeqId, before: usize) {
        let _ = self.model.seq_rm(seq, before);
        if before > self.model.pool().reserved(seq) {
            let _ = self.model.try_reserve(seq, before);
        }
    }

    /// Reserves, then runs the step. `grown` collects (sequence, reservation before) of every
    /// sequence this call gave more rows.
    fn step(
        &mut self,
        inputs: &[SeqInput<'_>],
        grown: &mut Vec<(SeqId, usize)>,
    ) -> Result<Vec<Range<usize>>, BackendError> {
        for i in inputs {
            if i.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            let before = self.model.pool().reserved(i.seq);
            let need = self.model.pool().seq_len(i.seq) + i.tokens.len();
            self.reserve(i.seq, need)?;
            if self.model.pool().reserved(i.seq) != before {
                grown.push((i.seq, before));
            }
        }
        let evals: Vec<SeqEval<'_>> = inputs
            .iter()
            .map(|i| SeqEval {
                seq: i.seq,
                tokens: i.tokens,
                all_logits: i.all_logits,
            })
            .collect();
        self.model.decode(&evals, &mut self.logits).map_err(failed)
    }
}

fn failed(e: ModelError) -> BackendError {
    BackendError::Failed(e.to_string())
}

impl BatchBackend for DenseBackend {
    fn n_vocab(&self) -> usize {
        self.model.n_vocab()
    }

    fn n_seq_max(&self) -> usize {
        self.model.n_seq_max()
    }

    fn n_ctx_seq(&self) -> usize {
        self.model.pool().capacity()
    }

    fn kv_capacity(&self) -> usize {
        self.model.pool().capacity()
    }

    fn kv_free(&self) -> usize {
        self.model.pool().free()
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        self.model.pool().seq_len(seq)
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.reserve(seq, n)
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        self.model.seq_rm(seq, keep).map_err(failed)
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        match self.model.try_seq_cp(src, dst, n) {
            Ok(true) => Ok(()),
            Ok(false) => Err(self.kv_full(dst, n)),
            Err(e) => Err(failed(e)),
        }
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.spans.clear();
        self.logits.clear();
        if inputs.is_empty() {
            return Ok(Outputs::new(
                self.model.n_vocab(),
                &self.logits,
                &self.spans,
            ));
        }
        // Every input's rows are reserved first. A failure, here or in the step, gives back what
        // this call reserved, so lengths and free rows are as they were.
        let mut grown: Vec<(SeqId, usize)> = Vec::new();
        match self.step(inputs, &mut grown) {
            Ok(spans) => self.spans = spans,
            Err(e) => {
                for (seq, before) in grown.into_iter().rev() {
                    self.give_back(seq, before);
                }
                self.logits.clear();
                return Err(e);
            }
        }
        Ok(Outputs::new(
            self.model.n_vocab(),
            &self.logits,
            &self.spans,
        ))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.model.seq_snapshot(seq).map_err(failed)
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        match self.model.try_seq_restore(seq, bytes, n) {
            Ok(true) => Ok(()),
            Ok(false) => Err(self.kv_full(seq, n)),
            Err(e) => Err(failed(e)),
        }
    }

    fn kv_format(&self) -> String {
        self.model.kv_format().to_string()
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }

    fn counters(&self) -> Vec<(&'static str, f64)> {
        vec![("step_graphs_built", self.model.step_graphs_built() as f64)]
    }
}

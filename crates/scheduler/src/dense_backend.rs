//! The Qwen3 dense builder as a [`BatchBackend`] (ADR 0065, 0066): sequences over the builder's
//! KV pool, decoded together in one step.

use crate::backend::{BackendError, BatchBackend, Outputs, SeqId, SeqInput, step_groups};
use std::ops::Range;
use tetraeder_model::{BackendChoice, DenseOptions, ModelError, Qwen3, SeqEval};
use tetraeder_plan::Topology;

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

    /// The model loaded across the ranks of `topo` on the executor (ADR 0069, 0070).
    pub fn with_topology(
        path: &std::path::Path,
        topo: &Topology,
        opts: &DenseOptions,
    ) -> Result<Self, ModelError> {
        let on_gpu = matches!(opts.backend, BackendChoice::Gpu { .. });
        Ok(Self::new(Qwen3::load_sharded(path, topo, opts)?, on_gpu))
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

    /// Reserves, then runs the step.
    fn step(&mut self, inputs: &[SeqInput<'_>]) -> Result<Vec<Range<usize>>, BackendError> {
        for i in inputs {
            if i.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            let need = self.model.pool().seq_len(i.seq) + i.tokens.len();
            self.reserve(i.seq, need)?;
        }
        let evals: Vec<SeqEval<'_>> = inputs
            .iter()
            .map(|i| SeqEval {
                seq: i.seq,
                tokens: i.tokens,
                all_logits: i.all_logits,
            })
            .collect();
        let groups = step_groups(inputs, self.on_gpu);
        if groups.len() == 1 {
            return self.model.decode(&evals, &mut self.logits).map_err(failed);
        }
        // One model call per group; `decode` rolls the earlier groups back on a failure.
        let mut spans = Vec::with_capacity(inputs.len());
        let mut rows = Vec::new();
        for range in &groups {
            let s = self
                .model
                .decode(&evals[range.clone()], &mut rows)
                .map_err(failed)?;
            let base = self.logits.len() / self.model.n_vocab().max(1);
            spans.extend(s.into_iter().map(|r| r.start + base..r.end + base));
            self.logits.extend_from_slice(&rows);
        }
        Ok(spans)
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

    /// The snapshot pool sizes its spares from this; the trait's 0 left the dense engine's
    /// first save without a spare (`20261117e`, ADR 0074).
    fn kv_row_bytes(&self) -> usize {
        self.model.kv_row_bytes()
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

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.model.pool().reserved(seq)
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
        // Every input's rows are reserved first. A failure, here or in any group of the step,
        // puts every input back to its length and reservation from before the call (a rollback
        // never grows a reservation, so it cannot fail for lack of rows).
        let before: Vec<(SeqId, usize, usize)> = inputs
            .iter()
            .map(|i| (i.seq, self.seq_len(i.seq), self.seq_reserved(i.seq)))
            .collect();
        match self.step(inputs) {
            Ok(spans) => self.spans = spans,
            Err(e) => {
                for &(seq, len, reserved) in before.iter().rev() {
                    if let Err(r) = self.model.rollback(seq, len, reserved) {
                        tracing::warn!("cannot roll back sequence {seq}: {r}");
                    }
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

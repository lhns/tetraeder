//! The Qwen3 dense builder as a [`BatchBackend`] (ADR 0065, 0066): sequences over the builder's
//! KV pool, decoded together in one step.

use crate::backend::{
    BackendError, BatchBackend, Feed, Outputs, SegmentBackend, SegmentInfo, SeqId, SeqInput,
    StepOut, step_groups,
};
use std::ops::Range;
use tetraeder_model::{BackendChoice, DenseOptions, ModelError, Qwen3, SeqEval};
use tetraeder_plan::Topology;

pub struct DenseBackend {
    model: Qwen3,
    on_gpu: bool,
    /// Floats per output row: `n_vocab`, or `n_embd` for a segment without the head.
    width: usize,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl DenseBackend {
    pub fn new(model: Qwen3, on_gpu: bool) -> Self {
        let width = model.out_width();
        Self {
            model,
            on_gpu,
            width,
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

    /// Reserves, then runs the step. `hidden` is the input of a segment without the embedding.
    fn run_step(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<Vec<Range<usize>>, BackendError> {
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
            return self
                .model
                .decode_hidden(&evals, hidden, &mut self.logits)
                .map_err(failed);
        }
        // One model call per group; `decode` rolls the earlier groups back on a failure.
        let n_embd = self.model.hparams().attn.n_embd as usize;
        let mut spans = Vec::with_capacity(inputs.len());
        let mut rows = Vec::new();
        let mut done = 0;
        for range in &groups {
            let n: usize = inputs[range.clone()].iter().map(|i| i.tokens.len()).sum();
            let h = hidden.map(|h| &h[done * n_embd..(done + n) * n_embd]);
            let s = self
                .model
                .decode_hidden(&evals[range.clone()], h, &mut rows)
                .map_err(failed)?;
            let base = self.logits.len() / self.width.max(1);
            spans.extend(s.into_iter().map(|r| r.start + base..r.end + base));
            self.logits.extend_from_slice(&rows);
            done += n;
        }
        Ok(spans)
    }

    /// One step with an input the caller took from the previous segment, if any.
    fn decode_feed(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<Outputs<'_>, BackendError> {
        self.spans.clear();
        self.logits.clear();
        if inputs.is_empty() {
            return Ok(Outputs::new(self.width, &self.logits, &self.spans));
        }
        // Every input's rows are reserved first. A failure, here or in any group of the step,
        // puts every input back to its length and reservation from before the call (a rollback
        // never grows a reservation, so it cannot fail for lack of rows).
        let before: Vec<(SeqId, usize, usize)> = inputs
            .iter()
            .map(|i| (i.seq, self.seq_len(i.seq), self.seq_reserved(i.seq)))
            .collect();
        match self.run_step(inputs, hidden) {
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
        Ok(Outputs::new(self.width, &self.logits, &self.spans))
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
        self.decode_feed(inputs, None)
    }

    /// Only a share of the layers is a segment; the whole model is served as before.
    fn as_segment(&mut self) -> Option<&mut dyn SegmentBackend> {
        let n_layer = self.model.hparams().attn.n_layer;
        if self.model.segment().is_whole(n_layer) {
            None
        } else {
            Some(self)
        }
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

impl SegmentBackend for DenseBackend {
    fn segment(&self) -> SegmentInfo {
        let s = self.model.segment();
        SegmentInfo {
            embed: s.embed,
            head: s.head,
            n_embd: self.model.hparams().attn.n_embd as usize,
            layers: (s.layers.start, s.layers.end),
        }
    }

    fn step(
        &mut self,
        inputs: &[SeqInput<'_>],
        feed: Feed<'_>,
    ) -> Result<StepOut<'_>, BackendError> {
        let hidden = match feed {
            Feed::Tokens => None,
            Feed::Hidden(h) => Some(h),
        };
        let head = self.model.segment().head;
        let out = self.decode_feed(inputs, hidden)?;
        Ok(if head {
            StepOut::Logits(out)
        } else {
            StepOut::Hidden(out)
        })
    }
}

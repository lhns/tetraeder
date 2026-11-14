//! The engines as seen by the scheduler: sequences in, logits out (ADR 0046, 0065).

use std::ops::Range;
use tetraeder_engine::{Engine, EngineError, SeqEval};
use tetraeder_ggml_sys::llama::LlamaError;
use tetraeder_model::{ModelError, Qwen3Moe, SeqEval as ModelSeqEval};

/// A sequence slot of a [`BatchBackend`] (ADR 0065).
pub type SeqId = u32;

/// One sequence's share of a step.
#[derive(Debug, Clone, Copy)]
pub struct SeqInput<'a> {
    pub seq: SeqId,
    /// Appended at positions `seq_len(seq)..` (never empty).
    pub tokens: &'a [i32],
    /// Logits of every token (speculative verification), else of the last one.
    pub all_logits: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BackendError {
    /// Not enough KV rows; nothing changed.
    #[error("kv full for sequence {seq}: needed {needed} rows, {free} free")]
    KvFull {
        seq: SeqId,
        needed: usize,
        free: usize,
    },
    /// The step failed; every sequence in it is back at its length before the step.
    #[error("{0}")]
    Failed(String),
}

/// Logits of the last decode, borrowed until the next call: `data` holds `n_vocab` floats per
/// row, `spans` the row range of each input in input order.
#[derive(Debug, Clone, Copy)]
pub struct Outputs<'a> {
    n_vocab: usize,
    data: &'a [f32],
    spans: &'a [Range<usize>],
}

impl<'a> Outputs<'a> {
    /// `spans[i]` are the row indices (not float offsets) of input `i` in `data`.
    pub fn new(n_vocab: usize, data: &'a [f32], spans: &'a [Range<usize>]) -> Self {
        debug_assert!(
            spans.iter().all(|s| s.end * n_vocab <= data.len()),
            "span beyond data"
        );
        Self {
            n_vocab,
            data,
            spans,
        }
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Number of inputs of the decode.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Rows of input `i`, position-major: one row, or `tokens.len()` rows with `all_logits`.
    pub fn rows(&self, i: usize) -> &'a [f32] {
        let s = &self.spans[i];
        &self.data[s.start * self.n_vocab..s.end * self.n_vocab]
    }

    /// The last row of input `i`.
    pub fn last(&self, i: usize) -> &'a [f32] {
        let rows = self.rows(i);
        &rows[rows.len() - self.n_vocab..]
    }
}

/// Many sequences over one KV pool, decoded together in one step (ADR 0065). Positions are
/// implicit: a sequence only grows at its end or is cut by `seq_rm`. A KV row belongs to at most
/// one sequence. Growing past `n_ctx_seq` is `KvFull` too. A `decode` that grows a sequence past its reservation reserves the missing
/// rows first and fails with `KvFull` (nothing changed) when they are not free.
pub trait BatchBackend {
    fn n_vocab(&self) -> usize;
    fn n_seq_max(&self) -> usize;
    /// Most positions one sequence may hold.
    fn n_ctx_seq(&self) -> usize;
    /// KV rows in the pool, and rows not reserved by any sequence.
    fn kv_capacity(&self) -> usize;
    fn kv_free(&self) -> usize;
    fn seq_len(&self, seq: SeqId) -> usize;
    /// Guarantees rows for `seq` to grow to `n` positions. KvFull changes nothing.
    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError>;
    /// Keeps the first `keep` positions; 0 frees the sequence and its reservation.
    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError>;
    /// `dst` (empty) becomes the first `n` positions of `src`, in rows of its own.
    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError>;
    /// One step: each input's tokens appended to its sequence (sequences distinct).
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError>;
    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError>;
    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError>;
    /// The KV cache format the snapshots are in, keying the prefix cache (ADR 0050 b).
    fn kv_format(&self) -> String {
        "f16/f16".into()
    }
    /// Whether steps run on a GPU, so wall time counts as device time.
    fn on_gpu(&self) -> bool;
    /// Weight bytes read from storage since load; `None` when the backend does not stream.
    fn bytes_streamed_total(&self) -> Option<u64>;
    /// Cumulative counters for the decode profile (`TETRAEDER_DECODE_PROFILE=1`), such as graph
    /// builds against reuses; the scheduler reports their change over a request's decode.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
    /// Most tokens one `decode` takes in total (libllama's `n_batch`); the scheduler keeps its
    /// step plan under it. Additive to ADR 0065.
    fn max_step_tokens(&self) -> usize {
        usize::MAX
    }
}

/// Tokens per step of the MoE builder; the scheduler plans under `max_step_tokens`, and
/// `decode` splits a larger step itself.
pub const MOE_CHUNK: usize = 512;

/// The libllama engine, loaded on CPU or GPU, as a [`BatchBackend`] over its unified KV pool
/// (ADR 0065). Reservations are bookkeeping against `n_ctx`: the pool finds cells one by one, so
/// a reservation under capacity cannot fail on fragmentation. `seq_cp` shares cells in libllama
/// but counts `n` rows for `dst`, which over-counts and never under-counts.
pub struct LlamaBackend {
    engine: Engine,
    on_gpu: bool,
    n_vocab: usize,
    /// Rows reserved per sequence (at least its length).
    reserved: Vec<usize>,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl LlamaBackend {
    pub fn new(engine: Engine, on_gpu: bool) -> Self {
        let n_vocab = usize::try_from(engine.model().n_vocab()).unwrap_or(0);
        let reserved = vec![0; engine.n_seq_max()];
        Self {
            engine,
            on_gpu,
            n_vocab,
            reserved,
            logits: Vec::new(),
            spans: Vec::new(),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    fn index(&self, seq: SeqId) -> Result<usize, BackendError> {
        let i = seq as usize;
        if i < self.reserved.len() {
            Ok(i)
        } else {
            Err(BackendError::Failed(format!(
                "sequence {seq} does not exist: the context holds {}",
                self.reserved.len()
            )))
        }
    }

    fn reserved_total(&self) -> usize {
        self.reserved.iter().sum()
    }

    fn kv_full(&self, seq: SeqId, needed: usize) -> BackendError {
        BackendError::KvFull {
            seq,
            needed,
            free: self.kv_free(),
        }
    }

    fn map_err(&self, seq: SeqId, needed: usize, e: EngineError) -> BackendError {
        match e {
            EngineError::Llama(LlamaError::NoKvSlot) => self.kv_full(seq, needed),
            e => BackendError::Failed(e.to_string()),
        }
    }
}

impl BatchBackend for LlamaBackend {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn n_seq_max(&self) -> usize {
        self.reserved.len()
    }

    fn n_ctx_seq(&self) -> usize {
        // Unified pool: one sequence may use all of it.
        self.engine.n_ctx()
    }

    fn kv_capacity(&self) -> usize {
        self.engine.n_ctx()
    }

    fn kv_free(&self) -> usize {
        self.kv_capacity().saturating_sub(self.reserved_total())
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        i32::try_from(seq).map_or(0, |s| self.engine.seq_len(s))
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        let i = self.index(seq)?;
        let need = n.saturating_sub(self.reserved[i]);
        if n > self.n_ctx_seq() || need > self.kv_free() {
            return Err(self.kv_full(seq, need));
        }
        self.reserved[i] = self.reserved[i].max(n);
        Ok(())
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        let i = self.index(seq)?;
        self.engine
            .seq_rm(seq as i32, keep)
            .map_err(|e| BackendError::Failed(e.to_string()))?;
        if keep == 0 {
            self.reserved[i] = 0;
        }
        Ok(())
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        let (s, d) = (self.index(src)?, self.index(dst)?);
        if s == d || self.seq_len(dst) != 0 || n > self.seq_len(src) {
            return Err(BackendError::Failed(format!(
                "seq_cp {src} -> {dst} of {n}: dst must be empty and distinct, n within src"
            )));
        }
        let need = n.saturating_sub(self.reserved[d]);
        if need > self.kv_free() {
            return Err(self.kv_full(dst, need));
        }
        self.engine
            .seq_cp(src as i32, dst as i32, n)
            .map_err(|e| BackendError::Failed(e.to_string()))?;
        self.reserved[d] = self.reserved[d].max(n);
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.spans.clear();
        self.logits.clear();
        // Validate everything before touching state.
        let mut seen: Vec<usize> = Vec::with_capacity(inputs.len());
        let mut need_total = 0;
        for inp in inputs {
            let i = self.index(inp.seq)?;
            if seen.contains(&i) {
                return Err(BackendError::Failed(format!("sequence {} twice", inp.seq)));
            }
            seen.push(i);
            if inp.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            let after = self.seq_len(inp.seq) + inp.tokens.len();
            if after > self.n_ctx_seq() {
                return Err(self.kv_full(inp.seq, after.saturating_sub(self.reserved[i])));
            }
            need_total += after.saturating_sub(self.reserved[i]);
        }
        if need_total > self.kv_free() {
            let first = inputs
                .iter()
                .find(|inp| {
                    self.seq_len(inp.seq) + inp.tokens.len() > self.reserved[inp.seq as usize]
                })
                .map_or(0, |inp| inp.seq);
            return Err(self.kv_full(first, need_total));
        }
        let before: Vec<(usize, usize)> = seen
            .iter()
            .map(|&i| (self.engine.seq_len(i as i32), self.reserved[i]))
            .collect();
        for (inp, &i) in inputs.iter().zip(&seen) {
            self.reserved[i] = self.reserved[i].max(self.seq_len(inp.seq) + inp.tokens.len());
        }
        if let Err(e) = self.run(inputs) {
            for (&i, &(len, res)) in seen.iter().zip(&before) {
                let _ = self.engine.seq_rm(i as i32, len);
                self.reserved[i] = res;
            }
            self.logits.clear();
            self.spans.clear();
            let total = inputs.iter().map(|i| i.tokens.len()).sum();
            return Err(self.map_err(inputs[0].seq, total, e));
        }
        Ok(Outputs::new(self.n_vocab, &self.logits, &self.spans))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.index(seq)?;
        self.engine
            .seq_snapshot(seq as i32)
            .map_err(|e| BackendError::Failed(e.to_string()))
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        let i = self.index(seq)?;
        let need = n.saturating_sub(self.reserved[i]);
        if n > self.n_ctx_seq() || need > self.kv_free() {
            return Err(self.kv_full(seq, need));
        }
        let before = self.reserved[i];
        self.reserved[i] = before.max(n);
        if let Err(e) = self.engine.seq_restore(seq as i32, bytes, n) {
            let _ = self.engine.seq_rm(seq as i32, 0);
            self.reserved[i] = before;
            return Err(self.map_err(seq, need, e));
        }
        Ok(())
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }

    /// libllama's perf counters: `n_reused` of `n_eval` decodes reused the previous graph.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        let p = self.engine.perf();
        vec![
            ("n_eval", f64::from(p.n_eval)),
            ("n_reused", f64::from(p.n_reused)),
            ("eval_s", p.eval_ms / 1e3),
        ]
    }

    fn max_step_tokens(&self) -> usize {
        self.engine.n_batch()
    }
}

impl LlamaBackend {
    /// Decodes `inputs` in as few engine calls as `n_batch` allows (one in the normal case: the
    /// scheduler plans under `max_step_tokens`) and collects the rows of every input.
    fn run(&mut self, inputs: &[SeqInput<'_>]) -> Result<(), EngineError> {
        let cap = self.engine.n_batch();
        let mut group: Vec<SeqEval<'_>> = Vec::new();
        // For each piece of the group: (input index, whether its rows are kept).
        let mut pieces: Vec<(usize, bool)> = Vec::new();
        let mut used = 0;
        let mut next_row = 0;
        let mut spans = Vec::with_capacity(inputs.len());
        for (n_in, inp) in inputs.iter().enumerate() {
            let rows = if inp.all_logits { inp.tokens.len() } else { 1 };
            spans.push(next_row..next_row + rows);
            next_row += rows;
            let mut rest = inp.tokens;
            while !rest.is_empty() {
                if used == cap {
                    self.flush(&mut group, &mut pieces)?;
                    used = 0;
                }
                let take = rest.len().min(cap - used);
                let (head, tail) = rest.split_at(take);
                let last = tail.is_empty();
                group.push(SeqEval {
                    seq: inp.seq as i32,
                    tokens: head,
                    all_logits: inp.all_logits,
                });
                // Rows are kept for all_logits inputs, and for the last piece otherwise.
                pieces.push((n_in, inp.all_logits || last));
                used += take;
                rest = tail;
            }
        }
        self.flush(&mut group, &mut pieces)?;
        debug_assert_eq!(self.logits.len(), next_row * self.n_vocab);
        self.spans = spans;
        Ok(())
    }

    fn flush(
        &mut self,
        group: &mut Vec<SeqEval<'_>>,
        pieces: &mut Vec<(usize, bool)>,
    ) -> Result<(), EngineError> {
        if group.is_empty() {
            return Ok(());
        }
        self.engine.decode_seqs(group)?;
        let mut k = 0;
        for (piece, &(_, keep)) in group.iter().zip(pieces.iter()) {
            let rows = if piece.all_logits {
                piece.tokens.len()
            } else {
                1
            };
            if keep {
                for r in 0..rows {
                    let row = self.engine.output_logits(k + r)?;
                    self.logits.extend_from_slice(row);
                }
            }
            k += rows;
        }
        group.clear();
        pieces.clear();
        Ok(())
    }
}

/// The Qwen3 MoE builder as a [`BatchBackend`] over its KV pool (ADR 0065, 0067). A step takes
/// at most `chunk` tokens in one graph; a larger `decode` runs as several.
pub struct MoeBackend {
    model: Qwen3Moe,
    on_gpu: bool,
    n_vocab: usize,
    logits: Vec<f32>,
    /// The rows of one model step, before the kept ones are copied to `logits`.
    step_logits: Vec<f32>,
    spans: Vec<Range<usize>>,
    chunk: usize,
}

impl MoeBackend {
    pub fn new(model: Qwen3Moe, on_gpu: bool) -> Self {
        Self::with_chunk(model, on_gpu, MOE_CHUNK)
    }

    /// Like `new`, with `chunk` tokens per step (the plan's ubatch).
    pub fn with_chunk(model: Qwen3Moe, on_gpu: bool, chunk: usize) -> Self {
        let n_vocab = model.n_vocab();
        Self {
            model,
            on_gpu,
            n_vocab,
            logits: Vec::new(),
            step_logits: Vec::new(),
            spans: Vec::new(),
            chunk: chunk.max(1),
        }
    }

    pub fn model(&self) -> &Qwen3Moe {
        &self.model
    }

    fn check_seq(&self, seq: SeqId) -> Result<(), BackendError> {
        if (seq as usize) < self.model.n_seq_max() {
            Ok(())
        } else {
            Err(BackendError::Failed(format!(
                "sequence {seq} does not exist: the pool holds {}",
                self.model.n_seq_max()
            )))
        }
    }

    fn kv_full(&self, seq: SeqId, needed: usize) -> BackendError {
        BackendError::KvFull {
            seq,
            needed,
            free: self.model.pool().free(),
        }
    }

    /// Rows for `seq` up to `n` positions: `KvFull` (nothing changed) when the pool has none.
    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.check_seq(seq)?;
        let have = self.model.pool().reserved(seq);
        if n <= have {
            return Ok(());
        }
        if n > self.model.pool().capacity() || !self.model.try_reserve(seq, n).map_err(failed)? {
            return Err(self.kv_full(seq, n - have));
        }
        Ok(())
    }

    /// Puts every sequence of a failed step back to its length and reservation before it.
    /// `seq_rm` shrinks to the positions held (rounded up to 256 rows) or frees an empty
    /// sequence; growing again takes the rows the shrink just released.
    fn restore(&mut self, saved: &[(SeqId, usize, usize)]) {
        for &(seq, len, reserved) in saved {
            if self.model.seq_len(seq) == len && self.model.pool().reserved(seq) == reserved {
                continue;
            }
            let _ = self.model.seq_rm(seq, len);
            if reserved > self.model.pool().reserved(seq) {
                let _ = self.model.try_reserve(seq, reserved);
            }
        }
    }

    /// Runs the pieces of `inputs`, at most `chunk` tokens per model step, and collects the rows
    /// of every input in `self.logits`.
    fn run(&mut self, inputs: &[SeqInput<'_>]) -> Result<(), BackendError> {
        let cap = self.chunk;
        let mut group: Vec<ModelSeqEval<'_>> = Vec::new();
        // Per piece of the group: whether its rows are kept.
        let mut keep: Vec<bool> = Vec::new();
        let mut used = 0;
        let mut next_row = 0;
        let mut spans = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let rows = if inp.all_logits { inp.tokens.len() } else { 1 };
            spans.push(next_row..next_row + rows);
            next_row += rows;
            let mut rest = inp.tokens;
            while !rest.is_empty() {
                if used == cap {
                    self.flush(&mut group, &mut keep)?;
                    used = 0;
                }
                let take = rest.len().min(cap - used);
                let (head, tail) = rest.split_at(take);
                group.push(ModelSeqEval {
                    seq: inp.seq,
                    tokens: head,
                    all_logits: inp.all_logits,
                });
                // Rows are kept for all_logits inputs, and for the last piece otherwise.
                keep.push(inp.all_logits || tail.is_empty());
                used += take;
                rest = tail;
            }
        }
        self.flush(&mut group, &mut keep)?;
        debug_assert_eq!(self.logits.len(), next_row * self.n_vocab);
        self.spans = spans;
        Ok(())
    }

    fn flush(
        &mut self,
        group: &mut Vec<ModelSeqEval<'_>>,
        keep: &mut Vec<bool>,
    ) -> Result<(), BackendError> {
        if group.is_empty() {
            return Ok(());
        }
        let mut buf = std::mem::take(&mut self.step_logits);
        let spans = self.model.decode(group, &mut buf).map_err(failed);
        let spans = match spans {
            Ok(s) => s,
            Err(e) => {
                self.step_logits = buf;
                return Err(e);
            }
        };
        for (r, &k) in spans.iter().zip(keep.iter()) {
            if k {
                self.logits
                    .extend_from_slice(&buf[r.start * self.n_vocab..r.end * self.n_vocab]);
            }
        }
        self.step_logits = buf;
        group.clear();
        keep.clear();
        Ok(())
    }
}

fn failed(e: ModelError) -> BackendError {
    BackendError::Failed(e.to_string())
}

impl BatchBackend for MoeBackend {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn n_seq_max(&self) -> usize {
        self.model.n_seq_max()
    }

    fn n_ctx_seq(&self) -> usize {
        // One sequence may use all of the pool.
        self.model.n_ctx()
    }

    fn kv_capacity(&self) -> usize {
        self.model.pool().capacity()
    }

    fn kv_free(&self) -> usize {
        self.model.pool().free()
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        self.model.seq_len(seq)
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.reserve(seq, n)
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        self.check_seq(seq)?;
        self.model.seq_rm(seq, keep).map_err(failed)
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        self.check_seq(src)?;
        self.check_seq(dst)?;
        if src == dst || self.model.seq_len(dst) != 0 || n > self.model.seq_len(src) {
            return Err(BackendError::Failed(format!(
                "seq_cp {src} -> {dst} of {n}: dst must be empty and distinct, n within src"
            )));
        }
        if !self.model.try_seq_cp(src, dst, n).map_err(failed)? {
            return Err(self.kv_full(dst, n));
        }
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.spans.clear();
        self.logits.clear();
        // Validate everything and reserve before the step. A failure, here or in the step, gives
        // back what this call reserved, so lengths and free rows are as they were.
        let mut saved: Vec<(SeqId, usize, usize)> = Vec::with_capacity(inputs.len());
        for inp in inputs {
            self.check_seq(inp.seq)?;
            if saved.iter().any(|s| s.0 == inp.seq) {
                return Err(BackendError::Failed(format!("sequence {} twice", inp.seq)));
            }
            if inp.tokens.is_empty() {
                return Err(BackendError::Failed("no tokens".into()));
            }
            saved.push((
                inp.seq,
                self.model.seq_len(inp.seq),
                self.model.pool().reserved(inp.seq),
            ));
        }
        let mut result = Ok(());
        for (inp, &(_, len, _)) in inputs.iter().zip(&saved) {
            result = self.reserve(inp.seq, len + inp.tokens.len());
            if result.is_err() {
                break;
            }
        }
        if result.is_ok() {
            result = self.run(inputs);
        }
        if let Err(e) = result {
            self.restore(&saved);
            self.logits.clear();
            self.spans.clear();
            return Err(e);
        }
        Ok(Outputs::new(self.n_vocab, &self.logits, &self.spans))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.check_seq(seq)?;
        self.model.seq_snapshot(seq).map_err(failed)
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        self.check_seq(seq)?;
        if n > self.model.n_ctx() {
            return Err(self.kv_full(seq, n));
        }
        if !self.model.try_seq_restore(seq, bytes, n).map_err(failed)? {
            return Err(self.kv_full(seq, n));
        }
        Ok(())
    }

    fn kv_format(&self) -> String {
        self.model.kv_format().to_string()
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.model.expert_stats().map(|s| s.io.bytes_read)
    }

    /// The builder's step profile (host timers) and whole-step graph builds.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        let p = self.model.profile();
        vec![
            ("steps", p.steps as f64),
            ("graphs", p.graphs as f64),
            ("step_graphs_built", self.model.step_graphs_built() as f64),
            ("inputs_s", p.inputs.as_secs_f64()),
            ("build_s", p.build.as_secs_f64()),
            ("alloc_s", p.alloc.as_secs_f64()),
            ("compute_s", p.compute.as_secs_f64()),
            ("readback_s", p.readback.as_secs_f64()),
            ("streamer_s", p.streamer.as_secs_f64()),
            ("total_s", p.total.as_secs_f64()),
        ]
    }

    fn max_step_tokens(&self) -> usize {
        self.chunk
    }
}

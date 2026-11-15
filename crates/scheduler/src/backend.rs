//! The engines as seen by the scheduler: tokens in, last-position logits out (ADR 0046).

use std::ops::Range;
use tetraeder_engine::{Engine, EngineError, SeqEval};
use tetraeder_ggml_sys::llama::LlamaError;
use tetraeder_model::Qwen3Moe;

/// Positions are token counts of the single cached sequence.
pub trait Backend {
    /// Context length in tokens.
    fn n_ctx(&self) -> usize;
    /// Positions currently cached.
    fn n_past(&self) -> usize;
    /// Keeps the first `keep` cached positions.
    fn truncate(&mut self, keep: usize) -> Result<(), String>;
    /// Appends `tokens` and returns the logits of the last one (`n_vocab` values, valid until
    /// the next call). Prefill is chunked inside.
    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String>;
    /// The cached positions as host bytes, for the prefix cache (ADR 0049).
    fn snapshot(&mut self) -> Result<Vec<u8>, String>;
    /// Replaces the cache with a [`Backend::snapshot`] of `n` positions.
    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String>;
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
}

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
    /// Cumulative counters for the decode profile, as [`Backend::counters`].
    fn counters(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
    /// KV bytes per position (0 when unknown), to size `seq_cp` in the transfer metric.
    fn kv_row_bytes(&self) -> usize {
        0
    }
    /// Most tokens one `decode` takes in total (libllama's `n_batch`); the scheduler keeps its
    /// step plan under it. Additive to ADR 0065.
    fn max_step_tokens(&self) -> usize {
        usize::MAX
    }
}

/// Inputs shorter than this are decoders (1 token, or 1 + drafts); longer ones are prompt chunks.
/// It is ggml's CPU flash-attention query tile (`GGML_FA_TILE_Q`).
pub(crate) const SHORT_INPUT: usize = 64;

/// The input ranges a step runs as, in input order. On the CPU, decoders and prompt chunks go in
/// separate model calls: in one call, a query tile holding both runs every decoder's history
/// through the tiled attention kernel (mixing 3 decoders into a 402-token chunk took 130 ms more
/// than running them apart on Qwen3-0.6B, loop-bench step shapes). On a GPU one call.
pub(crate) fn step_groups(inputs: &[SeqInput<'_>], on_gpu: bool) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for (i, inp) in inputs.iter().enumerate() {
        let short = inp.tokens.len() < SHORT_INPUT;
        match out.last_mut() {
            Some(r) if on_gpu || (inputs[r.start].tokens.len() < SHORT_INPUT) == short => {
                r.end = i + 1;
            }
            _ => out.push(i..i + 1),
        }
    }
    out
}

/// [`BatchBackend`] over the one-sequence [`Backend`] (until C4 implements the trait natively).
/// Only sequence 0 exists. The pool is the context: `n_ctx` rows, reserved by sequence 0.
/// `seq_cp` always fails: a copy needs a second, empty sequence and there is none.
pub struct SingleSeq<B: Backend> {
    inner: B,
    n_vocab: usize,
    reserved: usize,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl<B: Backend> SingleSeq<B> {
    /// `n_vocab` is the length of the logits `inner.eval` returns.
    pub fn new(inner: B, n_vocab: usize) -> Self {
        Self {
            inner,
            n_vocab,
            reserved: 0,
            logits: Vec::new(),
            spans: Vec::new(),
        }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    pub fn into_inner(self) -> B {
        self.inner
    }

    fn check_seq(seq: SeqId) -> Result<(), BackendError> {
        if seq == 0 {
            Ok(())
        } else {
            Err(BackendError::Failed(format!(
                "sequence {seq} does not exist: single-sequence backend"
            )))
        }
    }
}

impl<B: Backend> BatchBackend for SingleSeq<B> {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn n_seq_max(&self) -> usize {
        1
    }

    fn n_ctx_seq(&self) -> usize {
        self.inner.n_ctx()
    }

    fn kv_capacity(&self) -> usize {
        self.inner.n_ctx()
    }

    fn kv_free(&self) -> usize {
        self.inner.n_ctx().saturating_sub(self.reserved)
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        if seq == 0 { self.inner.n_past() } else { 0 }
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        Self::check_seq(seq)?;
        if n > self.inner.n_ctx() {
            return Err(BackendError::KvFull {
                seq,
                needed: n,
                free: self.kv_free(),
            });
        }
        self.reserved = self.reserved.max(n);
        Ok(())
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        Self::check_seq(seq)?;
        self.inner
            .truncate(keep.min(self.inner.n_past()))
            .map_err(BackendError::Failed)?;
        if keep == 0 {
            self.reserved = 0;
        }
        Ok(())
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, _n: usize) -> Result<(), BackendError> {
        Self::check_seq(src)?;
        Self::check_seq(dst)?;
        Err(BackendError::Failed(
            "seq_cp needs two sequences: single-sequence backend".into(),
        ))
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        if inputs.len() > 1 {
            return Err(BackendError::Failed(
                "single-sequence backend: one input per step".into(),
            ));
        }
        self.spans.clear();
        self.logits.clear();
        let Some(input) = inputs.first() else {
            return Ok(Outputs::new(self.n_vocab, &self.logits, &self.spans));
        };
        Self::check_seq(input.seq)?;
        if input.tokens.is_empty() {
            return Err(BackendError::Failed("no tokens".into()));
        }
        let before = self.inner.n_past();
        let after = before + input.tokens.len();
        if after > self.inner.n_ctx() {
            return Err(BackendError::KvFull {
                seq: input.seq,
                needed: after - self.reserved.max(before),
                free: self.kv_free(),
            });
        }
        let reserved_before = self.reserved;
        self.reserved = self.reserved.max(after);
        let run = (|| {
            if input.all_logits {
                // The old trait returns the last row only: one token at a time.
                for t in input.tokens {
                    let row = self.inner.eval(std::slice::from_ref(t))?;
                    self.logits.extend_from_slice(row);
                }
            } else {
                let row = self.inner.eval(input.tokens)?;
                self.logits.extend_from_slice(row);
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = run {
            // Back to the length before the step.
            let _ = self.inner.truncate(before);
            self.reserved = reserved_before;
            self.logits.clear();
            return Err(BackendError::Failed(e));
        }
        let rows = self.logits.len() / self.n_vocab.max(1);
        self.spans.push(0..rows);
        Ok(Outputs::new(self.n_vocab, &self.logits, &self.spans))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        Self::check_seq(seq)?;
        self.inner.snapshot().map_err(BackendError::Failed)
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        Self::check_seq(seq)?;
        if n > self.inner.n_ctx() {
            return Err(BackendError::KvFull {
                seq,
                needed: n,
                free: self.kv_free(),
            });
        }
        self.inner.restore(bytes, n).map_err(BackendError::Failed)?;
        self.reserved = self.reserved.max(n);
        Ok(())
    }

    fn kv_format(&self) -> String {
        self.inner.kv_format()
    }

    fn on_gpu(&self) -> bool {
        self.inner.on_gpu()
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.inner.bytes_streamed_total()
    }

    fn counters(&self) -> Vec<(&'static str, f64)> {
        self.inner.counters()
    }
}

/// Tokens per `eval_last` call of the MoE builder; the scheduler also chunks to stay
/// cancellable, this keeps the backend safe on its own.
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
    /// KV bytes per position, learned from the last snapshot (0 before the first).
    row_bytes: usize,
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
            row_bytes: 0,
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
        let bytes = self
            .engine
            .seq_snapshot(seq as i32)
            .map_err(|e| BackendError::Failed(e.to_string()))?;
        if let Some(per_row) = bytes.len().checked_div(self.seq_len(seq)) {
            self.row_bytes = per_row;
        }
        Ok(bytes)
    }

    fn kv_row_bytes(&self) -> usize {
        self.row_bytes
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
        let starts: Vec<usize> = step_groups(inputs, self.on_gpu)
            .into_iter()
            .map(|r| r.start)
            .collect();
        for (n_in, inp) in inputs.iter().enumerate() {
            if n_in > 0 && starts.contains(&n_in) {
                self.flush(&mut group, &mut pieces)?;
                used = 0;
            }
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

/// The Qwen3 MoE builder.
pub struct MoeBackend {
    model: Qwen3Moe,
    on_gpu: bool,
    logits: Vec<f32>,
    chunk: usize,
}

impl MoeBackend {
    pub fn new(model: Qwen3Moe, on_gpu: bool) -> Self {
        Self::with_chunk(model, on_gpu, MOE_CHUNK)
    }

    /// Like `new`, with `chunk` tokens per `eval_last` call (the plan's ubatch).
    pub fn with_chunk(model: Qwen3Moe, on_gpu: bool, chunk: usize) -> Self {
        Self {
            model,
            on_gpu,
            logits: Vec::new(),
            chunk: chunk.max(1),
        }
    }

    pub fn model(&self) -> &Qwen3Moe {
        &self.model
    }
}

impl Backend for MoeBackend {
    fn n_ctx(&self) -> usize {
        self.model.n_ctx()
    }

    fn n_past(&self) -> usize {
        self.model.n_past()
    }

    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        let keep = keep.min(self.model.n_past());
        self.model.rewind(keep).map_err(|e| e.to_string())
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        if tokens.is_empty() {
            return Err("no tokens".into());
        }
        for chunk in tokens.chunks(self.chunk) {
            // Into the kept buffer: no 600 KB allocation per decode token.
            self.model
                .eval_last_into(chunk, &mut self.logits)
                .map_err(|e| e.to_string())?;
        }
        Ok(&self.logits)
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.model.kv_snapshot().map_err(|e| e.to_string())
    }

    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.model.kv_restore(bytes, n).map_err(|e| e.to_string())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups(lens: &[usize], on_gpu: bool) -> Vec<Range<usize>> {
        let toks = vec![0; lens.iter().copied().max().unwrap_or(0)];
        let inputs: Vec<SeqInput<'_>> = lens
            .iter()
            .enumerate()
            .map(|(i, &n)| SeqInput {
                seq: i as SeqId,
                tokens: &toks[..n],
                all_logits: false,
            })
            .collect();
        step_groups(&inputs, on_gpu)
    }

    #[test]
    fn cpu_steps_run_decoders_and_prompt_chunks_apart() {
        assert_eq!(groups(&[1, 3, 1, 402], false), vec![0..3, 3..4]);
        assert_eq!(groups(&[1, 300, 200], false), vec![0..1, 1..3]);
        assert_eq!(groups(&[1, 1, 1, 1], false), vec![0..4]);
        assert_eq!(groups(&[402], false), vec![0..1]);
        assert_eq!(groups(&[63, 64], false), vec![0..1, 1..2]);
        assert!(groups(&[], false).is_empty());
        // One call on a GPU.
        assert_eq!(groups(&[1, 1, 402], true), vec![0..3]);
    }
}

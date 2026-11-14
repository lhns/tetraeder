//! The engines as seen by the scheduler: tokens in, last-position logits out (ADR 0046).

use std::ops::Range;
use tetraeder_engine::Engine;
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

/// The libllama engine, loaded on CPU or GPU.
pub struct LlamaBackend {
    engine: Engine,
    on_gpu: bool,
}

impl LlamaBackend {
    pub fn new(engine: Engine, on_gpu: bool) -> Self {
        Self { engine, on_gpu }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

impl Backend for LlamaBackend {
    fn n_ctx(&self) -> usize {
        self.engine.n_ctx()
    }

    fn n_past(&self) -> usize {
        self.engine.n_past()
    }

    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        self.engine.truncate(keep).map_err(|e| e.to_string())
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        self.engine.eval(tokens).map_err(|e| e.to_string())
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.engine
            .snapshot()
            .map(|(bytes, _)| bytes)
            .map_err(|e| e.to_string())
    }

    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.engine.restore(bytes, n).map_err(|e| e.to_string())
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

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
    /// Rows currently reserved for `seq` (at least its length, 0 when it has none). The scheduler
    /// reads it after a restore or a cut, which may change the reservation. The default is the
    /// length, which makes the scheduler reserve again before the sequence grows.
    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.seq_len(seq)
    }
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
    /// Whether `seq` held rows that the backend lost (a restarted worker process) and has not
    /// been freed since. Its contents are gone and using it fails; `seq_rm(seq, 0)` clears the
    /// mark. The scheduler drops idle prefixes on lost sequences. Default: nothing is ever lost.
    fn seq_lost(&self, _seq: SeqId) -> bool {
        false
    }
    /// Whether `seq_rm(seq, keep)` with `keep > 0` can cut a suffix. A recurrent or hybrid model
    /// keeps one state per sequence and cannot, so the scheduler reuses only a prefix it keeps
    /// whole and otherwise clears the sequence and prefills the prompt.
    fn can_cut_suffix(&self) -> bool {
        true
    }
    /// The KV cache format the snapshots are in, keying the prefix cache (ADR 0050 b).
    fn kv_format(&self) -> String {
        "f16/f16".into()
    }
    /// Whether [`BatchBackend::seq_shift`] works (`--context-shift`, ADR 0090): a cache that can
    /// drop a middle range and move the rest down. Default: no.
    fn can_shift(&self) -> bool {
        false
    }
    /// The context shift: drops positions `keep..keep + discard` of `seq` and moves the ones
    /// after them down by `discard`; the sequence is `discard` shorter. The reservation stays.
    fn seq_shift(
        &mut self,
        _seq: SeqId,
        _keep: usize,
        _discard: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::Failed(
            "this backend cannot shift a context".into(),
        ))
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
    /// The expert caches' counters since load, cumulative, for the judged round (ADR 0092): per
    /// layer of the model the compute tier's hits, the experts computed on the CPU, the layer
    /// steps with every routed expert resident and those without, and the RAM tier's hits and
    /// misses; the bytes read from storage; the slots per layer of each tier and the experts
    /// per layer. `None` when the backend has no expert cache. The caller takes the difference
    /// of two readings over the window it times.
    fn expert_stats_json(&self) -> Option<serde_json::Value> {
        None
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
    /// This backend as a segment of the layers (ADR 0091), when it is one. The worker serves
    /// hidden rows through it; a backend that holds the whole model is none.
    fn as_segment(&mut self) -> Option<&mut dyn SegmentBackend> {
        None
    }
}

/// What a segment's backend holds (ADR 0091, step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentInfo {
    /// Holds the embedding: takes tokens, else hidden rows.
    pub embed: bool,
    /// Holds the head: returns logits, else hidden rows.
    pub head: bool,
    /// Floats of one hidden row.
    pub n_embd: usize,
    /// Layers computed, `[start, end)`.
    pub layers: (u32, u32),
}

/// What a step of a segment consumes: the tokens (it holds the embedding), or the hidden rows of
/// every token of every input, `n_embd` F32 each, in input order.
#[derive(Debug, Clone, Copy)]
pub enum Feed<'a> {
    Tokens,
    Hidden(&'a [f32]),
}

/// What a step of a segment returns. `Hidden` holds one row of `n_embd` floats per token of
/// every input (whatever `all_logits` says), in `Outputs` whose row width is `n_embd`.
#[derive(Debug, Clone, Copy)]
pub enum StepOut<'a> {
    Logits(Outputs<'a>),
    Hidden(Outputs<'a>),
}

impl<'a> StepOut<'a> {
    /// The rows, logits or hidden.
    pub fn rows(&self) -> Outputs<'a> {
        match *self {
            Self::Logits(o) | Self::Hidden(o) => o,
        }
    }
}

/// A backend that runs part of the layers. [`BatchBackend::decode`] is `step` with
/// `Feed::Tokens` on a segment that holds the embedding and the head; every other call of the
/// trait works on the segment's own KV (its layers only), so a snapshot is that segment's share.
pub trait SegmentBackend: BatchBackend {
    fn segment(&self) -> SegmentInfo;
    /// One step with the contract of [`BatchBackend::decode`] (sequences appended, rolled back
    /// to their length and reservation on a failure).
    fn step(
        &mut self,
        inputs: &[SeqInput<'_>],
        feed: Feed<'_>,
    ) -> Result<StepOut<'_>, BackendError>;
    /// The expert-cache counters of the segment's backend (see
    /// [`BatchBackend::expert_stats_json`]); a remote segment asks its worker. `None` when it has
    /// none or the worker cannot answer.
    fn expert_stats(&mut self) -> Option<serde_json::Value> {
        self.expert_stats_json()
    }
    /// The time the segment itself spent on the last step, apart from any network, when it
    /// knows (a remote segment reports its worker's time).
    fn last_step_time(&self) -> Option<std::time::Duration> {
        None
    }
    /// Whether the segment computes a step apart from the caller (a worker on another node), so
    /// [`start_step`](Self::start_step) returns before the step is done and a pipelined prefill
    /// (ADR 0077) can overlap segments. Default: no, `step` is the only call.
    fn can_overlap(&self) -> bool {
        false
    }
    /// Sends a step and returns. Up to a few steps may be started before the first is finished;
    /// each [`finish_step`](Self::finish_step) takes the oldest, in order, and the segment runs
    /// them in that order. No other call to the segment between a start and its finish.
    fn start_step(
        &mut self,
        _inputs: &[SeqInput<'_>],
        _feed: Feed<'_>,
    ) -> Result<(), BackendError> {
        Err(BackendError::Failed(
            "this segment cannot start a step apart from its finish".into(),
        ))
    }
    /// Whether the oldest started step has its result, waiting up to `wait` for it (a worker's
    /// reply, read off the link). A step that cannot come (the worker is gone, its deadline is
    /// over) counts as ready: [`finish_step`](Self::finish_step) then returns the error. Lets the
    /// caller serve whichever segment answers first. Default: ready.
    fn reply_ready(&mut self, _wait: std::time::Duration) -> bool {
        true
    }
    /// The result of the oldest started step, as `step` returns it.
    fn finish_step(&mut self) -> Result<StepOut<'_>, BackendError> {
        Err(BackendError::Failed("no step was started".into()))
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

/// Tokens per model step of the MoE builder; `decode` splits a larger scheduler step itself.
pub const MOE_CHUNK: usize = 512;

/// The sequence a failed step names in `KvFull`: libllama cannot say which one found no cells,
/// so the one that grew the most past its reservation (the first on a tie). `growth` is
/// (sequence, rows beyond its reservation) per input, never empty.
fn neediest(growth: &[(SeqId, usize)]) -> SeqId {
    growth
        .iter()
        .rev()
        .max_by_key(|&&(_, over)| over)
        .map_or(0, |&(seq, _)| seq)
}

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
    /// False for a recurrent or hybrid model, whose memory cannot drop a suffix.
    can_cut: bool,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
}

impl LlamaBackend {
    pub fn new(engine: Engine, on_gpu: bool) -> Self {
        let n_vocab = usize::try_from(engine.model().n_vocab()).unwrap_or(0);
        let reserved = vec![0; engine.n_seq_max()];
        let can_cut = !(engine.model().is_recurrent() || engine.model().is_hybrid());
        Self {
            engine,
            on_gpu,
            n_vocab,
            reserved,
            row_bytes: 0,
            can_cut,
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

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.reserved.get(seq as usize).copied().unwrap_or(0)
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

    fn can_cut_suffix(&self) -> bool {
        self.can_cut
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
                // Unconditional: cells a failed step wrote past `len` must go even when the
                // length says nothing is there.
                let _ = self.engine.seq_truncate_cells(i as i32, len);
                self.reserved[i] = res;
            }
            self.logits.clear();
            self.spans.clear();
            let total = inputs.iter().map(|i| i.tokens.len()).sum();
            let growth: Vec<(SeqId, usize)> = inputs
                .iter()
                .zip(&before)
                .map(|(inp, &(len, res))| (inp.seq, (len + inp.tokens.len()).saturating_sub(res)))
                .collect();
            return Err(self.map_err(neediest(&growth), total, e));
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

    fn can_shift(&self) -> bool {
        self.can_cut && self.engine.can_shift()
    }

    fn seq_shift(&mut self, seq: SeqId, keep: usize, discard: usize) -> Result<(), BackendError> {
        let _ = self.index(seq)?;
        self.engine
            .seq_shift(seq as i32, keep, discard)
            .map_err(|e| BackendError::Failed(e.to_string()))
    }

    /// The context's real K and V types: they decide the bytes of a snapshot, which the prefix
    /// cache keys by this string.
    fn kv_format(&self) -> String {
        let (k, v) = self.engine.kv_types();
        format!("{}/{}", k.name(), v.name())
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

/// The Qwen3 MoE builder as a [`BatchBackend`] over its KV pool (ADR 0065, 0067). A step takes
/// at most `chunk` tokens in one graph; a larger `decode` runs as several. On the CPU decoders
/// and prompt chunks of one `decode` run apart (`step_groups`), like the other backends.
pub struct MoeBackend {
    model: Qwen3Moe,
    on_gpu: bool,
    n_vocab: usize,
    /// Floats per output row: `n_vocab`, or `n_embd` for a segment without the head.
    width: usize,
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
        let width = model.out_width();
        Self {
            model,
            on_gpu,
            n_vocab,
            width,
            logits: Vec::new(),
            step_logits: Vec::new(),
            spans: Vec::new(),
            chunk: chunk.max(1),
        }
    }

    pub fn model(&self) -> &Qwen3Moe {
        &self.model
    }

    /// The builder on the executor's ranks of `topo` (ADR 0069, 0070): every expert resident,
    /// one rank or many. `chunk` tokens per step as in [`MoeBackend::with_chunk`].
    pub fn with_topology(
        path: &std::path::Path,
        topo: &tetraeder_plan::Topology,
        opts: &tetraeder_model::LoadOptions,
        chunk: usize,
    ) -> Result<Self, ModelError> {
        let model = Qwen3Moe::load_sharded(path, topo, opts)?;
        let on_gpu = topo.ranks.iter().any(|r| r.gpu_index.is_some());
        Ok(Self::with_chunk(model, on_gpu, chunk))
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

    /// Puts every sequence of a failed step back to its length and reservation before it. A
    /// rollback never grows a reservation, so it cannot fail for lack of rows; a sequence the
    /// step moved stays where it moved.
    fn restore(&mut self, saved: &[(SeqId, usize, usize)]) {
        for &(seq, len, reserved) in saved.iter().rev() {
            if let Err(e) = self.model.rollback(seq, len, reserved) {
                tracing::warn!("cannot roll back sequence {seq}: {e}");
            }
        }
    }

    /// Runs `inputs` as `step_groups` splits them (on the CPU decoders and prompt chunks apart),
    /// and collects the rows of every input in `self.logits`, `self.spans` in input order.
    fn run(&mut self, inputs: &[SeqInput<'_>], hidden: Option<&[f32]>) -> Result<(), BackendError> {
        self.spans.clear();
        let n_embd = self.model.hparams().n_embd as usize;
        let mut done = 0;
        for range in step_groups(inputs, self.on_gpu) {
            let n: usize = inputs[range.clone()].iter().map(|i| i.tokens.len()).sum();
            let rows = hidden.map(|h| &h[done * n_embd..(done + n) * n_embd]);
            self.run_group(&inputs[range], rows)?;
            done += n;
        }
        Ok(())
    }

    /// Runs the pieces of `inputs`, at most `chunk` tokens per model step, appends the rows of
    /// every input to `self.logits` and their spans to `self.spans`.
    fn run_group(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        let cap = self.chunk;
        let n_embd = self.model.hparams().n_embd as usize;
        // A segment without the head returns every row: the next segment takes them all.
        let every = !self.model.segment().head;
        // The hidden rows of the tokens `done..done + used`.
        let rows =
            |done: usize, used: usize| hidden.map(|h| &h[done * n_embd..(done + used) * n_embd]);
        let mut done = 0;
        // Only a call with a prompt chunk builds the CPU hot set: a long input, or several
        // tokens without logits for every row. A group of decoders is not a prefill however many
        // sequences it holds, and neither is a verify step (a pending token and drafts, with the
        // logits of every row).
        let prefill = inputs
            .iter()
            .any(|i| i.tokens.len() >= SHORT_INPUT || (i.tokens.len() > 1 && !i.all_logits));
        let mut group: Vec<ModelSeqEval<'_>> = Vec::new();
        // Per piece of the group: whether its rows are kept.
        let mut keep: Vec<bool> = Vec::new();
        let mut used = 0;
        let mut next_row = self.logits.len() / self.width.max(1);
        let mut spans = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let all = inp.all_logits || every;
            let n_rows = if all { inp.tokens.len() } else { 1 };
            spans.push(next_row..next_row + n_rows);
            next_row += n_rows;
            let mut rest = inp.tokens;
            while !rest.is_empty() {
                if used == cap {
                    self.flush(&mut group, &mut keep, prefill, rows(done, used))?;
                    done += used;
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
                keep.push(all || tail.is_empty());
                used += take;
                rest = tail;
            }
        }
        self.flush(&mut group, &mut keep, prefill, rows(done, used))?;
        debug_assert_eq!(self.logits.len(), next_row * self.width);
        self.spans.extend(spans);
        Ok(())
    }

    /// One step: `hidden` is the input of a segment without the embedding.
    fn decode_feed(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<Outputs<'_>, BackendError> {
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
            result = self.run(inputs, hidden);
        }
        if let Err(e) = result {
            self.restore(&saved);
            self.logits.clear();
            self.spans.clear();
            return Err(e);
        }
        Ok(Outputs::new(self.width, &self.logits, &self.spans))
    }

    fn flush(
        &mut self,
        group: &mut Vec<ModelSeqEval<'_>>,
        keep: &mut Vec<bool>,
        prefill: bool,
        hidden: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        if group.is_empty() {
            return Ok(());
        }
        let mut buf = std::mem::take(&mut self.step_logits);
        let spans = self
            .model
            .decode_group_hidden(group, hidden, &mut buf, prefill)
            .map_err(failed);
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
                    .extend_from_slice(&buf[r.start * self.width..r.end * self.width]);
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

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.model.pool().reserved(seq)
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
        self.decode_feed(inputs, None)
    }

    /// Only a share of the layers is a segment; the whole model is served as before.
    fn as_segment(&mut self) -> Option<&mut dyn SegmentBackend> {
        let n_layer = self.model.hparams().n_layer;
        if self.model.segment().is_whole(n_layer) {
            None
        } else {
            Some(self)
        }
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

    fn expert_stats_json(&self) -> Option<serde_json::Value> {
        let st = self.model.expert_stats()?;
        let (vram, ram) = self.model.tier_slots()?;
        let layers: Vec<serde_json::Value> = st
            .layers
            .iter()
            .map(|l| {
                let ram = l.ram.unwrap_or_default();
                serde_json::json!({
                    "compute_hits": l.compute.hits,
                    "cpu": l.cpu,
                    "full_steps": l.full_steps,
                    "partial_steps": l.partial_steps,
                    "ram_hits": ram.hits,
                    "ram_misses": ram.misses,
                    "has_ram": l.ram.is_some(),
                })
            })
            .collect();
        Some(serde_json::json!({
            "n_expert": self.model.hparams().n_expert,
            "vram_slots": vram,
            "ram_slots": ram,
            "bytes_read": st.io.bytes_read,
            "layers": layers,
        }))
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

    fn kv_row_bytes(&self) -> usize {
        self.model.kv_row_bytes()
    }

    /// No cap: `decode` runs a larger step as model steps of `chunk` tokens, so the scheduler's
    /// budget (not the chunk) decides how many waiting prompts share a step.
    fn max_step_tokens(&self) -> usize {
        usize::MAX
    }
}

impl SegmentBackend for MoeBackend {
    fn segment(&self) -> SegmentInfo {
        let s = self.model.segment();
        SegmentInfo {
            embed: s.embed,
            head: s.head,
            n_embd: self.model.hparams().n_embd as usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_step_names_the_sequence_that_grew_most_past_its_reservation() {
        // Sequence 3 comes first but fits its reservation; 5 and 7 outgrow theirs.
        assert_eq!(neediest(&[(3, 0), (5, 40), (7, 90)]), 7);
        // Ties go to the first.
        assert_eq!(neediest(&[(3, 0), (5, 40), (7, 40)]), 5);
        assert_eq!(neediest(&[(4, 0), (2, 0)]), 4);
    }

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

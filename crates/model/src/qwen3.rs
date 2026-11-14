//! Qwen3 dense graph builder (ADR 0066), op for op like libllama's `qwen3` graph, on the shared
//! attention module and `kvcache`'s pool.
//!
//! Every step is one graph over the sequences of the step (ADR 0036): embedding, per layer
//! `attn::attention` (shared), the FFN `down(swiglu_split(gate(x), up(x)))`, output norm and
//! head. At the last layer the rows whose logits are wanted are selected after attention, as
//! libllama does, so the FFN and the head run on those rows only.
//!
//! The KV cache is one pool per layer ([`KvPool`] over the chunked layout, ADR 0067): the pool
//! plans positions, destination rows, `n_kv` and the mask of a step, this module runs it.

use crate::attn::{
    AttnHparams, AttnStep, AttnWeights, KV_PAD, Kv, SeqEval, alloc_kv, attention, copy_rows,
    rms_norm_mul,
};
use crate::cpu_poll;
use crate::gguf::{Gguf, GgufError};
use crate::hparams::names;
use crate::qwen3moe::{BackendChoice, MemoryUse, ModelError, Result};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tetraeder_ggml_sys::ggml::{
    Backend, Context, Graph, OwnedGraph, OwnedTensor, Sched, Tensor, Type,
};
use tetraeder_kvcache::{Chunked, KvError, KvPool, RowCopy, StepInput};
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_plan::KvFormat;
use tetraeder_telemetry::{Phase, StepSample, Telemetry};

/// `general.architecture` of a Qwen3 dense model.
pub const ARCH_DENSE: &str = "qwen3";

/// Upper bound of nodes per layer in a whole-step graph (about 45 are used).
const STEP_NODES_PER_LAYER: usize = 128;

/// Hyperparameters of a `qwen3` model (keys as llama.cpp writes them).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseHparams {
    pub attn: AttnHparams,
    pub n_ff: u32,
    /// True when the model has no `output.weight` and reuses the token embedding (Qwen3-0.6B).
    pub tied_output: bool,
    pub name: String,
}

impl DenseHparams {
    pub fn from_gguf(g: &Gguf) -> std::result::Result<Self, GgufError> {
        let arch = g.str("general.architecture")?;
        if arch != ARCH_DENSE {
            return Err(GgufError::Corrupt(format!(
                "architecture `{arch}`, expected `{ARCH_DENSE}`"
            )));
        }
        let k = |s: &str| format!("{ARCH_DENSE}.{s}");
        let n_head = g.u64(&k("attention.head_count"))? as u32;
        let n_embd = g.u64(&k("embedding_length"))? as u32;
        let head_dim =
            g.u64_or(&k("attention.key_length"), (n_embd / n_head.max(1)) as u64)? as u32;
        let value_len = g.u64_or(&k("attention.value_length"), head_dim as u64)? as u32;
        if value_len != head_dim {
            return Err(GgufError::Corrupt("key and value lengths differ".into()));
        }
        if g.get(&k("rope.scaling.type"))
            .is_some_and(|v| !matches!(v, crate::gguf::Value::Str(s) if s == "none"))
        {
            return Err(GgufError::Corrupt("rope scaling is not supported".into()));
        }
        let tok = g.tensor(names::TOK_EMBD)?;
        Ok(DenseHparams {
            attn: AttnHparams {
                n_layer: g.u64(&k("block_count"))? as u32,
                n_embd,
                n_head,
                n_head_kv: g.u64(&k("attention.head_count_kv"))? as u32,
                head_dim,
                n_rot: g.u64_or(&k("rope.dimension_count"), head_dim as u64)? as u32,
                n_ctx_train: g.u64(&k("context_length"))? as u32,
                n_vocab: tok.ne[1] as u32,
                rms_eps: g.f32(&k("attention.layer_norm_rms_epsilon"))?,
                rope_freq_base: g.f32_or(&k("rope.freq_base"), 10000.0)?,
            },
            n_ff: g.u64(&k("feed_forward_length"))? as u32,
            tied_output: !g.has_tensor(names::OUTPUT),
            name: g.str("general.name").unwrap_or(ARCH_DENSE).to_string(),
        })
    }
}

/// How a [`Qwen3`] is loaded.
#[derive(Clone)]
pub struct DenseOptions {
    pub backend: BackendChoice,
    /// KV pool size in rows, shared by all sequences (rounded up to a multiple of 256).
    pub n_ctx_pool: usize,
    /// Sequence ids `0..n_seq_max` may be used.
    pub n_seq_max: usize,
    pub kv: KvFormat,
    /// `None`: Hadamard rotation exactly when libllama rotates (a quantized cache).
    pub kv_rotation: Option<bool>,
    /// CPU threads; `None`: those of `BackendChoice::Cpu`, else the available parallelism.
    pub threads: Option<i32>,
    /// Keep the step graph while (n_tokens, n_kv, n_outputs) repeat.
    pub reuse_step_graph: bool,
    pub telemetry: Option<Arc<Telemetry>>,
}

impl DenseOptions {
    pub fn new(backend: BackendChoice) -> Self {
        Self {
            backend,
            n_ctx_pool: 4096,
            n_seq_max: 1,
            kv: KvFormat::default(),
            kv_rotation: None,
            threads: None,
            reuse_step_graph: true,
            telemetry: None,
        }
    }
}

impl std::fmt::Debug for DenseOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseOptions")
            .field("backend", &self.backend)
            .field("n_ctx_pool", &self.n_ctx_pool)
            .field("n_seq_max", &self.n_seq_max)
            .field("kv", &self.kv)
            .field("kv_rotation", &self.kv_rotation)
            .field("threads", &self.threads)
            .field("reuse_step_graph", &self.reuse_step_graph)
            .finish_non_exhaustive()
    }
}

struct Layer {
    attn: AttnWeights,
    ffn_norm: OwnedTensor,
    gate: OwnedTensor,
    up: OwnedTensor,
    down: OwnedTensor,
}

struct Weights {
    tok_embd: OwnedTensor,
    out_norm: OwnedTensor,
    /// `None` when the output projection is tied to the token embedding.
    output: Option<OwnedTensor>,
    layers: Vec<Layer>,
}

/// Inputs of one step; reallocated when its sizes change.
struct State {
    n_tokens: usize,
    n_kv: usize,
    n_out: usize,
    tokens: OwnedTensor,
    pos: OwnedTensor,
    rows: OwnedTensor,
    mask: OwnedTensor,
    outputs: OwnedTensor,
}

/// A step graph kept across steps. It holds the contexts of the weights, KV cache and state it
/// reads, so the state must be dropped after it.
struct StepGraph {
    /// (n_tokens, n_kv, n_outputs) of the build.
    key: (usize, usize, usize),
    graph: OwnedGraph,
    logits: OwnedTensor,
}

/// A Qwen3 dense model serving several sequences from one KV pool.
pub struct Qwen3 {
    h: DenseHparams,
    sched: Sched,
    primary: Backend,
    weights_bytes: u64,
    kv_bytes: u64,
    weights: Weights,
    kv: Kv,
    pool: KvPool,
    n_seq_max: usize,
    step: Option<StepGraph>,
    state: Option<State>,
    reuse_step_graph: bool,
    step_graphs_built: u64,
    metrics: MemoryMetrics,
    telemetry: Option<Arc<Telemetry>>,
}

impl std::fmt::Debug for Qwen3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3")
            .field("hparams", &self.h)
            .field("backend", &self.primary)
            .field("n_seq_max", &self.n_seq_max)
            .finish()
    }
}

fn pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

fn kv_err(e: KvError) -> ModelError {
    ModelError::Invalid(e.to_string())
}

fn step_nodes(h: &AttnHparams) -> usize {
    STEP_NODES_PER_LAYER * (h.n_layer as usize + 1)
}

impl Qwen3 {
    pub fn load(path: impl AsRef<Path>, opts: &DenseOptions) -> Result<Qwen3> {
        let gguf = Gguf::open(path.as_ref())?;
        let h = DenseHparams::from_gguf(&gguf)?;
        let a = &h.attn;
        if a.n_head % a.n_head_kv != 0 || a.n_rot != a.head_dim {
            return Err(ModelError::Invalid("unsupported hyperparameters".into()));
        }
        if opts.n_seq_max == 0 {
            return Err(ModelError::Invalid("n_seq_max must be at least 1".into()));
        }
        let cpu_threads = |own: Option<usize>| -> i32 {
            opts.threads.map(|t| t.max(1) as usize).or(own).map_or_else(
                || std::thread::available_parallelism().map_or(1, usize::from) as i32,
                |t| t as i32,
            )
        };
        let (n, gpu) = match opts.backend {
            BackendChoice::Cpu { threads } => (cpu_threads(threads), None),
            BackendChoice::Gpu { index } => (cpu_threads(None), Some(Backend::gpu(index)?)),
        };
        let poll = cpu_poll::detect(None, n as usize);
        let cpu = Backend::cpu_with_poll(n, Some(poll.level))?;
        if let Some(t) = &opts.telemetry {
            t.metrics().cpu_poll(poll.level, poll.auto);
        }
        let (primary, backends) = match gpu {
            None => (cpu.clone(), vec![cpu]),
            Some(gpu) => (gpu.clone(), vec![gpu, cpu]),
        };
        let sched = Sched::new(&backends, step_nodes(a) * 4, false)?;
        let (weights, weights_bytes) = load_weights(&gguf, &h, &primary)?;
        let rows = pad(opts.n_ctx_pool.max(1), KV_PAD);
        let (kv, kv_bytes) = alloc_kv(a, rows, &primary, opts.kv, opts.kv_rotation)?;
        let pool = KvPool::new(Box::new(Chunked::new(rows)));
        tracing::debug!(
            backend = primary.name(),
            n_layer = a.n_layer,
            rows,
            "qwen3 dense loaded"
        );
        Ok(Qwen3 {
            h,
            sched,
            primary,
            weights_bytes: weights_bytes as u64,
            kv_bytes: kv_bytes as u64,
            weights,
            kv,
            pool,
            n_seq_max: opts.n_seq_max,
            step: None,
            state: None,
            reuse_step_graph: opts.reuse_step_graph,
            step_graphs_built: 0,
            metrics: MemoryMetrics::new(opts.telemetry.clone()),
            telemetry: opts.telemetry.clone(),
        })
    }

    pub fn hparams(&self) -> &DenseHparams {
        &self.h
    }

    pub fn n_vocab(&self) -> usize {
        self.h.attn.n_vocab as usize
    }

    pub fn n_seq_max(&self) -> usize {
        self.n_seq_max
    }

    pub fn pool(&self) -> &KvPool {
        &self.pool
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv.fmt
    }

    /// Whether K and V are Hadamard-rotated.
    pub fn kv_rotated(&self) -> (bool, bool) {
        (self.kv.rot_k.is_some(), self.kv.rot_v.is_some())
    }

    /// Whole-step graphs built so far (a reused graph does not count).
    pub fn step_graphs_built(&self) -> u64 {
        self.step_graphs_built
    }

    pub fn set_reuse_step_graph(&mut self, on: bool) {
        self.reuse_step_graph = on;
        if !on {
            self.step = None;
        }
    }

    pub fn memory(&self) -> Vec<MemoryUse> {
        let tier = if self.primary.is_cpu() { "ram" } else { "vram" };
        let device = self.primary.name();
        let use_ = |kind, capacity: u64| MemoryUse {
            device: device.to_string(),
            tier,
            kind,
            format: (kind == "kv").then(|| self.kv.fmt.to_string()),
            capacity,
            used: capacity,
        };
        let work = self.sched.buffer_size(&self.primary) as u64;
        vec![
            use_("weights", self.weights_bytes),
            use_("kv", self.kv_bytes),
            use_("workspace", work),
        ]
    }

    /// Publishes memory use, KV tokens held and the pool's blocks; also after a sequence is cut,
    /// so the gauges do not keep its length.
    fn report_memory(&self) {
        let Some(t) = &self.telemetry else { return };
        let held = self.held_tokens() as u64;
        for m in self.memory() {
            match &m.format {
                Some(f) => self.metrics.kv(
                    &m.device,
                    m.tier,
                    f,
                    m.used,
                    m.capacity,
                    held,
                    self.pool.capacity() as u64,
                ),
                None => self
                    .metrics
                    .buffer(&m.device, m.tier, m.kind, m.used, m.capacity),
            }
        }
        let b = self.pool.blocks();
        let (device, tier) = (
            self.primary.name(),
            if self.primary.is_cpu() { "ram" } else { "vram" },
        );
        let m = t.metrics();
        m.kv_blocks(&device, tier, "used", b.used as u64);
        m.kv_blocks(&device, tier, "reserved", b.reserved as u64);
        m.kv_blocks(&device, tier, "free", b.free as u64);
    }

    fn held_tokens(&self) -> usize {
        (0..self.n_seq_max as u32)
            .map(|s| self.pool.seq_len(s))
            .sum()
    }

    fn check_seq(&self, seq: u32) -> Result<()> {
        if (seq as usize) < self.n_seq_max {
            Ok(())
        } else {
            Err(ModelError::Invalid(format!(
                "sequence {seq} outside 0..{}",
                self.n_seq_max
            )))
        }
    }

    /// Copies the rows a layout move or `seq_cp` asks for, on the device. `moves` counts them as
    /// relocations of a grown sequence (not those of a copy).
    fn apply(&mut self, copies: &[RowCopy], moves: bool) -> Result<()> {
        let copies: Vec<_> = copies.iter().map(|c| (c.src, c.dst, c.n)).collect();
        copy_rows(&self.kv, &mut self.sched, &copies)?;
        if let Some(t) = self.telemetry.as_ref().filter(|_| moves) {
            for _ in &copies {
                t.metrics().kv_relocation();
            }
        }
        Ok(())
    }

    /// Guarantees rows for `seq` to grow to `n` positions. `Ok(false)` when the pool cannot
    /// give them (nothing changed). A sequence the layout has to move is copied on the device.
    pub fn try_reserve(&mut self, seq: u32, n: usize) -> Result<bool> {
        self.check_seq(seq)?;
        match self.pool.reserve(seq, n) {
            Ok(copies) => {
                self.apply(&copies, true)?;
                self.report_memory();
                Ok(true)
            }
            Err(KvError::Full { .. }) => Ok(false),
            Err(e) => Err(kv_err(e)),
        }
    }

    /// Like [`Qwen3::try_reserve`], a full pool is an error.
    pub fn seq_reserve(&mut self, seq: u32, n: usize) -> Result<()> {
        if self.try_reserve(seq, n)? {
            Ok(())
        } else {
            Err(ModelError::Invalid(format!(
                "KV pool full: {n} rows for sequence {seq}, {} free",
                self.pool.free()
            )))
        }
    }

    /// Keeps the first `keep` positions of `seq`. The reservation shrinks to `keep` (rounded up
    /// to 256 rows); `usize::MAX` frees only the slack beyond the positions held.
    pub fn seq_rm(&mut self, seq: u32, keep: usize) -> Result<()> {
        self.check_seq(seq)?;
        self.pool.seq_rm(seq, keep);
        self.report_memory();
        Ok(())
    }

    /// `dst` (empty) becomes the first `n` positions of `src`. `Ok(false)` when the pool has no
    /// rows for it (nothing changed).
    pub fn try_seq_cp(&mut self, src: u32, dst: u32, n: usize) -> Result<bool> {
        self.check_seq(src)?;
        self.check_seq(dst)?;
        match self.pool.seq_cp(src, dst, n) {
            Ok(copies) => {
                self.apply(&copies, false)?;
                self.report_memory();
                Ok(true)
            }
            Err(KvError::Full { .. }) => Ok(false),
            Err(e) => Err(kv_err(e)),
        }
    }

    pub fn seq_cp(&mut self, src: u32, dst: u32, n: usize) -> Result<()> {
        if self.try_seq_cp(src, dst, n)? {
            Ok(())
        } else {
            Err(ModelError::Invalid(format!(
                "KV pool full: {n} rows for sequence {dst}, {} free",
                self.pool.free()
            )))
        }
    }

    /// Bytes of one cached position across all layers (K and V).
    fn kv_row_bytes(&self) -> usize {
        self.kv
            .k
            .iter()
            .chain(&self.kv.v)
            .map(|t| t.get().nbytes() / self.kv.n_rows)
            .sum()
    }

    /// The K and V rows of the positions of `seq` as host bytes, tensor by tensor, positions in
    /// order (ADR 0049). The format does not depend on where the rows sit in the pool.
    pub fn seq_snapshot(&self, seq: u32) -> Result<Vec<u8>> {
        self.check_seq(seq)?;
        let n = self.pool.seq_len(seq);
        let runs = self.pool.runs(seq, n);
        let mut out = Vec::with_capacity(n * self.kv_row_bytes());
        for t in self.kv.k.iter().chain(&self.kv.v) {
            let row = t.get().nbytes() / self.kv.n_rows;
            for &(start, len) in &runs {
                out.extend_from_slice(&t.get().get_bytes(start * row, len * row)?);
            }
        }
        Ok(out)
    }

    /// Replaces `seq` by a [`Qwen3::seq_snapshot`] of `n` positions. `Ok(false)` when the pool
    /// has no rows for them (nothing changed).
    pub fn try_seq_restore(&mut self, seq: u32, bytes: &[u8], n: usize) -> Result<bool> {
        self.check_seq(seq)?;
        if bytes.len() != n * self.kv_row_bytes() {
            return Err(ModelError::Invalid(format!(
                "a KV snapshot of {} bytes does not hold {n} positions",
                bytes.len()
            )));
        }
        // Check the pool can hold `n` rows before anything is dropped, so a full pool changes
        // nothing; then start from an empty sequence, which gets a run at least as good.
        if n > 0 && n > self.pool.reserved(seq) && !self.try_reserve(seq, n)? {
            return Ok(false);
        }
        self.pool.seq_rm(seq, 0);
        if n > 0 && !self.try_reserve(seq, n)? {
            return Err(ModelError::Invalid(format!(
                "KV pool lost the rows of sequence {seq} while it was replaced"
            )));
        }
        let runs = self.pool.runs(seq, n);
        let mut rest = bytes;
        for t in self.kv.k.iter().chain(&self.kv.v) {
            let row = t.get().nbytes() / self.kv.n_rows;
            for &(start, len) in &runs {
                let (head, tail) = rest.split_at(len * row);
                t.get().set_bytes(head, start * row)?;
                rest = tail;
            }
        }
        if n > 0 {
            self.pool.commit(&[StepInput {
                seq,
                n_tokens: n,
                all_logits: false,
            }]);
        }
        self.report_memory();
        Ok(true)
    }

    pub fn seq_restore(&mut self, seq: u32, bytes: &[u8], n: usize) -> Result<()> {
        if self.try_seq_restore(seq, bytes, n)? {
            Ok(())
        } else {
            Err(ModelError::Invalid(format!(
                "KV pool full: {n} rows for sequence {seq}, {} free",
                self.pool.free()
            )))
        }
    }

    /// Positions held by sequence 0.
    pub fn n_past(&self) -> usize {
        self.pool.seq_len(0)
    }

    /// Forgets sequence 0.
    pub fn reset(&mut self) {
        self.pool.seq_rm(0, 0);
        self.report_memory();
    }

    /// Rolls sequence 0 back to its first `n_past` positions.
    pub fn rewind(&mut self, n_past: usize) -> Result<()> {
        if n_past > self.n_past() {
            return Err(ModelError::Invalid(format!(
                "cannot rewind to {n_past}: only {} tokens cached",
                self.n_past()
            )));
        }
        self.seq_rm(0, n_past)
    }

    /// Runs `tokens` after sequence 0's cache and returns the logits of every position,
    /// `tokens.len() * n_vocab` values, position-major.
    pub fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.eval_seq0(tokens, true)
    }

    /// Like [`Qwen3::eval`] for the last position only (`n_vocab` values).
    pub fn eval_last(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.eval_seq0(tokens, false)
    }

    fn eval_seq0(&mut self, tokens: &[i32], all_logits: bool) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.decode(
            &[SeqEval {
                seq: 0,
                tokens,
                all_logits,
            }],
            &mut out,
        )?;
        Ok(out)
    }

    /// One step: each input's tokens are appended to its sequence (sequences distinct, tokens
    /// never empty). Rows missing for a sequence are reserved first (a full pool is an error and
    /// leaves every sequence's length as it was). The logits of the last token of each input, or
    /// of all its tokens with `all_logits`, go into `out`, `n_vocab` floats per row in input and
    /// position order; the result is the row range of each input.
    pub fn decode(
        &mut self,
        inputs: &[SeqEval<'_>],
        out: &mut Vec<f32>,
    ) -> Result<Vec<Range<usize>>> {
        let t0 = Instant::now();
        if inputs.is_empty() {
            return Err(ModelError::Invalid("no sequences".into()));
        }
        let n_vocab = self.h.attn.n_vocab;
        let mut step = Vec::with_capacity(inputs.len());
        for inp in inputs {
            self.check_seq(inp.seq)?;
            if inp.tokens.is_empty() {
                return Err(ModelError::Invalid("no tokens".into()));
            }
            if let Some(&t) = inp.tokens.iter().find(|&&t| t < 0 || t as u32 >= n_vocab) {
                return Err(ModelError::Invalid(format!(
                    "token {t} outside the vocabulary"
                )));
            }
            step.push(StepInput {
                seq: inp.seq,
                n_tokens: inp.tokens.len(),
                all_logits: inp.all_logits,
            });
        }
        for s in &step {
            let need = self.pool.seq_len(s.seq) + s.n_tokens;
            if self.pool.reserved(s.seq) < need {
                self.seq_reserve(s.seq, need)?;
            }
        }
        let plan = self.pool.plan_step(&step).map_err(kv_err)?;
        let n = plan.pos.len();
        let n_out = plan.outputs.len();
        let key = (n, plan.n_kv, n_out);
        if self
            .state
            .as_ref()
            .is_none_or(|s| (s.n_tokens, s.n_kv, s.n_out) != key)
        {
            // The kept graph holds the old state's context: drop it first, then the buffers.
            self.step = None;
            self.state = None;
            self.state = Some(alloc_state(key, &self.primary)?);
        }
        let st = self.state.as_ref().expect("state allocated above");
        let tokens: Vec<i32> = inputs
            .iter()
            .flat_map(|i| i.tokens.iter().copied())
            .collect();
        st.tokens.get().set_i32(&tokens)?;
        st.pos.get().set_i32(&plan.pos)?;
        st.rows.get().set_i32(&plan.rows)?;
        st.mask.get().set_f16_bits(&plan.mask)?;
        st.outputs.get().set_i32(&plan.outputs)?;

        let reusable = self.reuse_step_graph
            && self
                .step
                .as_ref()
                .is_some_and(|s| s.key == key && self.sched.is_allocated(&s.graph.get()));
        let fresh;
        let graph: &StepGraph = if reusable {
            self.step.as_ref().expect("checked above")
        } else {
            self.step = None;
            let nodes = step_nodes(&self.h.attn);
            let ctx = Rc::new(Context::new(nodes, nodes)?);
            let (graph, logits) = build_step(&self.h, &self.weights, &self.kv, st, &ctx);
            let mut deps = vec![
                &self.weights.tok_embd,
                &self.kv.k[0],
                &self.kv.v[0],
                &st.tokens,
            ];
            deps.extend(self.kv.rot_k.iter().chain(&self.kv.rot_v));
            fresh = StepGraph {
                key,
                logits: ctx.keep(logits)?,
                graph: ctx.keep_graph(graph, &deps)?,
            };
            self.sched.alloc_graph(&fresh.graph.get())?;
            self.step_graphs_built += 1;
            if self.reuse_step_graph {
                self.step.insert(fresh)
            } else {
                &fresh
            }
        };
        self.sched.compute(&graph.graph.get())?;
        graph.logits.get().get_f32_into(out)?;

        let mut row = 0;
        let ranges = step
            .iter()
            .map(|s| {
                let k = if s.all_logits { s.n_tokens } else { 1 };
                row += k;
                row - k..row
            })
            .collect();
        self.pool.commit(&step);
        if let Some(t) = &self.telemetry {
            t.record_step(StepSample {
                phase: if step.iter().any(|s| s.n_tokens > 1) {
                    Phase::Prefill
                } else {
                    Phase::Decode
                },
                stage: 0,
                graphs: false,
                duration: t0.elapsed(),
                tokens: n as u32,
                segments: 1,
            });
            self.report_memory();
        }
        Ok(ranges)
    }
}

fn alloc_state(key: (usize, usize, usize), backend: &Backend) -> Result<State> {
    let (n, n_kv, n_out) = key;
    let ctx = Rc::new(Context::new(8, 0)?);
    let state = State {
        n_tokens: n,
        n_kv,
        n_out,
        tokens: ctx.new_persistent(Type::I32, &[n as i64]),
        pos: ctx.new_persistent(Type::I32, &[n as i64]),
        rows: ctx.new_persistent(Type::I32, &[n as i64]),
        mask: ctx.new_persistent(Type::F16, &[n_kv as i64, n as i64]),
        outputs: ctx.new_persistent(Type::I32, &[n_out as i64]),
    };
    ctx.alloc_tensors(backend)?;
    Ok(state)
}

fn load_weights(gguf: &Gguf, h: &DenseHparams, backend: &Backend) -> Result<(Weights, usize)> {
    let n_tensors = 3 + h.attn.n_layer as usize * 11;
    let ctx = Rc::new(Context::new(n_tensors + 4, 0)?);
    let mut loaded: Vec<(OwnedTensor, usize, u64, u64)> = Vec::new();
    let mut make = |name: &str| -> Result<OwnedTensor> {
        let info = gguf.tensor(name)?;
        let ne: Vec<i64> = info.ne.iter().map(|&x| x as i64).collect();
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let t = ctx.new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        if t.get().nbytes() as u64 != info.nbytes {
            return Err(ModelError::Invalid(format!("size mismatch for `{name}`")));
        }
        loaded.push((t.clone(), info.shard, info.offset, info.nbytes));
        Ok(t)
    };
    let tok_embd = make(names::TOK_EMBD)?;
    let out_norm = make(names::OUTPUT_NORM)?;
    let output = if h.tied_output {
        None
    } else {
        Some(make(names::OUTPUT)?)
    };
    let mut layers = Vec::new();
    for il in 0..h.attn.n_layer {
        let mut l = |w: &str| make(&names::layer(il, w));
        layers.push(Layer {
            attn: AttnWeights {
                attn_norm: l("attn_norm")?,
                wq: l("attn_q")?,
                wk: l("attn_k")?,
                wv: l("attn_v")?,
                wo: l("attn_output")?,
                q_norm: l("attn_q_norm")?,
                k_norm: l("attn_k_norm")?,
            },
            ffn_norm: l("ffn_norm")?,
            gate: l("ffn_gate")?,
            up: l("ffn_up")?,
            down: l("ffn_down")?,
        });
    }
    let buffer_size = ctx.alloc_weights(backend)?;
    let mut shards = gguf
        .paths
        .iter()
        .map(std::fs::File::open)
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut buf = vec![0u8; 16 << 20];
    for (t, shard, offset, nbytes) in loaded {
        let f = &mut shards[shard];
        f.seek(SeekFrom::Start(offset))?;
        let mut done = 0u64;
        while done < nbytes {
            let len = ((nbytes - done) as usize).min(buf.len());
            f.read_exact(&mut buf[..len])?;
            t.get().set_bytes(&buf[..len], done as usize)?;
            done += len as u64;
        }
    }
    Ok((
        Weights {
            tok_embd,
            out_norm,
            output,
            layers,
        },
        buffer_size,
    ))
}

/// The whole step as one graph. Returns the graph and the logits `[n_vocab, n_outputs]`.
fn build_step<'a>(
    h: &DenseHparams,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    ctx: &'a Context,
) -> (Graph<'a>, Tensor<'a>) {
    let a = &h.attn;
    let mut g = ctx.new_graph(step_nodes(a));
    let mut x = ctx.get_rows(w.tok_embd.get(), st.tokens.get());
    let step = AttnStep {
        pos: st.pos.get(),
        rows: st.rows.get(),
        mask: st.mask.get(),
        n_tokens: st.n_tokens,
        n_kv: st.n_kv,
    };
    for (il, lw) in w.layers.iter().enumerate() {
        let mut ffn_inp = attention(a, ctx, &mut g, &lw.attn, kv, il, &step, x);
        if il + 1 == w.layers.len() && st.n_out < st.n_tokens {
            // libllama selects the output rows after attention and before the residual add;
            // `attention` adds the residual itself and a row-wise add commutes with the
            // selection, so the values are the same.
            ffn_inp = ctx.get_rows(ffn_inp, st.outputs.get());
        }
        let cur = rms_norm_mul(ctx, ffn_inp, lw.ffn_norm.get(), a.rms_eps);
        let ffn = ctx.swiglu_split(
            ctx.mul_mat(lw.gate.get(), cur),
            ctx.mul_mat(lw.up.get(), cur),
        );
        x = ctx.add(ctx.mul_mat(lw.down.get(), ffn), ffn_inp);
    }
    let cur = rms_norm_mul(ctx, x, w.out_norm.get(), a.rms_eps);
    let head = w.output.as_ref().unwrap_or(&w.tok_embd);
    let logits = ctx.mul_mat(head.get(), cur).set_output();
    g.expand(logits);
    (g, logits)
}

#[cfg(test)]
mod graph_order {
    use super::*;
    use std::path::PathBuf;

    fn model(n_seq: usize) -> Qwen3 {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3-f16.gguf");
        let opts = DenseOptions {
            n_seq_max: n_seq,
            n_ctx_pool: 512,
            threads: Some(1),
            ..DenseOptions::new(BackendChoice::Cpu { threads: Some(1) })
        };
        Qwen3::load(path, &opts).expect("fixture loads")
    }

    /// The ops after the attention block of a non-last layer are libllama's `build_ffn` with
    /// `LLM_FFN_SILU` and `LLM_FFN_PAR` (gate, up, swiglu_split, down), between the FFN norm
    /// and the residual add; the output rows are selected once, in the last layer.
    #[test]
    fn layer_ops_follow_libllama_order() {
        let m = model(1);
        for (n, n_out) in [(1usize, 1usize), (5, 1), (5, 5)] {
            let st = alloc_state((n, KV_PAD, n_out), &m.primary).unwrap();
            let nodes = step_nodes(&m.h.attn);
            let ctx = Context::new(nodes, nodes).unwrap();
            let (g, _) = build_step(&m.h, &m.weights, &m.kv, &st, &ctx);
            let ops: Vec<String> = (0..g.n_nodes()).map(|i| g.node(i).op_name()).collect();
            let get_rows = ops.iter().filter(|o| *o == "GET_ROWS").count();
            // Embedding, plus the output selection when fewer rows than tokens are wanted.
            assert_eq!(get_rows, 1 + usize::from(n_out < n), "n={n} out={n_out}");
            let ffn = [
                "RMS_NORM", "MUL", "MUL_MAT", "MUL_MAT", "GLU", "MUL_MAT", "ADD",
            ];
            let at = ops
                .windows(ffn.len())
                .position(|w| w == ffn)
                .unwrap_or_else(|| panic!("no FFN window in {ops:?}"));
            // Layer 0's FFN starts right after layer 0's attention output: `wo` then the add.
            assert_eq!(
                &ops[at - 2..at],
                ["MUL_MAT", "ADD"],
                "layer 0 FFN follows wo"
            );
            // The head ends the graph: output norm, then the projection.
            assert_eq!(&ops[ops.len() - 3..], ["RMS_NORM", "MUL", "MUL_MAT"]);
        }
    }

    /// Many sequences in one step: the inputs come from the pool's plan.
    #[test]
    fn decode_validates_inputs() {
        let mut m = model(2);
        let mut out = Vec::new();
        let seq = |seq, tokens| SeqEval {
            seq,
            tokens,
            all_logits: false,
        };
        assert!(m.decode(&[], &mut out).is_err());
        assert!(m.decode(&[seq(0, &[])], &mut out).is_err());
        assert!(m.decode(&[seq(2, &[1])], &mut out).is_err());
        assert!(m.decode(&[seq(0, &[-1])], &mut out).is_err());
        assert!(m.decode(&[seq(0, &[1]), seq(0, &[2])], &mut out).is_err());
        assert_eq!(m.n_past(), 0);
    }
}

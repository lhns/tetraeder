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

mod ranked;

use crate::attn::{
    AttnHparams, AttnStep, AttnWeights, KV_PAD, Kv, MASK_ROW_PAD, SeqEval, alloc_kv,
    attention_seqs, copy_rows, mask_tail, mask_view, rms_norm_mul,
};
use crate::cpu_poll;
use crate::gguf::{Gguf, GgufError};
use crate::hparams::names;
use crate::qwen3moe::{BackendChoice, MemoryUse, ModelError, Result};
use crate::readback::{Readback, STAGING_HALF, append_direct, row_spans, write_direct};
use crate::shard::{gather_rows, scatter_rows};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tetraeder_executor::ExecCounters;
use tetraeder_ggml_sys::ggml::{
    Backend, Context, Graph, OwnedGraph, OwnedTensor, Sched, Tensor, Type, cpu,
};
use tetraeder_kvcache::{Chunked, KvError, KvPool, RowCopy, SeqAttn, StepInput, StepPlan};
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_plan::{KvFormat, Topology, micro_batch_sizes};
use tetraeder_telemetry::{Phase, StepSample, Telemetry};

/// `general.architecture` of a Qwen3 dense model.
pub const ARCH_DENSE: &str = "qwen3";

/// Upper bound of nodes per layer in a whole-step graph (about 45 are used), plus per sequence
/// with attention per sequence (7: Q, K, V and mask views, permute, attention, join).
const STEP_NODES_PER_LAYER: usize = 128;
const STEP_NODES_PER_SEQ: usize = 8;

/// Env var that turns on [`DenseOptions::use_executor`] in [`DenseOptions::new`] (`1`).
pub const EXECUTOR_ENV: &str = "TETRAEDER_DENSE_EXECUTOR";

/// Env var read by [`DenseOptions::new`]: `1` turns on [`DenseOptions::per_seq_attn`], `0` turns
/// off [`DenseOptions::per_seq_attn_single`] (the masked attention for every step, as before
/// 2026-11-19).
pub const PER_SEQ_ATTN_ENV: &str = "TETRAEDER_DENSE_PER_SEQ_ATTN";

/// Env var: `1` logs a timing line per prefill step and per step over 50 ms (target
/// `tetraeder::trace`), and the snapshot spare thread's work (diagnosis of ttft5-20261119).
pub const STEP_TRACE_ENV: &str = "TETRAEDER_STEP_TRACE";

/// Whether [`STEP_TRACE_ENV`] is `1` (read once).
pub fn step_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(STEP_TRACE_ENV).is_ok_and(|v| v == "1"))
}

/// Minor page faults of the calling thread so far (Linux `/proc/thread-self/stat`), else `None`.
pub fn thread_minor_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
    // Fields after the command name: state is field 3, minflt field 10.
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(7)?.parse().ok()
}

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
    /// Attention per sequence over its own run of rows instead of one masked attention over
    /// rows `0..n_kv` of the pool. `new` takes it from `TETRAEDER_DENSE_PER_SEQ_ATTN=1`.
    pub per_seq_attn: bool,
    /// A step with one input attends over that input's run only, even with `per_seq_attn` off:
    /// the masked attention reads every row below the run too (another sequence's), which made
    /// prefill and decode of a sequence above another up to 1.45x slower than libllama, whose
    /// unified cache attends over the sequence's own cells (`fused_kv_rows`; ttft5-20261119,
    /// ADR 0066 History). One attention op either way. Default on; `0` in the env turns it off.
    pub per_seq_attn_single: bool,
    /// On a CPU backend, matmul weights in the CPU's extra buffer types (AMX, repack) where they
    /// run, as libllama's `use_extra_bufts` (default true there and here).
    pub extra_bufts: bool,
    pub telemetry: Option<Arc<Telemetry>>,
    /// Run a one-GPU [`Qwen3::load`] on the executor with one rank (ADR 0069) instead of the
    /// scheduler. Default off, or on with `TETRAEDER_DENSE_EXECUTOR=1`, until the executor has
    /// shown no regression on a GPU. [`Qwen3::load_sharded`] always uses the executor; the CPU
    /// backend always uses the scheduler through `load`.
    pub use_executor: bool,
    /// Run only the first N layers and the head (calibration anchors, ADR 0068); the logits
    /// are meaningless, the timing is real. Executor path, one stage only.
    pub layer_limit: Option<u32>,
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
            per_seq_attn: std::env::var(PER_SEQ_ATTN_ENV).is_ok_and(|v| v == "1"),
            per_seq_attn_single: !std::env::var(PER_SEQ_ATTN_ENV).is_ok_and(|v| v == "0"),
            extra_bufts: true,
            telemetry: None,
            use_executor: std::env::var(EXECUTOR_ENV).is_ok_and(|v| v == "1"),
            layer_limit: None,
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
            .field("per_seq_attn", &self.per_seq_attn)
            .field("per_seq_attn_single", &self.per_seq_attn_single)
            .field("extra_bufts", &self.extra_bufts)
            .field("use_executor", &self.use_executor)
            .field("layer_limit", &self.layer_limit)
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
    /// One weight of each extra buffer type's context, so a kept step graph holds those too.
    extra_anchors: Vec<OwnedTensor>,
}

/// Sizes of a step's inputs: tokens, `n_kv` (0 with attention per sequence), outputs and the
/// mask buffer's shape: `[n_kv, pad(n_tokens, MASK_ROW_PAD)]`, or per sequence `[blocks + tail,
/// 1]` with the tail [`mask_tail`] of the last block. The padding is never written and stays 0.
type StateKey = (usize, usize, usize, [usize; 2]);

/// The [`StateKey`] of a step plan.
fn state_key(plan: &StepPlan) -> StateKey {
    let (n, n_out) = (plan.pos.len(), plan.outputs.len());
    match plan.seqs.last() {
        None => (n, plan.n_kv, n_out, [plan.n_kv, pad(n, MASK_ROW_PAD)]),
        Some(s) => (
            n,
            0,
            n_out,
            [plan.mask.len() + mask_tail(s.n_tokens, s.n_kv), 1],
        ),
    }
}

/// Inputs of one step; reallocated when its sizes change.
struct State {
    key: StateKey,
    tokens: OwnedTensor,
    pos: OwnedTensor,
    rows: OwnedTensor,
    mask: OwnedTensor,
    outputs: OwnedTensor,
}

/// A step graph kept across steps. It holds the contexts of the weights, KV cache and state it
/// reads, so the state must be dropped after it.
struct StepGraph {
    /// The state's sizes and, with attention per sequence, every input's rows and tokens.
    key: (StateKey, Vec<SeqAttn>),
    graph: OwnedGraph,
    logits: OwnedTensor,
}

/// The scheduler path: weights, KV and step graph on one backend through [`Sched`].
struct SchedEngine {
    sched: Sched,
    primary: Backend,
    weights_bytes: u64,
    kv_bytes: u64,
    /// CPU extra buffer types holding weights (empty: none).
    weight_bufts: Vec<String>,
    weights: Weights,
    kv: Kv,
    step: Option<StepGraph>,
    state: Option<State>,
    /// Pinned staging for snapshots of a GPU cache (ADR 0074); `None` reads straight.
    readback: Option<Readback>,
}

enum Engine {
    Sched(Box<SchedEngine>),
    /// Per-rank weights, KV and graphs issued by the executor (ADR 0069).
    Ranked(Box<ranked::Ranked>),
}

/// A Qwen3 dense model serving several sequences from one KV pool.
pub struct Qwen3 {
    h: DenseHparams,
    engine: Engine,
    pool: KvPool,
    /// Rows of every KV tensor.
    n_rows: usize,
    n_seq_max: usize,
    kv_fmt: KvFormat,
    kv_rotated: (bool, bool),
    reuse_step_graph: bool,
    per_seq_attn: bool,
    per_seq_attn_single: bool,
    step_graphs_built: u64,
    /// KV rows copied by layout moves and copies since the last step, and the time they took
    /// (copies wait for the device), for [`STEP_TRACE_ENV`].
    trace_moves: (usize, std::time::Duration),
    metrics: MemoryMetrics,
    telemetry: Option<Arc<Telemetry>>,
}

impl std::fmt::Debug for Qwen3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let engine = match &self.engine {
            Engine::Sched(e) => e.primary.name(),
            Engine::Ranked(r) => format!("{} ranks on the executor", r.n_ranks()),
        };
        f.debug_struct("Qwen3")
            .field("hparams", &self.h)
            .field("engine", &engine)
            .field("n_seq_max", &self.n_seq_max)
            .finish()
    }
}

/// The GGUF and its hyperparameters, checked.
fn open(path: &Path, opts: &DenseOptions) -> Result<(Gguf, DenseHparams)> {
    let gguf = Gguf::open(path)?;
    let h = DenseHparams::from_gguf(&gguf)?;
    let a = &h.attn;
    if a.n_head % a.n_head_kv != 0 || a.n_rot != a.head_dim {
        return Err(ModelError::Invalid("unsupported hyperparameters".into()));
    }
    if opts.n_seq_max == 0 {
        return Err(ModelError::Invalid("n_seq_max must be at least 1".into()));
    }
    Ok((gguf, h))
}

fn pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

fn kv_err(e: KvError) -> ModelError {
    ModelError::Invalid(e.to_string())
}

fn step_nodes(h: &AttnHparams, n_seq: usize) -> usize {
    (STEP_NODES_PER_LAYER + STEP_NODES_PER_SEQ * n_seq) * (h.n_layer as usize + 1)
}

impl Qwen3 {
    /// Loads the whole model on one backend, through the scheduler; on a GPU with
    /// [`DenseOptions::use_executor`] as one rank of the executor.
    pub fn load(path: impl AsRef<Path>, opts: &DenseOptions) -> Result<Qwen3> {
        let (gguf, h) = open(path.as_ref(), opts)?;
        if let (BackendChoice::Gpu { index }, true) = (&opts.backend, opts.use_executor) {
            let topo = Topology::single(&format!("gpu{index}"), Some(*index), h.attn.n_layer);
            return Self::load_ranked(&gguf, h, &topo, opts);
        }
        if opts.layer_limit.is_some() {
            return Err(ModelError::Invalid(
                "layer_limit runs on the executor, not on the scheduler path".into(),
            ));
        }
        let a = &h.attn;
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
        let sched = Sched::new(&backends, step_nodes(a, opts.n_seq_max) * 4, false)?;
        let (weights, weights_bytes, extra) = load_weights(&gguf, &h, &primary, opts.extra_bufts)?;
        let rows = pad(opts.n_ctx_pool.max(1), KV_PAD);
        let (kv, kv_bytes) = alloc_kv(a, rows, &primary, opts.kv, opts.kv_rotation)?;
        let pool = KvPool::new(Box::new(Chunked::new(rows)));
        tracing::debug!(
            backend = primary.name(),
            n_layer = a.n_layer,
            rows,
            ?extra,
            "qwen3 dense loaded"
        );
        let kv_rotated = (kv.rot_k.is_some(), kv.rot_v.is_some());
        let readback = if primary.is_cpu() {
            None
        } else {
            Some(Readback::new(&primary, STAGING_HALF)?)
        };
        Ok(Qwen3 {
            h,
            engine: Engine::Sched(Box::new(SchedEngine {
                sched,
                primary,
                weights_bytes: weights_bytes as u64,
                kv_bytes: kv_bytes as u64,
                weights,
                kv,
                step: None,
                state: None,
                weight_bufts: extra,
                readback,
            })),
            pool,
            n_rows: rows,
            n_seq_max: opts.n_seq_max,
            kv_fmt: opts.kv,
            kv_rotated,
            reuse_step_graph: opts.reuse_step_graph,
            per_seq_attn: opts.per_seq_attn,
            per_seq_attn_single: opts.per_seq_attn_single,
            step_graphs_built: 0,
            trace_moves: Default::default(),
            metrics: MemoryMetrics::new(opts.telemetry.clone()),
            telemetry: opts.telemetry.clone(),
        })
    }

    /// Loads the model across the ranks of `topo` on the executor (ADR 0069, 0070): per rank its
    /// slices of the weights, the KV of its layers and KV heads, one KV pool for all. With
    /// `BackendChoice::Cpu` the ranks are CPU backends of `threads` threads each (CI stands CPU
    /// ranks in for GPUs); with `Gpu` rank i is the GPU `RankSpec::gpu_index` (else i).
    pub fn load_sharded(
        path: impl AsRef<Path>,
        topo: &Topology,
        opts: &DenseOptions,
    ) -> Result<Qwen3> {
        let (gguf, h) = open(path.as_ref(), opts)?;
        Self::load_ranked(&gguf, h, topo, opts)
    }

    fn load_ranked(
        gguf: &Gguf,
        h: DenseHparams,
        topo: &Topology,
        opts: &DenseOptions,
    ) -> Result<Qwen3> {
        let rows = pad(opts.n_ctx_pool.max(1), KV_PAD);
        let ranked = ranked::Ranked::load(gguf, &h, topo, opts, rows)?;
        let kv_rotated = ranked.kv_rotated();
        tracing::debug!(
            ranks = ranked.n_ranks(),
            comm = ranked.comm_path(),
            rows,
            "qwen3 dense loaded on the executor"
        );
        Ok(Qwen3 {
            h,
            engine: Engine::Ranked(Box::new(ranked)),
            // The executor has no attention per sequence.
            per_seq_attn: false,
            per_seq_attn_single: false,
            pool: KvPool::new(Box::new(Chunked::new(rows))),
            n_rows: rows,
            n_seq_max: opts.n_seq_max,
            kv_fmt: opts.kv,
            kv_rotated,
            reuse_step_graph: opts.reuse_step_graph,
            step_graphs_built: 0,
            trace_moves: Default::default(),
            metrics: MemoryMetrics::new(opts.telemetry.clone()),
            telemetry: opts.telemetry.clone(),
        })
    }

    /// Ranks of the executor; 1 on the scheduler path.
    pub fn n_ranks(&self) -> usize {
        match &self.engine {
            Engine::Sched(_) => 1,
            Engine::Ranked(r) => r.n_ranks(),
        }
    }

    /// The all-reduce path of the executor (`nccl`, `internal`, `fallback`); `None` on the
    /// scheduler path.
    pub fn comm_path(&self) -> Option<&'static str> {
        match &self.engine {
            Engine::Sched(_) => None,
            Engine::Ranked(r) => Some(r.comm_path()),
        }
    }

    /// Ranges launched and bytes moved by the executor since load or the last
    /// [`Qwen3::reset_exec_counters`]; `None` on the scheduler path.
    pub fn exec_counters(&self) -> Option<&ExecCounters> {
        match &self.engine {
            Engine::Sched(_) => None,
            Engine::Ranked(r) => Some(r.counters()),
        }
    }

    pub fn reset_exec_counters(&mut self) {
        if let Engine::Ranked(r) = &mut self.engine {
            r.reset_counters();
        }
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

    /// Names of the CPU extra buffer types that hold weights (empty when none does).
    pub fn weight_bufts(&self) -> &[String] {
        match &self.engine {
            Engine::Sched(e) => &e.weight_bufts,
            Engine::Ranked(_) => &[],
        }
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv_fmt
    }

    /// Whether K and V are Hadamard-rotated.
    pub fn kv_rotated(&self) -> (bool, bool) {
        self.kv_rotated
    }

    /// Whole-step graphs built so far (a reused graph does not count).
    pub fn step_graphs_built(&self) -> u64 {
        self.step_graphs_built
    }

    /// Switches [`DenseOptions::per_seq_attn`]; the next step builds its graph.
    pub fn set_per_seq_attn(&mut self, on: bool) {
        self.per_seq_attn = on;
    }

    pub fn set_reuse_step_graph(&mut self, on: bool) {
        self.reuse_step_graph = on;
        if !on {
            match &mut self.engine {
                Engine::Sched(e) => e.step = None,
                Engine::Ranked(r) => r.drop_step(),
            }
        }
    }

    pub fn memory(&self) -> Vec<MemoryUse> {
        let e = match &self.engine {
            Engine::Ranked(r) => return r.memory(),
            Engine::Sched(e) => e,
        };
        let tier = if e.primary.is_cpu() { "ram" } else { "vram" };
        let device = e.primary.name();
        let use_ = |kind, capacity: u64| MemoryUse {
            device: device.to_string(),
            tier,
            kind,
            format: (kind == "kv").then(|| self.kv_fmt.to_string()),
            capacity,
            used: capacity,
        };
        let work = e.sched.buffer_size(&e.primary) as u64;
        let mut out = vec![
            use_("weights", e.weights_bytes),
            use_("kv", e.kv_bytes),
            use_("workspace", work),
        ];
        if let Some(rb) = &e.readback {
            let bytes = rb.bytes() as u64;
            out.push(MemoryUse {
                device: "host".into(),
                tier: "ram",
                kind: "staging",
                format: None,
                capacity: bytes,
                used: bytes,
            });
        }
        out
    }

    /// The snapshot staging (ADR 0074) as (bytes, pinned); `None` when snapshots read straight.
    pub fn snapshot_staging(&self) -> Option<(usize, bool)> {
        match &self.engine {
            Engine::Sched(e) => e.readback.as_ref().map(|rb| (rb.bytes(), rb.pinned())),
            Engine::Ranked(_) => None,
        }
    }

    /// Replaces the snapshot staging by two halves of `half` bytes, or drops it (`None`: read
    /// straight). For tests and paired timings; the scheduler path only.
    pub fn set_snapshot_staging(&mut self, half: Option<usize>) -> Result<()> {
        let Engine::Sched(e) = &mut self.engine else {
            return Err(ModelError::Invalid(
                "snapshot staging is on the scheduler path only".into(),
            ));
        };
        e.readback = half.map(|h| Readback::new(&e.primary, h)).transpose()?;
        self.report_memory();
        Ok(())
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
        let (device, tier) = match &self.engine {
            Engine::Sched(e) => (
                e.primary.name(),
                if e.primary.is_cpu() { "ram" } else { "vram" },
            ),
            Engine::Ranked(r) => r.primary(),
        };
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
        let t = Instant::now();
        match &mut self.engine {
            Engine::Sched(e) => copy_rows(&e.kv, &mut e.sched, &copies)?,
            Engine::Ranked(r) => r.copy_rows(&copies)?,
        }
        self.trace_moves.0 += copies.iter().map(|c| c.2).sum::<usize>();
        self.trace_moves.1 += t.elapsed();
        // One relocation per move, however many chunks it copies in.
        if let Some(t) = self
            .telemetry
            .as_ref()
            .filter(|_| moves && !copies.is_empty())
        {
            t.metrics().kv_relocation();
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

    /// Undoes a failed step: `seq` goes back to `len` positions and at most `reserved` rows. It
    /// never grows, so unlike `seq_rm` followed by a reserve it cannot fail on a fragmented pool.
    pub fn rollback(&mut self, seq: u32, len: usize, reserved: usize) -> Result<()> {
        self.check_seq(seq)?;
        self.pool.rollback(seq, len, reserved);
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

    /// The cache tensors in snapshot order (all K layers, then all V layers), each as the
    /// slices of one layer across ranks in head order.
    fn kv_groups(&self) -> Vec<Vec<Tensor<'_>>> {
        match &self.engine {
            Engine::Sched(e) => {
                e.kv.k
                    .iter()
                    .chain(&e.kv.v)
                    .map(|t| vec![t.get()])
                    .collect()
            }
            Engine::Ranked(r) => r.kv_groups(),
        }
    }

    fn kv_rows(&self) -> usize {
        self.n_rows
    }

    /// Bytes of one cached position across all layers (K and V).
    pub fn kv_row_bytes(&self) -> usize {
        let n_rows = self.kv_rows();
        self.kv_groups()
            .iter()
            .flatten()
            .map(|t| t.nbytes() / n_rows)
            .sum()
    }

    /// The K and V rows of the positions of `seq` as host bytes, tensor by tensor, positions in
    /// order (ADR 0049). The format does not depend on where the rows sit in the pool, nor on
    /// the shape: a tensor cut across ranks is joined in head order.
    pub fn seq_snapshot(&self, seq: u32) -> Result<Vec<u8>> {
        self.check_seq(seq)?;
        let n = self.pool.seq_len(seq);
        let runs = self.pool.runs(seq, n);
        let mut out = crate::snapshot_pool::global().take(n * self.kv_row_bytes());
        if let Engine::Sched(e) = &self.engine {
            let tensors = e.kv.k.iter().chain(&e.kv.v).map(|t| t.get());
            let spans = row_spans(tensors, self.n_rows, &runs);
            match &e.readback {
                Some(rb) => rb.read(&spans, &mut out)?,
                None => append_direct(&spans, &mut out)?,
            }
            return Ok(out);
        }
        for group in self.kv_groups() {
            gather_rows(&group, self.kv_rows(), &runs, &mut out)?;
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
        if let Engine::Sched(e) = &self.engine {
            let tensors = e.kv.k.iter().chain(&e.kv.v).map(|t| t.get());
            let spans = row_spans(tensors, self.n_rows, &runs);
            match &e.readback {
                Some(rb) => rb.write(&spans, bytes)?,
                None => write_direct(&spans, bytes)?,
            }
        } else {
            let mut rest = bytes;
            for group in self.kv_groups() {
                scatter_rows(&group, self.kv_rows(), &runs, &mut rest)?;
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
        let trace = step_trace();
        let faults0 = if trace { thread_minor_faults() } else { None };
        let t_reserve = Instant::now();
        for s in &step {
            let need = self.pool.seq_len(s.seq) + s.n_tokens;
            if self.pool.reserved(s.seq) < need {
                self.seq_reserve(s.seq, need)?;
            }
        }
        let t_plan = Instant::now();
        // A one-sequence prefill on a pipelined topology runs as micro-batches, each planned as
        // the same chunk as a step of its own (ADR 0077).
        let micro = match &self.engine {
            Engine::Ranked(r) => r.micro_batches(),
            Engine::Sched(_) => 1,
        };
        let plans = match step.as_slice() {
            [one] if micro > 1 && one.n_tokens > 1 && !one.all_logits => {
                let sizes = micro_batch_sizes(one.n_tokens, micro);
                self.pool.plan_chunks(*one, &sizes).map_err(kv_err)?
            }
            _ => {
                // Attention per sequence needs each input in one run of rows (chunked layout).
                let per_seq = (self.per_seq_attn || (self.per_seq_attn_single && step.len() == 1))
                    && matches!(self.engine, Engine::Sched(_));
                let per_seq = if per_seq {
                    self.pool.plan_step_per_seq(&step).map_err(kv_err)?
                } else {
                    None
                };
                vec![match per_seq {
                    Some(p) => p,
                    None => self.pool.plan_step(&step).map_err(kv_err)?,
                }]
            }
        };
        let t_run = Instant::now();
        let n: usize = step.iter().map(|s| s.n_tokens).sum();
        let tokens: Vec<i32> = inputs
            .iter()
            .flat_map(|i| i.tokens.iter().copied())
            .collect();
        let mut times = RunTimes::default();
        match &mut self.engine {
            Engine::Sched(e) => e.run(
                &self.h,
                &plans[0],
                &tokens,
                self.reuse_step_graph,
                &mut self.step_graphs_built,
                out,
                &mut times,
            )?,
            Engine::Ranked(r) => r.run(
                &self.h,
                &plans,
                &tokens,
                self.reuse_step_graph,
                &mut self.step_graphs_built,
                self.telemetry.as_deref().map(Telemetry::metrics),
                out,
            )?,
        }
        if trace && (n > 1 || t0.elapsed().as_millis() >= 50) {
            // Milliseconds to 0.01, MiB to 0.1.
            let ms = |d: std::time::Duration| (d.as_secs_f64() * 1e5).round() / 100.0;
            let p = &plans[0];
            let mask_elems: usize = plans.iter().map(|p| p.mask.len()).sum();
            let faults = faults0
                .zip(thread_minor_faults())
                .map(|(a, b)| b.saturating_sub(a));
            tracing::info!(
                target: "tetraeder::trace",
                n_tokens = n,
                inputs = step.len(),
                n_kv = p.n_kv,
                per_seq = !p.seqs.is_empty(),
                kv_start = p.seqs.first().map_or(0, |s| s.kv_start),
                attn_rows = p.seqs.first().map_or(p.n_kv, |s| s.n_kv),
                mask_mib = (mask_elems as f64 * 2.0 / 104857.6).round() / 10.0,
                moved_rows = self.trace_moves.0,
                move_ms = ms(self.trace_moves.1),
                reserve_ms = ms(t_plan - t_reserve),
                plan_ms = ms(t_run - t_plan),
                state_ms = ms(times.state),
                upload_ms = ms(times.upload),
                build_ms = ms(times.build),
                compute_ms = ms(times.compute),
                read_ms = ms(times.read),
                total_ms = ms(t0.elapsed()),
                minflt = faults,
                "step"
            );
        }
        self.trace_moves = Default::default();
        for p in plans {
            self.pool.recycle(p);
        }

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

/// Where a scheduler step's time goes, for [`STEP_TRACE_ENV`]. `compute` waits for the device.
#[derive(Default)]
struct RunTimes {
    state: std::time::Duration,
    upload: std::time::Duration,
    build: std::time::Duration,
    compute: std::time::Duration,
    read: std::time::Duration,
}

impl SchedEngine {
    /// One step through the scheduler: the inputs, the kept or fresh whole-step graph, the
    /// logits of the output rows into `out`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        h: &DenseHparams,
        plan: &StepPlan,
        tokens: &[i32],
        reuse: bool,
        built: &mut u64,
        out: &mut Vec<f32>,
        times: &mut RunTimes,
    ) -> Result<()> {
        let t = Instant::now();
        let state_key = state_key(plan);
        let key = (state_key, plan.seqs.clone());
        if self.state.as_ref().is_none_or(|s| s.key != state_key) {
            // The kept graph holds the old state's context: drop it first, then the buffers.
            self.step = None;
            self.state = None;
            self.state = Some(alloc_state(state_key, &self.primary)?);
        }
        let st = self.state.as_ref().expect("state allocated above");
        let t_upload = Instant::now();
        st.tokens.get().set_i32(tokens)?;
        st.pos.get().set_i32(&plan.pos)?;
        st.rows.get().set_i32(&plan.rows)?;
        st.mask.get().set_f16_bits(&plan.mask)?;
        st.outputs.get().set_i32(&plan.outputs)?;
        let t_build = Instant::now();
        times.state = t_upload - t;
        times.upload = t_build - t_upload;

        let reusable = reuse
            && self
                .step
                .as_ref()
                .is_some_and(|s| s.key == key && self.sched.is_allocated(&s.graph.get()));
        let fresh;
        let graph: &StepGraph = if reusable {
            self.step.as_ref().expect("checked above")
        } else {
            self.step = None;
            let nodes = step_nodes(&h.attn, plan.seqs.len());
            let ctx = Rc::new(Context::new(nodes, nodes)?);
            let (graph, logits) = build_step(h, &self.weights, &self.kv, st, &plan.seqs, &ctx);
            let mut deps = vec![
                &self.weights.tok_embd,
                &self.kv.k[0],
                &self.kv.v[0],
                &st.tokens,
            ];
            deps.extend(self.kv.rot_k.iter().chain(&self.kv.rot_v));
            deps.extend(&self.weights.extra_anchors);
            fresh = StepGraph {
                key,
                logits: ctx.keep(logits)?,
                graph: ctx.keep_graph(graph, &deps)?,
            };
            self.sched.alloc_graph(&fresh.graph.get())?;
            *built += 1;
            if reuse {
                self.step.insert(fresh)
            } else {
                &fresh
            }
        };
        let t_compute = Instant::now();
        self.sched.compute(&graph.graph.get())?;
        let t_read = Instant::now();
        graph.logits.get().get_f32_into(out)?;
        times.build = t_compute - t_build;
        times.compute = t_read - t_compute;
        times.read = t_read.elapsed();
        Ok(())
    }
}

fn alloc_state(key: StateKey, backend: &Backend) -> Result<State> {
    let (n, _, n_out, mask) = key;
    let ctx = Rc::new(Context::new(8, 0)?);
    let state = State {
        key,
        tokens: ctx.new_persistent(Type::I32, &[n as i64]),
        pos: ctx.new_persistent(Type::I32, &[n as i64]),
        rows: ctx.new_persistent(Type::I32, &[n as i64]),
        mask: ctx.new_persistent(Type::F16, &[mask[0] as i64, mask[1] as i64]),
        outputs: ctx.new_persistent(Type::I32, &[n_out as i64]),
    };
    ctx.alloc_tensors(backend)?;
    // The mask padding is read by ggml-cuda (MASK_ROW_PAD) but never written: keep it finite.
    ctx.clear_buffer(0);
    Ok(state)
}

/// Loads the weights. On a CPU backend with `extra_bufts`, a matmul weight goes into the first
/// CPU extra buffer type that runs its matmul (AMX, repack), as libllama does with
/// `use_extra_bufts`; everything else into the backend's buffer. Returns the bytes of all buffers
/// and the extra buffer types used.
fn load_weights(
    gguf: &Gguf,
    h: &DenseHparams,
    backend: &Backend,
    extra_bufts: bool,
) -> Result<(Weights, usize, Vec<String>)> {
    let n_tensors = 3 + h.attn.n_layer as usize * 11;
    let extra = if extra_bufts && backend.is_cpu() {
        cpu::extra_bufts()
    } else {
        Vec::new()
    };
    // Context 0: the backend's buffer; context i + 1: extra buffer type i.
    let ctxs = (0..=extra.len())
        .map(|_| Context::new(n_tensors + 4, 0).map(Rc::new))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // (tensor, shard, offset, bytes, context index)
    let mut loaded: Vec<(OwnedTensor, usize, u64, u64, usize)> = Vec::new();
    let mut make = |name: &str, matmul: bool| -> Result<OwnedTensor> {
        let info = gguf.tensor(name)?;
        let ne: Vec<i64> = info.ne.iter().map(|&x| x as i64).collect();
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let at = if matmul && nd == 2 {
            extra
                .iter()
                .position(|b| b.supports_mul_mat(info.ty, [ne[0], ne[1]]))
                .map_or(0, |i| i + 1)
        } else {
            0
        };
        let t = ctxs[at].new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        if t.get().nbytes() as u64 != info.nbytes {
            return Err(ModelError::Invalid(format!("size mismatch for `{name}`")));
        }
        loaded.push((t.clone(), info.shard, info.offset, info.nbytes, at));
        Ok(t)
    };
    let tok_embd = make(names::TOK_EMBD, false)?;
    let out_norm = make(names::OUTPUT_NORM, false)?;
    let output = if h.tied_output {
        None
    } else {
        Some(make(names::OUTPUT, true)?)
    };
    let mut layers = Vec::new();
    for il in 0..h.attn.n_layer {
        let mut l = |w: &str, matmul: bool| make(&names::layer(il, w), matmul);
        layers.push(Layer {
            attn: AttnWeights {
                attn_norm: l("attn_norm", false)?,
                wq: l("attn_q", true)?,
                wk: l("attn_k", true)?,
                wv: l("attn_v", true)?,
                wo: l("attn_output", true)?,
                q_norm: l("attn_q_norm", false)?,
                k_norm: l("attn_k_norm", false)?,
            },
            ffn_norm: l("ffn_norm", false)?,
            gate: l("ffn_gate", true)?,
            up: l("ffn_up", true)?,
            down: l("ffn_down", true)?,
        });
    }
    let mut buffer_size = ctxs[0].alloc_weights(backend)?;
    let (mut used, mut extra_anchors) = (Vec::new(), Vec::new());
    for (i, (b, ctx)) in extra.iter().zip(&ctxs[1..]).enumerate() {
        if let Some(l) = loaded.iter().find(|l| l.4 == i + 1) {
            buffer_size += ctx.alloc_weights_extra(*b, backend)?;
            used.push(b.name());
            extra_anchors.push(l.0.clone());
        }
    }
    let mut shards = gguf
        .paths
        .iter()
        .map(std::fs::File::open)
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut buf = vec![0u8; 16 << 20];
    for (t, shard, offset, nbytes, at) in loaded {
        let f = &mut shards[shard];
        f.seek(SeekFrom::Start(offset))?;
        if at > 0 {
            // An extra buffer type converts a weight as it is written: in one piece.
            if buf.len() < nbytes as usize {
                buf.resize(nbytes as usize, 0);
            }
            f.read_exact(&mut buf[..nbytes as usize])?;
            t.get().set_bytes(&buf[..nbytes as usize], 0)?;
            continue;
        }
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
            extra_anchors,
        },
        buffer_size,
        used,
    ))
}

/// The whole step as one graph. Returns the graph and the logits `[n_vocab, n_outputs]`.
/// `seqs` empty: masked attention over rows `0..n_kv`; else attention per sequence.
fn build_step<'a>(
    h: &DenseHparams,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    seqs: &[SeqAttn],
    ctx: &'a Context,
) -> (Graph<'a>, Tensor<'a>) {
    let a = &h.attn;
    let mut g = ctx.new_graph(step_nodes(a, seqs.len()));
    let mut x = ctx.get_rows(w.tok_embd.get(), st.tokens.get());
    let (n_tokens, n_kv, n_out, _) = st.key;
    let mask = if seqs.is_empty() {
        mask_view(ctx, st.mask.get(), n_kv, n_tokens)
    } else {
        st.mask.get()
    };
    let step = AttnStep {
        pos: st.pos.get(),
        rows: st.rows.get(),
        mask,
        n_tokens,
        n_kv,
    };
    for (il, lw) in w.layers.iter().enumerate() {
        let mut ffn_inp = attention_seqs(a, ctx, &mut g, &lw.attn, kv, il, &step, seqs, x);
        if il + 1 == w.layers.len() && n_out < n_tokens {
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
        model_pool(n_seq, 512)
    }

    fn model_pool(n_seq: usize, n_ctx_pool: usize) -> Qwen3 {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3-f16.gguf");
        let opts = DenseOptions {
            n_seq_max: n_seq,
            n_ctx_pool,
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
        let Engine::Sched(e) = &m.engine else {
            panic!("the CPU backend uses the scheduler")
        };
        for (n, n_out) in [(1usize, 1usize), (5, 1), (5, 5)] {
            let st = alloc_state((n, KV_PAD, n_out, [KV_PAD, n]), &e.primary).unwrap();
            let nodes = step_nodes(&m.h.attn, 1);
            let ctx = Context::new(nodes, nodes).unwrap();
            let (g, _) = build_step(&m.h, &e.weights, &e.kv, &st, &[], &ctx);
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

    /// hardware: cpu. The step of the session 20261116pc CUDA crash on the tiny model: four
    /// sequences that held 533 positions keep a few (one keeps 406 and decodes), then 1207
    /// tokens in one step. ggml-cuda reads the mask in whole 64-row query tiles once a call has
    /// 1024 queries (`attn::MASK_ROW_PAD`), so the mask buffer must cover `pad(1207, 64)` rows;
    /// the padding stays zero, and each sequence's logits match it run alone.
    #[test]
    fn long_step_mask_covers_whole_query_tiles() {
        let n_vocab = model(1).n_vocab() as i32;
        let toks = |seed: usize, n: usize| -> Vec<i32> {
            (0..n)
                .map(|i| ((seed * 7919 + i * 31 + 1) as i32) % n_vocab)
                .collect()
        };
        let keep = [406usize, 3, 4, 3];
        let step_len = [1usize, 401, 402, 403];
        let history: Vec<Vec<i32>> = (0..4).map(|s| toks(s, 533)).collect();
        let step: Vec<Vec<i32>> = (0..4).map(|s| toks(s + 10, step_len[s])).collect();
        fn last(seq: u32, tokens: &[i32]) -> SeqEval<'_> {
            SeqEval {
                seq,
                tokens,
                all_logits: false,
            }
        }

        let mut m = model_pool(4, 4096);
        let mut out = Vec::new();
        for (s, h) in history.iter().enumerate() {
            m.decode(&[last(s as u32, h)], &mut out).unwrap();
        }
        for (s, &k) in keep.iter().enumerate() {
            m.seq_rm(s as u32, k).unwrap();
        }
        let inputs: Vec<SeqEval<'_>> = (0..4).map(|s| last(s as u32, &step[s])).collect();
        let ranges = m.decode(&inputs, &mut out).unwrap();
        let n: usize = step_len.iter().sum();
        assert_eq!(n, 1207);

        let Engine::Sched(e) = &m.engine else {
            panic!("the CPU backend uses the scheduler")
        };
        let st = e.state.as_ref().expect("a step ran");
        let (n_tokens, n_kv, _, _) = st.key;
        assert_eq!(n_tokens, n);
        // The widest query tile of ggml-cuda's flash attention at the pin.
        let rows = pad(n, 64);
        let mask = st.mask.get();
        assert!(
            mask.nbytes() >= rows * n_kv * 2,
            "mask buffer of {} bytes, the CUDA scan reads {} ({rows} rows of {n_kv})",
            mask.nbytes(),
            rows * n_kv * 2
        );
        let tail = mask.get_bytes(n * n_kv * 2, (rows - n) * n_kv * 2).unwrap();
        assert!(tail.iter().all(|&b| b == 0), "mask padding is not zero");

        let nv = m.n_vocab();
        let top = out.iter().fold(0f32, |a, x| a.max(x.abs()));
        for (s, r) in ranges.iter().enumerate() {
            let mut alone = model_pool(1, 1024);
            let mut want = Vec::new();
            alone
                .decode(&[last(0, &history[s][..keep[s]])], &mut want)
                .unwrap();
            alone.decode(&[last(0, &step[s])], &mut want).unwrap();
            let got = &out[r.start * nv..r.end * nv];
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            // The masked path's batch dependence (dense_per_seq.rs): loosely bounded.
            assert!(worst <= 1e-2 * top, "sequence {s}: max |delta| {worst:e}");
        }
    }

    fn model_single(n_seq: usize, n_ctx_pool: usize, per_seq_single: bool) -> Qwen3 {
        let mut m = model_pool(n_seq, n_ctx_pool);
        m.per_seq_attn = false;
        m.per_seq_attn_single = per_seq_single;
        m
    }

    fn eval(m: &mut Qwen3, seq: u32, tokens: &[i32], all_logits: bool) -> Vec<f32> {
        let mut out = Vec::new();
        m.decode(
            &[SeqEval {
                seq,
                tokens,
                all_logits,
            }],
            &mut out,
        )
        .unwrap();
        out
    }

    fn tokens(seed: usize, n: usize, n_vocab: usize) -> Vec<i32> {
        (0..n)
            .map(|i| ((seed * 7919 + i * 31 + 1) % n_vocab) as i32)
            .collect()
    }

    /// hardware: cpu. A step with one input attends over that input's run only, not over the
    /// rows of the sequence below it (ttft5-20261119: 1.35x to 1.45x slower decode and up to
    /// 0.4 s more prefill above a 22k-row run; libllama's unified cache reads the sequence's own
    /// cells). Its logits are those of the sequence alone at row 0, bit for bit.
    #[test]
    fn single_input_attends_over_its_own_run() {
        let mut m = model_single(2, 4096, true);
        let nv = m.n_vocab();
        let (below, own) = (tokens(1, 300, nv), tokens(2, 100, nv));
        eval(&mut m, 0, &below, false);
        let mut got = eval(&mut m, 1, &own[..60], true);
        got.extend(eval(&mut m, 1, &own[60..], false));
        assert_eq!(
            m.pool().row(1, 0),
            Some(512),
            "sequence 1 sits above sequence 0"
        );
        let Engine::Sched(e) = &m.engine else {
            panic!("the CPU backend uses the scheduler")
        };
        let (n_tokens, n_kv, _, mask) = e.state.as_ref().expect("a step ran").key;
        assert_eq!(n_tokens, 40);
        assert_eq!(n_kv, 0, "attention per sequence, not over rows 0..n_kv");
        assert_eq!(mask, [256 * 40 + mask_tail(40, 256), 1], "its own 256 rows");

        let mut alone = model_single(1, 1024, true);
        let mut want = eval(&mut alone, 0, &own[..60], true);
        want.extend(eval(&mut alone, 0, &own[60..], false));
        assert_eq!(got, want, "bit-equal to the sequence alone");
    }

    /// hardware: cpu. For a sequence at row 0 attention over its own run is the masked
    /// attention over rows 0..n_kv (same rows, same mask values): logits bit for bit, prefill,
    /// chunks with every logit, decodes and a step across a 256-row boundary.
    #[test]
    fn single_input_at_row_zero_matches_the_masked_path() {
        let mut logits = Vec::new();
        for per_seq_single in [false, true] {
            let mut m = model_single(1, 1024, per_seq_single);
            let toks = tokens(3, 300, m.n_vocab());
            let mut out = eval(&mut m, 0, &toks[..250], false);
            out.extend(eval(&mut m, 0, &toks[250..290], true));
            for t in &toks[290..] {
                out.extend(eval(&mut m, 0, std::slice::from_ref(t), false));
            }
            logits.push(out);
        }
        assert_eq!(logits[0], logits[1]);
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

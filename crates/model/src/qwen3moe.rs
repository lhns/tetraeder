//! Qwen3 MoE graph builder (ADR 0030), op for op like libllama's `qwen3moe` graph.
//!
//! With every expert resident a step is one graph (ADR 0036). Otherwise it runs as `n_layer + 1`
//! graph segments. Segment `l` finishes layer `l - 1` (expert FFN with the host-mapped expert
//! ids, residual) and starts layer `l` (attention, router, top-k); it ends at the top-k. The
//! host reads the ids, maps experts to slots and passes the mapped ids to the next segment
//! through a state tensor. Activations between segments stay in backend memory, in two
//! alternating state slots.
//!
//! With fewer expert slots than experts (ADR 0033) the expert tensors are slot tensors; between
//! segments the [`Streamer`] makes the routed experts resident and the host writes slot ids.
//! When a layer step routes to more distinct experts than there are slots, its (token, choice)
//! pairs are split into pair groups, one per chunk of resident experts (ADR 0035). A group
//! gathers the inputs of its pairs, runs the expert matmuls on them and scatters the unweighted
//! outputs into the slot's `acc`. The last group runs inside the next segment, which then
//! weights and sums `acc` with the same ops as the resident graph.
//!
//! With `MissPolicy::Cpu` the experts not resident in the compute slots form pair groups over the
//! RAM slot tensors, which sit in a CPU buffer marked as weights, so the scheduler computes them
//! on the CPU. Those groups and the group of the compute-slot hits share the next segment.

use crate::attn::{AttnStep, AttnWeights, KV_PAD, Kv, SeqEval, alloc_kv, attention, copy_rows};
use crate::convert::{ConvertError, ConvertOptions, ConvertStats};
use crate::cpu_poll::{self, CpuPoll};
use crate::experts::{ExpertStats, MissPolicy, RoutingStep, StagedLayer, Streamer, StreamerConfig};
use crate::gguf::GgufError;
use crate::hparams::{Hparams, ModelFile, names};
use crate::qwen3moe_shard::{self, Sharded};
use crate::upload::AdmitUpload;
use tetraeder_ggml_sys::ggml::{
    Backend, Context, GgmlError, Graph, HugepageAdvice, OwnedGraph, OwnedTensor, Sched, Tensor,
    Type,
};
use tetraeder_kvcache::{Chunked, KvError, KvPool, RowCopy, StepInput, StepPlan};
use tetraeder_memory::IoOptions;
use tetraeder_memory::io::IoFactory;
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_plan::{Format, KvFormat, ModelDesc, Topology};
use tetraeder_telemetry::{Phase, StepSample, Telemetry};

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
    #[error(transparent)]
    Convert(#[from] ConvertError),
    #[error("cannot read weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

/// Where the wall time of `eval` went, summed over steps since load or the last
/// [`Qwen3Moe::take_profile`]. Host-side timers only; `compute` includes waiting for the backend.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepProfile {
    pub steps: u64,
    /// Graphs computed: segments plus standalone group graphs.
    pub graphs: u64,
    /// Step inputs (tokens, positions, mask) and state allocation.
    pub inputs: Duration,
    /// Context creation and graph building.
    pub build: Duration,
    /// `sched.alloc_graph` and the group index writes.
    pub alloc: Duration,
    /// `sched.compute` of the segments and reading the logits.
    pub compute: Duration,
    /// Reading the router ids (and weights when recording) and writing the mapped ids.
    pub readback: Duration,
    /// Streamer work between segments: planning, staging, uploads and standalone group graphs.
    pub streamer: Duration,
    pub total: Duration,
    /// Overlapped decode (ADR 0054), all inside `streamer` so `total` still adds up. The CPU
    /// experts of layer steps with reads in flight: the first group graph (the experts already
    /// resident) and the later ones (experts computed as they land); the time the main thread
    /// blocked waiting for an expert; and the host copies around them (`xn` out, the group
    /// outputs into the accumulator's source).
    pub cpu_ready: Duration,
    pub read_wait: Duration,
    pub cpu_landed: Duration,
    pub copy: Duration,
    /// `sched.compute` of the segments only (not the logits readback), split by whether the
    /// segment runs a RAM-tier expert group (the CPU computes it) or not, with the segment
    /// counts. Both are part of `compute`.
    pub seg_compute_cpu: Duration,
    pub seg_compute_gpu: Duration,
    pub segs_cpu: u64,
    pub segs_gpu: u64,
    /// Routed (token, expert) pairs computed on the CPU from the RAM tier (`MissPolicy::Cpu`).
    pub cpu_pairs: u64,
    /// The part of `cpu_pairs` computed inside segment graphs, as opposed to group graphs and
    /// overlap groups (ADR 0064, step 0).
    pub seg_pairs: u64,
}

/// One buffer the builder owns (ADR 0037).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryUse {
    pub device: String,
    /// `vram` for the primary backend's memory, `ram` for host memory (also the primary
    /// backend's when it is the CPU).
    pub tier: &'static str,
    /// `weights`, `slots`, `kv` or `workspace`.
    pub kind: &'static str,
    /// The KV format (`q8_0/q4_0`) of the `kv` buffer.
    pub format: Option<String>,
    /// Size of the buffer.
    pub capacity: u64,
    /// Bytes holding data: for slots the resident experts, otherwise the whole buffer (the
    /// workspace is as big as the largest graph so far).
    pub used: u64,
}

/// Slots per layer for a tier from its count or its byte budget (ADR 0037).
fn resolve_tier(
    file: &ModelFile,
    tier: &str,
    count: Option<usize>,
    bytes: Option<u64>,
) -> Result<Option<usize>> {
    let Some(bytes) = bytes else {
        return Ok(count);
    };
    if count.is_some() {
        return Err(ModelError::Invalid(format!(
            "set {tier}_slots or {tier}_slot_bytes, not both"
        )));
    }
    let n = file.slots_for_bytes(bytes)?;
    if n == 0 {
        let min: u64 = file.expert_bytes_per_layer()?.iter().sum();
        return Err(ModelError::Invalid(format!(
            "{tier}_slot_bytes {bytes} buys no slot; one slot per layer needs {min} bytes"
        )));
    }
    Ok(Some(n as usize))
}

/// Where the weights, the KV cache and the compute run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// CPU with this many threads (`None`: the available parallelism).
    Cpu { threads: Option<usize> },
    /// The `index`-th GPU (CUDA or Vulkan, whatever the build has); the CPU is the fallback.
    Gpu { index: usize },
}

#[derive(Clone)]
pub struct LoadOptions {
    pub backend: BackendChoice,
    /// Threads of the CPU backend: the whole compute for `BackendChoice::Cpu`, the RAM tier and
    /// CPU groups under `Gpu`. `None`: `BackendChoice::Cpu { threads }` if given, else the
    /// available parallelism. When set it wins over `Cpu { threads }`.
    pub cpu_threads: Option<usize>,
    /// KV cache length in tokens (rounded up to a multiple of 256).
    pub n_ctx: usize,
    /// Expert slots per MoE layer on the compute backend (ADR 0033). `None`, or at least the
    /// expert count: every expert is loaded at load time and stays resident. Fewer: experts
    /// are read from the GGUF on demand.
    pub expert_slots: Option<usize>,
    /// Host-memory expert slots per layer between the file and the compute slots. Only with
    /// fewer compute slots than experts.
    pub ram_slots: Option<usize>,
    /// Compute-tier slot bytes instead of `expert_slots` (ADR 0037): the budget buys
    /// `floor(bytes / sum over layers of one expert's bytes)` slots per layer. This is the
    /// tier's slot bytes only, not weights, KV cache or compute buffers. At least the expert
    /// count's worth means every expert is resident. Exclusive with `expert_slots`.
    pub expert_slot_bytes: Option<u64>,
    /// RAM-tier slot bytes instead of `ram_slots`, counted like `expert_slot_bytes`.
    pub ram_slot_bytes: Option<u64>,
    /// Direct I/O settings for expert reads.
    pub io: IoOptions,
    /// Step and expert cache metrics go here when set.
    pub telemetry: Option<Arc<Telemetry>>,
    /// Keep the router's choices (see [`Qwen3Moe::take_routing_log`]); off by default.
    pub record_routing: bool,
    /// What a routed expert that is not in a compute slot costs: an upload or a CPU group
    /// (ADR 0035). `Cpu` needs `ram_slots`.
    pub miss: MissPolicy,
    /// Routed pairs per layer and expert from an earlier run ([`Qwen3Moe::routing_counts`]).
    /// With `MissPolicy::Cpu` the hot set is built from it at load; without one, after the
    /// first multi-token step.
    pub routing_profile: Option<Vec<Vec<u64>>>,
    /// Passed to ggml's scheduler, which then runs large-batch ops whose weights sit in host
    /// memory on the GPU, copying the weights per op, as llama.cpp does by default. Only
    /// matters with a GPU backend.
    pub op_offload: bool,
    /// With `op_offload`, the fewest tokens a streamed step needs to offload (ADR 0044). ggml
    /// counts a pair group's rows (tokens times experts) as its batch, so without this gate a
    /// 7-token step at top-8 would copy every routed RAM expert to the GPU. Default 32, the
    /// CUDA threshold applied to tokens, as libllama gets it.
    pub op_offload_min_tokens: usize,
    /// Keep the whole-step graph and its scheduler allocation across steps with the same token
    /// count, padded KV length, output count and routing recording, so ggml-cuda can replay a
    /// captured CUDA graph (ADR 0036). Only with every expert resident; on by default (30B decode
    /// 160 to 203 tok/s on the RTX 3090, greedy identical).
    pub reuse_step_graph: bool,
    /// With a streamer, size the schedulers for the whole-step graph as before ADR 0064 step 1
    /// instead of one segment. Only for the dispatch benchmark and its parity test; off by
    /// default.
    pub whole_step_sched: bool,
    /// With `MissPolicy::Cpu` and a RAM tier: predict each layer's experts from an earlier
    /// layer and read the predicted ones into the RAM tier while earlier layers compute (ADR
    /// 0038). Otherwise no effect. On by default (235B at 32K: decode 2.83 to 3.14 tok/s, greedy
    /// identical).
    pub prefetch: bool,
    /// Target format per GGUF tensor name (ADR 0047). A tensor whose target differs from its
    /// type in the file is converted at load, through the disk cache; empty loads the file as is.
    /// Only downward conversions; the router, norms and formats that need an importance matrix
    /// are refused.
    pub formats: BTreeMap<String, Format>,
    /// Conversion cache directory; `None`: `$TETRAEDER_CACHE_DIR`, else `.tetraeder-cache` next
    /// to the model's first shard.
    pub convert_cache: Option<PathBuf>,
    /// Layers ahead that `prefetch` predicts and reads, 1 or 2 (ADR 0038). Depth 2 also
    /// predicts layer L+2 from layer L's FFN input, tops up L+1 from the fresher depth-1
    /// prediction and reads L+2 meanwhile.
    pub prefetch_depth: usize,
    /// Polling level (0 to 100) of the CPU backend's ggml threadpool: how long idle workers
    /// spin before they sleep. `None`: auto (ADR 0061), 0 when the cgroup CPU quota binds (below
    /// the host's online CPUs, or at most as many whole cores as threads), else ggml's default
    /// (50). [`Qwen3Moe::cpu_poll`] returns the level in effect.
    pub cpu_poll: Option<u32>,
    /// Keep the polling level but put the CPU threadpool to sleep where no CPU graph follows
    /// soon: before a GPU-only segment and before waiting on expert reads (ADR 0061 addendum).
    /// Off by default; the logits do not change.
    pub pool_pause: bool,
    /// Compute threads of CPU graphs while a prefetch read is in flight (at most the full
    /// count, which the threadpool keeps). `None`: the full count.
    pub prefetch_threads: Option<usize>,
    /// Format of the K and V cache of every layer (ADR 0050 b). Quantized formats need a head
    /// size that is a multiple of their block size (32). Per-layer formats are not supported.
    pub kv: KvFormat,
    /// Decode computes the CPU experts of a layer in arrival order (ADR 0054): the resident ones
    /// first, the others as their reads land, instead of waiting for the whole layer's reads.
    /// Needs `io.service`. Applies to `MissPolicy::Cpu` steps without op offload that have one
    /// CPU chunk; other steps run as without it. Logits are bitwise unchanged.
    pub overlap: bool,
    /// With `overlap`, a layer step submits its own misses (urgent) before the prefetch of the
    /// next layers, so the service's workers take them first (ADR 0054 addendum). Off: the
    /// prefetch goes first, as measured on the 235B. Cache behaviour and logits are unchanged.
    pub urgent_first: bool,
    /// Test only: builds the expert reads in place of the I/O service, for example a simulator
    /// on a virtual clock (`tetraeder_memory::io::sim::SimIo`), so reads are still in flight
    /// when a layer is planned without any sleeping.
    #[doc(hidden)]
    pub io_factory: Option<IoFactory>,
    /// Hadamard rotation of Q, K, V and the attention output, as libllama does for a quantized
    /// cache (ADR 0050 addendum). `None`: on exactly when the format is quantized (libllama's
    /// default), off for F16. `Some(true)` needs a head size that is a multiple of 64.
    pub kv_rotation: Option<bool>,
    /// Sequences the KV pool serves (ADR 0065); `n_ctx` is the rows they share. Default 1.
    pub n_seq_max: usize,
    /// With `MissPolicy::Cpu`: the share (0 to 1) of a decode step's compute-tier misses that
    /// are uploaded into a compute slot after their CPU group is staged, evicting the least
    /// recently used unpinned slot; the hot set is then the tier's initial contents, unpinned
    /// (ADR 0059). `None` or 0: the static hot set. The step's groups are unchanged, so logits
    /// are too (bitwise on the CPU backend). Otherwise no effect.
    pub admit_share: Option<f64>,
    /// With `admit_share`: how an admitted expert is copied into its compute slot (ADR 0059
    /// addendum 2026-11-12b). `Sync` (default): a blocking copy straight from the RAM slot.
    /// `Async`: through a pinned staging ring on a stream of its own, waited for only before the
    /// layer's compute slots are next used. Logits are the same in both.
    pub admit_upload: AdmitUpload,
    /// Advise the kernel to back the RAM tier's host buffer with transparent huge pages
    /// (`MADV_HUGEPAGE`, Linux, before first touch), against TLB misses of the CPU expert
    /// matmuls over a tier of tens of GB (ADR 0035 addendum). No effect on results. Default off.
    pub ram_hugepages: bool,
    /// RAM-tier policy (ADR 0050 (c)): `Lru` (default), or `Freq { share }`, which pins the top
    /// share of each layer's RAM slots by routing count during decode, refreshed every 16 decode
    /// steps and at each hot-set refresh. Fewer disk reads per token on a slow disk; results are
    /// unchanged (bitwise on the CPU backend). Needs `ram_slots`.
    pub ram_policy: tetraeder_memory::RamPolicy,
    /// Run only the first N layers plus the head (calibration anchors, ADR 0069): the logits are
    /// meaningless, the timing is real. Only for [`Qwen3Moe::load_sharded`].
    pub layer_limit: Option<u32>,
}

impl std::fmt::Debug for LoadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadOptions")
            .field("backend", &self.backend)
            .field("cpu_threads", &self.cpu_threads)
            .field("n_ctx", &self.n_ctx)
            .field("expert_slots", &self.expert_slots)
            .field("ram_slots", &self.ram_slots)
            .field("expert_slot_bytes", &self.expert_slot_bytes)
            .field("ram_slot_bytes", &self.ram_slot_bytes)
            .field("miss", &self.miss)
            .field("telemetry", &self.telemetry.is_some())
            .field("op_offload", &self.op_offload)
            .field("op_offload_min_tokens", &self.op_offload_min_tokens)
            .field("reuse_step_graph", &self.reuse_step_graph)
            .field("prefetch", &self.prefetch)
            .field("formats", &self.formats.len())
            .field("prefetch_depth", &self.prefetch_depth)
            .field("cpu_poll", &self.cpu_poll)
            .field("pool_pause", &self.pool_pause)
            .field("prefetch_threads", &self.prefetch_threads)
            .field("kv", &self.kv)
            .field("io", &self.io)
            .field("overlap", &self.overlap)
            .field("urgent_first", &self.urgent_first)
            .field("io_factory", &self.io_factory.is_some())
            .field("kv_rotation", &self.kv_rotation)
            .field("n_seq_max", &self.n_seq_max)
            .field("admit_share", &self.admit_share)
            .field("admit_upload", &self.admit_upload)
            .field("ram_hugepages", &self.ram_hugepages)
            .field("ram_policy", &self.ram_policy)
            .field("layer_limit", &self.layer_limit)
            .finish_non_exhaustive()
    }
}

impl LoadOptions {
    pub fn new(backend: BackendChoice) -> Self {
        Self {
            backend,
            cpu_threads: None,
            n_ctx: 2048,
            expert_slots: None,
            ram_slots: None,
            expert_slot_bytes: None,
            ram_slot_bytes: None,
            io: IoOptions::default(),
            telemetry: None,
            record_routing: false,
            miss: MissPolicy::Upload,
            routing_profile: None,
            op_offload: false,
            op_offload_min_tokens: 32,
            reuse_step_graph: true,
            whole_step_sched: false,
            formats: BTreeMap::new(),
            convert_cache: None,
            prefetch: true,
            prefetch_depth: 1,
            cpu_poll: None,
            pool_pause: false,
            prefetch_threads: None,
            kv: KvFormat::default(),
            overlap: false,
            urgent_first: false,
            io_factory: None,
            kv_rotation: None,
            n_seq_max: 1,
            admit_share: None,
            admit_upload: AdmitUpload::Sync,
            ram_hugepages: false,
            ram_policy: tetraeder_memory::RamPolicy::Lru,
            layer_limit: None,
        }
    }
}

/// Upper bound of nodes in one segment; the scheduler's hash tables are sized for it.
const SEGMENT_NODES: usize = 1024;
/// Upper bound of nodes per layer in a whole-step graph (about 70 are used).
const STEP_NODES_PER_LAYER: usize = 128;

fn kv_err(e: KvError) -> ModelError {
    ModelError::Invalid(e.to_string())
}

fn pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

pub(crate) struct Layer {
    pub(crate) attn: AttnWeights,
    pub(crate) ffn_norm: OwnedTensor,
    pub(crate) gate_inp: OwnedTensor,
    /// Stacked expert weights, or slot tensors with fewer entries when streaming.
    pub(crate) gate_exps: OwnedTensor,
    pub(crate) up_exps: OwnedTensor,
    pub(crate) down_exps: OwnedTensor,
}

impl Layer {
    fn exps(&self) -> [Tensor<'_>; 3] {
        [
            self.gate_exps.get(),
            self.up_exps.get(),
            self.down_exps.get(),
        ]
    }

    fn exps_owned(&self) -> [OwnedTensor; 3] {
        [
            self.gate_exps.clone(),
            self.up_exps.clone(),
            self.down_exps.clone(),
        ]
    }
}

struct Weights {
    tok_embd: OwnedTensor,
    out_norm: OwnedTensor,
    /// `None` when the output projection is tied to the token embedding.
    output: Option<OwnedTensor>,
    layers: Vec<Layer>,
}

/// Activations that cross segment boundaries, one slot per layer parity.
struct Slot {
    resid: OwnedTensor,
    xn: OwnedTensor,
    /// Normalized router weights `[1, n_used, n]`.
    w: OwnedTensor,
    /// Top-k expert ids as the router chose them `[n_used, n]`, read by the host.
    ids: OwnedTensor,
    /// The ids the expert matmuls use, written by the host.
    ids_map: OwnedTensor,
    /// The next layer's predicted top-k ids `[n_used, n]` (ADR 0038), with prefetch on.
    pred: OwnedTensor,
    /// Layer L+2's, with prefetch depth 2.
    pred2: OwnedTensor,
    /// Unweighted expert outputs per (token, choice), filled by pair groups (ADR 0035).
    acc: OwnedTensor,
    /// The CPU experts' outputs of an overlapped layer step, in the order they were computed
    /// (ADR 0054), `[n_embd, n_used * n_tokens]`; with overlap only.
    cpu_out: Option<OwnedTensor>,
}

/// Host buffers of overlapped decode, in CPU memory: the normed hidden state the CPU groups
/// read and their output rows.
struct HostBufs {
    xn: OwnedTensor,
    out: OwnedTensor,
}

/// Per-step inputs and slots; reallocated when the token count or the padded KV length changes.
struct State {
    n_tokens: usize,
    n_kv: usize,
    /// Tokens whose logits the step returns.
    n_out: usize,
    tokens: OwnedTensor,
    pos: OwnedTensor,
    /// KV rows the step writes (`set_rows` indices).
    rows: OwnedTensor,
    mask: OwnedTensor,
    /// Indices of the tokens whose logits are returned.
    outputs: OwnedTensor,
    slots: [Slot; 2],
    /// The last layer's slot, sized by `n_out`, when fewer rows than tokens have outputs: that
    /// layer's router and expert FFN run on the output rows only, as in the whole-step graph.
    last: Option<Slot>,
    host: Option<HostBufs>,
}

impl State {
    /// The slot layer `il` writes and the next segment reads.
    fn slot(&self, il: usize, n_layer: usize) -> &Slot {
        match &self.last {
            Some(s) if il + 1 == n_layer => s,
            _ => &self.slots[il % 2],
        }
    }
}

/// A step state with the whole-step graph kept for it (`LoadOptions::reuse_step_graph`). The
/// graph reads the state, so it is declared first and dropped first.
struct CachedStep {
    step: Option<StepGraph>,
    /// The graph is allocated on `sched_alt` instead of `sched`: a scheduler holds one allocation,
    /// so the two most recent graphs need one each.
    alt: bool,
    state: State,
}

/// A whole-step graph kept across steps. It holds the contexts of the weights, KV cache and state
/// it reads, so the state must be dropped after it.
struct StepGraph {
    /// Token count, padded KV length, output count and routing recording of the build.
    key: (usize, usize, usize, bool),
    graph: OwnedGraph,
    logits: OwnedTensor,
    /// Per layer, the router's ids and weights (with routing recording).
    routed: Vec<(OwnedTensor, OwnedTensor)>,
}

/// A Qwen3 MoE model. The experts are all resident in backend memory or streamed through slots.
pub struct Qwen3Moe {
    file: ModelFile,
    sched: Sched,
    /// The second scheduler of the two-entry step cache (built when first needed, from
    /// `sched_proto`).
    sched_alt: Option<Sched>,
    sched_proto: (Vec<Backend>, usize, bool),
    /// The scheduler without op offload for small streamed steps (ADR 0044), with `op_offload`
    /// on a GPU backend.
    sched_small: Option<Sched>,
    /// CPU-only scheduler for the group graphs of overlapped decode (ADR 0054).
    sched_cpu: Option<Sched>,
    primary: Backend,
    cpu: Backend,
    /// Size of the weights buffer on the primary backend (slot tensors included) and of the KV
    /// buffer.
    weights_bytes: u64,
    kv_bytes: u64,
    weights: Weights,
    kv: Kv,
    /// Most recent first: with every expert resident, `reuse_step_graph` and the CPU as primary,
    /// the two most recent step shapes (a CPU step alternates decoders and a prompt chunk);
    /// otherwise one.
    steps: Vec<CachedStep>,
    pool: KvPool,
    n_seq_max: usize,
    reuse_step_graph: bool,
    step_graphs_built: u64,
    segments_run: u64,
    groups_run: u64,
    group_graphs_run: u64,
    max_splits: usize,
    policy: MissPolicy,
    /// Overlapped decode is possible: the option is on and the model streams through a RAM tier
    /// under the cpu miss policy.
    overlap: bool,
    /// `LoadOptions::urgent_first`.
    urgent_first: bool,
    /// Each segment also predicts the experts of this many layers ahead (ADR 0038), 0 for none:
    /// only with prefetch, the cpu miss policy and a RAM tier, where the reads can be prefetched.
    predict_depth: usize,
    /// Threads of the CPU backend, and the count while a prefetch read is in flight.
    cpu_threads: i32,
    prefetch_threads: Option<i32>,
    cpu_poll: CpuPoll,
    pool_pause: bool,
    /// Pauses already reported to telemetry.
    pauses_reported: u64,
    /// Pair groups may run on a GPU backend (the primary is not the CPU).
    gpu: bool,
    op_offload: bool,
    op_offload_min_tokens: usize,
    /// The hot set still has to be built from the first prefill's routing.
    hot_pending: bool,
    /// The current `decode` call contains a prompt chunk (see [`Qwen3Moe::decode_group`]).
    call_prefill: bool,
    streamer: Option<Streamer>,
    metrics: MemoryMetrics,
    telemetry: Option<Arc<Telemetry>>,
    routing: Option<Vec<RoutingStep>>,
    profile: StepProfile,
    convert: ConvertStats,
    /// Set by [`Qwen3Moe::load_sharded`]: the weights, KV tensors and steps live on the
    /// executor's ranks, and the fields above (scheduler, weights, KV) are placeholders.
    shard: Option<Box<Sharded>>,
}

impl std::fmt::Debug for Qwen3Moe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Moe")
            .field("hparams", &self.file.hparams)
            .field("backend", &self.primary)
            .field("n_past", &self.n_past())
            .finish()
    }
}

/// Adds the time since `t` to `acc` and returns now.
fn lap(acc: &mut Duration, t: Instant) -> Instant {
    let now = Instant::now();
    *acc += now - t;
    now
}

impl Qwen3Moe {
    pub fn load(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Qwen3Moe> {
        let mut file = ModelFile::open(path.as_ref())?;
        // `cpu` holds the RAM tier and runs the CPU groups.
        let threads_for = |own: Option<usize>| {
            opts.cpu_threads
                .or(own)
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
        };
        let convert_threads = match opts.backend {
            BackendChoice::Cpu { threads } => threads_for(threads),
            BackendChoice::Gpu { .. } => threads_for(None),
        };
        let convert = crate::convert::convert(
            &mut file.gguf,
            &opts.formats,
            &ConvertOptions {
                cache: opts.convert_cache.clone(),
                threads: convert_threads,
                telemetry: opts.telemetry.clone(),
                chunk_rows: None,
            },
        )?;
        let h = &file.hparams;
        if h.n_head % h.n_head_kv != 0 || h.n_expert_used > h.n_expert || h.n_rot != h.head_dim {
            return Err(ModelError::Invalid("unsupported hyperparameters".into()));
        }
        let expert_slots =
            resolve_tier(&file, "expert", opts.expert_slots, opts.expert_slot_bytes)?;
        let ram_slots = resolve_tier(&file, "ram", opts.ram_slots, opts.ram_slot_bytes)?;
        if !(1..=2).contains(&opts.prefetch_depth) {
            return Err(ModelError::Invalid(format!(
                "prefetch_depth must be 1 or 2, not {}",
                opts.prefetch_depth
            )));
        }
        if opts.overlap && !opts.io.service {
            return Err(ModelError::Invalid("overlap needs io.service".into()));
        }
        if opts.miss == MissPolicy::Cpu && ram_slots.is_none() {
            return Err(ModelError::Invalid(
                "the cpu miss policy needs ram_slots".into(),
            ));
        }
        let (n, gpu) = match opts.backend {
            BackendChoice::Cpu { threads } => (threads_for(threads), None),
            BackendChoice::Gpu { index } => (threads_for(None), Some(Backend::gpu(index)?)),
        };
        let mut cpu_poll = cpu_poll::detect(opts.cpu_poll, n);
        let cpu = Backend::cpu_with_poll(n as i32, Some(cpu_poll.level))?;
        cpu_poll.level = cpu.cpu_poll().unwrap_or(cpu_poll.level);
        tracing::info!(
            level = cpu_poll.level,
            auto = cpu_poll.auto,
            quota = ?cpu_poll.quota,
            online = ?cpu_poll.online,
            threads = n,
            "cpu threadpool polling"
        );
        if let Some(t) = &opts.telemetry {
            t.metrics().cpu_poll(cpu_poll.level, cpu_poll.auto);
        }
        let cpu_threads = n as i32;
        let (primary, backends) = match gpu {
            None => (cpu.clone(), vec![cpu.clone()]),
            Some(gpu) => (gpu.clone(), vec![gpu, cpu.clone()]),
        };
        let slots = match expert_slots {
            Some(0) => {
                return Err(ModelError::Invalid(
                    "expert_slots must be at least 1".into(),
                ));
            }
            Some(n) if n < h.n_expert as usize => Some(n),
            _ => None,
        };
        // With a streamer every graph is one segment (ADR 0064, step 1): a small scheduler
        // keeps `ggml_backend_sched_reset` at about 82 KB instead of 1.3 MB per segment.
        let sched_nodes = if slots.is_some() && !opts.whole_step_sched {
            SEGMENT_NODES * 4
        } else {
            step_nodes(h).max(SEGMENT_NODES) * 4
        };
        let sched = Sched::new(&backends, sched_nodes, opts.op_offload)?;
        let sched_proto = (backends.clone(), sched_nodes, opts.op_offload);
        // Streamed steps below `op_offload_min_tokens` use a scheduler without op offload
        // (ADR 0044); its compute buffers are sized by those small steps.
        let sched_small = if opts.op_offload && !backends[0].is_cpu() {
            Some(Sched::new(&backends, sched_nodes, false)?)
        } else {
            None
        };
        let (weights, weights_bytes) = load_weights(&file, &primary, slots)?;
        if opts.n_seq_max == 0 {
            return Err(ModelError::Invalid("n_seq_max must be at least 1".into()));
        }
        let n_seq_max = opts.n_seq_max;
        let kv_rows = pad(opts.n_ctx.max(1), KV_PAD);
        let (kv, kv_bytes) = alloc_kv(&h.attn(), kv_rows, &primary, opts.kv, opts.kv_rotation)?;
        let metrics = MemoryMetrics::new(opts.telemetry.clone());
        let streamer = slots
            .map(|n| {
                let tensors: Vec<_> = weights.layers.iter().map(Layer::exps_owned).collect();
                let cfg = StreamerConfig {
                    slots: n,
                    ram_slots: ram_slots.map(|r| r.max(1)),
                    io: opts.io,
                    profile: opts.routing_profile.clone(),
                    prefetch: opts.prefetch,
                    overlap: opts.overlap,
                    io_factory: opts.io_factory.clone(),
                    admit_share: opts
                        .admit_share
                        .filter(|&s| s > 0.0 && opts.miss == MissPolicy::Cpu),
                    admit_upload: opts.admit_upload,
                    ram_hugepages: opts.ram_hugepages,
                    ram_policy: opts.ram_policy,
                };
                Streamer::new(
                    &file,
                    &tensors,
                    &primary,
                    &cpu,
                    cfg,
                    metrics.clone(),
                    primary.name().to_string(),
                )
            })
            .transpose()?;
        tracing::debug!(
            backend = primary.name(),
            n_layer = h.n_layer,
            n_expert = h.n_expert,
            "qwen3moe loaded"
        );
        let cpu_hot = opts.miss == MissPolicy::Cpu && streamer.is_some();
        let overlap = opts.overlap && cpu_hot && ram_slots.is_some();
        let sched_cpu = if overlap {
            Some(Sched::new(
                std::slice::from_ref(&cpu),
                SEGMENT_NODES * 4,
                false,
            )?)
        } else {
            None
        };
        let opts_primary_is_cpu = primary.is_cpu();
        let mut model = Qwen3Moe {
            file,
            sched,
            sched_alt: None,
            sched_proto,
            sched_small,
            sched_cpu,
            primary,
            cpu,
            weights_bytes: weights_bytes as u64,
            kv_bytes: kv_bytes as u64,
            weights,
            kv,
            steps: Vec::new(),
            pool: KvPool::new(Box::new(Chunked::new(kv_rows))),
            n_seq_max,
            reuse_step_graph: opts.reuse_step_graph,
            step_graphs_built: 0,
            segments_run: 0,
            groups_run: 0,
            group_graphs_run: 0,
            max_splits: 0,
            policy: opts.miss,
            overlap,
            urgent_first: opts.urgent_first,
            predict_depth: if opts.prefetch && cpu_hot {
                opts.prefetch_depth
            } else {
                0
            },
            cpu_threads,
            prefetch_threads: opts
                .prefetch_threads
                .map(|t| (t as i32).clamp(1, cpu_threads)),
            cpu_poll,
            pool_pause: opts.pool_pause,
            pauses_reported: 0,
            gpu: !opts_primary_is_cpu,
            op_offload: opts.op_offload,
            op_offload_min_tokens: opts.op_offload_min_tokens,
            hot_pending: cpu_hot && opts.routing_profile.is_none(),
            call_prefill: false,
            streamer,
            metrics,
            telemetry: opts.telemetry.clone(),
            routing: opts.record_routing.then(Vec::new),
            profile: StepProfile::default(),
            convert,
            shard: None,
        };
        if cpu_hot && opts.routing_profile.is_some() {
            model.refresh_hot_set()?;
        }
        Ok(model)
    }

    /// Loads the model onto the ranks of `topo` and runs it on the executor (ADR 0069, 0070):
    /// every expert resident, tensor, expert and layer parallel and tensor times pipeline. Rank
    /// `i` of a stage reads only its slice of each tensor. Ranks without a `gpu_index` are CPU
    /// ranks (tests). Refused, naming the option: streaming and tiers (`expert_slots` below the
    /// expert count, `ram_slots`, the slot byte budgets), `op_offload`, `miss = cpu`, `formats`,
    /// `record_routing`, `overlap`, `admit_share` and `routing_profile`; the streamed and CPU
    /// expert paths stay on [`Qwen3Moe::load`].
    pub fn load_sharded(
        path: impl AsRef<Path>,
        topo: &Topology,
        opts: &LoadOptions,
    ) -> Result<Qwen3Moe> {
        let file = ModelFile::open(path.as_ref())?;
        let h = file.hparams.clone();
        qwen3moe_shard::check_options(opts, &h)?;
        if h.n_head % h.n_head_kv != 0 || h.n_expert_used > h.n_expert || h.n_rot != h.head_dim {
            return Err(ModelError::Invalid("unsupported hyperparameters".into()));
        }
        if opts.n_seq_max == 0 {
            return Err(ModelError::Invalid("n_seq_max must be at least 1".into()));
        }
        let threads = opts
            .cpu_threads
            .or(match opts.backend {
                BackendChoice::Cpu { threads } => threads,
                BackendChoice::Gpu { .. } => None,
            })
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
            .max(1);
        let kv_rows = pad(opts.n_ctx.max(1), KV_PAD);
        let shard = Sharded::load(&file, topo, opts, threads as i32, kv_rows)?;

        // Placeholders for the fields the `Sched` paths use.
        let cpu_poll = cpu_poll::detect(opts.cpu_poll, threads);
        let cpu = Backend::cpu(1)?;
        let backends = vec![cpu.clone()];
        let sched = Sched::new(&backends, SEGMENT_NODES, false)?;
        let ph = Rc::new(Context::new(2, 0)?);
        let weights = Weights {
            tok_embd: ph.new_persistent(Type::F32, &[1]),
            out_norm: ph.new_persistent(Type::F32, &[1]),
            output: None,
            layers: Vec::new(),
        };
        let kv = Kv {
            k: Vec::new(),
            v: Vec::new(),
            n_rows: kv_rows,
            fmt: opts.kv,
            rot_k: None,
            rot_v: None,
        };
        let sched_proto = (backends, SEGMENT_NODES, false);
        Ok(Qwen3Moe {
            file,
            sched,
            sched_alt: None,
            sched_proto,
            sched_small: None,
            sched_cpu: None,
            primary: cpu.clone(),
            cpu,
            weights_bytes: 0,
            kv_bytes: 0,
            weights,
            kv,
            steps: Vec::new(),
            pool: KvPool::new(Box::new(Chunked::new(kv_rows))),
            n_seq_max: opts.n_seq_max,
            reuse_step_graph: opts.reuse_step_graph,
            step_graphs_built: 0,
            segments_run: 0,
            groups_run: 0,
            group_graphs_run: 0,
            max_splits: 0,
            policy: opts.miss,
            overlap: false,
            urgent_first: false,
            predict_depth: 0,
            cpu_threads: threads as i32,
            prefetch_threads: None,
            cpu_poll,
            pool_pause: false,
            pauses_reported: 0,
            gpu: false,
            op_offload: false,
            op_offload_min_tokens: opts.op_offload_min_tokens,
            hot_pending: false,
            call_prefill: false,
            streamer: None,
            metrics: MemoryMetrics::new(opts.telemetry.clone()),
            telemetry: opts.telemetry.clone(),
            routing: None,
            profile: StepProfile::default(),
            convert: ConvertStats::default(),
            shard: Some(Box::new(shard)),
        })
    }

    /// What the load-time format conversion did (ADR 0047).
    pub fn convert_stats(&self) -> &ConvertStats {
        &self.convert
    }

    pub fn hparams(&self) -> &Hparams {
        &self.file.hparams
    }

    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// The placement description (units and sizes) of this model.
    pub fn model_desc(&self) -> Result<ModelDesc> {
        Ok(self.file.model_desc()?)
    }

    pub fn n_vocab(&self) -> usize {
        self.file.hparams.n_vocab as usize
    }

    /// Tokens held by sequence 0.
    pub fn n_past(&self) -> usize {
        self.pool.seq_len(0)
    }

    /// Sequences the KV pool serves.
    pub fn n_seq_max(&self) -> usize {
        self.n_seq_max
    }

    /// The KV pool: lengths, reservations and free rows.
    pub fn pool(&self) -> &KvPool {
        &self.pool
    }

    /// Format of the K and V cache.
    pub fn kv_format(&self) -> KvFormat {
        self.kv.fmt
    }

    /// Whether Q, K and V are rotated before they meet the cache (see
    /// [`LoadOptions::kv_rotation`]): the K side and the V side.
    pub fn kv_rotated(&self) -> (bool, bool) {
        (self.kv.rot_k.is_some(), self.kv.rot_v.is_some())
    }

    pub fn n_ctx(&self) -> usize {
        self.kv.n_rows
    }

    /// Graphs computed so far for the steps: `n_layer + 1` segments per step when streaming,
    /// one whole-step graph per step when every expert is resident (ADR 0036).
    pub fn segments_run(&self) -> u64 {
        self.segments_run
    }

    /// Whole-step graphs built and allocated so far (resident experts only); with
    /// [`LoadOptions::reuse_step_graph`] fewer than the steps run.
    pub fn step_graphs_built(&self) -> u64 {
        self.step_graphs_built
    }

    /// Turns the reuse of the whole-step graph on or off between steps; off drops the kept graph.
    pub fn set_reuse_step_graph(&mut self, on: bool) {
        self.reuse_step_graph = on;
        if !on {
            for e in &mut self.steps {
                e.step = None;
            }
        }
    }

    /// Pair groups computed because a layer step needed more distinct experts than there are
    /// slots (ADR 0035), inside a segment or as a standalone graph.
    pub fn expert_groups(&self) -> u64 {
        self.groups_run
    }

    /// Standalone group graphs run: graphs beyond the `n_layer + 1` segments of a step.
    pub fn group_graphs(&self) -> u64 {
        self.group_graphs_run
    }

    /// The most scheduler splits (runs on one backend) any graph had so far; more than one
    /// means a graph ran on both the GPU and the CPU.
    pub fn max_splits(&self) -> usize {
        self.max_splits
    }

    /// Routed pairs per layer and expert since load, plus the routing profile if one was given
    /// (a profile for [`LoadOptions::routing_profile`]); empty when every expert is resident.
    pub fn routing_counts(&self) -> Vec<Vec<u64>> {
        self.streamer
            .as_ref()
            .map(|s| s.counts().to_vec())
            .unwrap_or_default()
    }

    /// What `LoadOptions::ram_hugepages` did to the RAM tier's buffer; `None` when the option is
    /// off or there is no RAM tier.
    pub fn ram_hugepages(&self) -> Option<HugepageAdvice> {
        self.streamer.as_ref().and_then(|s| s.ram_hugepages())
    }

    /// With admission on and `AdmitUpload::Async`: whether its staging ring got the device's
    /// pinned host memory (else plain host memory, and the copies block while staged).
    pub fn admit_staging_pinned(&self) -> Option<bool> {
        self.streamer
            .as_ref()
            .and_then(|s| s.admit_staging_pinned())
    }

    /// Rebuilds the hot set from the routing counts (ADR 0035): per layer the most routed
    /// experts, as many as there are compute slots, pinned and made resident. Needs
    /// `MissPolicy::Cpu`; call it between requests. Does nothing when every expert is resident.
    pub fn refresh_hot_set(&mut self) -> Result<()> {
        if self.policy != MissPolicy::Cpu {
            return Err(ModelError::Invalid(
                "the hot set needs the cpu miss policy".into(),
            ));
        }
        let Some(streamer) = &mut self.streamer else {
            return Ok(());
        };
        let layers = &self.weights.layers;
        streamer.refresh_hot_set(&self.file, |il| layers[il].exps())?;
        if let Some(log) = &mut self.routing {
            log.push(RoutingStep::refresh_marker());
        }
        Ok(())
    }

    /// Ranks of the executor; 1 on the scheduler path.
    pub fn n_ranks(&self) -> usize {
        self.shard.as_ref().map_or(1, |s| s.n_ranks())
    }

    /// The all-reduce path of the executor (`nccl`, `internal`, `fallback`); `None` on the
    /// scheduler path.
    pub fn comm_path(&self) -> Option<&'static str> {
        self.shard.as_ref().map(|s| s.comm_path())
    }

    /// Ranges launched and bytes moved by the executor since load or the last
    /// [`Qwen3Moe::reset_exec_counters`]; `None` on the scheduler path.
    pub fn exec_counters(&self) -> Option<&tetraeder_executor::ExecCounters> {
        self.shard.as_ref().map(|s| s.counters())
    }

    pub fn reset_exec_counters(&mut self) {
        if let Some(s) = self.shard.as_mut() {
            s.clear_counters();
        }
    }

    /// Every buffer the builder owns with its capacity and used bytes (ADR 0037): the weights,
    /// the compute slot tensors, the KV cache, the RAM tier and the scheduler's compute buffers.
    /// Cheap: it reads sizes and the slot tables.
    pub fn memory(&self) -> Vec<MemoryUse> {
        if let Some(sh) = &self.shard {
            return sh.memory();
        }
        let tier = if self.primary.is_cpu() { "ram" } else { "vram" };
        let device = self.primary.name();
        let mut out = Vec::new();
        let mut push = |device: &str, tier, kind, capacity: u64, used: u64| {
            out.push(MemoryUse {
                device: device.to_string(),
                tier,
                kind,
                format: (kind == "kv").then(|| self.kv.fmt.to_string()),
                capacity,
                used,
            });
        };
        let slots = self.streamer.as_ref().map(Streamer::slot_use);
        let slot_cap = slots.as_ref().map_or(0, |s| s.compute_capacity);
        let weights = self.weights_bytes.saturating_sub(slot_cap);
        push(&device, tier, "weights", weights, weights);
        if let Some(s) = &slots {
            push(&device, tier, "slots", s.compute_capacity, s.compute_used);
        }
        push(&device, tier, "kv", self.kv_bytes, self.kv_bytes);
        if let Some((cap, used)) = slots.and_then(|s| s.ram) {
            push("host", "ram", "slots", cap, used);
        }
        if let Some(st) = &self.streamer {
            let io = st.bounce_bytes();
            push("host", "ram", "io", io, io);
        }
        let scheds = || {
            std::iter::once(&self.sched)
                .chain(&self.sched_alt)
                .chain(&self.sched_small)
        };
        let mut work: u64 = scheds().map(|s| s.buffer_size(&self.primary) as u64).sum();
        if self.primary.is_cpu() {
            work += self
                .sched_cpu
                .iter()
                .map(|s| s.buffer_size(&self.cpu) as u64)
                .sum::<u64>();
        }
        push(&device, tier, "workspace", work, work);
        if !self.primary.is_cpu() {
            let work = scheds()
                .chain(&self.sched_cpu)
                .map(|s| s.buffer_size(&self.cpu) as u64)
                .sum();
            push(&self.cpu.name(), "ram", "workspace", work, work);
        }
        out
    }

    /// The CPU threadpool's polling level in effect, whether it was chosen automatically, and
    /// the quota and online CPUs it was chosen from (ADR 0061).
    /// Threadpool pauses issued so far (`LoadOptions::pool_pause`).
    pub fn cpu_pool_pauses(&self) -> u64 {
        self.cpu.cpu_pool_pauses()
    }

    pub fn cpu_poll(&self) -> CpuPoll {
        self.cpu_poll
    }

    /// The expert file I/O of the first shard, e.g. `direct overlapped`; `None` when every
    /// expert is resident.
    pub fn io_backend(&self) -> Option<String> {
        self.streamer.as_ref().map(Streamer::io_backend)
    }

    /// Expert cache and read counters; `None` when every expert is resident.
    pub fn expert_stats(&self) -> Option<ExpertStats> {
        self.streamer.as_ref().map(Streamer::stats)
    }

    /// Slots per layer of the compute tier and of the RAM tier, as resolved at load; `None` when
    /// every expert is resident.
    pub fn tier_slots(&self) -> Option<(usize, Option<usize>)> {
        self.streamer.as_ref().map(Streamer::tier_slots)
    }

    /// Layers ahead that the prefetch predicts and reads: 0 when prefetch is off or does not
    /// apply (ADR 0038).
    pub fn prefetch_depth(&self) -> usize {
        self.predict_depth
    }

    /// The routing recorded since the last call (empty unless `record_routing` is set), one
    /// entry per layer step in execution order.
    pub fn take_routing_log(&mut self) -> Vec<RoutingStep> {
        self.routing
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// The time breakdown since load or the last [`Qwen3Moe::take_profile`].
    pub fn profile(&self) -> StepProfile {
        self.profile
    }

    /// The time breakdown since load or the last call, then starts a new one.
    pub fn take_profile(&mut self) -> StepProfile {
        std::mem::take(&mut self.profile)
    }

    /// Publishes memory use, the KV tokens held and the pool's blocks; also after the cache
    /// shrinks, so the gauge does not keep a dropped sequence's length.
    fn report_memory(&self) {
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
                    self.kv.n_rows as u64,
                ),
                None => self
                    .metrics
                    .buffer(&m.device, m.tier, m.kind, m.used, m.capacity),
            }
        }
        if let Some(t) = &self.telemetry {
            let b = self.pool.blocks();
            let device = self.primary.name();
            let tier = if self.primary.is_cpu() { "ram" } else { "vram" };
            let m = t.metrics();
            m.kv_blocks(&device, tier, "used", b.used as u64);
            m.kv_blocks(&device, tier, "reserved", b.reserved as u64);
            m.kv_blocks(&device, tier, "free", b.free as u64);
        }
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

    /// Copies the rows a layout move or `seq_cp` asks for, on the device. Applied right after
    /// the pool call that returned them: the layout has already changed.
    fn apply(&mut self, copies: &[RowCopy]) -> Result<()> {
        let list: Vec<_> = copies.iter().map(|c| (c.src, c.dst, c.n)).collect();
        match self.shard.as_mut() {
            Some(sh) => sh.copy_rows(&list)?,
            None => copy_rows(&self.kv, &mut self.sched, &list)?,
        }
        if let Some(t) = &self.telemetry {
            for _ in copies {
                t.metrics().kv_relocation();
            }
        }
        Ok(())
    }

    /// Guarantees rows for `seq` to grow to `n` positions. `Ok(false)` when the pool cannot give
    /// them (nothing changed). A sequence the layout has to move is copied on the device.
    pub fn try_reserve(&mut self, seq: u32, n: usize) -> Result<bool> {
        self.check_seq(seq)?;
        match self.pool.reserve(seq, n) {
            Ok(copies) => {
                self.apply(&copies)?;
                self.report_memory();
                Ok(true)
            }
            Err(KvError::Full { .. }) => Ok(false),
            Err(e) => Err(kv_err(e)),
        }
    }

    /// Like [`Qwen3Moe::try_reserve`]; a full pool is an error.
    pub fn seq_reserve(&mut self, seq: u32, n: usize) -> Result<()> {
        if self.try_reserve(seq, n)? {
            Ok(())
        } else {
            Err(self.pool_full(seq, n))
        }
    }

    fn pool_full(&self, seq: u32, n: usize) -> ModelError {
        ModelError::Invalid(format!(
            "KV pool full: {n} rows for sequence {seq}, {} free",
            self.pool.free()
        ))
    }

    /// Keeps the first `keep` positions of `seq`. The reservation shrinks to `keep` (rounded up
    /// to 256 rows); 0 frees the sequence.
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
                self.apply(&copies)?;
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
            Err(self.pool_full(dst, n))
        }
    }

    /// Positions held by `seq`.
    pub fn seq_len(&self, seq: u32) -> usize {
        self.pool.seq_len(seq)
    }

    /// Forgets the KV cache of sequence 0; the next `eval` starts at position 0.
    pub fn reset(&mut self) {
        self.pool.seq_rm(0, 0);
        self.report_memory();
    }

    /// Rolls sequence 0 back to its first `n_past` tokens. Rows above stay in the KV buffers but
    /// the mask hides them and the next step overwrites them, so nothing needs clearing
    /// (ADR 0043).
    pub fn rewind(&mut self, n_past: usize) -> Result<()> {
        if n_past > self.n_past() {
            return Err(ModelError::Invalid(format!(
                "cannot rewind to {n_past}: only {} tokens cached",
                self.n_past()
            )));
        }
        self.seq_rm(0, n_past)
    }

    /// Bytes of one cached position across all layers (K and V).
    pub fn kv_row_bytes(&self) -> usize {
        if let Some(sh) = &self.shard {
            return sh.kv_row_bytes();
        }
        self.kv
            .k
            .iter()
            .chain(&self.kv.v)
            .map(|t| t.get().nbytes() / self.kv.n_rows)
            .sum()
    }

    /// The K and V rows of the positions of `seq` as host bytes, tensor by tensor, positions in
    /// order (ADR 0049). The bytes do not depend on where the rows sit in the pool.
    pub fn seq_snapshot(&self, seq: u32) -> Result<Vec<u8>> {
        self.check_seq(seq)?;
        let n = self.pool.seq_len(seq);
        let runs = self.pool.runs(seq, n);
        if let Some(sh) = &self.shard {
            return sh.snapshot(&runs);
        }
        let mut out = Vec::with_capacity(n * self.kv_row_bytes());
        for t in self.kv.k.iter().chain(&self.kv.v) {
            let row = t.get().nbytes() / self.kv.n_rows;
            for &(start, len) in &runs {
                out.extend_from_slice(&t.get().get_bytes(start * row, len * row)?);
            }
        }
        Ok(out)
    }

    /// Replaces `seq` by a [`Qwen3Moe::seq_snapshot`] of `n` positions. `Ok(false)` when the
    /// pool has no rows for them (nothing changed).
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
        if let Some(sh) = &self.shard {
            sh.restore(&runs, bytes)?;
        } else {
            let mut rest = bytes;
            for t in self.kv.k.iter().chain(&self.kv.v) {
                let row = t.get().nbytes() / self.kv.n_rows;
                for &(start, len) in &runs {
                    let (head, tail) = rest.split_at(len * row);
                    t.get().set_bytes(head, start * row)?;
                    rest = tail;
                }
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
            Err(self.pool_full(seq, n))
        }
    }

    /// Sequence 0's cache as host bytes (ADR 0049).
    pub fn kv_snapshot(&self) -> Result<Vec<u8>> {
        self.seq_snapshot(0)
    }

    /// Writes a [`Qwen3Moe::kv_snapshot`] of `n` positions back and sets sequence 0's length.
    pub fn kv_restore(&mut self, bytes: &[u8], n: usize) -> Result<()> {
        self.seq_restore(0, bytes, n)
    }

    /// Runs `tokens` after the tokens already in sequence 0 and returns the logits of every
    /// position, `tokens.len() * n_vocab` values, position-major.
    pub fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.eval_seq0(tokens, true, &mut out)?;
        Ok(out)
    }

    /// Like [`Qwen3Moe::eval`] but only for the last position (`n_vocab` values): prefill does
    /// not need the head for the other positions.
    pub fn eval_last(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.eval_seq0(tokens, false, &mut out)?;
        Ok(out)
    }

    /// Like [`Qwen3Moe::eval_last`] but into `out`, whose allocation is reused: a decode step
    /// then reads its 151K logits without allocating.
    pub fn eval_last_into(&mut self, tokens: &[i32], out: &mut Vec<f32>) -> Result<()> {
        self.eval_seq0(tokens, false, out)
    }

    fn eval_seq0(&mut self, tokens: &[i32], all_logits: bool, out: &mut Vec<f32>) -> Result<()> {
        let n_past = self.pool.seq_len(0);
        if n_past + tokens.len() > self.kv.n_rows {
            return Err(ModelError::Invalid(format!(
                "context full: {n_past} cached + {} new > {}",
                tokens.len(),
                self.kv.n_rows
            )));
        }
        self.decode(
            &[SeqEval {
                seq: 0,
                tokens,
                all_logits,
            }],
            out,
        )?;
        Ok(())
    }

    /// One step: each input's tokens are appended to its sequence (sequences distinct, tokens
    /// never empty). Rows missing for a sequence are reserved first (a full pool is an error
    /// and leaves every sequence's length as it was). The logits of the last token of each
    /// input, or of all its tokens with `all_logits`, go into `out`, `n_vocab` floats per row in
    /// input and position order; the result is the row range of each input.
    pub fn decode(
        &mut self,
        inputs: &[SeqEval<'_>],
        out: &mut Vec<f32>,
    ) -> Result<Vec<Range<usize>>> {
        let prefill = inputs.iter().map(|i| i.tokens.len()).sum::<usize>() > 1;
        self.decode_group(inputs, out, prefill)
    }

    /// [`Qwen3Moe::decode`] that says whether the call holds a prompt chunk. Only such a call
    /// builds the CPU hot set from its routing (ADR 0035); a group of decoders with more than
    /// one token in all is not a prefill. `decode` takes more than one token as a prefill.
    pub fn decode_group(
        &mut self,
        inputs: &[SeqEval<'_>],
        out: &mut Vec<f32>,
        prefill: bool,
    ) -> Result<Vec<Range<usize>>> {
        self.call_prefill = prefill;
        let t0 = Instant::now();
        if inputs.is_empty() {
            return Err(ModelError::Invalid("no sequences".into()));
        }
        let n_vocab = self.file.hparams.n_vocab;
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
        let tokens: Vec<i32> = inputs
            .iter()
            .flat_map(|i| i.tokens.iter().copied())
            .collect();
        self.run(&tokens, &step, &plan, out, t0)?;
        let mut row = 0;
        Ok(step
            .iter()
            .map(|s| {
                let k = if s.all_logits { s.n_tokens } else { 1 };
                row += k;
                row - k..row
            })
            .collect())
    }

    /// The cached state of this shape, taken out of the cache (most recent first), or a new one.
    /// A new one makes room first: the least recent entry goes, graph before state, and the new
    /// entry takes the scheduler the other entry does not use.
    fn take_step_state(&mut self, key: (usize, usize, usize)) -> Result<CachedStep> {
        let at = self
            .steps
            .iter()
            .position(|e| (e.state.n_tokens, e.state.n_kv, e.state.n_out) == key);
        if let Some(i) = at {
            return Ok(self.steps.remove(i));
        }
        let cap = self.step_cache_len();
        self.steps.truncate(cap - 1);
        let alt = self.steps.first().is_some_and(|e| !e.alt);
        if alt && self.sched_alt.is_none() {
            let (backends, nodes, offload) = &self.sched_proto;
            self.sched_alt = Some(Sched::new(backends, *nodes, *offload)?);
        }
        let host = self.overlap.then_some(&self.cpu);
        let state = alloc_state(&self.file.hparams, key, &self.primary, host)?;
        Ok(CachedStep {
            step: None,
            alt,
            state,
        })
    }

    /// Entries the step cache holds: two with the whole-step graph kept on the CPU (where a step
    /// alternates decoders and a prompt chunk), else one. A second entry costs a second scheduler
    /// with its own compute buffers.
    fn step_cache_len(&self) -> usize {
        if self.streamer.is_none() && self.reuse_step_graph && self.primary.is_cpu() {
            2
        } else {
            1
        }
    }

    fn run(
        &mut self,
        tokens: &[i32],
        inputs: &[StepInput],
        plan: &StepPlan,
        logits: &mut Vec<f32>,
        t0: Instant,
    ) -> Result<()> {
        if self.shard.is_some() {
            return self.run_sharded(tokens, inputs, plan, logits, t0);
        }
        let key = (tokens.len(), plan.n_kv, plan.outputs.len());
        let mut entry = self.take_step_state(key)?;
        let r = self.run_in(&mut entry, tokens, inputs, plan, logits, t0);
        self.steps.insert(0, entry);
        self.steps.truncate(self.step_cache_len());
        r
    }

    /// A step on the executor's ranks ([`Qwen3Moe::load_sharded`]).
    fn run_sharded(
        &mut self,
        tokens: &[i32],
        inputs: &[StepInput],
        plan: &StepPlan,
        logits: &mut Vec<f32>,
        t0: Instant,
    ) -> Result<()> {
        let mut prof = self.profile;
        prof.steps += 1;
        let t = Instant::now();
        let reuse = self.reuse_step_graph;
        let sh = self.shard.as_mut().expect("checked by the caller");
        sh.step(tokens, plan, reuse, logits)?;
        self.step_graphs_built = sh.built;
        self.segments_run += 1;
        prof.graphs += 1;
        prof.compute += t.elapsed();
        self.finish_step(inputs, t0, prof, 1)
    }

    fn run_in(
        &mut self,
        entry: &mut CachedStep,
        tokens: &[i32],
        inputs: &[StepInput],
        plan: &StepPlan,
        logits: &mut Vec<f32>,
        t0: Instant,
    ) -> Result<()> {
        let h = self.file.hparams.clone();
        let n = tokens.len();
        let n_kv = plan.n_kv;
        let n_out = plan.outputs.len();
        let st = &entry.state;
        st.tokens.get().set_i32(tokens)?;
        st.pos.get().set_i32(&plan.pos)?;
        st.rows.get().set_i32(&plan.rows)?;
        // `[n_kv, n]` as libllama builds it at the pin (no row padding since GGML_KQ_MASK_PAD
        // went away): a decode step writes and uploads one row, not 32.
        st.mask.get().set_f16_bits(&plan.mask)?;
        st.outputs.get().set_i32(&plan.outputs)?;
        let mut prof = self.profile;
        prof.steps += 1;
        prof.inputs += t0.elapsed();

        if self.streamer.is_none() {
            // Every expert resident: no host work between layers, so the step is one graph
            // (ADR 0036).
            let mut t = Instant::now();
            let record = self.routing.is_some();
            let key = (n, n_kv, n_out, record);
            let sched = if entry.alt {
                self.sched_alt.as_mut().expect("made with the entry")
            } else {
                &mut self.sched
            };
            let reusable = self.reuse_step_graph
                && entry
                    .step
                    .as_ref()
                    .is_some_and(|s| s.key == key && sched.is_allocated(&s.graph.get()));
            // A kept graph is computed again with the new inputs: no build, no allocation.
            // Otherwise the graph is built and allocated; it is kept only with reuse on.
            let fresh;
            let step: &StepGraph = if reusable {
                entry.step.as_ref().expect("checked above")
            } else {
                entry.step = None;
                let ctx = Rc::new(Context::new(step_nodes(&h), step_nodes(&h))?);
                let (graph, out, routed) =
                    build_step(&h, &self.weights, &self.kv, st, &ctx, record);
                let mut deps = vec![
                    &self.weights.tok_embd,
                    &self.kv.k[0],
                    &self.kv.v[0],
                    &st.tokens,
                ];
                deps.extend(self.kv.rot_k.iter().chain(&self.kv.rot_v));
                let routed = routed
                    .into_iter()
                    .map(|(ids, w)| Ok((ctx.keep(ids)?, ctx.keep(w)?)))
                    .collect::<Result<Vec<_>>>()?;
                fresh = StepGraph {
                    key,
                    logits: ctx.keep(out)?,
                    routed,
                    graph: ctx.keep_graph(graph, &deps)?,
                };
                t = lap(&mut prof.build, t);
                sched.alloc_graph(&fresh.graph.get())?;
                self.max_splits = self.max_splits.max(sched.n_splits());
                self.step_graphs_built += 1;
                t = lap(&mut prof.alloc, t);
                if self.reuse_step_graph {
                    entry.step.insert(fresh)
                } else {
                    &fresh
                }
            };
            sched.compute(&step.graph.get())?;
            self.segments_run += 1;
            prof.graphs += 1;
            step.logits.get().get_f32_into(logits)?;
            let t = lap(&mut prof.compute, t);
            if let Some(log) = &mut self.routing {
                for (layer, (ids, w)) in step.routed.iter().enumerate() {
                    log.push(RoutingStep {
                        layer,
                        ids: ids.get().get_i32()?,
                        weights: w.get().get_f32()?,
                        n_tokens: n,
                        predicted: None,
                        predicted2: None,
                        refresh: false,
                    });
                }
            }
            for _ in 0..h.n_layer {
                self.metrics.layer_step(true);
            }
            lap(&mut prof.readback, t);
            return self.finish_step(inputs, t0, prof, 1);
        }

        let graphs_before = self.group_graphs_run;
        let nu = h.n_expert_used as usize;
        let n_layer = h.n_layer as usize;
        // The pair groups of the previous layer step that run inside this segment.
        let mut pending: Vec<PendingGroup> = Vec::new();
        // The predictions of the experts of a layer (ADR 0038): the next layer's from the
        // previous segment, and, at depth 2, per target layer the one made two segments ago.
        let mut predicted: Option<Vec<i32>> = None;
        let mut predicted2: Vec<Option<Vec<i32>>> = vec![None; n_layer + 2];
        if let Some(streamer) = &mut self.streamer {
            // A step that failed may have left a prefetch in flight.
            streamer.begin_step(n);
        }
        self.cpu.set_cpu_threads(self.cpu_threads);
        let offload = offloads(self.op_offload, self.op_offload_min_tokens, n);
        let sched = match &mut self.sched_small {
            Some(small) if !offload => small,
            _ => &mut self.sched,
        };
        for seg in 0..=n_layer {
            let groups = std::mem::take(&mut pending);
            let cpu_seg = groups.iter().any(|g| matches!(g.src, ExpertSrc::Ram));
            {
                let t = Instant::now();
                let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
                let specs: Vec<GroupSpec<'_>> = groups
                    .iter()
                    .map(|g| {
                        let n = g.pairs.pairs.len();
                        match g.src {
                            ExpertSrc::Copied => GroupSpec::Copied(n),
                            src => {
                                let lw = &self.weights.layers[seg - 1];
                                let exps = group_exps(src, lw, self.streamer.as_ref(), seg - 1);
                                GroupSpec::Exps(exps, n)
                            }
                        }
                    })
                    .collect();
                let (graph, out, inputs) = build_segment(
                    &h,
                    &self.weights,
                    &self.kv,
                    st,
                    &ctx,
                    seg,
                    self.predict_depth,
                    self.routing.is_some(),
                    &specs,
                );
                let t = lap(&mut prof.build, t);
                sched.alloc_graph(&graph)?;
                self.max_splits = self.max_splits.max(sched.n_splits());
                for (g, i) in groups.iter().zip(&inputs) {
                    i.write(&g.pairs)?;
                    self.groups_run += 1;
                }
                let t = lap(&mut prof.alloc, t);
                // A GPU-only segment leaves the CPU idle until the next layer's groups.
                if self.pool_pause && self.gpu && !cpu_seg {
                    self.cpu.pause_cpu_pool();
                }
                let tc = Instant::now();
                sched.compute(&graph)?;
                if cpu_seg {
                    prof.seg_compute_cpu += tc.elapsed();
                    prof.segs_cpu += 1;
                } else {
                    prof.seg_compute_gpu += tc.elapsed();
                    prof.segs_gpu += 1;
                }
                self.segments_run += 1;
                prof.graphs += 1;
                if let Some(l) = out {
                    l.get_f32_into(logits)?;
                }
                lap(&mut prof.compute, t);
            }
            if seg == n_layer {
                continue;
            }
            let t = Instant::now();
            let slot = st.slot(seg, n_layer);
            // The last layer's slot holds the output rows only (`State::last`).
            let rows_out = st.last.is_some() && seg + 1 == n_layer;
            let chosen = slot.ids.get().get_i32()?;
            let streamer = self
                .streamer
                .as_mut()
                .expect("a resident step returned above");
            // The predictions cover all rows: score the output rows against the choice.
            let out_rows = |p: &[i32]| -> Vec<i32> {
                if rows_out {
                    plan.outputs
                        .iter()
                        .flat_map(|&r| p[r as usize * nu..][..nu].iter().copied())
                        .collect()
                } else {
                    p.to_vec()
                }
            };
            if let Some(p) = predicted.take() {
                streamer.score_prediction(seg, 1, &out_rows(&p), &chosen, nu);
            }
            if let Some(p) = &predicted2[seg] {
                streamer.score_prediction(seg, 2, &out_rows(p), &chosen, nu);
            }
            // The predicted experts of the next layers, read back with this layer's routing.
            if self.predict_depth >= 1 && seg + 1 < n_layer {
                predicted = Some(slot.pred.get().get_i32()?);
            }
            if self.predict_depth >= 2 && seg + 2 < n_layer {
                predicted2[seg + 2] = Some(slot.pred2.get().get_i32()?);
            }
            if let Some(log) = &mut self.routing {
                // This layer's predictions (for layers seg + 1 and seg + 2), as read above.
                let full = &st.slots[seg % 2];
                let (ids, weights) = if rows_out {
                    (full.ids.get().get_i32()?, full.w.get().get_f32()?)
                } else {
                    (chosen.clone(), slot.w.get().get_f32()?)
                };
                log.push(RoutingStep {
                    layer: seg,
                    ids,
                    weights,
                    n_tokens: n,
                    predicted: predicted.clone(),
                    predicted2: predicted2.get(seg + 2).cloned().flatten(),
                    refresh: false,
                });
            }
            lap(&mut prof.readback, t);
            // Under overlap a layer does not wait for its reads before it is planned (ADR 0054).
            let gate = self.overlap && self.policy == MissPolicy::Cpu && !offload;
            if !gate {
                // The reads started for this layer must have landed before it is planned.
                if self.pool_pause {
                    self.cpu.pause_cpu_pool();
                }
                streamer.finish_prefetch(seg)?;
                set_compute_threads(&self.cpu, self.cpu_threads, self.prefetch_threads, streamer);
            }
            let layer = &self.weights.layers[seg];
            if self.policy == MissPolicy::Cpu {
                // Resident experts run from the compute slots, the others from the RAM slots
                // on the CPU, in chunks of the RAM slot count (ADR 0035).
                let plan = streamer.plan_cpu(seg, &chosen)?;
                prof.cpu_pairs += chosen
                    .iter()
                    .filter(|&&e| !plan.compute.contains(&(e as u32)))
                    .count() as u64;
                let overlapped = gate && plan.cpu.len() == 1;
                if gate && !overlapped && !plan.cpu.is_empty() {
                    if self.pool_pause {
                        self.cpu.pause_cpu_pool();
                    }
                    streamer.finish_prefetch(seg)?;
                }
                // With `urgent_first` an overlapped step submits its misses first: the workers
                // pick chunks in deadline order but cannot take back the ones they started.
                // Staging touches only layer `seg`, the prefetch only `seg + 1` and `seg + 2`.
                let early = if overlapped && self.urgent_first {
                    Some(streamer.stage_overlap(&self.file, seg, &plan.cpu[0])?)
                } else {
                    None
                };
                // Otherwise the next layers' reads start before this layer's own misses are
                // read: the reader has direct files of its own, and the device serves both.
                prefetch_ahead(streamer, &self.file, seg, &predicted, &predicted2)?;
                set_compute_threads(&self.cpu, self.cpu_threads, self.prefetch_threads, streamer);
                if plan.cpu.is_empty() {
                    let ids = map_ids(&chosen, &plan.compute, &plan.compute_slots);
                    slot.ids_map.get().set_i32(&ids)?;
                    continue;
                }
                let cap_ram = group_cap(ExpertSrc::Ram, self.gpu, offload);
                let cap_compute = group_cap(ExpertSrc::Compute, self.gpu, offload);
                if overlapped {
                    let chunk = &plan.cpu[0];
                    let staged = match early {
                        Some(s) => s,
                        None => streamer.stage_overlap(&self.file, seg, chunk)?,
                    };
                    set_compute_threads(
                        &self.cpu,
                        self.cpu_threads,
                        self.prefetch_threads,
                        streamer,
                    );
                    if staged.pending.is_empty() {
                        // Everything is in its slot: the usual path, the group in the next segment.
                        streamer.finish_overlap(seg);
                        let pairs = group_pairs(&chosen, nu, chunk, &staged.slots);
                        for pairs in split_group(pairs, cap_ram) {
                            prof.seg_pairs += pairs.pairs.len() as u64;
                            pending.push(PendingGroup {
                                src: ExpertSrc::Ram,
                                pairs,
                            });
                        }
                    } else {
                        let sched_cpu = self
                            .sched_cpu
                            .as_mut()
                            .expect("overlap has a cpu scheduler");
                        let host = st.host.as_ref().expect("overlap has host buffers");
                        let ctx = OverlapCtx {
                            h: &h,
                            chosen: &chosen,
                            nu,
                            chunk,
                            slots: &staged.slots,
                            host,
                            metrics: &self.metrics,
                            pause: self.pool_pause.then_some(&self.cpu),
                        };
                        let (pairs, graphs) = run_overlap_layer(
                            &ctx, streamer, seg, sched_cpu, slot, &staged, &mut prof,
                        )?;
                        self.groups_run += graphs;
                        self.group_graphs_run += graphs;
                        pending.push(PendingGroup {
                            src: ExpertSrc::Copied,
                            pairs,
                        });
                    }
                    if n == 1 {
                        // Every expert of the chunk has landed (ADR 0059).
                        streamer.admit(seg, chunk, &plan.compute, Some(&self.cpu))?;
                    }
                    if !plan.compute.is_empty() {
                        let pairs = group_pairs(&chosen, nu, &plan.compute, &plan.compute_slots);
                        for pairs in split_group(pairs, cap_compute) {
                            pending.push(PendingGroup {
                                src: ExpertSrc::Compute,
                                pairs,
                            });
                        }
                    }
                    continue;
                }
                let last = plan.cpu.len() - 1;
                for (i, chunk) in plan.cpu.iter().enumerate() {
                    let slots = streamer.stage_ram(&self.file, seg, chunk)?;
                    if n == 1 {
                        // Before the next chunk's reads can take these RAM slots (ADR 0059).
                        streamer.admit(seg, chunk, &plan.compute, Some(&self.cpu))?;
                    }
                    let pairs = group_pairs(&chosen, nu, chunk, &slots);
                    for pairs in split_group(pairs, cap_ram) {
                        if i == last {
                            prof.seg_pairs += pairs.pairs.len() as u64;
                            pending.push(PendingGroup {
                                src: ExpertSrc::Ram,
                                pairs,
                            });
                            continue;
                        }
                        let exps = streamer.ram_exps(seg).expect("RAM tier");
                        let splits = run_group_graph(&h, sched, exps, slot, &pairs, &mut prof)?;
                        self.max_splits = self.max_splits.max(splits);
                        self.groups_run += 1;
                        self.group_graphs_run += 1;
                    }
                }
                if !plan.compute.is_empty() {
                    let pairs = group_pairs(&chosen, nu, &plan.compute, &plan.compute_slots);
                    for pairs in split_group(pairs, cap_compute) {
                        pending.push(PendingGroup {
                            src: ExpertSrc::Compute,
                            pairs,
                        });
                    }
                }
                continue;
            }
            let chunks = streamer.plan(seg, &chosen)?;
            if let [chunk] = chunks.as_slice() {
                let slots = streamer.make_resident(&self.file, seg, &layer.exps(), chunk)?;
                slot.ids_map
                    .get()
                    .set_i32(&map_ids(&chosen, chunk, &slots))?;
                continue;
            }
            // More distinct experts than slots: one pair group per chunk (ADR 0035). All
            // but the last run as graphs of their own; the last goes into the next segment.
            let cap = group_cap(ExpertSrc::Compute, self.gpu, offload);
            let last = chunks.len() - 1;
            for (i, chunk) in chunks.iter().enumerate() {
                let slots = streamer.make_resident(&self.file, seg, &layer.exps(), chunk)?;
                let pairs = group_pairs(&chosen, nu, chunk, &slots);
                for pairs in split_group(pairs, cap) {
                    if i == last {
                        pending.push(PendingGroup {
                            src: ExpertSrc::Compute,
                            pairs,
                        });
                        continue;
                    }
                    let splits = run_group_graph(&h, sched, layer.exps(), slot, &pairs, &mut prof)?;
                    self.max_splits = self.max_splits.max(splits);
                    self.groups_run += 1;
                    self.group_graphs_run += 1;
                }
            }
        }
        if self.overlap {
            // Mispredicted prefetches are still pinned and uncounted: collect them, so the
            // counters and the cache are settled when the step returns (ADR 0054).
            if let Some(streamer) = &mut self.streamer {
                streamer.settle();
            }
        }
        // Segment graphs plus the standalone group graphs of this step (ADR 0035).
        let graphs = h.n_layer + 1 + (self.group_graphs_run - graphs_before) as u32;
        prof.graphs += self.group_graphs_run - graphs_before;
        // The rest of this step's time is the streamer's.
        let timed = |p: &StepProfile| p.inputs + p.build + p.alloc + p.compute + p.readback;
        let others = timed(&prof) - timed(&self.profile);
        prof.streamer += t0.elapsed().saturating_sub(others);
        self.finish_step(inputs, t0, prof, graphs)
    }

    /// Bookkeeping after a step that computed `graphs` graphs.
    fn finish_step(
        &mut self,
        step: &[StepInput],
        t0: Instant,
        mut prof: StepProfile,
        graphs: u32,
    ) -> Result<()> {
        let n: usize = step.iter().map(|s| s.n_tokens).sum();
        self.pool.commit(step);
        prof.total += t0.elapsed();
        self.profile = prof;
        if let Some(t) = &self.telemetry {
            t.record_step(StepSample {
                phase: if step.iter().all(|s| s.n_tokens == 1) {
                    Phase::Decode
                } else {
                    Phase::Prefill
                },
                stage: 0,
                graphs: false,
                duration: t0.elapsed(),
                tokens: n as u32,
                segments: graphs,
            });
            self.report_memory();
            let pauses = self.cpu.cpu_pool_pauses();
            t.metrics().cpu_pool_pauses(pauses - self.pauses_reported);
            self.pauses_reported = pauses;
        }
        if n > 1 && self.call_prefill && self.hot_pending {
            // The first prefill's routing decides the hot set (ADR 0035).
            self.hot_pending = false;
            self.refresh_hot_set()?;
        }
        Ok(())
    }
}

/// The slot of each chosen expert; every chosen expert is in `chunk` (the single-chunk case).
fn map_ids(chosen: &[i32], chunk: &[u32], slots: &[u32]) -> Vec<i32> {
    chosen
        .iter()
        .map(|&e| {
            let i = chunk
                .iter()
                .position(|&c| c as i32 == e)
                .expect("a single chunk holds every chosen expert");
            slots[i] as i32
        })
        .collect()
}

/// The pairs of one expert group (ADR 0035), in the order the resident `mul_mat_id` visits them.
#[derive(Debug, Default, PartialEq, Eq)]
struct PairGroup {
    /// The pair's token: the row of the normed hidden state to gather.
    tokens: Vec<i32>,
    /// `token * n_used + choice`: the pair's row in the accumulator.
    pairs: Vec<i32>,
    /// The slot of the pair's expert.
    ids: Vec<i32>,
}

/// The pairs of `chosen` (`n_used` per token) whose expert is in `experts`, with the experts' slots
/// in `slots`. Walking token-major, choice-minor gives each expert all its routed rows in the
/// order the resident layer gives them; the CPU's K-quant `mul_mat_id` picks its kernel per expert
/// by that row count and order, so this keeps the group bitwise equal to the resident layer
/// (ADR 0034, 0035).
fn group_pairs(chosen: &[i32], n_used: usize, experts: &[u32], slots: &[u32]) -> PairGroup {
    let mut g = PairGroup::default();
    for (pair, &e) in chosen.iter().enumerate() {
        if let Some(i) = experts.iter().position(|&c| c as i32 == e) {
            g.tokens.push((pair / n_used) as i32);
            g.pairs.push(pair as i32);
            g.ids.push(slots[i] as i32);
        }
    }
    g
}

/// Most (token, choice) rows of one group on CUDA: `mul_mat_id`'s id helper asserts
/// `n_rows * 4 B <= shared memory per block` (about 25k rows on Ampere); ubatch 4096 at top-8 is
/// 32k rows and aborted on the box.
const MAX_GROUP_ROWS: usize = 16384;

/// Whether a streamed step of `n` tokens runs with op offload (ADR 0044).
fn offloads(op_offload: bool, min_tokens: usize, n: usize) -> bool {
    op_offload && n >= min_tokens
}

/// The row cap of a group that can run on CUDA: compute-tier groups on a GPU backend, and RAM
/// groups when `op_offload` lets ggml run their matmuls there. CPU-only groups are never split:
/// the CPU K-quant `mul_mat_id` picks its kernel by per-expert row count, so splitting would break
/// the bitwise equality with the resident layer (ADR 0034, 0035).
fn group_cap(src: ExpertSrc, gpu: bool, op_offload: bool) -> Option<usize> {
    let on_gpu = gpu
        && match src {
            ExpertSrc::Compute => true,
            ExpertSrc::Ram => op_offload,
            ExpertSrc::Copied => false,
        };
    on_gpu.then_some(MAX_GROUP_ROWS)
}

/// `g` as consecutive sub-groups of at most `cap` pairs, token-major order kept; the pairs'
/// destination rows are disjoint, so the sub-groups can run in any graph.
fn split_group(g: PairGroup, cap: Option<usize>) -> Vec<PairGroup> {
    let Some(cap) = cap.filter(|&c| g.pairs.len() > c) else {
        return vec![g];
    };
    (0..g.pairs.len())
        .step_by(cap)
        .map(|from| {
            let to = (from + cap).min(g.pairs.len());
            PairGroup {
                tokens: g.tokens[from..to].to_vec(),
                pairs: g.pairs[from..to].to_vec(),
                ids: g.ids[from..to].to_vec(),
            }
        })
        .collect()
}

/// Which tensors a pair group's experts are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertSrc {
    /// The layer's expert tensors: all experts, or the compute slots.
    Compute,
    /// The RAM slot tensors, computed on the CPU.
    Ram,
    /// Outputs the CPU already computed in arrival order and copied to the slot's `cpu_out`
    /// (ADR 0054); the segment only scatters them into `acc`.
    Copied,
}

/// A pair group as the segment builder sees it.
#[derive(Clone, Copy)]
enum GroupSpec<'a> {
    /// Expert matmuls over these tensors for this many pairs.
    Exps([Tensor<'a>; 3], usize),
    /// This many rows of `cpu_out`.
    Copied(usize),
}

/// A pair group waiting to be built into the next segment.
struct PendingGroup {
    src: ExpertSrc,
    pairs: PairGroup,
}

fn group_exps<'a>(
    src: ExpertSrc,
    lw: &'a Layer,
    streamer: Option<&'a Streamer>,
    il: usize,
) -> [Tensor<'a>; 3] {
    match src {
        ExpertSrc::Compute => lw.exps(),
        ExpertSrc::Ram => streamer
            .and_then(|s| s.ram_exps(il))
            .expect("a RAM group needs the RAM tier"),
        ExpertSrc::Copied => unreachable!("a copied group has no expert tensors"),
    }
}

/// Runs one pair group as a graph of its own. Returns the scheduler's split count; the time goes
/// to `prof`'s build, alloc and compute like a segment's.
fn run_group_graph(
    h: &Hparams,
    sched: &mut Sched,
    exps: [Tensor<'_>; 3],
    s: &Slot,
    group: &PairGroup,
    prof: &mut StepProfile,
) -> Result<usize> {
    let t = Instant::now();
    let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let acc = ctx.reshape(s.acc.get(), &[h.n_embd as i64, acc_rows(s)]);
    let (done, inputs) = add_group(h, &ctx, exps, s, acc, group.pairs.len());
    g.expand(done);
    let t = lap(&mut prof.build, t);
    sched.alloc_graph(&g)?;
    let splits = sched.n_splits();
    inputs.write(group)?;
    let t = lap(&mut prof.alloc, t);
    sched.compute(&g)?;
    lap(&mut prof.compute, t);
    Ok(splits)
}

/// Index tensors of a group in a graph; filled after the graph is allocated. A copied group has
/// only `pairs`.
struct GroupInputs<'a> {
    tokens: Option<Tensor<'a>>,
    ids: Option<Tensor<'a>>,
    pairs: Tensor<'a>,
}

impl GroupInputs<'_> {
    fn write(&self, g: &PairGroup) -> Result<()> {
        if let Some(t) = self.tokens {
            t.set_i32(&g.tokens)?;
        }
        if let Some(t) = self.ids {
            t.set_i32(&g.ids)?;
        }
        self.pairs.set_i32(&g.pairs)?;
        Ok(())
    }
}

/// Rows of the slot's accumulator viewed as `[n_embd, n_used * n_tokens]`.
fn acc_rows(s: &Slot) -> i64 {
    let ne = s.acc.get().ne();
    ne[1] * ne[2]
}

/// The expert FFN of `n_pairs` pairs: gather the pairs' rows of `x` (`[n_embd, n_tokens]`), run
/// the expert matmuls with one expert per row, and return the unweighted outputs
/// `[n_embd, n_pairs]` with the token and expert id index tensors to fill. One definition for
/// the groups of the segments and of the CPU graphs, so they run the same ops.
fn expert_rows<'a>(
    h: &Hparams,
    ctx: &'a Context,
    exps: [Tensor<'a>; 3],
    x: Tensor<'a>,
    n_pairs: usize,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let [gate_exps, up_exps, down_exps] = exps;
    let p = n_pairs as i64;
    let index = |ne: &[i64]| ctx.new_tensor(Type::I32, ne).set_input();
    let (tokens, ids) = (index(&[p]), index(&[1, p]));
    let xg = ctx.reshape(ctx.get_rows(x, tokens), &[h.n_embd as i64, 1, p]);
    let up = ctx.mul_mat_id(up_exps, xg, ids);
    let gate = ctx.mul_mat_id(gate_exps, xg, ids);
    let act = ctx.swiglu_split(gate, up);
    let out = ctx.mul_mat_id(down_exps, act, ids);
    (ctx.reshape(out, &[h.n_embd as i64, p]), tokens, ids)
}

/// Adds the expert FFN of a group of `n_pairs` pairs: gather the pairs' inputs, run the expert
/// matmuls with one expert per row, and write the unweighted outputs into the rows `pairs` of
/// `dst` (`[n_embd, n_used * n_tokens]`). Returns the written `dst`, to order later reads after
/// the write, and the index tensors to fill.
fn add_group<'a>(
    h: &Hparams,
    ctx: &'a Context,
    exps: [Tensor<'a>; 3],
    s: &'a Slot,
    dst: Tensor<'a>,
    n_pairs: usize,
) -> (Tensor<'a>, GroupInputs<'a>) {
    let (out, tokens, ids) = expert_rows(h, ctx, exps, s.xn.get(), n_pairs);
    let pairs = ctx.new_tensor(Type::I32, &[n_pairs as i64]).set_input();
    (
        ctx.set_rows(dst, out, pairs),
        GroupInputs {
            tokens: Some(tokens),
            ids: Some(ids),
            pairs,
        },
    )
}

/// Scatters `n_pairs` rows of the slot's `cpu_out` into the rows `pairs` of `dst`.
fn add_copied<'a>(
    h: &Hparams,
    ctx: &'a Context,
    s: &'a Slot,
    dst: Tensor<'a>,
    n_pairs: usize,
) -> (Tensor<'a>, GroupInputs<'a>) {
    let cpu_out = s.cpu_out.as_ref().expect("overlap allocates cpu_out").get();
    let row = h.n_embd as usize * 4;
    let src = ctx.view(cpu_out, &[h.n_embd as i64, n_pairs as i64], &[row], 0);
    let pairs = ctx.new_tensor(Type::I32, &[n_pairs as i64]).set_input();
    (
        ctx.set_rows(dst, src, pairs),
        GroupInputs {
            tokens: None,
            ids: None,
            pairs,
        },
    )
}

/// What the CPU groups of an overlapped layer step need.
struct OverlapCtx<'a> {
    h: &'a Hparams,
    chosen: &'a [i32],
    nu: usize,
    /// The CPU experts of the layer step and their RAM slots, same order.
    chunk: &'a [u32],
    slots: &'a [u32],
    host: &'a HostBufs,
    metrics: &'a MemoryMetrics,
    /// The CPU backend, when its threadpool is paused while waiting for reads.
    pause: Option<&'a Backend>,
}

/// One CPU group graph on the CPU scheduler: the experts' outputs for `group` land in rows
/// `row0..` of the host output buffer. Runs the ops of `add_group` on the host copy of `xn`.
fn run_cpu_group(
    h: &Hparams,
    sched: &mut Sched,
    exps: [Tensor<'_>; 3],
    host: &HostBufs,
    group: &PairGroup,
    row0: usize,
) -> Result<()> {
    let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let p = group.pairs.len();
    let (out, tokens, ids) = expert_rows(h, &ctx, exps, host.xn.get(), p);
    let row = h.n_embd as usize * 4;
    let dst = ctx.view(
        host.out.get(),
        &[h.n_embd as i64, p as i64],
        &[row],
        row0 * row,
    );
    g.expand(ctx.cpy(out, dst));
    sched.alloc_graph(&g)?;
    tokens.set_i32(&group.tokens)?;
    ids.set_i32(&group.ids)?;
    sched.compute(&g)?;
    Ok(())
}

/// Computes `subset` (experts of the layer's CPU chunk) as one group and appends its pairs to
/// `all`, whose rows are the host output buffer's rows in order. Returns the time it took.
fn compute_subset(
    c: &OverlapCtx<'_>,
    streamer: &Streamer,
    seg: usize,
    sched: &mut Sched,
    subset: &[u32],
    all: &mut PairGroup,
) -> Result<Duration> {
    let t = Instant::now();
    let slots: Vec<u32> = subset
        .iter()
        .map(|e| {
            let i = c
                .chunk
                .iter()
                .position(|x| x == e)
                .expect("a subset of the chunk");
            c.slots[i]
        })
        .collect();
    let g = group_pairs(c.chosen, c.nu, subset, &slots);
    let exps = streamer.ram_exps(seg).expect("RAM tier");
    run_cpu_group(c.h, sched, exps, c.host, &g, all.pairs.len())?;
    all.tokens.extend(&g.tokens);
    all.pairs.extend(&g.pairs);
    all.ids.extend(&g.ids);
    Ok(t.elapsed())
}

/// The CPU experts of an overlapped layer step (ADR 0054): `xn` to the host, the ready experts
/// as one group, then the others in groups as they land, the outputs into `cpu_out` of the
/// slot. Returns the concatenated pairs in the order their rows were written, and the number of
/// graphs run.
fn run_overlap_layer(
    c: &OverlapCtx<'_>,
    streamer: &mut Streamer,
    seg: usize,
    sched: &mut Sched,
    slot: &Slot,
    staged: &StagedLayer,
    prof: &mut StepProfile,
) -> Result<(PairGroup, u64)> {
    let t = Instant::now();
    let xn = slot.xn.get();
    c.host
        .xn
        .get()
        .set_bytes(&xn.get_bytes(0, xn.nbytes())?, 0)?;
    let mut copy = t.elapsed();
    let mut all = PairGroup::default();
    let mut graphs = 0;
    if !staged.ready.is_empty() {
        let d = compute_subset(c, streamer, seg, sched, &staged.ready, &mut all)?;
        prof.cpu_ready += d;
        c.metrics.layer_part("cpu_ready", d);
        graphs += 1;
    }
    let mut pending = staged.pending.clone();
    while !pending.is_empty() {
        let t = Instant::now();
        if let Some(cpu) = c.pause {
            cpu.pause_cpu_pool();
        }
        let landed = streamer.wait_landed(seg, &pending)?;
        let w = t.elapsed();
        prof.read_wait += w;
        c.metrics.layer_part("read_wait", w);
        if landed.is_empty() {
            return Err(ModelError::Invalid("no expert landed".into()));
        }
        pending.retain(|e| !landed.contains(e));
        let d = compute_subset(c, streamer, seg, sched, &landed, &mut all)?;
        prof.cpu_landed += d;
        c.metrics.layer_part("cpu_landed", d);
        graphs += 1;
    }
    let t = Instant::now();
    let bytes = c
        .host
        .out
        .get()
        .get_bytes(0, all.pairs.len() * c.h.n_embd as usize * 4)?;
    slot.cpu_out
        .as_ref()
        .expect("overlap allocates cpu_out")
        .get()
        .set_bytes(&bytes, 0)?;
    copy += t.elapsed();
    prof.copy += copy;
    c.metrics.layer_part("copy", copy);
    streamer.finish_overlap(seg);
    Ok((all, graphs))
}

/// With `slots`, the expert tensors have that many entries and are left empty (zeros) for the
/// streamer to fill; everything else is read from the file.
/// The weights and the size of their buffer.
fn load_weights(
    file: &ModelFile,
    backend: &Backend,
    slots: Option<usize>,
) -> Result<(Weights, usize)> {
    let h = &file.hparams;
    let n_tensors = 3 + h.n_layer as usize * 12;
    let ctx = Rc::new(Context::new(n_tensors + 4, 0)?);
    let mut loaded: Vec<(OwnedTensor, usize, u64, u64)> = Vec::new();
    let mut make = |name: &str| -> Result<OwnedTensor> {
        let info = file.gguf.tensor(name)?;
        let mut ne: Vec<i64> = info.ne.iter().map(|&x| x as i64).collect();
        let streamed = slots.filter(|_| name.contains("_exps."));
        if let Some(n) = streamed {
            ne[2] = n as i64;
        }
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let t = ctx.new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        if streamed.is_none() {
            if t.get().nbytes() as u64 != info.nbytes {
                return Err(ModelError::Invalid(format!("size mismatch for `{name}`")));
            }
            loaded.push((t.clone(), info.shard, info.offset, info.nbytes));
        }
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
    for il in 0..h.n_layer {
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
            gate_inp: l("ffn_gate_inp")?,
            gate_exps: l("ffn_gate_exps")?,
            up_exps: l("ffn_up_exps")?,
            down_exps: l("ffn_down_exps")?,
        });
    }
    let buffer_size = ctx.alloc_weights(backend)?;
    if slots.is_some() {
        // Defensive: slots never loaded hold zeros, not garbage.
        ctx.clear_buffer(0);
    }
    let mut shards = file
        .gguf
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
    let weights = Weights {
        tok_embd,
        out_norm,
        output,
        layers,
    };
    Ok((weights, buffer_size))
}

/// Starts the prefetch reads once layer `seg` is planned, before its own misses are read: a
/// top-up of layer `seg + 1` from the fresh prediction, then layer `seg + 2` at depth 2.
fn prefetch_ahead(
    streamer: &mut Streamer,
    file: &ModelFile,
    seg: usize,
    predicted: &Option<Vec<i32>>,
    predicted2: &[Option<Vec<i32>>],
) -> Result<()> {
    if let Some(p) = predicted {
        streamer.start_prefetch(file, seg + 1, p, 1)?;
    }
    if let Some(p) = &predicted2[seg + 2] {
        streamer.start_prefetch(file, seg + 2, p, 2)?;
    }
    Ok(())
}

/// The CPU backend's compute threads for the next graphs: `reduced` while a prefetch read is in
/// flight (`LoadOptions::prefetch_threads`), `full` otherwise.
fn set_compute_threads(cpu: &Backend, full: i32, reduced: Option<i32>, streamer: &Streamer) {
    let n = match reduced {
        Some(t) if streamer.prefetch_in_flight() => t,
        _ => full,
    };
    cpu.set_cpu_threads(n);
}

/// `host`: with overlapped decode, the CPU backend that holds the host buffers (ADR 0054).
fn alloc_state(
    h: &Hparams,
    (n, n_kv, n_out): (usize, usize, usize),
    backend: &Backend,
    host: Option<&Backend>,
) -> Result<State> {
    let ctx = Rc::new(Context::new(48, 0)?);
    let (ne, nu, nt) = (h.n_embd as i64, h.n_expert_used as i64, n as i64);
    let slot = |i: usize, nt: i64| {
        let t = |ty, dims: &[i64], what: &str| {
            let t = ctx.new_persistent(ty, dims);
            t.get().set_name(&format!("{what}_{i}"));
            t
        };
        Slot {
            resid: t(Type::F32, &[ne, nt], "resid"),
            xn: t(Type::F32, &[ne, nt], "xn"),
            w: t(Type::F32, &[1, nu, nt], "weights"),
            ids: t(Type::I32, &[nu, nt], "ids"),
            ids_map: t(Type::I32, &[nu, nt], "ids_mapped"),
            pred: t(Type::I32, &[nu, nt], "predicted"),
            pred2: t(Type::I32, &[nu, nt], "predicted2"),
            acc: t(Type::F32, &[ne, nu, nt], "moe_acc"),
            cpu_out: host.map(|_| t(Type::F32, &[ne, nu * nt], "cpu_out")),
        }
    };
    let state = State {
        n_tokens: n,
        n_kv,
        n_out,
        tokens: ctx.new_persistent(Type::I32, &[nt]),
        pos: ctx.new_persistent(Type::I32, &[nt]),
        rows: ctx.new_persistent(Type::I32, &[nt]),
        mask: ctx.new_persistent(Type::F16, &[n_kv as i64, nt]),
        outputs: ctx.new_persistent(Type::I32, &[n_out as i64]),
        slots: [slot(0, nt), slot(1, nt)],
        last: (n_out < n).then(|| slot(2, n_out as i64)),
        host: None,
    };
    ctx.alloc_tensors(backend)?;
    let host = match host {
        Some(cpu) => {
            let hctx = Rc::new(Context::new(4, 0)?);
            let bufs = HostBufs {
                xn: hctx.new_persistent(Type::F32, &[ne, nt]),
                out: hctx.new_persistent(Type::F32, &[ne, nu * nt]),
            };
            bufs.xn.get().set_name("xn_host");
            bufs.out.get().set_name("cpu_out_host");
            hctx.alloc_tensors(cpu)?;
            Some(bufs)
        }
        None => None,
    };
    let state = State { host, ..state };
    Ok(state)
}

/// Builds segment `seg`. Returns the graph and, for the last segment, the logits tensor.
#[allow(clippy::too_many_arguments)]
fn build_segment<'a>(
    h: &Hparams,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    ctx: &'a Context,
    seg: usize,
    predict_depth: usize,
    record: bool,
    groups: &[GroupSpec<'a>],
) -> (Graph<'a>, Option<Tensor<'a>>, Vec<GroupInputs<'a>>) {
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let n_layer = h.n_layer as usize;

    let mut inputs = Vec::new();
    let x = if seg == 0 {
        ctx.get_rows(w.tok_embd.get(), st.tokens.get())
    } else {
        let s = st.slot(seg - 1, n_layer);
        let moe = if !groups.is_empty() {
            // The earlier groups are in `acc`; these complete it, then the same weighting
            // and sum as a single pass (ADR 0034).
            let mut acc = ctx.reshape(s.acc.get(), &[h.n_embd as i64, acc_rows(s)]);
            for &spec in groups {
                let (done, i) = match spec {
                    GroupSpec::Exps(exps, n_pairs) => add_group(h, ctx, exps, s, acc, n_pairs),
                    GroupSpec::Copied(n_pairs) => add_copied(h, ctx, s, acc, n_pairs),
                };
                acc = done;
                inputs.push(i);
            }
            let done = ctx.reshape(acc, &s.acc.get().ne()[..3]);
            sum_experts(h, ctx, ctx.mul(done, s.w.get()))
        } else {
            moe_ffn(h, ctx, &w.layers[seg - 1], s)
        };
        ctx.add(moe, s.resid.get())
    };

    if seg == n_layer {
        let logits = head(h, ctx, w, st, x);
        g.expand(logits);
        return (g, Some(logits), inputs);
    }

    let s = st.slot(seg, n_layer);
    let full = attention_block(h, ctx, &mut g, w, kv, st, seg, x);
    // The whole-step graph's rule: the last layer continues on the output rows only.
    let select = st.last.is_some() && seg + 1 == n_layer;
    let ffn_inp = if select {
        ctx.get_rows(full, st.outputs.get())
    } else {
        full
    };
    let (xn, weights, ids) = router(h, ctx, &w.layers[seg], ffn_inp);
    g.expand(ctx.cpy(ffn_inp, s.resid.get()));
    g.expand(ctx.cpy(xn, s.xn.get()));
    g.expand(ctx.cpy(weights, s.w.get()));
    g.expand(ctx.cpy(ids, s.ids.get()));
    if select && record {
        // The log keeps every row's routing: a second router on all rows, after the real one,
        // feeds the full-size slot of this parity; the host reads it for the log only.
        let f = &st.slots[seg % 2];
        let (_, lw, li) = router(h, ctx, &w.layers[seg], full);
        g.expand(lw);
        g.expand(ctx.cpy(lw, f.w.get()));
        g.expand(ctx.cpy(li, f.ids.get()));
    }
    if predict_depth >= 1 && seg + 1 < n_layer {
        // After the real router so its nodes stay consecutive for the CUDA topk-moe fusion.
        let normed = ctx.rms_norm(ffn_inp, h.rms_eps);
        add_prediction(h, ctx, &mut g, &w.layers[seg + 1], normed, s.pred.get());
        if predict_depth >= 2 && seg + 2 < n_layer {
            add_prediction(h, ctx, &mut g, &w.layers[seg + 2], normed, s.pred2.get());
        }
    }
    (g, None, inputs)
}

/// The top-k experts `layer`'s router would choose for `normed` (the RMS-normalized FFN input
/// of an earlier layer), copied to `out`.
fn add_prediction<'a>(
    h: &Hparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    layer: &'a Layer,
    normed: Tensor<'a>,
    out: Tensor<'a>,
) {
    let xp = ctx.mul(normed, layer.ffn_norm.get());
    let probs = ctx.soft_max(ctx.mul_mat(layer.gate_inp.get(), xp));
    let pred = ctx.argsort_top_k(probs, h.n_expert_used as i32);
    g.expand(ctx.cpy(pred, out));
}

/// Nodes and tensors of a whole-step graph.
fn step_nodes(h: &Hparams) -> usize {
    STEP_NODES_PER_LAYER * (h.n_layer as usize + 1)
}

/// The whole step as one graph when every expert is resident (ADR 0036): the ops of the
/// segments, with the router's ids and weights feeding the expert matmuls directly. Returns the
/// graph, the logits and, with `record`, each layer's ids and weights as outputs.
#[allow(clippy::too_many_arguments)]
fn build_step<'a>(
    h: &Hparams,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    ctx: &'a Context,
    record: bool,
) -> (Graph<'a>, Tensor<'a>, Vec<(Tensor<'a>, Tensor<'a>)>) {
    let mut g = ctx.new_graph(step_nodes(h));
    let mut x = ctx.get_rows(w.tok_embd.get(), st.tokens.get());
    let mut routed = Vec::new();
    let n_layer = h.n_layer as usize;
    for il in 0..n_layer {
        let full = attention_block(h, ctx, &mut g, w, kv, st, il, x);
        // libllama selects the output rows after the last layer's attention (`inp_out_ids`), so
        // that layer's router and expert FFN run on the output rows only. CUDA's matmul kernel
        // depends on the row count, so running all rows differs in the last bits.
        let select = il + 1 == n_layer && st.n_out < st.n_tokens;
        let ffn_inp = if select {
            ctx.get_rows(full, st.outputs.get())
        } else {
            full
        };
        let (xn, weights, ids) = router(h, ctx, &w.layers[il], ffn_inp);
        // The router's nodes enter the graph before the expert FFN, consecutive as in libllama's
        // `build_moe_ffn` and in the segments, so CUDA's topk-moe fusion fires the same way.
        g.expand(weights);
        if record {
            // The log keeps every row's routing: when rows were selected, a second router on all
            // rows, after the real one, feeds the log only.
            let (li, lw) = if select {
                let (_, lw, li) = router(h, ctx, &w.layers[il], full);
                g.expand(lw); // the router's nodes stay consecutive, as above
                (li, lw)
            } else {
                (ids, weights)
            };
            let r = (ctx.cont(li).set_output(), ctx.cont(lw).set_output());
            g.expand(r.0);
            g.expand(r.1);
            routed.push(r);
        }
        let experts = weighted_experts(h, ctx, &w.layers[il], xn, ids, weights);
        x = ctx.add(sum_experts(h, ctx, experts), ffn_inp);
    }
    let logits = head(h, ctx, w, st, x);
    g.expand(logits);
    (g, logits, routed)
}

/// Output norm and projection; only the positions in `st.outputs` unless every token has one or
/// `x` already holds only those rows.
fn head<'a>(
    h: &Hparams,
    ctx: &'a Context,
    w: &'a Weights,
    st: &'a State,
    x: Tensor<'a>,
) -> Tensor<'a> {
    let mut cur = ctx.mul(ctx.rms_norm(x, h.rms_eps), w.out_norm.get());
    // The whole-step graph already selected the rows in the last layer; `outputs` indexes the
    // original token rows, so gathering again would read out of range.
    if st.n_out < st.n_tokens && x.ne()[1] as usize == st.n_tokens {
        cur = ctx.get_rows(cur, st.outputs.get());
    }
    let head = w.output.as_ref().unwrap_or(&w.tok_embd);
    ctx.mul_mat(head.get(), cur).set_output()
}

/// Layer `il`'s attention with Q/K norm, RoPE, the KV cache and the residual: `ffn_inp`.
#[allow(clippy::too_many_arguments)]
fn attention_block<'a>(
    h: &Hparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    il: usize,
    x: Tensor<'a>,
) -> Tensor<'a> {
    let lw = &w.layers[il];
    let step = AttnStep {
        pos: st.pos.get(),
        rows: st.rows.get(),
        mask: st.mask.get(),
        n_tokens: st.n_tokens,
        n_kv: st.n_kv,
    };
    attention(&h.attn(), ctx, g, &lw.attn, kv, il, &step, x)
}

/// The FFN norm and the router (softmax, top-k, normalized weights) of one layer for the
/// hidden state `ffn_inp`. Returns `(normed, weights, ids)`.
pub(crate) fn router<'a>(
    h: &Hparams,
    ctx: &'a Context,
    lw: &'a Layer,
    ffn_inp: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let n = ffn_inp.ne()[1];
    let xn = ctx.mul(ctx.rms_norm(ffn_inp, h.rms_eps), lw.ffn_norm.get());
    let (ne, nu) = (h.n_expert as i64, h.n_expert_used as i64);
    let probs = ctx.soft_max(ctx.mul_mat(lw.gate_inp.get(), xn));
    let ids = ctx.argsort_top_k(probs, nu as i32);
    let sel = ctx.get_rows(ctx.reshape(probs, &[1, ne, n]), ids);
    let sel = ctx.reshape(sel, &[nu, n]);
    // Normalize over the selected experts; the clamp keeps the divisor representable in F16.
    let sum = ctx.clamp(ctx.sum_rows(sel), 6.103_515_6e-5, f32::INFINITY);
    let weights = ctx.reshape(ctx.div(sel, sum), &[1, nu, n]);
    (xn, weights, ids)
}

/// Expert FFN of layer `il` for the ids in the slot's `ids_map`: `down(silu(gate) * up)`, scaled
/// by the router weights and summed over the selected experts.
fn moe_ffn<'a>(h: &Hparams, ctx: &'a Context, lw: &'a Layer, s: &'a Slot) -> Tensor<'a> {
    let experts = weighted_experts(h, ctx, lw, s.xn.get(), s.ids_map.get(), s.w.get());
    sum_experts(h, ctx, experts)
}

/// `down(silu(gate) * up)` per selected expert of the normed input `xn` `[n_embd, n]`, scaled by
/// the router weights `w` `[1, n_used, n]`: `[n_embd, n_used, n]`.
pub(crate) fn weighted_experts<'a>(
    h: &Hparams,
    ctx: &'a Context,
    lw: &'a Layer,
    xn: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
) -> Tensor<'a> {
    let n = xn.ne()[1];
    let cur = ctx.reshape(xn, &[h.n_embd as i64, 1, n]);
    let up = ctx.mul_mat_id(lw.up_exps.get(), cur, ids);
    let gate = ctx.mul_mat_id(lw.gate_exps.get(), cur, ids);
    let act = ctx.swiglu_split(gate, up);
    let experts = ctx.mul_mat_id(lw.down_exps.get(), act, ids);
    ctx.mul(experts, w)
}

/// Sum over the selected experts, in choice order: `[n_embd, n_used, n]` to `[n_embd, n]`.
pub(crate) fn sum_experts<'a>(h: &Hparams, ctx: &'a Context, experts: Tensor<'a>) -> Tensor<'a> {
    let n = experts.ne()[2];
    let [_, nb1, nb2, _] = experts.nb();
    let n_embd = h.n_embd as i64;
    let view = |i: usize| ctx.view(experts, &[n_embd, n], &[nb2], i * nb1);
    (1..h.n_expert_used as usize).fold(view(0), |acc, i| ctx.add(acc, view(i)))
}

#[cfg(test)]
mod tests {
    use super::{ExpertSrc, MAX_GROUP_ROWS, group_cap, group_pairs, offloads, split_group};

    /// Top-8 of 32 experts, 24 tokens, skewed like a real router: low ids are popular.
    fn routing() -> (Vec<i32>, usize) {
        let (n_expert, n_used, n_tokens) = (32usize, 8usize, 24usize);
        let mut x = 12345u64;
        let mut chosen = Vec::new();
        for _ in 0..n_tokens {
            let mut token: Vec<i32> = Vec::new();
            while token.len() < n_used {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (x >> 33) as f64 / (1u64 << 31) as f64;
                let e = (u * u * n_expert as f64) as i32;
                if !token.contains(&e) {
                    token.push(e);
                }
            }
            chosen.extend(token);
        }
        (chosen, n_used)
    }

    /// Every pair lands in exactly one group, in increasing pair order, and each expert has all
    /// its routed rows in its group: the CPU's K-quant `mul_mat_id` picks its kernel per expert by
    /// that row count (ADR 0034), so this keeps the groups bitwise equal to a single layer.
    #[test]
    fn groups_partition_the_pairs_and_keep_the_rows_of_every_expert() {
        let (chosen, n_used) = routing();
        let mut order: Vec<u32> = Vec::new();
        for &e in &chosen {
            if !order.contains(&(e as u32)) {
                order.push(e as u32);
            }
        }
        let rows = |e: u32| chosen.iter().filter(|&&c| c as u32 == e).count();
        for size in [4usize, 8, 12] {
            let chunks: Vec<&[u32]> = order.chunks(size).collect();
            assert!(chunks.len() > 1, "chunks of {size} need several groups");
            let mut seen = vec![0u32; chosen.len()];
            for chunk in chunks {
                // Slot numbers unrelated to the expert ids.
                let slots: Vec<u32> = (0..chunk.len() as u32).rev().collect();
                let g = group_pairs(&chosen, n_used, chunk, &slots);
                assert!(g.pairs.windows(2).all(|w| w[0] < w[1]), "pairs increase");
                for (i, &pair) in g.pairs.iter().enumerate() {
                    let pair = pair as usize;
                    seen[pair] += 1;
                    assert_eq!(g.tokens[i] as usize, pair / n_used);
                    let k = slots.iter().position(|&s| s as i32 == g.ids[i]).unwrap();
                    assert_eq!(chunk[k] as i32, chosen[pair], "slot of pair {pair}");
                }
                for (&e, &slot) in chunk.iter().zip(&slots) {
                    let got = g.ids.iter().filter(|&&i| i == slot as i32).count();
                    assert_eq!(got, rows(e), "size {size}: expert {e} in slot {slot}");
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "size {size}: each pair once");
        }
    }

    proptest::proptest! {
        /// Rule 5, every unit computed exactly once: for random routings and a random split of
        /// the distinct experts into the compute-tier group and the CPU groups of an overlapped
        /// layer (ready, then arrival order), the groups' pairs partition all pairs, and each
        /// expert's pairs are all in the group of that expert (ADR 0054).
        #[test]
        fn overlap_groups_partition_all_pairs(
            n_tokens in 1usize..12,
            n_used in 1usize..6,
            n_groups in 1usize..5,
            seed in proptest::prelude::any::<u64>(),
        ) {
            let mut x = seed | 1;
            let mut next = |m: u64| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (x >> 33) % m
            };
            let mut chosen = Vec::new();
            for _ in 0..n_tokens {
                let mut token: Vec<i32> = Vec::new();
                while token.len() < n_used {
                    let e = next(12) as i32;
                    if !token.contains(&e) {
                        token.push(e);
                    }
                }
                chosen.extend(token);
            }
            let mut distinct: Vec<u32> = Vec::new();
            for &e in &chosen {
                if !distinct.contains(&(e as u32)) {
                    distinct.push(e as u32);
                }
            }
            // Group 0 is the compute tier, 1.. the ready group and the arrival groups.
            let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_groups + 1];
            for &e in &distinct {
                members[next(n_groups as u64 + 1) as usize].push(e);
            }
            let mut seen = vec![0u32; chosen.len()];
            for (gi, experts) in members.iter().enumerate() {
                // Slots are arbitrary numbers, unrelated to the expert ids.
                let slots: Vec<u32> = (0..experts.len() as u32).rev().collect();
                let g = group_pairs(&chosen, n_used, experts, &slots);
                for &pair in &g.pairs {
                    seen[pair as usize] += 1;
                    proptest::prop_assert!(
                        experts.contains(&(chosen[pair as usize] as u32)),
                        "group {} holds a pair of another expert", gi
                    );
                }
                for &e in experts {
                    let want = chosen.iter().filter(|&&c| c as u32 == e).count();
                    let got = g.pairs.iter().filter(|&&p| chosen[p as usize] as u32 == e).count();
                    proptest::prop_assert_eq!(got, want);
                }
            }
            proptest::prop_assert!(seen.iter().all(|&n| n == 1));
        }
    }

    /// A group over the cap splits into consecutive sub-groups that concatenate to the original.
    #[test]
    fn split_keeps_order_and_covers_every_pair() {
        let (chosen, n_used) = routing();
        let experts: Vec<u32> = (0..32).collect();
        let g = group_pairs(&chosen, n_used, &experts, &experts);
        let whole = group_pairs(&chosen, n_used, &experts, &experts);
        for cap in [1usize, 7, 64, 191] {
            let parts = split_group(group_pairs(&chosen, n_used, &experts, &experts), Some(cap));
            assert_eq!(parts.len(), g.pairs.len().div_ceil(cap));
            assert!(
                parts
                    .iter()
                    .all(|p| p.pairs.len() <= cap && !p.pairs.is_empty())
            );
            let join = |f: fn(&super::PairGroup) -> &Vec<i32>| -> Vec<i32> {
                parts.iter().flat_map(|p| f(p).clone()).collect()
            };
            assert_eq!(join(|p| &p.pairs), whole.pairs);
            assert_eq!(join(|p| &p.tokens), whole.tokens);
            assert_eq!(join(|p| &p.ids), whole.ids);
        }
        // At or under the cap, or no cap: untouched.
        assert_eq!(
            split_group(
                group_pairs(&chosen, n_used, &experts, &experts),
                Some(g.pairs.len())
            )
            .len(),
            1
        );
        assert_eq!(
            split_group(group_pairs(&chosen, n_used, &experts, &experts), None).len(),
            1
        );
    }

    /// Only groups that can run on CUDA are capped.
    #[test]
    fn cap_applies_to_cuda_groups_only() {
        let c = Some(MAX_GROUP_ROWS);
        assert_eq!(group_cap(ExpertSrc::Compute, true, false), c);
        assert_eq!(group_cap(ExpertSrc::Ram, true, true), c);
        assert_eq!(group_cap(ExpertSrc::Ram, true, false), None);
        assert_eq!(group_cap(ExpertSrc::Ram, false, true), None);
        assert_eq!(group_cap(ExpertSrc::Compute, false, false), None);
    }

    /// Op offload follows the step's tokens, not its pair rows (ADR 0044): a 7-token
    /// speculative verify step (56 pairs at top-8) stays on the CPU, a prefill ubatch offloads.
    #[test]
    fn offload_is_gated_by_step_tokens() {
        assert!(!offloads(true, 32, 1));
        assert!(!offloads(true, 32, 7));
        assert!(offloads(true, 32, 32));
        assert!(offloads(true, 32, 2048));
        assert!(!offloads(false, 32, 2048));
        assert!(offloads(true, 0, 1));
    }

    /// Two tokens, top-2: the group of expert 5 holds the pairs that chose it.
    #[test]
    fn group_of_one_expert() {
        let g = group_pairs(&[5, 1, 2, 5], 2, &[5], &[3]);
        assert_eq!(g.tokens, [0, 1]);
        assert_eq!(g.pairs, [0, 3]);
        assert_eq!(g.ids, [3, 3]);
    }
}

/// The op order of the graphs where ggml-cuda decides its fusions. CUDA fuses by matching
/// consecutive nodes, so the same ops in another order run other kernels and give other bits.
/// libllama orders the router so the topk-moe fusion fires (`build_moe_ffn` expands the weights
/// early); the builder's graphs must do the same, checked here without a GPU (ADR 0036).
#[cfg(test)]
mod graph_order {
    use super::*;
    use std::path::PathBuf;

    /// `ggml_cuda_topk_moe_fusion` with softmax gating and normalized weights.
    const TOPK_MOE: [&str; 10] = [
        "SOFT_MAX", "RESHAPE", "ARGSORT", "VIEW", "GET_ROWS", "RESHAPE", "SUM_ROWS", "CLAMP",
        "DIV", "RESHAPE",
    ];

    type Ops = Vec<(String, [i64; 4])>;

    fn model() -> Qwen3Moe {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
        let opts = LoadOptions::new(BackendChoice::Cpu { threads: Some(1) });
        Qwen3Moe::load(path, &opts).expect("fixture loads")
    }

    /// Op name and shape of every node, minus the nodes `skip` drops.
    fn ops(g: &Graph<'_>, skip: impl Fn(Tensor<'_>) -> bool) -> Ops {
        (0..g.n_nodes())
            .map(|i| g.node(i))
            .filter(|&t| !skip(t))
            .map(|t| (t.op_name(), t.ne()))
            .collect()
    }

    /// Every router is the topk-moe window with the links the CUDA matcher checks, and every
    /// weighted expert output is followed by its views and adds (`moe_weighted_reduction`).
    fn assert_fusable(g: &Graph<'_>, n_layer: usize, extra: usize, n_used: usize, what: &str) {
        let n = |i: usize| g.node(i);
        let src_is = |i: usize, s: usize, t: Tensor<'_>| n(i).src(s).is_some_and(|x| x.is(t));
        let (mut routers, mut reductions) = (0, 0);
        for i in 0..g.n_nodes() {
            match n(i).op_name().as_str() {
                "SOFT_MAX" => {
                    routers += 1;
                    let got: Vec<String> = (i..(i + TOPK_MOE.len()).min(g.n_nodes()))
                        .map(|j| n(j).op_name())
                        .collect();
                    assert_eq!(got, TOPK_MOE, "{what}: router at node {i}");
                    // (node, source, node it must be) as in ggml_cuda_topk_moe_fusion.
                    let links = [
                        (i + 1, 0, i),
                        (i + 2, 0, i),
                        (i + 3, 0, i + 2),
                        (i + 4, 0, i + 1),
                        (i + 4, 1, i + 3),
                        (i + 5, 0, i + 4),
                        (i + 6, 0, i + 5),
                        (i + 7, 0, i + 6),
                        (i + 8, 0, i + 5),
                        (i + 8, 1, i + 7),
                        (i + 9, 0, i + 8),
                    ];
                    for (node, s, from) in links {
                        assert!(
                            src_is(node, s, n(from)),
                            "{what}: router at node {i}, {node}"
                        );
                    }
                }
                "MUL" if n(i).src(0).is_some_and(|e| e.op_name() == "MUL_MAT_ID") => {
                    reductions += 1;
                    let (mut views, mut adds) = (0, 0);
                    for j in i + 1..i + 2 * n_used {
                        match n(j).op_name().as_str() {
                            "VIEW" if src_is(j, 0, n(i)) => views += 1,
                            "ADD" => adds += 1,
                            op => panic!("{what}: node {j} after the weighting is {op}"),
                        }
                    }
                    assert_eq!((views, adds), (n_used, n_used - 1), "{what}: node {i}");
                }
                _ => {}
            }
        }
        assert_eq!((routers, reductions), (n_layer + extra, n_layer), "{what}");
    }

    /// The segments' ops in order, without the copies into the state slots.
    fn segment_ops(m: &Qwen3Moe, st: &State, record: bool) -> Ops {
        let h = &m.file.hparams;
        let n_layer = h.n_layer as usize;
        let to_state = |t: Tensor<'_>| {
            t.op_name() == "CPY"
                && t.src(1).is_some_and(|d| {
                    let name = d.name();
                    ["resid_", "xn_", "weights_", "ids_"]
                        .iter()
                        .any(|p| name.starts_with(p))
                })
        };
        let mut all = Vec::new();
        for seg in 0..=n_layer {
            let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES).unwrap();
            let (g, _, _) = build_segment(h, &m.weights, &m.kv, st, &ctx, seg, 0, record, &[]);
            if seg > 0 && seg < n_layer {
                // A middle segment holds the previous layer's reduction and one router, and
                // with output rows selected the last layer's also the log's router.
                let what = format!("segment {seg}");
                let extra = usize::from(record && seg + 1 == n_layer && st.last.is_some());
                assert_fusable(&g, 1, extra, h.n_expert_used as usize, &what);
            }
            all.extend(ops(&g, to_state));
        }
        all
    }

    /// The expert prediction (ADR 0038) only appends nodes to a segment: the segment's own ops,
    /// and with them the CUDA fusion windows of the router and the norms, keep their order.
    #[test]
    fn prediction_only_appends_to_a_segment() {
        let m = model();
        let h = m.file.hparams.clone();
        let n_layer = h.n_layer as usize;
        for (n, n_out) in [(1, 1), (5, 5), (5, 1)] {
            let st = alloc_state(&h, (n, pad(n, KV_PAD), n_out), &m.primary, None).unwrap();
            for seg in 0..=n_layer {
                let build = |depth: usize| {
                    let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES).unwrap();
                    let (g, _, _) =
                        build_segment(&h, &m.weights, &m.kv, &st, &ctx, seg, depth, false, &[]);
                    ops(&g, |_| false)
                };
                let plain = build(0);
                for depth in [1, 2] {
                    let pred = build(depth);
                    assert_eq!(
                        &pred[..plain.len()],
                        &plain[..],
                        "segment {seg}, n {n}, depth {depth}"
                    );
                }
            }
        }
    }

    /// The attention mask is `[n_kv, n]` like libllama's `attn_inp_kq_mask` at the pin: a decode
    /// step at 100K positions writes and uploads 200 KB, not 6.4 MB (32 padding rows before).
    #[test]
    fn mask_has_one_row_per_token_like_libllama() {
        let m = model();
        let h = m.file.hparams.clone();
        for n in [1, 5, 33] {
            let st = alloc_state(&h, (n, pad(n, KV_PAD), 1), &m.primary, None).unwrap();
            assert_eq!(st.mask.get().ne(), [pad(n, KV_PAD) as i64, n as i64, 1, 1]);
        }
        // Decode after a prefill still matches a single evaluation of the whole sequence.
        let mut a = model();
        let whole = a.eval(&[1, 2, 3, 4]).unwrap();
        let mut b = model();
        b.eval(&[1, 2, 3]).unwrap();
        let last = b.eval(&[4]).unwrap();
        let nv = a.n_vocab();
        let want = &whole[3 * nv..];
        let diff = want
            .iter()
            .zip(&last)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "max logit difference {diff}");
    }

    #[test]
    fn whole_step_graph_orders_ops_like_the_segments_and_libllama() {
        let m = model();
        let h = m.file.hparams.clone();
        for (n, n_out) in [(1, 1), (5, 5), (5, 1)] {
            let st = alloc_state(&h, (n, pad(n, KV_PAD), n_out), &m.primary, None).unwrap();
            for record in [false, true] {
                let segments = segment_ops(&m, &st, record);
                let what = format!("step n={n} record={record}");
                let ctx = Context::new(step_nodes(&h), step_nodes(&h)).unwrap();
                let (g, _, _) = build_step(&h, &m.weights, &m.kv, &st, &ctx, record);
                let (n_layer, n_used) = (h.n_layer as usize, h.n_expert_used as usize);
                // With output rows selected, a recording step has one more router, for the log.
                let extra = usize::from(record && n_out < n);
                assert_fusable(&g, n_layer, extra, n_used, &what);
                // The routing outputs are extra nodes; the rest is the segments' order.
                let step = ops(&g, |t| t.op_name() == "CONT");
                assert_eq!(step, segments, "{what}");
            }
        }
    }

    /// Like libllama's `inp_out_ids`, the last layer's router and expert FFN run on the output
    /// rows only when there are fewer outputs than tokens; earlier layers always run all rows.
    #[test]
    fn last_layer_ffn_runs_on_the_output_rows_only() {
        let m = model();
        let h = m.file.hparams.clone();
        let n_layer = h.n_layer as usize;
        for (n, n_out) in [(1, 1), (5, 5), (5, 1), (5, 2)] {
            let st = alloc_state(&h, (n, pad(n, KV_PAD), n_out), &m.primary, None).unwrap();
            let ctx = Context::new(step_nodes(&h), step_nodes(&h)).unwrap();
            let (g, _, _) = build_step(&h, &m.weights, &m.kv, &st, &ctx, false);
            let ids: Vec<[i64; 4]> = (0..g.n_nodes())
                .map(|i| g.node(i))
                .filter(|t| t.op_name() == "MUL_MAT_ID")
                .map(|t| t.ne())
                .collect();
            let gets = (0..g.n_nodes())
                .filter(|&i| g.node(i).op_name() == "GET_ROWS")
                .count();
            // Token embedding, router selection per layer, and one output selection in all.
            assert_eq!(
                gets,
                1 + n_layer + usize::from(n_out < n),
                "n={n} out={n_out}"
            );
            assert_eq!(ids.len(), 3 * n_layer);
            for (i, ne) in ids.iter().enumerate() {
                let want = if i / 3 + 1 == n_layer { n_out } else { n } as i64;
                assert_eq!(ne[2], want, "n={n} out={n_out}: MUL_MAT_ID {i}");
            }
        }
    }

    /// The segments select the output rows like the whole-step graph: the last layer's expert
    /// matmuls see `n_out` rows, there is one output selection, and the head does not gather.
    #[test]
    fn last_segment_ffn_runs_on_the_output_rows_only() {
        let m = model();
        let h = m.file.hparams.clone();
        let n_layer = h.n_layer as usize;
        for (n, n_out) in [(1, 1), (5, 5), (5, 1), (5, 2)] {
            let st = alloc_state(&h, (n, pad(n, KV_PAD), n_out), &m.primary, None).unwrap();
            let (mut rows, mut gets) = (Vec::new(), 0);
            for seg in 0..=n_layer {
                let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES).unwrap();
                let (g, _, _) = build_segment(&h, &m.weights, &m.kv, &st, &ctx, seg, 0, false, &[]);
                for i in 0..g.n_nodes() {
                    let t = g.node(i);
                    match t.op_name().as_str() {
                        "MUL_MAT_ID" => rows.push(t.ne()[2]),
                        "GET_ROWS" => gets += 1,
                        _ => {}
                    }
                }
            }
            assert_eq!(rows.len(), 3 * n_layer);
            for (i, &r) in rows.iter().enumerate() {
                let want = if i / 3 + 1 == n_layer { n_out } else { n } as i64;
                assert_eq!(r, want, "n={n} out={n_out}: MUL_MAT_ID {i}");
            }
            // Token embedding, router selection per layer, and one output selection in all.
            assert_eq!(
                gets,
                1 + n_layer + usize::from(n_out < n),
                "n={n} out={n_out}"
            );
        }
    }
}

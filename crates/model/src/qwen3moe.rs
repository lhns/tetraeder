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

use crate::convert::{ConvertError, ConvertOptions, ConvertStats};
use crate::experts::{ExpertStats, MissPolicy, RoutingStep, Streamer, StreamerConfig};
use crate::gguf::GgufError;
use crate::hparams::{Hparams, ModelFile, names};
use tetraeder_ggml_sys::ggml::{
    Backend, Context, GgmlError, Graph, OwnedGraph, OwnedTensor, RopeParams, Sched, Tensor, Type,
};
use tetraeder_memory::IoOptions;
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_plan::{Format, ModelDesc};
use tetraeder_telemetry::{Phase, StepSample, Telemetry};

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
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
    /// count, padded KV length, `last_only` and routing recording, so ggml-cuda can replay a
    /// captured CUDA graph (ADR 0036). Only with every expert resident; on by default (30B decode
    /// 160 to 203 tok/s on the RTX 3090, greedy identical).
    pub reuse_step_graph: bool,
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
    /// spin before they sleep. `None`: ggml's default (50). A lower level leaves cores to the
    /// prefetch reader when the CPU quota is tight.
    pub cpu_poll: Option<u32>,
    /// Compute threads of CPU graphs while a prefetch read is in flight (at most the full
    /// count, which the threadpool keeps). `None`: the full count.
    pub prefetch_threads: Option<usize>,
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
            .field("prefetch_threads", &self.prefetch_threads)
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
            formats: BTreeMap::new(),
            convert_cache: None,
            prefetch: true,
            prefetch_depth: 1,
            cpu_poll: None,
            prefetch_threads: None,
        }
    }
}

const KV_PAD: usize = 256;
const MASK_PAD: usize = 32;
/// Upper bound of nodes in one segment; the scheduler's hash tables are sized for it.
const SEGMENT_NODES: usize = 1024;
/// Upper bound of nodes per layer in a whole-step graph (about 70 are used).
const STEP_NODES_PER_LAYER: usize = 128;
const F16_NEG_INF: u16 = 0xFC00;

fn pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

struct Layer {
    attn_norm: OwnedTensor,
    wq: OwnedTensor,
    wk: OwnedTensor,
    wv: OwnedTensor,
    wo: OwnedTensor,
    q_norm: OwnedTensor,
    k_norm: OwnedTensor,
    ffn_norm: OwnedTensor,
    gate_inp: OwnedTensor,
    /// Stacked expert weights, or slot tensors with fewer entries when streaming.
    gate_exps: OwnedTensor,
    up_exps: OwnedTensor,
    down_exps: OwnedTensor,
}

impl Layer {
    fn exps(&self) -> [Tensor<'_>; 3] {
        [
            self.gate_exps.get(),
            self.up_exps.get(),
            self.down_exps.get(),
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

/// Contiguous per-layer F16 KV cache: `[n_embd_kv, n_ctx]` for K and for V.
struct Kv {
    k: Vec<OwnedTensor>,
    v: Vec<OwnedTensor>,
    n_ctx: usize,
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
}

/// Per-step inputs and slots; reallocated when the token count or the padded KV length changes.
struct State {
    n_tokens: usize,
    n_kv: usize,
    tokens: OwnedTensor,
    pos: OwnedTensor,
    mask: OwnedTensor,
    slots: [Slot; 2],
}

/// A whole-step graph kept across steps (`LoadOptions::reuse_step_graph`). It holds the contexts
/// of the weights, KV cache and state it reads, so the state must be dropped after it.
struct StepGraph {
    /// Token count, padded KV length, `last_only` and routing recording of the build.
    key: (usize, usize, bool, bool),
    graph: OwnedGraph,
    logits: OwnedTensor,
    /// Per layer, the router's ids and weights (with routing recording).
    routed: Vec<(OwnedTensor, OwnedTensor)>,
}

/// A Qwen3 MoE model. The experts are all resident in backend memory or streamed through slots.
pub struct Qwen3Moe {
    file: ModelFile,
    sched: Sched,
    /// The scheduler without op offload for small streamed steps (ADR 0044), with `op_offload`
    /// on a GPU backend.
    sched_small: Option<Sched>,
    primary: Backend,
    cpu: Backend,
    /// Size of the weights buffer on the primary backend (slot tensors included) and of the KV
    /// buffer.
    weights_bytes: u64,
    kv_bytes: u64,
    weights: Weights,
    kv: Kv,
    step: Option<StepGraph>,
    state: Option<State>,
    n_past: usize,
    reuse_step_graph: bool,
    step_graphs_built: u64,
    segments_run: u64,
    groups_run: u64,
    group_graphs_run: u64,
    max_splits: usize,
    policy: MissPolicy,
    /// Each segment also predicts the experts of this many layers ahead (ADR 0038), 0 for none:
    /// only with prefetch, the cpu miss policy and a RAM tier, where the reads can be prefetched.
    predict_depth: usize,
    /// Threads of the CPU backend, and the count while a prefetch read is in flight.
    cpu_threads: i32,
    prefetch_threads: Option<i32>,
    /// Pair groups may run on a GPU backend (the primary is not the CPU).
    gpu: bool,
    op_offload: bool,
    op_offload_min_tokens: usize,
    /// The hot set still has to be built from the first prefill's routing.
    hot_pending: bool,
    streamer: Option<Streamer>,
    metrics: MemoryMetrics,
    telemetry: Option<Arc<Telemetry>>,
    routing: Option<Vec<RoutingStep>>,
    profile: StepProfile,
    convert: ConvertStats,
}

impl std::fmt::Debug for Qwen3Moe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Moe")
            .field("hparams", &self.file.hparams)
            .field("backend", &self.primary)
            .field("n_past", &self.n_past)
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
        if opts.miss == MissPolicy::Cpu && ram_slots.is_none() {
            return Err(ModelError::Invalid(
                "the cpu miss policy needs ram_slots".into(),
            ));
        }
        let (primary, cpu, backends, cpu_threads) = match opts.backend {
            BackendChoice::Cpu { threads } => {
                let n = threads_for(threads) as i32;
                let cpu = Backend::cpu_with_poll(n, opts.cpu_poll)?;
                (cpu.clone(), cpu.clone(), vec![cpu], n)
            }
            BackendChoice::Gpu { index } => {
                let gpu = Backend::gpu(index)?;
                let n = threads_for(None) as i32;
                let cpu = Backend::cpu_with_poll(n, opts.cpu_poll)?;
                (gpu.clone(), cpu.clone(), vec![gpu, cpu], n)
            }
        };
        let sched = Sched::new(
            &backends,
            step_nodes(h).max(SEGMENT_NODES) * 4,
            opts.op_offload,
        )?;
        // Streamed steps below `op_offload_min_tokens` use a scheduler without op offload
        // (ADR 0044); its compute buffers are sized by those small steps.
        let sched_small = if opts.op_offload && !backends[0].is_cpu() {
            Some(Sched::new(
                &backends,
                step_nodes(h).max(SEGMENT_NODES) * 4,
                false,
            )?)
        } else {
            None
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
        let (weights, weights_bytes) = load_weights(&file, &primary, slots)?;
        let (kv, kv_bytes) = alloc_kv(h, pad(opts.n_ctx.max(1), KV_PAD), &primary)?;
        let metrics = MemoryMetrics::new(opts.telemetry.clone());
        let streamer = slots
            .map(|n| {
                let tensors: Vec<_> = weights.layers.iter().map(Layer::exps).collect();
                let cfg = StreamerConfig {
                    slots: n,
                    ram_slots: ram_slots.map(|r| r.max(1)),
                    io: opts.io,
                    profile: opts.routing_profile.clone(),
                    prefetch: opts.prefetch,
                };
                Streamer::new(
                    &file,
                    &tensors,
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
        let opts_primary_is_cpu = primary.is_cpu();
        let mut model = Qwen3Moe {
            file,
            sched,
            sched_small,
            primary,
            cpu,
            weights_bytes: weights_bytes as u64,
            kv_bytes: kv_bytes as u64,
            weights,
            kv,
            step: None,
            state: None,
            n_past: 0,
            reuse_step_graph: opts.reuse_step_graph,
            step_graphs_built: 0,
            segments_run: 0,
            groups_run: 0,
            group_graphs_run: 0,
            max_splits: 0,
            policy: opts.miss,
            predict_depth: if opts.prefetch && cpu_hot {
                opts.prefetch_depth
            } else {
                0
            },
            cpu_threads,
            prefetch_threads: opts
                .prefetch_threads
                .map(|t| (t as i32).clamp(1, cpu_threads)),
            gpu: !opts_primary_is_cpu,
            op_offload: opts.op_offload,
            op_offload_min_tokens: opts.op_offload_min_tokens,
            hot_pending: cpu_hot && opts.routing_profile.is_none(),
            streamer,
            metrics,
            telemetry: opts.telemetry.clone(),
            routing: opts.record_routing.then(Vec::new),
            profile: StepProfile::default(),
            convert,
        };
        if cpu_hot && opts.routing_profile.is_some() {
            model.refresh_hot_set()?;
        }
        Ok(model)
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

    /// Tokens in the KV cache.
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn n_ctx(&self) -> usize {
        self.kv.n_ctx
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
            self.step = None;
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

    /// Every buffer the builder owns with its capacity and used bytes (ADR 0037): the weights,
    /// the compute slot tensors, the KV cache, the RAM tier and the scheduler's compute buffers.
    /// Cheap: it reads sizes and the slot tables.
    pub fn memory(&self) -> Vec<MemoryUse> {
        let tier = if self.primary.is_cpu() { "ram" } else { "vram" };
        let device = self.primary.name();
        let mut out = Vec::new();
        let mut push = |device: &str, tier, kind, capacity: u64, used: u64| {
            out.push(MemoryUse {
                device: device.to_string(),
                tier,
                kind,
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
        let scheds = || std::iter::once(&self.sched).chain(&self.sched_small);
        let work = scheds().map(|s| s.buffer_size(&self.primary) as u64).sum();
        push(&device, tier, "workspace", work, work);
        if !self.primary.is_cpu() {
            let work = scheds().map(|s| s.buffer_size(&self.cpu) as u64).sum();
            push(&self.cpu.name(), "ram", "workspace", work, work);
        }
        out
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

    /// The time breakdown since load or the last call, then starts a new one.
    pub fn take_profile(&mut self) -> StepProfile {
        std::mem::take(&mut self.profile)
    }

    /// Forgets the KV cache; the next `eval` starts at position 0.
    pub fn reset(&mut self) {
        self.n_past = 0;
    }

    /// Rolls the cache back to its first `n_past` tokens. Rows above stay in the KV buffers but
    /// `run()` masks every column above `n_past + i` and the next step overwrites them, so
    /// nothing needs clearing (ADR 0043).
    pub fn rewind(&mut self, n_past: usize) -> Result<()> {
        if n_past > self.n_past {
            return Err(ModelError::Invalid(format!(
                "cannot rewind to {n_past}: only {} tokens cached",
                self.n_past
            )));
        }
        self.n_past = n_past;
        Ok(())
    }

    /// Bytes of one cached position across all layers (K and V).
    fn kv_row_bytes(&self) -> usize {
        self.kv
            .k
            .iter()
            .chain(&self.kv.v)
            .map(|t| t.get().nbytes() / self.kv.n_ctx)
            .sum()
    }

    /// Copies the K and V rows of the cached positions to host bytes, layer by layer (ADR 0049).
    /// The cache is `[n_embd_kv, n_ctx]` per layer, so the rows of positions `0..n_past` are
    /// one contiguous range.
    pub fn kv_snapshot(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.n_past * self.kv_row_bytes());
        for t in self.kv.k.iter().chain(&self.kv.v) {
            let row = t.get().nbytes() / self.kv.n_ctx;
            out.extend_from_slice(&t.get().get_bytes(0, self.n_past * row)?);
        }
        Ok(out)
    }

    /// Writes a [`Qwen3Moe::kv_snapshot`] of `n` positions back and sets the cache length to `n`.
    pub fn kv_restore(&mut self, bytes: &[u8], n: usize) -> Result<()> {
        if n > self.kv.n_ctx || bytes.len() != n * self.kv_row_bytes() {
            return Err(ModelError::Invalid(format!(
                "a KV snapshot of {} bytes does not hold {n} positions",
                bytes.len()
            )));
        }
        let mut rest = bytes;
        for t in self.kv.k.iter().chain(&self.kv.v) {
            let len = n * (t.get().nbytes() / self.kv.n_ctx);
            let (head, tail) = rest.split_at(len);
            t.get().set_bytes(head, 0)?;
            rest = tail;
        }
        self.n_past = n;
        Ok(())
    }

    /// Runs `tokens` after the tokens already in the cache and returns the logits of every
    /// position, `tokens.len() * n_vocab` values, position-major.
    pub fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.run(tokens, false)
    }

    /// Like [`Qwen3Moe::eval`] but only for the last position (`n_vocab` values): prefill does
    /// not need the head for the other positions.
    pub fn eval_last(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.run(tokens, true)
    }

    fn run(&mut self, tokens: &[i32], last_only: bool) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let h = self.file.hparams.clone();
        let n = tokens.len();
        if n == 0 {
            return Err(ModelError::Invalid("no tokens".into()));
        }
        if self.n_past + n > self.kv.n_ctx {
            return Err(ModelError::Invalid(format!(
                "context full: {} cached + {n} new > {}",
                self.n_past, self.kv.n_ctx
            )));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t < 0 || t as u32 >= h.n_vocab) {
            return Err(ModelError::Invalid(format!(
                "token {t} outside the vocabulary"
            )));
        }
        let n_kv = pad(self.n_past + n, KV_PAD);
        if self
            .state
            .as_ref()
            .is_none_or(|s| s.n_tokens != n || s.n_kv != n_kv)
        {
            // The kept graph holds the old state's context: drop it first, then free the buffers.
            self.step = None;
            self.state = None;
            self.state = Some(alloc_state(&h, n, n_kv, &self.primary)?);
        }
        let st = self.state.as_ref().expect("state allocated above");
        st.tokens.get().set_i32(tokens)?;
        let pos: Vec<i32> = (self.n_past..self.n_past + n).map(|p| p as i32).collect();
        st.pos.get().set_i32(&pos)?;
        let mut mask = vec![0u16; n_kv * pad(n, MASK_PAD)];
        for i in 0..n {
            for j in (self.n_past + i + 1)..n_kv {
                mask[i * n_kv + j] = F16_NEG_INF;
            }
        }
        st.mask.get().set_f16_bits(&mask)?;
        let mut prof = self.profile;
        prof.steps += 1;
        prof.inputs += t0.elapsed();

        if self.streamer.is_none() {
            // Every expert resident: no host work between layers, so the step is one graph
            // (ADR 0036).
            let mut t = Instant::now();
            let record = self.routing.is_some();
            let key = (n, n_kv, last_only, record);
            let reusable = self.reuse_step_graph
                && self
                    .step
                    .as_ref()
                    .is_some_and(|s| s.key == key && self.sched.is_allocated(&s.graph.get()));
            // A kept graph is computed again with the new inputs: no build, no allocation.
            // Otherwise the graph is built and allocated; it is kept only with reuse on.
            let fresh;
            let step: &StepGraph = if reusable {
                self.step.as_ref().expect("checked above")
            } else {
                self.step = None;
                let ctx = Rc::new(Context::new(step_nodes(&h), step_nodes(&h))?);
                let (graph, out, routed) =
                    build_step(&h, &self.weights, &self.kv, st, &ctx, last_only, record);
                let deps = [
                    &self.weights.tok_embd,
                    &self.kv.k[0],
                    &self.kv.v[0],
                    &st.tokens,
                ];
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
                self.sched.alloc_graph(&fresh.graph.get())?;
                self.max_splits = self.max_splits.max(self.sched.n_splits());
                self.step_graphs_built += 1;
                t = lap(&mut prof.alloc, t);
                if self.reuse_step_graph {
                    self.step.insert(fresh)
                } else {
                    &fresh
                }
            };
            self.sched.compute(&step.graph.get())?;
            self.segments_run += 1;
            prof.graphs += 1;
            let logits = step.logits.get().get_f32()?;
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
            return self.finish_step(n, t0, prof, 1, logits);
        }

        let graphs_before = self.group_graphs_run;
        let nu = h.n_expert_used as usize;
        let n_layer = h.n_layer as usize;
        let mut logits = Vec::new();
        // The pair groups of the previous layer step that run inside this segment.
        let mut pending: Vec<PendingGroup> = Vec::new();
        // The predictions of the experts of a layer (ADR 0038): the next layer's from the
        // previous segment, and, at depth 2, per target layer the one made two segments ago.
        let mut predicted: Option<Vec<i32>> = None;
        let mut predicted2: Vec<Option<Vec<i32>>> = vec![None; n_layer + 2];
        if let Some(streamer) = &mut self.streamer {
            // A step that failed may have left a prefetch in flight.
            streamer.settle();
        }
        self.cpu.set_cpu_threads(self.cpu_threads);
        let offload = offloads(self.op_offload, self.op_offload_min_tokens, n);
        let sched = match &mut self.sched_small {
            Some(small) if !offload => small,
            _ => &mut self.sched,
        };
        for seg in 0..=n_layer {
            let groups = std::mem::take(&mut pending);
            {
                let t = Instant::now();
                let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
                let specs: Vec<([Tensor<'_>; 3], usize)> = groups
                    .iter()
                    .map(|g| {
                        let lw = &self.weights.layers[seg - 1];
                        let exps = group_exps(g.src, lw, self.streamer.as_ref(), seg - 1);
                        (exps, g.pairs.pairs.len())
                    })
                    .collect();
                let (graph, out, inputs) = build_segment(
                    &h,
                    &self.weights,
                    &self.kv,
                    st,
                    &ctx,
                    seg,
                    last_only,
                    self.predict_depth,
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
                sched.compute(&graph)?;
                self.segments_run += 1;
                prof.graphs += 1;
                if let Some(l) = out {
                    logits = l.get_f32()?;
                }
                lap(&mut prof.compute, t);
            }
            if seg == n_layer {
                continue;
            }
            let t = Instant::now();
            let slot = &st.slots[seg % 2];
            let chosen = slot.ids.get().get_i32()?;
            let streamer = self
                .streamer
                .as_mut()
                .expect("a resident step returned above");
            if let Some(p) = predicted.take() {
                streamer.score_prediction(seg, 1, &p, &chosen, nu);
            }
            if let Some(p) = &predicted2[seg] {
                streamer.score_prediction(seg, 2, p, &chosen, nu);
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
                log.push(RoutingStep {
                    layer: seg,
                    ids: chosen.clone(),
                    weights: slot.w.get().get_f32()?,
                    n_tokens: n,
                    predicted: predicted.clone(),
                    predicted2: predicted2.get(seg + 2).cloned().flatten(),
                    refresh: false,
                });
            }
            lap(&mut prof.readback, t);
            // The reads started for this layer must have landed before it is planned.
            streamer.finish_prefetch(seg)?;
            set_compute_threads(&self.cpu, self.cpu_threads, self.prefetch_threads, streamer);
            let layer = &self.weights.layers[seg];
            if self.policy == MissPolicy::Cpu {
                // Resident experts run from the compute slots, the others from the RAM slots
                // on the CPU, in chunks of the RAM slot count (ADR 0035).
                let plan = streamer.plan_cpu(seg, &chosen)?;
                // The next layers' reads start before this layer's own misses are read: the
                // reader has direct files of its own, and the device serves both at once.
                prefetch_ahead(streamer, &self.file, seg, &predicted, &predicted2)?;
                set_compute_threads(&self.cpu, self.cpu_threads, self.prefetch_threads, streamer);
                if plan.cpu.is_empty() {
                    let ids = map_ids(&chosen, &plan.compute, &plan.compute_slots);
                    slot.ids_map.get().set_i32(&ids)?;
                    continue;
                }
                let cap_ram = group_cap(ExpertSrc::Ram, self.gpu, offload);
                let cap_compute = group_cap(ExpertSrc::Compute, self.gpu, offload);
                let last = plan.cpu.len() - 1;
                for (i, chunk) in plan.cpu.iter().enumerate() {
                    let slots = streamer.stage_ram(&self.file, seg, chunk)?;
                    let pairs = group_pairs(&chosen, nu, chunk, &slots);
                    for pairs in split_group(pairs, cap_ram) {
                        if i == last {
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
        // Segment graphs plus the standalone group graphs of this step (ADR 0035).
        let graphs = h.n_layer + 1 + (self.group_graphs_run - graphs_before) as u32;
        prof.graphs += self.group_graphs_run - graphs_before;
        // The rest of this step's time is the streamer's.
        let timed = |p: &StepProfile| p.inputs + p.build + p.alloc + p.compute + p.readback;
        let others = timed(&prof) - timed(&self.profile);
        prof.streamer += t0.elapsed().saturating_sub(others);
        self.finish_step(n, t0, prof, graphs, logits)
    }

    /// Bookkeeping after a step that computed `graphs` graphs.
    fn finish_step(
        &mut self,
        n: usize,
        t0: Instant,
        mut prof: StepProfile,
        graphs: u32,
        logits: Vec<f32>,
    ) -> Result<Vec<f32>> {
        self.n_past += n;
        prof.total += t0.elapsed();
        self.profile = prof;
        if let Some(t) = &self.telemetry {
            t.record_step(StepSample {
                phase: if n > 1 { Phase::Prefill } else { Phase::Decode },
                stage: 0,
                graphs: false,
                duration: t0.elapsed(),
                tokens: n as u32,
                segments: graphs,
            });
            for m in self.memory() {
                self.metrics
                    .buffer(&m.device, m.tier, m.kind, m.used, m.capacity);
            }
        }
        if n > 1 && self.hot_pending {
            // The first prefill's routing decides the hot set (ADR 0035).
            self.hot_pending = false;
            self.refresh_hot_set()?;
        }
        Ok(logits)
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

/// Index tensors of a group in a graph; filled after the graph is allocated.
struct GroupInputs<'a> {
    tokens: Tensor<'a>,
    ids: Tensor<'a>,
    pairs: Tensor<'a>,
}

impl GroupInputs<'_> {
    fn write(&self, g: &PairGroup) -> Result<()> {
        self.tokens.set_i32(&g.tokens)?;
        self.ids.set_i32(&g.ids)?;
        self.pairs.set_i32(&g.pairs)?;
        Ok(())
    }
}

/// Rows of the slot's accumulator viewed as `[n_embd, n_used * n_tokens]`.
fn acc_rows(s: &Slot) -> i64 {
    let ne = s.acc.get().ne();
    ne[1] * ne[2]
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
    let [gate_exps, up_exps, down_exps] = exps;
    let p = n_pairs as i64;
    let index = |ne: &[i64]| ctx.new_tensor(Type::I32, ne).set_input();
    let (tokens, ids, pairs) = (index(&[p]), index(&[1, p]), index(&[p]));
    let xg = ctx.reshape(ctx.get_rows(s.xn.get(), tokens), &[h.n_embd as i64, 1, p]);
    let up = ctx.mul_mat_id(up_exps, xg, ids);
    let gate = ctx.mul_mat_id(gate_exps, xg, ids);
    let act = ctx.swiglu_split(gate, up);
    let out = ctx.mul_mat_id(down_exps, act, ids);
    let out = ctx.reshape(out, &[h.n_embd as i64, p]);
    (
        ctx.set_rows(dst, out, pairs),
        GroupInputs { tokens, ids, pairs },
    )
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
            attn_norm: l("attn_norm")?,
            wq: l("attn_q")?,
            wk: l("attn_k")?,
            wv: l("attn_v")?,
            wo: l("attn_output")?,
            q_norm: l("attn_q_norm")?,
            k_norm: l("attn_k_norm")?,
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
        streamer.start_prefetch(file, seg + 1, p)?;
    }
    if let Some(p) = &predicted2[seg + 2] {
        streamer.start_prefetch(file, seg + 2, p)?;
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

fn alloc_kv(h: &Hparams, n_ctx: usize, backend: &Backend) -> Result<(Kv, usize)> {
    let ctx = Rc::new(Context::new(2 * h.n_layer as usize + 4, 0)?);
    let n_embd_kv = (h.n_head_kv * h.head_dim) as i64;
    let mk = |kind: &str, il: u32| {
        let t = ctx.new_persistent(Type::F16, &[n_embd_kv, n_ctx as i64]);
        t.get().set_name(&format!("{kind}_cache_{il}"));
        t
    };
    let k = (0..h.n_layer).map(|il| mk("k", il)).collect();
    let v = (0..h.n_layer).map(|il| mk("v", il)).collect();
    let bytes = ctx.alloc_tensors(backend)?;
    // Padding rows are masked, but 0 * NaN is NaN: never leave them uninitialized.
    ctx.clear_buffer(0);
    Ok((Kv { k, v, n_ctx }, bytes))
}

fn alloc_state(h: &Hparams, n: usize, n_kv: usize, backend: &Backend) -> Result<State> {
    let ctx = Rc::new(Context::new(32, 0)?);
    let (ne, nu, nt) = (h.n_embd as i64, h.n_expert_used as i64, n as i64);
    let slot = |i: usize| {
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
        }
    };
    let state = State {
        n_tokens: n,
        n_kv,
        tokens: ctx.new_persistent(Type::I32, &[nt]),
        pos: ctx.new_persistent(Type::I32, &[nt]),
        mask: ctx.new_persistent(Type::F16, &[n_kv as i64, pad(n, MASK_PAD) as i64]),
        slots: [slot(0), slot(1)],
    };
    ctx.alloc_tensors(backend)?;
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
    last_only: bool,
    predict_depth: usize,
    groups: &[([Tensor<'a>; 3], usize)],
) -> (Graph<'a>, Option<Tensor<'a>>, Vec<GroupInputs<'a>>) {
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let n_layer = h.n_layer as usize;

    let mut inputs = Vec::new();
    let x = if seg == 0 {
        ctx.get_rows(w.tok_embd.get(), st.tokens.get())
    } else {
        let s = &st.slots[(seg - 1) % 2];
        let moe = if !groups.is_empty() {
            // The earlier groups are in `acc`; these complete it, then the same weighting
            // and sum as a single pass (ADR 0034).
            let mut acc = ctx.reshape(s.acc.get(), &[h.n_embd as i64, acc_rows(s)]);
            for &(exps, n_pairs) in groups {
                let (done, i) = add_group(h, ctx, exps, s, acc, n_pairs);
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
        let logits = head(h, ctx, w, st, x, last_only);
        g.expand(logits);
        return (g, Some(logits), inputs);
    }

    let s = &st.slots[seg % 2];
    let (ffn_inp, xn, weights, ids) = attention_and_router(h, ctx, &mut g, w, kv, st, seg, x);
    g.expand(ctx.cpy(ffn_inp, s.resid.get()));
    g.expand(ctx.cpy(xn, s.xn.get()));
    g.expand(ctx.cpy(weights, s.w.get()));
    g.expand(ctx.cpy(ids, s.ids.get()));
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
    last_only: bool,
    record: bool,
) -> (Graph<'a>, Tensor<'a>, Vec<(Tensor<'a>, Tensor<'a>)>) {
    let mut g = ctx.new_graph(step_nodes(h));
    let mut x = ctx.get_rows(w.tok_embd.get(), st.tokens.get());
    let mut routed = Vec::new();
    for il in 0..h.n_layer as usize {
        let (ffn_inp, xn, weights, ids) = attention_and_router(h, ctx, &mut g, w, kv, st, il, x);
        // The router's nodes enter the graph before the expert FFN, consecutive as in libllama's
        // `build_moe_ffn` and in the segments, so CUDA's topk-moe fusion fires the same way.
        g.expand(weights);
        if record {
            let r = (ctx.cont(ids).set_output(), ctx.cont(weights).set_output());
            g.expand(r.0);
            g.expand(r.1);
            routed.push(r);
        }
        let experts = weighted_experts(h, ctx, &w.layers[il], xn, ids, weights);
        x = ctx.add(sum_experts(h, ctx, experts), ffn_inp);
    }
    let logits = head(h, ctx, w, st, x, last_only);
    g.expand(logits);
    (g, logits, routed)
}

/// Output norm and projection; only the last position when `last_only`.
fn head<'a>(
    h: &Hparams,
    ctx: &'a Context,
    w: &'a Weights,
    st: &'a State,
    x: Tensor<'a>,
    last_only: bool,
) -> Tensor<'a> {
    let mut cur = ctx.mul(ctx.rms_norm(x, h.rms_eps), w.out_norm.get());
    if last_only && st.n_tokens > 1 {
        let nb1 = cur.nb()[1];
        cur = ctx.view(cur, &[h.n_embd as i64, 1], &[nb1], (st.n_tokens - 1) * nb1);
    }
    let head = w.output.as_ref().unwrap_or(&w.tok_embd);
    ctx.mul_mat(head.get(), cur).set_output()
}

/// Layer `il`: attention with Q/K norm, RoPE and the KV cache, the residual, the FFN norm and the
/// router (softmax, top-k, normalized weights). Returns `(ffn_inp, normed, weights, ids)`.
#[allow(clippy::too_many_arguments)]
fn attention_and_router<'a>(
    h: &Hparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    il: usize,
    x: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let lw = &w.layers[il];
    let n = st.n_tokens as i64;
    let (hd, nh, nkv) = (h.head_dim as i64, h.n_head as i64, h.n_head_kv as i64);
    let rope = RopeParams {
        n_dims: h.n_rot as i32,
        mode: 2, // NeoX
        n_ctx_orig: h.n_ctx_train as i32,
        freq_base: h.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };

    let cur = ctx.mul(ctx.rms_norm(x, h.rms_eps), lw.attn_norm.get());
    let q = ctx.reshape(ctx.mul_mat(lw.wq.get(), cur), &[hd, nh, n]);
    let k = ctx.reshape(ctx.mul_mat(lw.wk.get(), cur), &[hd, nkv, n]);
    let v = ctx.mul_mat(lw.wv.get(), cur);
    let q = ctx.mul(ctx.rms_norm(q, h.rms_eps), lw.q_norm.get());
    let q = ctx.rope_ext(q, st.pos.get(), &rope);
    let k = ctx.mul(ctx.rms_norm(k, h.rms_eps), lw.k_norm.get());
    let k = ctx.rope_ext(k, st.pos.get(), &rope);

    // Store K and V in the cache rows of these positions; the cache must be written before the
    // attention reads it, so the stores enter the graph first.
    let (kc, vc) = (kv.k[il].get(), kv.v[il].get());
    // `set_rows` with the positions as row indices, as libllama does: unlike a copy into a view
    // at `n_past`, the node is the same every step, so ggml-cuda can replay a captured CUDA
    // graph (ADR 0036).
    let row = kc.nb()[1];
    let n_embd_kv = hd * nkv;
    for (src, cache) in [(k, kc), (v, vc)] {
        let src = ctx.reshape(src, &[n_embd_kv, n]);
        g.expand(ctx.set_rows(cache, src, st.pos.get()));
    }

    let n_kv = st.n_kv as i64;
    let kview = ctx.view(kc, &[hd, n_kv, nkv], &[row, hd as usize * 2], 0);
    let vview = ctx.view(vc, &[hd, n_kv, nkv], &[row, hd as usize * 2], 0);
    let q = ctx.permute(q, [0, 2, 1, 3]);
    let attn = ctx.flash_attn_ext(q, kview, vview, st.mask.get(), 1.0 / (hd as f32).sqrt());
    let attn = ctx.reshape(attn, &[hd * nh, n]);
    let ffn_inp = ctx.add(ctx.mul_mat(lw.wo.get(), attn), x);

    let xn = ctx.mul(ctx.rms_norm(ffn_inp, h.rms_eps), lw.ffn_norm.get());
    let (ne, nu) = (h.n_expert as i64, h.n_expert_used as i64);
    let probs = ctx.soft_max(ctx.mul_mat(lw.gate_inp.get(), xn));
    let ids = ctx.argsort_top_k(probs, nu as i32);
    let sel = ctx.get_rows(ctx.reshape(probs, &[1, ne, n]), ids);
    let sel = ctx.reshape(sel, &[nu, n]);
    // Normalize over the selected experts; the clamp keeps the divisor representable in F16.
    let sum = ctx.clamp(ctx.sum_rows(sel), 6.103_515_6e-5, f32::INFINITY);
    let weights = ctx.reshape(ctx.div(sel, sum), &[1, nu, n]);
    (ffn_inp, xn, weights, ids)
}

/// Expert FFN of layer `il` for the ids in the slot's `ids_map`: `down(silu(gate) * up)`, scaled
/// by the router weights and summed over the selected experts.
fn moe_ffn<'a>(h: &Hparams, ctx: &'a Context, lw: &'a Layer, s: &'a Slot) -> Tensor<'a> {
    let experts = weighted_experts(h, ctx, lw, s.xn.get(), s.ids_map.get(), s.w.get());
    sum_experts(h, ctx, experts)
}

/// `down(silu(gate) * up)` per selected expert of the normed input `xn` `[n_embd, n]`, scaled by
/// the router weights `w` `[1, n_used, n]`: `[n_embd, n_used, n]`.
fn weighted_experts<'a>(
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
fn sum_experts<'a>(h: &Hparams, ctx: &'a Context, experts: Tensor<'a>) -> Tensor<'a> {
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
    fn assert_fusable(g: &Graph<'_>, n_layer: usize, n_used: usize, what: &str) {
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
        assert_eq!((routers, reductions), (n_layer, n_layer), "{what}");
    }

    /// The segments' ops in order, without the copies into the state slots.
    fn segment_ops(m: &Qwen3Moe, st: &State, last_only: bool) -> Ops {
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
            let (g, _, _) = build_segment(h, &m.weights, &m.kv, st, &ctx, seg, last_only, 0, &[]);
            if seg > 0 && seg < n_layer {
                // A middle segment holds the previous layer's reduction and one router.
                let what = format!("segment {seg}");
                assert_fusable(&g, 1, h.n_expert_used as usize, &what);
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
        for (n, last_only) in [(1, false), (5, true)] {
            let st = alloc_state(&h, n, pad(n, KV_PAD), &m.primary).unwrap();
            for seg in 0..=n_layer {
                let build = |depth: usize| {
                    let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES).unwrap();
                    let (g, _, _) =
                        build_segment(&h, &m.weights, &m.kv, &st, &ctx, seg, last_only, depth, &[]);
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

    #[test]
    fn whole_step_graph_orders_ops_like_the_segments_and_libllama() {
        let m = model();
        let h = m.file.hparams.clone();
        for (n, last_only) in [(1, false), (5, true)] {
            let st = alloc_state(&h, n, pad(n, KV_PAD), &m.primary).unwrap();
            let segments = segment_ops(&m, &st, last_only);
            for record in [false, true] {
                let what = format!("step n={n} record={record}");
                let ctx = Context::new(step_nodes(&h), step_nodes(&h)).unwrap();
                let (g, _, _) = build_step(&h, &m.weights, &m.kv, &st, &ctx, last_only, record);
                let (n_layer, n_used) = (h.n_layer as usize, h.n_expert_used as usize);
                assert_fusable(&g, n_layer, n_used, &what);
                // The routing outputs are extra nodes; the rest is the segments' order.
                let step = ops(&g, |t| t.op_name() == "CONT");
                assert_eq!(step, segments, "{what}");
            }
        }
    }
}

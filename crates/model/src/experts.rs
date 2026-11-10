//! Expert streaming for the builder (ADR 0033, 0035): the slot caches of both tiers, the loads
//! from the GGUF through direct I/O, the routing counts behind the hot set, and the counters.
//! No graph work: the builder asks for the experts of a layer step to be resident and gets slot
//! ids back, or, under [`MissPolicy::Cpu`], a split into compute-tier hits and CPU chunks.

use crate::hparams::ModelFile;
use crate::qwen3moe::{ModelError, Result};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedTensor, Tensor};
use tetraeder_memory::io::{
    Deadline, Io, IoFactory, IoService, Part, ServiceOptions, UnitDone, UnitId,
};
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_memory::reader::{HostDst, ReadJob, Reader, Ticket, copy_sink, tensor_dst};
use tetraeder_memory::replay::prefetch_plan;
use tetraeder_memory::slots::{LayerStats, Lookup, SlotCache, Tier, hot_from_counts};
use tetraeder_memory::{DirectFile, IoOptions, Range};

/// What a layer step does with a routed expert that is not resident in the compute tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MissPolicy {
    /// Load it into a compute slot (from the RAM tier or the file). More distinct experts than
    /// slots become several compute groups.
    #[default]
    Upload,
    /// Compute it on the CPU from its RAM slot, reading the file first on a RAM miss. The
    /// compute slots hold a pinned hot set (ADR 0035); decode never uploads. Needs `ram_slots`.
    Cpu,
}

/// Reads from the model file and writes into the compute tier since load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoCounters {
    /// `read_batch_with` calls, one per shard and request.
    pub batches: u64,
    /// Experts read from the file (three slabs each).
    pub experts_read: u64,
    /// Bytes of those experts.
    pub bytes_read: u64,
    /// Wall time of the device reads (`read_batch_with`, without the copies into the slots).
    pub read_time: Duration,
    /// Experts written into the compute tier, from any source.
    pub uploads: u64,
    /// Prefetch (ADR 0038). Experts read ahead of their layer step into the RAM tier. These reads
    /// are also in `batches`, `experts_read` and `bytes_read`, but not in `read_time`.
    pub prefetch_issued: u64,
    /// Prefetched experts later routed while still resident.
    pub prefetch_used: u64,
    /// Prefetched experts evicted before they were routed (best effort: never-used ones still
    /// resident count in neither).
    pub prefetch_wasted: u64,
    pub prefetch_bytes: u64,
    pub prefetch_wasted_bytes: u64,
    /// Worker time of the prefetch reads, and the time the step waited for them to finish.
    pub prefetch_read_time: Duration,
    pub prefetch_wait_time: Duration,
    /// Overlapped decode (ADR 0054). Time the main thread blocked waiting for an expert of the
    /// layer being computed (`read_wait`); layer steps that had experts still in flight when the
    /// CPU group started, and those experts; the summed spans of those steps (from planning to
    /// the last landing) and the part of them the main thread waited. `1 - wait / span` is the
    /// hidden share. Their reads are counted in `experts_read` and `bytes_read` as usual.
    /// Despite its name, `urgent_wait_time` is every wait for a pending expert, a late prefetch
    /// as well as a miss (it equals `overlap_wait`).
    pub urgent_wait_time: Duration,
    pub pending_steps: u64,
    pub pending_experts: u64,
    pub overlap_span: Duration,
    pub overlap_wait: Duration,
    /// The pending experts by where their read came from: misses of the layer step itself
    /// (urgent), and prefetches issued two layer steps ahead (depth 2); the rest are depth-1
    /// prefetches. `pending_lead` sums, over the pending prefetches, the time from their submit
    /// to the layer's plan.
    pub pending_urgent: u64,
    pub pending_ahead: u64,
    pub pending_lead: Duration,
    /// The pending steps that had at least one urgent miss, with their span and wait (the other
    /// pending steps only waited for late prefetches).
    pub urgent_steps: u64,
    pub urgent_steps_span: Duration,
    pub urgent_steps_wait: Duration,
    /// CPU expert groups run because their reads had landed (after the resident group). One
    /// group per wake-up of a wait: experts that land together form one.
    pub landed_groups: u64,
}

/// Cache counters of one MoE layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerExpertStats {
    /// Compute tier (VRAM on a GPU backend).
    pub compute: LayerStats,
    /// RAM tier, when there is one.
    pub ram: Option<LayerStats>,
    /// Layer steps with every routed expert resident in the compute tier.
    pub full_steps: u64,
    /// Layer steps that had to load at least one expert or compute one on the CPU.
    pub partial_steps: u64,
    /// Routed experts not resident in the compute tier and computed on the CPU, once per layer
    /// step ([`MissPolicy::Cpu`]).
    pub cpu: u64,
    /// Layer steps with compute-tier hits and CPU experts at once.
    pub mixed_steps: u64,
    /// Prediction scores (ADR 0038): per token, the predicted top-k experts also in the real
    /// top-k, and the predicted slots scored (tokens times k). Zero without prefetch.
    pub predicted_hits: u64,
    pub predicted_total: u64,
    /// The same for the prediction made two layers ahead (`prefetch_depth` 2).
    pub predicted2_hits: u64,
    pub predicted2_total: u64,
}

impl LayerExpertStats {
    /// Share of predicted experts that the router chose, `None` when nothing was scored.
    pub fn prediction_accuracy(&self) -> Option<f64> {
        (self.predicted_total > 0).then(|| self.predicted_hits as f64 / self.predicted_total as f64)
    }

    /// [`Self::prediction_accuracy`] of the two-layer-ahead prediction.
    pub fn prediction2_accuracy(&self) -> Option<f64> {
        (self.predicted2_total > 0)
            .then(|| self.predicted2_hits as f64 / self.predicted2_total as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStats {
    pub layers: Vec<LayerExpertStats>,
    pub io: IoCounters,
}

/// The top-k ids the router chose in one layer step (`[n_used, n_tokens]`, token-major), kept
/// when `LoadOptions::record_routing` is set.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingStep {
    pub layer: usize,
    pub ids: Vec<i32>,
    /// The router weights of `ids`, same layout.
    pub weights: Vec<f32>,
    /// Tokens of the step (`ids` is `[n_used, n_tokens]`).
    pub n_tokens: usize,
    /// The experts this layer predicted for the next layer, and for the layer after (ADR 0038),
    /// same layout as `ids`; `None` where the step made no prediction (prefetch off, the last
    /// layers, or the resident whole-step path).
    pub predicted: Option<Vec<i32>>,
    pub predicted2: Option<Vec<i32>>,
    /// A marker, not a layer step: the hot set was rebuilt here (`refresh_hot_set`); `ids` and
    /// the rest are empty. The routing replay (ADR 0050) rebuilds its VRAM tier at markers.
    pub refresh: bool,
}

impl RoutingStep {
    /// The hot-set refresh marker.
    pub(crate) fn refresh_marker() -> Self {
        Self {
            layer: 0,
            ids: Vec::new(),
            weights: Vec::new(),
            n_tokens: 0,
            predicted: None,
            predicted2: None,
            refresh: true,
        }
    }
}

/// Host copies of experts between the file and the compute tier: per layer a gate, up and down
/// tensor shaped like the GGUF ones with `slots` experts, in a CPU buffer marked as weights so
/// the scheduler computes on them there.
struct RamTier {
    cache: SlotCache,
    tensors: Vec<[OwnedTensor; 3]>,
    slots: usize,
    /// Size of the slot buffer.
    bytes: usize,
}

impl RamTier {
    fn exps(&self, il: usize) -> [Tensor<'_>; 3] {
        let [g, u, d] = &self.tensors[il];
        [g.get(), u.get(), d.get()]
    }
}

/// The experts of a layer step under [`MissPolicy::Cpu`].
pub(crate) struct CpuPlan {
    /// Routed experts resident in the compute tier, and their slots.
    pub compute: Vec<u32>,
    pub compute_slots: Vec<u32>,
    /// The other routed experts in chunks of at most the RAM slot count, RAM hits first.
    pub cpu: Vec<Vec<u32>>,
}

/// Bytes of the slot tiers, see [`Streamer::slot_use`].
pub(crate) struct SlotUse {
    pub compute_capacity: u64,
    pub compute_used: u64,
    /// `(capacity, used)` of the RAM tier.
    pub ram: Option<(u64, u64)>,
}

/// A prefetch read in flight into the RAM slots of layer `il`.
struct InFlight {
    il: usize,
    ticket: Ticket,
    experts: Vec<u32>,
}

/// One expert's read through the I/O service, from submit until its layer step is done with it.
struct InFlightUnit {
    il: usize,
    expert: u32,
    unit: UnitId,
    /// 0: a miss of the layer being computed (urgent); 1 or 2: read ahead for the prediction
    /// made that many layer steps before.
    depth: u8,
    submitted: Instant,
    /// The layer step routed to it.
    routed: bool,
    /// The bytes are in the slot and counted; the entry stays (its expert stays pinned and
    /// counts as in flight for `prefetch_plan`, as in the reader path) until the layer's step
    /// has finished with it.
    landed: bool,
}

/// What reads the prefetch (and, with overlap, the misses) of a decode step (ADR 0054). Declared
/// first in the streamer so it drops, waiting for its reads, before the RAM tier is freed.
enum Prefetcher {
    /// One background thread, one job per prefetch call (ADR 0038).
    Reader {
        reader: Reader,
        /// Its own direct files, one per shard, so its reads never wait for the bounce buffer of
        /// a synchronous read on the main thread (prefetch depth 2 overlaps them).
        dio: Vec<Arc<DirectFile>>,
        /// Reads in flight, oldest first (the reader runs them in this order).
        inflight: VecDeque<InFlight>,
    },
    /// Persistent workers, one unit per expert, earliest deadline first.
    Service {
        io: Box<dyn Io>,
        inflight: Vec<InFlightUnit>,
    },
}

/// The experts of a layer step under overlap (ADR 0054), split by whether their bytes are in
/// the RAM slots yet. `slots` is per expert in the order given to `stage_overlap`; `ready` and
/// `pending` partition those experts.
pub(crate) struct StagedLayer {
    pub slots: Vec<u32>,
    pub ready: Vec<u32>,
    pub pending: Vec<u32>,
}

/// Timing of the current layer step's reads for the hidden share.
struct OverlapSpan {
    start: Instant,
    last: Instant,
    wait: Duration,
    pending: u64,
    urgent: u64,
    ahead: u64,
    lead: Duration,
}

/// Cap on the reader's bounce buffer per shard: its reads are a few experts, not a layer step.
const READER_BOUNCE: usize = 64 << 20;
/// The reader's device read size and queue depth: a prefetch is a few experts of a few MB, which
/// as whole-slab reads keep only about three requests in flight (box 58958565: 3.2 GB/s for the
/// reader against 5.2 GB/s for the synchronous reads of the same layer steps).
const READER_CHUNK: usize = 1 << 20;
const READER_QUEUE_DEPTH: usize = 16;

pub(crate) struct Streamer {
    /// The background reads write into the RAM tier's memory: it comes first so that it is
    /// dropped, which waits for its jobs, before the RAM tier is freed.
    prefetcher: Option<Prefetcher>,
    /// Steps begun since load, for the service's deadlines.
    step: u64,
    span: Option<OverlapSpan>,
    /// Per layer and expert: read by a prefetch and not routed since.
    prefetched: Vec<Vec<bool>>,
    /// Prediction scores per layer: hits and slots scored.
    predicted: Vec<(u64, u64)>,
    /// The same for the prediction two layers ahead.
    predicted2: Vec<(u64, u64)>,
    /// One direct file per shard.
    dio: Vec<Arc<DirectFile>>,
    /// Bytes of one expert's gate, up and down slab, per layer: quantization mixes such as
    /// Q4_K_M give some layers wider formats.
    sizes: Vec<[usize; 3]>,
    n_expert: u32,
    /// Compute slots per layer.
    slots: usize,
    compute: SlotCache,
    ram: Option<RamTier>,
    metrics: MemoryMetrics,
    device: String,
    io: IoCounters,
    full: Vec<u64>,
    partial: Vec<u64>,
    cpu: Vec<u64>,
    mixed: Vec<u64>,
    /// Routed pairs per layer and expert, for the hot set.
    counts: Vec<Vec<u64>>,
}

pub(crate) struct StreamerConfig {
    pub slots: usize,
    pub ram_slots: Option<usize>,
    pub io: IoOptions,
    /// Routing counts to start from, `[layer][expert]`.
    pub profile: Option<Vec<Vec<u64>>>,
    /// Prefetch into the RAM tier when there is one (ADR 0038).
    pub prefetch: bool,
    /// Decode computes experts as they land (ADR 0054): the service reads the misses too.
    pub overlap: bool,
    /// Test only: builds the reads in place of the I/O service (`LoadOptions::io_factory`).
    pub io_factory: Option<IoFactory>,
}

fn invalid(msg: impl Into<String>) -> ModelError {
    ModelError::Invalid(msg.into())
}

impl Streamer {
    /// `slot_tensors[il]` are the gate, up and down slot tensors of layer `il`, `cfg.slots`
    /// deep. The RAM tier is allocated on `cpu`.
    pub(crate) fn new(
        file: &ModelFile,
        slot_tensors: &[[Tensor<'_>; 3]],
        cpu: &Backend,
        cfg: StreamerConfig,
        metrics: MemoryMetrics,
        device: String,
    ) -> Result<Streamer> {
        let h = &file.hparams;
        let layers = h.n_layer as usize;
        let mut sizes = Vec::with_capacity(layers);
        for (il, t) in slot_tensors.iter().enumerate() {
            let slabs = file.expert_slabs(il as u32, 0)?;
            for (tensor, part) in t.iter().zip(&slabs.parts) {
                if tensor.nb()[2] as u64 != part.len {
                    return Err(invalid(format!(
                        "layer {il}: slot size does not match the expert slab"
                    )));
                }
            }
            sizes.push(slabs.parts.map(|p| p.len as usize));
        }
        let n_expert = h.n_expert;
        let ram = cfg
            .ram_slots
            .map(|n| -> Result<RamTier> {
                let ctx = Rc::new(Context::new(3 * layers + 1, 0)?);
                let tensors = slot_tensors
                    .iter()
                    .enumerate()
                    .map(|(il, t)| {
                        t.map(|src| {
                            let ne = src.ne();
                            let t = ctx.new_persistent(src.ty(), &[ne[0], ne[1], n as i64]);
                            t.get().set_name(&format!("ram_{}_{il}", src.name()));
                            t
                        })
                    })
                    .collect();
                // No zero clear: a CPU group only reads RAM slots staged for it, and touching
                // every page at load would commit the whole tier up front (ADR 0035).
                let bytes = ctx.alloc_weights(cpu)?;
                Ok(RamTier {
                    cache: SlotCache::new(Tier::Ram, h.n_layer, n_expert, n as u32),
                    tensors,
                    slots: n,
                    bytes,
                })
            })
            .transpose()?;
        let counts = match cfg.profile {
            Some(p) => {
                if p.len() != layers || p.iter().any(|c| c.len() != n_expert as usize) {
                    return Err(invalid(format!(
                        "routing profile must be {layers} layers of {n_expert} counts"
                    )));
                }
                p
            }
            None => vec![vec![0; n_expert as usize]; layers],
        };
        let prefetcher = if ram.is_some() && cfg.io.service && (cfg.prefetch || cfg.overlap) {
            let io: Box<dyn Io> = match &cfg.io_factory {
                Some(make) => make(&file.gguf.paths)?,
                None => Box::new(IoService::new(
                    &file.gguf.paths,
                    ServiceOptions {
                        workers: cfg.io.workers,
                        chunk_bytes: cfg.io.chunk_bytes,
                        buffered: cfg.io.buffered,
                    },
                )?),
            };
            Some(Prefetcher::Service {
                io,
                inflight: Vec::new(),
            })
        } else if cfg.prefetch && ram.is_some() {
            let cap = match cfg.io.bounce_bytes {
                0 => READER_BOUNCE,
                n => n.min(READER_BOUNCE),
            };
            let io = IoOptions {
                bounce_bytes: cap,
                chunk_bytes: match cfg.io.chunk_bytes {
                    0 => READER_CHUNK,
                    n => n,
                },
                queue_depth: cfg.io.queue_depth.max(READER_QUEUE_DEPTH),
                ..cfg.io
            };
            Some(Prefetcher::Reader {
                reader: Reader::new(),
                dio: file
                    .gguf
                    .paths
                    .iter()
                    .map(|p| DirectFile::open_with(p, io).map(Arc::new))
                    .collect::<std::io::Result<Vec<_>>>()?,
                inflight: VecDeque::new(),
            })
        } else {
            None
        };
        Ok(Streamer {
            prefetcher,
            step: 0,
            span: None,
            prefetched: vec![vec![false; n_expert as usize]; layers],
            predicted: vec![(0, 0); layers],
            predicted2: vec![(0, 0); layers],
            dio: file
                .gguf
                .paths
                .iter()
                .map(|p| DirectFile::open_with(p, cfg.io).map(Arc::new))
                .collect::<std::io::Result<Vec<_>>>()?,
            sizes,
            n_expert,
            slots: cfg.slots,
            compute: SlotCache::new(Tier::Vram, h.n_layer, n_expert, cfg.slots as u32),
            ram,
            metrics,
            device,
            io: IoCounters::default(),
            full: vec![0; layers],
            partial: vec![0; layers],
            cpu: vec![0; layers],
            mixed: vec![0; layers],
            counts,
        })
    }

    /// Bytes held by the shards' bounce buffers for direct reads, the reader's or the service's
    /// included.
    pub(crate) fn bounce_bytes(&self) -> u64 {
        let own: u64 = self.dio.iter().map(|d| d.bounce_capacity() as u64).sum();
        own + match &self.prefetcher {
            Some(Prefetcher::Reader { dio, .. }) => {
                dio.iter().map(|d| d.bounce_capacity() as u64).sum()
            }
            Some(Prefetcher::Service { io, .. }) => io.bounce_bytes(),
            None => 0,
        }
    }

    /// Slots per layer of the compute tier and of the RAM tier.
    pub(crate) fn tier_slots(&self) -> (usize, Option<usize>) {
        (self.slots, self.ram.as_ref().map(|r| r.slots))
    }

    pub(crate) fn io_backend(&self) -> String {
        let d = &self.dio[0];
        let mode = if d.is_direct() { "direct" } else { "buffered" };
        format!("{mode} {}", d.backend())
    }

    pub(crate) fn stats(&self) -> ExpertStats {
        ExpertStats {
            layers: (0..self.full.len())
                .map(|il| LayerExpertStats {
                    compute: self.compute.layer(il).stats(),
                    ram: self.ram.as_ref().map(|r| r.cache.layer(il).stats()),
                    full_steps: self.full[il],
                    partial_steps: self.partial[il],
                    cpu: self.cpu[il],
                    mixed_steps: self.mixed[il],
                    predicted_hits: self.predicted[il].0,
                    predicted_total: self.predicted[il].1,
                    predicted2_hits: self.predicted2[il].0,
                    predicted2_total: self.predicted2[il].1,
                })
                .collect(),
            io: self.io,
        }
    }

    /// Capacity and bytes in use of the compute slot tensors and of the RAM tier. Used bytes are
    /// the resident slots (one expert's bytes of the layer each); reads only the slot tables.
    pub(crate) fn slot_use(&self) -> SlotUse {
        let per_expert: Vec<u64> = self
            .sizes
            .iter()
            .map(|s| s.iter().sum::<usize>() as u64)
            .collect();
        let used = |cache: &SlotCache| -> u64 {
            per_expert
                .iter()
                .enumerate()
                .map(|(il, b)| cache.layer(il).resident() as u64 * b)
                .sum()
        };
        SlotUse {
            compute_capacity: per_expert.iter().sum::<u64>() * self.slots as u64,
            compute_used: used(&self.compute),
            ram: self.ram.as_ref().map(|r| (r.bytes as u64, used(&r.cache))),
        }
    }

    /// Routed pairs per layer and expert since load (plus the profile, if one was given).
    pub(crate) fn counts(&self) -> &[Vec<u64>] {
        &self.counts
    }

    /// The RAM slot tensors of layer `il` (gate, up, down).
    pub(crate) fn ram_exps(&self, il: usize) -> Option<[Tensor<'_>; 3]> {
        self.ram.as_ref().map(|r| r.exps(il))
    }

    /// The distinct experts of a layer step in order of first appearance; counts the pairs.
    fn distinct(&mut self, il: usize, chosen: &[i32]) -> Result<Vec<u32>> {
        let mut seen = vec![false; self.n_expert as usize];
        let mut distinct = Vec::new();
        for &e in chosen {
            let Some(s) = usize::try_from(e).ok().and_then(|e| seen.get_mut(e)) else {
                return Err(invalid(format!("router chose expert {e}, out of range")));
            };
            self.counts[il][e as usize] += 1;
            if !std::mem::replace(s, true) {
                distinct.push(e as u32);
            }
        }
        Ok(distinct)
    }

    /// Splits a layer step into chunks of experts that fit the compute slots together: the
    /// distinct experts in order of first appearance, resident ones first. One chunk is the
    /// usual case; more mean one pair group per chunk (ADR 0035).
    pub(crate) fn plan(&mut self, il: usize, chosen: &[i32]) -> Result<Vec<Vec<u32>>> {
        let distinct = self.distinct(il, chosen)?;
        let layer = self.compute.layer(il);
        let (hits, misses): (Vec<u32>, Vec<u32>) = distinct
            .into_iter()
            .partition(|&e| layer.slot_of(e).is_some());
        if misses.is_empty() {
            self.full[il] += 1;
        } else {
            self.partial[il] += 1;
        }
        self.metrics.layer_step(misses.is_empty());
        let active = layer.active().max(1) as usize;
        let order: Vec<u32> = hits.into_iter().chain(misses).collect();
        Ok(order.chunks(active).map(<[u32]>::to_vec).collect())
    }

    /// [`MissPolicy::Cpu`]: the routed experts resident in the compute tier are looked up (LRU
    /// and hit counts); the others get no slot and are counted once as `cpu`. They are chunked by
    /// the RAM slot count, RAM hits first so one chunk's reads cannot evict a hit of a later one.
    pub(crate) fn plan_cpu(&mut self, il: usize, chosen: &[i32]) -> Result<CpuPlan> {
        let n = self
            .ram
            .as_ref()
            .ok_or_else(|| invalid("the cpu miss policy needs ram_slots"))?
            .slots
            .max(1);
        let distinct = self.distinct(il, chosen)?;
        let (compute, rest): (Vec<u32>, Vec<u32>) = distinct
            .into_iter()
            .partition(|&e| self.compute.layer(il).slot_of(e).is_some());
        let mut compute_slots = Vec::with_capacity(compute.len());
        for &e in &compute {
            let res = self.compute.layer_mut(il).lookup(e);
            self.metrics.lookup(il as u32, Tier::Vram, &res);
            match res {
                Lookup::Hit(s) => compute_slots.push(s),
                _ => return Err(invalid("a resident expert lost its slot")),
            }
        }
        for _ in &rest {
            self.metrics.cpu_lookup(il as u32, Tier::Vram);
        }
        self.cpu[il] += rest.len() as u64;
        if rest.is_empty() {
            self.full[il] += 1;
        } else {
            self.partial[il] += 1;
            if !compute.is_empty() {
                self.mixed[il] += 1;
            }
        }
        self.metrics.layer_step(rest.is_empty());
        let layer = self.ram.as_ref().expect("RAM tier").cache.layer(il);
        let order = hits_first(&rest, |&e| e, |e| layer.slot_of(e).is_some());
        Ok(CpuPlan {
            compute,
            compute_slots,
            cpu: order.chunks(n).map(<[u32]>::to_vec).collect(),
        })
    }

    /// Scores the prediction made `depth` (1 or 2) layer steps ago for layer `il` against the
    /// real routing: per token, the predicted ids that the router also chose (both
    /// `[n_used, n_tokens]`).
    pub(crate) fn score_prediction(
        &mut self,
        il: usize,
        depth: u32,
        predicted: &[i32],
        actual: &[i32],
        n_used: usize,
    ) {
        let (mut hits, mut total) = (0u64, 0u64);
        for (p, a) in predicted.chunks(n_used).zip(actual.chunks(n_used)) {
            hits += p.iter().filter(|e| a.contains(e)).count() as u64;
            total += p.len() as u64;
        }
        let score = if depth == 1 {
            &mut self.predicted[il]
        } else {
            &mut self.predicted2[il]
        };
        score.0 += hits;
        score.1 += total;
        self.metrics
            .prediction_accuracy(il as u32, depth, hits, total);
    }

    /// Starts a decode or prefill step: new deadlines for the service, and waits for the reads
    /// of an earlier step (a step that failed, or experts mispredicted under overlap, which are
    /// long landed), so nothing in flight spans two steps.
    pub(crate) fn begin_step(&mut self) {
        self.settle();
        self.step += 1;
    }

    /// Reads ahead the experts of layer `il` that `predicted` (`[n_used, n_tokens]`) names and
    /// that no tier holds yet, into free or least recently used RAM slots, on the background
    /// reads (ADR 0038). Experts the prediction needs that are resident already are pinned, so
    /// the new ones never evict them; at most `RAM slots - those` are read. The slots stay
    /// pinned until [`Streamer::finish_prefetch`]. Reads already in flight for the layer count as
    /// resident (their slots are assigned) and stay pinned, so a later call tops the layer up.
    /// The background reads have their own direct files, so the main thread's reads may run
    /// meanwhile. `depth`: the layer steps between the prediction and layer `il` (1 or 2), for
    /// the counters only.
    pub(crate) fn start_prefetch(
        &mut self,
        file: &ModelFile,
        il: usize,
        predicted: &[i32],
        depth: u8,
    ) -> Result<()> {
        let (Some(pf), Some(ram)) = (&mut self.prefetcher, &mut self.ram) else {
            return Ok(());
        };
        let inflight: Vec<u32> = match &*pf {
            Prefetcher::Reader { inflight, .. } => inflight
                .iter()
                .filter(|f| f.il == il)
                .flat_map(|f| f.experts.iter().copied())
                .collect(),
            Prefetcher::Service { inflight, .. } => inflight
                .iter()
                .filter(|u| u.il == il)
                .map(|u| u.expert)
                .collect(),
        };
        // The experts to pin (resident predicted ones and reads still in flight for the layer)
        // and to read; the selection is shared with the replay (ADR 0050).
        let (mut pins, missing) = prefetch_plan(
            predicted,
            self.n_expert,
            self.compute.layer(il),
            ram.cache.layer(il),
            &inflight,
            ram.slots,
        );
        if missing.is_empty() {
            return Ok(());
        }
        // Field borrows, so the slot tables stay writable while the tensors are read.
        let [dg, du, dd] = &ram.tensors[il];
        let dst = [dg.get(), du.get(), dd.get()];
        let sizes = self.sizes[il];
        let total: usize = sizes.iter().sum();
        pins.extend(&missing);
        let layer = ram.cache.layer_mut(il);
        layer.set_hot_set(&pins);
        // Per missing expert its slabs: shard, range and destination.
        let mut staged_experts: Vec<Vec<(usize, Range, HostDst)>> =
            Vec::with_capacity(missing.len());
        let mut staged = Ok(());
        for &e in &missing {
            let slot = match layer.lookup(e) {
                Lookup::Miss { slot, evicted } => {
                    if let Some(old) = evicted {
                        let flags = &mut self.prefetched[il];
                        note_wasted(flags, il, old, total, &mut self.io, &self.metrics);
                    }
                    slot as usize
                }
                _ => {
                    staged = Err(invalid("no RAM slot for a prefetch (all pinned)"));
                    break;
                }
            };
            self.prefetched[il][e as usize] = true;
            match file.expert_slabs(il as u32, e) {
                Ok(slabs) => {
                    let mut parts = Vec::with_capacity(3);
                    for (p, part) in slabs.parts.iter().enumerate() {
                        // The slot is pinned and nothing else touches it until
                        // `finish_prefetch` has waited; the RAM tier outlives the reads.
                        match tensor_dst(dst[p], slot * sizes[p], sizes[p]) {
                            Ok(d) => parts.push((
                                part.shard,
                                Range {
                                    offset: part.offset,
                                    len: part.len as usize,
                                },
                                d,
                            )),
                            Err(err) => {
                                staged = Err(err.into());
                                break;
                            }
                        }
                    }
                    if staged.is_err() {
                        break;
                    }
                    staged_experts.push(parts);
                }
                Err(err) => {
                    staged = Err(err.into());
                    break;
                }
            }
        }
        if let Err(e) = staged {
            // Reads of this layer in flight write into slots about to be cleared: wait first.
            let _ = self.finish_prefetch(il);
            self.clear_layer(il);
            return Err(e);
        }
        self.io.prefetch_issued += missing.len() as u64;
        self.metrics
            .prefetch_issued(il as u32, missing.len() as u64);
        match pf {
            Prefetcher::Reader {
                reader,
                dio,
                inflight,
            } => {
                let n_shards = dio.len();
                let mut ranges: Vec<Vec<Range>> = (0..n_shards).map(|_| Vec::new()).collect();
                let mut dsts: Vec<Vec<HostDst>> = (0..n_shards).map(|_| Vec::new()).collect();
                for parts in staged_experts {
                    for (shard, range, d) in parts {
                        ranges[shard].push(range);
                        dsts[shard].push(d);
                    }
                }
                let jobs: Vec<ReadJob> = dio
                    .iter()
                    .zip(ranges)
                    .zip(dsts)
                    .filter(|((_, r), _)| !r.is_empty())
                    .map(|((f, ranges), dsts)| ReadJob {
                        file: f.clone(),
                        ranges,
                        sink: copy_sink(dsts),
                    })
                    .collect();
                inflight.push_back(InFlight {
                    il,
                    ticket: reader.submit(jobs),
                    experts: missing,
                });
            }
            Prefetcher::Service { io, inflight } => {
                let order = self.step * self.sizes.len() as u64 + il as u64;
                self.io.batches += 1;
                for (&expert, parts) in missing.iter().zip(staged_experts) {
                    let unit = submit_expert(io.as_ref(), order, false, parts);
                    self.metrics.io_inflight(false, io.inflight());
                    inflight.push(InFlightUnit {
                        il,
                        expert,
                        unit,
                        depth,
                        submitted: Instant::now(),
                        routed: false,
                        landed: false,
                    });
                }
            }
        }
        Ok(())
    }

    /// A background read is running or queued.
    pub(crate) fn prefetch_in_flight(&self) -> bool {
        match &self.prefetcher {
            Some(Prefetcher::Reader { inflight, .. }) => !inflight.is_empty(),
            Some(Prefetcher::Service { io, .. }) => io.inflight() > 0,
            None => false,
        }
    }

    /// Waits for every prefetch read of layer `il`, if any, and unpins its slots. The reader is
    /// FIFO, so reads of later layers keep running. On an error the layer's tables are cleared
    /// (slots were assigned before the bytes arrived) and the first error returned.
    pub(crate) fn finish_prefetch(&mut self, il: usize) -> Result<()> {
        let mut failure = None;
        match &mut self.prefetcher {
            None => return Ok(()),
            Some(Prefetcher::Reader {
                reader, inflight, ..
            }) => {
                let (mine, rest): (VecDeque<_>, VecDeque<_>) = std::mem::take(inflight)
                    .into_iter()
                    .partition(|f| f.il == il);
                *inflight = rest;
                if mine.is_empty() {
                    return Ok(());
                }
                for f in mine {
                    let t0 = Instant::now();
                    let done = reader.wait(f.ticket);
                    self.io.prefetch_wait_time += t0.elapsed();
                    match done {
                        Ok(d) => {
                            let experts = f.experts.len();
                            self.io.batches += 1;
                            self.io.experts_read += experts as u64;
                            self.io.bytes_read += d.stats.requested;
                            self.io.prefetch_bytes += d.stats.requested;
                            self.io.prefetch_read_time += d.took;
                            self.metrics
                                .tier_read(&self.device, "nvme", d.stats.requested);
                            let took = d.took / experts.max(1) as u32;
                            for _ in 0..experts {
                                self.metrics.expert_load(Tier::Ram, "nvme", took);
                            }
                        }
                        Err(e) => failure = failure.or(Some(e)),
                    }
                }
            }
            Some(Prefetcher::Service { io, inflight }) => {
                let ids: Vec<UnitId> = inflight
                    .iter()
                    .filter(|u| u.il == il && !u.landed)
                    .map(|u| u.unit)
                    .collect();
                let had = inflight.iter().any(|u| u.il == il);
                inflight.retain(|u| u.il != il);
                if !had {
                    return Ok(());
                }
                let t0 = Instant::now();
                let done = io.wait_all(&ids);
                if !ids.is_empty() {
                    self.io.prefetch_wait_time += t0.elapsed();
                }
                // Units that landed before were counted when they were reaped.
                for (_, r) in done {
                    match r {
                        Ok(d) => account_unit(&mut self.io, &self.metrics, &self.device, true, &d),
                        Err(e) => failure = failure.or(Some(e)),
                    }
                }
            }
        }
        if let Some(ram) = &mut self.ram {
            ram.cache.layer_mut(il).set_hot_set(&[]);
        }
        match failure {
            None => Ok(()),
            Some(e) => {
                self.clear_layer(il);
                Err(e.into())
            }
        }
    }

    /// [`Streamer::finish_prefetch`] for every layer, for code that must not race with the
    /// reads and has no use for the result (the start of a step, the hot set refresh).
    pub(crate) fn settle(&mut self) {
        let layers: Vec<usize> = match &self.prefetcher {
            Some(Prefetcher::Reader { inflight, .. }) => {
                let mut v: Vec<usize> = inflight.iter().map(|f| f.il).collect();
                v.dedup();
                v
            }
            Some(Prefetcher::Service { inflight, .. }) => {
                let mut v: Vec<usize> = inflight.iter().map(|u| u.il).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
            None => return,
        };
        for il in layers {
            let _ = self.finish_prefetch(il);
        }
        self.span = None;
    }

    /// Counts the units of the service that have landed without blocking. On a failed unit the
    /// layer it belongs to is drained and cleared and the error returned.
    fn reap(&mut self) -> Result<()> {
        let Some(Prefetcher::Service { io, inflight }) = &mut self.prefetcher else {
            return Ok(());
        };
        let ids: Vec<UnitId> = inflight
            .iter()
            .filter(|u| !u.landed)
            .map(|u| u.unit)
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        let done = io.poll(&ids);
        let failed = land(inflight, done, &mut self.io, &self.metrics, &self.device);
        match failed {
            None => Ok(()),
            Some((il, e)) => {
                self.drain_layer(il);
                Err(e.into())
            }
        }
    }

    /// Waits for every unit of layer `il` and forgets them, then clears the layer's tables: the
    /// recovery after a failed read, with no unit left that could write into a cleared slot.
    fn drain_layer(&mut self, il: usize) {
        if let Some(Prefetcher::Service { io, inflight }) = &mut self.prefetcher {
            let ids: Vec<UnitId> = inflight
                .iter()
                .filter(|u| u.il == il && !u.landed)
                .map(|u| u.unit)
                .collect();
            let _ = io.wait_all(&ids);
            inflight.retain(|u| u.il != il);
        }
        self.span = None;
        self.clear_layer(il);
    }

    /// Overlapped decode (ADR 0054): the RAM lookups of the CPU experts of layer `il`, with the
    /// counters and order of `fill_ram`, without waiting for any read. An expert whose prefetch
    /// is in flight has its slot already and is a hit; the misses get urgent units. Experts
    /// whose bytes are in their slots are `ready`, the others `pending`. Needs the service.
    /// On an error the layer's units are waited for and its tables cleared.
    pub(crate) fn stage_overlap(
        &mut self,
        file: &ModelFile,
        il: usize,
        experts: &[u32],
    ) -> Result<StagedLayer> {
        let t0 = Instant::now();
        self.reap()?;
        let r = self.stage_overlap_inner(file, il, experts, t0);
        if r.is_err() {
            self.drain_layer(il);
        }
        r
    }

    fn stage_overlap_inner(
        &mut self,
        file: &ModelFile,
        il: usize,
        experts: &[u32],
        t0: Instant,
    ) -> Result<StagedLayer> {
        let order = self.step * self.sizes.len() as u64 + il as u64;
        let Some(Prefetcher::Service { io, inflight }) = &mut self.prefetcher else {
            return Err(invalid("overlap needs the I/O service"));
        };
        let ram = self.ram.as_mut().expect("RAM tier");
        let sizes = self.sizes[il];
        // Landed reads of this layer are done: unpin everything except the reads in flight, as
        // `finish_prefetch` does before its lookups, so the victims are the same.
        inflight.retain(|u| !(u.il == il && u.landed));
        // Reads in flight for experts this step does not route (mispredicted) stay pinned, so
        // their slots cannot be reused while they write. If that leaves too few slots for the
        // step's own experts, wait for them first (small RAM tiers only: the tier holds many
        // more experts than a layer step routes plus predicts).
        let stranded: Vec<UnitId> = inflight
            .iter()
            .filter(|u| u.il == il && !u.landed && !experts.contains(&u.expert))
            .map(|u| u.unit)
            .collect();
        if experts.len() + stranded.len() > ram.slots {
            let t = Instant::now();
            let done = io.wait_all(&stranded);
            self.io.prefetch_wait_time += t.elapsed();
            if let Some((_, e)) = land(inflight, done, &mut self.io, &self.metrics, &self.device) {
                return Err(e.into());
            }
            inflight.retain(|u| !(u.il == il && u.landed));
        }
        let mut pins: Vec<u32> = inflight
            .iter()
            .filter(|u| u.il == il)
            .map(|u| u.expert)
            .collect();
        let layer = ram.cache.layer_mut(il);
        layer.set_hot_set(&pins);
        let mut idx: Vec<usize> = (0..experts.len()).collect();
        idx.sort_by_key(|&i| layer.slot_of(experts[i]).is_none());
        let mut slots = vec![0u32; experts.len()];
        let mut misses: Vec<usize> = Vec::new();
        let mut pending = vec![false; experts.len()];
        for &i in &idx {
            let e = experts[i];
            let res = layer.lookup(e);
            self.metrics.lookup(il as u32, Tier::Ram, &res);
            match res {
                Lookup::Hit(s) => {
                    if std::mem::take(&mut self.prefetched[il][e as usize]) {
                        self.io.prefetch_used += 1;
                        self.metrics.prefetch_used(il as u32);
                    }
                    slots[i] = s;
                    if let Some(u) = inflight.iter_mut().find(|u| u.il == il && u.expert == e) {
                        u.routed = true;
                        pending[i] = true;
                    }
                }
                Lookup::Miss { slot, evicted } => {
                    if let Some(old) = evicted {
                        let bytes = sizes.iter().sum();
                        note_wasted(
                            &mut self.prefetched[il],
                            il,
                            old,
                            bytes,
                            &mut self.io,
                            &self.metrics,
                        );
                    }
                    slots[i] = slot;
                    misses.push(i);
                    pending[i] = true;
                    // Pinned from the moment it has a slot: its unit writes there.
                    pins.push(e);
                    layer.set_hot_set(&pins);
                }
                Lookup::Bypass => return Err(invalid("no free RAM expert slot (all pinned)")),
            }
        }
        if !misses.is_empty() {
            let [dg, du, dd] = &ram.tensors[il];
            let dst = [dg.get(), du.get(), dd.get()];
            self.io.batches += 1;
            for &i in &misses {
                let slabs = file.expert_slabs(il as u32, experts[i])?;
                let mut parts = Vec::with_capacity(3);
                for (p, part) in slabs.parts.iter().enumerate() {
                    // The slot is pinned and nothing else touches it until the unit has landed
                    // (`wait_landed`, `drain_layer` or `finish_prefetch`); the RAM tier
                    // outlives the service.
                    parts.push((
                        part.shard,
                        Range {
                            offset: part.offset,
                            len: part.len as usize,
                        },
                        tensor_dst(dst[p], slots[i] as usize * sizes[p], sizes[p])?,
                    ));
                }
                let unit = submit_expert(io.as_ref(), order, true, parts);
                self.metrics.io_inflight(true, io.inflight());
                inflight.push(InFlightUnit {
                    il,
                    expert: experts[i],
                    unit,
                    depth: 0,
                    submitted: Instant::now(),
                    routed: true,
                    landed: false,
                });
            }
        }
        // An in-flight unit may have landed since the reap: its expert is ready.
        let mut ready = Vec::new();
        let mut waiting = Vec::new();
        let (mut urgent, mut ahead, mut lead) = (0, 0, Duration::ZERO);
        let now = Instant::now();
        for (i, &e) in experts.iter().enumerate() {
            let unit = inflight
                .iter()
                .find(|u| u.il == il && u.expert == e && !u.landed);
            match unit.filter(|_| pending[i]) {
                Some(u) => {
                    waiting.push(e);
                    match u.depth {
                        0 => urgent += 1,
                        d => {
                            ahead += u64::from(d >= 2);
                            lead += now.saturating_duration_since(u.submitted);
                        }
                    }
                }
                None => ready.push(e),
            }
        }
        self.span = (!waiting.is_empty()).then(|| OverlapSpan {
            start: t0,
            last: now,
            wait: Duration::ZERO,
            pending: waiting.len() as u64,
            urgent,
            ahead,
            lead,
        });
        Ok(StagedLayer {
            slots,
            ready,
            pending: waiting,
        })
    }

    /// Blocks until at least one of the `pending` experts of layer `il` has landed and returns
    /// every one of them that has (and so is ready to compute). The time blocked is
    /// `urgent_wait_time`. On an error the layer's units are waited for, its tables cleared.
    pub(crate) fn wait_landed(&mut self, il: usize, pending: &[u32]) -> Result<Vec<u32>> {
        let r = self.wait_landed_inner(il, pending);
        if r.is_err() {
            self.drain_layer(il);
        }
        r
    }

    fn wait_landed_inner(&mut self, il: usize, pending: &[u32]) -> Result<Vec<u32>> {
        let Some(Prefetcher::Service { io, inflight }) = &mut self.prefetcher else {
            return Err(invalid("overlap needs the I/O service"));
        };
        let is_open = |u: &InFlightUnit| u.il == il && !u.landed && pending.contains(&u.expert);
        let ids: Vec<UnitId> = inflight
            .iter()
            .filter(|u| is_open(u))
            .map(|u| u.unit)
            .collect();
        let t0 = Instant::now();
        let done = io.wait_any(&ids);
        let waited = t0.elapsed();
        let failed = land(inflight, done, &mut self.io, &self.metrics, &self.device);
        let landed: Vec<u32> = pending
            .iter()
            .copied()
            .filter(|&e| {
                !inflight
                    .iter()
                    .any(|u| u.il == il && u.expert == e && !u.landed)
            })
            .collect();
        self.io.urgent_wait_time += waited;
        self.io.landed_groups += u64::from(!landed.is_empty());
        if let Some(sp) = &mut self.span {
            sp.wait += waited;
            sp.last = Instant::now();
        }
        match failed {
            None => Ok(landed),
            Some((_, e)) => Err(e.into()),
        }
    }

    /// Ends the layer's overlapped step: counts its read span, forgets the experts that landed
    /// and unpins everything but reads still in flight (mispredicted prefetches stay pinned
    /// until they land and the next `begin_step` or `finish_prefetch` of the layer collects
    /// them).
    pub(crate) fn finish_overlap(&mut self, il: usize) {
        if let Some(sp) = self.span.take() {
            let span = sp.last.saturating_duration_since(sp.start);
            self.io.pending_steps += 1;
            self.io.pending_experts += sp.pending;
            self.io.overlap_span += span;
            self.io.overlap_wait += sp.wait;
            self.io.pending_urgent += sp.urgent;
            self.io.pending_ahead += sp.ahead;
            self.io.pending_lead += sp.lead;
            if sp.urgent > 0 {
                self.io.urgent_steps += 1;
                self.io.urgent_steps_span += span;
                self.io.urgent_steps_wait += sp.wait;
            }
            self.metrics.read_hidden(sp.wait, span);
        }
        let Some(Prefetcher::Service { inflight, .. }) = &mut self.prefetcher else {
            return;
        };
        inflight.retain(|u| !(u.il == il && u.landed));
        let pins: Vec<u32> = inflight
            .iter()
            .filter(|u| u.il == il)
            .map(|u| u.expert)
            .collect();
        if let Some(ram) = &mut self.ram {
            ram.cache.layer_mut(il).set_hot_set(&pins);
        }
    }

    /// Makes `experts` (at most the RAM slot count) resident in the RAM slots of layer `il` and
    /// returns their slots, in order. On an error both tiers' tables of the layer are cleared.
    pub(crate) fn stage_ram(
        &mut self,
        file: &ModelFile,
        il: usize,
        experts: &[u32],
    ) -> Result<Vec<u32>> {
        match self.fill_ram(file, il, experts) {
            Ok(f) => Ok(f.into_iter().map(|(slot, _)| slot as u32).collect()),
            Err(e) => {
                self.clear_layer(il);
                Err(e)
            }
        }
    }

    fn clear_layer(&mut self, il: usize) {
        self.prefetched[il].fill(false);
        self.compute.layer_mut(il).clear();
        if let Some(ram) = &mut self.ram {
            ram.cache.layer_mut(il).clear();
        }
    }

    /// Makes `experts` (at most the RAM slot count) resident in the RAM tier: RAM hits first,
    /// then one batch from the file for the misses, copied into their slots. Returns per expert,
    /// in input order, its RAM slot and whether it was read from the file.
    fn fill_ram(
        &mut self,
        file: &ModelFile,
        il: usize,
        experts: &[u32],
    ) -> Result<Vec<(usize, bool)>> {
        let t0 = Instant::now();
        let ram = self.ram.as_mut().expect("RAM tier");
        let sizes = self.sizes[il];
        let mut order: Vec<usize> = (0..experts.len()).collect();
        order.sort_by_key(|&i| ram.cache.layer(il).slot_of(experts[i]).is_none());
        let mut out = vec![(0usize, false); experts.len()];
        for &i in &order {
            let res = ram.cache.layer_mut(il).lookup(experts[i]);
            self.metrics.lookup(il as u32, Tier::Ram, &res);
            out[i] = match res {
                Lookup::Hit(s) => {
                    if std::mem::take(&mut self.prefetched[il][experts[i] as usize]) {
                        self.io.prefetch_used += 1;
                        self.metrics.prefetch_used(il as u32);
                    }
                    (s as usize, false)
                }
                Lookup::Miss { slot, evicted } => {
                    if let Some(e) = evicted {
                        let bytes = sizes.iter().sum();
                        note_wasted(
                            &mut self.prefetched[il],
                            il,
                            e,
                            bytes,
                            &mut self.io,
                            &self.metrics,
                        );
                    }
                    (slot as usize, true)
                }
                Lookup::Bypass => return Err(invalid("no free RAM expert slot (all pinned)")),
            };
        }
        let reads: Vec<usize> = (0..experts.len()).filter(|&i| out[i].1).collect();
        if !reads.is_empty() {
            let loads: Vec<(u32, u32)> = reads
                .iter()
                .map(|&i| (experts[i], out[i].0 as u32))
                .collect();
            let bytes = read_experts(
                &self.dio,
                file,
                il,
                &loads,
                &ram.exps(il),
                &sizes,
                &mut self.io,
            )?;
            self.metrics.tier_read(&self.device, "nvme", bytes);
            let took = t0.elapsed() / reads.len() as u32;
            for _ in &reads {
                self.metrics.expert_load(Tier::Ram, "nvme", took);
            }
        }
        Ok(out)
    }

    /// Makes `experts` resident in the compute slots of layer `il` and returns their slots, in
    /// order. On an error the layer's slot tables are cleared: slots were assigned before the
    /// bytes arrived and must not be served.
    pub(crate) fn make_resident(
        &mut self,
        file: &ModelFile,
        il: usize,
        dst: &[Tensor<'_>; 3],
        experts: &[u32],
    ) -> Result<Vec<u32>> {
        let r = self.make_resident_inner(file, il, dst, experts);
        if r.is_err() {
            self.clear_layer(il);
        }
        r
    }

    fn make_resident_inner(
        &mut self,
        file: &ModelFile,
        il: usize,
        dst: &[Tensor<'_>; 3],
        experts: &[u32],
    ) -> Result<Vec<u32>> {
        let mut slots = Vec::with_capacity(experts.len());
        let mut loads: Vec<(u32, u32)> = Vec::new();
        for &e in experts {
            let res = self.compute.layer_mut(il).lookup(e);
            self.metrics.lookup(il as u32, Tier::Vram, &res);
            match res {
                Lookup::Hit(s) => slots.push(s),
                Lookup::Miss { slot, .. } => {
                    slots.push(slot);
                    loads.push((e, slot));
                }
                Lookup::Bypass => return Err(invalid("no free expert slot (all pinned)")),
            }
        }
        if !loads.is_empty() {
            match self.ram.is_some() {
                false => self.load_from_file(file, il, dst, &loads)?,
                true => {
                    let ram = self.ram.as_ref().expect("RAM tier");
                    let n = ram.slots.max(1);
                    let layer = ram.cache.layer(il);
                    let ordered = hits_first(&loads, |l| l.0, |e| layer.slot_of(e).is_some());
                    for group in ordered.chunks(n) {
                        self.load_via_ram(file, il, dst, group)?;
                    }
                }
            }
        }
        Ok(slots)
    }

    /// The hot set (ADR 0035): per layer the top experts by routing count, as many as there are
    /// compute slots, pinned and made resident (from the RAM tier or the file). Run it between
    /// requests; decode then finds them resident and never uploads. `dst(il)` are the compute
    /// slot tensors of layer `il`.
    pub(crate) fn refresh_hot_set<'a>(
        &mut self,
        file: &ModelFile,
        dst: impl Fn(usize) -> [Tensor<'a>; 3],
    ) -> Result<()> {
        self.settle();
        for il in 0..self.counts.len() {
            let active = self.compute.layer(il).active();
            let hot = hot_from_counts(&self.counts[il], active);
            let change = self.compute.layer_mut(il).set_hot_set(&hot);
            if !change.missing.is_empty() {
                self.make_resident(file, il, &dst(il), &change.missing)?;
            }
        }
        Ok(())
    }

    /// No RAM tier: one batch from the file, each slab copied straight into its compute slot.
    fn load_from_file(
        &mut self,
        file: &ModelFile,
        il: usize,
        dst: &[Tensor<'_>; 3],
        loads: &[(u32, u32)],
    ) -> Result<()> {
        let t0 = Instant::now();
        let sizes = self.sizes[il];
        let bytes = read_experts(&self.dio, file, il, loads, dst, &sizes, &mut self.io)?;
        self.metrics.tier_read(&self.device, "nvme", bytes);
        self.io.uploads += loads.len() as u64;
        self.record_loads(loads.len(), "nvme", t0);
        Ok(())
    }

    /// RAM tier: RAM hits first, one batch from the file for the RAM misses, then every expert
    /// of the group is uploaded from its RAM slot. At most `ram.slots` experts, so no expert
    /// of the group can evict another.
    fn load_via_ram(
        &mut self,
        file: &ModelFile,
        il: usize,
        dst: &[Tensor<'_>; 3],
        group: &[(u32, u32)],
    ) -> Result<()> {
        let t0 = Instant::now();
        let experts: Vec<u32> = group.iter().map(|g| g.0).collect();
        let filled = self.fill_ram(file, il, &experts)?;
        let ram = self.ram.as_ref().expect("RAM tier");
        let sizes = self.sizes[il];
        let total: usize = sizes.iter().sum();
        let src = ram.exps(il);
        for (&(_, cslot), &(rslot, _)) in group.iter().zip(&filled) {
            let bytes = read_slot(&src, &sizes, rslot as u32)?;
            write_slot(dst, &sizes, cslot, &bytes)?;
            self.io.uploads += 1;
        }
        let n_ram = filled.iter().filter(|f| !f.1).count();
        if n_ram > 0 {
            self.metrics
                .tier_read(&self.device, "ram", (n_ram * total) as u64);
        }
        // The batch time is shared by the experts of the group (ADR 0033).
        let share = t0.elapsed() / group.len().max(1) as u32;
        for &(_, from_file) in &filled {
            let source = if from_file { "nvme" } else { "ram" };
            self.metrics.expert_load(Tier::Vram, source, share);
        }
        Ok(())
    }

    /// `expert.load.duration` into the compute tier: the time since `t0` shared by the experts
    /// of the batch.
    fn record_loads(&self, n: usize, source: &str, t0: Instant) {
        let took = t0.elapsed() / n.max(1) as u32;
        for _ in 0..n {
            self.metrics.expert_load(Tier::Vram, source, took);
        }
    }
}

/// Counts a landed unit: one expert read through the service.
fn account_unit(
    counters: &mut IoCounters,
    m: &MemoryMetrics,
    device: &str,
    prefetch: bool,
    d: &UnitDone,
) {
    counters.experts_read += 1;
    counters.bytes_read += d.requested;
    if prefetch {
        counters.prefetch_bytes += d.requested;
        counters.prefetch_read_time += d.took;
    } else {
        counters.read_time += d.took;
    }
    m.tier_read(device, "nvme", d.requested);
    m.expert_load(Tier::Ram, "nvme", d.took);
}

/// Marks the entries of the units in `done` landed and counts them. Returns the layer and error
/// of the first unit that failed.
fn land(
    inflight: &mut [InFlightUnit],
    done: Vec<(UnitId, std::io::Result<UnitDone>)>,
    counters: &mut IoCounters,
    m: &MemoryMetrics,
    device: &str,
) -> Option<(usize, std::io::Error)> {
    let mut failed = None;
    for (id, r) in done {
        let Some(u) = inflight.iter_mut().find(|u| u.unit == id) else {
            continue;
        };
        u.landed = true;
        match r {
            Ok(d) => account_unit(counters, m, device, u.depth > 0, &d),
            Err(e) => {
                failed.get_or_insert((u.il, e));
            }
        }
    }
    failed
}

/// One expert's slabs as a unit of the service.
fn submit_expert(
    io: &dyn Io,
    order: u64,
    urgent: bool,
    parts: Vec<(usize, Range, HostDst)>,
) -> UnitId {
    io.submit(
        Deadline { order, urgent },
        parts
            .into_iter()
            .map(|(shard, range, dst)| Part { shard, range, dst })
            .collect(),
    )
}

/// A prefetched expert left its slot before it was routed: count it as wasted.
fn note_wasted(
    flags: &mut [bool],
    il: usize,
    expert: u32,
    bytes: usize,
    io: &mut IoCounters,
    m: &MemoryMetrics,
) {
    if std::mem::take(&mut flags[expert as usize]) {
        io.prefetch_wasted += 1;
        io.prefetch_wasted_bytes += bytes as u64;
        m.prefetch_wasted(il as u32, bytes as u64);
    }
}

/// `items` with those whose expert `resident` holds first (stable). Across all loads of a chunk,
/// so the misses of one RAM group cannot evict a hit that waits in a later group.
fn hits_first<T: Copy>(
    items: &[T],
    expert: impl Fn(&T) -> u32,
    resident: impl Fn(u32) -> bool,
) -> Vec<T> {
    let (hits, misses): (Vec<_>, Vec<_>) = items.iter().partition(|t| resident(expert(t)));
    hits.into_iter().chain(misses).copied().collect()
}

/// Reads the three slabs of each `(expert, slot)` of `loads` from the file into slot `slot` of
/// `dst` (gate, up, down tensors): one batch per shard that holds any of them, each slab
/// copied once, from the bounce buffer into the slot. Returns the bytes read.
fn read_experts(
    dio: &[Arc<DirectFile>],
    file: &ModelFile,
    il: usize,
    loads: &[(u32, u32)],
    dst: &[Tensor<'_>; 3],
    sizes: &[usize; 3],
    io: &mut IoCounters,
) -> Result<u64> {
    // Per shard the ranges and, for each, which tensor and slot it fills.
    let mut ranges: Vec<Vec<Range>> = (0..dio.len()).map(|_| Vec::new()).collect();
    let mut targets: Vec<Vec<(usize, u32)>> = (0..dio.len()).map(|_| Vec::new()).collect();
    for &(e, slot) in loads {
        let slabs = file.expert_slabs(il as u32, e)?;
        for (p, part) in slabs.parts.iter().enumerate() {
            ranges[part.shard].push(Range {
                offset: part.offset,
                len: part.len as usize,
            });
            targets[part.shard].push((p, slot));
        }
    }
    let mut requested = 0;
    for ((shard, ranges), targets) in dio.iter().zip(&ranges).zip(&targets) {
        if ranges.is_empty() {
            continue;
        }
        let t0 = Instant::now();
        let mut copy = Duration::ZERO;
        let mut failure = None;
        let res = shard.read_batch_with(ranges, |i, bytes| {
            let (p, slot) = targets[i];
            let t = Instant::now();
            let r = dst[p].set_bytes(bytes, slot as usize * sizes[p]);
            copy += t.elapsed();
            r.map_err(|e| {
                failure = Some(e);
                std::io::Error::other("write into the expert slot failed")
            })
        });
        // The device time only: the copies into the slots are part of the step, not of the read.
        io.read_time += t0.elapsed().saturating_sub(copy);
        io.batches += 1;
        match res {
            Ok(stats) => requested += stats.requested,
            Err(e) => return Err(failure.map_or_else(|| e.into(), ModelError::from)),
        }
    }
    io.experts_read += loads.len() as u64;
    io.bytes_read += requested;
    Ok(requested)
}

/// Writes one expert (gate, up, down slabs back to back in `src`) into slot `slot`.
fn write_slot(dst: &[Tensor<'_>; 3], sizes: &[usize; 3], slot: u32, src: &[u8]) -> Result<()> {
    let mut at = 0;
    for (t, &size) in dst.iter().zip(sizes) {
        t.set_bytes(&src[at..at + size], slot as usize * size)?;
        at += size;
    }
    Ok(())
}

/// The bytes of slot `slot` of the three tensors, back to back.
fn read_slot(src: &[Tensor<'_>; 3], sizes: &[usize; 3], slot: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(sizes.iter().sum());
    for (t, &size) in src.iter().zip(sizes) {
        out.extend(t.get_bytes(slot as usize * size, size)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAM reads for `loads` in groups of the RAM slot count, RAM holding `resident` (least
    /// recent first).
    fn ram_reads(order: impl Fn(&[(u32, u32)], &SlotCache) -> Vec<(u32, u32)>) -> u32 {
        let mut ram = SlotCache::new(Tier::Ram, 1, 8, 2);
        for e in [7, 2] {
            ram.layer_mut(0).lookup(e);
        }
        let loads = [(0, 0), (1, 1), (2, 2)];
        let mut reads = 0;
        for group in order(&loads, &ram).chunks(2) {
            for &(e, _) in group {
                if matches!(ram.layer_mut(0).lookup(e), Lookup::Miss { .. }) {
                    reads += 1;
                }
            }
        }
        reads
    }

    /// RAM slots 2 holding [7, 2]; loads [0, 1, 2] where 2 is a RAM hit. Two misses first
    /// evict 2 and it is read again; hits first read only 0 and 1.
    #[test]
    fn ram_hits_first_across_groups() {
        let in_order = |l: &[(u32, u32)], _: &SlotCache| l.to_vec();
        let hits_first = |l: &[(u32, u32)], c: &SlotCache| {
            super::hits_first(l, |l| l.0, |e| c.layer(0).slot_of(e).is_some())
        };
        assert_eq!(ram_reads(in_order), 3);
        assert_eq!(ram_reads(hits_first), 2);
    }
}

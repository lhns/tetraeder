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
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_memory::reader::{HostDst, ReadJob, Reader, Ticket, copy_sink, tensor_dst};
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

/// Cap on the reader's bounce buffer per shard: its reads are a few experts, not a layer step.
const READER_BOUNCE: usize = 64 << 20;
/// The reader's device read size and queue depth: a prefetch is a few experts of a few MB, which
/// as whole-slab reads keep only about three requests in flight (box 58958565: 3.2 GB/s for the
/// reader against 5.2 GB/s for the synchronous reads of the same layer steps).
const READER_CHUNK: usize = 1 << 20;
const READER_QUEUE_DEPTH: usize = 16;

pub(crate) struct Streamer {
    /// The background reader writes into the RAM tier's memory: it comes first so that it is
    /// dropped, which waits for its jobs, before the RAM tier is freed.
    reader: Option<Reader>,
    /// Reads in flight, oldest first (the reader runs them in this order).
    inflight: VecDeque<InFlight>,
    /// The reader's own direct files, one per shard, so its reads never wait for the bounce
    /// buffer of a synchronous read on the main thread (prefetch depth 2 overlaps them).
    reader_dio: Vec<Arc<DirectFile>>,
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
        let prefetch_reads = cfg.prefetch && ram.is_some();
        Ok(Streamer {
            reader: prefetch_reads.then(Reader::new),
            inflight: VecDeque::new(),
            reader_dio: if prefetch_reads {
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
                file.gguf
                    .paths
                    .iter()
                    .map(|p| DirectFile::open_with(p, io).map(Arc::new))
                    .collect::<std::io::Result<Vec<_>>>()?
            } else {
                Vec::new()
            },
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

    /// Bytes held by the shards' bounce buffers for direct reads, the reader's included.
    pub(crate) fn bounce_bytes(&self) -> u64 {
        self.dio
            .iter()
            .chain(&self.reader_dio)
            .map(|d| d.bounce_capacity() as u64)
            .sum()
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

    /// Reads ahead the experts of layer `il` that `predicted` (`[n_used, n_tokens]`) names and
    /// that no tier holds yet, into free or least recently used RAM slots, on the background
    /// reader (ADR 0038). Experts the prediction needs that are resident already are pinned, so
    /// the new ones never evict them; at most `RAM slots - those` are read. The slots stay
    /// pinned until [`Streamer::finish_prefetch`]. Reads already in flight for the layer count as
    /// resident (their slots are assigned) and stay pinned, so a later call tops the layer up.
    /// The reader has its own direct files, so the main thread's reads may run meanwhile.
    pub(crate) fn start_prefetch(
        &mut self,
        file: &ModelFile,
        il: usize,
        predicted: &[i32],
    ) -> Result<()> {
        let (Some(reader), Some(ram)) = (&self.reader, &mut self.ram) else {
            return Ok(());
        };
        // Distinct predicted experts, the most predicted first.
        let mut votes = vec![0u32; self.n_expert as usize];
        for &e in predicted {
            if let Some(v) = usize::try_from(e).ok().and_then(|e| votes.get_mut(e)) {
                *v += 1;
            }
        }
        let mut wanted: Vec<u32> = (0..self.n_expert)
            .filter(|&e| votes[e as usize] > 0)
            .collect();
        wanted.sort_by_key(|&e| std::cmp::Reverse(votes[e as usize]));
        let in_compute = self.compute.layer(il);
        wanted.retain(|&e| in_compute.slot_of(e).is_none());
        let layer = ram.cache.layer(il);
        let (resident, mut missing): (Vec<u32>, Vec<u32>) = wanted
            .into_iter()
            .partition(|&e| layer.slot_of(e).is_some());
        // Everything pinned before this call: the predicted resident experts and every read of
        // this layer still in flight.
        let mut pins = resident;
        for f in self.inflight.iter().filter(|f| f.il == il) {
            for &e in &f.experts {
                if !pins.contains(&e) {
                    pins.push(e);
                }
            }
        }
        missing.truncate(ram.slots.saturating_sub(pins.len()));
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
        let n_shards = self.reader_dio.len();
        let mut ranges: Vec<Vec<Range>> = (0..n_shards).map(|_| Vec::new()).collect();
        let mut dsts: Vec<Vec<HostDst>> = (0..n_shards).map(|_| Vec::new()).collect();
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
                    for (p, part) in slabs.parts.iter().enumerate() {
                        // The slot is pinned and nothing else touches it until
                        // `finish_prefetch` has waited; the RAM tier outlives the reader.
                        match tensor_dst(dst[p], slot * sizes[p], sizes[p]) {
                            Ok(d) => dsts[part.shard].push(d),
                            Err(err) => {
                                staged = Err(err.into());
                                break;
                            }
                        }
                        ranges[part.shard].push(Range {
                            offset: part.offset,
                            len: part.len as usize,
                        });
                    }
                    if staged.is_err() {
                        break;
                    }
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
        let jobs: Vec<ReadJob> = self
            .reader_dio
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
        self.io.prefetch_issued += missing.len() as u64;
        self.metrics
            .prefetch_issued(il as u32, missing.len() as u64);
        self.inflight.push_back(InFlight {
            il,
            ticket: reader.submit(jobs),
            experts: missing,
        });
        Ok(())
    }

    /// A prefetch read is in flight.
    pub(crate) fn prefetch_in_flight(&self) -> bool {
        !self.inflight.is_empty()
    }

    /// Waits for every prefetch read of layer `il`, if any, and unpins its slots. The reader is
    /// FIFO, so reads of later layers keep running. On an error the layer's tables are cleared
    /// (slots were assigned before the bytes arrived) and the first error returned.
    pub(crate) fn finish_prefetch(&mut self, il: usize) -> Result<()> {
        let (mine, rest): (VecDeque<_>, VecDeque<_>) = std::mem::take(&mut self.inflight)
            .into_iter()
            .partition(|f| f.il == il);
        self.inflight = rest;
        if mine.is_empty() {
            return Ok(());
        }
        let reader = self.reader.as_ref().expect("a prefetch needs the reader");
        let mut failure = None;
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
    /// reader and has no use for the result (the start of a step, the hot set refresh).
    pub(crate) fn settle(&mut self) {
        while let Some(il) = self.inflight.front().map(|f| f.il) {
            let _ = self.finish_prefetch(il);
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

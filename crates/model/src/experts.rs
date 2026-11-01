//! Expert streaming for the builder (ADR 0033, 0035): the slot caches of both tiers, the loads
//! from the GGUF through direct I/O, the routing counts behind the hot set, and the counters.
//! No graph work: the builder asks for the experts of a layer step to be resident and gets slot
//! ids back, or, under [`MissPolicy::Cpu`], a split into compute-tier hits and CPU chunks.

use crate::hparams::ModelFile;
use crate::qwen3moe::{ModelError, Result};
use std::rc::Rc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedTensor, Tensor};
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_memory::slots::{LayerStats, Lookup, SlotCache, Tier, hot_from_counts};
use tetraeder_memory::{DirectFile, IoOptions, ReadReq};

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
    /// `read_batch` calls.
    pub batches: u64,
    /// Experts read from the file (three slabs each).
    pub experts_read: u64,
    /// Bytes of those experts.
    pub bytes_read: u64,
    /// Wall time spent inside `read_batch`.
    pub read_time: Duration,
    /// Experts written into the compute tier, from any source.
    pub uploads: u64,
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

pub(crate) struct Streamer {
    dio: DirectFile,
    /// Bytes of one expert's gate, up and down slab, per layer: quantization mixes such as
    /// Q4_K_M give some layers wider formats.
    sizes: Vec<[usize; 3]>,
    n_expert: u32,
    compute: SlotCache,
    ram: Option<RamTier>,
    metrics: MemoryMetrics,
    device: String,
    staging: Vec<u8>,
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
                if tensor.nb()[2] as u64 != part.1 {
                    return Err(invalid(format!(
                        "layer {il}: slot size does not match the expert slab"
                    )));
                }
            }
            sizes.push(slabs.parts.map(|p| p.1 as usize));
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
                ctx.alloc_weights(cpu)?;
                ctx.clear_buffer(0);
                Ok(RamTier {
                    cache: SlotCache::new(Tier::Ram, h.n_layer, n_expert, n as u32),
                    tensors,
                    slots: n,
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
        Ok(Streamer {
            dio: DirectFile::open_with(&file.path, cfg.io)?,
            sizes,
            n_expert,
            compute: SlotCache::new(Tier::Vram, h.n_layer, n_expert, cfg.slots as u32),
            ram,
            metrics,
            device,
            staging: Vec::new(),
            io: IoCounters::default(),
            full: vec![0; layers],
            partial: vec![0; layers],
            cpu: vec![0; layers],
            mixed: vec![0; layers],
            counts,
        })
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
                })
                .collect(),
            io: self.io,
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
        let order = hits_first(&rest, |e| layer.slot_of(e).is_some());
        Ok(CpuPlan {
            compute,
            compute_slots,
            cpu: order.chunks(n).map(<[u32]>::to_vec).collect(),
        })
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
        let total: usize = sizes.iter().sum();
        let mut order: Vec<usize> = (0..experts.len()).collect();
        order.sort_by_key(|&i| ram.cache.layer(il).slot_of(experts[i]).is_none());
        let mut out = vec![(0usize, false); experts.len()];
        for &i in &order {
            let res = ram.cache.layer_mut(il).lookup(experts[i]);
            self.metrics.lookup(il as u32, Tier::Ram, &res);
            out[i] = match res {
                Lookup::Hit(s) => (s as usize, false),
                Lookup::Miss { slot, .. } => (slot as usize, true),
                Lookup::Bypass => return Err(invalid("no free RAM expert slot (all pinned)")),
            };
        }
        let reads: Vec<usize> = (0..experts.len()).filter(|&i| out[i].1).collect();
        if !reads.is_empty() {
            self.staging.resize(reads.len() * total, 0);
            let mut staging = std::mem::take(&mut self.staging);
            let items: Vec<(u32, &mut [u8])> = reads
                .iter()
                .map(|&i| experts[i])
                .zip(staging.chunks_mut(total))
                .collect();
            let read = read_experts(&self.dio, file, il, items, &mut self.io);
            let dst = ram.exps(il);
            let res = read.and_then(|bytes| {
                self.metrics.tier_read(&self.device, "nvme", bytes);
                for (&i, src) in reads.iter().zip(staging.chunks(total)) {
                    write_slot(&dst, &sizes, out[i].0 as u32, src)?;
                }
                Ok(())
            });
            self.staging = staging;
            res?;
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
                    let ordered = ram_hits_first(&loads, |e| layer.slot_of(e).is_some());
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

    /// No RAM tier: one batch from the file into a staging buffer, then the uploads.
    fn load_from_file(
        &mut self,
        file: &ModelFile,
        il: usize,
        dst: &[Tensor<'_>; 3],
        loads: &[(u32, u32)],
    ) -> Result<()> {
        let t0 = Instant::now();
        let sizes = self.sizes[il];
        let total: usize = sizes.iter().sum();
        self.staging.resize(loads.len() * total, 0);
        let mut staging = std::mem::take(&mut self.staging);
        let items: Vec<(u32, &mut [u8])> = loads
            .iter()
            .map(|l| l.0)
            .zip(staging.chunks_mut(total))
            .collect();
        let read = read_experts(&self.dio, file, il, items, &mut self.io);
        let res = read.and_then(|bytes| {
            self.metrics.tier_read(&self.device, "nvme", bytes);
            for (&(_, slot), src) in loads.iter().zip(staging.chunks(total)) {
                write_slot(dst, &sizes, slot, src)?;
                self.io.uploads += 1;
            }
            Ok(())
        });
        self.staging = staging;
        res?;
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

/// `experts` with those for which `resident` holds first (stable).
fn hits_first(experts: &[u32], resident: impl Fn(u32) -> bool) -> Vec<u32> {
    let (hits, misses): (Vec<_>, Vec<_>) = experts.iter().partition(|&&e| resident(e));
    hits.into_iter().chain(misses).copied().collect()
}

/// RAM hits first across all loads of a chunk, then the misses, so the misses of one RAM group
/// cannot evict a hit that waits in a later group (ADR 0033).
fn ram_hits_first(loads: &[(u32, u32)], in_ram: impl Fn(u32) -> bool) -> Vec<(u32, u32)> {
    let (hits, misses): (Vec<_>, Vec<_>) = loads.iter().partition(|&&(e, _)| in_ram(e));
    hits.into_iter().chain(misses).copied().collect()
}

/// Reads the three slabs of each expert into its buffer (`total` bytes: gate, up, down) with
/// one batch. Returns the bytes read.
fn read_experts(
    dio: &DirectFile,
    file: &ModelFile,
    il: usize,
    items: Vec<(u32, &mut [u8])>,
    io: &mut IoCounters,
) -> Result<u64> {
    let mut reqs: Vec<ReadReq<'_>> = Vec::with_capacity(items.len() * 3);
    let n = items.len() as u64;
    for (e, buf) in items {
        let slabs = file.expert_slabs(il as u32, e)?;
        let (gate, rest) = buf.split_at_mut(slabs.parts[0].1 as usize);
        let (up, down) = rest.split_at_mut(slabs.parts[1].1 as usize);
        for (dst, part) in [gate, up, down].into_iter().zip(slabs.parts) {
            reqs.push(ReadReq {
                offset: part.0,
                dst,
            });
        }
    }
    let t0 = Instant::now();
    let stats = dio.read_batch(&mut reqs)?;
    io.read_time += t0.elapsed();
    io.batches += 1;
    io.experts_read += n;
    io.bytes_read += stats.requested;
    Ok(stats.requested)
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
            ram_hits_first(l, |e| c.layer(0).slot_of(e).is_some())
        };
        assert_eq!(ram_reads(in_order), 3);
        assert_eq!(ram_reads(hits_first), 2);
    }
}

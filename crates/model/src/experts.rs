//! Expert streaming for the builder (ADR 0033): the slot caches of both tiers, the loads from
//! the GGUF through direct I/O, and the counters. No graph work: the builder asks for the
//! experts of a layer step to be resident and gets slot ids back.

use crate::hparams::ModelFile;
use crate::qwen3moe::{ModelError, Result};
use std::time::Instant;
use tetraeder_ggml_sys::ggml::Tensor;
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_memory::slots::{LayerStats, Lookup, SlotCache, Tier};
use tetraeder_memory::{DirectFile, IoOptions, ReadReq};

/// Reads from the model file since load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoCounters {
    /// `read_batch` calls.
    pub batches: u64,
    /// Experts read from the file (three slabs each).
    pub experts_read: u64,
    /// Bytes of those experts.
    pub bytes_read: u64,
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
    /// Layer steps that had to load at least one expert.
    pub partial_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStats {
    pub layers: Vec<LayerExpertStats>,
    pub io: IoCounters,
}

/// The top-k ids the router chose in one layer step (`[n_used, n_tokens]`, token-major), kept
/// when `LoadOptions::record_routing` is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingStep {
    pub layer: usize,
    pub ids: Vec<i32>,
}

/// Host copies of experts between the file and the compute tier: `slots` per layer, each
/// holding the three slabs of one expert back to back.
struct RamTier {
    cache: SlotCache,
    arenas: Vec<Vec<u8>>,
    slots: usize,
}

pub(crate) struct Streamer {
    dio: DirectFile,
    /// Bytes of one expert's gate, up and down slab.
    sizes: [usize; 3],
    total: usize,
    n_expert: u32,
    compute: SlotCache,
    ram: Option<RamTier>,
    metrics: MemoryMetrics,
    device: String,
    staging: Vec<u8>,
    io: IoCounters,
    full: Vec<u64>,
    partial: Vec<u64>,
}

fn invalid(msg: impl Into<String>) -> ModelError {
    ModelError::Invalid(msg.into())
}

impl Streamer {
    /// `slot_tensors[il]` are the gate, up and down slot tensors of layer `il`, `slots` deep.
    pub(crate) fn new(
        file: &ModelFile,
        slot_tensors: &[[Tensor<'_>; 3]],
        slots: usize,
        ram_slots: Option<usize>,
        io: IoOptions,
        metrics: MemoryMetrics,
        device: String,
    ) -> Result<Streamer> {
        let h = &file.hparams;
        let layers = h.n_layer as usize;
        let first = file.expert_slabs(0, 0)?;
        let sizes = first.parts.map(|p| p.1 as usize);
        for (il, t) in slot_tensors.iter().enumerate() {
            let slabs = file.expert_slabs(il as u32, 0)?;
            for m in 0..3 {
                if t[m].nb()[2] as u64 != slabs.parts[m].1 || slabs.parts[m].1 as usize != sizes[m]
                {
                    return Err(invalid(format!(
                        "layer {il}: slot size does not match the expert slab"
                    )));
                }
            }
        }
        let total: usize = sizes.iter().sum();
        let n_expert = h.n_expert;
        let ram = ram_slots.map(|n| RamTier {
            cache: SlotCache::new(Tier::Ram, h.n_layer, n_expert, n as u32),
            arenas: (0..layers).map(|_| vec![0u8; n * total]).collect(),
            slots: n,
        });
        Ok(Streamer {
            dio: DirectFile::open_with(&file.path, io)?,
            sizes,
            total,
            n_expert,
            compute: SlotCache::new(Tier::Vram, h.n_layer, n_expert, slots as u32),
            ram,
            metrics,
            device,
            staging: Vec::new(),
            io: IoCounters::default(),
            full: vec![0; layers],
            partial: vec![0; layers],
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
                })
                .collect(),
            io: self.io,
        }
    }

    /// Splits a layer step into chunks of experts that fit the compute slots together: the
    /// distinct experts in order of first appearance, resident ones first. One chunk is the
    /// usual case; more mean the overflow passes of ADR 0033.
    pub(crate) fn plan(&mut self, il: usize, chosen: &[i32]) -> Result<Vec<Vec<u32>>> {
        let mut seen = vec![false; self.n_expert as usize];
        let mut distinct = Vec::new();
        for &e in chosen {
            let Some(s) = usize::try_from(e).ok().and_then(|e| seen.get_mut(e)) else {
                return Err(invalid(format!("router chose expert {e}, out of range")));
            };
            if !std::mem::replace(s, true) {
                distinct.push(e as u32);
            }
        }
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
            self.compute.layer_mut(il).clear();
            if let Some(ram) = &mut self.ram {
                ram.cache.layer_mut(il).clear();
            }
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
                    let n = self.ram.as_ref().map_or(1, |r| r.slots.max(1));
                    for group in loads.chunks(n) {
                        self.load_via_ram(file, il, dst, group)?;
                    }
                }
            }
        }
        Ok(slots)
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
        let total = self.total;
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
                upload(dst, &self.sizes, slot, src)?;
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
        let ram = self.ram.as_mut().expect("RAM tier");
        let total = self.total;
        let (hits, misses): (Vec<_>, Vec<_>) = group
            .iter()
            .partition(|&&(e, _)| ram.cache.layer(il).slot_of(e).is_some());
        // (expert, compute slot, ram slot, read from the file)
        let mut plan: Vec<(u32, u32, usize, bool)> = Vec::with_capacity(group.len());
        for &(e, cslot) in hits.into_iter().chain(misses) {
            let res = ram.cache.layer_mut(il).lookup(e);
            self.metrics.lookup(il as u32, Tier::Ram, &res);
            match res {
                Lookup::Hit(s) => plan.push((e, cslot, s as usize, false)),
                Lookup::Miss { slot, .. } => plan.push((e, cslot, slot as usize, true)),
                Lookup::Bypass => return Err(invalid("no free RAM expert slot (all pinned)")),
            }
        }
        let arena = &mut ram.arenas[il];
        let n_read = plan.iter().filter(|p| p.3).count();
        if n_read > 0 {
            let mut chunks: Vec<Option<&mut [u8]>> = arena.chunks_mut(total).map(Some).collect();
            let items: Vec<(u32, &mut [u8])> = plan
                .iter()
                .filter(|p| p.3)
                .map(|p| (p.0, chunks[p.2].take().expect("distinct RAM slots")))
                .collect();
            let bytes = read_experts(&self.dio, file, il, items, &mut self.io)?;
            self.metrics.tier_read(&self.device, "nvme", bytes);
            let took = t0.elapsed() / n_read as u32;
            for _ in 0..n_read {
                self.metrics.expert_load(Tier::Ram, "nvme", took);
            }
        }
        let n_ram = plan.len() - n_read;
        for &(_, cslot, rslot, _) in &plan {
            upload(
                dst,
                &self.sizes,
                cslot,
                &arena[rslot * total..(rslot + 1) * total],
            )?;
        }
        if n_ram > 0 {
            self.metrics
                .tier_read(&self.device, "ram", (n_ram * total) as u64);
        }
        for (_, _, _, from_file) in &plan {
            let source = if *from_file { "nvme" } else { "ram" };
            self.record_loads(1, source, t0);
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
    let stats = dio.read_batch(&mut reqs)?;
    io.batches += 1;
    io.experts_read += n;
    io.bytes_read += stats.requested;
    Ok(stats.requested)
}

/// Writes one expert (gate, up, down slabs back to back in `src`) into slot `slot`.
fn upload(dst: &[Tensor<'_>; 3], sizes: &[usize; 3], slot: u32, src: &[u8]) -> Result<()> {
    let mut at = 0;
    for (t, &size) in dst.iter().zip(sizes) {
        t.set_bytes(&src[at..at + size], slot as usize * size)?;
        at += size;
    }
    Ok(())
}

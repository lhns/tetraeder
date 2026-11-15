//! Executor topology of a valid plan: ranks, stages and the slicing rule that the planner, the
//! executor and the builders share (ADR 0069, 0070). No ggml here.

use crate::{Expanded, Plan, Span};
use std::collections::BTreeSet;
use std::ops::Range;
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// Index into [`Topology::ranks`].
pub type RankId = usize;

/// One GPU backend of the executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankSpec {
    pub device: String,
    /// NVML index of the device, when the hardware graph knows it.
    pub gpu_index: Option<usize>,
}

/// A pipeline stage: a layer span computed by a group of ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSpec {
    pub layers: Span,
    /// The group in `Stage::devices` order; rank i of the group holds slice i.
    pub ranks: Vec<RankId>,
    pub tp: u32,
    pub ep: u32,
    /// The first stage holds the embedding.
    pub embed: bool,
    /// The last stage holds the output head.
    pub head: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    pub ranks: Vec<RankSpec>,
    pub stages: Vec<StageSpec>,
    pub n_layer: u32,
}

impl Topology {
    /// One replica only. Ranks in stage order, then `Stage::devices` order; a device in two
    /// stages is one rank. Collects every reason it cannot run.
    pub fn from_plan(
        plan: &Plan,
        x: &Expanded,
        hw: &HardwareGraph,
    ) -> Result<Topology, Vec<String>> {
        let mut errs: Vec<String> = Vec::new();
        if plan.replicas.len() != 1 {
            errs.push(format!(
                "{} replicas: the executor runs one replica",
                plan.replicas.len()
            ));
        }
        let Some(rep) = plan.replicas.first() else {
            return Err(errs);
        };

        let mut ranks: Vec<RankSpec> = Vec::new();
        let mut stages: Vec<StageSpec> = Vec::new();
        let mut next_layer = 0u32;
        let n = rep.stages.len();
        for (s, st) in rep.stages.iter().enumerate() {
            let (tp, ep) = (st.tp.unwrap_or(1), st.ep.unwrap_or(1));
            if st.layers.is_empty() || st.layers.0 != next_layer {
                errs.push(format!(
                    "stage {s}: layers [{}, {}) do not continue at layer {next_layer}",
                    st.layers.0, st.layers.1
                ));
            }
            next_layer = st.layers.1;
            if st.devices.len() != tp as usize {
                errs.push(format!(
                    "stage {s}: {} devices for tp = {tp}; the executor runs one tp group per stage",
                    st.devices.len()
                ));
            }
            if ep != 1 && ep != tp {
                errs.push(format!("stage {s}: ep = {ep} needs tp = ep, got tp = {tp}"));
            }
            if !st.expert_cache.is_empty() {
                errs.push(format!(
                    "stage {s}: slot caches are not executable (GPU-resident stages only)"
                ));
            }
            let mut group: Vec<RankId> = Vec::new();
            for (k, d) in st.devices.iter().enumerate() {
                if st.devices[..k].contains(d) {
                    errs.push(format!("stage {s}: device {d} listed twice"));
                    continue;
                }
                let Some(dev) = hw.device(d) else {
                    errs.push(format!("stage {s}: unknown device {d}"));
                    continue;
                };
                if dev.kind != DeviceKind::Gpu {
                    errs.push(format!(
                        "stage {s}: device {d} is not a GPU; the executor runs GPU stages only"
                    ));
                    continue;
                }
                let rank = match ranks.iter().position(|r| r.device == *d) {
                    Some(r) => r,
                    None => {
                        ranks.push(RankSpec {
                            device: d.clone(),
                            gpu_index: dev.index.map(|i| i as usize),
                        });
                        ranks.len() - 1
                    }
                };
                group.push(rank);
            }
            stages.push(StageSpec {
                layers: st.layers,
                ranks: group,
                tp,
                ep,
                embed: s == 0,
                head: s + 1 == n,
            });
        }

        // Every weight lives on its stage's GPUs and computes there.
        let mut shard_errs: BTreeSet<String> = BTreeSet::new();
        for p in x
            .shards
            .iter()
            .filter(|(k, _)| k.replica == 0)
            .map(|(_, p)| p)
        {
            let s = p.stage;
            match rep.stages.get(s as usize) {
                None => {
                    shard_errs.insert(format!("a shard names stage {s}, which the plan lacks"));
                }
                Some(st) => {
                    if !st.devices.contains(&p.home) {
                        shard_errs.insert(format!(
                            "stage {s}: weights homed on {}, outside the stage's devices",
                            p.home
                        ));
                    }
                    if p.compute != p.home {
                        shard_errs.insert(format!(
                            "stage {s}: a shard homed on {} computes on {}",
                            p.home, p.compute
                        ));
                    }
                }
            }
        }
        errs.extend(shard_errs);

        if errs.is_empty() {
            Ok(Topology {
                ranks,
                stages,
                n_layer: next_layer,
            })
        } else {
            Err(errs)
        }
    }

    /// One rank, one stage over all layers.
    pub fn single(device: &str, gpu_index: Option<usize>, n_layer: u32) -> Topology {
        Topology {
            ranks: vec![RankSpec {
                device: device.to_string(),
                gpu_index,
            }],
            stages: vec![StageSpec {
                layers: Span(0, n_layer),
                ranks: vec![0],
                tp: 1,
                ep: 1,
                embed: true,
                head: true,
            }],
            n_layer,
        }
    }

    /// The stage that computes layer `il`. Panics if `il >= n_layer`.
    pub fn stage_of_layer(&self, il: u32) -> &StageSpec {
        self.stages
            .iter()
            .find(|s| s.layers.contains(il))
            .unwrap_or_else(|| panic!("layer {il} is outside the {} layers", self.n_layer))
    }

    pub fn is_single(&self) -> bool {
        self.ranks.len() == 1 && self.stages.len() == 1
    }
}

/// Range of `len` elements for slice `i` of `n`, whole blocks of `block`, extra blocks to ranks
/// `(rot + j) % n` for j in 0..extra (ADR 0070). Errors if any slice is empty or `len` is not a
/// multiple of `block`.
pub fn block_slice(len: u64, block: u64, n: u32, i: u32, rot: u32) -> Result<Range<u64>, String> {
    if block == 0 || n == 0 {
        return Err("block and slice count must be positive".to_string());
    }
    if i >= n {
        return Err(format!("slice {i} of {n}"));
    }
    if !len.is_multiple_of(block) {
        return Err(format!("{len} elements are not whole blocks of {block}"));
    }
    let (n, blocks) = (u64::from(n), len / block);
    let (base, extra) = (blocks / n, blocks % n);
    if base == 0 {
        return Err(format!(
            "{blocks} blocks cannot give {n} slices one block each"
        ));
    }
    // Position of rank r among the ranks that get an extra block first: (r - rot) mod n.
    let rot = u64::from(rot) % n;
    let blocks_of = |r: u64| base + u64::from((r + n - rot) % n < extra);
    let start: u64 = (0..u64::from(i)).map(blocks_of).sum();
    let end = start + blocks_of(u64::from(i));
    Ok(start * block..end * block)
}

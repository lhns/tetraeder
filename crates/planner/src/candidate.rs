//! Executable candidate plans for one GPU plus the CPU (ADR 0040).
//!
//! [`enumerate`] depends only on capacities, the model layout and the workload, never on a
//! calibration or cost model, so the measured set of a fixture is fixed before scoring.

use crate::layout::ModelLayout;
use serde::{Deserialize, Serialize};
use tetraeder_plan::{Assign, DeviceBytes, Knobs, Plan, Replica, Span, Stage, UnitKind, UnitSel};

/// Device ids in every candidate plan.
pub const GPU: &str = "gpu0";
pub const CPU: &str = "cpu0";

/// Fixed CUDA context, cuBLAS and allocator slack on the GPU.
pub const CUDA_RESERVE: u64 = 600 << 20;
/// Fixed host slack (process, mmap bookkeeping, graph memory) on top of CPU-side weights, KV.
pub const HOST_RESERVE: u64 = 2 << 30;
/// Context length is rounded up to this.
pub const CTX_ROUND: u32 = 256;
/// Most candidates one enumeration returns.
pub const MAX_CANDIDATES: usize = 24;
/// Placements per enumeration (MoE: three steps down from the most experts on the GPU, plus all
/// experts on the CPU).
pub const PLACEMENTS: usize = 4;

pub const UBATCHES: [u32; 2] = [512, 2048];

/// Memory the engine may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Free VRAM for the engine.
    pub gpu_bytes: u64,
    pub ram_bytes: u64,
    /// CPU thread quota (not the visible core count).
    pub threads: u32,
}

/// The request shape the candidates are sized and measured for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workload {
    pub prompt: u32,
    pub r#gen: u32,
}

impl Workload {
    /// Context length of a request, rounded up to [`CTX_ROUND`].
    pub fn n_ctx(&self) -> u64 {
        u64::from((self.prompt + self.r#gen).div_ceil(CTX_ROUND) * CTX_ROUND)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    LlamaBench,
}

/// Where the weights live. Both keep the embedding on the CPU, as llama.cpp does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Placement {
    /// The last `gpu_layers` layers and the output head on the GPU, the rest on the CPU.
    Dense { gpu_layers: u32 },
    /// Attention, FFN and head of all layers on the GPU; the experts of layers
    /// `[0, gpu_expert_layers)` on the GPU, the experts of the others on the CPU.
    Moe { gpu_expert_layers: u32 },
}

impl Placement {
    pub fn describe(&self) -> String {
        match self {
            Self::Dense { gpu_layers } => format!("dense gpu_layers={gpu_layers}"),
            Self::Moe { gpu_expert_layers } => format!("moe gpu_expert_layers={gpu_expert_layers}"),
        }
    }

    /// Larger means more weight bytes on the GPU.
    pub fn gpu_share(&self) -> u32 {
        match *self {
            Self::Dense { gpu_layers } => gpu_layers,
            Self::Moe { gpu_expert_layers } => gpu_expert_layers,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// `c00`, `c01`, ... in enumeration order.
    pub id: String,
    /// A complete plan; the run settings are in `plan.knobs`.
    pub plan: Plan,
    pub engine: EngineKind,
    pub placement: Placement,
    /// The request this candidate was sized for and is measured with.
    pub prompt: u32,
    pub r#gen: u32,
    pub notes: Vec<String>,
}

impl Candidate {
    /// Run settings; `enumerate` and `build_candidate` always set all three.
    pub fn threads(&self) -> u32 {
        self.plan.knobs.threads.unwrap_or(1)
    }

    pub fn ubatch(&self) -> u32 {
        self.plan.knobs.ubatch.unwrap_or(512)
    }

    pub fn op_offload(&self) -> bool {
        self.plan.knobs.op_offload.unwrap_or(true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Enumeration {
    pub candidates: Vec<Candidate>,
    /// What was left out: (description, reason).
    pub dropped: Vec<(String, String)>,
    /// Feasible placements of the candidates, most GPU weights first. Anchors use the ends.
    pub placements: Vec<Placement>,
}

/// Estimated bytes of a candidate on each side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Footprint {
    pub weights_gpu: u64,
    pub weights_host: u64,
    pub kv_gpu: u64,
    pub kv_host: u64,
    /// Compute buffers and logits on the GPU.
    pub compute_gpu: u64,
    /// Everything on the GPU, with the CUDA reserve; 0 when nothing runs there.
    pub gpu: u64,
    /// Everything in host RAM, with the host reserve.
    pub host: u64,
}

/// Conservative memory estimate of a placement.
///
/// - weights: from the layout, per layer; the embedding is always on the host. A dense
///   placement with at least one GPU layer also puts the output head on the GPU.
/// - KV: f16, `2 * n_head_kv * head_dim * 2 B` per token per layer, for `n_ctx` tokens
///   ([`Workload::n_ctx`]), on the side of its layer (MoE: all on the GPU).
/// - compute buffers (GPU): `ubatch * (4 * n_vocab + 64 * n_embd)` bytes, an upper bound for
///   logits and activations with flash attention.
/// - reserves: [`CUDA_RESERVE`] on the GPU, [`HOST_RESERVE`] on the host.
///
/// Host page cache of mmapped GPU weights is reclaimable and not counted.
pub fn footprint(
    layout: &ModelLayout,
    placement: Placement,
    ubatch: u32,
    workload: &Workload,
) -> Footprint {
    let l = layout.n_layer as usize;
    let mut f = Footprint {
        weights_host: layout.embed_bytes,
        ..Default::default()
    };
    let layer_kv = layout.kv_bytes_per_token_per_layer() * workload.n_ctx();
    let gpu_layers = match placement {
        Placement::Dense { gpu_layers } => {
            let g = (gpu_layers as usize).min(l);
            for (i, b) in layout.layers.iter().enumerate() {
                let all = b.attn_bytes + b.ffn_bytes + b.expert_bytes;
                if i >= l - g {
                    f.weights_gpu += all;
                } else {
                    f.weights_host += all;
                }
            }
            if g > 0 {
                f.weights_gpu += layout.output_bytes;
            } else {
                f.weights_host += layout.output_bytes;
            }
            g
        }
        Placement::Moe { gpu_expert_layers } => {
            for (i, b) in layout.layers.iter().enumerate() {
                f.weights_gpu += b.dense_bytes();
                if i < gpu_expert_layers as usize {
                    f.weights_gpu += b.expert_bytes;
                } else {
                    f.weights_host += b.expert_bytes;
                }
            }
            f.weights_gpu += layout.output_bytes;
            l
        }
    };
    f.kv_gpu = layer_kv * gpu_layers as u64;
    f.kv_host = layer_kv * (l - gpu_layers) as u64;
    if gpu_layers > 0 {
        f.compute_gpu =
            u64::from(ubatch) * (4 * u64::from(layout.n_vocab) + 64 * u64::from(layout.n_embd));
        f.gpu = f.weights_gpu + f.kv_gpu + f.compute_gpu + CUDA_RESERVE;
    }
    f.host = f.weights_host + f.kv_host + HOST_RESERVE;
    f
}

fn fits(b: &Budget, f: &Footprint) -> Result<(), String> {
    if f.gpu > b.gpu_bytes {
        return Err(format!(
            "gpu estimate {} B > budget {} B",
            f.gpu, b.gpu_bytes
        ));
    }
    if f.host > b.ram_bytes {
        return Err(format!(
            "ram estimate {} B > budget {} B",
            f.host, b.ram_bytes
        ));
    }
    Ok(())
}

fn assign(
    kinds: &[UnitKind],
    layers: Option<Span>,
    device: &str,
    fmt: tetraeder_plan::Format,
) -> Assign {
    Assign {
        units: UnitSel {
            kinds: kinds.to_vec(),
            layers,
            experts: None,
        },
        home: Some(vec![device.to_string()]),
        compute: None,
        format: Some(fmt),
    }
}

/// The complete one-replica, one-stage plan of a placement.
pub fn build_plan(
    layout: &ModelLayout,
    placement: Placement,
    knobs: Knobs,
    ubatch: u32,
    workload: &Workload,
) -> Plan {
    use UnitKind::{Attn, Embed, Expert, Head, Mlp};
    let (l, fmt) = (layout.n_layer, layout.main_format());
    let mut rules = vec![assign(&[Embed], None, CPU, fmt)];
    match placement {
        Placement::Dense { gpu_layers } => {
            let g = gpu_layers.min(l);
            if g < l {
                rules.push(assign(&[Attn, Mlp], Some(Span(0, l - g)), CPU, fmt));
            }
            if g > 0 {
                rules.push(assign(&[Attn, Mlp], Some(Span(l - g, l)), GPU, fmt));
            }
            rules.push(assign(&[Head], None, if g > 0 { GPU } else { CPU }, fmt));
        }
        Placement::Moe { gpu_expert_layers } => {
            let k = gpu_expert_layers.min(l);
            rules.push(assign(&[Attn, Mlp, Head], None, GPU, fmt));
            if k > 0 {
                rules.push(assign(&[Expert], Some(Span(0, k)), GPU, fmt));
            }
            if k < l {
                rules.push(assign(&[Expert], Some(Span(k, l)), CPU, fmt));
            }
        }
    }
    let f = footprint(layout, placement, ubatch, workload);
    let bytes = |device: &str, bytes: u64| DeviceBytes {
        device: device.to_string(),
        bytes,
    };
    let kv = [(GPU, f.kv_gpu), (CPU, f.kv_host)]
        .into_iter()
        .filter(|(_, b)| *b > 0)
        .map(|(d, b)| bytes(d, b))
        .collect();
    Plan {
        model: layout.model.clone(),
        knobs,
        replicas: vec![Replica {
            stages: vec![Stage {
                layers: Span(0, l),
                devices: vec![GPU.to_string(), CPU.to_string()],
                tp: Some(1),
                ep: Some(1),
                assign: rules,
                kv: Some(kv),
                workspace: if f.compute_gpu > 0 {
                    vec![bytes(GPU, f.compute_gpu)]
                } else {
                    vec![]
                },
                ..Default::default()
            }],
        }],
    }
}

/// One candidate with the given run settings; `enumerate` and the campaign anchors use it.
pub fn build_candidate(
    id: &str,
    layout: &ModelLayout,
    placement: Placement,
    (threads, ubatch, op_offload): (u32, u32, bool),
    workload: &Workload,
) -> Candidate {
    let knobs = Knobs {
        threads: Some(threads),
        ubatch: Some(ubatch),
        op_offload: Some(op_offload),
        ..Default::default()
    };
    let f = footprint(layout, placement, ubatch, workload);
    Candidate {
        id: id.to_string(),
        plan: build_plan(layout, placement, knobs, ubatch, workload),
        engine: EngineKind::LlamaBench,
        placement,
        prompt: workload.prompt,
        r#gen: workload.r#gen,
        notes: vec![format!(
            "{}; estimate gpu {} MiB, host {} MiB",
            placement.describe(),
            f.gpu >> 20,
            f.host >> 20
        )],
    }
}

/// Placement step in layers.
pub fn step(n_layer: u32) -> u32 {
    ((f64::from(n_layer) / 12.0).round() as u32).max(2)
}

/// Thread counts: the quota and half of it (at least 1).
pub fn thread_values(quota: u32) -> Vec<u32> {
    let q = quota.max(1);
    let mut v = vec![q, (q / 2).max(1)];
    v.dedup();
    v
}

/// Candidate placements, most GPU weights first, plus what was dropped.
///
/// Dense: the largest GPU layer count that fits at the smallest ubatch, then [`PLACEMENTS`] - 1
/// steps of `step(L)` layers down. MoE: the largest number of leading layers whose experts fit
/// on the GPU, two steps of `step(L)` down while above 0, and always 0 (all experts on the CPU).
fn placements(
    budget: &Budget,
    layout: &ModelLayout,
    workload: &Workload,
    dropped: &mut Vec<(String, String)>,
) -> Vec<Placement> {
    let l = layout.n_layer;
    let ub = UBATCHES[0];
    let mk = |x: u32| {
        if layout.is_moe() {
            Placement::Moe {
                gpu_expert_layers: x,
            }
        } else {
            Placement::Dense { gpu_layers: x }
        }
    };
    let lo = u32::from(!layout.is_moe());
    let ok = |x: u32| fits(budget, &footprint(layout, mk(x), ub, workload));
    let Some(max) = (lo..=l).rev().find(|&x| ok(x).is_ok()) else {
        let why = ok(lo).err().unwrap_or_default();
        dropped.push(("all placements".into(), why));
        return vec![];
    };
    let st = step(l);
    let mut xs: Vec<u32> = if layout.is_moe() {
        (0..3)
            .filter_map(|i| max.checked_sub(i * st))
            .filter(|&x| x > 0)
            .chain([0])
            .collect()
    } else {
        (0..PLACEMENTS as u32)
            .filter_map(|i| max.checked_sub(i * st))
            .filter(|&x| x >= 1)
            .collect()
    };
    xs.dedup();
    let mut out = Vec::new();
    for x in xs {
        match ok(x) {
            Ok(()) => out.push(mk(x)),
            Err(e) => dropped.push((mk(x).describe(), e)),
        }
    }
    out
}

/// One row of the knob grid: indices into the placement list and the knob value lists.
#[derive(Clone, Copy)]
struct Row {
    p: usize,
    t: usize,
    u: usize,
    o: usize,
}

impl Row {
    /// How many settings differ from the first value of their list.
    fn non_default(&self) -> usize {
        self.t + self.u + self.o
    }
}

/// Thin to `cap` rows, deterministically. Round by round, every placement in order gives up
/// one row: the one with the most non-first knob values (ties: the later one in enumeration
/// order). A row is kept if removing it would leave a placement or a knob value unrepresented.
fn thin(mut rows: Vec<Row>, n_placements: usize, cap: usize) -> Vec<Row> {
    let covered = |rows: &[Row]| {
        let mut c = [[0usize; 2]; 3];
        for r in rows {
            c[0][r.t] += 1;
            c[1][r.u] += 1;
            c[2][r.o] += 1;
        }
        c
    };
    loop {
        let mut removed = false;
        for p in 0..n_placements {
            if rows.len() <= cap {
                return rows;
            }
            let mut order: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].p == p).collect();
            order.sort_by_key(|&i| std::cmp::Reverse((rows[i].non_default(), i)));
            let before = covered(&rows);
            for i in order {
                let r = rows[i];
                let last_of_placement = rows.iter().filter(|x| x.p == p).count() == 1;
                let last_value = before[0][r.t] == 1 || before[1][r.u] == 1 || before[2][r.o] == 1;
                if !last_of_placement && !last_value {
                    rows.remove(i);
                    removed = true;
                    break;
                }
            }
        }
        if !removed {
            return rows;
        }
    }
}

/// The candidate set of a fixture for one GPU plus the CPU, deterministic in its inputs.
///
/// Placements ([`footprint`]-checked at the smallest ubatch) times threads in {quota, quota / 2},
/// ubatch in {512, 2048}, op offload in {on, off}. A combination whose estimate exceeds the
/// budget is dropped. If more than [`MAX_CANDIDATES`] remain they are thinned by the rule
/// of `thin`: every placement and every knob value stays represented. Ids `c00`... follow the
/// final order: placement (most GPU weights first), then threads, ubatch, op offload on before off.
pub fn enumerate(budget: &Budget, layout: &ModelLayout, workload: &Workload) -> Enumeration {
    let mut dropped = Vec::new();
    let places = placements(budget, layout, workload, &mut dropped);
    let threads = thread_values(budget.threads);
    let ops = [true, false];
    let mut rows = Vec::new();
    for (p, &pl) in places.iter().enumerate() {
        for (t, &th) in threads.iter().enumerate() {
            for (u, &ub) in UBATCHES.iter().enumerate() {
                for (o, &op) in ops.iter().enumerate() {
                    let desc =
                        format!("{} threads={th} ubatch={ub} op_offload={op}", pl.describe());
                    match fits(budget, &footprint(layout, pl, ub, workload)) {
                        Ok(()) => rows.push(Row { p, t, u, o }),
                        Err(e) => dropped.push((desc, e)),
                    }
                }
            }
        }
    }
    let n = rows.len();
    let kept = thin(rows.clone(), places.len(), MAX_CANDIDATES);
    if kept.len() < n {
        for r in &rows {
            let key = |x: &Row| (x.p, x.t, x.u, x.o);
            if !kept.iter().any(|k| key(k) == key(r)) {
                let desc = format!(
                    "{} threads={} ubatch={} op_offload={}",
                    places[r.p].describe(),
                    threads[r.t],
                    UBATCHES[r.u],
                    ops[r.o]
                );
                dropped.push((desc, format!("thinned to {MAX_CANDIDATES}")));
            }
        }
    }
    let candidates = kept
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let knobs = (threads[r.t], UBATCHES[r.u], ops[r.o]);
            build_candidate(&format!("c{i:02}"), layout, places[r.p], knobs, workload)
        })
        .collect();
    Enumeration {
        candidates,
        dropped,
        placements: places,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::fixtures;
    use std::collections::{BTreeMap, BTreeSet};
    use tetraeder_plan::{Context, Phase, validate};
    use tetraeder_profiler::HardwareGraph;

    const BUDGET: Budget = Budget {
        gpu_bytes: 14_000_000_000,
        ram_bytes: 60_000_000_000,
        threads: 15,
    };
    const WORK: Workload = Workload {
        prompt: 2048,
        r#gen: 128,
    };

    fn hw(b: &Budget) -> HardwareGraph {
        HardwareGraph::from_toml(&format!(
            r#"
[[devices]]
id = "cpu0"
kind = "cpu"
name = "cpu"
tier = "ram"
capacity_bytes = {ram}
cores = 8
threads = 16
numa_nodes = 1
bandwidth_bytes_per_s = {{ value = 5.0e10, source = "prior" }}
compute = {{ flops_per_s = 1.0e12, c0_graph_s = 2.0e-6, c0_eager_s = 5.0e-6 }}

[[devices]]
id = "gpu0"
kind = "gpu"
name = "gpu"
tier = "vram"
capacity_bytes = {gpu}
index = 0
bandwidth_bytes_per_s = {{ value = 9.0e11, source = "prior" }}
compute = {{ flops_per_s = 3.0e13, c0_graph_s = 5.0e-6, c0_eager_s = 2.0e-5 }}

[[links]]
id = "pcie-gpu0"
a = "gpu0"
b = "cpu0"
kind = "pcie"
alpha_s = {{ value = 1.0e-5, source = "prior" }}
beta_bytes_per_s = {{ value = 2.5e10, source = "prior" }}
"#,
            ram = b.ram_bytes,
            gpu = b.gpu_bytes
        ))
        .unwrap()
    }

    fn check(layout: &ModelLayout) -> Enumeration {
        let en = enumerate(&BUDGET, layout, &WORK);
        assert!(en.candidates.len() >= 20, "{}", en.candidates.len());
        assert!(en.candidates.len() <= MAX_CANDIDATES);
        assert_eq!(en, enumerate(&BUDGET, layout, &WORK), "deterministic");
        let ids: Vec<_> = en.candidates.iter().map(|c| c.id.clone()).collect();
        let want: Vec<_> = (0..ids.len()).map(|i| format!("c{i:02}")).collect();
        assert_eq!(ids, want);

        let (hw, model) = (hw(&BUDGET), layout.to_model_desc());
        let reserve = BTreeMap::from([
            ("gpu0".to_string(), CUDA_RESERVE),
            ("cpu0".to_string(), HOST_RESERVE),
        ]);
        let ctx = Context {
            hw: &hw,
            model: &model,
            phase: Phase(3),
            reserve: &reserve,
            hints: None,
        };
        let (mut th, mut ub, mut op, mut pl) = (
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        for c in &en.candidates {
            let f = footprint(layout, c.placement, c.ubatch(), &WORK);
            assert!(
                f.gpu <= BUDGET.gpu_bytes && f.host <= BUDGET.ram_bytes,
                "{}",
                c.id
            );
            if let Err(e) = validate(&c.plan, &ctx) {
                panic!("{} {:?}: {e:#?}", c.id, c.placement);
            }
            th.insert(c.threads());
            ub.insert(c.ubatch());
            op.insert(c.op_offload());
            pl.insert(c.placement);
        }
        assert_eq!(th, BTreeSet::from([7, 15]));
        assert_eq!(ub, BTreeSet::from([512, 2048]));
        assert_eq!(op, BTreeSet::from([false, true]));
        assert_eq!(pl.len(), PLACEMENTS);
        assert_eq!(pl.len(), en.placements.len());
        en
    }

    #[test]
    fn dense_candidates() {
        let l = fixtures::dense();
        let en = check(&l);
        let gs: Vec<u32> = en.placements.iter().map(Placement::gpu_share).collect();
        // strictly descending in even steps of round(64 / 12) = 5
        assert!(gs.windows(2).all(|w| w[0] - w[1] == 5), "{gs:?}");
        assert!(gs[0] < 64 && gs[3] >= 1);
        // the largest placement is the largest that fits at ubatch 512
        let bigger = Placement::Dense {
            gpu_layers: gs[0] + 1,
        };
        let f = footprint(&l, bigger, 512, &WORK);
        assert!(f.gpu > BUDGET.gpu_bytes);
        assert!(en.dropped.iter().any(|(_, why)| why.contains("thinned")));
    }

    #[test]
    fn moe_candidates() {
        let l = fixtures::moe();
        let en = check(&l);
        let ks: Vec<u32> = en.placements.iter().map(Placement::gpu_share).collect();
        assert_eq!(*ks.last().unwrap(), 0, "all experts on the CPU is last");
        assert!(ks[0] > 0 && ks[0] < 48);
        assert!(ks.windows(2).all(|w| w[0] > w[1]));
        // non-expert weights of every layer are on the GPU in every candidate
        let st = &en.candidates[0].plan.replicas[0].stages[0];
        assert!(
            st.assign
                .iter()
                .any(|a| a.units.kinds.contains(&UnitKind::Attn)
                    && a.home.as_deref() == Some(&["gpu0".to_string()][..]))
        );
    }

    #[test]
    fn nothing_fits_is_reported() {
        let l = fixtures::moe();
        let b = Budget {
            gpu_bytes: 1 << 30,
            ..BUDGET
        };
        let en = enumerate(&b, &l, &WORK);
        assert!(en.candidates.is_empty() && !en.dropped.is_empty());
    }

    #[test]
    fn edge_placements_drop_the_big_ubatch_not_the_placement() {
        let l = fixtures::dense();
        let en = check(&l);
        let top = en.placements[0];
        // a budget just above the ubatch 512 estimate of the top placement but below 2048's
        let f512 = footprint(&l, top, 512, &WORK).gpu;
        let f2048 = footprint(&l, top, 2048, &WORK).gpu;
        let b = Budget {
            gpu_bytes: f512,
            ..BUDGET
        };
        let en = enumerate(&b, &l, &WORK);
        assert!(f2048 > f512);
        assert!(
            en.candidates
                .iter()
                .all(|c| footprint(&l, c.placement, c.ubatch(), &WORK).gpu <= f512)
        );
        assert!(
            en.dropped
                .iter()
                .any(|(_, why)| why.starts_with("gpu estimate"))
        );
    }

    #[test]
    fn small_quota_dedupes_threads() {
        assert_eq!(thread_values(1), vec![1]);
        assert_eq!(thread_values(15), vec![15, 7]);
        assert_eq!(step(48), 4);
        assert_eq!(step(64), 5);
        assert_eq!(step(8), 2);
    }
}

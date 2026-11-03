//! Candidates for the own MoE builder: expert slot counts per tier plus run knobs (ADR 0040,
//! "or `exit_regime` environment for the builder path"; slots from ADR 0033, 0034, 0037).
//!
//! The builder keeps every dense weight on the GPU and streams experts through two slot tiers:
//! `vram_slots` per layer on the GPU (the compute tier) and `ram_slots` per layer in host RAM.
//! Slot counts are at least the experts per token (ADR 0034). Like [`crate::enumerate`] the set
//! depends only on capacities, the layout and the workload, so it is fixed before scoring.

use crate::candidate::{
    Budget, Candidate, EngineKind, Enumeration, Footprint, MAX_CANDIDATES, Placement, Row,
    Workload, build_plan, footprint, thin, thread_values,
};
use crate::layout::ModelLayout;
use tetraeder_plan::Knobs;

/// Ubatch values of builder candidates.
pub const BUILDER_UBATCHES: [u32; 2] = [256, 2048];

/// Bytes of one slot in every layer: the sum over layers of one expert's bytes (ADR 0037), the
/// unit that slot byte budgets are divided by.
pub fn slot_row_bytes(layout: &ModelLayout) -> u64 {
    let n = u64::from(layout.n_expert.max(1));
    layout
        .layers
        .iter()
        .map(|l| l.expert_bytes.div_ceil(n))
        .sum()
}

/// VRAM bytes of `slots` slots per layer: the slots plus the `n_used` zero parking entries that
/// masked pairs use in overflow passes (ADR 0034).
pub fn vram_slot_bytes(layout: &ModelLayout, slots: u32) -> u64 {
    u64::from(slots + layout.n_expert_used) * slot_row_bytes(layout)
}

/// Host bytes of `slots` RAM-tier slots per layer.
pub fn ram_slot_bytes(layout: &ModelLayout, slots: u32) -> u64 {
    u64::from(slots) * slot_row_bytes(layout)
}

/// VRAM slot counts per layer: top-k, then doubling (2x, 4x, ...) while below the expert count,
/// then all experts. Ascending.
pub fn vram_slot_tiers(layout: &ModelLayout) -> Vec<u32> {
    let (k, n) = (layout.n_expert_used.max(1), layout.n_expert);
    let mut v = Vec::new();
    let mut s = k;
    while s < n {
        v.push(s);
        s = s.saturating_mul(2);
    }
    v.push(n.max(k));
    v
}

/// RAM-tier slot counts for a VRAM tier: a quarter, half and all of the experts not in VRAM,
/// each at least top-k and at most the expert count. Ascending, distinct.
pub fn ram_slot_options(layout: &ModelLayout, vram_slots: u32) -> Vec<u32> {
    let (k, n) = (layout.n_expert_used.max(1), layout.n_expert.max(1));
    let rest = n.saturating_sub(vram_slots);
    let mut v: Vec<u32> = [rest / 4, rest / 2, rest]
        .into_iter()
        .map(|x| x.clamp(k.min(n), n))
        .collect();
    v.dedup();
    v
}

fn fits_at(
    budget: &Budget,
    layout: &ModelLayout,
    pl: Placement,
    ubatch: u32,
    w: &Workload,
) -> Result<Footprint, String> {
    let f = footprint(layout, pl, ubatch, w);
    if f.gpu > budget.gpu_bytes {
        return Err(format!(
            "gpu estimate {} B > budget {} B",
            f.gpu, budget.gpu_bytes
        ));
    }
    if f.host > budget.ram_bytes {
        return Err(format!(
            "ram estimate {} B > budget {} B",
            f.host, budget.ram_bytes
        ));
    }
    Ok(f)
}

/// VRAM slot counts whose smallest-ubatch estimate fits the budget (with the RAM tier at its
/// smallest), ascending. What does not fit goes to `dropped`.
pub fn feasible_vram_tiers(
    budget: &Budget,
    layout: &ModelLayout,
    w: &Workload,
    dropped: &mut Vec<(String, String)>,
) -> Vec<u32> {
    let mut out = Vec::new();
    for v in vram_slot_tiers(layout) {
        let r = ram_slot_options(layout, v)[0];
        let pl = Placement::Builder {
            vram_slots: v,
            ram_slots: r,
        };
        match fits_at(budget, layout, pl, BUILDER_UBATCHES[0], w) {
            Ok(_) => out.push(v),
            Err(e) => dropped.push((pl.describe(), e)),
        }
    }
    out
}

/// One builder candidate with the given run settings.
pub fn build_builder_candidate(
    id: &str,
    layout: &ModelLayout,
    (vram_slots, ram_slots): (u32, u32),
    (threads, ubatch, op_offload): (u32, u32, bool),
    workload: &Workload,
) -> Candidate {
    let placement = Placement::Builder {
        vram_slots,
        ram_slots,
    };
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
        engine: EngineKind::Builder,
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

/// The builder candidate set of a fixture for one GPU plus the CPU, deterministic in its inputs.
///
/// VRAM slot tiers ([`vram_slot_tiers`], those that fit) times RAM options
/// ([`ram_slot_options`]) times threads in {quota, quota / 2}, ubatch in {256, 2048}, op offload
/// in {on, off}. A combination whose estimate exceeds the budget is dropped. If more than
/// [`MAX_CANDIDATES`] remain they are thinned by the rule of the libllama enumerator: every VRAM
/// tier and every knob value stays represented. Ids `c00`... follow the final order: VRAM slots
/// descending, then RAM slots, threads, ubatch, op offload on before off.
pub fn enumerate_builder(budget: &Budget, layout: &ModelLayout, w: &Workload) -> Enumeration {
    let mut dropped = Vec::new();
    if !layout.is_moe() {
        dropped.push(("all placements".into(), "the model is not a MoE".into()));
        return Enumeration {
            candidates: vec![],
            dropped,
            placements: vec![],
        };
    }
    let mut tiers = feasible_vram_tiers(budget, layout, w, &mut dropped);
    tiers.reverse();
    let threads = thread_values(budget.threads);
    let ops = [true, false];
    // (vram, ram, threads, ubatch, op) of a row.
    let desc_of = |r: &Row, rams: &[Vec<u32>]| {
        (
            tiers[r.p],
            rams[r.p][r.k[3]],
            threads[r.k[0]],
            BUILDER_UBATCHES[r.k[1]],
            ops[r.k[2]],
        )
    };
    let describe = |d: (u32, u32, u32, u32, bool)| {
        format!(
            "builder vram_slots={} ram_slots={} threads={} ubatch={} op_offload={}",
            d.0, d.1, d.2, d.3, d.4
        )
    };
    let rams: Vec<Vec<u32>> = tiers.iter().map(|&v| ram_slot_options(layout, v)).collect();
    let mut rows = Vec::new();
    for (p, &v) in tiers.iter().enumerate() {
        for (ri, &r) in rams[p].iter().enumerate() {
            let pl = Placement::Builder {
                vram_slots: v,
                ram_slots: r,
            };
            for (t, _) in threads.iter().enumerate() {
                for (u, &ub) in BUILDER_UBATCHES.iter().enumerate() {
                    for (o, _) in ops.iter().enumerate() {
                        let row = Row {
                            p,
                            k: [t, u, o, ri],
                        };
                        match fits_at(budget, layout, pl, ub, w) {
                            Ok(_) => rows.push(row),
                            Err(e) => dropped.push((describe(desc_of(&row, &rams)), e)),
                        }
                    }
                }
            }
        }
    }
    let n = rows.len();
    let kept = thin(rows.clone(), tiers.len(), MAX_CANDIDATES);
    if kept.len() < n {
        for r in &rows {
            let key = |x: &Row| (x.p, x.k);
            if !kept.iter().any(|k| key(k) == key(r)) {
                dropped.push((
                    describe(desc_of(r, &rams)),
                    format!("thinned to {MAX_CANDIDATES}"),
                ));
            }
        }
    }
    let mut placements: Vec<Placement> = Vec::new();
    let candidates = kept
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let (v, ram, t, ub, op) = desc_of(r, &rams);
            let c = build_builder_candidate(&format!("c{i:02}"), layout, (v, ram), (t, ub, op), w);
            if !placements.contains(&c.placement) {
                placements.push(c.placement);
            }
            c
        })
        .collect();
    Enumeration {
        candidates,
        dropped,
        placements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{CPU, CUDA_RESERVE, GPU, HOST_RESERVE};
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

    // hardware: none
    #[test]
    fn tiers_start_at_top_k_and_end_at_all_experts() {
        let l = fixtures::moe();
        assert_eq!(vram_slot_tiers(&l), [8, 16, 32, 64, 128]);
        // a quarter, half and all of the 120 experts outside VRAM
        assert_eq!(ram_slot_options(&l, 8), [30, 60, 120]);
        // nothing left outside VRAM: the floor is top-k
        assert_eq!(ram_slot_options(&l, 128), [8]);
        assert_eq!(ram_slot_options(&l, 64), [16, 32, 64]);
        // one slot row is one expert of every layer
        let e = 3 * tetraeder_plan::Format::Q4_K.bytes(2048 * 768);
        assert_eq!(slot_row_bytes(&l), 48 * e);
    }

    // hardware: none
    #[test]
    fn enumeration_is_deterministic_valid_and_at_least_top_k() {
        let l = fixtures::moe();
        let en = enumerate_builder(&BUDGET, &l, &WORK);
        assert_eq!(en, enumerate_builder(&BUDGET, &l, &WORK), "deterministic");
        assert!(
            (20..=MAX_CANDIDATES).contains(&en.candidates.len()),
            "{}",
            en.candidates.len()
        );
        let ids: Vec<_> = en.candidates.iter().map(|c| c.id.clone()).collect();
        let want: Vec<_> = (0..ids.len()).map(|i| format!("c{i:02}")).collect();
        assert_eq!(ids, want);

        let (hw, model) = (hw(&BUDGET), l.to_model_desc());
        let reserve = BTreeMap::from([
            (GPU.to_string(), CUDA_RESERVE),
            (CPU.to_string(), HOST_RESERVE),
        ]);
        let ctx = Context {
            hw: &hw,
            model: &model,
            phase: Phase(3),
            reserve: &reserve,
            hints: None,
        };
        let (mut th, mut ub, mut op, mut vr) = (
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        let k = l.n_expert_used;
        for c in &en.candidates {
            assert_eq!(c.engine, EngineKind::Builder);
            let Placement::Builder {
                vram_slots,
                ram_slots,
            } = c.placement
            else {
                panic!("{}: {:?}", c.id, c.placement)
            };
            assert!(vram_slots >= k && ram_slots >= k, "{}", c.id);
            assert!(
                vram_slots <= l.n_expert && ram_slots <= l.n_expert,
                "{}",
                c.id
            );
            let st = &c.plan.replicas[0].stages[0];
            let slots: BTreeMap<_, _> = st
                .expert_cache
                .iter()
                .map(|s| (s.device.as_str(), s.slots))
                .collect();
            assert_eq!(slots, BTreeMap::from([(GPU, vram_slots), (CPU, ram_slots)]));
            let f = footprint(&l, c.placement, c.ubatch(), &WORK);
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
            vr.insert(vram_slots);
        }
        assert_eq!(th, BTreeSet::from([7, 15]));
        assert_eq!(ub, BTreeSet::from([256, 2048]));
        assert_eq!(op, BTreeSet::from([false, true]));
        // 128 slots per layer do not fit in 14 GB; every smaller tier is represented
        assert_eq!(vr, BTreeSet::from([8, 16, 32, 64]));
        assert!(
            en.dropped
                .iter()
                .any(|(d, why)| d.contains("vram_slots=128") && why.starts_with("gpu estimate"))
        );
        assert!(en.dropped.iter().any(|(_, why)| why.contains("thinned")));
    }

    // hardware: none
    #[test]
    fn dense_and_tiny_budgets_give_no_candidates() {
        let en = enumerate_builder(&BUDGET, &fixtures::dense(), &WORK);
        assert!(en.candidates.is_empty() && !en.dropped.is_empty());
        let b = Budget {
            gpu_bytes: 1 << 30,
            ..BUDGET
        };
        let en = enumerate_builder(&b, &fixtures::moe(), &WORK);
        assert!(en.candidates.is_empty() && !en.dropped.is_empty());
    }

    // hardware: none
    #[test]
    fn small_ram_budget_drops_big_ram_tiers() {
        let l = fixtures::moe();
        let b = Budget {
            ram_bytes: 12_000_000_000,
            ..BUDGET
        };
        let en = enumerate_builder(&b, &l, &WORK);
        assert!(!en.candidates.is_empty());
        for c in &en.candidates {
            let f = footprint(&l, c.placement, c.ubatch(), &WORK);
            assert!(f.host <= b.ram_bytes, "{}", c.id);
        }
    }
}

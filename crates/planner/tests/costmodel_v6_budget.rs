//! Cost model 6 (ADR 0092), step 7: the planner's RAM budget under the balloon is measured; the
//! 59.9 GB the 64 GB arm really had rejects the pick that OOMed.
//!
//! Hardware label: none.

mod v6_common;

use proptest::prelude::*;
use tetraeder_plan::{Hints, KvFormat, Phase, segment_plan, segments};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    Placement, Workload, enumerate_cluster, footprint, segment_layout, single_node_runs,
};
use tetraeder_profiler::HardwareGraph;
use v6_common::{layout_of, session};

const LAN: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");
const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

/// The planner's budget of a device: its capacity less the 1% margin (`candidate::Budget`).
fn budget(capacity: u64) -> u64 {
    capacity / 100 * 99
}

fn with_capacity(g: &mut HardwareGraph, node: Option<&str>, gpu: u64, ram: u64) {
    for d in g.devices.iter_mut().filter(|d| d.node.as_deref() == node) {
        if d.id.ends_with("gpu0") {
            d.capacity_bytes = gpu;
        } else if d.id.ends_with("cpu0") {
            d.capacity_bytes = ram;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Rule 5, ADR 0092 section 4: whatever RAM and VRAM the node really has (the balloon leaves
    /// less than the nominal), no placement the single-node pick returns needs more than the
    /// graph's budget, pinned or not.
    #[test]
    fn a_single_node_pick_never_exceeds_the_measured_budget(
        ram_gb in 24u64..=96,
        gpu_gb in 12u64..=48,
        pin in prop::option::of((0u32..=32, 0u32..=128)),
    ) {
        let layout = fixtures::moe_235b();
        let mut g = HardwareGraph::from_toml(LAN).unwrap().node_graph(Some("a"));
        let (gpu, ram) = (gpu_gb * 1_000_000_000, ram_gb * 1_000_000_000);
        with_capacity(&mut g, None, gpu, ram);
        let (runs, _) = single_node_runs(&g, &layout, &W, pin, None).unwrap();
        for r in &runs {
            let f = footprint(
                &layout,
                Placement::Builder { vram_slots: r.vram_slots, ram_slots: r.ram_slots },
                r.ubatch,
                &W,
                KvFormat::default(),
            );
            prop_assert!(f.host <= budget(ram), "v{} r{} needs {} of {}", r.vram_slots, r.ram_slots, f.host, budget(ram));
            prop_assert!(f.gpu <= budget(gpu), "v{} r{} needs {} of {}", r.vram_slots, r.ram_slots, f.gpu, budget(gpu));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// The same for every segment of every plan across nodes: each stage fits the RAM and VRAM
    /// of its own node.
    #[test]
    fn a_cluster_plan_never_exceeds_the_budget_of_a_node(
        ram_a in 40u64..=96,
        ram_b in 40u64..=96,
        gpu_gb in 20u64..=48,
    ) {
        let layout = fixtures::moe_235b();
        let mut hw = HardwareGraph::from_toml(LAN).unwrap();
        let gpu = gpu_gb * 1_000_000_000;
        with_capacity(&mut hw, Some("a"), gpu, ram_a * 1_000_000_000);
        with_capacity(&mut hw, Some("b"), gpu, ram_b * 1_000_000_000);
        let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
        for c in &cands {
            let ub = c.plan.knobs.ubatch.unwrap();
            for seg in segments(&c.plan, 0, &hw).unwrap().iter().filter(|s| !s.stages.is_empty()) {
                let node = seg.node.as_deref().unwrap();
                let st = &segment_plan(&c.plan, 0, seg).replicas[0].stages[0];
                let slots = |id: &str| st.expert_cache.iter().find(|e| e.device == id).map(|e| e.slots);
                let (Some(vram_slots), Some(ram_slots)) = (slots("gpu0"), slots("cpu0")) else { continue };
                let sub = segment_layout(&layout, seg.layers, seg.embed, seg.head);
                let f = footprint(&sub, Placement::Builder { vram_slots, ram_slots }, ub, &W, KvFormat::default());
                let ram = if node == "a" { ram_a } else { ram_b } * 1_000_000_000;
                prop_assert!(f.host <= budget(ram), "{} on {node}: {} of {}", c.id, f.host, budget(ram));
                prop_assert!(f.gpu <= budget(gpu), "{} on {node}: {} of {}", c.id, f.gpu, budget(gpu));
            }
        }
    }
}

/// ADR 0092 section 4: the measured RAM budget of the 64 GB arm (59.9 GB) rejects the pick that
/// OOMed and accepts the one that ran.
#[test]
fn a_59_9_gb_budget_rejects_16_56_and_accepts_16_52() {
    let Some(dir) = session() else { return };
    let mut hw = HardwareGraph::from_toml(
        &std::fs::read_to_string(dir.join("predictions/hw-D.toml")).unwrap(),
    )
    .unwrap();
    hw.devices
        .iter_mut()
        .find(|d| d.id == "cpu0")
        .unwrap()
        .capacity_bytes = 59_900_000_000;
    let layout = layout_of(&dir);
    let w = Workload {
        prompt: 2048,
        r#gen: 128,
    };
    let (rejected, why) = single_node_runs(&hw, &layout, &w, Some((16, 56)), None).unwrap();
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(!why.is_empty());
    let (accepted, _) = single_node_runs(&hw, &layout, &w, Some((16, 52)), None).unwrap();
    assert!(!accepted.is_empty());
}

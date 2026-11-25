//! The pinned plan of the Case S dry run (`tests/fixtures/phase6/tiny-case-s-plan.toml`, used by
//! `scripts/phase6_exit_s.sh` with TINY=1): the eight-layer tiny MoE cut 4 and 4 across nodes `a`
//! (the leader) and `b`, 4 compute-tier and 4 RAM-tier slots per layer on each, which cover its 8
//! experts. A CI runner has no GPU, so the planner prices no builder placement for the tiny
//! layout and the plan is written by hand; this test keeps it a plan the rest of the code reads.
//!
//! Hardware label: none.

use tetraeder_plan::{Plan, UnitKind, segments};
use tetraeder_profiler::HardwareGraph;

const PLAN: &str = include_str!("../../../tests/fixtures/phase6/tiny-case-s-plan.toml");
const HW: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");
/// Experts per layer of `tests/fixtures/tiny-qwen3moe-l8-f16.gguf`.
const N_EXPERT: u32 = 8;

// hardware: none
#[test]
fn the_pinned_tiny_plan_is_a_head_led_covering_two_stage_plan() {
    let plan = Plan::from_toml(PLAN).unwrap();
    assert_eq!(Plan::from_toml(&plan.to_toml().unwrap()).unwrap(), plan);
    let st = &plan.replicas[0].stages;
    assert_eq!(st.len(), 2);
    // One stage per node, the leader first, the layers partitioned.
    assert!(st[0].devices.iter().all(|d| d.starts_with("a/")));
    assert!(st[1].devices.iter().all(|d| d.starts_with("b/")));
    assert_eq!((st[0].layers.0, st[0].layers.1), (0, 4));
    assert_eq!((st[1].layers.0, st[1].layers.1), (4, 8));
    // Covering: VRAM plus RAM slots hold every expert of each layer, on both nodes.
    for s in st {
        let slots: u32 = s.expert_cache.iter().map(|c| c.slots).sum();
        assert_eq!(slots, N_EXPERT, "{:?}", s.expert_cache);
    }
    // The embedding is on the leader, the output head on the last stage.
    let kinds = |i: usize| -> Vec<UnitKind> {
        st[i]
            .assign
            .iter()
            .flat_map(|a| a.units.kinds.clone())
            .collect()
    };
    assert!(kinds(0).contains(&UnitKind::Embed) && !kinds(0).contains(&UnitKind::Head));
    assert!(kinds(1).contains(&UnitKind::Head) && !kinds(1).contains(&UnitKind::Embed));
}

// hardware: none
#[test]
fn the_pinned_tiny_plan_cuts_into_one_segment_per_node_on_a_cluster_graph() {
    let plan = Plan::from_toml(PLAN).unwrap();
    let hw = HardwareGraph::from_toml(HW).unwrap();
    let segs = segments(&plan, 0, &hw).unwrap();
    let got: Vec<(Option<&str>, u32, u32, bool, bool)> = segs
        .iter()
        .map(|s| (s.node.as_deref(), s.layers.0, s.layers.1, s.embed, s.head))
        .collect();
    assert_eq!(
        got,
        vec![
            (Some("a"), 0, 4, true, false),
            (Some("b"), 4, 8, false, true)
        ]
    );
}

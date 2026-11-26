//! Tree node `planner-footprint-validator-gap`: the cluster planner and the plan validator agree
//! on memory, so the search never assembles a plan the validator rejects for it.
//!
//! The cause: the validator sizes every unit of a kind at the whole model's mean (one expert size
//! for all layers), the footprint at each layer's bytes. The 235B's Q6_K expert layers sit at
//! both ends, so a short middle segment of light layers counts more slot-cache bytes in the
//! validator than in the footprint, by more than the footprint's 1% margin on a 24 GiB card.
//!
//! Hardware label: none (simulated hardware fixtures only).

use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_plan::{Context, Hints, Issue, KvFormat, Phase, segment_plan, segments, validate};
use tetraeder_planner::cluster::stage_memory;
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{Placement, Workload, enumerate_cluster, footprint, segment_layout};
use tetraeder_profiler::HardwareGraph;

const LAN: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");

/// The committed cluster fixtures (AGENTS rule 4), healthy and faulty, then the planner's
/// single-node hardware fixtures.
const FIXTURES: [&str; 9] = [
    LAN,
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_samehost.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan_slowlink.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan_slowlink.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan_degraded.toml"),
    include_str!("../fixtures/rtx3090_prior.toml"),
    include_str!("../fixtures/cpu_only.toml"),
];

const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

/// `n` copies of the first node of `fixture`, named `n0`.., joined pairwise by the LAN
/// fixture's network link.
fn n_nodes(fixture: &str, n: usize) -> HardwareGraph {
    let lan = HardwareGraph::from_toml(LAN).unwrap();
    let base = HardwareGraph::from_toml(fixture).unwrap();
    let node = base.node_graph(base.nodes()[0]);
    let net = lan.links.iter().find(|l| l.id == "net-a-b").unwrap();
    let nodes: Vec<(String, HardwareGraph)> =
        (0..n).map(|i| (format!("n{i}"), node.clone())).collect();
    let mut links = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            links.push(tetraeder_profiler::Link {
                id: format!("net-n{i}-n{j}"),
                a: format!("n{i}/cpu0"),
                b: format!("n{j}/cpu0"),
                ..net.clone()
            });
        }
    }
    HardwareGraph::merge_cluster(&nodes, links, vec![])
}

/// Every GPU of `g` at `gpu` bytes and every CPU at `ram` bytes (`None`: as the fixture says).
fn with_capacity(g: &mut HardwareGraph, gpu: Option<u64>, ram: Option<u64>) {
    for d in &mut g.devices {
        if d.id.ends_with("gpu0")
            && let Some(b) = gpu
        {
            d.capacity_bytes = b;
        } else if d.id.ends_with("cpu0")
            && let Some(b) = ram
        {
            d.capacity_bytes = b;
        }
    }
}

/// The search's prunes that are the validator's memory verdict on an assembled plan.
fn memory_rejections(pruned: &[(String, String)]) -> Vec<&(String, String)> {
    pruned
        .iter()
        .filter(|(_, why)| why.starts_with("invalid") && why.contains("MemoryExceeded"))
        .collect()
}

/// Without zero entries.
fn nonzero(m: BTreeMap<String, u64>) -> BTreeMap<String, u64> {
    m.into_iter().filter(|(_, b)| *b > 0).collect()
}

/// Rule 5 on every plan the search builds for the 235B on `hw`: none is pruned for memory; for
/// every candidate the planner's gate ([`stage_memory`] summed over the stages) counts exactly
/// what the validator counts on each device, and each builder segment's footprint is within its
/// node's capacity on the GPU and the host (the footprint carries the reserves the validator
/// subtracts). Returns the number of candidates.
fn check_candidates(hw: &HardwareGraph) -> Result<usize, TestCaseError> {
    let layout = fixtures::moe_235b();
    let desc = layout.to_model_desc();
    let (cands, pruned) = enumerate_cluster(hw, &layout, &Hints::default(), Phase(6));
    let bad = memory_rejections(&pruned);
    prop_assert!(
        bad.is_empty(),
        "{} plans rejected for memory, first {:?}",
        bad.len(),
        bad.first()
    );
    let reserve = BTreeMap::new();
    let ctx = Context {
        hw,
        model: &desc,
        phase: Phase(6),
        reserve: &reserve,
        hints: None,
    };
    for c in &cands {
        let validator = validate(&c.plan, &ctx).unwrap().memory_per_device();
        let mut gate: BTreeMap<String, u64> = BTreeMap::new();
        for st in &c.plan.replicas[0].stages {
            for (d, b) in stage_memory(&desc, st) {
                *gate.entry(d).or_default() += b;
            }
        }
        prop_assert_eq!(nonzero(gate), nonzero(validator), "{}", c.id);
    }
    for c in &cands {
        let ub = c.plan.knobs.ubatch.unwrap();
        for seg in segments(&c.plan, 0, hw)
            .unwrap()
            .iter()
            .filter(|s| !s.stages.is_empty())
        {
            let node = seg.node.as_deref().unwrap();
            let st = &segment_plan(&c.plan, 0, seg).replicas[0].stages[0];
            let slots = |id: &str| {
                st.expert_cache
                    .iter()
                    .find(|e| e.device == id)
                    .map(|e| e.slots)
            };
            let (Some(vram_slots), Some(ram_slots)) = (slots("gpu0"), slots("cpu0")) else {
                continue;
            };
            let sub = segment_layout(&layout, seg.layers, seg.embed, seg.head);
            let f = footprint(
                &sub,
                Placement::Builder {
                    vram_slots,
                    ram_slots,
                },
                ub,
                &W,
                KvFormat::default(),
            );
            let cap = |id: &str| hw.device(&format!("{node}/{id}")).unwrap().capacity_bytes;
            prop_assert!(
                f.gpu <= cap("gpu0"),
                "{} on {node}: gpu {} of {}",
                c.id,
                f.gpu,
                cap("gpu0")
            );
            prop_assert!(
                f.host <= cap("cpu0"),
                "{} on {node}: host {} of {}",
                c.id,
                f.host,
                cap("cpu0")
            );
        }
    }
    Ok(cands.len())
}

/// The defect: seven 24 GiB nodes, every segment fits its node by the footprint, yet the
/// validator rejected the assembled plans with MemoryExceeded.
#[test]
fn seven_24_gib_nodes_build_no_plan_the_validator_rejects() {
    let n = check_candidates(&n_nodes(LAN, 7)).unwrap();
    assert!(n > 0, "no candidate on seven 24 GiB nodes");
}

/// The filter of `memory_rejections` matches the validator's memory issue as the search prints
/// it (`invalid: {:?}`).
#[test]
fn the_filter_matches_the_validators_memory_issue() {
    let i = Issue::MemoryExceeded {
        device: "gpu0".into(),
        used: 2,
        capacity: 1,
        reserve: 0,
    };
    assert!(format!("{i:?}").contains("MemoryExceeded"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// AGENTS rules 4 and 5: on every fixture, as committed or with its first node copied to up
    /// to seven nodes, at any GPU and RAM size, the planner emits only plans whose memory the
    /// validator accepts and whose footprint fits each node.
    #[test]
    fn no_fixture_gets_a_plan_the_validator_rejects_for_memory(
        fixture in 0..FIXTURES.len(),
        copies in prop::option::weighted(0.75, 2usize..=7),
        gpu_gb in prop::option::of(16u64..=48),
        ram_gb in prop::option::of(40u64..=128),
    ) {
        let mut hw = match copies {
            Some(n) => n_nodes(FIXTURES[fixture], n),
            None => HardwareGraph::from_toml(FIXTURES[fixture]).unwrap(),
        };
        with_capacity(&mut hw, gpu_gb.map(|g| g * 1_000_000_000), ram_gb.map(|r| r * 1_000_000_000));
        check_candidates(&hw)?;
    }
}

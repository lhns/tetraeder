//! Cluster planner tests (ADR 0091 step 2; AGENTS.md rules 4 and 5): on every simulated cluster
//! fixture and fault, every candidate validates, stays inside every device's and node's memory
//! (the RAM budget included), computes every unit exactly once, is cut into segments that cover
//! the layers in order, and computes on every node; single-node plans predict as before.
//!
//! Hardware label: none (simulated hardware fixtures only).

use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use tetraeder_costmodel::cluster::predict_cluster;
use tetraeder_plan::{
    Context, Forbid, Hints, Issue, Phase, Plan, Segment, UnitId, WorkloadHints, segment_plan,
    segments, validate,
};
use tetraeder_planner::cluster::{candidate_segments, segment_layout};
use tetraeder_planner::layout::{ModelLayout, fixtures};
use tetraeder_planner::serve::{predict_plan_tier, uniform_tier};
use tetraeder_planner::{
    ClusterCandidate, Workload, cluster_features, enumerate_cluster, enumerate_cluster_covering,
    predict_cluster_plan, single_node_references, single_node_runs,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const FIXTURES: [(&str, &str); 7] = [
    (
        "lan",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"),
    ),
    (
        "wan",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"),
    ),
    (
        "samehost",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_samehost.toml"),
    ),
    (
        "three_mixed",
        include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml"),
    ),
    (
        "lan_slowlink",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan_slowlink.toml"),
    ),
    (
        "wan_slowlink",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan_slowlink.toml"),
    ),
    (
        "lan_degraded",
        include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan_degraded.toml"),
    ),
];

fn graphs() -> Vec<(&'static str, HardwareGraph)> {
    FIXTURES
        .iter()
        .map(|(n, t)| (*n, HardwareGraph::from_toml(t).unwrap()))
        .collect()
}

const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

fn reserve(hw: &HardwareGraph) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let r = match d.kind {
                DeviceKind::Gpu => 600 << 20,
                DeviceKind::Cpu => (2 << 30) + (256 << 20),
                DeviceKind::Disk => 0,
            };
            (d.id.clone(), r)
        })
        .collect()
}

/// Units a segment holds.
fn units_of(layout: &ModelLayout, s: &Segment) -> Vec<UnitId> {
    layout
        .to_model_desc()
        .units()
        .filter(|u| match u {
            UnitId::Embed => s.embed,
            UnitId::Head => s.head,
            _ => u.layer().is_some_and(|l| s.layers.contains(l)),
        })
        .collect()
}

/// Everything a candidate promises.
fn check(
    c: &ClusterCandidate,
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
    nodes: &BTreeSet<String>,
    w: &Workload,
    what: &str,
) {
    let desc = layout.to_model_desc();
    let rs = reserve(hw);
    let ctx = Context {
        hw,
        model: &desc,
        phase: Phase(6),
        reserve: &rs,
        hints: Some(hints),
    };
    let e = validate(&c.plan, &ctx).unwrap_or_else(|i| panic!("{what} {}: {i:#?}", c.id));

    // No device or node budget exceeded, the RAM budget included.
    let used = e.memory_per_device();
    let mut per_node: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for (dev, bytes) in &used {
        let d = hw.device(dev).unwrap();
        let limit = d.capacity_bytes.saturating_sub(rs[dev]);
        assert!(*bytes <= limit, "{what} {}: {dev} {bytes} > {limit}", c.id);
        if d.kind != DeviceKind::Disk {
            let n = per_node.entry(d.node.as_deref().unwrap()).or_default();
            n.0 += bytes;
            n.1 += d.capacity_bytes;
        }
    }
    for (n, (u, cap)) in &per_node {
        assert!(u <= cap, "{what} {}: node {n} holds {u} of {cap}", c.id);
    }

    // Every unit is computed exactly once per replica, on one shard.
    let mut count: BTreeMap<UnitId, u32> = BTreeMap::new();
    for k in e.shards.keys() {
        assert_eq!(k.of, 1);
        *count.entry(k.unit).or_default() += 1;
    }
    assert_eq!(count.len(), desc.units().count(), "{what} {}", c.id);
    assert!(count.values().all(|&n| n == 1), "{what} {}", c.id);

    // Stages are single-node and the segments cover [0, L) in order.
    let segs = segments(&c.plan, 0, hw).unwrap();
    let mut next = 0;
    for s in &segs {
        assert_eq!(s.layers.0, next, "{what} {}", c.id);
        next = s.layers.1;
    }
    assert_eq!(next, layout.n_layer);
    assert!(segs[0].embed && segs.iter().filter(|s| s.embed).count() == 1);
    assert_eq!(segs.iter().filter(|s| s.head).count(), 1);
    // Every unit lives in exactly one segment, and the segment plans keep the stages.
    let mut held: BTreeMap<UnitId, u32> = BTreeMap::new();
    for s in &segs {
        for u in units_of(layout, s) {
            *held.entry(u).or_default() += 1;
        }
    }
    assert_eq!(held.len(), count.len());
    assert!(held.values().all(|&n| n == 1));
    let stages: usize = segs
        .iter()
        .map(|s| {
            segment_plan(&c.plan, 0, s).replicas[0].stages.len() - usize::from(s.stages.is_empty())
        })
        .sum();
    assert_eq!(stages, c.plan.replicas[0].stages.len());

    // Case S: every usable node computes layers; a head alone does not count.
    let computing: BTreeSet<String> = segs
        .iter()
        .filter(|s| !s.layers.is_empty())
        .map(|s| s.node.clone().unwrap())
        .collect();
    assert_eq!(&computing, nodes, "{what} {}: spans every node", c.id);

    // The stored prediction is the plan's, and the plan survives TOML.
    assert_eq!(
        predict_cluster_plan(hw, layout, &c.plan, w).unwrap(),
        c.prediction
    );
    assert_eq!(Plan::from_toml(&c.plan.to_toml().unwrap()).unwrap(), c.plan);
}

fn all_nodes(hw: &HardwareGraph) -> BTreeSet<String> {
    hw.nodes().iter().map(|n| n.unwrap().to_string()).collect()
}

fn sorted_by_prediction(cands: &[ClusterCandidate]) -> bool {
    cands
        .windows(2)
        .all(|w| w[0].prediction.request_s <= w[1].prediction.request_s)
}

#[test]
fn every_fixture_has_valid_spanning_candidates_for_the_30b() {
    let layout = fixtures::moe();
    let hints = Hints::default();
    for (name, hw) in graphs() {
        let (cands, pruned) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
        assert!(!cands.is_empty(), "{name}: {pruned:#?}");
        assert!(sorted_by_prediction(&cands), "{name}");
        for (i, c) in cands.iter().enumerate() {
            assert_eq!(c.id, format!("c{i:02}"));
            check(c, &hw, &layout, &hints, &all_nodes(&hw), &W, name);
        }
    }
}

#[test]
fn the_235b_spans_both_nodes_on_every_two_node_fixture() {
    let layout = fixtures::moe_235b();
    // About 142 GB, more than a node's 24 GB of VRAM and 64 GB of RAM.
    let total = layout.total_bytes();
    assert!((135e9..150e9).contains(&(total as f64)), "{total}");
    let hints = Hints::default();
    for (name, hw) in graphs() {
        let (cands, pruned) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
        assert!(!cands.is_empty(), "{name}: {pruned:#?}");
        for c in cands.iter().take(6) {
            check(c, &hw, &layout, &hints, &all_nodes(&hw), &W, name);
        }
        // The single node is a reference, never a candidate.
        for r in single_node_references(&hw, &layout, &hints) {
            assert!(r.id.starts_with("ref-"));
            assert_eq!(segments(&r.plan, 0, &hw).unwrap().len(), 1);
            assert!(!cands.iter().any(|c| c.plan == r.plan));
        }
    }
}

/// Judged Case S candidates (ADR 0091 History 2026-11-25) on every fixture: the same properties
/// as any candidate, plus every stage's VRAM and RAM slots hold all experts of its layers, and
/// every covering candidate is also one of the unrestricted search.
#[test]
fn covering_candidates_hold_the_properties_and_cover_every_expert() {
    let layout = fixtures::moe_235b();
    let hints = Hints::default();
    let mut with_candidates = 0;
    for (name, hw) in graphs() {
        let (cands, pruned) = enumerate_cluster_covering(&hw, &layout, &hints, Phase(6));
        if cands.is_empty() {
            // Nothing covers: the reason says what is missing.
            assert!(
                pruned
                    .iter()
                    .any(|(_, w)| w.contains("no covering placement")),
                "{name}: {pruned:#?}"
            );
            continue;
        }
        with_candidates += 1;
        let (all, _) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
        for c in cands.iter().take(6) {
            check(c, &hw, &layout, &hints, &all_nodes(&hw), &W, name);
            for st in &c.plan.replicas[0].stages {
                let slots: u32 = st.expert_cache.iter().map(|e| e.slots).sum();
                assert!(slots >= layout.n_expert, "{name} {}: {slots} slots", c.id);
            }
            assert!(all.iter().any(|a| a.plan == c.plan), "{name} {}", c.id);
        }
    }
    assert!(with_candidates > 0, "no fixture has a covering candidate");
}

#[test]
fn forbid_node_removes_the_node_from_every_candidate() {
    let layout = fixtures::moe();
    let (_, hw) = graphs().remove(3);
    let hints = Hints {
        forbid: vec![Forbid {
            units: None,
            device: None,
            node: Some("c".into()),
            tier: None,
        }],
        ..Default::default()
    };
    let (cands, pruned) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
    assert!(!cands.is_empty());
    assert!(pruned.iter().any(|p| p.1.contains("forbid node")));
    let nodes: BTreeSet<String> = ["a".to_string(), "b".to_string()].into();
    for c in &cands {
        check(c, &hw, &layout, &hints, &nodes, &W, "three_mixed without c");
        assert!(
            candidate_segments(&hw, c)
                .iter()
                .all(|s| s.node.as_deref() != Some("c"))
        );
    }
    // Forbidding one of two nodes leaves no spanning plan, only the reference.
    let (_, lan) = graphs().remove(0);
    let one = Hints {
        forbid: vec![Forbid {
            node: Some("b".into()),
            ..hints.forbid[0].clone()
        }],
        ..Default::default()
    };
    let (cands, pruned) = enumerate_cluster(&lan, &layout, &one, Phase(6));
    assert!(cands.is_empty());
    assert!(pruned.iter().any(|p| p.1.contains("reference only")));
}

#[test]
fn before_phase_six_nothing_spans_nodes() {
    let layout = fixtures::moe();
    let (_, hw) = graphs().remove(0);
    let (cands, pruned) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(5));
    assert!(cands.is_empty());
    assert!(
        pruned.iter().any(|p| p.1.contains("multi_node")),
        "{:?}",
        pruned.first()
    );
}

#[test]
fn a_node_whose_gpu_is_too_small_runs_its_segment_on_the_cpu() {
    let layout = fixtures::moe();
    let (_, mut hw) = graphs().remove(0);
    for d in &mut hw.devices {
        if d.id == "b/gpu0" {
            d.capacity_bytes = 400 << 20; // under the CUDA reserve: no GPU placement fits
        }
    }
    let hints = Hints::default();
    let (cands, pruned) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
    assert!(!cands.is_empty(), "{pruned:#?}");
    for c in &cands {
        check(c, &hw, &layout, &hints, &all_nodes(&hw), &W, "small gpu");
        let on_b: Vec<_> = c.plan.replicas[0]
            .stages
            .iter()
            .filter(|s| s.devices.iter().all(|d| d.starts_with("b/")))
            .collect();
        assert!(on_b.iter().all(|s| s.devices == ["b/cpu0"]), "{}", c.id);
        assert!(!on_b.is_empty());
    }
}

#[test]
fn a_dense_model_is_refused_with_a_reason() {
    let (_, hw) = graphs().remove(0);
    let (cands, pruned) = enumerate_cluster(&hw, &fixtures::dense(), &Hints::default(), Phase(6));
    assert!(cands.is_empty() && pruned.iter().any(|p| p.1.contains("MoE")));
}

/// One node's plan predicts through the cluster path exactly as the single-node path does with
/// the same placement, knobs and routing prior (ADR 0091, C7).
#[test]
fn a_reference_predicts_as_the_single_node_path_does() {
    let layout = fixtures::moe();
    let hints = Hints::default();
    for (name, hw) in graphs() {
        for r in single_node_references(&hw, &layout, &hints) {
            let f = cluster_features(&hw, &layout, &r.plan).unwrap();
            assert_eq!(f.segments.len(), 1);
            let seg = segments(&r.plan, 0, &hw).unwrap().remove(0);
            let sp = segment_plan(&r.plan, 0, &seg);
            let st = &sp.replicas[0].stages[0];
            if st.expert_cache.is_empty() {
                continue; // a CPU-only node: today's CPU placement, covered by the features test
            }
            let slots = |id: &str| {
                st.expert_cache
                    .iter()
                    .find(|c| c.device == id)
                    .unwrap()
                    .slots
            };
            let (v, ram) = (slots("gpu0"), slots("cpu0"));
            // The same plan the single-node serve path builds, predicted with the same tier.
            let node = hw.node_graph(seg.node.as_deref());
            let one = tetraeder_planner::build_builder_candidate(
                "ref",
                &layout,
                (v, ram),
                (12, r.plan.knobs.ubatch.unwrap(), true),
                &W,
            );
            let want = predict_plan_tier(
                &node,
                &layout,
                &one.plan,
                &W,
                Some(&uniform_tier(&layout, &[(v, ram)])),
            )
            .unwrap();
            assert_eq!(r.prediction.segments, vec![want], "{name}");
            let c = predict_cluster(&hw, &f, W.prompt, W.r#gen).unwrap();
            assert_eq!(c, r.prediction, "{name}");
        }
    }
}

/// The single-node prediction tool (`node-predict`) and the cluster reference price the same plan
/// bit for bit, pinned to the reference's pair and unpinned (same pick). Pins the diagnosis in
/// `docs/notes/stream-predict-diagnosis.md`: the two numbers differ by the hardware graph only.
#[test]
fn a_reference_equals_its_node_run() {
    let layout = fixtures::moe_235b();
    let mut checked = 0;
    for (name, hw) in graphs() {
        for r in single_node_references(&hw, &layout, &Hints::default()) {
            let seg = segments(&r.plan, 0, &hw).unwrap().remove(0);
            let sp = segment_plan(&r.plan, 0, &seg);
            let st = &sp.replicas[0].stages[0];
            if st.expert_cache.is_empty() {
                continue; // a CPU-only node: node_run needs a GPU
            }
            let slots = |id: &str| {
                st.expert_cache
                    .iter()
                    .find(|c| c.device == id)
                    .unwrap()
                    .slots
            };
            let (v, ram, ub) = (slots("gpu0"), slots("cpu0"), r.plan.knobs.ubatch.unwrap());
            let one = hw.node_graph(seg.node.as_deref());
            if one.device("nvme0").is_none() {
                continue; // node_run needs the builder disk
            }
            let (pinned, _) = single_node_runs(&one, &layout, &W, Some((v, ram)), None).unwrap();
            let run = pinned.iter().find(|x| x.ubatch == ub).unwrap();
            let want = &r.prediction;
            assert_eq!(run.pp_tps.to_bits(), want.pp_tps.to_bits(), "{name}");
            assert_eq!(run.tg_tps.to_bits(), want.tg_tps.to_bits(), "{name}");
            assert_eq!(run.request_s.to_bits(), want.request_s.to_bits(), "{name}");
            let (all, _) = single_node_runs(&one, &layout, &W, None, None).unwrap();
            assert_eq!(
                all[0].request_s.to_bits(),
                want.request_s.to_bits(),
                "{name}: same pick"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no reference had a GPU and a disk");
}

#[test]
fn a_whole_model_segment_layout_is_the_layout() {
    let l = fixtures::moe();
    assert_eq!(
        segment_layout(&l, tetraeder_plan::Span(0, l.n_layer), true, true),
        l
    );
    let part = segment_layout(&l, tetraeder_plan::Span(10, 20), false, false);
    assert_eq!(
        (part.n_layer, part.embed_bytes, part.output_bytes),
        (10, 0, 0)
    );
    assert_eq!(part.layers, l.layers[10..20]);
}

#[test]
fn faults_move_the_prediction_the_right_way() {
    let layout = fixtures::moe_235b();
    let hints = Hints::default();
    let best = |i: usize| {
        let (_, hw) = graphs().remove(i);
        enumerate_cluster(&hw, &layout, &hints, Phase(6))
            .0
            .remove(0)
    };
    // The same nodes, a ten times slower link: never faster.
    let (lan, slow) = (best(0), best(4));
    assert!(slow.prediction.tg_tps <= lan.prediction.tg_tps);
    assert!(slow.prediction.pp_tps <= lan.prediction.pp_tps);
    let (wan, wan_slow) = (best(1), best(5));
    assert!(wan_slow.prediction.tg_tps <= wan.prediction.tg_tps);
    // A node at reduced capacity: never faster than the full nodes on the same link.
    let degraded = best(6);
    assert!(degraded.prediction.request_s >= lan.prediction.request_s);
    // Faster links, faster plans.
    let same = best(2);
    assert!(same.prediction.tg_tps >= lan.prediction.tg_tps);
    assert!(lan.prediction.tg_tps >= wan.prediction.tg_tps);
}

fn arb_hints() -> impl Strategy<Value = Hints> {
    (
        256u32..8192,
        prop::option::of(prop::sample::select(vec!["a", "b", "c"])),
    )
        .prop_map(|(prompt, forbidden)| Hints {
            workload: Some(WorkloadHints {
                typical_prompt: Some(prompt),
                ..Default::default()
            }),
            forbid: forbidden
                .into_iter()
                .map(|n| Forbid {
                    units: None,
                    device: None,
                    node: Some(n.into()),
                    tier: None,
                })
                .collect(),
            ..Default::default()
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Rule 5 over random workloads and node trust rules on every fixture.
    #[test]
    fn candidates_hold_their_properties_under_random_hints(which in 0usize..7, hints in arb_hints()) {
        let layout = fixtures::moe();
        let (name, hw) = graphs().remove(which);
        let usable: BTreeSet<String> = all_nodes(&hw)
            .into_iter()
            .filter(|n| !hints.forbid.iter().any(|f| f.node.as_deref() == Some(n)))
            .collect();
        let (cands, _) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
        if usable.len() < 2 {
            prop_assert!(cands.is_empty());
        } else {
            prop_assert!(!cands.is_empty());
        }
        for c in &cands {
            // The prompt hint is the workload the candidates are priced for.
            let w = Workload { prompt: hints.workload.as_ref().unwrap().typical_prompt.unwrap(), r#gen: 128 };
            check(c, &hw, &layout, &hints, &usable, &w, name);
        }
    }
}

#[test]
fn an_invalid_cluster_plan_is_caught() {
    // The validator, not the enumerator, is the safety net: a plan whose stage spans nodes.
    let layout = fixtures::moe();
    let (_, hw) = graphs().remove(0);
    let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    let mut plan = cands[0].plan.clone();
    plan.replicas[0].stages[0].devices.push("b/gpu0".into());
    let desc = layout.to_model_desc();
    let rs = reserve(&hw);
    let ctx = Context {
        hw: &hw,
        model: &desc,
        phase: Phase(6),
        reserve: &rs,
        hints: None,
    };
    let issues = validate(&plan, &ctx).unwrap_err();
    assert!(
        issues
            .iter()
            .any(|i| matches!(i, Issue::StageSpansNodes { .. }))
    );
    assert!(segments(&plan, 0, &hw).is_err());
}

/// The planner's first prediction for the exit test (ADR 0091 "Expected numbers"): pp2048 and
/// tg128 of the best spanning plan of the 235B on each link, and of one node alone. Printed with
/// `--nocapture`; the assertions are the relations the ADR reads off the table.
#[test]
fn the_235b_exit_fixture_prediction() {
    let layout = fixtures::moe_235b();
    let hints = Hints::default();
    let mut rows = Vec::new();
    for (i, name) in [(2usize, "samehost"), (0, "lan"), (1, "wan")] {
        let (_, hw) = graphs().remove(i);
        let (cands, _) = enumerate_cluster(&hw, &layout, &hints, Phase(6));
        let best = &cands[0];
        let t = |p: &tetraeder_costmodel::cluster::ClusterPrediction| {
            f64::from(W.prompt) / p.pp_tps + f64::from(W.r#gen) / p.tg_tps
        };
        println!(
            "{name:9} spanning {}: pp {:.1} tg {:.2} tok/s, T {:.1} s, hops {:.0} ms/token",
            best.notes[0],
            best.prediction.pp_tps,
            best.prediction.tg_tps,
            t(&best.prediction),
            best.prediction.hop_s_per_token * 1e3
        );
        let slots = |p: &Plan| {
            p.replicas[0]
                .stages
                .iter()
                .map(|s| {
                    let n = |i: usize| s.expert_cache.get(i).map_or(0, |c| c.slots);
                    format!(
                        "layers {}..{} vram {} ram {}",
                        s.layers.0,
                        s.layers.1,
                        n(0),
                        n(1)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        };
        println!("{name:9} slots: {}", slots(&best.plan));
        let r = single_node_references(&hw, &layout, &hints).remove(0);
        println!("{name:9} reference slots: {}", slots(&r.plan));
        println!(
            "{name:9} one node {}: pp {:.1} tg {:.2} tok/s, T {:.1} s",
            r.id,
            r.prediction.pp_tps,
            r.prediction.tg_tps,
            t(&r.prediction)
        );
        rows.push((best.prediction.clone(), r.prediction));
    }
    // A faster link never loses decode or prefill rate; the single node does not depend on it.
    assert!(rows[0].0.tg_tps >= rows[1].0.tg_tps && rows[1].0.tg_tps >= rows[2].0.tg_tps);
    assert!(rows[0].0.pp_tps >= rows[1].0.pp_tps && rows[1].0.pp_tps >= rows[2].0.pp_tps);
    assert_eq!(rows[0].1.tg_tps, rows[2].1.tg_tps);
}

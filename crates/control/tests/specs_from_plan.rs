//! The replica specs derived from a cluster plan (ADR 0091, step 6; rule 5): over the planner's
//! candidates on every cluster fixture, each replica's segments cover the layers exactly once,
//! the embedding and the head once, sit on configured nodes, and the leader is the node of the
//! first segment. A plan that names a node the cluster does not have is refused.
//!
//! Hardware label: none (simulated hardware fixtures only).

use tetraeder_control::ClusterConfig;
use tetraeder_control::head::specs_from_plan;
use tetraeder_plan::{Hints, Phase, Plan, segment_shape};
use tetraeder_planner::cluster::enumerate_cluster;
use tetraeder_planner::layout::fixtures;
use tetraeder_profiler::HardwareGraph;

const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

const FIXTURES: [&str; 4] = [
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan_degraded.toml"),
];

fn config(hw: &HardwareGraph) -> ClusterConfig {
    let nodes: Vec<&str> = hw.nodes().into_iter().flatten().collect();
    let mut t = format!("cluster = \"fx\"\nhead = \"{}\"\n", nodes[0]);
    for (i, n) in nodes.iter().enumerate() {
        t += &format!(
            "[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{}\"\nanchor = \"{n}.pem\"\n",
            9000 + i
        );
    }
    ClusterConfig::parse(&t).unwrap()
}

// hardware: none
#[test]
fn derived_replicas_cover_the_model_once_on_configured_nodes() {
    let layout = fixtures::moe();
    let mut checked = 0;
    for text in FIXTURES {
        let hw = HardwareGraph::from_toml(text).unwrap();
        let cfg = config(&hw);
        let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
        assert!(!cands.is_empty(), "the fixture has candidates");
        for c in cands {
            let specs = specs_from_plan(&c.plan, &hw, &cfg, SHA, &[]).unwrap();
            assert_eq!(specs.len(), c.plan.replicas.len());
            for (r, leader, spec) in specs {
                assert!(
                    cfg.node(&leader).is_some(),
                    "the leader is a configured node"
                );
                assert_eq!(spec.model_sha256, SHA);
                if spec.segments.is_empty() {
                    // Whole on its leader: the one segment's plan covers every layer.
                    let sub = Plan::from_toml(&spec.plan).unwrap();
                    let s = segment_shape(&sub, layout.n_layer).unwrap();
                    assert_eq!((s.layers.0, s.layers.1), (0, layout.n_layer));
                    assert!(s.embed && s.head);
                    continue;
                }
                assert_eq!(
                    spec.segments[0].peer.id, leader,
                    "replica {r}: leader first"
                );
                // The leader's engine reads the plan's knobs (prefill micro-batches, ubatch): the
                // replica's plan travels with a split replica too (rehearsal 2: it was empty, so
                // the leader warned "prefill is not pipelined" and ran its defaults).
                let sent = Plan::from_toml(&spec.plan)
                    .unwrap_or_else(|e| panic!("replica {r}: the plan is not sent: {e}"));
                assert_eq!(
                    sent.knobs.prefill_micro_batches,
                    c.plan.knobs.prefill_micro_batches
                );
                assert_eq!(sent.knobs.ubatch, c.plan.knobs.ubatch);
                let (mut next, mut embeds, mut heads) = (0u32, 0, 0);
                for seg in &spec.segments {
                    assert!(
                        cfg.node(&seg.peer.id).is_some(),
                        "a segment on a configured node"
                    );
                    let sub = Plan::from_toml(&seg.plan).unwrap();
                    let s = segment_shape(&sub, layout.n_layer).unwrap();
                    if s.layers.0 != s.layers.1 {
                        assert_eq!(s.layers.0, next, "layers are contiguous and not repeated");
                        next = s.layers.1;
                    }
                    embeds += u32::from(s.embed);
                    heads += u32::from(s.head);
                    // A segment's devices carry no node prefix: its worker sees one node.
                    for st in &sub.replicas[0].stages {
                        assert!(
                            st.devices.iter().all(|d| !d.contains('/')),
                            "{:?}",
                            st.devices
                        );
                    }
                }
                assert_eq!(next, layout.n_layer, "every layer is computed");
                assert_eq!((embeds, heads), (1, 1), "the embedding and the head once");
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "{checked} replicas across nodes checked");
    println!(
        "specs from plan fixtures {} replicas_checked {checked}",
        FIXTURES.len()
    );
}

// hardware: none
#[test]
fn a_plan_on_a_node_the_cluster_lacks_is_refused() {
    let hw = HardwareGraph::from_toml(FIXTURES[0]).unwrap();
    let (cands, _) = enumerate_cluster(&hw, &fixtures::moe(), &Hints::default(), Phase(6));
    // A cluster file with one of the plan's two nodes missing.
    let cfg = ClusterConfig::parse(
        "cluster = \"fx\"\nhead = \"a\"\n[[nodes]]\nid = \"a\"\naddr = \"127.0.0.1:9000\"\nanchor = \"a.pem\"\n",
    )
    .unwrap();
    let err = specs_from_plan(&cands[0].plan, &hw, &cfg, SHA, &[]).unwrap_err();
    assert!(err.contains("not a node of cluster.toml"), "{err}");
}

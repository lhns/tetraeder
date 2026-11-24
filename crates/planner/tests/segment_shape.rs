//! `segment_shape` reads the layers, embed and head of a worker's sub-plan (ADR 0091, step 4);
//! on every candidate of every cluster fixture it agrees with `segments`, whichever way the
//! head was placed.
//!
//! Hardware labels (AGENTS rule 9): all tests are `cpu`.

use tetraeder_plan::{Hints, Phase, Plan, Replica, Stage, segment_plan, segment_shape, segments};
use tetraeder_planner::enumerate_cluster;
use tetraeder_planner::layout::fixtures;
use tetraeder_profiler::HardwareGraph;

const FIXTURES: [&str; 4] = [
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_samehost.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml"),
];

/// hardware: cpu
#[test]
fn the_shape_of_every_sub_plan_matches_its_segment() {
    let layout = fixtures::moe();
    let mut head_apart = 0;
    let mut seen = 0;
    for text in FIXTURES {
        let hw = HardwareGraph::from_toml(text).unwrap();
        let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
        assert!(!cands.is_empty());
        for c in &cands {
            let segs = segments(&c.plan, 0, &hw).unwrap();
            head_apart += segs.iter().filter(|s| s.stages.is_empty()).count();
            for s in &segs {
                let sub = segment_plan(&c.plan, 0, s);
                let shape = segment_shape(&sub, layout.n_layer).unwrap();
                assert_eq!(
                    (shape.layers, shape.embed, shape.head),
                    (s.layers, s.embed, s.head),
                    "{} segment {:?}",
                    c.id,
                    s.stages
                );
                seen += 1;
            }
        }
    }
    // The fixtures hold plans with the head on the leader, and plans with the head on the last.
    assert!(head_apart > 0 && seen > head_apart);
}

/// hardware: cpu
#[test]
fn a_plan_that_is_not_one_segment_is_refused() {
    let none = Plan {
        model: "m".into(),
        knobs: Default::default(),
        replicas: vec![],
    };
    assert!(segment_shape(&none, 4).is_err());
    let empty = Plan {
        replicas: vec![Replica {
            stages: Vec::<Stage>::new(),
        }],
        ..none.clone()
    };
    assert!(segment_shape(&empty, 4).is_err());
}

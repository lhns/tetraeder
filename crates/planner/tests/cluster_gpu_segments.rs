//! A segment that sits all on its node's GPU is priced (`cluster-plan-source-gpu-segments`,
//! ADR 0091 History). The rehearsal ran the 30B as two stages, one per 3090, written by hand
//! (`scripts/phase6_plan.py`: devices `["b/gpu0"]`, no expert cache); the cost model refused it
//! ("a segment is a builder placement or all on the CPU"), so the head could neither price it
//! nor replan from it.
//!
//! An all-GPU segment is read as the builder placement with every expert resident in VRAM and
//! none in RAM. Run on the rehearsal's own hardware graph and layout, and as a property over
//! the cluster fixtures and cuts.
//!
//! Hardware label: none (the graph and layout are copies of the rehearsal's `a/hw.json` and `a/layout.json`,
//! kept in `fixtures/phase6-rehearsal` because boxes do not get `bench/results`).

use proptest::prelude::*;
use tetraeder_costmodel::cluster::predict_cluster;
use tetraeder_plan::{Forbid, Hints, Phase, Plan};
use tetraeder_planner::layout::{ModelLayout, fixtures};
use tetraeder_planner::multi_cli::{parse_hw, parse_layout};
use tetraeder_planner::{
    Workload, cluster_features, enumerate_cluster, predict_cluster_plan, single_node_references,
};
use tetraeder_profiler::HardwareGraph;

const HW: &str = include_str!("../fixtures/phase6-rehearsal/hw.json");
const LAYOUT: &str = include_str!("../fixtures/phase6-rehearsal/layout.json");
const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

const FIXTURES: [&str; 4] = [
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_samehost.toml"),
    include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml"),
];

fn stage(lo: u32, hi: u32, dev: &str, kind: &str) -> String {
    format!(
        "[[replicas.stages]]\nlayers = [{lo}, {hi}]\ndevices = [\"{dev}\"]\ntp = 1\nep = 1\n\n\
         [[replicas.stages.assign]]\nunits = {{ layers = [{lo}, {hi}] }}\nhome = [\"{dev}\"]\ncompute = [\"{dev}\"]\n\n\
         [[replicas.stages.assign]]\nunits = {{ kinds = [\"{kind}\"] }}\nhome = [\"{dev}\"]\ncompute = [\"{dev}\"]\n\n"
    )
}

/// What `scripts/phase6_plan.py` writes: stage 0 with the embedding, the last with the head,
/// each wholly on one GPU.
fn hand_plan(layout: &ModelLayout, cut: u32, first: &str, second: &str) -> Plan {
    let t = format!(
        "model = \"{}\"\n\n[[replicas]]\n\n{}{}",
        layout.model,
        stage(0, cut, first, "embed"),
        stage(cut, layout.n_layer, second, "head")
    );
    Plan::from_toml(&t).unwrap()
}

fn rehearsal() -> (HardwareGraph, ModelLayout) {
    (parse_hw(HW).unwrap(), parse_layout(LAYOUT).unwrap())
}

// hardware: none
#[test]
fn the_rehearsals_hand_written_split_is_priced() {
    let (hw, layout) = rehearsal();
    let plan = hand_plan(&layout, 24, "b/gpu0", "a/gpu0");
    let p = predict_cluster_plan(&hw, &layout, &plan, &W)
        .unwrap_or_else(|e| panic!("the all-GPU split is not priced: {e}"));
    assert!(p.pp_tps > 0.0 && p.tg_tps > 0.0, "{p:?}");
    println!(
        "gpu_segments hand_split pp_tps={:.1} tg_tps={:.2}",
        p.pp_tps, p.tg_tps
    );
}

// hardware: none
#[test]
fn the_planner_proposes_a_split_it_can_price_again() {
    let (hw, layout) = rehearsal();
    let (cands, pruned) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    assert!(
        !cands.is_empty(),
        "no proposal for two GPU nodes: {pruned:?}"
    );
    let best = &cands[0];
    let f = cluster_features(&hw, &layout, &best.plan).expect("a proposal is priced again");
    assert_eq!(f.segments.len(), 2);
    println!(
        "gpu_segments proposed={} best_pp_tps={:.1} best_tg_tps={:.2} note={}",
        cands.len(),
        best.prediction.pp_tps,
        best.prediction.tg_tps,
        best.notes.join(" ")
    );
}

// hardware: none
#[test]
fn a_node_that_left_leaves_a_single_node_reference() {
    let (hw, layout) = rehearsal();
    let hints = Hints {
        forbid: vec![Forbid {
            units: None,
            device: None,
            node: Some("b".into()),
            tier: None,
        }],
        ..Hints::default()
    };
    let refs = single_node_references(&hw, &layout, &hints);
    assert_eq!(refs.len(), 1, "node a alone holds the 30B");
    assert!(refs[0].id.ends_with('a'));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// On every fixture and every cut, a plan with each stage wholly on one GPU is priced, and
    /// the prediction is finite and positive.
    #[test]
    fn an_all_gpu_segment_is_priced_on_every_fixture(f in 0usize..FIXTURES.len(), cut in 4u32..44) {
        let hw = HardwareGraph::from_toml(FIXTURES[f]).unwrap();
        let layout = fixtures::moe();
        let nodes: Vec<String> = hw.nodes().into_iter().flatten().map(str::to_owned).collect();
        prop_assume!(nodes.len() >= 2);
        let gpu = |n: &str| format!("{n}/gpu0");
        prop_assume!(hw.device(&gpu(&nodes[0])).is_some() && hw.device(&gpu(&nodes[1])).is_some());
        let cut = cut.min(layout.n_layer - 1);
        let plan = hand_plan(&layout, cut, &gpu(&nodes[0]), &gpu(&nodes[1]));
        let feat = cluster_features(&hw, &layout, &plan)
            .map_err(|e| TestCaseError::fail(format!("the all-GPU plan is not priced: {e}")))?;
        let p = predict_cluster(&hw, &feat, W.prompt, W.r#gen).unwrap();
        prop_assert!(p.request_s.is_finite() && p.request_s > 0.0);
        prop_assert!(p.pp_tps > 0.0 && p.tg_tps > 0.0);
    }
}

// hardware: none
#[test]
fn candidates_cluster_writes_the_plans_the_head_serves() {
    let dir = std::env::temp_dir().join(format!("tetraeder-cluster-plans-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (hw, layout) = (dir.join("hw.json"), dir.join("layout.json"));
    std::fs::write(&hw, HW).unwrap();
    std::fs::write(&layout, LAYOUT).unwrap();
    let plans = dir.join("plans");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_tetraeder-plan"))
        .args(["candidates", "--cluster", "--hw"])
        .arg(&hw)
        .arg("--layout")
        .arg(&layout)
        .arg("--plans-dir")
        .arg(&plans)
        .arg("--out")
        .arg(dir.join("summary.json"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let first = std::fs::read_dir(&plans)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .min()
        .expect("a plan file");
    // The plan reads back and prices again, as `serve --cluster-plan` needs.
    let plan = Plan::from_toml(&std::fs::read_to_string(&first).unwrap()).unwrap();
    let (hwg, lay) = rehearsal();
    predict_cluster_plan(&hwg, &lay, &plan, &W).expect("the written plan is priced");
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
    assert!(
        summary["candidates"]
            .as_array()
            .is_some_and(|c| !c.is_empty())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

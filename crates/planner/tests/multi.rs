//! The Phase 5 candidate set and its predictions (ADR 0068, 0070), on the simulated 2x and 4x
//! RTX 3090 boxes without peer access and on the older fixtures (AGENTS.md rule 4), with a
//! slow-link fault. Hardware label: none (simulated hardware only).

use std::collections::BTreeMap;
use tetraeder_costmodel::predict::{Coll, Phase as RankPhase, RunFeatures, predict};
use tetraeder_plan::Hints;
use tetraeder_planner::layout::fixtures::{dense, moe};
use tetraeder_planner::multi_cli::{self, PHASE};
use tetraeder_planner::{
    ModelLayout, candidate_for, enumerate_multi_with, features_multi, predict_candidate,
    topology_of,
};
use tetraeder_profiler::{HardwareGraph, LinkKind, RangeCosts};

const TWO: &str = include_str!("../../costmodel/fixtures/two_3090_hoststaged.toml");
const FOUR: &str = include_str!("../../costmodel/fixtures/four_3090_hoststaged.toml");
const TWO_GPU: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
const RTX3090: &str = include_str!("../fixtures/rtx3090_prior.toml");
const CPU_ONLY: &str = include_str!("../fixtures/cpu_only.toml");

const QUOTA: u32 = 8;

fn hw(toml: &str) -> HardwareGraph {
    let g = HardwareGraph::from_toml(toml).unwrap();
    g.validate().unwrap();
    g
}

fn ids(hw: &HardwareGraph, l: &ModelLayout) -> Vec<String> {
    let en = enumerate_multi_with(hw, l, &Hints::default(), PHASE, QUOTA);
    en.candidates.into_iter().map(|c| c.id).collect()
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn the_candidate_set_is_adr_0068s_table() {
    let (two, four) = (hw(TWO), hw(FOUR));
    assert_eq!(ids(&two, &dense()), strs(&["single", "layer-2", "tp-2"]));
    assert_eq!(
        ids(&two, &moe()),
        strs(&["single", "layer-2", "tp-2", "ep-2"])
    );
    assert_eq!(
        ids(&four, &dense()),
        strs(&["single", "layer-4", "tp-4", "tp2-pp2"])
    );
    // tp-4 cuts the 768-wide experts into 3 blocks: pruned with the reason
    assert_eq!(
        ids(&four, &moe()),
        strs(&["single", "layer-4", "ep-4", "tp2-pp2"])
    );
    let en = enumerate_multi_with(&four, &moe(), &Hints::default(), PHASE, QUOTA);
    let tp4 = en.pruned.iter().find(|p| p.0 == "tp-4").unwrap();
    assert!(tp4.1.contains("3 blocks over 4 ranks"), "{}", tp4.1);
}

#[test]
fn enumeration_is_deterministic() {
    let four = hw(FOUR);
    for l in [dense(), moe()] {
        let a = enumerate_multi_with(&four, &l, &Hints::default(), PHASE, QUOTA);
        let b = enumerate_multi_with(&four, &l, &Hints::default(), PHASE, QUOTA);
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
}

/// Every fixture with GPUs: each candidate is a valid plan whose stages tile the layers once,
/// and its prediction is finite.
#[test]
fn every_candidate_on_every_fixture_is_valid_and_finite() {
    let all = [
        ("two_3090_hoststaged", hw(TWO)),
        ("four_3090_hoststaged", hw(FOUR)),
        ("two_gpu_cpu_nvme", hw(TWO_GPU)),
        ("rtx3090_prior", hw(RTX3090)),
    ];
    for (name, g) in &all {
        for l in [dense(), moe()] {
            let en = enumerate_multi_with(g, &l, &Hints::default(), PHASE, QUOTA);
            assert!(
                en.candidates.iter().any(|c| c.id == "single"),
                "{name}/{}: {:?}",
                l.model,
                en.pruned
            );
            for c in &en.candidates {
                let t = topology_of(g, &l, &c.plan, PHASE).unwrap();
                // every layer is computed by exactly one stage
                let mut next = 0;
                for s in &t.stages {
                    assert_eq!(s.layers.0, next, "{name}/{}/{}", l.model, c.id);
                    next = s.layers.1;
                }
                assert_eq!(next, l.n_layer);
                let p = predict_candidate(g, &l, &c.plan, PHASE, (2048, 128), None)
                    .unwrap_or_else(|e| panic!("{name}/{}/{}: {e}", l.model, c.id));
                assert!(
                    p.pp_tok_s.is_finite() && p.pp_tok_s > 0.0,
                    "{name}/{}/{}",
                    l.model,
                    c.id
                );
                assert!(p.tg_tok_s.is_finite() && p.tg_tok_s > 0.0);
            }
        }
    }
    // no GPU: everything pruned with one reason
    let none = enumerate_multi_with(&hw(CPU_ONLY), &dense(), &Hints::default(), PHASE, QUOTA);
    assert!(none.candidates.is_empty() && !none.pruned.is_empty());
}

fn round(x: f64) -> String {
    format!("{x:.3e}")
}

/// The predicted rates per candidate and fixture, committed; a change must update the file on
/// purpose (`UPDATE_SNAPSHOT=1 cargo test -p tetraeder-planner --test multi`).
#[test]
fn predictions_match_the_committed_snapshot() {
    let mut got: BTreeMap<String, String> = BTreeMap::new();
    for (name, g) in [("two_3090", hw(TWO)), ("four_3090", hw(FOUR))] {
        for l in [dense(), moe()] {
            let en = enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA);
            for c in &en.candidates {
                let p = predict_candidate(&g, &l, &c.plan, PHASE, (2048, 128), None).unwrap();
                got.insert(
                    format!("{name}/{}/{}", l.model, c.id),
                    format!("pp {} tg {}", round(p.pp_tok_s), round(p.tg_tok_s)),
                );
            }
        }
    }
    let text = serde_json::to_string_pretty(&got).unwrap() + "\n";
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/snapshots/multi_predictions.json"
    );
    if std::env::var("UPDATE_SNAPSHOT").is_ok() {
        std::fs::write(path, &text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(path).unwrap_or_default();
    assert_eq!(
        want.replace("\r\n", "\n"),
        text,
        "predictions changed; if on purpose, write this to {path}:\n{text}"
    );
}

/// A slow host link (fault): staged copies and the all-reduce curves get ten times slower. Rules
/// out a model that prices tensor parallel without its collectives.
#[test]
fn a_slow_host_link_hurts_tensor_parallel_and_leaves_single_alone() {
    let base = hw(TWO);
    let mut slow = base.clone();
    for l in &mut slow.links {
        if l.kind == LinkKind::HostStaged {
            l.beta_bytes_per_s.value /= 10.0;
            l.alpha_s.value *= 10.0;
        }
    }
    for c in &mut slow.collectives {
        for p in &mut c.points {
            p.1 *= 10.0;
        }
    }
    let l = dense();
    let rate = |g: &HardwareGraph, id: &str| {
        let c = candidate_for(g, &l, &Hints::default(), PHASE, id).unwrap();
        predict_candidate(g, &l, &c.plan, PHASE, (2048, 128), None).unwrap()
    };
    let (b, s) = (rate(&base, "tp-2"), rate(&slow, "tp-2"));
    assert!(
        s.tg_tok_s < b.tg_tok_s * 0.9,
        "{} vs {}",
        s.tg_tok_s,
        b.tg_tok_s
    );
    assert!(s.pp_tok_s < b.pp_tok_s);
    let (b1, s1) = (rate(&base, "single"), rate(&slow, "single"));
    assert_eq!(b1.tg_tok_s.to_bits(), s1.tg_tok_s.to_bits());
    // the layer split moves one hidden state per token and suffers far less
    let (bl, sl) = (rate(&base, "layer-2"), rate(&slow, "layer-2"));
    assert!(sl.tg_tok_s / bl.tg_tok_s > s.tg_tok_s / b.tg_tok_s);
}

#[test]
fn tensor_parallel_beats_one_gpu_only_when_collectives_are_cheap() {
    // With NVLink-class collectives tp-2 halves the weight reads; with the 12 GB/s host path it
    // cannot reach twice the speed of one GPU.
    let g = hw(TWO);
    let l = dense();
    let r = |id: &str| {
        let c = candidate_for(&g, &l, &Hints::default(), PHASE, id).unwrap();
        predict_candidate(&g, &l, &c.plan, PHASE, (2048, 128), None).unwrap()
    };
    let (one, two) = (r("single"), r("tp-2"));
    assert!(two.tg_tok_s < 2.0 * one.tg_tok_s);
    assert!(two.pp_tok_s < 2.0 * one.pp_tok_s);
}

/// The anchor plumbing end to end: rows generated from known `c_range` and `c_step` on the 2x
/// fixture are read back by `calibrate`.
#[test]
fn calibrate_recovers_c_range_and_c_step_from_anchor_rows() {
    let prior = hw(TWO);
    let mut truth = prior.clone();
    truth.range_costs = Some(RangeCosts {
        c_range_graph_s: 1e-4,
        c_range_eager_s: 4e-5,
        c_step_s: 8e-4,
        // at the fit's prior (ADR 0078): c_range and the all-reduce latency share one range
        allreduce_latency_s: Some(1.2e-5),
        ..RangeCosts::PRIOR
    });
    let l = dense();
    let mut rows = Vec::new();
    for id in ["single", "tp-2"] {
        let c = candidate_for(&truth, &l, &Hints::default(), PHASE, id).unwrap();
        let topo = topology_of(&truth, &l, &c.plan, PHASE).unwrap();
        for limit in [8u32, 16, 32] {
            let f = features_multi(&l, &topo, &c.plan.knobs, Some(limit), None).unwrap();
            let p = predict(&truth, &f, 2048, 128).unwrap();
            rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                "n_prompt": 2048, "n_gen": 0, "avg_ts": p.pp_tok_s}));
            rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                "n_prompt": 0, "n_gen": 128, "avg_ts": p.tg_tok_s}));
        }
    }
    let text = serde_json::to_string(&rows).unwrap();
    let fit = multi_cli::calibrate(&prior, &[(l, vec![text])]).unwrap();
    assert_eq!(fit.anchors.len(), 6);
    let c = fit.hw.range_costs.expect("fitted");
    let rel = |a: f64, b: f64| (a / b - 1.0).abs();
    assert!(rel(c.c_range_graph_s, 1e-4) < 0.05, "{c:?}");
    assert!(rel(c.c_step_s, 8e-4) < 0.05, "{c:?}");
}

fn features_of(l: &ModelLayout, id: &str, limit: Option<u32>) -> RunFeatures {
    let g = hw(TWO);
    let c = candidate_for(&g, l, &Hints::default(), PHASE, id).unwrap();
    let topo = topology_of(&g, l, &c.plan, PHASE).unwrap();
    features_multi(l, &topo, &c.plan.knobs, limit, None).unwrap()
}

/// MODEL_VERSION 2: `single` is a one-rank plan on the ranks path, with the step cost and the
/// logit readback of every other shape (ADR 0070).
#[test]
fn single_is_priced_on_the_ranks_path() {
    for l in [dense(), moe()] {
        let f = features_of(&l, "single", Some(8));
        let r = f.ranks.as_ref().expect("single has ranks");
        assert!(r.step_fixed);
        let ranges: u32 = r.phases.iter().map(|p| p.repeat * p.ranges_per_rank).sum();
        assert_eq!(ranges, 1, "{}: one range per step", l.model);
        let head = r.phases.last().unwrap();
        assert!(head.out_rows);
        assert!(matches!(
            &head.after,
            Some(Coll::Copy { per_step: true, pairs, .. }) if pairs == &[("gpu0".to_string(), "cpu0".to_string())]
        ));
        let p = predict(&hw(TWO), &f, 2048, 128).unwrap();
        assert!(p.tg_breakdown.iter().any(|e| e.0 == "step"));
    }
}

/// The head (and the dense builder's last FFN) on the output rows only, on every shape.
#[test]
fn the_head_runs_on_the_output_rows() {
    for l in [dense(), moe()] {
        for id in ["single", "layer-2", "tp-2"] {
            let f = features_of(&l, id, None);
            let r = f.ranks.unwrap();
            let out: Vec<&RankPhase> = r.phases.iter().filter(|p| p.out_rows).collect();
            let head_bytes: f64 = out
                .iter()
                .flat_map(|p| p.parts.iter().take(1))
                .map(|p| p.weight_bytes_per_token)
                .sum();
            let tp = if id == "tp-2" { 2.0 } else { 1.0 };
            let ffn = if l.is_moe() {
                0.0
            } else {
                l.layers.last().unwrap().ffn_bytes as f64 / tp
            };
            let want = l.output_bytes as f64 / tp + ffn;
            assert!(
                (head_bytes / want - 1.0).abs() < 1e-9,
                "{}/{id}: {head_bytes} vs {want}",
                l.model
            );
        }
    }
}

/// A MoE prefill reads every expert (its column share on tp-2), decode the top-k share; the
/// FLOPs follow the chosen experts either way.
#[test]
fn moe_prefill_reads_every_expert_on_single_and_tp2() {
    let l = moe();
    let ex: f64 = l.layers.iter().map(|b| b.expert_bytes as f64).sum();
    let r = f64::from(l.n_expert_used) / f64::from(l.n_expert);
    for id in ["single", "tp-2"] {
        let f = features_of(&l, id, None);
        let (mut dec, mut pre, mut dec_p, mut pre_p) = (0.0, 0.0, 0.0, 0.0);
        for ph in &f.ranks.unwrap().phases {
            let k = f64::from(ph.repeat);
            let parts = ph.prefill_parts.as_ref().unwrap_or(&ph.parts);
            // summed over the ranks: the column shares add up to the whole expert
            dec += k * ph
                .parts
                .iter()
                .map(|p| p.expert_bytes_per_token)
                .sum::<f64>();
            pre += k * parts.iter().map(|p| p.expert_bytes_per_token).sum::<f64>();
            dec_p += k * ph
                .parts
                .iter()
                .map(|p| p.expert_params_per_token)
                .sum::<f64>();
            pre_p += k * parts.iter().map(|p| p.expert_params_per_token).sum::<f64>();
        }
        assert!((pre / ex - 1.0).abs() < 1e-9, "{id}: prefill {pre} of {ex}");
        assert!((dec / (r * ex) - 1.0).abs() < 1e-9, "{id}: decode {dec}");
        assert!((pre_p / dec_p - 1.0).abs() < 1e-9, "{id}: params");
    }
}

#[test]
fn anchor_file_names_give_the_model() {
    assert_eq!(
        multi_cli::anchor_model("anchors-qwen3-32b-r2.json").as_deref(),
        Some("qwen3-32b")
    );
    assert_eq!(multi_cli::anchor_model("anchors-x.json"), None);
    assert_eq!(multi_cli::anchor_model("other-x-r1.json"), None);
}

#[test]
fn predict_file_carries_the_hashes_and_all_candidates() {
    let g = hw(TWO);
    let l = moe();
    let file = multi_cli::candidates_multi(&g, &l);
    let p =
        multi_cli::predict_multi(&g, &l, &file, ("abc".into(), None, None), "deadbeef").unwrap();
    assert_eq!(p.candidates.len(), file.candidates.len());
    assert_eq!(
        (p.hw_sha256.as_str(), p.git_commit.as_str()),
        ("abc", "deadbeef")
    );
    assert!(
        p.candidates
            .iter()
            .all(|r| (r.t_s - (2048.0 / r.pp_tps + 128.0 / r.tg_tps)).abs() < 1e-9)
    );
    assert_eq!(multi_cli::sha256_hex(b"abc").len(), 64);
}

#[test]
fn the_comm_probe_report_is_read_as_written() {
    // As `tetraeder comm-probe` serializes it (E5): PascalCase ops, extra fields.
    let text = r#"{"comm_path":"internal","devices":["gpu0","gpu1"],"repeats":20,"curves":[
        {"op":"AllReduce","ranks":["gpu0","gpu1"],"comm":"internal","points":[[4096.0,3e-5],[16384.0,4e-5]]},
        {"op":"Copy","ranks":["gpu0","gpu1"],"comm":"copy","points":[[4096.0,2e-5],[16384.0,3e-5]]}]}"#;
    let mut g = hw(TWO);
    let n = g.collectives.len();
    let path = multi_cli::merge_comm(&mut g, text).unwrap();
    assert_eq!(path, "internal");
    // the prior curves of the same op and ranks are replaced
    assert_eq!(g.collectives.len(), n);
    let t = g.collective_time(
        tetraeder_profiler::CollectiveOp::AllReduce,
        &["gpu1".to_string(), "gpu0".to_string()],
        4096.0,
    );
    assert_eq!(t, Some(3e-5));
}

/// The layer counts of the stages of candidate `id`.
fn stage_layers(g: &HardwareGraph, l: &ModelLayout, id: &str) -> Vec<u32> {
    let en = enumerate_multi_with(g, l, &Hints::default(), PHASE, QUOTA);
    let c = en.candidates.iter().find(|c| c.id == id).expect(id);
    let t = topology_of(g, l, &c.plan, PHASE).unwrap();
    t.stages.iter().map(|s| s.layers.1 - s.layers.0).collect()
}

/// The split balances one pipelined prefill step (ADR 0070 History 2026-11-19): the head runs one
/// row per step, so it no longer pushes layers off the last stage (28/20 on the 30B before).
#[test]
fn the_layer_split_balances_the_prefill_step() {
    let (two, four) = (hw(TWO), hw(FOUR));
    assert_eq!(stage_layers(&two, &moe(), "layer-2"), [24, 24]);
    assert_eq!(stage_layers(&two, &dense(), "layer-2"), [32, 32]);
    assert_eq!(stage_layers(&four, &moe(), "layer-4"), [12, 12, 12, 12]);
    assert_eq!(stage_layers(&four, &dense(), "layer-4"), [16, 16, 16, 16]);
    assert_eq!(stage_layers(&four, &moe(), "tp2-pp2"), [24, 24]);
    assert_eq!(stage_layers(&four, &dense(), "tp2-pp2"), [32, 32]);
}

/// Every simulated fixture with identical GPUs: the stages of every pipeline candidate tile the
/// layers once and differ by at most one layer.
#[test]
fn every_fixture_splits_within_one_layer_of_equal() {
    for (name, g) in [("two", hw(TWO)), ("four", hw(FOUR))] {
        for l in [dense(), moe()] {
            let en = enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA);
            for c in en
                .candidates
                .iter()
                .filter(|c| c.id.contains("layer") || c.id.contains("pp"))
            {
                let n = stage_layers(&g, &l, &c.id);
                let what = format!("{name}/{}/{}: {n:?}", l.model, c.id);
                assert_eq!(n.iter().sum::<u32>(), l.n_layer, "{what}");
                let (lo, hi) = (n.iter().min().unwrap(), n.iter().max().unwrap());
                assert!(hi - lo <= 1, "{what}");
            }
        }
    }
}

mod memory_property {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Shrinking the cards: whatever the planner still offers tiles the layers once and
        /// passed the validator's memory check (`output_bytes` included); the rest is pruned,
        /// never a plan over capacity.
        #[test]
        fn layer_splits_never_exceed_memory(gb in 10u64..26, fixture in 0usize..2) {
            let mut g = hw([TWO, FOUR][fixture]);
            for d in g.devices.iter_mut().filter(|d| d.id.starts_with("gpu")) {
                d.capacity_bytes = gb * (1 << 30);
            }
            for l in [dense(), moe()] {
                let en = enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA);
                for c in &en.candidates {
                    let t = topology_of(&g, &l, &c.plan, PHASE).unwrap();
                    let mut next = 0;
                    for s in &t.stages {
                        prop_assert_eq!(s.layers.0, next);
                        next = s.layers.1;
                    }
                    prop_assert_eq!(next, l.n_layer);
                }
            }
        }
    }
}

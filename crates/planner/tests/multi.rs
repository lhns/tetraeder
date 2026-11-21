//! The Phase 5 candidate set and its predictions (ADR 0068, 0070), on the simulated 2x and 4x
//! RTX 3090 boxes without peer access and on the older fixtures (AGENTS.md rule 4), with a
//! slow-link fault. Hardware label: none (simulated hardware only).

use std::collections::BTreeMap;
use tetraeder_costmodel::predict::{Coll, Phase as RankPhase, RunFeatures, predict};
use tetraeder_plan::Hints;
use tetraeder_planner::layout::fixtures::{dense, moe};
use tetraeder_planner::multi_cli::{self, PHASE};
use tetraeder_planner::{
    ModelLayout, Routing, candidate_for, enumerate_multi_with, expected_max_local, features_multi,
    predict_candidate, topology_of,
};
use tetraeder_profiler::{HardwareGraph, LinkKind, RangeCosts};

const TWO: &str = include_str!("../../costmodel/fixtures/two_3090_hoststaged.toml");
const FOUR: &str = include_str!("../../costmodel/fixtures/four_3090_hoststaged.toml");
const TWO_GPU: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
const RTX3090: &str = include_str!("../fixtures/rtx3090_prior.toml");
const CPU_ONLY: &str = include_str!("../fixtures/cpu_only.toml");
/// The `decode_clocks` block of host 20210 (ADR 0082), appended to a graph fixture.
const CLOCKS: &str = include_str!("../../costmodel/fixtures/decode_clocks_host20210.toml");

const QUOTA: u32 = 8;

fn hw(toml: &str) -> HardwareGraph {
    let g = HardwareGraph::from_toml(toml).unwrap();
    g.validate().unwrap();
    g
}

fn with_clocks(toml: &str) -> HardwareGraph {
    hw(&format!("{toml}\n{CLOCKS}"))
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

/// Every candidate of the fixtures predicted, as the snapshot files hold them.
fn snapshot_text(fixtures: [(&str, HardwareGraph); 2]) -> String {
    let mut got: BTreeMap<String, String> = BTreeMap::new();
    for (name, g) in fixtures {
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
    serde_json::to_string_pretty(&got).unwrap() + "\n"
}

/// The predicted rates per candidate and fixture, committed; a change must update the file on
/// purpose (`UPDATE_SNAPSHOT=1 cargo test -p tetraeder-planner --test multi`).
#[test]
fn predictions_match_the_committed_snapshot() {
    let text = snapshot_text([("two_3090", hw(TWO)), ("four_3090", hw(FOUR))]);
    check_snapshot("multi_predictions.json", &text);
}

/// The same fixtures with host 20210's decode clocks (ADR 0082), committed beside the first.
#[test]
fn predictions_with_decode_clocks_match_the_committed_snapshot() {
    let text = snapshot_text([
        ("two_3090_clocks", with_clocks(TWO)),
        ("four_3090_clocks", with_clocks(FOUR)),
    ]);
    check_snapshot("multi_predictions_clocks.json", &text);
}

fn check_snapshot(file: &str, text: &str) {
    let path = format!("{}/tests/snapshots/{file}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var("UPDATE_SNAPSHOT").is_ok() {
        std::fs::write(&path, text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        want.replace("\r\n", "\n"),
        text,
        "predictions changed; if on purpose, write this to {path}:\n{text}"
    );
}

/// ADR 0082 on the 2x fixture: with the clocks block no prediction moves (every GPU is busy
/// more than 0.32 of a decode step), and the block does not change any prefill number.
#[test]
fn decode_clocks_change_nothing_on_two_gpus() {
    let (plain, clocked) = (hw(TWO), with_clocks(TWO));
    for l in [dense(), moe()] {
        for c in enumerate_multi_with(&plain, &l, &Hints::default(), PHASE, QUOTA).candidates {
            let w = (2048, 128);
            let a = predict_candidate(&plain, &l, &c.plan, PHASE, w, None).unwrap();
            let b = predict_candidate(&clocked, &l, &c.plan, PHASE, w, None).unwrap();
            assert_eq!(a.pp_tok_s.to_bits(), b.pp_tok_s.to_bits(), "{}", c.id);
            assert_eq!(
                a.tg_tok_s.to_bits(),
                b.tg_tok_s.to_bits(),
                "{}/{}",
                l.model,
                c.id
            );
            assert!(a.decode_clocks.is_empty() && !b.decode_clocks.is_empty());
        }
    }
}

/// ADR 0082 on the 4x fixture: only 30B decode falls below the duty threshold somewhere; the 32B
/// gets no factor (its busy clock is under the low one), prefill never moves.
#[test]
fn decode_clocks_slow_the_low_duty_30b_decode_on_four_gpus() {
    let (plain, clocked) = (hw(FOUR), with_clocks(FOUR));
    let mut moved = Vec::new();
    for l in [dense(), moe()] {
        for c in enumerate_multi_with(&plain, &l, &Hints::default(), PHASE, QUOTA).candidates {
            let w = (2048, 128);
            let a = predict_candidate(&plain, &l, &c.plan, PHASE, w, None).unwrap();
            let b = predict_candidate(&clocked, &l, &c.plan, PHASE, w, None).unwrap();
            assert_eq!(a.pp_tok_s.to_bits(), b.pp_tok_s.to_bits(), "{}", c.id);
            if a.tg_tok_s.to_bits() != b.tg_tok_s.to_bits() {
                assert!(b.tg_tok_s < a.tg_tok_s);
                moved.push(format!("{}/{}", l.model, c.id));
            }
            for d in &b.decode_clocks {
                assert_eq!(d.factor > 1.0, d.duty < 0.32 && l.is_moe(), "{}", c.id);
            }
        }
    }
    assert!(
        moved.iter().any(|m| m == "qwen3-30b-a3b/layer-4"),
        "{moved:?}"
    );
    assert!(
        moved.iter().all(|m| m.starts_with("qwen3-30b-a3b/")),
        "{moved:?}"
    );
}

/// ADR 0082: `layer-4@L24` is the layer anchor, 6/6/6 and the rest of the 30B on the last
/// stage, and only boxes with 3 or more GPUs have it.
#[test]
fn the_layer_anchor_cuts_six_six_six_and_the_rest() {
    let (g, l) = (hw(FOUR), moe());
    let c = candidate_for(&g, &l, &Hints::default(), PHASE, "layer-4@L24").unwrap();
    assert_eq!(c.id, "layer-4@L24");
    let topo = topology_of(&g, &l, &c.plan, PHASE).unwrap();
    let spans: Vec<(u32, u32)> = topo
        .stages
        .iter()
        .map(|s| (s.layers.0, s.layers.1))
        .collect();
    assert_eq!(spans, [(0, 6), (6, 12), (12, 18), (18, 48)]);
    // limited to 24 layers every stage computes 6
    let f = features_multi(&l, &topo, &c.plan.knobs, Some(24), None).unwrap();
    let r = f.ranks.unwrap();
    assert_eq!(r.model.as_deref(), Some("qwen3-30b-a3b"));
    let ops: Vec<f64> = r
        .phases
        .iter()
        .filter(|p| !p.out_rows)
        .map(|p| p.parts[0].ops_per_token)
        .collect();
    assert_eq!(ops.len(), 4);
    assert!(ops.windows(2).all(|w| w[0] == w[1]), "{ops:?}");
    for id in ["layer-4@L24", "layer-2@L24"] {
        assert!(candidate_for(&hw(TWO), &l, &Hints::default(), PHASE, id).is_err());
    }
    // not in the box's fixed set, and a limit that 4 does not divide is refused
    let en = enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA);
    assert!(en.candidates.iter().all(|c| !c.id.contains('@')));
    assert!(candidate_for(&g, &l, &Hints::default(), PHASE, "layer-4@L25").is_err());
}

/// ADR 0082: the layer anchor is held out. The fit is the same with and without its rows (and
/// with the clocks in the graph), and its tg residual is reported.
#[test]
fn calibrate_holds_the_layer_anchor_out() {
    let prior = hw(FOUR);
    let clocked = with_clocks(FOUR);
    let l = moe();
    let rows_of = |g: &HardwareGraph, id: &str, limits: &[u32], tg_only: bool| {
        let c = candidate_for(g, &l, &Hints::default(), PHASE, id).unwrap();
        let topo = topology_of(g, &l, &c.plan, PHASE).unwrap();
        let mut rows = Vec::new();
        for &limit in limits {
            let f = features_multi(&l, &topo, &c.plan.knobs, Some(limit), None).unwrap();
            let p = predict(g, &f, 2048, 128).unwrap();
            if !tg_only {
                rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                    "n_prompt": 2048, "n_gen": 0, "avg_ts": p.pp_tok_s}));
            }
            rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                "n_prompt": 0, "n_gen": 128, "avg_ts": p.tg_tok_s}));
        }
        rows
    };
    let single = rows_of(&clocked, "single", &[8, 16, 24], false);
    let mut with_layer = single.clone();
    with_layer.extend(rows_of(&clocked, "layer-4@L24", &[24], true));
    let text = |v: &[serde_json::Value]| serde_json::to_string(v).unwrap();
    let without = multi_cli::calibrate(&prior, &[(moe(), vec![text(&single)])]).unwrap();
    let with = multi_cli::calibrate(&clocked, &[(moe(), vec![text(&with_layer)])]).unwrap();
    assert_eq!(with.anchors, without.anchors);
    assert_eq!(with.residuals, without.residuals);
    assert!(without.held_out.is_empty());
    let mut fitted = with.hw.clone();
    assert_eq!(fitted.decode_clocks, clocked.decode_clocks);
    fitted.decode_clocks = None;
    assert_eq!(fitted, without.hw);
    assert_eq!(with.held_out.len(), 1);
    let (id, r) = &with.held_out[0];
    assert_eq!(id, "qwen3-30b-a3b/layer-4@L24/L24:tg");
    assert!(r.is_finite(), "{r}");
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
/// fixture are read back by `calibrate`. The anchors fix the fixed cost of a tp phase (one range
/// and one all-reduce), `c_range + a`, not its split (ADR 0078); `c_step` shares the single's one
/// range with `c_range`, hence the wider tolerance.
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
    let per_phase = c.c_range_graph_s + c.allreduce_latency_s.unwrap();
    assert!(rel(per_phase, 1e-4 + 1.2e-5) < 0.05, "{c:?}");
    assert!(rel(c.c_step_s, 8e-4) < 0.10, "{c:?}");
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

/// ADR 0080: the routed slots' activation bytes per token and MoE layer. Every expert-parallel
/// rank pays all of them, a tp-2 rank its column share of the n_ff_exp part only, a dense model
/// none; graphs without `expert_slot_traffic` (fitted before cost model 4) price none.
#[test]
fn routed_slot_bytes_are_not_split_by_expert_parallel() {
    let l = moe();
    let (k, ff, embd) = (
        f64::from(l.n_expert_used),
        f64::from(l.ff_exp_dim()),
        f64::from(l.n_embd),
    );
    let full = k * (25.125 * ff + 14.25 * embd) + (8.0 + 12.0 * (k - 1.0)) * embd;
    // the two tp-2 ranks together: the n_ff_exp part once, the rest on both
    let pair = 2.0 * full - k * 25.125 * ff;
    let moe_layers = l.layers.iter().filter(|b| b.expert_bytes > 0).count() as f64;
    // per rank, summed over the phases, prefill parts
    let per_rank = |id: &str| {
        let f = features_of(&l, id, None);
        let mut by: BTreeMap<String, f64> = BTreeMap::new();
        for ph in &f.ranks.unwrap().phases {
            for p in ph.prefill_parts.as_ref().unwrap_or(&ph.parts) {
                *by.entry(p.device.clone()).or_default() +=
                    f64::from(ph.repeat) * p.expert_slot_bytes_per_token;
            }
        }
        by.into_values().collect::<Vec<_>>()
    };
    let close = |a: f64, b: f64| (a / b - 1.0).abs() < 1e-9;
    assert!(close(per_rank("single")[0], moe_layers * full));
    for r in per_rank("ep-2") {
        assert!(close(r, moe_layers * full), "{r}");
    }
    let tp = per_rank("tp-2");
    assert_eq!(tp.len(), 2);
    assert!(close(tp.iter().sum(), moe_layers * pair), "{tp:?}");
    assert!(tp.iter().all(|&r| r < moe_layers * full), "{tp:?}");
    let f = features_of(&dense(), "tp-2", None);
    assert!(f.ranks.unwrap().phases.iter().all(|ph| {
        ph.parts
            .iter()
            .all(|p| p.expert_slot_bytes_per_token == 0.0)
    }));
    // priced only on cost model 4 graphs, and then the slower
    let mut old = hw(TWO);
    old.range_costs = Some(RangeCosts::PRIOR);
    let mut new = old.clone();
    new.range_costs = Some(RangeCosts {
        expert_slot_traffic: true,
        ..RangeCosts::PRIOR
    });
    let f = features_of(&l, "ep-2", None);
    let (a, b) = (
        predict(&old, &f, 2048, 128).unwrap(),
        predict(&new, &f, 2048, 128).unwrap(),
    );
    assert!(b.pp_tok_s < a.pp_tok_s && b.tg_tok_s < a.tg_tok_s);
    let mut zero = f.clone();
    for ph in &mut zero.ranks.as_mut().unwrap().phases {
        for p in ph
            .parts
            .iter_mut()
            .chain(ph.prefill_parts.iter_mut().flatten())
        {
            p.expert_slot_bytes_per_token = 0.0;
        }
    }
    let z = predict(&new, &zero, 2048, 128).unwrap();
    assert_eq!(z.pp_tok_s.to_bits(), a.pp_tok_s.to_bits());
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
    let p = multi_cli::predict_multi(&g, &l, &file, ("abc".into(), None, None), None, "deadbeef")
        .unwrap();
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

// ---------------------------------------------------------------------------------------------
// Routing frequencies (ADR 0070, 0068 History 2026-11-19), on both simulated boxes (rule 4)

/// Counts per layer: `hot` on the first `span` experts (rank 0's under expert parallel), 1
/// elsewhere.
fn skewed(l: &ModelLayout, span: usize, hot: u64) -> Routing {
    let e = l.n_expert as usize;
    let row: Vec<u64> = (0..e).map(|i| if i < span { hot } else { 1 }).collect();
    let t = vec![row; l.n_layer as usize];
    Routing::from_counts(&t, &t).unwrap()
}

fn uniform_routing(l: &ModelLayout) -> Routing {
    let t = vec![vec![16u64; l.n_expert as usize]; l.n_layer as usize];
    Routing::from_counts(&t, &t).unwrap()
}

fn predict_with(g: &HardwareGraph, l: &ModelLayout, id: &str, r: Option<&Routing>) -> (u64, u64) {
    let c = candidate_for(g, l, &Hints::default(), PHASE, id).unwrap();
    let p = predict_candidate(g, l, &c.plan, PHASE, (2048, 128), r).unwrap();
    (p.pp_tok_s.to_bits(), p.tg_tok_s.to_bits())
}

#[test]
fn uniform_routing_reproduces_the_prediction_without_routing_exactly() {
    let l = moe();
    let u = uniform_routing(&l);
    assert!(u.prefill.iter().flatten().all(|&x| x == 1.0));
    for g in [hw(TWO), hw(FOUR)] {
        for c in enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA).candidates {
            assert_eq!(
                predict_with(&g, &l, &c.id, Some(&u)),
                predict_with(&g, &l, &c.id, None),
                "{}",
                c.id
            );
        }
    }
}

#[test]
fn skewed_routing_leaves_single_layer_tp_and_tp_pp_alone() {
    let l = moe();
    for g in [hw(TWO), hw(FOUR)] {
        let r = skewed(&l, 16, 9);
        for c in enumerate_multi_with(&g, &l, &Hints::default(), PHASE, QUOTA).candidates {
            if c.id.starts_with("ep-") {
                continue;
            }
            assert_eq!(
                predict_with(&g, &l, &c.id, Some(&r)),
                predict_with(&g, &l, &c.id, None),
                "{}",
                c.id
            );
        }
    }
}

/// Sums of the expert terms of the expert-parallel FFN phases: decode expert bytes, prefill
/// expert params (the routed rows' FLOPs), each the busiest rank's.
fn ep_expert_terms(f: &RunFeatures) -> (f64, f64) {
    let ranks = f.ranks.as_ref().unwrap();
    let (mut dec, mut pre) = (0.0, 0.0);
    for ph in &ranks.phases {
        let Some(pp) = &ph.prefill_parts else {
            continue;
        };
        let k = f64::from(ph.repeat);
        dec += k * ph
            .parts
            .iter()
            .map(|p| p.expert_bytes_per_token)
            .fold(0.0, f64::max);
        pre += k * pp
            .iter()
            .map(|p| p.expert_params_per_token)
            .fold(0.0, f64::max);
    }
    (dec, pre)
}

#[test]
fn skewed_routing_prices_expert_parallel_at_its_slowest_rank() {
    let l = moe();
    let e = l.n_expert as usize;
    for (g, n) in [(hw(TWO), 2usize), (hw(FOUR), 4)] {
        let id = format!("ep-{n}");
        let span = e / n;
        let hot = 5u64;
        let r = skewed(&l, span, hot);
        let c = candidate_for(&g, &l, &Hints::default(), PHASE, &id).unwrap();
        let topo = topology_of(&g, &l, &c.plan, PHASE).unwrap();
        let feats =
            |r: Option<&Routing>| features_multi(&l, &topo, &c.plan.knobs, None, r).unwrap();
        let (dec_u, pre_u) = ep_expert_terms(&feats(None));
        let (dec_s, pre_s) = ep_expert_terms(&feats(Some(&r)));
        let rel = |a: f64, b: f64| (a / b - 1.0).abs();
        // prefill: the busiest rank's share of the routed rows, against 1 / n
        let hot_rows = hot as f64 * span as f64;
        let share = hot_rows / (hot_rows + (e - span) as f64);
        assert!(
            rel(pre_s / pre_u, share * n as f64) < 1e-9,
            "{id}: {pre_s} {pre_u}"
        );
        // decode: the Monte Carlo of the recorded frequencies, against the uniform one
        let mc = |f: &[f64]| expected_max_local(f, l.n_expert_used as usize, n, 1000, 20_261_115);
        let flat = vec![1.0; e];
        let want = mc(&r.decode[0]) / mc(&flat);
        assert!(
            want > 1.0 && rel(dec_s / dec_u, want) < 1e-9,
            "{id}: {dec_s} {dec_u}"
        );
        // and the shape gets slower on both rates
        let (pu, tu) = predict_with(&g, &l, &id, None);
        let (ps, ts) = predict_with(&g, &l, &id, Some(&r));
        assert!(f64::from_bits(ps) < f64::from_bits(pu), "{id}");
        assert!(f64::from_bits(ts) < f64::from_bits(tu), "{id}");
    }
}

#[test]
fn a_routing_file_must_match_the_layout() {
    let l = moe();
    let (n, e) = (l.n_layer as usize, l.n_expert as usize);
    let file = |layers: usize, experts: usize| multi_cli::RoutingFile {
        model: "qwen3-30b-a3b".into(),
        n_expert: experts as u32,
        top_k: 8,
        prefill_tokens: 2048,
        decode_tokens: 128,
        prefill: vec![vec![16; experts]; layers],
        decode: vec![vec![1; experts]; layers],
    };
    let text = |f: &multi_cli::RoutingFile| serde_json::to_string(f).unwrap();
    let r = multi_cli::parse_routing(&text(&file(n, e)), &l).unwrap();
    assert_eq!((r.prefill.len(), r.decode[0].len()), (n, e));
    assert!(r.decode.iter().flatten().all(|&x| x == 1.0));
    let err = |f: &multi_cli::RoutingFile, l: &ModelLayout| {
        multi_cli::parse_routing(&text(f), l).unwrap_err()
    };
    assert!(err(&file(n - 1, e), &l).contains("layers"));
    assert!(err(&file(n, e / 2), &l).contains("experts"));
    assert!(err(&file(n, e), &dense()).contains("no experts"));
    let mut none = file(n, e);
    none.decode[3] = vec![0; e];
    assert!(err(&none, &l).contains("no selections"));
}

#[test]
fn the_predict_file_names_the_routing_it_used() {
    let g = hw(TWO);
    let l = moe();
    let file = multi_cli::candidates_multi(&g, &l);
    let r = skewed(&l, 64, 3);
    let hashes = || ("abc".to_string(), None, None);
    let with =
        multi_cli::predict_multi(&g, &l, &file, hashes(), Some((&r, "f00d".into())), "x").unwrap();
    let without = multi_cli::predict_multi(&g, &l, &file, hashes(), None, "x").unwrap();
    assert_eq!(with.routing_sha256.as_deref(), Some("f00d"));
    assert!(
        !serde_json::to_string(&without)
            .unwrap()
            .contains("routing_sha256")
    );
    let t = |p: &multi_cli::PredictFile, id: &str| {
        p.candidates.iter().find(|c| c.id == id).unwrap().t_s
    };
    assert!(t(&with, "ep-2") > t(&without, "ep-2"));
    assert_eq!(
        t(&with, "single").to_bits(),
        t(&without, "single").to_bits()
    );
}

/// ADR 0078: with `tp-2` and `tp-4` anchors the all-reduces span two group sizes, and the fit
/// gains the latency per ring step `b`, which only the difference between the two identifies.
#[test]
fn a_tp4_anchor_fits_the_latency_per_ring_step() {
    let mut prior = hw(FOUR);
    prior.range_costs = Some(RangeCosts {
        c_range_graph_s: 1e-9,
        ..RangeCosts::PRIOR
    });
    let mut truth = prior.clone();
    truth.range_costs = Some(RangeCosts {
        c_range_graph_s: 1e-9,
        c_step_s: 8e-4,
        allreduce_latency_s: Some(6e-6),
        allreduce_step_s: Some(3e-6),
        ..RangeCosts::PRIOR
    });
    let l = dense();
    let rows_of = |ids: &[&str]| {
        let mut rows = Vec::new();
        for id in ids {
            let c = candidate_for(&truth, &l, &Hints::default(), PHASE, id).unwrap();
            let topo = topology_of(&truth, &l, &c.plan, PHASE).unwrap();
            for limit in [16u32, 32] {
                let f = features_multi(&l, &topo, &c.plan.knobs, Some(limit), None).unwrap();
                let p = predict(&truth, &f, 2048, 128).unwrap();
                rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                    "n_prompt": 2048, "n_gen": 0, "avg_ts": p.pp_tok_s}));
                rows.push(serde_json::json!({"plan_id": id, "layer_limit": limit,
                    "n_prompt": 0, "n_gen": 128, "avg_ts": p.tg_tok_s}));
            }
        }
        serde_json::to_string(&rows).unwrap()
    };
    let two =
        multi_cli::calibrate(&prior, &[(l.clone(), vec![rows_of(&["single", "tp-2"])])]).unwrap();
    assert_eq!(two.hw.range_costs_or_prior().allreduce_step_s, None);
    let four = multi_cli::calibrate(
        &prior,
        &[(l.clone(), vec![rows_of(&["single", "tp-2", "tp-4"])])],
    )
    .unwrap();
    assert_eq!(four.anchors.len(), 6);
    let b = four
        .hw
        .range_costs_or_prior()
        .allreduce_step_s
        .expect("fitted");
    assert!(b > 1e-6, "{b}: the prior is 1 us");
    // the full tp-4 decode, which no anchor is, lands near the truth
    let tg = |g: &HardwareGraph| {
        let c = candidate_for(g, &l, &Hints::default(), PHASE, "tp-4").unwrap();
        predict_candidate(g, &l, &c.plan, PHASE, (2048, 128), None)
            .unwrap()
            .tg_tok_s
    };
    let err = |g: &HardwareGraph| (tg(g) / tg(&truth) - 1.0).abs();
    assert!(err(&four.hw) < 0.02, "{} {}", tg(&four.hw), tg(&truth));
    assert!(
        err(&four.hw) < err(&two.hw),
        "{} {}",
        err(&four.hw),
        err(&two.hw)
    );
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

//! Cost model 6 (ADR 0092), step 5: the reader's copy rate is one thread's measured rate. Rule 4
//! on every simulated fixture with a disk, and the in-sample re-pricing of
//! `stream-predict-20261124` (arm D2 is the done line, arm R is report only).
//!
//! Hardware label: none (the graphs are committed fixtures).

mod v6_common;

use tetraeder_costmodel::predict::ReadPath;
use tetraeder_costmodel::predict::{RunFeatures, predict};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::score::{BUILDER_DISK, TierOutcome, builder_features_tier, builder_run};
use tetraeder_planner::{RoutingSource, Workload, read_tier_replay, single_node_runs};
use tetraeder_profiler::HardwareGraph;
use v6_common::{features, json, layout_of, outcome, session};

// ---- rule 4: every simulated hardware fixture that has a disk

const DISK_FIXTURES: [(&str, &str); 4] = [
    (
        "costmodel/two_gpu_cpu_nvme",
        include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    ),
    (
        "analysis/rtx3090_box",
        include_str!("../../analysis/fixtures/rtx3090_box.toml"),
    ),
    (
        "planner/rtx3090_prior",
        include_str!("../fixtures/rtx3090_prior.toml"),
    ),
    (
        "profiler/two_gpu_host",
        include_str!("../../profiler/fixtures/two_gpu_host.toml"),
    ),
];

/// Decode tokens per second of the reader path, `None` when the graph cannot price the run.
fn reader_tg(hw: &HardwareGraph, o: &TierOutcome) -> Option<f64> {
    let l = fixtures::moe();
    let f = features(&l, o).ok()?;
    predict(hw, &f, 512, 64).ok().map(|p| p.tg_tok_s)
}

/// A graph whose GPU is the faster side for expert work: moving routed pairs to it cannot cost
/// time, which the share monotonicity needs.
fn gpu_is_faster(hw: &HardwareGraph) -> bool {
    let (Some(g), Some(c)) = (hw.device("gpu0"), hw.device("cpu0")) else {
        return false;
    };
    let (Some(gc), Some(gb), Some((cf, cb))) =
        (g.compute, g.bandwidth_bytes_per_s, c.cpu_rates(12))
    else {
        return false;
    };
    gc.flops_per_s >= cf && gb.value >= cb
}

#[test]
fn on_every_disk_fixture_more_share_copy_rate_and_ram_slots_never_slow_decode() {
    let l = fixtures::moe();
    let mut checked = 0;
    for (name, toml) in DISK_FIXTURES {
        let hw = HardwareGraph::from_toml(toml).unwrap();
        if !gpu_is_faster(&hw) || reader_tg(&hw, &outcome(32, 96, 0.3, None)).is_none() {
            eprintln!("skip {name}: no GPU, CPU and priced disk");
            continue;
        }
        checked += 1;
        let mut last = 0.0;
        for share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let t = reader_tg(&hw, &outcome(32, 96, share, None)).unwrap();
            assert!(
                t >= last * (1.0 - 1e-12),
                "{name}: share {share}: {t} < {last}"
            );
            last = t;
        }
        let mut last = 0.0;
        for copy in [1e9, 2e9, 4e9, 8e9, 16e9] {
            let mut g = hw.clone();
            g.devices
                .iter_mut()
                .find(|d| d.id == "cpu0")
                .unwrap()
                .copy_bytes_per_s = Some(copy);
            let t = reader_tg(&g, &outcome(32, 96, 0.3, None)).unwrap();
            assert!(
                t >= last * (1.0 - 1e-12),
                "{name}: copy {copy}: {t} < {last}"
            );
            last = t;
        }
        let mut last = 0.0;
        for ram in [30, 60, 120] {
            let o = tetraeder_planner::uniform_tier(&l, &[(8, ram)]).remove(0);
            let t = reader_tg(&hw, &o).unwrap();
            assert!(t >= last * (1.0 - 1e-12), "{name}: ram {ram}: {t} < {last}");
            last = t;
        }
        // a tier-less run and a zero-byte tier keep their numbers (cost model 5)
        let mut zero = outcome(32, 96, 0.3, None);
        zero.decode.sync_miss_bytes_per_token = 0.0;
        zero.decode.prefetch_bytes_per_token = 0.0;
        zero.prefill.disk_bytes_per_token = 0.0;
        let f = features(&l, &zero).unwrap();
        let base = RunFeatures {
            tier: None,
            ..f.clone()
        };
        assert_eq!(
            predict(&hw, &f, 512, 64).unwrap().tg_tok_s,
            predict(&hw, &base, 512, 64).unwrap().tg_tok_s,
            "{name}"
        );
    }
    assert!(checked >= 1, "no fixture priced the disk");
}

/// ADR 0092 section 6, the done line: arm D2 decode, priced from the committed frozen graph
/// (prior rates), the committed replay and the 8 GB/s copy prior, is within 0.15 of the measured
/// median. Labelled in sample: the session chose the forms, nothing was fitted.
#[test]
fn arm_d2_decode_is_within_015_in_sample() {
    let Some(dir) = session() else { return };
    let p = dir.join("predictions/d2");
    let hw =
        HardwareGraph::from_toml(&std::fs::read_to_string(p.join("hw-D2.toml")).unwrap()).unwrap();
    let replay = read_tier_replay(&p.join("replay-D2.json")).unwrap();
    let w = Workload {
        prompt: 2048,
        r#gen: 128,
    };
    let (runs, _) =
        single_node_runs(&hw, &layout_of(&dir), &w, Some((16, 52)), Some(&replay)).unwrap();
    let pick = runs.iter().find(|r| r.ubatch == 2048).expect("ubatch 2048");
    assert_eq!(pick.routing, RoutingSource::Trace);
    let errors = json(&dir.join("errors.json"));
    let m = &errors["D2/D2-trace"];
    let (pp_meas, tg_meas) = (
        m["pp_meas"].as_f64().unwrap(),
        m["tg_meas"].as_f64().unwrap(),
    );
    let (e_pp, e_tg) = (pick.pp_tps / pp_meas - 1.0, pick.tg_tps / tg_meas - 1.0);
    eprintln!(
        "IN_SAMPLE D2 trace v6: pp {:.3} tok/s (meas {pp_meas:.3}, e {e_pp:+.3}), tg {:.4} tok/s (meas {tg_meas:.4}, e {e_tg:+.3}), binding {}, tg {:?}",
        pick.pp_tps, pick.tg_tps, pick.binding, pick.tg_breakdown
    );
    assert!(e_tg.abs() <= 0.15, "arm D2 decode error {e_tg:+.3}");
}

/// Report only: arm R priced at the routing its own rounds measured (`vram_pair_share`, its IO
/// counters), the frozen graph `hw-R.toml`, uncalibrated.
#[test]
fn arm_r_is_reported_at_its_measured_routing() {
    let Some(dir) = session() else { return };
    let hw = HardwareGraph::from_toml(
        &std::fs::read_to_string(dir.join("predictions/hw-R.toml")).unwrap(),
    )
    .unwrap();
    let files: Vec<serde_json::Value> = (1..=3)
        .map(|k| json(&dir.join(format!("arms/exit-R-r{k}.json"))))
        .collect();
    let run = builder_run(&files);
    let o = run
        .measured_outcome(32, 96)
        .expect("arm R measured a pair share");
    let f = builder_features_tier(
        &layout_of(&dir),
        32,
        &o,
        (61, 2048, true),
        ReadPath::Reader,
        (16, 16),
        BUILDER_DISK,
    )
    .unwrap();
    let p = predict(&hw, &f, 2048, 128).unwrap();
    let (pp_meas, tg_meas) = (run.pp_tok_s.unwrap(), run.tg_tok_s.unwrap());
    eprintln!(
        "IN_SAMPLE R measured v6: pp {:.2} tok/s (meas {pp_meas:.2}, e {:+.3}), tg {:.3} tok/s (meas {tg_meas:.3}, e {:+.3}), pair share {:.3}",
        p.pp_tok_s,
        p.pp_tok_s / pp_meas - 1.0,
        p.tg_tok_s,
        p.tg_tok_s / tg_meas - 1.0,
        o.decode.vram_share
    );
    assert!(p.pp_tok_s.is_finite() && p.tg_tok_s.is_finite());
}

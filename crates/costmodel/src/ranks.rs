//! Hand-computed checks of the multi-rank step time (ADR 0070).

use crate::allreduce_time;
use crate::predict::fixture::features;
use crate::predict::{Coll, DevicePart, Phase, RankFeatures, RunFeatures, predict};
use tetraeder_profiler::{
    CollectiveCurve, CollectiveOp, ComputeSpec, Device, DeviceKind, HardwareGraph, Link, LinkKind,
    Measurement, RangeCosts, Tier,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 * b.abs().max(1e-12), "{a} vs {b}");
}

fn graph() -> HardwareGraph {
    let gpu = |id: &str| {
        let mut d = Device::new(id, DeviceKind::Gpu, "g", Tier::Vram, 24 << 30);
        d.compute = Some(ComputeSpec {
            flops_per_s: 1e13,
            c0_graph_s: 1e-6,
            c0_eager_s: 4e-6,
            expert_flops_per_s: None,
        });
        d.bandwidth_bytes_per_s = Some(Measurement::prior(1e11));
        d
    };
    HardwareGraph {
        devices: vec![gpu("gpu0"), gpu("gpu1")],
        links: vec![Link {
            id: "staged".into(),
            a: "gpu0".into(),
            b: "gpu1".into(),
            kind: LinkKind::HostStaged,
            alpha_s: Measurement::prior(1e-5),
            beta_bytes_per_s: Measurement::prior(1e10),
            lane_group: None,
        }],
        ..Default::default()
    }
}

fn part(dev: &str, bytes: f64) -> DevicePart {
    DevicePart {
        device: dev.into(),
        weight_bytes_per_token: bytes,
        params_per_token: 0.0,
        resident_matmul_bytes: bytes,
        ops_per_token: 10.0,
        kv_bytes_per_ctx_token: 0.0,
        attn_flops_per_ctx_token: 0.0,
        expert_bytes_per_token: 0.0,
        expert_params_per_token: 0.0,
    }
}

fn group() -> Vec<String> {
    vec!["gpu0".to_string(), "gpu1".to_string()]
}

fn two_phase() -> RunFeatures {
    let a = Phase {
        repeat: 3,
        // uneven parts: the slower rank sets the phase time
        parts: vec![part("gpu0", 1e9), part("gpu1", 2e9)],
        prefill_parts: None,
        ranges_per_rank: 1,
        after: Some(Coll::AllReduce {
            group: group(),
            bytes_per_token: 1000.0,
        }),
        out_rows: false,
    };
    let b = Phase {
        repeat: 1,
        parts: vec![part("gpu0", 1e9), part("gpu1", 1e9)],
        prefill_parts: None,
        ranges_per_rank: 1,
        after: None,
        out_rows: false,
    };
    RunFeatures {
        parts: vec![part("gpu0", 1e9), part("gpu1", 1e9)],
        ranks: Some(RankFeatures {
            phases: vec![a, b],
            step_fixed: true,
        }),
        ..features(10.0, 0.0)
    }
}

#[test]
fn two_phase_decode_matches_the_formula() {
    let hw = graph();
    let p = predict(&hw, &two_phase(), 512, 128).unwrap();
    let c = RangeCosts::PRIOR;
    let slow = 1e-6 * 10.0 + 2e9 / 1e11; // gpu1 of phase a
    let b = 1e-6 * 10.0 + 1e9 / 1e11;
    let ar = allreduce_time(2, 1e-5, 1e10, 1000.0);
    let t = 3.0 * (slow + c.c_range_graph_s + ar) + (b + c.c_range_graph_s) + c.c_step_s;
    close(p.tg_tok_s, 1.0 / t);
    let term = |l: &str| p.tg_breakdown.iter().find(|e| e.0 == l).unwrap().1;
    close(term("ranges"), 4.0 * c.c_range_graph_s);
    close(term("allreduce:gpu0+gpu1"), 3.0 * ar);
    close(term("step"), c.c_step_s);
    assert_eq!(p.binding, "compute:gpu1");
}

#[test]
fn prefill_collective_bytes_scale_with_the_ubatch_and_c_range_follows_the_mode() {
    let mut hw = graph();
    hw.range_costs = Some(RangeCosts {
        c_range_graph_s: 1e-5,
        c_range_eager_s: 5e-5,
        c_step_s: 1e-3,
        allreduce_latency_share: 1.0,
    });
    let mut f = two_phase();
    let a = predict(&hw, &f, 1024, 8).unwrap();
    // two ubatches of 512: the all-reduce moves 512 * 1000 bytes each time
    let ar = allreduce_time(2, 1e-5, 1e10, 512.0 * 1000.0);
    let got = a.pp_breakdown.iter().find(|e| e.0 == "allreduce:gpu0+gpu1");
    close(got.unwrap().1, 2.0 * 3.0 * ar);
    f.graphs = false;
    let e = predict(&hw, &f, 1024, 8).unwrap();
    let r =
        |p: &crate::predict::Prediction| p.tg_breakdown.iter().find(|x| x.0 == "ranges").unwrap().1;
    close(r(&a), 4.0 * 1e-5);
    close(r(&e), 4.0 * 5e-5);
}

#[test]
fn a_measured_curve_replaces_the_ring_prior() {
    let mut hw = graph();
    hw.collectives = vec![CollectiveCurve {
        op: CollectiveOp::AllReduce,
        ranks: group(),
        comm: "internal".into(),
        points: vec![(1.0, 5e-5), (1e12, 5e-5)],
    }];
    let p = predict(&hw, &two_phase(), 512, 128).unwrap();
    let t = p
        .tg_breakdown
        .iter()
        .find(|e| e.0 == "allreduce:gpu0+gpu1")
        .unwrap()
        .1;
    close(t, 3.0 * 5e-5);
}

#[test]
fn a_copy_without_a_link_is_an_error() {
    let mut hw = graph();
    hw.links.clear();
    let mut f = two_phase();
    if let Some(r) = f.ranks.as_mut() {
        r.phases[1].after = Some(Coll::Copy {
            pairs: vec![("gpu0".into(), "gpu1".into())],
            bytes_per_token: 100.0,
            per_step: false,
        });
        r.phases[0].after = None;
    }
    assert!(predict(&hw, &f, 512, 128).is_err());
    hw.links = graph().links;
    assert!(predict(&hw, &f, 512, 128).is_ok());
}

#[test]
fn the_logit_readback_is_per_step() {
    let hw = graph();
    let mut f = two_phase();
    if let Some(r) = f.ranks.as_mut() {
        r.phases[1].after = Some(Coll::Copy {
            pairs: vec![("gpu0".into(), "gpu1".into())],
            bytes_per_token: 1e6,
            per_step: true,
        });
    }
    let p = predict(&hw, &f, 1024, 8).unwrap();
    let per = |v: &[(String, f64)]| v.iter().find(|e| e.0 == "copy:gpu0-gpu1").unwrap().1;
    // one output per step: the same bytes in decode and in each of the two prefill steps
    close(per(&p.tg_breakdown), 1e-5 + 1e6 / 1e10);
    close(per(&p.pp_breakdown), 2.0 * (1e-5 + 1e6 / 1e10));
}

// ---- MODEL_VERSION 2 (ADR 0070, 2026-11-18)

/// One rank, one phase: `p` on gpu0.
fn one_phase(p: DevicePart, out_rows: bool) -> RunFeatures {
    RunFeatures {
        parts: vec![p.clone()],
        ranks: Some(RankFeatures {
            phases: vec![Phase {
                repeat: 1,
                parts: vec![p],
                prefill_parts: None,
                ranges_per_rank: 0,
                after: None,
                out_rows,
            }],
            step_fixed: false,
        }),
        ..features(10.0, 0.0)
    }
}

/// Seconds of a 512-token prompt (one ubatch).
fn prefill_s(hw: &HardwareGraph, f: &RunFeatures) -> f64 {
    512.0 / predict(hw, f, 512, 8).unwrap().pp_tok_s
}

fn with_expert_rate(f_exp: Option<f64>) -> HardwareGraph {
    let mut hw = graph();
    for d in &mut hw.devices {
        d.compute.as_mut().unwrap().expert_flops_per_s = f_exp;
    }
    hw
}

#[test]
fn the_expert_rate_prices_expert_matmuls_only() {
    let mut p = part("gpu0", 1e8);
    p.params_per_token = 2e8;
    p.expert_bytes_per_token = 4e8;
    p.expert_params_per_token = 1e8;
    let f = one_phase(p.clone(), false);
    // dense and experts compute bound at 512 rows, the experts at their own rate
    let dense = 2.0 * 2e8 * 512.0 / 1e13;
    close(
        prefill_s(&with_expert_rate(Some(2.5e12)), &f),
        1e-6 * 10.0 + dense + 2.0 * 1e8 * 512.0 / 2.5e12,
    );
    // without a fitted rate the experts run at F
    close(
        prefill_s(&with_expert_rate(None), &f),
        1e-6 * 10.0 + dense + 2.0 * 1e8 * 512.0 / 1e13,
    );
    // a part without expert work does not see the rate
    let mut dense_only = p;
    dense_only.expert_bytes_per_token = 0.0;
    dense_only.expert_params_per_token = 0.0;
    let f = one_phase(dense_only, false);
    assert_eq!(
        prefill_s(&with_expert_rate(Some(2.5e12)), &f).to_bits(),
        prefill_s(&graph(), &f).to_bits()
    );
}

#[test]
fn each_op_class_has_its_own_roofline() {
    // dense compute bound at 512 rows, experts bound by their bytes: the times add
    let mut p = part("gpu0", 1e8);
    p.params_per_token = 2e8;
    p.expert_bytes_per_token = 4e9;
    p.expert_params_per_token = 1e6;
    let t = prefill_s(&graph(), &one_phase(p, false));
    let dense = 2.0 * 2e8 * 512.0 / 1e13;
    close(t, 1e-6 * 10.0 + dense + 4e9 / 1e11);
    // one roofline over the sums would hide the expert bytes behind the dense FLOPs
    let mixed = (dense + 2.0 * 1e6 * 512.0 / 1e13).max((1e8 + 4e9) / 1e11);
    assert!(t > 1e-6 * 10.0 + mixed);
}

#[test]
fn an_output_row_phase_computes_one_row_per_prefill_step() {
    // the head: 1e9 bytes of weights, 2e9 params
    let mut head = part("gpu0", 1e9);
    head.params_per_token = 2e9;
    let all = prefill_s(&graph(), &one_phase(head.clone(), false));
    close(all, 1e-6 * 10.0 + 2.0 * 2e9 * 512.0 / 1e13);
    // one row: bound by reading the weights once
    let out = prefill_s(&graph(), &one_phase(head.clone(), true));
    close(out, 1e-6 * 10.0 + (2.0 * 2e9 / 1e13f64).max(1e9 / 1e11));
    // decode is one row either way
    let tg = |o: bool| {
        predict(&graph(), &one_phase(head.clone(), o), 512, 8)
            .unwrap()
            .tg_tok_s
    };
    assert_eq!(tg(true).to_bits(), tg(false).to_bits());
}

#[test]
fn the_allreduce_pays_its_latency_share_and_keeps_its_bytes() {
    let ar = |hw: &HardwareGraph| {
        let p = predict(hw, &two_phase(), 512, 128).unwrap();
        let t = |v: &[(String, f64)]| v.iter().find(|e| e.0 == "allreduce:gpu0+gpu1").unwrap().1;
        (t(&p.tg_breakdown), t(&p.pp_breakdown))
    };
    let share = |s: f64, with_curve: bool| {
        let mut hw = graph();
        if with_curve {
            hw.collectives = vec![CollectiveCurve {
                op: CollectiveOp::AllReduce,
                ranks: group(),
                comm: "nccl".into(),
                points: vec![(4096.0, 3e-5), (1e6, 1e-4)],
            }];
        }
        hw.range_costs = Some(RangeCosts {
            allreduce_latency_share: s,
            ..RangeCosts::PRIOR
        });
        ar(&hw)
    };
    // identity at 1: the curve as measured (1000 bytes is below its first point)
    let (tg1, pp1) = share(1.0, true);
    close(tg1, 3.0 * 3e-5);
    // share 0.25: each all-reduce sheds 0.75 of the zero-byte time, decode and prefill alike
    let (tg, pp) = share(0.25, true);
    close(tg, 3.0 * 0.25 * 3e-5);
    close(pp, pp1 - 3.0 * 0.75 * 3e-5);
    // without a curve the ring prior's latency 2 (p - 1) alpha is the zero-byte time
    let (tg, _) = share(0.5, false);
    close(
        tg,
        3.0 * (allreduce_time(2, 1e-5, 1e10, 1000.0) - 0.5 * 2.0 * 1e-5),
    );
}

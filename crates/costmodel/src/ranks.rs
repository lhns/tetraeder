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
    };
    let b = Phase {
        repeat: 1,
        parts: vec![part("gpu0", 1e9), part("gpu1", 1e9)],
        prefill_parts: None,
        ranges_per_rank: 1,
        after: None,
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

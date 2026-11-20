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
        expert_slot_bytes_per_token: 0.0,
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
            prefill_micro_batches: 1,
            model: None,
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
        allreduce_latency_s: None,
        allreduce_step_s: None,
        expert_slot_traffic: false,
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
            prefill_micro_batches: 1,
            model: None,
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

/// ADR 0078: with a fitted latency `a` (and step latency `b`) an all-reduce over `p` ranks in a
/// step costs the curve's byte part plus `a + 2 (p - 1) b`, whatever the probe's zero-byte time
/// of that group was; the share is then not used. Without `b` every group pays `a`.
#[test]
fn the_allreduce_pays_its_in_step_latency_and_keeps_its_bytes() {
    let ids = |n: usize| (0..n).map(|i| format!("gpu{i}")).collect::<Vec<_>>();
    let mut hw = graph();
    hw.collectives = vec![
        CollectiveCurve {
            op: CollectiveOp::AllReduce,
            ranks: ids(2),
            comm: "nccl".into(),
            points: vec![(4096.0, 4e-5), (1e6, 2e-4)],
        },
        CollectiveCurve {
            op: CollectiveOp::AllReduce,
            ranks: ids(4),
            comm: "nccl".into(),
            points: vec![(4096.0, 3e-5), (1e6, 3e-4)],
        },
    ];
    let t = |hw: &HardwareGraph, p: usize, m: f64| {
        let c = Coll::AllReduce {
            group: ids(p),
            bytes_per_token: 1000.0,
        };
        crate::predict::coll_time(hw, &c, m).unwrap().1
    };
    let bytes = |a: f64, b: f64| a + (b - a) * (512e3 - 4096.0) / (1e6 - 4096.0);
    // `a` alone: below the first point only the latency is left, the same on two and four ranks
    hw.range_costs = Some(RangeCosts {
        allreduce_latency_share: 0.25,
        allreduce_latency_s: Some(1.2e-5),
        ..RangeCosts::PRIOR
    });
    close(t(&hw, 2, 1.0), 1.2e-5);
    close(t(&hw, 4, 1.0), 1.2e-5);
    // the byte part stays as measured
    close(t(&hw, 4, 512.0), bytes(3e-5, 3e-4) - 3e-5 + 1.2e-5);
    // with `b`: 2 ring steps on two ranks, 6 on four
    hw.range_costs = Some(RangeCosts {
        allreduce_latency_share: 0.25,
        allreduce_latency_s: Some(1.2e-5),
        allreduce_step_s: Some(1e-6),
        ..RangeCosts::PRIOR
    });
    close(t(&hw, 2, 1.0), 1.2e-5 + 2e-6);
    close(t(&hw, 4, 1.0), 1.2e-5 + 6e-6);
    close(t(&hw, 2, 512.0), bytes(4e-5, 2e-4) - 4e-5 + 1.4e-5);
}

/// ADR 0080 History: on graphs of cost model 4 a curve whose head point is not below its
/// successor (the probe's floor) reads the head as the next segment extended back; a rising
/// head and graphs before cost model 4 read the curve as measured.
#[test]
fn a_floor_head_takes_the_next_segment_in_cost_model_4() {
    let ids: Vec<String> = (0..4).map(|i| format!("gpu{i}")).collect();
    let floor = vec![(4096.0, 7.8e-5), (16384.0, 7.7e-5), (65536.0, 1.03e-4)];
    let rising = vec![(4096.0, 6.4e-5), (16384.0, 7.6e-5), (65536.0, 1.03e-4)];
    let t = |points: &[(f64, f64)], slots: bool, bytes: f64| {
        let mut hw = graph();
        hw.collectives = vec![CollectiveCurve {
            op: CollectiveOp::AllReduce,
            ranks: ids.clone(),
            comm: "nccl".into(),
            points: points.to_vec(),
        }];
        hw.range_costs = Some(RangeCosts {
            allreduce_latency_s: Some(3e-6),
            allreduce_step_s: Some(4e-6),
            expert_slot_traffic: slots,
            ..RangeCosts::PRIOR
        });
        let c = Coll::AllReduce {
            group: ids.clone(),
            bytes_per_token: bytes,
        };
        crate::predict::coll_time(&hw, &c, 1.0).unwrap().1
    };
    let lat = 3e-6 + 6.0 * 4e-6;
    let slope = (1.03e-4 - 7.7e-5) / (65536.0 - 16384.0);
    let head = 7.7e-5 - slope * (16384.0 - 4096.0);
    // 8 KiB and 20 KiB, the decode all-reduces of n_embd 2048 and 5120
    for bytes in [8192.0, 20480.0] {
        let on_line = 7.7e-5 - slope * (16384.0 - bytes);
        close(t(&floor, true, bytes), on_line - head + lat);
        let flat = if bytes < 16384.0 {
            0.0
        } else {
            7.7e-5 + slope * (bytes - 16384.0) - 7.8e-5
        };
        close(t(&floor, false, bytes), flat.max(0.0) + lat);
        assert_eq!(
            t(&rising, true, bytes).to_bits(),
            t(&rising, false, bytes).to_bits()
        );
    }
}

/// A `layer-2` plan: two equal one-rank stages and a hop of `n_embd x 4` bytes per token.
fn layer2(micro: u32) -> RunFeatures {
    let stage = |dev: &str, after: Option<Coll>| {
        let mut p = part(dev, 1e8);
        p.params_per_token = 1e8;
        Phase {
            repeat: 1,
            parts: vec![p],
            prefill_parts: None,
            ranges_per_rank: 0,
            after,
            out_rows: false,
        }
    };
    let hop = Coll::Copy {
        pairs: vec![("gpu0".into(), "gpu1".into())],
        bytes_per_token: 20480.0,
        per_step: false,
    };
    RunFeatures {
        parts: vec![part("gpu0", 1e8), part("gpu1", 1e8)],
        ranks: Some(RankFeatures {
            phases: vec![stage("gpu0", Some(hop)), stage("gpu1", None)],
            step_fixed: false,
            prefill_micro_batches: micro,
            model: None,
        }),
        ..features(10.0, 0.0)
    }
}

/// ADR 0077: with M chunks the stages overlap (GPipe). Two stages of chunk time t and a hop h:
/// one ubatch per step costs `4 (2 t + h)` for 2048 tokens, four chunks in one step
/// `(4 + 2 - 1) t + (2 - 1) h`, the pipeline fill being one chunk time on each end stage.
#[test]
fn pipelined_prefill_overlaps_the_stages() {
    let hw = graph();
    let t = 1e-6 * 10.0 + 2.0 * 1e8 * 512.0 / 1e13;
    let h = 1e-5 + 512.0 * 20480.0 / 1e10;
    let pp_s = |m: u32| 2048.0 / predict(&hw, &layer2(m), 2048, 8).unwrap().pp_tok_s;
    let (serial, piped) = (pp_s(1), pp_s(4));
    close(serial, 4.0 * (2.0 * t + h));
    close(piped, 5.0 * t + h);
    close(serial - piped, 3.0 * (t + h));
    // decode does not pipeline
    let tg = |m: u32| predict(&hw, &layer2(m), 2048, 8).unwrap().tg_tok_s;
    assert_eq!(tg(1).to_bits(), tg(4).to_bits());
    // one stage has nothing to pipeline
    let mut one = layer2(4);
    if let Some(r) = one.ranks.as_mut() {
        r.phases[0].after = None;
        r.phases.truncate(1);
    }
    let mut one1 = one.clone();
    one1.ranks.as_mut().unwrap().prefill_micro_batches = 1;
    let pp = |f: &RunFeatures| predict(&hw, f, 2048, 8).unwrap().pp_tok_s;
    assert_eq!(pp(&one).to_bits(), pp(&one1).to_bits());
}

// ---- MODEL_VERSION 5 (ADR 0082): decode priced at the clock a GPU's duty gets

/// Four GPUs, no links: the phases below have no collective.
fn graph4() -> HardwareGraph {
    let mut g = graph();
    let gpu = g.devices[0].clone();
    g.devices = (0..4)
        .map(|i| Device {
            id: format!("gpu{i}"),
            ..gpu.clone()
        })
        .collect();
    g.links.clear();
    g
}

/// One-rank stages in turn; stage `i` costs `ops[i] x U` per decode token, `U = c0 + 1e-8`
/// (`c0` per op plus the routed-expert roofline at 1e3 bytes per op), all of it clock-bound.
fn stages(ops: &[f64], model: Option<&str>) -> RunFeatures {
    let mk = |i: usize| DevicePart {
        expert_bytes_per_token: ops[i] * 1e3,
        ops_per_token: ops[i],
        ..part(&format!("gpu{i}"), 0.0)
    };
    let phases = (0..ops.len())
        .map(|i| Phase {
            repeat: 1,
            parts: vec![mk(i)],
            prefill_parts: None,
            ranges_per_rank: 0,
            after: None,
            out_rows: false,
        })
        .collect();
    RunFeatures {
        parts: (0..ops.len()).map(mk).collect(),
        ranks: Some(RankFeatures {
            phases,
            step_fixed: false,
            prefill_micro_batches: 1,
            model: model.map(str::to_string),
        }),
        ..features(10.0, 0.0)
    }
}

fn clocks(low: f64, busy: &[(&str, f64)]) -> tetraeder_profiler::DecodeClocks {
    tetraeder_profiler::DecodeClocks {
        low_mhz: low,
        busy_mhz: busy.iter().map(|(m, x)| (m.to_string(), *x)).collect(),
        duty_threshold: 0.32,
    }
}

const U: f64 = 1e-6 + 1e-8;

#[test]
fn four_low_duty_gpus_decode_at_the_factor_and_prefill_is_unchanged() {
    let mut hw = graph4();
    let f = stages(&[10.0; 4], Some("m"));
    let base = predict(&hw, &f, 512, 128).unwrap();
    assert!(base.decode_clocks.is_empty());
    hw.decode_clocks = Some(clocks(1695.0, &[("m", 1905.0)]));
    let p = predict(&hw, &f, 512, 128).unwrap();
    let k = 1905.0 / 1695.0;
    close(1.0 / p.tg_tok_s, 40.0 * U * k);
    assert_eq!(p.pp_tok_s.to_bits(), base.pp_tok_s.to_bits());
    assert_eq!(p.decode_clocks.len(), 4);
    for (i, c) in p.decode_clocks.iter().enumerate() {
        assert_eq!(c.device, format!("gpu{i}"));
        close(c.duty, 0.25);
        close(c.factor, k);
        assert_eq!(c.mhz, 1695.0);
    }
}

#[test]
fn only_the_gpus_under_the_threshold_are_scaled() {
    let mut hw = graph4();
    hw.decode_clocks = Some(clocks(1695.0, &[("m", 1905.0)]));
    let p = predict(&hw, &stages(&[60.0, 10.0, 10.0, 10.0], Some("m")), 512, 128).unwrap();
    let k = 1905.0 / 1695.0;
    close(1.0 / p.tg_tok_s, 60.0 * U + 30.0 * U * k);
    let c = &p.decode_clocks;
    close(c[0].duty, 60.0 / 90.0);
    assert_eq!((c[0].factor, c[0].mhz), (1.0, 1905.0));
    close(c[1].duty, 10.0 / 90.0);
    close(c[1].factor, k);
    assert_eq!(c[1].mhz, 1695.0);
}

#[test]
fn a_gpu_at_duty_one_third_keeps_the_busy_clock() {
    let mut hw = graph4();
    let f = stages(&[10.0; 3], Some("m"));
    let base = predict(&hw, &f, 512, 128).unwrap();
    hw.decode_clocks = Some(clocks(1695.0, &[("m", 1905.0)]));
    let p = predict(&hw, &f, 512, 128).unwrap();
    assert_eq!(p.tg_tok_s.to_bits(), base.tg_tok_s.to_bits());
    assert!(p.decode_clocks.iter().all(|c| c.factor == 1.0));
    close(p.decode_clocks[0].duty, 1.0 / 3.0);
}

#[test]
fn no_gap_a_missing_model_or_no_model_leave_every_bit() {
    let base = predict(&graph4(), &stages(&[10.0; 4], Some("m")), 512, 128).unwrap();
    for (low, busy, model) in [
        (1695.0, 1600.0, Some("m")),
        (1695.0, 1695.0 + 15.0, Some("m")),
        (1695.0, 1905.0, Some("other")),
        (1695.0, 1905.0, None),
    ] {
        let mut hw = graph4();
        hw.decode_clocks = Some(clocks(low, &[("m", busy)]));
        let p = predict(&hw, &stages(&[10.0; 4], model), 512, 128).unwrap();
        assert_eq!(
            p.tg_tok_s.to_bits(),
            base.tg_tok_s.to_bits(),
            "{busy} {model:?}"
        );
        assert_eq!(p.pp_tok_s.to_bits(), base.pp_tok_s.to_bits());
    }
    // one clock step over the low clock is a gap
    let mut hw = graph4();
    hw.decode_clocks = Some(clocks(1695.0, &[("m", 1695.0 + 15.1)]));
    let p = predict(&hw, &stages(&[10.0; 4], Some("m")), 512, 128).unwrap();
    assert!(p.tg_tok_s < base.tg_tok_s);
}

#[test]
fn an_all_reduce_counts_in_the_duty_of_its_group_only() {
    // gpu0 and gpu1 all-reduce after their phase; the other two are not in the group.
    let mut f = stages(&[10.0; 4], Some("m"));
    let r = f.ranks.as_mut().unwrap();
    r.phases[0].parts.push(DevicePart {
        ops_per_token: 10.0,
        expert_bytes_per_token: 1e4,
        ..part("gpu1", 0.0)
    });
    r.phases[0].after = Some(Coll::AllReduce {
        group: vec!["gpu0".into(), "gpu1".into()],
        bytes_per_token: 1000.0,
    });
    let mut hw = graph4();
    hw.links = graph().links;
    hw.decode_clocks = Some(clocks(1695.0, &[("m", 1905.0)]));
    let p = predict(&hw, &f, 512, 128).unwrap();
    let ar = allreduce_time(2, 1e-5, 1e10, 1000.0);
    let c = &p.decode_clocks;
    // gpu1 is in two phases: gpu0's and its own; the step is the slower of the first two parts
    // plus the all-reduce, then three more phases
    let total = 40.0 * U + ar;
    close(c[0].duty, (10.0 * U + ar) / total);
    close(c[1].duty, (20.0 * U + ar) / total);
    close(c[2].duty, 10.0 * U / total);
}

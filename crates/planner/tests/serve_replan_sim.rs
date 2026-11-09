//! Replanning on simulated hardware and faults (ADR 0055, AGENTS.md rule 4). Every graph with a
//! `cpu0`, both model layouts, both engines: no fault, a CPU co-tenant, a GPU throttle and a VRAM
//! shrink; the proposals lower, fit, and move work away from the slowed device.

mod common;

use common::{graphs, has_gpu, input, scale};
use tetraeder_plan::{KvFormat, Plan};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    ModelLayout, Priors, ServeEngine, ServeInput, Workload, candidates, correct_for_drift,
    fill_from_priors, footprint, lower, pick, predict_plan, predict_plan_for, propose, recognize,
    regret, uniform_tier,
};
use tetraeder_profiler::{ComputeSpec, Device, DeviceKind, HardwareGraph, Tier};

struct Case {
    name: String,
    hw: HardwareGraph,
    layout: ModelLayout,
    engine: ServeEngine,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (g, hw) in graphs() {
        for (l, layout) in [("dense", fixtures::dense()), ("moe", fixtures::moe())] {
            let engines: &[ServeEngine] = if layout.is_moe() {
                &[ServeEngine::Llama, ServeEngine::Builder]
            } else {
                &[ServeEngine::Llama]
            };
            for &engine in engines {
                out.push(Case {
                    name: format!("{g}/{l}/{engine:?}"),
                    hw: hw.clone(),
                    layout: layout.clone(),
                    engine,
                });
            }
        }
    }
    out
}

fn sizing(i: &ServeInput) -> Workload {
    Workload {
        prompt: i.ctx - i.workload.r#gen,
        r#gen: i.workload.r#gen,
    }
}

fn fits(i: &ServeInput, p: &Plan) -> bool {
    let (pl, k) = recognize(p, &i.layout).unwrap();
    let f = footprint(
        &i.layout,
        pl,
        k.ubatch.unwrap(),
        &sizing(i),
        KvFormat::default(),
    );
    f.gpu <= i.budget.gpu_bytes && f.host <= i.budget.ram_bytes
}

fn request_s(hw: &HardwareGraph, i: &ServeInput, p: &Plan) -> f64 {
    predict_plan(hw, &i.layout, p, &i.workload)
        .unwrap()
        .request_s
}

#[test]
fn no_fault_keeps_the_plan() {
    for c in cases() {
        let inp = input(&c.hw, &c.layout, c.engine);
        let cur = pick(&inp).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let pred = cur
            .prediction
            .clone()
            .unwrap_or_else(|| panic!("{}: {:?}", c.name, cur.notes));
        assert!(propose(&inp, &cur.plan).is_none(), "{}", c.name);
        assert_eq!(regret(&inp, &cur.plan), Some(0.0), "{}", c.name);
        // a measurement equal to the prediction changes no pick
        if let Some(g) = correct_for_drift(&c.hw, &pred, 1.0 / pred.tg_tok_s) {
            let again = ServeInput {
                hw: g,
                ..inp.clone()
            };
            assert_eq!(pick(&again).unwrap().plan, cur.plan, "{}", c.name);
            assert!(propose(&again, &cur.plan).is_none(), "{}", c.name);
        }
    }
}

fn fault(device: &str) {
    let mut proposals = 0;
    for c in cases() {
        let inp = input(&c.hw, &c.layout, c.engine);
        let cur = pick(&inp).unwrap();
        let pred = cur.prediction.clone().unwrap();
        let truth = scale(&c.hw, device, 0.5);
        let measured = 1.0
            / predict_plan(&truth, &c.layout, &cur.plan, &common::WORK)
                .unwrap()
                .tg_tok_s;
        let Some(corrected) = correct_for_drift(&c.hw, &pred, measured) else {
            continue;
        };
        let inp2 = ServeInput {
            hw: corrected,
            ..inp.clone()
        };
        let Some(p) = propose(&inp2, &cur.plan) else {
            continue;
        };
        proposals += 1;
        let what = format!("{} fault {device}", c.name);
        assert!(lower(&p.plan, &c.layout, c.engine).is_ok(), "{what}");
        assert!(fits(&inp2, &p.plan), "{what}");
        assert!(candidates(&inp2).contains(&p.plan), "{what}");
        assert!(p.gain > 0.0 && p.gain <= 1.0, "{what}");
        let binding_device = pred.binding.split_once(':').map(|b| b.1);
        if binding_device == Some(device) {
            let (t_new, t_cur) = (
                request_s(&truth, &inp, &p.plan),
                request_s(&truth, &inp, &cur.plan),
            );
            assert!(t_new <= t_cur * 1.0001, "{what}: truth {t_new} > {t_cur}");
        }
        if device == "cpu0" {
            let share = |q: &Plan| recognize(q, &c.layout).unwrap().0.gpu_share();
            assert!(share(&p.plan) >= share(&cur.plan), "{what}");
        }
    }
    eprintln!("{device}: {proposals} proposals");
}

#[test]
fn cpu_co_tenant() {
    fault("cpu0");
}

#[test]
fn gpu_throttle() {
    fault("gpu0");
}

#[test]
fn vram_shrink() {
    for c in cases().into_iter().filter(|c| has_gpu(&c.hw)) {
        let inp = input(&c.hw, &c.layout, c.engine);
        let cur = pick(&inp).unwrap();
        let mut small = inp.clone();
        small.budget.gpu_bytes = inp.budget.gpu_bytes / 10 * 6;
        let what = &c.name;
        if fits(&small, &cur.plan) {
            // same behaviour as no fault: nothing is forced, and a switch needs a real gain
            if let Some(p) = propose(&small, &cur.plan) {
                assert!(p.gain > 0.0 && p.gain < 1.0, "{what}");
            }
        } else {
            let p = propose(&small, &cur.plan).unwrap_or_else(|| panic!("{what}: no proposal"));
            assert_eq!(p.gain, 1.0, "{what}");
            assert!(fits(&small, &p.plan), "{what}");
            assert!(lower(&p.plan, &c.layout, c.engine).is_ok(), "{what}");
        }
    }
}

#[test]
fn a_missing_gpu_forces_a_switch_to_the_cpu() {
    for c in cases().into_iter().filter(|c| has_gpu(&c.hw)) {
        let inp = input(&c.hw, &c.layout, c.engine);
        let cur = pick(&inp).unwrap();
        let cpu_only = ServeInput {
            hw: common::without_gpus(&c.hw),
            budget: tetraeder_planner::budget_of(
                &common::without_gpus(&c.hw),
                common::QUOTA,
                None,
                None,
            ),
            ..inp
        };
        let p = propose(&cpu_only, &cur.plan).unwrap_or_else(|| panic!("{}: no proposal", c.name));
        assert_eq!(p.gain, 1.0, "{}", c.name);
        assert_eq!(
            recognize(&p.plan, &c.layout).unwrap().0.gpu_share(),
            0,
            "{}",
            c.name
        );
    }
}

#[test]
fn quota_cut_forces_fewer_threads() {
    for c in cases() {
        let inp = input(&c.hw, &c.layout, c.engine);
        let cur = pick(&inp).unwrap();
        let t = cur.plan.knobs.threads.unwrap();
        if t < 2 {
            continue;
        }
        let mut cut = inp.clone();
        cut.budget.threads = t / 2;
        let p = propose(&cut, &cur.plan).unwrap_or_else(|| panic!("{}: no proposal", c.name));
        assert_eq!(p.gain, 1.0, "{}", c.name);
        assert!(p.plan.knobs.threads.unwrap() <= t / 2, "{}", c.name);
    }
}

/// The builder on the RTX 3090 prior with a (uniform stand-in) tier replay: the candidates,
/// their slot pairs, and the running plan, the fastest one the disk binds.
fn builder_with_tier() -> (ServeInput, Plan) {
    let (_, hw) = graphs()
        .into_iter()
        .find(|g| g.0 == "rtx3090_prior")
        .unwrap();
    let layout = fixtures::moe();
    let mut inp = input(&hw, &layout, ServeEngine::Builder);
    let slots_of = |p: &Plan, l: &ModelLayout| match recognize(p, l).unwrap().0 {
        tetraeder_planner::Placement::Builder {
            vram_slots,
            ram_slots,
        } => (vram_slots, ram_slots),
        other => panic!("{other:?}"),
    };
    let pairs: Vec<(u32, u32)> = candidates(&inp)
        .iter()
        .map(|p| slots_of(p, &layout))
        .collect();
    inp.tier = Some(uniform_tier(&layout, &pairs));
    let cur = candidates(&inp)
        .into_iter()
        .filter_map(|p| {
            let pred = predict_plan_for(&inp, &p).ok()?;
            (pred.binding == "read:nvme0").then_some((pred.tg_tok_s, p))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("a plan the disk binds")
        .1;
    (inp, cur)
}

fn builder_slots(inp: &ServeInput, p: &Plan) -> (u32, u32) {
    match recognize(p, &inp.layout).unwrap().0 {
        tetraeder_planner::Placement::Builder {
            vram_slots,
            ram_slots,
        } => (vram_slots, ram_slots),
        other => panic!("{other:?}"),
    }
}

fn disk_bytes(inp: &ServeInput, p: &Plan) -> f64 {
    let (v, r) = builder_slots(inp, p);
    let t = inp.tier.as_ref().unwrap();
    t.iter()
        .find(|o| o.vram_slots == v && o.ram_slots == r)
        .unwrap()
        .decode
        .disk_bytes_per_token
}

/// A slow NVMe (curve x0.25) shows in the step time as a `read:nvme0` term; the drift
/// correction scales the curve down by the measured factor, and the planner then proposes a plan
/// that reads fewer bytes from the disk (more RAM slots, or more VRAM slots).
// hardware: none
#[test]
fn a_slow_disk_moves_the_builder_plan_to_fewer_disk_bytes() {
    let (inp, cur) = builder_with_tier();
    let pred = predict_plan_for(&inp, &cur).unwrap();
    assert_eq!(pred.binding, "read:nvme0", "{:?}", pred.tg_breakdown);
    // the box's disk delivers a quarter of the curve
    let mut truth = inp.hw.clone();
    for d in &mut truth.devices {
        for q in &mut d.queue_curve {
            q.bytes_per_s *= 0.25;
        }
    }
    let slow = ServeInput {
        hw: truth,
        ..inp.clone()
    };
    let measured = 1.0 / predict_plan_for(&slow, &cur).unwrap().tg_tok_s;
    assert!(measured > 1.0 / pred.tg_tok_s * 1.2, "the drift is visible");
    let corrected = correct_for_drift(&inp.hw, &pred, measured).expect("a read binding corrects");
    let curve = |g: &HardwareGraph| g.device("nvme0").unwrap().queue_curve.clone();
    for (old, new) in curve(&inp.hw).iter().zip(curve(&corrected)) {
        assert!(new.bytes_per_s < old.bytes_per_s, "{old:?} {new:?}");
    }
    let inp2 = ServeInput {
        hw: corrected,
        ..inp.clone()
    };
    let p = propose(&inp2, &cur).expect("a proposal");
    let (v0, r0) = builder_slots(&inp, &cur);
    let (v1, r1) = builder_slots(&inp, &p.plan);
    assert!(v1 > v0 || r1 > r0, "{:?} -> {:?}", (v0, r0), (v1, r1));
    assert!(disk_bytes(&inp, &p.plan) < disk_bytes(&inp, &cur));
    assert!(p.gain > 0.0);
}

/// `read:<disk>` binding on a graph with a plain bandwidth (no curve) scales the bandwidth.
// hardware: none
#[test]
fn a_read_binding_scales_a_disk_bandwidth_without_a_curve() {
    let (inp, cur) = builder_with_tier();
    let mut hw = inp.hw.clone();
    let d = hw.devices.iter_mut().find(|d| d.id == "nvme0").unwrap();
    d.queue_curve.clear();
    d.bandwidth_bytes_per_s = Some(tetraeder_profiler::Measurement::prior(3.0e9));
    let inp = ServeInput { hw, ..inp };
    let pred = predict_plan_for(&inp, &cur).unwrap();
    assert_eq!(pred.binding, "read:nvme0");
    let t = 1.0 / pred.tg_tok_s;
    let g = correct_for_drift(&inp.hw, &pred, t * 1.5).unwrap();
    let bw = |g: &HardwareGraph| {
        g.device("nvme0")
            .unwrap()
            .bandwidth_bytes_per_s
            .unwrap()
            .value
    };
    assert!(bw(&g) < bw(&inp.hw));
}

/// Without a tier file the builder candidates tie on RAM slots and nothing changes; with one,
/// candidates without an outcome are left out of the enumeration.
// hardware: none
#[test]
fn a_tier_file_filters_candidates_without_an_outcome() {
    let (inp, _) = builder_with_tier();
    let all = candidates(&inp);
    let mut few = inp.clone();
    let keep = uniform_tier(&inp.layout, &[builder_slots(&inp, &all[0])]);
    few.tier = Some(keep);
    let some = candidates(&few);
    assert_eq!(some.len(), 1);
    assert_eq!(some[0], all[0]);
    let mut none = inp.clone();
    none.tier = None;
    assert_eq!(candidates(&none).len(), all.len());
}

#[test]
fn deterministic() {
    for c in cases() {
        let inp = input(&c.hw, &c.layout, c.engine);
        let (a, b) = (pick(&inp).unwrap(), pick(&inp).unwrap());
        assert_eq!(a.plan, b.plan, "{}", c.name);
        assert_eq!(a.prediction, b.prediction, "{}", c.name);
        let mut small = inp.clone();
        small.budget.threads = 3;
        let (x, y) = (propose(&small, &a.plan), propose(&small, &a.plan));
        assert_eq!(x.map(|p| p.plan), y.map(|p| p.plan), "{}", c.name);
    }
}

#[test]
fn fill_from_priors_names_the_gpu() {
    let mut hw = HardwareGraph::default();
    let mut gpu = Device::new(
        "gpu0",
        DeviceKind::Gpu,
        "NVIDIA GeForce RTX 3090",
        Tier::Vram,
        24 << 30,
    );
    gpu.index = Some(0);
    let mut odd = Device::new(
        "gpu1",
        DeviceKind::Gpu,
        "Some Unknown Card",
        Tier::Vram,
        8 << 30,
    );
    odd.index = Some(1);
    let mut cpu = Device::new("cpu0", DeviceKind::Cpu, "cpu", Tier::Ram, 64 << 30);
    cpu.threads = Some(8);
    hw.devices = vec![cpu, gpu, odd];
    let notes = fill_from_priors(&mut hw, &Priors::builtin());
    let g = hw.device("gpu0").unwrap();
    assert_eq!(
        g.compute,
        Some(ComputeSpec {
            flops_per_s: 7.1e13,
            c0_graph_s: 5.0e-6,
            c0_eager_s: 2.0e-5
        })
    );
    assert_eq!(g.bandwidth_bytes_per_s.unwrap().value, 9.36e11);
    assert!(hw.device("gpu1").unwrap().compute.is_none());
    assert!(
        hw.device("cpu0").unwrap().compute.is_none(),
        "CPUs are not filled"
    );
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("gpu1"));
}

#[test]
fn an_unpredictable_graph_runs_the_first_candidate_without_a_prediction() {
    // a discovered CPU has no compute spec
    let mut hw = common::graphs().pop().unwrap().1;
    hw.devices[0].compute = None;
    let inp = input(&hw, &fixtures::moe(), ServeEngine::Llama);
    let p = pick(&inp).unwrap();
    assert!(p.prediction.is_none() && !p.notes.is_empty());
    assert_eq!(p.plan, candidates(&inp)[0]);
    assert!(propose(&inp, &p.plan).is_none());
    assert!(regret(&inp, &p.plan).is_none());
}

#[test]
fn nothing_fits_is_an_error() {
    let hw = common::graphs().pop().unwrap().1;
    let mut inp = input(&hw, &fixtures::moe(), ServeEngine::Llama);
    inp.budget.ram_bytes = 1 << 30;
    let e = pick(&inp).unwrap_err();
    assert!(e.contains("no executable plan"), "{e}");
}

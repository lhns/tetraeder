//! Replanning on simulated hardware and faults (ADR 0055, AGENTS.md rule 4). Every graph with a
//! `cpu0`, both model layouts, both engines: no fault, a CPU co-tenant, a GPU throttle and a VRAM
//! shrink; the proposals lower, fit, and move work away from the slowed device.

mod common;

use common::{graphs, has_gpu, input, scale};
use tetraeder_plan::{KvFormat, Plan};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    ModelLayout, Priors, ServeEngine, ServeInput, Workload, candidates, correct_for_drift,
    fill_from_priors, footprint, lower, pick, predict_plan, propose, recognize, regret,
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

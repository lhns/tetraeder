//! One test per detector rule, the verdict, the intervals, the advisor and the JSON format.

use tetraeder_analysis::advise::{ActionKind, PolicyClass, advise};
use tetraeder_analysis::detect::{FaultClass, Regime, Verdict, detect};
use tetraeder_analysis::observation::{Observation, ThrottleReason};
use tetraeder_analysis::simfault::{SimSpec, simulate};
use tetraeder_profiler::HardwareGraph;

const BOX: &str = include_str!("../fixtures/rtx3090_box.toml");
const WAN: &str = include_str!("../../costmodel/fixtures/wan_two_node.toml");
const EXAMPLE: &str = include_str!("../fixtures/observation_example.json");

fn healthy(seed: u64) -> Observation {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    let mut o = simulate(&hw, &SimSpec::new(seed, &[])).unwrap();
    o.env.throttle.clear();
    o
}

fn classes(o: &Observation) -> Vec<FaultClass> {
    detect(o).faults.iter().map(|f| f.class).collect()
}

#[test]
fn healthy_run_has_no_faults_and_is_balanced() {
    let o = healthy(1);
    let r = detect(&o);
    assert!(r.faults.is_empty(), "{:?}", r.faults);
    assert_eq!(r.regime, Regime::Serialized);
    assert_eq!(r.verdict, Verdict::Balanced);
    let total: f64 = r.limits.iter().map(|l| l.share).sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn a_dominant_resource_is_the_verdict() {
    let hw = HardwareGraph::from_toml(WAN).unwrap();
    let r = detect(&simulate(&hw, &SimSpec::new(1, &[])).unwrap());
    assert_eq!(r.verdict, Verdict::Limited("link:wan".into()));
    assert!(r.faults.is_empty(), "a limit is not a fault");
    assert!(r.limits[0].ci_low > r.limits[1].ci_high);
}

#[test]
fn load_regime_uses_utilization_against_the_busiest() {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    let mut spec = SimSpec::new(2, &[]);
    spec.batch = 8;
    let r = detect(&simulate(&hw, &spec).unwrap());
    assert_eq!(r.regime, Regime::Load);
    // The busiest resource can change between windows, so its mean is just below 1.
    assert!(r.limits[0].share > 0.95 && r.limits[0].share <= 1.0);
    assert!(r.faults.is_empty());
}

#[test]
fn intervals_are_deterministic_and_narrow_with_more_windows() {
    let o = healthy(3);
    assert_eq!(detect(&o).limits, detect(&o).limits);
    let width = |o: &Observation| {
        let l = &detect(o).limits[0];
        l.ci_high - l.ci_low
    };
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    let mut many = SimSpec::new(3, &[]);
    many.windows = 32;
    assert!(width(&simulate(&hw, &many).unwrap()) < width(&o));
    let mut one = o.clone();
    one.windows.truncate(1);
    let r = detect(&one);
    assert_eq!(r.limits[0].ci_low, r.limits[0].ci_high);
    assert!(r.notes.iter().any(|n| n.contains("windows")));
}

#[test]
fn throttle_reason_alone_is_not_a_fault_when_the_rate_is_fine() {
    let mut o = healthy(1);
    o.env
        .throttle
        .push(tetraeder_analysis::observation::Throttle {
            device: "gpu0".into(),
            reason: ThrottleReason::Power,
        });
    assert!(classes(&o).is_empty());
}

#[test]
fn throttle_without_calibration_still_fires_on_a_material_gpu() {
    let mut o = healthy(1);
    o.expected = None;
    o.env
        .throttle
        .push(tetraeder_analysis::observation::Throttle {
            device: "gpu0".into(),
            reason: ThrottleReason::Thermal,
        });
    assert_eq!(classes(&o), vec![FaultClass::GpuThrottle]);
}

#[test]
fn env_fault_on_an_immaterial_resource_is_ignored() {
    let mut o = healthy(1);
    // Starve the GPU of time: it no longer carries 10% of the step.
    for w in &mut o.windows {
        w.busy_s.insert("gpu:gpu0".into(), 0.001);
        w.wall_s = w.busy_s.values().sum();
    }
    o.expected = None;
    o.env
        .throttle
        .push(tetraeder_analysis::observation::Throttle {
            device: "gpu0".into(),
            reason: ThrottleReason::Power,
        });
    assert!(classes(&o).is_empty());
}

#[test]
fn slow_gpu_without_a_reason_is_a_model_mismatch() {
    let mut o = healthy(1);
    for w in &mut o.windows {
        *w.busy_s.get_mut("gpu:gpu0").unwrap() *= 1.6;
        w.wall_s = w.busy_s.values().sum();
    }
    assert_eq!(classes(&o), vec![FaultClass::ModelMismatch]);
}

#[test]
fn slow_disk_with_and_without_visible_contention_is_one_disk_slow() {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    for seed in [0, 1] {
        let o = simulate(&hw, &SimSpec::new(seed, &[FaultClass::DiskSlow])).unwrap();
        let r = detect(&o);
        assert_eq!(r.faults.len(), 1);
        assert_eq!(r.faults[0].class, FaultClass::DiskSlow);
        let io_max = r.faults[0].evidence.iter().any(|e| e.contains("io.max"));
        assert_eq!(io_max, o.env.io_max_bytes_per_s.is_some());
    }
}

#[test]
fn cpu_cotenant_needs_a_slow_cpu() {
    let mut o = healthy(1);
    o.env.cpu_other_cores = o.env.cpu_quota_cores / 2.0;
    assert!(classes(&o).is_empty(), "co-tenant that costs nothing");
    for w in &mut o.windows {
        *w.busy_s.get_mut("cpu").unwrap() *= 2.0;
        w.wall_s = w.busy_s.values().sum();
    }
    assert_eq!(classes(&o), vec![FaultClass::CpuCotenant]);
}

fn slow_cpu(o: &mut Observation, factor: f64) {
    for w in &mut o.windows {
        *w.busy_s.get_mut("cpu").unwrap() *= factor;
        w.wall_s = w.busy_s.values().sum();
    }
}

#[test]
fn quota_throttle_needs_throttled_periods_and_a_slow_cpu() {
    let mut o = healthy(1);
    o.env.cpu_throttled_share = Some(0.9);
    assert!(classes(&o).is_empty(), "throttled but the CPU rate is fine");
    slow_cpu(&mut o, 1.3);
    let r = detect(&o);
    assert_eq!(classes(&o), vec![FaultClass::CpuQuotaThrottle], "{r:?}");
    assert_eq!(r.faults[0].resource, "cpu_quota");
    assert!(r.faults[0].evidence[0].contains("90%"), "{:?}", r.faults[0]);
    o.env.cpu_throttled_share = Some(0.2);
    assert!(classes(&o).is_empty(), "below the 30% threshold");
    o.env.cpu_throttled_share = None;
    assert!(classes(&o).is_empty(), "no cgroup signal, no finding");
}

#[test]
fn quota_throttle_without_calibration_needs_a_material_cpu() {
    let mut o = healthy(1);
    o.expected = None;
    o.env.cpu_throttled_share = Some(0.9);
    assert_eq!(classes(&o), vec![FaultClass::CpuQuotaThrottle]);
    for w in &mut o.windows {
        w.busy_s.insert("cpu".into(), 0.0);
    }
    assert!(classes(&o).is_empty(), "an idle CPU is not throttled work");
}

#[test]
fn quota_throttle_is_not_a_cotenant_and_not_a_model_mismatch() {
    let mut o = healthy(1);
    o.env.cpu_throttled_share = Some(0.9);
    slow_cpu(&mut o, 1.3);
    let c = classes(&o);
    assert!(!c.contains(&FaultClass::CpuCotenant), "{c:?}");
    assert!(!c.contains(&FaultClass::ModelMismatch), "{c:?}");
    // Other processes using the cores are the co-tenant's evidence; both can hold at once.
    o.env.cpu_other_cores = o.env.cpu_quota_cores / 2.0;
    let c = classes(&o);
    assert!(
        c.contains(&FaultClass::CpuCotenant) && c.contains(&FaultClass::CpuQuotaThrottle),
        "{c:?}"
    );
}

#[test]
fn quota_throttle_advice_lowers_polling_and_threads() {
    let mut o = healthy(1);
    o.env.cpu_throttled_share = Some(0.9);
    slow_cpu(&mut o, 1.3);
    let a = advise(&detect(&o), &o, &[]);
    let t = a
        .iter()
        .find(|a| a.kind == ActionKind::Threads && a.target == "cpu_quota")
        .expect("threads action");
    assert!(t.why.contains("cpu_poll"), "{}", t.why);
}

#[test]
fn simulated_quota_throttle_is_ranked_first() {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    for seed in 0..10 {
        let o = simulate(&hw, &SimSpec::new(seed, &[FaultClass::CpuQuotaThrottle])).unwrap();
        assert!(o.env.cpu_throttled_share.unwrap() >= 0.8);
        assert_eq!(
            classes(&o),
            vec![FaultClass::CpuQuotaThrottle],
            "seed {seed}"
        );
        let h = simulate(&hw, &SimSpec::new(seed, &[])).unwrap();
        assert!(h.env.cpu_throttled_share.unwrap() < 0.05);
    }
}

#[test]
fn vram_pressure_needs_slots_or_hit_rate_and_waits() {
    let mut o = healthy(1);
    o.env.vram_other_bytes.insert("gpu0".into(), 8_000_000_000);
    assert!(classes(&o).is_empty(), "VRAM held but no effect on slots");
    o.env.expert_hit_rate = Some(0.5);
    assert_eq!(classes(&o), vec![FaultClass::VramPressure]);
    o.env.expert_hit_rate = Some(0.9);
    o.env.slots_actual = Some(20);
    assert_eq!(classes(&o), vec![FaultClass::VramPressure]);
    // No expert work at all: nothing to pressure.
    for w in &mut o.windows {
        w.split_s.insert("expert_load_wait".into(), 0.0);
        w.busy_s.insert("cpu".into(), 0.0);
    }
    assert!(classes(&o).is_empty());
}

#[test]
fn pcie_below_max_fires_only_on_a_material_slow_link() {
    let mut o = healthy(1);
    o.env.pcie[0].gen_current = 3;
    assert!(classes(&o).is_empty(), "lower gen but the rate is fine");
    o.expected = None;
    assert_eq!(classes(&o), vec![FaultClass::PcieDegraded]);
}

#[test]
fn swapping_needs_paging_and_a_stall() {
    let mut o = healthy(1);
    assert_eq!(o.env.memory_stall_share, None, "odd seeds have no PSI");
    o.env.swap_in_bytes_per_s = 3e8;
    assert!(
        classes(&o).is_empty(),
        "swap-in while the CPU runs at its rate"
    );
    // PSI alone can show the stall.
    o.env.memory_stall_share = Some(0.4);
    let r = detect(&o);
    assert_eq!(classes(&o), vec![FaultClass::Swapping]);
    assert!((r.faults[0].cost_share - 0.4).abs() < 1e-9);
    assert_eq!(r.faults[0].resource, "ram");
    // The stall lands in compute on the CPU, not in host time (the CPU testbed run); PSI may
    // count only a little of it (5% at a 5x slowdown on the testbed).
    o.env.memory_stall_share = Some(0.05);
    for w in &mut o.windows {
        *w.busy_s.get_mut("cpu").unwrap() *= 3.0;
        w.wall_s = w.busy_s.values().sum();
    }
    let r = detect(&o);
    assert_eq!(classes(&o), vec![FaultClass::Swapping], "{r:?}");
    assert!(r.faults[0].cost_share > 0.3, "{r:?}");
    o.env.memory_stall_share = None;
    assert_eq!(classes(&o), vec![FaultClass::Swapping]);
    // A stray page is not paging.
    o.env.swap_in_bytes_per_s = 1e6;
    assert!(!classes(&o).contains(&FaultClass::Swapping));
    // Refaults of a mapped file without swap: major faults count only with a PSI memory stall.
    o.env.major_faults_per_s = 1e5;
    assert!(!classes(&o).contains(&FaultClass::Swapping));
    o.env.memory_stall_share = Some(0.0);
    assert!(!classes(&o).contains(&FaultClass::Swapping));
    o.env.memory_stall_share = Some(0.05);
    assert_eq!(classes(&o), vec![FaultClass::Swapping]);
}

#[test]
fn network_link_slow_is_link_slow() {
    let hw = HardwareGraph::from_toml(WAN).unwrap();
    let o = simulate(&hw, &SimSpec::new(1, &[FaultClass::LinkSlow])).unwrap();
    let r = detect(&o);
    assert_eq!(r.faults[0].class, FaultClass::LinkSlow);
    assert_eq!(r.faults[0].resource, "link:wan");
}

#[test]
fn faults_are_ranked_by_cost() {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    let o = simulate(
        &hw,
        &SimSpec::new(2, &[FaultClass::GpuThrottle, FaultClass::DiskSlow]),
    )
    .unwrap();
    let f = detect(&o).faults;
    assert!(f.windows(2).all(|w| w[0].cost_share >= w[1].cost_share));
    assert_eq!(
        f[0].class,
        FaultClass::DiskSlow,
        "70% slower beats 40% slower"
    );
}

#[test]
fn advisor_ranks_by_gain_minus_cost_and_applies_policy() {
    let hw = HardwareGraph::from_toml(BOX).unwrap();
    let o = simulate(&hw, &SimSpec::new(2, &[FaultClass::DiskSlow])).unwrap();
    let r = detect(&o);
    let none = advise(&r, &o, &[]);
    assert!(none.windows(2).all(|w| w[0].score() >= w[1].score()));
    assert_eq!(none[0].kind, ActionKind::FixEnvironment);
    assert_eq!(none[0].policy, PolicyClass::Report);
    let hot = none
        .iter()
        .find(|a| a.kind == ActionKind::GrowHotSet)
        .unwrap();
    assert_eq!(hot.policy, PolicyClass::Ask, "not in hints.auto_apply");
    let auto = advise(&r, &o, &["grow_hot_set".to_string()]);
    let hot = auto
        .iter()
        .find(|a| a.kind == ActionKind::GrowHotSet)
        .unwrap();
    assert_eq!(hot.policy, PolicyClass::Auto);
}

#[test]
fn advisor_proposes_scaling_when_balanced() {
    let o = healthy(1);
    let a = advise(&detect(&o), &o, &[]);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].kind, ActionKind::ScaleOut);
}

#[test]
fn advisor_maps_a_limit_to_roofline_actions() {
    let hw = HardwareGraph::from_toml(WAN).unwrap();
    let o = simulate(&hw, &SimSpec::new(1, &[])).unwrap();
    let a = advise(&detect(&o), &o, &[]);
    assert_eq!(a[0].kind, ActionKind::Replan);
    assert_eq!(a[0].target, "link:wan");
    assert!(a[0].predicted_gain_share > 0.2);
}

#[test]
fn observation_example_round_trips() {
    let o: Observation = serde_json::from_str(EXAMPLE).unwrap();
    assert_eq!(o.windows.len(), 2);
    assert_eq!(o.env.pcie[0].resource(), "link:pcie-gpu0");
    assert_eq!(o.env.throttle[0].reason, ThrottleReason::Sw);
    let again: Observation = serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
    assert_eq!(o, again);
    let r = detect(&o);
    assert!(r.faults.is_empty(), "{:?}", r.faults);
}

#[test]
fn a_sampler_may_omit_everything_but_windows() {
    let o: Observation = serde_json::from_str(
        r#"{"id":"min","windows":[{"tokens":8,"wall_s":1.0,"busy_s":{"cpu":0.7,"host":0.3}}]}"#,
    )
    .unwrap();
    assert_eq!(o.batch, 1);
    let r = detect(&o);
    assert_eq!(r.verdict, Verdict::Limited("cpu".into()));
    assert!(r.faults.is_empty());
}

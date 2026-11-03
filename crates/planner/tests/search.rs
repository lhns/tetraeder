//! Tests for the search layer (AGENTS.md rules 4 and 5): on every simulated hardware fixture and
//! several models, a plan is found, validates, stays inside memory, computes every unit exactly
//! once and migrates from a fixed starting plan to itself; properties over random hints.

use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_plan::*;
use tetraeder_planner::objective::{devices_used, pareto_front, violations};
use tetraeder_planner::{
    BitsQuality, NullQuality, QualityModel, SearchInput, SearchResult, TableQuality, search,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph, Tier};

const FIXTURES: [(&str, &str); 4] = [
    (
        "two_gpu_cpu_nvme",
        include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    ),
    (
        "wan_two_node",
        include_str!("../../costmodel/fixtures/wan_two_node.toml"),
    ),
    (
        "rtx3090_box",
        include_str!("../../analysis/fixtures/rtx3090_box.toml"),
    ),
    (
        "rtx3090_prior",
        include_str!("../fixtures/rtx3090_prior.toml"),
    ),
];
const SMALL_MOE: &str = include_str!("../../plan/fixtures/model_small_moe.toml");

fn hardware() -> Vec<(&'static str, HardwareGraph)> {
    FIXTURES
        .iter()
        .map(|(n, t)| (*n, HardwareGraph::from_toml(t).unwrap()))
        .collect()
}

fn size(params: u64) -> UnitSize {
    UnitSize {
        params,
        flops_per_token: 2.0 * params as f64,
        source_format: Format::F16,
        roles: Vec::new(),
    }
}

fn dense(layers: u32, attn: u64, mlp: u64, io: u64) -> ModelDesc {
    ModelDesc {
        name: "dense".into(),
        layers,
        heads: 32,
        kv_heads: 8,
        experts: 0,
        experts_used: 0,
        embed: size(io),
        attn: size(attn),
        mlp: size(mlp),
        expert: None,
        head: size(io),
        kv_bytes_per_token_per_layer: 4096,
        activation_bytes_per_token: 8192,
    }
}

fn moe(layers: u32, experts: u32, expert: u64) -> ModelDesc {
    ModelDesc {
        name: "moe".into(),
        layers,
        heads: 16,
        kv_heads: 8,
        experts,
        experts_used: 2,
        embed: size(100_000_000),
        attn: size(30_000_000),
        mlp: size(4_000_000),
        expert: Some(size(expert)),
        head: size(100_000_000),
        kv_bytes_per_token_per_layer: 4096,
        activation_bytes_per_token: 8192,
    }
}

/// Dense, small enough for any fixture.
fn dense_small() -> ModelDesc {
    dense(8, 20_000_000, 60_000_000, 50_000_000)
}

/// Dense, 57 GB at F16: does not fit any single GPU.
fn dense_big() -> ModelDesc {
    dense(40, 200_000_000, 500_000_000, 200_000_000)
}

fn moe_small() -> ModelDesc {
    toml::from_str(SMALL_MOE).unwrap()
}

/// MoE, 40 GB at F16: does not fit one 24 GB GPU.
fn moe_big() -> ModelDesc {
    moe(24, 16, 50_000_000)
}

fn models() -> Vec<(&'static str, ModelDesc)> {
    vec![
        ("dense_small", dense_small()),
        ("moe_small", moe_small()),
        ("dense_big", dense_big()),
        ("moe_big", moe_big()),
    ]
}

fn reserve(hw: &HardwareGraph) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let r = match d.kind {
                DeviceKind::Gpu => 600 << 20,
                DeviceKind::Cpu => 2 << 30,
                DeviceKind::Disk => 0,
            };
            (d.id.clone(), r)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    hw: &HardwareGraph,
    m: &ModelDesc,
    hints: &Hints,
    phase: u8,
    q: &dyn QualityModel,
    routing: Option<&[f64]>,
    linux: bool,
) -> SearchResult {
    search(&SearchInput {
        hw,
        model: m,
        hints,
        phase: Phase(phase),
        workload: (512, 128, 1),
        quality: q,
        reserve: reserve(hw),
        routing,
        linux,
    })
}

fn ctx<'a>(
    hw: &'a HardwareGraph,
    m: &'a ModelDesc,
    phase: u8,
    reserve: &'a BTreeMap<String, u64>,
    hints: &'a Hints,
) -> Context<'a> {
    Context {
        hw,
        model: m,
        phase: Phase(phase),
        reserve,
        hints: Some(hints),
    }
}

/// The invariants of every returned plan: validates, memory inside capacity minus reserve, every
/// unit has all its shards exactly once.
fn check_plan(
    plan: &Plan,
    hw: &HardwareGraph,
    m: &ModelDesc,
    phase: u8,
    reserve: &BTreeMap<String, u64>,
    hints: &Hints,
) -> Expanded {
    let e = validate(plan, &ctx(hw, m, phase, reserve, hints)).unwrap_or_else(|i| panic!("{i:#?}"));
    for (dev, used) in e.memory_per_device() {
        let cap = hw.device(&dev).unwrap().capacity_bytes;
        let limit = cap.saturating_sub(reserve.get(&dev).copied().unwrap_or(0));
        assert!(used <= limit, "{dev}: {used} > {limit}");
    }
    let mut by_unit: BTreeMap<UnitId, Vec<(u32, u32)>> = BTreeMap::new();
    for k in e.shards.keys() {
        by_unit.entry(k.unit).or_default().push((k.shard, k.of));
    }
    assert_eq!(by_unit.len(), m.units().count());
    for u in m.units() {
        let shards = &by_unit[&u];
        let of = shards[0].1;
        let want: Vec<(u32, u32)> = (0..of).map(|i| (i, of)).collect();
        assert_eq!(shards, &want, "{u}");
    }
    e
}

/// Everything a search result promises.
fn check_result(
    res: &SearchResult,
    hw: &HardwareGraph,
    m: &ModelDesc,
    phase: u8,
    hints: &Hints,
) -> Option<usize> {
    let rsv = reserve(hw);
    assert!(
        res.pruned
            .iter()
            .all(|(_, why)| !why.starts_with("invalid")),
        "{:#?}",
        res.pruned
    );
    for s in &res.scored {
        check_plan(&s.plan, hw, m, phase, &rsv, hints);
        assert!(s.tok_s.is_finite() && s.tok_s > 0.0, "{}", s.desc);
        assert!(s.ttft_s.is_finite() && s.ttft_s > 0.0, "{}", s.desc);
        assert!(s.quality_loss >= 0.0 && s.resource_cost >= 0.0);
        assert!(!s.stage_times.is_empty());
    }
    let c = &res.choice;
    let i = c.chosen?;
    let chosen = &res.scored[i];
    if c.feasible {
        assert!(c.failing.is_empty() && c.closest.is_none());
        assert!(violations(chosen, hints).is_empty());
    } else {
        assert!(!c.failing.is_empty());
        assert_eq!(c.closest, Some(i));
    }
    // The front is not dominated by any plan in its pool.
    for &f in &c.front {
        let a = &res.scored[f];
        for (j, b) in res.scored.iter().enumerate() {
            let pool_ok = !c.feasible || violations(b, hints).is_empty();
            if pool_ok && j != f {
                let dominates = (b.tok_s > a.tok_s && b.quality_loss <= a.quality_loss + 1e-12)
                    || (b.tok_s >= a.tok_s && b.quality_loss < a.quality_loss - 1e-12);
                assert!(!dominates, "front plan {f} dominated by {j}");
            }
        }
    }
    // Migration from a fixed starting plan (the first one found) to the chosen plan ends in it.
    let rsv = reserve(hw);
    let from = check_plan(&res.scored[0].plan, hw, m, phase, &rsv, hints);
    let to = check_plan(&chosen.plan, hw, m, phase, &rsv, hints);
    let dag = MigrationDag::from_diff(&diff(&from, &to));
    for seed in [0usize, 7, 12345] {
        let mut k = seed;
        let order = dag
            .order_with(|_| {
                k = k
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                k >> 33
            })
            .expect("acyclic");
        let mut state = State::from(&from);
        for s in order {
            state.apply(&dag.steps[s]).unwrap();
        }
        assert_eq!(state, State::from(&to));
    }
    Some(i)
}

#[test]
fn every_fixture_and_model_gets_a_valid_plan() {
    for (hn, hw) in hardware() {
        for (mn, m) in models() {
            let q = BitsQuality::new(&m, 1.0);
            for phase in [1u8, 3, 5] {
                let hints = Hints::default();
                let res = run(&hw, &m, &hints, phase, &q, None, true);
                // Phase 1 cannot convert formats: a model larger than the fixture memory has no
                // plan there, and the result says so.
                let i = check_result(&res, &hw, &m, phase, &hints);
                if phase == 1 && i.is_none() {
                    assert!(!res.pruned.is_empty() && !res.choice.failing.is_empty());
                    continue;
                }
                assert!(i.is_some(), "{hn} {mn} phase {phase}: {:#?}", res.pruned);
                if phase < 5 {
                    for s in &res.scored {
                        for st in &s.plan.replicas[0].stages {
                            assert_eq!((st.tp, st.ep), (Some(1), Some(1)), "{hn} {mn}");
                        }
                    }
                }
                if phase == 1 {
                    for s in &res.scored {
                        let e = validate(&s.plan, &ctx(&hw, &m, 1, &reserve(&hw), &hints)).unwrap();
                        assert!(
                            e.shards.iter().all(|(k, p)| {
                                p.format == m.size(k.unit).unwrap().source_format
                            })
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn search_is_deterministic_and_serializable() {
    let (_, hw) = &hardware()[0];
    let m = moe_big();
    let q = BitsQuality::new(&m, 1.0);
    let a = run(hw, &m, &Hints::default(), 3, &q, None, true);
    let b = run(hw, &m, &Hints::default(), 3, &q, None, true);
    assert_eq!(a.pruned, b.pruned);
    assert_eq!(a.choice, b.choice);
    assert_eq!(a.scored.len(), b.scored.len());
    for (x, y) in a.scored.iter().zip(&b.scored) {
        assert_eq!(x.desc, y.desc);
        assert_eq!(x.plan, y.plan, "{}", x.desc);
        assert_eq!(
            (x.tok_s, x.ttft_s, x.tpot_s, x.simulated),
            (y.tok_s, y.ttft_s, y.tpot_s, y.simulated),
            "{}",
            x.desc
        );
        assert_eq!(x.stage_times, y.stage_times, "{}", x.desc);
        assert_eq!(
            (x.quality_loss, x.resource_cost, x.headroom_bytes),
            (y.quality_loss, y.resource_cost, y.headroom_bytes)
        );
    }
    assert_eq!(a, b);
    let json = serde_json::to_string(&a).unwrap();
    // serde_json's float parsing may differ in the last digit: compare numbers with a tolerance.
    let back = serde_json::from_str::<SearchResult>(&json).unwrap();
    assert_eq!(back.pruned, a.pruned);
    assert_eq!(back.choice, a.choice);
    assert_eq!(back.scored.len(), a.scored.len());
    let near = |x: f64, y: f64| (x - y).abs() <= 1e-12 * x.abs().max(y.abs());
    for (x, y) in back.scored.iter().zip(&a.scored) {
        assert_eq!((&x.desc, &x.plan), (&y.desc, &y.plan));
        assert_eq!(
            (x.simulated, x.headroom_bytes),
            (y.simulated, y.headroom_bytes)
        );
        assert!(near(x.tok_s, y.tok_s) && near(x.ttft_s, y.ttft_s));
        assert!(near(x.quality_loss, y.quality_loss) && near(x.resource_cost, y.resource_cost));
        assert_eq!(x.tpot_s.is_some(), y.tpot_s.is_some());
        if let (Some(p), Some(q)) = (x.tpot_s, y.tpot_s) {
            assert!(near(p, q));
        }
        assert_eq!(x.stage_times.len(), y.stage_times.len());
        for (s, u) in x.stage_times.iter().zip(&y.stage_times) {
            assert_eq!(
                (s.stage, s.layers, &s.devices),
                (u.stage, u.layers, &u.devices)
            );
            assert!(near(s.compute_s, u.compute_s) && near(s.hop_s, u.hop_s));
        }
    }
}

#[test]
fn experts_spill_to_nvme_when_ram_is_full() {
    // 8 GiB of RAM next to a 24 GB GPU: a 40 GB MoE has to put experts on the NVMe.
    let mut hw = hardware()[0].1.clone();
    hw.devices
        .iter_mut()
        .find(|d| d.id == "cpu0")
        .unwrap()
        .capacity_bytes = 8 << 30;
    let m = moe_big();
    let q = NullQuality::new(&m);
    let hints = Hints::default();
    let res = run(&hw, &m, &hints, 3, &q, None, true);
    check_result(&res, &hw, &m, 3, &hints).unwrap();
    let on_nvme = |s: &tetraeder_planner::Scored| {
        s.plan.replicas[0].stages.iter().any(|st| {
            st.assign.iter().any(|a| {
                a.home.as_deref() == Some(&["nvme0".to_string()][..])
                    && a.compute.as_deref() == Some(&[st.devices[0].clone()][..])
            }) && (!st.expert_cache.is_empty() || st.devices[0] == "cpu0")
        })
    };
    assert!(res.scored.iter().any(on_nvme), "{:#?}", res.pruned);
    // Forbidding the tier removes those plans; what is left (or nothing) still validates.
    let hints = Hints {
        forbid: vec![Forbid {
            units: None,
            device: None,
            tier: Some(Tier::Nvme),
        }],
        ..Default::default()
    };
    let res = run(&hw, &m, &hints, 3, &q, None, true);
    check_result(&res, &hw, &m, 3, &hints);
    assert!(!res.scored.iter().any(on_nvme));
    // Phase 1 has no expert tiers: nothing on the NVMe, whatever the memory.
    let hints = Hints::default();
    let res = run(&hw, &m, &hints, 1, &q, None, true);
    check_result(&res, &hw, &m, 1, &hints);
    assert!(!res.scored.iter().any(on_nvme));
}

#[test]
fn the_big_models_use_the_lower_tiers_and_compression() {
    let hw = &hardware()[0].1; // two GPUs, CPU, NVMe
    let m = moe_big();
    let q = BitsQuality::new(&m, 1.0);
    let res = run(hw, &m, &Hints::default(), 3, &q, None, true);
    let i = check_result(&res, hw, &m, 3, &Hints::default()).unwrap();
    let used = devices_used(&res.scored[i].plan);
    assert!(!used.is_empty());
    // Some plan keeps source formats and spills experts to RAM or NVMe; some compresses.
    assert!(res.scored.iter().any(|s| s.quality_loss == 0.0));
    assert!(res.scored.iter().any(|s| s.quality_loss > 0.0));
    assert!(res.choice.front.len() >= 2, "{:?}", res.choice.front);
    // Streaming plans carry a slot cache and name their NVMe/RAM homes explicitly.
    let streamed = res.scored.iter().any(|s| {
        s.plan.replicas[0]
            .stages
            .iter()
            .any(|st| !st.expert_cache.is_empty())
    });
    assert!(streamed);
}

#[test]
fn hot_experts_follow_the_routing_frequency() {
    let hw = &hardware()[2].1; // rtx3090_box
    let m = moe_big();
    let q = BitsQuality::new(&m, 1.0);
    let mut routing = vec![1.0; 16];
    routing[5] = 40.0;
    routing[11] = 30.0;
    routing[2] = 10.0;
    let hints = Hints::default();
    let res = run(hw, &m, &hints, 3, &q, Some(&routing), true);
    let rsv = reserve(hw);
    let mut checked = 0;
    for s in &res.scored {
        let e = check_plan(&s.plan, hw, &m, 3, &rsv, &hints);
        for l in 0..m.layers {
            let home = |x: u32| {
                e.shards[&ShardKey {
                    replica: 0,
                    unit: UnitId::Expert(l, x),
                    shard: 0,
                    of: 1,
                }]
                    .home
                    .clone()
            };
            let on_gpu: Vec<u32> = (0..16).filter(|&x| home(x) == "gpu0").collect();
            let off_gpu: Vec<u32> = (0..16).filter(|&x| home(x) != "gpu0").collect();
            for &x in &on_gpu {
                for &y in &off_gpu {
                    assert!(
                        routing[x as usize] >= routing[y as usize],
                        "layer {l}: expert {x} on the GPU, {y} off it"
                    );
                }
            }
            checked += usize::from(!on_gpu.is_empty() && !off_gpu.is_empty());
        }
    }
    assert!(checked > 0, "no plan mixes GPU and lower tiers");
}

#[test]
fn infeasible_limits_name_the_failing_limit_and_return_a_valid_plan() {
    let hw = &hardware()[1].1; // wan_two_node: 2 x 24 GB, no CPU
    let m = dense_big(); // 57 GB: needs compression
    let q = BitsQuality::new(&m, 1.0);
    let hints = Hints {
        max_quality_loss: Some(1e-9),
        target_tok_s: Some(1e9),
        ..Default::default()
    };
    let res = run(hw, &m, &hints, 3, &q, None, true);
    let i = check_result(&res, hw, &m, 3, &hints).unwrap();
    assert!(!res.choice.feasible);
    let f = res.choice.failing.join("; ");
    assert!(
        f.contains("max_quality_loss") && f.contains("target_tok_s"),
        "{f}"
    );
    assert_eq!(res.choice.closest, Some(i));

    // A model that fits nowhere has no plan: the pruned list says why.
    let huge = dense(400, 200_000_000, 500_000_000, 200_000_000);
    let null = NullQuality::new(&huge);
    let res = run(hw, &huge, &Hints::default(), 3, &null, None, true);
    assert!(res.scored.is_empty() && res.choice.chosen.is_none());
    assert!(!res.choice.failing.is_empty() && !res.pruned.is_empty());
}

#[test]
fn phase_gates_parallel_degrees_and_formats() {
    let hw = &hardware()[0].1;
    let m = dense_small();
    let q = BitsQuality::new(&m, 1.0);
    let tp_plans = |phase: u8, linux: bool| {
        let res = run(hw, &m, &Hints::default(), phase, &q, None, linux);
        check_result(&res, hw, &m, phase, &Hints::default()).unwrap();
        let n = res
            .scored
            .iter()
            .filter(|s| s.plan.replicas[0].stages.iter().any(|st| st.tp > Some(1)))
            .count();
        (n, res)
    };
    let (n3, r3) = tp_plans(3, true);
    assert_eq!(n3, 0);
    assert!(r3.pruned.iter().any(|(_, w)| w.contains("cannot execute")));
    let (n5, _) = tp_plans(5, true);
    assert!(n5 > 0);
    let (n5w, r5w) = tp_plans(5, false);
    assert_eq!(n5w, 0);
    assert!(r5w.pruned.iter().any(|(_, w)| w.contains("Linux")));

    // No format conversion before phase 2: only the source format, whatever the quality model.
    let res = run(hw, &m, &Hints::default(), 1, &q, None, true);
    assert!(res.scored.iter().all(|s| s.quality_loss == 0.0));
    let res = run(hw, &m, &Hints::default(), 2, &q, None, true);
    assert!(res.scored.iter().any(|s| s.quality_loss > 0.0));
}

#[test]
fn moe_parallel_stage_spreads_experts_evenly() {
    let hw = &hardware()[0].1;
    let m = moe_small();
    let q = BitsQuality::new(&m, 1.0);
    let res = run(hw, &m, &Hints::default(), 5, &q, None, true);
    check_result(&res, hw, &m, 5, &Hints::default()).unwrap();
    assert!(res.scored.iter().any(|s| {
        s.plan.replicas[0]
            .stages
            .iter()
            .any(|st| st.ep > Some(1) || st.tp > Some(1))
    }));
}

#[test]
fn quality_ladder_comes_from_the_quality_model() {
    // Only Q8_0 and Q4_K are known, per layer span and kind; others must never be chosen.
    let hw = &hardware()[2].1;
    let m = dense_big();
    let mut q = TableQuality::new(&m, 0.0);
    for kind in [UnitKind::Attn, UnitKind::Mlp] {
        q = q.with(kind, Span(0, 40), Format::Q8_0, 0.02).with(
            kind,
            Span(0, 40),
            Format::Q4_K,
            0.2,
        );
    }
    q = q
        .with(UnitKind::Embed, Span(0, 0), Format::Q8_0, 0.001)
        .with(UnitKind::Head, Span(0, 0), Format::Q8_0, 0.001);
    let hints = Hints::default();
    let res = run(hw, &m, &hints, 3, &q, None, true);
    check_result(&res, hw, &m, 3, &hints).unwrap();
    let rsv = reserve(hw);
    for s in &res.scored {
        let e = check_plan(&s.plan, hw, &m, 3, &rsv, &hints);
        for (k, p) in &e.shards {
            assert!(
                [Format::F16, Format::Q8_0, Format::Q4_K].contains(&p.format),
                "{k:?} {:?}",
                p.format
            );
        }
    }
    // A floor above the known formats keeps units precise.
    let hints = Hints {
        min_format: Some(Format::Q8_0),
        ..Default::default()
    };
    let res = run(hw, &m, &hints, 3, &q, None, true);
    check_result(&res, hw, &m, 3, &hints).unwrap();
}

#[test]
fn without_a_speed_limit_the_fastest_plan_wins_and_a_target_picks_the_cheapest_hardware() {
    let hw = &hardware()[0].1;
    let m = dense_small();
    let q = NullQuality::new(&m);
    let base = run(hw, &m, &Hints::default(), 3, &q, None, true);
    let chosen = &base.scored[base.choice.chosen.unwrap()];
    // All plans tie on quality: the choice is within the speed band of the fastest plan, and the
    // cheapest of those.
    let fastest = base.scored.iter().map(|s| s.tok_s).fold(0.0, f64::max);
    assert!(chosen.tok_s >= fastest * (1.0 - tetraeder_planner::objective::SPEED_BAND));
    let band_cost = base
        .scored
        .iter()
        .filter(|s| s.tok_s >= fastest * (1.0 - tetraeder_planner::objective::SPEED_BAND))
        .map(|s| s.resource_cost)
        .fold(f64::MAX, f64::min);
    assert_eq!(chosen.resource_cost, band_cost);
    // With a target the cheapest hardware that reaches it wins.
    let want = fastest * 0.5;
    let hints = Hints {
        target_tok_s: Some(want),
        ..Default::default()
    };
    let res = run(hw, &m, &hints, 3, &q, None, true);
    let i = check_result(&res, hw, &m, 3, &hints).unwrap();
    assert!(res.choice.feasible && res.scored[i].tok_s >= want);
    let ok_cost = res
        .scored
        .iter()
        .filter(|s| s.tok_s >= want)
        .map(|s| s.resource_cost)
        .fold(f64::MAX, f64::min);
    assert_eq!(res.scored[i].resource_cost, ok_cost);
}

#[test]
fn pins_and_forbids_are_honored() {
    let hw = &hardware()[0].1;
    let m = dense_small();
    let q = NullQuality::new(&m);
    let hints = Hints {
        pin: vec![Pin {
            units: UnitSel {
                layers: Some(Span(0, 2)),
                ..Default::default()
            },
            device: "gpu1".into(),
        }],
        forbid: vec![Forbid {
            units: None,
            device: Some("cpu0".into()),
            tier: None,
        }],
        ..Default::default()
    };
    let res = run(hw, &m, &hints, 3, &q, None, true);
    check_result(&res, hw, &m, 3, &hints).unwrap();
    for s in &res.scored {
        assert!(!devices_used(&s.plan).contains(&"cpu0".to_string()));
        let first = &s.plan.replicas[0].stages[0];
        assert!(first.devices.contains(&"gpu1".to_string()));
    }
    // A tier forbid for the whole model.
    let hints = Hints {
        forbid: vec![Forbid {
            units: None,
            device: None,
            tier: Some(Tier::Nvme),
        }],
        ..Default::default()
    };
    let m = moe_big();
    let q = BitsQuality::new(&m, 1.0);
    let res = run(hw, &m, &hints, 3, &q, None, true);
    check_result(&res, hw, &m, 3, &hints).unwrap();
    assert!(
        res.scored
            .iter()
            .all(|s| !devices_used(&s.plan).contains(&"nvme0".to_string()))
    );
}

#[test]
fn tradeoff_ends_pick_the_fastest_and_the_best_quality_front_points() {
    let hw = &hardware()[2].1;
    let m = moe_big();
    let q = BitsQuality::new(&m, 1.0);
    let at = |t: f64| {
        let hints = Hints {
            tradeoff: Some(t),
            ..Default::default()
        };
        let res = run(hw, &m, &hints, 3, &q, None, true);
        let i = check_result(&res, hw, &m, 3, &hints).unwrap();
        (res, i)
    };
    let (r0, i0) = at(0.0);
    let (r1, i1) = at(1.0);
    assert_eq!(r0.scored, r1.scored);
    let s = &r0.scored;
    assert!(r0.choice.front.len() >= 2);
    let max_tok = s.iter().map(|x| x.tok_s).fold(0.0, f64::max);
    let min_loss = s.iter().map(|x| x.quality_loss).fold(f64::MAX, f64::min);
    assert_eq!(s[i0].tok_s, max_tok);
    assert_eq!(i0, r0.choice.front[0]);
    assert_eq!(s[i1].quality_loss, min_loss);
    assert_eq!(Some(&i1), r1.choice.front.last());
    // Of the plans with the best quality, the chosen one is the fastest.
    let best_q_tok = s
        .iter()
        .filter(|x| x.quality_loss == min_loss)
        .map(|x| x.tok_s)
        .fold(0.0, f64::max);
    assert_eq!(s[i1].tok_s, best_q_tok);
    assert_eq!(
        pareto_front(s, &(0..s.len()).collect::<Vec<_>>()),
        r0.choice.front
    );
}

// ------------------------------------------------------------------ properties

fn pick<T: Clone>(xs: &[T], i: usize) -> Option<T> {
    xs.get(i).cloned()
}

#[derive(Debug, Clone)]
struct Knob {
    hw: usize,
    model: usize,
    layers: u32,
    phase: u8,
    pin: usize,
    pin_to: u32,
    forbid_dev: usize,
    forbid_tier: usize,
    cost_dev: usize,
    cost: f64,
    target: f64,
    ttft_ms: f64,
    qloss: f64,
    tradeoff: f64,
}

fn knobs() -> impl Strategy<Value = Knob> {
    (
        (0usize..4, 0usize..4, 2u32..9, 1u8..=5),
        (0usize..9, 1u32..4, 0usize..9, 0usize..8),
        (0usize..9, 0.0f64..6.0),
        (0.0f64..120.0, 0.0f64..800.0, 0.0f64..0.5, 0.0f64..1.0),
    )
        .prop_map(
            |((hw, model, layers, phase), (pin, pin_to, fd, ft), (cd, cost), (t, ttft, ql, tr))| {
                Knob {
                    hw,
                    model,
                    layers,
                    phase,
                    pin,
                    pin_to,
                    forbid_dev: fd,
                    forbid_tier: ft,
                    cost_dev: cd,
                    cost,
                    target: t,
                    ttft_ms: ttft,
                    qloss: ql,
                    tradeoff: tr,
                }
            },
        )
}

fn build(k: &Knob) -> (HardwareGraph, ModelDesc, Hints) {
    let hw = hardware()[k.hw].1.clone();
    let ids: Vec<String> = hw.devices.iter().map(|d| d.id.clone()).collect();
    let mut m = match k.model {
        0 => dense(k.layers, 20_000_000, 60_000_000, 50_000_000),
        1 => {
            let mut m = moe_small();
            m.layers = k.layers.min(6);
            m
        }
        2 => dense(k.layers * 4, 200_000_000, 500_000_000, 200_000_000),
        _ => moe(k.layers * 2, 16, 50_000_000),
    };
    m.name = format!("m{}", k.model);
    let mut h = Hints::default();
    if let Some(d) = pick(&ids, k.pin) {
        h.pin.push(Pin {
            units: UnitSel {
                layers: Some(Span(0, k.pin_to)),
                ..Default::default()
            },
            device: d,
        });
    }
    if let Some(d) = pick(&ids, k.forbid_dev) {
        h.forbid.push(Forbid {
            units: None,
            device: Some(d),
            tier: None,
        });
    }
    if k.forbid_tier == 0 {
        h.forbid.push(Forbid {
            units: Some(UnitSel {
                kinds: vec![UnitKind::Expert],
                ..Default::default()
            }),
            device: None,
            tier: Some(Tier::Nvme),
        });
    }
    if let Some(d) = pick(&ids, k.cost_dev) {
        h.cost.insert(d, k.cost);
    }
    h.target_tok_s = (k.target > 40.0).then_some(k.target);
    h.p95_ttft_ms = (k.ttft_ms > 400.0).then_some(k.ttft_ms);
    h.max_quality_loss = (k.qloss > 0.3).then_some(k.qloss * 0.1);
    h.tradeoff = (k.tradeoff > 0.5).then_some(k.tradeoff);
    (hw, m, h)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Whatever the hints, every plan found validates (memory, exactly-once, hard hints), the
    /// choice meets every limit when it says it is feasible, and the migration from the first plan
    /// to the chosen one ends in it.
    #[test]
    fn plans_found_under_random_hints_are_valid(k in knobs()) {
        let (hw, m, h) = build(&k);
        prop_assume!(h.check().is_ok());
        let q = BitsQuality::new(&m, 1.0);
        let res = run(&hw, &m, &h, k.phase, &q, None, true);
        let chosen = check_result(&res, &hw, &m, k.phase, &h);
        // Nothing found means every skeleton was pruned with a reason.
        if chosen.is_none() {
            prop_assert!(!res.pruned.is_empty());
        }
    }

    /// A device forbidden outright appears in no plan.
    #[test]
    fn forbidden_devices_never_appear(k in knobs(), dev in 0usize..4) {
        let (hw, m, mut h) = build(&k);
        h.pin.clear();
        h.forbid.clear();
        let Some(d) = pick(&hw.devices.iter().map(|d| d.id.clone()).collect::<Vec<_>>(), dev) else {
            return Ok(());
        };
        h.forbid.push(Forbid { units: None, device: Some(d.clone()), tier: None });
        h.tradeoff = None;
        let q = BitsQuality::new(&m, 1.0);
        let res = run(&hw, &m, &h, k.phase, &q, None, true);
        check_result(&res, &hw, &m, k.phase, &h);
        for s in &res.scored {
            prop_assert!(!devices_used(&s.plan).contains(&d), "{} in {}", d, s.desc);
        }
    }

    /// With a tighter quality limit the chosen plan's loss does not increase, provided both
    /// choices meet their limits' companions (the looser run was feasible).
    #[test]
    fn a_tighter_quality_limit_never_raises_the_chosen_loss(k in knobs(), q1 in 0.001f64..0.3, f in 0.05f64..1.0) {
        let (hw, m, mut h) = build(&k);
        h.pin.clear();
        h.tradeoff = None;
        let q = BitsQuality::new(&m, 1.0);
        h.max_quality_loss = Some(q1);
        let loose = run(&hw, &m, &h, k.phase, &q, None, true);
        h.max_quality_loss = Some(q1 * f);
        let tight = run(&hw, &m, &h, k.phase, &q, None, true);
        // The searches see the same plans; only the limit differs.
        prop_assert_eq!(loose.scored.len(), tight.scored.len());
        if let (Some(a), Some(b)) = (loose.choice.chosen, tight.choice.chosen)
            && loose.choice.feasible
        {
            prop_assert!(
                tight.scored[b].quality_loss <= loose.scored[a].quality_loss + 1e-12,
                "{} -> {}", loose.scored[a].quality_loss, tight.scored[b].quality_loss
            );
        }
    }

    /// Raising a device's price changes nothing about the plans found. With no limits and no
    /// tradeoff (so quality, then cost decide): the new choice costs no more than the old one
    /// under the new prices, and a choice that did not use the device stays the choice.
    #[test]
    fn a_higher_price_never_attracts_the_choice(k in knobs(), dev in 0usize..4, mult in 1.5f64..10.0) {
        let (hw, m, mut h) = build(&k);
        h.pin.clear();
        h.forbid.clear();
        h.cost.clear();
        h.target_tok_s = None;
        h.p95_ttft_ms = None;
        h.max_quality_loss = None;
        h.tradeoff = None;
        let q = BitsQuality::new(&m, 1.0);
        let base = run(&hw, &m, &h, k.phase, &q, None, true);
        let Some(d) = pick(&hw.devices.iter().map(|d| d.id.clone()).collect::<Vec<_>>(), dev) else {
            return Ok(());
        };
        let kind = hw.device(&d).unwrap().kind;
        let price = tetraeder_planner::objective::default_cost(kind).max(0.25) * mult;
        h.cost.insert(d.clone(), price);
        let dear = run(&hw, &m, &h, k.phase, &q, None, true);
        prop_assert_eq!(base.scored.len(), dear.scored.len());
        for (a, b) in base.scored.iter().zip(&dear.scored) {
            prop_assert_eq!(&a.plan, &b.plan);
            prop_assert_eq!(a.tok_s, b.tok_s);
            prop_assert_eq!(a.quality_loss, b.quality_loss);
        }
        let (Some(a), Some(b)) = (base.choice.chosen, dear.choice.chosen) else {
            return Ok(());
        };
        prop_assert!(dear.scored[b].resource_cost <= dear.scored[a].resource_cost + 1e-9);
        prop_assert_eq!(dear.scored[b].quality_loss, dear.scored[a].quality_loss);
        if !devices_used(&base.scored[a].plan).contains(&d) {
            prop_assert_eq!(a, b);
        }
    }
}

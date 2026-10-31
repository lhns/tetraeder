//! Property tests on plans (AGENTS.md rule 5): memory never exceeded, every unit computed exactly
//! once, every migration DAG ends in its target plan. Each case runs on every hardware fixture
//! (rule 4). Cases are built from a seed through a small deterministic generator, so a failing
//! seed reproduces exactly.

use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_plan::testing::{Rng, gen_model, gen_plan};
use tetraeder_plan::*;
use tetraeder_profiler::HardwareGraph;

const HARDWARE: [&str; 3] = [
    include_str!("../../profiler/fixtures/two_gpu_host.toml"),
    include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    include_str!("../../costmodel/fixtures/wan_two_node.toml"),
];

fn hardware() -> Vec<HardwareGraph> {
    HARDWARE
        .iter()
        .map(|t| HardwareGraph::from_toml(t).unwrap())
        .collect()
}

fn validate_with(
    plan: &Plan,
    m: &ModelDesc,
    hw: &HardwareGraph,
    phase: u8,
    reserve: &BTreeMap<String, u64>,
) -> Result<Expanded, Vec<Issue>> {
    validate(
        plan,
        &Context {
            hw,
            model: m,
            phase: Phase(phase),
            reserve,
            hints: None,
        },
    )
}

fn params_of(m: &ModelDesc, u: UnitId) -> UnitSize {
    *m.size(u).unwrap()
}

/// Property 2, the checks on an accepted plan, written without `memory_per_device`.
fn check_accepted(
    plan: &Plan,
    e: &Expanded,
    m: &ModelDesc,
    hw: &HardwareGraph,
    reserve: &BTreeMap<String, u64>,
) {
    let mut used: BTreeMap<&str, u64> = BTreeMap::new();
    for (k, p) in &e.shards {
        *used.entry(&p.home).or_default() += p.bytes;
        let size = params_of(m, k.unit);
        assert_eq!(
            p.bytes,
            p.format.bytes(size.params).div_ceil(u64::from(k.of))
        );
        assert!(size.source_format.at_least(p.format), "upconvert {k:?}");
    }
    for (k, b) in &e.buffers {
        *used.entry(&k.device).or_default() += b;
    }
    for (dev, bytes) in &used {
        let cap = hw.device(dev).unwrap().capacity_bytes;
        let limit = cap.saturating_sub(reserve.get(*dev).copied().unwrap_or(0));
        assert!(*bytes <= limit, "{dev}: {bytes} > {limit}");
    }

    // Phase budgets: inside the arenas (so KV is never borrowed), and weights plus KV plus a
    // phase's bytes inside capacity minus reserve.
    let arena = |k: &BudgetKey, kind| {
        let key = BufferKey {
            replica: k.replica,
            stage: k.stage,
            kind,
            device: k.device.clone(),
        };
        e.buffers.get(&key).copied().unwrap_or(0)
    };
    let mut explicit = 0;
    for rep in &plan.replicas {
        explicit += rep
            .stages
            .iter()
            .map(|s| s.phase_budgets.len())
            .sum::<usize>();
    }
    assert!(e.budgets.len() >= explicit);
    let mut phase_used: BTreeMap<&str, [u64; 2]> = BTreeMap::new();
    for (k, b) in &e.budgets {
        let (slots, ws) = (
            arena(k, BufferKind::SlotCache),
            arena(k, BufferKind::Workspace),
        );
        for (i, p) in [b.prefill, b.decode].into_iter().enumerate() {
            assert!(
                p.slots <= slots && p.slots + p.workspace <= slots + ws,
                "{k:?}"
            );
            phase_used.entry(&k.device).or_default()[i] += p.slots + p.workspace;
        }
    }
    for (dev, phases) in phase_used {
        let arenas: u64 = e
            .budgets
            .keys()
            .filter(|k| k.device == dev)
            .map(|k| arena(k, BufferKind::SlotCache) + arena(k, BufferKind::Workspace))
            .sum();
        let cap = hw.device(dev).unwrap().capacity_bytes;
        let limit = cap.saturating_sub(reserve.get(dev).copied().unwrap_or(0));
        for p in phases {
            // used - arenas = weights + KV
            assert!(used[dev] - arenas + p <= limit, "{dev}");
        }
    }

    let mut by_unit: BTreeMap<(u32, UnitId), Vec<(u32, u32)>> = BTreeMap::new();
    for k in e.shards.keys() {
        by_unit
            .entry((k.replica, k.unit))
            .or_default()
            .push((k.shard, k.of));
    }
    for r in 0..plan.replicas.len() as u32 {
        for u in m.units() {
            let shards = &by_unit[&(r, u)];
            let of = shards[0].1;
            let want: Vec<(u32, u32)> = (0..of).map(|i| (i, of)).collect();
            assert_eq!(shards, &want, "replica {r} unit {u}");
        }
    }
    assert_eq!(by_unit.len(), plan.replicas.len() * m.units().count());
}

fn only_memory(issues: &[Issue]) -> bool {
    issues.iter().all(|i| {
        matches!(
            i,
            Issue::MemoryExceeded { .. } | Issue::PhaseMemoryExceeded { .. }
        )
    })
}

fn random_reserve(rng: &mut Rng, hw: &HardwareGraph, big: bool) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let hi = if big { d.capacity_bytes } else { 1 << 20 };
            (d.id.clone(), rng.range(0, hi))
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn roundtrip_and_accepted_plans(seed in any::<u64>()) {
        for hw in hardware() {
            let mut rng = Rng(seed);
            let m = gen_model(&mut rng);
            let phase = rng.range(2, 6) as u8;
            let oversize = rng.chance(5);
            let plan = gen_plan(&mut rng, &m, &hw, phase, oversize);
            prop_assert_eq!(&Plan::from_toml(&plan.to_toml().unwrap()).unwrap(), &plan);
            prop_assert_eq!(&Plan::from_json(&plan.to_json().unwrap()).unwrap(), &plan);

            let big = rng.chance(4);
            let reserve = random_reserve(&mut rng, &hw, big);
            match validate_with(&plan, &m, &hw, phase, &reserve) {
                Ok(e) => {
                    prop_assert!(!oversize);
                    check_accepted(&plan, &e, &m, &hw, &reserve);
                }
                Err(issues) => {
                    // Structure is valid by construction: only memory may fail.
                    prop_assert!(only_memory(&issues), "{:#?}", issues);
                    prop_assert!(oversize || big, "{:#?}", issues);
                }
            }
            // Without a large reserve the plan fits unless a budget was made too big.
            let none = BTreeMap::new();
            let r = validate_with(&plan, &m, &hw, phase, &none);
            prop_assert_eq!(r.is_ok(), !oversize, "{:?}", r.err());
            if let Ok(e) = r {
                check_accepted(&plan, &e, &m, &hw, &none);
            }
        }
    }

    #[test]
    fn mutations_are_rejected(seed in any::<u64>()) {
        for hw in hardware() {
            let mut rng = Rng(seed);
            let m = gen_model(&mut rng);
            let phase = rng.range(2, 6) as u8;
            let plan = gen_plan(&mut rng, &m, &hw, phase, false);
            let none = BTreeMap::new();
            let e = validate_with(&plan, &m, &hw, phase, &none).unwrap();

            let mut pick_rule = |p: &mut Plan| {
                let r = rng.below(p.replicas.len());
                let s = rng.below(p.replicas[r].stages.len());
                let n = p.replicas[r].stages[s].assign.len();
                (r, s, rng.below(n))
            };

            let mut p = plan.clone();
            let (r, s, i) = pick_rule(&mut p);
            p.replicas[r].stages[s].assign.remove(i);
            let issues = validate_with(&p, &m, &hw, phase, &none).unwrap_err();
            let found = issues.iter().any(|i| matches!(i, Issue::Missing { .. }));
            prop_assert!(found);

            let mut p = plan.clone();
            let (r, s, i) = pick_rule(&mut p);
            let dup = p.replicas[r].stages[s].assign[i].clone();
            p.replicas[r].stages[s].assign.push(dup);
            let issues = validate_with(&p, &m, &hw, phase, &none).unwrap_err();
            let found = issues.iter().any(|i| matches!(i, Issue::Duplicate { .. }));
            prop_assert!(found);

            let used = e.memory_per_device();
            let (dev, _) = used.iter().find(|(_, b)| **b > 0).unwrap();
            let cap = hw.device(dev).unwrap().capacity_bytes;
            let reserve = BTreeMap::from([(dev.clone(), cap)]);
            let issues = validate_with(&plan, &m, &hw, phase, &reserve).unwrap_err();
            let found = issues
                .iter()
                .any(|i| matches!(i, Issue::MemoryExceeded { device, .. } if device == dev));
            prop_assert!(found);
        }
    }

    #[test]
    fn migration_ends_in_the_target_plan(
        model_seed in any::<u64>(),
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
        picks in proptest::collection::vec(any::<usize>(), 1..64),
    ) {
        for hw in hardware() {
            let m = gen_model(&mut Rng(model_seed));
            let none = BTreeMap::new();
            let build = |seed: u64| {
                let mut rng = Rng(seed);
                let phase = rng.range(2, 6) as u8;
                let plan = gen_plan(&mut rng, &m, &hw, phase, false);
                validate_with(&plan, &m, &hw, 6, &none).unwrap()
            };
            let (ea, eb) = (build(seed_a), build(seed_b));

            prop_assert!(diff(&ea, &ea).is_empty());
            let d = diff(&ea, &eb);
            prop_assert_eq!(d.is_empty(), ea == eb);
            let dag = MigrationDag::from_diff(&d);
            prop_assert_eq!(&dag.steps[0], &Step::Drain);
            prop_assert_eq!(dag.steps.last(), Some(&Step::Commit));

            let mut it = picks.iter().cycle();
            let order = dag.order_with(|_| *it.next().unwrap()).expect("acyclic");
            prop_assert_eq!(order.len(), dag.steps.len());
            prop_assert_eq!(order[0], 0);
            prop_assert_eq!(*order.last().unwrap(), dag.steps.len() - 1);
            let at: BTreeMap<usize, usize> =
                order.iter().enumerate().map(|(pos, s)| (*s, pos)).collect();
            for (before, after) in &dag.edges {
                prop_assert!(at[before] < at[after]);
            }

            let mut state = State::from(&ea);
            for i in order {
                state.apply(&dag.steps[i]).unwrap();
            }
            prop_assert_eq!(state, State::from(&eb));
        }
    }
}

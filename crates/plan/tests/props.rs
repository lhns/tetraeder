//! Property tests on plans (AGENTS.md rule 5): memory never exceeded, every unit computed exactly
//! once, every migration DAG ends in its target plan. Each case runs on every hardware fixture
//! (rule 4). Cases are built from a seed through a small deterministic generator, so a failing
//! seed reproduces exactly.

use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use tetraeder_plan::*;
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const HARDWARE: [&str; 3] = [
    include_str!("../../profiler/fixtures/two_gpu_host.toml"),
    include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    include_str!("../../costmodel/fixtures/wan_two_node.toml"),
];
const SMALL_MOE: &str = include_str!("../fixtures/model_small_moe.toml");

fn hardware() -> Vec<HardwareGraph> {
    HARDWARE
        .iter()
        .map(|t| HardwareGraph::from_toml(t).unwrap())
        .collect()
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }

    /// True once in `n`.
    fn chance(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

fn gen_model(rng: &mut Rng) -> ModelDesc {
    if rng.chance(4) {
        return toml::from_str(SMALL_MOE).unwrap();
    }
    let experts = *rng.pick(&[0u32, 4, 8]);
    let (heads, kv_heads) = *rng.pick(&[(8u32, 4u32), (4, 2), (6, 3), (3, 3), (2, 1)]);
    let source_format = *rng.pick(&[Format::F16, Format::BF16, Format::Q8_0, Format::F32]);
    let mut size = |hi: u64| UnitSize {
        params: rng.range(1000, hi),
        flops_per_token: 1.0e6,
        source_format,
    };
    let (embed, attn, mlp, expert, head) = (
        size(2_000_000),
        size(2_000_000),
        size(500_000),
        size(1_000_000),
        size(2_000_000),
    );
    ModelDesc {
        name: "gen".into(),
        layers: rng.range(1, 6) as u32,
        heads,
        kv_heads,
        experts,
        experts_used: if experts > 0 { 2 } else { 0 },
        embed,
        attn,
        mlp,
        expert: (experts > 0).then_some(expert),
        head,
        kv_bytes_per_token_per_layer: 1024,
        activation_bytes_per_token: 1024,
    }
}

fn sel(kinds: &[UnitKind], experts: Option<Span>) -> UnitSel {
    UnitSel {
        kinds: kinds.to_vec(),
        layers: None,
        experts,
    }
}

fn ids(hw: &HardwareGraph, f: impl Fn(DeviceKind) -> bool) -> Vec<String> {
    let it = hw.devices.iter().filter(|d| f(d.kind));
    it.map(|d| d.id.clone()).collect()
}

/// A valid complete plan for `phase`. With `oversize`, one KV budget is larger than its device,
/// so memory (and only memory) fails.
fn gen_plan(rng: &mut Rng, m: &ModelDesc, hw: &HardwareGraph, phase: u8, oversize: bool) -> Plan {
    let replicas = if phase >= 6 && rng.chance(2) { 2 } else { 1 };
    let mut big = oversize;
    let replicas = (0..replicas)
        .map(|_| Replica {
            stages: gen_stages(rng, m, hw, phase, &mut big),
        })
        .collect();
    Plan {
        model: m.name.clone(),
        knobs: Knobs {
            prefill_micro_batches: rng.range(1, 8) as u32,
            cuda_graphs: rng.chance(2),
            max_batch: rng.chance(2).then(|| rng.range(1, 64) as u32),
        },
        replicas,
    }
}

fn gen_stages(
    rng: &mut Rng,
    m: &ModelDesc,
    hw: &HardwareGraph,
    phase: u8,
    oversize: &mut bool,
) -> Vec<Stage> {
    let compute = ids(hw, |k| k != DeviceKind::Disk);
    let gpus = ids(hw, |k| k == DeviceKind::Gpu);
    let all = ids(hw, |_| true);
    let disks = ids(hw, |k| k == DeviceKind::Disk);
    let source = m.attn.source_format;
    let formats: Vec<Format> = Format::ALL
        .iter()
        .copied()
        .filter(|f| source.at_least(*f))
        .collect();

    let k = 1 + rng.below(m.layers.min(3) as usize);
    let mut cuts = BTreeSet::new();
    while cuts.len() < k - 1 {
        cuts.insert(1 + rng.below(m.layers as usize - 1) as u32);
    }
    let bounds: Vec<u32> = std::iter::once(0)
        .chain(cuts)
        .chain(std::iter::once(m.layers))
        .collect();
    let can_tp =
        phase >= 5 && gpus.len() >= 2 && m.heads.is_multiple_of(2) && m.kv_heads.is_multiple_of(2);

    let mut stages = Vec::new();
    for (s, w) in bounds.windows(2).enumerate() {
        let tp = if can_tp && rng.chance(2) { 2 } else { 1 };
        let group: Vec<String> = if tp == 2 {
            gpus[..2].to_vec()
        } else {
            let first = rng.pick(&compute).clone();
            let mut g = vec![first];
            for d in &compute {
                if !g.contains(d) && rng.chance(2) {
                    g.push(d.clone());
                }
            }
            g
        };
        let mut assign = Vec::new();

        // Attention and MLP: tp shards on the group's devices.
        let (home, comp) = if tp == 2 {
            (group.clone(), rng.chance(2).then(|| group.clone()))
        } else if phase >= 5 && !disks.is_empty() && rng.chance(4) {
            (vec![disks[0].clone()], Some(vec![rng.pick(&group).clone()]))
        } else {
            let d = rng.pick(&group).clone();
            let c = (phase >= 5 && rng.chance(3)).then(|| vec![rng.pick(&group).clone()]);
            (vec![d], c)
        };
        assign.push(Assign {
            units: sel(&[UnitKind::Attn, UnitKind::Mlp], None),
            home: Some(home),
            compute: comp,
            format: Some(*rng.pick(&formats)),
        });
        if s == 0 || s + 2 == bounds.len() {
            assign.push(Assign {
                units: sel(&[UnitKind::Embed, UnitKind::Head], None),
                home: Some(vec![rng.pick(&group).clone()]),
                compute: None,
                format: Some(*rng.pick(&formats)),
            });
        }
        if m.is_moe() {
            let cut = if rng.chance(2) {
                0
            } else {
                rng.range(1, u64::from(m.experts) - 1) as u32
            };
            let spans = if cut == 0 {
                vec![Span(0, m.experts)]
            } else {
                vec![Span(0, cut), Span(cut, m.experts)]
            };
            for span in spans {
                let (home, comp) = match rng.below(3) {
                    0 => (rng.pick(&group).clone(), None),
                    1 => (rng.pick(&all).clone(), Some(rng.pick(&group).clone())),
                    _ => (rng.pick(&group).clone(), Some(rng.pick(&group).clone())),
                };
                assign.push(Assign {
                    units: sel(&[UnitKind::Expert], Some(span)),
                    home: Some(vec![home]),
                    compute: comp.map(|c| vec![c]),
                    format: Some(*rng.pick(&formats)),
                });
            }
        }

        let mut kv = Vec::new();
        for _ in 0..rng.range(1, 2) {
            let device = rng.pick(&group).clone();
            let bytes = if std::mem::take(oversize) {
                hw.device(&device).unwrap().capacity_bytes + 1
            } else {
                rng.range(1, 64 << 20)
            };
            kv.push(DeviceBytes { device, bytes });
        }
        let workspace = if rng.chance(2) {
            vec![DeviceBytes {
                device: rng.pick(&group).clone(),
                bytes: rng.range(1, 32 << 20),
            }]
        } else {
            vec![]
        };
        let expert_cache = if m.is_moe() && rng.chance(2) {
            vec![SlotCache {
                device: rng.pick(&group).clone(),
                slots: rng.range(1, 3) as u32,
            }]
        } else {
            vec![]
        };
        stages.push(Stage {
            layers: Span(w[0], w[1]),
            devices: group,
            tp: Some(tp),
            ep: Some(1),
            assign,
            kv: Some(kv),
            workspace,
            expert_cache,
        });
    }
    stages
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
    for (dev, bytes) in used {
        let cap = hw.device(dev).unwrap().capacity_bytes;
        let limit = cap.saturating_sub(reserve.get(dev).copied().unwrap_or(0));
        assert!(bytes <= limit, "{dev}: {bytes} > {limit}");
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
    issues
        .iter()
        .all(|i| matches!(i, Issue::MemoryExceeded { .. }))
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

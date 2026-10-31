//! Arenas sized from expanded plans never exceed the device budget (ADR 0031), on a fake device
//! for plans from the plan crate's generator, and on ggml's CPU backend for a small plan.

use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_memory::fake::{FakeAllocator, FakeDevice};
use tetraeder_memory::{
    ArenaBuffer, ArenaError, Arenas, Phase, Purpose, arena_sizes, available_for_arenas,
    measure_reserve, phase_bytes, reserve_map,
};
use tetraeder_plan::testing::{Rng, gen_model, gen_plan};
use tetraeder_plan::{
    BudgetKey, Budgets, BufferKey, BufferKind, Context, Expanded, Phase as PlanPhase, PhaseBytes,
    validate,
};
use tetraeder_profiler::HardwareGraph;

const HARDWARE: [&str; 3] = [
    include_str!("../../profiler/fixtures/two_gpu_host.toml"),
    include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    include_str!("../../costmodel/fixtures/wan_two_node.toml"),
];

fn fake_for(hw: &HardwareGraph) -> FakeAllocator {
    FakeAllocator::new(
        hw.devices
            .iter()
            .map(|d| (d.id.clone(), FakeDevice::new(d.capacity_bytes))),
    )
}

fn buffer(e: &mut Expanded, stage: u32, kind: BufferKind, device: &str, bytes: u64) {
    let key = BufferKey {
        replica: 0,
        stage,
        kind,
        device: device.into(),
    };
    e.buffers.insert(key, bytes);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn arenas_never_exceed_the_device_budget(seed in any::<u64>()) {
        for text in HARDWARE {
            let hw = HardwareGraph::from_toml(text).unwrap();
            let mut rng = Rng(seed);
            let m = gen_model(&mut rng);
            let phase = rng.range(2, 6) as u8;
            let plan = gen_plan(&mut rng, &m, &hw, phase, false);
            let reserve: BTreeMap<String, u64> = hw
                .devices
                .iter()
                .map(|d| (d.id.clone(), rng.range(0, 1 << 20)))
                .collect();
            let ctx = Context {
                hw: &hw,
                model: &m,
                phase: PlanPhase(phase),
                reserve: &reserve,
                hints: None,
            };
            let Ok(e) = validate(&plan, &ctx) else { continue };

            let fake = fake_for(&hw);
            let avail = available_for_arenas(&e, &hw, &reserve);
            let wanted = arena_sizes(&e);
            let mut requested: BTreeMap<&str, u64> = BTreeMap::new();
            for ((dev, _), b) in &wanted {
                *requested.entry(dev).or_default() += b;
            }
            match Arenas::allocate(&fake, &e, &avail) {
                Ok(arenas) => {
                    let allocated = arenas.allocated();
                    for (dev, bytes) in &allocated {
                        prop_assert!(*bytes <= avail[dev], "{dev}: {bytes} > {}", avail[dev]);
                        prop_assert!(*bytes >= requested[dev.as_str()]);
                        prop_assert_eq!(fake.used(dev), *bytes);
                        let cap = hw.device(dev).unwrap().capacity_bytes;
                        prop_assert!(*bytes <= cap - reserve[dev]);
                    }
                    for a in arenas.iter() {
                        prop_assert_eq!(a.buffer.size() % 256, 0);
                        prop_assert!(a.buffer.size() >= a.requested && a.requested > 0);
                    }
                }
                // The validator counted the plan's bytes; only alignment can push it over.
                Err(ArenaError::OverBudget { device, needed, available }) => {
                    prop_assert!(needed > available);
                    prop_assert!(requested[device.as_str()] <= available);
                    prop_assert!(needed - requested[device.as_str()] < 256 * 3);
                }
                Err(other) => prop_assert!(false, "{other:?}"),
            }
        }
    }
}

#[test]
fn over_budget_allocates_nothing() {
    let mut e = Expanded::default();
    buffer(&mut e, 0, BufferKind::Kv, "gpu0", 600);
    buffer(&mut e, 0, BufferKind::SlotCache, "gpu0", 600);
    let fake = FakeAllocator::new([("gpu0".to_string(), FakeDevice::new(10_000))]);
    let avail = BTreeMap::from([("gpu0".to_string(), 1000)]);
    // 600 rounds up to 768 each: 1536 > 1000.
    let err = Arenas::allocate(&fake, &e, &avail).unwrap_err();
    assert_eq!(
        err,
        ArenaError::OverBudget {
            device: "gpu0".into(),
            needed: 1536,
            available: 1000
        }
    );
    assert_eq!(fake.used("gpu0"), 0);
}

#[test]
fn stages_sum_into_one_arena_per_purpose() {
    let mut e = Expanded::default();
    buffer(&mut e, 0, BufferKind::Kv, "gpu0", 1000);
    buffer(&mut e, 1, BufferKind::Kv, "gpu0", 2000);
    buffer(&mut e, 1, BufferKind::Workspace, "gpu0", 0);
    buffer(&mut e, 1, BufferKind::SlotCache, "gpu1", 512);
    let fake = FakeAllocator::new([
        ("gpu0".to_string(), FakeDevice::new(1 << 20)),
        ("gpu1".to_string(), FakeDevice::new(1 << 20)),
    ]);
    let a = Arenas::allocate(&fake, &e, &BTreeMap::new()).unwrap();
    assert_eq!(a.iter().count(), 2, "zero-size arena skipped");
    let kv = a.get("gpu0", Purpose::Kv).unwrap();
    assert_eq!((kv.requested, kv.buffer.size()), (3000, 3072));
    assert_eq!(a.get("gpu1", Purpose::Slots).unwrap().buffer.size(), 512);
    assert!(a.get("gpu0", Purpose::Reserve).is_none());
}

#[test]
fn backend_limits_are_errors() {
    let mut e = Expanded::default();
    buffer(&mut e, 0, BufferKind::Kv, "gpu0", 5000);
    let mut dev = FakeDevice::new(1 << 20);
    dev.max_size = 4096;
    let fake = FakeAllocator::new([("gpu0".to_string(), dev)]);
    assert!(matches!(
        Arenas::allocate(&fake, &e, &BTreeMap::new()),
        Err(ArenaError::TooLarge { .. })
    ));
    // The backend itself says no (device smaller than the budget claims).
    let fake = FakeAllocator::new([("gpu0".to_string(), FakeDevice::new(4096))]);
    assert!(matches!(
        Arenas::allocate(&fake, &e, &BTreeMap::new()),
        Err(ArenaError::Alloc { .. })
    ));
    assert!(matches!(
        Arenas::allocate(&fake, &e, &BTreeMap::new()).map(|_| ()),
        Err(ArenaError::Alloc {
            purpose: Purpose::Kv,
            ..
        })
    ));
    let e2 = Expanded::default();
    let unknown = FakeAllocator::new([]);
    assert!(Arenas::allocate(&unknown, &e2, &BTreeMap::new()).is_ok());
    buffer(&mut e, 0, BufferKind::Kv, "nope", 1);
    assert!(matches!(
        Arenas::allocate(&unknown, &e, &BTreeMap::new()),
        Err(ArenaError::UnknownDevice(_))
    ));
}

#[test]
fn phase_bytes_sum_over_stages() {
    let mut e = Expanded::default();
    let key = |stage| BudgetKey {
        replica: 0,
        stage,
        device: "gpu0".into(),
    };
    let b = |ps, pw, ds, dw| Budgets {
        prefill: PhaseBytes {
            slots: ps,
            workspace: pw,
        },
        decode: PhaseBytes {
            slots: ds,
            workspace: dw,
        },
    };
    e.budgets.insert(key(0), b(0, 90, 60, 30));
    e.budgets.insert(key(1), b(10, 20, 40, 5));
    let pre = phase_bytes(&e, "gpu0", Phase::Prefill);
    let dec = phase_bytes(&e, "gpu0", Phase::Decode);
    assert_eq!((pre.slots, pre.workspace), (10, 110));
    assert_eq!((dec.slots, dec.workspace), (100, 35));
    assert_eq!(
        phase_bytes(&e, "gpu1", Phase::Decode),
        PhaseBytes::default()
    );
}

#[test]
fn reserve_is_others_plus_init_plus_overhead() {
    let mut dev = FakeDevice::new(1 << 30);
    dev.used_by_others = 100 << 20;
    dev.init_cost = 300 << 20;
    dev.alloc_overhead = 4096;
    let fake = FakeAllocator::new([("gpu0".to_string(), dev)]);
    let r = measure_reserve(&fake, "gpu0", 1 << 20).unwrap();
    assert_eq!(r.total, 1 << 30);
    assert_eq!(r.others(), 100 << 20);
    assert_eq!(r.init_cost(), 300 << 20);
    assert_eq!(r.alloc_overhead(), 4096);
    assert_eq!(r.reserve(), (400 << 20) + 4096);
    // The test allocation is gone.
    assert_eq!(fake.used("gpu0"), (100 << 20) + (300 << 20));
    assert_eq!(reserve_map(&[r])["gpu0"], (400 << 20) + 4096);
}

#[cfg(feature = "ggml")]
mod cpu {
    use super::*;
    use tetraeder_memory::ggml::GgmlAllocator;

    #[test]
    fn arenas_on_the_cpu_backend() {
        let mut e = Expanded::default();
        buffer(&mut e, 0, BufferKind::Kv, "cpu0", (3 << 20) + 5);
        buffer(&mut e, 0, BufferKind::SlotCache, "cpu0", 1 << 20);
        buffer(&mut e, 0, BufferKind::Workspace, "cpu0", 512 << 10);
        let mut alloc = GgmlAllocator::new();
        assert!(GgmlAllocator::available().iter().any(|n| n == "CPU"));
        alloc.bind_cpu("cpu0").unwrap();
        assert!(alloc.bind("gpu9", "no-such-device").is_err());

        let avail = BTreeMap::from([("cpu0".to_string(), 8 << 20)]);
        let arenas = Arenas::allocate(&alloc, &e, &avail).unwrap();
        assert_eq!(arenas.iter().count(), 3);
        let total: u64 = arenas.allocated()["cpu0"];
        assert!(((4 << 20) + (512 << 10) + 5..=8 << 20).contains(&total));
        for a in arenas.iter() {
            assert!(a.buffer.size() >= a.requested);
            let base = a.buffer.base();
            assert!(!base.is_null());
            // SAFETY: the buffer owns size() bytes at base for as long as `arenas` lives.
            let mem = unsafe { std::slice::from_raw_parts_mut(base, a.buffer.size() as usize) };
            mem.fill(0xAB);
            assert_eq!(mem[mem.len() - 1], 0xAB);
        }

        let over = BTreeMap::from([("cpu0".to_string(), 1 << 20)]);
        assert!(matches!(
            Arenas::allocate(&alloc, &e, &over),
            Err(ArenaError::OverBudget { .. })
        ));
    }

    #[test]
    fn reserve_on_the_cpu_backend() {
        let mut alloc = GgmlAllocator::new();
        alloc.bind_cpu("cpu0").unwrap();
        let r = measure_reserve(&alloc, "cpu0", 1 << 20).unwrap();
        eprintln!("{r:?} reserve={}", r.reserve());
        assert!(r.total > 0 && r.test_bytes >= 1 << 20);
        // Free memory moves under us, so only the shape is checked.
        assert!(r.free_before <= r.total && r.reserve() >= r.others());
    }
}

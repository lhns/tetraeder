//! Phase budgets (ADR 0031). In the fixture, stage 1 on gpu1 has a slot arena of
//! 4 x Q4_K(5M) bytes and no workspace; stage 0 on gpu0 has a 256 MiB workspace and no slots.

use std::collections::BTreeMap;
use tetraeder_plan::*;
use tetraeder_profiler::HardwareGraph;

const MODEL: &str = include_str!("../fixtures/model_small_moe.toml");
const PLAN: &str = include_str!("../fixtures/plan_small_moe_phase2.toml");
const HW: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");

fn plan() -> Plan {
    Plan::from_toml(PLAN).unwrap()
}

fn check(plan: &Plan, phase: u8, reserve: &BTreeMap<String, u64>) -> Result<Expanded, Vec<Issue>> {
    let hw = HardwareGraph::from_toml(HW).unwrap();
    let model: ModelDesc = toml::from_str(MODEL).unwrap();
    validate(
        plan,
        &Context {
            hw: &hw,
            model: &model,
            phase: Phase(phase),
            reserve,
            hints: None,
        },
    )
}

fn ok(plan: &Plan) -> Expanded {
    check(plan, 2, &BTreeMap::new()).unwrap_or_else(|e| panic!("{e:#?}"))
}

fn issues(plan: &Plan, phase: u8) -> Vec<Issue> {
    check(plan, phase, &BTreeMap::new()).expect_err("plan should be rejected")
}

fn slot_arena() -> u64 {
    4 * Format::Q4_K.bytes(5_000_000)
}

fn bytes(slots: u64, workspace: u64) -> PhaseBytes {
    PhaseBytes { slots, workspace }
}

fn with_budget(prefill: PhaseBytes, decode: PhaseBytes) -> Plan {
    let mut p = plan();
    p.replicas[0].stages[1].phase_budgets = vec![PhaseBudget {
        device: "gpu1".into(),
        prefill,
        decode,
    }];
    p
}

fn budget_issue(found: &[Issue], phase: &str) -> bool {
    found
        .iter()
        .any(|i| matches!(i, Issue::PhaseBudget { phase: p, .. } if *p == phase))
}

fn key(stage: u32, device: &str) -> BudgetKey {
    BudgetKey {
        replica: 0,
        stage,
        device: device.into(),
    }
}

#[test]
fn budgets_default_to_the_arenas_and_roundtrip() {
    let e = ok(&plan());
    let ws = bytes(0, 268_435_456);
    assert_eq!(e.budgets[&key(0, "gpu0")].prefill, ws);
    assert_eq!(e.budgets[&key(0, "gpu0")].decode, ws);
    assert_eq!(e.budgets[&key(1, "gpu1")].prefill, bytes(slot_arena(), 0));
    assert_eq!(e.budgets.len(), 2);

    let p = with_budget(
        bytes(0, slot_arena()),
        bytes(slot_arena() / 2, slot_arena() / 2),
    );
    assert_eq!(Plan::from_toml(&p.to_toml().unwrap()).unwrap(), p);
    assert_eq!(Plan::from_json(&p.to_json().unwrap()).unwrap(), p);
}

#[test]
fn prefill_borrows_slot_bytes_for_workspace() {
    let e = ok(&with_budget(bytes(0, slot_arena()), bytes(slot_arena(), 0)));
    let b = e.budgets[&key(1, "gpu1")];
    assert_eq!(b.prefill, bytes(0, slot_arena()));
    assert_eq!(b.decode, bytes(slot_arena(), 0));
}

#[test]
fn budgets_cannot_exceed_the_arenas() {
    // More bytes than the slot and workspace arenas hold: KV would have to be borrowed.
    let p = with_budget(bytes(0, slot_arena() + 1), bytes(slot_arena(), 0));
    let found = issues(&p, 2);
    assert!(budget_issue(&found, "prefill"), "{found:#?}");
    assert!(!budget_issue(&found, "decode"), "{found:#?}");
    // More slots than the slot arena.
    let found = issues(&with_budget(bytes(0, 0), bytes(slot_arena() + 1, 0)), 2);
    assert!(budget_issue(&found, "decode"), "{found:#?}");
    // A device without arenas has nothing to borrow.
    let mut p = plan();
    p.replicas[0].stages[1].phase_budgets = vec![PhaseBudget {
        device: "cpu0".into(),
        prefill: bytes(1, 0),
        decode: bytes(0, 0),
    }];
    let found = issues(&p, 2);
    assert!(budget_issue(&found, "prefill"), "{found:#?}");
}

#[test]
fn budget_given_twice_is_rejected() {
    let mut p = with_budget(bytes(0, 0), bytes(0, 0));
    let again = p.replicas[0].stages[1].phase_budgets[0].clone();
    p.replicas[0].stages[1].phase_budgets.push(again);
    let found = issues(&p, 2);
    assert!(budget_issue(&found, "both"), "{found:#?}");
}

#[test]
fn phase_budgets_count_against_capacity_minus_reserve() {
    let p = with_budget(bytes(0, slot_arena()), bytes(slot_arena(), 0));
    let hw = HardwareGraph::from_toml(HW).unwrap();
    let cap = hw.device("gpu1").unwrap().capacity_bytes;
    let reserve = BTreeMap::from([("gpu1".to_string(), cap)]);
    let found = check(&p, 2, &reserve).unwrap_err();
    for phase in ["prefill", "decode"] {
        assert!(
            found.iter().any(|i| matches!(
                i,
                Issue::PhaseMemoryExceeded { device, phase: p, .. }
                    if device == "gpu1" && *p == phase
            )),
            "{phase}: {found:#?}"
        );
    }
    // Exactly the used bytes left over: both phases fit.
    let used = ok(&p).memory_per_device()["gpu1"];
    let reserve = BTreeMap::from([("gpu1".to_string(), cap - used)]);
    check(&p, 2, &reserve).unwrap();
}

#[test]
fn capability_phase_budgets_from_phase_2() {
    assert!(!Phase(1).capabilities().phase_budgets);
    assert!((2..=9).all(|p| Phase(p).capabilities().phase_budgets));
    let p = with_budget(bytes(0, slot_arena()), bytes(slot_arena(), 0));
    let found = issues(&p, 1);
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::Phase { field, .. } if *field == "phase_budgets")),
        "{found:#?}"
    );
}

#[test]
fn budget_changes_show_in_diff_and_migration() {
    let a = ok(&plan());
    let b = ok(&with_budget(bytes(0, slot_arena()), bytes(slot_arena(), 0)));
    let d = diff(&a, &b);
    assert_eq!(d.budgets.len(), 1);
    assert!(!d.is_empty() && d.shards.is_empty() && d.buffers.is_empty());
    let dag = MigrationDag::from_diff(&d);
    let mut s = State::from(&a);
    for i in dag.order_with(|_| 0).unwrap() {
        s.apply(&dag.steps[i]).unwrap();
    }
    assert_eq!(s, State::from(&b));
}

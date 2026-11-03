//! Per-role formats (ADR 0047): bytes, validation, hints, the per-expert capability, diff.

use std::collections::BTreeMap;
use tetraeder_plan::*;
use tetraeder_profiler::HardwareGraph;

const MODEL: &str = include_str!("../fixtures/model_small_moe.toml");
const PLAN: &str = include_str!("../fixtures/plan_small_moe_phase2.toml");
const HW: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");

fn model() -> ModelDesc {
    toml::from_str(MODEL).unwrap()
}

fn plan() -> Plan {
    Plan::from_toml(PLAN).unwrap()
}

fn check(
    plan: &Plan,
    model: &ModelDesc,
    phase: u8,
    hints: Option<&Hints>,
) -> Result<Expanded, Vec<Issue>> {
    let hw = HardwareGraph::from_toml(HW).unwrap();
    validate(
        plan,
        &Context {
            hw: &hw,
            model,
            phase: Phase(phase),
            reserve: &BTreeMap::new(),
            hints,
        },
    )
}

fn assign(p: &mut Plan, stage: usize, rule: usize) -> &mut Assign {
    &mut p.replicas[0].stages[stage].assign[rule]
}

fn set(a: &mut Assign, role: &str, f: Format) {
    a.role_formats.insert(role.into(), f);
}

fn shard(unit: UnitId) -> ShardKey {
    ShardKey {
        replica: 0,
        unit,
        shard: 0,
        of: 1,
    }
}

fn has_unknown_role(found: &[Issue], name: &str) -> bool {
    found
        .iter()
        .any(|i| matches!(i, Issue::UnknownRole { role, .. } if role == name))
}

#[test]
fn role_formats_set_bytes_and_placement_roles() {
    let mut p = plan();
    set(assign(&mut p, 0, 0), "attn.v", Format::Q4_K);
    let e = check(&p, &model(), 2, None).unwrap();
    let a = &e.shards[&shard(UnitId::Attn(0))];
    assert_eq!(a.format, Format::Q8_0);
    assert_eq!(a.roles["attn.v"], Format::Q4_K);
    assert_eq!(a.roles["attn.q"], Format::Q8_0);
    assert_eq!(
        a.bytes,
        Format::Q8_0.bytes(18_000_000) + Format::Q4_K.bytes(2_000_000)
    );
    // Experts of stage 0 have no override: three roles at the unit format.
    let x = &e.shards[&shard(UnitId::Expert(0, 3))];
    let want = [1_700_000, 1_700_000, 1_600_000].map(|n| Format::Q8_0.bytes(n));
    assert_eq!(x.bytes, want.iter().sum::<u64>());
}

#[test]
fn uncovered_params_stay_f32() {
    let mut m = model();
    m.attn.roles.pop();
    m.attn.roles[0].params = 7_000_000;
    // Roles cover 7M + 2M + 2M of 20M; the other 9M are f32.
    let e = check(&plan(), &m, 2, None).unwrap();
    let want = Format::Q8_0.bytes(7_000_000)
        + Format::Q8_0.bytes(2_000_000) * 2
        + Format::F32.bytes(9_000_000);
    assert_eq!(e.shards[&shard(UnitId::Attn(0))].bytes, want);
}

#[test]
fn unknown_roles_are_rejected() {
    let mut p = plan();
    set(assign(&mut p, 0, 0), "attn.zzz", Format::Q4_K);
    let found = check(&p, &model(), 2, None).unwrap_err();
    assert!(has_unknown_role(&found, "attn.zzz"));

    // A valid name, but the rule selects only experts.
    let mut p = plan();
    set(assign(&mut p, 1, 1), "attn.v", Format::Q4_K);
    let found = check(&p, &model(), 2, None).unwrap_err();
    assert!(has_unknown_role(&found, "attn.v"));
}

#[test]
fn role_formats_only_go_downward() {
    let mut m = model();
    m.attn.roles[2].source_format = Format::Q4_K;
    let mut p = plan();
    set(assign(&mut p, 0, 0), "attn.v", Format::Q8_0);
    let found = check(&p, &m, 2, None).unwrap_err();
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::RoleUpconvert { role, .. } if role == "attn.v"))
    );
}

/// Layer split and overrides, source formats only (valid in phase 1).
fn libllama_plan() -> Plan {
    let mut p = plan();
    assign(&mut p, 0, 0).format = Some(Format::F16);
    let s = &mut p.replicas[0].stages[1];
    s.expert_cache.clear();
    s.assign.truncate(2);
    s.assign[0].format = Some(Format::F16);
    s.assign[1].units.experts = None;
    s.assign[1].home = Some(vec!["cpu0".into()]);
    s.assign[1].format = Some(Format::F16);
    p
}

#[test]
fn role_conversion_needs_the_capability() {
    let mut p = libllama_plan();
    check(&p, &model(), 1, None).unwrap();
    set(assign(&mut p, 0, 0), "attn.v", Format::Q4_K);
    let found = check(&p, &model(), 1, None).unwrap_err();
    assert!(found.iter().any(|i| matches!(
        i,
        Issue::Phase {
            field: "format_conversion",
            ..
        }
    )));
    check(&p, &model(), 2, None).unwrap();
}

#[test]
fn hints_use_the_least_precise_role() {
    let mut p = plan();
    set(assign(&mut p, 1, 0), "attn.v", Format::Q2_K);
    let min = Hints::from_toml("min_format = \"Q4_K\"").unwrap();
    assert!(check(&plan(), &model(), 2, Some(&min)).is_ok());
    let found = check(&p, &model(), 2, Some(&min)).unwrap_err();
    assert!(found.iter().all(|i| matches!(i, Issue::Hint { .. })));

    let mut p = plan();
    set(assign(&mut p, 0, 0), "attn.v", Format::Q2_K);
    let floor = Hints::from_toml(
        "[[format_floor]]\nunits = { kinds = [\"attn\"], layers = [0, 2] }\nformat = \"Q8_0\"\n",
    )
    .unwrap();
    assert!(check(&plan(), &model(), 2, Some(&floor)).is_ok());
    let found = check(&p, &model(), 2, Some(&floor)).unwrap_err();
    assert!(found.iter().any(|i| matches!(i, Issue::Hint { .. })));
}

#[test]
fn experts_of_a_layer_must_share_formats() {
    let per_expert = |found: &[Issue]| {
        found
            .iter()
            .filter(|i| matches!(i, Issue::PerExpertFormats { .. }))
            .count()
    };
    // Different role formats between expert groups 0..4 and 4..8 (layers 2 and 3).
    let mut p = plan();
    set(assign(&mut p, 1, 1), "expert.down", Format::Q6_K);
    for phase in [2, 6] {
        let found = check(&p, &model(), phase, None).unwrap_err();
        assert_eq!(per_expert(&found), 2, "{found:#?}");
    }
    // The same override on both groups is fine.
    set(assign(&mut p, 1, 2), "expert.down", Format::Q6_K);
    check(&p, &model(), 2, None).unwrap();

    // Without roles the unit formats decide.
    let mut m = model();
    m.expert.as_mut().unwrap().roles.clear();
    let mut p = plan();
    assign(&mut p, 1, 2).format = Some(Format::Q8_0);
    let found = check(&p, &m, 2, None).unwrap_err();
    assert_eq!(per_expert(&found), 2, "{found:#?}");
}

#[test]
fn capability_per_expert_formats_in_no_phase() {
    assert!((1..=9).all(|p| !Phase(p).capabilities().per_expert_formats));
}

#[test]
fn role_format_change_is_a_shard_change() {
    let a = check(&plan(), &model(), 2, None).unwrap();
    let mut p = plan();
    set(assign(&mut p, 0, 0), "attn.v", Format::Q4_K);
    let b = check(&p, &model(), 2, None).unwrap();

    let d = diff(&a, &b);
    // Only the attention units of layers 0 and 1 hold an attn.v.
    assert_eq!(d.shards.len(), 2);
    for c in &d.shards {
        let ShardChange::Change { from, to, .. } = c else {
            panic!("{c:?}");
        };
        assert_eq!(from.format, to.format);
        assert_ne!(from.roles, to.roles);
    }
    assert_eq!(
        d.moved_bytes(),
        2 * (Format::Q8_0.bytes(18_000_000) + Format::Q4_K.bytes(2_000_000))
    );
    let dag = MigrationDag::from_diff(&d);
    let mut s = State::from(&a);
    for i in dag.order_with(|_| 0).unwrap() {
        s.apply(&dag.steps[i]).unwrap();
    }
    assert_eq!(s, State::from(&b));
}

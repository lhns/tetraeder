//! Example-based tests for the validator, the capability table, diff and migration.
//! Property tests live in `props.rs`.

use std::collections::BTreeMap;
use tetraeder_plan::*;
use tetraeder_profiler::HardwareGraph;

const MODEL: &str = include_str!("../fixtures/model_small_moe.toml");
const PLAN: &str = include_str!("../fixtures/plan_small_moe_phase2.toml");
const HW: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");

fn model() -> ModelDesc {
    toml::from_str(MODEL).unwrap()
}

fn hw() -> HardwareGraph {
    HardwareGraph::from_toml(HW).unwrap()
}

fn plan() -> Plan {
    Plan::from_toml(PLAN).unwrap()
}

fn check(
    plan: &Plan,
    phase: u8,
    reserve: &BTreeMap<String, u64>,
    hints: Option<&Hints>,
) -> Result<Expanded, Vec<Issue>> {
    let (hw, model) = (hw(), model());
    validate(
        plan,
        &Context {
            hw: &hw,
            model: &model,
            phase: Phase(phase),
            reserve,
            hints,
        },
    )
}

fn ok(plan: &Plan, phase: u8) -> Expanded {
    check(plan, phase, &BTreeMap::new(), None).unwrap_or_else(|e| panic!("{e:#?}"))
}

fn issues(plan: &Plan, phase: u8) -> Vec<Issue> {
    check(plan, phase, &BTreeMap::new(), None).expect_err("plan should be rejected")
}

fn stage(p: &mut Plan, s: usize) -> &mut Stage {
    &mut p.replicas[0].stages[s]
}

#[test]
fn fixture_roundtrips_and_validates() {
    let p = plan();
    assert_eq!(Plan::from_toml(&p.to_toml().unwrap()).unwrap(), p);
    assert_eq!(Plan::from_json(&p.to_json().unwrap()).unwrap(), p);

    let e = ok(&p, 2);
    // embed + 4 layers x (attn, mlp, 8 experts) + head, one shard each.
    assert_eq!(e.shards.len(), 42);
    let key = |unit| ShardKey {
        replica: 0,
        unit,
        shard: 0,
        of: 1,
    };
    let embed = &e.shards[&key(UnitId::Embed)];
    assert_eq!((embed.stage, embed.home.as_str()), (0, "gpu0"));
    assert_eq!(embed.bytes, Format::Q8_0.bytes(100_000_000));
    let cold = &e.shards[&key(UnitId::Expert(3, 6))];
    assert_eq!(
        (cold.home.as_str(), cold.compute.as_str(), cold.format),
        ("nvme0", "cpu0", Format::IQ2_XXS)
    );
    assert_eq!(e.shards[&key(UnitId::Head)].stage, 1);

    // Slot cache: 2 slots x 2 MoE layers x largest expert in the stage (Q4_K).
    let slots = BufferKey {
        replica: 0,
        stage: 1,
        kind: BufferKind::SlotCache,
        device: "gpu1".into(),
    };
    assert_eq!(e.buffers[&slots], 4 * Format::Q4_K.bytes(5_000_000));

    let mem = e.memory_per_device();
    let gpu0 = Format::Q8_0.bytes(100_000_000)
        + 2 * (Format::Q8_0.bytes(20_000_000)
            + Format::Q8_0.bytes(1_000_000)
            + 8 * Format::Q8_0.bytes(5_000_000))
        + 1_073_741_824
        + 268_435_456;
    assert_eq!(mem["gpu0"], gpu0);
    assert_eq!(mem["nvme0"], 8 * Format::IQ2_XXS.bytes(5_000_000));
}

#[test]
fn phase_1_rejects_conversion_tiers_and_compute_override() {
    let found = issues(&plan(), 1);
    for field in ["format", "expert_tiers", "compute"] {
        assert!(
            found
                .iter()
                .any(|i| matches!(i, Issue::Phase { field: f, .. } if f.contains(field))),
            "no phase issue for {field}: {found:#?}"
        );
    }
}

#[test]
fn missing_and_duplicate_units() {
    let mut p = plan();
    stage(&mut p, 1).assign.pop();
    let found = issues(&p, 2);
    let missing = found
        .iter()
        .filter(|i| matches!(i, Issue::Missing { .. }))
        .count();
    assert_eq!(missing, 8, "{found:#?}");

    let mut p = plan();
    let dup = stage(&mut p, 1).assign[1].clone();
    stage(&mut p, 1).assign.push(dup);
    let found = issues(&p, 2);
    assert!(found.contains(&Issue::Duplicate {
        at: At {
            replica: 0,
            stage: Some(1)
        },
        unit: UnitId::Expert(2, 0)
    }));
}

#[test]
fn spans_must_tile_the_model() {
    let mut p = plan();
    stage(&mut p, 1).layers = Span(3, 4);
    assert!(
        issues(&p, 2)
            .iter()
            .any(|i| matches!(i, Issue::Spans { .. }))
    );
}

#[test]
fn memory_counts_the_measured_reserve() {
    let p = plan();
    let cap = hw().device("gpu1").unwrap().capacity_bytes;
    let reserve = BTreeMap::from([("gpu1".to_string(), cap - 1024)]);
    let found = check(&p, 2, &reserve, None).unwrap_err();
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::MemoryExceeded { device, .. } if device == "gpu1")),
        "{found:#?}"
    );
}

#[test]
fn huge_budgets_do_not_wrap_the_memory_check() {
    let mut p = plan();
    // Two budgets on one device sum past u64::MAX.
    stage(&mut p, 0).kv = Some(vec![
        DeviceBytes {
            device: "gpu0".into(),
            bytes: u64::MAX,
        },
        DeviceBytes {
            device: "gpu0".into(),
            bytes: 2,
        },
    ]);
    let found = issues(&p, 2);
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::MemoryExceeded { device, .. } if device == "gpu0")),
        "{found:#?}"
    );
}

#[test]
fn formats_only_convert_downward() {
    let mut p = plan();
    stage(&mut p, 0).assign[0].format = Some(Format::F32);
    assert!(issues(&p, 2).iter().any(|i| matches!(
        i,
        Issue::Upconvert {
            target: Format::F32,
            ..
        }
    )));
}

#[test]
fn disk_home_needs_a_compute_device() {
    let mut p = plan();
    stage(&mut p, 1).assign[2].compute = None;
    assert!(
        issues(&p, 2)
            .iter()
            .any(|i| matches!(i, Issue::NoCompute { .. }))
    );
}

#[test]
fn compute_device_must_be_in_the_group() {
    let mut p = plan();
    stage(&mut p, 1).assign[2].compute = Some(vec!["gpu0".into()]);
    assert!(
        issues(&p, 2)
            .iter()
            .any(|i| matches!(i, Issue::NotInGroup { .. }))
    );
}

/// Stage 0 split over both GPUs with tensor parallel 2; stage 1 keeps gpu1 and cpu0.
fn tp_plan(tp: u32, homes: [&str; 2]) -> Plan {
    let mut p = plan();
    let s = stage(&mut p, 0);
    s.devices = vec!["gpu0".into(), "gpu1".into()];
    s.tp = Some(tp);
    s.assign = vec![
        Assign {
            units: UnitSel {
                kinds: vec![UnitKind::Attn, UnitKind::Mlp],
                ..Default::default()
            },
            home: Some(homes.iter().map(|h| h.to_string()).collect()),
            compute: None,
            format: Some(Format::Q8_0),
        },
        Assign {
            units: UnitSel {
                kinds: vec![UnitKind::Embed, UnitKind::Expert],
                ..Default::default()
            },
            home: Some(vec!["gpu0".into()]),
            compute: None,
            format: Some(Format::Q8_0),
        },
    ];
    p
}

#[test]
fn tensor_parallel_shards() {
    let e = ok(&tp_plan(2, ["gpu0", "gpu1"]), 5);
    let shard = |s| {
        &e.shards[&ShardKey {
            replica: 0,
            unit: UnitId::Attn(1),
            shard: s,
            of: 2,
        }]
    };
    assert_eq!(shard(0).home, "gpu0");
    assert_eq!(shard(1).home, "gpu1");
    assert_eq!(shard(0).bytes, Format::Q8_0.bytes(20_000_000).div_ceil(2));

    let found = issues(&tp_plan(2, ["gpu0", "gpu1"]), 4);
    assert!(
        found.iter().any(|i| matches!(i, Issue::Phase { .. })),
        "{found:#?}"
    );

    let found = issues(&tp_plan(2, ["gpu0", "gpu0"]), 5);
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::SharedShardDevice { .. })),
        "{found:#?}"
    );
}

#[test]
fn degrees_divide_heads_experts_and_devices() {
    // tp 3 divides neither 8 heads, 4 KV heads nor 2 devices.
    let found = issues(&tp_plan(3, ["gpu0", "gpu1"]), 5);
    for what in ["heads", "kv_heads", "devices"] {
        assert!(
            found
                .iter()
                .any(|i| matches!(i, Issue::Degree { degree: "tp", what: w, .. } if *w == what)),
            "no tp issue for {what}: {found:#?}"
        );
    }
    let mut p = plan();
    stage(&mut p, 1).ep = Some(3);
    let found = issues(&p, 5);
    assert!(
        found.iter().any(|i| matches!(
            i,
            Issue::Degree {
                degree: "ep",
                what: "experts",
                ..
            }
        )),
        "{found:#?}"
    );
}

#[test]
fn expert_parallel_spreads_evenly() {
    // ep 2 on gpu1 and cpu0: experts 0..4 compute on gpu1, 4..8 on cpu0. Even, so valid.
    let mut p = plan();
    stage(&mut p, 1).ep = Some(2);
    ok(&p, 5);
    // Move one more expert to cpu0: 3 versus 5 is uneven.
    let s = stage(&mut p, 1);
    s.assign[1].units.experts = Some(Span(0, 3));
    s.assign[2].units.experts = Some(Span(3, 8));
    assert!(
        issues(&p, 5)
            .iter()
            .any(|i| matches!(i, Issue::ExpertSpread { .. }))
    );
}

#[test]
fn hard_hints_hold() {
    let p = plan();
    let hints: Hints = Hints::from_toml(
        r#"
min_format = "Q4_K"

[[pin]]
units = { layers = [0, 2] }
device = "gpu1"

[[forbid]]
units = { kinds = ["expert"] }
tier = "nvme"
"#,
    )
    .unwrap();
    let found = check(&p, 2, &BTreeMap::new(), Some(&hints)).unwrap_err();
    let hint_issues = found
        .iter()
        .filter(|i| matches!(i, Issue::Hint { .. }))
        .count();
    assert!(hint_issues >= 3, "{found:#?}");
    assert!(found.iter().all(|i| matches!(i, Issue::Hint { .. })));
}

#[test]
fn partial_plan_with_open_fields() {
    let mut p = plan();
    let s = stage(&mut p, 1);
    s.devices.clear();
    s.tp = None;
    s.ep = None;
    s.kv = None;
    s.expert_cache.clear();
    s.assign.truncate(1);
    s.assign[0].home = None;
    s.assign[0].format = None;

    let (hw, model, reserve) = (hw(), model(), BTreeMap::new());
    let ctx = Context {
        hw: &hw,
        model: &model,
        phase: Phase(2),
        reserve: &reserve,
        hints: None,
    };
    validate_partial(&p, &ctx).unwrap_or_else(|e| panic!("{e:#?}"));
    let found = validate(&p, &ctx).unwrap_err();
    assert!(found.iter().any(|i| matches!(i, Issue::Open { .. })));
    assert!(found.iter().any(|i| matches!(i, Issue::Missing { .. })));

    // A set field must still be valid in a partial plan.
    stage(&mut p, 1).devices = vec!["gpu7".into()];
    let found = validate_partial(&p, &ctx).unwrap_err();
    assert!(
        found
            .iter()
            .any(|i| matches!(i, Issue::UnknownDevice { .. }))
    );
}

#[test]
fn replicas_and_remote_need_phase_6() {
    let mut p = plan();
    p.replicas.push(p.replicas[0].clone());
    let found = issues(&p, 5);
    assert!(
        found.iter().any(|i| matches!(i, Issue::Phase { .. })),
        "{found:#?}"
    );
    // Two replicas on the same devices double the memory, which still fits here.
    let e = ok(&p, 6);
    assert_eq!(e.shards.len(), 84);
}

// One test per row of docs/plan.md "What each phase can execute".

/// Layer split plus tensor-to-buffer overrides (experts to CPU), source formats only.
fn libllama_plan() -> Plan {
    let mut p = plan();
    stage(&mut p, 0).assign[0].format = Some(Format::F16);
    let s = stage(&mut p, 1);
    s.expert_cache.clear();
    s.assign.truncate(2);
    s.assign[0].format = Some(Format::F16);
    s.assign[1].units.experts = None;
    s.assign[1].home = Some(vec!["cpu0".into()]);
    s.assign[1].format = Some(Format::F16);
    p
}

#[test]
fn capability_layer_split_and_overrides_all_phases() {
    for phase in 1..=9 {
        ok(&libllama_plan(), phase);
    }
}

#[test]
fn capability_expert_tiers_from_phase_2() {
    assert!(!Phase(1).capabilities().expert_tiers);
    assert!((2..=9).all(|p| Phase(p).capabilities().expert_tiers));
}

#[test]
fn capability_compute_device_per_shard() {
    assert_eq!(
        Phase(1).capabilities().compute_override,
        ComputeOverride::None
    );
    for p in 2..=4 {
        assert_eq!(
            Phase(p).capabilities().compute_override,
            ComputeOverride::ExpertsOnly
        );
    }
    assert_eq!(
        Phase(5).capabilities().compute_override,
        ComputeOverride::Any
    );
}

#[test]
fn capability_tensor_and_expert_parallel_from_phase_5() {
    assert!((1..=4).all(|p| !Phase(p).capabilities().tp_ep));
    assert!((5..=9).all(|p| Phase(p).capabilities().tp_ep));
}

#[test]
fn capability_replicas_and_nodes_from_phase_6() {
    assert!((1..=5).all(|p| !Phase(p).capabilities().multi_node));
    assert!((6..=9).all(|p| Phase(p).capabilities().multi_node));
}

#[test]
fn capability_format_conversion_from_phase_2() {
    assert!(!Phase(1).capabilities().convert_formats);
    assert!((2..=9).all(|p| Phase(p).capabilities().convert_formats));
}

#[test]
fn diff_and_migration_move_cold_experts_to_ram() {
    let a = plan();
    let mut b = plan();
    stage(&mut b, 1).assign[2].home = Some(vec!["cpu0".into()]);
    stage(&mut b, 1).assign[2].compute = None;
    let (ea, eb) = (ok(&a, 2), ok(&b, 2));

    assert!(diff(&ea, &ea).is_empty());
    let d = diff(&ea, &eb);
    assert_eq!(d.shards.len(), 8);
    assert!(
        d.shards
            .iter()
            .all(|c| matches!(c, ShardChange::Change { .. }))
    );
    assert_eq!(d.moved_bytes(), 8 * Format::IQ2_XXS.bytes(5_000_000));

    let dag = MigrationDag::from_diff(&d);
    assert_eq!(dag.steps.first(), Some(&Step::Drain));
    let order = dag.order_with(|_| 0).expect("acyclic");
    assert_eq!(order.len(), dag.steps.len());
    assert_eq!(dag.steps[*order.last().unwrap()], Step::Commit);
    let mut s = State::from(&ea);
    for i in order {
        s.apply(&dag.steps[i]).unwrap();
    }
    assert_eq!(s, State::from(&eb));
}

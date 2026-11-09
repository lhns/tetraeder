//! KV cache format per attention unit (ADR 0050 b): IR, sizes, validator, token capacity.
//!
//! Hardware label: none (simulated hardware fixture only).

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

fn issues(p: &Plan) -> Vec<Issue> {
    let (hw, m) = (HardwareGraph::from_toml(HW).unwrap(), model());
    let reserve = BTreeMap::new();
    let ctx = Context {
        hw: &hw,
        model: &m,
        phase: Phase(2),
        reserve: &reserve,
        hints: None,
    };
    validate(p, &ctx).err().unwrap_or_default()
}

fn kv_issue(i: &[Issue]) -> bool {
    i.iter().any(|x| matches!(x, Issue::KvFormat { .. }))
}

#[test]
fn kv_format_serde_and_display() {
    for kv in KvFormat::ALLOWED {
        let s = serde_json::to_string(&kv).unwrap();
        assert_eq!(serde_json::from_str::<KvFormat>(&s).unwrap(), kv);
    }
    assert_eq!(KvFormat::default(), KvFormat::both(Format::F16));
    assert_eq!(KvFormat::Q8_0_Q4_0.to_string(), "q8_0/q4_0");
    assert_eq!(KvFormat::F16.to_string(), "f16/f16");
    assert!(KvFormat::Q4_0.is_allowed());
    assert!(
        !KvFormat {
            k: Format::Q4_0,
            v: Format::Q8_0
        }
        .is_allowed()
    );
}

#[test]
fn kv_bytes_for_the_235b_shape() {
    // Qwen3-235B-A22B: 4 kv heads x 128 = 512 = n_embd_kv; the field is K plus V in F16.
    let m = ModelDesc {
        kv_bytes_per_token_per_layer: 4 * 512,
        ..model()
    };
    assert_eq!(m.kv_bytes_per_token_per_layer_as(KvFormat::F16), 2048);
    assert_eq!(m.kv_bytes_per_token_per_layer_as(KvFormat::Q8_0), 1088);
    assert_eq!(m.kv_bytes_per_token_per_layer_as(KvFormat::Q8_0_Q4_0), 832);
    assert_eq!(KvFormat::Q8_0.bytes(512), 1088);
}

#[test]
fn old_plans_parse_and_validate_unchanged() {
    let p = plan();
    assert!(
        p.replicas[0]
            .stages
            .iter()
            .flat_map(|s| &s.assign)
            .all(|a| a.kv_format.is_none())
    );
    assert!(!p.to_toml().unwrap().contains("kv_format"));
    assert!(issues(&p).is_empty());
    let kv = p.kv_formats(&model(), 0);
    assert_eq!(kv.len(), 4);
    assert!(kv.values().all(|f| *f == KvFormat::F16));
}

#[test]
fn validator_checks_the_pair_and_the_rule() {
    let mut p = plan();
    p.replicas[0].stages[1].assign[0].kv_format = Some(KvFormat::Q8_0_Q4_0);
    assert!(issues(&p).is_empty());

    let q4k = KvFormat::both(Format::Q4_K);
    p.replicas[0].stages[1].assign[0].kv_format = Some(q4k);
    assert!(kv_issue(&issues(&p)));

    p.replicas[0].stages[1].assign[0].kv_format = None;
    p.replicas[0].stages[1].assign[1].kv_format = Some(KvFormat::Q8_0);
    assert!(kv_issue(&issues(&p)), "expert-only rule");
}

#[test]
fn kv_formats_follow_the_rules() {
    let mut p = plan();
    p.replicas[0].stages[1].assign[0].kv_format = Some(KvFormat::Q8_0);
    let kv = p.kv_formats(&model(), 0);
    assert_eq!(kv[&0], KvFormat::F16);
    assert_eq!(kv[&2], KvFormat::Q8_0);
    assert_eq!(kv[&3], KvFormat::Q8_0);
}

#[test]
fn kv_tokens_grow_with_the_format() {
    let mut p = plan();
    let m = model();
    // Stage 1: 2 layers, 512 MiB on gpu1, 4096 B per token and layer in F16.
    assert_eq!(p.kv_tokens(&m, 0, 1, "gpu1"), (512 << 20) / (2 * 4096));
    p.replicas[0].stages[1].assign[0].kv_format = Some(KvFormat::Q8_0);
    let per_layer = m.kv_bytes_per_token_per_layer_as(KvFormat::Q8_0);
    assert_eq!(p.kv_tokens(&m, 0, 1, "gpu1"), (512 << 20) / (2 * per_layer));
    assert_eq!(
        p.kv_tokens(&m, 0, 1, "gpu0"),
        0,
        "no budget on gpu0 in stage 1"
    );
}

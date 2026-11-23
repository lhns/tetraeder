//! Plans across nodes (ADR 0091, step 2; AGENTS.md rules 4 and 5): the validator's node rules,
//! `forbid node`, and segments on simulated clusters built from the committed node fixtures.
//!
//! Hardware label: none (simulated hardware fixtures only).

use std::collections::BTreeMap;
use tetraeder_plan::testing::{Rng, gen_model, gen_plan};
use tetraeder_plan::*;
use tetraeder_profiler::{
    CollectiveCurve, CollectiveOp, HardwareGraph, Link, LinkKind, Measurement,
};

const NODE: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
const SMALL_MOE: &str = include_str!("../fixtures/model_small_moe.toml");

fn net(a: &str, b: &str) -> Link {
    Link {
        id: format!("net-{a}-{b}"),
        a: format!("{a}/cpu0"),
        b: format!("{b}/cpu0"),
        kind: LinkKind::Network,
        alpha_s: Measurement::prior(0.01),
        beta_bytes_per_s: Measurement::prior(1.0e8),
        lane_group: None,
    }
}

/// Nodes `a` and `b` (Linux) and `c` (Windows), each the two-GPU fixture box.
fn cluster() -> HardwareGraph {
    let box_ = HardwareGraph::from_toml(NODE).unwrap();
    let mut g = HardwareGraph::merge_cluster(
        &[
            ("a".into(), box_.clone()),
            ("b".into(), box_.clone()),
            ("c".into(), box_),
        ],
        vec![net("a", "b"), net("a", "c"), net("b", "c")],
        vec![CollectiveCurve {
            op: CollectiveOp::Copy,
            ranks: vec!["a/cpu0".into(), "b/cpu0".into()],
            comm: "tls".into(),
            points: vec![(4096.0, 0.01), (1048576.0, 0.05)],
        }],
    );
    for d in &mut g.devices {
        d.os = Some(
            if d.node.as_deref() == Some("c") {
                "windows"
            } else {
                "linux"
            }
            .into(),
        );
    }
    g.validate().unwrap();
    g
}

fn model() -> ModelDesc {
    toml::from_str(SMALL_MOE).unwrap()
}

fn rule(kinds: &str, home: &str, compute: Option<&str>) -> String {
    let compute = compute.map_or(String::new(), |c| format!("compute = [\"{c}\"]\n"));
    format!(
        "[[replicas.stages.assign]]\nunits = {{ kinds = [{kinds}] }}\nhome = [\"{home}\"]\n{compute}format = \"F16\"\n"
    )
}

/// One stage: layers, device group, its rules.
fn stage(layers: (u32, u32), devices: &[&str], kv: &str, rules: &[String]) -> String {
    let ds = devices
        .iter()
        .map(|d| format!("\"{d}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[[replicas.stages]]\nlayers = [{}, {}]\ndevices = [{ds}]\ntp = 1\nep = 1\nkv = [{{ device = \"{kv}\", bytes = 1048576 }}]\n{}",
        layers.0,
        layers.1,
        rules.concat()
    )
}

fn plan_of(stages: &[String]) -> Plan {
    let text = format!("model = \"small-moe\"\n[[replicas]]\n{}", stages.concat());
    Plan::from_toml(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

const EMB: &str = "\"embed\"";
const REST: &str = "\"attn\", \"mlp\", \"expert\"";
const HEAD: &str = "\"head\"";

/// Layers 0..2 on a, 2..4 on b; the head on b's stage.
fn two_nodes() -> Plan {
    plan_of(&[
        stage(
            (0, 2),
            &["a/gpu0"],
            "a/gpu0",
            &[rule(EMB, "a/gpu0", None), rule(REST, "a/gpu0", None)],
        ),
        stage(
            (2, 4),
            &["b/gpu0"],
            "b/gpu0",
            &[rule(REST, "b/gpu0", None), rule(HEAD, "b/gpu0", None)],
        ),
    ])
}

fn check(plan: &Plan, hw: &HardwareGraph, phase: u8, hints: Option<&Hints>) -> Vec<Issue> {
    let reserve = BTreeMap::new();
    let m = model();
    let ctx = Context {
        hw,
        model: &m,
        phase: Phase(phase),
        reserve: &reserve,
        hints,
    };
    validate(plan, &ctx).err().unwrap_or_default()
}

#[test]
fn one_stage_per_node_is_valid_from_phase_six() {
    let hw = cluster();
    assert_eq!(check(&two_nodes(), &hw, 6, None), vec![]);
    // Before Phase 6 the engine cannot run a plan on two nodes.
    let early = check(&two_nodes(), &hw, 5, None);
    assert!(
        early.iter().any(|i| matches!(
            i,
            Issue::Phase {
                field: "multi_node",
                ..
            }
        )),
        "{early:?}"
    );
}

#[test]
fn a_group_never_spans_nodes() {
    let hw = cluster();
    let p = plan_of(&[stage(
        (0, 4),
        &["a/gpu0", "b/gpu0"],
        "a/gpu0",
        &[
            rule(EMB, "a/gpu0", None),
            rule("\"attn\", \"mlp\", \"expert\", \"head\"", "a/gpu0", None),
        ],
    )]);
    let issues = check(&p, &hw, 6, None);
    assert!(
        issues.iter().any(|i| matches!(
            i,
            Issue::StageSpansNodes { nodes, .. } if nodes == &["a".to_string(), "b".to_string()]
        )),
        "{issues:?}"
    );
}

#[test]
fn forbid_node_is_a_hard_hint() {
    let hw = cluster();
    let hints = Hints {
        forbid: vec![Forbid {
            units: None,
            device: None,
            node: Some("b".into()),
            tier: None,
        }],
        ..Default::default()
    };
    hints.check().unwrap();
    let issues = check(&two_nodes(), &hw, 6, Some(&hints));
    assert_eq!(
        issues,
        vec![Issue::ForbiddenNode {
            at: At {
                replica: 0,
                stage: Some(1)
            },
            node: "b".into()
        }]
    );
    // Only some units: the node may hold the others.
    let only_head = Hints {
        forbid: vec![Forbid {
            units: Some(UnitSel {
                kinds: vec![UnitKind::Head],
                ..Default::default()
            }),
            device: None,
            node: Some("a".into()),
            tier: None,
        }],
        ..Default::default()
    };
    assert_eq!(check(&two_nodes(), &hw, 6, Some(&only_head)), vec![]);
    // The hint round-trips through TOML.
    let again = Hints::from_toml(&hints.to_toml().unwrap()).unwrap();
    assert_eq!(again, hints);
}

/// Layers 0..2 on a, 2..4 on b, the head computing on `head_dev`.
fn head_on(head_dev: &str) -> Plan {
    plan_of(&[
        stage(
            (0, 2),
            &["a/gpu0"],
            "a/gpu0",
            &[rule(EMB, "a/gpu0", None), rule(REST, "a/gpu0", None)],
        ),
        stage(
            (2, 4),
            &["b/gpu0"],
            "b/gpu0",
            &[
                rule(REST, "b/gpu0", None),
                rule(HEAD, head_dev, Some(head_dev)),
            ],
        ),
    ])
}

#[test]
fn the_head_may_compute_on_the_leaders_node_only() {
    let hw = cluster();
    // The leader is a (first stage): allowed, and segments show a head-only segment on a.
    let p = head_on("a/gpu0");
    assert_eq!(check(&p, &hw, 6, None), vec![]);
    let segs = segments(&p, 0, &hw).unwrap();
    let nodes: Vec<_> = segs.iter().map(|s| s.node.as_deref()).collect();
    assert_eq!(nodes, [Some("a"), Some("b"), Some("a")]);
    assert_eq!(segs[2].stages, 2..2);
    assert_eq!((segs[1].head, segs[2].head), (false, true));
    // Another node's GPU is refused.
    let issues = check(&head_on("c/gpu0"), &hw, 6, None);
    assert!(
        issues
            .iter()
            .any(|i| matches!(i, Issue::CrossNode { device, .. } if device == "c/gpu0")),
        "{issues:?}"
    );
}

/// Layers 2..4 on b with the experts 4..8 homed and computed on `dev`.
fn remote_experts(dev: &str) -> Plan {
    plan_of(&[
        stage(
            (0, 2),
            &["a/gpu0"],
            "a/gpu0",
            &[rule(EMB, "a/gpu0", None), rule(REST, "a/gpu0", None)],
        ),
        stage(
            (2, 4),
            &["b/gpu0"],
            "b/gpu0",
            &[
                "[[replicas.stages.assign]]\nunits = { kinds = [\"expert\"], experts = [0, 4] }\nhome = [\"b/gpu0\"]\nformat = \"F16\"\n".to_string(),
                format!(
                    "[[replicas.stages.assign]]\nunits = {{ kinds = [\"expert\"], experts = [4, 8] }}\nhome = [\"{dev}\"]\ncompute = [\"{dev}\"]\nformat = \"F16\"\n"
                ),
                rule("\"attn\", \"mlp\", \"head\"", "b/gpu0", None),
            ],
        ),
    ])
}

#[test]
fn experts_compute_on_another_node_only_between_linux_nodes() {
    let hw = cluster();
    assert_eq!(check(&remote_experts("a/gpu1"), &hw, 6, None), vec![]);
    let issues = check(&remote_experts("c/gpu1"), &hw, 6, None);
    assert!(
        issues.iter().any(|i| matches!(
            i,
            Issue::CrossNode { detail, .. } if detail.contains("Linux")
        )),
        "{issues:?}"
    );
}

#[test]
fn single_node_graphs_have_no_node_rules() {
    // The same stage group on a graph without nodes is plain `NotInGroup`, as before.
    let hw = HardwareGraph::from_toml(NODE).unwrap();
    let p = plan_of(&[stage(
        (0, 4),
        &["gpu0"],
        "gpu0",
        &[
            rule(EMB, "gpu0", None),
            rule("\"attn\", \"mlp\", \"expert\"", "gpu0", None),
            rule(HEAD, "gpu1", Some("gpu1")),
        ],
    )]);
    let issues = check(&p, &hw, 6, None);
    assert!(
        issues.iter().all(|i| matches!(i, Issue::NotInGroup { .. })) && !issues.is_empty(),
        "{issues:?}"
    );
}

#[test]
fn node_graph_gives_back_the_box() {
    let hw = cluster();
    let box_ = HardwareGraph::from_toml(NODE).unwrap();
    for n in ["a", "b", "c"] {
        let g = hw.node_graph(Some(n));
        g.validate().unwrap();
        let mut want = box_.clone();
        for d in &mut want.devices {
            d.os = Some(if n == "c" { "windows" } else { "linux" }.into());
        }
        assert_eq!(g, want, "node {n}");
    }
    // No node on any device: the graph itself.
    assert_eq!(box_.node_graph(None), box_);
    assert_eq!(hw.node_of("b/gpu1"), Some("b"));
    assert_eq!(hw.nodes(), [Some("a"), Some("b"), Some("c")]);
    assert!(hw.network_link("b", "a").is_some());
    assert!(hw.network_curve("a", "b").is_some() && hw.network_curve("b", "a").is_none());
}

#[test]
fn segments_of_a_one_node_plan_is_the_plan() {
    let hw = cluster();
    let p = plan_of(&[stage(
        (0, 4),
        &["a/gpu0"],
        "a/gpu0",
        &[
            rule(EMB, "a/gpu0", None),
            rule("\"attn\", \"mlp\", \"expert\", \"head\"", "a/gpu0", None),
        ],
    )]);
    assert_eq!(check(&p, &hw, 6, None), vec![]);
    let segs = segments(&p, 0, &hw).unwrap();
    assert_eq!(
        segs,
        [Segment {
            node: Some("a".into()),
            stages: 0..1,
            layers: Span(0, 4),
            embed: true,
            head: true
        }]
    );
    let sp = segment_plan(&p, 0, &segs[0]);
    let mut want = p.clone();
    // Ids lose the node prefix.
    let strip = |s: &str| s.strip_prefix("a/").unwrap().to_string();
    let st = &mut want.replicas[0].stages[0];
    st.devices = st.devices.iter().map(|d| strip(d)).collect();
    st.kv = st.kv.as_ref().map(|k| {
        k.iter()
            .map(|b| DeviceBytes {
                device: strip(&b.device),
                bytes: b.bytes,
            })
            .collect()
    });
    for a in &mut st.assign {
        a.home = a
            .home
            .as_ref()
            .map(|h| h.iter().map(|d| strip(d)).collect());
    }
    assert_eq!(sp, want);
}

#[test]
fn a_stage_on_no_single_node_has_no_segments() {
    let hw = cluster();
    let p = plan_of(&[stage(
        (0, 4),
        &["a/gpu0", "b/gpu0"],
        "a/gpu0",
        &[rule(EMB, "a/gpu0", None), rule(REST, "a/gpu0", None)],
    )]);
    assert!(segments(&p, 0, &hw).is_err());
}

#[test]
fn head_apart_needs_its_own_rule() {
    let hw = cluster();
    let p = plan_of(&[
        stage(
            (0, 2),
            &["a/gpu0"],
            "a/gpu0",
            &[rule(EMB, "a/gpu0", None), rule(REST, "a/gpu0", None)],
        ),
        stage(
            (2, 4),
            &["b/gpu0"],
            "b/gpu0",
            &[rule(
                "\"attn\", \"mlp\", \"expert\", \"head\"",
                "a/gpu0",
                Some("a/gpu0"),
            )],
        ),
    ]);
    assert!(segments(&p, 0, &hw).is_err());
}

/// Units a segment holds: the layers' units, the embedding in the first, the head in one.
fn units_of(m: &ModelDesc, s: &Segment) -> Vec<UnitId> {
    m.units()
        .filter(|u| match u {
            UnitId::Embed => s.embed,
            UnitId::Head => s.head,
            _ => u.layer().is_some_and(|l| s.layers.contains(l)),
        })
        .collect()
}

/// Rule 5: generated plans on the cluster graph. Every plan the generator makes on one node of
/// it has one segment equal to the plan; stitched together by hand, two nodes' plans cover the
/// layers in order, with the embedding first and the head once.
#[test]
fn generated_plans_have_covering_segments() {
    let hw = cluster();
    let mut seen = 0;
    for seed in 0..60u64 {
        let mut rng = Rng(seed);
        let m = gen_model(&mut rng);
        for node in ["a", "b", "c"] {
            let g = hw.node_graph(Some(node));
            let plan = gen_plan(&mut rng, &m, &g, 5, false);
            // Move the generated plan onto the node: prefix every device id.
            let moved = prefix(&plan, node);
            let segs = segments(&moved, 0, &hw).unwrap();
            assert_eq!(segs.len(), 1, "seed {seed} node {node}");
            let s = &segs[0];
            assert_eq!(s.layers, Span(0, m.layers));
            assert!(s.embed && s.head);
            assert_eq!(units_of(&m, s).len(), m.units().count());
            seen += 1;
        }
    }
    assert!(seen >= 150, "only {seen} generated plans");
}

fn prefix(plan: &Plan, node: &str) -> Plan {
    let p = |d: &String| format!("{node}/{d}");
    let all = |v: &Vec<String>| v.iter().map(p).collect::<Vec<_>>();
    let bytes = |v: &[DeviceBytes]| {
        v.iter()
            .map(|b| DeviceBytes {
                device: p(&b.device),
                bytes: b.bytes,
            })
            .collect::<Vec<_>>()
    };
    let mut out = plan.clone();
    for rep in &mut out.replicas {
        for st in &mut rep.stages {
            st.devices = all(&st.devices);
            st.kv = st.kv.as_deref().map(bytes);
            st.workspace = bytes(&st.workspace);
            for c in &mut st.expert_cache {
                c.device = p(&c.device);
            }
            for b in &mut st.phase_budgets {
                b.device = p(&b.device);
            }
            for a in &mut st.assign {
                a.home = a.home.as_ref().map(all);
                a.compute = a.compute.as_ref().map(all);
            }
        }
    }
    out
}

#[test]
fn two_node_segments_cover_the_model_in_order() {
    let hw = cluster();
    let m = model();
    for p in [two_nodes(), head_on("a/gpu0")] {
        let segs = segments(&p, 0, &hw).unwrap();
        let mut next = 0;
        let (mut embeds, mut heads) = (0, 0);
        for s in &segs {
            assert_eq!(s.layers.0, next);
            next = s.layers.1;
            embeds += usize::from(s.embed);
            heads += usize::from(s.head);
        }
        assert_eq!(next, m.layers);
        assert_eq!((embeds, heads), (1, 1));
        assert!(segs[0].embed);
        // Every unit is held by exactly one segment.
        let mut count: BTreeMap<UnitId, u32> = BTreeMap::new();
        for s in &segs {
            for u in units_of(&m, s) {
                *count.entry(u).or_default() += 1;
            }
        }
        assert_eq!(count.len(), m.units().count());
        assert!(count.values().all(|&c| c == 1));
        // Concatenating the segment plans gives back the stages: same layers, same rules.
        let stages: usize = segs
            .iter()
            .map(|s| segment_plan(&p, 0, s).replicas[0].stages.len())
            .sum();
        assert_eq!(
            stages,
            p.replicas[0].stages.len() + usize::from(segs.len() > p.replicas[0].stages.len())
        );
    }
}

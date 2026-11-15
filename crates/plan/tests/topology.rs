//! Topology and slicing tests (ADR 0069, 0070), on literal 1x to 4x GPU hardware and the small
//! MoE model. Property tests cover random stage layouts and block slices.
//!
//! Hardware label: none (simulated hardware only).

use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_plan::testing::Rng;
use tetraeder_plan::*;
use tetraeder_profiler::HardwareGraph;

const MODEL: &str = include_str!("../fixtures/model_small_moe.toml");

fn model() -> ModelDesc {
    toml::from_str(MODEL).unwrap()
}

/// `n` GPUs (NVML index 0..n, device ids gpu0..) and one CPU.
fn hw(n: usize) -> HardwareGraph {
    let mut s = String::from(
        "[[devices]]\nid = \"cpu0\"\nkind = \"cpu\"\nname = \"cpu\"\ntier = \"ram\"\n\
         capacity_bytes = 68719476736\n",
    );
    for i in 0..n {
        s += &format!(
            "[[devices]]\nid = \"gpu{i}\"\nkind = \"gpu\"\nname = \"gpu\"\ntier = \"vram\"\n\
             capacity_bytes = 25769803776\nindex = {i}\n"
        );
    }
    HardwareGraph::from_toml(&s).unwrap()
}

fn rule(kinds: &[UnitKind], experts: Option<Span>, home: &[&str]) -> Assign {
    Assign {
        units: UnitSel {
            kinds: kinds.to_vec(),
            layers: None,
            experts,
        },
        home: Some(home.iter().map(|h| h.to_string()).collect()),
        compute: None,
        format: Some(Format::Q8_0),
        role_formats: BTreeMap::new(),
        kv_format: None,
    }
}

/// A stage over `devs` with tp = devs.len(); experts are cut (ep = 1) or whole (ep = tp).
fn stage(layers: (u32, u32), devs: &[&str], ep: u32, embed: bool, head: bool) -> Stage {
    let tp = devs.len() as u32;
    let mut assign = vec![rule(&[UnitKind::Attn, UnitKind::Mlp], None, devs)];
    if embed {
        assign.push(rule(&[UnitKind::Embed], None, &devs[..1]));
    }
    if head {
        assign.push(rule(&[UnitKind::Head], None, &devs[..1]));
    }
    if ep > 1 {
        let per = 8 / ep;
        for (i, d) in devs.iter().enumerate() {
            let span = Span(i as u32 * per, (i as u32 + 1) * per);
            assign.push(rule(&[UnitKind::Expert], Some(span), &[d]));
        }
    } else {
        assign.push(rule(&[UnitKind::Expert], None, devs));
    }
    Stage {
        layers: Span(layers.0, layers.1),
        devices: devs.iter().map(|d| d.to_string()).collect(),
        tp: Some(tp),
        ep: Some(ep),
        assign,
        kv: Some(vec![DeviceBytes {
            device: devs[0].to_string(),
            bytes: 1 << 20,
        }]),
        ..Default::default()
    }
}

fn plan_of(stages: Vec<Stage>) -> Plan {
    Plan {
        model: "small-moe".into(),
        knobs: Knobs::default(),
        replicas: vec![Replica { stages }],
    }
}

fn topo(n_gpu: usize, stages: Vec<Stage>) -> Topology {
    let (hw, m, plan) = (hw(n_gpu), model(), plan_of(stages));
    let reserve = BTreeMap::new();
    let ctx = Context {
        hw: &hw,
        model: &m,
        phase: Phase(5),
        reserve: &reserve,
        hints: None,
    };
    let x = validate(&plan, &ctx).unwrap_or_else(|e| panic!("{e:#?}"));
    Topology::from_plan(&plan, &x, &hw).unwrap_or_else(|e| panic!("{e:#?}"))
}

fn names(t: &Topology) -> Vec<(&str, Option<usize>)> {
    t.ranks
        .iter()
        .map(|r| (r.device.as_str(), r.gpu_index))
        .collect()
}

#[test]
fn single_matches_a_one_stage_plan() {
    let t = topo(1, vec![stage((0, 4), &["gpu0"], 1, true, true)]);
    assert_eq!(t, Topology::single("gpu0", Some(0), 4));
    assert!(t.is_single());
    assert_eq!(t.stage_of_layer(3).layers, Span(0, 4));
}

#[test]
fn layer_2_has_two_stages_with_embed_first_and_head_last() {
    let t = topo(
        2,
        vec![
            stage((0, 1), &["gpu0"], 1, true, false),
            stage((1, 4), &["gpu1"], 1, false, true),
        ],
    );
    assert_eq!(names(&t), [("gpu0", Some(0)), ("gpu1", Some(1))]);
    assert!(!t.is_single());
    let flags: Vec<_> = t.stages.iter().map(|s| (s.embed, s.head)).collect();
    assert_eq!(flags, [(true, false), (false, true)]);
    assert_eq!(t.stages[0].ranks, [0]);
    assert_eq!(t.stages[1].ranks, [1]);
    assert_eq!(t.stage_of_layer(0).ranks, [0]);
    assert_eq!(t.stage_of_layer(1).ranks, [1]);
    assert_eq!(t.n_layer, 4);
}

#[test]
fn tp_2_and_ep_2_are_one_stage_over_two_ranks() {
    for ep in [1, 2] {
        let t = topo(2, vec![stage((0, 4), &["gpu0", "gpu1"], ep, true, true)]);
        assert_eq!(t.ranks.len(), 2);
        assert_eq!(t.stages.len(), 1);
        let s = &t.stages[0];
        assert_eq!((s.tp, s.ep, s.embed, s.head), (2, ep, true, true));
        assert_eq!(s.ranks, [0, 1]);
    }
}

#[test]
fn rank_order_follows_the_stage_devices_order() {
    let t = topo(2, vec![stage((0, 4), &["gpu1", "gpu0"], 1, true, true)]);
    assert_eq!(names(&t), [("gpu1", Some(1)), ("gpu0", Some(0))]);
}

#[test]
fn tp2_pp2_on_four_gpus() {
    let t = topo(
        4,
        vec![
            stage((0, 2), &["gpu0", "gpu1"], 1, true, false),
            stage((2, 4), &["gpu2", "gpu3"], 1, false, true),
        ],
    );
    assert_eq!(t.ranks.len(), 4);
    assert_eq!(t.stages[0].ranks, [0, 1]);
    assert_eq!(t.stages[1].ranks, [2, 3]);
    assert_eq!(t.stages.iter().map(|s| s.tp).collect::<Vec<_>>(), [2, 2]);
    assert_eq!(t.stage_of_layer(2).ranks, [2, 3]);
}

#[test]
fn a_device_in_two_stages_is_one_rank() {
    let t = topo(
        1,
        vec![
            stage((0, 2), &["gpu0"], 1, true, false),
            stage((2, 4), &["gpu0"], 1, false, true),
        ],
    );
    assert_eq!(t.ranks.len(), 1);
    assert_eq!(t.stages.len(), 2);
    assert!(!t.is_single());
}

fn errors(plan: &Plan, x: &Expanded, n_gpu: usize) -> Vec<String> {
    Topology::from_plan(plan, x, &hw(n_gpu)).expect_err("plan should not run")
}

#[test]
fn every_error_is_collected() {
    // Stage 0: a CPU, and tp = 1 with ep = 2. Stage 1 starts at layer 2 and names an unknown
    // device.
    let mut a = stage((0, 1), &["cpu0"], 1, true, false);
    a.ep = Some(2);
    let b = stage((2, 4), &["gpu0", "gpu9"], 1, false, true);
    let e = errors(&plan_of(vec![a, b]), &Expanded::default(), 1);
    let has = |s: &str| e.iter().any(|m| m.contains(s));
    assert!(has("stage 0: device cpu0 is not a GPU"), "{e:#?}");
    assert!(has("stage 0: ep = 2 needs tp = ep"), "{e:#?}");
    assert!(
        has("stage 1: layers [2, 4) do not continue at layer 1"),
        "{e:#?}"
    );
    assert!(has("stage 1: unknown device gpu9"), "{e:#?}");
    assert!(e.len() >= 4);

    let mut p = plan_of(vec![stage((0, 4), &["gpu0"], 1, true, true)]);
    p.replicas.push(p.replicas[0].clone());
    assert!(errors(&p, &Expanded::default(), 1)[0].contains("2 replicas"));
    p.replicas.clear();
    assert!(errors(&p, &Expanded::default(), 1)[0].contains("0 replicas"));
}

#[test]
fn shards_off_the_stage_gpus_are_rejected() {
    let (hw2, m) = (hw(2), model());
    let plan = plan_of(vec![stage((0, 4), &["gpu0"], 1, true, true)]);
    let reserve = BTreeMap::new();
    let ctx = Context {
        hw: &hw2,
        model: &m,
        phase: Phase(5),
        reserve: &reserve,
        hints: None,
    };
    let mut x = validate(&plan, &ctx).unwrap();
    Topology::from_plan(&plan, &x, &hw2).unwrap();
    for p in x.shards.values_mut() {
        p.home = "cpu0".into();
        p.compute = "gpu0".into();
    }
    let e = errors(&plan, &x, 2);
    assert!(
        e.iter().any(|m| m.contains("outside the stage's devices")),
        "{e:#?}"
    );
    assert!(e.iter().any(|m| m.contains("computes on gpu0")), "{e:#?}");
    // One line per reason, not per shard.
    assert!(e.len() <= 3, "{e:#?}");
}

#[test]
fn slot_caches_are_rejected() {
    let mut s = stage((0, 4), &["gpu0"], 1, true, true);
    s.expert_cache = vec![SlotCache {
        device: "gpu0".into(),
        slots: 1,
    }];
    let e = errors(&plan_of(vec![s]), &Expanded::default(), 1);
    assert!(e.iter().any(|m| m.contains("slot caches")), "{e:#?}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For random stage layouts every layer maps to exactly one stage and every rank belongs to
    /// at least one stage; a device in two stages is one rank.
    #[test]
    fn layers_and_ranks_are_covered_exactly(seed in any::<u64>()) {
        let mut rng = Rng(seed);
        let n_layer = rng.range(1, 24) as u32;
        let k = 1 + rng.below(n_layer.min(6) as usize);
        let mut cuts = std::collections::BTreeSet::new();
        while cuts.len() < k - 1 {
            cuts.insert(1 + rng.below(n_layer as usize - 1) as u32);
        }
        let bounds: Vec<u32> = std::iter::once(0)
            .chain(cuts)
            .chain(std::iter::once(n_layer))
            .collect();
        let stages: Vec<Stage> = bounds
            .windows(2)
            .map(|w| {
                let tp = *rng.pick(&[1usize, 2, 4]);
                let mut pool: Vec<usize> = (0..4).collect();
                let mut devices = Vec::new();
                for _ in 0..tp {
                    devices.push(format!("gpu{}", pool.remove(rng.below(pool.len()))));
                }
                let ep = if tp > 1 && rng.chance(2) { tp as u32 } else { 1 };
                Stage {
                    layers: Span(w[0], w[1]),
                    devices,
                    tp: Some(tp as u32),
                    ep: Some(ep),
                    ..Default::default()
                }
            })
            .collect();
        let plan = plan_of(stages.clone());
        let t = Topology::from_plan(&plan, &Expanded::default(), &hw(4)).unwrap();
        prop_assert_eq!(t.n_layer, n_layer);
        prop_assert_eq!(t.stages.len(), stages.len());
        for il in 0..n_layer {
            let n = t.stages.iter().filter(|s| s.layers.contains(il)).count();
            prop_assert_eq!(n, 1);
            prop_assert!(t.stage_of_layer(il).layers.contains(il));
        }
        for r in 0..t.ranks.len() {
            prop_assert!(t.stages.iter().any(|s| s.ranks.contains(&r)));
        }
        let mut devices: Vec<&str> = t.ranks.iter().map(|r| r.device.as_str()).collect();
        let before = devices.len();
        devices.sort_unstable();
        devices.dedup();
        prop_assert_eq!(devices.len(), before);
        for (s, st) in t.stages.iter().zip(&stages) {
            prop_assert_eq!(s.ranks.len(), st.devices.len());
            for (r, d) in s.ranks.iter().zip(&st.devices) {
                prop_assert_eq!(&t.ranks[*r].device, d);
            }
            prop_assert_eq!(s.embed, s.layers.0 == 0);
            prop_assert_eq!(s.head, s.layers.1 == n_layer);
        }
    }

    /// Slices tile `[0, len)` exactly, in whole blocks, and differ by at most one block; the
    /// extra blocks go to ranks `(rot + j) % n`.
    #[test]
    fn block_slices_tile_the_range(
        block in 1u64..512,
        n in 1u32..9,
        extra_blocks in 0u64..200,
        rot in 0u32..40,
    ) {
        let blocks = u64::from(n) + extra_blocks;
        let len = blocks * block;
        let mut at = 0;
        let (base, extra) = (blocks / u64::from(n), blocks % u64::from(n));
        for i in 0..n {
            let r = block_slice(len, block, n, i, rot).unwrap();
            prop_assert_eq!(r.start, at);
            prop_assert_eq!((r.end - r.start) % block, 0);
            let got = (r.end - r.start) / block;
            let pos = (u64::from(i) + u64::from(n) - u64::from(rot) % u64::from(n))
                % u64::from(n);
            prop_assert_eq!(got, base + u64::from(pos < extra));
            at = r.end;
        }
        prop_assert_eq!(at, len);
        prop_assert!(block_slice(len, block, n, n, rot).is_err());
    }
}

#[test]
fn block_slice_examples() {
    // Qwen3-30B-A3B experts: 3 blocks of 256 over 2 ranks.
    assert_eq!(block_slice(768, 256, 2, 0, 0).unwrap(), 0..512);
    assert_eq!(block_slice(768, 256, 2, 1, 0).unwrap(), 512..768);
    // The rotation swaps who gets the extra block.
    assert_eq!(block_slice(768, 256, 2, 0, 1).unwrap(), 0..256);
    assert_eq!(block_slice(768, 256, 2, 1, 1).unwrap(), 256..768);
    // Rotation wraps over n.
    assert_eq!(block_slice(768, 256, 2, 0, 2).unwrap(), 0..512);
    // 3 blocks over 4 ranks leaves one empty.
    assert!(block_slice(768, 256, 4, 0, 0).is_err());
    // Qwen3-32B FFN: 100 blocks over 4.
    for i in 0..4 {
        let r = block_slice(25600, 256, 4, i, 7).unwrap();
        assert_eq!(r, u64::from(i) * 6400..u64::from(i + 1) * 6400);
    }
    // Not whole blocks; degenerate arguments.
    assert!(block_slice(700, 256, 2, 0, 0).is_err());
    assert!(block_slice(768, 0, 2, 0, 0).is_err());
    assert!(block_slice(768, 256, 0, 0, 0).is_err());
}

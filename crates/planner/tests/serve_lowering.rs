//! Lowering of executable plans (ADR 0055, AGENTS.md rule 5): `recognize` inverts `build_plan`
//! for the shapes the enumerators emit, every candidate lowers, the patterns and slot counts say
//! what the plan says, projection is idempotent, and every served candidate validates and fits.

mod common;

use common::{CTX, WORK, graphs, has_gpu, input};
use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use tetraeder_plan::{Context, Knobs, KvFormat, Phase, Plan, UnitKind, validate};
use tetraeder_planner::candidate::{CPU, CUDA_RESERVE, GPU, HOST_RESERVE};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    Applies, Budget, Lowered, ModelLayout, Placement, ServeEngine, Workload, build_plan,
    candidates, enumerate, enumerate_builder, enumerate_cpu, footprint, lower, recognize,
};

/// Tiny matcher for the regex subset `range_regex` emits.
fn matches(pattern: &str, n: u32) -> bool {
    let s = n.to_string();
    pattern.split('|').any(|alt| {
        let mut cls = Vec::new();
        let b = alt.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'[' {
                cls.push((b[i + 1], b[i + 3]));
                i += 5;
            } else {
                cls.push((b[i], b[i]));
                i += 1;
            }
        }
        cls.len() == s.len()
            && s.bytes()
                .zip(&cls)
                .all(|(d, &(lo, hi))| (lo..=hi).contains(&d))
    })
}

/// Layers the `-ot` pattern sends to the CPU: `exps=CPU` is all of them, otherwise the layer
/// alternatives inside `blk\.(...)\.`.
fn pattern_layers(patterns: &[String], n_layer: u32) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for p in patterns {
        if p == "exps=CPU" {
            out.extend(0..n_layer);
            continue;
        }
        let alts = p
            .strip_prefix(r"blk\.(")
            .and_then(|r| r.split_once(r")\."))
            .unwrap_or_else(|| panic!("unexpected pattern {p}"))
            .0;
        out.extend((0..n_layer).filter(|&n| matches(alts, n)));
    }
    out
}

/// Layers whose experts the plan assigns to `cpu0`.
fn cpu_expert_layers(plan: &Plan, n_layer: u32) -> BTreeSet<u32> {
    let st = &plan.replicas[0].stages[0];
    let mut out = BTreeSet::new();
    for a in st
        .assign
        .iter()
        .filter(|a| a.units.kinds.contains(&UnitKind::Expert))
    {
        if a.home.as_deref() == Some(&[CPU.to_string()][..]) {
            let span = a.units.layers.unwrap_or(tetraeder_plan::Span(0, n_layer));
            out.extend(span.0..span.1);
        }
    }
    out
}

fn arb_knobs() -> impl Strategy<Value = Knobs> {
    (
        1u32..=64,
        prop::sample::select(vec![128u32, 256, 512, 2048]),
        any::<bool>(),
    )
        .prop_map(|(t, ub, op)| Knobs {
            threads: Some(t),
            ubatch: Some(ub),
            op_offload: Some(op),
            ..Default::default()
        })
}

fn arb_dense_placement() -> impl Strategy<Value = Placement> {
    prop_oneof![
        (0u32..=64).prop_map(|g| Placement::Dense { gpu_layers: g }),
        Just(Placement::Cpu),
    ]
}

fn arb_moe_placement() -> impl Strategy<Value = Placement> {
    prop_oneof![
        (0u32..=48).prop_map(|k| Placement::Moe {
            gpu_expert_layers: k
        }),
        (1u32..=128, 0u32..=128).prop_map(|(v, r)| Placement::Builder {
            vram_slots: v,
            ram_slots: r
        }),
        Just(Placement::Cpu),
    ]
}

proptest! {
    #[test]
    fn recognize_inverts_build_plan_on_dense(p in arb_dense_placement(), k in arb_knobs()) {
        let l = fixtures::dense();
        let plan = build_plan(&l, p, k, k.ubatch.unwrap(), &WORK, KvFormat::default());
        prop_assert_eq!(recognize(&plan, &l), Ok((p, k)));
    }

    #[test]
    fn recognize_inverts_build_plan_on_moe(p in arb_moe_placement(), k in arb_knobs()) {
        let l = fixtures::moe();
        let plan = build_plan(&l, p, k, k.ubatch.unwrap(), &WORK, KvFormat::default());
        prop_assert_eq!(recognize(&plan, &l), Ok((p, k)));
    }
}

fn layouts() -> Vec<(&'static str, ModelLayout)> {
    vec![("dense", fixtures::dense()), ("moe", fixtures::moe())]
}

/// Checks one candidate against its lowering.
fn check_lowering(plan: &Plan, layout: &ModelLayout, engine: ServeEngine, what: &str) {
    let (placement, knobs) = recognize(plan, layout).unwrap_or_else(|e| panic!("{what}: {e}"));
    let lowered = lower(plan, layout, engine).unwrap_or_else(|e| panic!("{what}: {e}"));
    match lowered {
        Lowered::Llama {
            cpu_patterns,
            threads,
            ubatch,
            op_offload,
            ..
        } => {
            let want = if layout.is_moe() {
                cpu_expert_layers(plan, layout.n_layer)
            } else {
                BTreeSet::new()
            };
            let got = pattern_layers(&cpu_patterns, layout.n_layer);
            // A Cpu or Moe plan: the pattern matches exactly the CPU experts. Dense MoE-free
            // plans have none.
            if matches!(placement, Placement::Cpu | Placement::Moe { .. }) {
                if layout.is_moe() && placement == Placement::Cpu {
                    // the Cpu placement runs without -ngl: nothing to override
                    assert!(cpu_patterns.is_empty(), "{what}");
                } else {
                    assert_eq!(got, want, "{what}");
                }
            }
            assert_eq!(Some(threads), knobs.threads, "{what}");
            assert_eq!(Some(ubatch), knobs.ubatch, "{what}");
            assert_eq!(Some(op_offload), knobs.op_offload, "{what}");
        }
        Lowered::Builder {
            gpu,
            expert_slots,
            ram_slots,
            ..
        } => {
            let st = &plan.replicas[0].stages[0];
            let slots = |d: &str| {
                st.expert_cache
                    .iter()
                    .find(|c| c.device == d)
                    .map(|c| c.slots)
            };
            match placement {
                Placement::Builder {
                    vram_slots,
                    ram_slots: r,
                } => {
                    assert_eq!(gpu, Some(0), "{what}");
                    assert_eq!(slots(GPU), Some(vram_slots), "{what}");
                    assert_eq!(slots(CPU), Some(r), "{what}");
                    assert_eq!(
                        expert_slots.unwrap_or(layout.n_expert),
                        vram_slots.min(layout.n_expert),
                        "{what}"
                    );
                    assert_eq!(ram_slots.unwrap_or(0), r, "{what}");
                }
                Placement::Cpu => {
                    assert_eq!((gpu, expert_slots, ram_slots), (None, None, None), "{what}");
                }
                p => panic!("{what}: {p:?} lowered to the builder"),
            }
        }
    }
}

fn reserve() -> BTreeMap<String, u64> {
    BTreeMap::from([
        (GPU.to_string(), CUDA_RESERVE),
        (CPU.to_string(), HOST_RESERVE),
    ])
}

#[test]
fn every_enumerated_candidate_lowers() {
    for (gname, hw) in graphs() {
        for (lname, layout) in layouts() {
            let budget: Budget = common::input(&hw, &layout, ServeEngine::Llama).budget;
            let w = Workload {
                prompt: CTX - WORK.r#gen,
                r#gen: WORK.r#gen,
            };
            let mut sets: Vec<(&str, ServeEngine, Vec<Plan>)> = Vec::new();
            let plans = |en: tetraeder_planner::Enumeration| {
                en.candidates
                    .into_iter()
                    .map(|c| c.plan)
                    .collect::<Vec<_>>()
            };
            if has_gpu(&hw) {
                sets.push((
                    "enumerate",
                    ServeEngine::Llama,
                    plans(enumerate(&budget, &layout, &w)),
                ));
                if layout.is_moe() {
                    sets.push((
                        "enumerate_builder",
                        ServeEngine::Builder,
                        plans(enumerate_builder(&budget, &layout, &w)),
                    ));
                }
            }
            sets.push((
                "enumerate_cpu",
                ServeEngine::Llama,
                plans(enumerate_cpu(&budget, &layout, &w)),
            ));
            if layout.is_moe() {
                sets.push((
                    "enumerate_cpu",
                    ServeEngine::Builder,
                    plans(enumerate_cpu(&budget, &layout, &w)),
                ));
            }
            for (which, engine, ps) in sets {
                for (i, p) in ps.iter().enumerate() {
                    let what = format!("{gname}/{lname}/{which}/{engine:?}/c{i:02}");
                    check_lowering(p, &layout, engine, &what);
                    for applies in [Applies::today(), Applies::all()] {
                        let (once, _) = applies.project(p, engine);
                        let (twice, notes) = applies.project(&once, engine);
                        assert_eq!(once, twice, "{what}: project is idempotent");
                        assert!(notes.is_empty(), "{what}");
                        assert_eq!(once.replicas.len(), p.replicas.len(), "{what}");
                        assert_eq!(
                            once.replicas, p.replicas,
                            "{what}: project keeps the placement"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn served_candidates_validate_and_fit() {
    for (gname, hw) in graphs() {
        for (lname, layout) in layouts() {
            let engines: &[ServeEngine] = if layout.is_moe() {
                &[ServeEngine::Llama, ServeEngine::Builder]
            } else {
                &[ServeEngine::Llama]
            };
            for &engine in engines {
                let inp = input(&hw, &layout, engine);
                let cands = candidates(&inp);
                assert!(
                    !cands.is_empty(),
                    "{gname}/{lname}/{engine:?}: no candidate"
                );
                let model = layout.to_model_desc();
                let reserve = reserve();
                let ctx = Context {
                    hw: &hw,
                    model: &model,
                    phase: Phase(3),
                    reserve: &reserve,
                    hints: None,
                };
                let sizing = Workload {
                    prompt: CTX - WORK.r#gen,
                    r#gen: WORK.r#gen,
                };
                for (i, p) in cands.iter().enumerate() {
                    let what = format!("{gname}/{lname}/{engine:?}/{i}");
                    if let Err(e) = validate(p, &ctx) {
                        panic!("{what}: {e:#?}");
                    }
                    let (pl, k) = recognize(p, &layout).unwrap();
                    let f = footprint(&layout, pl, k.ubatch.unwrap(), &sizing, KvFormat::default());
                    assert!(f.gpu <= inp.budget.gpu_bytes, "{what}");
                    assert!(f.host <= inp.budget.ram_bytes, "{what}");
                    assert!(k.threads.unwrap() <= inp.budget.threads, "{what}");
                    check_lowering(p, &layout, engine, &what);
                }
            }
        }
    }
}

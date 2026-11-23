//! `kv = auto` (ADR 0086, AGENTS.md rules 4 and 5): on every simulated hardware fixture the
//! planner takes the first of F16, Q8_0, Q8_0/Q4_0 that has a feasible plan at the context; the
//! chosen plan never exceeds VRAM or RAM with the KV bytes of the chosen format counted.
//!
//! The oracle is the footprint of the lightest placement of a side (fewest non-KV bytes), not the
//! enumerators, so the two are checked against each other. Contexts come from the oracle's
//! boundaries, not from constants.

mod common;

use common::{CTX, WORK, graphs, has_gpu, input};
use proptest::prelude::*;
use std::collections::BTreeMap;
use tetraeder_plan::{Context, KvFormat, Phase, Plan, UnitKind, validate};
use tetraeder_planner::builder::{BUILDER_UBATCHES, ram_slot_options};
use tetraeder_planner::candidate::{CPU, CUDA_RESERVE, GPU, HOST_RESERVE, UBATCHES};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    AUTO_ORDER, Budget, KvChoice, ModelLayout, Placement, ServeEngine, ServeInput, Workload,
    enumerate_choice, enumerate_kv, footprint, pick_for_placement, pick_kv,
};
use tetraeder_profiler::HardwareGraph;

const STEP: u32 = 256;
const MAX_CTX: u32 = 1 << 24;

fn layouts() -> Vec<(&'static str, ModelLayout)> {
    vec![("dense", fixtures::dense()), ("moe", fixtures::moe())]
}

fn sizing(ctx: u32) -> Workload {
    Workload {
        prompt: ctx.saturating_sub(WORK.r#gen).max(1),
        r#gen: WORK.r#gen,
    }
}

fn fits_light(
    layout: &ModelLayout,
    b: &Budget,
    side: Option<ServeEngine>,
    kv: KvFormat,
    ctx: u32,
) -> bool {
    let fits = |pl, ub| {
        let f = footprint(layout, pl, ub, &sizing(ctx), kv);
        f.gpu <= b.gpu_bytes && f.host <= b.ram_bytes
    };
    if side == Some(ServeEngine::Builder) {
        // GPU and host bytes pull in opposite directions with the VRAM tier: any tier may do.
        return (layout.n_expert_used..=layout.n_expert).any(|v| {
            let pl = Placement::Builder {
                vram_slots: v,
                ram_slots: ram_slot_options(layout, v)[0],
            };
            fits(pl, BUILDER_UBATCHES[0])
        });
    }
    if side.is_some() {
        // libllama: more GPU layers move weights and KV off the host, so any count may do.
        let lo = u32::from(!layout.is_moe());
        return (lo..=layout.n_layer).any(|x| {
            let pl = if layout.is_moe() {
                Placement::Moe {
                    gpu_expert_layers: x,
                }
            } else {
                Placement::Dense { gpu_layers: x }
            };
            fits(pl, UBATCHES[0])
        });
    }
    fits(Placement::Cpu, UBATCHES[0])
}

/// The format `auto` must pick and whether it runs on the CPU, by the oracle.
fn expected(inp: &ServeInput, hw: &HardwareGraph, ctx: u32) -> Option<(KvFormat, bool)> {
    let (l, b) = (&inp.layout, &inp.budget);
    if has_gpu(hw) {
        for kv in AUTO_ORDER {
            if fits_light(l, b, Some(inp.engine), kv, ctx) {
                return Some((kv, false));
            }
        }
    }
    AUTO_ORDER
        .into_iter()
        .find(|&kv| fits_light(l, b, None, kv, ctx))
        .map(|kv| (kv, true))
}

/// Largest context (a multiple of [`STEP`]) where `ok` holds, 0 when not even [`STEP`] does.
fn max_ctx(ok: impl Fn(u32) -> bool) -> u32 {
    if !ok(STEP) {
        return 0;
    }
    let (mut lo, mut hi) = (STEP, MAX_CTX);
    while lo < hi {
        let mid = (lo + (hi - lo).div_ceil(2)).div_ceil(STEP) * STEP;
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid - STEP;
        }
    }
    lo
}

fn engines(layout: &ModelLayout) -> Vec<ServeEngine> {
    if layout.is_moe() {
        vec![ServeEngine::Llama, ServeEngine::Builder]
    } else {
        vec![ServeEngine::Llama]
    }
}

/// Contexts around every boundary of every format and side, plus two small ones.
fn probes(inp: &ServeInput, hw: &HardwareGraph) -> Vec<u32> {
    let mut v = vec![STEP, CTX];
    let sides: Vec<Option<ServeEngine>> = if has_gpu(hw) {
        vec![Some(inp.engine), None]
    } else {
        vec![None]
    };
    for side in sides {
        for kv in AUTO_ORDER {
            let m = max_ctx(|c| fits_light(&inp.layout, &inp.budget, side, kv, c));
            v.extend(
                [m, m + STEP, m.saturating_sub(STEP)]
                    .into_iter()
                    .filter(|&c| c >= STEP),
            );
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn kv_of(plan: &Plan) -> Option<KvFormat> {
    let mut found = None;
    for a in plan.replicas[0].stages[0]
        .assign
        .iter()
        .filter(|a| a.units.kinds.contains(&UnitKind::Attn))
    {
        match (found, a.kv_format) {
            (None, f) => found = Some(f),
            (Some(prev), f) => assert_eq!(prev, f, "attention rules disagree on the KV format"),
        }
    }
    found.flatten()
}

/// Every candidate fits the budget with the KV bytes of `kv` counted, and its plan says `kv`.
fn check_fits(inp: &ServeInput, en: &tetraeder_planner::Enumeration, kv: KvFormat, what: &str) {
    assert!(!en.candidates.is_empty(), "{what}: no candidate");
    let w = sizing(inp.ctx);
    for c in &en.candidates {
        let f = footprint(&inp.layout, c.placement, c.ubatch(), &w, kv);
        assert!(f.gpu <= inp.budget.gpu_bytes, "{what}/{}: gpu", c.id);
        assert!(f.host <= inp.budget.ram_bytes, "{what}/{}: host", c.id);
        let say = (kv != KvFormat::default()).then_some(kv);
        assert_eq!(kv_of(&c.plan), say, "{what}/{}: plan kv format", c.id);
    }
}

#[test]
fn auto_follows_the_oracle_on_every_fixture() {
    // Which outcomes the sweep saw: F16, Q8_0, Q8_0/Q4_0, none.
    let mut seen = [false; 4];
    for (gname, hw) in graphs() {
        for (lname, layout) in layouts() {
            for engine in engines(&layout) {
                let mut inp = input(&hw, &layout, engine);
                for ctx in probes(&inp, &hw) {
                    inp.ctx = ctx;
                    let what = format!("{gname}/{lname}/{engine:?}/ctx {ctx}");
                    match (
                        enumerate_choice(&inp, KvChoice::Auto),
                        expected(&inp, &hw, ctx),
                    ) {
                        (Ok((en, Some(pick))), Some((kv, cpu))) => {
                            assert_eq!(pick.format, kv, "{what}: {}", pick.reason);
                            check_fits(&inp, &en, kv, &what);
                            assert_eq!(pick.reason.contains("on the CPU"), cpu && has_gpu(&hw));
                            assert!(pick.reason.contains("fits"), "{what}: {}", pick.reason);
                            for c in &en.candidates {
                                assert!(c.notes.contains(&pick.explain()), "{what}");
                            }
                            let i = AUTO_ORDER.iter().position(|&f| f == kv).unwrap();
                            seen[i] = true;
                            assert_eq!(pick.skipped.len(), i, "{what}");
                        }
                        (Err(e), None) => {
                            assert!(e.contains("no KV format fits"), "{what}: {e}");
                            seen[3] = true;
                        }
                        (got, want) => {
                            panic!("{what}: got {:?}, oracle {want:?}", got.map(|g| g.1))
                        }
                    }
                }
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "outcomes seen {seen:?}");
}

#[test]
fn chosen_plans_validate() {
    for (gname, hw) in graphs() {
        for (lname, layout) in layouts() {
            for engine in engines(&layout) {
                let mut inp = input(&hw, &layout, engine);
                let model = layout.to_model_desc();
                let reserve: BTreeMap<String, u64> = [
                    (GPU.to_string(), CUDA_RESERVE),
                    (CPU.to_string(), HOST_RESERVE),
                ]
                .into_iter()
                .collect();
                for ctx in probes(&inp, &hw) {
                    inp.ctx = ctx;
                    let Ok((en, Some(pick))) = enumerate_choice(&inp, KvChoice::Auto) else {
                        continue;
                    };
                    let vctx = Context {
                        hw: &hw,
                        model: &model,
                        phase: Phase(3),
                        reserve: &reserve,
                        hints: None,
                    };
                    for c in &en.candidates {
                        if let Err(e) = validate(&c.plan, &vctx) {
                            panic!("{gname}/{lname}/{engine:?}/ctx {ctx}/{}: {e:#?}", c.id);
                        }
                    }
                    assert!(en.candidates.iter().all(|c| kv_of(&c.plan)
                        == (pick.format != KvFormat::default()).then_some(pick.format)));
                }
            }
        }
    }
}

#[test]
fn only_q8_fits_names_the_numbers_and_records_the_format() {
    let (_, hw) = graphs()
        .into_iter()
        .find(|(n, _)| *n == "rtx3090_prior")
        .unwrap();
    let layout = fixtures::moe();
    let mut inp = input(&hw, &layout, ServeEngine::Builder);
    let m16 = max_ctx(|c| {
        fits_light(
            &layout,
            &inp.budget,
            Some(ServeEngine::Builder),
            KvFormat::F16,
            c,
        )
    });
    let m8 = max_ctx(|c| {
        fits_light(
            &layout,
            &inp.budget,
            Some(ServeEngine::Builder),
            KvFormat::Q8_0,
            c,
        )
    });
    assert!(
        m16 + STEP <= m8,
        "a context band where only Q8_0 fits: {m16} {m8}"
    );
    inp.ctx = m16 + STEP;
    let (en, pick) = enumerate_choice(&inp, KvChoice::Auto).unwrap();
    let pick = pick.unwrap();
    assert_eq!(pick.format, KvFormat::Q8_0);
    assert!(pick.reason.starts_with("f16/f16 needs "), "{}", pick.reason);
    assert!(pick.reason.contains(" GB KV, "), "{}", pick.reason);
    assert!(
        pick.reason.contains(" GB free; q8_0/q8_0 fits"),
        "{}",
        pick.reason
    );
    assert!(
        en.candidates
            .iter()
            .all(|c| kv_of(&c.plan) == Some(KvFormat::Q8_0))
    );
    // The default does not change: F16 has no candidate there, so the CPU is the fallback.
    let (def, none) = enumerate_choice(&inp, KvChoice::Default).unwrap();
    assert!(none.is_none());
    assert!(def.candidates.iter().all(|c| kv_of(&c.plan).is_none()));
}

#[test]
fn default_and_fixed_enumerate_as_before() {
    for (gname, hw) in graphs().into_iter().filter(|(_, h)| has_gpu(h)) {
        for (lname, layout) in layouts() {
            let inp = input(&hw, &layout, ServeEngine::Llama);
            let (en, pick) = enumerate_choice(&inp, KvChoice::Default).unwrap();
            assert!(pick.is_none());
            let w = sizing(inp.ctx);
            let old = tetraeder_planner::enumerate(&inp.budget, &layout, &w);
            if !old.candidates.is_empty() {
                assert_eq!(en, old, "{gname}/{lname}");
            }
            let (q8, _) = enumerate_choice(&inp, KvChoice::Fixed(KvFormat::Q8_0)).unwrap();
            let want = enumerate_kv(&inp.budget, &layout, &w, KvFormat::Q8_0);
            if !want.candidates.is_empty() {
                assert_eq!(q8, want, "{gname}/{lname}");
            }
        }
    }
}

#[test]
fn bf16_and_q4_0_are_never_picked() {
    let (_, hw) = graphs()
        .into_iter()
        .find(|(n, _)| *n == "rtx3090_prior")
        .unwrap();
    let layout = fixtures::moe();
    let mut inp = input(&hw, &layout, ServeEngine::Builder);
    for ctx in probes(&inp, &hw) {
        inp.ctx = ctx;
        if let Ok((_, Some(p))) = enumerate_choice(&inp, KvChoice::Auto) {
            assert!(AUTO_ORDER.contains(&p.format));
            assert!(![KvFormat::BF16, KvFormat::Q4_0].contains(&p.format));
        }
    }
    assert_eq!(
        AUTO_ORDER,
        [KvFormat::F16, KvFormat::Q8_0, KvFormat::Q8_0_Q4_0]
    );
}

#[test]
fn quantized_formats_need_a_head_size_of_32s() {
    let mut l = fixtures::moe();
    l.head_dim = 80;
    let r = pick_kv(&l, |kv| {
        if kv == KvFormat::F16 {
            Err("f16 too big".into())
        } else {
            Ok("fits".into())
        }
    });
    let e = r.unwrap_err();
    assert!(e.contains("multiple of 32"), "{e}");
    // F16 still wins when it fits.
    assert_eq!(
        pick_kv(&l, |_| Ok("fits".into())).unwrap().format,
        KvFormat::F16
    );
}

#[test]
fn choice_parses() {
    assert_eq!("auto".parse::<KvChoice>().unwrap(), KvChoice::Auto);
    assert_eq!("AUTO".parse::<KvChoice>().unwrap(), KvChoice::Auto);
    assert_eq!(
        "q8_0".parse::<KvChoice>().unwrap(),
        KvChoice::Fixed(KvFormat::Q8_0)
    );
    assert_eq!(
        "q8_0/q4_0".parse::<KvChoice>().unwrap(),
        KvChoice::Fixed(KvFormat::Q8_0_Q4_0)
    );
    assert!("q4_k".parse::<KvChoice>().is_err());
    assert!("q4_0/q8_0".parse::<KvChoice>().is_err());
    assert_eq!(KvChoice::default().fixed(), Some(KvFormat::F16));
    assert_eq!(KvChoice::Auto.fixed(), None);
}

#[test]
fn a_fixed_placement_picks_by_its_footprint() {
    let layout = fixtures::moe();
    let at = (
        Placement::Moe {
            gpu_expert_layers: 0,
        },
        UBATCHES[0],
    );
    let big = Budget {
        gpu_bytes: 24 << 30,
        ram_bytes: 64 << 30,
        threads: 8,
    };
    let w = |ctx| sizing(ctx);
    let (kv, p) = pick_for_placement(&big, &layout, &w(CTX), at, KvChoice::Auto).unwrap();
    assert_eq!((kv, p.is_some()), (KvFormat::F16, true));
    let m16 = max_ctx(|c| {
        let f = footprint(&layout, at.0, at.1, &w(c), KvFormat::F16);
        f.gpu <= big.gpu_bytes && f.host <= big.ram_bytes
    });
    let (kv, _) = pick_for_placement(&big, &layout, &w(m16 + STEP), at, KvChoice::Auto).unwrap();
    assert_eq!(kv, KvFormat::Q8_0);
    let (kv, p) = pick_for_placement(
        &big,
        &layout,
        &w(MAX_CTX),
        at,
        KvChoice::Fixed(KvFormat::Q4_0),
    )
    .unwrap();
    assert_eq!((kv, p), (KvFormat::Q4_0, None));
    let err = pick_for_placement(&big, &layout, &w(MAX_CTX), at, KvChoice::Auto).unwrap_err();
    assert!(
        err.contains("no KV format fits") && err.contains("needs"),
        "{err}"
    );
}

fn arb_case() -> impl Strategy<Value = (usize, usize, u64, u64, u32)> {
    (0..4usize, 0..3usize, 1u64..80, 1u64..256, 1u32..8192)
}

proptest! {
    /// Rule 5: whatever the budget and context, the chosen plan never exceeds VRAM or RAM with the
    /// KV bytes of the chosen format counted, no earlier format would have fit, and an error
    /// means no format fits anywhere.
    #[test]
    fn the_chosen_plan_never_exceeds_memory((g, e, gpu_gb, ram_gb, steps) in arb_case()) {
        let graphs = graphs();
        let (gname, hw) = &graphs[g];
        let layouts = layouts();
        let (lname, layout) = &layouts[e % layouts.len()];
        let engs = engines(layout);
        let engine = engs[e % engs.len()];
        let mut inp = input(hw, layout, engine);
        inp.budget.gpu_bytes = gpu_gb << 30;
        inp.budget.ram_bytes = ram_gb << 30;
        inp.ctx = steps * STEP;
        let what = format!("{gname}/{lname}/{engine:?}/gpu {gpu_gb}/ram {ram_gb}/ctx {}", inp.ctx);
        match (enumerate_choice(&inp, KvChoice::Auto), expected(&inp, hw, inp.ctx)) {
            (Ok((en, Some(pick))), Some((kv, _))) => {
                prop_assert_eq!(pick.format, kv, "{}", what);
                check_fits(&inp, &en, kv, &what);
            }
            (Err(_), None) => {}
            (got, want) => prop_assert!(false, "{}: got {:?}, oracle {:?}", what, got.map(|g| g.1), want),
        }
    }
}

//! Low-RAM modes (ADR 0085 class 1, node `planner-low-ram-modes`): what the planner does for a
//! streamed MoE model (the Qwen3-235B-A22B shape, about 133 GB at Q4_K) when the RAM budget has
//! no room for an expert RAM tier, as Strata's `--ram-experts-gb 0` modes run it:
//!
//! 1. experts in VRAM slots and on NVMe only (GPU, no RAM tier);
//! 2. CPU compute with the weights on NVMe only (no GPU).
//!
//! Finding: neither mode has a plan today. The builder's smallest RAM tier is top-k slots per
//! layer (ADR 0034) and the CPU placement holds all weights in RAM, so a budget below those
//! floors gets a refusal, now one that names the missing tier. A plan with no RAM tier
//! (`ram_slots` 0, which the engine runs when `--ram-slots` is absent) is not enumerated; that is
//! the gap, and needs `lower.rs` and `score.rs` to carry it.

mod common;

use common::{CTX, WORK, graphs, has_gpu, input};
use tetraeder_plan::Format;
use tetraeder_planner::builder::{BUILDER_UBATCHES, ram_slot_options};
use tetraeder_planner::candidate::UBATCHES;
use tetraeder_planner::{
    ModelLayout, ModelMeta, Placement, ServeEngine, TensorRec, Workload, build_layout, candidates,
    footprint, pick, recognize,
};

fn rec(name: &str, ty: &str, dims: &[u64]) -> TensorRec {
    let f: Format = serde_json::from_str(&format!("\"{ty}\"")).unwrap();
    TensorRec {
        name: name.into(),
        ty: ty.into(),
        dims: dims.to_vec(),
        bytes: f.bytes(dims.iter().product()),
    }
}

/// Qwen3-235B-A22B at Q4_K: 94 layers, 64 heads and 4 KV heads of 128, 128 experts of 1536, top 8.
fn qwen3_235b() -> ModelLayout {
    let (e, v, ex, q, kv, ne, n) = (4096u64, 151_936u64, 1536u64, 8192u64, 512u64, 128u64, 94);
    let meta = ModelMeta {
        model: "qwen3-235b-a22b".into(),
        arch: "qwen3moe".into(),
        n_layer: n as u32,
        n_embd: e as u32,
        n_head: 64,
        n_head_kv: 4,
        head_dim: 128,
        n_expert: ne as u32,
        n_expert_used: 8,
    };
    let mut t = vec![rec("token_embd.weight", "Q4_K", &[e, v])];
    for i in 0..n {
        let b = |s: &str| format!("blk.{i}.{s}.weight");
        t.push(rec(&b("attn_norm"), "F32", &[e]));
        t.push(rec(&b("attn_q"), "Q4_K", &[e, q]));
        t.push(rec(&b("attn_k"), "Q4_K", &[e, kv]));
        t.push(rec(&b("attn_v"), "Q4_K", &[e, kv]));
        t.push(rec(&b("attn_output"), "Q4_K", &[q, e]));
        t.push(rec(&b("ffn_norm"), "F32", &[e]));
        t.push(rec(&b("ffn_gate_inp"), "F32", &[e, ne]));
        t.push(rec(&b("ffn_gate_exps"), "Q4_K", &[e, ex, ne]));
        t.push(rec(&b("ffn_up_exps"), "Q4_K", &[e, ex, ne]));
        t.push(rec(&b("ffn_down_exps"), "Q4_K", &[ex, e, ne]));
    }
    t.push(rec("output_norm.weight", "F32", &[e]));
    t.push(rec("output.weight", "Q6_K", &[e, v]));
    build_layout(&meta, &t).unwrap()
}

fn sizing() -> Workload {
    Workload {
        prompt: CTX - WORK.r#gen,
        r#gen: WORK.r#gen,
    }
}

/// The least host RAM any placement of the engine needs on a GPU budget, by brute force over the
/// VRAM slot counts: the builder's RAM tier shrinks as more experts sit in VRAM.
fn host_floor(layout: &ModelLayout, gpu_bytes: u64, builder: bool) -> u64 {
    let kv = tetraeder_plan::KvFormat::default();
    if !builder {
        return footprint(layout, Placement::Cpu, UBATCHES[0], &sizing(), kv).host;
    }
    (layout.n_expert_used..=layout.n_expert)
        .filter_map(|v| {
            let pl = Placement::Builder {
                vram_slots: v,
                ram_slots: ram_slot_options(layout, v)[0],
            };
            let f = footprint(layout, pl, BUILDER_UBATCHES[0], &sizing(), kv);
            (f.gpu <= gpu_bytes).then_some(f.host)
        })
        .min()
        .expect("some VRAM tier fits")
}

#[test]
fn the_235b_layout_is_the_streamed_size() {
    let l = qwen3_235b();
    let gb = l.total_bytes() as f64 / 1e9;
    assert!((120.0..150.0).contains(&gb), "{gb}");
    assert_eq!((l.n_layer, l.n_expert, l.n_expert_used), (94, 128, 8));
}

/// Mode 1: a GPU, experts streamed from NVMe, no RAM for an expert tier.
#[test]
fn gpu_and_nvme_only_is_refused_naming_the_ram_tier() {
    let layout = qwen3_235b();
    for (gname, hw) in graphs().into_iter().filter(|(_, h)| has_gpu(h)) {
        let mut inp = input(&hw, &layout, ServeEngine::Builder);
        let floor = host_floor(&layout, inp.budget.gpu_bytes, true);
        // The RAM that is left once the OS and the dense host side are paid: no room for slots.
        inp.budget.ram_bytes = floor / 2;
        assert!(candidates(&inp).is_empty(), "{gname}");
        let e = pick(&inp).unwrap_err();
        assert!(
            e.starts_with("no executable plan fits the budget"),
            "{gname}: {e}"
        );
        assert!(
            e.contains("missing tier") && e.contains("RAM tier"),
            "{gname}: {e}"
        );
        assert!(e.contains("expert slots per layer"), "{gname}: {e}");
    }
}

/// The floor itself: with room for the top-k RAM slots a plan exists and keeps them.
#[test]
fn the_smallest_builder_ram_tier_is_top_k_slots() {
    let layout = qwen3_235b();
    for (gname, hw) in graphs().into_iter().filter(|(_, h)| has_gpu(h)) {
        let mut inp = input(&hw, &layout, ServeEngine::Builder);
        let floor = host_floor(&layout, inp.budget.gpu_bytes, true);
        inp.budget.ram_bytes = floor;
        let plans = candidates(&inp);
        assert!(!plans.is_empty(), "{gname}: a plan at the floor");
        for p in &plans {
            let (pl, _) = recognize(p, &layout).unwrap();
            let Placement::Builder { ram_slots, .. } = pl else {
                panic!("{gname}: not a builder placement: {pl:?}");
            };
            assert!(ram_slots >= layout.n_expert_used, "{gname}: {ram_slots}");
        }
        inp.budget.ram_bytes = floor - 1;
        assert!(
            candidates(&inp).is_empty(),
            "{gname}: one byte under the floor"
        );
    }
}

/// The gap: no budget, however small, yields a plan with no RAM tier.
#[test]
fn no_ram_tier_plans_are_not_enumerated_yet() {
    let layout = qwen3_235b();
    let (_, hw) = graphs()
        .into_iter()
        .find(|(n, _)| *n == "rtx3090_prior")
        .unwrap();
    let gpu = input(&hw, &layout, ServeEngine::Builder).budget.gpu_bytes;
    let floor = host_floor(&layout, gpu, true);
    for ram in [0, floor / 4, floor, 4 * floor, 64 * floor] {
        let mut inp = input(&hw, &layout, ServeEngine::Builder);
        inp.budget.ram_bytes = ram;
        for p in candidates(&inp) {
            if let (Placement::Builder { ram_slots, .. }, _) = recognize(&p, &layout).unwrap() {
                assert!(ram_slots > 0, "ram {ram}");
            }
        }
    }
}

/// Mode 2: no GPU, the weights would stream from NVMe, no RAM for them.
#[test]
fn cpu_and_nvme_only_is_refused_naming_the_ram_tier() {
    let layout = qwen3_235b();
    let floor = host_floor(&layout, 0, false);
    for (gname, hw) in graphs().into_iter().filter(|(_, h)| !has_gpu(h)) {
        let mut inp = input(&hw, &layout, ServeEngine::Llama);
        assert!(
            inp.budget.ram_bytes < floor,
            "{gname}: the fixture is too small for 235B"
        );
        assert!(candidates(&inp).is_empty(), "{gname}");
        let e = pick(&inp).unwrap_err();
        assert!(
            e.contains("missing tier") && e.contains("CPU placement"),
            "{gname}: {e}"
        );
        assert!(e.contains("no RAM tier"), "{gname}: {e}");
        // Room for all weights: the CPU plans appear, one byte less and they do not.
        inp.budget.ram_bytes = floor;
        assert!(!candidates(&inp).is_empty(), "{gname}: at the floor");
        inp.budget.ram_bytes = floor - 1;
        assert!(candidates(&inp).is_empty(), "{gname}: under the floor");
    }
}

/// A box that fits keeps the old message: no hint where no tier is short.
#[test]
fn a_fitting_box_has_no_missing_tier() {
    let layout = tetraeder_planner::layout::fixtures::moe();
    for (gname, hw) in graphs() {
        let inp = input(&hw, &layout, ServeEngine::Llama);
        assert!(
            tetraeder_planner::serve::missing_tier(&inp).is_none(),
            "{gname}"
        );
    }
}

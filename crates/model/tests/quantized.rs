//! The builder on a quantized GGUF: the tiny fixture quantized to Q8_0 by libllama, so the
//! quantized weight paths (get_rows on `token_embd`, mul_mat, mul_mat_id and expert slabs of
//! block-quantized rows) run locally, against libllama on the same file.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU; `gpu_*` tests skip cleanly without one.

mod common;

use common::*;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Ftype, QuantizeParams, gguf_tensor_types, quantize};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

/// hardware: cpu. Resident and streaming (with and without a RAM tier, with overflow passes)
/// match libllama on the Q8_0 copy. Same kernels on both sides, so the CPU tolerance applies.
#[test]
fn cpu_q8_0_matches_libllama() {
    let path = std::env::temp_dir().join(format!("tetraeder-q8-{}.gguf", std::process::id()));
    let mut qp = QuantizeParams::new(Ftype::Q8_0);
    qp.nthread = 2;
    quantize(fixture(), &path, &qp).unwrap();
    let types: std::collections::HashMap<_, _> =
        gguf_tensor_types(&path).unwrap().into_iter().collect();
    assert_eq!(types["token_embd.weight"], "q8_0");
    assert_eq!(types["blk.0.ffn_down_exps.weight"], "q8_0");
    assert_eq!(types["blk.0.attn_q.weight"], "q8_0");
    assert_eq!(types["blk.0.attn_norm.weight"], "f32");

    let r = libllama_reference_at(&path, &PROMPT, N_GREEDY);
    for (slots, ram) in [
        (None, None),
        (Some(4), None),
        (Some(2), Some(3)),
        (Some(2), None),
    ] {
        let o = LoadOptions {
            expert_slots: slots,
            ram_slots: ram,
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("q8_0 slots {slots:?} ram {ram:?}: max |delta| {worst:e}");
        assert_eq!(m.expert_stats().is_some(), slots.is_some());
    }
    std::fs::remove_file(&path).ok();
}

/// hardware: gpu (CUDA build; skips without a GPU device). A 24-token prompt is above the
/// batch-8 limit of CUDA's vector kernel, so the expert matmuls take the MMQ path the real
/// Q4_K_M model takes in prefill. Overflow passes (2 and 4 slots) against the resident builder on
/// the same GPU: only the summation order differs.
#[test]
fn gpu_q8_0_overflow_passes_match_resident() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    let path = std::env::temp_dir().join(format!("tetraeder-q8g-{}.gguf", std::process::id()));
    let mut qp = QuantizeParams::new(Ftype::Q8_0);
    qp.nthread = 2;
    quantize(fixture(), &path, &qp).unwrap();
    let gpu = BackendChoice::Gpu { index: 0 };
    let run = |slots: Option<usize>| {
        let o = LoadOptions {
            expert_slots: slots,
            ..LoadOptions::new(gpu)
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        let prompt: Vec<i32> = (0..24).map(|i| (i * 37 + 5) % m.n_vocab() as i32).collect();
        let logits = m.eval(&prompt).unwrap();
        (logits, m.overflow_passes())
    };
    let (base, _) = run(None);
    for slots in [4, 2] {
        let (got, passes) = run(Some(slots));
        let worst = base
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let nan = got.iter().any(|x| !x.is_finite());
        eprintln!(
            "q8_0 gpu slots {slots}: passes {passes}, max |delta| {worst:e}, non-finite {nan}"
        );
        assert!(!nan, "slots {slots}: non-finite logits");
        assert!(
            passes > 0,
            "slots {slots}: 24 tokens route to more than {slots} experts"
        );
        assert!(worst < 1e-3, "slots {slots}: max |delta| {worst:e}");
    }
    std::fs::remove_file(&path).ok();
}

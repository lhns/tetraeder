//! The builder on a quantized GGUF: the tiny fixture quantized to Q8_0 by libllama, so the
//! quantized weight paths (get_rows on `token_embd`, mul_mat, mul_mat_id and expert slabs of
//! block-quantized rows) run locally, against libllama on the same file.
//!
//! Hardware label: all tests are `cpu_*`.

mod common;

use common::*;
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

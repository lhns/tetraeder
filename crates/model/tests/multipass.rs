//! Overflow passes (ADR 0033, 0034) on a K-quant model shaped like Qwen3-30B-A3B where the
//! expert path is concerned: top-8 of 32 experts, dims of 256, Q4_K_M (q4_K and q6_K experts).
//! A prefill that needs several passes must equal the resident builder bit for bit: logits and
//! the router weights of every layer.
//!
//! The CPU's tiled K-quant kernel, which once made the passes inexact (ADR 0034), exists only in
//! AVX builds. Without AVX these tests pass either way; the row-count invariant behind the fix is
//! checked everywhere by a unit test in `qwen3moe.rs`.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use std::path::Path;
use tetraeder_ggml_sys::llama::{Ftype, gguf_tensor_types};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;

struct Prefill {
    logits: Vec<f32>,
    /// Router weights per layer, token-major.
    weights: Vec<Vec<f32>>,
    passes: u64,
}

fn prefill(path: &Path, slots: Option<usize>) -> Prefill {
    let o = LoadOptions {
        expert_slots: slots,
        record_routing: true,
        ..LoadOptions::new(CPU)
    };
    let mut m = Qwen3Moe::load(path, &o).unwrap();
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let logits = m.eval(&prompt).unwrap();
    let mut weights = vec![Vec::new(); m.hparams().n_layer as usize];
    for step in m.take_routing_log() {
        weights[step.layer].extend(step.weights);
    }
    Prefill {
        logits,
        weights,
        passes: m.overflow_passes(),
    }
}

/// The first layer whose router weights differ bitwise and the tokens that differ there.
fn first_weight_diff(a: &Prefill, b: &Prefill) -> Option<(usize, Vec<usize>)> {
    a.weights
        .iter()
        .zip(&b.weights)
        .enumerate()
        .find_map(|(l, (x, y))| {
            let nu = x.len() / N_PROMPT;
            let tokens: Vec<usize> = (0..N_PROMPT)
                .filter(|&t| {
                    let (p, q) = (&x[t * nu..(t + 1) * nu], &y[t * nu..(t + 1) * nu]);
                    p.iter().zip(q).any(|(p, q)| p.to_bits() != q.to_bits())
                })
                .collect();
            (!tokens.is_empty()).then_some((l, tokens))
        })
}

/// Streaming with `slots` against the resident builder; panics on any bit of difference.
fn check_exact(path: &Path, slot_counts: &[usize]) {
    let base = prefill(path, None);
    let mut failures = Vec::new();
    for &slots in slot_counts {
        let run = prefill(path, Some(slots));
        assert!(run.passes > 0, "slots {slots}: a prefill needs passes");
        let max_delta = base
            .logits
            .iter()
            .zip(&run.logits)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let diff = first_weight_diff(&base, &run);
        let bitwise = base
            .logits
            .iter()
            .zip(&run.logits)
            .all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!(
            "slots {slots}: passes {}, max |logit delta| {max_delta:e}, first router weight diff {diff:?}",
            run.passes
        );
        if !bitwise || diff.is_some() {
            failures.push(slots);
        }
    }
    assert!(
        failures.is_empty(),
        "not bitwise exact with slots {failures:?}"
    );
}

/// hardware: cpu. Q4_K_M: q4_K gate and up, q4_K and q6_K down depending on the layer, like
/// the real model. Overflow passes at 8, 12 and 16 slots equal the resident builder bitwise.
#[test]
fn cpu_k_quant_overflow_passes_are_exact() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km");
    let types = gguf_tensor_types(&path).unwrap();
    let downs: Vec<&str> = types
        .iter()
        .filter(|(n, _)| n.ends_with("ffn_down_exps.weight"))
        .map(|(_, t)| t.as_str())
        .collect();
    eprintln!("ffn_down_exps types: {downs:?}");
    assert!(downs.contains(&"q4_K") && downs.contains(&"q6_K"));
    check_exact(&path, &[8, 12, 16]);
}

/// hardware: cpu. The same on a Q8_0 copy (a type without the CPU's tiled K-quant kernel).
#[test]
fn cpu_q8_0_overflow_passes_are_exact() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "q8_0");
    check_exact(&path, &[8, 16]);
}

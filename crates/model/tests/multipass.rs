//! Expert pair groups (ADR 0035) on a K-quant model shaped like Qwen3-30B-A3B where the expert
//! path is concerned: top-8 of 32 experts, dims of 256, Q4_K_M (q4_K and q6_K experts). A prefill
//! and the decode steps after it that need several groups must equal the resident builder bit for
//! bit: logits and the router weights of every layer.
//!
//! The CPU's tiled K-quant kernel, which picks its kernel per expert by row count (ADR 0034),
//! exists only in AVX builds. Without AVX the K-quant test passes either way; the row-count
//! invariant behind it is checked everywhere by a unit test in `qwen3moe.rs`.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use std::path::Path;
use tetraeder_ggml_sys::llama::{Ftype, gguf_tensor_types};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 3;

struct Run {
    /// Prefill logits, then the logits of each greedy decode step.
    logits: Vec<Vec<f32>>,
    /// Router weights per layer, token-major, over the prefill.
    weights: Vec<Vec<f32>>,
    groups: u64,
    group_graphs: u64,
}

fn argmax(l: &[f32]) -> i32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn run(path: &Path, slots: Option<usize>) -> Run {
    let o = LoadOptions {
        expert_slots: slots,
        record_routing: true,
        ..LoadOptions::new(CPU)
    };
    let mut m = Qwen3Moe::load(path, &o).unwrap();
    let n_vocab = m.n_vocab();
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let prefill = m.eval(&prompt).unwrap();
    let mut weights = vec![Vec::new(); m.hparams().n_layer as usize];
    for step in m.take_routing_log() {
        weights[step.layer].extend(step.weights);
    }
    let mut next = argmax(&prefill[(N_PROMPT - 1) * n_vocab..]);
    let mut logits = vec![prefill];
    for _ in 0..N_DECODE {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l);
        logits.push(l);
    }
    Run {
        logits,
        weights,
        groups: m.expert_groups(),
        group_graphs: m.group_graphs(),
    }
}

/// The first layer whose router weights differ bitwise and the tokens that differ there.
fn first_weight_diff(a: &Run, b: &Run) -> Option<(usize, Vec<usize>)> {
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
    let base = run(path, None);
    let mut failures = Vec::new();
    for &slots in slot_counts {
        let got = run(path, Some(slots));
        assert!(got.groups > 0, "slots {slots}: a prefill needs groups");
        // Top-8 needs several groups at every decode step too with at most 4 slots.
        let max_delta = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let deltas: Vec<f32> = base
            .logits
            .iter()
            .zip(&got.logits)
            .map(|(a, b)| max_delta(a, b))
            .collect();
        let bitwise = base
            .logits
            .iter()
            .zip(&got.logits)
            .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        let diff = first_weight_diff(&base, &got);
        eprintln!(
            "slots {slots}: groups {}, group graphs {}, max |logit delta| prefill then decode {deltas:?}, first router weight diff {diff:?}",
            got.groups, got.group_graphs
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
/// the real model. Pair groups at 4, 8, 12 and 16 slots (4 is below the 8 experts per token)
/// equal the resident builder bitwise, in prefill and in decode.
#[test]
fn cpu_k_quant_pair_groups_are_exact() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km");
    let types = gguf_tensor_types(&path).unwrap();
    let downs: Vec<&str> = types
        .iter()
        .filter(|(n, _)| n.ends_with("ffn_down_exps.weight"))
        .map(|(_, t)| t.as_str())
        .collect();
    eprintln!("ffn_down_exps types: {downs:?}");
    assert!(downs.contains(&"q4_K") && downs.contains(&"q6_K"));
    check_exact(&path, &[4, 8, 12, 16]);
}

/// hardware: cpu. The same on a Q8_0 copy (a type without the CPU's tiled K-quant kernel).
#[test]
fn cpu_q8_0_pair_groups_are_exact() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "q8_0");
    check_exact(&path, &[4, 8, 16]);
}

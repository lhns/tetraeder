//! Parity of the own Qwen3 MoE builder against libllama on the synthetic fixture (ADR 0030).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU; `gpu_*` tests skip cleanly when
//! the build has no GPU device (a CPU-only build, or no card).

mod common;

use common::*;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

fn check_parity(backend: BackendChoice, tol: Tol) {
    let r = libllama_reference(&PROMPT, N_GREEDY);
    let mut m = Qwen3Moe::load(fixture(), &LoadOptions::new(backend)).unwrap();
    let worst = check_against_tol(&mut m, &r, tol);
    eprintln!(
        "max |delta| over the top-{TOP_K} logits of {} positions: {worst:e}",
        PROMPT.len()
    );
    // Steps: the full prefill, the last-only prefill and one per greedy token; with every
    // expert resident each runs as one graph (ADR 0036).
    assert_eq!(m.segments_run(), 2 + N_GREEDY as u64);
}

/// hardware: cpu
#[test]
fn cpu_matches_libllama() {
    check_parity(BackendChoice::Cpu { threads: Some(2) }, Tol::abs(TOL_CPU));
}

/// hardware: cpu. The last layer runs its expert FFN on the output rows only (libllama's
/// `inp_out_ids`): the last-only prefill gives the last row of the full prefill.
#[test]
fn cpu_last_only_prefill_equals_last_row_of_full_prefill() {
    let mut m = Qwen3Moe::load(
        fixture(),
        &LoadOptions::new(BackendChoice::Cpu { threads: Some(2) }),
    )
    .unwrap();
    let nv = m.n_vocab();
    let full = m.eval(&PROMPT).unwrap();
    m.reset();
    let last = m.eval_last(&PROMPT).unwrap();
    for (a, b) in full[(PROMPT.len() - 1) * nv..].iter().zip(&last) {
        assert!((a - b).abs() <= TOL_CPU, "{a} vs {b}");
    }
}

/// hardware: cpu. A prompt split over several `eval` calls gives the same logits as one call:
/// the KV cache carries over.
#[test]
fn cpu_chunked_prefill_equals_single_prefill() {
    let mut m = Qwen3Moe::load(
        fixture(),
        &LoadOptions::new(BackendChoice::Cpu { threads: Some(2) }),
    )
    .unwrap();
    let whole = m.eval_last(&PROMPT).unwrap();
    m.reset();
    m.eval(&PROMPT[..4]).unwrap();
    let tail = m.eval_last(&PROMPT[4..]).unwrap();
    for (a, b) in whole.iter().zip(&tail) {
        assert!((a - b).abs() <= TOL_CPU, "{a} vs {b}");
    }
    assert_eq!(argmax(&whole), argmax(&tail));
}

/// hardware: cpu
#[test]
fn cpu_rejects_bad_input() {
    let mut m = Qwen3Moe::load(
        fixture(),
        &LoadOptions::new(BackendChoice::Cpu { threads: Some(1) }),
    )
    .unwrap();
    assert!(m.eval(&[]).is_err());
    assert!(m.eval(&[-1]).is_err());
    assert!(m.eval(&[m.n_vocab() as i32]).is_err());
}

/// hardware: gpu (CUDA build; skips without a GPU device)
#[test]
fn gpu_matches_libllama() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    // libllama stays on the CPU as the reference; the builder runs on GPU 0.
    check_parity(BackendChoice::Gpu { index: 0 }, GPU_TOL);
}

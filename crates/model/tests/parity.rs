//! Parity of the own Qwen3 MoE builder against libllama on the synthetic fixture (ADR 0030).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU; `gpu_*` tests skip cleanly when
//! the build has no GPU device (a CPU-only build, or no card).

use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

/// Largest allowed absolute difference of a reference top-k logit.
/// CPU: both sides run the same F16 weights, F16 KV and ggml kernels in the same op order, and
/// the measured difference is 0; 1e-4 leaves room for a different thread count or SIMD level
/// (summation order, about 1e-6 at the fixture's logit scale of 1 to 5).
const TOL_CPU: f32 = 1e-4;
/// GPU builder against the CPU reference: different kernels and F16 rounding of activations,
/// about 1e-3 relative at that scale.
const TOL_GPU: f32 = 5e-3;
const TOP_K: usize = 5;
const PROMPT: [i32; 6] = [12, 77, 3, 200, 41, 9];
const N_GREEDY: usize = 12;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|p| p.0)
        .unwrap()
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

struct Reference {
    n_vocab: usize,
    /// Logits of every prompt position.
    prompt_logits: Vec<f32>,
    greedy: Vec<i32>,
}

fn libllama_reference(prompt: &[i32], n_greedy: usize) -> Reference {
    let model = Arc::new(
        Model::load(
            fixture(),
            &ModelParams {
                n_gpu_layers: 0,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the fixture"),
    );
    let n_vocab = model.n_vocab() as usize;
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_ubatch: 64,
            n_threads: 2,
            n_threads_batch: 2,
            flash_attn: FlashAttn::On,
            ..ContextParams::default()
        },
    )
    .unwrap();
    let mut batch = Batch::new(64);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, true).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let mut prompt_logits = Vec::new();
    for i in 0..prompt.len() {
        prompt_logits.extend_from_slice(ctx.logits_ith(i as i32).unwrap());
    }
    let mut greedy = Vec::new();
    let mut next = argmax(&prompt_logits[(prompt.len() - 1) * n_vocab..]) as i32;
    for step in 0..n_greedy {
        greedy.push(next);
        batch.clear();
        batch
            .push(next, (prompt.len() + step) as i32, 0, true)
            .unwrap();
        ctx.decode(&batch).unwrap();
        next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
    }
    Reference {
        n_vocab,
        prompt_logits,
        greedy,
    }
}

fn check_parity(backend: BackendChoice, tol: f32) {
    let r = libllama_reference(&PROMPT, N_GREEDY);
    let mut m = Qwen3Moe::load(fixture(), &LoadOptions::new(backend)).unwrap();
    assert_eq!(m.n_vocab(), r.n_vocab);

    // Logits of every prompt position in one prefill step.
    let got = m.eval(&PROMPT).unwrap();
    assert_eq!(got.len(), r.prompt_logits.len());
    assert!(got.iter().all(|x| x.is_finite()));
    let mut worst = 0f32;
    for p in 0..PROMPT.len() {
        let (want, have) = (
            &r.prompt_logits[p * r.n_vocab..(p + 1) * r.n_vocab],
            &got[p * r.n_vocab..(p + 1) * r.n_vocab],
        );
        let top = top_k(want, TOP_K);
        for &i in &top {
            worst = worst.max((want[i] - have[i]).abs());
            assert!(
                (want[i] - have[i]).abs() <= tol,
                "position {p}, token {i}: libllama {} vs builder {}",
                want[i],
                have[i]
            );
        }
        assert_eq!(top[0], argmax(have), "top-1 differs at position {p}");
    }
    eprintln!(
        "max |delta| over the top-{TOP_K} logits of {} positions: {worst:e}",
        PROMPT.len()
    );

    // Greedy sequence: prefill the prompt, then decode one token per step (n_tokens = 1 graphs,
    // incremental KV cache).
    m.reset();
    let last = m.eval_last(&PROMPT).unwrap();
    assert_eq!(last.len(), r.n_vocab);
    let mut next = argmax(&last) as i32;
    let mut seq = Vec::new();
    for _ in 0..N_GREEDY {
        seq.push(next);
        next = argmax(&m.eval(&[next]).unwrap()) as i32;
    }
    assert_eq!(seq, r.greedy);
    // Steps: the full prefill, the last-only prefill and one per greedy token; each runs one
    // segment per MoE layer plus the head (ADR 0030).
    let layers = m.hparams().n_layer as u64;
    assert_eq!(m.segments_run(), (layers + 1) * (2 + N_GREEDY as u64));
}

/// hardware: cpu
#[test]
fn cpu_matches_libllama() {
    check_parity(BackendChoice::Cpu { threads: Some(2) }, TOL_CPU);
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
    check_parity(BackendChoice::Gpu { index: 0 }, TOL_GPU);
}

//! Shared by the model tests: the fixture, libllama as the reference and the logit checks.
#![allow(dead_code)]

pub mod split;
pub mod synth;
pub mod trace;

/// The CPU quota of the container (cgroup v2 `cpu.max`), if any.
pub fn cpu_quota() -> Option<f64> {
    let s = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut it = s.split_whitespace();
    let (q, p) = (it.next()?, it.next()?);
    Some(q.parse::<f64>().ok()? / p.parse::<f64>().ok()?)
}

/// Thread counts for a paired benchmark: `TETRAEDER_BENCH_THREADS` (comma list) if set, else
/// the container CPU quota rounded down, else the available parallelism (same rule as
/// `scripts/ot_pair2.sh`; the host core count in a container is not what it may use).
pub fn bench_thread_list() -> Vec<usize> {
    let from_env: Vec<usize> = std::env::var("TETRAEDER_BENCH_THREADS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    if !from_env.is_empty() {
        return from_env;
    }
    let quota = cpu_quota().map(|q| q as usize).filter(|&q| q >= 1);
    vec![quota.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))]
}

/// The one thread count both engines get: the first of [`bench_thread_list`].
pub fn bench_threads() -> usize {
    bench_thread_list()[0]
}

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::Qwen3Moe;
use tetraeder_plan::KvFormat;

pub fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

/// Largest allowed absolute difference of a reference top-k logit.
/// CPU: both sides run the same F16 weights, F16 KV and ggml kernels in the same op order, and
/// the measured difference is 0; 1e-4 leaves room for a different thread count or SIMD level
/// (summation order, about 1e-6 at the fixture's logit scale of 1 to 5).
pub const TOL_CPU: f32 = 1e-4;
/// GPU builder against the CPU reference: different kernels and F16 rounding of activations,
/// about 1e-3 relative at that scale.
pub const TOL_GPU: f32 = 5e-3;
pub const TOP_K: usize = 5;
pub const PROMPT: [i32; 6] = [12, 77, 3, 200, 41, 9];
pub const N_GREEDY: usize = 12;

/// Prefetch counters between two stats snapshots, per `per` (tokens): experts issued, used and
/// wasted, bytes, the worker's read time against the time the step waited for it, and the
/// prediction accuracy (hits over slots scored) per layer and overall, for the prediction one
/// layer ahead and, with prefetch depth 2, two layers ahead (ADR 0038).
pub fn prefetch_json(
    a: &tetraeder_model::ExpertStats,
    b: &tetraeder_model::ExpertStats,
    per: f64,
) -> serde_json::Value {
    let (x, y) = (&a.io, &b.io);
    let read = (y.prefetch_read_time - x.prefetch_read_time).as_secs_f64();
    let wait = (y.prefetch_wait_time - x.prefetch_wait_time).as_secs_f64();
    let (mut hits, mut total) = (0u64, 0u64);
    let (mut hits2, mut total2) = (0u64, 0u64);
    let accuracy2: Vec<Option<f64>> = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(p, q)| {
            let (h, t) = (
                q.predicted2_hits - p.predicted2_hits,
                q.predicted2_total - p.predicted2_total,
            );
            hits2 += h;
            total2 += t;
            (t > 0).then(|| h as f64 / t as f64)
        })
        .collect();
    let accuracy: Vec<Option<f64>> = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(p, q)| {
            let (h, t) = (
                q.predicted_hits - p.predicted_hits,
                q.predicted_total - p.predicted_total,
            );
            hits += h;
            total += t;
            (t > 0).then(|| h as f64 / t as f64)
        })
        .collect();
    serde_json::json!({
        "issued": y.prefetch_issued - x.prefetch_issued,
        "used": y.prefetch_used - x.prefetch_used,
        "wasted": y.prefetch_wasted - x.prefetch_wasted,
        "bytes": y.prefetch_bytes - x.prefetch_bytes,
        "wasted_bytes": y.prefetch_wasted_bytes - x.prefetch_wasted_bytes,
        "read_ms_per_token": read * 1e3 / per,
        "wait_ms_per_token": wait * 1e3 / per,
        "overlap": if read > 0.0 { Some(1.0 - wait / read) } else { None },
        "accuracy_mean": if total > 0 { Some(hits as f64 / total as f64) } else { None },
        "accuracy": accuracy,
        "accuracy2_mean": if total2 > 0 { Some(hits2 as f64 / total2 as f64) } else { None },
        "accuracy2": accuracy2,
    })
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|p| p.0)
        .unwrap()
}

pub fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

pub struct Reference {
    pub n_vocab: usize,
    /// Logits of every prompt position.
    pub prompt_logits: Vec<f32>,
    pub greedy: Vec<i32>,
}

pub fn libllama_reference(prompt: &[i32], n_greedy: usize) -> Reference {
    libllama_reference_at(&fixture(), prompt, n_greedy)
}

/// Like [`libllama_reference`] for another GGUF (same shape, for example a quantized copy).
pub fn libllama_reference_at(path: &Path, prompt: &[i32], n_greedy: usize) -> Reference {
    libllama_reference_kv(path, prompt, n_greedy, KvFormat::default())
}

/// Like [`libllama_reference_at`] with the K and V cache types of `kv` (ADR 0050 b). libllama's
/// Hadamard rotation of a quantized cache is off: the builder stores plain values, so only then
/// do the logits compare.
pub fn libllama_reference_kv(
    path: &Path,
    prompt: &[i32],
    n_greedy: usize,
    kv: KvFormat,
) -> Reference {
    let ty = |f| tetraeder_model::convert::ggml_type(f).expect("ggml type of the KV format");
    tetraeder_ggml_sys::llama::disable_attn_rot();
    let model = Arc::new(
        Model::load(
            path,
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
            type_k: ty(kv.k),
            type_v: ty(kv.v),
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

/// Prefill of `PROMPT` (logits of every position), then `N_GREEDY` greedy decode steps, against
/// the reference. Panics on a difference above `tol` in a reference top-k logit or in the
/// greedy sequence. Returns the largest logit difference seen.
pub fn check_against(m: &mut Qwen3Moe, r: &Reference, tol: f32) -> f32 {
    m.reset();
    assert_eq!(m.n_vocab(), r.n_vocab);
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
    worst
}

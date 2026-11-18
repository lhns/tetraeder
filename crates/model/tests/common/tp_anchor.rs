//! libllama's own tensor-parallel gap, the anchor of the tensor-parallel and expert-parallel
//! shapes on real models (ADR 0071, "real models need their own anchor" addendum).

use super::{GPU_TOL, TOP_K, Tol, argmax, largest, top_k};
use std::path::Path;
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Model, ModelParams, SplitMode,
};

/// Top-5 of `want` in `got`: the worst |delta| and its bound `tol.of(want)`.
pub fn top5_gap(got: &[f32], want: &[f32], tol: Tol) -> (f32, f32) {
    let worst = top_k(want, TOP_K)
        .into_iter()
        .map(|i| (got[i] - want[i]).abs())
        .fold(0f32, f32::max);
    (worst, tol.of(want))
}

/// The prompt's logits (every position) and the logits before each of `tokens` (teacher forced,
/// the first is the prompt's last row) from libllama with every layer on the GPU:
/// `SplitMode::None` on GPU 0, or `SplitMode::Tensor` over every visible GPU (`-sm tensor`,
/// ADR 0069's step-0 baseline). The prompt must fit one batch of 64.
pub fn libllama_run(
    path: &Path,
    split: SplitMode,
    prompt: &[i32],
    tokens: &[i32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers: -1,
                split_mode: split,
                main_gpu: 0,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the model"),
    );
    let nv = model.n_vocab() as usize;
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 512,
            n_batch: 64,
            n_ubatch: 64,
            flash_attn: FlashAttn::On,
            ..ContextParams::default()
        },
    )
    .expect("libllama context");
    let mut batch = Batch::new(64);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, true).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let mut rows = Vec::with_capacity(prompt.len() * nv);
    for i in 0..prompt.len() {
        rows.extend_from_slice(ctx.logits_ith(i as i32).unwrap());
    }
    let mut steps = vec![rows[(prompt.len() - 1) * nv..].to_vec()];
    let fed = &tokens[..tokens.len().saturating_sub(1)];
    for (k, &t) in fed.iter().enumerate() {
        batch.clear();
        batch.push(t, (prompt.len() + k) as i32, 0, true).unwrap();
        ctx.decode(&batch).unwrap();
        steps.push(ctx.logits_ith(-1).unwrap().to_vec());
    }
    (rows, steps)
}

/// libllama's own tensor-parallel gap on this model and box: `-sm tensor` against one GPU on the
/// same tokens, the worst top-5 |delta| of a row as a share of its largest logit. The rows of
/// the steps (the prompt's last row and one per token) always count; every prompt position
/// counts with `all_prompt_rows`. Every comparison is printed (steps every 16th). The bound of a
/// shape that reduces partial results across ranks is max(`GPU_TOL`, 1.5x this share); the
/// anchor's split degree is every visible GPU.
pub fn libllama_tp_anchor(
    path: &Path,
    prompt: &[i32],
    tokens: &[i32],
    all_prompt_rows: bool,
) -> f32 {
    let (p1, s1) = libllama_run(path, SplitMode::None, prompt, tokens);
    let (pt, st) = libllama_run(path, SplitMode::Tensor, prompt, tokens);
    let nv = s1[0].len();
    let mut worst = 0f32;
    let mut row = |name: String, got: &[f32], want: &[f32], print: bool| {
        let (d, _) = top5_gap(got, want, GPU_TOL);
        let rel = d / largest(want);
        worst = worst.max(rel);
        if print {
            let top1 = argmax(got) == argmax(want);
            eprintln!(
                "libllama tensor / one GPU: {name}: top-5 max |delta| {d:e} ({rel:.2e} of largest), top-1 equal {top1}"
            );
        }
    };
    if all_prompt_rows {
        for p in 0..prompt.len() {
            let r = p * nv..(p + 1) * nv;
            row(format!("prompt position {p}"), &pt[r.clone()], &p1[r], true);
        }
    }
    for (k, (g, w)) in st.iter().zip(&s1).enumerate() {
        row(format!("step {k}"), g, w, k % 16 == 0);
    }
    eprintln!("libllama tensor / one GPU: worst {worst:.2e} of the largest logit");
    worst
}

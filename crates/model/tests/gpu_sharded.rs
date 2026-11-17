//! The sharded dense builder on CUDA (ADR 0069, 0070, 0071, 0075, brief E6): one GPU rank against
//! the scheduler path bit for bit, then every shape against `single` on Qwen3-8B Q4_K_M.
//!
//! Hardware labels: `gpu_` (one GPU), `gpu2_` (two), `gpu4_` (four). A missing GPU skips; with
//! `TETRAEDER_REQUIRE_GPUS=<n>` it fails (`Backend::require_gpus`). The model comes from
//! `TETRAEDER_MODELS_DIR` (`Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`); without it the tests
//! skip unless `TETRAEDER_REQUIRE_MODELS` is set. Every comparison is printed before the
//! assertions of its shape run, and the shapes run to completion (ADR 0071).

mod common;

use common::multi_seq::max_diff;
use common::shapes::{layer_g, single, tp, tp2pp2};
use common::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Model, ModelParams, SplitMode,
};
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};
use tetraeder_plan::Topology;

const GPU: BackendChoice = BackendChoice::Gpu { index: 0 };
const N_TOKENS: usize = 64;

fn model() -> Option<PathBuf> {
    let found = std::env::var_os("TETRAEDER_MODELS_DIR")
        .map(|d| PathBuf::from(d).join("Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"))
        .filter(|p| p.is_file());
    if found.is_none() {
        let test = std::thread::current().name().unwrap_or("?").to_string();
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_MODELS").is_none(),
            "{test}: Qwen3-8B Q4_K_M not found (TETRAEDER_REQUIRE_MODELS is set)"
        );
        eprintln!("skip: {test}: Qwen3-8B Q4_K_M not found");
    }
    found
}

fn opts() -> DenseOptions {
    DenseOptions {
        n_ctx_pool: 512,
        ..DenseOptions::new(GPU)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// hardware: gpu, model. One GPU rank on the executor equals the scheduler path bit for bit:
/// the logits of every prompt position and 64 greedy tokens (the scheduler's, teacher forced).
/// Ranges are views of the rank graph (ADR 0075), so the executor runs the scheduler's fused
/// kernels; the 6-token prompt and the decode steps stay on MMVQ, where no fusion needs the
/// scheduler's `graph_optimize`. Every row is printed before the assertion.
#[test]
fn gpu_one_rank_is_bit_equal_to_the_scheduler() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU");
        return;
    }
    let Some(path) = model() else { return };
    let sched = DenseOptions {
        use_executor: false,
        ..opts()
    };
    let exec = DenseOptions {
        use_executor: true,
        ..opts()
    };
    let mut a = Qwen3::load(&path, &sched).unwrap();
    let mut b = Qwen3::load(&path, &exec).unwrap();
    assert_eq!(a.n_ranks(), 1);
    assert_eq!(b.n_ranks(), 1);
    assert!(a.comm_path().is_none() && b.comm_path().is_some());
    let nv = a.n_vocab();
    let mut fails = Vec::new();
    let (la, lb) = (a.eval(&PROMPT).unwrap(), b.eval(&PROMPT).unwrap());
    for p in 0..PROMPT.len() {
        let (x, y) = (&la[p * nv..(p + 1) * nv], &lb[p * nv..(p + 1) * nv]);
        let d = max_diff(x, y);
        eprintln!(
            "prompt position {p}: max |delta| {d:e} ({:.2e} of largest), bit-equal {}",
            d / largest(x),
            bits(x) == bits(y)
        );
        if bits(x) != bits(y) {
            fails.push(format!("prompt position {p}: max |delta| {d:e}"));
        }
    }
    a.reset();
    b.reset();
    let mut la = a.eval_last(&PROMPT).unwrap();
    let mut lb = b.eval_last(&PROMPT).unwrap();
    for step in 0..N_TOKENS {
        let (d, equal) = (max_diff(&la, &lb), bits(&la) == bits(&lb));
        if step % 16 == 0 || !equal {
            eprintln!(
                "step {step}: max |delta| {d:e} ({:.2e} of largest), bit-equal {equal}",
                d / largest(&la)
            );
        }
        if !equal {
            fails.push(format!("step {step}: max |delta| {d:e}"));
        }
        let t = argmax(&la) as i32;
        la = a.eval_last(&[t]).unwrap();
        lb = b.eval_last(&[t]).unwrap();
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// Top-5 of `want` in `got`: the worst |delta| and its bound `tol.of(want)`.
fn top5_gap(got: &[f32], want: &[f32], tol: Tol) -> (f32, f32) {
    let worst = top_k(want, TOP_K)
        .into_iter()
        .map(|i| (got[i] - want[i]).abs())
        .fold(0f32, f32::max);
    (worst, tol.of(want))
}

/// The prompt's logits and the logits before each of `tokens` (teacher forced, the first is
/// the prompt's last row) from libllama with every layer on the GPU: `SplitMode::None` on GPU
/// 0, or `SplitMode::Tensor` over every visible GPU (`-sm tensor`, ADR 0069's step-0 baseline).
fn libllama_run(path: &Path, split: SplitMode, tokens: &[i32]) -> (Vec<f32>, Vec<Vec<f32>>) {
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
    for (i, &t) in PROMPT.iter().enumerate() {
        batch.push(t, i as i32, 0, true).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let mut prompt = Vec::with_capacity(PROMPT.len() * nv);
    for i in 0..PROMPT.len() {
        prompt.extend_from_slice(ctx.logits_ith(i as i32).unwrap());
    }
    let mut steps = vec![prompt[(PROMPT.len() - 1) * nv..].to_vec()];
    let fed = &tokens[..tokens.len().saturating_sub(1)];
    for (k, &t) in fed.iter().enumerate() {
        batch.clear();
        batch.push(t, (PROMPT.len() + k) as i32, 0, true).unwrap();
        ctx.decode(&batch).unwrap();
        steps.push(ctx.logits_ith(-1).unwrap().to_vec());
    }
    (prompt, steps)
}

/// libllama's own tensor-parallel gap on this model and box (ADR 0071's anchored rule for real
/// models, 2026-11-16 addendum): `-sm tensor` against one GPU on the same tokens, the worst
/// top-5 |delta| of a row as a share of its largest logit. The tensor-parallel shapes are
/// bounded by max(`GPU_TOL`, 1.5x this share); the anchor's split degree is every visible GPU.
fn libllama_tp_anchor(path: &Path, tokens: &[i32]) -> f32 {
    let (p1, s1) = libllama_run(path, SplitMode::None, tokens);
    let (pt, st) = libllama_run(path, SplitMode::Tensor, tokens);
    let nv = s1[0].len();
    let mut worst = 0f32;
    let mut row = |name: String, got: &[f32], want: &[f32], print: bool| {
        let (d, _) = top5_gap(got, want, GPU_TOL);
        let rel = d / largest(want);
        worst = worst.max(rel);
        if print {
            let top1 = argmax(got) == argmax(want);
            eprintln!(
                "libllama tensor / one GPU: {name}: top-5 max |delta| {d:e}                  ({rel:.2e} of largest), top-1 equal {top1}"
            );
        }
    };
    for p in 0..PROMPT.len() {
        let r = p * nv..(p + 1) * nv;
        row(format!("prompt position {p}"), &pt[r.clone()], &p1[r], true);
    }
    for (k, (g, w)) in st.iter().zip(&s1).enumerate() {
        row(format!("step {k}"), g, w, k % 16 == 0);
    }
    eprintln!("libllama tensor / one GPU: worst {worst:.2e} of the largest logit");
    worst
}

/// One shape against `single` on the model: the prompt's logits and 64 greedy tokens, teacher
/// forced with the reference's, top-5 within `tol` and the greedy token equal or a near tie
/// (twice the bound). Returns the failures; ranges per decode token are checked against ADR
/// 0070 for a tensor stage.
fn check_shape(
    path: &std::path::Path,
    name: &str,
    topo: &Topology,
    tol: Tol,
    want_prompt: &[f32],
    want_steps: &[(i32, Vec<f32>)],
) -> Vec<String> {
    let mut fails = Vec::new();
    let mut m = Qwen3::load_sharded(path, topo, &opts()).unwrap();
    eprintln!(
        "{name}: {} ranks, all-reduce path {:?}, bound max({:e}, {:.2e} of the largest logit)",
        m.n_ranks(),
        m.comm_path(),
        tol.abs,
        tol.rel
    );
    let nv = m.n_vocab();
    let got = m.eval(&PROMPT).unwrap();
    for p in 0..PROMPT.len() {
        let (g, w) = (
            &got[p * nv..(p + 1) * nv],
            &want_prompt[p * nv..(p + 1) * nv],
        );
        let (d, bound) = top5_gap(g, w, tol);
        eprintln!("{name}: prompt position {p}: top-5 max |delta| {d:e}, bound {bound:e}");
        if d > bound {
            fails.push(format!("{name}: position {p}: {d} > {bound}"));
        }
    }
    m.reset();
    let mut logits = m.eval_last(&PROMPT).unwrap();
    m.reset_exec_counters();
    for (k, (t, want)) in want_steps.iter().enumerate() {
        let (d, bound) = top5_gap(&logits, want, tol);
        let (g, w) = (argmax(&logits), argmax(want));
        if k % 16 == 0 || d > bound || g != w {
            eprintln!(
                "{name}: step {k}: top-5 max |delta| {d:e}, bound {bound:e}, top-1 equal {}",
                g == w
            );
        }
        if d > bound {
            fails.push(format!("{name}: step {k}: {d} > {bound}"));
        }
        if g != w && (want[w] - want[g]).abs() > 2.0 * bound {
            fails.push(format!("{name}: step {k}: greedy {g} against {w}"));
        }
        logits = m.eval_last(&[*t]).unwrap();
    }
    let c = m.exec_counters().unwrap().clone();
    let steps = want_steps.len() as u64;
    eprintln!("{name}: ranges {:?} over {steps} decode tokens", c.ranges);
    let tp_stage = topo.stages.iter().map(|s| s.ranks.len()).max().unwrap_or(1) > 1;
    if tp_stage {
        // Per token: attention and FFN per layer, and the head after the last reduce; a
        // pipeline stage adds nothing to its ranks' count.
        let layers_of = |r: usize| {
            let s = topo.stages.iter().find(|s| s.ranks.contains(&r)).unwrap();
            u64::from(s.layers.1 - s.layers.0)
        };
        for (r, &n) in c.ranges.iter().enumerate() {
            let want = steps * (2 * layers_of(r) + 1);
            if n != want {
                fails.push(format!("{name}: rank {r}: {n} ranges, expected {want}"));
            }
        }
    } else if topo.stages.len() > 1 {
        for (r, &n) in c.ranges.iter().enumerate() {
            if n != steps {
                fails.push(format!("{name}: rank {r}: {n} ranges, expected {steps}"));
            }
        }
    }
    fails
}

/// Every shape against `single`. A shape with a tensor stage sums partial results across ranks
/// (other rounding than one rank's matmul, in F32 for these sizes: NCCL reduces in BF16 only
/// from 32768 elements on two ranks, `ggml-cuda.cu:1349`), so it is bounded by libllama's own
/// tensor-parallel gap; layer shapes compute what `single` computes and keep `GPU_TOL`.
fn shapes_against_single(shapes: Vec<(String, Topology)>) {
    let Some(path) = model() else { return };
    let mut one = Qwen3::load_sharded(&path, &single(36), &opts()).unwrap();
    let n_layer = one.hparams().attn.n_layer;
    let want_prompt = one.eval(&PROMPT).unwrap();
    one.reset();
    let mut logits = one.eval_last(&PROMPT).unwrap();
    let mut steps = Vec::new();
    for _ in 0..N_TOKENS {
        let t = argmax(&logits) as i32;
        let next = one.eval_last(&[t]).unwrap();
        steps.push((t, logits));
        logits = next;
    }
    drop(one);
    let has_tp = shapes
        .iter()
        .any(|(_, t)| t.stages.iter().any(|s| s.ranks.len() > 1));
    let tp_tol = if has_tp {
        let tokens: Vec<i32> = steps.iter().map(|(t, _)| *t).collect();
        let own = libllama_tp_anchor(&path, &tokens);
        Tol {
            abs: TOL_GPU,
            rel: GPU_TOL.rel.max(1.5 * own),
        }
    } else {
        GPU_TOL
    };
    let mut fails = Vec::new();
    for (name, topo) in &shapes {
        assert_eq!(topo.n_layer, n_layer);
        let tensor = topo.stages.iter().any(|s| s.ranks.len() > 1);
        let tol = if tensor { tp_tol } else { GPU_TOL };
        fails.extend(check_shape(&path, name, topo, tol, &want_prompt, &steps));
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// hardware: gpu2, model. `tp-2` and `layer-2` at the first, middle and last layer boundary
/// against `single`.
#[test]
fn gpu2_shapes_match_single() {
    if !Backend::require_gpus(2) {
        return;
    }
    let mut shapes = vec![("tp-2".to_string(), tp(36, 2))];
    for at in [1, 18, 35] {
        shapes.push((
            format!("layer-2 at {at}"),
            common::shapes::layer_split(36, at),
        ));
    }
    shapes_against_single(shapes);
}

/// hardware: gpu4, model. `tp-4`, `layer-4` and `tp2-pp2` against `single`.
#[test]
fn gpu4_shapes_match_single() {
    if !Backend::require_gpus(4) {
        return;
    }
    shapes_against_single(vec![
        ("tp-4".to_string(), tp(36, 4)),
        ("layer-4".to_string(), layer_g(36, 4)),
        ("tp2-pp2".to_string(), tp2pp2(36, 18)),
    ]);
}

/// hardware: gpu2, model. `layer_limit` runs on one and two ranks: the timing is real, so the
/// ranges per token are those of the truncated model.
#[test]
fn gpu2_layer_limit_runs() {
    if !Backend::require_gpus(2) {
        return;
    }
    let Some(path) = model() else { return };
    for (name, topo, ranges) in [
        ("single", single(36), vec![1u64]),
        ("tp-2", tp(36, 2), vec![2 * 4 + 1; 2]),
    ] {
        let o = DenseOptions {
            layer_limit: Some(4),
            ..opts()
        };
        let mut m = Qwen3::load_sharded(&path, &topo, &o).unwrap();
        m.eval_last(&PROMPT).unwrap();
        m.reset_exec_counters();
        let l = m.eval_last(&[5]).unwrap();
        assert!(l.iter().all(|x| x.is_finite()), "{name}");
        assert_eq!(m.exec_counters().unwrap().ranges, ranges, "{name}");
    }
}

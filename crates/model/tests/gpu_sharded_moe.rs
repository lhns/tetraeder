//! The sharded MoE builder on GPUs (ADR 0069, 0070, 0071; Phase 5, E7): Qwen3-30B-A3B Q4_K_M,
//! every shape against `single` (the same builder on one GPU), 64 greedy tokens teacher-forced
//! from `single`. Shapes that reduce partial sums or expert outputs across ranks (tensor and
//! expert parallel) are bounded by max(`GPU_TOL`, 1.5x libllama's own `-sm tensor` gap on the
//! same tokens), the rest by `GPU_TOL` (ADR 0071, 2026-11-18). Every shape runs to completion
//! and every comparison is printed before the assertion.
//!
//! Hardware label: `gpu2_` (two GPUs) and `gpu4_` (four). They need a CUDA build, the model file
//! named by `TETRAEDER_MOE_GGUF`, and patch 0004 in the build for expert parallel; they skip
//! cleanly without them (`TETRAEDER_REQUIRE_GPUS=<n>` makes missing GPUs a failure).

mod common;

use common::multi_seq::prompt;
use common::tp_anchor::libllama_tp_anchor;
use common::{GPU_TOL, TOL_GPU, Tol, argmax, top_k};
use std::path::PathBuf;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_model::{BackendChoice, LoadOptions, ModelFile, Qwen3Moe};
use tetraeder_plan::{RankSpec, Span, StageSpec, Topology};

const N_PROMPT: usize = 64;
const N_GREEDY: usize = 64;

fn model() -> Option<PathBuf> {
    let Some(path) = std::env::var_os("TETRAEDER_MOE_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_MOE_GGUF not set");
        return None;
    };
    assert!(path.is_file(), "TETRAEDER_MOE_GGUF names no file: {path:?}");
    Some(path)
}

fn opts() -> LoadOptions {
    LoadOptions {
        n_ctx: 512,
        ..LoadOptions::new(BackendChoice::Gpu { index: 0 })
    }
}

fn topo(n_layer: u32, n_ranks: usize, stages: &[(u32, u32, Vec<usize>, bool)]) -> Topology {
    let n = stages.len();
    Topology {
        ranks: (0..n_ranks)
            .map(|i| RankSpec {
                device: format!("gpu{i}"),
                gpu_index: Some(i),
            })
            .collect(),
        stages: stages
            .iter()
            .enumerate()
            .map(|(i, (a, b, rs, ep))| StageSpec {
                layers: Span(*a, *b),
                ranks: rs.clone(),
                tp: rs.len() as u32,
                ep: if *ep { rs.len() as u32 } else { 1 },
                embed: i == 0,
                head: i + 1 == n,
            })
            .collect(),
        n_layer,
        prefill_micro_batches: 1,
    }
}

fn whole(n_layer: u32, g: usize, ep: bool) -> Topology {
    topo(n_layer, g, &[(0, n_layer, (0..g).collect(), ep)])
}

fn layer_split(n_layer: u32, g: usize) -> Topology {
    let stages: Vec<_> = (0..g)
        .map(|i| {
            (
                n_layer * i as u32 / g as u32,
                n_layer * (i as u32 + 1) / g as u32,
                vec![i],
                false,
            )
        })
        .collect();
    topo(n_layer, g, &stages)
}

fn tp2_pp2(n_layer: u32) -> Topology {
    let mid = n_layer / 2;
    topo(
        n_layer,
        4,
        &[
            (0, mid, vec![0, 1], false),
            (mid, n_layer, vec![2, 3], false),
        ],
    )
}

/// The greedy run of `single`: tokens and the logits after the prompt and after each token.
struct Run {
    toks: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

fn single_run(path: &PathBuf) -> (Run, u32) {
    let n_layer = ModelFile::open(path).unwrap().hparams.n_layer;
    let mut m =
        Qwen3Moe::load_sharded(path, &Topology::single("gpu0", Some(0), n_layer), &opts()).unwrap();
    let p = prompt(1, N_PROMPT);
    let mut l = m.eval_last(&p).unwrap();
    let mut toks = p;
    let mut logits = vec![l.clone()];
    for _ in 0..N_GREEDY {
        let t = argmax(&l) as i32;
        toks.push(t);
        l = m.eval_last(&[t]).unwrap();
        logits.push(l.clone());
    }
    (Run { toks, logits }, n_layer)
}

/// `t` follows `single` teacher-forced: top-5 logits within `tol_shape`, greedy equal or a near
/// tie. Every row is printed; returns the failures. Routing is not printed: a sharded load
/// refuses `record_routing` (the router runs inside the rank graphs, and a recording hook would
/// change the executor's programs), so a near-tie router flip cannot be told from a defect here;
/// the margin between `single`'s top two logits is printed instead.
fn follows(path: &PathBuf, t: &Topology, want: &Run, tol_shape: Tol, what: &str) -> Vec<String> {
    let mut m = Qwen3Moe::load_sharded(path, t, &opts())
        .unwrap_or_else(|e| panic!("{what}: load_sharded: {e}"));
    let mut got = m.eval_last(&want.toks[..N_PROMPT]).unwrap();
    let mut fails = Vec::new();
    let mut over = 0;
    let mut ratio = 0f32;
    eprintln!(
        "{what}: bound max({:e}, {:.2e} of the largest logit)",
        tol_shape.abs, tol_shape.rel
    );
    for (j, w) in want.logits.iter().enumerate() {
        let tol = tol_shape.of(w);
        let worst = top_k(w, 5)
            .into_iter()
            .map(|i| (got[i] - w[i]).abs())
            .fold(0f32, f32::max);
        let (g, a) = (argmax(&got), argmax(w));
        let near = (w[a] - w[g]).abs() <= 2.0 * tol;
        let top = top_k(w, 2);
        let margin = w[top[0]] - w[top[1]];
        eprintln!(
            "{what}, step {j}: top-5 max |diff| {worst:e} (tolerance {tol:e}), greedy equal {}, single's top-2 margin {margin:e}",
            g == a
        );
        ratio = ratio.max(worst / tol);
        if worst > tol {
            over += 1;
            fails.push(format!("{what}: step {j}: top-5 diff {worst} > {tol}"));
        }
        if g != a && !near {
            fails.push(format!(
                "{what}: step {j}: greedy {g} against {a}, not a near tie"
            ));
        }
        if j + 1 < want.logits.len() {
            got = m.eval_last(&[want.toks[N_PROMPT + j]]).unwrap();
        }
    }
    eprintln!(
        "{what}: {over} of {} steps over the bound, worst {ratio:.2}x the bound",
        want.logits.len()
    );
    fails
}

/// Every shape against `single`; the shapes marked true sum partial results across ranks and are
/// bounded by libllama's own tensor-parallel gap. libllama has no expert-parallel split mode
/// (`SplitMode` is none, layer, row, tensor), so `-sm tensor` anchors the ep shapes too.
fn run_shapes(path: &PathBuf, want: &Run, shapes: &[(&str, Topology, bool)]) -> Vec<String> {
    let tol_tp = if shapes.iter().any(|s| s.2) {
        // 65 rows: the prompt's last row and one per token; the last token is `single`'s pick.
        let mut tokens = want.toks[N_PROMPT..].to_vec();
        tokens.push(argmax(want.logits.last().unwrap()) as i32);
        let own = libllama_tp_anchor(path, &want.toks[..N_PROMPT], &tokens, false);
        Tol {
            abs: TOL_GPU,
            rel: GPU_TOL.rel.max(1.5 * own),
        }
    } else {
        GPU_TOL
    };
    let mut fails = Vec::new();
    for (name, t, reduces) in shapes {
        let tol = if *reduces { tol_tp } else { GPU_TOL };
        fails.extend(follows(path, t, want, tol, name));
    }
    fails
}

/// hardware: gpu2 (two GPUs, the model). `tp-2`, `ep-2` and `layer-2` against `single`.
#[test]
fn gpu2_moe_shapes_match_single() {
    let Some(path) = model() else { return };
    if !Backend::require_gpus(2) {
        return;
    }
    let (want, n) = single_run(&path);
    let shapes = [
        ("tp-2", whole(n, 2, false), true),
        ("ep-2", whole(n, 2, true), true),
        ("layer-2", layer_split(n, 2), false),
    ];
    let fails = run_shapes(&path, &want, &shapes);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// hardware: gpu4 (four GPUs, the model). `tp2-pp2`, `ep-4` and `layer-4` against `single`;
/// `tp-4` is refused at load with the block reason (3 blocks of 256 in the expert down rows).
#[test]
fn gpu4_moe_shapes_match_single() {
    let Some(path) = model() else { return };
    if !Backend::require_gpus(4) {
        return;
    }
    let (want, n) = single_run(&path);
    let shapes = [
        ("tp2-pp2", tp2_pp2(n), true),
        ("ep-4", whole(n, 4, true), true),
        ("layer-4", layer_split(n, 4), false),
    ];
    let mut fails = run_shapes(&path, &want, &shapes);
    match Qwen3Moe::load_sharded(&path, &whole(n, 4, false), &opts()) {
        Err(e) if e.to_string().contains("blocks") => {}
        other => fails.push(format!("tp-4 refusal: {:?}", other.map(|_| ()))),
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// hardware: gpu2 (two GPUs, the model). ADR 0077: `layer-2` with the prompt as one step of 4
/// pipelined micro-batches against the same chunks as serial steps (M = 1): the last prompt row
/// and 16 greedy tokens (the serial run's, teacher forced) within `GPU_TOL`. Prints bit-equality
/// and the warm prefill time of both.
#[test]
fn gpu2_moe_pipelined_prefill_matches() {
    let Some(path) = model() else { return };
    if !Backend::require_gpus(2) {
        return;
    }
    const M: u32 = 4;
    let n_layer = ModelFile::open(&path).unwrap().hparams.n_layer;
    let base = layer_split(n_layer, 2);
    let piped_topo = Topology {
        prefill_micro_batches: M,
        ..base.clone()
    };
    let mut serial = Qwen3Moe::load_sharded(&path, &base, &opts()).unwrap();
    let mut piped = Qwen3Moe::load_sharded(&path, &piped_topo, &opts()).unwrap();
    let p = prompt(2, 400);
    let sizes = tetraeder_plan::micro_batch_sizes(p.len(), M);
    let run_serial = |m: &mut Qwen3Moe| {
        let mut at = 0;
        let mut l = Vec::new();
        for &n in &sizes {
            l = m.eval_last(&p[at..at + n]).unwrap();
            at += n;
        }
        l
    };
    run_serial(&mut serial);
    serial.reset();
    piped.eval_last(&p).unwrap();
    piped.reset();
    let t = std::time::Instant::now();
    let mut ls = run_serial(&mut serial);
    let ts = t.elapsed();
    let t = std::time::Instant::now();
    let mut lp = piped.eval_last(&p).unwrap();
    let tp = t.elapsed();
    eprintln!(
        "moe pipelined prefill: {} tokens, serial {:.1} ms, pipelined {:.1} ms ({:.2}x)",
        p.len(),
        ts.as_secs_f64() * 1e3,
        tp.as_secs_f64() * 1e3,
        ts.as_secs_f64() / tp.as_secs_f64()
    );
    let mut fails = Vec::new();
    for step in 0..=16 {
        let tol = GPU_TOL.of(&ls);
        let worst = top_k(&ls, 5)
            .into_iter()
            .map(|i| (lp[i] - ls[i]).abs())
            .fold(0f32, f32::max);
        let (g, w) = (argmax(&lp), argmax(&ls));
        let equal = lp.iter().zip(&ls).all(|(a, b)| a.to_bits() == b.to_bits());
        eprintln!(
            "step {step}: top-5 max |diff| {worst:e}, tolerance {tol:e}, bit-equal {equal}, greedy equal {}",
            g == w
        );
        if worst > tol {
            fails.push(format!("step {step}: {worst} > {tol}"));
        }
        if g != w && (ls[w] - ls[g]).abs() > 2.0 * tol {
            fails.push(format!("step {step}: greedy {g} against {w}"));
        }
        if step < 16 {
            let t = w as i32;
            ls = serial.eval_last(&[t]).unwrap();
            lp = piped.eval_last(&[t]).unwrap();
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

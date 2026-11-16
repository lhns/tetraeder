//! The sharded MoE builder on GPUs (ADR 0069, 0070, 0071; Phase 5, E7): Qwen3-30B-A3B Q4_K_M,
//! every shape against `single` (the same builder on one GPU) within the GPU rule, 64 greedy
//! tokens teacher-forced from `single`.
//!
//! Hardware label: `gpu2_` (two GPUs) and `gpu4_` (four). They need a CUDA build, the model file
//! named by `TETRAEDER_MOE_GGUF`, and patch 0004 in the build for expert parallel; they skip
//! cleanly without them (`TETRAEDER_REQUIRE_GPUS=<n>` makes missing GPUs a failure).

mod common;

use common::multi_seq::prompt;
use common::{GPU_TOL, argmax, top_k};
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

/// `t` follows `single` teacher-forced: top-5 logits within the GPU rule, greedy equal or a near
/// tie. Every row is printed before any assertion.
fn follows(path: &PathBuf, t: &Topology, want: &Run, what: &str) {
    let mut m = Qwen3Moe::load_sharded(path, t, &opts())
        .unwrap_or_else(|e| panic!("{what}: load_sharded: {e}"));
    let mut got = m.eval_last(&want.toks[..N_PROMPT]).unwrap();
    let mut fails = Vec::new();
    for (j, w) in want.logits.iter().enumerate() {
        let tol = GPU_TOL.of(w);
        let worst = top_k(w, 5)
            .into_iter()
            .map(|i| (got[i] - w[i]).abs())
            .fold(0f32, f32::max);
        let (g, a) = (argmax(&got), argmax(w));
        let near = (w[a] - w[g]).abs() <= 2.0 * tol;
        eprintln!(
            "{what}, step {j}: top-5 max |diff| {worst:e} (tolerance {tol:e}), greedy equal {}",
            g == a
        );
        if worst > tol {
            fails.push(format!("step {j}: top-5 diff {worst} > {tol}"));
        }
        if g != a && !near {
            fails.push(format!("step {j}: greedy {g} against {a}, not a near tie"));
        }
        if j + 1 < want.logits.len() {
            got = m.eval_last(&[want.toks[N_PROMPT + j]]).unwrap();
        }
    }
    assert!(fails.is_empty(), "{what}: {fails:?}");
}

/// hardware: gpu2 (two GPUs, the model). `tp-2`, `ep-2` and `layer-2` against `single`.
#[test]
fn gpu2_moe_shapes_match_single() {
    let Some(path) = model() else { return };
    if !Backend::require_gpus(2) {
        return;
    }
    let (want, n) = single_run(&path);
    follows(&path, &whole(n, 2, false), &want, "tp-2");
    follows(&path, &whole(n, 2, true), &want, "ep-2");
    follows(&path, &layer_split(n, 2), &want, "layer-2");
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
    follows(&path, &tp2_pp2(n), &want, "tp2-pp2");
    follows(&path, &whole(n, 4, true), &want, "ep-4");
    follows(&path, &layer_split(n, 4), &want, "layer-4");
    let err = Qwen3Moe::load_sharded(&path, &whole(n, 4, false), &opts())
        .unwrap_err()
        .to_string();
    assert!(err.contains("blocks"), "tp-4 refusal: {err}");
}

//! Batched parity of the own builders on the real verdict models (ADR 0065 "Verdict", ADR 0071):
//! three sequences, a prompt chunk mixed into the decode steps of the others, against libllama
//! on CUDA with the same step shapes. `gpu_dense.rs` checks Qwen3-8B with one sequence only.
//!
//! The bound is anchored to libllama's own noise on the same model and card (ADR 0071): its
//! batched run against split, and against the same batched steps with every sequence's KV cells
//! shifted. The second is needed on real models: the builder reserves KV rows per sequence where
//! libllama allocates cells in step order, so CUDA flash attention sums in another order. With
//! Q4_K weights the activations are rounded to q8_1 before every matmul, which can grow such a
//! last-bit difference over the layers; the first box run (`bench/results/20261116d2a/phase4/`)
//! measured 1.3e-2 to 3e-2 of the largest logit between builder and libllama on Qwen3-8B, flat in
//! position and sequence, while the one row whose cells and step shapes coincide (sequence 0 in
//! the first step) was bit-equal. The shifted run measures that sensitivity on libllama itself.
//!
//! Hardware label: `gpu_*`, model (`cpu_*` runs the same body on the tiny fixtures). Skips
//! cleanly when there is no GPU (a failure with `TETRAEDER_REQUIRE_GPU`) or the model file is
//! absent from `TETRAEDER_MODELS_DIR`. libllama and the builder are never on the GPU at the same
//! time: the libllama logits are plain values and its model is dropped before the builder loads.

mod common;

use common::multi_seq::{Step, load, prompt, run_steps, schedule, script_items, steps_logits};
use common::{
    GPU_TOL, Item, Logits, TOL_CPU, Tol, builder_script, compare_scripts_tol, dense_fixture,
    fixture, libllama_script_on, part, print_script_rows, run_all,
};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, Qwen3};
use tetraeder_plan::KvFormat;

const GPU: BackendChoice = BackendChoice::Gpu { index: 0 };

/// The model file, or None (skip) without a GPU or the file.
fn model(rel: &str) -> Option<PathBuf> {
    if !Backend::gpu_available() {
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_GPU").is_none(),
            "TETRAEDER_REQUIRE_GPU is set but there is no GPU device"
        );
        eprintln!("skip: no GPU");
        return None;
    }
    let Some(dir) = std::env::var_os("TETRAEDER_MODELS_DIR") else {
        eprintln!("skip: TETRAEDER_MODELS_DIR is not set");
        return None;
    };
    let path = PathBuf::from(dir).join(rel);
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

/// Three prompts of 20, 24 and 40 tokens plus 6 more tokens each; sequence 2 joins in the second
/// step with a 16-token chunk next to the decode tokens of the others. Every step has at most 64
/// tokens: `libllama_script_on` runs libllama with `n_batch` 64, so a larger step would be split
/// into ubatches there and no longer match the builder's step shapes.
fn script() -> (Vec<Vec<i32>>, Vec<Step>) {
    let lens = [20usize, 24, 40];
    let toks: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let mut t = prompt(1 + k as i32, n);
            t.extend(prompt(11 + k as i32, 6));
            t
        })
        .collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    assert!(
        steps
            .iter()
            .all(|s| s.iter().map(|(_, r)| r.len()).sum::<usize>() <= 64)
    );
    (toks, steps)
}

/// KV cells a fourth, dummy sequence takes ahead of the script in the shifted libllama run.
/// Not a multiple of any kernel tile, so every sequence's cells move within the KV tiles.
const SHIFT: usize = 7;

/// libllama batched (as [`libllama_script_on`]) after a dummy sequence of [`SHIFT`] tokens took
/// the first cells of the unified cache: the same step shapes and kernels, only the KV cells of
/// every sequence moved. Its gap to the unshifted run is libllama's own sensitivity to where a
/// sequence sits in the cache, which the builder's per-sequence row reservation differs in.
fn libllama_shifted(path: &Path, toks: &[Vec<i32>], items: &[Vec<Item>], ngl: i32) -> Logits {
    let dummy = toks.len() as u32;
    let mut streams = toks.to_vec();
    streams.push(prompt(99, SHIFT));
    let mut steps = vec![vec![Item {
        seq: dummy,
        range: 0..SHIFT,
        all: false,
    }]];
    steps.extend(items.iter().cloned());
    let mut out = libllama_script_on(path, &streams, &steps, KvFormat::F16, ngl);
    out.retain(|(s, _), _| *s != dummy);
    out
}

/// Every item of `steps` its own call.
fn split_items(steps: &[Vec<Item>]) -> Vec<Vec<Item>> {
    steps.iter().flatten().map(|i| vec![i.clone()]).collect()
}

/// The five runs of the anchor (ADR 0071) for one model.
struct Runs {
    ll_batched: Logits,
    ll_split: Logits,
    ll_shifted: Logits,
    ours_batched: Logits,
    ours_split: Logits,
}

/// libllama's three runs (batched, split, batched with the KV cells shifted), one model load
/// each; the model is dropped before the builder loads.
fn libllama_runs(path: &Path, toks: &[Vec<i32>], items: &[Vec<Item>], ngl: i32) -> [Logits; 3] {
    let kv = KvFormat::F16;
    [
        libllama_script_on(path, toks, items, kv, ngl),
        libllama_script_on(path, toks, &split_items(items), kv, ngl),
        libllama_shifted(path, toks, items, ngl),
    ]
}

/// Prints every comparison, then asserts. libllama's own gaps (batched against split, and
/// against the shifted cells) are recorded, not asserted; the bound is `tol` widened to 1.5
/// times the larger of them (ADR 0071). The builder batched against split is asserted within
/// that bound; the builder against libllama at the same step shapes within it with
/// `anchored`, else within `tol`. On CUDA the builder's KV rows are not libllama's cells
/// (rows reserved per sequence against cells allocated in step order), so flash attention sums
/// in another order there and the same-shape gap is noise of libllama's own size; on the CPU
/// the order does not depend on the cells and the gap is 0, so `tol` stays.
fn judge(what: &str, tol: Tol, anchored: bool, r: &Runs) {
    let pairs = [
        (
            "libllama batched / libllama split",
            &r.ll_split,
            &r.ll_batched,
        ),
        (
            "libllama batched / libllama shifted cells",
            &r.ll_batched,
            &r.ll_shifted,
        ),
        (
            "builder batched / builder split",
            &r.ours_split,
            &r.ours_batched,
        ),
        (
            "builder batched / libllama batched",
            &r.ll_batched,
            &r.ours_batched,
        ),
        ("builder split / libllama split", &r.ll_split, &r.ours_split),
    ];
    let worst: Vec<f32> = pairs
        .iter()
        .map(|(name, want, have)| print_script_rows(&format!("{what} {name}"), want, have))
        .collect();
    for ((name, _, _), w) in pairs.iter().zip(&worst) {
        eprintln!("{what} {name}: worst {w:.2e} of the largest logit");
    }
    let own = Tol {
        abs: tol.abs,
        rel: tol.rel.max(1.5 * worst[0].max(worst[1])),
    };
    let same = if anchored { own } else { tol };
    eprintln!(
        "{what}: bound max({:e}, {:.2e} of the largest logit), same shapes max({:e}, {:.2e})",
        own.abs, own.rel, same.abs, same.rel
    );
    let parts = pairs[2..]
        .iter()
        .enumerate()
        .map(|(i, (name, want, have))| {
            let t = if i == 0 { own } else { same };
            part(*name, move || {
                compare_scripts_tol(want, have, t, &format!("{what} {name}"));
            })
        })
        .collect();
    run_all(parts);
}

/// The dense builder on `backend` against libllama on the same device.
fn dense_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let anchored = matches!(backend, BackendChoice::Gpu { .. });
    let (toks, steps) = script();
    let items = script_items(&steps);
    let [ll_batched, ll_split, ll_shifted] = libllama_runs(path, &toks, &items, ngl);
    let o = DenseOptions {
        n_seq_max: 3,
        n_ctx_pool: 1024, // rows are reserved in steps of 256 per sequence
        // as libllama_script_on loads the model
        extra_bufts: false,
        ..DenseOptions::new(backend)
    };
    let mut m = Qwen3::load(path, &o).unwrap();
    let ours_batched = builder_script(&mut m, &toks, &items, true);
    for k in 0..toks.len() as u32 {
        m.seq_rm(k, 0).unwrap();
    }
    let ours_split = builder_script(&mut m, &toks, &items, false);
    drop(m);
    let runs = Runs {
        ll_batched,
        ll_split,
        ll_shifted,
        ours_batched,
        ours_split,
    };
    judge(what, tol, anchored, &runs);
}

/// The MoE builder (resident) on `backend` against libllama on the same device.
fn moe_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let anchored = matches!(backend, BackendChoice::Gpu { .. });
    let (toks, steps) = script();
    let [ll_batched, ll_split, ll_shifted] = libllama_runs(path, &toks, &script_items(&steps), ngl);
    let o = LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(backend)
    };
    let mut m = load(path, &o, 3);
    let ours_batched = steps_logits(run_steps(&mut m, &toks, &steps), &steps);
    for k in 0..toks.len() as u32 {
        m.seq_rm(k, 0).unwrap();
    }
    let split: Vec<Step> = steps.iter().flatten().map(|x| vec![x.clone()]).collect();
    let ours_split = steps_logits(run_steps(&mut m, &toks, &split), &split);
    drop(m);
    let runs = Runs {
        ll_batched,
        ll_split,
        ll_shifted,
        ours_batched,
        ours_split,
    };
    judge(what, tol, anchored, &runs);
}

/// hardware: gpu, model. Qwen3-8B Q4_K_M, the dense builder batched against libllama batched.
#[test]
fn gpu_qwen3_8b_batched_matches_libllama() {
    let Some(path) = model("Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf") else {
        return;
    };
    dense_vs_libllama(
        &path,
        GPU,
        -1,
        GPU_TOL,
        "qwen3-8b batched / libllama batched",
    );
}

/// hardware: gpu, model. Qwen3-30B-A3B Q4_K_M, the MoE builder (resident) batched against
/// libllama batched.
#[test]
fn gpu_qwen3_30b_batched_matches_libllama() {
    let Some(path) = model("Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf") else {
        return;
    };
    moe_vs_libllama(
        &path,
        GPU,
        -1,
        GPU_TOL,
        "qwen3-30b batched / libllama batched",
    );
}

/// hardware: cpu. The body of the two tests above on the tiny fixtures, so CI runs the script,
/// the step sizes and the comparison; builder and libllama run the same kernels on the CPU.
#[test]
fn cpu_fixtures_batched_match_libllama() {
    let cpu = || BackendChoice::Cpu { threads: Some(2) };
    let tol = Tol::from(TOL_CPU);
    dense_vs_libllama(
        &dense_fixture(),
        cpu(),
        0,
        tol,
        "cpu dense fixture batched / libllama",
    );
    moe_vs_libllama(
        &fixture(),
        cpu(),
        0,
        tol,
        "cpu moe fixture batched / libllama",
    );
}

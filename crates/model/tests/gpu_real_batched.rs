//! Batched parity of the own builders on the real verdict models (ADR 0065 "Verdict", ADR 0071):
//! three sequences, a prompt chunk mixed into the decode steps of the others, against libllama
//! on CUDA with the same step shapes. `gpu_dense.rs` checks Qwen3-8B with one sequence only.
//!
//! Hardware label: `gpu_*`, model (`cpu_*` runs the same body on the tiny fixtures). Skips
//! cleanly when there is no GPU (a failure with `TETRAEDER_REQUIRE_GPU`) or the model file is
//! absent from `TETRAEDER_MODELS_DIR`. libllama and the builder are never on the GPU at the same
//! time: the libllama logits are plain values and its model is dropped before the builder loads.

mod common;

use common::multi_seq::{Step, load, prompt, run_steps, schedule, script_items, steps_logits};
use common::{
    GPU_TOL, Logits, TOL_CPU, Tol, builder_script, compare_scripts_tol, dense_fixture, fixture,
    libllama_script_on, print_script_rows,
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

fn check(what: &str, tol: Tol, want: &Logits, have: &Logits) {
    print_script_rows(what, want, have);
    let worst = compare_scripts_tol(want, have, tol, what);
    eprintln!("{what}: max |delta| {worst:e}");
}

/// The dense builder on `backend` batched against libllama batched on the same device.
fn dense_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let (toks, steps) = script();
    let items = script_items(&steps);
    let want = libllama_script_on(path, &toks, &items, KvFormat::F16, ngl);
    let o = DenseOptions {
        n_seq_max: 3,
        n_ctx_pool: 1024, // rows are reserved in steps of 256 per sequence
        // as libllama_script_on loads the model
        extra_bufts: false,
        ..DenseOptions::new(backend)
    };
    let mut m = Qwen3::load(path, &o).unwrap();
    let have = builder_script(&mut m, &toks, &items, true);
    check(what, tol, &want, &have);
}

/// The MoE builder (resident) on `backend` batched against libllama batched on the same device.
fn moe_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let (toks, steps) = script();
    let want = libllama_script_on(path, &toks, &script_items(&steps), KvFormat::F16, ngl);
    let o = LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(backend)
    };
    let mut m = load(path, &o, 3);
    let have = steps_logits(run_steps(&mut m, &toks, &steps), &steps);
    check(what, tol, &want, &have);
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

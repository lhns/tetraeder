//! The dense builder on CUDA (ADR 0066): the tiny dense fixture against the CPU reference within
//! `TOL_GPU`, one and three sequences; Qwen3-8B Q4_K_M top-1 and 64 greedy tokens against
//! libllama when `TETRAEDER_MODELS_DIR` has it.
//!
//! Hardware label: `gpu_*`. Without a GPU the tests skip; with `TETRAEDER_REQUIRE_GPU` set a
//! missing GPU fails them (`Backend::gpu_available`).

mod common;

use common::*;
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};
use tetraeder_plan::KvFormat;

const GPU: BackendChoice = BackendChoice::Gpu { index: 0 };
const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

fn load(backend: BackendChoice, kv: KvFormat, n_seq_max: usize) -> Qwen3 {
    let o = DenseOptions {
        kv,
        n_seq_max,
        n_ctx_pool: 256 * n_seq_max,
        ..DenseOptions::new(backend)
    };
    Qwen3::load(dense_fixture(), &o).unwrap()
}

fn stream(seed: u32, len: usize) -> Vec<i32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 16) % 256) as i32
        })
        .collect()
}

/// hardware: gpu. One sequence: prefill logits and greedy steps against libllama on the CPU,
/// F16 cache and the Q8_0 cache with libllama's default rotation.
#[test]
fn gpu_one_sequence_matches_the_cpu_reference() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU");
        return;
    }
    for kv in [KvFormat::F16, KvFormat::Q8_0] {
        let r = libllama_reference_kv_default_rot(&dense_fixture(), &PROMPT, N_GREEDY, kv);
        let mut m = load(GPU, kv, 1);
        let worst = check_against(&mut m, &r, TOL_GPU);
        eprintln!("{kv} on GPU: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: gpu. Three sequences in steps with a prefill chunk mixed in, against the same
/// script on the CPU builder.
#[test]
fn gpu_three_sequences_match_the_cpu_builder() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU");
        return;
    }
    let streams: Vec<Vec<i32>> = [12usize, 13, 11]
        .iter()
        .enumerate()
        .map(|(i, &n)| stream(7 + i as u32, n))
        .collect();
    let it = |seq, range, all| Item { seq, range, all };
    let steps = vec![
        vec![it(0, 0..6, false), it(1, 0..5, false), it(2, 0..4, false)],
        vec![it(0, 6..7, false), it(1, 5..9, true), it(2, 4..5, false)],
        vec![it(0, 7..8, false), it(1, 9..10, false), it(2, 5..6, false)],
        vec![it(0, 8..9, false), it(1, 10..11, false), it(2, 6..7, false)],
    ];
    let kv = KvFormat::F16;
    let want = builder_script(&mut load(CPU, kv, 3), &streams, &steps, true);
    let have = builder_script(&mut load(GPU, kv, 3), &streams, &steps, true);
    let worst = compare_scripts(&want, &have, TOL_GPU, false, "gpu batched / cpu batched");
    eprintln!("three sequences on GPU: max |delta| {worst:e}");
}

/// libllama on the GPU: 64 greedy tokens after `PROMPT`.
fn libllama_greedy(path: &std::path::Path) -> Vec<i32> {
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers: -1,
                ..ModelParams::default()
            },
        )
        .unwrap(),
    );
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
    .unwrap();
    let mut batch = Batch::new(64);
    for (i, &t) in PROMPT.iter().enumerate() {
        batch.push(t, i as i32, 0, i + 1 == PROMPT.len()).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let mut next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
    let mut greedy = vec![next];
    for step in 0..63 {
        batch.clear();
        batch
            .push(next, (PROMPT.len() + step) as i32, 0, true)
            .unwrap();
        ctx.decode(&batch).unwrap();
        next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
        greedy.push(next);
    }
    greedy
}

/// hardware: gpu, model. Qwen3-8B Q4_K_M: top-1 equal and 64 greedy tokens equal to libllama's.
#[test]
fn gpu_qwen3_8b_matches_libllama() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU");
        return;
    }
    let Some(dir) = std::env::var_os("TETRAEDER_MODELS_DIR") else {
        eprintln!("skip: TETRAEDER_MODELS_DIR is not set");
        return;
    };
    let path = PathBuf::from(dir).join("Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf");
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let reference = libllama_greedy(&path);
    let o = DenseOptions {
        n_ctx_pool: 512,
        ..DenseOptions::new(GPU)
    };
    let mut m = Qwen3::load(&path, &o).unwrap();
    let mut next = argmax(&m.eval_last(&PROMPT).unwrap()) as i32;
    let mut ours = vec![next];
    for _ in 0..63 {
        next = argmax(&m.eval_last(&[next]).unwrap()) as i32;
        ours.push(next);
    }
    assert_eq!(ours[0], reference[0], "top-1 of the prompt");
    let diverged = ours.iter().zip(&reference).position(|(a, b)| a != b);
    assert!(
        diverged.is_none(),
        "greedy diverges at {diverged:?}: {ours:?} vs {reference:?}"
    );
}

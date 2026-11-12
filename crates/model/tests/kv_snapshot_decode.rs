//! Snapshot, reload, restore, decode (the multi-arm exit harness, `exit_regime.rs`
//! `TETRAEDER_EXIT_ARMS`): a fresh model that only ran the first ubatch (so its hot set is built)
//! and then `kv_restore`s the prefill's KV must decode the same greedy tokens and logits as the
//! model that prefilled. The model is streamed (small VRAM and RAM tiers, prefetch, the I/O
//! service and overlap on a virtual clock), so the tier state differs between the two runs and
//! only the KV carries the prompt. Also in place: restore after decoding, then decode again.
//!
//! Why it holds: the KV is a plain `[n_embd_kv, n_ctx]` buffer per layer, the step inputs
//! (tokens, positions, mask) are rebuilt from `n_past` every step, and the expert tiers, the
//! prefetch prediction and the hot set are outside the KV, so they change timing, not values.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` need no GPU; `gpu_*` skip without a GPU device.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_memory::IoOptions;
use tetraeder_memory::io::IoFactory;
use tetraeder_memory::io::sim::{SimClock, SimIo};
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 64;
const UBATCH: usize = 32;
const N_DECODE: usize = 16;

type Decoded = (Vec<i32>, Vec<Vec<f32>>);

fn sim_io() -> IoFactory {
    Arc::new(|paths| {
        let io = SimIo::flat(paths, SimClock::new(), 1e8, 4, 4096)?;
        Ok(Box::new(io))
    })
}

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-prefetch")
}

fn prompt() -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect()
}

fn load(path: &Path, backend: BackendChoice) -> Qwen3Moe {
    let o = LoadOptions {
        expert_slots: Some(8),
        ram_slots: Some(12),
        miss: MissPolicy::Cpu,
        prefetch: true,
        io: IoOptions {
            service: true,
            workers: 4,
            ..IoOptions::default()
        },
        overlap: true,
        io_factory: Some(sim_io()),
        n_ctx: 256,
        ..LoadOptions::new(backend)
    };
    Qwen3Moe::load(path, &o).unwrap()
}

/// The prefill in ubatches; returns the last logits.
fn prefill(m: &mut Qwen3Moe) -> Vec<f32> {
    let mut last = Vec::new();
    for c in prompt().chunks(UBATCH) {
        last = m.eval_last(c).unwrap();
    }
    last
}

/// Greedy decode from the prefill's last logits: tokens and every step's logits.
fn decode(m: &mut Qwen3Moe, last: &[f32]) -> Decoded {
    let mut next = argmax(last) as i32;
    let (mut toks, mut all) = (Vec::new(), Vec::new());
    for _ in 0..N_DECODE {
        toks.push(next);
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        all.push(l);
    }
    (toks, all)
}

fn assert_same(want: &Decoded, got: &Decoded, tol: f32) {
    assert_eq!(want.0, got.0, "greedy tokens");
    for (i, (a, b)) in want.1.iter().zip(&got.1).enumerate() {
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= tol, "step {i}: max |delta| {worst:e} > {tol:e}");
    }
}

fn check(backend: BackendChoice, tol: f32) {
    let path = fixture();
    let mut a = load(&path, backend);
    let last = prefill(&mut a);
    let snap = a.kv_snapshot().unwrap();
    let want = decode(&mut a, &last);

    // in place: restore the snapshot and decode again
    a.kv_restore(&snap, N_PROMPT).unwrap();
    assert_eq!(a.n_past(), N_PROMPT);
    assert_same(&want, &decode(&mut a, &last), tol);

    // reload: a fresh model, the first ubatch only (builds the hot set), then the restore
    drop(a);
    let mut b = load(&path, backend);
    b.eval_last(&prompt()[..UBATCH]).unwrap();
    b.kv_restore(&snap, N_PROMPT).unwrap();
    assert_same(&want, &decode(&mut b, &last), tol);
}

/// hardware: cpu. Bitwise on the CPU backend (same kernels, same op order).
#[test]
fn cpu_snapshot_reload_restore_decodes_like_prefill() {
    check(CPU, 0.0);
}

/// hardware: gpu (skips without a GPU device).
#[test]
fn gpu_snapshot_reload_restore_decodes_like_prefill() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    check(BackendChoice::Gpu { index: 0 }, common::TOL_GPU);
}

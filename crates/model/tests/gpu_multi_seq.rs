//! `multi_seq.rs` on a GPU (ADR 0065, 0067; phase 4, C4): several sequences in one step, resident
//! and streamed experts, `seq_cp`, snapshots and a chunked move mid-decode, within `TOL_GPU` of
//! each sequence alone. This is also where CUDA flash attention with sequences mixed in one
//! query tile is checked against single-sequence runs (ADR 0065, "To verify on the box").
//!
//! Hardware label: `gpu_*` (a GPU device in the build); skips cleanly without one, and fails
//! when `TETRAEDER_REQUIRE_GPU` is set.

mod common;

use common::multi_seq::*;
use common::synth::{MoeSpec, quantized_moe};
use common::{TOL_GPU, fixture};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy};
use tetraeder_plan::KvFormat;

fn base() -> Option<LoadOptions> {
    if !Backend::gpu_available() {
        // The box's gpu-parity stage sets this: no GPU there is a failure, not a skip.
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_GPU").is_none(),
            "TETRAEDER_REQUIRE_GPU is set but there is no GPU device"
        );
        eprintln!("skip: no GPU device in this build or on this machine");
        return None;
    }
    Some(LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(BackendChoice::Gpu { index: 0 })
    })
}

/// hardware: gpu.
#[test]
fn gpu_resident_batched_equals_alone() {
    let Some(o) = base() else { return };
    batched_equals_alone(&fixture(), &o, TOL_GPU, "gpu resident");
}

/// hardware: gpu. 2 of 8 experts per layer on the card, uploads on a miss.
#[test]
fn gpu_streamed_upload_batched_equals_alone() {
    let Some(o) = base() else { return };
    let o = LoadOptions {
        expert_slots: Some(2),
        ..o
    };
    let m = batched_equals_alone(&fixture(), &o, TOL_GPU, "gpu upload");
    assert!(m.expert_stats().expect("streaming").io.experts_read > 0);
}

/// hardware: gpu. A RAM tier; the misses run as CPU groups next to the GPU segments.
#[test]
fn gpu_streamed_cpu_miss_batched_equals_alone() {
    let Some(o) = base() else { return };
    let o = LoadOptions {
        expert_slots: Some(2),
        ram_slots: Some(3),
        miss: MissPolicy::Cpu,
        ..o
    };
    batched_equals_alone(&fixture(), &o, TOL_GPU, "gpu cpu miss");
}

/// hardware: gpu.
#[test]
fn gpu_seq_cp_snapshot_and_move() {
    let Some(o) = base() else { return };
    seq_cp_continues(&fixture(), &o, TOL_GPU, "gpu");
    snapshot_roundtrip(&fixture(), &o, TOL_GPU, "gpu");
    let big = LoadOptions {
        n_ctx: 2048,
        ..o.clone()
    };
    move_mid_decode(&fixture(), &big, TOL_GPU, "gpu");
    // The copy kernel on a quantized cache.
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8");
    let q = LoadOptions {
        n_ctx: 2048,
        kv: KvFormat::Q8_0,
        kv_rotation: Some(true),
        ..o
    };
    move_mid_decode(&path, &q, TOL_GPU, "gpu q8_0 kv");
}

/// hardware: gpu. The same batched steps on the GPU builder and on the CPU builder, and the GPU
/// run against each sequence alone on the CPU.
#[test]
fn gpu_batched_matches_the_cpu_builder() {
    let Some(o) = base() else { return };
    let cpu = LoadOptions {
        backend: BackendChoice::Cpu { threads: Some(2) },
        ..o.clone()
    };
    let path = fixture();
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let alones: Vec<Alone> = prompts.iter().map(|p| alone(&path, &cpu, p, 6)).collect();
    let toks: Vec<Vec<i32>> = alones.iter().map(|a| a.toks.clone()).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    let on_gpu = run_steps(&mut load(&path, &o, 3), &toks, &steps);
    let on_cpu = run_steps(&mut load(&path, &cpu, 3), &toks, &steps);
    for (si, (g, c)) in on_gpu.iter().zip(&on_cpu).enumerate() {
        for (ii, (x, y)) in g.iter().zip(c).enumerate() {
            let what = format!("gpu against cpu, step {si}, input {ii}");
            close(x, y, TOL_GPU, &what);
            same_token(x, y, TOL_GPU, &what);
        }
    }
    check_steps(
        &on_gpu,
        &steps,
        &alones,
        TOL_GPU,
        "gpu against alone on cpu",
    );
}

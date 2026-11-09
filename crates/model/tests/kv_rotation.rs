//! Hadamard rotation of a quantized KV cache (ADR 0050 addendum): the builder against libllama
//! with its default rotation ON, on the Q8_0 synthetic model (head size 64), resident and
//! streamed. This binary never turns libllama's rotation off (it is a process-wide setting; the
//! rotation-off parity lives in `kv_format.rs`).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use common::*;
use std::path::PathBuf;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const FORMATS: [KvFormat; 3] = [KvFormat::Q8_0, KvFormat::Q8_0_Q4_0, KvFormat::Q4_0];

fn model_path() -> PathBuf {
    assert!(
        std::env::var("LLAMA_ATTN_ROT_DISABLE").map_or(true, |v| v != "1"),
        "LLAMA_ATTN_ROT_DISABLE must not be set for the rotation-on parity tests"
    );
    quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8")
}

fn opts(kv: KvFormat, streamed: bool) -> LoadOptions {
    let base = LoadOptions {
        kv,
        ..LoadOptions::new(CPU)
    };
    if streamed {
        LoadOptions {
            expert_slots: Some(8),
            ram_slots: Some(16),
            miss: MissPolicy::Cpu,
            ..base
        }
    } else {
        base
    }
}

/// hardware: cpu. The default follows libllama: a quantized cache is rotated on both sides, F16
/// is not, and the flag overrides both ways.
#[test]
fn cpu_rotation_default_follows_the_format() {
    let path = model_path();
    let rotated = |o: &LoadOptions| Qwen3Moe::load(&path, o).unwrap().kv_rotated();
    assert_eq!(rotated(&opts(KvFormat::F16, false)), (false, false));
    for kv in FORMATS {
        assert_eq!(rotated(&opts(kv, false)), (true, true), "{kv}");
        let off = LoadOptions {
            kv_rotation: Some(false),
            ..opts(kv, false)
        };
        assert_eq!(rotated(&off), (false, false), "{kv}");
    }
    let on = LoadOptions {
        kv_rotation: Some(true),
        ..opts(KvFormat::F16, false)
    };
    assert_eq!(rotated(&on), (true, true));
}

/// hardware: cpu. Rotation on a head size that is not a multiple of 64 is a load error; the
/// default silently leaves it off, as libllama does.
#[test]
fn cpu_rotation_needs_head_size_64() {
    let err = Qwen3Moe::load(
        fixture(),
        &LoadOptions {
            kv_rotation: Some(true),
            ..LoadOptions::new(CPU)
        },
    )
    .expect_err("head size 16");
    assert!(err.to_string().contains("multiple of 64"), "{err}");
    let m = Qwen3Moe::load(fixture(), &LoadOptions::new(CPU)).unwrap();
    assert_eq!(m.kv_rotated(), (false, false));
}

/// hardware: cpu. Same logits as libllama with its rotation on.
#[test]
fn cpu_rotated_kv_matches_libllama_resident() {
    let path = model_path();
    for kv in FORMATS {
        let r = libllama_reference_kv_default_rot(&path, &PROMPT, N_GREEDY, kv);
        let mut m = Qwen3Moe::load(&path, &opts(kv, false)).unwrap();
        assert_eq!(m.kv_rotated(), (true, true));
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv} rotated: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: cpu. Experts in two tiers, the RAM misses computed on the CPU.
#[test]
fn cpu_rotated_kv_matches_libllama_streamed() {
    let path = model_path();
    for kv in FORMATS {
        let r = libllama_reference_kv_default_rot(&path, &PROMPT, N_GREEDY, kv);
        let mut m = Qwen3Moe::load(&path, &opts(kv, true)).unwrap();
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv} rotated, streamed: max |delta|: {worst:e}");
    }
}

/// hardware: cpu. The rotation is in effect: the logits differ from the unrotated cache.
#[test]
fn cpu_rotation_changes_the_logits() {
    let path = model_path();
    let run = |rotation| {
        let o = LoadOptions {
            kv_rotation: Some(rotation),
            ..opts(KvFormat::Q4_0, false)
        };
        Qwen3Moe::load(&path, &o)
            .unwrap()
            .eval_last(&PROMPT)
            .unwrap()
    };
    let (off, on) = (run(false), run(true));
    assert!(off.iter().zip(&on).any(|(a, b)| a != b));
    assert!(on.iter().all(|x| x.is_finite()));
}

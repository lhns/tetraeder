//! Low-bit KV cache of the builder (ADR 0050 b) on the K-quant synthetic model (head size 64, so
//! Q8_0 and Q4_0 blocks divide it): parity against libllama with the same `type_k` and `type_v`,
//! resident and streamed, the prefix-cache snapshot and the memory report.
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
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "kv")
}

fn streamed(kv: KvFormat) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(8),
        ram_slots: Some(16),
        miss: MissPolicy::Cpu,
        kv,
        ..LoadOptions::new(CPU)
    }
}

/// hardware: cpu. The reference really runs the quantized cache: its logits differ from F16's.
#[test]
fn cpu_reference_kv_types_take_effect() {
    let path = model_path();
    let a = libllama_reference_kv(&path, &PROMPT, N_GREEDY, KvFormat::F16);
    let b = libllama_reference_kv(&path, &PROMPT, N_GREEDY, KvFormat::Q8_0);
    assert!(
        a.prompt_logits
            .iter()
            .zip(&b.prompt_logits)
            .any(|(x, y)| x != y)
    );
}

/// hardware: cpu. The control: with the F16 cache this model matches libllama too.
#[test]
fn cpu_f16_kv_matches_libllama_on_the_synthetic_model() {
    let path = model_path();
    let r = libllama_reference_kv(&path, &PROMPT, N_GREEDY, KvFormat::F16);
    let mut m = Qwen3Moe::load(&path, &LoadOptions::new(CPU)).unwrap();
    let worst = check_against(&mut m, &r, TOL_CPU);
    eprintln!("f16: max |delta| over the top-{TOP_K} logits: {worst:e}");
}

/// hardware: cpu
#[test]
fn cpu_quantized_kv_matches_libllama_resident() {
    let path = model_path();
    for kv in FORMATS {
        let r = libllama_reference_kv(&path, &PROMPT, N_GREEDY, kv);
        let o = LoadOptions {
            kv,
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv}: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: cpu. Experts in two tiers, the RAM misses computed on the CPU.
#[test]
fn cpu_quantized_kv_matches_libllama_streamed() {
    let path = model_path();
    for kv in FORMATS {
        let r = libllama_reference_kv(&path, &PROMPT, N_GREEDY, kv);
        let mut m = Qwen3Moe::load(&path, &streamed(kv)).unwrap();
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv} streamed: max |delta|: {worst:e}");
    }
}

/// hardware: cpu. A quantized cache really differs from F16 (the formats are in effect).
#[test]
fn cpu_quantized_kv_changes_the_logits() {
    let path = model_path();
    let run = |kv| {
        let o = LoadOptions {
            kv,
            ..LoadOptions::new(CPU)
        };
        Qwen3Moe::load(&path, &o)
            .unwrap()
            .eval_last(&PROMPT)
            .unwrap()
    };
    let (f16, q4) = (run(KvFormat::F16), run(KvFormat::Q4_0));
    assert!(f16.iter().zip(&q4).any(|(a, b)| a != b));
    assert!(q4.iter().all(|x| x.is_finite()));
}

/// hardware: cpu. `kv_snapshot` and `kv_restore` carry quantized rows: the next step's logits
/// are bitwise equal after a restore. The memory report counts the real buffer.
#[test]
fn cpu_quantized_kv_snapshot_roundtrip_and_memory() {
    let path = model_path();
    for kv in FORMATS {
        let o = LoadOptions {
            kv,
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        m.eval(&PROMPT).unwrap();
        let snap = m.kv_snapshot().unwrap();
        let first = m.eval(&[7]).unwrap();
        m.kv_restore(&snap, PROMPT.len()).unwrap();
        let again = m.eval(&[7]).unwrap();
        assert!(
            first
                .iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{kv}"
        );

        let desc = tetraeder_model::ModelFile::open(&path)
            .unwrap()
            .model_desc()
            .unwrap();
        let per_layer = desc.kv_bytes_per_token_per_layer_as(kv);
        let expect = u64::from(desc.layers) * m.n_ctx() as u64 * per_layer;
        let mem = m.memory();
        let u = mem.iter().find(|u| u.kind == "kv").unwrap();
        assert_eq!((u.capacity, u.used), (expect, expect), "{kv}");
        assert_eq!(u.format.as_deref(), Some(kv.to_string().as_str()));
        assert!(
            mem.iter()
                .filter(|u| u.kind != "kv")
                .all(|u| u.format.is_none())
        );
    }
}

/// hardware: cpu. A head size that is not a whole number of blocks, and a pair no kernel
/// supports, are load errors.
#[test]
fn cpu_bad_kv_formats_are_refused() {
    let err = Qwen3Moe::load(
        fixture(),
        &LoadOptions {
            kv: KvFormat::Q8_0,
            ..LoadOptions::new(CPU)
        },
    )
    .expect_err("head size 16 is not a multiple of 32");
    assert!(err.to_string().contains("block size"), "{err}");
    let err = Qwen3Moe::load(
        model_path(),
        &LoadOptions {
            kv: KvFormat::both(tetraeder_plan::Format::Q4_K),
            ..LoadOptions::new(CPU)
        },
    )
    .expect_err("Q4_K is not a KV format");
    assert!(err.to_string().contains("KV format"), "{err}");
}

/// Mean KL(p || q) over the vocabulary of two sets of logit rows, and the top-1 agreement.
fn mean_kl(p: &[f32], q: &[f32], n_vocab: usize) -> (f64, f64) {
    let lse = |x: &[f32]| {
        let mx = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        mx + x
            .iter()
            .map(|&v| (f64::from(v) - mx).exp())
            .sum::<f64>()
            .ln()
    };
    let (mut kl, mut agree, mut rows) = (0.0, 0usize, 0usize);
    for (a, b) in p.chunks(n_vocab).zip(q.chunks(n_vocab)) {
        let (la, lb) = (lse(a), lse(b));
        kl += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let lp = f64::from(x) - la;
                lp.exp() * (lp - (f64::from(y) - lb))
            })
            .sum::<f64>();
        agree += usize::from(argmax(a) == argmax(b));
        rows += 1;
    }
    (kl / rows as f64, agree as f64 / rows as f64)
}

/// hardware: cpu. Informative: KL of the quantized caches against F16 at 2K context on the
/// random-weight model (flat logits, so the numbers say little about a real model; the
/// acceptance is `kv_quality.rs`).
#[test]
fn cpu_quantized_kv_kl_at_2k() {
    let path = model_path();
    let tokens: Vec<i32> = (0..2048i32).map(|i| (i * 37 + 5) % 256).collect();
    let run = |kv| {
        let o = LoadOptions {
            kv,
            n_ctx: 2048,
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        m.eval_last(&tokens[..1792]).unwrap();
        (m.n_vocab(), m.eval(&tokens[1792..]).unwrap())
    };
    let (n_vocab, base) = run(KvFormat::F16);
    for kv in FORMATS {
        let (_, q) = run(kv);
        let (kl, top1) = mean_kl(&base, &q, n_vocab);
        eprintln!("KV {kv} vs f16 at 2K, last 256 positions: mean KL {kl:.6}, top-1 {top1:.4}");
        assert!(kl.is_finite() && kl >= 0.0, "{kv}: KL {kl}");
    }
}

//! Hadamard rotation of a quantized KV cache on the dense builder (ADR 0050 addendum, 0066):
//! the builder against libllama with its default rotation ON on the tiny dense fixture (head
//! size 64). This binary never turns libllama's rotation off (it is a process-wide setting; the
//! rotation-off parity lives in `dense_parity.rs`).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::*;
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3, SeqEval};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const FORMATS: [KvFormat; 3] = [KvFormat::Q8_0, KvFormat::Q8_0_Q4_0, KvFormat::Q4_0];

fn load(kv: KvFormat, n_seq_max: usize) -> Qwen3 {
    assert!(
        std::env::var("LLAMA_ATTN_ROT_DISABLE").map_or(true, |v| v != "1"),
        "LLAMA_ATTN_ROT_DISABLE must not be set for the rotation-on parity tests"
    );
    let o = DenseOptions {
        kv,
        n_seq_max,
        n_ctx_pool: 512,
        ..DenseOptions::new(CPU)
    };
    Qwen3::load(dense_fixture(), &o).unwrap()
}

/// hardware: cpu. The default follows libllama: a quantized cache is rotated on both sides, F16
/// is not, and the flag overrides both ways.
#[test]
fn cpu_rotation_default_follows_the_format() {
    assert_eq!(load(KvFormat::F16, 1).kv_rotated(), (false, false));
    for kv in FORMATS {
        assert_eq!(load(kv, 1).kv_rotated(), (true, true), "{kv}");
        let off = DenseOptions {
            kv,
            kv_rotation: Some(false),
            ..DenseOptions::new(CPU)
        };
        let m = Qwen3::load(dense_fixture(), &off).unwrap();
        assert_eq!(m.kv_rotated(), (false, false), "{kv}");
    }
    let on = DenseOptions {
        kv_rotation: Some(true),
        ..DenseOptions::new(CPU)
    };
    let m = Qwen3::load(dense_fixture(), &on).unwrap();
    assert_eq!(m.kv_rotated(), (true, true));
}

/// hardware: cpu. Prefill logits of every position and 12 greedy steps against libllama with
/// rotation on.
#[test]
fn cpu_rotated_cache_matches_libllama() {
    for kv in FORMATS {
        let r = libllama_reference_kv_default_rot(&dense_fixture(), &PROMPT, N_GREEDY, kv);
        let mut m = load(kv, 1);
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv} rotated: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: cpu. Two sequences in one step equal each alone with the rotation on.
#[test]
fn cpu_rotated_cache_batched_equals_alone() {
    let kv = KvFormat::Q8_0;
    let (a, b) = ([5, 6, 7, 8, 9], [50, 51, 52]);
    let mut m = load(kv, 2);
    let mut out = Vec::new();
    let input = |seq, tokens| SeqEval {
        seq,
        tokens,
        all_logits: true,
    };
    let spans = m.decode(&[input(0, &a), input(1, &b)], &mut out).unwrap();
    let nv = m.n_vocab();
    for (span, toks) in spans.into_iter().zip([&a[..], &b[..]]) {
        let want = load(kv, 1).eval(toks).unwrap();
        let worst = want
            .iter()
            .zip(&out[span.start * nv..span.end * nv])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("rotated batched vs alone: {worst:e}");
        assert!(worst <= TOL_CPU, "{worst:e}");
    }
}

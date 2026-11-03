//! `Engine::eval` and `Engine::truncate` on the tiny Qwen3 MoE fixture, CPU only.

use std::path::PathBuf;
use tetraeder_engine::{ContextConfig, Engine, Placement};

fn engine(n_batch: u32) -> Engine {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    let cfg = ContextConfig {
        n_ctx: 256,
        n_batch,
        n_ubatch: n_batch,
        threads: Some(2),
        ..ContextConfig::default()
    };
    Engine::load(path, &Placement::cpu(), &cfg, None).expect("fixture loads")
}

fn argmax(l: &[f32]) -> usize {
    l.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0
}

#[test]
fn truncate_then_reeval_reproduces_the_logits() {
    let mut e = engine(64);
    assert_eq!(e.n_past(), 0);
    e.eval(&[10, 20, 30, 40]).unwrap();
    assert_eq!(e.n_past(), 4);
    let first = e.eval(&[50]).unwrap().to_vec();
    assert_eq!(e.n_past(), 5);

    e.truncate(4).unwrap();
    assert_eq!(e.n_past(), 4);
    let again = e.eval(&[50]).unwrap().to_vec();
    assert_eq!(first, again);

    // Dropping more than the cache holds is a no-op; dropping to zero empties it.
    e.truncate(100).unwrap();
    assert_eq!(e.n_past(), 5);
    e.truncate(0).unwrap();
    assert_eq!(e.n_past(), 0);
    let cold = e.eval(&[10, 20, 30, 40, 50]).unwrap().to_vec();
    assert_eq!(argmax(&cold), argmax(&first));
}

#[test]
fn prefill_is_chunked_by_n_batch() {
    let toks: Vec<i32> = (1..=9).collect();
    let whole = engine(64).eval(&toks).unwrap().to_vec();
    let mut small = engine(4);
    let chunked = small.eval(&toks).unwrap().to_vec();
    assert_eq!(small.n_past(), 9);
    assert_eq!(argmax(&whole), argmax(&chunked));
    for (a, b) in whole.iter().zip(&chunked) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

#[test]
fn eval_rejects_empty_and_overlong_input() {
    let mut e = engine(64);
    assert!(e.eval(&[]).is_err());
    let too_long = vec![1; 300];
    assert!(e.eval(&too_long).is_err());
    assert_eq!(e.n_past(), 0);
}

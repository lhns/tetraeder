//! Engine tests on Qwen3-0.6B Q8_0, CPU only. Skipped without the model file.

mod common;

use common::{cpu_ctx, cpu_engine};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;
use tetraeder_engine::{
    BufferOverride, Engine, EngineError, GenerateRequest, Placement, Prompt, Sampling, StopReason,
};
use tetraeder_telemetry::{Phase, Telemetry, TelemetryConfig};

fn req(prompt: &str, n: u32) -> GenerateRequest {
    GenerateRequest::new(Prompt::Text(prompt.into()), n)
}

fn go() -> ControlFlow<()> {
    ControlFlow::Continue(())
}

#[test]
fn placement_errors_are_errors_not_panics() {
    // Fails in placement resolution, before any file is touched.
    let p = Placement {
        overrides: vec![BufferOverride::parse("x=NOPE").unwrap()],
        ..Placement::cpu()
    };
    let e = Engine::load("missing.gguf", &p, &cpu_ctx(), None)
        .err()
        .unwrap();
    assert!(matches!(e, EngineError::Placement(_)), "{e}");

    let e = Engine::load("missing.gguf", &Placement::cpu(), &cpu_ctx(), None)
        .err()
        .unwrap();
    assert!(matches!(e, EngineError::Llama(_)), "{e}");
}

#[test]
fn greedy_generation_is_deterministic_and_streams_the_text() {
    let path = require_model!();
    let mut eng = cpu_engine(&path, None);
    let mut pieces = String::new();
    let a = eng
        .generate(&req("The capital of France is", 12), |ev| {
            pieces.push_str(&ev.piece);
            go()
        })
        .unwrap();
    assert_eq!(pieces, a.text);
    assert_eq!(a.tokens.len(), 12);
    assert_eq!(a.stop_reason, StopReason::MaxTokens);
    assert_eq!(a.decode_steps, 11);
    assert!(a.ttft > Duration::ZERO && a.total >= a.ttft);

    let b = eng
        .generate(&req("The capital of France is", 12), |_| go())
        .unwrap();
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.text, b.text);
    assert!(!a.text.is_empty());
}

#[test]
fn callback_can_cancel() {
    let path = require_model!();
    let mut eng = cpu_engine(&path, None);
    let r = eng
        .generate(&req("Once upon a time", 20), |ev| {
            if ev.index == 2 {
                ControlFlow::Break(())
            } else {
                go()
            }
        })
        .unwrap();
    assert_eq!(r.tokens.len(), 3);
    assert_eq!(r.stop_reason, StopReason::Cancelled);
}

#[test]
fn top_logits_start_with_the_greedy_token() {
    let path = require_model!();
    let mut eng = cpu_engine(&path, None);
    let mut r = req("def fibonacci(n):", 4);
    r.top_k_logits = 5;
    let rep = eng.generate(&r, |_| go()).unwrap();
    assert_eq!(rep.top.len(), rep.tokens.len());
    for (t, top) in rep.tokens.iter().zip(&rep.top) {
        assert_eq!(top.len(), 5);
        assert_eq!(top[0].0, *t);
        assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));
    }
}

#[test]
fn seeded_random_sampling_repeats() {
    let path = require_model!();
    let mut eng = cpu_engine(&path, None);
    let mut r = req("Once upon a time", 8);
    r.sampling = Sampling::Random {
        seed: 7,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
    };
    let a = eng.generate(&r, |_| go()).unwrap();
    let b = eng.generate(&r, |_| go()).unwrap();
    assert_eq!(a.tokens, b.tokens);
}

#[test]
fn bad_requests_are_errors() {
    let path = require_model!();
    let mut eng = cpu_engine(&path, None);
    // The Qwen tokenizer adds no BOS, so an empty prompt has no tokens.
    let e = eng.generate(&req("", 4), |_| go()).unwrap_err();
    assert!(matches!(e, EngineError::EmptyPrompt), "{e}");
    let e = eng.generate(&req("hi", 100_000), |_| go()).unwrap_err();
    assert!(matches!(e, EngineError::ContextTooSmall { .. }), "{e}");
}

#[test]
fn telemetry_sees_prefill_and_decode_steps() {
    let path = require_model!();
    let tel = Arc::new(Telemetry::new(&TelemetryConfig::default()).unwrap());
    let mut eng = cpu_engine(&path, Some(Arc::clone(&tel)));
    let r = eng
        .generate(&req("The capital of France is", 6), |_| go())
        .unwrap();
    let steps = tel.ring().snapshot();
    let prefill: u32 = steps
        .iter()
        .filter(|s| s.phase == Phase::Prefill)
        .map(|s| s.tokens)
        .sum();
    let decode: Vec<_> = steps.iter().filter(|s| s.phase == Phase::Decode).collect();
    assert_eq!(prefill as usize, r.prompt_tokens.len());
    assert_eq!(decode.len(), r.decode_steps);
    assert_eq!(decode.len(), r.tokens.len() - 1);
    assert!(decode.iter().all(|s| s.tokens == 1 && !s.graphs));
    assert!(steps.iter().all(|s| s.duration > Duration::ZERO));
}

#[test]
fn bench_loops_run_and_repeat_after_reset() {
    let path = require_model!();
    let tel = Arc::new(Telemetry::new(&TelemetryConfig::default()).unwrap());
    let mut eng = cpu_engine(&path, Some(Arc::clone(&tel)));
    eng.bench_warmup(8, 1).unwrap();
    for _ in 0..2 {
        eng.bench_reset();
        let p = eng.bench_prompt(48).unwrap();
        let g = eng.bench_gen(4).unwrap();
        assert!(p > Duration::ZERO && g > Duration::ZERO);
    }
    // warmup: 1 prefill + 1 decode; per rep: 1 prefill + 4 decode.
    assert_eq!(tel.ring().stats().count, 2 + 2 * 5);
}

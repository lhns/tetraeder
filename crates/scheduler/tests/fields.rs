//! Group B of the llama-server parity matrix at the engine: log-probabilities, grammars, timings
//! and seed offsets on the tiny fixtures (ADR 0083). Pure CPU.
//!
//! Hardware label: cpu.

use std::path::PathBuf;
use tetraeder_engine::{ContextConfig, Placement};
use tetraeder_gateway::engine::{
    Engine, EngineError, FinishReason, GenEvent, GenerateRequest, GrammarSource, SamplingExtras,
    SamplingParams, TokenProbs,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::sample::{SampleParams, Sampler, SamplerEnv};
use tetraeder_scheduler::{Scheduler, SchedulerConfig, load_llama, load_moe};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn vocab_only() -> Model {
    Model::load(
        fixture(),
        &ModelParams {
            vocab_only: true,
            ..ModelParams::default()
        },
    )
    .unwrap()
}

fn logits(i: usize, n: usize) -> Vec<f32> {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 6.0
        })
        .collect()
}

/// The text of a token as llama-server writes it (control tokens as nothing).
fn piece(model: &Model, t: u32) -> String {
    String::from_utf8_lossy(&model.token_to_piece(t as i32, false).unwrap_or_default()).into_owned()
}

#[test]
fn grammar_keeps_every_sampled_token_inside_the_grammar() {
    let model = vocab_only();
    let n = model.n_vocab() as usize;
    let eog: Vec<u32> = (0..n as u32).filter(|&t| model.is_eog(t as i32)).collect();
    let p = SampleParams {
        grammar: Some("root ::= [a-f]{3}".into()),
        ..SampleParams::default()
    };
    let env = SamplerEnv {
        n_vocab: n,
        eog: &eog,
        model: Some(&model),
    };
    for seed in 0..5u32 {
        let mut s = Sampler::new(&p, Some(seed), &env).unwrap();
        let mut text = String::new();
        let mut ended = false;
        for i in 0..40 {
            let t = s
                .sample(&logits(i + seed as usize, n), None)
                .expect("a token");
            if eog.contains(&t) {
                ended = true;
                break;
            }
            text.push_str(&piece(&model, t));
        }
        // three letters from a to f, then only the end of generation is allowed
        assert!(ended, "seed {seed}: no end after {text:?}");
        assert!(
            text.len() == 3 && text.chars().all(|c| ('a'..='f').contains(&c)),
            "seed {seed}: {text:?}"
        );
    }
}

#[test]
fn grammar_composes_with_a_token_mask() {
    let model = vocab_only();
    let n = model.n_vocab() as usize;
    let p = SampleParams {
        grammar: Some("root ::= [a-z]+".into()),
        ..SampleParams::default()
    };
    let env = SamplerEnv {
        n_vocab: n,
        eog: &[],
        model: Some(&model),
    };
    // only tokens that start with a vowel pass the mask; the grammar then keeps them lowercase
    let allowed: Vec<u32> = (0..n as u32)
        .filter(|&t| piece(&model, t).starts_with(['a', 'e', 'i', 'o', 'u']))
        .collect();
    assert!(!allowed.is_empty());
    let mut words = vec![0u32; n.div_ceil(32)];
    for &a in &allowed {
        words[a as usize / 32] |= 1 << (a % 32);
    }
    let mask = tetraeder_gateway::engine::TokenMask { words };
    let mut s = Sampler::new(&p, Some(1), &env).unwrap();
    for i in 0..20 {
        let t = s.sample(&logits(i, n), Some(&mask)).expect("a token");
        assert!(allowed.contains(&t));
        assert!(piece(&model, t).chars().all(|c| c.is_ascii_lowercase()));
    }
}

#[test]
fn a_mask_the_grammar_rejects_gives_no_token_instead_of_aborting() {
    let model = vocab_only();
    let n = model.n_vocab() as usize;
    let p = SampleParams {
        grammar: Some("root ::= [a-z]+".into()),
        ..SampleParams::default()
    };
    let env = SamplerEnv {
        n_vocab: n,
        eog: &[],
        model: Some(&model),
    };
    // only tokens that start with a capital pass the mask, and the grammar wants lowercase
    let allowed: Vec<u32> = (0..n as u32)
        .filter(|&t| piece(&model, t).starts_with(|c: char| c.is_ascii_uppercase()))
        .collect();
    assert!(!allowed.is_empty());
    let mut words = vec![0u32; n.div_ceil(32)];
    for &a in &allowed {
        words[a as usize / 32] |= 1 << (a % 32);
    }
    let mask = tetraeder_gateway::engine::TokenMask { words };
    let mut s = Sampler::new(&p, Some(1), &env).unwrap();
    assert_eq!(s.sample(&logits(0, n), Some(&mask)), None);
}

#[test]
fn a_bad_grammar_is_an_error() {
    let model = vocab_only();
    let p = SampleParams {
        grammar: Some("root ::= \"ab".into()),
        ..SampleParams::default()
    };
    let env = SamplerEnv {
        n_vocab: model.n_vocab() as usize,
        eog: &[],
        model: Some(&model),
    };
    assert!(Sampler::new(&p, Some(1), &env).is_err());
}

fn start_llama() -> Scheduler {
    let ctx = ContextConfig {
        n_ctx: 256,
        n_batch: 64,
        n_ubatch: 64,
        threads: Some(2),
        ..ContextConfig::default()
    };
    let path = fixture();
    Scheduler::start(SchedulerConfig::new(fixture()), move || {
        load_llama(&path, &Placement::cpu(), &ctx, None)
    })
    .unwrap()
}

fn start_moe() -> Scheduler {
    let path = fixture();
    Scheduler::start(SchedulerConfig::new(fixture()), move || {
        let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
        o.n_ctx = 256;
        load_moe(&path, &o)
    })
    .unwrap()
}

struct Run {
    tokens: Vec<u32>,
    probs: Vec<TokenProbs>,
    /// The order of events: `P` for probabilities, `T` for a token.
    order: String,
    reason: FinishReason,
    stats: tetraeder_gateway::engine::EngineStats,
}

fn run(s: &Scheduler, params: SamplingParams) -> Run {
    let id = s.models()[0].id.clone();
    let prompt = s.tokenize(&id, "Hello world").unwrap();
    let mut rx = s
        .generate(GenerateRequest {
            model: id,
            prompt,
            params,
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .unwrap();
    let mut out = Run {
        tokens: vec![],
        probs: vec![],
        order: String::new(),
        reason: FinishReason::Cancelled,
        stats: Default::default(),
    };
    loop {
        match rx.blocking_recv().expect("stream ended") {
            GenEvent::Token(t) => {
                out.tokens.push(t);
                out.order.push('T');
            }
            GenEvent::Probs(p) => {
                out.probs.push(p);
                out.order.push('P');
            }
            GenEvent::Finished { reason, stats } => {
                out.reason = reason;
                out.stats = stats;
                return out;
            }
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("engine error: {e}"),
        }
    }
}

fn greedy(max_tokens: u32, extra: SamplingExtras) -> SamplingParams {
    SamplingParams {
        max_tokens,
        temperature: Some(0.0),
        extra,
        ..SamplingParams::default()
    }
}

#[test]
fn logprobs_come_before_each_token_on_both_backends() {
    for s in [start_llama(), start_moe()] {
        let extra = SamplingExtras {
            logprobs: Some(3),
            ..SamplingExtras::default()
        };
        let r = run(&s, greedy(6, extra));
        // every token is preceded by its probabilities; the end-of-generation token has only those
        let want: String = "PT".repeat(r.tokens.len());
        let want = if r.reason == FinishReason::EndOfGeneration {
            want + "P"
        } else {
            want
        };
        assert_eq!(r.order, want);
        for (t, p) in r.tokens.iter().zip(&r.probs) {
            assert_eq!(p.token, *t);
            assert_eq!(p.top.len(), 3);
            // greedy picks the most likely token, so it heads the list
            assert_eq!(p.top[0].0, *t);
            assert!((p.top[0].1 - p.logprob).abs() < 1e-6);
            assert!(p.top.windows(2).all(|w| w[0].1 >= w[1].1));
            assert!(p.logprob <= 0.0);
        }
        // off by default
        let r = run(&s, greedy(4, SamplingExtras::default()));
        assert!(r.probs.is_empty());
    }
}

#[test]
fn stats_carry_the_timings() {
    let s = start_llama();
    let r = run(&s, greedy(5, SamplingExtras::default()));
    assert!(r.stats.prompt_ms > 0.0, "{:?}", r.stats);
    assert!(r.stats.predicted_ms > 0.0, "{:?}", r.stats);
}

#[test]
fn a_grammar_request_follows_the_grammar_through_the_engine() {
    let s = start_llama();
    let id = s.models()[0].id.clone();
    assert!(matches!(
        s.check_grammar(&id, "root ::= \"ab"),
        Err(EngineError::Invalid(_))
    ));
    s.check_grammar(&id, "root ::= [a-f]{3}").unwrap();
    let extra = SamplingExtras {
        grammar: Some(GrammarSource::Gbnf("root ::= [a-f]{3}".into())),
        ..SamplingExtras::default()
    };
    let r = run(&s, greedy(12, extra));
    let model = vocab_only();
    let text: String = r.tokens.iter().map(|&t| piece(&model, t)).collect();
    assert_eq!(r.reason, FinishReason::EndOfGeneration, "{text:?}");
    assert!(
        text.len() == 3 && text.chars().all(|c| ('a'..='f').contains(&c)),
        "{text:?}"
    );
}

#[test]
fn json_schema_converts_through_the_engine() {
    let s = start_llama();
    let id = s.models()[0].id.clone();
    let g = s
        .json_schema_to_grammar(&id, r#"{"type": "object"}"#)
        .unwrap();
    assert!(g.contains("root ::="), "{g}");
    s.check_grammar(&id, &g).unwrap();
    assert!(matches!(
        s.json_schema_to_grammar(&id, "{nope"),
        Err(EngineError::Invalid(_))
    ));
}

#[test]
fn seed_offsets_give_different_draws_from_one_seed() {
    let s = start_llama();
    let at = |offset: u32| {
        let p = SamplingParams {
            max_tokens: 12,
            temperature: Some(2.0),
            seed: Some(7),
            extra: SamplingExtras {
                seed_offset: offset,
                ..SamplingExtras::default()
            },
            ..SamplingParams::default()
        };
        run(&s, p).tokens
    };
    assert_eq!(at(0), at(0));
    assert_eq!(at(1), at(1));
    assert_ne!(at(0), at(1));
}

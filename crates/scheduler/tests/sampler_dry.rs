//! DRY needs the model's vocabulary: the scheduler's sampler against a hand-built libllama
//! chain, and a DRY request through both scheduler backends. Tiny fixture, no GPU.
//!
//! Hardware label: cpu.

use std::path::PathBuf;
use tetraeder_engine::ContextConfig;
use tetraeder_engine::Placement;
use tetraeder_gateway::engine::{
    Engine, GenEvent, GenerateRequest, SamplingExtras, SamplingParams,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams, Sampler as Chain};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::sample::{SampleParams, Sampler, SamplerEnv};
use tetraeder_scheduler::{Scheduler, SchedulerConfig, load_llama, load_moe};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn row(i: usize, n: usize) -> Vec<f32> {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0
        })
        .collect()
}

#[test]
fn dry_matches_a_hand_built_chain() {
    let model = Model::load(
        fixture(),
        &ModelParams {
            vocab_only: true,
            ..ModelParams::default()
        },
    )
    .unwrap();
    let n = model.n_vocab() as usize;
    let p = SampleParams {
        dry_multiplier: 0.8,
        dry_allowed_length: 1,
        ..SampleParams::default()
    };
    let env = SamplerEnv {
        n_vocab: n,
        eog: &[],
        model: Some(&model),
    };
    let seed = 99;
    let mut ours = Sampler::new(&p, Some(seed), &env).unwrap();
    let mut by_hand = Chain::chain();
    by_hand = by_hand
        .dry(
            &model,
            0.8,
            1.75,
            1,
            64,
            &["\n", ":", "\"", "*"].map(String::from),
        )
        .unwrap()
        .top_k(40)
        .top_p(0.95, 0)
        .min_p(0.05, 0)
        .temperature_ext(0.8, 0.0, 1.0)
        .dist(seed);
    // A repetitive history, so DRY has sequences to punish.
    for t in [3, 4, 5, 3, 4, 5, 3, 4] {
        ours.accept_all(&[t]);
        by_hand.accept(t);
    }
    for i in 0..100 {
        let l = row(i % 7, n);
        let a = ours.sample(&l, None).unwrap();
        let b = by_hand.sample_logits(&l).unwrap() as u32;
        assert_eq!(a, b, "draw {i}");
    }
}

#[test]
fn dry_request_runs_on_both_backends() {
    let p = fixture();
    let ctx = ContextConfig {
        n_ctx: 256,
        n_batch: 64,
        n_ubatch: 64,
        threads: Some(2),
        ..ContextConfig::default()
    };
    let path = p.clone();
    let llama = Scheduler::start(SchedulerConfig::new(p.clone()), move || {
        load_llama(&path, &Placement::cpu(), &ctx, None)
    })
    .unwrap();
    let path = p.clone();
    let moe = Scheduler::start(SchedulerConfig::new(p), move || {
        let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
        o.n_ctx = 256;
        load_moe(&path, &o)
    })
    .unwrap();
    for s in [llama, moe] {
        let id = s.models()[0].id.clone();
        let prompt = s.tokenize(&id, "Hello world").unwrap();
        let mut rx = s
            .generate(GenerateRequest {
                model: id,
                prompt,
                params: SamplingParams {
                    max_tokens: 8,
                    seed: Some(5),
                    extra: SamplingExtras {
                        dry_multiplier: Some(0.8),
                        ..SamplingExtras::default()
                    },
                    ..SamplingParams::default()
                },
                constraint: None,
                cancel: CancellationToken::new(),
            })
            .unwrap();
        loop {
            match rx.blocking_recv().expect("stream ended") {
                GenEvent::Token(_) => {}
                GenEvent::Finished { .. } => break,
                GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("engine error: {e}"),
            }
        }
    }
}

//! After the backend restarts (the worker of `serve --worker`), idle sequences are lost. A new
//! request that would have reused one recomputes instead of failing (Phase 5, E9 follow-up).
//!
//! Hardware label: none (the tokenizer comes from the tiny fixture).

mod fake;

use fake::{FakeKv, Probe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{Loaded, Scheduler, SchedulerConfig};
use tokio_util::sync::CancellationToken;

fn start() -> (Scheduler, Arc<Probe>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    let sc = SchedulerConfig::new(&path);
    let cell: Arc<std::sync::OnceLock<Arc<Probe>>> = Arc::default();
    let c = Arc::clone(&cell);
    let s = Scheduler::start(sc, move || {
        let tokenizer = Arc::new(
            Model::load(
                &path,
                &ModelParams {
                    vocab_only: true,
                    ..ModelParams::default()
                },
            )
            .map_err(|e| e.to_string())?,
        );
        let kv = FakeKv::new(tokenizer.n_vocab() as usize, 2, 512, 1024);
        let _ = c.set(kv.probe());
        Ok(Loaded {
            backend: Box::new(kv),
            tokenizer,
        })
    })
    .expect("scheduler over FakeKv");
    (s, cell.get().cloned().expect("probe"))
}

/// Runs one greedy request; returns its tokens and cached prompt tokens, or the error.
fn run(s: &Scheduler, prompt: Vec<u32>) -> Result<(Vec<u32>, u32), String> {
    let mut rx = s
        .generate(GenerateRequest {
            model: s.models()[0].id.clone(),
            prompt,
            params: SamplingParams {
                max_tokens: 4,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .expect("generate");
    let mut tokens = Vec::new();
    while let Some(ev) = rx.blocking_recv() {
        match ev {
            GenEvent::Token(t) => tokens.push(t),
            GenEvent::Finished { stats, .. } => return Ok((tokens, stats.cached_tokens)),
            GenEvent::Probs(_) => {}
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => return Err(e),
        }
    }
    panic!("closed without a terminal event");
}

#[test]
fn a_request_after_a_restart_recomputes_instead_of_reusing_a_lost_idle_sequence() {
    let prompt: Vec<u32> = (0..40).map(|i| 10 + (i * 7) % 150).collect();
    let mut longer = prompt.clone();
    longer.extend([11, 12, 13]);

    let (fresh, _) = start();
    let (want, _) = run(&fresh, longer.clone()).expect("fresh run");

    let (s, probe) = start();
    run(&s, prompt).expect("first request");
    probe.restart.store(true, Ordering::SeqCst);
    let (got, cached) = run(&s, longer).expect("the request after the restart must not fail");
    assert_eq!(cached, 0, "nothing may be reused from a lost sequence");
    assert_eq!(got, want, "recomputed tokens equal a fresh run");
}

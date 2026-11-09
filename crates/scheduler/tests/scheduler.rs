//! The scheduler on the tiny Qwen3 MoE fixture, CPU only, through both backends.
//!
//! Hardware label: cpu.

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tetraeder_engine::{ContextConfig, Engine as LlamaEngine, GenerateRequest as LlamaRequest};
use tetraeder_engine::{Placement, Prompt};
use tetraeder_gateway::engine::{
    Engine, EngineStats, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenConstraint,
    TokenMask,
};
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::{Scheduler, SchedulerConfig, load_llama, load_moe};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

const CTX: u32 = 256;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn ctx_cfg() -> ContextConfig {
    ContextConfig {
        n_ctx: CTX,
        n_batch: 64,
        n_ubatch: 64,
        threads: Some(2),
        ..ContextConfig::default()
    }
}

fn llama_sched() -> Scheduler {
    let p = fixture();
    let path = p.clone();
    Scheduler::start(SchedulerConfig::new(p), move || {
        load_llama(&path, &Placement::cpu(), &ctx_cfg(), None)
    })
    .expect("llama scheduler")
}

fn moe_sched() -> Scheduler {
    let p = fixture();
    let path = p.clone();
    Scheduler::start(SchedulerConfig::new(p), move || {
        let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
        o.n_ctx = CTX as usize;
        load_moe(&path, &o)
    })
    .expect("moe scheduler")
}

fn greedy(max: u32) -> SamplingParams {
    SamplingParams {
        max_tokens: max,
        temperature: Some(0.0),
        ..SamplingParams::default()
    }
}

struct Out {
    tokens: Vec<u32>,
    reason: FinishReason,
    stats: EngineStats,
}

fn start(
    s: &Scheduler,
    prompt: &[u32],
    params: SamplingParams,
    constraint: Option<Box<dyn TokenConstraint>>,
    cancel: CancellationToken,
) -> tokio::sync::mpsc::Receiver<GenEvent> {
    let id = s.models()[0].id.clone();
    s.generate(GenerateRequest {
        model: id,
        prompt: prompt.to_vec(),
        params,
        constraint,
        cancel,
    })
    .unwrap()
}

fn drain(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Out {
    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().expect("the worker ended the stream") {
            GenEvent::Token(t) => tokens.push(t),
            GenEvent::Finished { reason, stats } => {
                return Out {
                    tokens,
                    reason,
                    stats,
                };
            }
            GenEvent::Error(e) => panic!("engine error: {e}"),
        }
    }
}

fn run(s: &Scheduler, prompt: &[u32], max: u32) -> Out {
    drain(start(
        s,
        prompt,
        greedy(max),
        None,
        CancellationToken::new(),
    ))
}

fn prompt(s: &Scheduler, text: &str) -> Vec<u32> {
    let id = s.models()[0].id.clone();
    let t = s.tokenize(&id, text).unwrap();
    assert!(t.len() >= 2, "{text:?} -> {t:?}");
    t
}

#[test]
fn model_info_and_vocab_come_from_libllama() {
    for s in [llama_sched(), moe_sched()] {
        let m = &s.models()[0];
        assert_eq!(m.id, "tiny-qwen3moe-f16");
        assert_eq!(m.context_length, CTX);
        let v = s.vocab(&m.id).unwrap();
        assert!(v.len() >= 256);
        let a = s.tokenize(&m.id, "A").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(v.bytes(a[0]), b"A");
        assert!(s.vocab("other").is_err());
        assert!(s.tokenize("other", "x").is_err());
    }
}

#[test]
fn greedy_sequence_equals_engine_generate() {
    let s = llama_sched();
    let p = prompt(&s, "Hello, tiny world");
    let got = run(&s, &p, 12);
    assert_eq!(got.stats.prompt_tokens as usize, p.len());
    assert_eq!(got.stats.cached_tokens, 0);

    let mut eng = LlamaEngine::load(fixture(), &Placement::cpu(), &ctx_cfg(), None).unwrap();
    let toks: Vec<i32> = p.iter().map(|&t| t as i32).collect();
    let mut req = LlamaRequest::new(Prompt::Tokens(toks), 12);
    req.add_special = false;
    let want = eng
        .generate(&req, |_| std::ops::ControlFlow::Continue(()))
        .unwrap();
    let want: Vec<u32> = want.tokens.iter().map(|&t| t as u32).collect();
    assert_eq!(got.tokens, want);
}

#[test]
fn moe_backend_agrees_with_libllama_backend() {
    let (l, m) = (llama_sched(), moe_sched());
    let p = prompt(&l, "Hello, tiny world");
    let a = run(&l, &p, 16);
    let b = run(&m, &p, 16);
    assert_eq!(a.tokens, b.tokens);
    assert!(a.stats.bytes_streamed.is_none());
    assert!(b.stats.gpu_seconds.is_none() && a.stats.gpu_seconds.is_none());
}

#[test]
fn shared_prefix_is_reused_and_output_is_unchanged() {
    for make in [llama_sched as fn() -> Scheduler, moe_sched] {
        let s = make();
        let p1 = prompt(&s, "Hello, tiny world");
        let first = run(&s, &p1, 6);
        assert_eq!(first.stats.cached_tokens, 0);

        // Same prompt again: everything but the last token is cached.
        let again = run(&s, &p1, 6);
        assert_eq!(again.stats.cached_tokens as usize, p1.len() - 1);
        assert_eq!(again.tokens, first.tokens);

        // A conversation turn: the old prompt and its answer, then more text.
        let mut p2 = p1.clone();
        p2.extend(&first.tokens);
        p2.extend(prompt(&s, "more text"));
        let warm = run(&s, &p2, 6);
        assert!(
            warm.stats.cached_tokens as usize > p1.len(),
            "cached {} of {}",
            warm.stats.cached_tokens,
            p2.len()
        );
        let cold = run(&make(), &p2, 6);
        assert_eq!(cold.stats.cached_tokens, 0);
        assert_eq!(warm.tokens, cold.tokens);

        // A different prompt drops the cache from the first difference.
        let other = run(&s, &prompt(&s, "Zebra crossing"), 4);
        assert!(other.stats.cached_tokens < 3);
    }
}

struct Only(Vec<u32>, usize);

impl TokenConstraint for Only {
    fn mask(&mut self) -> Result<Option<TokenMask>, String> {
        let mut words = vec![0u32; 8];
        for &t in &self.0 {
            words[t as usize / 32] |= 1 << (t % 32);
        }
        Ok(Some(TokenMask { words }))
    }

    fn commit(&mut self, token: u32) -> Result<(), String> {
        assert!(self.0.contains(&token));
        self.1 += 1;
        Ok(())
    }
}

#[test]
fn constraint_mask_is_honoured() {
    for s in [llama_sched(), moe_sched()] {
        let p = prompt(&s, "Hello");
        for temperature in [0.0, 1.0] {
            let out = drain(start(
                &s,
                &p,
                SamplingParams {
                    max_tokens: 12,
                    temperature: Some(temperature),
                    seed: Some(9),
                    ..SamplingParams::default()
                },
                Some(Box::new(Only(vec![65, 66, 67], 0))),
                CancellationToken::new(),
            ));
            assert_eq!(out.tokens.len(), 12);
            assert!(out.tokens.iter().all(|t| [65, 66, 67].contains(t)));
            assert_eq!(out.reason, FinishReason::Length);
        }
    }
}

#[test]
fn cancellation_stops_generation_and_the_worker_goes_on() {
    for s in [llama_sched(), moe_sched()] {
        let p = prompt(&s, "Hello");

        // Cancelled before it starts.
        let c = CancellationToken::new();
        c.cancel();
        let out = drain(start(&s, &p, greedy(5), None, c));
        assert_eq!(out.reason, FinishReason::Cancelled);
        assert!(out.tokens.is_empty());

        // Cancelled while running.
        let c = CancellationToken::new();
        let mut rx = start(&s, &p, greedy(200), None, c.clone());
        for _ in 0..2 {
            assert!(matches!(rx.blocking_recv(), Some(GenEvent::Token(_))));
        }
        c.cancel();
        let mut n = 2;
        let reason = loop {
            match rx.blocking_recv().unwrap() {
                GenEvent::Token(_) => n += 1,
                GenEvent::Finished { reason, stats } => {
                    assert!(stats.generated_tokens < 100);
                    break reason;
                }
                GenEvent::Error(e) => panic!("{e}"),
            }
        };
        assert_eq!(reason, FinishReason::Cancelled);
        assert!(n < 200);

        let next = run(&s, &p, 4);
        assert_eq!(next.reason, FinishReason::Length);
        assert_eq!(next.tokens.len(), 4);
    }
}

#[test]
fn dropped_receiver_stops_generation() {
    for s in [llama_sched(), moe_sched()] {
        let p = prompt(&s, "Hello");
        let mut rx = start(&s, &p, greedy(200), None, CancellationToken::new());
        assert!(matches!(rx.blocking_recv(), Some(GenEvent::Token(_))));
        drop(rx);
        let next = run(&s, &p, 4);
        assert_eq!(next.tokens.len(), 4);
    }
}

#[test]
fn context_limit_ends_with_length() {
    let s = moe_sched();
    let p = prompt(&s, "Hello");
    // The mask keeps the fixture from sampling its end-of-generation token.
    let out = drain(start(
        &s,
        &p,
        greedy(10_000),
        Some(Box::new(Only(vec![65, 66, 67], 0))),
        CancellationToken::new(),
    ));
    assert_eq!(out.reason, FinishReason::Length);
    // The last token is sampled but never evaluated, so it needs no KV slot.
    assert_eq!(p.len() + out.tokens.len(), CTX as usize + 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gateway_serves_completions_over_the_scheduler() {
    let s = moe_sched();
    let id = s.models()[0].id.clone();
    let gw = Arc::new(Gateway::new(Arc::new(s), GatewayConfig::default()));
    let app = tetraeder_gateway::server::router(gw);
    let body =
        serde_json::json!({"model": id, "prompt": "Hello", "max_tokens": 5, "temperature": 0});
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(120), app.oneshot(req))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(resp.status(), 200);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["model"], id.as_str());
    assert_eq!(v["usage"]["completion_tokens"], 5, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

// ---- prefix cache in host RAM (ADR 0049) ----

type Make = fn(&dyn Fn(&mut SchedulerConfig)) -> Scheduler;

fn llama_with(f: &dyn Fn(&mut SchedulerConfig)) -> Scheduler {
    let p = fixture();
    let path = p.clone();
    let mut cfg = SchedulerConfig::new(p);
    f(&mut cfg);
    Scheduler::start(cfg, move || {
        load_llama(&path, &Placement::cpu(), &ctx_cfg(), None)
    })
    .expect("llama scheduler")
}

fn moe_with(f: &dyn Fn(&mut SchedulerConfig)) -> Scheduler {
    let p = fixture();
    let path = p.clone();
    let mut cfg = SchedulerConfig::new(p);
    f(&mut cfg);
    Scheduler::start(cfg, move || {
        let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
        o.n_ctx = CTX as usize;
        load_moe(&path, &o)
    })
    .expect("moe scheduler")
}

/// Small thresholds so prompts that fit the 256-token fixture context cross them.
fn small_cache(bytes: usize) -> impl Fn(&mut SchedulerConfig) {
    move |c| {
        c.cache_ram_bytes = bytes;
        c.cache_save_min = 32;
        c.cache_restore_gain = 16;
    }
}

/// `n` tokens of text that starts with `lead`, so prompts with different leads share no prefix.
fn long_prompt(s: &Scheduler, lead: &str, n: usize) -> Vec<u32> {
    let text: String = (0..n).map(|i| format!("{lead}{i} ")).collect();
    let mut t = prompt(s, &text);
    assert!(t.len() >= n, "{} tokens", t.len());
    t.truncate(n);
    t
}

const MEG: usize = 1 << 20;

#[test]
fn ram_restore_serves_the_evicted_prefix_and_matches_a_cold_run() {
    for make in [llama_with as Make, moe_with] {
        let s = make(&small_cache(64 * MEG));
        let a = long_prompt(&s, "alpha", 100);
        let b = long_prompt(&s, "zulu", 100);
        run(&s, &a, 6);
        let other = run(&s, &b, 6);
        assert!(other.stats.cached_tokens < 16);
        let mut a2 = a.clone();
        a2.extend(prompt(&s, "and some more words"));
        let warm = run(&s, &a2, 8);
        assert!(
            warm.stats.cached_tokens as usize >= a.len() - 1,
            "cached {} of {}",
            warm.stats.cached_tokens,
            a2.len()
        );
        let cold = run(&make(&small_cache(0)), &a2, 8);
        assert_eq!(cold.stats.cached_tokens, 0);
        assert_eq!(warm.tokens, cold.tokens);
    }
}

#[test]
fn a_zero_budget_disables_the_cache() {
    for make in [llama_with as Make, moe_with] {
        let s = make(&small_cache(0));
        let a = long_prompt(&s, "alpha", 100);
        run(&s, &a, 4);
        run(&s, &long_prompt(&s, "zulu", 100), 4);
        let mut a2 = a.clone();
        a2.extend(prompt(&s, "and some more words"));
        assert!(run(&s, &a2, 4).stats.cached_tokens < 16);
    }
}

#[test]
fn least_recently_used_sequences_are_evicted() {
    for (i, make) in [llama_with as Make, moe_with].into_iter().enumerate() {
        // Size one entry from a snapshot, then allow one and a half.
        let probe = make(&small_cache(0));
        let a = long_prompt(&probe, "alpha", 100);
        let size = snapshot_size(i, &a);
        let s = make(&small_cache(size * 3 / 2));
        let (a, b, c) = (
            long_prompt(&s, "alpha", 100),
            long_prompt(&s, "zulu", 100),
            long_prompt(&s, "mike", 100),
        );
        run(&s, &a, 4);
        run(&s, &b, 4); // saves A
        run(&s, &c, 4); // saves B, which evicts A
        let mut a2 = a.clone();
        a2.extend(prompt(&s, "tail one"));
        assert!(run(&s, &a2, 4).stats.cached_tokens < 16, "A was kept");
        // Running A2 saved C and evicted B.
        let mut c2 = c.clone();
        c2.extend(prompt(&s, "tail two"));
        let hit = run(&s, &c2, 4);
        assert!(
            hit.stats.cached_tokens as usize >= c.len() - 1,
            "backend {i}: cached {} of {}, entry size {size}",
            hit.stats.cached_tokens,
            c2.len()
        );
    }
}

fn snapshot_size(backend: usize, prompt: &[u32]) -> usize {
    let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut b = backends().remove(backend);
    b.eval(&toks).unwrap();
    b.snapshot().unwrap().len()
}

fn backends() -> Vec<Box<dyn tetraeder_scheduler::Backend>> {
    let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
    o.n_ctx = CTX as usize;
    vec![
        load_llama(&fixture(), &Placement::cpu(), &ctx_cfg(), None)
            .unwrap()
            .backend,
        load_moe(&fixture(), &o).unwrap().backend,
    ]
}

#[test]
fn snapshot_and_restore_round_trip_gives_identical_logits() {
    let toks: Vec<i32> = (1..=60).collect();
    for i in 0..2 {
        let mut x = backends().remove(i);
        x.eval(&toks).unwrap();
        let snap = x.snapshot().unwrap();
        let want = x.eval(&[7]).unwrap().to_vec();
        // Same backend after a rollback, and a fresh one.
        x.restore(&snap, toks.len()).unwrap();
        assert_eq!(x.n_past(), toks.len());
        assert_eq!(x.eval(&[7]).unwrap(), &want[..]);
        let mut y = backends().remove(i);
        y.restore(&snap, toks.len()).unwrap();
        assert_eq!(y.eval(&[7]).unwrap(), &want[..]);
    }
}

/// `decode_profile` on: every decode step is counted, the stages are timed and the backend's
/// counters change over the decode (the builder keeps its step graph, libllama reuses its own).
#[test]
fn decode_profile_reports_steps_stages_and_graph_reuse() {
    let p = fixture();
    let llama = {
        let path = p.clone();
        let mut cfg = SchedulerConfig::new(p.clone());
        cfg.decode_profile = true;
        Scheduler::start(cfg, move || {
            load_llama(&path, &Placement::cpu(), &ctx_cfg(), None)
        })
        .unwrap()
    };
    let moe = {
        let path = p.clone();
        let mut cfg = SchedulerConfig::new(p.clone());
        cfg.decode_profile = true;
        Scheduler::start(cfg, move || {
            let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
            o.n_ctx = CTX as usize;
            load_moe(&path, &o)
        })
        .unwrap()
    };
    for (s, reuse_key) in [(&llama, "n_reused"), (&moe, "steps")] {
        let pr = prompt(s, "Once upon a time");
        let out = run(s, &pr, 8);
        let prof = out.stats.profile.expect("profile on");
        // The first token comes from the prefill step and each later one from a decode step; at
        // the length limit no further step runs, an end-of-generation token takes one more.
        let n = out.stats.generated_tokens;
        let want = if out.reason == FinishReason::Length {
            n - 1
        } else {
            n
        };
        assert_eq!(prof.decode_steps, want, "{:?}", out.reason);
        assert!(n >= 3, "too short to check: {n} tokens");
        assert!(prof.eval_s > 0.0 && prof.sample_s > 0.0 && prof.prefill_s > 0.0);
        let get = |k: &str| prof.backend.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
        assert!(
            get(reuse_key).is_some_and(|v| v > 0.0),
            "{:?}",
            prof.backend
        );
    }
    // Off by default (unless the environment turns it on).
    if std::env::var_os("TETRAEDER_DECODE_PROFILE").is_none() {
        let s = llama_sched();
        let pr = prompt(&s, "Once upon a time");
        assert!(run(&s, &pr, 3).stats.profile.is_none());
    }
}

/// The builder backend reads each step's logits into one kept buffer: decode allocates no
/// 600 KB vector per token, and the values equal a fresh `eval_last`.
#[test]
fn moe_backend_reuses_its_logits_buffer() {
    let mut b = backends().remove(1);
    let toks: Vec<i32> = (1..=12).collect();
    let first = b.eval(&toks).unwrap().as_ptr();
    let a = b.eval(&[5]).unwrap().to_vec();
    let again = b.eval(&[6]).unwrap().as_ptr();
    assert_eq!(first, again, "the logits buffer moved");
    let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
    o.n_ctx = CTX as usize;
    let mut m = tetraeder_model::Qwen3Moe::load(fixture(), &o).unwrap();
    m.eval_last(&toks).unwrap();
    assert_eq!(m.eval_last(&[5]).unwrap(), a);
}

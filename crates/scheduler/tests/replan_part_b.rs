//! ADR 0055 part B on the tiny fixture, CPU only: libllama op offload, the builder prefill chunk
//! and the worker join.
//!
//! Hardware label: cpu.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tetraeder_engine::{ContextConfig, Placement};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::{
    Backend, Loaded, Scheduler, SchedulerConfig, load_llama, load_moe_chunked,
};
use tokio_util::sync::CancellationToken;

const CTX: u32 = 256;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn greedy_tokens(s: &Scheduler, text: &str, max: u32) -> Vec<u32> {
    let id = s.models()[0].id.clone();
    let prompt = s.tokenize(&id, text).unwrap();
    let mut rx = s
        .generate(GenerateRequest {
            model: id,
            prompt,
            params: SamplingParams {
                max_tokens: max,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .unwrap();
    let mut out = Vec::new();
    loop {
        match rx.blocking_recv().expect("stream ended") {
            GenEvent::Token(t) => out.push(t),
            GenEvent::Finished { .. } => return out,
            GenEvent::Error(e) => panic!("engine error: {e}"),
        }
    }
}

#[test]
fn llama_greedy_output_is_the_same_with_op_offload_off() {
    let run = |op_offload: bool| {
        let path = fixture();
        let cfg = ContextConfig {
            n_ctx: CTX,
            n_batch: 64,
            n_ubatch: 64,
            threads: Some(2),
            op_offload,
            ..ContextConfig::default()
        };
        let s = Scheduler::start(SchedulerConfig::new(&path), {
            let path = path.clone();
            move || load_llama(&path, &Placement::cpu(), &cfg, None)
        })
        .expect("llama scheduler");
        greedy_tokens(&s, "Hello world, hello world", 12)
    };
    let (on, off) = (run(true), run(false));
    assert!(!on.is_empty());
    assert_eq!(on, off);
}

#[test]
fn builder_greedy_output_is_the_same_for_chunk_64_and_512() {
    let run = |chunk: usize| {
        let path = fixture();
        let mut cfg = SchedulerConfig::new(&path);
        cfg.prefill_chunk = chunk;
        let s = Scheduler::start(cfg, move || {
            let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
            o.n_ctx = CTX as usize;
            load_moe_chunked(&path, &o, chunk)
        })
        .expect("builder scheduler");
        // Long enough to need several 64-token chunks.
        let text = "the quick brown fox jumps over the lazy dog ".repeat(4);
        greedy_tokens(&s, &text, 8)
    };
    let (small, large) = (run(64), run(512));
    assert!(!small.is_empty());
    assert_eq!(small, large);
}

/// A backend that records the size of every `eval` call and sets a flag when dropped.
struct Fake {
    logits: Vec<f32>,
    calls: Arc<Mutex<Vec<usize>>>,
    dropped: Arc<AtomicBool>,
    n_past: Arc<AtomicUsize>,
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl Backend for Fake {
    fn n_ctx(&self) -> usize {
        CTX as usize
    }
    fn n_past(&self) -> usize {
        self.n_past.load(Ordering::SeqCst)
    }
    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        self.n_past.store(keep, Ordering::SeqCst);
        Ok(())
    }
    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        self.calls.lock().unwrap().push(tokens.len());
        self.n_past.fetch_add(tokens.len(), Ordering::SeqCst);
        Ok(&self.logits)
    }
    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        Ok(Vec::new())
    }
    fn restore(&mut self, _: &[u8], n: usize) -> Result<(), String> {
        self.n_past.store(n, Ordering::SeqCst);
        Ok(())
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
}

struct Rig {
    sched: Scheduler,
    calls: Arc<Mutex<Vec<usize>>>,
    dropped: Arc<AtomicBool>,
}

fn fake_sched(chunk: Option<usize>) -> Rig {
    let path = fixture();
    let calls: Arc<Mutex<Vec<usize>>> = Arc::default();
    let dropped = Arc::new(AtomicBool::new(false));
    let mut cfg = SchedulerConfig::new(&path);
    if let Some(c) = chunk {
        cfg.prefill_chunk = c;
    }
    let (c, d) = (Arc::clone(&calls), Arc::clone(&dropped));
    let sched = Scheduler::start(cfg, move || {
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
        let mut logits = vec![0.0; tokenizer.n_vocab() as usize];
        logits[65] = 1.0;
        Ok(Loaded {
            backend: Box::new(Fake {
                logits,
                calls: c,
                dropped: d,
                n_past: Arc::default(),
            }),
            tokenizer,
        })
    })
    .expect("fake scheduler");
    Rig {
        sched,
        calls,
        dropped,
    }
}

#[test]
fn the_worker_evaluates_at_most_the_prefill_chunk() {
    let r = fake_sched(Some(64));
    let text = "abcdefghij".repeat(15);
    greedy_tokens(&r.sched, &text, 2);
    let calls = r.calls.lock().unwrap().clone();
    assert!(calls.iter().any(|&n| n > 1), "{calls:?}");
    assert!(calls.iter().all(|&n| n <= 64), "{calls:?}");
    let total: usize = calls.iter().sum();
    assert!(total >= 100, "{calls:?}");
}

#[test]
fn shutdown_returns_after_the_backend_is_dropped() {
    let r = fake_sched(None);
    greedy_tokens(&r.sched, "Hello", 2);
    assert!(!r.dropped.load(Ordering::SeqCst));
    r.sched.shutdown().unwrap();
    assert!(r.dropped.load(Ordering::SeqCst));
}

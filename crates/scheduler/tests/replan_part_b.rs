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
    BackendError, BatchBackend, Loaded, Outputs, Scheduler, SchedulerConfig, SeqId, SeqInput,
    load_llama, load_moe_chunked,
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
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("engine error: {e}"),
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
        cfg.step_tokens = chunk;
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

/// A backend that records the size of every input of a `decode` and sets a flag when dropped.
struct Fake {
    logits: Vec<f32>,
    rows: Vec<f32>,
    spans: Vec<std::ops::Range<usize>>,
    calls: Arc<Mutex<Vec<usize>>>,
    dropped: Arc<AtomicBool>,
    n_past: Arc<AtomicUsize>,
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl BatchBackend for Fake {
    fn n_vocab(&self) -> usize {
        self.logits.len()
    }
    fn n_seq_max(&self) -> usize {
        1
    }
    fn n_ctx_seq(&self) -> usize {
        CTX as usize
    }
    fn kv_capacity(&self) -> usize {
        CTX as usize
    }
    fn kv_free(&self) -> usize {
        CTX as usize - self.n_past.load(Ordering::SeqCst)
    }
    fn seq_len(&self, seq: SeqId) -> usize {
        if seq == 0 {
            self.n_past.load(Ordering::SeqCst)
        } else {
            0
        }
    }
    fn seq_reserve(&mut self, _: SeqId, n: usize) -> Result<(), BackendError> {
        if n > CTX as usize {
            return Err(BackendError::KvFull {
                seq: 0,
                needed: n,
                free: self.kv_free(),
            });
        }
        Ok(())
    }
    fn seq_rm(&mut self, _: SeqId, keep: usize) -> Result<(), BackendError> {
        self.n_past
            .store(keep.min(self.seq_len(0)), Ordering::SeqCst);
        Ok(())
    }
    fn seq_cp(&mut self, _: SeqId, _: SeqId, _: usize) -> Result<(), BackendError> {
        Err(BackendError::Failed("single sequence".into()))
    }
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.rows.clear();
        self.spans.clear();
        for inp in inputs {
            self.calls.lock().unwrap().push(inp.tokens.len());
            self.n_past.fetch_add(inp.tokens.len(), Ordering::SeqCst);
            let n = if inp.all_logits { inp.tokens.len() } else { 1 };
            let from = self.rows.len() / self.logits.len();
            for _ in 0..n {
                self.rows.extend_from_slice(&self.logits);
            }
            self.spans.push(from..from + n);
        }
        Ok(Outputs::new(self.logits.len(), &self.rows, &self.spans))
    }
    fn seq_snapshot(&mut self, _: SeqId) -> Result<Vec<u8>, BackendError> {
        Ok(Vec::new())
    }
    fn seq_restore(&mut self, _: SeqId, _: &[u8], n: usize) -> Result<(), BackendError> {
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
        cfg.step_tokens = c;
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
                rows: Vec::new(),
                spans: Vec::new(),
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

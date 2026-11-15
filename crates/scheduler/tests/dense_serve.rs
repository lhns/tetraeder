//! `Scheduler` over the Qwen3 dense builder (ADR 0065, 0066) on the tiny dense fixture: four
//! concurrent requests finish in batches above one sequence, and P1 (every request ends in
//! exactly one terminal event) holds under random prompt lengths, `max_tokens` and cancels.
//!
//! Hardware label: `cpu_*`.

use proptest::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, DenseOptions};
use tetraeder_scheduler::{
    BackendError, BatchBackend, DenseBackend, Loaded, Outputs, Scheduler, SchedulerConfig, SeqId,
    SeqInput,
};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-f16.gguf")
}

/// Passes everything through and records the largest number of inputs of one decode.
struct Recording {
    inner: DenseBackend,
    widest: Arc<AtomicUsize>,
}

impl BatchBackend for Recording {
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_seq_max(&self) -> usize {
        self.inner.n_seq_max()
    }
    fn n_ctx_seq(&self) -> usize {
        self.inner.n_ctx_seq()
    }
    fn kv_capacity(&self) -> usize {
        self.inner.kv_capacity()
    }
    fn kv_free(&self) -> usize {
        self.inner.kv_free()
    }
    fn seq_len(&self, seq: SeqId) -> usize {
        self.inner.seq_len(seq)
    }
    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.inner.seq_reserve(seq, n)
    }
    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        self.inner.seq_rm(seq, keep)
    }
    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        self.inner.seq_cp(src, dst, n)
    }
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.widest.fetch_max(inputs.len(), Ordering::Relaxed);
        self.inner.decode(inputs)
    }
    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.inner.seq_snapshot(seq)
    }
    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        self.inner.seq_restore(seq, bytes, n)
    }
    fn kv_format(&self) -> String {
        self.inner.kv_format()
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
    fn max_step_tokens(&self) -> usize {
        self.inner.max_step_tokens()
    }
}

fn start(parallel: usize, pool: usize) -> (Arc<Scheduler>, Arc<AtomicUsize>) {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = parallel;
    sc.step_tokens = 64;
    let widest = Arc::new(AtomicUsize::new(0));
    let w = Arc::clone(&widest);
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
        let opts = DenseOptions {
            n_seq_max: parallel,
            n_ctx_pool: pool,
            threads: Some(2),
            ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
        };
        let model = tetraeder_model::Qwen3::load(&path, &opts).map_err(|e| e.to_string())?;
        Ok(Loaded {
            backend: Box::new(Recording {
                inner: DenseBackend::new(model, false),
                widest: w,
            }),
            tokenizer,
        })
    })
    .expect("scheduler over the dense builder");
    (Arc::new(s), widest)
}

#[derive(Debug, Clone)]
struct Spec {
    id: u32,
    prompt_len: usize,
    max_tokens: u32,
    cancel_after: Option<usize>,
}

/// One request to its end; panics when the stream breaks the protocol (an event after the
/// terminal one, or none). Returns the number of tokens.
fn consume(s: &Scheduler, spec: &Spec) -> usize {
    let cancel = CancellationToken::new();
    let id = s.models()[0].id.clone();
    let mut rx = s
        .generate(GenerateRequest {
            model: id,
            prompt: (0..spec.prompt_len)
                .map(|i| 10 + (spec.id * 31 + i as u32 * 13) % 150)
                .collect(),
            params: SamplingParams {
                max_tokens: spec.max_tokens,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: cancel.clone(),
        })
        .expect("generate");
    let (mut tokens, mut ended) = (0usize, false);
    while let Some(ev) = rx.blocking_recv() {
        assert!(!ended, "event after the terminal one: {spec:?}");
        match ev {
            GenEvent::Token(_) => {
                tokens += 1;
                if Some(tokens) == spec.cancel_after {
                    cancel.cancel();
                }
            }
            GenEvent::Finished { .. } | GenEvent::Error(_) => ended = true,
        }
    }
    assert!(ended, "closed without a terminal event: {spec:?}");
    tokens
}

fn run_all(s: &Arc<Scheduler>, specs: &[Spec]) -> Vec<usize> {
    let (tx, rx) = std::sync::mpsc::channel();
    let threads: Vec<_> = specs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, spec)| {
            let (s, tx) = (Arc::clone(s), tx.clone());
            std::thread::spawn(move || {
                let _ = tx.send((i, consume(&s, &spec)));
            })
        })
        .collect();
    drop(tx);
    let mut out = vec![0; specs.len()];
    for _ in specs {
        let (i, n) = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("the scheduler hung or a request thread panicked");
        out[i] = n;
    }
    for t in threads {
        t.join().expect("request thread");
    }
    out
}

fn shutdown(s: Arc<Scheduler>) {
    Arc::try_unwrap(s)
        .unwrap_or_else(|_| panic!("a request thread still holds the scheduler"))
        .shutdown()
        .expect("the worker ends once every request has");
}

/// hardware: cpu. Four concurrent requests all finish and share decode steps.
#[test]
fn cpu_four_concurrent_requests_batch() {
    let (s, widest) = start(4, 2048);
    let specs: Vec<Spec> = (0..4)
        .map(|id| Spec {
            id,
            prompt_len: 20 + 5 * id as usize,
            max_tokens: 160,
            cancel_after: None,
        })
        .collect();
    let counts = run_all(&s, &specs);
    shutdown(s);
    assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
    assert!(
        widest.load(Ordering::Relaxed) > 1,
        "no step held more than one sequence"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4, failure_persistence: None, ..ProptestConfig::default() })]

    /// hardware: cpu. P1 under a pool smaller than the load (preemption) and cancels.
    #[test]
    fn cpu_p1_every_request_ends_once(
        specs in prop::collection::vec(
            (1usize..120, 1u32..16, prop::option::of(1usize..6)), 2..6),
        pool in prop::sample::select(vec![512usize, 1024]),
    ) {
        let (s, _) = start(3, pool);
        let specs: Vec<Spec> = specs
            .into_iter()
            .enumerate()
            .map(|(id, (prompt_len, max_tokens, cancel_after))| Spec {
                id: id as u32,
                prompt_len,
                max_tokens,
                cancel_after,
            })
            .collect();
        let counts = run_all(&s, &specs);
        shutdown(s);
        for (c, sp) in counts.iter().zip(&specs) {
            prop_assert!(*c <= sp.max_tokens as usize, "{sp:?}: {c} tokens");
        }
    }
}

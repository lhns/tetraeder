//! `Scheduler` over the Qwen3 dense builder (ADR 0065, 0066) on the tiny dense fixture: four
//! concurrent requests finish in batches above one sequence, and P1 (every request ends in
//! exactly one terminal event) holds under random prompt lengths, `max_tokens` and cancels.
//!
//! Hardware label: `cpu_*`.

use proptest::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tetraeder_gateway::engine::{
    Engine, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenConstraint, TokenMask,
};
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
    /// Successful `seq_cp` calls as (src, dst, n).
    cps: Arc<Mutex<Vec<(SeqId, SeqId, usize)>>>,
    /// `seq_cp` fails without trying, as a pool without room would.
    fail_cp: Arc<AtomicBool>,
    decodes: Arc<AtomicUsize>,
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
        if self.fail_cp.load(Ordering::Relaxed) {
            return Err(BackendError::Failed("injected".into()));
        }
        self.inner.seq_cp(src, dst, n)?;
        self.cps.lock().unwrap().push((src, dst, n));
        Ok(())
    }
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.widest.fetch_max(inputs.len(), Ordering::Relaxed);
        self.decodes.fetch_add(1, Ordering::Relaxed);
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

struct Probe {
    widest: Arc<AtomicUsize>,
    cps: Arc<Mutex<Vec<(SeqId, SeqId, usize)>>>,
    fail_cp: Arc<AtomicBool>,
    decodes: Arc<AtomicUsize>,
}

fn start(parallel: usize, pool: usize) -> (Arc<Scheduler>, Arc<AtomicUsize>) {
    let (s, p) = start_with(parallel, pool, true);
    (s, p.widest)
}

fn start_with(parallel: usize, pool: usize, prefix_reuse: bool) -> (Arc<Scheduler>, Probe) {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = parallel;
    sc.step_tokens = 64;
    sc.prefix_reuse = prefix_reuse;
    sc.cache_restore_gain = 16;
    let probe = Probe {
        widest: Arc::default(),
        cps: Arc::default(),
        fail_cp: Arc::default(),
        decodes: Arc::default(),
    };
    let (w, cps, fail_cp, decodes) = (
        Arc::clone(&probe.widest),
        Arc::clone(&probe.cps),
        Arc::clone(&probe.fail_cp),
        Arc::clone(&probe.decodes),
    );
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
                cps,
                fail_cp,
                decodes,
            }),
            tokenizer,
        })
    })
    .expect("scheduler over the dense builder");
    (Arc::new(s), probe)
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
            GenEvent::Finished { .. } | GenEvent::Error(_) | GenEvent::ContextExceeded(_) => {
                ended = true
            }
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

/// Allows a few tokens, so the tiny vocabulary's end-of-generation tokens never end a run.
struct Only(Vec<u32>);

impl TokenConstraint for Only {
    fn mask(&mut self) -> Result<Option<TokenMask>, String> {
        let mut words = vec![0u32; 16];
        for &t in &self.0 {
            words[t as usize / 32] |= 1 << (t % 32);
        }
        Ok(Some(TokenMask { words }))
    }

    fn commit(&mut self, _: u32) -> Result<(), String> {
        Ok(())
    }
}

type Events = tokio::sync::mpsc::Receiver<GenEvent>;

fn generate(s: &Scheduler, prompt: Vec<u32>, max_tokens: u32) -> Events {
    s.generate(GenerateRequest {
        model: s.models()[0].id.clone(),
        prompt,
        params: SamplingParams {
            max_tokens,
            temperature: Some(0.0),
            ..SamplingParams::default()
        },
        constraint: Some(Box::new(Only(vec![65, 66, 67]))),
        cancel: CancellationToken::new(),
    })
    .expect("generate")
}

/// Reads to the end; returns the generated tokens and `cached_tokens`.
fn finish(mut rx: Events) -> (Vec<u32>, u32) {
    let mut toks = Vec::new();
    loop {
        match rx.blocking_recv().expect("terminal event") {
            GenEvent::Token(t) => toks.push(t),
            GenEvent::Finished { reason, stats } => {
                assert_eq!(reason, FinishReason::Length);
                return (toks, stats.cached_tokens);
            }
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("{e}"),
        }
    }
}

/// The first `share` tokens are common to all seeds.
fn shared_prompt(n: usize, seed: u32, share: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            if i < share {
                10 + (i as u32 * 7) % 150
            } else {
                10 + (seed * 31 + i as u32 * 13) % 150
            }
        })
        .collect()
}

/// A second request sharing a long prefix with a running one (its reader is not reading, so it
/// stays busy): returns the second one's tokens and `cached_tokens`, and the successful copies.
fn second_over_a_busy_one(
    prefix_reuse: bool,
    fail_cp: bool,
) -> (Vec<u32>, u32, Vec<(SeqId, SeqId, usize)>) {
    let (s, probe) = start_with(2, 2048, prefix_reuse);
    probe.fail_cp.store(fail_cp, Ordering::Relaxed);
    let first = generate(&s, shared_prompt(600, 1, 600), 400);
    let t0 = std::time::Instant::now();
    while probe.decodes.load(Ordering::Relaxed) < 12 {
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "first never decodes"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let (toks, cached) = finish(generate(&s, shared_prompt(640, 2, 600), 12));
    let cps = probe.cps.lock().unwrap().clone();
    drop(first);
    shutdown(s);
    (toks, cached, cps)
}

/// hardware: cpu. The own pool takes a copy into the destination the scheduler reserved: the
/// second request starts from the shared prefix and generates what a fresh prefill does.
#[test]
fn cpu_a_busy_sequence_lends_its_prefix() {
    let (with, cached, cps) = second_over_a_busy_one(true, false);
    let (without, none, _) = second_over_a_busy_one(false, false);
    assert!(cached >= 500, "cached {cached}");
    assert!(cps.iter().any(|&(_, _, n)| n >= 500), "no copy: {cps:?}");
    assert_eq!(none, 0);
    assert_eq!(with, without);
}

/// hardware: cpu. A copy that fails does not lose the request: it prefills from scratch (no
/// idle or RAM source here) and still generates the same tokens.
#[test]
fn cpu_a_failed_busy_copy_falls_back() {
    let (got, cached, cps) = second_over_a_busy_one(true, true);
    let (fresh, _, _) = second_over_a_busy_one(false, false);
    assert!(cps.is_empty());
    assert_eq!(cached, 0);
    assert_eq!(got, fresh);
}

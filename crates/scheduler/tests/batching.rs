//! Continuous batching through the gateway `Engine` trait on `Scheduler` over `FakeKv`
//! (ADR 0065, "Property tests, first"): P1 (every request ends in exactly one terminal event),
//! P4 (batching never mixes sequences: tokens equal a solo run) and P5 (progress within a bound
//! on steps), under random arrivals, prompt lengths, `max_tokens`, cancellations, slow consumers,
//! `KvFull` and failures. P2 and P3 are `FakeKv`'s own checks (`tests/batch_backend.rs`).
//!
//! Hardware label: none (the tokenizer comes from the tiny fixture).

mod fake;

use fake::{FakeKv, Probe, logits_for};
use proptest::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tetraeder_gateway::engine::{
    Engine, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenConstraint, TokenMask,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{Loaded, Scheduler, SchedulerConfig};
use tokio_util::sync::CancellationToken;

/// Longest prompt plus generation any test request needs.
const N_CTX_SEQ: usize = 3200;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

#[derive(Clone)]
struct Setup {
    parallel: usize,
    step_tokens: usize,
    /// Rows the pool shows; below the sum of the requests it forces preemption.
    capacity: usize,
    fail_every: Option<u32>,
}

fn start(cfg: &Setup, metrics: Option<tetraeder_telemetry::Metrics>) -> (Scheduler, Arc<Probe>) {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = cfg.parallel;
    sc.step_tokens = cfg.step_tokens;
    // Small thresholds so short prompts exercise saving and restoring.
    sc.cache_save_min = 32;
    sc.cache_restore_gain = 16;
    sc.metrics = metrics;
    let (parallel, capacity, fail_every) = (cfg.parallel, cfg.capacity, cfg.fail_every);
    let probe_cell: Arc<std::sync::OnceLock<Arc<Probe>>> = Arc::default();
    let cell = Arc::clone(&probe_cell);
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
        let n_vocab = tokenizer.n_vocab() as usize;
        let kv = FakeKv::new(n_vocab, parallel, N_CTX_SEQ, capacity).with_fail_every(fail_every);
        let _ = cell.set(kv.probe());
        Ok(Loaded {
            backend: Box::new(kv),
            tokenizer,
        })
    })
    .expect("scheduler over FakeKv");
    (s, probe_cell.get().cloned().expect("probe"))
}

#[derive(Debug, Clone)]
struct Spec {
    prompt_len: usize,
    /// The first `share` tokens are the same in every request (prefix reuse).
    share: usize,
    id: u32,
    max_tokens: u32,
    greedy: bool,
    seed: u64,
    /// Waits before sending the request.
    arrival_ms: u64,
    /// Cancels after this many received tokens.
    cancel_after: Option<usize>,
    /// Waits before reading anything: the channel fills and the sequence is paused.
    read_after_ms: u64,
}

impl Spec {
    fn prompt(&self) -> Vec<u32> {
        (0..self.prompt_len)
            .map(|i| {
                if i < self.share {
                    10 + (i as u32 * 7) % 150
                } else {
                    10 + (self.id * 31 + i as u32 * 13) % 150
                }
            })
            .collect()
    }

    fn params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.greedy.then_some(0.0),
            seed: Some(self.seed),
            ..SamplingParams::default()
        }
    }
}

#[derive(Debug)]
struct Outcome {
    tokens: Vec<u32>,
    /// `Ok(reason)` for `Finished`, `Err(message)` for `Error`.
    end: Result<FinishReason, String>,
}

/// Runs one request to its end. Panics when the stream breaks the protocol (P1): an event after
/// the terminal one, or a stream that closes without one.
fn consume(s: &Scheduler, spec: &Spec, constraint: Option<Box<dyn TokenConstraint>>) -> Outcome {
    std::thread::sleep(Duration::from_millis(spec.arrival_ms));
    let cancel = CancellationToken::new();
    let id = s.models()[0].id.clone();
    let mut rx = s
        .generate(GenerateRequest {
            model: id,
            prompt: spec.prompt(),
            params: spec.params(),
            constraint,
            cancel: cancel.clone(),
        })
        .expect("generate");
    std::thread::sleep(Duration::from_millis(spec.read_after_ms));
    let mut tokens = Vec::new();
    let mut end = None;
    while let Some(ev) = rx.blocking_recv() {
        assert!(end.is_none(), "event after the terminal one: {spec:?}");
        match ev {
            GenEvent::Token(t) => {
                tokens.push(t);
                if Some(tokens.len()) == spec.cancel_after {
                    cancel.cancel();
                }
            }
            GenEvent::Finished { reason, stats } => {
                assert_eq!(stats.generated_tokens as usize, tokens.len(), "{spec:?}");
                end = Some(Ok(reason));
            }
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => end = Some(Err(e)),
        }
    }
    Outcome {
        tokens,
        end: end.unwrap_or_else(|| panic!("closed without a terminal event: {spec:?}")),
    }
}

/// All requests concurrently, each on its own thread; a watchdog turns a hang into a failure.
fn run_all(s: &Arc<Scheduler>, specs: &[Spec]) -> Vec<Outcome> {
    let (tx, rx) = std::sync::mpsc::channel();
    let threads: Vec<_> = specs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, spec)| {
            let (s, tx) = (Arc::clone(s), tx.clone());
            std::thread::spawn(move || {
                let _ = tx.send((i, consume(&s, &spec, None)));
            })
        })
        .collect();
    drop(tx);
    let mut out: Vec<Option<Outcome>> = specs.iter().map(|_| None).collect();
    for _ in specs {
        let (i, o) = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("the scheduler hung or a request thread panicked");
        out[i] = Some(o);
    }
    // The threads drop their handle on the scheduler as they end.
    for t in threads {
        t.join().expect("request thread");
    }
    out.into_iter().map(Option::unwrap).collect()
}

/// The same requests one after the other on a scheduler with one sequence and room for all.
fn solo(specs: &[Spec]) -> Vec<Outcome> {
    let cfg = Setup {
        parallel: 1,
        step_tokens: 512,
        capacity: N_CTX_SEQ,
        fail_every: None,
    };
    let (s, _) = start(&cfg, None);
    specs
        .iter()
        .map(|sp| {
            let sp = Spec {
                arrival_ms: 0,
                cancel_after: None,
                read_after_ms: 0,
                ..sp.clone()
            };
            consume(&s, &sp, None)
        })
        .collect()
}

fn shutdown(s: Arc<Scheduler>) {
    Arc::try_unwrap(s)
        .ok()
        .expect("no thread keeps the scheduler")
        .shutdown()
        .expect("the worker ends once every request has");
}

fn spec_strategy(id: u32) -> impl Strategy<Value = Spec> {
    (
        prop_oneof![3 => 1usize..60, 1 => 1usize..3000],
        prop_oneof![3 => 0usize..40, 1 => 0usize..400],
        1u32..90,
        any::<bool>(),
        any::<u64>(),
        0u64..15,
        prop_oneof![3 => Just(None), 1 => (0usize..6).prop_map(Some)],
        prop_oneof![3 => Just(0u64), 1 => 5u64..40],
    )
        .prop_map(
            move |(prompt_len, share, max_tokens, greedy, seed, arrival_ms, cancel_after, wait)| {
                Spec {
                    prompt_len,
                    share: share.min(prompt_len),
                    id,
                    max_tokens,
                    greedy,
                    seed,
                    arrival_ms,
                    cancel_after,
                    read_after_ms: wait,
                }
            },
        )
}

fn specs_strategy() -> impl Strategy<Value = Vec<Spec>> {
    (1usize..=8).prop_flat_map(|n| (0..n as u32).map(spec_strategy).collect::<Vec<_>>())
}

fn setup_strategy() -> impl Strategy<Value = (usize, usize, u32, bool)> {
    (
        1usize..=4,
        // Step budget: from "barely above parallel" to bigger than any prompt.
        prop_oneof![Just(24usize), Just(64), Just(512), Just(4096)],
        2u32..8,
        any::<bool>(),
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, failure_persistence: None, ..ProptestConfig::default() })]

    /// P1 and P4 with a pool smaller than the load (preemption, eviction, restore from RAM) and
    /// injected failures.
    #[test]
    fn p1_p4_hold_under_pressure_and_failures(
        specs in specs_strategy(),
        (parallel, step, fail_n, inject) in setup_strategy(),
        extra in 0usize..400,
    ) {
        let need = specs.iter().map(|s| s.prompt_len + s.max_tokens as usize).max().unwrap();
        // One request always fits alone; several together often do not.
        let cfg = Setup {
            parallel,
            step_tokens: step.max(parallel + 8),
            capacity: (need + extra).min(N_CTX_SEQ * parallel),
            fail_every: inject.then_some(fail_n),
        };
        let (s, probe) = start(&cfg, None);
        let s = Arc::new(s);
        let got = run_all(&s, &specs);
        shutdown(s);
        let want = solo(&specs);
        for ((spec, g), w) in specs.iter().zip(&got).zip(&want) {
            prop_assert!(w.end.is_ok(), "solo run failed: {:?}", w.end);
            match &g.end {
                Ok(FinishReason::Cancelled) | Err(_) => {
                    prop_assert!(
                        w.tokens.starts_with(&g.tokens),
                        "{spec:?}: {:?} is not a prefix of {:?}", g.tokens, w.tokens
                    );
                    if g.end.is_err() {
                        prop_assert!(inject, "{spec:?}: error without an injected failure: {:?}", g.end);
                    }
                }
                Ok(reason) => {
                    prop_assert_eq!(&g.tokens, &w.tokens, "{:?}", spec);
                    prop_assert_eq!(Some(reason), w.end.as_ref().ok());
                }
            }
        }
        prop_assert!(probe.max_inputs.load(Ordering::SeqCst) <= parallel);
        prop_assert!(probe.max_step_tokens.load(Ordering::SeqCst) <= cfg.step_tokens.max(parallel));
    }

    /// P5: without failures, cancellations or slow readers every request finishes, within as many
    /// decode calls as the requests would take one after the other (the oldest live request gets
    /// a full chunk or its token in every step).
    #[test]
    fn p5_progress_is_bounded(
        specs in specs_strategy(),
        (parallel, step, _, _) in setup_strategy(),
    ) {
        let specs: Vec<Spec> = specs
            .into_iter()
            .map(|s| Spec { cancel_after: None, read_after_ms: 0, ..s })
            .collect();
        let cfg = Setup {
            parallel,
            step_tokens: step.max(parallel + 8),
            capacity: N_CTX_SEQ * parallel,
            fail_every: None,
        };
        let (s, probe) = start(&cfg, None);
        let s = Arc::new(s);
        let got = run_all(&s, &specs);
        shutdown(s);
        let chunk = cfg.step_tokens - parallel;
        let mut bound = 0;
        for (spec, g) in specs.iter().zip(&got) {
            prop_assert!(
                matches!(g.end, Ok(FinishReason::EndOfGeneration | FinishReason::Length)),
                "{spec:?}: {:?}", g.end
            );
            bound += spec.prompt_len.div_ceil(chunk) + spec.max_tokens as usize;
        }
        let decodes = probe.decodes.load(Ordering::SeqCst) as usize;
        prop_assert!(decodes <= bound, "{decodes} decode calls, bound {bound}");
    }
}

/// Allows only a few tokens, so the tiny vocabulary's end-of-generation tokens never end a run.
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

/// Two long generations in a pool that holds one: the newer is preempted when the older grows
/// past the pool, requeued with its generated tokens, and ends with the tokens of a solo run.
#[test]
fn preemption_requeues_and_output_equals_a_solo_run() {
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let metrics = tetraeder_telemetry::Metrics::new(&mp.meter("test"));
    let mk = |id| Spec {
        prompt_len: 100,
        share: 0,
        id,
        max_tokens: 1500,
        greedy: true,
        seed: 1,
        arrival_ms: 0,
        cancel_after: None,
        read_after_ms: 0,
    };
    let specs = [mk(1), mk(2)];
    // Both are admitted (2 x 1124 rows fit), the older outgrows the pool at 1176 positions.
    let cfg = Setup {
        parallel: 2,
        step_tokens: 512,
        capacity: 2300,
        fail_every: None,
    };
    let (s, _) = start(&cfg, Some(metrics));
    let s = Arc::new(s);
    let (tx, rx) = std::sync::mpsc::channel();
    let threads: Vec<_> = specs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, sp)| {
            let (s, tx) = (Arc::clone(&s), tx.clone());
            std::thread::spawn(move || {
                let _ = tx.send((i, consume(&s, &sp, Some(Box::new(Only(vec![65, 66, 67]))))));
            })
        })
        .collect();
    let mut got: Vec<Option<Outcome>> = vec![None, None];
    for _ in 0..2 {
        let (i, o) = rx.recv_timeout(Duration::from_secs(120)).expect("hung");
        got[i] = Some(o);
    }
    for t in threads {
        t.join().expect("request thread");
    }
    shutdown(s);
    let (solo_sched, _) = start(
        &Setup {
            parallel: 1,
            step_tokens: 512,
            capacity: N_CTX_SEQ,
            fail_every: None,
        },
        None,
    );
    for (sp, g) in specs.iter().zip(got) {
        let g = g.unwrap();
        let w = consume(&solo_sched, sp, Some(Box::new(Only(vec![65, 66, 67]))));
        assert_eq!(g.end, Ok(FinishReason::Length));
        assert_eq!(g.tokens.len(), 1500);
        assert_eq!(g.tokens, w.tokens);
    }
    mp.force_flush().unwrap();
    let mut preemptions = 0;
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for m in sm.metrics() {
                if let (
                    "tetraeder.scheduler.preemptions",
                    AggregatedMetrics::U64(MetricData::Sum(x)),
                ) = (m.name(), m.data())
                {
                    preemptions += x.data_points().map(|d| d.value()).sum::<u64>();
                }
            }
        }
    }
    assert!(preemptions >= 1, "no preemption happened");
}

/// A reader that never reads does not stall the others: its sequence is paused with a full
/// channel while a second request runs to its end.
#[test]
fn a_slow_consumer_does_not_stall_the_batch() {
    let cfg = Setup {
        parallel: 2,
        step_tokens: 512,
        capacity: 2 * 400,
        fail_every: None,
    };
    let (s, _) = start(&cfg, None);
    let s = Arc::new(s);
    let id = s.models()[0].id.clone();
    let stalled = s
        .generate(GenerateRequest {
            model: id,
            prompt: vec![20, 21, 22],
            params: SamplingParams {
                max_tokens: 200,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: Some(Box::new(Only(vec![65, 66, 67]))),
            cancel: CancellationToken::new(),
        })
        .unwrap();
    let spec = Spec {
        prompt_len: 30,
        share: 0,
        id: 5,
        max_tokens: 100,
        greedy: true,
        seed: 3,
        arrival_ms: 50,
        cancel_after: None,
        read_after_ms: 0,
    };
    let done = consume(&s, &spec, Some(Box::new(Only(vec![65, 66, 67]))));
    assert_eq!(done.end, Ok(FinishReason::Length));
    assert_eq!(done.tokens.len(), 100);
    // The slow reader then gets everything, in order, once.
    let mut rx = stalled;
    let mut n = 0;
    loop {
        match rx.blocking_recv().expect("terminal event") {
            GenEvent::Token(_) => n += 1,
            GenEvent::Finished { reason, .. } => {
                assert_eq!(reason, FinishReason::Length);
                break;
            }
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("{e}"),
        }
    }
    assert_eq!(n, 200);
    drop(rx);
    shutdown(s);
}

/// `FakeKv`'s logits really depend on the whole sequence, so P4 can catch a mix-up.
#[test]
fn fake_logits_depend_on_every_token() {
    assert_ne!(logits_for(16, &[1, 2, 3]), logits_for(16, &[1, 2, 4]));
    assert_ne!(logits_for(16, &[1, 2, 3]), logits_for(16, &[2, 1, 3]));
}

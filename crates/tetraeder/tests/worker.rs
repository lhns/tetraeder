//! `tetraeder worker --fake` (a `FakeKv`) behind `RemoteBackend` (ADR 0069, "Worker process, one
//! per box" and "Failure handling"): the scheduler's batching properties P1 (every request ends
//! in exactly one terminal event) and P4 (batching never mixes sequences) through the pipe, and
//! the supervision cases: a child that dies mid-step, hangs, answers garbage or half a frame, or
//! reports a sticky error, and the restart cap.
//!
//! Faults are `--fake-fault KIND:TOKEN` and fire on a `decode` input of two or more tokens that
//! holds TOKEN, so sampled tokens (one-token inputs) never trigger them.
//!
//! Hardware label: cpu (the tokenizer comes from the tiny fixture).

#[allow(clippy::all, clippy::pedantic)]
#[path = "../../scheduler/tests/fake/mod.rs"]
mod fake;

use fake::{FakeKv, logits_for};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tetraeder::remote_backend::{RemoteBackend, RemoteConfig};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Loaded, Scheduler, SchedulerConfig, SeqInput,
};
use tokio_util::sync::CancellationToken;

const VOCAB: usize = 64;
const CTX: usize = 3200;
const FLOOR: Duration = Duration::from_secs(1);

fn worker_config(vocab: usize, parallel: usize, extra: &[&str]) -> RemoteConfig {
    let mut args: Vec<OsString> = vec![
        "worker".into(),
        "--fake".into(),
        "--fake-vocab".into(),
        vocab.to_string().into(),
        "--parallel".into(),
        parallel.to_string().into(),
        "--ctx".into(),
        CTX.to_string().into(),
    ];
    args.extend(extra.iter().map(OsString::from));
    let mut c = RemoteConfig::new(env!("CARGO_BIN_EXE_tetraeder"), args);
    c.deadline_floor = FLOOR;
    c.backoff_base = Duration::from_millis(10);
    c.startup_timeout = Duration::from_secs(60);
    c
}

fn remote(extra: &[&str]) -> RemoteBackend {
    RemoteBackend::start(worker_config(VOCAB, 4, extra)).expect("worker starts")
}

type Step<'a> = &'a [(u32, Vec<i32>, bool)];

/// One decode; the rows of every input, or the error.
fn step(b: &mut dyn BatchBackend, ins: Step<'_>) -> Result<Vec<Vec<f32>>, BackendError> {
    let inputs: Vec<SeqInput<'_>> = ins
        .iter()
        .map(|(seq, tokens, all_logits)| SeqInput {
            seq: *seq,
            tokens,
            all_logits: *all_logits,
        })
        .collect();
    let out = b.decode(&inputs)?;
    Ok((0..out.len()).map(|i| out.rows(i).to_vec()).collect())
}

fn failed_with(r: Result<Vec<Vec<f32>>, BackendError>, what: &str) {
    match r {
        Err(BackendError::Failed(m)) => assert!(m.contains(what), "{m:?} lacks {what:?}"),
        other => panic!("expected Failed({what}), got {other:?}"),
    }
}

fn same_state(a: &dyn BatchBackend, b: &dyn BatchBackend) {
    assert_eq!(a.kv_free(), b.kv_free());
    for s in 0..a.n_seq_max() as u32 {
        assert_eq!(a.seq_len(s), b.seq_len(s), "len of {s}");
        assert_eq!(a.seq_reserved(s), b.seq_reserved(s), "reserved of {s}");
    }
}

// hardware: cpu
#[test]
fn the_pipe_gives_the_answers_of_the_backend_itself() {
    let mut r = remote(&[]);
    let mut l = FakeKv::new(VOCAB, 4, CTX, CTX);
    assert_eq!(r.n_vocab(), l.n_vocab());
    assert_eq!(r.n_seq_max(), l.n_seq_max());
    assert_eq!(r.n_ctx_seq(), l.n_ctx_seq());
    assert_eq!(r.kv_capacity(), l.kv_capacity());
    assert_eq!(r.kv_format(), l.kv_format());
    assert_eq!(r.kv_row_bytes(), l.kv_row_bytes());
    assert_eq!(r.max_step_tokens(), l.max_step_tokens());
    assert_eq!((r.on_gpu(), r.bytes_streamed_total()), (false, None));
    same_state(&r, &l);

    for b in [&mut r as &mut dyn BatchBackend, &mut l] {
        b.seq_reserve(0, 10).unwrap();
    }
    same_state(&r, &l);
    let steps: [Step<'_>; 3] = [
        &[(0, vec![1, 2, 3], false), (1, vec![4, 5], true)],
        &[(0, vec![9], false), (1, vec![7, 8], false)],
        &[(2, vec![1, 2, 3], false)],
    ];
    for s in steps {
        assert_eq!(step(&mut r, s).unwrap(), step(&mut l, s).unwrap());
        same_state(&r, &l);
    }
    // Errors keep their kind and numbers.
    assert_eq!(r.seq_reserve(3, CTX + 1), l.seq_reserve(3, CTX + 1));
    assert!(matches!(
        r.seq_reserve(3, CTX + 1),
        Err(BackendError::KvFull { .. })
    ));
    assert_eq!(r.seq_cp(0, 1, 2), l.seq_cp(0, 1, 2));
    assert_eq!(r.seq_cp(0, 3, 2), l.seq_cp(0, 3, 2));
    same_state(&r, &l);
    // Snapshots are raw bytes in the payload.
    let (sr, sl) = (r.seq_snapshot(0).unwrap(), l.seq_snapshot(0).unwrap());
    assert_eq!(sr, sl);
    assert_eq!(r.seq_restore(1, &sr, 4), l.seq_restore(1, &sl, 4));
    assert_eq!(r.seq_rm(0, 1), l.seq_rm(0, 1));
    same_state(&r, &l);
    let s: Step<'_> = &[(1, vec![3], false)];
    assert_eq!(step(&mut r, s).unwrap(), step(&mut l, s).unwrap());
    assert_eq!(r.restarts(), 0);
}

// hardware: cpu
#[test]
fn a_failed_step_that_is_not_sticky_keeps_the_worker() {
    let mut b = remote(&["--fake-fail-every", "2"]);
    step(&mut b, &[(0, vec![1, 2, 3], false)]).unwrap();
    failed_with(step(&mut b, &[(0, vec![4], false)]), "injected failure");
    // The rollback contract holds across the pipe, and nothing restarted.
    assert_eq!(b.seq_len(0), 3);
    step(&mut b, &[(0, vec![4], false)]).unwrap();
    assert_eq!((b.seq_len(0), b.restarts()), (4, 0));
}

/// The state after a death: nothing held, live sequences refused, others usable, and the first
/// call that needs the worker starts it again.
fn assert_recovers(b: &mut RemoteBackend) {
    assert_eq!((b.seq_len(0), b.seq_reserved(0)), (0, 0));
    assert_eq!(b.kv_free(), b.kv_capacity());
    // Sequence 0 held rows; its tokens are gone and it must not continue.
    failed_with(step(b, &[(0, vec![4], false)]), "lost");
    // Freeing it must not start the worker; sequence 1 held nothing and works at once.
    b.seq_rm(0, 0).unwrap();
    assert_eq!(b.restarts(), 0);
    let rows = step(b, &[(1, vec![1, 2, 3], false)]).unwrap();
    assert_eq!(rows[0], logits_for(VOCAB, &[1, 2, 3]));
    assert_eq!(b.restarts(), 1);
    let rows = step(b, &[(0, vec![1, 2], false)]).unwrap();
    assert_eq!(rows[0], logits_for(VOCAB, &[1, 2]));
    assert_eq!((b.seq_len(0), b.seq_len(1)), (2, 3));
}

fn fault_case(fault: &str, expect: &str) -> Duration {
    let mut b = remote(&["--fake-fault", fault]);
    step(&mut b, &[(0, vec![1, 2, 3], false)]).unwrap();
    assert_eq!(b.seq_len(0), 3);
    let t0 = Instant::now();
    failed_with(step(&mut b, &[(1, vec![60, 61], false)]), expect);
    let took = t0.elapsed();
    assert_recovers(&mut b);
    took
}

// hardware: cpu
#[test]
fn a_child_killed_mid_step_fails_that_step_and_later_calls_succeed() {
    fault_case("crash:60", "closed its output");
}

// hardware: cpu
#[test]
fn a_child_that_hangs_is_killed_at_the_deadline() {
    let took = fault_case("hang:60", "no reply within");
    assert!(took >= FLOOR, "{took:?}");
    assert!(took < Duration::from_secs(30), "{took:?}");
}

// hardware: cpu
#[test]
fn a_sticky_error_restarts_the_worker() {
    fault_case("sticky:60", "CUDA error");
}

// hardware: cpu
#[test]
fn a_malformed_frame_restarts_the_worker() {
    fault_case("garbage:60", "bad header");
}

// hardware: cpu
#[test]
fn half_a_frame_restarts_the_worker() {
    fault_case("truncate:60", "truncated");
}

// hardware: cpu
#[test]
fn a_worker_that_cannot_start_is_a_startup_error() {
    let err = |c| RemoteBackend::start(c).err().expect("startup error");
    let e = err(RemoteConfig::new("tetraeder-no-such-program", Vec::new()));
    assert!(e.contains("cannot start"), "{e}");
    let mut c = worker_config(VOCAB, 4, &["--no-such-flag"]);
    c.startup_timeout = Duration::from_secs(30);
    let e = err(c);
    assert!(e.contains("did not start"), "{e}");
}

/// Crashes the worker on every call, past the cap.
fn crash_until_the_cap(b: &mut RemoteBackend) -> usize {
    let mut failed = 0;
    for _ in 0..8 {
        if step(b, &[(1, vec![60, 61], false)]).is_err() {
            failed += 1;
        }
    }
    failed
}

// hardware: cpu
#[test]
fn the_restart_cap_gives_up_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let mut cfg = worker_config(VOCAB, 4, &["--fake-fault", "crash:60"]);
    cfg.on_give_up = Box::new(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
    });
    let mut b = RemoteBackend::start(cfg).unwrap();
    // Five restarts are allowed: the sixth death is the last worker, the next call gives up.
    for _ in 0..6 {
        failed_with(
            step(&mut b, &[(1, vec![60, 61], false)]),
            "closed its output",
        );
    }
    assert_eq!((b.restarts(), calls.load(Ordering::SeqCst)), (5, 0));
    failed_with(step(&mut b, &[(1, vec![60, 61], false)]), "giving up");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(crash_until_the_cap(&mut b), 8);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "gives up once");
}

// hardware: cpu
#[test]
fn the_restart_cap_ends_the_process() {
    if std::env::var_os("TETRAEDER_CAP_CHILD").is_some() {
        // The default handler exits with status 1.
        let mut b =
            RemoteBackend::start(worker_config(VOCAB, 4, &["--fake-fault", "crash:60"])).unwrap();
        crash_until_the_cap(&mut b);
        panic!("the process should have exited");
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "the_restart_cap_ends_the_process", "--nocapture"])
        .env("TETRAEDER_CAP_CHILD", "1")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(1));
}

// ---- the scheduler over the pipe ----

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

/// A scheduler over a worker (`extra` worker flags), or over an in-process `FakeKv`.
fn scheduler(parallel: usize, extra: Option<&'static [&'static str]>) -> Scheduler {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = parallel;
    sc.step_tokens = 512;
    sc.cache_save_min = 8;
    sc.cache_restore_gain = 4;
    Scheduler::start(sc, move || {
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
        let backend: Box<dyn BatchBackend> = match extra {
            Some(extra) => Box::new(RemoteBackend::start(worker_config(
                n_vocab, parallel, extra,
            ))?),
            None => Box::new(FakeKv::new(n_vocab, parallel, CTX, CTX)),
        };
        Ok(Loaded { backend, tokenizer })
    })
    .expect("scheduler")
}

#[derive(Debug)]
struct Outcome {
    tokens: Vec<u32>,
    /// `Err` for an `Error` event.
    ok: Result<(), String>,
}

/// Runs one request; panics when the stream breaks P1 (an event after the terminal one, or no
/// terminal one).
fn generate(s: &Scheduler, prompt: Vec<u32>, max_tokens: u32) -> Outcome {
    let mut rx = s
        .generate(GenerateRequest {
            model: s.models()[0].id.clone(),
            prompt,
            params: SamplingParams {
                max_tokens,
                temperature: Some(0.0),
                seed: Some(1),
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .expect("generate");
    let mut tokens = Vec::new();
    let mut end = None;
    while let Some(ev) = rx.blocking_recv() {
        assert!(end.is_none(), "event after the terminal one");
        match ev {
            GenEvent::Token(t) => tokens.push(t),
            GenEvent::Finished { .. } => end = Some(Ok(())),
            GenEvent::Probs(_) => {}
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => end = Some(Err(e)),
        }
    }
    Outcome {
        tokens,
        ok: end.expect("closed without a terminal event"),
    }
}

/// Prompt of `len` tokens whose first `share` are common to all requests; none is a fault token.
fn prompt(id: u32, len: usize, share: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            if i < share {
                10 + (i as u32 * 7) % 80
            } else {
                10 + (id * 31 + i as u32 * 13) % 80
            }
        })
        .collect()
}

fn requests() -> Vec<(Vec<u32>, u32)> {
    [
        (5usize, 12u32),
        (40, 30),
        (200, 8),
        (3, 40),
        (90, 20),
        (1, 25),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(len, max))| (prompt(i as u32, len, len.min(24) / 2), max))
    .collect()
}

fn solo(reqs: &[(Vec<u32>, u32)]) -> Vec<Outcome> {
    let s = scheduler(1, None);
    reqs.iter()
        .map(|(p, m)| generate(&s, p.clone(), *m))
        .collect()
}

/// All requests at once, each on its own thread, with a watchdog.
fn together(s: &Arc<Scheduler>, reqs: &[(Vec<u32>, u32)]) -> Vec<Outcome> {
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, (p, m)) in reqs.iter().cloned().enumerate() {
        let (s, tx) = (Arc::clone(s), tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send((i, generate(&s, p, m)));
        });
    }
    drop(tx);
    let mut out: Vec<Option<Outcome>> = reqs.iter().map(|_| None).collect();
    for _ in reqs {
        let (i, o) = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("the scheduler hung or a request thread panicked");
        out[i] = Some(o);
    }
    out.into_iter().map(Option::unwrap).collect()
}

fn shutdown(s: Arc<Scheduler>) {
    // The request threads drop their handles as they end.
    for _ in 0..100 {
        if Arc::strong_count(&s) == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Arc::try_unwrap(s)
        .ok()
        .expect("no thread keeps the scheduler")
        .shutdown()
        .expect("the worker ends once every request has");
}

// hardware: cpu
#[test]
fn p1_and_p4_hold_through_the_pipe() {
    let reqs = requests();
    let want = solo(&reqs);
    assert!(want.iter().all(|w| w.ok.is_ok() && !w.tokens.is_empty()));
    let s = Arc::new(scheduler(3, Some(&[])));
    let got = together(&s, &reqs);
    shutdown(s);
    for (g, w) in got.iter().zip(&want) {
        assert!(g.ok.is_ok(), "{:?}", g.ok);
        assert_eq!(g.tokens, w.tokens);
    }
}

// hardware: cpu
#[test]
fn p1_and_p4_hold_through_the_pipe_with_failing_steps() {
    let reqs = requests();
    let want = solo(&reqs);
    let s = Arc::new(scheduler(3, Some(&["--fake-fail-every", "4"])));
    let got = together(&s, &reqs);
    shutdown(s);
    for (g, w) in got.iter().zip(&want) {
        if g.ok.is_ok() {
            assert_eq!(g.tokens, w.tokens);
        } else {
            assert!(w.tokens.starts_with(&g.tokens));
        }
    }
}

/// A request that kills (or wedges) the worker fails; the requests before and after it do not.
fn survives(fault: &'static [&'static str]) {
    let reqs = requests();
    let want = solo(&reqs);
    let s = Arc::new(scheduler(1, Some(fault)));
    let first = generate(&s, reqs[1].0.clone(), reqs[1].1);
    assert!(first.ok.is_ok());
    assert_eq!(first.tokens, want[1].tokens);
    // The fault token 99 is only in this prompt.
    let mut bad = prompt(7, 20, 0);
    bad[10] = 99;
    let broken = generate(&s, bad, 10);
    assert!(broken.ok.is_err(), "{broken:?}");
    // Later requests succeed after the restart, with and without a prefix to reuse.
    for i in [1, 4, 0, 2] {
        let o = generate(&s, reqs[i].0.clone(), reqs[i].1);
        assert!(o.ok.is_ok(), "request {i}: {:?}", o.ok);
        assert_eq!(o.tokens, want[i].tokens, "request {i}");
    }
    shutdown(s);
}

// hardware: cpu
#[test]
fn a_crash_mid_step_fails_that_request_and_the_queue_survives() {
    survives(&["--fake-fault", "crash:99"]);
}

// hardware: cpu
#[test]
fn a_hang_fails_that_request_and_the_queue_survives() {
    survives(&["--fake-fault", "hang:99"]);
}

// hardware: cpu
#[test]
fn a_sticky_error_fails_that_request_and_the_queue_survives() {
    survives(&["--fake-fault", "sticky:99"]);
}

// hardware: cpu
#[test]
fn a_malformed_reply_fails_that_request_and_the_queue_survives() {
    survives(&["--fake-fault", "garbage:99"]);
}

// hardware: cpu
#[test]
fn a_truncated_reply_fails_that_request_and_the_queue_survives() {
    survives(&["--fake-fault", "truncate:99"]);
}

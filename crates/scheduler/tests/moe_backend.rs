//! `MoeBackend` as a native `BatchBackend` over the builder's KV pool (ADR 0065, phase 4 C4) on
//! the tiny fixture, CPU only: several sequences per step, the step cap, `KvFull`, `seq_cp`,
//! snapshots and the scheduler serving concurrent requests with the same greedy tokens as alone.
//!
//! Hardware label: cpu.

use std::path::PathBuf;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Scheduler, SchedulerConfig, SeqInput, load_moe, load_moe_chunked,
};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn opts(n_ctx: usize, n_seq_max: usize) -> LoadOptions {
    LoadOptions {
        n_ctx,
        n_seq_max,
        ..LoadOptions::new(BackendChoice::Cpu { threads: Some(2) })
    }
}

fn backend(n_ctx: usize, n_seq_max: usize, chunk: usize) -> Box<dyn BatchBackend> {
    load_moe_chunked(&fixture(), &opts(n_ctx, n_seq_max), chunk)
        .expect("fixture loads")
        .backend
}

fn prompt(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32)
        .map(|i| 4 + (seed * 31 + i * 7) % 200)
        .collect()
}

fn close(a: &[f32], b: &[f32], what: &str) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = 1e-4f32.max(1e-3 * scale);
    assert!(worst <= tol, "{what}: {worst} > {tol}");
}

fn input(seq: u32, tokens: &[i32]) -> SeqInput<'_> {
    SeqInput {
        seq,
        tokens,
        all_logits: false,
    }
}

#[test]
fn cpu_two_sequences_in_one_step_equal_each_alone() {
    let (a, b) = (prompt(1, 12), prompt(2, 30));
    let mut together = backend(512, 2, 512);
    assert_eq!((together.n_seq_max(), together.kv_capacity()), (2, 512));
    let out = together.decode(&[input(0, &a), input(1, &b)]).unwrap();
    let (la, lb) = (out.last(0).to_vec(), out.last(1).to_vec());
    for (seq, p, want) in [(0, &a, &la), (1, &b, &lb)] {
        let mut alone = backend(512, 1, 512);
        let l = alone.decode(&[input(0, p)]).unwrap().last(0).to_vec();
        close(want, &l, &format!("sequence {seq}"));
    }
    assert_eq!((together.seq_len(0), together.seq_len(1)), (12, 30));
    assert_eq!(together.kv_free(), 0, "two 256-row chunks of 512");
}

#[test]
fn cpu_a_step_over_the_cap_is_split_and_gives_the_same_rows() {
    let p = prompt(3, 100);
    let mut by_hand = backend(512, 1, 512);
    let mut split = backend(512, 1, 16);
    assert_eq!(split.max_step_tokens(), 16);
    // The same pieces, one decode call each, against one call the backend splits itself: the
    // pieces are the same batches, so the rows are bit-equal. (Against the unsplit prefill they
    // differ by chunking, as for any chunked prefill.)
    let mut want = Vec::new();
    for piece in p.chunks(16) {
        want = by_hand.decode(&[input(0, piece)]).unwrap().last(0).to_vec();
    }
    let got = split.decode(&[input(0, &p)]).unwrap().last(0).to_vec();
    assert_eq!(got, want);
    assert_eq!(split.seq_len(0), 100);
    // all_logits keeps one row per token across pieces.
    let t = &p[..40];
    let out = split
        .decode(&[SeqInput {
            seq: 0,
            tokens: t,
            all_logits: true,
        }])
        .unwrap();
    assert_eq!(out.rows(0).len(), 40 * out.n_vocab());
}

#[test]
fn cpu_kv_full_changes_nothing_and_a_freed_sequence_makes_room() {
    let mut b = backend(256, 2, 64);
    let p = prompt(4, 100);
    b.decode(&[input(0, &p)]).unwrap();
    assert_eq!(b.kv_free(), 0);
    let q = prompt(5, 5);
    let err = b.decode(&[input(1, &q)]).err().unwrap();
    assert!(matches!(err, BackendError::KvFull { seq: 1, .. }), "{err}");
    assert_eq!((b.seq_len(0), b.seq_len(1), b.kv_free()), (100, 0, 0));
    assert!(matches!(
        b.seq_reserve(1, 1),
        Err(BackendError::KvFull { .. })
    ));
    // A step that grows sequence 0 past its rows and needs a sequence-1 chunk too fails whole.
    let long = prompt(6, 200);
    assert!(b.decode(&[input(0, &long)]).is_err());
    assert_eq!(b.seq_len(0), 100);
    b.seq_rm(0, 0).unwrap();
    assert_eq!(b.kv_free(), 256);
    b.decode(&[input(1, &q)]).unwrap();
    assert_eq!(b.seq_len(1), 5);
    // Unknown sequence, twice in a step, empty tokens: Failed, lengths unchanged.
    for bad in [
        vec![input(2, &q)],
        vec![input(1, &q), input(1, &q)],
        vec![input(1, &[])],
    ] {
        assert!(matches!(b.decode(&bad), Err(BackendError::Failed(_))));
    }
    assert_eq!(b.seq_len(1), 5);
    // The pool is intact.
    b.decode(&[input(1, &[9])]).unwrap();
    assert_eq!(b.seq_len(1), 6);
}

#[test]
fn cpu_seq_cp_and_snapshot_keep_the_logits() {
    let mut b = backend(768, 3, 512);
    let p = prompt(7, 40);
    b.decode(&[input(0, &p)]).unwrap();
    b.seq_cp(0, 1, 40).unwrap();
    assert_eq!(b.seq_len(1), 40);
    let snap = b.seq_snapshot(0).unwrap();
    let tok = [11];
    let l0 = b.decode(&[input(0, &tok)]).unwrap().last(0).to_vec();
    let l1 = b.decode(&[input(1, &tok)]).unwrap().last(0).to_vec();
    close(&l1, &l0, "copy");
    b.seq_restore(2, &snap, 40).unwrap();
    let l2 = b.decode(&[input(2, &tok)]).unwrap().last(0).to_vec();
    close(&l2, &l0, "restored");
    assert!(b.seq_cp(0, 1, 5).is_err(), "dst must be empty");
    assert!(b.seq_cp(0, 5, 5).is_err(), "no such sequence");
}

fn greedy_params(max: u32) -> SamplingParams {
    SamplingParams {
        max_tokens: max,
        temperature: Some(0.0),
        ..SamplingParams::default()
    }
}

fn generate(s: &Scheduler, text: &str, max: u32) -> tokio::sync::mpsc::Receiver<GenEvent> {
    let id = s.models()[0].id.clone();
    let prompt = s.tokenize(&id, text).unwrap();
    s.generate(GenerateRequest {
        model: id,
        prompt,
        params: greedy_params(max),
        constraint: None,
        cancel: CancellationToken::new(),
    })
    .unwrap()
}

fn tokens(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Vec<u32> {
    let mut out = Vec::new();
    loop {
        match rx.blocking_recv().expect("the stream ends with Finished") {
            GenEvent::Token(t) => out.push(t),
            GenEvent::Finished { .. } => return out,
            GenEvent::Error(e) => panic!("engine error: {e}"),
        }
    }
}

fn sched(parallel: usize) -> Scheduler {
    let path = fixture();
    let mut cfg = SchedulerConfig::new(&path);
    cfg.parallel = parallel;
    Scheduler::start(cfg, move || load_moe(&path, &opts(1024, parallel))).expect("scheduler")
}

/// Greedy requests served concurrently by the MoE backend produce the tokens they produce one
/// at a time (ADR 0043's near-tie rule does not trigger on the fixture).
#[test]
fn cpu_scheduler_serves_concurrent_requests_like_one_at_a_time() {
    let texts = [
        "The quick brown fox",
        "Hello, world! How are you today?",
        "abcdefghij abcdefghij abcdefghij abcdefghij",
    ];
    let one = sched(1);
    let want: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| tokens(generate(&one, t, 12)))
        .collect();
    assert!(want.iter().any(|w| w.len() > 1), "{want:?}");
    one.shutdown().unwrap();
    let many = sched(3);
    let rxs: Vec<_> = texts.iter().map(|t| generate(&many, t, 12)).collect();
    let got: Vec<Vec<u32>> = rxs.into_iter().map(tokens).collect();
    assert_eq!(got, want);
    many.shutdown().unwrap();
}

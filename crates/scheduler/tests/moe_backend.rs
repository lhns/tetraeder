//! `MoeBackend` as a native `BatchBackend` over the builder's KV pool (ADR 0065, phase 4 C4) on
//! the tiny fixture, CPU only: several sequences per step, the step cap, `KvFull`, `seq_cp`,
//! snapshots and the scheduler serving concurrent requests with the same greedy tokens as alone.
//!
//! Hardware label: cpu.

#[allow(dead_code)]
#[path = "../../model/tests/common/synth.rs"]
mod synth;

use std::path::PathBuf;
use synth::{MoeSpec, quantized_moe};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};
use tetraeder_scheduler::{
    BackendError, BatchBackend, MoeBackend, Scheduler, SchedulerConfig, SeqInput, load_moe,
    load_moe_chunked,
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
    // The scheduler may plan past the chunk: decode splits.
    assert_eq!(split.max_step_tokens(), usize::MAX);
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

/// hardware: cpu. A step mixing decoders and a prompt chunk (run as separate model calls on the
/// CPU) gives the logits and lengths of each sequence decoded alone, rows in input order.
#[test]
fn cpu_mixed_step_equals_alone() {
    let hist: Vec<Vec<i32>> = (0..3).map(|s| prompt(s, 5 + s as usize)).collect();
    let next: Vec<Vec<i32>> = vec![vec![11], prompt(7, 3), vec![13], prompt(9, 100)];
    fn step(s: usize, t: &[i32]) -> SeqInput<'_> {
        SeqInput {
            seq: s as u32,
            tokens: t,
            all_logits: s == 1,
        }
    }
    let mut batch = backend(2048, 4, 512);
    for (s, h) in hist.iter().enumerate() {
        batch.decode(&[input(s as u32, h)]).unwrap();
    }
    let inputs: Vec<_> = next.iter().enumerate().map(|(s, t)| step(s, t)).collect();
    let out = batch.decode(&inputs).unwrap();
    let got: Vec<Vec<f32>> = (0..4).map(|i| out.rows(i).to_vec()).collect();
    for (s, t) in next.iter().enumerate() {
        let mut one = backend(256, 1, 512);
        if let Some(h) = hist.get(s) {
            one.decode(&[input(0, h)]).unwrap();
        }
        let out = one
            .decode(&[SeqInput {
                seq: 0,
                tokens: t,
                all_logits: s == 1,
            }])
            .unwrap();
        assert_eq!(out.rows(0).len(), got[s].len(), "sequence {s}");
        close(&got[s], out.rows(0), &format!("sequence {s}"));
    }
    assert_eq!(
        (0..4).map(|s| batch.seq_len(s)).collect::<Vec<_>>(),
        vec![6, 9, 8, 100]
    );
}

/// hardware: cpu. The prompt of a mixed step finds no rows: `KvFull` before any call, the decoders
/// are still at their lengths (a failure inside a later call goes through the same `restore`).
#[test]
fn cpu_mixed_step_kv_full_changes_nothing() {
    // Two runs of 256 rows: sequences 0 and 1 hold them; sequence 2 has nowhere to go.
    let mut b = backend(512, 3, 512);
    let (p0, p1) = (prompt(1, 10), prompt(2, 10));
    b.decode(&[input(0, &p0), input(1, &p1)]).unwrap();
    let big = prompt(3, 100);
    let err = b
        .decode(&[input(0, &[5]), input(1, &[6]), input(2, &big)])
        .err()
        .unwrap();
    assert!(matches!(err, BackendError::KvFull { seq: 2, .. }), "{err}");
    assert_eq!((b.seq_len(0), b.seq_len(1), b.seq_len(2)), (10, 10, 0));
    // Same lengths and the pool still works.
    b.decode(&[input(0, &[5]), input(1, &[6])]).unwrap();
    assert_eq!((b.seq_len(0), b.seq_len(1)), (11, 11));
}

/// hardware: cpu. The prompt call of a mixed step fails after the decoders' call ran (a token
/// outside the vocabulary): every input is back at its length and reservation, and the pool works.
#[test]
fn cpu_failure_in_the_second_call_restores_lengths_and_reservations() {
    let mut b = backend(1024, 4, 512);
    for s in 0..2 {
        b.decode(&[input(s, &prompt(s as i32, 10))]).unwrap();
    }
    b.seq_reserve(1, 300).unwrap();
    let before: Vec<_> = (0..4).map(|s| (b.seq_len(s), b.seq_reserved(s))).collect();
    let free = b.kv_free();
    let mut bad = prompt(3, 100);
    bad[50] = 1_000_000;
    let err = b
        .decode(&[input(0, &[5]), input(1, &[6]), input(2, &bad)])
        .err()
        .unwrap();
    assert!(matches!(err, BackendError::Failed(_)), "{err}");
    let after: Vec<_> = (0..4).map(|s| (b.seq_len(s), b.seq_reserved(s))).collect();
    assert_eq!((after, b.kv_free()), (before, free));
    b.decode(&[input(0, &[5]), input(1, &[6])]).unwrap();
    assert_eq!((b.seq_len(0), b.seq_len(1)), (11, 11));
}

/// hardware: cpu. A prompt chunk comes before a decoder in input order: the two run as separate
/// calls and every input gets the logits it gets alone, rows in input order.
#[test]
fn cpu_prompt_chunk_before_a_decoder_equals_alone() {
    let hist: Vec<Vec<i32>> = (0..4).map(|s| prompt(s, 6 + s as usize)).collect();
    let next: Vec<Vec<i32>> = vec![prompt(11, 100), vec![21], prompt(12, 70), vec![22]];
    let mut batch = backend(2048, 4, 512);
    for (s, h) in hist.iter().enumerate() {
        batch.decode(&[input(s as u32, h)]).unwrap();
    }
    let inputs: Vec<_> = next
        .iter()
        .enumerate()
        .map(|(s, t)| SeqInput {
            seq: s as u32,
            tokens: t,
            all_logits: s == 2,
        })
        .collect();
    let out = batch.decode(&inputs).unwrap();
    let got: Vec<Vec<f32>> = (0..4).map(|i| out.rows(i).to_vec()).collect();
    for (s, t) in next.iter().enumerate() {
        let mut one = backend(512, 1, 512);
        one.decode(&[input(0, &hist[s])]).unwrap();
        let out = one
            .decode(&[SeqInput {
                seq: 0,
                tokens: t,
                all_logits: s == 2,
            }])
            .unwrap();
        assert_eq!(out.rows(0).len(), got[s].len(), "sequence {s}");
        close(&got[s], out.rows(0), &format!("sequence {s}"));
    }
    assert_eq!(
        (0..4).map(|s| batch.seq_len(s)).collect::<Vec<_>>(),
        vec![106, 8, 78, 10]
    );
}

/// hardware: cpu. A step that grows sequence 0 past its run moves it (the pool copies its rows to
/// a run with room), then finds no rows for sequence 2 and fails: every sequence is back at its
/// length and reservation, the moved run stays where it moved, and the next step gives the
/// logits of a backend that never failed.
#[test]
fn cpu_failed_step_after_a_moved_reservation_rolls_back() {
    let (p0, p1) = (prompt(1, 10), prompt(2, 10));
    let mut b = backend(2048, 3, 512);
    let mut reference = backend(2048, 3, 512);
    for x in [&mut b, &mut reference] {
        x.decode(&[input(0, &p0), input(1, &p1)]).unwrap();
    }
    let (free, reserved) = (b.kv_free(), (b.seq_reserved(0), b.seq_reserved(1)));
    // Sequence 0 needs 310 rows: its run (rows 0..256) is followed by sequence 1's, so it moves
    // to a run of 1536 rows (rows 512..2048). Sequence 2 then needs 512 rows and the only gap
    // left is 256 rows.
    let grow = prompt(3, 300);
    let err = b
        .decode(&[input(0, &grow), input(2, &prompt(8, 300))])
        .err()
        .unwrap();
    assert!(matches!(err, BackendError::KvFull { seq: 2, .. }), "{err}");
    assert_eq!((b.seq_len(0), b.seq_len(1), b.seq_len(2)), (10, 10, 0));
    assert_eq!((b.seq_reserved(0), b.seq_reserved(1)), reserved);
    assert_eq!((b.seq_reserved(2), b.kv_free()), (0, free));
    // The moved rows are intact and the pool works: same logits as the reference.
    let step = [input(0, &[5]), input(1, &[6])];
    let want: Vec<Vec<f32>> = {
        let out = reference.decode(&step).unwrap();
        (0..2).map(|i| out.rows(i).to_vec()).collect()
    };
    let out = b.decode(&step).unwrap();
    for (i, w) in want.iter().enumerate() {
        close(out.rows(i), w, &format!("sequence {i}"));
    }
}

/// hardware: cpu. The same move, then a failure inside the model call (a token outside the
/// vocabulary, after the decoders' call ran): the decoder is back at its length too.
#[test]
fn cpu_failure_in_a_call_after_a_moved_reservation_rolls_back() {
    let (p0, p1) = (prompt(1, 10), prompt(2, 10));
    let mut b = backend(2048, 2, 512);
    b.decode(&[input(0, &p0), input(1, &p1)]).unwrap();
    let before: Vec<_> = (0..2).map(|s| (b.seq_len(s), b.seq_reserved(s))).collect();
    let free = b.kv_free();
    let mut bad = prompt(4, 300);
    bad[200] = 1_000_000;
    assert!(matches!(
        b.decode(&[input(1, &[8]), input(0, &bad)]),
        Err(BackendError::Failed(_))
    ));
    let after: Vec<_> = (0..2).map(|s| (b.seq_len(s), b.seq_reserved(s))).collect();
    assert_eq!((after, b.kv_free()), (before, free));
    b.decode(&[input(0, &[5]), input(1, &[6])]).unwrap();
    assert_eq!((b.seq_len(0), b.seq_len(1)), (11, 11));
}

/// hardware: cpu. The CPU hot set comes from the routing of a prompt chunk, not of a group of
/// decoders that happens to hold more than one token (the first call of a busy server).
#[test]
fn cpu_hot_set_is_built_by_a_prompt_chunk_not_a_decoder_group() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "moe-backend-hot");
    let o = LoadOptions {
        n_ctx: 1024,
        n_seq_max: 4,
        expert_slots: Some(8),
        ram_slots: Some(32),
        miss: MissPolicy::Cpu,
        ..LoadOptions::new(BackendChoice::Cpu { threads: Some(2) })
    };
    let uploads = |b: &MoeBackend| b.model().expert_stats().map_or(0, |s| s.io.uploads);
    // A long prompt chunk and a short prompt (under 64 tokens, still a prefill).
    for n_prompt in [100, 10] {
        let mut b = MoeBackend::new(Qwen3Moe::load(&path, &o).unwrap(), false);
        for round in 0..2 {
            let toks: Vec<[i32; 1]> = (0..4).map(|s| [10 + 10 * round + s]).collect();
            let inputs: Vec<_> = toks
                .iter()
                .enumerate()
                .map(|(s, t)| input(s as u32, t))
                .collect();
            b.decode(&inputs).unwrap();
            assert_eq!(uploads(&b), 0, "four decoders are no prefill");
        }
        // Decoders and a prompt in one step: the prompt builds the hot set.
        let new = prompt(5, n_prompt);
        b.seq_rm(3, 0).unwrap();
        b.decode(&[input(0, &[30]), input(1, &[31]), input(3, &new)])
            .unwrap();
        assert!(
            uploads(&b) > 0,
            "a prompt of {n_prompt} tokens built the hot set"
        );
    }
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
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("engine error: {e}"),
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

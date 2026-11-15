//! `DenseBackend` (ADR 0065, 0066) through the `BatchBackend` contract on the tiny dense
//! fixture: an error changes nothing (lengths and free rows), `KvFull` is typed, a batch equals
//! its sequences alone.
//!
//! Hardware label: `cpu_*`.

use proptest::prelude::*;
use std::path::PathBuf;
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};
use tetraeder_plan::KvFormat;
use tetraeder_scheduler::{BackendError, BatchBackend, DenseBackend, SeqId, SeqInput};

const SEQS: u32 = 4;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-f16.gguf")
}

fn backend(pool: usize, n_seq_max: usize) -> DenseBackend {
    let o = DenseOptions {
        n_ctx_pool: pool,
        n_seq_max,
        kv: KvFormat::F16,
        threads: Some(2),
        ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
    };
    DenseBackend::new(Qwen3::load(fixture(), &o).unwrap(), false)
}

fn lens(b: &DenseBackend) -> Vec<usize> {
    (0..SEQS).map(|s| b.seq_len(s)).collect()
}

fn tokens(seed: usize, n: usize) -> Vec<i32> {
    (0..n).map(|k| ((seed * 31 + k * 7) % 200) as i32).collect()
}

fn input(seq: SeqId, tokens: &[i32], all: bool) -> SeqInput<'_> {
    SeqInput {
        seq,
        tokens,
        all_logits: all,
    }
}

/// hardware: cpu. Four sequences decoded in one step give the logits of each decoded alone.
#[test]
fn cpu_batch_equals_alone() {
    let toks: Vec<Vec<i32>> = (0..4).map(|s| tokens(s, 3 + 2 * s)).collect();
    let mut batch = backend(1024, 4);
    let inputs: Vec<_> = toks
        .iter()
        .enumerate()
        .map(|(s, t)| input(s as u32, t, s % 2 == 0))
        .collect();
    let out = batch.decode(&inputs).unwrap();
    let got: Vec<Vec<f32>> = (0..4).map(|i| out.rows(i).to_vec()).collect();
    for (s, t) in toks.iter().enumerate() {
        let mut one = backend(256, 1);
        let out = one.decode(&[input(0, t, s % 2 == 0)]).unwrap();
        let worst = out
            .rows(0)
            .iter()
            .zip(&got[s])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(out.rows(0).len(), got[s].len());
        // Other row counts take other matmul kernels (ADR 0043): not bitwise.
        let top = out.rows(0).iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(worst <= 1e-4f32.max(1e-3 * top), "sequence {s}: {worst:e}");
    }
    assert_eq!(lens(&batch), vec![3, 5, 7, 9]);
}

/// hardware: cpu. A step mixing decoders and a prompt chunk (run as separate model calls on the
/// CPU) gives the logits and lengths of each sequence decoded alone.
#[test]
fn cpu_mixed_step_equals_alone() {
    let hist: Vec<Vec<i32>> = (0..3).map(|s| tokens(s, 5 + s)).collect();
    let next: Vec<Vec<i32>> = vec![vec![11], tokens(7, 3), vec![13], tokens(9, 100)];
    let mut batch = backend(2048, 4);
    for (s, h) in hist.iter().enumerate() {
        batch.decode(&[input(s as u32, h, false)]).unwrap();
    }
    let inputs: Vec<_> = next
        .iter()
        .enumerate()
        .map(|(s, t)| input(s as u32, t, s == 1))
        .collect();
    let out = batch.decode(&inputs).unwrap();
    let got: Vec<Vec<f32>> = (0..4).map(|i| out.rows(i).to_vec()).collect();
    for (s, t) in next.iter().enumerate() {
        let mut one = backend(256, 1);
        if let Some(h) = hist.get(s) {
            one.decode(&[input(0, h, false)]).unwrap();
        }
        let out = one.decode(&[input(0, t, s == 1)]).unwrap();
        assert_eq!(out.rows(0).len(), got[s].len());
        let worst = out
            .rows(0)
            .iter()
            .zip(&got[s])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top = out.rows(0).iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(worst <= 1e-4f32.max(1e-3 * top), "sequence {s}: {worst:e}");
    }
    assert_eq!(lens(&batch), vec![6, 9, 8, 100]);
}

/// hardware: cpu. `KvFull` names the shortfall and changes nothing, also when an earlier input of
/// the same decode could be served.
#[test]
fn cpu_kv_full_changes_nothing() {
    // Two runs of 256 rows: sequence 0 holds the first, sequence 1 the second.
    let mut b = backend(512, 4);
    let t = tokens(1, 10);
    b.decode(&[input(0, &t, false), input(1, &t, false)])
        .unwrap();
    let free = b.kv_free();
    assert_eq!(free, 0);
    // Sequence 2 needs a run and none is free; sequence 0 would have grown within its run.
    let err = b
        .decode(&[input(0, &t, false), input(2, &t, false)])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, BackendError::KvFull { seq: 2, .. }), "{err}");
    assert_eq!((lens(&b), b.kv_free()), (vec![10, 10, 0, 0], free));
    // Growing sequence 1 past its run cannot move it anywhere.
    b.seq_reserve(1, 256).unwrap();
    assert!(matches!(
        b.seq_reserve(1, 300),
        Err(BackendError::KvFull { seq: 1, .. })
    ));
    assert!(matches!(
        b.seq_reserve(0, 100_000),
        Err(BackendError::KvFull { .. })
    ));
    // A copy needs rows too.
    assert!(matches!(
        b.seq_cp(0, 3, 10),
        Err(BackendError::KvFull { seq: 3, .. })
    ));
    assert_eq!((lens(&b), b.kv_free()), (vec![10, 10, 0, 0], free));
    // Freeing a sequence makes room.
    b.seq_rm(1, 0).unwrap();
    b.seq_cp(0, 3, 10).unwrap();
    assert_eq!(b.seq_len(3), 10);
}

/// hardware: cpu. A failed decode gives back the rows it reserved, also for a sequence that
/// holds positions and grew inside its slack: `reserved` and the free rows are as before.
#[test]
fn cpu_failed_decode_restores_reservations() {
    let mut b = backend(1280, 4);
    b.seq_reserve(0, 512).unwrap(); // rows 0..512
    b.seq_reserve(1, 256).unwrap(); // 512..768
    b.seq_reserve(3, 256).unwrap(); // 768..1024
    b.seq_rm(1, 0).unwrap(); // 512..768 free again
    let t = tokens(2, 100);
    b.decode(&[input(0, &t, false)]).unwrap();
    let (free, reserved) = (b.kv_free(), b.model().pool().reserved(0));
    assert_eq!((free, reserved), (512, 512));
    // Sequence 0 would grow to 768 rows in place, then sequence 2 finds 256 free rows, not 512.
    let (grow, other) = (tokens(3, 450), tokens(4, 300));
    let err = b
        .decode(&[input(0, &grow, false), input(2, &other, false)])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, BackendError::KvFull { seq: 2, .. }), "{err}");
    assert_eq!(b.seq_len(0), 100);
    assert_eq!(b.model().pool().reserved(0), reserved);
    assert_eq!(b.kv_free(), free);
    assert_eq!(b.model().pool().reserved(2), 0);
}

/// hardware: cpu. On the CPU a step with decoders and a prompt chunk is two model calls. When the
/// second fails, every input goes back to its length and reservation from before the call: the
/// decoders of the first call (which did not grow, one of them empty with a reservation), too.
#[test]
fn cpu_failed_second_group_restores_every_input() {
    let mut b = backend(2048, 4);
    b.seq_reserve(0, 600).unwrap(); // 3 blocks, slack beyond the 5 positions
    b.seq_reserve(2, 256).unwrap(); // reserved, empty
    b.decode(&[
        input(0, &tokens(1, 5), false),
        input(1, &tokens(2, 5), false),
    ])
    .unwrap();
    let snap = |b: &DenseBackend| {
        (
            lens(b),
            (0..SEQS).map(|s| b.seq_reserved(s)).collect::<Vec<_>>(),
            b.kv_free(),
        )
    };
    let before = snap(&b);
    assert_eq!(before.1[0], 768);
    assert_eq!(before.1[2], 256);
    let (d0, d1, d2) = (tokens(3, 1), tokens(4, 1), tokens(5, 1));
    let mut chunk = tokens(6, 100);
    chunk[40] = -1; // outside the vocabulary: the model rejects the chunk's call
    let err = b
        .decode(&[
            input(0, &d0, false),
            input(1, &d1, false),
            input(2, &d2, false),
            input(3, &chunk, false),
        ])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, BackendError::Failed(_)), "{err}");
    assert_eq!(snap(&b), before);
    // Still usable.
    chunk[40] = 7;
    b.decode(&[input(0, &d0, false), input(3, &chunk, false)])
        .unwrap();
    assert_eq!((b.seq_len(0), b.seq_len(3)), (6, 100));
}

#[derive(Debug, Clone)]
enum Op {
    Reserve { seq: SeqId, n: usize },
    Rm { seq: SeqId, keep: usize },
    Cp { src: SeqId, dst: SeqId, n: usize },
    Decode(Vec<(SeqId, usize, bool)>),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..SEQS, 0usize..700).prop_map(|(seq, n)| Op::Reserve { seq, n }),
        (0..SEQS, 0usize..40).prop_map(|(seq, keep)| Op::Rm { seq, keep }),
        (0..SEQS, 0..SEQS, 0usize..40).prop_map(|(src, dst, n)| Op::Cp { src, dst, n }),
        prop::collection::vec((0..SEQS, 1usize..12, any::<bool>()), 1..=4).prop_map(Op::Decode),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]

    /// P2 and P3 on the real backend: after any operation the pool is not over-subscribed, and
    /// an operation that fails changes neither a length nor the free rows.
    #[test]
    fn cpu_contract_holds(ops in prop::collection::vec(op_strategy(), 1..14)) {
        let mut b = backend(1024, SEQS as usize);
        let nv = b.n_vocab();
        for (step, op) in ops.iter().enumerate() {
            let (l0, f0) = (lens(&b), b.kv_free());
            let ok = match op {
                Op::Reserve { seq, n } => b.seq_reserve(*seq, *n).is_ok(),
                Op::Rm { seq, keep } => b.seq_rm(*seq, *keep).is_ok(),
                Op::Cp { src, dst, n } => {
                    let n = n % (b.seq_len(*src) + 1);
                    b.seq_cp(*src, *dst, n).is_ok()
                }
                Op::Decode(items) => {
                    let mut seen = Vec::new();
                    let toks: Vec<(SeqId, Vec<i32>, bool)> = items
                        .iter()
                        .filter(|(s, ..)| !seen.contains(s) && { seen.push(*s); true })
                        .map(|&(s, n, all)| (s, tokens(step * 5 + s as usize, n), all))
                        .collect();
                    let inputs: Vec<_> = toks.iter().map(|(s, t, a)| input(*s, t, *a)).collect();
                    match b.decode(&inputs) {
                        Ok(o) => {
                            assert_eq!(o.len(), inputs.len());
                            for (i, inp) in inputs.iter().enumerate() {
                                let rows = if inp.all_logits { inp.tokens.len() } else { 1 };
                                assert_eq!(o.rows(i).len(), rows * nv);
                                assert!(o.rows(i).iter().all(|x| x.is_finite()));
                            }
                            for (s, t, _) in &toks {
                                assert_eq!(b.seq_len(*s), l0[*s as usize] + t.len());
                            }
                            true
                        }
                        Err(_) => false,
                    }
                }
            };
            if !ok {
                prop_assert_eq!(lens(&b), l0, "lengths changed by a failed {:?}", op);
                prop_assert_eq!(b.kv_free(), f0, "free rows changed by a failed {:?}", op);
            }
            prop_assert!(b.kv_free() <= b.kv_capacity());
            let total: usize = lens(&b).iter().sum();
            prop_assert!(total <= b.kv_capacity());
        }
    }
}

/// hardware: cpu. A step that grows sequence 0 past its run moves it (the pool copies its rows to
/// a run with room), then finds no rows for sequence 2 and fails: every sequence is back at its
/// length and reservation, the moved run stays where it moved, and the next step gives the
/// logits of a backend that never failed.
#[test]
fn cpu_failed_step_after_a_moved_reservation_rolls_back() {
    let (t0, t1) = (tokens(1, 10), tokens(2, 10));
    let mut b = backend(2048, 4);
    let mut reference = backend(2048, 4);
    for x in [&mut b, &mut reference] {
        x.decode(&[input(0, &t0, false), input(1, &t1, false)])
            .unwrap();
    }
    let (free, reserved) = (b.kv_free(), (b.seq_reserved(0), b.seq_reserved(1)));
    // Sequence 0 needs 310 rows: its run (rows 0..256) is followed by sequence 1's, so it moves
    // to a run of 1536 rows (rows 512..2048). Sequence 2 then needs 512 rows and the only gap
    // left is 256 rows.
    let grow = tokens(3, 300);
    let big = tokens(8, 300);
    let err = b
        .decode(&[input(0, &grow, false), input(2, &big, false)])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, BackendError::KvFull { seq: 2, .. }), "{err}");
    assert_eq!(lens(&b)[..3], [10, 10, 0]);
    assert_eq!((b.seq_reserved(0), b.seq_reserved(1)), reserved);
    assert_eq!((b.seq_reserved(2), b.kv_free()), (0, free));
    // The moved rows are intact and the pool works: same logits as the reference.
    let step = [input(0, &[5], false), input(1, &[6], false)];
    let want: Vec<Vec<f32>> = {
        let out = reference.decode(&step).unwrap();
        (0..2).map(|i| out.rows(i).to_vec()).collect()
    };
    let out = b.decode(&step).unwrap();
    for (i, w) in want.iter().enumerate() {
        let worst = out
            .rows(i)
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top = w.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(worst <= 1e-4f32.max(1e-3 * top), "sequence {i}: {worst:e}");
    }
}

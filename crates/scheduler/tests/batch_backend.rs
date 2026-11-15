//! The `BatchBackend` contract (ADR 0065, "Property tests, first") on `FakeKv`.
//!
//! Covers P2 (rows disjoint, `seq_cp` destination owns equal contents), P3 (reservations never
//! exceed capacity, `KvFull` changes nothing) and "a failed decode leaves every sequence at its
//! length before the step". P1, P4 and P5 need the scheduler loop and come with it (C2b).

mod fake;

use fake::{FakeKv, logits_for};
use proptest::prelude::*;
use tetraeder_scheduler::{BackendError, BatchBackend, SeqId, SeqInput};

const N_VOCAB: usize = 8;
const SEQ_IDS: u32 = 5; // one more than the 4 slots: the last id does not exist

#[derive(Debug, Clone)]
enum Op {
    Reserve { seq: SeqId, n: usize },
    Rm { seq: SeqId, keep: usize },
    Cp { src: SeqId, dst: SeqId, n: usize },
    Decode(Vec<(SeqId, usize, bool)>),
}

fn op_strategy(ids: u32, max_inputs: usize) -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..ids, 0usize..70).prop_map(|(seq, n)| Op::Reserve { seq, n }),
        (0..ids, 0usize..40).prop_map(|(seq, keep)| Op::Rm { seq, keep }),
        (0..ids, 0..ids, 0usize..40).prop_map(|(src, dst, n)| Op::Cp { src, dst, n }),
        prop::collection::vec((0..ids, 1usize..12, any::<bool>()), 1..=max_inputs)
            .prop_map(Op::Decode),
    ]
}

struct State {
    lens: Vec<usize>,
    free: usize,
}

fn state<B: BatchBackend>(b: &B) -> State {
    State {
        lens: (0..SEQ_IDS).map(|s| b.seq_len(s)).collect(),
        free: b.kv_free(),
    }
}

/// Runs `ops` and checks the contract that holds for every backend after every call. `after`
/// adds backend-specific checks (row sets).
fn run<B: BatchBackend>(b: &mut B, ops: &[Op], mut after: impl FnMut(&B, &Op, bool)) {
    for (step, op) in ops.iter().enumerate() {
        let before = state(b);
        let ok = match op {
            Op::Reserve { seq, n } => {
                let r = b.seq_reserve(*seq, *n);
                settle(b, &before, r)
            }
            Op::Rm { seq, keep } => {
                let r = b.seq_rm(*seq, *keep);
                settle(b, &before, r)
            }
            Op::Cp { src, dst, n } => {
                let n = n % (b.seq_len(*src) + 1);
                let r = b.seq_cp(*src, *dst, n);
                settle(b, &before, r)
            }
            Op::Decode(items) => {
                // Distinct sequences, tokens a function of the step.
                let mut seen = Vec::new();
                let toks: Vec<(SeqId, Vec<i32>, bool)> = items
                    .iter()
                    .filter(|(s, ..)| {
                        !seen.contains(s) && {
                            seen.push(*s);
                            true
                        }
                    })
                    .map(|&(s, n, all)| {
                        let t = (0..n).map(|k| ((step * 31 + k * 7 + s as usize) % 50) as i32);
                        (s, t.collect(), all)
                    })
                    .collect();
                let inputs: Vec<SeqInput<'_>> = toks
                    .iter()
                    .map(|(seq, tokens, all)| SeqInput {
                        seq: *seq,
                        tokens,
                        all_logits: *all,
                    })
                    .collect();
                let res = b.decode(&inputs).map(|o| {
                    assert_eq!(o.len(), inputs.len());
                    for (i, inp) in inputs.iter().enumerate() {
                        let rows = if inp.all_logits { inp.tokens.len() } else { 1 };
                        assert_eq!(o.rows(i).len(), rows * N_VOCAB, "rows of input {i}");
                        assert_eq!(o.last(i), &o.rows(i)[(rows - 1) * N_VOCAB..]);
                    }
                });
                let ok = settle(b, &before, res);
                if ok {
                    for inp in &inputs {
                        let s = inp.seq as usize;
                        assert_eq!(b.seq_len(inp.seq), before.lens[s] + inp.tokens.len());
                    }
                }
                ok
            }
        };
        // Always: the pool is never over-subscribed and sequences stay within their context.
        assert!(b.kv_free() <= b.kv_capacity());
        let total: usize = (0..SEQ_IDS).map(|s| b.seq_len(s)).sum();
        assert!(total <= b.kv_capacity(), "written rows beyond capacity");
        for s in 0..SEQ_IDS {
            assert!(b.seq_len(s) <= b.n_ctx_seq());
        }
        after(b, op, ok);
    }
}

/// An error changes nothing (lengths and free rows), `KvFull` included.
fn settle<B: BatchBackend>(b: &B, before: &State, r: Result<(), BackendError>) -> bool {
    match r {
        Ok(()) => true,
        Err(e) => {
            let now = state(b);
            assert_eq!(now.lens, before.lens, "lengths changed by {e}");
            assert_eq!(now.free, before.free, "free rows changed by {e}");
            false
        }
    }
}

/// P2 and P3 on `FakeKv`'s row sets.
fn check_rows(b: &FakeKv, op: &Op, ok: bool) {
    let mut all: Vec<usize> = Vec::new();
    for s in 0..4 {
        let rows = b.rows_of(s);
        assert!(b.seq_len(s) <= rows.len(), "written beyond reservation");
        for &r in rows {
            assert!(!b.is_free_row(r), "reserved row {r} also free");
        }
        all.extend_from_slice(rows);
    }
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "a row belongs to two sequences (P2)");
    assert_eq!(n, b.reserved_total());
    assert!(b.reserved_total() <= b.kv_capacity(), "over-reserved (P3)");
    for s in 0..4 {
        let toks: Vec<Option<i32>> = b.tokens_of(s).iter().map(|&t| Some(t)).collect();
        assert_eq!(b.contents(s), toks, "pool rows hold the sequence's tokens");
    }
    if let (Op::Cp { src, dst, .. }, true) = (op, ok) {
        let n = b.seq_len(*dst);
        assert_eq!(b.contents(*dst), b.contents(*src)[..n], "copy content");
        let (rs, rd) = (b.rows_of(*src), b.rows_of(*dst));
        assert!(rd.iter().all(|r| !rs.contains(r)), "copy shares rows (P2)");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn fake_kv_contract(
        ops in prop::collection::vec(op_strategy(SEQ_IDS, 4), 1..60),
        capacity in 8usize..160,
        kv_full_at in prop::option::of(1usize..120),
        fail_every in prop::option::of(1u32..5),
    ) {
        let mut b = FakeKv::new(N_VOCAB, 4, 64, capacity)
            .with_kv_full_at(kv_full_at)
            .with_fail_every(fail_every);
        run(&mut b, &ops, check_rows);
    }
}

#[test]
fn fake_kv_kv_full_changes_nothing_and_names_the_shortfall() {
    let mut b = FakeKv::new(N_VOCAB, 4, 64, 10);
    b.seq_reserve(0, 6).unwrap();
    let err = b.seq_reserve(1, 5).unwrap_err();
    assert_eq!(
        err,
        BackendError::KvFull {
            seq: 1,
            needed: 5,
            free: 4
        }
    );
    assert_eq!(b.kv_free(), 4);
    // A decode needing more than is free fails whole: sequence 0 is not partly advanced.
    let t = [1, 2, 3, 4, 5, 6, 7];
    let inputs = [
        SeqInput {
            seq: 0,
            tokens: &t[..2],
            all_logits: false,
        },
        SeqInput {
            seq: 1,
            tokens: &t,
            all_logits: false,
        },
    ];
    assert!(matches!(
        b.decode(&inputs),
        Err(BackendError::KvFull { .. })
    ));
    assert_eq!((b.seq_len(0), b.seq_len(1), b.kv_free()), (0, 0, 4));
}

#[test]
fn fake_kv_logits_do_not_depend_on_the_batch() {
    let a = [3, 1, 4, 1, 5];
    let c = [9, 2, 6];
    let mut together = FakeKv::new(N_VOCAB, 4, 64, 64);
    let out = together
        .decode(&[
            SeqInput {
                seq: 0,
                tokens: &a,
                all_logits: true,
            },
            SeqInput {
                seq: 2,
                tokens: &c,
                all_logits: false,
            },
        ])
        .unwrap();
    let (ra, rc) = (out.rows(0).to_vec(), out.last(1).to_vec());
    let mut alone = FakeKv::new(N_VOCAB, 4, 64, 64);
    let out = alone
        .decode(&[SeqInput {
            seq: 3,
            tokens: &c,
            all_logits: false,
        }])
        .unwrap();
    assert_eq!(out.last(0), rc);
    assert_eq!(rc, logits_for(N_VOCAB, &c));
    assert_eq!(ra[4 * N_VOCAB..], logits_for(N_VOCAB, &a));
}

#[test]
fn fake_kv_failed_decode_rolls_back_a_partly_applied_step() {
    let mut b = FakeKv::new(N_VOCAB, 4, 64, 64).with_fail_every(Some(2));
    let t = [1, 2, 3];
    let step = |b: &mut FakeKv| {
        b.decode(&[
            SeqInput {
                seq: 0,
                tokens: &t,
                all_logits: false,
            },
            SeqInput {
                seq: 1,
                tokens: &t,
                all_logits: true,
            },
        ])
        .map(|_| ())
    };
    step(&mut b).unwrap();
    let free = b.kv_free();
    assert!(matches!(step(&mut b), Err(BackendError::Failed(_))));
    assert_eq!((b.seq_len(0), b.seq_len(1)), (3, 3));
    assert_eq!(b.kv_free(), free);
    step(&mut b).unwrap();
    assert_eq!((b.seq_len(0), b.seq_len(1)), (6, 6));
}

#[test]
fn fake_kv_seq_cp_then_diverge() {
    let mut b = FakeKv::new(N_VOCAB, 4, 64, 64);
    b.decode(&[SeqInput {
        seq: 0,
        tokens: &[1, 2, 3, 4],
        all_logits: false,
    }])
    .unwrap();
    b.seq_cp(0, 1, 3).unwrap();
    assert_eq!(b.contents(1), b.contents(0)[..3]);
    // The copy is independent: cutting the source does not touch it.
    b.seq_rm(0, 0).unwrap();
    assert_eq!(b.tokens_of(1), [1, 2, 3]);
    // dst must be empty.
    assert!(b.seq_cp(1, 1, 1).is_err());
    b.seq_cp(1, 2, 3).unwrap();
    assert!(b.seq_cp(1, 2, 1).is_err());
}

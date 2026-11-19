//! With one sequence at row 0 the plan is what the single-sequence builder computes today
//! (`rows == pos`, `n_kv = pad(len + n, 256)`, the causal mask), which keeps C4 bit-equal.

use tetraeder_kvcache::{
    Chunked, F16_NEG_INF, KvError, KvPool, Layout, Paged, ROW_ALIGN, RowCopy, StepInput,
};

fn pad(n: usize) -> usize {
    n.div_ceil(ROW_ALIGN) * ROW_ALIGN
}

/// The mask loop of `Qwen3Moe::run` (crates/model/src/qwen3moe.rs), the oracle.
fn causal_mask(n_past: usize, n: usize, n_kv: usize) -> Vec<u16> {
    let mut mask = vec![0u16; n_kv * n];
    for i in 0..n {
        for j in (n_past + i + 1)..n_kv {
            mask[i * n_kv + j] = F16_NEG_INF;
        }
    }
    mask
}

fn layouts(capacity: usize) -> Vec<KvPool> {
    vec![
        KvPool::new(Box::new(Chunked::new(capacity))),
        KvPool::new(Box::new(Paged::new(capacity, 256))),
    ]
}

#[test]
fn one_sequence_matches_the_causal_oracle() {
    for mut pool in layouts(4096) {
        let name = pool.layout_name();
        let mut len = 0;
        for n in [37usize, 1, 1, 300, 1, 5] {
            assert!(pool.reserve(0, len + n).unwrap().is_empty());
            let input = [StepInput {
                seq: 0,
                n_tokens: n,
                all_logits: n == 5,
            }];
            let plan = pool.plan_step(&input).unwrap();
            let pos: Vec<i32> = (len..len + n).map(|p| p as i32).collect();
            assert_eq!(plan.pos, pos, "{name}");
            assert_eq!(plan.rows, pos, "{name}: rows == pos");
            assert_eq!(plan.n_kv, pad(len + n), "{name}");
            assert_eq!(plan.mask, causal_mask(len, n, plan.n_kv), "{name}");
            let outputs: Vec<i32> = if n == 5 {
                (0..5).collect()
            } else {
                vec![n as i32 - 1]
            };
            assert_eq!(plan.outputs, outputs, "{name}");
            pool.commit(&input);
            len += n;
        }
        assert_eq!(pool.seq_len(0), len);
    }
}

#[test]
fn two_sequences_read_only_their_own_rows() {
    let mut pool = KvPool::new(Box::new(Chunked::new(2048)));
    pool.reserve(1, 300).unwrap();
    pool.reserve(2, 10).unwrap();
    let a = [StepInput {
        seq: 1,
        n_tokens: 2,
        all_logits: false,
    }];
    pool.plan_step(&a).unwrap();
    pool.commit(&a);
    let both = [
        StepInput {
            seq: 1,
            n_tokens: 1,
            all_logits: false,
        },
        StepInput {
            seq: 2,
            n_tokens: 3,
            all_logits: false,
        },
    ];
    let plan = pool.plan_step(&both).unwrap();
    // seq 1 at rows 0..300, seq 2 at 512..768 (aligned, after the 300-row run's 512)
    assert_eq!(plan.pos, vec![2, 0, 1, 2]);
    assert_eq!(plan.rows, vec![2, 512, 513, 514]);
    assert_eq!(plan.n_kv, 768);
    let open = |t: usize| -> Vec<usize> {
        (0..plan.n_kv)
            .filter(|&j| plan.mask[t * plan.n_kv + j] == 0)
            .collect()
    };
    assert_eq!(open(0), vec![0, 1, 2]);
    assert_eq!(open(1), vec![512]);
    assert_eq!(open(3), vec![512, 513, 514]);
}

#[test]
fn errors() {
    let mut pool = KvPool::new(Box::new(Chunked::new(512)));
    let step = [StepInput {
        seq: 9,
        n_tokens: 1,
        all_logits: false,
    }];
    assert!(matches!(pool.plan_step(&step), Err(KvError::UnknownSeq(9))));
    pool.reserve(9, 10).unwrap();
    pool.commit(&step);
    pool.reserve(9, 10).unwrap();
    // reservation is 256 rows: a step of 300 tokens is not covered
    let big = [StepInput {
        seq: 9,
        n_tokens: 300,
        all_logits: false,
    }];
    assert!(matches!(pool.plan_step(&big), Err(KvError::Invalid(_))));
    assert!(matches!(pool.reserve(8, 1000), Err(KvError::Full { .. })));
    assert!(matches!(pool.seq_cp(9, 9, 1), Err(KvError::Invalid(_))));
    assert!(matches!(pool.seq_cp(7, 8, 1), Err(KvError::UnknownSeq(7))));
}

#[test]
fn chunked_grows_in_place_then_moves_with_grow() {
    let mut c = Chunked::new(8192);
    c.reserve(1, 256).unwrap();
    c.reserve(2, 256).unwrap();
    // seq 1 is blocked by seq 2: it moves to a run of 512 + 1024 rows after it
    let moves = c.reserve(1, 512).unwrap();
    assert_eq!(moves.len(), 1);
    assert_eq!((moves[0].src, moves[0].n, moves[0].dst), (0, 256, 512));
    assert_eq!(c.reserved(1), 512 + 1024);
    // the hole at 0 is used first by a small sequence
    c.reserve(3, 256).unwrap();
    assert_eq!(c.row(3, 0), 0);
    // seq 2 can grow in place into nothing: its neighbour starts at 512
    assert_eq!(c.reserve(2, 300).unwrap().len(), 1);
}

/// ttft4-20261119 #39: a lone sequence whose run neither grows in place nor fits a gap slides
/// down over itself into the free rows below it, in chunks no longer than the slide.
#[test]
fn chunked_slides_a_run_down_when_no_gap_fits() {
    let mut c = Chunked::new(3584);
    c.reserve(1, 1024).unwrap();
    c.reserve(2, 2048).unwrap();
    c.shrink(1, 0);
    // seq 2 at 1024..3072: 512 rows after it, 1024 below it, no gap of 3072.
    let moves = c.reserve(2, 3072).unwrap();
    assert_eq!(
        moves,
        vec![
            RowCopy {
                src: 1024,
                dst: 0,
                n: 1024
            },
            RowCopy {
                src: 2048,
                dst: 1024,
                n: 1024
            },
        ]
    );
    assert_eq!((c.run_start(2), c.reserved(2)), (Some(0), 3072));
    // The pool keeps only the positions in use, chunk by chunk.
    let mut pool = KvPool::new(Box::new(Chunked::new(3584)));
    pool.reserve(1, 1024).unwrap();
    pool.reserve(2, 2048).unwrap();
    pool.commit(&[StepInput {
        seq: 2,
        n_tokens: 1500,
        all_logits: false,
    }]);
    pool.seq_rm(1, 0);
    let moves = pool.reserve(2, 3072).unwrap();
    assert_eq!(
        moves
            .iter()
            .map(|c| (c.src, c.dst, c.n))
            .collect::<Vec<_>>(),
        vec![(1024, 0, 1024), (2048, 1024, 476)]
    );
    // Too big for the rows below and after it together: Full, nothing changed.
    assert!(matches!(c.reserve(2, 3840), Err(KvError::Full { .. })));
    assert_eq!((c.run_start(2), c.reserved(2)), (Some(0), 3072));
}

#[test]
fn seq_cp_gives_dst_rows_of_its_own() {
    for mut pool in layouts(4096) {
        pool.reserve(0, 600).unwrap();
        let step = [StepInput {
            seq: 0,
            n_tokens: 600,
            all_logits: false,
        }];
        pool.plan_step(&step).unwrap();
        pool.commit(&step);
        let copies = pool.seq_cp(0, 1, 500).unwrap();
        assert_eq!(copies.iter().map(|c| c.n).sum::<usize>(), 500);
        assert_eq!(pool.seq_len(1), 500);
        let src: Vec<usize> = (0..600).map(|p| pool.row(0, p).unwrap()).collect();
        for p in 0..500 {
            assert!(!src.contains(&pool.row(1, p).unwrap()));
        }
        assert_eq!(pool.runs(1, 500).iter().map(|r| r.1).sum::<usize>(), 500);
    }
}

/// The scheduler reserves a destination's rows before the copy; an empty destination that
/// already reserves rows takes the copy, keeping (at least) its reservation. One holding
/// positions is still refused.
#[test]
fn seq_cp_into_a_reserved_but_empty_destination() {
    for mut pool in layouts(4096) {
        pool.reserve(0, 600).unwrap();
        let step = [StepInput {
            seq: 0,
            n_tokens: 600,
            all_logits: false,
        }];
        pool.plan_step(&step).unwrap();
        pool.commit(&step);
        pool.reserve(1, 1000).unwrap();
        let before = pool.reserved(1);
        let copies = pool.seq_cp(0, 1, 500).unwrap();
        assert_eq!(copies.iter().map(|c| c.n).sum::<usize>(), 500);
        assert_eq!(pool.seq_len(1), 500);
        assert!(pool.reserved(1) >= before);
        let src: Vec<usize> = (0..600).map(|p| pool.row(0, p).unwrap()).collect();
        for p in 0..500 {
            assert!(!src.contains(&pool.row(1, p).unwrap()));
        }
        assert!(matches!(pool.seq_cp(0, 1, 10), Err(KvError::Invalid(_))));
    }
}

/// A pipelined prefill (ADR 0077) plans each micro-batch as the same chunk would be planned as a
/// step of its own after the chunks before it; lengths are unchanged on return, so the step
/// still commits once.
#[test]
fn chunk_plans_equal_serial_steps() {
    for mut pool in layouts(4096) {
        let name = pool.layout_name();
        // Another sequence first, so the rows of sequence 0 are not its positions.
        let other = [StepInput {
            seq: 1,
            n_tokens: 40,
            all_logits: false,
        }];
        pool.reserve(1, 40).unwrap();
        pool.commit(&other);
        pool.reserve(0, 1000).unwrap();
        let first = [StepInput {
            seq: 0,
            n_tokens: 300,
            all_logits: false,
        }];
        pool.commit(&first);
        let sizes = [175usize, 175, 175, 175];
        let input = StepInput {
            seq: 0,
            n_tokens: 700,
            all_logits: false,
        };
        let chunks = pool.plan_chunks(input, &sizes).unwrap();
        assert_eq!(pool.seq_len(0), 300, "{name}: lengths unchanged");
        let mut serial = Vec::new();
        for &n in &sizes {
            let s = [StepInput {
                n_tokens: n,
                ..input
            }];
            serial.push(pool.plan_step(&s).unwrap());
            pool.commit(&s);
        }
        assert_eq!(chunks, serial, "{name}");
        assert_eq!(pool.seq_len(0), 1000, "{name}");
    }
}

/// Chunk plans refuse what the step would, and sizes that do not add up; lengths stay.
#[test]
fn chunk_plans_refuse_what_the_step_would() {
    let mut pool = KvPool::new(Box::new(Chunked::new(1024)));
    pool.reserve(0, 100).unwrap();
    let input = StepInput {
        seq: 0,
        n_tokens: 300,
        all_logits: false,
    };
    // The reservation covers 256 rows.
    assert!(pool.plan_chunks(input, &[150, 150]).is_err());
    assert_eq!(pool.seq_len(0), 0);
    pool.reserve(0, 300).unwrap();
    assert!(pool.plan_chunks(input, &[150, 100]).is_err());
    assert!(pool.plan_chunks(input, &[]).is_err());
    assert_eq!(pool.seq_len(0), 0);
    assert_eq!(pool.plan_chunks(input, &[150, 150]).unwrap().len(), 2);
}

//! With one sequence at row 0 the plan is what the single-sequence builder computes today
//! (`rows == pos`, `n_kv = pad(len + n, 256)`, the causal mask), which keeps C4 bit-equal.

use tetraeder_kvcache::{
    Chunked, F16_NEG_INF, KvError, KvPool, Layout, Paged, ROW_ALIGN, StepInput,
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

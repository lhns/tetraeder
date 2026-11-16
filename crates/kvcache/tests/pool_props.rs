//! Properties of the pool over both layouts (ADR 0065 P2 and P3, ADR 0067): rows of live
//! sequences are disjoint, a failed reservation changes nothing, every planned row lies inside
//! its sequence's reservation, masks open exactly a token's own prefix, and moves and copies
//! preserve the content of every position.

use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use tetraeder_kvcache::{
    BlockCounts, Chunked, F16_NEG_INF, KvError, KvPool, Paged, ROW_ALIGN, RowCopy, SeqId, StepInput,
};

#[derive(Debug, Clone, Copy)]
enum Kind {
    Chunked,
    Paged(usize),
}

fn pool(kind: Kind, capacity: usize) -> KvPool {
    match kind {
        Kind::Chunked => KvPool::new(Box::new(Chunked::new(capacity))),
        Kind::Paged(b) => KvPool::new(Box::new(Paged::new(capacity, b))),
    }
}

#[derive(Debug, Clone)]
enum Op {
    Reserve(SeqId, usize),
    Step(Vec<(SeqId, usize, bool)>),
    Rm(SeqId, usize),
    Cp(SeqId, SeqId, usize),
}

fn op() -> impl Strategy<Value = Op> {
    let seq = 0u32..5;
    prop_oneof![
        (seq.clone(), 1usize..1800).prop_map(|(s, n)| Op::Reserve(s, n)),
        proptest::collection::vec((seq.clone(), 1usize..48, any::<bool>()), 1..4)
            .prop_map(Op::Step),
        (seq.clone(), 0usize..1800).prop_map(|(s, k)| Op::Rm(s, k)),
        (seq.clone(), seq, 0usize..1800).prop_map(|(a, b, n)| Op::Cp(a, b, n)),
    ]
}

/// Physical rows (one value per row) and the expected value of every position.
struct Sim {
    pool: KvPool,
    store: Vec<u64>,
    model: HashMap<SeqId, Vec<u64>>,
    next: u64,
}

impl Sim {
    fn apply(&mut self, copies: &[RowCopy]) {
        for c in copies {
            let src: Vec<u64> = self.store[c.src..c.src + c.n].to_vec();
            self.store[c.dst..c.dst + c.n].copy_from_slice(&src);
        }
    }

    fn snapshot(&self) -> (usize, Vec<(SeqId, usize, Vec<usize>)>) {
        let mut seqs: Vec<_> = (0..5u32)
            .map(|s| {
                let r = self.pool.reserved(s);
                let rows = (0..r).map(|p| self.pool.row(s, p).unwrap()).collect();
                (s, r, rows)
            })
            .collect();
        seqs.sort();
        (self.pool.free(), seqs)
    }

    /// The per-sequence plan has the masked plan's tokens, rows and outputs; each input reads a
    /// run of its own reservation, its mask block opening exactly the token's prefix. Paged
    /// layouts have no runs and get none.
    fn check_per_seq(&self, inputs: &[StepInput], plan: &tetraeder_kvcache::StepPlan) {
        let Some(ps) = self.pool.plan_step_per_seq(inputs).unwrap() else {
            assert_eq!(self.pool.layout_name(), "paged");
            return;
        };
        assert_eq!(self.pool.layout_name(), "chunked");
        assert_eq!(
            (&ps.pos, &ps.rows, &ps.outputs),
            (&plan.pos, &plan.rows, &plan.outputs)
        );
        assert_eq!(ps.seqs.len(), inputs.len());
        let mut t = 0;
        let mut off = 0;
        for (inp, s) in inputs.iter().zip(&ps.seqs) {
            let len = self.pool.seq_len(inp.seq);
            assert_eq!(
                (s.first_token, s.n_tokens, s.mask_offset),
                (t, inp.n_tokens, off)
            );
            assert_eq!(s.n_kv % ROW_ALIGN, 0);
            assert!(len + inp.n_tokens <= s.n_kv && s.n_kv <= self.pool.reserved(inp.seq));
            for p in 0..s.n_kv {
                assert_eq!(self.pool.row(inp.seq, p), Some(s.kv_start + p));
            }
            for i in 0..inp.n_tokens {
                for j in 0..s.n_kv {
                    let v = ps.mask[off + i * s.n_kv + j];
                    assert_eq!(v == 0, j <= len + i, "token {} col {j}", t + i);
                    assert!(v == 0 || v == F16_NEG_INF);
                }
            }
            t += inp.n_tokens;
            off += s.n_kv * inp.n_tokens;
        }
        assert_eq!(ps.mask.len(), off);
    }

    fn check_invariants(&self) {
        let cap = self.pool.capacity();
        let mut seen = HashSet::new();
        let mut reserved = 0;
        for s in 0..5u32 {
            let r = self.pool.reserved(s);
            reserved += r;
            for p in 0..r {
                let row = self.pool.row(s, p).unwrap();
                assert!(row < cap, "row {row} outside the pool of {cap}");
                assert!(seen.insert(row), "row {row} owned twice (P2)");
            }
        }
        assert!(
            reserved <= cap,
            "reserved {reserved} above capacity {cap} (P3)"
        );
        assert_eq!(self.pool.free(), cap - reserved);
        for (&s, vals) in &self.model {
            assert_eq!(self.pool.seq_len(s), vals.len());
            assert!(self.pool.reserved(s) >= vals.len());
            for (p, v) in vals.iter().enumerate() {
                let row = self.pool.row(s, p).unwrap();
                assert_eq!(self.store[row], *v, "seq {s} pos {p} lost its content");
            }
        }
        let b: BlockCounts = self.pool.blocks();
        assert!(b.used + b.reserved + b.free >= cap / ROW_ALIGN);
    }

    fn run(&mut self, op: Op) {
        match op {
            Op::Reserve(s, n) => {
                let before = self.snapshot();
                match self.pool.reserve(s, n) {
                    Ok(copies) => self.apply(&copies),
                    Err(KvError::Full { .. }) => {
                        assert_eq!(self.snapshot(), before, "Full changed state")
                    }
                    Err(e) => panic!("{e}"),
                }
            }
            Op::Step(items) => self.step(items),
            Op::Rm(s, keep) => {
                self.pool.seq_rm(s, keep);
                if let Some(v) = self.model.get_mut(&s) {
                    v.truncate(keep);
                    if v.is_empty() {
                        self.model.remove(&s);
                    }
                }
            }
            Op::Cp(a, b, n) => {
                let before = self.snapshot();
                let ok_input = a != b
                    && self.pool.reserved(a) > 0
                    && self.pool.seq_len(a) >= n
                    && self.pool.seq_len(b) == 0;
                match self.pool.seq_cp(a, b, n) {
                    Ok(copies) => {
                        assert!(ok_input);
                        self.apply(&copies);
                        if n > 0 {
                            let prefix = self.model[&a][..n].to_vec();
                            self.model.insert(b, prefix);
                        }
                        // dst rows are its own (P2): checked by check_invariants
                    }
                    Err(KvError::Full { .. }) => {
                        assert!(ok_input);
                        assert_eq!(self.snapshot(), before, "Full changed state");
                    }
                    Err(_) => assert!(!ok_input),
                }
            }
        }
    }

    fn step(&mut self, items: Vec<(SeqId, usize, bool)>) {
        let mut seen = HashSet::new();
        let items: Vec<_> = items.into_iter().filter(|i| seen.insert(i.0)).collect();
        for &(s, k, _) in &items {
            let need = self.pool.seq_len(s) + k;
            match self.pool.reserve(s, need) {
                Ok(copies) => self.apply(&copies),
                Err(KvError::Full { .. }) => return,
                Err(e) => panic!("{e}"),
            }
        }
        let inputs: Vec<StepInput> = items
            .iter()
            .map(|&(seq, n_tokens, all_logits)| StepInput {
                seq,
                n_tokens,
                all_logits,
            })
            .collect();
        let plan = self.pool.plan_step(&inputs).unwrap();
        self.check_per_seq(&inputs, &plan);
        let n_tokens: usize = inputs.iter().map(|i| i.n_tokens).sum();
        assert_eq!(plan.pos.len(), n_tokens);
        assert_eq!(plan.rows.len(), n_tokens);
        assert_eq!(plan.mask.len(), plan.n_kv * n_tokens);
        assert_eq!(plan.n_kv % ROW_ALIGN, 0);

        let mut t = 0;
        let mut outputs = Vec::new();
        for inp in &inputs {
            let len = self.pool.seq_len(inp.seq);
            for i in 0..inp.n_tokens {
                assert_eq!(plan.pos[t], (len + i) as i32);
                let row = plan.rows[t] as usize;
                // P3: inside the reservation, which is inside the pool and below n_kv.
                assert!(row < plan.n_kv);
                assert_eq!(Some(row), self.pool.row(inp.seq, len + i));
                assert!(len + i < self.pool.reserved(inp.seq));
                let open: HashSet<usize> = (0..=len + i)
                    .map(|p| self.pool.row(inp.seq, p).unwrap())
                    .collect();
                for j in 0..plan.n_kv {
                    let v = plan.mask[t * plan.n_kv + j];
                    assert_eq!(v == 0, open.contains(&j), "token {t} row {j}");
                    assert!(v == 0 || v == F16_NEG_INF);
                }
                if inp.all_logits || i + 1 == inp.n_tokens {
                    outputs.push(t as i32);
                }
                self.next += 1;
                self.store[row] = self.next;
                self.model.entry(inp.seq).or_default().push(self.next);
                t += 1;
            }
        }
        assert_eq!(plan.outputs, outputs);
        self.pool.commit(&inputs);
    }
}

fn run_case(kind: Kind, units: usize, ops: Vec<Op>) {
    let capacity = units * ROW_ALIGN;
    let mut sim = Sim {
        pool: pool(kind, capacity),
        store: vec![0; capacity],
        model: HashMap::new(),
        next: 0,
    };
    sim.check_invariants();
    for op in ops {
        sim.run(op);
        sim.check_invariants();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn chunked(units in 8usize..24, ops in proptest::collection::vec(op(), 1..40)) {
        run_case(Kind::Chunked, units, ops);
    }

    #[test]
    fn paged_64(units in 8usize..24, ops in proptest::collection::vec(op(), 1..40)) {
        run_case(Kind::Paged(64), units, ops);
    }

    #[test]
    fn paged_256(units in 8usize..24, ops in proptest::collection::vec(op(), 1..40)) {
        run_case(Kind::Paged(256), units, ops);
    }
}

/// Whether `n` rows can be had for `seq` in `pool`: in place, or in a free gap of `n` rows with
/// the sequence's own run still occupied (a move copies from it).
fn chunked_has_room(pool: &KvPool, seq: SeqId, n: usize) -> bool {
    let mut runs: Vec<(usize, usize, SeqId)> = (0..5u32)
        .filter(|&s| pool.reserved(s) > 0)
        .map(|s| (pool.row(s, 0).unwrap(), pool.reserved(s), s))
        .collect();
    runs.sort();
    let mut cursor = 0;
    for (i, &(start, size, s)) in runs.iter().enumerate() {
        if start - cursor >= n {
            return true;
        }
        if s == seq {
            let next = runs.get(i + 1).map_or(pool.capacity(), |r| r.0);
            if start + n <= next {
                return true;
            }
        }
        cursor = start + size;
    }
    pool.capacity() - cursor >= n
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A sequence that has to move succeeds whenever a free gap of the rounded need exists, even
    /// when the gap is smaller than need + GROW (the extra rows are a courtesy, not a demand).
    #[test]
    fn chunked_reserve_succeeds_whenever_a_gap_fits(
        capacity in prop::sample::select(vec![2048usize, 3072, 4096, 6144]),
        ops in proptest::collection::vec((0u32..5, 1usize..1800), 1..40),
    ) {
        let mut pool = pool(Kind::Chunked, capacity);
        for (s, n) in ops {
            let need = n.div_ceil(ROW_ALIGN) * ROW_ALIGN;
            let fits = pool.reserved(s) >= need || chunked_has_room(&pool, s, need);
            match pool.reserve(s, n) {
                Ok(_) => prop_assert!(fits, "reserved without room?"),
                Err(KvError::Full { .. }) => {
                    prop_assert!(!fits, "seq {s} refused {need} rows with a gap that fits");
                }
                Err(e) => prop_assert!(false, "{e}"),
            }
        }
    }
}

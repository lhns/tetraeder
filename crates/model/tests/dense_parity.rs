//! The Qwen3 dense builder (ADR 0066) against libllama on the tiny dense fixture
//! (`cargo xtask tiny-dense`: head size 64, so the quantized KV formats run): prefill logits of
//! every position and greedy steps, the quantized caches with rotation off (`dense_rotation.rs`
//! has it on), several sequences in one step, `seq_cp`, a chunked move and snapshots.
//! `cpu_per_seq_*` repeat the libllama comparisons with attention per sequence; the whole file
//! runs with it under `TETRAEDER_DENSE_PER_SEQ_ATTN=1` (`dense_per_seq.rs` has the rest).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::*;
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3, SeqEval};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const FORMATS: [KvFormat; 3] = [KvFormat::F16, KvFormat::Q8_0, KvFormat::Q8_0_Q4_0];

fn opts(kv: KvFormat, n_seq_max: usize) -> DenseOptions {
    DenseOptions {
        kv,
        kv_rotation: Some(false),
        n_seq_max,
        // A sequence reserves whole 256-row units.
        n_ctx_pool: 256 * n_seq_max,
        extra_bufts: false,
        ..DenseOptions::new(CPU)
    }
}

fn load(kv: KvFormat, n_seq_max: usize) -> Qwen3 {
    load_attn(kv, n_seq_max, None)
}

/// `per_seq`: `None` keeps `DenseOptions::new`'s choice (the env var).
fn load_attn(kv: KvFormat, n_seq_max: usize, per_seq: Option<bool>) -> Qwen3 {
    let mut o = opts(kv, n_seq_max);
    o.per_seq_attn = per_seq.unwrap_or(o.per_seq_attn);
    Qwen3::load(dense_fixture(), &o).unwrap()
}

/// Prefill logits of every position and 12 greedy steps, top-5 within `TOL_CPU`, for the F16
/// cache and the quantized formats with rotation off on both sides.
fn matches_libllama(per_seq: Option<bool>) {
    for kv in FORMATS {
        let r = libllama_reference_kv(&dense_fixture(), &PROMPT, N_GREEDY, kv);
        let mut m = load_attn(kv, 1, per_seq);
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("{kv} per_seq {per_seq:?}: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: cpu.
#[test]
fn cpu_matches_libllama() {
    matches_libllama(None);
}

/// hardware: cpu.
#[test]
fn cpu_per_seq_matches_libllama() {
    matches_libllama(Some(true));
}

/// hardware: cpu. A step graph kept for decode gives the same logits as one built per step.
#[test]
fn cpu_step_graph_reuse_is_bit_equal() {
    let run = |reuse: bool| {
        let o = DenseOptions {
            reuse_step_graph: reuse,
            ..opts(KvFormat::F16, 1)
        };
        let mut m = Qwen3::load(dense_fixture(), &o).unwrap();
        let mut all = m.eval_last(&PROMPT).unwrap();
        for t in [5, 6, 7, 8] {
            all.extend(m.eval_last(&[t]).unwrap());
        }
        (all, m.step_graphs_built())
    };
    let ((a, built_a), (b, built_b)) = (run(true), run(false));
    assert_eq!(a, b);
    assert_eq!((built_a, built_b), (2, 5));
}

/// hardware: cpu. The kept graph of a batched step gives the logits of a graph built for it.
#[test]
fn cpu_batched_step_graph_reuse_is_bit_equal() {
    let (streams, steps) = script();
    let run = |reuse: bool| {
        let o = DenseOptions {
            reuse_step_graph: reuse,
            ..opts(KvFormat::F16, 3)
        };
        let mut m = Qwen3::load(dense_fixture(), &o).unwrap();
        let rows = builder_script(&mut m, &streams, &steps, true);
        (rows, m.step_graphs_built())
    };
    let ((a, built_a), (b, built_b)) = (run(true), run(false));
    assert_eq!(a, b);
    assert!(
        built_a < built_b,
        "the kept graph was reused: {built_a} < {built_b}"
    );
}

/// A deterministic stream of byte tokens.
fn stream(seed: u32, len: usize) -> Vec<i32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 16) % 256) as i32
        })
        .collect()
}

fn item(seq: u32, range: std::ops::Range<usize>, all: bool) -> Item {
    Item { seq, range, all }
}

/// Three sequences of different lengths: all three prefill chunks in the first step, then a
/// step where the middle sequence's second prefill chunk (every token verified) sits among two
/// decodes, then decodes until each ends, so the batch shrinks from 3 to 1.
fn script() -> (Vec<Vec<i32>>, Vec<Vec<Item>>) {
    let lens = [12usize, 13, 11];
    let streams: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(i, &n)| stream(7 + i as u32, n))
        .collect();
    let mut steps = vec![
        vec![
            item(0, 0..6, false),
            item(1, 0..5, false),
            item(2, 0..4, false),
        ],
        vec![
            item(0, 6..7, false),
            item(1, 5..9, true),
            item(2, 4..5, false),
        ],
    ];
    let mut at = [7usize, 9, 5];
    loop {
        let mut step = Vec::new();
        for s in 0..3usize {
            if at[s] < lens[s] {
                step.push(item(s as u32, at[s]..at[s] + 1, false));
                at[s] += 1;
            }
        }
        if step.is_empty() {
            break;
        }
        steps.push(step);
    }
    (streams, steps)
}

/// The script restricted to one sequence: each step keeps only that sequence's item.
fn alone(steps: &[Vec<Item>], seq: u32) -> Vec<Vec<Item>> {
    steps
        .iter()
        .filter_map(|s| {
            let v: Vec<Item> = s.iter().filter(|i| i.seq == seq).cloned().collect();
            (!v.is_empty()).then_some(v)
        })
        .collect()
}

/// hardware: cpu. Batched against each sequence alone through the builder, and against libllama
/// with `n_seq_max = 3` and a unified KV buffer, greedy equal except at a near-tie (ADR 0043).
/// Against libllama (same step shapes) `TOL_CPU` holds; against alone runs `tol_batch`.
#[test]
fn cpu_multi_sequence_matches_alone_and_libllama() {
    multi_sequence(None);
}

/// hardware: cpu. The same with attention per sequence.
#[test]
fn cpu_per_seq_multi_sequence_matches_alone_and_libllama() {
    multi_sequence(Some(true));
}

fn multi_sequence(per_seq: Option<bool>) {
    let (streams, steps) = script();
    for kv in FORMATS {
        let load = |kv, n| load_attn(kv, n, per_seq);
        let mut m = load(kv, 3);
        let batched = builder_script(&mut m, &streams, &steps, true);
        // Each alone: a model of its own, one sequence.
        let mut alone_rows = Logits::new();
        for s in 0..3u32 {
            let mut one = load(kv, 1);
            let sub: Vec<Vec<Item>> = alone(&steps, s)
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|mut i| {
                            i.seq = 0;
                            i
                        })
                        .collect()
                })
                .collect();
            let streams0 = [streams[s as usize].clone()];
            for ((_, p), row) in builder_script(&mut one, &streams0, &sub, true) {
                alone_rows.insert((s, p), row);
            }
        }
        let reference = libllama_script(&dense_fixture(), &streams, &steps, kv);
        let mut m2 = load(kv, 3);
        let split = builder_script(&mut m2, &streams, &steps, false);
        // Print every comparison before asserting any, so a failure on a host this cannot be
        // reproduced on says which pairs differ (batch size, step shape or row placement).
        for (name, a, b) in [
            ("batched/alone", &alone_rows, &batched),
            ("batched/libllama", &reference, &batched),
            ("split/alone", &alone_rows, &split),
            ("split/libllama", &reference, &split),
        ] {
            let (worst, first) = script_delta(a, b);
            eprintln!("{kv} {name}: max |delta| {worst:e}, first differing row {first:?}");
        }
        let w = compare_scripts(
            &alone_rows,
            &batched,
            TOL_CPU,
            true,
            &format!("{kv} batched/alone"),
        );
        let l = compare_scripts(
            &reference,
            &batched,
            TOL_CPU,
            false,
            &format!("{kv} batched/libllama"),
        );
        // The same steps one item at a time on the shared pool.
        let s = compare_scripts(
            &alone_rows,
            &split,
            TOL_CPU,
            false,
            &format!("{kv} split/alone"),
        );
        eprintln!(
            "{kv}: {} rows, max |delta| batched vs alone {w:e}, vs libllama {l:e}, split vs alone {s:e}",
            batched.len()
        );
    }
}

/// hardware: cpu. `seq_cp` of a prefilled prompt, then both sequences continue in one step
/// with different tokens; each equals a fresh run of its own tokens (within `tol_batch`).
#[test]
fn cpu_seq_cp_continuation() {
    let kv = KvFormat::F16;
    let (a, b) = ([21, 22, 23, 24], [31, 32, 33]);
    let mut m = load(kv, 2);
    m.eval_last(&PROMPT).unwrap();
    m.seq_cp(0, 1, PROMPT.len()).unwrap();
    assert_eq!((m.pool().seq_len(0), m.pool().seq_len(1)), (6, 6));
    let mut out = Vec::new();
    let spans = m
        .decode(
            &[
                SeqEval {
                    seq: 0,
                    tokens: &a,
                    all_logits: true,
                },
                SeqEval {
                    seq: 1,
                    tokens: &b,
                    all_logits: true,
                },
            ],
            &mut out,
        )
        .unwrap();
    let nv = m.n_vocab();
    for (span, cont) in spans.into_iter().zip([&a[..], &b[..]]) {
        let mut fresh = load(kv, 1);
        fresh.eval_last(&PROMPT).unwrap();
        let want = fresh.eval(cont).unwrap();
        let have = &out[span.start * nv..span.end * nv];
        let worst = want
            .iter()
            .zip(have)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("seq_cp continuation: max |delta| {worst:e}");
        assert!(worst <= tol_batch(&want), "{worst:e}");
    }
    // A copy into a sequence that holds positions, or from an empty one, is refused.
    assert!(m.seq_cp(0, 1, 2).is_err());
    assert!(load(kv, 2).seq_cp(0, 1, 1).is_err());
}

/// hardware: cpu. A snapshot restores into the same or another sequence id: the next step's
/// logits are bitwise equal in place and within `TOL_CPU` elsewhere in the pool.
#[test]
fn cpu_snapshot_roundtrip() {
    for kv in FORMATS {
        let mut m = load(kv, 3);
        m.eval_last(&PROMPT).unwrap();
        let bytes = m.seq_snapshot(0).unwrap();
        let next = m.eval_last(&[40]).unwrap();
        // In place.
        m.seq_rm(0, 0).unwrap();
        m.seq_restore(0, &bytes, PROMPT.len()).unwrap();
        assert_eq!(m.n_past(), PROMPT.len());
        assert_eq!(m.eval_last(&[40]).unwrap(), next, "{kv} in place");
        // Another id, another row range.
        m.seq_restore(2, &bytes, PROMPT.len()).unwrap();
        let mut out = Vec::new();
        m.decode(
            &[SeqEval {
                seq: 2,
                tokens: &[40],
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
        let worst = next
            .iter()
            .zip(&out)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(worst <= TOL_CPU, "{kv} elsewhere: {worst:e}");
        // A snapshot of the wrong size is refused.
        assert!(m.seq_restore(1, &bytes[1..], PROMPT.len()).is_err());
    }
}

/// hardware: cpu. A snapshot read through staging halves (ADR 0074; small ones cut the ranges
/// and take many syncs) has the bytes of the straight read and restores bit-exact in place; a
/// CPU model has no staging unless asked, and reports it as host memory when it has.
#[test]
fn cpu_staged_snapshot_roundtrip() {
    for kv in FORMATS {
        let mut m = load(kv, 3);
        assert_eq!(m.snapshot_staging(), None, "{kv}");
        let mut out = Vec::new();
        m.decode(
            &[
                SeqEval {
                    seq: 1,
                    tokens: &[7, 8, 9, 10],
                    all_logits: false,
                },
                SeqEval {
                    seq: 0,
                    tokens: &PROMPT,
                    all_logits: false,
                },
            ],
            &mut out,
        )
        .unwrap();
        let direct = m.seq_snapshot(0).unwrap();
        assert!(!direct.is_empty(), "{kv}");
        for half in [1, 333, 4096, 1 << 20] {
            m.set_snapshot_staging(Some(half)).unwrap();
            assert_eq!(m.snapshot_staging(), Some((2 * half, false)), "{kv}");
            assert!(
                m.memory()
                    .iter()
                    .any(|u| u.kind == "staging" && u.capacity == 2 * half as u64),
                "{kv}"
            );
            assert_eq!(m.seq_snapshot(0).unwrap(), direct, "{kv} half {half}");
        }
        // A restore through the staging halves (ADR 0074) writes the same KV: the snapshot
        // read back is byte-identical and the next step's logits equal the unrestored run's.
        let next = m.eval_last(&[40]).unwrap();
        for half in [1, 333, 4096, 1 << 20] {
            m.set_snapshot_staging(Some(half)).unwrap();
            m.seq_rm(0, 0).unwrap();
            m.seq_restore(0, &direct, PROMPT.len()).unwrap();
            assert_eq!(
                m.seq_snapshot(0).unwrap(),
                direct,
                "{kv} restore half {half}"
            );
            assert_eq!(
                m.eval_last(&[40]).unwrap(),
                next,
                "{kv} restore half {half}"
            );
        }
        m.set_snapshot_staging(None).unwrap();
        m.seq_rm(0, 0).unwrap();
        m.seq_restore(0, &direct, PROMPT.len()).unwrap();
        assert_eq!(m.seq_snapshot(0).unwrap(), direct, "{kv} bytes roundtrip");
        assert_eq!(m.eval_last(&[40]).unwrap(), next, "{kv} in place");
        m.set_snapshot_staging(None).unwrap();
        assert!(m.memory().iter().all(|u| u.kind != "staging"), "{kv}");
    }
}

/// hardware: cpu. A sequence that outgrows its run is moved by the layout and copied on the
/// device mid-decode; the logits stay within `TOL_CPU` of the unmoved run.
#[test]
fn cpu_chunked_move_keeps_logits() {
    let pool = |n_seq_max| DenseOptions {
        n_ctx_pool: 4096,
        ..opts(KvFormat::F16, n_seq_max)
    };
    let (long, other) = (stream(3, 300), stream(4, 10));
    let mut moved = Qwen3::load(dense_fixture(), &pool(2)).unwrap();
    moved.eval_last(&long[..200]).unwrap();
    // The second sequence takes the rows after the first's run, so the first must move.
    moved.eval_last_seq(1, &other);
    let row0 = moved.pool().row(0, 0).unwrap();
    let got = moved.eval(&long[200..]).unwrap();
    assert_ne!(moved.pool().row(0, 0).unwrap(), row0, "the sequence moved");
    let mut plain = Qwen3::load(dense_fixture(), &pool(1)).unwrap();
    plain.eval_last(&long[..200]).unwrap();
    let want = plain.eval(&long[200..]).unwrap();
    let worst = want
        .iter()
        .zip(&got)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("chunked move: max |delta| {worst:e}");
    assert!(worst <= TOL_CPU, "{worst:e}");
}

trait EvalSeq {
    fn eval_last_seq(&mut self, seq: u32, tokens: &[i32]);
}

impl EvalSeq for Qwen3 {
    fn eval_last_seq(&mut self, seq: u32, tokens: &[i32]) {
        let mut out = Vec::new();
        self.decode(
            &[SeqEval {
                seq,
                tokens,
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
    }
}

/// hardware: cpu. A full pool is reported without changing any length, and `seq_rm` gives the
/// rows back.
#[test]
fn cpu_full_pool_and_seq_rm() {
    let o = DenseOptions {
        n_ctx_pool: 256,
        ..opts(KvFormat::F16, 2)
    };
    let mut m = Qwen3::load(dense_fixture(), &o).unwrap();
    m.eval_last_seq(0, &stream(1, 200));
    assert!(
        !m.try_reserve(1, 100).unwrap(),
        "no run of 100 rows is left"
    );
    let mut out = Vec::new();
    let inp = [SeqEval {
        seq: 0,
        tokens: &[1; 100],
        all_logits: false,
    }];
    assert!(m.decode(&inp, &mut out).is_err());
    assert_eq!(m.pool().seq_len(0), 200);
    m.seq_rm(0, 0).unwrap();
    assert!(m.try_reserve(1, 100).unwrap());
}

/// hardware: cpu. Weights in the CPU's extra buffer types (AMX, repack; only where the host has
/// one that runs these matmuls) against plain weights, prefill and decode: bit-equal when no
/// extra buffer type takes a weight, else within 5% of the largest logit (a gross-error bound:
/// AMX's F16 kernel keeps activations in F32 where the plain one rounds them to F16; CI's AMX
/// runner measured 1.8e-2 and 1.4e-1 on the random fixture). Printed for the record: the prompt
/// logits against libllama with its extra buffer types on, which runs the same kernels.
#[test]
fn cpu_extra_bufts_match_plain_weights() {
    for kv in [KvFormat::F16, KvFormat::Q8_0] {
        let run = |extra: bool| {
            let o = DenseOptions {
                extra_bufts: extra,
                ..opts(kv, 1)
            };
            let mut m = Qwen3::load(dense_fixture(), &o).unwrap();
            let mut all = m.eval(&PROMPT).unwrap();
            for t in [5, 6, 7] {
                all.extend(m.eval_last(&[t]).unwrap());
            }
            (all, m.weight_bufts().to_vec())
        };
        let ((on, bufts), (off, none)) = (run(true), run(false));
        assert!(none.is_empty());
        let worst = on
            .iter()
            .zip(&off)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("{kv} extra buffer types {bufts:?}: max |delta| vs plain {worst:e}");
        if bufts.is_empty() {
            assert_eq!(on, off, "{kv}");
        } else {
            let top = off.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(worst <= 5e-2 * top, "{kv}: {worst:e} (largest logit {top})");
            let r = libllama_reference_bufts(&dense_fixture(), &PROMPT, 0, kv, true);
            let n = r.prompt_logits.len();
            let lw = r.prompt_logits[..n]
                .iter()
                .zip(&on[..n])
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            eprintln!("{kv} prompt logits vs libllama with extra buffer types: max |delta| {lw:e}");
        }
    }
}

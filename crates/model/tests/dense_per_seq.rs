//! Attention per sequence (`DenseOptions::per_seq_attn`) against the masked attention over the
//! whole pool, on the tiny dense fixture: sequences longer than 512 rows (the CPU split-KV
//! kernel) at non-zero run starts with others resident, prefill chunks mixed with decodes, and a
//! kept step graph that must not be reused when a sequence's rows change.
//! `dense_parity.rs` has the comparisons with libllama.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::*;
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3, SeqEval};
use tetraeder_plan::KvFormat;

const FORMATS: [KvFormat; 3] = [KvFormat::F16, KvFormat::Q8_0, KvFormat::Q8_0_Q4_0];

fn load(kv: KvFormat, per_seq: bool) -> Qwen3 {
    let o = DenseOptions {
        kv,
        kv_rotation: Some(false),
        n_seq_max: 3,
        n_ctx_pool: 4096,
        per_seq_attn: per_seq,
        // Off: the masked path for one-input steps too, the reference these tests compare with.
        per_seq_attn_single: per_seq,
        extra_bufts: false,
        ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
    };
    Qwen3::load(dense_fixture(), &o).unwrap()
}

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

/// Sequence 0 prefills 520 tokens (rows 0..768), sequence 1 300 (768..1280), sequence 2 grows
/// in place to 610 (from row 1280); a 500-token chunk of sequence 2 runs with a decode of
/// sequence 0, then all three decode, then sequence 1 alone above the others.
fn script() -> (Vec<Vec<i32>>, Vec<Vec<Item>>) {
    let streams = vec![stream(11, 540), stream(12, 330), stream(13, 620)];
    let mut steps = vec![
        vec![item(0, 0..520, false)],
        vec![item(1, 0..300, false), item(2, 0..100, true)],
        vec![item(0, 520..521, false), item(2, 100..600, true)],
    ];
    for k in 0..10 {
        steps.push(vec![
            item(0, 521 + k..522 + k, false),
            item(1, 300 + k..301 + k, false),
            item(2, 600 + k..601 + k, false),
        ]);
    }
    for k in 310..320 {
        steps.push(vec![item(1, k..k + 1, false)]);
    }
    (streams, steps)
}

/// Each sequence alone in a model of its own (one sequence, its rows from 0).
fn alone(kv: KvFormat, per_seq: bool, streams: &[Vec<i32>], steps: &[Vec<Item>]) -> Logits {
    let mut out = Logits::new();
    for s in 0..streams.len() as u32 {
        let sub: Vec<Vec<Item>> = steps
            .iter()
            .filter_map(|st| {
                let v: Vec<Item> = st
                    .iter()
                    .filter(|i| i.seq == s)
                    .map(|i| item(0, i.range.clone(), i.all))
                    .collect();
                (!v.is_empty()).then_some(v)
            })
            .collect();
        let rows = builder_script(
            &mut load(kv, per_seq),
            &[streams[s as usize].clone()],
            &sub,
            true,
        );
        out.extend(rows.into_iter().map(|((_, p), r)| ((s, p), r)));
    }
    out
}

/// hardware: cpu. Against each sequence alone (where both paths build the same graph), per
/// KV format: attention per sequence one input per step within `TOL_CPU`, batched within
/// `tol_batch` (bit-equal on some CPUs; others take other matmul kernels for other row counts,
/// ADR 0043: CI Windows measured 2.4e-3). The masked path batched is only printed and loosely bounded: with an F16 cache a
/// decode row inside a 500-token step runs ggml's tiled kernel (F32 accumulation) over the whole
/// pool, while alone it runs the split-KV kernel (F16 V accumulation), measured 0.118 at a largest
/// logit of 26 here. That is the masked path's (and libllama unified's) batch dependence, which
/// attention per sequence removes.
#[test]
fn cpu_per_seq_matches_alone_on_long_sequences() {
    let (streams, steps) = script();
    for kv in FORMATS {
        let reference = alone(kv, false, &streams, &steps);
        let (w, _) = script_delta(&reference, &alone(kv, true, &streams, &steps));
        assert!(w <= TOL_CPU, "{kv}: alone per_seq/masked {w:e}");
        let masked = builder_script(&mut load(kv, false), &streams, &steps, true);
        let (wm, fm) = script_delta(&reference, &masked);
        eprintln!("{kv} masked batched vs alone: max |delta| {wm:e}, first differing row {fm:?}");
        let top = reference
            .values()
            .flatten()
            .fold(0f32, |m, x| m.max(x.abs()));
        eprintln!("{kv} largest logit {top}");
        assert!(wm <= 1e-2 * top, "{kv} masked batched/alone {wm:e}");
        for batched in [true, false] {
            let mut m = load(kv, true);
            let rows = builder_script(&mut m, &streams, &steps, batched);
            let (worst, first) = script_delta(&reference, &rows);
            eprintln!(
                "{kv} per_seq batched {batched} vs alone: max |delta| {worst:e}, first differing row {first:?}, {} graphs",
                m.step_graphs_built()
            );
            compare_scripts(
                &reference,
                &rows,
                TOL_CPU,
                batched,
                &format!("{kv} per_seq batched {batched}"),
            );
            assert_ne!(m.pool().row(2, 0), Some(0), "sequence 2 starts above row 0");
            assert!(m.pool().seq_len(2) > 512);
        }
    }
}

fn decode1(m: &mut Qwen3, seq: u32, tok: i32) -> Vec<f32> {
    let mut out = Vec::new();
    m.decode(
        &[SeqEval {
            seq,
            tokens: &[tok],
            all_logits: false,
        }],
        &mut out,
    )
    .unwrap();
    out
}

/// hardware: cpu. Two sequences of the same length in different runs give steps of the same
/// sizes (tokens, outputs, mask): the kept graph of one must not serve the other, and a repeated
/// shape reuses it. Each step matches the masked path bit for bit or within `TOL_CPU`.
#[test]
fn cpu_per_seq_graph_key_follows_the_rows() {
    let kv = KvFormat::F16;
    let (mut on, mut off) = (load(kv, true), load(kv, false));
    let (a, b) = (stream(21, 10), stream(22, 10));
    for m in [&mut on, &mut off] {
        let mut out = Vec::new();
        m.decode(
            &[SeqEval {
                seq: 0,
                tokens: &a,
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
        m.decode(
            &[SeqEval {
                seq: 1,
                tokens: &b,
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
    }
    assert_eq!(on.pool().row(1, 0), Some(256));
    let built = on.step_graphs_built();
    let mut worst = 0f32;
    for (k, seq) in [1u32, 1, 0, 0, 1].into_iter().enumerate() {
        let tok = 40 + k as i32;
        let got = decode1(&mut on, seq, tok);
        let want = decode1(&mut off, seq, tok);
        let d = want
            .iter()
            .zip(&got)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        worst = worst.max(d);
        assert!(d <= TOL_CPU, "step {k} seq {seq}: {d:e}");
    }
    eprintln!("graph key: max |delta| vs masked {worst:e}");
    // Seq 1, seq 1 (kept), seq 0 (other rows: built), seq 0 (kept), seq 1 (built).
    assert_eq!(on.step_graphs_built() - built, 3);
}

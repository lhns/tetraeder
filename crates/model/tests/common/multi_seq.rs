//! Scenarios of `multi_seq.rs` and `gpu_multi_seq.rs` (ADR 0065, C4): several sequences in one
//! step on the MoE builder's KV pool, each against the same sequence alone, whatever the expert
//! path (resident, slots with uploads, RAM tier, prefetch, I/O service with overlap).
//!
//! A batched step uses other kernels than a one-row step, so the logits are equal within a
//! tolerance, not bitwise (ADR 0043); the greedy token is equal or the alone run's top two are a
//! near tie. Every batched run is teacher-forced with the alone run's tokens, so one flipped
//! near-tie does not make the logits incomparable afterwards.
#![allow(dead_code)]

use super::{Item, Logits, Tol, anchor_pairs, argmax, largest, libllama_script_on};
use std::path::Path;
use tetraeder_model::{LoadOptions, Qwen3Moe, SeqEval};

/// Largest |difference| of two rows.
pub fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Equal within `tol.of(want)`. A bare `f32` is that floor or 1e-3 of the largest logit: which
/// kernels a CPU variant uses for one row and for several decides the gap (CI's runners show
/// ~9e-3 where a dev box shows less). The GPU passes `GPU_TOL`.
pub fn close(got: &[f32], want: &[f32], tol: impl Into<Tol>, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let worst = max_diff(got, want);
    let scale = largest(want);
    let tol = tol.into().of(want);
    assert!(
        worst <= tol,
        "{what}: max |diff| {worst} > {tol} (largest logit {scale})"
    );
}

/// The greedy token of `got` equals the one of `want`, or `want` has a near tie there.
pub fn same_token(got: &[f32], want: &[f32], tol: impl Into<Tol>, what: &str) {
    let (g, w) = (argmax(got), argmax(want));
    if g != w {
        let gap = (want[w] - want[g]).abs();
        assert!(
            gap <= 2.0 * tol.into().of(want),
            "{what}: greedy {g} against {w}, gap {gap}"
        );
    }
}

/// One line per compared row (`what`, max |diff|, that over the largest logit, greedy equal),
/// printed before any assertion so a failure shows whether the gap is flat or grows.
pub fn print_row(what: &str, got: &[f32], want: &[f32]) {
    let d = max_diff(got, want);
    eprintln!(
        "{what}: max |diff| {d:e} ({:.2e} of largest {}), greedy equal {}",
        d / largest(want),
        largest(want),
        argmax(got) == argmax(want)
    );
}

pub fn prompt(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32)
        .map(|i| 4 + (seed * 31 + i * 7) % 200)
        .collect()
}

pub fn load(path: &Path, opts: &LoadOptions, n_seq_max: usize) -> Qwen3Moe {
    let o = LoadOptions {
        n_seq_max,
        ..opts.clone()
    };
    Qwen3Moe::load(path, &o).unwrap()
}

/// One sequence alone: `toks` is the prompt followed by `steps` greedy tokens; `logits[j]` are
/// the logits after the first `prompt.len() + j` tokens of it.
pub struct Alone {
    pub prompt_len: usize,
    pub toks: Vec<i32>,
    pub logits: Vec<Vec<f32>>,
}

pub fn alone(path: &Path, opts: &LoadOptions, prompt: &[i32], steps: usize) -> Alone {
    let mut m = load(path, opts, 1);
    alone_on(&mut m, prompt, steps)
}

pub fn alone_on(m: &mut Qwen3Moe, prompt: &[i32], steps: usize) -> Alone {
    m.reset();
    let mut l = m.eval_last(prompt).unwrap();
    let mut toks = prompt.to_vec();
    let mut logits = vec![l.clone()];
    for _ in 0..steps {
        let t = argmax(&l) as i32;
        toks.push(t);
        l = m.eval_last(&[t]).unwrap();
        logits.push(l.clone());
    }
    Alone {
        prompt_len: prompt.len(),
        toks,
        logits,
    }
}

/// A step of the schedule: `(sequence, token range of its stream)` per input.
pub type Step = Vec<(usize, std::ops::Range<usize>)>;

/// Sequence 0 and 1 prefill in the first step; sequence 2 joins in the second with a prefill
/// chunk of `chunk` tokens next to the decode tokens of the others, and finishes its prompt in
/// the third. Then every sequence decodes one token per step until its stream is consumed.
pub fn schedule(lens: &[usize], total: &[usize], chunk: usize) -> Vec<Step> {
    let mut cursor = vec![0usize; lens.len()];
    let mut steps = Vec::new();
    for step in 0.. {
        let mut s = Step::new();
        for k in 0..lens.len() {
            let joined = k < 2 || step >= 1;
            if !joined || cursor[k] >= total[k] {
                continue;
            }
            let end = if cursor[k] < lens[k] {
                if k == 2 && step == 1 {
                    chunk.min(lens[k])
                } else {
                    lens[k]
                }
            } else {
                cursor[k] + 1
            };
            s.push((k, cursor[k]..end));
            cursor[k] = end;
        }
        if s.is_empty() {
            break;
        }
        steps.push(s);
    }
    steps
}

/// Runs `steps` on `m` (sequence ids 0..) and returns, per step and input, the logits of its
/// last token.
pub fn run_steps(m: &mut Qwen3Moe, toks: &[Vec<i32>], steps: &[Step]) -> Vec<Vec<Vec<f32>>> {
    let nv = m.n_vocab();
    let mut out = Vec::new();
    for s in steps {
        let inputs: Vec<SeqEval> = s
            .iter()
            .map(|(k, r)| SeqEval {
                seq: *k as u32,
                tokens: &toks[*k][r.clone()],
                all_logits: false,
            })
            .collect();
        let mut buf = Vec::new();
        let spans = m.decode(&inputs, &mut buf).unwrap();
        assert_eq!(spans.len(), s.len());
        out.push(
            spans
                .iter()
                .map(|r| {
                    assert_eq!(r.len(), 1);
                    buf[r.start * nv..r.end * nv].to_vec()
                })
                .collect(),
        );
    }
    out
}

/// Compares the logits of `got` (from [`run_steps`]) with the alone runs wherever a step ends
/// at or after a prompt.
pub fn check_steps(
    got: &[Vec<Vec<f32>>],
    steps: &[Step],
    alones: &[Alone],
    tol: impl Into<Tol>,
    what: &str,
) -> usize {
    let tol = tol.into();
    let mut pairs = Vec::new();
    for (si, (s, rows)) in steps.iter().zip(got).enumerate() {
        for ((k, r), l) in s.iter().zip(rows) {
            let a = &alones[*k];
            if r.end < a.prompt_len {
                continue;
            }
            let want = &a.logits[r.end - a.prompt_len];
            let what = format!("{what}: step {si}, sequence {k}, {} tokens", r.end);
            print_row(&what, l, want);
            pairs.push((what, l, want));
        }
    }
    for (what, l, want) in &pairs {
        close(l, want, tol, what);
        same_token(l, want, tol, what);
    }
    pairs.len()
}

/// Three sequences with a prefill chunk mixed into decode steps, each against itself alone.
pub fn batched_equals_alone(
    path: &Path,
    opts: &LoadOptions,
    tol: impl Into<Tol> + Copy,
    what: &str,
) -> Qwen3Moe {
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let n = 6;
    let alones: Vec<Alone> = prompts.iter().map(|p| alone(path, opts, p, n)).collect();
    let toks: Vec<Vec<i32>> = alones.iter().map(|a| a.toks.clone()).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    assert!(
        steps.iter().any(|s| s.iter().any(|(_, r)| r.len() > 1)
            && s.iter().filter(|(_, r)| r.len() == 1).count() >= 2),
        "a prefill chunk shares a step with decode tokens"
    );
    let mut m = load(path, opts, 3);
    let got = run_steps(&mut m, &toks, &steps);
    let compared = check_steps(&got, &steps, &alones, tol, what);
    assert!(compared >= 3 * n, "{what}: {compared} comparisons");
    for (k, t) in toks.iter().enumerate() {
        assert_eq!(
            m.seq_len(k as u32),
            t.len(),
            "{what}: length of sequence {k}"
        );
    }
    m
}

/// The same batched steps on two configurations of one model (for example the serial streamed
/// path and the one with prefetch, the I/O service and overlap): the logits agree within `tol`.
/// This is for models where batched against alone is not a fair test: with 32 experts and top-8
/// on random weights, the 1% gap between a one-row and a several-row quantized kernel on some
/// CPUs flips a near-tied expert choice, and the logits move by a few percent.
pub fn batched_paths_agree(
    path: &Path,
    a: &LoadOptions,
    b: &LoadOptions,
    tol: impl Into<Tol> + Copy,
    what: &str,
) -> Qwen3Moe {
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let toks: Vec<Vec<i32>> = prompts.iter().map(|p| alone(path, b, p, 6).toks).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    let mut m = load(path, a, 3);
    let got = run_steps(&mut m, &toks, &steps);
    let want = run_steps(&mut load(path, b, 3), &toks, &steps);
    for (si, (g, w)) in got.iter().zip(&want).enumerate() {
        for (ii, (x, y)) in g.iter().zip(w).enumerate() {
            let what = format!("{what}: step {si}, input {ii}");
            close(x, y, tol, &what);
            same_token(x, y, tol, &what);
        }
    }
    m
}

/// `seq_cp` of a whole sequence and of a prefix, then continuing the copy, against a fresh
/// prefill of the same tokens; the source continues too.
pub fn seq_cp_continues(path: &Path, opts: &LoadOptions, tol: impl Into<Tol> + Copy, what: &str) {
    let p = prompt(8, 25);
    let extra = [11, 12, 13];
    let mut all = p.clone();
    all.extend(extra);
    let want = {
        let mut m = load(path, opts, 1);
        m.eval_last(&all).unwrap()
    };
    let mut m = load(path, opts, 3);
    m.eval_last(&p).unwrap();
    m.seq_cp(0, 1, p.len()).unwrap();
    assert_eq!(m.seq_len(1), p.len());
    let rows = |m: &Qwen3Moe, s| (m.pool().row(s, 0), m.pool().row(s, p.len() - 1));
    assert_ne!(rows(&m, 0).0, rows(&m, 1).0, "the copy has rows of its own");
    let mut buf = Vec::new();
    m.decode(
        &[SeqEval {
            seq: 1,
            tokens: &extra,
            all_logits: false,
        }],
        &mut buf,
    )
    .unwrap();
    let mut checks = vec![("copy continues", buf.clone())];
    // The source is untouched and continues alone.
    assert_eq!(m.seq_len(0), p.len());
    m.decode(
        &[SeqEval {
            seq: 0,
            tokens: &extra,
            all_logits: false,
        }],
        &mut buf,
    )
    .unwrap();
    checks.push(("source continues", buf.clone()));
    // A prefix copy into sequence 2.
    m.seq_cp(0, 2, 10).unwrap();
    m.decode(
        &[SeqEval {
            seq: 2,
            tokens: &all[10..],
            all_logits: false,
        }],
        &mut buf,
    )
    .unwrap();
    checks.push(("prefix copy", buf.clone()));
    for (name, got) in &checks {
        print_row(&format!("{what}: {name}"), got, &want);
    }
    for (name, got) in &checks {
        close(got, &want, tol, &format!("{what}: {name}"));
    }
    // dst must be empty, src long enough, ids in range.
    assert!(m.seq_cp(0, 1, 5).is_err());
    assert!(m.seq_cp(0, 0, 5).is_err());
    assert!(m.seq_cp(1, 2, 999).is_err());
    assert!(m.seq_cp(0, 3, 1).is_err());
}

/// A sequence's snapshot next to live ones: restored into another slot it decodes the same, its
/// bytes survive the roundtrip, and the others are untouched.
pub fn snapshot_roundtrip(path: &Path, opts: &LoadOptions, tol: impl Into<Tol> + Copy, what: &str) {
    let (a, b) = (prompt(9, 20), prompt(10, 14));
    let mut m = load(path, opts, 3);
    let mut buf = Vec::new();
    m.decode(
        &[
            SeqEval {
                seq: 0,
                tokens: &a,
                all_logits: false,
            },
            SeqEval {
                seq: 2,
                tokens: &b,
                all_logits: false,
            },
        ],
        &mut buf,
    )
    .unwrap();
    let snap = m.seq_snapshot(2).unwrap();
    let step = |m: &mut Qwen3Moe, seq| {
        let mut out = Vec::new();
        m.decode(
            &[SeqEval {
                seq,
                tokens: &[5],
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
        out
    };
    let want = step(&mut m, 2);
    m.seq_rm(2, 0).unwrap();
    assert_eq!((m.seq_len(2), m.seq_len(0)), (0, a.len()));
    m.seq_restore(1, &snap, b.len()).unwrap();
    assert_eq!(m.seq_len(1), b.len());
    assert_eq!(m.seq_snapshot(1).unwrap(), snap, "{what}: bytes roundtrip");
    close(&step(&mut m, 1), &want, tol, &format!("{what}: restored"));
    // Restoring over a live sequence replaces it.
    m.seq_restore(0, &snap, b.len()).unwrap();
    assert_eq!(m.seq_len(0), b.len());
    close(&step(&mut m, 0), &want, tol, &format!("{what}: replaced"));
    assert!(m.seq_restore(0, &snap[1..], b.len()).is_err());
}

/// A layout move in the middle of a decode (a neighbour stops sequence 0 growing in place). The
/// move copies the rows on the device: sequence 0's snapshot is byte-equal before and after it.
/// The decode after the move is compared with a reference that gets the same bytes at the same
/// rows from the host (`seq_restore`, two neighbours in the first 512 rows), so both run the same
/// kernels on the same rows and `n_kv`: equal within `tol`, bitwise on a deterministic backend.
/// Before the move the sequence decodes as it does alone (same rows, same `n_kv`).
///
/// The reference is not a run with sequence 0 at row 512 from the start: attention sums in an
/// order that depends on the row offset and `n_kv` (ggml's CPU flash attention splits a
/// one-query step into `n_kv / threads` chunks from `n_kv` 512 and accumulates V in F16; CUDA
/// splits over KV blocks), so such a run differs by kernel noise before the move, and a fixture
/// with near-tied experts turns that into percent (the first CUDA runs, ADR 0071).
///
/// The move is the reserve a decode of the 257th token makes (`decode` calls the same
/// `seq_reserve`), taken explicitly so the bytes compare at an unchanged length. `opts` must have
/// a pool of at least 1024 rows (the move takes the gap behind the neighbour, 512 rows, when no
/// gap holds the extra GROW).
pub fn move_mid_decode(path: &Path, opts: &LoadOptions, tol: impl Into<Tol> + Copy, what: &str) {
    assert!(opts.n_ctx >= 1024);
    let p = prompt(4, 200);
    let neighbour = prompt(5, 10);
    let n = 80;
    let lone = alone(path, opts, &p, n);
    let toks = &lone.toks;
    let decode_one = |m: &mut Qwen3Moe, t: i32| {
        let mut buf = Vec::new();
        m.decode(
            &[SeqEval {
                seq: 0,
                tokens: &[t],
                all_logits: false,
            }],
            &mut buf,
        )
        .unwrap();
        buf
    };

    let mut m = load(path, opts, 2);
    m.eval_seq(0, &p);
    // The neighbour takes the rows behind sequence 0's 256.
    m.eval_seq(1, &neighbour);
    assert_eq!(m.pool().row(0, 0), Some(0));
    let nb_before = m.seq_snapshot(1).unwrap();
    // Decode until sequence 0 fills its rows: (logits, the alone run's after the same tokens).
    let mut before_move: Vec<(Vec<f32>, &Vec<f32>)> = Vec::new();
    while m.seq_len(0) < m.pool().reserved(0) {
        let j = before_move.len();
        let g = decode_one(&mut m, toks[p.len() + j]);
        before_move.push((g, &lone.logits[j + 1]));
    }
    let moved_at = before_move.len();
    let len = m.seq_len(0);
    assert_eq!(len, 256, "{what}: sequence 0 fills its first run");
    let bytes = m.seq_snapshot(0).unwrap();
    m.seq_reserve(0, len + 1).unwrap();
    assert_eq!(m.pool().row(0, 0), Some(512), "{what}: sequence 0 moved");
    let same_bytes = m.seq_snapshot(0).unwrap() == bytes;

    // The reference: two neighbours in the first 512 rows, the same bytes behind them.
    let mut r = load(path, opts, 3);
    r.eval_seq(1, &neighbour);
    r.eval_seq(2, &neighbour);
    r.seq_restore(0, &bytes, len).unwrap();
    assert_eq!(r.pool().row(0, 0), Some(512), "{what}: the reference's rows");
    let after_move: Vec<(Vec<f32>, Vec<f32>)> = (moved_at..n)
        .map(|j| {
            let t = toks[p.len() + j];
            (decode_one(&mut m, t), decode_one(&mut r, t))
        })
        .collect();

    let rows: Vec<(String, &[f32], &[f32])> = before_move
        .iter()
        .enumerate()
        .map(|(j, (g, w))| (format!("{what}: step {j}, alone"), &g[..], &w[..]))
        .chain(after_move.iter().enumerate().map(|(k, (g, w))| {
            let j = moved_at + k;
            (format!("{what}: step {j}, moved"), &g[..], &w[..])
        }))
        .collect();
    for (what, g, want) in &rows {
        print_row(what, g, want);
    }
    assert!(same_bytes, "{what}: the moved rows hold other bytes");
    for (what, g, want) in &rows {
        close(g, want, tol, what);
        same_token(g, want, tol, what);
    }
    assert_eq!(m.seq_len(0), p.len() + n);
    assert_eq!(
        m.seq_snapshot(1).unwrap(),
        nb_before,
        "{what}: the neighbour"
    );
}

trait EvalSeq {
    fn eval_seq(&mut self, seq: u32, tokens: &[i32]);
}

impl EvalSeq for Qwen3Moe {
    fn eval_seq(&mut self, seq: u32, tokens: &[i32]) {
        let mut buf = Vec::new();
        self.decode(
            &[SeqEval {
                seq,
                tokens,
                all_logits: false,
            }],
            &mut buf,
        )
        .unwrap();
    }
}

/// Every input of `steps` as a step of its own: the same token ranges, one sequence per call.
pub fn split_steps(steps: &[Step]) -> Vec<Step> {
    steps.iter().flatten().map(|i| vec![i.clone()]).collect()
}

/// `steps` as a libllama script (logits of each input's last token).
pub fn script_items(steps: &[Step]) -> Vec<Vec<Item>> {
    steps
        .iter()
        .map(|s| {
            s.iter()
                .map(|(k, r)| Item {
                    seq: *k as u32,
                    range: r.clone(),
                    all: false,
                })
                .collect()
        })
        .collect()
}

/// [`run_steps`] output keyed like a script: (sequence, position of the input's last token).
pub fn steps_logits(rows: Vec<Vec<Vec<f32>>>, steps: &[Step]) -> Logits {
    let mut out = Logits::new();
    for (s, rows) in steps.iter().zip(rows) {
        for ((k, r), l) in s.iter().zip(rows) {
            out.insert((*k as u32, r.end - 1), l);
        }
    }
    out
}

/// The anchor of the batched tolerance: libllama (`n_gpu_layers`) and the builder (`opts`) run
/// the schedule of [`batched_equals_alone`] batched and split (every input its own call), and
/// libllama batched on the CPU. Prints every comparison, then asserts each within `tol`:
/// libllama batched against split is llama.cpp's own batching noise on that backend; the
/// builder against libllama with the same step shapes runs the same kernels, so a mask, row or
/// position bug shows there while kernel noise mostly cancels. Returns the worst relative gap
/// per comparison, in print order.
pub fn libllama_anchor(
    path: &Path,
    opts: &LoadOptions,
    n_gpu_layers: i32,
    tol: Tol,
    what: &str,
) -> Vec<f32> {
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let toks: Vec<Vec<i32>> = prompts
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let mut t = p.clone();
            t.extend(prompt(11 + k as i32, 6));
            t
        })
        .collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    let split = split_steps(&steps);
    let kv = opts.kv;
    let ll = |s: &[Step], ngl| libllama_script_on(path, &toks, &script_items(s), kv, ngl);
    let ours = |s: &[Step]| steps_logits(run_steps(&mut load(path, opts, 3), &toks, s), s);
    let (ll_b, ll_s) = (ll(&steps[..], n_gpu_layers), ll(&split[..], n_gpu_layers));
    let ll_cpu = ll(&steps[..], 0);
    let (our_b, our_s) = (ours(&steps[..]), ours(&split[..]));
    anchor_pairs(what, &ll_b, &ll_s, &ll_cpu, &our_b, &our_s, tol)
}

//! Several sequences in one libllama context (ADR 0065, C2): `decode_seqs` against each
//! sequence alone, `seq_cp`, snapshots and error handling, on the tiny Qwen3 MoE fixture.
//!
//! Hardware label: cpu.

use std::path::PathBuf;
use tetraeder_engine::{ContextConfig, Engine, Placement, SeqEval};

/// Batched matmul kernels differ from single-row ones (ADR 0043): equal within this, not bitwise.
const TOL_CPU: f32 = 1e-4;

fn engine(n_seq_max: u32, n_batch: u32) -> Engine {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    let cfg = ContextConfig {
        n_ctx: 1024,
        n_batch,
        n_ubatch: n_batch,
        threads: Some(2),
        n_seq_max,
        kv_unified: true,
        ..ContextConfig::default()
    };
    Engine::load(path, &Placement::cpu(), &cfg, None).expect("fixture loads")
}

fn argmax(l: &[f32]) -> i32 {
    l.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0 as i32
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    // Which kernels a CPU variant uses for one row and for several decides the gap: CI's runners
    // show ~9e-3 where the dev box shows less. Equal within TOL_CPU or 1e-3 of the largest logit.
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = TOL_CPU.max(1e-3 * scale);
    assert!(
        worst <= tol,
        "max |diff| {worst} > {tol} (largest logit {scale})"
    );
}

fn prompt(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32)
        .map(|i| 4 + (seed * 31 + i * 7) % 200)
        .collect()
}

/// Greedy tokens and last logits of `prompt` then `steps` decoded tokens, alone on sequence 0.
fn alone(prompt: &[i32], steps: usize) -> (Vec<i32>, Vec<f32>) {
    let mut e = engine(1, 64);
    let mut l = e.eval(prompt).unwrap().to_vec();
    let mut out = Vec::new();
    for _ in 0..steps {
        let t = argmax(&l);
        out.push(t);
        l = e.eval(&[t]).unwrap().to_vec();
    }
    (out, l)
}

#[test]
fn four_sequences_decoded_together_equal_each_alone() {
    let prompts = [prompt(1, 5), prompt(2, 17), prompt(3, 1), prompt(4, 40)];
    let steps = 6;
    let want: Vec<_> = prompts.iter().map(|p| alone(p, steps)).collect();

    let mut e = engine(4, 64);
    assert_eq!(e.n_seq_max(), 4);
    // Prefill of sequences 0..3 as one step, with a further prefill chunk mixed into the
    // decode steps below.
    let inputs: Vec<SeqEval> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| SeqEval {
            seq: i as i32,
            tokens: p,
            all_logits: false,
        })
        .collect();
    e.decode_seqs(&inputs).unwrap();
    assert_eq!(e.n_outputs(), 4);
    let mut next: Vec<i32> = (0..4)
        .map(|k| argmax(e.output_logits(k).unwrap()))
        .collect();
    let mut got: Vec<Vec<i32>> = vec![Vec::new(); 4];
    let mut last: Vec<Vec<f32>> = vec![Vec::new(); 4];
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(e.seq_len(i as i32), p.len());
    }
    for _ in 0..steps {
        for k in 0..4 {
            got[k].push(next[k]);
        }
        let toks: Vec<[i32; 1]> = next.iter().map(|&t| [t]).collect();
        let inputs: Vec<SeqEval> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| SeqEval {
                seq: i as i32,
                tokens: t,
                all_logits: false,
            })
            .collect();
        e.decode_seqs(&inputs).unwrap();
        for k in 0..4 {
            last[k] = e.output_logits(k).unwrap().to_vec();
            next[k] = argmax(&last[k]);
        }
    }
    for k in 0..4 {
        assert_eq!(got[k], want[k].0, "greedy tokens of sequence {k}");
        close(&last[k], &want[k].1);
    }
}

#[test]
fn a_prefill_chunk_mixed_with_decode_rows_changes_nothing() {
    let (a, b) = (prompt(5, 12), prompt(6, 30));
    let (want_a, la) = alone(&a, 3);
    let (_, lb) = alone(&b, 0);

    let mut e = engine(2, 64);
    // Sequence 0 prefills, then decodes while sequence 1 prefills in two chunks.
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &a,
        all_logits: false,
    }])
    .unwrap();
    let mut got_a = Vec::new();
    let mut t = argmax(e.output_logits(0).unwrap());
    let split = 20;
    for step in 0..3 {
        got_a.push(t);
        let chunk: &[i32] = if step == 0 {
            &b[..split]
        } else if step == 1 {
            &b[split..]
        } else {
            &[]
        };
        let tok = [t];
        let mut inputs = vec![SeqEval {
            seq: 0,
            tokens: &tok,
            all_logits: false,
        }];
        if !chunk.is_empty() {
            inputs.push(SeqEval {
                seq: 1,
                tokens: chunk,
                all_logits: false,
            });
        }
        e.decode_seqs(&inputs).unwrap();
        let l = e.output_logits(0).unwrap().to_vec();
        t = argmax(&l);
        if step == 2 {
            close(&l, &la);
        }
        if step == 1 {
            // Sequence 1's prompt is complete: its last logits equal the lone prefill's.
            close(e.output_logits(1).unwrap(), &lb);
        }
    }
    assert_eq!(got_a, want_a);
    assert_eq!(e.seq_len(1), b.len());
}

#[test]
fn all_logits_returns_every_row_in_order() {
    let p = prompt(7, 9);
    let mut e = engine(2, 64);
    e.decode_seqs(&[
        SeqEval {
            seq: 1,
            tokens: &p,
            all_logits: true,
        },
        SeqEval {
            seq: 0,
            tokens: &p[..3],
            all_logits: false,
        },
    ])
    .unwrap();
    assert_eq!(e.n_outputs(), 9 + 1);
    // Row k of the first input is the lone logits after the first k+1 tokens.
    for k in [0usize, 4, 8] {
        let mut lone = engine(1, 64);
        let want = lone.eval(&p[..=k]).unwrap().to_vec();
        close(e.output_logits(k).unwrap(), &want);
    }
    assert!(e.output_logits(10).is_err());
}

#[test]
fn seq_cp_then_continue_equals_a_fresh_prefill() {
    let p = prompt(8, 25);
    let extra = [11, 12, 13];
    let mut fresh = engine(1, 64);
    let mut all = p.clone();
    all.extend(extra);
    let want = fresh.eval(&all).unwrap().to_vec();

    let mut e = engine(2, 64);
    e.eval(&p).unwrap();
    e.seq_cp(0, 1, p.len()).unwrap();
    assert_eq!(e.seq_len(1), p.len());
    e.decode_seqs(&[SeqEval {
        seq: 1,
        tokens: &extra,
        all_logits: false,
    }])
    .unwrap();
    close(e.output_logits(0).unwrap(), &want);
    // The source is untouched and can diverge.
    assert_eq!(e.seq_len(0), p.len());
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &extra,
        all_logits: false,
    }])
    .unwrap();
    close(e.output_logits(0).unwrap(), &want);

    // A prefix copy.
    let mut e = engine(2, 64);
    e.eval(&p).unwrap();
    e.seq_cp(0, 1, 10).unwrap();
    assert_eq!(e.seq_len(1), 10);
    e.decode_seqs(&[SeqEval {
        seq: 1,
        tokens: &p[10..],
        all_logits: false,
    }])
    .unwrap();
    let want = engine(1, 64).eval(&p).unwrap().to_vec();
    close(e.output_logits(0).unwrap(), &want);

    // dst must be empty, src long enough.
    assert!(e.seq_cp(0, 1, 5).is_err());
    assert!(e.seq_cp(0, 0, 5).is_err());
    assert!(e.seq_cp(1, 0, 99).is_err());
}

#[test]
fn snapshot_and_restore_a_sequence_next_to_others() {
    let (a, b) = (prompt(9, 20), prompt(10, 14));
    let mut e = engine(3, 64);
    e.decode_seqs(&[
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
    ])
    .unwrap();
    let snap = e.seq_snapshot(2).unwrap();
    e.decode_seqs(&[SeqEval {
        seq: 2,
        tokens: &[5],
        all_logits: false,
    }])
    .unwrap();
    let want = e.output_logits(0).unwrap().to_vec();
    e.seq_rm(2, 0).unwrap();
    assert_eq!(e.seq_len(2), 0);
    assert_eq!(e.seq_len(0), a.len(), "other sequences are untouched");
    e.seq_restore(1, &snap, b.len()).unwrap();
    assert_eq!(e.seq_len(1), b.len());
    e.decode_seqs(&[SeqEval {
        seq: 1,
        tokens: &[5],
        all_logits: false,
    }])
    .unwrap();
    close(e.output_logits(0).unwrap(), &want);
}

#[test]
fn errors_leave_lengths_unchanged() {
    let p = prompt(11, 10);
    let mut e = engine(2, 16);
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &p,
        all_logits: false,
    }])
    .unwrap();
    let ok = [1, 2];
    // More tokens than n_batch.
    let big = vec![3; 17];
    assert!(
        e.decode_seqs(&[
            SeqEval {
                seq: 0,
                tokens: &ok,
                all_logits: false
            },
            SeqEval {
                seq: 1,
                tokens: &big,
                all_logits: false
            },
        ])
        .is_err()
    );
    // A repeated sequence, an unknown one, an empty input, a sequence beyond the context.
    for bad in [
        vec![
            SeqEval {
                seq: 0,
                tokens: &ok,
                all_logits: false,
            },
            SeqEval {
                seq: 0,
                tokens: &ok,
                all_logits: false,
            },
        ],
        vec![SeqEval {
            seq: 2,
            tokens: &ok,
            all_logits: false,
        }],
        vec![SeqEval {
            seq: 1,
            tokens: &[],
            all_logits: false,
        }],
    ] {
        assert!(e.decode_seqs(&bad).is_err());
    }
    assert_eq!(e.seq_len(0), p.len());
    assert_eq!(e.seq_len(1), 0);
    assert!(e.seq_rm(5, 0).is_err());
    // The context still works.
    e.decode_seqs(&[SeqEval {
        seq: 1,
        tokens: &ok,
        all_logits: false,
    }])
    .unwrap();
    assert_eq!((e.seq_len(0), e.seq_len(1)), (p.len(), 2));
}

#[test]
fn single_sequence_api_is_sequence_zero() {
    let mut e = engine(3, 64);
    e.eval(&[10, 20, 30]).unwrap();
    assert_eq!(e.n_past(), 3);
    assert_eq!(e.seq_len(0), 3);
    e.truncate(1).unwrap();
    assert_eq!(e.seq_len(0), 1);
    let (bytes, n) = e.snapshot().unwrap();
    e.restore(&bytes, n).unwrap();
    assert_eq!(e.seq_len(0), 1);
}

#[test]
fn a_full_pool_fails_the_step_and_leaves_every_length() {
    use tetraeder_engine::EngineError;
    use tetraeder_ggml_sys::llama::LlamaError;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    let cfg = ContextConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 256,
        threads: Some(2),
        n_seq_max: 2,
        ..ContextConfig::default()
    };
    let mut e = Engine::load(path, &Placement::cpu(), &cfg, None).unwrap();
    // libllama may round the pool up.
    let n_ctx = e.n_ctx();
    let (a, b) = (prompt(12, n_ctx - 10), prompt(13, 40));
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &a,
        all_logits: false,
    }])
    .unwrap();
    // Each sequence fits the context alone, both together do not fit the pool.
    let err = e
        .decode_seqs(&[
            SeqEval {
                seq: 1,
                tokens: &b[..20],
                all_logits: false,
            },
            SeqEval {
                seq: 0,
                tokens: &[1],
                all_logits: false,
            },
        ])
        .unwrap_err();
    assert!(
        matches!(err, EngineError::Llama(LlamaError::NoKvSlot)),
        "{err}"
    );
    assert_eq!((e.seq_len(0), e.seq_len(1)), (n_ctx - 10, 0));
    // The pool is intact: sequence 0 continues as an engine that never saw the failed step.
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &[1],
        all_logits: false,
    }])
    .unwrap();
    let got = e.output_logits(0).unwrap().to_vec();
    let mut control = Engine::load(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf"),
        &Placement::cpu(),
        &cfg,
        None,
    )
    .unwrap();
    for t in [&a[..], &[1][..]] {
        control
            .decode_seqs(&[SeqEval {
                seq: 0,
                tokens: t,
                all_logits: false,
            }])
            .unwrap();
    }
    close(&got, control.output_logits(0).unwrap());
}

/// A step that fails on a later ubatch (the first one fits the pool, the second does not) leaves
/// no cells of the first behind: the sequence takes a decode at its old length.
#[test]
fn a_step_failing_on_a_later_ubatch_leaves_no_orphan_cells() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    let cfg = ContextConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 64,
        threads: Some(2),
        n_seq_max: 2,
        ..ContextConfig::default()
    };
    let mut e = Engine::load(path, &Placement::cpu(), &cfg, None).unwrap();
    let n_ctx = e.n_ctx();
    let (a, b) = (prompt(12, n_ctx - 126), prompt(13, 200));
    let step = |e: &mut Engine, seq, toks: &[i32]| {
        e.decode_seqs(&[SeqEval {
            seq,
            tokens: toks,
            all_logits: false,
        }])
    };
    step(&mut e, 0, &a).unwrap();
    // 126 rows are free: the first ubatch of 64 fits, the second does not.
    assert!(step(&mut e, 1, &b).is_err());
    assert_eq!(e.seq_len(1), 0);
    step(&mut e, 1, &b[..50]).unwrap();
    assert_eq!(e.seq_len(1), 50);
    // The rollback primitive cuts at the length it is given, also when that is the length held.
    e.seq_truncate_cells(1, 50).unwrap();
    step(&mut e, 1, &b[50..60]).unwrap();
    assert_eq!(e.seq_len(1), 60);
}

/// A restore that libllama rejects partway can leave cells the length does not count; clearing
/// the sequence must drop them even though `seq_len` is 0, so a decode at position 0 works.
#[test]
fn clearing_an_empty_looking_sequence_drops_cells_left_by_a_failed_restore() {
    let a = prompt(12, 40);
    let mut e = engine(2, 64);
    e.decode_seqs(&[SeqEval {
        seq: 0,
        tokens: &a,
        all_logits: false,
    }])
    .unwrap();
    let snap = e.seq_snapshot(0).unwrap();
    let mut saw_leftover = false;
    // Cut the snapshot at several points: the header and cell metadata may parse, the data not.
    for cut in (snap.len() / 8..snap.len()).step_by(snap.len() / 16) {
        let mut e = engine(2, 64);
        if e.seq_restore(1, &snap[..cut], a.len()).is_ok() {
            continue;
        }
        assert_eq!(e.seq_len(1), 0);
        e.seq_rm(1, 0).unwrap();
        let p = prompt(13, 5);
        e.decode_seqs(&[SeqEval {
            seq: 1,
            tokens: &p,
            all_logits: false,
        }])
        .expect("a cleared sequence starts again at position 0");
        saw_leftover = true;
    }
    assert!(saw_leftover, "no truncated snapshot was rejected");
}

//! The sharded dense builder on CPU ranks (ADR 0069, 0070, brief E6): every shape against the
//! one-rank builder on the tiny dense fixtures (2 layers, and 4 layers from
//! `cargo xtask tiny-dense --layers 4`), batched sequences, KV snapshots across shapes and
//! `seq_cp`.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU. CPU ranks stand in for GPUs; the
//! all-reduce runs on the executor's fallback path.

mod common;

use common::multi_seq::{close, print_row, same_token};
use common::shapes::{layer_split, single, tp, tp2pp2};
use common::*;
use std::path::Path;
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3, SeqEval};
use tetraeder_plan::{KvFormat, Topology, micro_batch_sizes};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

fn opts(n_seq_max: usize) -> DenseOptions {
    DenseOptions {
        kv: KvFormat::F16,
        kv_rotation: Some(false),
        // CPU ranks hold weights in file layout; the scheduler path would repack them in the
        // CPU's extra buffer types and differ in the last bits.
        extra_bufts: false,
        n_seq_max,
        // A sequence reserves whole 256-row units.
        n_ctx_pool: 256 * n_seq_max,
        ..DenseOptions::new(CPU)
    }
}

/// The scheduler path on the CPU backend: the reference of every comparison.
fn one_rank(path: &Path, n_seq_max: usize) -> Qwen3 {
    Qwen3::load(path, &opts(n_seq_max)).unwrap()
}

fn sharded(path: &Path, topo: &Topology, n_seq_max: usize) -> Qwen3 {
    Qwen3::load_sharded(path, topo, &opts(n_seq_max)).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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

/// hardware: cpu. One CPU rank on the executor equals the scheduler path bit for bit: the
/// logits of every prompt position and 12 greedy steps.
#[test]
fn cpu_one_rank_is_bit_equal_to_the_scheduler() {
    let path = dense_fixture();
    let mut a = one_rank(&path, 1);
    let mut b = sharded(&path, &single(2), 1);
    assert_eq!((a.n_ranks(), b.n_ranks()), (1, 1));
    assert!(a.exec_counters().is_none());
    assert_eq!(
        bits(&a.eval(&PROMPT).unwrap()),
        bits(&b.eval(&PROMPT).unwrap())
    );
    a.reset();
    b.reset();
    b.reset_exec_counters();
    let mut la = a.eval_last(&PROMPT).unwrap();
    let mut lb = b.eval_last(&PROMPT).unwrap();
    assert_eq!(bits(&la), bits(&lb));
    for step in 0..N_GREEDY {
        let t = argmax(&la) as i32;
        assert_eq!(t as usize, argmax(&lb));
        la = a.eval_last(&[t]).unwrap();
        lb = b.eval_last(&[t]).unwrap();
        assert_eq!(bits(&la), bits(&lb), "step {step}");
    }
    // One range per step, no collective.
    let c = b.exec_counters().unwrap();
    assert_eq!(c.ranges, vec![1 + N_GREEDY as u64]);
    assert!(c.collective_bytes.is_empty());
}

/// hardware: cpu. The kept step graph gives the logits of one built per step.
#[test]
fn cpu_executor_step_graph_reuse_is_bit_equal() {
    let run = |reuse: bool| {
        let o = DenseOptions {
            reuse_step_graph: reuse,
            ..opts(1)
        };
        let mut m = Qwen3::load_sharded(dense_fixture(), &tp(2, 2), &o).unwrap();
        let mut all = m.eval_last(&PROMPT).unwrap();
        for t in [5, 6, 7, 8] {
            all.extend(m.eval_last(&[t]).unwrap());
        }
        (bits(&all), m.step_graphs_built())
    };
    let ((a, built_a), (b, built_b)) = (run(true), run(false));
    assert_eq!(a, b);
    assert_eq!((built_a, built_b), (2, 5));
}

/// hardware: cpu. `tp-2` and `layer-2` (split after layer 1) on the 2-layer fixture: top-5
/// within `tol_batch`, greedy equal or a near tie (ADR 0070, "Numerics").
#[test]
fn cpu_two_layer_shapes_match_one_rank() {
    let path = dense_fixture();
    let reference = builder_reference(&mut one_rank(&path, 1));
    for (name, topo) in [("tp-2", tp(2, 2)), ("layer-2", layer_split(2, 1))] {
        let mut m = sharded(&path, &topo, 1);
        assert_eq!(m.n_ranks(), 2);
        let worst = check_against_tol(&mut m, &reference, Tol::from(TOL_CPU));
        eprintln!("{name}: max |delta| over the top-{TOP_K} logits: {worst:e}");
    }
}

/// hardware: cpu. The 4-layer fixture: `tp2-pp2` and a layer split at each layer boundary
/// (Rule 2's split points).
#[test]
fn cpu_four_layer_shapes_match_one_rank() {
    let path = dense_fixture_l4();
    let reference = builder_reference(&mut one_rank(&path, 1));
    let mut shapes = vec![("tp-2", tp(4, 2)), ("tp2-pp2", tp2pp2(4, 2))];
    for at in 1..4 {
        shapes.push(("layer-2", layer_split(4, at)));
    }
    for (name, topo) in shapes {
        let mut m = sharded(&path, &topo, 1);
        let worst = check_against_tol(&mut m, &reference, Tol::from(TOL_CPU));
        eprintln!(
            "{name} ({} ranks, stages {:?}): max |delta| {worst:e}",
            m.n_ranks(),
            topo.stages.iter().map(|s| s.layers).collect::<Vec<_>>()
        );
    }
}

/// hardware: cpu. A layer split keeps every op and its order, and the hop is a copy, so its
/// logits equal the one-rank run bit for bit.
#[test]
fn cpu_layer_split_is_bit_equal() {
    let path = dense_fixture_l4();
    let mut a = one_rank(&path, 1);
    let la = a.eval(&PROMPT).unwrap();
    for at in 1..4 {
        let mut b = sharded(&path, &layer_split(4, at), 1);
        let lb = b.eval(&PROMPT).unwrap();
        assert_eq!(bits(&la), bits(&lb), "split at {at}");
    }
}

/// hardware: cpu. Ranges per decode token are ADR 0070's: `2 * layers + 1` per rank on a tensor
/// stage (attention, FFN, and the head after the last reduce), one per rank of a layer split,
/// and the bytes the collectives move.
#[test]
fn cpu_ranges_and_collective_bytes_per_decode_token() {
    let path = dense_fixture_l4();
    let n_embd_bytes = 128u64 * 4;
    let decode = |topo: &Topology| {
        let mut m = sharded(&path, topo, 1);
        m.eval_last(&PROMPT).unwrap();
        m.reset_exec_counters();
        m.eval_last(&[5]).unwrap();
        m.exec_counters().unwrap().clone()
    };
    // tp-2: two all-reduces per layer, each summing both ranks' [n_embd, 1] tensors.
    let c = decode(&tp(4, 2));
    assert_eq!(c.ranges, vec![2 * 4 + 1; 2]);
    assert_eq!(
        c.collective_bytes.get(&("all_reduce", "fallback")),
        Some(&(2 * 4 * 2 * n_embd_bytes))
    );
    // layer-2: one range per rank and one hop of [n_embd, 1].
    let c = decode(&layer_split(4, 2));
    assert_eq!(c.ranges, vec![1, 1]);
    assert_eq!(
        c.collective_bytes.get(&("copy", "fallback")),
        Some(&n_embd_bytes)
    );
    // tp2-pp2: each stage has 2 layers (4 ranges and the last), one hop per pair.
    let c = decode(&tp2pp2(4, 2));
    assert_eq!(c.ranges, vec![2 * 2 + 1; 4]);
    assert_eq!(
        c.collective_bytes.get(&("all_reduce", "fallback")),
        Some(&(2 * 4 * 2 * n_embd_bytes))
    );
    assert_eq!(
        c.collective_bytes.get(&("copy", "fallback")),
        Some(&(2 * n_embd_bytes))
    );
}

/// hardware: cpu. Three sequences of different lengths in one step on `tp-2`, and then decoding
/// together, against each sequence alone on the same shape.
#[test]
fn cpu_three_sequences_batched_on_tp2_match_each_alone() {
    let path = dense_fixture();
    let topo = tp(2, 2);
    let lens = [6usize, 5, 4];
    let streams = [stream(7, 9), stream(8, 8), stream(9, 7)];
    let n_decode = 3;
    let nv = 260;
    // Rows of the last token per sequence and step: step 0 is the prefill chunk.
    let run = |seqs: &[usize], m: &mut Qwen3, seq_of: &dyn Fn(usize) -> u32| {
        let mut rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); seqs.len()];
        for step in 0..=n_decode {
            let items: Vec<SeqEval<'_>> = seqs
                .iter()
                .map(|&s| {
                    let tokens = if step == 0 {
                        &streams[s][..lens[s]]
                    } else {
                        &streams[s][lens[s] + step - 1..lens[s] + step]
                    };
                    SeqEval {
                        seq: seq_of(s),
                        tokens,
                        all_logits: false,
                    }
                })
                .collect();
            let mut out = Vec::new();
            let ranges = m.decode(&items, &mut out).unwrap();
            for (k, r) in ranges.iter().enumerate() {
                rows[k].push(out[r.start * nv..r.end * nv].to_vec());
            }
        }
        rows
    };
    let mut batched = sharded(&path, &topo, 3);
    assert_eq!(batched.n_vocab(), nv);
    let together = run(&[0, 1, 2], &mut batched, &|s| s as u32);
    #[allow(clippy::needless_range_loop)]
    for s in 0..3 {
        let mut alone = sharded(&path, &topo, 1);
        let rows = run(&[s], &mut alone, &|_| 0);
        for (step, want) in rows[0].iter().enumerate() {
            let got = &together[s][step];
            let what = format!("sequence {s} step {step}");
            print_row(&what, got, want);
            close(got, want, TOL_CPU, &what);
            same_token(got, want, TOL_CPU, &what);
        }
    }
}

/// The shapes whose snapshots must interchange.
fn snapshot_shapes() -> Vec<(&'static str, Option<Topology>)> {
    vec![
        ("one-rank", None),
        ("tp-2", Some(tp(2, 2))),
        ("layer-2", Some(layer_split(2, 1))),
    ]
}

fn load_shape(path: &Path, topo: &Option<Topology>, n_seq_max: usize) -> Qwen3 {
    match topo {
        None => one_rank(path, n_seq_max),
        Some(t) => sharded(path, t, n_seq_max),
    }
}

/// hardware: cpu. A `tp-2` and a `layer-2` snapshot have the one-rank layout (ADR 0069, "KV
/// snapshots and copies"): the same length, and byte for byte the same on layer 0, whose inputs
/// are identical on every shape.
#[test]
fn cpu_snapshots_have_the_one_rank_layout() {
    let path = dense_fixture();
    let n = PROMPT.len();
    let mut snaps = Vec::new();
    for (name, topo) in snapshot_shapes() {
        let mut m = load_shape(&path, &topo, 1);
        m.eval_last(&PROMPT).unwrap();
        let s = m.seq_snapshot(0).unwrap();
        eprintln!("{name}: {} bytes", s.len());
        snaps.push((name, s));
    }
    let n_layer = 2;
    let (_, base) = &snaps[0];
    let row = base.len() / (n * 2 * n_layer);
    assert_eq!(row, 2 * 2 * 64, "2 KV heads of 64, F16");
    let layer0 = |s: &[u8]| {
        // Tensor order: all K layers, then all V layers.
        let k = s[..n * row].to_vec();
        let v = s[n_layer * n * row..(n_layer + 1) * n * row].to_vec();
        (k, v)
    };
    for (name, s) in &snaps[1..] {
        assert_eq!(s.len(), base.len(), "{name}: length");
        assert_eq!(layer0(s), layer0(base), "{name}: layer 0");
    }
}

/// hardware: cpu. A snapshot taken under one shape restores under every shape, after which
/// decoding matches the run that never left the first shape within `tol_batch`; the same shape
/// round-trips bit for bit.
#[test]
fn cpu_snapshots_restore_across_shapes() {
    let path = dense_fixture();
    let n = PROMPT.len();
    for (from, ftopo) in snapshot_shapes() {
        let mut src = load_shape(&path, &ftopo, 1);
        src.eval_last(&PROMPT).unwrap();
        let snap = src.seq_snapshot(0).unwrap();
        let want = src.eval_last(&[5]).unwrap();
        for (to, ttopo) in snapshot_shapes() {
            let mut dst = load_shape(&path, &ttopo, 1);
            dst.seq_restore(0, &snap, n).unwrap();
            assert_eq!(dst.n_past(), n);
            let again = dst.seq_snapshot(0).unwrap();
            assert_eq!(again, snap, "{from} -> {to}: the restored bytes read back");
            let got = dst.eval_last(&[5]).unwrap();
            let what = format!("{from} -> {to}");
            print_row(&what, &got, &want);
            if from == to {
                assert_eq!(bits(&got), bits(&want), "{what}: same shape");
            } else {
                close(&got, &want, TOL_CPU, &what);
                same_token(&got, &want, TOL_CPU, &what);
            }
        }
    }
}

/// hardware: cpu. `seq_cp` under `tp-2` equals two separate prefills of the prefix.
#[test]
fn cpu_seq_cp_under_tp2_equals_a_separate_prefill() {
    let path = dense_fixture();
    let topo = tp(2, 2);
    let prefix = &PROMPT[..4];
    let mut m = sharded(&path, &topo, 2);
    let mut out = Vec::new();
    m.decode(
        &[SeqEval {
            seq: 0,
            tokens: &PROMPT,
            all_logits: false,
        }],
        &mut out,
    )
    .unwrap();
    m.seq_cp(0, 1, prefix.len()).unwrap();
    assert_eq!(m.pool().seq_len(1), prefix.len());
    m.decode(
        &[SeqEval {
            seq: 1,
            tokens: &[5],
            all_logits: false,
        }],
        &mut out,
    )
    .unwrap();

    let mut fresh = sharded(&path, &topo, 1);
    fresh.eval_last(prefix).unwrap();
    let want = fresh.eval_last(&[5]).unwrap();
    print_row("seq_cp", &out, &want);
    close(&out, &want, TOL_CPU, "seq_cp");
    same_token(&out, &want, TOL_CPU, "seq_cp");
}

/// hardware: cpu. `layer_limit` runs the first layers and the head: the timing is real, the
/// logits are not the model's. One range, one rank's graph only.
#[test]
fn cpu_layer_limit_runs_the_first_layers() {
    let path = dense_fixture_l4();
    let o = DenseOptions {
        layer_limit: Some(1),
        ..opts(1)
    };
    let mut m = Qwen3::load_sharded(&path, &single(4), &o).unwrap();
    let mut full = sharded(&path, &single(4), 1);
    let (a, b) = (
        m.eval_last(&PROMPT).unwrap(),
        full.eval_last(&PROMPT).unwrap(),
    );
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|x| x.is_finite()));
    assert_ne!(bits(&a), bits(&b));
    // Less KV and weights than the whole model.
    let bytes = |m: &Qwen3| m.memory().iter().map(|u| u.capacity).sum::<u64>();
    assert!(bytes(&m) < bytes(&full));
    // The scheduler path has no layer limit, and a limit needs one stage.
    assert!(Qwen3::load(&path, &o).is_err());
    assert!(Qwen3::load_sharded(&path, &layer_split(4, 2), &o).is_err());
}

/// hardware: cpu. Shapes the builder cannot run say why at load.
#[test]
fn cpu_load_refuses_what_it_cannot_run() {
    let path = dense_fixture();
    let err = |t: &Topology| {
        Qwen3::load_sharded(&path, t, &opts(1))
            .unwrap_err()
            .to_string()
    };
    // 2 KV heads over 4 ranks.
    assert!(err(&tp(2, 4)).contains("KV heads"), "{}", err(&tp(2, 4)));
    // The topology's layer count must be the model's.
    assert!(err(&single(3)).contains("layers"), "{}", err(&single(3)));
}

/// hardware: cpu. ADR 0077: a prompt sent as one step on a pipelined topology (M micro-batches
/// in one program) gives logits bit-equal to the same chunks sent as serial steps on the same
/// shape with M = 1, and greedy decode after it stays bit-equal (same KV rows). One step graph
/// is built for the pipelined prompt, M for the serial one. One stage ignores the knob.
#[test]
fn cpu_pipelined_prefill_is_bit_equal_to_serial_chunks() {
    let prompt = stream(11, 37);
    for (name, path, base) in [
        ("layer-2", dense_fixture(), layer_split(2, 1)),
        ("layer-2 at 3", dense_fixture_l4(), layer_split(4, 3)),
        ("tp2-pp2", dense_fixture_l4(), tp2pp2(4, 2)),
    ] {
        for m in [2u32, 4] {
            let mut serial = sharded(&path, &base, 1);
            let mut piped = sharded(
                &path,
                &Topology {
                    prefill_micro_batches: m,
                    ..base.clone()
                },
                1,
            );
            let mut ls = Vec::new();
            let mut at = 0;
            for n in micro_batch_sizes(prompt.len(), m) {
                ls = serial.eval_last(&prompt[at..at + n]).unwrap();
                at += n;
            }
            let mut lp = piped.eval_last(&prompt).unwrap();
            assert_eq!(bits(&ls), bits(&lp), "{name}, M = {m}: prompt");
            // Chunks of one size reuse a step graph; the serial run builds one per size.
            assert!(serial.step_graphs_built() >= 2, "{name}");
            assert_eq!(piped.step_graphs_built(), 1, "{name}");
            for step in 0..N_GREEDY {
                let t = argmax(&ls) as i32;
                ls = serial.eval_last(&[t]).unwrap();
                lp = piped.eval_last(&[t]).unwrap();
                assert_eq!(bits(&ls), bits(&lp), "{name}, M = {m}: step {step}");
            }
        }
    }
    // One stage: the knob changes nothing.
    let path = dense_fixture();
    let mut a = sharded(&path, &tp(2, 2), 1);
    let mut b = sharded(
        &path,
        &Topology {
            prefill_micro_batches: 4,
            ..tp(2, 2)
        },
        1,
    );
    assert_eq!(
        bits(&a.eval_last(&prompt).unwrap()),
        bits(&b.eval_last(&prompt).unwrap())
    );
    assert_eq!(a.exec_counters(), b.exec_counters());
}

/// hardware: cpu. Steps that do not pipeline on a pipelined topology build as with M = 1:
/// all-logits prefill, batched sequences and a one-token step.
#[test]
fn cpu_pipelined_topology_runs_other_steps_unpipelined() {
    let path = dense_fixture();
    let base = layer_split(2, 1);
    let piped = Topology {
        prefill_micro_batches: 4,
        ..base.clone()
    };
    let mut a = sharded(&path, &base, 2);
    let mut b = sharded(&path, &piped, 2);
    // Every position's logits.
    assert_eq!(
        bits(&a.eval(&PROMPT).unwrap()),
        bits(&b.eval(&PROMPT).unwrap())
    );
    // A one-token step, then two sequences in one step.
    let p1 = stream(4, 7);
    let run = |m: &mut Qwen3| {
        let mut out = Vec::new();
        m.eval_last(&[7]).unwrap();
        m.decode(
            &[
                SeqEval {
                    seq: 0,
                    tokens: &[5],
                    all_logits: false,
                },
                SeqEval {
                    seq: 1,
                    tokens: &p1,
                    all_logits: false,
                },
            ],
            &mut out,
        )
        .unwrap();
        bits(&out)
    };
    assert_eq!(run(&mut a), run(&mut b), "batched step");
    assert_eq!(a.step_graphs_built(), b.step_graphs_built());
}

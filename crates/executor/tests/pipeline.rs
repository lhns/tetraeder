//! Pipelined micro-batches (ADR 0077), label `cpu`: the GPipe issue order, a two-rank program of
//! several micro-batches in one graph per rank, and what validation refuses.

use tetraeder_executor::{
    Executor, Op, Program, RankDevice, RankGraph, Ranks, gpipe_sort, validate,
};
use tetraeder_ggml_sys::ggml::{Context, Tensor, Type};

const N: usize = 16;

/// Every (stage, micro-batch) group comes once and whole, after (s - 1, m) and (s, m - 1), with
/// its ops in their order; one micro-batch is stage order.
#[test]
fn gpipe_sort_is_a_wavefront() {
    for stages in 1..5usize {
        for micro in 1..6usize {
            // Two ops per group, built micro-batch major as the builders do.
            let mut ops = Vec::new();
            for m in 0..micro {
                for s in 0..stages {
                    ops.push(((s, m), (s, m, 0)));
                    ops.push(((s, m), (s, m, 1)));
                }
            }
            let sorted = gpipe_sort(ops);
            assert_eq!(sorted.len(), 2 * stages * micro);
            let at = |s: usize, m: usize, k: usize| {
                sorted
                    .iter()
                    .position(|&x| x == (s, m, k))
                    .unwrap_or_else(|| panic!("({s}, {m}, {k}) missing"))
            };
            for s in 0..stages {
                for m in 0..micro {
                    assert_eq!(at(s, m, 0) + 1, at(s, m, 1), "group ({s}, {m}) split");
                    if s > 0 {
                        assert!(at(s - 1, m, 1) < at(s, m, 0), "({s}, {m}) before its input");
                    }
                    if m > 0 {
                        assert!(at(s, m - 1, 1) < at(s, m, 0), "({s}, {m}) before {}", m - 1);
                    }
                    // A copy enqueues its wait on the destination stream when issued, so the hop
                    // into stage s for m (last op of (s - 1, m)) must come after stage s issued
                    // m - 1, or (s, m - 1) waits for (s - 1, m) to finish.
                    if s > 0 && m > 0 {
                        assert!(
                            at(s, m - 1, 1) < at(s - 1, m, 0),
                            "hop into ({s}, {m}) before ({s}, {})",
                            m - 1
                        );
                    }
                }
            }
            if micro == 1 {
                let order: Vec<usize> = sorted.iter().map(|x| x.0).collect();
                let want: Vec<usize> = (0..stages).flat_map(|s| [s, s]).collect();
                assert_eq!(order, want);
            }
            // The second stage starts micro-batch 0 before the first stage starts micro-batch 2.
            if stages > 1 && micro > 2 {
                assert!(at(1, 0, 0) < at(0, 2, 0));
            }
        }
    }
}

/// Rank 0 computes `a_m = 2 x_m + x_m` for each micro-batch m, rank 1 `o_m = 0.5 h_m + h_m` from
/// its hop input `h_m` (a graph input, allocated by the executor). One graph per rank, one range
/// per micro-batch.
struct Micro<'a> {
    graphs: Vec<RankGraph<'a>>,
    xs: Vec<Tensor<'a>>,
    srcs: Vec<Tensor<'a>>,
    hops: Vec<Tensor<'a>>,
    outs: Vec<Tensor<'a>>,
}

fn micro<'a>(
    data: &'a Context,
    g0: &'a Context,
    g1: &'a Context,
    m: usize,
    shared: bool,
) -> Micro<'a> {
    let n = N as i64;
    let xs: Vec<Tensor<'_>> = (0..m).map(|_| data.new_tensor(Type::F32, &[n])).collect();
    let mut graph0 = g0.new_graph(4 * m);
    let mut graph1 = g1.new_graph(4 * m);
    let (mut cuts0, mut cuts1) = (Vec::new(), Vec::new());
    let (mut srcs, mut hops, mut outs) = (Vec::new(), Vec::new(), Vec::new());
    let one_hop = g1.new_tensor(Type::F32, &[n]);
    for &x in &xs {
        let a = g0.add(g0.scale(x, 2.0), x);
        graph0.expand(a);
        cuts0.push(graph0.n_nodes());
        let h = if shared {
            one_hop
        } else {
            g1.new_tensor(Type::F32, &[n])
        };
        let o = g1.add(g1.scale(h, 0.5), h).set_output();
        graph1.expand(o);
        cuts1.push(graph1.n_nodes());
        srcs.push(a);
        hops.push(h);
        outs.push(o);
    }
    Micro {
        graphs: vec![
            RankGraph {
                rank: 0,
                graph: graph0,
                cuts: cuts0,
            },
            RankGraph {
                rank: 1,
                graph: graph1,
                cuts: cuts1,
            },
        ],
        xs,
        srcs,
        hops,
        outs,
    }
}

/// The ops of `p` tagged by (stage, micro-batch), in GPipe order.
fn ops<'a>(p: &Micro<'a>) -> Vec<Op<'a>> {
    let mut tagged = Vec::new();
    for m in 0..p.srcs.len() {
        tagged.push(((0, m), Op::Compute { rank: 0, range: m }));
        tagged.push((
            (0, m),
            Op::Copy {
                src: (0, p.srcs[m]),
                dst: (1, p.hops[m]),
            },
        ));
        tagged.push(((1, m), Op::Compute { rank: 1, range: m }));
    }
    gpipe_sort(tagged)
}

fn contexts(m: usize) -> [Context; 3] {
    [
        Context::new(m, 0).unwrap(),
        Context::new(4 * m + 4, 4 * m).unwrap(),
        Context::new(4 * m + 4, 4 * m).unwrap(),
    ]
}

/// Three micro-batches over two CPU ranks give each micro-batch's exact result, for two steps
/// with new inputs on the same kept graphs.
#[test]
fn pipelined_micro_batches_compute_each_micro_batch() {
    let m = 3;
    let [data, g0, g1] = contexts(m);
    let p = micro(&data, &g0, &g1, m, false);
    let mut exec = Executor::new(
        Ranks::open(&[
            RankDevice::Cpu { threads: 1 },
            RankDevice::Cpu { threads: 2 },
        ])
        .unwrap(),
    );
    data.alloc_tensors(exec.ranks().backend(0)).unwrap();
    let program = Program {
        ops: ops(&p),
        graphs: p.graphs,
        outputs: vec![1],
        prefill: false,
    };
    assert_eq!(validate(&program, 2), Ok(()));
    for step in 0..2 {
        for (k, x) in p.xs.iter().enumerate() {
            let v: Vec<f32> = (0..N).map(|i| (i + 10 * k + 100 * step) as f32).collect();
            x.set_f32(&v).unwrap();
        }
        exec.run(&program).unwrap();
        for (k, o) in p.outs.iter().enumerate() {
            let want: Vec<f32> = (0..N)
                .map(|i| 4.5 * (i + 10 * k + 100 * step) as f32)
                .collect();
            assert_eq!(o.get_f32().unwrap(), want, "step {step}, micro-batch {k}");
        }
    }
    let c = exec.take_counters();
    assert_eq!(c.ranges, vec![2 * m as u64, 2 * m as u64]);
    assert_eq!(
        c.collective_bytes.get(&("copy", "fallback")),
        Some(&(2 * m as u64 * N as u64 * 4))
    );
}

/// One hop tensor for two micro-batches is a race once copies run ahead of the next stage
/// (the second copy can land before the first micro-batch read it): refused.
#[test]
fn a_hop_shared_by_two_micro_batches_is_reported() {
    let [data, g0, g1] = contexts(2);
    let p = micro(&data, &g0, &g1, 2, true);
    let program = Program {
        ops: ops(&p),
        graphs: p.graphs,
        outputs: vec![1],
        prefill: false,
    };
    let errs = validate(&program, 2).unwrap_err().join("\n");
    assert!(errs.contains("destination of two copies"), "{errs}");
}

/// A micro-batch's copy issued before the range that computes its source is refused, in a
/// pipelined program as in a serial one.
#[test]
fn a_copy_ahead_of_its_micro_batch_is_reported() {
    let [data, g0, g1] = contexts(2);
    let p = micro(&data, &g0, &g1, 2, false);
    let mut o = ops(&p);
    // GPipe order: C(0,0) Copy0 C(1,0) C(0,1) Copy1 C(1,1). Move Copy1 before C(0,1).
    o.swap(3, 4);
    let program = Program {
        ops: o,
        graphs: p.graphs,
        outputs: vec![1],
        prefill: false,
    };
    let errs = validate(&program, 2).unwrap_err().join("\n");
    assert!(errs.contains("copy before range 1 of rank 0"), "{errs}");
}

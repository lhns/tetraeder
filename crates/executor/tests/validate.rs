//! Program validation (label `cpu`, no backend needed): property tests over random programs
//! (Rule 5) and one test per violation class of ADR 0069.

use proptest::prelude::*;
use tetraeder_executor::{Op, Program, RankGraph, RankId, validate};
use tetraeder_ggml_sys::ggml::{Context, Tensor, Type};

/// Per rank: a chain of `lens[r]` scale nodes over an input, cut where `masks[r]` says (then at
/// the end).
#[derive(Debug, Clone)]
struct Spec {
    lens: Vec<usize>,
    masks: Vec<Vec<bool>>,
    reduce: Vec<bool>,
}

fn spec() -> impl Strategy<Value = Spec> {
    (2usize..=4).prop_flat_map(|n| {
        (
            prop::collection::vec(2usize..=8, n),
            prop::collection::vec(prop::collection::vec(any::<bool>(), 7), n),
            prop::collection::vec(any::<bool>(), 8),
        )
            .prop_map(|(lens, masks, reduce)| Spec {
                lens,
                masks,
                reduce,
            })
    })
}

struct Built<'a> {
    graphs: Vec<RankGraph<'a>>,
    nodes: Vec<Vec<Tensor<'a>>>,
}

fn contexts(lens: &[usize]) -> Vec<Context> {
    lens.iter()
        .map(|&l| Context::new(l + 2, l).unwrap())
        .collect()
}

/// A chain graph per rank in `ctxs`.
fn build<'a>(ctxs: &'a [Context], lens: &[usize], masks: &[Vec<bool>]) -> Built<'a> {
    let mut graphs = Vec::new();
    let mut nodes = Vec::new();
    for (r, ctx) in ctxs.iter().enumerate() {
        let mut t = ctx.new_tensor(Type::F32, &[4]);
        let mut chain = Vec::new();
        for _ in 0..lens[r] {
            t = ctx.scale(t, 1.0);
            chain.push(t);
        }
        let mut graph = ctx.new_graph(lens[r]);
        graph.expand(t);
        assert_eq!(graph.n_nodes(), lens[r]);
        let mut cuts: Vec<usize> = (1..lens[r]).filter(|&j| masks[r][j - 1]).collect();
        cuts.push(lens[r]);
        graphs.push(RankGraph {
            rank: r,
            graph,
            cuts,
        });
        nodes.push(chain);
    }
    Built { graphs, nodes }
}

/// Stage `s` computes range `s` of every rank that has one; the ranks with a boundary there
/// all-reduce it when `reduce` says so.
fn program<'a>(b: Built<'a>, reduce: &[bool]) -> Program<'a> {
    let Built { graphs, nodes } = b;
    let stages = graphs.iter().map(|g| g.cuts.len()).max().unwrap();
    let mut ops = Vec::new();
    for s in 0..stages {
        for g in graphs.iter().filter(|g| s < g.cuts.len()) {
            ops.push(Op::Compute {
                rank: g.rank,
                range: s,
            });
        }
        let tensors: Vec<(RankId, Tensor<'a>)> = graphs
            .iter()
            .filter(|g| s + 1 < g.cuts.len())
            .map(|g| (g.rank, nodes[g.rank][g.cuts[s] - 1]))
            .collect();
        if reduce[s % reduce.len()] && tensors.len() >= 2 {
            ops.push(Op::AllReduce { tensors });
        }
    }
    let outputs = graphs.iter().map(|g| g.rank).collect();
    Program {
        graphs,
        ops,
        outputs,
        prefill: false,
    }
}

fn violations(p: &Program<'_>, n: usize) -> String {
    validate(p, n).err().unwrap_or_default().join("\n")
}

fn all_reduce_positions(p: &Program<'_>) -> Vec<usize> {
    (0..p.ops.len())
        .filter(|&i| matches!(p.ops[i], Op::AllReduce { .. }))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn valid_programs_pass(s in spec()) {
        let ctxs = contexts(&s.lens);
        let p = program(build(&ctxs, &s.lens, &s.masks), &s.reduce);
        prop_assert_eq!(validate(&p, s.lens.len()), Ok(()));
    }

    /// Every way to break a valid program is reported, with the reason named.
    #[test]
    fn broken_programs_are_reported(s in spec(), pick in 0usize..64) {
        let n = s.lens.len();
        let ctxs = contexts(&s.lens);
        let p = program(build(&ctxs, &s.lens, &s.masks), &s.reduce);
        let computes: Vec<usize> = (0..p.ops.len())
            .filter(|&i| matches!(p.ops[i], Op::Compute { .. }))
            .collect();

        // A range nobody computes.
        let mut q = Program { graphs: rebuild(&p), ops: p.ops.clone(), outputs: vec![], prefill: false };
        q.ops.remove(computes[pick % computes.len()]);
        prop_assert!(violations(&q, n).contains("never computed"), "{}", violations(&q, n));

        // A range computed twice.
        q.ops = p.ops.clone();
        let dup = p.ops[computes[pick % computes.len()]].clone();
        q.ops.insert(0, dup);
        prop_assert!(violations(&q, n).contains("computed twice"), "{}", violations(&q, n));

        // A cut list that does not end at the last node.
        q.ops = p.ops.clone();
        q.graphs[pick % n].cuts.pop();
        prop_assert!(violations(&q, n).contains("do not end"), "{}", violations(&q, n));

        // An all-reduce issued before the ranges that compute its tensors.
        let ars = all_reduce_positions(&p);
        if let Some(&i) = ars.get(pick % ars.len().max(1)) {
            let mut q = Program { graphs: rebuild(&p), ops: p.ops.clone(), outputs: vec![], prefill: false };
            let op = q.ops.remove(i);
            q.ops.insert(0, op);
            prop_assert!(violations(&q, n).contains("all-reduce before range"), "{}", violations(&q, n));

            // ... an all-reduce on a tensor inside a range ...
            let mut q = Program { graphs: rebuild(&p), ops: p.ops.clone(), outputs: vec![], prefill: false };
            if let Op::AllReduce { tensors } = &mut q.ops[i] {
                let (r, t) = tensors[0];
                let idx = (0..q.graphs[r].graph.n_nodes())
                    .find(|&k| q.graphs[r].graph.node(k).is(t))
                    .unwrap();
                if idx >= 1 && !q.graphs[r].cuts.contains(&idx) {
                    tensors[0].1 = q.graphs[r].graph.node(idx - 1);
                    prop_assert!(
                        violations(&q, n).contains("not the last node of a range"),
                        "{}",
                        violations(&q, n)
                    );
                }
            }

            // ... and one rank twice.
            let mut q = Program { graphs: rebuild(&p), ops: p.ops.clone(), outputs: vec![], prefill: false };
            if let Op::AllReduce { tensors } = &mut q.ops[i] {
                tensors[1] = tensors[0];
                prop_assert!(violations(&q, n).contains("names a rank twice"), "{}", violations(&q, n));
            }
        }

        // Two ranges of one rank swapped.
        for r in 0..n {
            let mine: Vec<usize> = computes
                .iter()
                .copied()
                .filter(|&i| matches!(p.ops[i], Op::Compute { rank, .. } if rank == r))
                .collect();
            if mine.len() >= 2 {
                let mut q = Program { graphs: rebuild(&p), ops: p.ops.clone(), outputs: vec![], prefill: false };
                q.ops.swap(mine[0], mine[1]);
                prop_assert!(violations(&q, n).contains("after a later range"), "{}", violations(&q, n));
                break;
            }
        }
    }
}

/// The same graphs again (a graph is not `Clone`): new graph objects over the same nodes.
fn rebuild<'a>(p: &Program<'a>) -> Vec<RankGraph<'a>> {
    // Leaked: the property tests build a few hundred tiny contexts and the OS reclaims them.
    p.graphs
        .iter()
        .map(|g| {
            let ctx: &'static Context =
                Box::leak(Box::new(Context::new(0, g.graph.n_nodes()).unwrap()));
            let mut graph = ctx.new_graph(g.graph.n_nodes());
            graph.expand(g.graph.node(g.graph.n_nodes() - 1));
            RankGraph {
                rank: g.rank,
                graph,
                cuts: g.cuts.clone(),
            }
        })
        .collect()
}

/// Rank 0 (nodes `a`, `b`, cut after each) hands `b` to rank 1, whose first node reads the
/// input `h`.
struct Pipe<'a> {
    graphs: Vec<RankGraph<'a>>,
    b: Tensor<'a>,
    h: Tensor<'a>,
    a: Tensor<'a>,
    c: Tensor<'a>,
}

fn pipe_contexts() -> [Context; 2] {
    [Context::new(4, 2).unwrap(), Context::new(4, 2).unwrap()]
}

fn pipe(ctxs: &[Context; 2]) -> Pipe<'_> {
    let x = ctxs[0].new_tensor(Type::F32, &[4]);
    let a = ctxs[0].scale(x, 1.0);
    let b = ctxs[0].scale(a, 1.0);
    let mut g0 = ctxs[0].new_graph(2);
    g0.expand(b);
    let h = ctxs[1].new_tensor(Type::F32, &[4]);
    let c = ctxs[1].scale(h, 1.0);
    let d = ctxs[1].scale(c, 1.0);
    let mut g1 = ctxs[1].new_graph(2);
    g1.expand(d);
    Pipe {
        graphs: vec![
            RankGraph {
                rank: 0,
                graph: g0,
                cuts: vec![1, 2],
            },
            RankGraph {
                rank: 1,
                graph: g1,
                cuts: vec![1, 2],
            },
        ],
        b,
        h,
        a,
        c,
    }
}

fn pipe_program<'a>(p: Pipe<'a>, order: &[usize], dst: Tensor<'a>, src: Tensor<'a>) -> Program<'a> {
    // 0: Compute(0,0)  1: Compute(0,1)  2: Copy  3: Compute(1,0)  4: Compute(1,1)
    let all = [
        Op::Compute { rank: 0, range: 0 },
        Op::Compute { rank: 0, range: 1 },
        Op::Copy {
            src: (0, src),
            dst: (1, dst),
        },
        Op::Compute { rank: 1, range: 0 },
        Op::Compute { rank: 1, range: 1 },
    ];
    Program {
        graphs: p.graphs,
        ops: order.iter().map(|&i| all[i].clone()).collect(),
        outputs: vec![1],
        prefill: false,
    }
}

#[test]
fn a_copy_hop_between_ranks_is_valid() {
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (b, h) = (p.b, p.h);
    let prog = pipe_program(p, &[0, 1, 2, 3, 4], h, b);
    assert_eq!(validate(&prog, 2), Ok(()));
}

#[test]
fn a_copy_before_its_source_or_after_its_reader_is_reported() {
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (b, h) = (p.b, p.h);
    let prog = pipe_program(p, &[0, 2, 1, 3, 4], h, b);
    assert!(violations(&prog, 2).contains("copy before range 1 of rank 0"));

    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (b, h) = (p.b, p.h);
    let prog = pipe_program(p, &[0, 1, 3, 2, 4], h, b);
    assert!(violations(&prog, 2).contains("range 0 of rank 1 reads the copy tensor"));
}

#[test]
fn a_copy_between_the_wrong_tensors_is_reported() {
    // The source is inside a range.
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let h = p.h;
    // Both nodes of rank 0 end a range here, so make the source a node of a longer range.
    let mut prog = pipe_program(p, &[0, 1, 2, 3, 4], h, h);
    prog.graphs[0].cuts = vec![2];
    prog.ops.remove(1);
    prog.ops[1] = Op::Copy {
        src: (0, prog.graphs[0].graph.node(0)),
        dst: (1, h),
    };
    assert!(violations(&prog, 2).contains("copy source on rank 0 is not the last node"));

    // The destination is computed by its rank's graph.
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (b, c) = (p.b, p.c);
    let prog = pipe_program(p, &[0, 1, 2, 3, 4], c, b);
    assert!(violations(&prog, 2).contains("is computed by its graph"));

    // Same rank, and different layouts.
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (a, b) = (p.a, p.b);
    let mut prog = pipe_program(p, &[0, 1, 2, 3, 4], a, b);
    prog.ops[2] = Op::Copy {
        src: (0, b),
        dst: (0, a),
    };
    assert!(violations(&prog, 2).contains("to itself"));
    let odd = ctxs[0].new_tensor(Type::F32, &[5]);
    prog.ops[2] = Op::Copy {
        src: (0, b),
        dst: (1, odd),
    };
    assert!(violations(&prog, 2).contains("different layouts"));
}

#[test]
fn ranks_graphs_and_all_reduce_shapes_are_checked() {
    let ctxs = pipe_contexts();
    let p = pipe(&ctxs);
    let (a, b, c) = (p.a, p.b, p.c);
    let mut prog = pipe_program(p, &[0, 1, 3, 4], b, b);
    // A graph for a rank that does not exist, a rank with two graphs.
    prog.graphs[1].rank = 5;
    assert!(violations(&prog, 2).contains("rank 5, but there are 2 ranks"));
    prog.graphs[1].rank = 0;
    assert!(violations(&prog, 2).contains("more than one graph"));
    prog.graphs[1].rank = 1;
    // An all-reduce of one tensor, of tensors of two shapes, and of a rank without a graph.
    prog.ops.push(Op::AllReduce {
        tensors: vec![(0, b)],
    });
    assert!(violations(&prog, 2).contains("all-reduce of 1 tensors"));
    let odd = ctxs[1].new_tensor(Type::F32, &[5]);
    prog.ops.pop();
    prog.ops.push(Op::AllReduce {
        tensors: vec![(0, b), (1, odd)],
    });
    assert!(violations(&prog, 2).contains("has shape"));
    prog.ops.pop();
    prog.ops.push(Op::AllReduce {
        tensors: vec![(0, a), (7, c)],
    });
    assert!(violations(&prog, 2).contains("rank 7, which has no usable graph"));
}

//! Executor on CPU ranks (label `cpu`): the fallback all-reduce, copies between ranks and the
//! reuse of a prepared program, all against the host or one graph on one backend.

mod common;

use common::{LEN, all_reduce_case, bits, cpu_ranks, pipeline_case, pipeline_case_with};
use tetraeder_executor::{
    ExecError, ExecOptions, Executor, Op, Program, RankDevice, RankGraph, Ranks,
};
use tetraeder_ggml_sys::ggml::{Backend, Context, Gallocr, Tensor, Type};

/// `((x + y) * y * 0.5)` normalized, plus `x`: five nodes.
fn chain<'a>(c: &'a Context, x: Tensor<'a>, y: Tensor<'a>) -> Tensor<'a> {
    let s = c.scale(c.mul(c.add(x, y), y), 0.5);
    c.add(c.rms_norm(s, 1e-6), x).set_output()
}

/// One rank and one range (and the same nodes cut in two or three) equal `Gallocr` plus
/// `compute` of the graph, bit for bit.
#[test]
fn one_rank_is_bit_equal_to_gallocr_and_compute() {
    let n = LEN as i64;
    let backend = Backend::cpu(1).unwrap();
    let lctx = Context::new(2, 0).unwrap();
    let (x, y) = (
        lctx.new_tensor(Type::F32, &[n]),
        lctx.new_tensor(Type::F32, &[n]),
    );
    lctx.alloc_tensors(&backend).unwrap();
    let xv: Vec<f32> = (0..LEN).map(|i| 0.37 * i as f32 - 3.0).collect();
    let yv: Vec<f32> = (0..LEN).map(|i| 1.5 - 0.11 * i as f32).collect();
    x.set_f32(&xv).unwrap();
    y.set_f32(&yv).unwrap();

    let gctx = Context::new(16, 8).unwrap();
    let e = chain(&gctx, x, y);
    let mut g = gctx.new_graph(8);
    g.expand(e);
    assert_eq!(g.n_nodes(), 5);

    let mut galloc = Gallocr::new(&backend).unwrap();
    galloc.alloc_graph(&g).unwrap();
    backend.compute_async(&g).unwrap();
    backend.synchronize();
    let want = bits(&e.get_f32().unwrap());

    let mut exec = Executor::new(Ranks::open(&[RankDevice::Cpu { threads: 1 }]).unwrap());
    for cuts in [vec![5], vec![2, 5], vec![1, 2, 5]] {
        // Fresh nodes without data, so the executor's own allocation is what runs.
        let rctx = Context::new(16, 8).unwrap();
        let e = chain(&rctx, x, y);
        let mut graph = rctx.new_graph(8);
        graph.expand(e);
        let ops = (0..cuts.len())
            .map(|range| Op::Compute { rank: 0, range })
            .collect();
        let p = Program {
            graphs: vec![RankGraph {
                rank: 0,
                graph,
                cuts: cuts.clone(),
            }],
            ops,
            outputs: vec![0],
            prefill: false,
        };
        exec.run(&p).unwrap();
        assert_eq!(bits(&e.get_f32().unwrap()), want, "cuts {cuts:?}");
        let c = exec.take_counters();
        assert_eq!(c.ranges, vec![cuts.len() as u64]);
        assert!(c.collective_bytes.is_empty());
    }
}

#[test]
fn cpu_ranks_report_the_fallback_path() {
    let ranks = Ranks::open(&cpu_ranks(2)).unwrap();
    assert_eq!((ranks.len(), ranks.comm_path()), (2, "fallback"));
    assert_eq!(ranks.gpu_ranks(), 0);
}

#[test]
fn all_reduce_over_two_cpu_ranks_is_the_host_sum() {
    let [first, _] = all_reduce_case(&cpu_ranks(2), &[vec![0, 1]]);
    let bytes = 2 * LEN as u64 * 4;
    assert_eq!(
        first.collective_bytes.get(&("all_reduce", "fallback")),
        Some(&bytes)
    );
}

#[test]
fn all_reduce_over_four_cpu_ranks_is_the_host_sum() {
    all_reduce_case(&cpu_ranks(4), &[vec![0, 1, 2, 3]]);
}

/// Groups that are not all the ranks in rank order: two pairs, a reversed pair with a rank of
/// its own, and a group that does not start at rank 0.
#[test]
fn all_reduce_over_groups_of_four_ranks() {
    all_reduce_case(&cpu_ranks(4), &[vec![0, 1], vec![2, 3]]);
    all_reduce_case(&cpu_ranks(4), &[vec![3, 1]]);
    all_reduce_case(&cpu_ranks(4), &[vec![1, 2, 3]]);
}

/// Several all-reduces in a row reuse one scratch set.
#[test]
fn consecutive_all_reduces_share_scratch() {
    let [first, second] = all_reduce_case(&cpu_ranks(3), &[vec![0, 1, 2], vec![2, 1, 0]]);
    // 3 graphs, and a scratch set per group.
    assert_eq!(first.allocations, 3 + 2);
    assert_eq!(second.allocations, 0);
}

#[test]
fn a_second_run_allocates_nothing() {
    let [first, second] = all_reduce_case(&cpu_ranks(2), &[vec![0, 1]]);
    assert!(first.allocations >= 3, "{first:?}");
    assert_eq!(second.allocations, 0);
    assert_eq!(second.ranges, vec![1, 1]);
}

/// A copy hop between two ranks, with each side cut in several ranges, equals one graph on one
/// backend bit for bit, for several steps.
#[test]
fn pipeline_equals_one_graph() {
    for (c0, c1) in [
        (vec![3], vec![3]),
        (vec![1, 3], vec![3]),
        (vec![3], vec![1, 2, 3]),
        (vec![1, 2, 3], vec![2, 3]),
    ] {
        let counters = pipeline_case(
            [
                RankDevice::Cpu { threads: 1 },
                RankDevice::Cpu { threads: 2 },
            ],
            &c0,
            &c1,
            3,
            |_| {},
        );
        // Counters of the three steps together.
        assert_eq!(
            counters.ranges,
            vec![3 * c0.len() as u64, 3 * c1.len() as u64]
        );
        assert_eq!(
            counters.collective_bytes.get(&("copy", "fallback")),
            Some(&(3 * LEN as u64 * 4))
        );
    }
}

/// Copies and all-reduces outside a program, and what they refuse.
#[test]
fn transfer_outside_a_program() {
    let mut exec = Executor::new(Ranks::open(&cpu_ranks(2)).unwrap());
    let ctxs = [Context::new(2, 0).unwrap(), Context::new(2, 0).unwrap()];
    let ts: Vec<_> = ctxs
        .iter()
        .map(|c| (c.new_tensor(Type::F32, &[8]), c.new_tensor(Type::I32, &[8])))
        .collect();
    for (r, c) in ctxs.iter().enumerate() {
        c.alloc_tensors(exec.ranks().backend(r)).unwrap();
    }
    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..8).map(|i| 10.0 * i as f32).collect();
    ts[0].0.set_f32(&a).unwrap();
    ts[1].0.set_f32(&b).unwrap();

    exec.transfer(&Op::Copy {
        src: (0, ts[0].0),
        dst: (1, ts[1].0),
    })
    .unwrap();
    assert_eq!(ts[1].0.get_f32().unwrap(), a);

    exec.transfer(&Op::AllReduce {
        tensors: vec![(0, ts[0].0), (1, ts[1].0)],
    })
    .unwrap();
    let want: Vec<f32> = a.iter().map(|v| 2.0 * v).collect();
    assert_eq!(ts[0].0.get_f32().unwrap(), want);
    assert_eq!(ts[1].0.get_f32().unwrap(), want);
    let c = exec.take_counters();
    assert_eq!(c.collective_bytes.get(&("copy", "fallback")), Some(&32));
    assert_eq!(
        c.collective_bytes.get(&("all_reduce", "fallback")),
        Some(&64)
    );

    // F32 only; one tensor is no group; a compute op needs a program.
    assert!(matches!(
        exec.transfer(&Op::AllReduce {
            tensors: vec![(0, ts[0].1), (1, ts[1].1)]
        }),
        Err(ExecError::Invalid(_))
    ));
    assert!(matches!(
        exec.transfer(&Op::AllReduce {
            tensors: vec![(0, ts[0].0)]
        }),
        Err(ExecError::Invalid(_))
    ));
    assert!(matches!(
        exec.transfer(&Op::Compute { rank: 0, range: 0 }),
        Err(ExecError::Invalid(_))
    ));
}

/// A program that breaks the rules is refused before anything runs (debug builds validate).
#[test]
fn run_refuses_an_invalid_program() {
    if !cfg!(debug_assertions) {
        return;
    }
    let mut exec = Executor::new(Ranks::open(&cpu_ranks(1)).unwrap());
    let ctx = Context::new(2, 2).unwrap();
    let x = ctx.new_tensor(Type::F32, &[4]);
    let mut g = ctx.new_graph(2);
    g.expand(ctx.scale(x, 2.0));
    let p = Program {
        graphs: vec![RankGraph {
            rank: 0,
            graph: g,
            cuts: vec![1],
        }],
        ops: vec![],
        outputs: vec![0],
        prefill: false,
    };
    match exec.run(&p) {
        Err(ExecError::Invalid(v)) => assert!(v[0].contains("never computed"), "{v:?}"),
        other => panic!("{other:?}"),
    }
}

/// The trace (ADR 0069 History 2026-11-19) writes one line per step with one record per op and
/// the output sync, in issue and in sync mode; results stay bit-equal to the reference
/// (`pipeline_case` checks every step), and range uids change nothing on CPU ranks.
#[test]
fn trace_records_every_op_per_step() {
    for (sync, uid) in [(false, false), (true, true)] {
        let path = std::env::temp_dir().join(format!(
            "tetraeder-exec-trace-{}-{sync}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let opts = ExecOptions {
            range_uid: uid,
            async_inputs: false,
            trace: Some(path.clone()),
            trace_sync: sync,
        };
        // Rank 0: two ranges, copy, rank 1: two ranges; then the sync of rank 1.
        pipeline_case_with(opts, [cpu_ranks(1)[0]; 2], &[1, 3], &[2, 3], 3, |_| {});
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        for (i, line) in lines.iter().enumerate() {
            assert!(
                line.starts_with(&format!("{{\"step\":{i},\"sync\":{sync},\"ev\":[")),
                "{line}"
            );
            let kinds: Vec<&str> = line
                .match_indices("[\"")
                .map(|(at, _)| {
                    let rest = &line[at + 2..];
                    &rest[..rest.find('"').unwrap()]
                })
                .collect();
            assert_eq!(
                kinds,
                ["compute", "compute", "copy", "compute", "compute", "sync"],
                "{line}"
            );
            assert!(line.contains("[\"copy\",0,1,"), "{line}");
        }
    }
}

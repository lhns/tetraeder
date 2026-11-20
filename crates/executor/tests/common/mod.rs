//! Program builders shared by the CPU and GPU tests. Each case builds its own contexts, runs the
//! executor and checks the numbers itself, so lifetimes stay inside one function.
#![allow(dead_code)]

use tetraeder_executor::{
    ExecCounters, ExecOptions, Executor, Op, Program, RankDevice, RankGraph, RankId, Ranks,
};
use tetraeder_ggml_sys::ggml::{Backend, Context, Gallocr, Tensor, Type};

pub const LEN: usize = 64;

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `n` small-integer inputs: sums of a few of them are exact in F32.
fn input(seed: usize, step: usize) -> Vec<f32> {
    (0..LEN)
        .map(|i| (((seed + 1) * (i % 7 + 1) + step) % 31) as f32 - 15.0)
        .collect()
}

/// One rank per device, each computing `y = x * 1` as a single range; the ranks of each group
/// all-reduce `y`; a rank in no group keeps `y = x`. Three steps with new inputs; the sums are
/// checked against the host. Returns the counters of the first and the second step.
pub fn all_reduce_case(devices: &[RankDevice], groups: &[Vec<RankId>]) -> [ExecCounters; 2] {
    let n = devices.len();
    let mut exec = Executor::new(Ranks::open(devices).unwrap());
    let lctx: Vec<Context> = (0..n).map(|_| Context::new(1, 0).unwrap()).collect();
    let xs: Vec<Tensor<'_>> = lctx
        .iter()
        .map(|c| c.new_tensor(Type::F32, &[LEN as i64]))
        .collect();
    for (r, c) in lctx.iter().enumerate() {
        c.alloc_tensors(exec.ranks().backend(r)).unwrap();
    }
    let gctx: Vec<Context> = (0..n).map(|_| Context::new(4, 4).unwrap()).collect();
    let ys: Vec<Tensor<'_>> = gctx
        .iter()
        .zip(&xs)
        .map(|(c, &x)| c.scale(x, 1.0).set_output())
        .collect();
    let graphs: Vec<RankGraph<'_>> = (0..n)
        .map(|r| {
            let mut graph = gctx[r].new_graph(4);
            graph.expand(ys[r]);
            RankGraph {
                rank: r,
                graph,
                cuts: vec![1],
            }
        })
        .collect();
    let mut ops: Vec<Op<'_>> = (0..n).map(|rank| Op::Compute { rank, range: 0 }).collect();
    for g in groups {
        ops.push(Op::AllReduce {
            tensors: g.iter().map(|&r| (r, ys[r])).collect(),
        });
    }
    let p = Program {
        graphs,
        ops,
        outputs: (0..n).collect(),
        prefill: false,
    };
    exec.prepare(&p).unwrap();
    let mut counters = Vec::new();
    for step in 0..3 {
        let inputs: Vec<Vec<f32>> = (0..n).map(|r| input(r, step)).collect();
        for (x, v) in xs.iter().zip(&inputs) {
            x.set_f32(v).unwrap();
        }
        exec.run(&p).unwrap();
        // The groups apply in order, so a rank in two of them sums sums.
        let mut want = inputs.clone();
        for g in groups {
            let sum: Vec<f32> = (0..LEN)
                .map(|i| g.iter().map(|&m| want[m][i]).sum())
                .collect();
            for &m in g {
                want[m].clone_from(&sum);
            }
        }
        for r in 0..n {
            assert_eq!(ys[r].get_f32().unwrap(), want[r], "step {step} rank {r}");
        }
        counters.push(exec.take_counters());
    }
    let c = counters.remove(0);
    let second = counters.remove(0);
    assert_eq!(c.ranges, vec![1; n]);
    assert_eq!(second.ranges, vec![1; n]);
    [c, second]
}

/// Rank 0 computes `a = (x + y) * y * 0.5`, copies it to rank 1's input `h`, and rank 1 computes
/// `rms_norm(h * 2) + z`. The same chain is one graph on one backend as the reference. Ranges are
/// cut at `cuts0` and `cuts1` (each ends at 3). `after_step` runs after every executor step.
/// The outputs must match the reference bit for bit at every one of `steps` steps.
pub fn pipeline_case(
    devices: [RankDevice; 2],
    cuts0: &[usize],
    cuts1: &[usize],
    steps: usize,
    after_step: impl FnMut(usize),
) -> ExecCounters {
    pipeline_case_with(
        ExecOptions::from_env(),
        devices,
        cuts0,
        cuts1,
        steps,
        after_step,
    )
}

/// [`pipeline_case`] on an executor with `opts`; the trace (if any) gets one line per step.
pub fn pipeline_case_with(
    opts: ExecOptions,
    devices: [RankDevice; 2],
    cuts0: &[usize],
    cuts1: &[usize],
    steps: usize,
    mut after_step: impl FnMut(usize),
) -> ExecCounters {
    let mut exec = Executor::with_options(Ranks::open(&devices).unwrap(), opts);
    let (b0, b1) = (
        exec.ranks().backend(0).clone(),
        exec.ranks().backend(1).clone(),
    );
    let n = LEN as i64;

    // Reference: everything on rank 0's device, one graph.
    let rctx = Context::new(8, 0).unwrap();
    let (rx, ry, rz) = (
        rctx.new_tensor(Type::F32, &[n]),
        rctx.new_tensor(Type::F32, &[n]),
        rctx.new_tensor(Type::F32, &[n]),
    );
    rctx.alloc_tensors(&b0).unwrap();
    let rg_ctx = Context::new(16, 8).unwrap();
    let a = rg_ctx.scale(rg_ctx.mul(rg_ctx.add(rx, ry), ry), 0.5);
    let rout = rg_ctx
        .add(rg_ctx.rms_norm(rg_ctx.scale(a, 2.0), 1e-6), rz)
        .set_output();
    let mut rgraph = rg_ctx.new_graph(8);
    rgraph.expand(rout);
    let mut galloc = Gallocr::new(&b0).unwrap();
    galloc.alloc_graph(&rgraph).unwrap();

    // Pipeline.
    let c0 = Context::new(2, 0).unwrap();
    let (x, y) = (
        c0.new_tensor(Type::F32, &[n]),
        c0.new_tensor(Type::F32, &[n]),
    );
    c0.alloc_tensors(&b0).unwrap();
    let c1 = Context::new(2, 0).unwrap();
    let (h, z) = (
        c1.new_tensor(Type::F32, &[n]),
        c1.new_tensor(Type::F32, &[n]),
    );
    c1.alloc_tensors(&b1).unwrap();
    let g0_ctx = Context::new(8, 3).unwrap();
    let a0 = g0_ctx.scale(g0_ctx.mul(g0_ctx.add(x, y), y), 0.5);
    let mut g0 = g0_ctx.new_graph(3);
    g0.expand(a0);
    let g1_ctx = Context::new(8, 3).unwrap();
    let out = g1_ctx
        .add(g1_ctx.rms_norm(g1_ctx.scale(h, 2.0), 1e-6), z)
        .set_output();
    let mut g1 = g1_ctx.new_graph(3);
    g1.expand(out);
    assert_eq!((g0.n_nodes(), g1.n_nodes()), (3, 3));

    let mut ops = Vec::new();
    for k in 0..cuts0.len() {
        ops.push(Op::Compute { rank: 0, range: k });
    }
    ops.push(Op::Copy {
        src: (0, a0),
        dst: (1, h),
    });
    for k in 0..cuts1.len() {
        ops.push(Op::Compute { rank: 1, range: k });
    }
    let p = Program {
        graphs: vec![
            RankGraph {
                rank: 0,
                graph: g0,
                cuts: cuts0.to_vec(),
            },
            RankGraph {
                rank: 1,
                graph: g1,
                cuts: cuts1.to_vec(),
            },
        ],
        ops,
        outputs: vec![1],
        prefill: false,
    };
    for step in 0..steps {
        let (xv, yv, zv) = (input(0, step), input(1, step), input(2, step));
        for (t, v) in [
            (rx, &xv),
            (ry, &yv),
            (rz, &zv),
            (x, &xv),
            (y, &yv),
            (z, &zv),
        ] {
            t.set_f32(v).unwrap();
        }
        b0.compute_async(&rgraph).unwrap();
        b0.synchronize();
        exec.run(&p).unwrap();
        assert_eq!(
            bits(&out.get_f32().unwrap()),
            bits(&rout.get_f32().unwrap()),
            "step {step} cuts {cuts0:?} {cuts1:?}"
        );
        exec.trace_flush();
        after_step(step);
    }
    exec.take_counters()
}

/// The comm path name of a CPU-only executor is "fallback".
pub fn cpu_ranks(n: usize) -> Vec<RankDevice> {
    vec![RankDevice::Cpu { threads: 1 }; n]
}

pub fn backend_of(d: RankDevice) -> Backend {
    match d {
        RankDevice::Gpu(i) => Backend::gpu(i).unwrap(),
        RankDevice::Cpu { threads } => Backend::cpu(threads).unwrap(),
    }
}

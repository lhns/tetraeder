//! Bindings the own executor needs (ADR 0069): range graphs, async compute, the graph allocator,
//! events and the comm API. Tests are labelled by hardware; GPU tests skip without a GPU
//! (`TETRAEDER_REQUIRE_GPUS=<n>` turns the skip into a failure on GPU boxes).

use tetraeder_ggml_sys::ggml::{
    Backend, Comm, Context, Event, Gallocr, GgmlError, Graph, RangeGraphs, Tensor, Type,
};
use tetraeder_ggml_sys::llama::cuda_graph_log_counts;

const N: usize = 8;

/// The chain's `x`, `y` and `out`, allocated in `ctx`'s own buffer on `backend`.
fn leaves<'a>(ctx: &'a Context, backend: &Backend) -> [Tensor<'a>; 3] {
    let t = [(); 3].map(|_| ctx.new_tensor(Type::F32, &[N as i64]));
    ctx.alloc_tensors(backend).unwrap();
    t
}

fn xs(seed: f32) -> Vec<f32> {
    (0..N).map(|i| seed + 0.37 * i as f32 - 1.0).collect()
}

/// The chain `add, mul, scale, rms_norm, add, cpy` in plain f32.
fn reference(x: &[f32], y: &[f32]) -> Vec<f32> {
    let c: Vec<f32> = x.iter().zip(y).map(|(x, y)| (x + y) * y * 0.5).collect();
    let ms = c.iter().map(|v| v * v).sum::<f32>() / N as f32;
    let s = 1.0 / (ms + 1e-6).sqrt();
    c.iter().zip(x).map(|(c, x)| c * s + x).collect()
}

/// Builds the 6-node chain over `x`, `y` and `out` in `gctx`; the last node is the output.
fn chain<'a>(gctx: &'a Context, x: Tensor<'a>, y: Tensor<'a>, out: Tensor<'a>) -> Graph<'a> {
    let a = gctx.add(x, y);
    let b = gctx.mul(a, y);
    let c = gctx.scale(b, 0.5);
    let d = gctx.rms_norm(c, 1e-6);
    let e = gctx.add(d, x);
    let f = gctx.cpy(e, out).set_output();
    let mut g = gctx.new_graph(8);
    g.expand(f);
    g
}

/// Three consecutive node ranges of the 6-node graph give the same output bits as the whole.
#[test]
fn range_graphs_match_the_whole_graph() {
    let cpu = Backend::cpu(2).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    let (xv, yv) = (xs(0.5), xs(2.0));
    x.set_f32(&xv).unwrap();
    y.set_f32(&yv).unwrap();

    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    assert_eq!(g.n_nodes(), 6);
    let mut galloc = Gallocr::new(&cpu).unwrap();
    galloc.alloc_graph(&g).unwrap();

    cpu.compute_async(&g).unwrap();
    cpu.synchronize();
    let whole = out.get_f32().unwrap();
    for (got, want) in whole.iter().zip(reference(&xv, &yv)) {
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{got} {want}"
        );
    }

    out.set_f32(&[0.0; N]).unwrap();
    let rctx = [(); 3].map(|_| Context::new(0, 2).unwrap());
    let parts: Vec<Graph<'_>> = rctx
        .iter()
        .enumerate()
        .map(|(i, c)| g.range(c, 2 * i, 2 * i + 2))
        .collect();
    for p in &parts {
        assert_eq!(p.n_nodes(), 2);
        cpu.compute_async(p).unwrap();
    }
    cpu.synchronize();
    let split = out.get_f32().unwrap();
    assert_eq!(
        whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        split.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // The allocation stays with the parent graph.
    assert!(galloc.is_allocated(&g));
    assert!(!galloc.is_allocated(&parts[0]));
}

/// One allocation serves repeated computes with new inputs.
#[test]
fn gallocr_allocates_once() {
    let cpu = Backend::cpu(1).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    let mut galloc = Gallocr::new(&cpu).unwrap();
    assert!(!galloc.is_allocated(&g));
    galloc.reserve(&g).unwrap();
    assert!(!galloc.is_allocated(&g), "reserve allocates no graph");
    galloc.alloc_graph(&g).unwrap();
    assert!(galloc.is_allocated(&g));
    assert!(galloc.buffer_size() > 0);
    let size = galloc.buffer_size();
    for seed in [0.5, 3.0, -2.0] {
        let (xv, yv) = (xs(seed), xs(seed + 1.5));
        x.set_f32(&xv).unwrap();
        y.set_f32(&yv).unwrap();
        cpu.compute_async(&g).unwrap();
        cpu.synchronize();
        for (got, want) in out.get_f32().unwrap().iter().zip(reference(&xv, &yv)) {
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{got} {want}"
            );
        }
        assert!(galloc.is_allocated(&g));
    }
    assert_eq!(galloc.buffer_size(), size);
}

/// A graph nobody allocated is refused, not computed on null data.
#[test]
fn compute_async_refuses_an_unallocated_graph() {
    let cpu = Backend::cpu(1).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    assert!(matches!(cpu.compute_async(&g), Err(GgmlError::Invalid(_))));
}

/// `copy_to_async` is a host-tensor to backend copy (`ggml_backend_tensor_set_async`); with two
/// CPU backends the second one's "stream" copies before returning.
#[test]
fn copy_to_async_between_cpu_backends() {
    let a = Backend::cpu(1).unwrap();
    let b = Backend::cpu(1).unwrap();
    let src_ctx = Context::new(1, 0).unwrap();
    let src = src_ctx.new_tensor(Type::I8, &[64]);
    src_ctx.alloc_tensors(&a).unwrap();
    let want: Vec<u8> = (0..64).map(|i| (i as u8).wrapping_mul(7)).collect();
    src.set_bytes(&want, 0).unwrap();
    let dst_ctx = Context::new(1, 0).unwrap();
    let dst = dst_ctx.new_tensor(Type::I8, &[64]);
    dst_ctx.alloc_tensors(&a).unwrap();
    dst.set_bytes(&[0; 64], 0).unwrap();
    src.copy_to_async(8, &b, dst, 16, 32).unwrap();
    b.synchronize();
    let got = dst.get_bytes(0, 64).unwrap();
    assert_eq!(got[16..48], want[8..40]);
    assert!(got[..16].iter().all(|&v| v == 0) && got[48..].iter().all(|&v| v == 0));
}

#[test]
fn cpu_has_no_comm_and_no_events() {
    let cpu = Backend::cpu(1).unwrap();
    assert!(Comm::new(std::slice::from_ref(&cpu)).is_none());
    assert!(Comm::new(&[cpu.clone(), Backend::cpu(1).unwrap()]).is_none());
    assert!(Comm::new(&[]).is_none());
    assert!(Event::new(&cpu).is_err());
    assert_eq!(cpu.gpu_index(), None);
}

/// hardware: gpu (skips without one). A comm over two backends of one GPU returns some path or
/// None without crashing; an event recorded on one backend orders the other.
#[test]
fn gpu_comm_and_event_on_one_gpu() {
    if !Backend::require_gpus(1) {
        return;
    }
    let a = Backend::gpu(0).unwrap();
    let b = a.sibling().unwrap();
    assert_eq!((a.gpu_index(), b.gpu_index()), (Some(0), Some(0)));
    match Comm::new(&[a.clone(), b.clone()]) {
        Some(c) => eprintln!("comm path on one GPU: {:?}", c.path()),
        None => eprintln!("no comm on one GPU"),
    }

    // Host data to the GPU on `a`'s stream, then `b` waits for it through an event.
    let host = Context::new(1, 0).unwrap();
    let src = host.new_tensor(Type::F32, &[N as i64]);
    host.alloc_host_staging(&a).unwrap();
    let dev = Context::new(1, 0).unwrap();
    let dst = dev.new_tensor(Type::F32, &[N as i64]);
    dev.alloc_tensors(&a).unwrap();
    let want = xs(1.0);
    src.set_f32(&want).unwrap();
    let ev = Event::new(&a).unwrap();
    src.copy_to_async(0, &a, dst, 0, N * 4).unwrap();
    ev.record(&a);
    ev.wait(&b);
    b.synchronize();
    ev.synchronize();
    assert_eq!(dst.get_f32().unwrap(), want);

    // The chain on the GPU through the allocator and async compute.
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &a);
    let (xv, yv) = (xs(0.5), xs(2.0));
    x.set_f32(&xv).unwrap();
    y.set_f32(&yv).unwrap();
    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    let mut galloc = Gallocr::new(&a).unwrap();
    galloc.alloc_graph(&g).unwrap();
    a.compute_async(&g).unwrap();
    ev.record(&a);
    ev.wait(&b);
    b.synchronize();
    for (got, want) in out.get_f32().unwrap().iter().zip(reference(&xv, &yv)) {
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "{got} {want}"
        );
    }
}

/// hardware: 2 gpus (skips with fewer). `[1..n]` and `[n..1]` sum to `n + 1` on both.
#[test]
fn gpu2_all_reduce_sums_across_gpus() {
    if !Backend::require_gpus(2) {
        return;
    }
    let backends = [Backend::gpu(0).unwrap(), Backend::gpu(1).unwrap()];
    assert_eq!(backends[1].gpu_index(), Some(1));
    let n = 4096usize;
    let ctxs: Vec<Context> = (0..2).map(|_| Context::new(1, 0).unwrap()).collect();
    let ts: Vec<Tensor<'_>> = ctxs
        .iter()
        .map(|c| c.new_tensor(Type::F32, &[n as i64]))
        .collect();
    for (c, b) in ctxs.iter().zip(&backends) {
        c.alloc_tensors(b).unwrap();
    }
    ts[0]
        .set_f32(&(1..=n).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    ts[1]
        .set_f32(&(1..=n).rev().map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    let comm = Comm::new(&backends).expect("a comm over two CUDA GPUs");
    eprintln!("all-reduce path: {:?}", comm.path());
    assert!(comm.all_reduce(&ts), "ggml declined the all-reduce");
    for b in &backends {
        b.synchronize();
    }
    let want = vec![(n + 1) as f32; n];
    assert_eq!(ts[0].get_f32().unwrap(), want);
    assert_eq!(ts[1].get_f32().unwrap(), want);
}

/// `copy_tensor_async` between two CPU backends copies before returning; a layout mismatch is an
/// error, not a ggml abort.
#[test]
fn copy_tensor_async_between_cpu_backends() {
    let (a, b) = (Backend::cpu(1).unwrap(), Backend::cpu(1).unwrap());
    let (sc, dc) = (Context::new(2, 0).unwrap(), Context::new(2, 0).unwrap());
    let src = sc.new_tensor(Type::F32, &[N as i64]);
    let other = sc.new_tensor(Type::F32, &[N as i64 + 1]);
    let dst = dc.new_tensor(Type::F32, &[N as i64]);
    sc.alloc_tensors(&a).unwrap();
    dc.alloc_tensors(&b).unwrap();
    let want = xs(3.0);
    src.set_f32(&want).unwrap();
    src.copy_tensor_async(&a, &b, dst).unwrap();
    b.synchronize();
    assert_eq!(dst.get_f32().unwrap(), want);
    assert!(matches!(
        other.copy_tensor_async(&a, &b, dst),
        Err(GgmlError::Invalid(_))
    ));
}

/// `RangeGraphs` cut at uneven points gives the whole graph's bits and refuses another graph,
/// bad cuts and a range out of bounds.
#[test]
fn range_graphs_cache_matches_the_whole_graph() {
    let cpu = Backend::cpu(1).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    x.set_f32(&xs(0.5)).unwrap();
    y.set_f32(&xs(2.0)).unwrap();
    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    let mut galloc = Gallocr::new(&cpu).unwrap();
    galloc.alloc_graph(&g).unwrap();
    cpu.compute_async(&g).unwrap();
    let whole = out.get_f32().unwrap();
    out.set_f32(&[0.0; N]).unwrap();

    assert!(RangeGraphs::new(&g, &[], false).is_err());
    assert!(RangeGraphs::new(&g, &[3, 3, 6], false).is_err());
    assert!(RangeGraphs::new(&g, &[2, 5], false).is_err());
    let ranges = RangeGraphs::new(&g, &[1, 4, 6], false).unwrap();
    assert_eq!(ranges.len(), 3);
    for i in 0..3 {
        cpu.compute_async(&ranges.get(&g, i).unwrap()).unwrap();
    }
    assert_eq!(ranges.get(&g, 1).unwrap().n_nodes(), 3);
    assert!(ranges.get(&g, 3).is_err());
    assert_eq!(
        out.get_f32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let gctx2 = Context::new(16, 8).unwrap();
    let g2 = chain(&gctx2, x, y, out);
    assert!(ranges.get(&g2, 0).is_err());
}

/// Ranges with a fresh uid (patch 0005; ggml-cuda then skips its node comparison) compute what
/// ranges without one compute, again and again; the CPU ignores the uid.
#[test]
fn ranges_with_a_uid_compute_as_without() {
    let cpu = Backend::cpu(1).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    x.set_f32(&xs(0.5)).unwrap();
    y.set_f32(&xs(2.0)).unwrap();
    let gctx = Context::new(16, 8).unwrap();
    let g = chain(&gctx, x, y, out);
    let mut galloc = Gallocr::new(&cpu).unwrap();
    galloc.alloc_graph(&g).unwrap();
    let mut got = Vec::new();
    for uid in [false, true] {
        let ranges = RangeGraphs::new(&g, &[1, 4, 6], uid).unwrap();
        for _ in 0..2 {
            out.set_f32(&[0.0; N]).unwrap();
            for i in 0..3 {
                cpu.compute_async(&ranges.get(&g, i).unwrap()).unwrap();
            }
            got.push(out.get_f32().unwrap());
        }
    }
    assert!(got.windows(2).all(|w| w[0] == w[1]), "{got:?}");
}

/// `set_i32_async` and `set_f16_bits_async` write what they are given on a CPU backend; a write
/// past the tensor's end is refused.
#[test]
fn async_setters_write_on_cpu() {
    let cpu = Backend::cpu(1).unwrap();
    let ctx = Context::new(2, 0).unwrap();
    let a = ctx.new_tensor(Type::I32, &[4]);
    let h = ctx.new_tensor(Type::F16, &[4]);
    ctx.alloc_tensors(&cpu).unwrap();
    a.set_i32_async(&cpu, &[3, -1, 7, 9]).unwrap();
    h.set_f16_bits_async(&cpu, &[0x3c00, 0xfc00, 0, 0x4000])
        .unwrap();
    cpu.synchronize();
    assert_eq!(a.get_i32().unwrap(), vec![3, -1, 7, 9]);
    // The four halves read back as two little-endian words.
    let words: Vec<u32> = h.get_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(words, vec![0xfc00_3c00, 0x4000_0000]);
    assert!(a.set_i32_async(&cpu, &[1, 2, 3, 4, 5]).is_err());
}

/// A range keeps the whole graph's use counts, so a backend fuses in it as in the whole graph
/// (ADR 0075). The CPU fuses `rms_norm` with the `mul` after it and then never writes the norm's
/// own output: a sentinel there survives the whole graph and a range holding both nodes. A range
/// filled with `ggml_graph_add_node` has no use counts, runs both ops and overwrites it; on CUDA
/// that ran other kernels than the scheduler (the one-rank mismatch on the S2 box).
#[test]
fn ranges_fuse_as_the_whole_graph() {
    let cpu = Backend::cpu(1).unwrap();
    let ctx = Context::new(8, 4).unwrap();
    let x = ctx.new_tensor(Type::F32, &[N as i64]);
    let w = ctx.new_tensor(Type::F32, &[N as i64]);
    let norm = ctx.rms_norm(x, 1e-6);
    let y = ctx.mul(norm, w);
    let z = ctx.scale(y, 2.0).set_output();
    let mut g = ctx.new_graph(4);
    g.expand(z);
    assert_eq!(g.n_nodes(), 3);
    // Every tensor in its own memory, so no result shares the norm's bytes.
    ctx.alloc_tensors(&cpu).unwrap();
    x.set_f32(&xs(0.5)).unwrap();
    w.set_f32(&xs(2.0)).unwrap();
    let sentinel = [7.0f32; N];
    norm.set_f32(&sentinel).unwrap();
    cpu.compute_async(&g).unwrap();
    cpu.synchronize();
    assert_eq!(
        norm.get_f32().unwrap(),
        sentinel,
        "the CPU fuses rms_norm and mul in the whole graph"
    );
    let whole = z.get_f32().unwrap();
    z.set_f32(&[0.0; N]).unwrap();

    let ranges = RangeGraphs::new(&g, &[2, 3], false).unwrap();
    for i in 0..2 {
        cpu.compute_async(&ranges.get(&g, i).unwrap()).unwrap();
    }
    cpu.synchronize();
    assert_eq!(
        norm.get_f32().unwrap(),
        sentinel,
        "a range fuses rms_norm and mul as the whole graph"
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&z.get_f32().unwrap()), bits(&whole));
}

/// `add_inplace` writes into its first operand, which is how the fallback all-reduce sums.
#[test]
fn add_inplace_writes_the_first_operand() {
    let cpu = Backend::cpu(1).unwrap();
    let ctx = Context::new(8, 4).unwrap();
    let a = ctx.new_tensor(Type::F32, &[N as i64]);
    let b = ctx.new_tensor(Type::F32, &[N as i64]);
    let sum = ctx.add_inplace(a, b);
    let mut g = ctx.new_graph(4);
    g.expand(sum);
    ctx.alloc_tensors(&cpu).unwrap();
    let (av, bv) = (xs(1.0), xs(5.0));
    a.set_f32(&av).unwrap();
    b.set_f32(&bv).unwrap();
    cpu.compute_async(&g).unwrap();
    let want: Vec<f32> = av.iter().zip(&bv).map(|(a, b)| a + b).collect();
    assert_eq!(a.get_f32().unwrap(), want);
    assert_eq!(b.get_f32().unwrap(), bv);
}

/// `set_no_capture` (patch 0006, ADR 0081) is callable on a graph, a view and a `RangeGraphs`,
/// and a flagged graph computes the same bits: the CPU backend ignores the flag.
#[test]
fn no_capture_flag_changes_nothing_on_cpu() {
    let cpu = Backend::cpu(1).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &cpu);
    x.set_f32(&xs(0.5)).unwrap();
    y.set_f32(&xs(2.0)).unwrap();
    let gctx = Context::new(16, 8).unwrap();
    let mut g = chain(&gctx, x, y, out);
    let mut galloc = Gallocr::new(&cpu).unwrap();
    galloc.alloc_graph(&g).unwrap();
    cpu.compute_async(&g).unwrap();
    let whole = out.get_f32().unwrap();

    g.set_no_capture(true);
    out.set_f32(&[0.0; N]).unwrap();
    cpu.compute_async(&g).unwrap();
    assert_eq!(out.get_f32().unwrap(), whole);
    g.set_no_capture(false);

    let vctx = Context::new(0, 1).unwrap();
    let mut view = g.range(&vctx, 0, 6);
    view.set_no_capture(true);
    out.set_f32(&[0.0; N]).unwrap();
    cpu.compute_async(&view).unwrap();
    assert_eq!(out.get_f32().unwrap(), whole);

    let mut ranges = RangeGraphs::new(&g, &[1, 4, 6], true).unwrap();
    for flag in [true, false, true] {
        ranges.set_no_capture(flag);
        out.set_f32(&[0.0; N]).unwrap();
        for i in 0..3 {
            cpu.compute_async(&ranges.get(&g, i).unwrap()).unwrap();
        }
        assert_eq!(out.get_f32().unwrap(), whole);
    }
}

/// hardware: gpu (skips without one). A graph flagged with `set_no_capture` is run directly on
/// every call, an unflagged copy of it is captured on its second call (ggml logs "CUDA graph
/// warmup complete" only with `GGML_CUDA_GRAPH_DEBUG=1`, read once by ggml, so the test needs it
/// in the environment). Other tests of this binary compute a graph once, which never warms up.
#[test]
fn gpu_no_capture_flag_skips_the_cuda_graph() {
    if !Backend::require_gpus(1) {
        return;
    }
    if std::env::var_os("GGML_CUDA_GRAPH_DEBUG").is_none() {
        let required = ["TETRAEDER_REQUIRE_GPU", "TETRAEDER_REQUIRE_GPUS"]
            .iter()
            .any(|k| std::env::var_os(k).is_some());
        assert!(!required, "set GGML_CUDA_GRAPH_DEBUG=1 for this test");
        eprintln!("skip: set GGML_CUDA_GRAPH_DEBUG=1");
        return;
    }
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some() {
        eprintln!("skip: CUDA graphs are disabled");
        return;
    }
    let gpu = Backend::gpu(0).unwrap();
    let lctx = Context::new(3, 0).unwrap();
    let [x, y, out] = leaves(&lctx, &gpu);
    let (xv, yv) = (xs(0.5), xs(2.0));
    x.set_f32(&xv).unwrap();
    y.set_f32(&yv).unwrap();

    // Four calls of the whole graph, or of a uid view of it in two ranges (a view and the whole
    // graph share a graph key, so they get a graph each); returns the captures counted.
    let run = |flag: bool, whole: bool| -> u64 {
        let gctx = Context::new(16, 8).unwrap();
        let mut g = chain(&gctx, x, y, out);
        let mut galloc = Gallocr::new(&gpu).unwrap();
        galloc.alloc_graph(&g).unwrap();
        let mut ranges = RangeGraphs::new(&g, &[3, 6], true).unwrap();
        g.set_no_capture(flag);
        ranges.set_no_capture(flag);
        let start = cuda_graph_log_counts().warmups;
        for _ in 0..4 {
            out.set_f32(&[0.0; N]).unwrap();
            if whole {
                gpu.compute_async(&g).unwrap();
            } else {
                for i in 0..2 {
                    gpu.compute_async(&ranges.get(&g, i).unwrap()).unwrap();
                }
            }
            gpu.synchronize();
            for (got, want) in out.get_f32().unwrap().iter().zip(reference(&xv, &yv)) {
                assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0));
            }
        }
        cuda_graph_log_counts().warmups - start
    };
    for whole in [true, false] {
        assert_eq!(run(true, whole), 0, "a flagged graph was captured");
    }
    for whole in [true, false] {
        assert!(
            run(false, whole) >= 1,
            "an unflagged graph was not captured"
        );
    }
}

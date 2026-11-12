//! Tests of the safe ggml surface on the CPU backend.

use std::rc::Rc;
use tetraeder_ggml_sys::ggml::{Backend, Context, GgmlError, OwnedGraph, Sched, Type};

fn cpu() -> (Backend, Sched) {
    let cpu = Backend::cpu(2).unwrap();
    let sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    (cpu, sched)
}

#[test]
fn mul_mat_matches_by_hand() {
    let (cpu, mut sched) = cpu();
    // Weights live in a persistent context with its own buffer.
    let wctx = Context::new(1, 0).unwrap();
    let w = wctx.new_tensor(Type::F32, &[3, 2]); // two rows of three
    wctx.alloc_tensors(&cpu).unwrap();
    w.set_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let gctx = Context::new(8, 64).unwrap();
    let x = gctx.new_tensor(Type::F32, &[3, 1]);
    let y = gctx.mul_mat(w, x).set_output();
    let mut g = gctx.new_graph(64);
    g.expand(y);
    sched.alloc_graph(&g).unwrap();
    x.set_f32(&[1.0, 0.5, 2.0]).unwrap();
    sched.compute(&g).unwrap();
    assert_eq!(y.get_f32().unwrap(), vec![8.0, 18.5]);
}

#[test]
fn top_k_ids_and_get_rows() {
    let (_cpu, mut sched) = cpu();
    let ctx = Context::new(8, 64).unwrap();
    let scores = ctx.new_tensor(Type::F32, &[4, 1]);
    let ids = ctx.argsort_top_k(scores, 2).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(ids);
    sched.alloc_graph(&g).unwrap();
    scores.set_f32(&[0.1, 0.9, 0.3, 0.5]).unwrap();
    sched.compute(&g).unwrap();
    assert_eq!(ids.get_i32().unwrap(), vec![1, 3]);
}

#[test]
fn compute_refuses_unallocated_graph() {
    let (_cpu, mut sched) = cpu();
    let ctx = Context::new(4, 16).unwrap();
    let a = ctx.new_tensor(Type::F32, &[2]);
    let b = ctx.add(a, a);
    let mut g = ctx.new_graph(16);
    g.expand(b);
    assert!(matches!(sched.compute(&g), Err(GgmlError::Invalid(_))));
    assert!(matches!(a.set_f32(&[1.0, 2.0]), Err(GgmlError::Invalid(_))));
}

#[test]
fn sched_needs_cpu_last() {
    if !Backend::gpu_available() {
        return;
    }
    let gpu = Backend::gpu(0).unwrap();
    assert!(Sched::new(&[gpu], 16, false).is_err());
}

#[test]
fn keep_refuses_a_foreign_tensor() {
    let a = Rc::new(Context::new(4, 0).unwrap());
    let b = Rc::new(Context::new(4, 0).unwrap());
    let t = a.new_tensor(Type::F32, &[2]);
    let u = b.new_tensor(Type::F32, &[2]);
    assert!(a.keep(t).is_ok());
    assert!(matches!(a.keep(u), Err(GgmlError::Invalid(_))));
}

/// `y = w * x` with `w` in a persistent context and `x` in another one.
#[test]
fn keep_graph_needs_the_contexts_it_reads() {
    let (cpu, _sched) = cpu();
    let wctx = Rc::new(Context::new(1, 0).unwrap());
    let w = wctx.new_persistent(Type::F32, &[2, 1]);
    let xctx = Rc::new(Context::new(1, 0).unwrap());
    let x = xctx.new_persistent(Type::F32, &[2, 1]);
    wctx.alloc_tensors(&cpu).unwrap();
    xctx.alloc_tensors(&cpu).unwrap();
    // Room for four small graphs.
    let gctx = Rc::new(Context::new(8, 64).unwrap());
    let y = gctx.mul(w.get(), x.get());
    let graph = || {
        let mut g = gctx.new_graph(16);
        g.expand(y);
        g
    };
    let refused = |r: Result<OwnedGraph, GgmlError>| matches!(r, Err(GgmlError::Invalid(_)));
    assert!(refused(gctx.keep_graph(graph(), &[])));
    assert!(refused(gctx.keep_graph(graph(), &[&w])));
    assert!(gctx.keep_graph(graph(), &[&w, &x, &w]).is_ok());
    // A graph of another context is refused.
    assert!(refused(wctx.keep_graph(graph(), &[&w, &x])));
}

/// A kept graph computes again with new inputs and no second `alloc_graph`.
#[test]
fn kept_graph_computes_twice() {
    let (cpu, mut sched) = cpu();
    let wctx = Rc::new(Context::new(1, 0).unwrap());
    let w = wctx.new_persistent(Type::F32, &[3, 2]);
    let xctx = Rc::new(Context::new(1, 0).unwrap());
    let x = xctx.new_persistent(Type::F32, &[3, 1]);
    wctx.alloc_tensors(&cpu).unwrap();
    xctx.alloc_tensors(&cpu).unwrap();
    w.get().set_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let gctx = Rc::new(Context::new(8, 64).unwrap());
    let y = gctx.mul_mat(w.get(), x.get()).set_output();
    let mut g = gctx.new_graph(64);
    g.expand(y);
    let y = gctx.keep(y).unwrap();
    let kept = gctx.keep_graph(g, &[&w, &x]).unwrap();
    assert!(!sched.is_allocated(&kept.get()));
    sched.alloc_graph(&kept.get()).unwrap();
    assert!(sched.is_allocated(&kept.get()));
    drop(gctx); // the kept graph holds its context
    for (input, want) in [
        ([1.0, 0.5, 2.0], [8.0, 18.5]),
        ([0.0, 1.0, 1.0], [5.0, 11.0]),
    ] {
        x.get().set_f32(&input).unwrap();
        sched.compute(&kept.get()).unwrap();
        assert_eq!(y.get().get_f32().unwrap(), want);
        assert!(sched.is_allocated(&kept.get()));
    }
}

/// hardware: cpu. Linux only: a tensor buffer of 32 MiB through the weights path, untouched, then
/// `MADV_HUGEPAGE` on it must return 0 for the aligned range inside. Elsewhere the call is a no-op.
#[test]
fn hugepage_advice_on_a_cpu_buffer() {
    let cpu = Backend::cpu(1).unwrap();
    let ctx = Context::new(1, 0).unwrap();
    let _t = ctx.new_tensor(Type::F32, &[8 << 20]);
    let bytes = ctx.alloc_weights(&cpu).unwrap();
    assert!(bytes >= 32 << 20);
    let a = ctx.advise_hugepages();
    if cfg!(target_os = "linux") {
        assert!(a.applied && a.errno == 0, "madvise failed: {a:?}");
        // 32 MiB minus up to 2 MiB lost to alignment at each end.
        assert!(a.bytes >= 28 << 20 && a.bytes <= bytes as u64, "{a:?}");
        assert_eq!(a.bytes % (2 << 20), 0);
    } else {
        eprintln!("skipped: MADV_HUGEPAGE is Linux only");
        assert!(!a.applied && a.bytes == 0);
    }
    // no buffer: nothing to advise
    assert!(!Context::new(1, 0).unwrap().advise_hugepages().applied);
}

/// A byte pattern for the copy tests.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// Host-to-backend copies straight from a host tensor (ADR 0059 addendum 2026-11-12b) into
/// `dst_backend`'s memory, the async one on `stream`.
fn host_copies(dst_backend: &Backend, stream: &Backend) {
    let host = Context::new(1, 0).unwrap();
    let src = host.new_tensor(Type::I8, &[256]);
    let (bytes, pinned) = host.alloc_host_staging(dst_backend).unwrap();
    assert!(bytes >= 256);
    if dst_backend.is_cpu() {
        assert!(!pinned, "the CPU device has no host buffer type of its own");
    }
    let want = pattern(256, 7);
    src.set_bytes(&want, 0).unwrap();
    let dev = Context::new(1, 0).unwrap();
    let dst = dev.new_tensor(Type::I8, &[256]);
    dev.alloc_tensors(dst_backend).unwrap();
    dst.set_bytes(&[0; 256], 0).unwrap();

    src.copy_to(16, dst, 100, 50).unwrap();
    src.copy_to_async(128, stream, dst, 0, 64).unwrap();
    stream.synchronize();
    let got = dst.get_bytes(0, 256).unwrap();
    assert_eq!(got[100..150], want[16..66]);
    assert_eq!(got[0..64], want[128..192]);
    assert!(got[64..100].iter().all(|&b| b == 0));

    // Out of range on either side is an error, never a ggml abort.
    assert!(matches!(
        src.copy_to(250, dst, 0, 10),
        Err(GgmlError::Invalid(_))
    ));
    assert!(matches!(
        src.copy_to_async(0, stream, dst, 250, 10),
        Err(GgmlError::Invalid(_))
    ));
}

/// hardware: cpu. `copy_to` and `copy_to_async` on the CPU backend, where the async copy is
/// synchronous; a CPU backend's sibling is itself.
#[test]
fn cpu_host_copies() {
    let cpu = Backend::cpu(1).unwrap();
    host_copies(&cpu, &cpu.sibling().unwrap());
}

/// hardware: gpu (skips without one). The async copy runs on a sibling instance's stream into a
/// GPU buffer from pinned staging; a GPU tensor is no host source, and a CPU buffer is no
/// destination for the GPU stream.
#[test]
fn gpu_host_copies_on_a_sibling_stream() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device");
        return;
    }
    let gpu = Backend::gpu(0).unwrap();
    let stream = gpu.sibling().unwrap();
    host_copies(&gpu, &stream);
    let dev = Context::new(2, 0).unwrap();
    let a = dev.new_tensor(Type::I8, &[16]);
    let b = dev.new_tensor(Type::I8, &[16]);
    dev.alloc_tensors(&gpu).unwrap();
    assert!(matches!(a.copy_to(0, b, 0, 16), Err(GgmlError::Invalid(_))));
    let host = Context::new(1, 0).unwrap();
    let h = host.new_tensor(Type::I8, &[16]);
    let (_, pinned) = host.alloc_host_staging(&gpu).unwrap();
    eprintln!("pinned staging: {pinned}");
    let cpu = Backend::cpu(1).unwrap();
    let plain = Context::new(1, 0).unwrap();
    let c = plain.new_tensor(Type::I8, &[16]);
    plain.alloc_tensors(&cpu).unwrap();
    assert!(matches!(
        h.copy_to_async(0, &stream, c, 0, 16),
        Err(GgmlError::Invalid(_))
    ));
}

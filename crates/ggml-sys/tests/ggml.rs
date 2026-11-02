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

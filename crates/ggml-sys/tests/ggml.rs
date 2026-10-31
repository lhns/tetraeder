//! Tests of the safe ggml surface on the CPU backend.

use tetraeder_ggml_sys::ggml::{Backend, Context, GgmlError, Sched, Type};

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

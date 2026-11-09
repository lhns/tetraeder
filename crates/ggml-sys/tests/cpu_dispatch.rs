//! Runtime CPU ISA dispatch (ADR 0052). Hardware: cpu, any x86; other CPUs skip.

use tetraeder_ggml_sys::cpu::{self, Dispatch};
use tetraeder_ggml_sys::ggml::{Backend, Context, Sched, Type};

fn x86() -> bool {
    if cfg!(target_arch = "x86_64") {
        return true;
    }
    eprintln!("skipped: x86-64 only (ggml dispatches CPU variants on x86)");
    false
}

/// The loaded CPU backend is the best one for this host: `missing` non-empty means the best
/// module was not loaded, `extra` non-empty means a variant this CPU cannot run was picked
/// (the MSVC VBMI scoring bug, patch 0001).
#[test]
fn cpu_backend_reports_variant() {
    if !x86() {
        return;
    }
    tetraeder_ggml_sys::llama::init();
    let info = cpu::backend_info().expect("CPU backend must load");
    eprintln!("{info:#?}");
    let dl = cfg!(ggml_dl);
    assert_eq!(info.dispatch, if dl { Dispatch::Dl } else { Dispatch::Static });
    assert_eq!(info.module_dir.is_some(), dl, "module dir only in a DL build");
    if info.forced.is_none() {
        assert!(info.missing.is_empty(), "best module not loaded: {info:?}");
    }
    assert!(info.extra.is_empty(), "wrong variant loaded: {info:?}");
}

#[test]
fn cpu_backend_computes() {
    let cpu = Backend::cpu(2).unwrap();
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    let ctx = Context::new(8, 64).unwrap();
    let a = ctx.new_tensor(Type::F32, &[4]);
    let b = ctx.new_tensor(Type::F32, &[4]);
    let c = ctx.add(a, b).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(c);
    sched.alloc_graph(&g).unwrap();
    a.set_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    b.set_f32(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    sched.compute(&g).unwrap();
    assert_eq!(c.get_f32().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);
}

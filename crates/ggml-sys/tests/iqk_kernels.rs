//! iqk AVX-512 VNNI decode kernels for Q4_K and Q6_K experts (patch 0002, ADR 0053).
//! Hardware label: `avx512_vnni`. Every test skips with a printed reason when the loaded CPU
//! backend is below that tier; otherwise it needs the `IQK_MM` feature, unless
//! `GGML_CPU_IQK_MM=0` is set (CI runs the file once that way: the reference checks must still
//! pass on ggml's own kernels, and `iqk_off_by_env` checks the feature is gone).
//!
//! Reference: weights dequantized by ggml, the activation through ggml's `Q8_K` quantizer and
//! back, a dot product in f64. Bound per output: 1e-5 * sum|w*x| + 1e-6.

use tetraeder_ggml_sys as sys;
use tetraeder_ggml_sys::cpu;
use tetraeder_ggml_sys::ggml::{Backend, Context, Sched, Tensor, Type, quantize, to_f32};

const EXPERTS: i64 = 16;
const USED: usize = 4;
const NE01: i64 = 96;

/// `Some(iqk expected)` when the tests apply here, `None` to skip.
fn gate() -> Option<bool> {
    let info = cpu::backend_info().expect("CPU backend must load");
    if !matches!(info.tier, "avx512_vnni" | "avx512_bf16" | "amx_int8") {
        eprintln!(
            "skipped: needs the avx512_vnni tier, loaded CPU backend is {} (variant forced: {:?})",
            info.tier, info.forced
        );
        return None;
    }
    let off = cpu::iqk_off_by_env();
    let has = info.isa.iter().any(|f| f == "IQK_MM");
    assert_eq!(
        has, !off,
        "IQK_MM feature must be present unless GGML_CPU_IQK_MM=0: {:?}",
        info.isa
    );
    Some(!off)
}

/// Deterministic floats in [-1, 1).
fn noise(n: usize, mut s: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// ggml's `Q8_K` round trip of one row, as the kernels see the activation.
fn q8k_roundtrip(x: &[f32]) -> Vec<f32> {
    let q8k = Type::from_name("q8_K").unwrap();
    let mut blocks = vec![0u8; q8k.row_size(x.len() as i64)];
    // SAFETY: `from_float_ref` of Q8_K reads `x.len()` floats and writes whole Q8_K blocks into
    // `blocks`, which holds exactly `x.len() / 256` of them.
    unsafe {
        let f = (*sys::ggml_get_type_traits(q8k.id() as _))
            .from_float_ref
            .expect("q8_K quantizer");
        f(x.as_ptr(), blocks.as_mut_ptr().cast(), x.len() as i64);
    }
    let mut out = vec![0f32; x.len()];
    to_f32(q8k, &blocks, &mut out).unwrap();
    out
}

struct Weights {
    ty: Type,
    ne00: usize,
    /// Dequantized, `[expert][row][k]`.
    dq: Vec<f32>,
    /// Quantized bytes, `[expert][row]`.
    bytes: Vec<u8>,
}

fn make_weights(ty: Type, ne00: usize, seed: u64) -> Weights {
    let rows = EXPERTS as usize * NE01 as usize;
    let src = noise(rows * ne00, seed);
    let mut bytes = vec![0u8; rows * ty.row_size(ne00 as i64)];
    quantize(ty, &src, rows, ne00, None, &mut bytes).unwrap();
    let mut dq = vec![0f32; rows * ne00];
    to_f32(ty, &bytes, &mut dq).unwrap();
    Weights {
        ty,
        ne00,
        dq,
        bytes,
    }
}

fn used_experts() -> Vec<i32> {
    vec![3, 15, 0, 9]
}

fn weight_tensor<'a>(wctx: &'a Context, cpu: &Backend, w: &Weights) -> Tensor<'a> {
    let t = wctx.new_tensor(w.ty, &[w.ne00 as i64, NE01, EXPERTS]);
    wctx.alloc_weights(cpu).unwrap();
    t.set_bytes(&w.bytes, 0).unwrap();
    t
}

/// Runs the graph on `threads` threads; returns the `[slot][row]` outputs. `use_ref` selects
/// ggml's own vec_dot path. `act` is `[rows][ne00]`, one row (gate/up) or one per slot (down).
fn run(w: &Weights, act: &[f32], threads: i32, use_ref: bool) -> Vec<f32> {
    let cpu = Backend::cpu(threads).unwrap();
    let wctx = Context::new(2, 0).unwrap();
    let wt = weight_tensor(&wctx, &cpu, w);
    let rows = act.len() / w.ne00;
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    let ctx = Context::new(8, 64).unwrap();
    let b = ctx.new_tensor(Type::F32, &[w.ne00 as i64, rows as i64, 1]);
    let ids = ctx.new_tensor(Type::I32, &[USED as i64, 1]);
    let y = ctx.mul_mat_id(wt, b, ids).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    sched.alloc_graph(&g).unwrap();
    b.set_f32(act).unwrap();
    ids.set_i32(&used_experts()).unwrap();
    cpu.set_cpu_use_ref(use_ref);
    sched.compute(&g).unwrap();
    y.get_f32().unwrap()
}

/// f64 reference, `[slot][row]`, and the per-output bound.
fn reference(w: &Weights, act: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let rows = act.len() / w.ne00;
    let xq: Vec<f32> = act.chunks(w.ne00).flat_map(q8k_roundtrip).collect();
    let mut y = Vec::new();
    let mut bound = Vec::new();
    for (slot, e) in used_experts().into_iter().enumerate() {
        let x = &xq[(slot % rows) * w.ne00..][..w.ne00];
        for r in 0..NE01 as usize {
            let wr = &w.dq[(e as usize * NE01 as usize + r) * w.ne00..][..w.ne00];
            let (mut s, mut a) = (0f64, 0f64);
            for (wv, xv) in wr.iter().zip(x) {
                let p = f64::from(*wv) * f64::from(*xv);
                s += p;
                a += p.abs();
            }
            y.push(s);
            bound.push(1e-5 * a + 1e-6);
        }
    }
    (y, bound)
}

fn max_err(y: &[f32], r: &[f64], bound: &[f64]) -> (f64, f64) {
    assert_eq!(y.len(), r.len());
    let mut worst = 0f64;
    let mut worst_ratio = 0f64;
    for ((v, r), b) in y.iter().zip(r).zip(bound) {
        let e = (f64::from(*v) - r).abs();
        assert!(
            e <= *b,
            "error {e} above bound {b} (value {v}, reference {r})"
        );
        worst = worst.max(e);
        worst_ratio = worst_ratio.max(e / b);
    }
    (worst, worst_ratio)
}

fn check_type(ty_name: &str) {
    let Some(iqk) = gate() else { return };
    let ty = Type::from_name(ty_name).unwrap();
    for ne00 in [256usize, 1536, 4096] {
        let w = make_weights(ty, ne00, 11 + ne00 as u64);
        // Gate/up: one activation row for all slots. Down: one row per slot.
        for rows in [1usize, USED] {
            let act = noise(rows * ne00, 99 + rows as u64);
            let (r, bound) = reference(&w, &act);
            let threads = 4;
            let y = run(&w, &act, threads, false);
            let y_ref = run(&w, &act, threads, true);
            let (e1, q1) = max_err(&y, &r, &bound);
            let (e2, q2) = max_err(&y_ref, &r, &bound);
            let diff = y
                .iter()
                .zip(&y_ref)
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0f64, f64::max);
            eprintln!(
                "{ty_name} ne00={ne00} rows={rows} iqk={iqk}: max err {e1:.3e} ({q1:.2} of bound), \
                 vec_dot max err {e2:.3e} ({q2:.2} of bound), max |iqk - vec_dot| {diff:.3e}"
            );
        }
    }
}

/// hardware: avx512_vnni.
#[test]
fn q4k_mul_mat_id_decode_matches_reference() {
    check_type("q4_K");
}

/// hardware: avx512_vnni.
#[test]
fn q6k_mul_mat_id_decode_matches_reference() {
    check_type("q6_K");
}

/// hardware: avx512_vnni. One row is computed by one thread whatever the split.
#[test]
fn iqk_is_deterministic_across_threads() {
    if gate().is_none() {
        return;
    }
    for name in ["q4_K", "q6_K"] {
        let w = make_weights(Type::from_name(name).unwrap(), 1536, 5);
        let act = noise(USED * 1536, 6);
        let base = run(&w, &act, 1, false);
        for threads in [3, 8] {
            let y = run(&w, &act, threads, false);
            let same = base.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{name}: {threads} threads differ from 1 thread");
        }
    }
}

/// hardware: avx512_vnni. CI runs this with `GGML_CPU_IQK_MM=0`.
#[test]
fn iqk_off_by_env() {
    if !cpu::iqk_off_by_env() {
        eprintln!("skipped: set GGML_CPU_IQK_MM=0 to run");
        return;
    }
    let info = cpu::backend_info().expect("CPU backend must load");
    assert!(
        !info.isa.iter().any(|f| f == "IQK_MM"),
        "IQK_MM still reported: {:?}",
        info.isa
    );
    assert!(
        !info.missing.iter().any(|f| f == "IQK_MM"),
        "switched off on purpose, not missing"
    );
}

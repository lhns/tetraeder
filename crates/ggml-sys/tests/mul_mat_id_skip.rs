//! `mul_mat_id` with expert id -1 (patch 0004, ADR 0070): the slot reads no weights and its output
//! row is exactly zero, as in the CPU kernel. Expert parallel (ADR 0070) relies on it for the
//! experts that live on other ranks.
//!
//! Hardware labels: `cpu_` tests run everywhere. `gpu_` tests need a CUDA device and return early
//! without one. On a box (session S1):
//! `cargo test --release -p tetraeder-ggml-sys --features cuda --test mul_mat_id_skip gpu_ -- --nocapture --test-threads=1`
//! The timing test prints a line `MUL_MAT_ID_SKIP {json}` and writes the same JSON to the file
//! named by `TETRAEDER_MMID_JSON` when set.

use std::time::Instant;
use tetraeder_ggml_sys::ggml::{Backend, Context, Sched, Type, quantize};

/// Relative to the largest reference magnitude; as `TOL_GPU` of the model tests.
const TOL_GPU: f32 = 5e-3;
/// CPU against a per-expert `mul_mat`: same kernels and quantized activations, only summation
/// order inside a tile may differ.
const TOL_CPU: f32 = 1e-4;

struct Shape {
    ty: Type,
    k: i64,
    n: i64,
    experts: i64,
    used: i64,
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn quant_type(name: &str) -> Type {
    Type::from_name(name).unwrap_or_else(|| panic!("unknown type {name}"))
}

/// Quantized bytes of every expert, `[n rows of k]` each.
fn expert_bytes(s: &Shape, distinct: bool, rng: &mut Rng) -> Vec<Vec<u8>> {
    let (rows, per_row) = (s.n as usize, s.k as usize);
    let mut make = || {
        let src: Vec<f32> = (0..rows * per_row).map(|_| rng.f32()).collect();
        let mut out = vec![0u8; rows * s.ty.row_size(s.k)];
        quantize(s.ty, &src, rows, per_row, None, &mut out).unwrap();
        out
    };
    if distinct {
        (0..s.experts).map(|_| make()).collect()
    } else {
        let one = make();
        (0..s.experts).map(|_| one.clone()).collect()
    }
}

/// Ids `[used, tokens]` (token-major): distinct experts per token, `neg_pct` percent of the slots
/// replaced by -1.
fn make_ids(s: &Shape, tokens: usize, neg_pct: usize, rng: &mut Rng) -> Vec<i32> {
    let mut ids = Vec::with_capacity(tokens * s.used as usize);
    for _ in 0..tokens {
        let mut pool: Vec<i32> = (0..s.experts as i32).collect();
        for j in 0..s.used as usize {
            let pick = j + rng.below(pool.len() - j);
            pool.swap(j, pick);
            ids.push(if rng.below(100) < neg_pct {
                -1
            } else {
                pool[j]
            });
        }
    }
    ids
}

/// `mul_mat_id` on `backends` (weights in the first backend's memory). The graph runs once with
/// `warm` ids when given (so the output buffer holds stale nonzero rows), then with `ids`.
/// `b` is `[k, b1, tokens]`; the result is `[n, used, tokens]`.
fn run_mul_mat_id(
    backends: &[Backend],
    s: &Shape,
    experts: &[Vec<u8>],
    b1: i64,
    b: &[f32],
    ids: &[i32],
    warm: Option<&[i32]>,
) -> Vec<f32> {
    let tokens = (ids.len() as i64) / s.used;
    let wctx = Context::new(1, 0).unwrap();
    let w = wctx.new_tensor(s.ty, &[s.k, s.n, s.experts]);
    wctx.alloc_tensors(&backends[0]).unwrap();
    let stride = experts[0].len();
    for (e, bytes) in experts.iter().enumerate() {
        w.set_bytes(bytes, e * stride).unwrap();
    }
    let ctx = Context::new(8, 64).unwrap();
    let bt = ctx.new_tensor(Type::F32, &[s.k, b1, tokens]);
    let it = ctx.new_tensor(Type::I32, &[s.used, tokens]);
    let y = ctx.mul_mat_id(w, bt, it).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    let mut sched = Sched::new(backends, 64, false).unwrap();
    sched.alloc_graph(&g).unwrap();
    bt.set_f32(b).unwrap();
    if let Some(warm) = warm {
        it.set_i32(warm).unwrap();
        sched.compute(&g).unwrap();
    }
    it.set_i32(ids).unwrap();
    sched.compute(&g).unwrap();
    y.get_f32().unwrap()
}

/// The expected output built from one plain `mul_mat` per expert on the CPU; zero rows for -1.
fn reference(
    cpu: &Backend,
    s: &Shape,
    experts: &[Vec<u8>],
    b1: i64,
    b: &[f32],
    ids: &[i32],
) -> Vec<f32> {
    let (k, n, used) = (s.k as usize, s.n as usize, s.used as usize);
    let mut out = vec![0.0f32; ids.len() * n];
    for (e, bytes) in experts.iter().enumerate() {
        let slots: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == e as i32).collect();
        if slots.is_empty() {
            continue;
        }
        let wctx = Context::new(1, 0).unwrap();
        let w = wctx.new_tensor(s.ty, &[s.k, s.n]);
        wctx.alloc_tensors(cpu).unwrap();
        w.set_bytes(bytes, 0).unwrap();
        let mut x = Vec::with_capacity(slots.len() * k);
        for &i in &slots {
            let (t, j) = (i / used, i % used);
            let col = t * b1 as usize + if b1 == 1 { 0 } else { j };
            x.extend_from_slice(&b[col * k..(col + 1) * k]);
        }
        let ctx = Context::new(8, 64).unwrap();
        let xt = ctx.new_tensor(Type::F32, &[s.k, slots.len() as i64]);
        let y = ctx.mul_mat(w, xt).set_output();
        let mut g = ctx.new_graph(64);
        g.expand(y);
        let mut sched = Sched::new(std::slice::from_ref(cpu), 64, false).unwrap();
        sched.alloc_graph(&g).unwrap();
        xt.set_f32(&x).unwrap();
        sched.compute(&g).unwrap();
        let rows = y.get_f32().unwrap();
        for (c, &i) in slots.iter().enumerate() {
            out[i * n..(i + 1) * n].copy_from_slice(&rows[c * n..(c + 1) * n]);
        }
    }
    out
}

/// Rows with id -1 are exactly zero; the rest match `want` within `tol` of its largest value.
fn check(got: &[f32], want: &[f32], ids: &[i32], n: usize, tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: output size");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, &id) in ids.iter().enumerate() {
        let (g, w) = (&got[i * n..(i + 1) * n], &want[i * n..(i + 1) * n]);
        if id < 0 {
            assert!(
                g.iter().all(|&v| v == 0.0),
                "{what}: slot {i} has id -1 but a nonzero row"
            );
        } else {
            let d = g
                .iter()
                .zip(w)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                d <= tol * scale,
                "{what}: slot {i}: diff {d} > {}",
                tol * scale
            );
        }
    }
}

const TYPES: [&str; 4] = ["q4_K", "q6_K", "q8_0", "f16"];
const TOKENS: [usize; 4] = [1, 8, 64, 512];

/// Runs every type and token count with ids that are 30% -1; `under_test` is the backend list
/// under test and `tol` its tolerance against the reference.
fn sweep(under_test: &[Backend], cpu: &Backend, tol: f32, against_cpu_mul_mat_id: bool) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for name in TYPES {
        let s = Shape {
            ty: quant_type(name),
            k: 512,
            n: 64,
            experts: 16,
            used: 4,
        };
        let experts = expert_bytes(&s, true, &mut rng);
        for tokens in TOKENS {
            for b1 in [1, s.used] {
                let b: Vec<f32> = (0..s.k as usize * b1 as usize * tokens)
                    .map(|_| rng.f32())
                    .collect();
                let ids = make_ids(&s, tokens, 30, &mut rng);
                let warm = make_ids(&s, tokens, 0, &mut rng);
                assert!(ids.contains(&-1) || tokens == 1, "ids without -1");
                let what = format!("{name} tokens={tokens} b1={b1}");
                let got = run_mul_mat_id(under_test, &s, &experts, b1, &b, &ids, Some(&warm));
                let want = if against_cpu_mul_mat_id {
                    run_mul_mat_id(std::slice::from_ref(cpu), &s, &experts, b1, &b, &ids, None)
                } else {
                    reference(cpu, &s, &experts, b1, &b, &ids)
                };
                check(&got, &want, &ids, s.n as usize, tol, &what);
            }
        }
    }
}

/// hardware: cpu.
#[test]
fn cpu_mul_mat_id_negative_ids_give_zero_rows() {
    let cpu = Backend::cpu(2).unwrap();
    sweep(std::slice::from_ref(&cpu), &cpu, TOL_CPU, false);
}

/// hardware: gpu (CUDA). Same cases on the GPU against the CPU `mul_mat_id`.
#[test]
fn gpu_mul_mat_id_negative_ids_match_cpu() {
    if !Backend::gpu_available() {
        return;
    }
    let cpu = Backend::cpu(2).unwrap();
    let gpu = Backend::gpu(0).unwrap();
    sweep(&[gpu, cpu.clone()], &cpu, TOL_GPU, true);
}

/// hardware: gpu (CUDA). Skipped slots cost nothing: the time falls with the share of -1 ids.
#[test]
fn gpu_mul_mat_id_time_falls_with_skipped_ids() {
    if !Backend::gpu_available() {
        return;
    }
    let cpu = Backend::cpu(2).unwrap();
    let gpu = Backend::gpu(0).unwrap();
    let backends = [gpu.clone(), cpu];
    // Qwen3-30B-A3B gate/up shape; one quantized expert copied to all slots.
    let s = Shape {
        ty: quant_type("q4_K"),
        k: 2048,
        n: 768,
        experts: 128,
        used: 8,
    };
    let tokens = 512usize;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let experts = expert_bytes(&s, false, &mut rng);

    let wctx = Context::new(1, 0).unwrap();
    let w = wctx.new_tensor(s.ty, &[s.k, s.n, s.experts]);
    wctx.alloc_tensors(&gpu).unwrap();
    for (e, bytes) in experts.iter().enumerate() {
        w.set_bytes(bytes, e * bytes.len()).unwrap();
    }
    let ctx = Context::new(8, 64).unwrap();
    let bt = ctx.new_tensor(Type::F32, &[s.k, 1, tokens as i64]);
    let it = ctx.new_tensor(Type::I32, &[s.used, tokens as i64]);
    let y = ctx.mul_mat_id(w, bt, it).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    let mut sched = Sched::new(&backends, 64, false).unwrap();
    sched.alloc_graph(&g).unwrap();
    let b: Vec<f32> = (0..s.k as usize * tokens).map(|_| rng.f32()).collect();
    bt.set_f32(&b).unwrap();

    let mut medians = Vec::new();
    for pct in [0usize, 50, 90] {
        let ids = make_ids(&s, tokens, pct, &mut rng);
        it.set_i32(&ids).unwrap();
        for _ in 0..3 {
            sched.compute(&g).unwrap();
        }
        let mut ms: Vec<f64> = (0..20)
            .map(|_| {
                let t = Instant::now();
                sched.compute(&g).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        medians.push((pct, ms[ms.len() / 2]));
    }
    let json = format!(
        "{{\"test\":\"mul_mat_id_skip\",\"gpu\":\"{}\",\"tokens\":{tokens},\"type\":\"q4_K\",\"median_ms\":{{{}}}}}",
        gpu.name(),
        medians
            .iter()
            .map(|(p, m)| format!("\"neg_{p}\":{m:.4}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("MUL_MAT_ID_SKIP {json}");
    if let Ok(path) = std::env::var("TETRAEDER_MMID_JSON") {
        std::fs::write(path, &json).unwrap();
    }
    assert!(
        medians[1].1 < medians[0].1 && medians[2].1 < medians[1].1,
        "time must fall with the share of -1 ids: {medians:?}"
    );
}

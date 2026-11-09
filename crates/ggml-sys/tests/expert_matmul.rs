//! Decode expert matmul on the CPU: ggml `mul_mat_id` with Q4_K experts (and the Q4_K_M mix, Q6_K down) at Qwen3-235B-A22B shapes
//! (n_embd 4096, expert ffn 1536, 8 of 128 experts, one token; gate, up and down), timed per call.
//! Diagnostic for the KTransformers decode gap (ADR 0045, "Diagnosing the KTransformers decode gap"):
//! build it with `TETRAEDER_GGML_NATIVE=OFF` and `=ON` and compare, to see what AVX-512 kernels
//! are worth on the box CPU.
//!
//! Hardware label: `cpu`. Ignored by default (needs about 1.5 GB, release only):
//! `cargo test --release -p tetraeder-ggml-sys --test expert_matmul -- --ignored --nocapture`
//!
//! Env: `TETRAEDER_BENCH_THREADS` (available parallelism), `TETRAEDER_BENCH_CALLS` (64),
//! `TETRAEDER_GGML_NATIVE` (only echoed into the result line). The last stdout line starting with
//! `EXPERT_MATMUL ` is the result as JSON, with the ggml CPU features this build was compiled with.

use std::time::Instant;
use tetraeder_ggml_sys::ggml::{Backend, Context, Sched, Tensor, Type, quantize};
use tetraeder_ggml_sys::llama::system_info;

const EMBD: i64 = 4096;
const FF: i64 = 1536;
const EXPERTS: i64 = 128;
const USED: i64 = 8;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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

/// Stacked `[k, n, EXPERTS]` Q4_K weights. One quantized expert is copied to all slots: the speed
/// of the kernels does not depend on the values, and distinct memory per expert keeps the reads
/// cold like the real model.
fn fill_experts(w: Tensor<'_>, k: i64, n: i64, seed: u64) {
    let ty = w.ty();
    let rows = n as usize;
    let per_row = k as usize;
    let src = noise(rows * per_row, seed);
    let mut one = vec![0u8; rows * ty.row_size(k)];
    quantize(ty, &src, rows, per_row, None, &mut one).unwrap();
    for e in 0..EXPERTS as usize {
        w.set_bytes(&one, e * one.len()).unwrap();
    }
}

struct Stats {
    median_ms: f64,
    min_ms: f64,
    mean_ms: f64,
    gb_s: f64,
}

/// Times `calls` computes of `mul_mat_id(w, b, ids)`, rotating the 8 used experts over all 128.
fn time_op(
    sched: &mut Sched,
    w: Tensor<'_>,
    b_shape: [i64; 3],
    k: i64,
    n: i64,
    calls: usize,
) -> Stats {
    let ctx = Context::new(8, 64).unwrap();
    let b = ctx.new_tensor(Type::F32, &b_shape);
    let ids = ctx.new_tensor(Type::I32, &[USED, 1]);
    let y = ctx.mul_mat_id(w, b, ids).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    sched.reset();
    sched.alloc_graph(&g).unwrap();
    b.set_f32(&noise((b_shape[0] * b_shape[1]) as usize, 7))
        .unwrap();
    let set = |i: usize| -> Vec<i32> {
        (0..USED as usize)
            .map(|j| ((i * USED as usize + j) % EXPERTS as usize) as i32)
            .collect()
    };
    for i in 0..3 {
        ids.set_i32(&set(i)).unwrap();
        sched.compute(&g).unwrap();
    }
    let mut ms = Vec::with_capacity(calls);
    for i in 0..calls {
        ids.set_i32(&set(i + 3)).unwrap();
        let t = Instant::now();
        sched.compute(&g).unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let bytes = (USED as usize * n as usize * w.ty().row_size(k)) as f64;
    Stats {
        median_ms: median,
        min_ms: sorted[0],
        mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
        gb_s: bytes / (median * 1e-3) / 1e9,
    }
}

fn cpu_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|m| m.trim().to_string())
}

/// hardware: cpu. All experts Q4_K.
#[test]
#[ignore = "diagnostic benchmark: about 1.5 GB, release only"]
fn cpu_decode_expert_matmul_q4k() {
    decode_expert_matmul("q4_K", "q4_K");
}

/// hardware: cpu. Q4_K_M mix: gate and up Q4_K, down Q6_K (ADR 0053).
#[test]
#[ignore = "diagnostic benchmark: about 1.7 GB, release only"]
fn cpu_decode_expert_matmul_q4km() {
    decode_expert_matmul("q4_K", "q6_K");
}

fn decode_expert_matmul(gate_up: &str, down_ty: &str) {
    let threads = env_usize(
        "TETRAEDER_BENCH_THREADS",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    );
    let calls = env_usize("TETRAEDER_BENCH_CALLS", 64);
    let cpu = Backend::cpu(threads as i32).unwrap();
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    let q4k = Type::from_name(gate_up).unwrap();
    let qdown = Type::from_name(down_ty).unwrap();

    let wctx = Context::new(4, 0).unwrap();
    let gate = wctx.new_tensor(q4k, &[EMBD, FF, EXPERTS]);
    let up = wctx.new_tensor(q4k, &[EMBD, FF, EXPERTS]);
    let down = wctx.new_tensor(qdown, &[FF, EMBD, EXPERTS]);
    wctx.alloc_weights(&cpu).unwrap();
    fill_experts(gate, EMBD, FF, 1);
    fill_experts(up, EMBD, FF, 2);
    fill_experts(down, FF, EMBD, 3);

    let g = time_op(&mut sched, gate, [EMBD, 1, 1], EMBD, FF, calls);
    let u = time_op(&mut sched, up, [EMBD, 1, 1], EMBD, FF, calls);
    let d = time_op(&mut sched, down, [FF, USED, 1], FF, EMBD, calls);
    let layer_ms = g.median_ms + u.median_ms + d.median_ms;

    let sysinfo = system_info();
    // Features of the loaded CPU backend (the variant ggml picked), not of the host.
    let info = tetraeder_ggml_sys::cpu::backend_info().unwrap();
    let has = |f: &str| i32::from(info.isa.iter().any(|x| x == f));
    let feat = serde_like(&[
        ("avx2", has("AVX2")),
        ("fma", has("FMA")),
        ("avx512", has("AVX512")),
        ("avx512_vnni", has("AVX512_VNNI")),
        ("avx512_bf16", has("AVX512_BF16")),
        ("amx_int8", has("AMX_INT8")),
        ("llamafile", has("LLAMAFILE")),
        ("iqk_mm", has("IQK_MM")),
    ]);
    let tiled_force = i32::from(std::env::var("GGML_CPU_TILED_MM_FORCE").is_ok_and(|v| v == "1"));
    let op = |s: &Stats| {
        format!(
            "{{\"median_ms\":{:.4},\"min_ms\":{:.4},\"mean_ms\":{:.4},\"gb_s\":{:.2}}}",
            s.median_ms, s.min_ms, s.mean_ms, s.gb_s
        )
    };
    println!(
        "EXPERT_MATMUL {{\"mix\":\"{gate_up}+{down_ty}\",\"tier\":\"{}\",\"iqk_mm\":{},\
         \"tiled_force\":{tiled_force},\"native\":\"{}\",\"threads\":{threads},\"calls\":{calls},\
         \"cpu\":\"{}\",\"ggml_features\":{feat},\"gate\":{},\"up\":{},\"down\":{},\"layer_ms\":{layer_ms:.4},\
         \"tokens_per_s_94_layers\":{:.3}}}",
        info.tier,
        has("IQK_MM"),
        std::env::var("TETRAEDER_GGML_NATIVE").unwrap_or_else(|_| "unset".into()),
        cpu_model().unwrap_or_default(),
        op(&g),
        op(&u),
        op(&d),
        1e3 / (layer_ms * 94.0),
    );
    println!("system_info: {sysinfo}");
    assert!(layer_ms.is_finite() && layer_ms > 0.0);
}

fn serde_like(items: &[(&str, i32)]) -> String {
    let body: Vec<String> = items.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    format!("{{{}}}", body.join(","))
}

//! Temporary diagnostic for the pin move (track-pin-adopt): single CPU ops on 7 rows against
//! the same ops on one row, bitwise. Prints only; runs with TETRAEDER_BATCH_DIAG=1.

use tetraeder_ggml_sys::ggml::{Backend, Context, Sched, Type};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `w` [k, m] in `wty` (cast from F32 in the graph), x [k, n]; returns [m, n].
fn mul_mat(wty: Type, k: usize, m: usize, x: &[f32], n: usize) -> Vec<f32> {
    let cpu = Backend::cpu(2).unwrap();
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    let ctx = Context::new(8, 64).unwrap();
    let w = ctx.new_tensor(Type::F32, &[k as i64, m as i64]).set_input();
    let xt = ctx.new_tensor(Type::F32, &[k as i64, n as i64]).set_input();
    let wc = if wty == Type::F32 {
        w
    } else {
        ctx.cast(w, wty)
    };
    let y = ctx.mul_mat(wc, xt).set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    sched.alloc_graph(&g).unwrap();
    w.set_f32(&rand(k * m, 7)).unwrap();
    xt.set_f32(x).unwrap();
    sched.compute(&g).unwrap();
    y.get_f32().unwrap()
}

/// q [d, n, h] rows `first..first+n` of a causal block over `n_kv` F16 entries.
fn fa(d: usize, h: usize, hkv: usize, n_kv: usize, q: &[f32], first: usize, n: usize) -> Vec<f32> {
    let cpu = Backend::cpu(2).unwrap();
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 64, false).unwrap();
    let ctx = Context::new(16, 64).unwrap();
    let qt = ctx
        .new_tensor(Type::F32, &[d as i64, n as i64, h as i64])
        .set_input();
    let k = ctx
        .new_tensor(Type::F32, &[d as i64, n_kv as i64, hkv as i64])
        .set_input();
    let v = ctx
        .new_tensor(Type::F32, &[d as i64, n_kv as i64, hkv as i64])
        .set_input();
    let npad = n.div_ceil(32) * 32;
    let mask = ctx
        .new_tensor(Type::F32, &[n_kv as i64, npad as i64])
        .set_input();
    let y = ctx
        .flash_attn_ext(
            qt,
            ctx.cast(k, Type::F16),
            ctx.cast(v, Type::F16),
            ctx.cast(mask, Type::F16),
            0.125,
        )
        .set_output();
    let mut g = ctx.new_graph(64);
    g.expand(y);
    sched.alloc_graph(&g).unwrap();
    // q is [d, n, h]: pick rows first..first+n of every head from the full [d, 7, h] q
    let mut qs = Vec::with_capacity(d * n * h);
    for hh in 0..h {
        for r in first..first + n {
            let off = (hh * 7 + r) * d;
            qs.extend_from_slice(&q[off..off + d]);
        }
    }
    qt.set_f32(&qs).unwrap();
    k.set_f32(&rand(d * n_kv * hkv, 11)).unwrap();
    v.set_f32(&rand(d * n_kv * hkv, 13)).unwrap();
    let mut m = vec![f32::NEG_INFINITY; n_kv * npad];
    for r in 0..n {
        for c in 0..=(24 + first + r).min(n_kv - 1) {
            m[r * n_kv + c] = 0.0;
        }
    }
    mask.set_f32(&m).unwrap();
    sched.compute(&g).unwrap();
    y.get_f32().unwrap()
}

fn report(what: &str, batched: &[f32], one: impl Fn(usize) -> Vec<f32>, row: usize) {
    for i in 0..7 {
        let o = one(i);
        let b = &batched[i * row..(i + 1) * row];
        let bits = o
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let diff = o.iter().zip(b).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let scale = o.iter().fold(0f32, |a, x| a.max(x.abs()));
        println!("diag op {what} row {i}: {bits} of {row} differ, max {diff:.3e} of {scale:.3e}");
    }
}

#[test]
fn batch_diag_ops() {
    if std::env::var("TETRAEDER_BATCH_DIAG").is_err() {
        return;
    }
    let (k, n) = (256, 7);
    let x = rand(k * n, 3);
    for (wty, m) in [
        (Type::F32, 32),
        (Type::F32, 256),
        (Type::F16, 256),
        (Type::from_id(8).unwrap(), 256),
    ] {
        let batched = mul_mat(wty, k, m, &x, n);
        report(
            &format!("mul_mat {} [{k}x{m}]", wty.name()),
            &batched,
            |i| mul_mat(wty, k, m, &x[i * k..(i + 1) * k], 1),
            m,
        );
    }
    let (d, h, hkv, n_kv) = (64, 4, 2, 31);
    let q = rand(d * 7 * h, 5);
    let batched = fa(d, h, hkv, n_kv, &q, 0, 7);
    // result is [d, h, n]: row i is d*h floats
    report(
        "flash_attn_ext f16 kv",
        &batched,
        |i| fa(d, h, hkv, n_kv, &q, i, 1),
        d * h,
    );
}

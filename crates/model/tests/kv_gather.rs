//! CPU micro-benchmark of ADR 0067 step 1: what a paged KV layout costs the attention op.
//!
//! Qwen3-8B attention shape (32 query heads, 8 KV heads, head 128, F16 K and V), sequence
//! length L in {2048, 8192, 32768}, 1 and 4 queries, blocks of 64 and 256 rows scattered in a
//! pool of 4L rows. Three variants, median of 20 at `bench_threads()`:
//!   (i)   flash attention over a contiguous L-row view of the pool;
//!   (ii)  `get_rows` of the sequence's rows, cast back to F16, flash attention;
//!   (iii) flash attention over the whole pool with a mask that leaves the L rows open.
//! From the times and the decode lengths of workload W (`sim.json` of `layout_sim`) it computes
//! y_gather = mean over W's decode inputs of (t(ii)/t(i) - 1) x share(L) and y_sparse likewise
//! with t(iii), share(L) = L x R / (Wt + L x R), ratios interpolated in log2 L, for 256-row
//! blocks (64 reported too). Verdict by ADR 0067's four outcome rules.
//!
//! Hardware label: `cpu`. Ignored by default (slow, release only):
//! `cargo test --release -p tetraeder-model --test kv_gather -- --ignored --nocapture`
//!
//! Env: `TETRAEDER_KV_SIM_OUT` the `sim.json` written by `layout_sim` (read; y values and
//! `verdict` are added to it), `TETRAEDER_KV_GATHER_OUT` the `kv-gather.json` to write,
//! `TETRAEDER_KV_GATHER_LENS` comma list of L (default 2048,8192,32768).

mod common;

use common::bench_threads;
use serde_json::{Value, json};
use std::time::Instant;
use tetraeder_ggml_sys::ggml::{Backend, Context, Graph, Sched, Tensor, Type};

const HEAD: i64 = 128;
const N_HEAD: i64 = 32;
const N_HEAD_KV: i64 = 8;
/// Elements of one K or V row over all KV heads.
const EMBD_KV: i64 = HEAD * N_HEAD_KV;
const F16_NEG_INF: u16 = 0xFC00;
/// K plus V bytes of one row over all layers of Qwen3-8B F16 (ADR 0067).
const R: f64 = 147_456.0;
/// Weight bytes of Qwen3-8B Q4_K_M (ADR 0067).
const WT: f64 = 5_021_827_072.0;
/// Thresholds of ADR 0067, fixed before measuring.
const X: f64 = 0.25;
const Y: f64 = 0.03;
const REPEATS: usize = 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// F16 bits of a finite value with magnitude in [1/8, 1) and a random sign.
    fn f16(&mut self) -> u16 {
        let r = self.next();
        let sign = (r & 1) as u16;
        let exp = 12 + ((r >> 1) % 3) as u16;
        let man = ((r >> 3) & 0x3ff) as u16;
        (sign << 15) | (exp << 10) | man
    }
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Fills `t` (F16, `len` elements) with random values in chunks, so no copy of the whole pool
/// is held. Returns the values when `keep` is set (small sizes).
fn fill(t: Tensor<'_>, len: usize, rng: &mut Rng, keep: bool) -> Vec<u16> {
    let mut kept = Vec::new();
    let chunk = 1 << 22;
    let mut off = 0;
    while off < len {
        let n = chunk.min(len - off);
        let v: Vec<u16> = (0..n).map(|_| rng.f16()).collect();
        t.set_bytes(&bytes(&v), off * 2).unwrap();
        if keep {
            kept.extend(v);
        }
        off += n;
    }
    kept
}

fn views<'a>(ctx: &'a Context, t: Tensor<'a>, first_row: usize, n_rows: usize) -> Tensor<'a> {
    // `[HEAD, n_rows, N_HEAD_KV]` over rows of `EMBD_KV` elements.
    ctx.view(
        t,
        &[HEAD, n_rows as i64, N_HEAD_KV],
        &[(EMBD_KV * 2) as usize, (HEAD * 2) as usize],
        first_row * (EMBD_KV * 2) as usize,
    )
}

/// Attention output of one variant, flattened.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Contig,
    Gather,
    Mask,
}

/// Logical row indices of an L-row sequence in blocks of `block` scattered in a pool of `rows`.
fn scatter(l: usize, rows: usize, block: usize, rng: &mut Rng) -> Vec<i32> {
    let n_blocks = rows / block;
    let mut ids: Vec<usize> = (0..n_blocks).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, (rng.next() as usize) % (i + 1));
    }
    (0..l)
        .map(|p| (ids[p / block] * block + p % block) as i32)
        .collect()
}

/// One variant's graph, allocated and loaded with its inputs.
struct Prepared<'a> {
    sched: Sched,
    graph: Graph<'a>,
    out: Tensor<'a>,
    load: Box<dyn Fn() + 'a>,
}

#[allow(clippy::too_many_arguments)]
fn prepare<'a>(
    ctx: &'a Context,
    backend: &Backend,
    variant: Variant,
    pool_k: Tensor<'a>,
    pool_v: Tensor<'a>,
    idx: &[i32],
    pool_rows: usize,
    n_q: usize,
    q_data: &[f32],
) -> Prepared<'a> {
    let l = idx.len();
    let mut sched = Sched::new(std::slice::from_ref(backend), 64, false).unwrap();
    let q = ctx
        .new_tensor(Type::F32, &[HEAD, n_q as i64, N_HEAD])
        .set_input();
    let scale = 1.0 / (HEAD as f32).sqrt();
    let (out, idx_t, mask) = match variant {
        Variant::Contig => {
            let mask = ctx
                .new_tensor(Type::F16, &[l as i64, n_q as i64])
                .set_input();
            let k = views(ctx, pool_k, 0, l);
            let v = views(ctx, pool_v, 0, l);
            (ctx.flash_attn_ext(q, k, v, mask, scale), None, mask)
        }
        Variant::Gather => {
            let mask = ctx
                .new_tensor(Type::F16, &[l as i64, n_q as i64])
                .set_input();
            let idx_t = ctx.new_tensor(Type::I32, &[l as i64]).set_input();
            let gathered = |pool: Tensor<'a>| {
                let rows = ctx.get_rows(pool, idx_t);
                let h = ctx.cast(rows, Type::F16);
                views(ctx, h, 0, l)
            };
            let k = gathered(pool_k);
            let v = gathered(pool_v);
            (ctx.flash_attn_ext(q, k, v, mask, scale), Some(idx_t), mask)
        }
        Variant::Mask => {
            let mask = ctx
                .new_tensor(Type::F16, &[pool_rows as i64, n_q as i64])
                .set_input();
            let k = views(ctx, pool_k, 0, pool_rows);
            let v = views(ctx, pool_v, 0, pool_rows);
            (ctx.flash_attn_ext(q, k, v, mask, scale), None, mask)
        }
    };
    let out = out.set_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    sched.alloc_graph(&graph).unwrap();

    // The scheduler may reuse an input's memory once its last reader has run, so every run
    // starts by writing the inputs again (outside the timed part).
    let q_data = q_data.to_vec();
    let idx = idx.to_vec();
    let mask_data: Vec<u16> = match variant {
        Variant::Mask => {
            let mut m = vec![F16_NEG_INF; pool_rows];
            for &r in &idx {
                m[r as usize] = 0;
            }
            (0..n_q).flat_map(|_| m.iter().copied()).collect()
        }
        _ => vec![0u16; l * n_q],
    };
    let load = Box::new(move || {
        q.set_f32(&q_data).unwrap();
        mask.set_f16_bits(&mask_data).unwrap();
        if let Some(t) = idx_t {
            t.set_i32(&idx).unwrap();
        }
    });
    Prepared {
        sched,
        graph,
        out,
        load,
    }
}

/// Times the variants with their repeats interleaved (one run of each per round), so a drifting
/// host hits all of them alike. Returns the median microseconds and the output of each.
fn measure(
    backend: &Backend,
    variants: &[(Variant, Tensor<'_>, Tensor<'_>)],
    idx: &[i32],
    pool_rows: usize,
    n_q: usize,
    repeats: usize,
    q_data: &[f32],
) -> Vec<(f64, Vec<f32>)> {
    let ctxs: Vec<Context> = variants
        .iter()
        .map(|_| Context::new(32, 64).unwrap())
        .collect();
    let mut prepared: Vec<Prepared<'_>> = ctxs
        .iter()
        .zip(variants)
        .map(|(ctx, &(variant, k, v))| {
            prepare(ctx, backend, variant, k, v, idx, pool_rows, n_q, q_data)
        })
        .collect();
    for p in &mut prepared {
        for _ in 0..2 {
            (p.load)();
            p.sched.compute(&p.graph).unwrap();
        }
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); prepared.len()];
    for _ in 0..repeats {
        for (p, t) in prepared.iter_mut().zip(&mut times) {
            (p.load)();
            let t0 = Instant::now();
            p.sched.compute(&p.graph).unwrap();
            t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    prepared
        .iter()
        .zip(times)
        .map(|(p, mut t)| {
            t.sort_by(f64::total_cmp);
            let median = if repeats % 2 == 1 {
                t[repeats / 2]
            } else {
                (t[repeats / 2 - 1] + t[repeats / 2]) / 2.0
            };
            (median, p.out.get_f32().unwrap())
        })
        .collect()
}

fn make_q(n_q: usize, rng: &mut Rng) -> Vec<f32> {
    (0..HEAD as usize * n_q * N_HEAD as usize)
        .map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

/// (ii) and (iii) give (i)'s attention output when (i) reads the same rows in order.
fn check_agreement(backend: &Backend) {
    let (l, rows, block, n_q) = (512, 2048, 64, 4);
    let mut rng = Rng(7);
    let idx = scatter(l, rows, block, &mut rng);
    let n = rows * EMBD_KV as usize;

    let pctx = Context::new(2, 0).unwrap();
    let (pk, pv) = (
        pctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]),
        pctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]),
    );
    pctx.alloc_tensors(backend).unwrap();
    let hk = fill(pk, n, &mut rng, true);
    let hv = fill(pv, n, &mut rng, true);

    // Reference pool: the sequence's rows in order at the start.
    let rctx = Context::new(2, 0).unwrap();
    let (rk, rv) = (
        rctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]),
        rctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]),
    );
    rctx.alloc_tensors(backend).unwrap();
    for (src, dst) in [(&hk, rk), (&hv, rv)] {
        let ordered: Vec<u16> = idx
            .iter()
            .flat_map(|&r| src[r as usize * EMBD_KV as usize..][..EMBD_KV as usize].to_vec())
            .collect();
        dst.set_bytes(&bytes(&ordered), 0).unwrap();
    }

    let q = make_q(n_q, &mut rng);
    let want = measure(
        backend,
        &[(Variant::Contig, rk, rv)],
        &idx,
        rows,
        n_q,
        1,
        &q,
    )
    .remove(0)
    .1;
    let got = measure(
        backend,
        &[(Variant::Gather, pk, pv), (Variant::Mask, pk, pv)],
        &idx,
        rows,
        n_q,
        1,
        &q,
    );
    for (variant, (_, got)) in ["gather", "mask"].iter().zip(got) {
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{variant} differs from contiguous by {worst}");
    }
}

#[test]
fn cpu_kv_gather_variants_agree() {
    let backend = Backend::cpu(2).unwrap();
    check_agreement(&backend);
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
        .unwrap_or_else(|| "unknown".into())
}

/// Ratio t(variant)/t(contiguous) at length `l`, interpolated linearly in log2 L between the
/// measured lengths and held at the ends. `pts`: (L, ratio) ascending.
fn ratio_at(pts: &[(f64, f64)], l: f64) -> f64 {
    if l <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        if l <= w[1].0 {
            let f = (l.log2() - w[0].0.log2()) / (w[1].0.log2() - w[0].0.log2());
            return w[0].1 + f * (w[1].1 - w[0].1);
        }
    }
    pts[pts.len() - 1].1
}

/// Mean over W's decode inputs of (ratio(L) - 1) x share(L).
fn y_of(pts: &[(f64, f64)], hist: &[(f64, f64)]) -> f64 {
    let total: f64 = hist.iter().map(|h| h.1).sum();
    hist.iter()
        .map(|&(l, n)| {
            let share = l * R / (WT + l * R);
            n * (ratio_at(pts, l) - 1.0) * share
        })
        .sum::<f64>()
        / total
}

/// ADR 0067 outcome rules 1 to 4.
fn verdict(saving: f64, y_mask: f64, y_gather: f64, y_sparse: f64) -> &'static str {
    if saving < X {
        "chunked"
    } else if y_mask.min(y_gather) <= Y {
        "paged_unmodified"
    } else if y_sparse <= Y {
        "step2"
    } else {
        "chunked"
    }
}

#[test]
#[ignore = "slow CPU benchmark; run with --ignored in release"]
fn cpu_kv_gather_bench() {
    let threads = bench_threads();
    let backend = Backend::cpu(threads as i32).unwrap();
    check_agreement(&backend);

    let lens: Vec<usize> = std::env::var("TETRAEDER_KV_GATHER_LENS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2048, 8192, 32768]);
    let mut cases: Vec<Value> = Vec::new();
    for &l in &lens {
        let rows = 4 * l;
        let mut rng = Rng(l as u64);
        let pctx = Context::new(2, 0).unwrap();
        let pk = pctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]);
        let pv = pctx.new_tensor(Type::F16, &[EMBD_KV, rows as i64]);
        pctx.alloc_tensors(&backend).unwrap();
        fill(pk, rows * EMBD_KV as usize, &mut rng, false);
        fill(pv, rows * EMBD_KV as usize, &mut rng, false);
        for n_q in [1usize, 4] {
            let q = make_q(n_q, &mut rng);
            for block in [64usize, 256] {
                let idx = scatter(l, rows, block, &mut rng);
                let r = measure(
                    &backend,
                    &[
                        (Variant::Contig, pk, pv),
                        (Variant::Gather, pk, pv),
                        (Variant::Mask, pk, pv),
                    ],
                    &idx,
                    rows,
                    n_q,
                    REPEATS,
                    &q,
                );
                let (contig, gather, mask) = (r[0].0, r[1].0, r[2].0);
                eprintln!(
                    "L {l:6} n_q {n_q} block {block:3}: contiguous {contig:10.1} us, gather {gather:10.1} us ({:.2}x), mask {mask:10.1} us ({:.2}x)",
                    gather / contig,
                    mask / contig
                );
                cases.push(json!({
                    "l": l, "n_q": n_q, "block": block,
                    "t_contig_us": contig, "t_gather_us": gather, "t_mask_us": mask,
                }));
            }
        }
    }

    // y values from W's decode lengths, n_q = 1.
    let tier = tetraeder_ggml_sys::ggml::cpu::backend_info().map_or("unknown", |i| i.tier);
    eprintln!("ggml CPU tier {tier}, {threads} threads");
    let mut doc = json!({
        "cpu": cpu_model(),
        "ggml_cpu_tier": tier,
        "threads": threads,
        "cases": cases.clone(),
    });
    let sim_path = std::env::var_os("TETRAEDER_KV_SIM_OUT");
    let sim: Option<Value> = sim_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok());
    if let Some(sim) = sim.as_ref().filter(|_| lens.len() > 1) {
        let hist: Vec<(f64, f64)> = sim["decode_len_hist"]
            .as_object()
            .expect("sim.json holds decode_len_hist")
            .iter()
            .map(|(k, v)| (k.parse::<f64>().unwrap() + 128.0, v.as_f64().unwrap()))
            .collect();
        let pts = |block: usize, field: &str| -> Vec<(f64, f64)> {
            lens.iter()
                .map(|&l| {
                    let c = cases
                        .iter()
                        .find(|c| c["l"] == l && c["n_q"] == 1 && c["block"] == block)
                        .unwrap();
                    (
                        l as f64,
                        c[field].as_f64().unwrap() / c["t_contig_us"].as_f64().unwrap(),
                    )
                })
                .collect()
        };
        let y_gather = y_of(&pts(256, "t_gather_us"), &hist);
        let y_sparse = y_of(&pts(256, "t_mask_us"), &hist);
        doc["y_gather"] = json!(y_gather);
        doc["y_sparse"] = json!(y_sparse);
        doc["y_gather_block64"] = json!(y_of(&pts(64, "t_gather_us"), &hist));
        doc["y_sparse_block64"] = json!(y_of(&pts(64, "t_mask_us"), &hist));
        let saving = sim["saving"].as_f64().unwrap();
        let y_mask = sim["y_mask"].as_f64().unwrap();
        let v = verdict(saving, y_mask, y_gather, y_sparse);
        // The same rules with y_mask measured with both layouts in a pool of the same size.
        let y_same = sim["y_mask_at_chunked_pool"].as_f64().unwrap();
        let v_same = verdict(saving, y_same, y_gather, y_sparse);
        eprintln!(
            "saving {saving:.4} (X {X}), y_mask {y_mask:.5} (same pool {y_same:.5}), y_gather {y_gather:.5}, y_sparse {y_sparse:.5} (Y {Y}): verdict {v} (same pool: {v_same})"
        );
        let mut sim = sim.clone();
        sim["y_gather"] = json!(y_gather);
        sim["y_sparse"] = json!(y_sparse);
        sim["verdict"] = json!(v);
        sim["verdict_at_chunked_pool"] = json!(v_same);
        std::fs::write(
            sim_path.unwrap(),
            serde_json::to_string_pretty(&sim).unwrap() + "\n",
        )
        .unwrap();
    }
    if let Some(path) = std::env::var_os("TETRAEDER_KV_GATHER_OUT") {
        std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    }
}

#[test]
fn cpu_kv_gather_rules_follow_the_adr() {
    assert_eq!(verdict(0.1, 0.0, 0.0, 0.0), "chunked");
    assert_eq!(verdict(0.3, 0.5, 0.02, 0.5), "paged_unmodified");
    assert_eq!(verdict(0.3, 0.02, 0.5, 0.5), "paged_unmodified");
    assert_eq!(verdict(0.3, 0.5, 0.5, 0.02), "step2");
    assert_eq!(verdict(0.3, 0.5, 0.5, 0.5), "chunked");
}

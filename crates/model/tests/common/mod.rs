//! Shared by the model tests: the fixture, libllama as the reference and the logit checks.
#![allow(dead_code)]

pub mod multi_seq;
pub mod split;
pub mod synth;
pub mod trace;

/// The CPU quota of the container (cgroup v2 `cpu.max`), if any.
pub fn cpu_quota() -> Option<f64> {
    let s = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut it = s.split_whitespace();
    let (q, p) = (it.next()?, it.next()?);
    Some(q.parse::<f64>().ok()? / p.parse::<f64>().ok()?)
}

/// Thread counts for a paired benchmark: `TETRAEDER_BENCH_THREADS` (comma list) if set, else
/// the container CPU quota rounded down, else the available parallelism (same rule as
/// `scripts/ot_pair2.sh`; the host core count in a container is not what it may use).
pub fn bench_thread_list() -> Vec<usize> {
    let from_env: Vec<usize> = std::env::var("TETRAEDER_BENCH_THREADS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    if !from_env.is_empty() {
        return from_env;
    }
    let quota = cpu_quota().map(|q| q as usize).filter(|&q| q >= 1);
    vec![quota.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))]
}

/// The one thread count both engines get: the first of [`bench_thread_list`].
pub fn bench_threads() -> usize {
    bench_thread_list()[0]
}

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::{Qwen3, Qwen3Moe};
use tetraeder_plan::KvFormat;

pub fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

/// The tiny dense fixture (`cargo xtask tiny-dense`, ADR 0066).
pub fn dense_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-f16.gguf")
}

/// Largest allowed absolute difference of a reference top-k logit.
/// CPU: both sides run the same F16 weights, F16 KV and ggml kernels in the same op order, and
/// the measured difference is 0; 1e-4 leaves room for a different thread count or SIMD level
/// (summation order, about 1e-6 at the fixture's logit scale of 1 to 5).
pub const TOL_CPU: f32 = 1e-4;
/// GPU builder against the CPU reference: different kernels and F16 rounding of activations,
/// about 1e-3 relative at a logit scale of 1 to 5. An absolute floor only: the F16 fixtures'
/// logits reach 11 to 15, so the GPU checks of `gpu_dense.rs` and `gpu_multi_seq.rs` use
/// [`GPU_TOL`].
pub const TOL_GPU: f32 = 5e-3;

/// A logit tolerance: an absolute floor and a factor of the reference row's largest |logit|.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub abs: f32,
    pub rel: f32,
}

impl Tol {
    /// An absolute bound only.
    pub const fn abs(abs: f32) -> Self {
        Tol { abs, rel: 0.0 }
    }

    /// The bound for a row whose reference logits are `want`.
    pub fn of(self, want: &[f32]) -> f32 {
        self.abs.max(self.rel * largest(want))
    }
}

/// `abs` floor and 1e-3 of the largest logit: the batched-against-alone rule ([`tol_batch`]).
impl From<f32> for Tol {
    fn from(abs: f32) -> Self {
        Tol { abs, rel: 1e-3 }
    }
}

/// CUDA against the CPU, or batched against alone on CUDA, with F16 weights:
/// `max(TOL_GPU, 5e-3 * largest logit)`. On compute capability 8.x an F16 matmul takes mmvf for
/// 1 row, mmf (tensor cores, F32 accumulation) for 2 to 16 rows, and cuBLAS `COMPUTE_16F` (half
/// accumulation and half output) above 16 rows or where mmf does not fit the shape (the 260-row
/// output head, the MoE fixture's 32-wide down experts). F16's relative half-ulp is 4.9e-4; the
/// first box run (RTX 3090) measured 1.4e-3 to 2.4e-3 of the largest logit. libllama runs the
/// same kernels (it sets no F32 precision on Qwen3 matmuls); `gpu_libllama_*` measure its own
/// batched-against-alone gap as the anchor. Greedy tokens stay equal or within ADR 0043's
/// near-tie rule.
pub const GPU_TOL: Tol = Tol {
    abs: TOL_GPU,
    rel: 5e-3,
};

/// Largest |logit| of a row.
pub fn largest(logits: &[f32]) -> f32 {
    logits.iter().fold(0f32, |m, x| m.max(x.abs()))
}
/// A batched step against the same tokens decoded alone: `max(TOL_CPU, 1e-3 * largest logit)`,
/// the rule of `engine/tests/multi_seq.rs`. ggml takes other matmul kernels for other row counts
/// (ADR 0043), so the last bits differ on some CPUs; CI measured 7.4e-3 (Linux) and 3.5e-4
/// (Windows) here. Same-shape comparisons (libllama, the same step split) keep `TOL_CPU`.
pub fn tol_batch(logits: &[f32]) -> f32 {
    Tol::from(TOL_CPU).of(logits)
}
pub const TOP_K: usize = 5;
pub const PROMPT: [i32; 6] = [12, 77, 3, 200, 41, 9];
pub const N_GREEDY: usize = 12;

/// Prefetch counters between two stats snapshots, per `per` (tokens): experts issued, used and
/// wasted, bytes, the worker's read time against the time the step waited for it, and the
/// prediction accuracy (hits over slots scored) per layer and overall, for the prediction one
/// layer ahead and, with prefetch depth 2, two layers ahead (ADR 0038).
pub fn prefetch_json(
    a: &tetraeder_model::ExpertStats,
    b: &tetraeder_model::ExpertStats,
    per: f64,
) -> serde_json::Value {
    let (x, y) = (&a.io, &b.io);
    let read = (y.prefetch_read_time - x.prefetch_read_time).as_secs_f64();
    let wait = (y.prefetch_wait_time - x.prefetch_wait_time).as_secs_f64();
    let (mut hits, mut total) = (0u64, 0u64);
    let (mut hits2, mut total2) = (0u64, 0u64);
    let accuracy2: Vec<Option<f64>> = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(p, q)| {
            let (h, t) = (
                q.predicted2_hits - p.predicted2_hits,
                q.predicted2_total - p.predicted2_total,
            );
            hits2 += h;
            total2 += t;
            (t > 0).then(|| h as f64 / t as f64)
        })
        .collect();
    let accuracy: Vec<Option<f64>> = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(p, q)| {
            let (h, t) = (
                q.predicted_hits - p.predicted_hits,
                q.predicted_total - p.predicted_total,
            );
            hits += h;
            total += t;
            (t > 0).then(|| h as f64 / t as f64)
        })
        .collect();
    serde_json::json!({
        "issued": y.prefetch_issued - x.prefetch_issued,
        "used": y.prefetch_used - x.prefetch_used,
        "wasted": y.prefetch_wasted - x.prefetch_wasted,
        "bytes": y.prefetch_bytes - x.prefetch_bytes,
        "wasted_bytes": y.prefetch_wasted_bytes - x.prefetch_wasted_bytes,
        "read_ms_per_token": read * 1e3 / per,
        "wait_ms_per_token": wait * 1e3 / per,
        "overlap": if read > 0.0 { Some(1.0 - wait / read) } else { None },
        "accuracy_mean": if total > 0 { Some(hits as f64 / total as f64) } else { None },
        "accuracy": accuracy,
        "accuracy2_mean": if total2 > 0 { Some(hits2 as f64 / total2 as f64) } else { None },
        "accuracy2": accuracy2,
    })
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|p| p.0)
        .unwrap()
}

/// Indices of the `k` largest values, largest first, ties by lower index (as a stable full sort).
/// Partial select, not a full sort: the exit harness calls it per decode token over 151,936
/// logits, where a full sort cost 5 to 8 ms of the measured token time (ADR 0060 addendum
/// 2026-11-14).
pub fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let cmp = |a: &usize, b: &usize| v[*b].total_cmp(&v[*a]).then(a.cmp(b));
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(idx.len());
    if k > 0 && k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
    }
    idx.truncate(k);
    idx.sort_unstable_by(cmp);
    idx
}

pub struct Reference {
    pub n_vocab: usize,
    /// Logits of every prompt position.
    pub prompt_logits: Vec<f32>,
    pub greedy: Vec<i32>,
}

pub fn libllama_reference(prompt: &[i32], n_greedy: usize) -> Reference {
    libllama_reference_at(&fixture(), prompt, n_greedy)
}

/// Like [`libllama_reference`] for another GGUF (same shape, for example a quantized copy).
pub fn libllama_reference_at(path: &Path, prompt: &[i32], n_greedy: usize) -> Reference {
    libllama_reference_kv(path, prompt, n_greedy, KvFormat::default())
}

/// Like [`libllama_reference_at`] with the K and V cache types of `kv` (ADR 0050 b). libllama's
/// Hadamard rotation of a quantized cache is off, so it compares with a builder run with
/// `kv_rotation: Some(false)`.
pub fn libllama_reference_kv(
    path: &Path,
    prompt: &[i32],
    n_greedy: usize,
    kv: KvFormat,
) -> Reference {
    tetraeder_ggml_sys::llama::disable_attn_rot();
    libllama_reference_kv_default_rot(path, prompt, n_greedy, kv)
}

/// Like [`libllama_reference_kv`] with libllama's default rotation (on for a quantized cache),
/// unless the process turned it off. It is a process-wide setting read when a context is
/// created: a test binary uses either this or [`libllama_reference_kv`], never both
/// (`kv_rotation.rs` is the one that keeps it on).
pub fn libllama_reference_kv_default_rot(
    path: &Path,
    prompt: &[i32],
    n_greedy: usize,
    kv: KvFormat,
) -> Reference {
    let ty = |f| tetraeder_model::convert::ggml_type(f).expect("ggml type of the KV format");
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers: 0,
                // The builder's CPU weights stay in file layout; without this an AMX or AVX2
                // CPU tier repacks libllama's and the reference runs other kernels (ADR 0052).
                extra_bufts: false,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the fixture"),
    );
    let n_vocab = model.n_vocab() as usize;
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_ubatch: 64,
            n_threads: 2,
            n_threads_batch: 2,
            flash_attn: FlashAttn::On,
            type_k: ty(kv.k),
            type_v: ty(kv.v),
            // A CPU reference: in a CUDA build libllama's scheduler otherwise runs host ops of
            // 32 or more rows (the Q/K norms of a 10-token step) on the GPU.
            op_offload: false,
            ..ContextParams::default()
        },
    )
    .unwrap();
    let mut batch = Batch::new(64);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, true).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let mut prompt_logits = Vec::new();
    for i in 0..prompt.len() {
        prompt_logits.extend_from_slice(ctx.logits_ith(i as i32).unwrap());
    }
    let mut greedy = Vec::new();
    let mut next = argmax(&prompt_logits[(prompt.len() - 1) * n_vocab..]) as i32;
    for step in 0..n_greedy {
        greedy.push(next);
        batch.clear();
        batch
            .push(next, (prompt.len() + step) as i32, 0, true)
            .unwrap();
        ctx.decode(&batch).unwrap();
        next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
    }
    Reference {
        n_vocab,
        prompt_logits,
        greedy,
    }
}

/// What `check_against` needs of a builder: sequence 0 evaluated from scratch.
pub trait Evaluator {
    fn reset(&mut self);
    /// Logits of every position of `t` after the cache.
    fn eval(&mut self, t: &[i32]) -> Vec<f32>;
    /// Logits of the last position only.
    fn eval_last(&mut self, t: &[i32]) -> Vec<f32>;
    fn n_vocab(&self) -> usize;
}

impl Evaluator for Qwen3Moe {
    fn reset(&mut self) {
        Qwen3Moe::reset(self);
    }
    fn eval(&mut self, t: &[i32]) -> Vec<f32> {
        Qwen3Moe::eval(self, t).unwrap()
    }
    fn eval_last(&mut self, t: &[i32]) -> Vec<f32> {
        Qwen3Moe::eval_last(self, t).unwrap()
    }
    fn n_vocab(&self) -> usize {
        Qwen3Moe::n_vocab(self)
    }
}

impl Evaluator for Qwen3 {
    fn reset(&mut self) {
        Qwen3::reset(self);
    }
    fn eval(&mut self, t: &[i32]) -> Vec<f32> {
        Qwen3::eval(self, t).unwrap()
    }
    fn eval_last(&mut self, t: &[i32]) -> Vec<f32> {
        Qwen3::eval_last(self, t).unwrap()
    }
    fn n_vocab(&self) -> usize {
        Qwen3::n_vocab(self)
    }
}

/// Prefill of `PROMPT` (logits of every position), then `N_GREEDY` greedy decode steps, against
/// the reference. Panics on a difference above `tol` in a reference top-k logit or in the
/// greedy sequence. Returns the largest logit difference seen.
pub fn check_against(m: &mut dyn Evaluator, r: &Reference, tol: f32) -> f32 {
    check_against_tol(m, r, Tol::abs(tol))
}

/// [`check_against`] with a bound per position (`tol.of` the reference row). Every position's
/// worst top-k difference is printed before any assertion, so one run shows whether the gap is
/// flat or grows along the prompt.
pub fn check_against_tol(m: &mut dyn Evaluator, r: &Reference, tol: Tol) -> f32 {
    m.reset();
    assert_eq!(m.n_vocab(), r.n_vocab);
    let got = m.eval(&PROMPT);
    assert_eq!(got.len(), r.prompt_logits.len());
    assert!(got.iter().all(|x| x.is_finite()));
    let nv = r.n_vocab;
    let mut worst = 0f32;
    let mut fails = Vec::new();
    for p in 0..PROMPT.len() {
        let (want, have) = (
            &r.prompt_logits[p * nv..(p + 1) * nv],
            &got[p * nv..(p + 1) * nv],
        );
        let top = top_k(want, TOP_K);
        let (i, d) = top
            .iter()
            .map(|&i| (i, (want[i] - have[i]).abs()))
            .fold((top[0], 0f32), |a, b| if b.1 > a.1 { b } else { a });
        let (bound, scale) = (tol.of(want), largest(want));
        eprintln!(
            "position {p}: top-{TOP_K} max |delta| {d:e} at token {i} ({:.2e} of largest {scale}), bound {bound:e}, top-1 equal {}",
            d / scale,
            top[0] == argmax(have)
        );
        worst = worst.max(d);
        if d > bound {
            fails.push(format!(
                "position {p}, token {i}: libllama {} vs builder {} (bound {bound:e})",
                want[i], have[i]
            ));
        }
    }
    // Top-1 equal, or the reference's top two within twice the bound (ADR 0043's near tie).
    for p in 0..PROMPT.len() {
        let (want, have) = (
            &r.prompt_logits[p * nv..(p + 1) * nv],
            &got[p * nv..(p + 1) * nv],
        );
        let (w, h) = (top_k(want, 1)[0], argmax(have));
        if w != h {
            let gap = want[w] - want[h];
            eprintln!("position {p}: top-1 {h} against {w}, reference gap {gap:e}");
            if gap > 2.0 * tol.of(want) {
                fails.push(format!(
                    "top-1 differs at position {p}: {h} against {w}, gap {gap:e}"
                ));
            }
        }
    }

    // Greedy sequence: prefill the prompt, then decode one token per step (n_tokens = 1 graphs,
    // incremental KV cache), teacher-forced with the reference's tokens so one near tie does not
    // make the rest incomparable. Each step's top-1 equals the reference token or the builder's
    // own top two are within twice the bound there.
    m.reset();
    let mut logits = m.eval_last(&PROMPT);
    assert_eq!(logits.len(), r.n_vocab);
    for (k, &t) in r.greedy.iter().enumerate() {
        let h = argmax(&logits);
        if h as i32 != t {
            let gap = logits[h] - logits[t as usize];
            eprintln!("greedy step {k}: builder {h} against reference {t}, builder gap {gap:e}");
            if gap > 2.0 * tol.of(&logits) {
                fails.push(format!("greedy step {k}: {h} against {t}, gap {gap:e}"));
            }
        }
        logits = m.eval(&[t]);
    }
    assert!(fails.is_empty(), "{}", fails.join("; "));
    worst
}

/// A named part of a test for [`run_all`].
pub type Part<'a> = (&'a str, Box<dyn FnOnce() + 'a>);

/// A [`Part`] from a closure.
pub fn part<'a>(name: &'a str, f: impl FnOnce() + 'a) -> Part<'a> {
    (name, Box::new(f))
}

/// Runs every part, then panics with the names of those that panicked, so one failing part does
/// not hide the numbers of the others (box runs print every comparison before asserting).
pub fn run_all(parts: Vec<Part<'_>>) {
    let failed: Vec<&str> = parts
        .into_iter()
        .filter_map(|(name, f)| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .is_err()
                .then_some(name)
        })
        .collect();
    assert!(failed.is_empty(), "failed: {}", failed.join(", "));
}

/// One step of a multi-sequence script: `(seq, first, last)` appends tokens `first..last` of the
/// sequence's stream; `all` asks for the logits of every token, else of the last.
#[derive(Clone, Debug)]
pub struct Item {
    pub seq: u32,
    pub range: std::ops::Range<usize>,
    pub all: bool,
}

/// Logits by (sequence, position) of a multi-sequence script.
pub type Logits = std::collections::BTreeMap<(u32, usize), Vec<f32>>;

/// Runs `steps` (token streams per sequence) on libllama with `n_seq_max` sequences in one
/// unified KV buffer, as `scheduler`'s batches do. Returns the logits of every requested token.
pub fn libllama_script(
    path: &Path,
    streams: &[Vec<i32>],
    steps: &[Vec<Item>],
    kv: KvFormat,
) -> Logits {
    tetraeder_ggml_sys::llama::disable_attn_rot();
    libllama_script_on(path, streams, steps, kv, 0)
}

/// [`libllama_script`] with `n_gpu_layers` (-1: all on the GPU), leaving libllama's rotation as
/// the process has it (it matters only for a quantized cache).
pub fn libllama_script_on(
    path: &Path,
    streams: &[Vec<i32>],
    steps: &[Vec<Item>],
    kv: KvFormat,
    n_gpu_layers: i32,
) -> Logits {
    let ty = |f| tetraeder_model::convert::ggml_type(f).expect("ggml type of the KV format");
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers,
                extra_bufts: false,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the fixture"),
    );
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 512,
            n_batch: 64,
            n_ubatch: 64,
            n_threads: 2,
            n_threads_batch: 2,
            flash_attn: FlashAttn::On,
            type_k: ty(kv.k),
            type_v: ty(kv.v),
            n_seq_max: streams.len() as u32,
            kv_unified: true,
            // `n_gpu_layers` 0 is a CPU run: no host op moved to a GPU of the build (see
            // `libllama_reference`).
            op_offload: n_gpu_layers != 0,
            ..ContextParams::default()
        },
    )
    .unwrap();
    let mut batch = Batch::new(64);
    let mut out = Logits::new();
    for step in steps {
        batch.clear();
        let mut wanted = Vec::new();
        for it in step {
            for p in it.range.clone() {
                let want = it.all || p + 1 == it.range.end;
                if want {
                    wanted.push((it.seq, p, batch.len() as i32));
                }
                batch
                    .push(streams[it.seq as usize][p], p as i32, it.seq as i32, want)
                    .unwrap();
            }
        }
        ctx.decode(&batch).unwrap();
        for (seq, p, i) in wanted {
            out.insert((seq, p), ctx.logits_ith(i).unwrap().to_vec());
        }
    }
    out
}

/// Runs `steps` on a builder with `n_seq_max` sequences; with `batched` off every item is its
/// own step. Returns the logits of every requested token.
pub fn builder_script(
    m: &mut Qwen3,
    streams: &[Vec<i32>],
    steps: &[Vec<Item>],
    batched: bool,
) -> Logits {
    use tetraeder_model::SeqEval;
    let nv = m.n_vocab();
    let mut out = Logits::new();
    let mut run = |m: &mut Qwen3, items: &[&Item]| {
        let inputs: Vec<SeqEval<'_>> = items
            .iter()
            .map(|it| SeqEval {
                seq: it.seq,
                tokens: &streams[it.seq as usize][it.range.clone()],
                all_logits: it.all,
            })
            .collect();
        let mut buf = Vec::new();
        let spans = m.decode(&inputs, &mut buf).unwrap();
        for (it, span) in items.iter().zip(spans) {
            let first = it.range.end - span.len();
            for (k, row) in span.enumerate() {
                out.insert((it.seq, first + k), buf[row * nv..(row + 1) * nv].to_vec());
            }
        }
    };
    for step in steps {
        if batched {
            run(m, &step.iter().collect::<Vec<_>>());
        } else {
            for it in step {
                run(m, &[it]);
            }
        }
    }
    out
}

/// The largest absolute logit difference over the rows both scripts have, and the first row
/// (in key order) that differs at all.
pub fn script_delta(want: &Logits, have: &Logits) -> (f32, Option<(u32, usize)>) {
    let mut worst = 0f32;
    let mut first = None;
    for (key, w) in want {
        let Some(h) = have.get(key) else { continue };
        let d = w
            .iter()
            .zip(h)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if d > 0.0 && first.is_none() {
            first = Some(*key);
        }
        worst = worst.max(d);
    }
    (worst, first)
}

/// Largest absolute logit difference between two scripts' rows over every logit, and the
/// number of rows compared. Panics when a row is missing or the argmax differs where the
/// reference's top-2 margin is above both ADR 0043's near-tie bound (1e-3 of the largest logit)
/// and twice the row's tolerance (a smaller margin can flip within it). With `rel` the
/// tolerance of a row is [`tol_batch`] of the reference row, else `tol`.
pub fn compare_scripts(want: &Logits, have: &Logits, tol: f32, rel: bool, what: &str) -> f32 {
    let tol = if rel {
        Tol::from(TOL_CPU)
    } else {
        Tol::abs(tol)
    };
    compare_scripts_tol(want, have, tol, what)
}

/// Every row both scripts have: `(key, max |delta|, that over the reference's largest |logit|,
/// argmax equal)`.
pub fn script_rows(want: &Logits, have: &Logits) -> Vec<((u32, usize), f32, f32, bool)> {
    want.iter()
        .filter_map(|(key, w)| {
            let h = have.get(key)?;
            let d = w
                .iter()
                .zip(h)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            Some((*key, d, d / largest(w), argmax(w) == argmax(h)))
        })
        .collect()
}

/// Prints [`script_rows`], one line per row, and returns the worst relative difference.
pub fn print_script_rows(what: &str, want: &Logits, have: &Logits) -> f32 {
    let rows = script_rows(want, have);
    for (key, d, rel, same) in &rows {
        eprintln!("{what}: {key:?} max |delta| {d:e} ({rel:.2e} of largest), argmax equal {same}");
    }
    let worst = rows.iter().fold(0f32, |m, r| m.max(r.2));
    eprintln!("{what}: {} rows, worst {worst:.2e} of largest", rows.len());
    worst
}

/// [`compare_scripts`] with a [`Tol`] per row (`tol.of` the reference row).
pub fn compare_scripts_tol(want: &Logits, have: &Logits, tol: Tol, what: &str) -> f32 {
    assert_eq!(
        want.keys().collect::<Vec<_>>(),
        have.keys().collect::<Vec<_>>(),
        "{what}: rows differ"
    );
    let mut worst = 0f32;
    for (key, w) in want {
        let h = &have[key];
        assert!(
            h.iter().all(|x| x.is_finite()),
            "{what}: {key:?} not finite"
        );
        let diff = w
            .iter()
            .zip(h)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst = worst.max(diff);
        let tol = tol.of(w);
        assert!(
            diff <= tol,
            "{what}: {key:?} differs by {diff:e} (tolerance {tol:e})"
        );
        let top = top_k(w, 2);
        let margin = w[top[0]] - w[top[1]];
        if margin > (1e-3 * w[top[0]].abs()).max(2.0 * tol) {
            assert_eq!(top[0], argmax(h), "{what}: argmax differs at {key:?}");
        }
    }
    worst
}

/// The comparisons of a batching anchor, all printed row by row before any assertion.
/// libllama batched against split (every input its own call) is llama.cpp's own batching noise
/// on that backend, and libllama on the backend against the CPU its own backend gap: both are
/// recorded, not asserted (they are not our code). The builder against libllama with the same
/// step shapes runs the same kernels, so a mask, row or position bug shows there while kernel
/// noise cancels: asserted within `tol` (on the CPU the measured gap is 0). The builder batched
/// against split is asserted within `tol` widened to 1.5 times libllama's own worst relative
/// gap, so the bound follows llama.cpp's behaviour (CI measured 1.07e-3 of the largest logit for
/// libllama on the CPU on the dense fixture, above `tol_batch`'s 1e-3) (ADR 0071). Returns the
/// worst gap of each comparison over the largest logit, in print order.
pub fn anchor_pairs(
    what: &str,
    ll_batched: &Logits,
    ll_split: &Logits,
    ll_cpu: &Logits,
    ours_batched: &Logits,
    ours_split: &Logits,
    tol: Tol,
) -> Vec<f32> {
    let pairs = [
        ("libllama batched / libllama split", ll_split, ll_batched),
        ("builder batched / builder split", ours_split, ours_batched),
        (
            "builder batched / libllama batched",
            ll_batched,
            ours_batched,
        ),
        ("builder split / libllama split", ll_split, ours_split),
        ("libllama batched / libllama on the cpu", ll_cpu, ll_batched),
    ];
    let worst: Vec<f32> = pairs
        .iter()
        .map(|(name, want, have)| print_script_rows(&format!("{what} {name}"), want, have))
        .collect();
    for ((name, _, _), w) in pairs.iter().zip(&worst) {
        eprintln!("{what} {name}: worst {w:.2e} of the largest logit");
    }
    let own = Tol {
        abs: tol.abs,
        rel: tol.rel.max(1.5 * worst[0]),
    };
    for (i, (name, want, have)) in pairs.into_iter().enumerate() {
        let what = format!("{what} {name}");
        match i {
            1 => compare_scripts_tol(want, have, own, &what),
            2 | 3 => compare_scripts_tol(want, have, tol, &what),
            _ => continue,
        };
    }
    worst
}

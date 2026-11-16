//! Batched parity of the own builders on the real verdict models (ADR 0065 "Verdict", ADR 0071):
//! three sequences, a prompt chunk mixed into the decode steps of the others, against libllama
//! on CUDA with the same step shapes. `gpu_dense.rs` checks Qwen3-8B with one sequence only.
//!
//! The bound is anchored to libllama's own noise on the same model and card (ADR 0071): its
//! batched run against split, and against the same batched steps with every sequence's KV cells
//! shifted. The second is needed on real models: the builder reserves KV rows per sequence where
//! libllama allocates cells in step order, so CUDA flash attention sums in another order. With
//! Q4_K weights the activations are rounded to q8_1 before every matmul, which can grow such a
//! last-bit difference over the layers; the first box run (`bench/results/20261116d2a/phase4/`)
//! measured 1.3e-2 to 3e-2 of the largest logit between builder and libllama on Qwen3-8B, flat in
//! position and sequence, while the one row whose cells and step shapes coincide (sequence 0 in
//! the first step) was bit-equal. The shifted run measures that sensitivity on libllama itself.
//!
//! Hardware label: `gpu_*`, model (`cpu_*` runs the same body on the tiny fixtures). Skips
//! cleanly when there is no GPU (a failure with `TETRAEDER_REQUIRE_GPU`) or the model file is
//! absent from `TETRAEDER_MODELS_DIR`. libllama and the builder are never on the GPU at the same
//! time: the libllama logits are plain values and its model is dropped before the builder loads.

mod common;

use common::multi_seq::{Step, load, prompt, run_steps, schedule, script_items, steps_logits};
use common::{
    GPU_TOL, Item, Logits, TOL_CPU, Tol, builder_script, compare_scripts_tol, dense_fixture,
    fixture, libllama_script_on, part, print_script_rows, run_all,
};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, Qwen3, Qwen3Moe};
use tetraeder_plan::KvFormat;

const GPU: BackendChoice = BackendChoice::Gpu { index: 0 };

/// The model file, or None (skip) without a GPU or the file.
fn model(rel: &str) -> Option<PathBuf> {
    if !Backend::gpu_available() {
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_GPU").is_none(),
            "TETRAEDER_REQUIRE_GPU is set but there is no GPU device"
        );
        eprintln!("skip: no GPU");
        return None;
    }
    let Some(dir) = std::env::var_os("TETRAEDER_MODELS_DIR") else {
        eprintln!("skip: TETRAEDER_MODELS_DIR is not set");
        return None;
    };
    let path = PathBuf::from(dir).join(rel);
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

/// Three prompts of 20, 24 and 40 tokens plus 6 more tokens each; sequence 2 joins in the second
/// step with a 16-token chunk next to the decode tokens of the others. Every step has at most 64
/// tokens: `libllama_script_on` runs libllama with `n_batch` 64, so a larger step would be split
/// into ubatches there and no longer match the builder's step shapes.
fn script() -> (Vec<Vec<i32>>, Vec<Step>) {
    let lens = [20usize, 24, 40];
    let toks: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let mut t = prompt(1 + k as i32, n);
            t.extend(prompt(11 + k as i32, 6));
            t
        })
        .collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    assert!(
        steps
            .iter()
            .all(|s| s.iter().map(|(_, r)| r.len()).sum::<usize>() <= 64)
    );
    (toks, steps)
}

/// KV cells a fourth, dummy sequence takes ahead of the script in the shifted libllama runs:
/// 7 moves every sequence within the KV tiles; 256 moves them by a whole padding block, so the
/// script ends above 256 cells and libllama's `n_kv` is 512 as the builder's (sequence 1's rows
/// start at 256 there).
const SHIFTS: [usize; 2] = [7, 256];

/// libllama batched (as [`libllama_script_on`]) after a dummy sequence of `shift` tokens (in
/// steps of at most 64) took the first cells of the unified cache: the same step shapes and
/// kernels, only the KV cells of every sequence moved. Its gap to the unshifted run is
/// libllama's own sensitivity to where a sequence sits in the cache, where the builder differs.
fn libllama_shifted(
    path: &Path,
    toks: &[Vec<i32>],
    items: &[Vec<Item>],
    ngl: i32,
    shift: usize,
) -> Logits {
    let dummy = toks.len() as u32;
    let mut streams = toks.to_vec();
    streams.push(prompt(99, shift));
    let mut steps: Vec<Vec<Item>> = (0..shift)
        .step_by(64)
        .map(|a| {
            vec![Item {
                seq: dummy,
                range: a..shift.min(a + 64),
                all: false,
            }]
        })
        .collect();
    steps.extend(items.iter().cloned());
    let mut out = libllama_script_on(path, &streams, &steps, KvFormat::F16, ngl);
    out.retain(|(s, _), _| *s != dummy);
    out
}

/// Every input of `steps` its own call.
fn split(steps: &[Step]) -> Vec<Step> {
    steps.iter().flatten().map(|x| vec![x.clone()]).collect()
}

/// Sequence 0 of [`script`] alone: its prompt, then one token per step. Its cells, `n_kv` and
/// step shapes are the same in libllama and the builder, so it is the strict control.
fn alone(toks: &[Vec<i32>], prompt_len: usize) -> Vec<Step> {
    let mut steps = vec![vec![(0, 0..prompt_len)]];
    steps.extend((prompt_len..toks[0].len()).map(|p| vec![(0, p..p + 1)]));
    steps
}

/// The runs of one model: libllama's and the builder's.
struct Runs {
    ll_batched: Logits,
    ll_split: Logits,
    /// Per entry of [`SHIFTS`].
    ll_shifted: Vec<Logits>,
    ll_alone: Logits,
    ours_batched: Logits,
    ours_split: Logits,
    ours_alone: Logits,
}

/// libllama's runs, one model load each; the model is dropped before the builder loads.
fn libllama_runs(path: &Path, toks: &[Vec<i32>], steps: &[Step], ngl: i32) -> Runs {
    let kv = KvFormat::F16;
    let items = script_items(steps);
    let alone_items = script_items(&alone(toks, steps[0][0].1.end));
    Runs {
        ll_batched: libllama_script_on(path, toks, &items, kv, ngl),
        ll_split: libllama_script_on(path, toks, &script_items(&split(steps)), kv, ngl),
        ll_shifted: SHIFTS
            .iter()
            .map(|&n| libllama_shifted(path, toks, &items, ngl, n))
            .collect(),
        ll_alone: libllama_script_on(path, &toks[..1], &alone_items, kv, ngl),
        ours_batched: Logits::new(),
        ours_split: Logits::new(),
        ours_alone: Logits::new(),
    }
}

/// Prints every comparison, then asserts (ADR 0071). libllama's own gaps (batched against
/// split and against shifted cells) are recorded, not asserted; `own` is `tol` widened to 1.5
/// times the largest of them. Asserted: the builder batched against split within `own`; the
/// builder against libllama at the same step shapes within `own` with `anchored`, else within
/// `tol`; sequence 0 alone, builder against libllama, within `tol` always.
///
/// On CUDA the builder's KV rows are not libllama's cells (rows reserved per sequence against
/// cells allocated in step order), so flash attention sums in another order and the batched
/// same-shape gap is noise of libllama's own size. Alone, cells and shapes coincide, and a
/// divergence of the builder on the real weights shows at the plain bound. On the CPU the
/// order does not depend on the cells, the gap is 0 and `tol` stays everywhere.
fn judge(what: &str, tol: Tol, anchored: bool, r: &Runs) {
    let mut pairs: Vec<(String, &Logits, &Logits)> = vec![(
        "libllama batched / libllama split".to_string(),
        &r.ll_split,
        &r.ll_batched,
    )];
    for (n, l) in SHIFTS.iter().zip(&r.ll_shifted) {
        pairs.push((
            format!("libllama batched / libllama shifted by {n} cells"),
            &r.ll_batched,
            l,
        ));
    }
    let n_own = pairs.len();
    pairs.extend([
        (
            "builder batched / builder split".to_string(),
            &r.ours_split,
            &r.ours_batched,
        ),
        (
            "builder batched / libllama batched".to_string(),
            &r.ll_batched,
            &r.ours_batched,
        ),
        (
            "builder split / libllama split".to_string(),
            &r.ll_split,
            &r.ours_split,
        ),
        (
            "builder alone / libllama alone".to_string(),
            &r.ll_alone,
            &r.ours_alone,
        ),
    ]);
    let worst: Vec<f32> = pairs
        .iter()
        .map(|(name, want, have)| print_script_rows(&format!("{what} {name}"), want, have))
        .collect();
    for ((name, _, _), w) in pairs.iter().zip(&worst) {
        eprintln!("{what} {name}: worst {w:.2e} of the largest logit");
    }
    let own = Tol {
        abs: tol.abs,
        rel: tol
            .rel
            .max(1.5 * worst[..n_own].iter().fold(0f32, |m, &w| m.max(w))),
    };
    let same = if anchored { own } else { tol };
    eprintln!(
        "{what}: bound max({:e}, {:.2e} of the largest logit), same shapes {:.2e}, alone {:.2e}",
        own.abs, own.rel, same.rel, tol.rel
    );
    let bounds = [own, same, same, tol];
    let parts = pairs[n_own..]
        .iter()
        .zip(bounds)
        .map(|((name, want, have), t)| {
            part(name, move || {
                compare_scripts_tol(want, have, t, &format!("{what} {name}"));
            })
        })
        .collect();
    run_all(parts);
}

/// Forgets every sequence of a builder.
macro_rules! forget {
    ($m:expr, $n:expr) => {
        for k in 0..$n as u32 {
            $m.seq_rm(k, 0).unwrap();
        }
    };
}

/// The dense builder on `backend` against libllama on the same device.
fn dense_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let anchored = matches!(backend, BackendChoice::Gpu { .. });
    let (toks, steps) = script();
    let mut r = libllama_runs(path, &toks, &steps, ngl);
    let o = DenseOptions {
        n_seq_max: 3,
        n_ctx_pool: 1024, // rows are reserved in steps of 256 per sequence
        // as libllama_script_on loads the model
        extra_bufts: false,
        ..DenseOptions::new(backend)
    };
    let mut m = Qwen3::load(path, &o).unwrap();
    let items = script_items(&steps);
    r.ours_batched = builder_script(&mut m, &toks, &items, true);
    forget!(m, toks.len());
    r.ours_split = builder_script(&mut m, &toks, &items, false);
    forget!(m, toks.len());
    let alone_items = script_items(&alone(&toks, steps[0][0].1.end));
    r.ours_alone = builder_script(&mut m, &toks, &alone_items, true);
    drop(m);
    judge(what, tol, anchored, &r);
}

/// The MoE builder (resident) on `backend` against libllama on the same device.
fn moe_vs_libllama(path: &Path, backend: BackendChoice, ngl: i32, tol: Tol, what: &str) {
    let anchored = matches!(backend, BackendChoice::Gpu { .. });
    let (toks, steps) = script();
    let mut r = libllama_runs(path, &toks, &steps, ngl);
    let o = LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(backend)
    };
    let mut m = load(path, &o, 3);
    let run = |m: &mut Qwen3Moe, s: &[Step]| steps_logits(run_steps(m, &toks, s), s);
    r.ours_batched = run(&mut m, &steps);
    forget!(m, toks.len());
    r.ours_split = run(&mut m, &split(&steps));
    forget!(m, toks.len());
    r.ours_alone = run(&mut m, &alone(&toks, steps[0][0].1.end));
    drop(m);
    judge(what, tol, anchored, &r);
}

/// hardware: gpu, model. Qwen3-8B Q4_K_M, the dense builder batched against libllama batched.
#[test]
fn gpu_qwen3_8b_batched_matches_libllama() {
    let Some(path) = model("Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf") else {
        return;
    };
    dense_vs_libllama(&path, GPU, -1, GPU_TOL, "qwen3-8b");
}

/// hardware: gpu, model. Qwen3-30B-A3B Q4_K_M, the MoE builder (resident) batched against
/// libllama batched.
#[test]
fn gpu_qwen3_30b_batched_matches_libllama() {
    let Some(path) = model("Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf") else {
        return;
    };
    moe_vs_libllama(&path, GPU, -1, GPU_TOL, "qwen3-30b");
}

/// hardware: cpu. The body of the two tests above on the tiny fixtures, so CI runs the script,
/// the step sizes and the comparison; builder and libllama run the same kernels on the CPU.
#[test]
fn cpu_fixtures_batched_match_libllama() {
    let cpu = || BackendChoice::Cpu { threads: Some(2) };
    let tol = Tol::from(TOL_CPU);
    dense_vs_libllama(&dense_fixture(), cpu(), 0, tol, "cpu dense fixture");
    moe_vs_libllama(&fixture(), cpu(), 0, tol, "cpu moe fixture");
}

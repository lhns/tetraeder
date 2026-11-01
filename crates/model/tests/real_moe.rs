//! The builder on a real quantized MoE model (ADR 0030, 0033): Qwen3-30B-A3B Q4_K_M, resident
//! and streaming, against libllama on the same GPU.
//!
//! Hardware label: `gpu_*` (CUDA build, a 24 GB card, the model file named by
//! `TETRAEDER_MOE_GGUF`); skips cleanly without either. libllama and the builder are never on
//! the GPU at the same time: the reference is reduced to plain values and dropped first.
//!
//! `TETRAEDER_REAL_OUT=<file>` also writes every number as JSON.

mod common;

use common::{TOP_K, argmax, top_k};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::{BackendChoice, LayerExpertStats, LoadOptions, Qwen3Moe};

const N_PROMPT: usize = 64;
const N_GREEDY: usize = 16;
const TEXT: &str = "The old lighthouse keeper climbed the spiral stairs every evening at dusk, \
carrying a lantern, a small loaf of bread and a worn notebook in which he recorded the weather, \
the passing ships and the strange colours of the sea. On the night the great storm arrived, the \
waves rose higher than the cliffs, the wind tore slates from the roof, and far below the harbour \
bell rang without anyone touching it. He wrote one more line in the notebook and then";

/// What libllama produced, as plain values.
struct Reference {
    prompt: Vec<i32>,
    /// Per prompt position, the top-k `(token, logit)`.
    top: Vec<Vec<(usize, f32)>>,
    greedy: Vec<i32>,
    load_s: f64,
    prefill_s: f64,
    decode_tps: f64,
}

/// Everything libllama holds (model, context, GPU buffers) is dropped on return.
fn libllama_on_gpu(path: &Path) -> Reference {
    let t0 = Instant::now();
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers: -1,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the model"),
    );
    let load_s = t0.elapsed().as_secs_f64();
    let mut prompt = model.tokenize(TEXT, model.add_bos(), false).unwrap();
    assert!(prompt.len() >= N_PROMPT, "prompt text too short");
    prompt.truncate(N_PROMPT);
    let n_vocab = model.n_vocab() as usize;
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 512,
            n_batch: 128,
            n_ubatch: 128,
            n_threads: 4,
            n_threads_batch: 4,
            flash_attn: FlashAttn::On,
            ..ContextParams::default()
        },
    )
    .unwrap();
    let mut batch = Batch::new(128);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, true).unwrap();
    }
    let t0 = Instant::now();
    ctx.decode(&batch).unwrap();
    let prefill_s = t0.elapsed().as_secs_f64();
    let mut top = Vec::new();
    for i in 0..prompt.len() {
        let l = ctx.logits_ith(i as i32).unwrap();
        assert_eq!(l.len(), n_vocab);
        top.push(top_k(l, TOP_K).into_iter().map(|t| (t, l[t])).collect());
    }
    let mut next = argmax(ctx.logits_ith(prompt.len() as i32 - 1).unwrap()) as i32;
    let mut greedy = Vec::new();
    let t0 = Instant::now();
    for step in 0..N_GREEDY {
        greedy.push(next);
        batch.clear();
        batch
            .push(next, (prompt.len() + step) as i32, 0, true)
            .unwrap();
        ctx.decode(&batch).unwrap();
        next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
    }
    let decode_tps = N_GREEDY as f64 / t0.elapsed().as_secs_f64();
    Reference {
        prompt,
        top,
        greedy,
        load_s,
        prefill_s,
        decode_tps,
    }
}

/// One builder configuration, loaded, checked and dropped. Returns its greedy tokens and the
/// numbers of the run.
fn run_config(
    name: &str,
    path: &Path,
    expert_slots: Option<usize>,
    ram_slots: Option<usize>,
    split: usize,
    r: &Reference,
) -> (Vec<i32>, Value, Vec<Vec<i32>>) {
    let o = LoadOptions {
        expert_slots,
        ram_slots,
        record_routing: true,
        ..LoadOptions::new(BackendChoice::Gpu { index: 0 })
    };
    let t0 = Instant::now();
    let mut m = Qwen3Moe::load(path, &o).unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    let layers = m.hparams().n_layer as u64;
    let n_vocab = m.n_vocab();

    let t0 = Instant::now();
    // Prefill in `split` evals: the same math in another summation order (the noise floor).
    let mut logits = Vec::with_capacity(r.prompt.len() * n_vocab);
    for part in r.prompt.chunks(r.prompt.len().div_ceil(split)) {
        logits.extend(m.eval(part).unwrap());
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    assert_eq!(logits.len(), r.prompt.len() * n_vocab);
    // Prefill routing per layer, token-major, whatever the split.
    let mut routing: Vec<Vec<i32>> = vec![Vec::new(); layers as usize];
    for step in m.take_routing_log() {
        routing[step.layer].extend(step.ids);
    }
    let mut worst = 0f32;
    // First prompt position whose top-5 logits differ by more than float noise.
    let mut first_off = None;
    for (p, want) in r.top.iter().enumerate() {
        for &(tok, v) in want {
            let d = (v - logits[p * n_vocab + tok]).abs();
            worst = worst.max(d);
            if d > 1e-2 && first_off.is_none() {
                first_off = Some(p);
            }
        }
    }
    let mut next = argmax(&logits[(r.prompt.len() - 1) * n_vocab..]) as i32;
    drop(logits);
    let mut greedy = Vec::new();
    let t0 = Instant::now();
    for _ in 0..N_GREEDY {
        greedy.push(next);
        next = argmax(&m.eval(&[next]).unwrap()) as i32;
    }
    let decode_s = t0.elapsed().as_secs_f64();

    let mut out = json!({
        "load_s": load_s,
        "prefill_s": prefill_s,
        "prefill_tok_s": r.prompt.len() as f64 / prefill_s,
        "decode_tok_s": N_GREEDY as f64 / decode_s,
        "max_top5_delta": worst,
        "first_prompt_pos_off": first_off,
        "overflow_passes": m.overflow_passes(),
        "greedy": greedy,
    });
    if let Some(s) = m.expert_stats() {
        let sum = |f: &dyn Fn(&LayerExpertStats) -> u64| -> u64 { s.layers.iter().map(f).sum() };
        let (full, partial) = (sum(&|l| l.full_steps), sum(&|l| l.partial_steps));
        let prefill_steps = split as u64;
        assert_eq!(
            full + partial,
            layers * (prefill_steps + N_GREEDY as u64),
            "{name}"
        );
        let read_s = s.io.read_time.as_secs_f64();
        let read_gb_s = if read_s > 0.0 {
            s.io.bytes_read as f64 / 1e9 / read_s
        } else {
            0.0
        };
        out["experts"] = json!({
            "compute_hits": sum(&|l| l.compute.hits),
            "compute_misses": sum(&|l| l.compute.misses),
            "full_steps": full,
            "partial_steps": partial,
            "io_bytes": s.io.bytes_read,
            "io_batches": s.io.batches,
            "experts_read": s.io.experts_read,
            "read_s": read_s,
            "read_gb_s": read_gb_s,
            "ram_hits": sum(&|l| l.ram.map_or(0, |x| x.hits)),
            "ram_misses": sum(&|l| l.ram.map_or(0, |x| x.misses)),
        });
    }
    eprintln!("{name}: {out}");
    (greedy, out, routing)
}

/// hardware: gpu (CUDA build, 24 GB; skips without a GPU or `TETRAEDER_MOE_GGUF`). Greedy
/// sequences of the resident and the streaming builder equal libllama's.
#[test]
fn gpu_real_moe_streaming() {
    let Some(path) = std::env::var_os("TETRAEDER_MOE_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_MOE_GGUF not set");
        return;
    };
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    assert!(path.is_file(), "TETRAEDER_MOE_GGUF names no file: {path:?}");

    let r = libllama_on_gpu(&path);
    eprintln!(
        "libllama: load {:.1} s, prefill {:.3} s, decode {:.1} tok/s (info only)",
        r.load_s, r.prefill_s, r.decode_tps
    );
    let mut report = json!({
        "model": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "prompt_tokens": r.prompt.len(),
        "greedy_tokens": N_GREEDY,
        "libllama": {
            "load_s": r.load_s,
            "prefill_s": r.prefill_s,
            "decode_tok_s": r.decode_tps,
            "greedy": r.greedy,
        },
    });
    // Checked after the report is written, so a failure still leaves the numbers.
    let mut failures = Vec::new();
    let mut resident: Option<Vec<i32>> = None;
    // resident_split2: the resident builder with the prefill in two evals, the noise floor of
    // the model on this GPU (summation order only). slots120: prefill fits without overflow passes.
    let configs = [
        ("resident", None, None, 1),
        ("resident_split2", None, None, 2),
        ("slots120", Some(120), None, 1),
        ("slots32", Some(32), None, 1),
        ("slots32_ram64", Some(32), Some(64), 1),
    ];
    let mut resident_routing: Option<Vec<Vec<i32>>> = None;
    for (name, slots, ram, split) in configs {
        let (greedy, mut v, routing) = run_config(name, &path, slots, ram, split, &r);
        // Prefill routing against the resident builder: per layer, the tokens whose expert set
        // differs. A late, single flip points at rounding; early, wide differences at a bug.
        if let Some(base) = &resident_routing {
            let diff: Vec<usize> = base
                .iter()
                .zip(&routing)
                .map(|(a, b)| {
                    let nu = a.len() / r.prompt.len();
                    a.chunks(nu)
                        .zip(b.chunks(nu))
                        .filter(|(x, y)| {
                            let (mut x, mut y) = (x.to_vec(), y.to_vec());
                            x.sort();
                            y.sort();
                            x != y
                        })
                        .count()
                })
                .collect();
            eprintln!("{name}: prefill tokens with other experts, per layer: {diff:?}");
            v["routing_diff_per_layer"] = json!(diff);
        }
        if resident_routing.is_none() {
            resident_routing = Some(routing);
        }
        if greedy != r.greedy {
            failures.push(format!("{name}: greedy sequence differs from libllama"));
        }
        if resident.as_ref().is_some_and(|base| *base != greedy) {
            failures.push(format!("{name}: differs from the resident builder"));
        }
        resident.get_or_insert(greedy);
        report[name] = v;
    }
    if let Some(out) = std::env::var_os("TETRAEDER_REAL_OUT") {
        let out = PathBuf::from(out);
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    assert!(failures.is_empty(), "{failures:?}");
}

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
use tetraeder_model::{BackendChoice, LayerExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

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
fn libllama_on_gpu(path: &Path, threads: usize) -> Reference {
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
            n_threads: threads as i32,
            n_threads_batch: threads as i32,
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

struct Run {
    greedy: Vec<i32>,
    report: Value,
    /// Prefill routing per layer, token-major.
    routing: Vec<Vec<i32>>,
    /// Prefill router weights per layer, same layout.
    weights: Vec<Vec<f32>>,
    /// Prefill logits of every prompt position.
    logits: Vec<f32>,
    n_vocab: usize,
}

/// Per layer, the prompt tokens whose expert set differs between two runs.
fn routing_diff(a: &Run, b: &Run, n_prompt: usize) -> Vec<usize> {
    a.routing
        .iter()
        .zip(&b.routing)
        .map(|(x, y)| {
            let nu = x.len() / n_prompt;
            x.chunks(nu)
                .zip(y.chunks(nu))
                .filter(|(p, q)| {
                    let (mut p, mut q) = (p.to_vec(), q.to_vec());
                    p.sort();
                    q.sort();
                    p != q
                })
                .count()
        })
        .collect()
}

/// Where two runs' prefills first differ: the first layer whose router weights differ bitwise
/// (the layer after the first different FFN output), its tokens, and per token the groups its
/// experts of the layer before fell into with `slots` (prefill from empty slots: distinct experts
/// in first-appearance order, chunks of `slots`).
fn first_weight_diff(a: &Run, b: &Run, n_prompt: usize, slots: usize) -> Value {
    for (l, (x, y)) in a.weights.iter().zip(&b.weights).enumerate() {
        let nu = x.len() / n_prompt;
        let tokens: Vec<usize> = (0..n_prompt)
            .filter(|&t| {
                x[t * nu..(t + 1) * nu]
                    .iter()
                    .zip(&y[t * nu..(t + 1) * nu])
                    .any(|(p, q)| p.to_bits() != q.to_bits())
            })
            .collect();
        if tokens.is_empty() {
            continue;
        }
        let groups = if l == 0 {
            Vec::new()
        } else {
            let ids = &a.routing[l - 1];
            let mut order: Vec<i32> = Vec::new();
            for &e in ids {
                if !order.contains(&e) {
                    order.push(e);
                }
            }
            let group_of = |e: i32| order.iter().position(|&o| o == e).unwrap() / slots;
            (0..n_prompt)
                .map(|t| {
                    let mut p: Vec<usize> = ids[t * nu..(t + 1) * nu]
                        .iter()
                        .map(|&e| group_of(e))
                        .collect();
                    p.sort();
                    p.dedup();
                    p
                })
                .collect::<Vec<_>>()
        };
        let single_group_tokens = groups.iter().filter(|p| p.len() == 1).count();
        let differing_groups: Vec<_> = tokens.iter().map(|&t| groups.get(t).cloned()).collect();
        return json!({
            "layer": l,
            "tokens": tokens,
            "their_groups_in_layer_before": differing_groups,
            "single_group_tokens_in_layer_before": single_group_tokens,
        });
    }
    json!(null)
}

/// Configurations named `*_cpupolicy` use `MissPolicy::Cpu` (ADR 0035): the hot set in the
/// compute slots, every other routed expert computed on the CPU from the RAM tier.
fn is_cpu_policy(name: &str) -> bool {
    name.ends_with("_cpupolicy")
}

/// One builder configuration, loaded, checked and dropped. Returns its greedy tokens and the
/// numbers of the run.
fn run_config(
    name: &str,
    path: &Path,
    expert_slots: Option<usize>,
    ram_slots: Option<usize>,
    split: usize,
    threads: usize,
    r: &Reference,
) -> Run {
    let o = LoadOptions {
        expert_slots,
        ram_slots,
        // `resident_norecord` measures speed: libllama records nothing.
        record_routing: name != "resident_norecord",
        miss: if is_cpu_policy(name) {
            MissPolicy::Cpu
        } else {
            MissPolicy::Upload
        },
        // `cpu_*` configs run the builder on the CPU backend.
        cpu_threads: Some(threads),
        ..LoadOptions::new(if name.starts_with("cpu_") {
            BackendChoice::Cpu { threads: None }
        } else {
            BackendChoice::Gpu { index: 0 }
        })
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
    let mut weights: Vec<Vec<f32>> = vec![Vec::new(); layers as usize];
    for step in m.take_routing_log() {
        routing[step.layer].extend(step.ids);
        weights[step.layer].extend(step.weights);
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
    let mut greedy = Vec::new();
    let uploads_before = m.expert_stats().map_or(0, |s| s.io.uploads);
    let t0 = Instant::now();
    for _ in 0..N_GREEDY {
        greedy.push(next);
        next = argmax(&m.eval(&[next]).unwrap()) as i32;
    }
    let decode_s = t0.elapsed().as_secs_f64();
    let uploads_decode = m.expert_stats().map_or(0, |s| s.io.uploads) - uploads_before;

    let mut out = json!({
        "load_s": load_s,
        "prefill_s": prefill_s,
        "prefill_tok_s": r.prompt.len() as f64 / prefill_s,
        "decode_tok_s": N_GREEDY as f64 / decode_s,
        "max_top5_delta": worst,
        "first_prompt_pos_off": first_off,
        "expert_groups": m.expert_groups(),
        "group_graphs": m.group_graphs(),
        "max_splits": m.max_splits(),
        "uploads_during_decode": uploads_decode,
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
            "cpu_experts": sum(&|l| l.cpu),
            "mixed_steps": sum(&|l| l.mixed_steps),
            "uploads": s.io.uploads,
            "ram_hits": sum(&|l| l.ram.map_or(0, |x| x.hits)),
            "ram_misses": sum(&|l| l.ram.map_or(0, |x| x.misses)),
        });
    }
    if is_cpu_policy(name) {
        // The run above starts with an empty RAM tier and reads experts from disk; llama.cpp's
        // `-ot` has the whole model in memory. Run the prompt again warm, with the time breakdown.
        m.reset();
        m.take_profile();
        let t0 = Instant::now();
        m.eval(&r.prompt).unwrap();
        let prefill_s = t0.elapsed().as_secs_f64();
        let prefill_profile = m.take_profile();
        let t0 = Instant::now();
        for &t in &greedy {
            m.eval(&[t]).unwrap();
        }
        let decode_s = t0.elapsed().as_secs_f64();
        out["warm"] = json!({
            "prefill_tok_s": r.prompt.len() as f64 / prefill_s,
            "decode_tok_s": greedy.len() as f64 / decode_s,
            "prefill_profile": format!("{prefill_profile:?}"),
            "decode_profile": format!("{:?}", m.take_profile()),
        });
    }
    eprintln!("{name}: {out}");
    Run {
        greedy,
        report: out,
        routing,
        weights,
        logits,
        n_vocab,
    }
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

    let threads = common::bench_threads();
    eprintln!("threads for both engines: {threads}");
    let r = libllama_on_gpu(&path, threads);
    eprintln!(
        "libllama: load {:.1} s, prefill {:.3} s, decode {:.1} tok/s (info only)",
        r.load_s, r.prefill_s, r.decode_tps
    );
    let mut report = json!({
        "model": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "prompt_tokens": r.prompt.len(),
        "greedy_tokens": N_GREEDY,
        "threads": threads,
        "cpu_quota": common::cpu_quota(),
        "libllama": {
            "load_s": r.load_s,
            "prefill_s": r.prefill_s,
            "decode_tok_s": r.decode_tps,
            "greedy": r.greedy,
        },
    });
    // Checked after the report is written, so a failure still leaves the numbers.
    let mut failures = Vec::new();
    // (name, compute slots, RAM slots, prefill evals, base config, exact).
    // Single-token evals take CUDA's vector kernel, whose result per row does not depend on the
    // number of experts in the tensor, so streaming must equal the resident builder bit for bit.
    // A prefill that needs pair groups takes MMQ, whose stream-k split depends on the number of
    // rows per expert; it is checked for gross errors only (ADR 0034, 0035). On the CPU the same
    // prefill is exact since a group gives every expert all its rows (ADR 0035), which separates
    // the kernel's rounding from the streaming logic.
    let n = r.prompt.len();
    let configs = [
        ("resident", None, None, 1, None, true),
        ("resident_norecord", None, None, 1, None, true),
        ("resident_split2", None, None, 2, Some("resident"), true),
        ("slots120", Some(120), None, 1, Some("resident"), true),
        ("resident_tokenwise", None, None, n, None, true),
        (
            "slots32_tokenwise",
            Some(32),
            None,
            n,
            Some("resident_tokenwise"),
            true,
        ),
        ("slots32", Some(32), None, 1, Some("resident"), false),
        (
            "slots32_ram64",
            Some(32),
            Some(64),
            1,
            Some("resident"),
            false,
        ),
        // Placement v1 (ADR 0035): 32 compute slots hold the hot set, the rest runs on the CPU
        // from 128 RAM slots; decode must not upload. The prefill takes MMQ for the compute
        // group and the CPU kernels for the others, so it is checked like `slots32`.
        (
            "slots32_ram128_cpupolicy",
            Some(32),
            Some(128),
            1,
            Some("resident"),
            false,
        ),
        // The exit regime (model larger than RAM): 32 RAM slots per layer, so CPU groups read
        // their experts from disk in every step, warm rerun included.
        (
            "slots32_ram32_cpupolicy",
            Some(32),
            Some(32),
            1,
            Some("resident"),
            false,
        ),
        ("cpu_resident", None, None, 1, None, true),
        ("cpu_slots32", Some(32), None, 1, Some("cpu_resident"), true),
    ];
    let mut runs: Vec<(&str, Run)> = Vec::new();
    for (name, slots, ram, split, base, exact) in configs {
        let mut run = run_config(name, &path, slots, ram, split, threads, &r);
        if (name == "resident" || name == "resident_norecord") && run.greedy != r.greedy {
            failures.push(format!("{name}: greedy sequence differs from libllama"));
        }
        if let Some(base) = base.and_then(|b| runs.iter().find(|(n, _)| *n == b)) {
            let b = &base.1;
            let diff = routing_diff(b, &run, n);
            let max_delta = b
                .logits
                .iter()
                .zip(&run.logits)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let top1 = |l: &[f32]| -> Vec<usize> { l.chunks(run.n_vocab).map(argmax).collect() };
            let agree = top1(&b.logits)
                .iter()
                .zip(top1(&run.logits))
                .filter(|(x, y)| **x == *y)
                .count() as f64
                / n as f64;
            eprintln!(
                "{name} vs {}: max |logit delta| {max_delta:e}, top-1 agreement {agree:.3}, routing diff per layer {diff:?}",
                base.0
            );
            if name == "cpu_slots32" {
                let d = first_weight_diff(b, &run, n, 32);
                eprintln!("{name} vs {}: first router weight difference {d}", base.0);
                run.report["first_weight_diff"] = d;
            }
            run.report["vs"] = json!({
                "base": base.0,
                "max_logit_delta": max_delta,
                "top1_agreement": agree,
                "routing_diff_per_layer": diff,
                "greedy_equal": run.greedy == b.greedy,
            });
            if exact {
                if max_delta != 0.0 || run.greedy != b.greedy {
                    failures.push(format!("{name}: not bitwise equal to {}", base.0));
                }
            } else if diff[0] != 0 || agree < 0.75 {
                failures.push(format!(
                    "{name}: layer-0 routing or top-1 agreement ({agree:.3}) off against {}",
                    base.0
                ));
            }
        }
        if is_cpu_policy(name) {
            let up = run.report["uploads_during_decode"].as_u64();
            eprintln!(
                "{name}: prefill {:.1} tok/s, decode {:.1} tok/s, uploads during decode {up:?}",
                run.report["prefill_tok_s"].as_f64().unwrap_or(0.0),
                run.report["decode_tok_s"].as_f64().unwrap_or(0.0),
            );
            if up != Some(0) {
                failures.push(format!("{name}: decode uploaded {up:?} experts"));
            }
        }
        report[name] = run.report.clone();
        runs.push((name, run));
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

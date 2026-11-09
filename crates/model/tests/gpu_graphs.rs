//! CUDA graph replay in decode (ADR 0036): libllama against the builder with the whole-step graph
//! kept across steps off and on, on a real MoE model.
//!
//! Hardware label: `gpu_*` (CUDA build, a 24 GB card, the model file named by
//! `TETRAEDER_MOE_GGUF`); skips cleanly without either. Run once per process setting of
//! `GGML_CUDA_DISABLE_GRAPHS` and `GGML_CUDA_GRAPH_DEBUG`: ggml reads both once. With the debug
//! variable set the CUDA graph events are counted from ggml's log.
//!
//! Environment: `TETRAEDER_GRAPHS_DECODE` (decode tokens, default 128), `TETRAEDER_GRAPHS_REPEATS`
//! (default 3), `TETRAEDER_GRAPHS_OUT` (JSON file). libllama runs first and is dropped before the
//! builder loads.

mod common;

use common::argmax;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, CudaGraphLogCounts, FlashAttn, Model, ModelParams,
    cuda_graph_log_counts,
};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

const N_PROMPT: usize = 64;
/// Decode steps left out of the steady rate: CUDA graph warmup and the capture.
const WARMUP_STEPS: usize = 4;
const TEXT: &str = "The old lighthouse keeper climbed the spiral stairs every evening at dusk, \
carrying a lantern, a small loaf of bread and a worn notebook in which he recorded the weather, \
the passing ships and the strange colours of the sea. On the night the great storm arrived, the \
waves rose higher than the cliffs, the wind tore slates from the roof, and far below the harbour \
bell rang without anyone touching it. He wrote one more line in the notebook and then";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Decode rates from the duration of each step: over all steps, and without the first few.
fn rates(steps: &[f64]) -> (f64, f64) {
    let all = steps.len() as f64 / steps.iter().sum::<f64>();
    let rest = &steps[WARMUP_STEPS.min(steps.len())..];
    let steady = rest.len() as f64 / rest.iter().sum::<f64>();
    (all, steady)
}

fn delta(a: CudaGraphLogCounts, b: CudaGraphLogCounts) -> Value {
    json!({
        "warmups": b.warmups - a.warmups,
        "resets": b.resets - a.resets,
        "reused": b.reused - a.reused,
        "update_failed": b.update_failed - a.update_failed,
    })
}

struct Reference {
    prompt: Vec<i32>,
    /// Per repeat: (decode tok/s, steady tok/s, greedy tokens).
    repeats: Vec<(f64, f64, Vec<i32>)>,
}

/// libllama on the GPU, every layer offloaded; everything it holds is dropped on return.
fn libllama_on_gpu(path: &Path, n_decode: usize, repeats: usize) -> Reference {
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
    let mut prompt = model.tokenize(TEXT, model.add_bos(), false).unwrap();
    assert!(prompt.len() >= N_PROMPT, "prompt text too short");
    prompt.truncate(N_PROMPT);
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: (N_PROMPT + n_decode + 64).max(512) as u32,
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
    let mut out = Vec::new();
    for _ in 0..repeats {
        ctx.memory_clear();
        batch.clear();
        for (i, &t) in prompt.iter().enumerate() {
            batch.push(t, i as i32, 0, i == prompt.len() - 1).unwrap();
        }
        ctx.decode(&batch).unwrap();
        let mut next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
        let (mut greedy, mut steps) = (Vec::new(), Vec::new());
        for step in 0..n_decode {
            greedy.push(next);
            batch.clear();
            batch
                .push(next, (prompt.len() + step) as i32, 0, true)
                .unwrap();
            let t0 = Instant::now();
            ctx.decode(&batch).unwrap();
            next = argmax(ctx.logits_ith(-1).unwrap()) as i32;
            steps.push(t0.elapsed().as_secs_f64());
        }
        let (all, steady) = rates(&steps);
        out.push((all, steady, greedy));
    }
    Reference {
        prompt,
        repeats: out,
    }
}

/// hardware: gpu (CUDA build, 24 GB; skips without a GPU or `TETRAEDER_MOE_GGUF`)
#[test]
fn gpu_cuda_graphs_decode() {
    let Some(path) = std::env::var_os("TETRAEDER_MOE_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_MOE_GGUF not set");
        return;
    };
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    assert!(path.is_file(), "TETRAEDER_MOE_GGUF names no file: {path:?}");
    let n_decode = env_usize("TETRAEDER_GRAPHS_DECODE", 128);
    let repeats = env_usize("TETRAEDER_GRAPHS_REPEATS", 3);
    assert!(n_decode > WARMUP_STEPS && repeats > 0);

    let r = libllama_on_gpu(&path, n_decode, repeats);
    let libllama: Vec<Value> = r
        .repeats
        .iter()
        .map(|(all, steady, greedy)| {
            json!({"decode_tok_s": all, "steady_tok_s": steady, "greedy": greedy})
        })
        .collect();
    eprintln!("libllama: {}", json!(libllama));

    let o = LoadOptions {
        n_ctx: N_PROMPT + n_decode + 64,
        record_routing: false,
        ..LoadOptions::new(BackendChoice::Gpu { index: 0 })
    };
    let mut m = Qwen3Moe::load(&path, &o).unwrap();
    let mut builder = Vec::new();
    let mut failures = Vec::new();
    let mut first_greedy: Option<Vec<i32>> = None;
    for rep in 0..repeats {
        for reuse in [false, true] {
            m.set_reuse_step_graph(reuse);
            m.reset();
            m.take_profile();
            let first = m.eval_last(&r.prompt).unwrap();
            m.take_profile();
            let (built, counts) = (m.step_graphs_built(), cuda_graph_log_counts());
            let mut next = argmax(&first) as i32;
            let (mut greedy, mut steps) = (Vec::new(), Vec::new());
            for _ in 0..n_decode {
                greedy.push(next);
                let t0 = Instant::now();
                next = argmax(&m.eval(&[next]).unwrap()) as i32;
                steps.push(t0.elapsed().as_secs_f64());
            }
            let (all, steady) = rates(&steps);
            let profile = m.take_profile();
            let run = json!({
                "repeat": rep,
                "reuse_step_graph": reuse,
                "decode_tok_s": all,
                "steady_tok_s": steady,
                "profile": format!("{profile:?}"),
                "step_graphs_built": m.step_graphs_built() - built,
                "cuda_graph_log": delta(counts, cuda_graph_log_counts()),
                "greedy": greedy,
            });
            eprintln!("builder: {run}");
            match &first_greedy {
                None => first_greedy = Some(greedy),
                Some(g) if *g != greedy => {
                    failures.push(format!("repeat {rep} reuse={reuse}: greedy tokens differ"));
                }
                Some(_) => {}
            }
            builder.push(run);
        }
    }
    let builder_greedy = first_greedy.expect("at least one run");
    let report = json!({
        "model": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "prompt_tokens": r.prompt.len(),
        "decode_tokens": n_decode,
        "repeats": repeats,
        "warmup_steps_excluded": WARMUP_STEPS,
        "GGML_CUDA_DISABLE_GRAPHS_set": std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some(),
        "GGML_CUDA_GRAPH_DEBUG_set": std::env::var_os("GGML_CUDA_GRAPH_DEBUG").is_some(),
        "libllama": libllama,
        "builder": builder,
        "builder_equals_libllama_greedy": r.repeats.iter().all(|(_, _, g)| *g == builder_greedy),
    });
    eprintln!("GRAPHS_JSON {report}");
    if let Some(out) = std::env::var_os("TETRAEDER_GRAPHS_OUT") {
        let out = PathBuf::from(out);
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    assert!(failures.is_empty(), "{failures:?}");
}

//! Serve decode without the gateway (docs/m1-open.md item 4): the libllama engine and the MoE
//! builder through the scheduler's `Backend`, the way `tetraeder serve` drives them, at several
//! context depths and with the serve sampler, so the A6000 gap (0.88x llama-server in the Claude
//! Code proxy run) can be reproduced and split without Claude Code. llama-bench `-d` gives the
//! raw llama.cpp number next to it (scripts/decode_gap.sh).
//!
//! Hardware label: `gpu_*` (CUDA build, a 48 GB card for 128K context with the 30B; the model
//! file named by `TETRAEDER_GAP_GGUF`); skips cleanly without either. The engines load one after
//! the other, so only one KV cache is allocated at a time.
//!
//! Env: `TETRAEDER_GAP_DEPTHS` (comma separated, default `0,20000,100000`), `TETRAEDER_GAP_CTX`
//! (131072), `TETRAEDER_GAP_DECODE` (tokens per depth, 128), `TETRAEDER_GAP_THREADS` (5, the proxy
//! run's), `TETRAEDER_GAP_ENGINES` (`llama,moe`), `TETRAEDER_GAP_OUT` (JSON file).

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tetraeder_engine::{ContextConfig, FlashAttn, Placement};
use tetraeder_ggml_sys::ggml::Backend as GgmlBackend;
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::sample::{SampleParams, Sampler, greedy};
use tetraeder_scheduler::{Loaded, PREFILL_CHUNK, load_llama, load_moe};

/// Decode steps left out of the steady numbers: graph capture after a depth change.
const WARMUP_STEPS: usize = 4;
const TEXT: &str = "fn main() { let mut total = 0; for i in 0..100 { total += i * 3 % 7; } \
println!(\"{total}\"); } // The quick brown fox jumps over the lazy dog while the compiler checks \
every borrow, every lifetime and every trait bound in the crate.\n";

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Serve's sampling defaults (llama-server's): what Claude Code requests got.
const SERVE_SAMPLING: SampleParams = SampleParams {
    temperature: 0.8,
    top_k: 40,
    top_p: 0.95,
};

fn load(engine: &str, path: &Path, n_ctx: usize, threads: i32) -> Loaded {
    match engine {
        // `tetraeder serve --engine llama` with the proxy run's flags.
        "llama" => load_llama(
            path,
            &Placement {
                n_gpu_layers: -1,
                ..Placement::default()
            },
            &ContextConfig {
                n_ctx: n_ctx as u32,
                n_batch: 2048,
                n_ubatch: 512,
                threads: Some(threads),
                threads_batch: None,
                flash_attn: FlashAttn::Auto,
                op_offload: true,
            },
            None,
        )
        .expect("libllama loads"),
        // `tetraeder serve` (auto picks the builder for qwen3moe), every expert resident.
        "moe" => {
            let mut o = LoadOptions::new(BackendChoice::Gpu { index: 0 });
            o.cpu_threads = Some(threads as usize);
            o.n_ctx = n_ctx;
            load_moe(path, &o).expect("the builder loads")
        }
        other => panic!("unknown engine {other}"),
    }
}

fn mean_ms(d: &[Duration]) -> f64 {
    d.iter().sum::<Duration>().as_secs_f64() * 1e3 / d.len().max(1) as f64
}

/// hardware: gpu (CUDA build, 48 GB; skips without a GPU or `TETRAEDER_GAP_GGUF`)
#[test]
fn gpu_serve_decode_by_depth() {
    let Some(path) = std::env::var_os("TETRAEDER_GAP_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_GAP_GGUF not set");
        return;
    };
    if !GgmlBackend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    assert!(path.is_file(), "TETRAEDER_GAP_GGUF names no file: {path:?}");
    let depths: Vec<usize> = env_or("TETRAEDER_GAP_DEPTHS", "0,20000,100000")
        .split(',')
        .map(|d| d.trim().parse().expect("depth"))
        .collect();
    let n_ctx = env_usize("TETRAEDER_GAP_CTX", 131072);
    let n_decode = env_usize("TETRAEDER_GAP_DECODE", 128);
    let threads = env_usize("TETRAEDER_GAP_THREADS", 5) as i32;
    assert!(n_decode > WARMUP_STEPS);
    let mut rows = Vec::new();
    for engine in env_or("TETRAEDER_GAP_ENGINES", "llama,moe").split(',') {
        let Loaded {
            mut backend,
            tokenizer,
        } = load(engine, &path, n_ctx, threads);
        let unit: Vec<i32> = tokenizer.tokenize(TEXT, false, false).unwrap();
        for &depth in &depths {
            // At least a short prompt: depth 0 means "nothing before the decode" as in tg128.
            let n_prompt = depth.max(16);
            assert!(
                n_prompt + n_decode < n_ctx,
                "depth {depth} does not fit {n_ctx}"
            );
            let prompt: Vec<i32> = unit.iter().copied().cycle().take(n_prompt).collect();
            backend.truncate(0).unwrap();
            let t = Instant::now();
            for chunk in prompt.chunks(PREFILL_CHUNK) {
                backend.eval(chunk).unwrap();
            }
            let prefill_s = t.elapsed().as_secs_f64();
            let mut sampler = Sampler::new(SERVE_SAMPLING, 42);
            let c0 = backend.counters();
            let (mut eval, mut sample, mut argmax) = (Vec::new(), Vec::new(), Vec::new());
            let mut next = tokenizer.bos().max(0);
            for _ in 0..n_decode {
                let t = Instant::now();
                let logits = backend.eval(&[next]).unwrap();
                eval.push(t.elapsed());
                let t = Instant::now();
                let g = greedy(logits, |_| true).unwrap();
                argmax.push(t.elapsed());
                let t = Instant::now();
                next = sampler.sample(logits, None).unwrap_or(g) as i32;
                sample.push(t.elapsed());
            }
            let counters: Value = backend
                .counters()
                .into_iter()
                .zip(c0)
                .map(|((k, v), (_, v0))| (k.to_owned(), json!(v - v0)))
                .collect::<serde_json::Map<_, _>>()
                .into();
            let steady = |d: &[Duration]| mean_ms(&d[WARMUP_STEPS..]);
            let row = json!({
                "engine": engine,
                "depth": depth,
                "prompt_tokens": n_prompt,
                "prefill_tok_s": n_prompt as f64 / prefill_s,
                "decode_tokens": n_decode,
                "eval_ms": mean_ms(&eval),
                "eval_ms_steady": steady(&eval),
                "sample_ms": mean_ms(&sample),
                "argmax_ms": mean_ms(&argmax),
                // Engine only, and what serve's worker loop pays per token (eval + sample).
                "eval_tok_s": 1e3 / steady(&eval),
                "serve_tok_s": 1e3 / (steady(&eval) + steady(&sample)),
                "counters": counters,
            });
            eprintln!("decode_gap: {row}");
            rows.push(row);
        }
    }
    let report = json!({
        "model": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "n_ctx": n_ctx,
        "threads": threads,
        "sampling": {"temperature": SERVE_SAMPLING.temperature, "top_k": SERVE_SAMPLING.top_k,
                     "top_p": SERVE_SAMPLING.top_p},
        "warmup_steps_excluded": WARMUP_STEPS,
        "rows": rows,
    });
    eprintln!("DECODE_GAP_JSON {report}");
    if let Some(out) = std::env::var_os("TETRAEDER_GAP_OUT") {
        let out = PathBuf::from(out);
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
}

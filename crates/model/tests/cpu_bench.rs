//! CPU speed of the builder against libllama on the same GGUF with the same thread count
//! (rule 3): prefill and decode tok/s, paired and interleaved, plus the builder's time breakdown.
//!
//! Hardware label: `cpu`. Ignored by default (slow, release only):
//! `cargo test --release -p tetraeder-model --test cpu_bench -- --ignored --nocapture`
//!
//! Env: `TETRAEDER_MOE_GGUF` a real model (default: a synthetic Q4_K_M MoE, generated once in
//! the temp dir), `TETRAEDER_BENCH_THREADS` comma list (default: the available parallelism),
//! `TETRAEDER_BENCH_PROMPT` (64), `TETRAEDER_BENCH_DECODE` (32), `TETRAEDER_BENCH_REPEATS` (3),
//! `TETRAEDER_BENCH_VARIANTS` (`libllama,builder,builder_cpupolicy`), `TETRAEDER_BENCH_OUT` (JSON
//! file), `TETRAEDER_BENCH_MACHINE` (a note). In a GPU build set `CUDA_VISIBLE_DEVICES=` so
//! libllama cannot offload large ops to the GPU.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe_at};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Ftype, Model, ModelParams, ThreadPool,
};
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, ModelFile, Qwen3Moe, StepProfile};

/// Qwen3-30B-A3B's depth (49 graph segments per step) and top-8 routing with small dims, so the
/// per-graph cost is visible next to the compute. About 340 MB as Q4_K_M.
const SPEC: MoeSpec = MoeSpec {
    layers: 48,
    embd: 512,
    heads: 8,
    heads_kv: 2,
    head_dim: 64,
    experts: 32,
    used: 8,
    ff: 256,
    seed: 20261101,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_path() -> (PathBuf, Value) {
    if let Some(p) = std::env::var_os("TETRAEDER_MOE_GGUF") {
        let p = PathBuf::from(p);
        let v = json!({"path": p.display().to_string()});
        return (p, v);
    }
    let s = SPEC;
    let name = format!(
        "tetraeder-bench-l{}-e{}-x{}-u{}-f{}-q4km.gguf",
        s.layers, s.embd, s.experts, s.used, s.ff
    );
    let p = quantized_moe_at(&s, Ftype::Q4KM, &std::env::temp_dir().join(name));
    let v = json!({
        "synthetic": format!("{s:?}"),
        "ftype": "Q4_K_M",
        "bytes": std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
    });
    (p, v)
}

/// The CPU quota of the container (cgroup v2 `cpu.max`), if any.
fn cpu_quota() -> Option<f64> {
    let s = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut it = s.split_whitespace();
    let (q, p) = (it.next()?, it.next()?);
    Some(q.parse::<f64>().ok()? / p.parse::<f64>().ok()?)
}

fn cpu_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|m| m.trim().to_string())
}

/// One timed run: prefill of the prompt, then greedy decode.
struct Sample {
    prefill: Duration,
    decode: Duration,
    tokens: Vec<i32>,
}

trait Engine {
    fn run(&mut self, prompt: &[i32], n_decode: usize) -> Sample;
    /// The builder's breakdown of the last run's prefill and decode.
    fn profiles(&mut self) -> Option<(StepProfile, StepProfile)> {
        None
    }
}

struct Llama {
    ctx: Context,
    batch: Batch,
}

impl Llama {
    fn new(model: &Arc<Model>, threads: usize, n_prompt: usize, n_ctx: usize) -> Llama {
        let t = threads as i32;
        let mut ctx = Context::new(
            model,
            &ContextParams {
                n_ctx: n_ctx as u32,
                n_batch: n_prompt as u32,
                n_ubatch: n_prompt as u32,
                n_threads: t,
                n_threads_batch: t,
                flash_attn: FlashAttn::On,
                ..ContextParams::default()
            },
        )
        .unwrap();
        // As llama-bench and the engine do.
        ctx.attach_threadpool(Arc::new(ThreadPool::new(t).unwrap()));
        Llama {
            ctx,
            batch: Batch::new(n_prompt),
        }
    }
}

impl Engine for Llama {
    fn run(&mut self, prompt: &[i32], n_decode: usize) -> Sample {
        self.ctx.memory_clear();
        self.batch.clear();
        for (i, &t) in prompt.iter().enumerate() {
            self.batch
                .push(t, i as i32, 0, i + 1 == prompt.len())
                .unwrap();
        }
        let t0 = Instant::now();
        self.ctx.decode(&self.batch).unwrap();
        let mut next = argmax(self.ctx.logits_ith(-1).unwrap()) as i32;
        let prefill = t0.elapsed();
        let mut tokens = Vec::new();
        let t0 = Instant::now();
        for step in 0..n_decode {
            tokens.push(next);
            self.batch.clear();
            self.batch
                .push(next, (prompt.len() + step) as i32, 0, true)
                .unwrap();
            self.ctx.decode(&self.batch).unwrap();
            next = argmax(self.ctx.logits_ith(-1).unwrap()) as i32;
        }
        Sample {
            prefill,
            decode: t0.elapsed(),
            tokens,
        }
    }
}

struct Builder {
    m: Qwen3Moe,
    last: Option<(StepProfile, StepProfile)>,
}

impl Engine for Builder {
    fn run(&mut self, prompt: &[i32], n_decode: usize) -> Sample {
        self.m.reset();
        self.m.take_profile();
        let t0 = Instant::now();
        let mut next = argmax(&self.m.eval_last(prompt).unwrap()) as i32;
        let prefill = t0.elapsed();
        let pp = self.m.take_profile();
        let mut tokens = Vec::new();
        let t0 = Instant::now();
        for _ in 0..n_decode {
            tokens.push(next);
            next = argmax(&self.m.eval(&[next]).unwrap()) as i32;
        }
        let decode = t0.elapsed();
        self.last = Some((pp, self.m.take_profile()));
        Sample {
            prefill,
            decode,
            tokens,
        }
    }

    fn profiles(&mut self) -> Option<(StepProfile, StepProfile)> {
        self.last.take()
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Per step (prefill) or per token (decode), in ms.
fn profile_json(p: &StepProfile, per: f64) -> Value {
    let ms = |d: Duration| d.as_secs_f64() * 1e3 / per;
    json!({
        "graphs": p.graphs as f64 / per,
        "inputs_ms": ms(p.inputs),
        "build_ms": ms(p.build),
        "alloc_ms": ms(p.alloc),
        "compute_ms": ms(p.compute),
        "readback_ms": ms(p.readback),
        "streamer_ms": ms(p.streamer),
        "total_ms": ms(p.total),
    })
}

fn add(a: &mut StepProfile, b: &StepProfile) {
    a.steps += b.steps;
    a.graphs += b.graphs;
    a.inputs += b.inputs;
    a.build += b.build;
    a.alloc += b.alloc;
    a.compute += b.compute;
    a.readback += b.readback;
    a.streamer += b.streamer;
    a.total += b.total;
}

/// hardware: cpu. Builder against libllama on the CPU, written as JSON (see the module docs).
#[test]
#[ignore = "slow CPU benchmark; run with --ignored in release"]
fn cpu_bench_builder_vs_libllama() {
    let n_prompt = env_usize("TETRAEDER_BENCH_PROMPT", 64);
    let n_decode = env_usize("TETRAEDER_BENCH_DECODE", 32);
    let repeats = env_usize("TETRAEDER_BENCH_REPEATS", 3).max(1);
    let avail = std::thread::available_parallelism().map_or(1, usize::from);
    let threads: Vec<usize> = std::env::var("TETRAEDER_BENCH_THREADS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![avail]);
    let variants: Vec<String> = std::env::var("TETRAEDER_BENCH_VARIANTS")
        .unwrap_or_else(|_| "libllama,builder,builder_cpupolicy".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let (path, model_info) = model_path();
    let n_ctx = (n_prompt + n_decode + 16).next_multiple_of(256);

    let llama_model = variants.iter().any(|v| v == "libllama").then(|| {
        Arc::new(
            Model::load(
                &path,
                &ModelParams {
                    n_gpu_layers: 0,
                    ..ModelParams::default()
                },
            )
            .unwrap(),
        )
    });
    let mut runs = Vec::new();
    let mut ratios = Vec::new();
    for &t in &threads {
        let mut engines: Vec<(String, Box<dyn Engine>)> = Vec::new();
        let mut prompt: Vec<i32> = Vec::new();
        for v in &variants {
            let e: Box<dyn Engine> = match v.as_str() {
                "libllama" => Box::new(Llama::new(
                    llama_model.as_ref().unwrap(),
                    t,
                    n_prompt,
                    n_ctx,
                )),
                "builder" | "builder_cpupolicy" => {
                    let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(t) });
                    o.n_ctx = n_ctx;
                    if v == "builder_cpupolicy" {
                        // A quarter of the experts in the compute tier (the hot set), every
                        // expert in the RAM tier: pair groups without disk reads.
                        let n_expert = ModelFile::open(&path).unwrap().hparams.n_expert;
                        o.expert_slots = Some((n_expert as usize / 4).max(1));
                        o.ram_slots = Some(n_expert as usize);
                        o.miss = MissPolicy::Cpu;
                    }
                    let m = Qwen3Moe::load(&path, &o).unwrap();
                    if prompt.is_empty() {
                        let n_vocab = m.n_vocab() as i64;
                        prompt = (0..n_prompt as i64)
                            .map(|i| ((i * 7919 + 13) % n_vocab.min(256)) as i32)
                            .collect();
                    }
                    Box::new(Builder { m, last: None })
                }
                other => panic!("unknown variant `{other}`"),
            };
            engines.push((v.clone(), e));
        }
        if prompt.is_empty() {
            prompt = (0..n_prompt as i32)
                .map(|i| (i * 7919 + 13) % 256)
                .collect();
        }
        // Warmup: fills caches, builds the hot set of the cpu policy from the first prefill.
        for (_, e) in &mut engines {
            e.run(&prompt, 4);
            e.profiles();
        }
        let n = engines.len();
        let mut samples: Vec<Vec<Sample>> = (0..n).map(|_| Vec::new()).collect();
        let mut profs = vec![(StepProfile::default(), StepProfile::default()); n];
        for r in 0..repeats {
            // Rotate the order so no variant always runs first.
            for k in 0..n {
                let i = (k + r) % n;
                let s = engines[i].1.run(&prompt, n_decode);
                if let Some((pp, dp)) = engines[i].1.profiles() {
                    add(&mut profs[i].0, &pp);
                    add(&mut profs[i].1, &dp);
                }
                samples[i].push(s);
            }
        }
        let reference = variants.iter().position(|v| v == "libllama");
        let mut medians = Vec::new();
        for (i, (name, _)) in engines.iter().enumerate() {
            let pre: Vec<f64> = samples[i]
                .iter()
                .map(|s| n_prompt as f64 / s.prefill.as_secs_f64())
                .collect();
            let dec: Vec<f64> = samples[i]
                .iter()
                .map(|s| n_decode as f64 / s.decode.as_secs_f64())
                .collect();
            let (mp, md) = (median(pre.clone()), median(dec.clone()));
            medians.push((mp, md));
            let mut run = json!({
                "threads": t,
                "variant": name,
                "prefill_tok_s": pre,
                "decode_tok_s": dec,
                "prefill_tok_s_median": mp,
                "decode_tok_s_median": md,
                "greedy_equals_libllama": reference.map(|r| samples[r][0].tokens == samples[i][0].tokens),
            });
            if profs[i].0.steps > 0 {
                let reps = repeats as f64;
                run["profile_prefill_per_step"] = profile_json(&profs[i].0, reps);
                run["profile_decode_per_token"] = profile_json(&profs[i].1, reps * n_decode as f64);
            }
            eprintln!("{run}");
            runs.push(run);
        }
        if let Some(r) = reference {
            for (i, (name, _)) in engines.iter().enumerate() {
                if i == r {
                    continue;
                }
                let ratio = json!({
                    "threads": t,
                    "variant": name,
                    "prefill_ratio": medians[i].0 / medians[r].0,
                    "decode_ratio": medians[i].1 / medians[r].1,
                });
                eprintln!("ratio vs libllama: {ratio}");
                ratios.push(ratio);
            }
        }
    }
    let out = json!({
        "machine": std::env::var("TETRAEDER_BENCH_MACHINE").ok(),
        "env": {
            "os": std::env::consts::OS,
            "cpu": cpu_model(),
            "available_parallelism": avail,
            "cpu_quota": cpu_quota(),
            "gpu_devices_visible": Backend::gpu_count(),
        },
        "model": model_info,
        "n_prompt": n_prompt,
        "n_decode": n_decode,
        "repeats": repeats,
        "ratios_builder_over_libllama": ratios,
        "runs": runs,
    });
    if let Some(p) = std::env::var_os("TETRAEDER_BENCH_OUT") {
        std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
}

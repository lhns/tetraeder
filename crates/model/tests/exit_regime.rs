//! The Phase 2 exit regime (ADR 0031, 0035, 0037): a MoE larger than the RAM given to the
//! expert tiers, `MissPolicy::Cpu`, a hot set in the compute tier, a RAM tier sized from a
//! byte budget, misses read from disk every step. Reports prefill and decode speed, the step
//! time split, read throughput, the per-layer full-hit rate during decode and memory (rule 3).
//!
//! Hardware label: `cpu` with the default synthetic model; `gpu_*` on the exit box
//! (`TETRAEDER_EXIT_GPU=1`, CUDA build). Ignored by default (slow, release only):
//! `cargo test --release -p tetraeder-model --test exit_regime -- --ignored --nocapture`
//!
//! Env (all optional):
//! - `TETRAEDER_EXIT_GGUF`: the model, the first shard of a split one. Default: a synthetic
//!   Q4_K_M MoE with realistic slab sizes, generated once in the temp dir.
//! - `TETRAEDER_EXIT_GPU=1`: GPU backend; the CPU runs the RAM-tier experts.
//! - `TETRAEDER_EXIT_VRAM_SLOT_BYTES` / `TETRAEDER_EXIT_RAM_SLOT_BYTES`: tier budgets
//!   (ADR 0037). Without them `TETRAEDER_EXIT_SLOTS` (2) and `TETRAEDER_EXIT_RAM_SLOTS`
//!   (a quarter of the experts) are slot counts.
//! - `TETRAEDER_EXIT_PREFILL` (256) prompt tokens, run in ubatches of each size in
//!   `TETRAEDER_EXIT_UBATCH` (comma list, "64,256"), one cold load per size.
//! - `TETRAEDER_EXIT_DECODE` (32) greedy tokens after the prefill.
//! - `TETRAEDER_EXIT_CTX` (prefill + decode, rounded up), `TETRAEDER_EXIT_THREADS`.
//! - `TETRAEDER_EXIT_OP_OFFLOAD=1`: large CPU-group matmuls run on the GPU (ggml op offload).
//! - `TETRAEDER_EXIT_REFRESH_HOT=1`: rebuild the hot set from the whole prefill before decode
//!   (by default it comes from the first ubatch).
//! - `TETRAEDER_EXIT_PROMPT_FILE`: text tokenized with the model's vocab (libllama, vocab
//!   only), repeated to length. Default: deterministic pseudo-random token ids.
//! - `TETRAEDER_EXIT_OUT` (JSON file), `TETRAEDER_EXIT_MACHINE` (a note).

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe_at};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Ftype, Model, ModelParams};
use tetraeder_memory::process;
use tetraeder_model::{
    BackendChoice, ExpertStats, IoCounters, LoadOptions, MissPolicy, ModelFile, Qwen3Moe,
    StepProfile,
};

/// Expert of about 2.6 MB (three slabs) like Qwen3-30B-A3B, top-4 of 16, so a RAM tier of a
/// quarter of the experts reads from disk every step. About 350 MB as Q4_K_M, 2.6 GB as the F32
/// intermediate (generated once).
const SPEC: MoeSpec = MoeSpec {
    layers: 8,
    embd: 2048,
    heads: 16,
    heads_kv: 4,
    head_dim: 128,
    experts: 16,
    used: 4,
    ff: 768,
    seed: 20261101,
};

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_flag(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == "1")
}

fn model_path() -> (PathBuf, Value) {
    if let Some(p) = std::env::var_os("TETRAEDER_EXIT_GGUF") {
        let p = PathBuf::from(p);
        return (p.clone(), json!({"path": p.display().to_string()}));
    }
    let s = SPEC;
    let name = format!(
        "tetraeder-exit-l{}-e{}-x{}-u{}-f{}-q4km.gguf",
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

/// `n` prompt tokens: the prompt file through the model's vocab, repeated, or a fixed
/// pseudo-random sequence.
fn prompt(path: &PathBuf, n: usize, n_vocab: usize) -> Vec<i32> {
    if let Some(f) = std::env::var_os("TETRAEDER_EXIT_PROMPT_FILE") {
        let text = std::fs::read_to_string(f).expect("prompt file");
        let vocab = Model::load(
            path,
            &ModelParams {
                vocab_only: true,
                ..ModelParams::default()
            },
        )
        .expect("libllama reads the vocab");
        let toks = vocab.tokenize(&text, vocab.add_bos(), false).unwrap();
        assert!(!toks.is_empty(), "empty prompt file");
        return toks.iter().copied().cycle().take(n).collect();
    }
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n_vocab as u64) as i32
        })
        .collect()
}

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

/// Reads between two counter snapshots, per `per` (tokens).
fn io_json(a: &IoCounters, b: &IoCounters, per: f64) -> Value {
    let bytes = b.bytes_read - a.bytes_read;
    let secs = (b.read_time - a.read_time).as_secs_f64();
    json!({
        "batches": b.batches - a.batches,
        "experts_read": b.experts_read - a.experts_read,
        "bytes": bytes,
        "bytes_per_token": bytes as f64 / per,
        "read_s": secs,
        "read_ms_per_token": secs * 1e3 / per,
        "read_gb_s": if secs > 0.0 { bytes as f64 / secs / 1e9 } else { 0.0 },
        "uploads": b.uploads - a.uploads,
    })
}

/// Per layer, the share of decode layer steps with every routed expert in the compute tier,
/// and the RAM-tier hit rate.
fn layers_json(a: &ExpertStats, b: &ExpertStats) -> Value {
    let mut full = Vec::new();
    let mut ram_hit = Vec::new();
    for (la, lb) in a.layers.iter().zip(&b.layers) {
        let f = lb.full_steps - la.full_steps;
        let p = lb.partial_steps - la.partial_steps;
        full.push(f as f64 / (f + p).max(1) as f64);
        if let (Some(ra), Some(rb)) = (la.ram, lb.ram) {
            let h = rb.hits - ra.hits;
            let m = rb.misses - ra.misses;
            ram_hit.push(h as f64 / (h + m).max(1) as f64);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({
        "full_hit_rate_mean": mean(&full),
        "ram_hit_rate_mean": mean(&ram_hit),
        "full_hit_rate": full,
        "ram_hit_rate": ram_hit,
    })
}

/// hardware: cpu (default) or gpu_* (`TETRAEDER_EXIT_GPU=1`). See the module docs.
#[test]
#[ignore = "slow exit-regime benchmark; run with --ignored in release"]
fn exit_regime() {
    let gpu = env_flag("TETRAEDER_EXIT_GPU");
    if gpu && !Backend::gpu_available() {
        eprintln!("skip: TETRAEDER_EXIT_GPU=1 but no GPU in this build");
        return;
    }
    let (path, model_info) = model_path();
    let file = ModelFile::open(&path).expect("model opens");
    let h = file.hparams.clone();
    let n_prefill = env_u64("TETRAEDER_EXIT_PREFILL").unwrap_or(256) as usize;
    let n_decode = env_u64("TETRAEDER_EXIT_DECODE").unwrap_or(32) as usize;
    let ubatches: Vec<usize> = std::env::var("TETRAEDER_EXIT_UBATCH")
        .unwrap_or_else(|_| "64,256".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let threads = env_u64("TETRAEDER_EXIT_THREADS")
        .map(|t| t as usize)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let n_ctx = env_u64("TETRAEDER_EXIT_CTX")
        .map(|c| c as usize)
        .unwrap_or(n_prefill + n_decode);
    let tokens = prompt(&path, n_prefill, h.n_vocab as usize);

    let mut opts = LoadOptions::new(if gpu {
        BackendChoice::Gpu { index: 0 }
    } else {
        BackendChoice::Cpu {
            threads: Some(threads),
        }
    });
    opts.n_ctx = n_ctx;
    opts.miss = MissPolicy::Cpu;
    opts.op_offload = env_flag("TETRAEDER_EXIT_OP_OFFLOAD");
    let refresh_hot = env_flag("TETRAEDER_EXIT_REFRESH_HOT");
    match env_u64("TETRAEDER_EXIT_VRAM_SLOT_BYTES") {
        Some(b) => opts.expert_slot_bytes = Some(b),
        None => opts.expert_slots = Some(env_u64("TETRAEDER_EXIT_SLOTS").unwrap_or(2) as usize),
    }
    match env_u64("TETRAEDER_EXIT_RAM_SLOT_BYTES") {
        Some(b) => opts.ram_slot_bytes = Some(b),
        None => {
            let n = env_u64("TETRAEDER_EXIT_RAM_SLOTS").unwrap_or(h.n_expert as u64 / 4);
            opts.ram_slots = Some(n as usize);
        }
    }
    let per_layer = file.expert_bytes_per_layer().expect("expert sizes");
    let slot_bytes_all_layers: u64 = per_layer.iter().sum();

    let mut runs = Vec::new();
    let mut io_backend = None;
    for &ub in &ubatches {
        let t0 = Instant::now();
        let mut m = Qwen3Moe::load(&path, &opts).expect("builder loads");
        let load_s = t0.elapsed().as_secs_f64();
        io_backend = m.io_backend();
        let s0 = m.expert_stats().expect("the exit regime streams experts");

        m.take_profile();
        let t0 = Instant::now();
        let mut ubatch_s = Vec::new();
        let mut last = Vec::new();
        for chunk in tokens.chunks(ub.max(1)) {
            let t = Instant::now();
            last = m.eval_last(chunk).expect("prefill");
            ubatch_s.push(t.elapsed().as_secs_f64());
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let pp = m.take_profile();
        // The hot set comes from the first ubatch; optionally rebuild it from the whole prefill.
        let t = Instant::now();
        if refresh_hot {
            m.refresh_hot_set().expect("hot set refresh");
        }
        let refresh_s = t.elapsed().as_secs_f64();
        let s1 = m.expert_stats().unwrap();

        let mut next = argmax(&last) as i32;
        let mut greedy = Vec::new();
        let t0 = Instant::now();
        for _ in 0..n_decode {
            greedy.push(next);
            next = argmax(&m.eval(&[next]).expect("decode")) as i32;
        }
        let decode_s = t0.elapsed().as_secs_f64();
        let tg = m.take_profile();
        let s2 = m.expert_stats().unwrap();

        let memory: Vec<Value> = m
            .memory()
            .iter()
            .map(|u| {
                json!({"device": u.device, "tier": u.tier, "kind": u.kind,
                       "capacity": u.capacity, "used": u.used})
            })
            .collect();
        let run = json!({
            "ubatch": ub,
            "load_s": load_s,
            "prefill_tokens": n_prefill,
            "prefill_s": prefill_s,
            "prefill_tok_s": n_prefill as f64 / prefill_s,
            "hot_refresh_s": refresh_s,
            "ubatch_s": ubatch_s,
            "prefill_profile_per_ubatch": profile_json(&pp, pp.steps.max(1) as f64),
            "prefill_io": io_json(&s0.io, &s1.io, n_prefill as f64),
            "decode_tokens": n_decode,
            "decode_tok_s": n_decode as f64 / decode_s,
            "decode_profile_per_token": profile_json(&tg, n_decode.max(1) as f64),
            "decode_io": io_json(&s1.io, &s2.io, n_decode.max(1) as f64),
            "decode_layers": layers_json(&s1, &s2),
            "greedy": greedy,
            "max_splits": m.max_splits(),
            "memory": memory,
            "rss_bytes": process::current_rss(),
            "peak_rss_bytes": process::peak_rss(),
        });
        eprintln!("{run}");
        runs.push(run);
    }
    let out = json!({
        "machine": std::env::var("TETRAEDER_EXIT_MACHINE").ok(),
        "env": {
            "os": std::env::consts::OS,
            "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
            "gpu_devices_visible": Backend::gpu_count(),
        },
        "model": model_info,
        "options": {
            "gpu": gpu,
            "threads": threads,
            "n_ctx": n_ctx,
            "op_offload": opts.op_offload,
            "refresh_hot": refresh_hot,
            "expert_slots": opts.expert_slots,
            "expert_slot_bytes": opts.expert_slot_bytes,
            "ram_slots": opts.ram_slots,
            "ram_slot_bytes": opts.ram_slot_bytes,
        },
        "io_backend": io_backend,
        "expert_bytes_per_layer": per_layer,
        "slot_bytes_all_layers": slot_bytes_all_layers,
        "runs": runs,
    });
    if let Some(p) = std::env::var_os("TETRAEDER_EXIT_OUT") {
        std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
}

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
//! - `TETRAEDER_EXIT_DECODE` (32) greedy tokens after the prefill, in
//!   `TETRAEDER_EXIT_DECODE_WINDOWS` (1) windows reported one by one (`decode_windows`).
//! - `TETRAEDER_EXIT_CTX` (prefill + decode, rounded up), `TETRAEDER_EXIT_THREADS`.
//! - `TETRAEDER_EXIT_OP_OFFLOAD=1`: large CPU-group matmuls run on the GPU (ggml op offload).
//! - `TETRAEDER_EXIT_PREFETCH`: `0` turns the predicted expert prefetch into the RAM tier off,
//!   `1` on; unset follows `LoadOptions` (on, ADR 0038). The JSON gets `prefill_prefetch` and `decode_prefetch` (counters, overlap and
//!   prediction accuracy per layer and mean).
//! - `TETRAEDER_EXIT_PREFETCH_DEPTH` (1): layers ahead that the prefetch predicts and reads, 1 or
//!   2; depth 2 adds `accuracy2` and `accuracy2_mean` to the prefetch JSON.
//! - `TETRAEDER_EXIT_IO_SERVICE`: `1` reads the prefetch (and, with overlap, the misses) through
//!   the persistent I/O service instead of the single reader thread (ADR 0054), `0` the reader;
//!   unset follows `LoadOptions`. `TETRAEDER_EXIT_IO_WORKERS` (default 16) sets its workers.
//! - `TETRAEDER_EXIT_OVERLAP`: `1` computes the CPU experts of a decode layer in arrival order
//!   (needs the service), `0` off; unset follows `LoadOptions`. The JSON gets `decode_overlap`:
//!   layer steps with experts in flight and those experts per token, `read_wait` ms per token,
//!   the hidden share (`1 - wait / span`) and the main thread's wait in `urgent_wait_ms` (every
//!   wait, despite the name). The pending experts split into `pending_urgent` (misses of the
//!   step), `pending_top_up` (depth-1 prefetch) and `pending_ahead` (depth 2), with the mean time
//!   from a pending prefetch's submit to its layer's plan (`pending_prefetch_lead_ms`); steps with
//!   an urgent miss get their own wait, span and hidden share (`urgent_steps_*`), the others
//!   `prefetch_steps_*`. `urgent_read_ms` is the mean submit-to-landing time of an urgent miss
//!   (units in flight together each count their whole latency, so it is not device time).
//! - `TETRAEDER_EXIT_URGENT_FIRST` (0/1, default 0): with overlap, a layer step submits its
//!   misses before the next layers' prefetch (`LoadOptions::urgent_first`).
//! - `TETRAEDER_EXIT_ADMIT_SHARE` (0 to 1, default off): decode admits that share of the
//!   compute-tier misses into VRAM slots (`LoadOptions::admit_share`, ADR 0059); `decode_io`
//!   gets `admitted`, `admitted_bytes` and `admit_ms_per_token` (the synchronous uploads).
//! - `TETRAEDER_EXIT_RAM_POLICY` (`lru` default, or `freq:<share>`): the RAM-tier policy
//!   (`LoadOptions::ram_policy`, ADR 0050 (c)); `freq` pins the top share of each layer's RAM
//!   slots by routing count during decode. Recorded in the options as `ram_policy`.
//! - `TETRAEDER_EXIT_RAM_HUGEPAGES` (0/1, default 0): `MADV_HUGEPAGE` on the RAM tier's buffer
//!   right after allocation (`LoadOptions::ram_hugepages`, ADR 0035 addendum). Each run records
//!   `ram_hugepages` (`applied`, `advised_bytes`, `errno`); the host sampler's `anon_huge_gb`
//!   shows whether the kernel backed the tier with huge pages.
//! - `TETRAEDER_EXIT_KV_ROT` (0/1): Hadamard rotation of Q, K and V (ADR 0050 addendum). Unset:
//!   on for a quantized KV format, off for f16, as libllama. The reference engine follows it;
//!   `TETRAEDER_EXIT_REF_ATTN_ROT` (0/1) overrides the reference alone. The JSON options record
//!   `kv_rotation` as resolved.
//! - `TETRAEDER_EXIT_KV`: K and V cache format, `f16` (default), `q8_0`, `q8_0/q4_0` or `q4_0`
//!   (ADR 0050 b); both engines use it (`LoadOptions::kv`, libllama `type_k` and `type_v`).
//! - `TETRAEDER_EXIT_CPU_POLL`: ggml threadpool polling level 0 to 100 (default: auto, ADR
//!   0061: 0 when the cgroup CPU quota binds, else ggml's 50). The JSON options record the level
//!   in effect (`cpu_poll`), `cpu_poll_auto` and `cpu_quota_detected`; null for the reference.
//! - `TETRAEDER_EXIT_POOL_PAUSE` (0/1): pause the CPU threadpool in GPU-only segments and read
//!   waits (ADR 0061 addendum; default off). The JSON options record `pool_pause`.
//! - `TETRAEDER_EXIT_PREFETCH_THREADS`: CPU compute threads while a prefetch read is in flight
//!   (default: all).
//! - `TETRAEDER_EXIT_REFRESH_HOT=1`: rebuild the hot set from the whole prefill before decode
//!   (by default it comes from the first ubatch).
//! - `TETRAEDER_EXIT_PROMPT_FILE`: text tokenized with the model's vocab (libllama, vocab
//!   only), repeated to length. Default: deterministic pseudo-random token ids.
//! - `TETRAEDER_EXIT_ENGINE=libllama`: the reference instead of the builder: libllama with
//!   every layer on the GPU and the overrides of `TETRAEDER_EXIT_REF_OT` (default `exps=CPU`,
//!   a `;` list of `regex=BUFFER_TYPE` as `llama-bench -ot`, first match wins). Same tokens,
//!   ubatch, context and threads, so it is also the llama.cpp baseline on the real prompt
//!   (llama-bench decodes random tokens); the JSON has the same speed and output fields.
//! - `TETRAEDER_EXIT_ROUTING_OUT`: write the routing of the run (with the predictions) as a
//!   `tetraeder-routing` trace for `tetraeder-bench replay` (ADR 0050 (c)). One file per ubatch
//!   size: with several, the size is inserted before the extension.
//! - `TETRAEDER_EXIT_OUT` (JSON file), `TETRAEDER_EXIT_MACHINE` (a note).
//! - `OTEL_EXPORTER_OTLP_ENDPOINT`: the builder exports its metrics there (ADR 0060), every
//!   `OTEL_METRIC_EXPORT_INTERVAL` ms, with `OTEL_RESOURCE_ATTRIBUTES`; unset, no telemetry.
//!
//! Every run records `prompt` (token count and an FNV-1a hash of the token ids), `first_top`
//! (the 10 largest logits after the prefill, `[id, logit]`) and `margins` (top-1 minus top-2
//! logit at each greedy step), so two runs can be shown to have had the same input and a
//! differing greedy token can be told apart from a near-tie.

mod common;

use common::synth::{MoeSpec, quantized_moe_at};
use common::trace::RoutingRecorder;
use common::{argmax, prefetch_json, top_k};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Ftype, Model, ModelParams, resolve_buffer_type,
};
use tetraeder_memory::process;
use tetraeder_model::convert::ggml_type;
use tetraeder_model::{
    BackendChoice, ExpertStats, IoCounters, LoadOptions, MissPolicy, ModelFile, Qwen3Moe,
    StepProfile,
};
use tetraeder_plan::KvFormat;
use tetraeder_telemetry::{Telemetry, TelemetryConfig, cgroup_cpu_quota};

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

fn git_out(args: &[&str]) -> Option<String> {
    let o = std::process::Command::new("git").args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// Who produced the result, for the calibration ledger (ADR 0056): the engine build, the
/// session and the host. The commit comes from `TETRAEDER_GIT_COMMIT` (the box checkout may not
/// be a git repo), else git.
fn provenance(threads: usize) -> Value {
    let commit = std::env::var("TETRAEDER_GIT_COMMIT")
        .ok()
        .filter(|c| !c.is_empty())
        .or_else(|| git_out(&["rev-parse", "HEAD"]));
    let host = std::env::var("CONTAINER_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| {
            ["HOSTNAME", "COMPUTERNAME"]
                .iter()
                .filter_map(|k| std::env::var(k).ok())
                .find(|v| !v.is_empty())
        })
        .or_else(|| {
            let o = std::process::Command::new("hostname").output().ok()?;
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        });
    // cgroup `cpu.max` (v1 cfs quota and period as fallback); null when unlimited or unknown
    let quota = cgroup_cpu_quota();
    json!({
        "engine_commit": commit,
        "engine_dirty": git_out(&["status", "--porcelain"]).map(|s| !s.is_empty()),
        "session": std::env::var("TETRAEDER_SESSION").ok().filter(|s| !s.is_empty()),
        "host_id": host,
        "cpu_quota": quota,
        "threads": threads,
        "cpu_variant": tetraeder_ggml_sys::cpu::backend_info().ok().map(|i| i.tier),
    })
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

/// FNV-1a over the token ids (little-endian i32), as hex.
fn tokens_hash(tokens: &[i32]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in tokens.iter().flat_map(|t| t.to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The 10 largest logits as `[id, logit]`.
fn top_json(logits: &[f32]) -> Value {
    json!(
        top_k(logits, 10)
            .into_iter()
            .map(|i| json!([i, logits[i]]))
            .collect::<Vec<_>>()
    )
}

/// Top-1 minus top-2 logit.
fn margin(logits: &[f32]) -> f32 {
    let t = top_k(logits, 2);
    logits[t[0]] - logits[t[1]]
}

/// Greedy decode of `n` tokens from `last` (the prefill's last logits) through `step`.
fn greedy_decode(
    last: &[f32],
    n: usize,
    mut step: impl FnMut(i32) -> Vec<f32>,
) -> (Vec<i32>, Vec<f32>) {
    let mut next = argmax(last) as i32;
    let mut margins = vec![margin(last)];
    let mut greedy = Vec::new();
    for _ in 0..n {
        greedy.push(next);
        let l = step(next);
        margins.push(margin(&l));
        next = argmax(&l) as i32;
    }
    margins.truncate(n);
    (greedy, margins)
}

/// The libllama reference run (`TETRAEDER_EXIT_ENGINE=libllama`), one per ubatch size.
fn reference_run(
    path: &PathBuf,
    tokens: &[i32],
    ub: usize,
    n_decode: usize,
    n_ctx: usize,
    threads: usize,
    kv: KvFormat,
) -> Value {
    // The reference rotates a quantized cache like the builder does (TETRAEDER_EXIT_KV_ROT), so
    // both store the same values; TETRAEDER_EXIT_REF_ATTN_ROT=0/1 overrides the reference alone.
    let ref_rot = env_u64("TETRAEDER_EXIT_REF_ATTN_ROT")
        .map(|v| v == 1)
        .or(env_u64("TETRAEDER_EXIT_KV_ROT").map(|v| v == 1))
        .unwrap_or(kv != KvFormat::F16);
    if !ref_rot {
        tetraeder_ggml_sys::llama::disable_attn_rot();
    }
    let patterns = std::env::var("TETRAEDER_EXIT_REF_OT").unwrap_or_else(|_| "exps=CPU".into());
    let overrides = patterns
        .split(';')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (re, buft) = p.rsplit_once('=').expect("override is regex=BUFFER_TYPE");
            (
                re.to_string(),
                resolve_buffer_type(buft).expect("buffer type"),
            )
        })
        .collect();
    let t0 = Instant::now();
    let model = Arc::new(
        Model::load(
            path,
            &ModelParams {
                n_gpu_layers: 999,
                buffer_overrides: overrides,
                ..ModelParams::default()
            },
        )
        .expect("libllama loads the model"),
    );
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: n_ctx as u32,
            n_batch: ub as u32,
            n_ubatch: ub as u32,
            n_threads: threads as i32,
            n_threads_batch: threads as i32,
            flash_attn: FlashAttn::On,
            type_k: ggml_type(kv.k).expect("K type"),
            type_v: ggml_type(kv.v).expect("V type"),
            ..ContextParams::default()
        },
    )
    .expect("libllama context");
    let load_s = t0.elapsed().as_secs_f64();
    let mut batch = Batch::new(ub.max(1));
    let t0 = Instant::now();
    let mut pos = 0;
    let n_chunks = tokens.len().div_ceil(ub.max(1));
    for (c, chunk) in tokens.chunks(ub.max(1)).enumerate() {
        batch.clear();
        for (i, &t) in chunk.iter().enumerate() {
            let last = c + 1 == n_chunks && i + 1 == chunk.len();
            batch.push(t, pos, 0, last).unwrap();
            pos += 1;
        }
        ctx.decode(&batch).expect("libllama prefill");
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let last = ctx.logits_ith(-1).unwrap().to_vec();
    let t0 = Instant::now();
    let (greedy, margins) = greedy_decode(&last, n_decode, |t| {
        batch.clear();
        batch.push(t, pos, 0, true).unwrap();
        pos += 1;
        ctx.decode(&batch).expect("libllama decode");
        ctx.logits_ith(-1).unwrap().to_vec()
    });
    let decode_s = t0.elapsed().as_secs_f64();
    let run = json!({
        "engine": "libllama",
        "ref_ot": patterns,
        "ubatch": ub,
        "load_s": load_s,
        "prefill_tokens": tokens.len(),
        "prefill_s": prefill_s,
        "prefill_tok_s": tokens.len() as f64 / prefill_s,
        "decode_tokens": n_decode,
        "decode_tok_s": n_decode as f64 / decode_s,
        "first_top": top_json(&last),
        "greedy": greedy,
        "margins": margins,
        "rss_bytes": process::current_rss(),
        "peak_rss_bytes": process::peak_rss(),
    });
    eprintln!("{run}");
    run
}

/// `base`, or with several ubatch sizes `stem.ub<size>.ext`.
fn trace_path(base: &std::path::Path, ub: usize, many: bool) -> PathBuf {
    if !many {
        return base.to_path_buf();
    }
    let stem = base
        .file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned());
    let ext = base
        .extension()
        .map_or(String::new(), |e| format!(".{}", e.to_string_lossy()));
    base.with_file_name(format!("{stem}.ub{ub}{ext}"))
}

fn add_profile(a: &mut StepProfile, b: &StepProfile) {
    a.steps += b.steps;
    a.graphs += b.graphs;
    a.inputs += b.inputs;
    a.build += b.build;
    a.alloc += b.alloc;
    a.compute += b.compute;
    a.readback += b.readback;
    a.streamer += b.streamer;
    a.total += b.total;
    a.cpu_ready += b.cpu_ready;
    a.read_wait += b.read_wait;
    a.cpu_landed += b.cpu_landed;
    a.copy += b.copy;
    a.seg_compute_cpu += b.seg_compute_cpu;
    a.seg_compute_gpu += b.seg_compute_gpu;
    a.segs_cpu += b.segs_cpu;
    a.segs_gpu += b.segs_gpu;
    a.cpu_pairs += b.cpu_pairs;
}

fn unix_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
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
        "cpu_ready_ms": ms(p.cpu_ready),
        "read_wait_ms": ms(p.read_wait),
        "cpu_landed_ms": ms(p.cpu_landed),
        "copy_ms": ms(p.copy),
        // `sched.compute` of the segments with a RAM-tier (CPU) group and of those without,
        // and the routed pairs the CPU computed, per `per`.
        "seg_compute_cpu_ms": ms(p.seg_compute_cpu),
        "seg_compute_gpu_ms": ms(p.seg_compute_gpu),
        "segs_cpu": p.segs_cpu as f64 / per,
        "segs_gpu": p.segs_gpu as f64 / per,
        "cpu_pairs": p.cpu_pairs as f64 / per,
    })
}

/// Reads between two counter snapshots, per `per` (tokens) over `wall_s` of wall time.
/// `waited_read_gb_s` is bytes (prefetch included) over the time spent in waited reads only, so it
/// overstates the disk rate; `bytes_per_s` is bytes over wall time.
fn io_json(a: &IoCounters, b: &IoCounters, per: f64, wall_s: f64) -> Value {
    let bytes = b.bytes_read - a.bytes_read;
    let secs = (b.read_time - a.read_time).as_secs_f64();
    json!({
        "batches": b.batches - a.batches,
        "experts_read": b.experts_read - a.experts_read,
        "bytes": bytes,
        "bytes_per_token": bytes as f64 / per,
        "read_s": secs,
        "read_ms_per_token": secs * 1e3 / per,
        "waited_read_gb_s": if secs > 0.0 { bytes as f64 / secs / 1e9 } else { 0.0 },
        "bytes_per_s": if wall_s > 0.0 { bytes as f64 / wall_s } else { 0.0 },
        "uploads": b.uploads - a.uploads,
        "admitted": b.admitted - a.admitted,
        "admitted_bytes": b.admitted_bytes - a.admitted_bytes,
        "admit_ms_per_token": (b.admit_time - a.admit_time).as_secs_f64() * 1e3 / per,
    })
}

/// Overlapped decode between two counter snapshots, per `per` (tokens): layer steps that had
/// experts in flight, how many, the main thread's wait and the hidden share of the read span.
fn overlap_json(a: &IoCounters, b: &IoCounters, per: f64) -> Value {
    let span = (b.overlap_span - a.overlap_span).as_secs_f64();
    let wait = (b.overlap_wait - a.overlap_wait).as_secs_f64();
    let u_span = (b.urgent_steps_span - a.urgent_steps_span).as_secs_f64();
    let u_wait = (b.urgent_steps_wait - a.urgent_steps_wait).as_secs_f64();
    let (p_span, p_wait) = (span - u_span, wait - u_wait);
    let hidden = |w: f64, s: f64| if s > 0.0 { Some(1.0 - w / s) } else { None };
    let pending = b.pending_experts - a.pending_experts;
    let urgent = b.pending_urgent - a.pending_urgent;
    let ahead = b.pending_ahead - a.pending_ahead;
    let prefetch = pending - urgent;
    let lead = (b.pending_lead - a.pending_lead).as_secs_f64();
    // With overlap every miss is an urgent unit; `read_time` sums their latencies.
    let misses =
        (b.experts_read - a.experts_read).saturating_sub(b.prefetch_issued - a.prefetch_issued);
    let read = (b.read_time - a.read_time).as_secs_f64();
    let steps = (b.pending_steps - a.pending_steps) as f64;
    let u_steps = (b.urgent_steps - a.urgent_steps) as f64;
    json!({
        "pending_urgent_per_token": urgent as f64 / per,
        "pending_top_up_per_token": (prefetch - ahead) as f64 / per,
        "pending_ahead_per_token": ahead as f64 / per,
        "pending_prefetch_lead_ms": if prefetch > 0 { Some(lead * 1e3 / prefetch as f64) } else { None },
        "urgent_read_ms": if misses > 0 { Some(read * 1e3 / misses as f64) } else { None },
        "urgent_steps_per_token": u_steps / per,
        "urgent_steps_wait_ms_per_token": u_wait * 1e3 / per,
        "urgent_steps_span_ms_per_token": u_span * 1e3 / per,
        "urgent_steps_hidden_share": hidden(u_wait, u_span),
        "prefetch_steps_per_token": (steps - u_steps) / per,
        "prefetch_steps_wait_ms_per_token": p_wait * 1e3 / per,
        "prefetch_steps_span_ms_per_token": p_span * 1e3 / per,
        "prefetch_steps_hidden_share": hidden(p_wait, p_span),
        "pending_steps_per_token": (b.pending_steps - a.pending_steps) as f64 / per,
        "pending_experts_per_token": (b.pending_experts - a.pending_experts) as f64 / per,
        "read_wait_ms_per_token": wait * 1e3 / per,
        "read_span_ms_per_token": span * 1e3 / per,
        "hidden_share": if span > 0.0 { Some(1.0 - wait / span) } else { None },
        "urgent_wait_ms_per_token": (b.urgent_wait_time - a.urgent_wait_time).as_secs_f64() * 1e3 / per,
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
    // The exit-JSON view of `tetraeder.expert.cache.requests`: compute-tier hits over hits plus
    // `cpu` lookups. A decode step has one token, so its distinct experts are its pairs.
    let hits: u64 = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(x, y)| y.compute.hits - x.compute.hits)
        .sum();
    let cpu: u64 = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(x, y)| y.cpu - x.cpu)
        .sum();
    json!({
        "vram_pair_share": if hits + cpu > 0 { Some(hits as f64 / (hits + cpu) as f64) } else { None },
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
    let n_windows = env_u64("TETRAEDER_EXIT_DECODE_WINDOWS").unwrap_or(1) as usize;
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
    let prompt_info = json!({"tokens": tokens.len(), "fnv1a64": tokens_hash(&tokens)});
    let reference = std::env::var("TETRAEDER_EXIT_ENGINE").is_ok_and(|v| v == "libllama");

    // ADR 0060: the builder's metrics to the box collector when the harness gives an endpoint
    let tel_cfg = TelemetryConfig::from_env();
    let telemetry = tel_cfg.otlp_endpoint.is_some().then(|| {
        let t = Arc::new(Telemetry::new(&tel_cfg).expect("OTLP exporter"));
        t.metrics().cpu_threads(threads as u64, cgroup_cpu_quota());
        t
    });

    let mut opts = LoadOptions::new(if gpu {
        BackendChoice::Gpu { index: 0 }
    } else {
        BackendChoice::Cpu {
            threads: Some(threads),
        }
    });
    opts.n_ctx = n_ctx;
    opts.miss = MissPolicy::Cpu;
    opts.telemetry = telemetry.clone();
    opts.op_offload = env_flag("TETRAEDER_EXIT_OP_OFFLOAD");
    // Unset follows LoadOptions (prefetch on); `0` turns it off, `1` on.
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_PREFETCH") {
        opts.prefetch = v == "1";
    }
    if let Some(d) = env_u64("TETRAEDER_EXIT_PREFETCH_DEPTH") {
        opts.prefetch_depth = d as usize;
    }
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_KV") {
        opts.kv = v
            .parse()
            .expect("TETRAEDER_EXIT_KV is f16, q8_0, q8_0/q4_0 or q4_0");
    }
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_IO_SERVICE") {
        opts.io.service = v == "1";
    }
    if let Some(w) = env_u64("TETRAEDER_EXIT_IO_WORKERS") {
        opts.io.workers = w as usize;
    }
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_OVERLAP") {
        opts.overlap = v == "1";
    }
    opts.urgent_first = env_flag("TETRAEDER_EXIT_URGENT_FIRST");
    opts.ram_hugepages = env_flag("TETRAEDER_EXIT_RAM_HUGEPAGES");
    opts.admit_share = std::env::var("TETRAEDER_EXIT_ADMIT_SHARE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_RAM_POLICY") {
        opts.ram_policy = tetraeder_memory::RamPolicy::parse(&v).unwrap_or_else(|e| panic!("{e}"));
    }
    opts.kv_rotation = env_u64("TETRAEDER_EXIT_KV_ROT").map(|v| v == 1);
    opts.cpu_poll = env_u64("TETRAEDER_EXIT_CPU_POLL").map(|p| p as u32);
    opts.pool_pause = env_flag("TETRAEDER_EXIT_POOL_PAUSE");
    opts.prefetch_threads = env_u64("TETRAEDER_EXIT_PREFETCH_THREADS").map(|t| t as usize);
    let refresh_hot = env_flag("TETRAEDER_EXIT_REFRESH_HOT");
    let routing_out = std::env::var_os("TETRAEDER_EXIT_ROUTING_OUT").map(PathBuf::from);
    opts.record_routing = routing_out.is_some() && !reference;
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
    let mut cpu_poll = None;
    for &ub in &ubatches {
        if reference {
            runs.push(reference_run(
                &path, &tokens, ub, n_decode, n_ctx, threads, opts.kv,
            ));
            continue;
        }
        let t0 = Instant::now();
        let mut m = Qwen3Moe::load(&path, &opts).expect("builder loads");
        let load_s = t0.elapsed().as_secs_f64();
        io_backend = m.io_backend();
        cpu_poll = Some(m.cpu_poll());
        let s0 = m.expert_stats().expect("the exit regime streams experts");
        let mut recorder = routing_out.as_ref().map(|p| {
            let path = trace_path(p, ub, ubatches.len() > 1);
            let hot = if refresh_hot {
                "refresh"
            } else {
                "first_prefill"
            };
            RoutingRecorder::create(
                &path,
                &m,
                ub,
                prompt_info["fnv1a64"].as_str().unwrap_or_default(),
                hot,
                None,
            )
        });

        m.take_profile();
        let t0 = Instant::now();
        let mut ubatch_s = Vec::new();
        let mut last = Vec::new();
        for chunk in tokens.chunks(ub.max(1)) {
            let t = Instant::now();
            last = m.eval_last(chunk).expect("prefill");
            ubatch_s.push(t.elapsed().as_secs_f64());
            if let Some(r) = &mut recorder {
                r.drain(&mut m);
            }
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let pp = m.take_profile();
        // The hot set comes from the first ubatch; optionally rebuild it from the whole prefill.
        let t = Instant::now();
        if refresh_hot {
            m.refresh_hot_set().expect("hot set refresh");
            if let Some(r) = &mut recorder {
                r.drain(&mut m);
            }
        }
        let refresh_s = t.elapsed().as_secs_f64();
        let s1 = m.expert_stats().unwrap();

        let first_top = top_json(&last);
        let mut next = argmax(&last) as i32;
        let mut margins = vec![margin(&last)];
        let mut greedy = Vec::new();
        // Decode in windows (each with its own profile and reads) so a detector run gets repeated
        // samples from one load; the wall-clock bounds let a host sampler align with them.
        let mut windows = Vec::new();
        let mut tg = StepProfile::default();
        let mut sw = s1.clone();
        let t0 = Instant::now();
        let per_window = n_decode.div_ceil(n_windows.max(1)).max(1);
        let mut done = 0;
        while done < n_decode {
            let n = per_window.min(n_decode - done);
            let start = unix_s();
            let tw = Instant::now();
            for _ in 0..n {
                greedy.push(next);
                let l = m.eval(&[next]).expect("decode");
                if let Some(r) = &mut recorder {
                    r.drain(&mut m);
                }
                margins.push(margin(&l));
                next = argmax(&l) as i32;
            }
            let wall = tw.elapsed().as_secs_f64();
            let p = m.take_profile();
            let se = m.expert_stats().unwrap();
            windows.push(json!({
                "start_unix_s": start,
                "end_unix_s": unix_s(),
                "tokens": n,
                "wall_s": wall,
                "profile_per_token": profile_json(&p, n as f64),
                "io": io_json(&sw.io, &se.io, n as f64, wall),
                "layers": layers_json(&sw, &se),
            }));
            add_profile(&mut tg, &p);
            sw = se;
            done += n;
        }
        margins.truncate(n_decode);
        let decode_s = t0.elapsed().as_secs_f64();
        let s2 = m.expert_stats().unwrap();

        if let Some(r) = recorder.take() {
            r.finish();
        }
        let memory: Vec<Value> = m
            .memory()
            .iter()
            .map(|u| {
                json!({"device": u.device, "tier": u.tier, "kind": u.kind,
                       "capacity": u.capacity, "used": u.used})
            })
            .collect();
        let run = json!({
            "engine": "tetraeder",
            "ubatch": ub,
            "load_s": load_s,
            "prefill_tokens": n_prefill,
            "prefill_s": prefill_s,
            "prefill_tok_s": n_prefill as f64 / prefill_s,
            "hot_refresh_s": refresh_s,
            "ubatch_s": ubatch_s,
            "prefill_profile_per_ubatch": profile_json(&pp, pp.steps.max(1) as f64),
            "prefill_io": io_json(&s0.io, &s1.io, n_prefill as f64, prefill_s + refresh_s),
            "prefill_prefetch": prefetch_json(&s0, &s1, n_prefill as f64),
            "decode_tokens": n_decode,
            "decode_tok_s": n_decode as f64 / decode_s,
            "decode_profile_per_token": profile_json(&tg, n_decode.max(1) as f64),
            "decode_io": io_json(&s1.io, &s2.io, n_decode.max(1) as f64, decode_s),
            "decode_prefetch": prefetch_json(&s1, &s2, n_decode.max(1) as f64),
            "decode_overlap": overlap_json(&s1.io, &s2.io, n_decode.max(1) as f64),
            "decode_layers": layers_json(&s1, &s2),
            "first_top": first_top,
            "decode_windows": windows,
            "greedy": greedy,
            "margins": margins,
            "max_splits": m.max_splits(),
            "memory": memory,
            "ram_hugepages": m.ram_hugepages().map(|a| json!({
                "applied": a.applied,
                "advised_bytes": a.bytes,
                "errno": a.errno,
            })),
            "rss_bytes": process::current_rss(),
            "peak_rss_bytes": process::peak_rss(),
        });
        eprintln!("{run}");
        runs.push(run);
    }
    let out = json!({
        "machine": std::env::var("TETRAEDER_EXIT_MACHINE").ok(),
        "provenance": provenance(threads),
        "env": {
            "os": std::env::consts::OS,
            "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
            "gpu_devices_visible": Backend::gpu_count(),
            // the CPU features ggml was compiled with (TETRAEDER_GGML_NATIVE)
            "ggml_system_info": tetraeder_ggml_sys::llama::system_info(),
        },
        "model": model_info,
        "prompt": prompt_info,
        "options": {
            "gpu": gpu,
            "threads": threads,
            "n_ctx": n_ctx,
            "op_offload": opts.op_offload,
            "refresh_hot": refresh_hot,
            "prefetch": opts.prefetch,
            "prefetch_depth": opts.prefetch_depth,
            "io_service": opts.io.service,
            "io_workers": opts.io.workers,
            "io_chunk_bytes": opts.io.chunk_bytes,
            "overlap": opts.overlap,
            "urgent_first": opts.urgent_first,
            "admit_share": opts.admit_share,
            "ram_policy": opts.ram_policy.name(),
            "ram_hugepages": opts.ram_hugepages,
            "kv": opts.kv.to_string(),
            "kv_rotation": opts.kv_rotation.unwrap_or(opts.kv != KvFormat::F16),
            "pool_pause": opts.pool_pause,
            "cpu_poll": cpu_poll.map(|p| p.level),
            "cpu_poll_auto": cpu_poll.map(|p| p.auto),
            "cpu_quota_detected": cpu_poll.and_then(|p| p.quota),
            "prefetch_threads": opts.prefetch_threads,
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
    // the last export before the process ends
    if let Some(t) = &telemetry {
        t.shutdown();
    }
}

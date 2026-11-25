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
//!   `TETRAEDER_EXIT_DECODE_WINDOWS` (1) windows reported one by one (`decode_windows`); each
//!   window's `gap` names its token with the most time outside `run()` and splits it.
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
//! - `TETRAEDER_EXIT_PREFILL_WARM=n` (default 0): the warm protocol (ADR 0092 section 3): `n`
//!   untimed full requests (prefill and decode) of the same prompt, their routing dropped, then
//!   the timed request from a cleared sequence. The harness drives the model directly, so every
//!   prompt token is computed (no prefix reuse, no prompt cache). The result has `warm` (the
//!   requests, whether the plan covers every expert, `timed_prefill_disk_bytes`, `unclean`: a
//!   covering plan whose timed prefill read from disk), `warm_requests` and `warm_prefill` (the
//!   first warm request's prefill).
//! - `TETRAEDER_EXIT_LAYERS=a..b`: run only layers `a` to `b` (half open), with no head: a span
//!   from layer 0 takes tokens, any other takes hidden rows from `TETRAEDER_EXIT_HIDDEN_IN` (a
//!   dump file) or a fixed seed (`TETRAEDER_EXIT_HIDDEN_SEED`, default 20261124: standard
//!   normal rows). `TETRAEDER_EXIT_HEAD=1` keeps the head when the span ends at the last layer
//!   (the two-span parity test). A span cannot pick tokens, so its decode steps use the token
//!   ids of the dump file, else fixed pseudo-random ones. The result's `segment` describes the
//!   span and `decode_layers` covers its layers (`first_layer`, `vram_pair_share_layers`).
//! - `TETRAEDER_EXIT_HIDDEN_OUT=layer,file`: a whole-model greedy run as two chained builders
//!   (layers below `layer` without the head, the rest with it) that writes the hidden rows
//!   entering `layer` for every prompt and decode token, with the token ids, to `file` (the
//!   input of a later span run). Report only: the chain has no counters or profile, and its
//!   routing trace (`TETRAEDER_EXIT_ROUTING_OUT`) is that of the whole model.
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
//! - `TETRAEDER_EXIT_ARMS` (`name:ENV=V,ENV=V;name2:...`): multi-arm mode, the test
//!   `exit_regime_arms` (run it by name): one prefill, a KV snapshot, then the arms interleaved
//!   for `TETRAEDER_EXIT_ARM_ROUNDS` rounds. Its docs list the other `TETRAEDER_EXIT_ARM*` and
//!   `TETRAEDER_EXIT_SNAP_*` variables.
//!
//! Every run records `prompt` (token count and an FNV-1a hash of the token ids), `first_top`
//! (the 10 largest logits after the prefill, `[id, logit]`) and `margins` (top-1 minus top-2
//! logit at each greedy step), so two runs can be shown to have had the same input and a
//! differing greedy token can be told apart from a near-tie.

mod common;

use common::exit_segment::{
    HiddenFile, HiddenIn, SegmentArgs, Workload, covers, fnv1a_bytes, resolve_inputs, run_protocol,
    warm_requests, warm_summary,
};
use common::synth::{MoeSpec, quantized_moe_at};
use common::trace::RoutingRecorder;
use common::{argmax, prefetch_json, top_k};
use serde_json::{Value, json};
use std::ops::Range;
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
    BackendChoice, ExpertStats, Hparams, IoCounters, LoadOptions, MissPolicy, ModelFile, Qwen3Moe,
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

/// Environment lookups with an overlay: a multi-arm run applies an arm's `ENV=V` pairs over the
/// process environment without touching it.
#[derive(Clone, Default)]
struct Env(std::collections::BTreeMap<String, String>);

impl Env {
    fn get(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned().or_else(|| std::env::var(key).ok())
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.parse().ok())
    }

    fn flag(&self, key: &str) -> bool {
        self.get(key).is_some_and(|v| v == "1")
    }
}

/// The builder's load options from the `TETRAEDER_EXIT_*` variables in `e` (the module docs list
/// them); the caller adds telemetry and routing recording.
fn load_options(e: &Env, gpu: bool, threads: usize, n_ctx: usize, n_expert: u64) -> LoadOptions {
    let mut opts = LoadOptions::new(if gpu {
        BackendChoice::Gpu { index: 0 }
    } else {
        BackendChoice::Cpu {
            threads: Some(threads),
        }
    });
    opts.n_ctx = n_ctx;
    opts.miss = MissPolicy::Cpu;
    opts.op_offload = e.flag("TETRAEDER_EXIT_OP_OFFLOAD");
    // Unset follows LoadOptions (prefetch on); `0` turns it off, `1` on.
    if let Some(v) = e.get("TETRAEDER_EXIT_PREFETCH") {
        opts.prefetch = v == "1";
    }
    if let Some(d) = e.u64("TETRAEDER_EXIT_PREFETCH_DEPTH") {
        opts.prefetch_depth = d as usize;
    }
    if let Some(v) = e.get("TETRAEDER_EXIT_KV") {
        opts.kv = v
            .parse()
            .expect("TETRAEDER_EXIT_KV is f16, q8_0, q8_0/q4_0 or q4_0");
    }
    if let Some(v) = e.get("TETRAEDER_EXIT_IO_SERVICE") {
        opts.io.service = v == "1";
    }
    if let Some(w) = e.u64("TETRAEDER_EXIT_IO_WORKERS") {
        opts.io.workers = w as usize;
    }
    if let Some(v) = e.get("TETRAEDER_EXIT_OVERLAP") {
        opts.overlap = v == "1";
    }
    opts.urgent_first = e.flag("TETRAEDER_EXIT_URGENT_FIRST");
    opts.ram_hugepages = e.flag("TETRAEDER_EXIT_RAM_HUGEPAGES");
    opts.admit_share = e
        .get("TETRAEDER_EXIT_ADMIT_SHARE")
        .and_then(|v| v.parse::<f64>().ok());
    if let Some(v) = e.get("TETRAEDER_EXIT_RAM_POLICY") {
        opts.ram_policy =
            tetraeder_memory::RamPolicy::parse(&v).unwrap_or_else(|err| panic!("{err}"));
    }
    opts.kv_rotation = e.u64("TETRAEDER_EXIT_KV_ROT").map(|v| v == 1);
    opts.cpu_poll = e.u64("TETRAEDER_EXIT_CPU_POLL").map(|p| p as u32);
    opts.pool_pause = e.flag("TETRAEDER_EXIT_POOL_PAUSE");
    opts.prefetch_threads = e.u64("TETRAEDER_EXIT_PREFETCH_THREADS").map(|t| t as usize);
    match e.u64("TETRAEDER_EXIT_VRAM_SLOT_BYTES") {
        Some(b) => opts.expert_slot_bytes = Some(b),
        None => opts.expert_slots = Some(e.u64("TETRAEDER_EXIT_SLOTS").unwrap_or(2) as usize),
    }
    match e.u64("TETRAEDER_EXIT_RAM_SLOT_BYTES") {
        Some(b) => opts.ram_slot_bytes = Some(b),
        None => {
            let n = e.u64("TETRAEDER_EXIT_RAM_SLOTS").unwrap_or(n_expert / 4);
            opts.ram_slots = Some(n as usize);
        }
    }
    opts
}

/// The `options` block of a result JSON.
fn options_json(
    gpu: bool,
    threads: usize,
    n_ctx: usize,
    opts: &LoadOptions,
    refresh_hot: bool,
    cpu_poll: Option<tetraeder_model::CpuPoll>,
) -> Value {
    json!({
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
    })
}

/// ADR 0060: the builder's metrics to the box collector when the harness gives an endpoint.
fn telemetry_for(threads: usize) -> Option<Arc<Telemetry>> {
    let cfg = TelemetryConfig::from_env();
    cfg.otlp_endpoint.is_some().then(|| {
        let t = Arc::new(Telemetry::new(&cfg).expect("OTLP exporter"));
        t.metrics().cpu_threads(threads as u64, cgroup_cpu_quota());
        t
    })
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
    // one pass, no allocation: it runs per decode token inside the measured time
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in logits {
        if x.total_cmp(&a).is_gt() {
            b = a;
            a = x;
        } else if x.total_cmp(&b).is_gt() {
            b = x;
        }
    }
    a - b
}

/// The calling thread's counters from `/proc/thread-self/stat` (Linux; `None` elsewhere).
#[derive(Clone, Copy, Default)]
struct ThreadStat {
    minflt: u64,
    majflt: u64,
    /// User and system CPU time in clock ticks (`USER_HZ`, 100 on Linux).
    utime: u64,
    stime: u64,
}

impl ThreadStat {
    fn now() -> Option<Self> {
        let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
        // Fields after the parenthesised name: state is field 3, minflt 10, majflt 12,
        // utime 14, stime 15.
        let f: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        let n = |i: usize| f.get(i)?.parse().ok();
        Some(Self {
            minflt: n(7)?,
            majflt: n(9)?,
            utime: n(11)?,
            stime: n(12)?,
        })
    }

    fn since(self, a: Self) -> Self {
        Self {
            minflt: self.minflt - a.minflt,
            majflt: self.majflt - a.majflt,
            utime: self.utime - a.utime,
            stime: self.stime - a.stime,
        }
    }

    fn json(self) -> Value {
        json!({"minflt": self.minflt, "majflt": self.majflt,
               "user_ms": self.utime * 10, "sys_ms": self.stime * 10})
    }
}

/// One part of a token outside `run()`: its wall time and the thread's counters over it. A
/// wall time far above the CPU time means the thread was blocked; close to `sys_ms`, the kernel
/// worked for it (faults, reclaim).
#[derive(Clone, Copy, Default)]
struct GapPart {
    wall: Duration,
    stat: Option<ThreadStat>,
}

impl GapPart {
    fn json(self) -> Value {
        json!({"ms": self.wall.as_secs_f64() * 1e3, "thread": self.stat.map(ThreadStat::json)})
    }
}

/// Per decode token, the wall time outside `run()` (`StepProfile::total`), split into the part
/// inside `eval` (step bookkeeping, telemetry, drops; its thread counters cover the whole
/// `eval`) and the harness's own (margin, argmax, routing drain): a window's `gap` block says
/// which token stalled and where.
#[derive(Default)]
struct GapTrace {
    sum: Duration,
    max: Duration,
    max_token: usize,
    in_eval: GapPart,
    harness: GapPart,
}

impl GapTrace {
    fn token(&mut self, i: usize, in_eval: GapPart, harness: GapPart) {
        let gap = in_eval.wall + harness.wall;
        self.sum += gap;
        if gap >= self.max {
            (self.max, self.max_token) = (gap, i);
            (self.in_eval, self.harness) = (in_eval, harness);
        }
    }

    fn json(&self) -> Value {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        json!({
            "sum_ms": ms(self.sum),
            "max_ms": ms(self.max),
            "max_token": self.max_token,
            "max_in_eval": self.in_eval.json(),
            "max_harness": self.harness.json(),
        })
    }
}

/// Decodes `token` as the window's `i`-th; `harness` gets the logits (and drops them).
fn traced_token(
    m: &mut Qwen3Moe,
    gaps: &mut GapTrace,
    i: usize,
    token: i32,
    harness: impl FnOnce(&mut Qwen3Moe, Vec<f32>),
) {
    let since = |a: Option<ThreadStat>, b: Option<ThreadStat>| a.zip(b).map(|(a, b)| b.since(a));
    let s0 = ThreadStat::now();
    let before = m.profile().total;
    let t = Instant::now();
    let l = m.eval(&[token]).expect("decode");
    let eval = t.elapsed();
    let in_run = m.profile().total.saturating_sub(before);
    let s1 = ThreadStat::now();
    let t = Instant::now();
    harness(m, l);
    let wall = t.elapsed();
    let s2 = ThreadStat::now();
    let in_eval = GapPart {
        wall: eval.saturating_sub(in_run),
        stat: since(s0, s1),
    };
    let harness = GapPart {
        wall,
        stat: since(s1, s2),
    };
    gaps.token(i, in_eval, harness);
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
    a.seg_pairs += b.seg_pairs;
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
        "seg_pairs": p.seg_pairs as f64 / per,
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

/// Per layer of `span`, the share of decode layer steps with every routed expert in the compute
/// tier, and the RAM-tier hit rate. The counter tables cover every layer of the model (zeros
/// outside a segment), so the arrays and means cover the span only; `first_layer` says where it
/// starts. `vram_pair_share_layers` is the per-layer form of `vram_pair_share` (compute-tier
/// hits over hits plus `cpu` lookups; null where the layer saw no pair) and `pairs_layers` its
/// weights: the aggregate is the pair-weighted mean of the layers.
fn layers_json(a: &ExpertStats, b: &ExpertStats, span: Range<usize>) -> Value {
    let mut full = Vec::new();
    let mut ram_hit = Vec::new();
    let mut share = Vec::new();
    let mut pairs = Vec::new();
    let (mut hits, mut cpu) = (0u64, 0u64);
    for (la, lb) in a
        .layers
        .iter()
        .zip(&b.layers)
        .skip(span.start)
        .take(span.len())
    {
        let f = lb.full_steps - la.full_steps;
        let p = lb.partial_steps - la.partial_steps;
        full.push(f as f64 / (f + p).max(1) as f64);
        if let (Some(ra), Some(rb)) = (la.ram, lb.ram) {
            let h = rb.hits - ra.hits;
            let m = rb.misses - ra.misses;
            ram_hit.push(h as f64 / (h + m).max(1) as f64);
        }
        // The exit-JSON view of `tetraeder.expert.cache.requests`. A decode step has one token,
        // so its distinct experts are its pairs.
        let (h, c) = (lb.compute.hits - la.compute.hits, lb.cpu - la.cpu);
        hits += h;
        cpu += c;
        pairs.push(h + c);
        share.push(if h + c > 0 {
            Some(h as f64 / (h + c) as f64)
        } else {
            None
        });
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({
        "vram_pair_share": if hits + cpu > 0 { Some(hits as f64 / (hits + cpu) as f64) } else { None },
        "first_layer": span.start,
        "vram_pair_share_layers": share,
        "pairs_layers": pairs,
        "full_hit_rate_mean": mean(&full),
        "ram_hit_rate_mean": mean(&ram_hit),
        "full_hit_rate": full,
        "ram_hit_rate": ram_hit,
    })
}

/// What one segment run needs (the flags of ADR 0092 steps E2 and E3).
struct SegmentCtx<'a> {
    path: &'a PathBuf,
    h: &'a Hparams,
    args: &'a SegmentArgs,
    /// The base load options; every segment loads them with its own `segment`.
    opts: &'a LoadOptions,
    tokens: &'a [i32],
    prompt_hash: &'a str,
    routing_out: Option<PathBuf>,
    ub: usize,
    n_decode: usize,
    n_warm: usize,
}

/// A run of a layer span (`TETRAEDER_EXIT_LAYERS`), or of the whole model as two chained
/// segments that dump the hidden rows at a layer boundary (`TETRAEDER_EXIT_HIDDEN_OUT`): the
/// warm protocol, then the timed request. The result has the fields of a whole-model run that a
/// segment can fill (a chain has no counters or profile: its two builders each hold half of
/// them), and `segment` describes the span. The dump is written after the timed request.
fn segment_run(c: &SegmentCtx<'_>) -> Value {
    let h = c.h;
    let (n_embd, n_vocab) = (h.n_embd as usize, h.n_vocab as usize);
    let specs = c.args.specs(h.n_layer);
    let inputs = resolve_inputs(c.args, &specs[0], n_embd, n_vocab, c.tokens, c.n_decode)
        .unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let mut models: Vec<Qwen3Moe> = specs
        .iter()
        .map(|s| {
            let mut o = c.opts.clone();
            o.segment = Some(s.clone());
            Qwen3Moe::load(c.path, &o).expect("segment loads")
        })
        .collect();
    let load_s = t0.elapsed().as_secs_f64();
    let mut recorder = c.routing_out.as_ref().map(|p| {
        RoutingRecorder::create(p, &models[0], c.ub, c.prompt_hash, "first_prefill", None)
    });
    let w = Workload {
        tokens: c.tokens,
        ub: c.ub,
        n_decode: c.n_decode,
        decode_tokens: inputs.decode_tokens.as_deref(),
        rows: inputs.rows.as_deref(),
    };
    let mut log = Vec::new();
    let mut refs: Vec<&mut Qwen3Moe> = models.iter_mut().collect();
    let covering = covers(refs[0].tier_slots(), h.n_expert as usize);
    let proto = run_protocol(
        &mut refs,
        &w,
        c.n_warm,
        c.args.tap_at(),
        &mut log,
        &mut |ms: &mut [&mut Qwen3Moe]| {
            if let Some(r) = recorder.as_mut() {
                r.drain_chain(ms);
            }
        },
    );
    // wall clock of the end of the timed decode, for the host sampler's windows (`decode_windows`)
    let ended_unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    drop(refs);
    if let Some(r) = recorder.take() {
        r.finish();
    }
    let mut t = proto.timed;
    let (n_prefill, n_decode) = (c.tokens.len(), c.n_decode);

    let hidden_out = c.args.hidden_out.as_ref().map(|o| {
        let mut tokens = c.tokens.to_vec();
        tokens.extend(&t.decode_fed);
        let f = HiddenFile {
            layer: o.layer,
            n_embd,
            n_prompt: n_prefill,
            tokens,
            rows: std::mem::take(&mut t.tap),
        };
        assert_eq!(
            f.rows.len(),
            f.tokens.len() * n_embd,
            "one hidden row per token"
        );
        let bytes = f.to_bytes();
        std::fs::write(&o.file, &bytes).expect("hidden-row file");
        json!({"layer": o.layer, "file": o.file.display().to_string(),
               "bytes": bytes.len(), "fnv1a64": fnv1a_bytes(&bytes), "rows": f.tokens.len()})
    });
    let hidden_in = c.args.hidden_in.as_ref().map(|i| match i {
        HiddenIn::File(p) => json!({"file": p.display().to_string()}),
        HiddenIn::Seed(s) => json!({"seed": s, "distribution": "standard normal"}),
    });
    let first = &specs[0];
    let span = first.layers.start as usize..specs[specs.len() - 1].layers.end as usize;
    let (prefill_io, decode_io, decode_layers) = match &t.snaps {
        Some(s) => (
            io_json(
                &s.start.io,
                &s.after_prefill.io,
                n_prefill as f64,
                t.prefill_s,
            ),
            io_json(
                &s.after_prefill.io,
                &s.end.io,
                n_decode.max(1) as f64,
                t.decode_s,
            ),
            layers_json(&s.after_prefill, &s.end, span.clone()),
        ),
        None => (Value::Null, Value::Null, Value::Null),
    };
    let warm: Vec<Value> = proto
        .warm
        .iter()
        .map(|r| {
            json!({
                "prefill_s": r.prefill_s,
                "prefill_tok_s": n_prefill as f64 / r.prefill_s,
                "decode_s": r.decode_s,
                "io": r.snaps.as_ref().map(|s| io_json(&s.start.io, &s.after_prefill.io, n_prefill as f64, r.prefill_s)),
            })
        })
        .collect();
    let memory: Vec<Value> = models
        .iter()
        .flat_map(|m| m.memory())
        .map(|u| {
            json!({"device": u.device, "tier": u.tier, "kind": u.kind,
                   "capacity": u.capacity, "used": u.used})
        })
        .collect();
    let run = json!({
        "engine": "tetraeder",
        "ubatch": c.ub,
        "load_s": load_s,
        "segment": {
            "layers": [span.start, span.end],
            "segments": specs.iter().map(|s| json!({
                "layers": [s.layers.start, s.layers.end], "embed": s.embed, "head": s.head,
            })).collect::<Vec<_>>(),
            "hidden_in": hidden_in,
            "hidden_out": hidden_out,
        },
        "warm_prefill": warm.first().cloned().unwrap_or(Value::Null),
        "warm_requests": warm,
        "warm": warm_summary(c.n_warm, covering, t.prefill_disk_bytes()),
        "prefill_tokens": n_prefill,
        "prefill_s": t.prefill_s,
        "prefill_tok_s": n_prefill as f64 / t.prefill_s,
        "ubatch_s": t.ubatch_s,
        "prefill_profile_per_ubatch": t.prefill_profile.as_ref().map(|p| profile_json(p, p.steps.max(1) as f64)),
        "prefill_io": prefill_io,
        "decode_tokens": n_decode,
        "decode_tok_s": n_decode as f64 / t.decode_s,
        "decode_profile_per_token": t.decode_profile.as_ref().map(|p| profile_json(p, n_decode.max(1) as f64)),
        "decode_io": decode_io,
        "decode_layers": decode_layers,
        "decode_windows": [{"start_unix_s": ended_unix_s - t.decode_s, "end_unix_s": ended_unix_s, "tokens": n_decode}],
        "hot_refresh_s": 0.0,
        // the tokens the decode steps were fed: greedy with the head on a whole-model run
        "decode_fed": t.decode_fed,
        "first_top": specs[specs.len() - 1].head.then(|| top_json(&t.last)),
        "greedy": if first.embed && specs[specs.len() - 1].head { json!(t.decode_fed) } else { Value::Null },
        "request_log": log,
        "max_splits": models[0].max_splits(),
        "memory": memory,
        "rss_bytes": process::current_rss(),
        "peak_rss_bytes": process::peak_rss(),
    });
    eprintln!("{run}");
    run
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
    let telemetry = telemetry_for(threads);

    let mut opts = load_options(&Env::default(), gpu, threads, n_ctx, h.n_expert as u64);
    opts.telemetry = telemetry.clone();
    let refresh_hot = env_flag("TETRAEDER_EXIT_REFRESH_HOT");
    let routing_out = std::env::var_os("TETRAEDER_EXIT_ROUTING_OUT").map(PathBuf::from);
    opts.record_routing = routing_out.is_some() && !reference;
    let per_layer = file.expert_bytes_per_layer().expect("expert sizes");
    let slot_bytes_all_layers: u64 = per_layer.iter().sum();

    let n_warm = env_u64("TETRAEDER_EXIT_PREFILL_WARM").unwrap_or(0) as usize;
    let seg_args =
        SegmentArgs::parse(|k| std::env::var(k).ok(), h.n_layer).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !(reference && seg_args.is_segment_run()),
        "TETRAEDER_EXIT_LAYERS and _HIDDEN_OUT run the builder, not the reference"
    );
    assert!(
        !(routing_out.is_some() && seg_args.layers.is_some()),
        "a routing trace needs a whole-model run or a hidden-row dump, not a span"
    );
    let mut runs = Vec::new();
    let mut io_backend = None;
    let mut cpu_poll = None;
    for &ub in &ubatches {
        if seg_args.is_segment_run() {
            let ctx = SegmentCtx {
                path: &path,
                h: &h,
                args: &seg_args,
                opts: &opts,
                tokens: &tokens,
                prompt_hash: prompt_info["fnv1a64"].as_str().unwrap_or_default(),
                routing_out: routing_out
                    .as_ref()
                    .map(|p| trace_path(p, ub, ubatches.len() > 1)),
                ub,
                n_decode,
                n_warm,
            };
            runs.push(segment_run(&ctx));
            continue;
        }
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
        let mut s0 = m.expert_stats().expect("the exit regime streams experts");
        // TETRAEDER_EXIT_PREFILL_WARM=n: n untimed full requests of the same prompt first (ADR
        // 0092 section 3), then the timed one from an empty sequence, so a covering tier is full
        // and the timed prefill reads nothing from disk. `warm_prefill` is the first request's
        // prefill (read by `sp_table.py`); `warm` has the protocol and the clean flag.
        let mut warm_prefill = Value::Null;
        let mut warm_requests_json = Vec::new();
        if n_warm > 0 {
            let w = Workload {
                tokens: &tokens,
                ub,
                n_decode,
                decode_tokens: None,
                rows: None,
            };
            let warm = warm_requests(&mut [&mut m], &w, n_warm, &mut Vec::new());
            for (k, r) in warm.iter().enumerate() {
                let snaps = r.snaps.as_ref().expect("single model has counters");
                let v = json!({
                    "prefill_s": r.prefill_s,
                    "prefill_tok_s": n_prefill as f64 / r.prefill_s,
                    "decode_s": r.decode_s,
                    "io": io_json(&snaps.start.io, &snaps.after_prefill.io, n_prefill as f64, r.prefill_s),
                });
                if k == 0 {
                    warm_prefill = v.clone();
                }
                warm_requests_json.push(v);
            }
            m.reset();
            s0 = m.expert_stats().unwrap();
        }
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
            let mut gaps = GapTrace::default();
            for i in 0..n {
                greedy.push(next);
                traced_token(&mut m, &mut gaps, i, next, |m, l| {
                    if let Some(r) = &mut recorder {
                        r.drain(m);
                    }
                    margins.push(margin(&l));
                    next = argmax(&l) as i32;
                });
            }
            let wall = tw.elapsed().as_secs_f64();
            let p = m.take_profile();
            let se = m.expert_stats().unwrap();
            windows.push(json!({
                "start_unix_s": start,
                "end_unix_s": unix_s(),
                "tokens": n,
                "wall_s": wall,
                "gap": gaps.json(),
                "profile_per_token": profile_json(&p, n as f64),
                "io": io_json(&sw.io, &se.io, n as f64, wall),
                "layers": layers_json(&sw, &se, 0..h.n_layer as usize),
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
        let prefill_io = io_json(&s0.io, &s1.io, n_prefill as f64, prefill_s + refresh_s);
        let run = json!({
            "engine": "tetraeder",
            "ubatch": ub,
            "load_s": load_s,
            "warm_prefill": warm_prefill,
            "warm_requests": warm_requests_json,
            "warm": warm_summary(
                n_warm,
                covers(m.tier_slots(), h.n_expert as usize),
                prefill_io["bytes"].as_u64().unwrap_or(0),
            ),
            "prefill_tokens": n_prefill,
            "prefill_s": prefill_s,
            "prefill_tok_s": n_prefill as f64 / prefill_s,
            "hot_refresh_s": refresh_s,
            "ubatch_s": ubatch_s,
            "prefill_profile_per_ubatch": profile_json(&pp, pp.steps.max(1) as f64),
            "prefill_io": prefill_io,
            "prefill_prefetch": prefetch_json(&s0, &s1, n_prefill as f64),
            "decode_tokens": n_decode,
            "decode_tok_s": n_decode as f64 / decode_s,
            "decode_profile_per_token": profile_json(&tg, n_decode.max(1) as f64),
            "decode_io": io_json(&s1.io, &s2.io, n_decode.max(1) as f64, decode_s),
            "decode_prefetch": prefetch_json(&s1, &s2, n_decode.max(1) as f64),
            "decode_overlap": overlap_json(&s1.io, &s2.io, n_decode.max(1) as f64),
            "decode_layers": layers_json(&s1, &s2, 0..h.n_layer as usize),
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
        "options": options_json(gpu, threads, n_ctx, &opts, refresh_hot, cpu_poll),
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

/// One arm of the multi-arm run: a name and the `ENV=V` pairs laid over the process environment.
struct Arm {
    name: String,
    env: Env,
}

/// `name:ENV=V,ENV=V;name2:...` (a bare `name` is an arm with no overrides).
fn parse_arms(s: &str) -> Vec<Arm> {
    let arms: Vec<Arm> = s
        .split(';')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(|a| {
            let (name, rest) = a.split_once(':').unwrap_or((a, ""));
            let mut env = Env::default();
            for kv in rest.split(',').filter(|p| !p.trim().is_empty()) {
                let (k, v) = kv.split_once('=').expect("an arm's env is ENV=V");
                env.0.insert(k.trim().into(), v.trim().into());
            }
            Arm {
                name: name.trim().into(),
                env,
            }
        })
        .collect();
    for (i, a) in arms.iter().enumerate() {
        assert!(
            !a.name.is_empty() && arms[..i].iter().all(|b| b.name != a.name),
            "arm names must be non-empty and unique: {:?}",
            a.name
        );
        assert!(
            !a.env.0.contains_key("TETRAEDER_EXIT_REFRESH_HOT")
                && !a.env.0.contains_key("TETRAEDER_EXIT_ROUTING_OUT"),
            "arm {}: the hot set refresh and the routing trace need the prefill, not a snapshot",
            a.name
        );
    }
    arms
}

/// The prefill's KV: in RAM when small, else a file (a 32K context of the 235B is about 6 GB,
/// which the RAM tier's reserve cannot carry for the whole run).
enum Snapshot {
    Ram(Vec<u8>),
    Disk(PathBuf, usize),
}

impl Snapshot {
    fn keep(bytes: Vec<u8>, ram_max: usize, dir: &std::path::Path, stamp: &str) -> Snapshot {
        if bytes.len() <= ram_max {
            return Snapshot::Ram(bytes);
        }
        std::fs::create_dir_all(dir).expect("snapshot dir");
        let p = dir.join(format!("kv-snapshot-{stamp}.bin"));
        std::fs::write(&p, &bytes).expect("snapshot file");
        Snapshot::Disk(p, bytes.len())
    }

    fn len(&self) -> usize {
        match self {
            Snapshot::Ram(b) => b.len(),
            Snapshot::Disk(_, n) => *n,
        }
    }

    fn bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Snapshot::Ram(b) => b.as_slice().into(),
            Snapshot::Disk(p, _) => std::fs::read(p).expect("snapshot file").into(),
        }
    }

    fn storage(&self) -> &'static str {
        match self {
            Snapshot::Ram(_) => "ram",
            Snapshot::Disk(..) => "disk",
        }
    }
}

/// hardware: cpu (default) or gpu_* (`TETRAEDER_EXIT_GPU=1`). Skips without `TETRAEDER_EXIT_ARMS`.
///
/// Multi-arm mode: the arms of a pairing in one process, so they sit minutes apart on the same
/// host instead of an hour. The prompt is prefilled once with the base options (the process
/// environment), the KV is saved, then for `TETRAEDER_EXIT_ARM_ROUNDS` rounds (default 1; odd
/// rounds in arm order, even rounds reversed, so a drift does not favour one arm) each arm
///   1. drops the model and loads it with the arm's options (every `LoadOptions` field is fixed at
///      load; nothing is switchable at runtime, so every arm reloads, alike),
///   2. runs the prompt's first ubatch (the hot set is built from the first prefill call, as in a
///      normal run), restores the KV snapshot (the prefill is not repeated); with
///      `TETRAEDER_EXIT_ARM_REPLAY_LAST=1` it then recomputes the last ubatch on top of the
///      restored rows, which refills the RAM tier like the prefill's end did,
///   3. decodes a discarded warm-up window of `TETRAEDER_EXIT_ARM_WARMUP` (16) tokens and
///   4. times `TETRAEDER_EXIT_DECODE` tokens in `TETRAEDER_EXIT_DECODE_WINDOWS` windows.
///
/// `TETRAEDER_EXIT_ARMS` is `name:ENV=V,ENV=V;name2:...` (`TETRAEDER_EXIT_REFRESH_HOT` and
/// `TETRAEDER_EXIT_ROUTING_OUT` are rejected: both need the prefill itself). Output, in
/// `TETRAEDER_EXIT_ARMS_DIR` (default `.`) with the stamp `TETRAEDER_EXIT_ARMS_STAMP`: one
/// `exit-<arm>-r<k>-<stamp>.json` per arm and round in the format of `exit_regime`, written as
/// each finishes, and `exit-<TETRAEDER_EXIT_ARMS_INDEX, default levers-snap>-<stamp>.json` with
/// `ok`, the shared prefill and snapshot numbers and, per arm and round, whether its greedy
/// tokens (warm-up and timed) equal the first run's. In an arm's JSON `prefill_s` is the arm's
/// preparation (load, first ubatch, restore, warm-up; `prefill_tokens` counts the tokens run in
/// it), so the host sampler's "prefill" block describes that; the real prefill is in
/// `snapshot.shared.prefill_*`. `TETRAEDER_EXIT_SNAP_DIR` (default `.`) and
/// `TETRAEDER_EXIT_SNAP_RAM_MAX_GB` (2) decide where the snapshot lives.
/// `TETRAEDER_EXIT_ARMS_EXACT=1` fails the test when greedy tokens differ between arms.
#[test]
#[ignore = "slow exit-regime benchmark; run with --ignored in release"]
fn exit_regime_arms() {
    let Ok(arms_s) = std::env::var("TETRAEDER_EXIT_ARMS") else {
        eprintln!("skip: TETRAEDER_EXIT_ARMS is not set");
        return;
    };
    let gpu = env_flag("TETRAEDER_EXIT_GPU");
    if gpu && !Backend::gpu_available() {
        eprintln!("skip: TETRAEDER_EXIT_GPU=1 but no GPU in this build");
        return;
    }
    let arms = parse_arms(&arms_s);
    assert!(!arms.is_empty(), "TETRAEDER_EXIT_ARMS names no arm");
    let rounds = env_u64("TETRAEDER_EXIT_ARM_ROUNDS").unwrap_or(1).max(1) as usize;
    let warmup = env_u64("TETRAEDER_EXIT_ARM_WARMUP").unwrap_or(16) as usize;
    let replay_last = env_flag("TETRAEDER_EXIT_ARM_REPLAY_LAST");
    let exact = env_flag("TETRAEDER_EXIT_ARMS_EXACT");
    let out_dir =
        PathBuf::from(std::env::var("TETRAEDER_EXIT_ARMS_DIR").unwrap_or_else(|_| ".".into()));
    let snap_dir =
        PathBuf::from(std::env::var("TETRAEDER_EXIT_SNAP_DIR").unwrap_or_else(|_| ".".into()));
    let index_name =
        std::env::var("TETRAEDER_EXIT_ARMS_INDEX").unwrap_or_else(|_| "levers-snap".into());
    let stamp = std::env::var("TETRAEDER_EXIT_ARMS_STAMP")
        .unwrap_or_else(|_| format!("{}", unix_s() as u64));
    let ram_max = (env_u64("TETRAEDER_EXIT_SNAP_RAM_MAX_GB").unwrap_or(2) << 30) as usize;

    let (path, model_info) = model_path();
    let file = ModelFile::open(&path).expect("model opens");
    let h = file.hparams.clone();
    let n_prefill = env_u64("TETRAEDER_EXIT_PREFILL").unwrap_or(256) as usize;
    let n_decode = env_u64("TETRAEDER_EXIT_DECODE").unwrap_or(32) as usize;
    let n_windows = env_u64("TETRAEDER_EXIT_DECODE_WINDOWS").unwrap_or(1) as usize;
    let ub = std::env::var("TETRAEDER_EXIT_UBATCH")
        .ok()
        .and_then(|s| s.split(',').next().and_then(|x| x.trim().parse().ok()))
        .unwrap_or(256usize)
        .max(1);
    let n_ctx = env_u64("TETRAEDER_EXIT_CTX")
        .map(|c| c as usize)
        .unwrap_or(n_prefill + warmup + n_decode);
    assert!(n_ctx >= n_prefill + warmup + n_decode, "context too small");
    let threads_of = |e: &Env| {
        e.u64("TETRAEDER_EXIT_THREADS")
            .map(|t| t as usize)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
    };
    let tokens = prompt(&path, n_prefill, h.n_vocab as usize);
    let prompt_info = json!({"tokens": tokens.len(), "fnv1a64": tokens_hash(&tokens)});
    let base_threads = threads_of(&Env::default());
    let telemetry = telemetry_for(base_threads);
    let n_expert = h.n_expert as u64;
    let per_layer = file.expert_bytes_per_layer().expect("expert sizes");
    let slot_bytes_all_layers: u64 = per_layer.iter().sum();

    // The one prefill, with the base options; its KV is the snapshot every arm restores.
    let mut opts = load_options(&Env::default(), gpu, base_threads, n_ctx, n_expert);
    opts.telemetry = telemetry.clone();
    let t0 = Instant::now();
    let mut m = Qwen3Moe::load(&path, &opts).expect("builder loads");
    let base_load_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut ubatch_s = Vec::new();
    let mut last = Vec::new();
    for chunk in tokens.chunks(ub) {
        let t = Instant::now();
        last = m.eval_last(chunk).expect("prefill");
        ubatch_s.push(t.elapsed().as_secs_f64());
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let first_top = top_json(&last);
    let first_next = argmax(&last) as i32;
    let t = Instant::now();
    let snapshot = Snapshot::keep(
        m.kv_snapshot().expect("kv snapshot"),
        ram_max,
        &snap_dir,
        &stamp,
    );
    let snapshot_s = t.elapsed().as_secs_f64();
    drop(m);
    let snapshot_json = json!({
        "bytes": snapshot.len(),
        "storage": snapshot.storage(),
        "positions": n_prefill,
        "save_s": snapshot_s,
        "prefill_s": prefill_s,
        "prefill_tok_s": n_prefill as f64 / prefill_s,
        "prefill_ubatch_s": ubatch_s,
        "base_load_s": base_load_s,
        "replay_last_ubatch": replay_last,
    });
    eprintln!("snapshot: {snapshot_json}");

    // (arm, round, greedy incl. warm-up, result file)
    let mut results: Vec<(String, usize, Vec<i32>, String)> = Vec::new();
    for round in 1..=rounds {
        let order: Vec<&Arm> = if round % 2 == 1 {
            arms.iter().collect()
        } else {
            arms.iter().rev().collect()
        };
        for arm in order {
            let threads = threads_of(&arm.env);
            let mut opts = load_options(&arm.env, gpu, threads, n_ctx, n_expert);
            opts.telemetry = telemetry.clone();
            let t_arm = Instant::now();
            let t0 = Instant::now();
            let mut m = Qwen3Moe::load(&path, &opts).expect("builder loads");
            let load_s = t0.elapsed().as_secs_f64();
            let io_backend = m.io_backend();
            let cpu_poll = Some(m.cpu_poll());
            let s0 = m.expert_stats().expect("the exit regime streams experts");
            m.take_profile();
            // the hot set comes from the first prefill call, as in a normal run
            let t = Instant::now();
            m.eval_last(&tokens[..ub.min(tokens.len())])
                .expect("first ubatch");
            let hot_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            m.kv_restore(&snapshot.bytes(), n_prefill)
                .expect("kv restore");
            let restore_s = t.elapsed().as_secs_f64();
            let mut replay_s = 0.0;
            if replay_last {
                let start = (n_prefill - 1) / ub * ub;
                let t = Instant::now();
                m.rewind(start).expect("rewind");
                let l = m.eval_last(&tokens[start..]).expect("replayed ubatch");
                replay_s = t.elapsed().as_secs_f64();
                assert_eq!(
                    argmax(&l),
                    argmax(&last),
                    "the replayed ubatch changed the top token"
                );
            }
            // discarded warm-up window
            let mut next = first_next;
            let mut cur_margin = margin(&last);
            let mut greedy_all = Vec::new();
            for _ in 0..warmup {
                greedy_all.push(next);
                let l = m.eval(&[next]).expect("warm-up decode");
                cur_margin = margin(&l);
                next = argmax(&l) as i32;
            }
            let pp = m.take_profile();
            let s1 = m.expert_stats().unwrap();
            let prep_s = t_arm.elapsed().as_secs_f64();
            let prep_tokens = ub.min(tokens.len()) + warmup;

            let mut margins = vec![cur_margin];
            let mut windows = Vec::new();
            let mut tg = StepProfile::default();
            let mut sw = s1.clone();
            let t0 = Instant::now();
            let per_window = n_decode.div_ceil(n_windows.max(1)).max(1);
            let mut done = 0;
            let mut greedy = Vec::new();
            while done < n_decode {
                let n = per_window.min(n_decode - done);
                let start = unix_s();
                let tw = Instant::now();
                let mut gaps = GapTrace::default();
                for i in 0..n {
                    greedy.push(next);
                    traced_token(&mut m, &mut gaps, i, next, |_, l| {
                        margins.push(margin(&l));
                        next = argmax(&l) as i32;
                    });
                }
                let wall = tw.elapsed().as_secs_f64();
                let p = m.take_profile();
                let se = m.expert_stats().unwrap();
                windows.push(json!({
                    "start_unix_s": start,
                    "end_unix_s": unix_s(),
                    "tokens": n,
                    "wall_s": wall,
                    "gap": gaps.json(),
                    "profile_per_token": profile_json(&p, n as f64),
                    "io": io_json(&sw.io, &se.io, n as f64, wall),
                    "layers": layers_json(&sw, &se, 0..h.n_layer as usize),
                }));
                add_profile(&mut tg, &p);
                sw = se;
                done += n;
            }
            margins.truncate(n_decode);
            let decode_s = t0.elapsed().as_secs_f64();
            let s2 = m.expert_stats().unwrap();
            greedy_all.extend(&greedy);

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
                // the arm's preparation, not a prefill: see the test's docs
                "prefill_tokens": prep_tokens,
                "prefill_s": prep_s,
                "prefill_tok_s": prep_tokens as f64 / prep_s,
                "hot_refresh_s": 0.0,
                "prefill_profile_per_ubatch": profile_json(&pp, pp.steps.max(1) as f64),
                "prefill_io": io_json(&s0.io, &s1.io, prep_tokens as f64, prep_s),
                "prefill_prefetch": prefetch_json(&s0, &s1, prep_tokens as f64),
                "decode_tokens": n_decode,
                "decode_tok_s": n_decode as f64 / decode_s,
                "decode_profile_per_token": profile_json(&tg, n_decode.max(1) as f64),
                "decode_io": io_json(&s1.io, &s2.io, n_decode.max(1) as f64, decode_s),
                "decode_prefetch": prefetch_json(&s1, &s2, n_decode.max(1) as f64),
                "decode_overlap": overlap_json(&s1.io, &s2.io, n_decode.max(1) as f64),
                "decode_layers": layers_json(&s1, &s2, 0..h.n_layer as usize),
                "first_top": first_top.clone(),
                "decode_windows": windows,
                "greedy": greedy,
                "warmup_greedy": &greedy_all[..warmup],
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
            let overlay: serde_json::Map<String, Value> = arm
                .env
                .0
                .iter()
                .map(|(k, v)| (k.clone(), json!(v)))
                .collect();
            let out = json!({
                "machine": std::env::var("TETRAEDER_EXIT_MACHINE").ok(),
                "provenance": provenance(threads),
                "env": {
                    "os": std::env::consts::OS,
                    "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
                    "gpu_devices_visible": Backend::gpu_count(),
                    "ggml_system_info": tetraeder_ggml_sys::llama::system_info(),
                },
                "model": model_info.clone(),
                "prompt": prompt_info.clone(),
                "arm": {"name": arm.name, "round": round, "rounds": rounds, "env": overlay},
                "snapshot": {
                    "restore_s": restore_s,
                    "first_ubatch_s": hot_s,
                    "replay_last_s": replay_s,
                    "warmup_tokens": warmup,
                    "prep_s": prep_s,
                    "shared": snapshot_json.clone(),
                },
                "options": options_json(gpu, threads, n_ctx, &opts, false, cpu_poll),
                "io_backend": io_backend,
                "expert_bytes_per_layer": per_layer.clone(),
                "slot_bytes_all_layers": slot_bytes_all_layers,
                "runs": [run],
            });
            let name = format!("exit-{}-r{round}-{stamp}.json", arm.name);
            std::fs::create_dir_all(&out_dir).expect("output dir");
            std::fs::write(
                out_dir.join(&name),
                serde_json::to_string_pretty(&out).unwrap(),
            )
            .expect("arm result");
            eprintln!(
                "arm {} r{round}: load {load_s:.1} s, prep {prep_s:.1} s, decode {:.3} tok/s -> {name}",
                arm.name,
                n_decode as f64 / decode_s
            );
            results.push((arm.name.clone(), round, greedy_all, name));
            drop(m);
        }
    }

    // greedy tokens against the first run's, per arm and round
    let reference = results[0].2.clone();
    let mut all_equal = true;
    let runs: Vec<Value> = results
        .iter()
        .map(|(name, round, g, file)| {
            let div = g.iter().zip(&reference).position(|(a, b)| a != b);
            all_equal &= div.is_none() && g.len() == reference.len();
            json!({"arm": name, "round": round, "file": file,
                   "greedy_equal_to_first": div.is_none(), "first_divergence": div})
        })
        .collect();
    let index = json!({
        "ok": true,
        "arms": arms.iter().map(|a| a.name.clone()).collect::<Vec<_>>(),
        "rounds": rounds,
        "prompt": prompt_info,
        "snapshot": snapshot_json,
        "greedy_equal_all": all_equal,
        "runs": runs,
    });
    std::fs::write(
        out_dir.join(format!("exit-{index_name}-{stamp}.json")),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .expect("index");
    if let Snapshot::Disk(p, _) = &snapshot {
        let _ = std::fs::remove_file(p);
    }
    if let Some(t) = &telemetry {
        t.shutdown();
    }
    if exact {
        assert!(all_equal, "greedy tokens differ between arms: {index}");
    }
}

/// CI tests of the segment and warm parts of the harness (ADR 0092, steps E2 and E3) on a tiny
/// synthetic MoE: the flags, the dump file, a two-span run against the whole run, the warm
/// protocol and the result JSON. The exit regime itself needs `--ignored` and a real model.
mod ci {
    use super::*;
    use crate::common::exit_segment::{
        DEFAULT_SEED, HiddenFile, is_unclean, no_hook, pseudo_tokens, run_request, seeded_rows,
    };
    use crate::common::synth::write_moe;
    use crate::common::{PROMPT, tol_batch};
    use tetraeder_model::SegmentSpec;

    const N_DEC: usize = 5;
    const N_EMBD: usize = 64;

    fn synth4(tag: &str) -> PathBuf {
        let spec = MoeSpec {
            layers: 4,
            embd: N_EMBD,
            heads: 4,
            heads_kv: 2,
            head_dim: 16,
            experts: 8,
            used: 2,
            ff: 64,
            seed: 20261125,
        };
        let p = std::env::temp_dir().join(format!(
            "tetraeder-exitseg-{tag}-{}.gguf",
            std::process::id()
        ));
        write_moe(&p, &spec);
        p
    }

    /// 3 VRAM slots and `ram` RAM slots of 8 experts, the cpu miss policy and prefetch: the
    /// path of the 235B on a node.
    fn opts(ram: usize, record: bool) -> LoadOptions {
        LoadOptions {
            kv: KvFormat::F16,
            kv_rotation: Some(false),
            n_ctx: 512,
            expert_slots: Some(3),
            ram_slots: Some(ram),
            miss: MissPolicy::Cpu,
            prefetch: true,
            record_routing: record,
            ..LoadOptions::new(BackendChoice::Cpu { threads: Some(2) })
        }
    }

    fn load(path: &PathBuf, o: &LoadOptions, seg: Option<SegmentSpec>) -> Qwen3Moe {
        let mut o = o.clone();
        o.segment = seg;
        Qwen3Moe::load(path, &o).expect("loads")
    }

    fn work<'a>(dt: Option<&'a [i32]>, rows: Option<&'a [f32]>) -> Workload<'a> {
        Workload {
            tokens: &PROMPT,
            ub: 4,
            n_decode: N_DEC,
            decode_tokens: dt,
            rows,
        }
    }

    fn close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d <= tol_batch(want), "{what}: {d:e}");
    }

    fn env_of<'a>(pairs: &'a [(&'a str, String)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone())
    }

    fn parse(pairs: &[(&str, &str)], n: u32) -> Result<SegmentArgs, String> {
        let owned: Vec<(&str, String)> =
            pairs.iter().map(|(k, v)| (*k, (*v).to_string())).collect();
        SegmentArgs::parse(env_of(&owned), n)
    }

    // hardware: none
    #[test]
    fn flags_parse_and_refuse_bad_combinations() {
        assert_eq!(parse(&[], 94).unwrap(), SegmentArgs::default());
        assert!(!parse(&[], 94).unwrap().is_segment_run());
        // the confirmation segment: layers 47 to 93, no head, fixed seed
        let a = parse(&[("TETRAEDER_EXIT_LAYERS", "47..94")], 94).unwrap();
        assert_eq!(a.layers, Some(47..94));
        assert!(!a.head);
        assert_eq!(a.hidden_in, Some(HiddenIn::Seed(DEFAULT_SEED)));
        let s = a.specs(94);
        assert_eq!(s.len(), 1);
        assert!(!s[0].embed && !s[0].head && s[0].layers == (47..94));
        let a = parse(
            &[
                ("TETRAEDER_EXIT_LAYERS", "47..94"),
                ("TETRAEDER_EXIT_HIDDEN_SEED", "7"),
            ],
            94,
        )
        .unwrap();
        assert_eq!(a.hidden_in, Some(HiddenIn::Seed(7)));
        let a = parse(
            &[
                ("TETRAEDER_EXIT_LAYERS", "47..94"),
                ("TETRAEDER_EXIT_HEAD", "1"),
                ("TETRAEDER_EXIT_HIDDEN_IN", "rows.bin"),
            ],
            94,
        )
        .unwrap();
        assert!(a.head && a.specs(94)[0].head);
        assert_eq!(a.hidden_in, Some(HiddenIn::File("rows.bin".into())));
        // a span from layer 0 has the embedding and takes tokens
        let a = parse(&[("TETRAEDER_EXIT_LAYERS", "0..47")], 94).unwrap();
        assert!(a.hidden_in.is_none() && a.specs(94)[0].embed);
        // a dump is the whole model cut in two at the layer
        let a = parse(&[("TETRAEDER_EXIT_HIDDEN_OUT", "47,rows.bin")], 94).unwrap();
        assert!(a.is_segment_run());
        assert_eq!(a.tap_at(), Some(0));
        let s = a.specs(94);
        assert_eq!(s.len(), 2);
        assert!(s[0].embed && !s[0].head && s[0].layers == (0..47));
        assert!(!s[1].embed && s[1].head && s[1].layers == (47..94));
        // refused
        let bad: [&[(&str, &str)]; 13] = [
            &[("TETRAEDER_EXIT_LAYERS", "5..5")],
            &[("TETRAEDER_EXIT_LAYERS", "5..95")],
            &[("TETRAEDER_EXIT_LAYERS", "a..b")],
            &[("TETRAEDER_EXIT_LAYERS", "7")],
            &[
                ("TETRAEDER_EXIT_LAYERS", "47..90"),
                ("TETRAEDER_EXIT_HEAD", "1"),
            ],
            &[
                ("TETRAEDER_EXIT_LAYERS", "0..47"),
                ("TETRAEDER_EXIT_HIDDEN_SEED", "1"),
            ],
            &[("TETRAEDER_EXIT_HIDDEN_SEED", "1")],
            &[("TETRAEDER_EXIT_HIDDEN_IN", "rows.bin")],
            &[
                ("TETRAEDER_EXIT_LAYERS", "47..94"),
                ("TETRAEDER_EXIT_HIDDEN_SEED", "x"),
            ],
            &[("TETRAEDER_EXIT_HIDDEN_OUT", "0,rows.bin")],
            &[("TETRAEDER_EXIT_HIDDEN_OUT", "94,rows.bin")],
            &[("TETRAEDER_EXIT_HIDDEN_OUT", "47")],
            &[
                ("TETRAEDER_EXIT_LAYERS", "0..47"),
                ("TETRAEDER_EXIT_HIDDEN_OUT", "20,rows.bin"),
            ],
        ];
        for b in bad {
            assert!(parse(b, 94).is_err(), "{b:?}");
        }
    }

    // hardware: none
    #[test]
    fn hidden_file_round_trips_and_the_seed_input_is_fixed() {
        let f = HiddenFile {
            layer: 3,
            n_embd: 2,
            n_prompt: 2,
            tokens: vec![5, 6, 7],
            rows: vec![0.5, -1.0, 2.0, 3.5, -0.25, 8.0],
        };
        let b = f.to_bytes();
        assert_eq!(HiddenFile::from_bytes(&b).unwrap(), f);
        assert!(HiddenFile::from_bytes(&b[..b.len() - 1]).is_err());
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(HiddenFile::from_bytes(&bad).is_err());
        let a = seeded_rows(9, 64, 128);
        assert_eq!(a, seeded_rows(9, 64, 128));
        assert_ne!(a, seeded_rows(10, 64, 128));
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.1, "{mean} {var}");
        assert_eq!(pseudo_tokens(6, 260), pseudo_tokens(6, 260));
        assert!(pseudo_tokens(50, 260).iter().all(|t| (0..260).contains(t)));
    }

    // hardware: cpu. Span [0,2) with the head off and [2,4) on its dumped rows give the whole
    // run's logits within tol_batch, the same greedy tokens, and the rows the chain dumped.
    #[test]
    fn cpu_two_span_run_matches_the_whole_run() {
        let path = synth4("parity");
        let h = ModelFile::open(&path).unwrap().hparams.clone();
        let o = opts(6, false);
        let mut whole = load(&path, &o, None);
        let want = run_request(
            &mut [&mut whole],
            &work(None, None),
            "whole",
            None,
            &mut Vec::new(),
            &mut no_hook,
        );
        let head_off = SegmentSpec {
            layers: 0..2,
            embed: true,
            head: false,
        };
        let tail = SegmentSpec {
            layers: 2..4,
            embed: false,
            head: true,
        };
        let mut a = load(&path, &o, Some(head_off));
        let mut b = load(&path, &o, Some(tail));
        let chain = run_request(
            &mut [&mut a, &mut b],
            &work(None, None),
            "chain",
            Some(0),
            &mut Vec::new(),
            &mut no_hook,
        );
        assert_eq!(chain.decode_fed, want.decode_fed);
        close(&chain.last, &want.last, "chain prefill");
        close(&chain.last_decode, &want.last_decode, "chain decode");
        assert_eq!(chain.tap.len(), (PROMPT.len() + N_DEC) * N_EMBD);

        // the dump, read back as the second span's input
        let mut tokens = PROMPT.to_vec();
        tokens.extend(&chain.decode_fed);
        let fpath =
            std::env::temp_dir().join(format!("tetraeder-exitseg-{}.rows", std::process::id()));
        HiddenFile {
            layer: 2,
            n_embd: N_EMBD,
            n_prompt: PROMPT.len(),
            tokens,
            rows: chain.tap.clone(),
        }
        .write(&fpath)
        .unwrap();
        let pairs = [
            ("TETRAEDER_EXIT_LAYERS", "2..4".to_string()),
            ("TETRAEDER_EXIT_HEAD", "1".to_string()),
            ("TETRAEDER_EXIT_HIDDEN_IN", fpath.display().to_string()),
        ];
        let args = SegmentArgs::parse(env_of(&pairs), 4).unwrap();
        let specs = args.specs(4);
        let nv = h.n_vocab as usize;
        let inp = resolve_inputs(&args, &specs[0], N_EMBD, nv, &PROMPT, N_DEC).unwrap();
        let mut span = load(&path, &o, Some(specs[0].clone()));
        let got = run_request(
            &mut [&mut span],
            &work(inp.decode_tokens.as_deref(), inp.rows.as_deref()),
            "span",
            None,
            &mut Vec::new(),
            &mut no_hook,
        );
        assert_eq!(got.decode_fed, want.decode_fed);
        close(&got.last, &want.last, "span prefill");
        close(&got.last_decode, &want.last_decode, "span decode");
        // a file for another prompt is refused
        assert!(resolve_inputs(&args, &specs[0], N_EMBD, nv, &[1, 2, 3], N_DEC).is_err());

        // the first span alone: its last prompt row is the chain's dumped one
        let pairs = [("TETRAEDER_EXIT_LAYERS", "0..2".to_string())];
        let args = SegmentArgs::parse(env_of(&pairs), 4).unwrap();
        let specs = args.specs(4);
        let inp = resolve_inputs(&args, &specs[0], N_EMBD, nv, &PROMPT, N_DEC).unwrap();
        assert!(inp.rows.is_none() && inp.decode_tokens.is_some());
        let mut first = load(&path, &o, Some(specs[0].clone()));
        let got = run_request(
            &mut [&mut first],
            &work(inp.decode_tokens.as_deref(), None),
            "first",
            None,
            &mut Vec::new(),
            &mut no_hook,
        );
        let n = PROMPT.len();
        close(
            &got.last,
            &chain.tap[(n - 1) * N_EMBD..n * N_EMBD],
            "first span",
        );
        let _ = std::fs::remove_file(&fpath);
    }

    // hardware: cpu. The warm protocol: n warm requests of the same prompt (prefill and
    // decode), their routing dropped, then a timed request that recomputes every token; the
    // unclean flag follows the timed prefill's disk bytes.
    #[test]
    fn cpu_warm_protocol_runs_warm_requests_then_the_timed_one() {
        let path = synth4("warm");
        let h = ModelFile::open(&path).unwrap().hparams.clone();
        let o = opts(6, true);
        let mut m = load(&path, &o, None);
        let w = work(None, None);
        let mut log = Vec::new();
        let p = run_protocol(&mut [&mut m], &w, 2, None, &mut log, &mut no_hook);
        assert_eq!(
            log,
            [
                "warm0:reset",
                "warm0:prefill",
                "warm0:decode",
                "warm1:reset",
                "warm1:prefill",
                "warm1:decode",
                "discard_routing",
                "timed:reset",
                "timed:prefill",
                "timed:decode",
            ]
        );
        assert_eq!(p.warm.len(), 2);
        // the timed request recomputed the whole prompt and decoded from it, the same way
        assert_eq!(m.seq_len(0), PROMPT.len() + N_DEC);
        assert!(p.warm.iter().all(|r| r.decode_fed == p.timed.decode_fed));
        // a trace holds the timed request only
        let routed = m.take_routing_log();
        let layer0 = routed.iter().filter(|r| !r.refresh && r.layer == 0).count();
        assert!(layer0 > 0 && layer0 <= 2 + N_DEC, "{layer0} layer-0 steps");

        let covering = covers(m.tier_slots(), h.n_expert as usize);
        assert!(covering, "3 + 6 slots of 8 experts");
        let bytes = p.timed.prefill_disk_bytes();
        let s = warm_summary(2, covering, bytes);
        assert_eq!(s["requests"], 2);
        assert_eq!(s["timed_prefill_disk_bytes"], bytes);
        assert_eq!(s["unclean"], bytes > 0);
        assert!(!s["prefix_reuse"].as_bool().unwrap());
        assert!(!s["prompt_cache"].as_bool().unwrap());
        // a cold first request reads; the warm one reads no more than that
        let mut cold = load(&path, &o, None);
        let c = run_protocol(&mut [&mut cold], &w, 0, None, &mut Vec::new(), &mut no_hook);
        assert!(c.timed.prefill_disk_bytes() > 0);
        assert!(bytes <= c.timed.prefill_disk_bytes());
        // the predicate: only a covering segment can be unclean
        assert!(is_unclean(true, 1) && !is_unclean(true, 0) && !is_unclean(false, 1));
        assert!(!covers(Some((3, Some(4))), 8) && covers(Some((3, Some(5))), 8));
        assert!(!covers(None, 8));
    }

    fn ctx<'a>(
        path: &'a PathBuf,
        h: &'a Hparams,
        args: &'a SegmentArgs,
        o: &'a LoadOptions,
        routing_out: Option<PathBuf>,
        n_warm: usize,
    ) -> SegmentCtx<'a> {
        SegmentCtx {
            path,
            h,
            args,
            opts: o,
            tokens: &PROMPT,
            prompt_hash: "test",
            routing_out,
            ub: 4,
            n_decode: N_DEC,
            n_warm,
        }
    }

    // hardware: cpu. The span result carries the warm block, the timed prefill's bytes and the
    // per-layer pair share of the span, consistent with the aggregate.
    #[test]
    fn cpu_span_result_has_the_warm_and_pair_share_fields() {
        let path = synth4("json");
        let h = ModelFile::open(&path).unwrap().hparams.clone();
        let o = opts(6, false);
        let pairs = [
            ("TETRAEDER_EXIT_LAYERS", "2..4".to_string()),
            ("TETRAEDER_EXIT_HEAD", "1".to_string()),
        ];
        let args = SegmentArgs::parse(env_of(&pairs), 4).unwrap();
        let v = segment_run(&ctx(&path, &h, &args, &o, None, 1));
        assert_eq!(v["segment"]["layers"], json!([2, 4]));
        assert_eq!(v["segment"]["hidden_in"]["seed"], DEFAULT_SEED);
        assert_eq!(v["warm"]["requests"], 1);
        assert!(v["warm"]["covering"].as_bool().unwrap());
        let bytes = v["prefill_io"]["bytes"].as_u64().unwrap();
        assert_eq!(v["warm"]["timed_prefill_disk_bytes"], bytes);
        assert_eq!(v["warm"]["unclean"], bytes > 0);
        assert_eq!(v["warm_requests"].as_array().unwrap().len(), 1);
        assert_eq!(
            v["request_log"],
            json!([
                "warm0:reset",
                "warm0:prefill",
                "warm0:decode",
                "discard_routing",
                "timed:reset",
                "timed:prefill",
                "timed:decode"
            ])
        );
        assert!(v["first_top"].is_array());
        assert!(v["decode_io"]["bytes_per_token"].is_number());
        let l = &v["decode_layers"];
        assert_eq!(l["first_layer"], 2);
        let share = l["vram_pair_share_layers"].as_array().unwrap();
        let weights = l["pairs_layers"].as_array().unwrap();
        assert_eq!((share.len(), weights.len()), (2, 2));
        let (mut hit, mut all) = (0.0, 0.0);
        for (s, p) in share.iter().zip(weights) {
            let (s, p) = (s.as_f64().unwrap(), p.as_f64().unwrap());
            assert!((0.0..=1.0).contains(&s) && p > 0.0);
            hit += s * p;
            all += p;
        }
        let agg = l["vram_pair_share"].as_f64().unwrap();
        assert!((hit / all - agg).abs() < 1e-9, "{} {agg}", hit / all);
        assert_eq!(l["full_hit_rate"].as_array().unwrap().len(), 2);
    }

    // hardware: cpu. A dump run is the whole model as two segments: it writes the rows and the
    // tokens, picks greedy tokens, and records one trace.
    #[test]
    fn cpu_dump_run_writes_the_hidden_rows() {
        let path = synth4("dump");
        let h = ModelFile::open(&path).unwrap().hparams.clone();
        let o = opts(6, true);
        let dir = std::env::temp_dir();
        let id = std::process::id();
        let rows = dir.join(format!("tetraeder-exitseg-dump-{id}.rows"));
        let trace = dir.join(format!("tetraeder-exitseg-dump-{id}.trace"));
        let pairs = [("TETRAEDER_EXIT_HIDDEN_OUT", format!("2,{}", rows.display()))];
        let args = SegmentArgs::parse(env_of(&pairs), 4).unwrap();
        let v = segment_run(&ctx(&path, &h, &args, &o, Some(trace.clone()), 1));
        let f = HiddenFile::read(&rows).unwrap();
        assert_eq!((f.layer, f.n_embd, f.n_prompt), (2, N_EMBD, PROMPT.len()));
        assert_eq!(f.tokens.len(), PROMPT.len() + N_DEC);
        assert_eq!(f.tokens[..PROMPT.len()], PROMPT);
        assert_eq!(f.rows.len(), f.tokens.len() * N_EMBD);
        assert_eq!(v["greedy"], json!(&f.tokens[PROMPT.len()..]));
        assert_eq!(v["segment"]["hidden_out"]["rows"], f.tokens.len());
        assert_eq!(v["segment"]["segments"].as_array().unwrap().len(), 2);
        assert!(v["prefill_io"].is_null() && v["first_top"].is_array());
        assert!(std::fs::metadata(&trace).unwrap().len() > 0);
        let _ = std::fs::remove_file(&rows);
        let _ = std::fs::remove_file(&trace);
    }
}

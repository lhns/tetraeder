//! CPU speed of the builder against libllama on the same GGUF with the same thread count
//! (rule 3): prefill and decode tok/s, paired and interleaved, plus the builder's time breakdown.
//!
//! Hardware label: `cpu`. Ignored by default (slow, release only):
//! `cargo test --release -p tetraeder-model --test cpu_bench -- --ignored --nocapture`
//!
//! `segment_overhead` (ADR 0064) measures the per-graph costs of a streamed decode step: the
//! CPU dispatch floor, scheduler allocation and the synthetic MoE decode before and after the
//! segment-sized scheduler. Env: `TETRAEDER_OVERHEAD_PARTS` (`floor,alloc,e2e`, default all),
//! `TETRAEDER_OVERHEAD_THREADS` (the n of the floor's 1, n/2, n; default the bench thread
//! count), `TETRAEDER_BENCH_OUT`.
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
use tetraeder_ggml_sys::ggml::{Backend, Context as GgmlContext, Graph, Sched, Tensor, Type};
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
        "cpu_pairs": p.cpu_pairs as f64 / per,
        "seg_pairs": p.seg_pairs as f64 / per,
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
    a.cpu_pairs += b.cpu_pairs;
    a.seg_pairs += b.seg_pairs;
}

/// hardware: cpu. Builder against libllama on the CPU, written as JSON (see the module docs).
#[test]
#[ignore = "slow CPU benchmark; run with --ignored in release"]
fn cpu_bench_builder_vs_libllama() {
    let n_prompt = env_usize("TETRAEDER_BENCH_PROMPT", 64);
    let n_decode = env_usize("TETRAEDER_BENCH_DECODE", 32);
    let repeats = env_usize("TETRAEDER_BENCH_REPEATS", 3).max(1);
    let threads = common::bench_thread_list();
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
            "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
            "cpu_quota": common::cpu_quota(),
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

// ---- segment_overhead (ADR 0064) ----

const D_EMBD: i64 = 64;
const D_FF: i64 = 32;
const D_EXPERTS: i64 = 8;
const D_USED: i64 = 4;

/// The leaves of the chain graphs, filled after allocation: F32 ones with zeros, the I32 ones
/// with valid ids.
#[derive(Default)]
struct Leaves<'a> {
    f32s: Vec<Tensor<'a>>,
    i32s: Vec<Tensor<'a>>,
}

impl Leaves<'_> {
    fn fill(&self) {
        for t in &self.f32s {
            t.set_f32(&vec![0.0; t.nbytes() / 4]).unwrap();
        }
        for t in &self.i32s {
            let ids: Vec<i32> = (0..t.nbytes() / 4)
                .map(|i| (i as i64 % D_EXPERTS) as i32)
                .collect();
            t.set_i32(&ids).unwrap();
        }
    }
}

/// `chains` copies of the 6-node CPU group shape (`get_rows`, gate, up and down `mul_mat_id`,
/// `swiglu`, `cpy`) on tiny tensors, sharing the weights. `extra` adds that many `add` nodes
/// to the graph, to give a second topology.
fn chain_graph<'a>(
    ctx: &'a GgmlContext,
    chains: usize,
    extra: usize,
    leaves: &mut Leaves<'a>,
) -> Graph<'a> {
    let mut g = ctx.new_graph(1024);
    let table = ctx.new_tensor(Type::F32, &[D_EMBD, 16]);
    let wg = ctx.new_tensor(Type::F32, &[D_EMBD, D_FF, D_EXPERTS]);
    let wu = ctx.new_tensor(Type::F32, &[D_EMBD, D_FF, D_EXPERTS]);
    let wd = ctx.new_tensor(Type::F32, &[D_FF, D_EMBD, D_EXPERTS]);
    let ids = ctx.new_tensor(Type::I32, &[D_USED, 1]);
    leaves.f32s.extend([table, wg, wu, wd]);
    leaves.i32s.push(ids);
    for _ in 0..chains {
        let tok = ctx.new_tensor(Type::I32, &[1]);
        let dst = ctx.new_tensor(Type::F32, &[D_EMBD, D_USED, 1]);
        leaves.i32s.push(tok);
        leaves.f32s.push(dst);
        let x = ctx.get_rows(table, tok);
        let gate = ctx.mul_mat_id(wg, x, ids);
        let up = ctx.mul_mat_id(wu, x, ids);
        let h = ctx.swiglu_split(gate, up);
        let down = ctx.mul_mat_id(wd, h, ids);
        g.expand(ctx.cpy(down, dst).set_output());
    }
    for _ in 0..extra {
        let a = ctx.new_tensor(Type::F32, &[D_EMBD]);
        let b = ctx.new_tensor(Type::F32, &[D_EMBD]);
        leaves.f32s.extend([a, b]);
        g.expand(ctx.add(a, b).set_output());
    }
    g
}

fn spin_us(us: u64) {
    if us == 0 {
        return;
    }
    let t = Instant::now();
    while t.elapsed() < Duration::from_micros(us) {
        std::hint::spin_loop();
    }
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q) as usize]
}

/// Part 1: us per compute of a 1-node and a 6-node CPU graph.
fn dispatch_floor(n: usize) -> Value {
    const COMPUTES: usize = 2000;
    let mut threads = vec![1, (n / 2).max(1), n.max(1)];
    threads.dedup();
    let mut rows: Vec<Value> = Vec::new();
    for &t in &threads {
        for poll in [0u32, 50] {
            let cpu = Backend::cpu_with_poll(t as i32, Some(poll)).unwrap();
            let mut sched = Sched::new(std::slice::from_ref(&cpu), 4096, false).unwrap();
            for (kind, chains, extra) in [("1_node", 0, 1), ("6_node", 1, 0)] {
                let ctx = GgmlContext::new(256, 1024).unwrap();
                let mut leaves = Leaves::default();
                let g = chain_graph(&ctx, chains, extra, &mut leaves);
                sched.alloc_graph(&g).unwrap();
                leaves.fill();
                for pause in [false, true] {
                    for gap in [0u64, 100, 500] {
                        for _ in 0..50 {
                            sched.compute(&g).unwrap();
                        }
                        let mut us = Vec::with_capacity(COMPUTES);
                        for _ in 0..COMPUTES {
                            spin_us(gap);
                            if pause {
                                cpu.pause_cpu_pool();
                            }
                            let t0 = Instant::now();
                            sched.compute(&g).unwrap();
                            us.push(t0.elapsed().as_secs_f64() * 1e6);
                        }
                        let mean = us.iter().sum::<f64>() / us.len() as f64;
                        us.sort_by(f64::total_cmp);
                        let row = json!({
                            "threads": t, "poll": poll, "pause": pause, "gap_us": gap,
                            "graph": kind, "nodes": g.n_nodes(), "computes": COMPUTES,
                            "mean_us": mean, "median_us": pct(&us, 0.5), "p90_us": pct(&us, 0.9),
                        });
                        eprintln!("{row}");
                        rows.push(row);
                    }
                }
            }
        }
    }
    // Per configuration: the 1-node graph is one kickoff and one barrier; each further node of
    // the 6-node graph adds a barrier, so (m6 - m1) / 5 is the cost per barrier.
    let mut derived = Vec::new();
    for a in rows.iter().filter(|r| r["graph"] == "1_node") {
        let b = rows.iter().find(|r| {
            r["graph"] == "6_node"
                && ["threads", "poll", "pause", "gap_us"]
                    .iter()
                    .all(|k| r[*k] == a[*k])
        });
        if let Some(b) = b {
            let m1 = a["median_us"].as_f64().unwrap();
            let m6 = b["median_us"].as_f64().unwrap();
            derived.push(json!({
                "threads": a["threads"], "poll": a["poll"], "pause": a["pause"],
                "gap_us": a["gap_us"],
                "kickoff_us": m1, "barrier_us": (m6 - m1) / 5.0,
            }));
        }
    }
    json!({"rows": rows, "per_kickoff_and_barrier": derived})
}

/// Part 2: us per `alloc_graph` of a segment-sized graph (25 chains) on schedulers of 4,096
/// and 48,640 nodes, one topology and two alternating, and per `reset` alone.
fn alloc_cost() -> Value {
    const ALLOCS: usize = 400;
    let ctx_a = GgmlContext::new(1024, 1024).unwrap();
    let ctx_b = GgmlContext::new(1024, 1024).unwrap();
    let (mut la, mut lb) = (Leaves::default(), Leaves::default());
    let ga = chain_graph(&ctx_a, 25, 0, &mut la);
    let gb = chain_graph(&ctx_b, 24, 5, &mut lb);
    let cpu = Backend::cpu(1).unwrap();
    let mut rows = Vec::new();
    for nodes in [4096usize, 48_640] {
        let mut sched = Sched::new(std::slice::from_ref(&cpu), nodes, false).unwrap();
        for k in 0..4 {
            let g = if k % 2 == 0 { &ga } else { &gb };
            sched.alloc_graph(g).unwrap();
        }
        for (topo, alternate) in [("one", false), ("two", true)] {
            let t0 = Instant::now();
            for i in 0..ALLOCS {
                let g = if alternate && i % 2 == 1 { &gb } else { &ga };
                sched.alloc_graph(g).unwrap();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / ALLOCS as f64;
            let row = json!({"sched_nodes": nodes, "topologies": topo,
                "graph_nodes": ga.n_nodes(), "alloc_us": us});
            eprintln!("{row}");
            rows.push(row);
        }
        let t0 = Instant::now();
        for _ in 0..ALLOCS {
            sched.reset();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / ALLOCS as f64;
        let row = json!({"sched_nodes": nodes, "reset_us": us});
        eprintln!("{row}");
        rows.push(row);
    }
    json!(rows)
}

/// Part 3: the `builder_cpupolicy` decode with the whole-step scheduler size and with the
/// segment size (ADR 0064 step 1), interleaved; greedy tokens must be equal.
fn end_to_end(threads: usize, path: &std::path::Path) -> Value {
    let n_prompt = env_usize("TETRAEDER_BENCH_PROMPT", 64);
    let n_decode = env_usize("TETRAEDER_BENCH_DECODE", 32);
    let repeats = env_usize("TETRAEDER_BENCH_REPEATS", 3).max(1);
    let n_ctx = (n_prompt + n_decode + 16).next_multiple_of(256);
    let n_expert = ModelFile::open(path).unwrap().hparams.n_expert as usize;
    let mut engines: Vec<(&str, Builder)> = Vec::new();
    for (name, whole) in [("whole_step_sched", true), ("segment_sched", false)] {
        let mut o = LoadOptions::new(BackendChoice::Cpu {
            threads: Some(threads),
        });
        o.n_ctx = n_ctx;
        o.expert_slots = Some((n_expert / 4).max(1));
        o.ram_slots = Some(n_expert);
        o.miss = MissPolicy::Cpu;
        o.whole_step_sched = whole;
        let m = Qwen3Moe::load(path, &o).unwrap();
        engines.push((name, Builder { m, last: None }));
    }
    let n_vocab = engines[0].1.m.n_vocab() as i64;
    let prompt: Vec<i32> = (0..n_prompt as i64)
        .map(|i| ((i * 7919 + 13) % n_vocab.min(256)) as i32)
        .collect();
    for (_, e) in &mut engines {
        e.run(&prompt, 4);
        e.profiles();
    }
    let mut tok_s: Vec<Vec<f64>> = vec![Vec::new(); 2];
    let mut profs = vec![StepProfile::default(); 2];
    let mut tokens: Vec<Vec<i32>> = vec![Vec::new(); 2];
    for r in 0..repeats {
        for k in 0..2 {
            let i = (k + r) % 2;
            let s = engines[i].1.run(&prompt, n_decode);
            tok_s[i].push(n_decode as f64 / s.decode.as_secs_f64());
            if let Some((_, dp)) = engines[i].1.profiles() {
                add(&mut profs[i], &dp);
            }
            tokens[i] = s.tokens;
        }
    }
    let tokens_equal = tokens[0] == tokens[1];
    let variants: Vec<Value> = (0..2)
        .map(|i| {
            json!({
                "variant": engines[i].0,
                "decode_tok_s": tok_s[i],
                "decode_tok_s_median": median(tok_s[i].clone()),
                "profile_decode_per_token": profile_json(&profs[i], (repeats * n_decode) as f64),
            })
        })
        .collect();
    eprintln!("end to end: tokens_equal {tokens_equal} {variants:?}");
    assert!(tokens_equal, "step 1 changed the greedy tokens");
    json!({"threads": threads, "n_prompt": n_prompt, "n_decode": n_decode,
        "repeats": repeats, "greedy_equal": tokens_equal, "variants": variants})
}

/// hardware: cpu. Per-graph costs of a streamed decode step (ADR 0064): dispatch floor,
/// allocation, end to end. Written as JSON to `TETRAEDER_BENCH_OUT`.
#[test]
#[ignore = "slow CPU benchmark; run with --ignored in release"]
fn segment_overhead() {
    let parts =
        std::env::var("TETRAEDER_OVERHEAD_PARTS").unwrap_or_else(|_| "floor,alloc,e2e".into());
    let has = |p: &str| parts.split(',').any(|x| x.trim() == p);
    let n = env_usize("TETRAEDER_OVERHEAD_THREADS", common::bench_threads());
    let mut out = json!({
        "env": {
            "os": std::env::consts::OS,
            "cpu": cpu_model(),
            "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
            "cpu_quota": common::cpu_quota(),
            "threads": n,
        },
        "parts": parts,
    });
    if has("floor") {
        out["dispatch_floor"] = dispatch_floor(n);
    }
    if has("alloc") {
        out["allocation"] = alloc_cost();
    }
    if has("e2e") {
        let (path, info) = model_path();
        out["model"] = info;
        out["end_to_end"] = end_to_end(n, &path);
    }
    if let Some(p) = std::env::var_os("TETRAEDER_BENCH_OUT") {
        std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
}

/// hardware: cpu. The graphs of `segment_overhead` build, allocate and compute on a small
/// pool, and both scheduler sizes allocate a segment-sized graph, so a shape error shows in CI
/// and not on a box.
#[test]
fn cpu_segment_overhead_graphs_compute() {
    let cpu = Backend::cpu_with_poll(2, Some(0)).unwrap();
    let mut sched = Sched::new(std::slice::from_ref(&cpu), 4096, false).unwrap();
    for (chains, extra, nodes) in [(0, 1, 1), (1, 0, 6)] {
        let ctx = GgmlContext::new(256, 1024).unwrap();
        let mut leaves = Leaves::default();
        let g = chain_graph(&ctx, chains, extra, &mut leaves);
        assert_eq!(g.n_nodes(), nodes);
        sched.alloc_graph(&g).unwrap();
        leaves.fill();
        sched.compute(&g).unwrap();
        sched.compute(&g).unwrap();
    }
    let ctx = GgmlContext::new(1024, 1024).unwrap();
    let mut leaves = Leaves::default();
    let g = chain_graph(&ctx, 25, 0, &mut leaves);
    for n in [4096usize, 48_640] {
        let mut sched = Sched::new(std::slice::from_ref(&cpu), n, false).unwrap();
        sched.alloc_graph(&g).unwrap();
        leaves.fill();
        sched.compute(&g).unwrap();
    }
}

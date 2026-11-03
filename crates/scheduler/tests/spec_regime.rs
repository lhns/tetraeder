//! Speculative decoding in the Phase 2 exit regime (ADR 0043), modeled on
//! `crates/model/tests/exit_regime.rs`: one prefill, then one long greedy decode whose blocks
//! rotate between plain decoding, the n-gram drafter and a draft model, so every method sees the
//! same cache history. A plain replay of the whole continuation then checks the tokens.
//!
//! Hardware label: `cpu` with the default synthetic model; `gpu_*` on the exit box
//! (`TETRAEDER_EXIT_GPU=1`, CUDA build). Ignored by default (slow, release only):
//! `cargo test --release -p tetraeder-scheduler --test spec_regime -- --ignored --nocapture`
//!
//! Env (all optional). The exit regime's knobs keep their meaning: `TETRAEDER_EXIT_GGUF` (the
//! same synthetic Q4_K_M model and file name as exit_regime, shared), `_GPU`, `_THREADS`,
//! `_VRAM_SLOT_BYTES` / `_SLOTS`, `_RAM_SLOT_BYTES` / `_RAM_SLOTS`, `_PREFILL` (256), `_UBATCH`
//! (first value, 256), `_CTX`, `_OP_OFFLOAD`, `_PREFETCH`, `_PROMPT_FILE`, `_OUT`, `_MACHINE`.
//! New:
//! - `TETRAEDER_EXIT_SPEC_VARIANTS` ("plain,ngram:a,draft:a"): each `plain` or
//!   `<ngram|draft>:<k|a>`, `a` = adaptive up to 6. Draft variants are skipped without a draft.
//! - `TETRAEDER_EXIT_SPEC_BLOCK` (32) tokens per block, `_ROUNDS` (4) rounds; the variant order
//!   rotates by one each round.
//! - `TETRAEDER_EXIT_SPEC_DRAFT_GGUF`, `_DRAFT_GPU_LAYERS` (99 with `TETRAEDER_EXIT_GPU=1`, else
//!   0), `_PMIN` (0.0), `_NGRAM_MIN` (2), `_NGRAM_MAX` (4).
//! - `TETRAEDER_EXIT_SPEC_REPLAY` (1): plain replay of the continuation.

#[allow(dead_code)]
#[path = "../../model/tests/common/synth.rs"]
mod synth;

use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use synth::{MoeSpec, quantized_moe_at};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Ftype, Model, ModelParams};
use tetraeder_memory::process;
use tetraeder_model::{
    BackendChoice, ExpertStats, IoCounters, LoadOptions, MissPolicy, ModelFile, Qwen3Moe,
    StepProfile,
};
use tetraeder_scheduler::spec::{
    DraftLen, DraftModel, Drafter, MAX_VERIFY, NgramDrafter, Sampling, Speculator, argmax,
};

/// Same model as exit_regime.rs, so the generated file is shared.
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
}

/// RAM-tier hits and misses summed over layers.
fn ram_counts(s: &ExpertStats) -> (u64, u64) {
    s.layers
        .iter()
        .filter_map(|l| l.ram)
        .fold((0, 0), |(h, m), r| (h + r.hits, m + r.misses))
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Plain,
    Ngram,
    Draft,
}

struct Variant {
    name: String,
    method: Method,
    len: DraftLen,
}

fn parse_variants(s: &str) -> Vec<Variant> {
    s.split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| {
            let (m, k) = v.split_once(':').unwrap_or((v, "a"));
            let method = match m {
                "plain" => Method::Plain,
                "ngram" => Method::Ngram,
                "draft" => Method::Draft,
                _ => panic!("unknown variant `{v}`"),
            };
            let len = match k {
                "a" => DraftLen::Adaptive {
                    max: MAX_VERIFY - 1,
                },
                k => DraftLen::Fixed(k.parse::<usize>().expect("k or a").min(MAX_VERIFY - 1)),
            };
            Variant {
                name: v.to_owned(),
                method,
                len: if method == Method::Plain {
                    DraftLen::Fixed(0)
                } else {
                    len
                },
            }
        })
        .collect()
}

/// Totals of one variant over all its blocks.
struct Acc {
    speculator: Speculator,
    tokens: u64,
    secs: f64,
    profile: StepProfile,
    bytes: u64,
    read_secs: f64,
    experts: u64,
    ram_hits: u64,
    ram_misses: u64,
}

fn io_delta(a: &IoCounters, b: &IoCounters) -> (u64, f64, u64) {
    (
        b.bytes_read - a.bytes_read,
        (b.read_time - a.read_time).as_secs_f64(),
        b.experts_read - a.experts_read,
    )
}

fn variant_json(v: &Variant, a: &Acc) -> Value {
    let st = a.speculator.stats();
    let per = a.tokens.max(1) as f64;
    let by_batch: serde_json::Map<String, Value> = st
        .by_batch
        .iter()
        .enumerate()
        .filter(|(_, b)| b.steps > 0)
        .map(|(i, b)| {
            (
                (i + 1).to_string(),
                json!({"mean_ms": b.time.as_secs_f64() * 1e3 / b.steps as f64, "count": b.steps}),
            )
        })
        .collect();
    let method = match v.method {
        Method::Plain => "plain",
        Method::Ngram => "ngram",
        Method::Draft => "draft",
    };
    json!({
        "name": v.name,
        "method": method,
        "k": format!("{:?}", v.len),
        "tokens": a.tokens,
        "seconds": a.secs,
        "tok_s": a.tokens as f64 / a.secs.max(1e-9),
        "steps": st.steps,
        "tokens_per_step": st.emitted as f64 / st.steps.max(1) as f64,
        "drafted": st.drafted,
        "accepted": st.accepted,
        "acceptance": if st.drafted > 0 { st.accepted as f64 / st.drafted as f64 } else { 0.0 },
        "draft_ms_per_token": st.draft_time.as_secs_f64() * 1e3 / per,
        "verify_ms_by_batch": by_batch,
        "profile_per_token": profile_json(&a.profile, per),
        "io_per_token": {
            "bytes": a.bytes as f64 / per,
            "read_ms": a.read_secs * 1e3 / per,
            "experts": a.experts as f64 / per,
        },
        "ram_hit_rate": if a.ram_hits + a.ram_misses > 0 {
            json!(a.ram_hits as f64 / (a.ram_hits + a.ram_misses) as f64)
        } else {
            Value::Null
        },
    })
}

/// hardware: cpu (default) or gpu_* (`TETRAEDER_EXIT_GPU=1`). See the module docs.
#[test]
#[ignore = "slow speculative-decoding benchmark; run with --ignored in release"]
fn spec_regime() {
    let gpu = env_flag("TETRAEDER_EXIT_GPU");
    if gpu && !Backend::gpu_available() {
        eprintln!("skip: TETRAEDER_EXIT_GPU=1 but no GPU in this build");
        return;
    }
    let (path, model_info) = model_path();
    let file = ModelFile::open(&path).expect("model opens");
    let h = file.hparams.clone();
    let n_prefill = env_u64("TETRAEDER_EXIT_PREFILL").unwrap_or(256) as usize;
    let ubatch = std::env::var("TETRAEDER_EXIT_UBATCH")
        .ok()
        .and_then(|s| s.split(',').find_map(|v| v.trim().parse::<usize>().ok()))
        .unwrap_or(256)
        .max(1);
    let threads = env_u64("TETRAEDER_EXIT_THREADS")
        .map(|t| t as usize)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let block = env_u64("TETRAEDER_EXIT_SPEC_BLOCK").unwrap_or(32) as usize;
    let rounds = env_u64("TETRAEDER_EXIT_SPEC_ROUNDS").unwrap_or(4) as usize;
    let replay = std::env::var("TETRAEDER_EXIT_SPEC_REPLAY").map_or(true, |v| v != "0");
    let p_min = std::env::var("TETRAEDER_EXIT_SPEC_PMIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);
    let ngram = NgramDrafter {
        min: env_u64("TETRAEDER_EXIT_SPEC_NGRAM_MIN").unwrap_or(2) as usize,
        max: env_u64("TETRAEDER_EXIT_SPEC_NGRAM_MAX").unwrap_or(4) as usize,
    };
    let draft_gguf = std::env::var_os("TETRAEDER_EXIT_SPEC_DRAFT_GGUF").map(PathBuf::from);
    let draft_gpu_layers = env_u64("TETRAEDER_EXIT_SPEC_DRAFT_GPU_LAYERS")
        .map(|n| n as i32)
        .unwrap_or(if gpu { 99 } else { 0 });

    // A fully offloaded draft needs no CPU threads; a second spinning pool would steal cores
    // from the builder's CPU experts.
    let draft_threads = if draft_gpu_layers > 0 {
        1
    } else {
        threads as i32
    };

    let mut variants = parse_variants(
        &std::env::var("TETRAEDER_EXIT_SPEC_VARIANTS")
            .unwrap_or_else(|_| "plain,ngram:a,draft:a".into()),
    );
    let mut notes = Vec::new();
    if draft_gguf.is_none() && variants.iter().any(|v| v.method == Method::Draft) {
        notes.push("draft variants skipped: TETRAEDER_EXIT_SPEC_DRAFT_GGUF is not set".to_owned());
        variants.retain(|v| v.method != Method::Draft);
    }
    assert!(!variants.is_empty(), "no variants to run");

    // Every block can overshoot its target by up to MAX_VERIFY - 1 tokens.
    let max_decode = rounds * variants.len() * (block + MAX_VERIFY);
    let n_ctx = env_u64("TETRAEDER_EXIT_CTX")
        .map(|c| c as usize)
        .unwrap_or(n_prefill + max_decode + 8)
        .max(n_prefill + max_decode + 8);
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
    // Unset follows LoadOptions (prefetch on); `0` turns it off, `1` on.
    if let Ok(v) = std::env::var("TETRAEDER_EXIT_PREFETCH") {
        opts.prefetch = v == "1";
    }
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

    let mut m = Qwen3Moe::load(&path, &opts).expect("builder loads");
    let mut draft = draft_gguf.as_ref().map(|p| {
        let d = DraftModel::load(p, n_ctx as u32, draft_gpu_layers, draft_threads, p_min)
            .unwrap_or_else(|e| {
                panic!(
                    "draft model loads: {e:?}. At 32K the draft needs about 4.4 GB of VRAM (0.6 GB of \
                     weights, 3.8 GB of F16 KV) beside the builder's expert slots: lower \
                     TETRAEDER_EXIT_VRAM_SLOT_BYTES by that much (scripts/spec_run.sh does)"
                )
            });
        d.check_vocab(m.n_vocab()).expect("draft shares the vocab");
        d
    });

    m.take_profile();
    let t0 = Instant::now();
    let mut last = Vec::new();
    for chunk in tokens.chunks(ubatch) {
        last = m.eval_last(chunk).expect("prefill");
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    m.take_profile();
    let mut history = tokens.clone();
    history.push(argmax(&last) as i32);

    // Decode the history once, untimed per block, so the first draft block does not pay for it.
    let mut draft_prefill_s = 0.0;
    if let Some(d) = draft.as_mut()
        && variants.iter().any(|v| v.method == Method::Draft)
    {
        let t = Instant::now();
        d.draft(&history, 1).expect("draft warmup");
        draft_prefill_s = t.elapsed().as_secs_f64();
    }

    let mut accs: Vec<Acc> = variants
        .iter()
        .map(|v| Acc {
            speculator: Speculator::new(v.len, Sampling::GREEDY, None),
            tokens: 0,
            secs: 0.0,
            profile: StepProfile::default(),
            bytes: 0,
            read_secs: 0.0,
            experts: 0,
            ram_hits: 0,
            ram_misses: 0,
        })
        .collect();
    let mut ngram_drafter = ngram;
    let mut replay_json = Value::Null;

    let write = |rounds_done: usize,
                 accs: &[Acc],
                 history: &[i32],
                 replay_v: &Value,
                 m: &Qwen3Moe| {
        let memory: Vec<Value> = m
            .memory()
            .iter()
            .map(|u| {
                json!({"device": u.device, "tier": u.tier, "kind": u.kind,
                       "capacity": u.capacity, "used": u.used})
            })
            .collect();
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
                "ubatch": ubatch,
                "op_offload": opts.op_offload,
                "prefetch": opts.prefetch,
                "expert_slots": opts.expert_slots,
                "expert_slot_bytes": opts.expert_slot_bytes,
                "ram_slots": opts.ram_slots,
                "ram_slot_bytes": opts.ram_slot_bytes,
                "variants": variants.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
                "block": block,
                "rounds": rounds,
                "rounds_done": rounds_done,
                "replay": replay,
                "draft_gguf": draft_gguf.as_ref().map(|p| p.display().to_string()),
                "draft_gpu_layers": draft_gpu_layers,
                "draft_threads": draft_threads,
                "p_min": p_min,
                "ngram_min": ngram.min,
                "ngram_max": ngram.max,
            },
            "notes": notes,
            "prefill_tokens": n_prefill,
            "prefill_s": prefill_s,
            "prefill_tok_s": n_prefill as f64 / prefill_s,
            "draft_prefill_s": draft_prefill_s,
            "variants": variants.iter().zip(accs).map(|(v, a)| variant_json(v, a)).collect::<Vec<_>>(),
            "continuation": &history[n_prefill + 1..],
            "replay": replay_v,
            "max_splits": m.max_splits(),
            "memory": memory,
            "rss_bytes": process::current_rss(),
            "peak_rss_bytes": process::peak_rss(),
        });
        if let Some(p) = std::env::var_os("TETRAEDER_EXIT_OUT") {
            std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        }
    };

    for round in 0..rounds {
        for j in 0..variants.len() {
            let i = (j + round) % variants.len();
            let v = &variants[i];
            let a = &mut accs[i];
            let drafter: Option<&mut dyn Drafter> = match v.method {
                Method::Plain => None,
                Method::Ngram => Some(&mut ngram_drafter),
                Method::Draft => draft.as_mut().map(|d| d as &mut dyn Drafter),
            };
            m.take_profile();
            let s0 = m.expert_stats().expect("the exit regime streams experts");
            let t = Instant::now();
            let n = a
                .speculator
                .generate(&mut m, drafter, &mut history, block)
                .expect("block");
            let secs = t.elapsed().as_secs_f64();
            let p = m.take_profile();
            let s1 = m.expert_stats().unwrap();
            let (bytes, read_secs, experts) = io_delta(&s0.io, &s1.io);
            let (h0, m0) = ram_counts(&s0);
            let (h1, m1) = ram_counts(&s1);
            a.tokens += n as u64;
            a.secs += secs;
            add_profile(&mut a.profile, &p);
            a.bytes += bytes;
            a.read_secs += read_secs;
            a.experts += experts;
            a.ram_hits += h1 - h0;
            a.ram_misses += m1 - m0;
            eprintln!(
                "round {round} {:<10} {n} tokens {:.2} tok/s, {:.1} MB read/token",
                v.name,
                n as f64 / secs,
                bytes as f64 / n as f64 / 1e6
            );
        }
        write(round + 1, &accs, &history, &replay_json, &m);
    }

    if replay {
        // Plain one-token steps from the same prefill; teacher-forced by the continuation until
        // the first token that differs.
        m.rewind(n_prefill).expect("rewind to the prefill");
        let cont = &history[n_prefill..];
        let mut first_div = None;
        let mut checked = 0;
        for i in 0..cont.len() - 1 {
            let l = m.eval(&[cont[i]]).expect("replay");
            let top = argmax(&l);
            checked += 1;
            if top as i32 != cont[i + 1] {
                let second = l
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != top)
                    .map(|(_, &x)| x)
                    .fold(f32::MIN, f32::max);
                first_div = Some(json!({
                    "index": i,
                    "margin": l[top] - second,
                    "replay_token": top,
                    "continuation_token": cont[i + 1],
                }));
                break;
            }
        }
        replay_json = json!({
            "equal": first_div.is_none(),
            "checked": checked,
            "first_divergence": first_div,
        });
        eprintln!("replay: {replay_json}");
    }
    write(rounds, &accs, &history, &replay_json, &m);
}

//! Step shapes on the CPU (loop-bench, ADR 0065): the wall time of one `BatchBackend::decode` for
//! the shapes continuous batching produces at 4 sequences, on both backends `serve` uses, so a
//! gap to llama-server can be split into KV width, mixing decoders into a prefill step, and the
//! rest.
//!
//! - `alone`: a prompt alone, the pool empty.
//! - `resident`: the same with three sequences resident below it (wider `n_kv`), not stepping.
//! - `mixedK`: `resident` with K of them decoding one token each in the same step (`_prompt_first`:
//!   the prompt chunk first).
//! - `splitK`: `mixedK` as two `decode` calls, the decoders then the prompt.
//! - `decode4x8`, `decode1x8_alone`, `decode1x8_resident`: eight decode steps of four sequences,
//!   or of one alone or above three resident ones.
//!
//! Engines: `llama` and `dense` as serve runs them; `_one_call` without the CPU split of decoders
//! and prompt chunks (`step_groups`); `llama_streams` with one KV stream per sequence.
//!
//! Hardware label: `cpu_model` (the GGUF named by `TETRAEDER_SHAPES_GGUF`); skips without it.
//! Env: `TETRAEDER_SHAPES_THREADS` (4), `TETRAEDER_SHAPES_PROMPT` (402), `TETRAEDER_SHAPES_REPS` (3).

use std::path::{Path, PathBuf};
use std::time::Instant;
use tetraeder_engine::{ContextConfig, Engine, FlashAttn, Placement};
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};
use tetraeder_scheduler::{BatchBackend, DenseBackend, LlamaBackend, SeqId, SeqInput};

const SEQS: u32 = 4;
const CTX: usize = 8192;
const RESIDENT: usize = 470;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn toks(seed: usize, n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| ((seed * 7919 + i * 104_729) % 30_000 + 200) as i32)
        .collect()
}

fn reset(b: &mut dyn BatchBackend) {
    for s in 0..SEQS {
        b.seq_rm(s, 0).unwrap();
    }
}

/// Sequences 0..3 hold `RESIDENT` tokens; sequence 3 is empty with room for the prompt.
fn fill(b: &mut dyn BatchBackend, prompt: usize) {
    reset(b);
    for s in 0..3 {
        b.seq_reserve(s, RESIDENT + 64).unwrap();
        let t = toks(s as usize, RESIDENT);
        b.decode(&[SeqInput {
            seq: s,
            tokens: &t,
            all_logits: false,
        }])
        .unwrap();
    }
    b.seq_reserve(3, prompt + 64).unwrap();
}

fn timed(b: &mut dyn BatchBackend, steps: &[Vec<(SeqId, Vec<i32>)>]) -> f64 {
    let t0 = Instant::now();
    for step in steps {
        let inputs: Vec<SeqInput<'_>> = step
            .iter()
            .map(|(s, t)| SeqInput {
                seq: *s,
                tokens: t,
                all_logits: false,
            })
            .collect();
        b.decode(&inputs).unwrap();
    }
    t0.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn shapes(name: &str, b: &mut dyn BatchBackend) {
    let prompt = env_usize("TETRAEDER_SHAPES_PROMPT", 402);
    let reps = env_usize("TETRAEDER_SHAPES_REPS", 3);
    let p = toks(99, prompt);
    let dec =
        |k: u32| -> Vec<(SeqId, Vec<i32>)> { (0..k).map(|s| (s, vec![300 + s as i32])).collect() };
    let mut run =
        |label: &str, setup: &dyn Fn(&mut dyn BatchBackend), steps: Vec<Vec<(SeqId, Vec<i32>)>>| {
            let ms: Vec<f64> = (0..reps)
                .map(|_| {
                    setup(b);
                    timed(b, &steps)
                })
                .collect();
            let m = median(ms);
            println!(
                "SHAPE engine={name} shape={label} ms={m:.1} ms_per_prompt_token={:.2}",
                m / prompt as f64
            );
        };
    let alone = |b: &mut dyn BatchBackend| {
        reset(b);
        b.seq_reserve(3, prompt + 64).unwrap();
    };
    let resident = |b: &mut dyn BatchBackend| fill(b, prompt);
    run("alone", &alone, vec![vec![(3, p.clone())]]);
    run("resident", &resident, vec![vec![(3, p.clone())]]);
    for k in 1..=3 {
        let mut mixed = dec(k);
        mixed.push((3, p.clone()));
        run(&format!("mixed{k}"), &resident, vec![mixed]);
        run(
            &format!("split{k}"),
            &resident,
            vec![dec(k), vec![(3, p.clone())]],
        );
        let mut last = vec![(3, p.clone())];
        last.extend(dec(k));
        run(&format!("mixed{k}_prompt_first"), &resident, vec![last]);
    }
    let four = |b: &mut dyn BatchBackend| {
        fill(b, 1);
        b.decode(&[SeqInput {
            seq: 3,
            tokens: &toks(5, RESIDENT),
            all_logits: false,
        }])
        .unwrap();
    };
    // Eight steps per sample: decode steps are short.
    run("decode4x8", &four, vec![dec(4); 8]);
    let one_alone = |b: &mut dyn BatchBackend| {
        reset(b);
        b.seq_reserve(3, RESIDENT + 64).unwrap();
        b.decode(&[SeqInput {
            seq: 3,
            tokens: &toks(5, RESIDENT),
            all_logits: false,
        }])
        .unwrap();
    };
    run("decode1x8_alone", &one_alone, vec![vec![(3, vec![300])]; 8]);
    run("decode1x8_resident", &four, vec![vec![(3, vec![300])]; 8]);
}

fn llama_engine(path: &Path, threads: usize, kv_unified: bool) -> Engine {
    Engine::load(
        path,
        &Placement {
            n_gpu_layers: 0,
            ..Placement::default()
        },
        &ContextConfig {
            n_ctx: CTX as u32,
            n_batch: 512,
            n_ubatch: 512,
            threads: Some(threads as i32),
            threads_batch: None,
            flash_attn: FlashAttn::Auto,
            op_offload: true,
            n_seq_max: SEQS,
            kv_unified,
        },
        None,
    )
    .expect("libllama loads")
}

// hardware: cpu_model
#[test]
fn step_shapes() {
    let Some(path) = std::env::var_os("TETRAEDER_SHAPES_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_SHAPES_GGUF is not set");
        return;
    };
    let threads = env_usize("TETRAEDER_SHAPES_THREADS", 4);
    // What serve runs: one KV pool, decoders and prompt chunks as separate calls on the CPU.
    shapes(
        "llama",
        &mut LlamaBackend::new(llama_engine(&path, threads, true), false),
    );
    // `on_gpu` true on a CPU placement: every step one call.
    shapes(
        "llama_one_call",
        &mut LlamaBackend::new(llama_engine(&path, threads, true), true),
    );
    // llama-server's layout: one KV stream per sequence (not serve's, ADR 0065).
    shapes(
        "llama_streams",
        &mut LlamaBackend::new(llama_engine(&path, threads, false), false),
    );
    let o = DenseOptions {
        n_ctx_pool: CTX,
        n_seq_max: SEQS as usize,
        threads: Some(threads as i32),
        ..DenseOptions::new(BackendChoice::Cpu {
            threads: Some(threads),
        })
    };
    shapes(
        "dense",
        &mut DenseBackend::new(Qwen3::load(&path, &o).expect("dense loads"), false),
    );
    shapes(
        "dense_one_call",
        &mut DenseBackend::new(Qwen3::load(&path, &o).expect("dense loads"), true),
    );
}

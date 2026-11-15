//! Step shapes on the CPU (loop-bench, ADR 0065): the wall time of one `BatchBackend::decode` for
//! the shapes continuous batching produces at 4 sequences, on both backends `serve` uses, so a
//! gap to llama-server can be split into KV width, mixing decoders into a prefill step, and the
//! rest.
//!
//! - `alone`: a prompt alone, the pool empty.
//! - `resident`: the same with three sequences resident below it (wider `n_kv`), not stepping.
//! - `mixed`: `resident` with the three decoding one token each in the same step.
//! - `split`: `mixed` as two calls, the decoders then the prompt.
//! - `decode4`: four decoders, no prompt.
//!
//! Hardware label: `cpu_model` (the GGUF named by `TETRAEDER_SHAPES_GGUF`); skips without it.
//! Env: `TETRAEDER_SHAPES_THREADS` (4), `TETRAEDER_SHAPES_PROMPT` (402), `TETRAEDER_SHAPES_REPS` (3).

use std::path::PathBuf;
use std::time::Instant;
use tetraeder_engine::{ContextConfig, FlashAttn, Placement};
use tetraeder_model::{BackendChoice, DenseOptions};
use tetraeder_scheduler::{BatchBackend, SeqId, SeqInput, load_dense, load_llama};

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
    run("decode4", &four, vec![dec(4)]);
}

// hardware: cpu_model
#[test]
fn step_shapes() {
    let Some(path) = std::env::var_os("TETRAEDER_SHAPES_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_SHAPES_GGUF is not set");
        return;
    };
    let threads = env_usize("TETRAEDER_SHAPES_THREADS", 4);
    let mut llama = load_llama(
        &path,
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
            kv_unified: true,
        },
        None,
    )
    .expect("libllama loads");
    shapes("llama", llama.backend.as_mut());
    drop(llama);
    let o = DenseOptions {
        n_ctx_pool: CTX,
        n_seq_max: SEQS as usize,
        threads: Some(threads as i32),
        ..DenseOptions::new(BackendChoice::Cpu {
            threads: Some(threads),
        })
    };
    let mut dense = load_dense(&path, &o).expect("the dense builder loads");
    shapes("dense", dense.backend.as_mut());
}

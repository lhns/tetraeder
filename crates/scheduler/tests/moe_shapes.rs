//! Step shapes of `--engine moe` against libllama on the same GGUF, CPU only (ADR 0065, loop-bench
//! style, see `step_shapes.rs` for the shape names): the wall time of one `BatchBackend::decode`
//! for the shapes continuous batching produces at 4 sequences.
//!
//! - `alone`, `resident`: a prompt alone, or above three resident sequences.
//! - `mixedK`, `splitK`, `mixedK_prompt_first`: K decoders with the prompt in one step, as two
//!   `decode` calls, or with the prompt first.
//! - `decode4x8`, `decode1x8_alone`, `decode1x8_resident`: eight decode steps of four sequences,
//!   or of one alone or above three resident ones.
//!
//! The model is a synthetic random-weight `qwen3moe` written by the test (8 layers, 32 experts,
//! top-8, Q8_0, 32768-token vocabulary, 8192 context), unless `TETRAEDER_SHAPES_GGUF` names a file.
//! It checks no numbers; it prints `SHAPE engine=... shape=... ms=...` lines. Parity is
//! `moe_backend.rs`'s job.
//!
//! Hardware label: `cpu`. Runs only with `TETRAEDER_MOE_SHAPES` set (it writes a model of about
//! 450 MB and takes minutes). Env: `TETRAEDER_SHAPES_THREADS` (4), `TETRAEDER_SHAPES_PROMPT` (402),
//! `TETRAEDER_SHAPES_REPS` (3), `TETRAEDER_SHAPES_ENGINES` (`llama`, `moe`; both).

#[allow(dead_code)]
#[path = "../../model/tests/common/synth.rs"]
mod synth;

use std::path::{Path, PathBuf};
use std::time::Instant;
use synth::{MoeSpec, quantized_moe_at_with};
use tetraeder_engine::{ContextConfig, Engine, FlashAttn, Placement};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};
use tetraeder_scheduler::{BatchBackend, LlamaBackend, MoeBackend, SeqId, SeqInput};

const SEQS: u32 = 4;
const CTX: usize = 8192;
const VOCAB: usize = 32768;
const RESIDENT: usize = 470;

const SPEC: MoeSpec = MoeSpec {
    layers: 8,
    embd: 512,
    heads: 8,
    heads_kv: 2,
    head_dim: 64,
    experts: 32,
    used: 8,
    ff: 256,
    seed: 20261115,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Token ids inside the byte-level part of the synthetic vocabulary.
fn toks(seed: usize, n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| ((seed * 7919 + i * 104_729) % 250 + 4) as i32)
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
        |k: u32| -> Vec<(SeqId, Vec<i32>)> { (0..k).map(|s| (s, vec![30 + s as i32])).collect() };
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
    // One untimed pass first: page faults of the weights and first graph builds.
    run("warmup", &alone, vec![vec![(3, p.clone())]]);
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
    run("decode1x8_alone", &one_alone, vec![vec![(3, vec![30])]; 8]);
    run("decode1x8_resident", &four, vec![vec![(3, vec![30])]; 8]);
    for (k, v) in b.counters() {
        println!("COUNTER engine={name} {k}={v:.4}");
    }
}

fn llama_engine(path: &Path, threads: usize) -> Engine {
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
            kv_unified: true,
            ..ContextConfig::default()
        },
        None,
    )
    .expect("libllama loads")
}

fn model_path() -> PathBuf {
    if let Some(p) = std::env::var_os("TETRAEDER_SHAPES_GGUF") {
        return PathBuf::from(p);
    }
    // The shape is in the name: a file of another spec is never reused.
    let name = format!(
        "tetraeder-moe-shapes-l{}-e{}-x{}-f{}-v{VOCAB}-c{CTX}.gguf",
        SPEC.layers, SPEC.embd, SPEC.experts, SPEC.ff
    );
    quantized_moe_at_with(
        &SPEC,
        Ftype::Q8_0,
        &std::env::temp_dir().join(name),
        CTX as u32,
        VOCAB,
    )
}

// hardware: cpu
#[test]
fn moe_step_shapes() {
    if std::env::var_os("TETRAEDER_MOE_SHAPES").is_none() {
        eprintln!("skip: TETRAEDER_MOE_SHAPES is not set");
        return;
    }
    let path = model_path();
    let threads = env_usize("TETRAEDER_SHAPES_THREADS", 4);
    let wanted = std::env::var("TETRAEDER_SHAPES_ENGINES").unwrap_or_default();
    let want = |name: &str| wanted.is_empty() || wanted.split(',').any(|w| w.trim() == name);
    if want("llama") {
        shapes(
            "llama",
            &mut LlamaBackend::new(llama_engine(&path, threads), false),
        );
    }
    if want("moe") {
        let o = LoadOptions {
            n_ctx: CTX,
            n_seq_max: SEQS as usize,
            ..LoadOptions::new(BackendChoice::Cpu {
                threads: Some(threads),
            })
        };
        let model = Qwen3Moe::load(&path, &o).expect("moe loads");
        shapes("moe", &mut MoeBackend::new(model, false));
    }
}

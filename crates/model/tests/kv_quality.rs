//! KL acceptance of the low-bit KV formats (ADR 0050 b), builder logits against the builder's own
//! F16 KV. Box only: needs a real model and takes long.
//!
//! Env:
//! - `TETRAEDER_KV_QUALITY_GGUF`: Qwen3-30B-A3B Q4_K_M (required; the test skips without it).
//! - `TETRAEDER_KV_QUALITY_OUT`: result JSON path.
//! - `TETRAEDER_KV_QUALITY_PROMPT`: text for the long-context runs (default
//!   `bench/prompts/exit_prompt.txt`), cycled to 8K and 32K tokens, the last 1024 positions scored.
//! - `TETRAEDER_KV_QUALITY_TEXT`: calibration text of ADR 0048 (wikitext-2 `wiki.train.raw`),
//!   16 chunks of 512 tokens, the second half of each chunk scored; skipped when unset.
//! - `TETRAEDER_KV_QUALITY_SLOTS`: expert slots per layer when the model does not fit (default:
//!   every expert resident).
//!
//! Bounds (mean KL over the full vocabulary, top-1 agreement), at every depth: Q8_0/Q8_0 at most
//! 0.003 and 0.99; Q8_0/Q4_0 at most 0.015 and 0.97.
//!
//! hardware: gpu (uses the first GPU when the build has one, else the CPU; slow either way)

use serde_json::json;
use std::path::PathBuf;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};
use tetraeder_plan::KvFormat;

const UBATCH: usize = 2048;
const SCORED_LONG: usize = 1024;
const CALIB_CTX: usize = 512;
const CALIB_CHUNKS: usize = 16;

/// One scored set: token sequences and the first scored position of each.
struct Set {
    name: String,
    seqs: Vec<(Vec<i32>, usize)>,
}

fn tokenize(gguf: &PathBuf, text: &str) -> Vec<i32> {
    let vocab = Model::load(
        gguf,
        &ModelParams {
            vocab_only: true,
            ..ModelParams::default()
        },
    )
    .expect("libllama reads the vocab");
    vocab.tokenize(text, vocab.add_bos(), false).unwrap()
}

/// Logits of the positions `from..` of `tokens`: the prefix runs in ubatches that keep only the
/// last row, the tail in one all-position step per ubatch.
fn scored_logits(m: &mut Qwen3Moe, tokens: &[i32], from: usize) -> Vec<f32> {
    m.reset();
    for c in tokens[..from].chunks(UBATCH) {
        m.eval_last(c).unwrap();
    }
    let mut out = Vec::new();
    for c in tokens[from..].chunks(UBATCH) {
        out.extend(m.eval(c).unwrap());
    }
    out
}

/// KL(p || q) of one position in f64 from two logit rows, and whether the argmax agrees.
fn kl_row(p: &[f32], q: &[f32]) -> (f64, bool) {
    let lse = |x: &[f32]| {
        let mx = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        mx + x
            .iter()
            .map(|&v| (f64::from(v) - mx).exp())
            .sum::<f64>()
            .ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    let kl = p
        .iter()
        .zip(q)
        .map(|(&a, &b)| {
            let lpa = f64::from(a) - lp;
            lpa.exp() * (lpa - (f64::from(b) - lq))
        })
        .sum();
    let top = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    (kl, top(p) == top(q))
}

fn load(gguf: &PathBuf, n_ctx: usize, kv: KvFormat) -> Qwen3Moe {
    let backend = if Backend::gpu_available() {
        BackendChoice::Gpu { index: 0 }
    } else {
        BackendChoice::Cpu { threads: None }
    };
    let mut o = LoadOptions::new(backend);
    o.n_ctx = n_ctx;
    o.kv = kv;
    if let Some(n) = std::env::var("TETRAEDER_KV_QUALITY_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        o.expert_slots = Some(n);
        o.ram_slots = Some(128);
        o.miss = tetraeder_model::MissPolicy::Cpu;
    }
    Qwen3Moe::load(gguf, &o).expect("builder loads")
}

#[test]
#[ignore = "box only: needs the 30B GGUF and a GPU; run with --ignored in release"]
fn kv_quality() {
    let Some(gguf) = std::env::var_os("TETRAEDER_KV_QUALITY_GGUF").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_KV_QUALITY_GGUF is not set");
        return;
    };
    let prompt = std::env::var_os("TETRAEDER_KV_QUALITY_PROMPT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts/exit_prompt.txt")
        });
    let base = tokenize(
        &gguf,
        &std::fs::read_to_string(prompt).expect("prompt file"),
    );
    let cycled = |n: usize| -> Vec<i32> { base.iter().copied().cycle().take(n).collect() };

    let mut sets = Vec::new();
    for depth in [8192usize, 32768] {
        sets.push(Set {
            name: format!("exit_prompt_{}k", depth / 1024),
            seqs: vec![(cycled(depth), depth - SCORED_LONG)],
        });
    }
    if let Some(text) = std::env::var_os("TETRAEDER_KV_QUALITY_TEXT") {
        let toks = tokenize(
            &gguf,
            &std::fs::read_to_string(text).expect("calibration text"),
        );
        let seqs = toks
            .chunks_exact(CALIB_CTX)
            .take(CALIB_CHUNKS)
            .map(|c| (c.to_vec(), CALIB_CTX / 2))
            .collect::<Vec<_>>();
        assert_eq!(seqs.len(), CALIB_CHUNKS, "calibration text too short");
        sets.push(Set {
            name: "calibration_512".into(),
            seqs,
        });
    }
    let n_ctx = sets
        .iter()
        .flat_map(|s| &s.seqs)
        .map(|(t, _)| t.len())
        .max()
        .unwrap();

    // The F16 reference logits of every scored position, kept on the host.
    let mut m = load(&gguf, n_ctx, KvFormat::F16);
    let n_vocab = m.n_vocab();
    let reference: Vec<Vec<Vec<f32>>> = sets
        .iter()
        .map(|s| {
            s.seqs
                .iter()
                .map(|(t, from)| scored_logits(&mut m, t, *from))
                .collect()
        })
        .collect();
    drop(m);

    let mut results = Vec::new();
    for kv in [KvFormat::Q8_0, KvFormat::Q8_0_Q4_0] {
        let (kl_max, top1_min) = if kv == KvFormat::Q8_0 {
            (0.003, 0.99)
        } else {
            (0.015, 0.97)
        };
        let mut m = load(&gguf, n_ctx, kv);
        for (set, refs) in sets.iter().zip(&reference) {
            let (mut kl_sum, mut agree, mut n) = (0.0f64, 0usize, 0usize);
            for ((t, from), p) in set.seqs.iter().zip(refs) {
                let q = scored_logits(&mut m, t, *from);
                for (pr, qr) in p.chunks(n_vocab).zip(q.chunks(n_vocab)) {
                    let (kl, same) = kl_row(pr, qr);
                    kl_sum += kl;
                    agree += usize::from(same);
                    n += 1;
                }
            }
            let (kl, top1) = (kl_sum / n as f64, agree as f64 / n as f64);
            eprintln!(
                "{kv} {}: mean KL {kl:.5}, top-1 {top1:.4} over {n} positions",
                set.name
            );
            results.push(json!({
                "kv": kv.to_string(),
                "set": set.name,
                "positions": n,
                "mean_kl": kl,
                "top1_agreement": top1,
                "bound_kl": kl_max,
                "bound_top1": top1_min,
                "pass": kl <= kl_max && top1 >= top1_min,
            }));
        }
    }
    if let Some(p) = std::env::var_os("TETRAEDER_KV_QUALITY_OUT") {
        let out = json!({"model": gguf.display().to_string(), "reference": "builder F16/F16 KV", "results": results});
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
    for r in &results {
        assert!(r["pass"].as_bool().unwrap(), "{r}");
    }
}

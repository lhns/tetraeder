//! KL acceptance of the low-bit KV formats (ADR 0050 b), builder logits against the builder's own
//! F16 KV, plus informative rows that say where a failure comes from.
//!
//! Env (box test `kv_quality`):
//! - `TETRAEDER_KV_QUALITY_GGUF`: Qwen3-30B-A3B Q4_K_M (required; the test skips without it).
//! - `TETRAEDER_KV_QUALITY_OUT`: result JSON path.
//! - `TETRAEDER_KV_QUALITY_PROMPT`: text for the long-context runs (default
//!   `bench/prompts/exit_prompt.txt`, about 25.8K tokens). See [`depth_seq`] for how it becomes
//!   8K and 32K tokens; the last 1024 positions are scored, the same text at both depths.
//! - `TETRAEDER_KV_QUALITY_TEXT`: calibration text of ADR 0048 (wikitext-2 `wiki.train.raw`),
//!   16 chunks of 512 tokens, the second half of each chunk scored; skipped when unset
//!   (`scripts/kv_quality.sh` always sets it).
//! - `TETRAEDER_KV_QUALITY_SLOTS`: expert slots per layer when the model does not fit (default:
//!   every expert resident).
//!
//! Any perturbation of this model, even F16 rounding, costs about 0.007 mean KL against the
//! reference (20261110b), so the gate is on the cost above that floor (ADR 0050 note
//! 2026-11-10b, bounds provisional): the floor row is the builder's F16 cache with the rotation
//! forced on, and each gating row (the builder's default rotation) is paired with it position by
//! position. At every depth, Q8_0/Q8_0 excess mean KL at most 0.003 and excess top-1 loss at most
//! 0.01; Q8_0/Q4_0 at most 0.015 and 0.03. The ADR's absolute bounds are reported (`abs_pass`).
//! Informative rows: the reference again (determinism) and at half the ubatch, both formats with
//! the rotation off, libllama with F16 and both formats (its default rotation), and each engine's
//! formats against that engine's own F16 row.
//!
//! Hardware labels (AGENTS rule 9) per test.

use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, FlashAttn, Model, ModelParams};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};
use tetraeder_plan::KvFormat;

const UBATCH: usize = 2048;
const SCORED_LONG: usize = 1024;
/// The shortest depth; its sequence is the start of the prompt, unchanged.
const SHORT: usize = 8192;
const DEPTHS: [usize; 2] = [SHORT, 32768];
const CALIB_CTX: usize = 512;
const CALIB_CHUNKS: usize = 16;
/// N-gram length of the repeat check, and the largest share of scored positions whose n-gram
/// may already occur in the context (the exit prompt quotes itself a little: about 2%).
const REPEAT_N: usize = 8;
const REPEAT_MAX: f64 = 0.1;

/// The `depth` tokens of a long-context set. The scored block (the last `SCORED_LONG`) must be
/// text the model has not seen before in that sequence: cycling a prompt shorter than `depth`
/// puts it in the repeat, where the model copies and every KV format scores KL near 0 (the
/// 20261110 run scored a verbatim repeat at 32K). So the sequence ends with the first `SHORT`
/// tokens of the prompt, and the rest of the prompt fills the front, cycled so that it ends with
/// the prompt's last token. At `SHORT` this is the plain prompt prefix; at every depth the last
/// `SHORT` tokens, and so the scored block and its local context, are the same.
fn depth_seq(prompt: &[i32], depth: usize) -> Result<Vec<i32>, String> {
    if depth < SHORT || prompt.len() <= SHORT {
        return Err(format!(
            "need a prompt longer than {SHORT} tokens and a depth of at least {SHORT}: prompt {}, depth {depth}",
            prompt.len()
        ));
    }
    let (head, rest) = prompt.split_at(SHORT);
    let n = depth - SHORT;
    let start = (rest.len() - n % rest.len()) % rest.len();
    let mut s: Vec<i32> = rest.iter().cycle().skip(start).take(n).copied().collect();
    s.extend_from_slice(head);
    Ok(s)
}

/// Share of the positions `from..` whose `n`-gram ending there already occurs in `seq[..from]`:
/// how much of the scored text the model can copy from its context.
fn repeat_fraction(seq: &[i32], from: usize, n: usize) -> f64 {
    let seen: HashSet<&[i32]> = seq[..from].windows(n).collect();
    let hits = (from..seq.len())
        .filter(|&j| j + 1 >= n && seen.contains(&seq[j + 1 - n..=j]))
        .count();
    hits as f64 / (seq.len() - from) as f64
}

/// FNV-1a over the token ids (little-endian i32), as hex (as `exit_regime.rs`).
fn tokens_hash(tokens: &[i32]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in tokens.iter().flat_map(|t| t.to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

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

/// Logits of the positions `from..` of `tokens`: the prefix runs in ubatches of `ubatch` that keep
/// only the last row, the tail in one all-position step per ubatch.
fn scored_logits(m: &mut Qwen3Moe, tokens: &[i32], from: usize, ubatch: usize) -> Vec<f32> {
    m.reset();
    for c in tokens[..from].chunks(ubatch) {
        m.eval_last(c).unwrap();
    }
    let mut out = Vec::new();
    for c in tokens[from..].chunks(ubatch) {
        out.extend(m.eval(c).unwrap());
    }
    out
}

/// The same for libllama, in a fresh context (its default rotation of a quantized cache).
fn libllama_logits(
    model: &Arc<Model>,
    kv: KvFormat,
    n_ctx: usize,
    tokens: &[i32],
    from: usize,
) -> Vec<f32> {
    let ty = |f| tetraeder_model::convert::ggml_type(f).expect("ggml type of the KV format");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as i32;
    let mut ctx = Context::new(
        model,
        &ContextParams {
            n_ctx: n_ctx as u32,
            n_batch: UBATCH as u32,
            n_ubatch: UBATCH as u32,
            n_threads: threads,
            n_threads_batch: threads,
            flash_attn: FlashAttn::On,
            type_k: ty(kv.k),
            type_v: ty(kv.v),
            ..ContextParams::default()
        },
    )
    .expect("libllama context");
    let mut batch = Batch::new(UBATCH);
    let mut out = Vec::new();
    for (ci, c) in tokens.chunks(UBATCH).enumerate() {
        batch.clear();
        for (i, &t) in c.iter().enumerate() {
            let pos = ci * UBATCH + i;
            batch.push(t, pos as i32, 0, pos >= from).unwrap();
        }
        ctx.decode(&batch).unwrap();
        for i in (0..c.len()).filter(|&i| ci * UBATCH + i >= from) {
            out.extend_from_slice(ctx.logits_ith(i as i32).unwrap());
        }
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

fn gpu() -> bool {
    Backend::gpu_available()
}

fn load(gguf: &PathBuf, n_ctx: usize, kv: KvFormat, rotation: Option<bool>) -> Qwen3Moe {
    let backend = if gpu() {
        BackendChoice::Gpu { index: 0 }
    } else {
        BackendChoice::Cpu { threads: None }
    };
    let mut o = LoadOptions::new(backend);
    o.n_ctx = n_ctx;
    o.kv = kv;
    o.kv_rotation = rotation;
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

/// Per-position KL and argmax agreement of one variant against a reference, over one set.
struct Scored {
    kl: Vec<f64>,
    agree: Vec<bool>,
}

fn score(refs: &[Vec<f32>], got: &[Vec<f32>], n_vocab: usize) -> Scored {
    let (mut kl, mut agree) = (Vec::new(), Vec::new());
    for (p, q) in refs.iter().zip(got) {
        assert_eq!(p.len(), q.len(), "scored rows differ");
        for (pr, qr) in p.chunks(n_vocab).zip(q.chunks(n_vocab)) {
            let (k, same) = kl_row(pr, qr);
            kl.push(k);
            agree.push(same);
        }
    }
    Scored { kl, agree }
}

fn top1(s: &Scored) -> f64 {
    s.agree.iter().filter(|&&a| a).count() as f64 / s.agree.len() as f64
}

/// Mean, median, p99 and max KL and the top-1 agreement.
fn summary(s: &Scored) -> Value {
    let n = s.kl.len();
    let mean = s.kl.iter().sum::<f64>() / n as f64;
    let mut kls = s.kl.clone();
    kls.sort_by(f64::total_cmp);
    json!({
        "positions": n,
        "mean_kl": mean,
        "median_kl": kls[n / 2],
        "p99_kl": kls[(n * 99 / 100).min(n - 1)],
        "max_kl": kls[n - 1],
        "top1_agreement": top1(s),
    })
}

/// Cost of a variant above the floor, both scored against the same reference on the same
/// positions: the mean per-position KL difference, its standard error, and the extra share of
/// positions that lose the reference's argmax.
fn excess(got: &Scored, floor: &Scored) -> (f64, f64, f64) {
    assert_eq!(got.kl.len(), floor.kl.len(), "scored positions differ");
    let d: Vec<f64> = got.kl.iter().zip(&floor.kl).map(|(a, b)| a - b).collect();
    let n = d.len() as f64;
    let mean = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt(), top1(floor) - top1(got))
}

/// hardware: cpu. The excess is the paired per-position difference.
#[test]
fn cpu_excess_is_the_paired_difference() {
    let floor = Scored {
        kl: vec![0.01, 0.02, 0.0, 0.03],
        agree: vec![true, true, true, false],
    };
    let got = Scored {
        kl: vec![0.02, 0.03, 0.01, 0.04],
        agree: vec![true, false, true, false],
    };
    let (mean, se, top) = excess(&got, &floor);
    assert!((mean - 0.01).abs() < 1e-12, "{mean}");
    assert!(se < 1e-9, "{se}");
    assert!((top - 0.25).abs() < 1e-12, "{top}");
    let got = Scored {
        kl: vec![0.03, 0.02, 0.0, 0.03],
        agree: floor.agree.clone(),
    };
    let (mean, se, top) = excess(&got, &floor);
    assert!((mean - 0.005).abs() < 1e-12, "{mean}");
    assert!((se - 0.005).abs() < 1e-12, "{se}");
    assert_eq!(top, 0.0);
}

/// hardware: cpu. At the short depth the sequence is the prompt prefix, as in the 20261110 run.
#[test]
fn cpu_short_depth_is_the_prompt_prefix() {
    let prompt: Vec<i32> = (0..25_848).collect();
    assert_eq!(depth_seq(&prompt, SHORT).unwrap(), prompt[..SHORT]);
}

/// hardware: cpu. Deeper sequences end with the short one, so every depth scores the same text.
#[test]
fn cpu_long_depth_ends_with_the_short_sequence() {
    let prompt: Vec<i32> = (0..25_848).collect();
    let long = depth_seq(&prompt, 32_768).unwrap();
    assert_eq!(long.len(), 32_768);
    assert_eq!(long[32_768 - SHORT..], prompt[..SHORT]);
    // The front is the rest of the prompt, cycled, ending with its last token.
    assert_eq!(long[32_768 - SHORT - 1], 25_847);
}

/// hardware: cpu. The scored block never repeats earlier context at any depth; plain cycling
/// (the 20261110 harness) scored a verbatim repeat at 32K.
#[test]
fn cpu_scored_block_is_not_a_repeat() {
    let prompt: Vec<i32> = (0..25_848).collect();
    for depth in DEPTHS {
        let s = depth_seq(&prompt, depth).unwrap();
        assert_eq!(
            repeat_fraction(&s, depth - SCORED_LONG, REPEAT_N),
            0.0,
            "{depth}"
        );
    }
    let cycled: Vec<i32> = prompt.iter().copied().cycle().take(32_768).collect();
    assert_eq!(
        repeat_fraction(&cycled, 32_768 - SCORED_LONG, REPEAT_N),
        1.0
    );
}

/// hardware: cpu. A prompt no longer than the short depth cannot fill the sets.
#[test]
fn cpu_short_prompt_is_an_error() {
    let prompt: Vec<i32> = (0..SHORT as i32).collect();
    assert!(depth_seq(&prompt, 32_768).is_err());
    assert!(depth_seq(&(0..30_000).collect::<Vec<_>>(), 4096).is_err());
}

/// What runs against the reference: the builder with a rotation setting (`None` = its default),
/// or libllama.
enum Engine {
    Builder(Option<bool>),
    Libllama,
}

impl Engine {
    fn name(&self) -> &'static str {
        match self {
            Engine::Builder(_) => "builder",
            Engine::Libllama => "libllama",
        }
    }
}

/// What a row is for.
enum Role {
    /// The builder's F16 cache with the rotation forced on: as close to the reference as the
    /// arithmetic allows, so its KL is the floor any perturbation reaches. Gating rows are
    /// paired with it.
    Floor,
    /// Gating on the cost above the floor (ADR 0050 note 2026-11-10b): excess mean KL at most,
    /// excess top-1 loss at most. The ADR's absolute bounds (mean KL at most, top-1 at least)
    /// are reported next to it, not gated.
    Gate {
        excess: (f64, f64),
        absolute: (f64, f64),
    },
    Info,
}

struct Variant {
    kv: KvFormat,
    engine: Engine,
    ubatch: usize,
    role: Role,
    /// Keep the logits as this engine's own baseline: later rows of the engine are also scored
    /// against it (same engine, same batching, only the KV format differs).
    base: bool,
}

/// hardware: gpu (uses the first GPU when the build has one, else the CPU; slow either way).
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

    let mut sets = Vec::new();
    let mut set_info = Vec::new();
    for depth in DEPTHS {
        let s = depth_seq(&base, depth).unwrap();
        let from = depth - SCORED_LONG;
        let rep = repeat_fraction(&s, from, REPEAT_N);
        let name = format!("exit_prompt_{}k", depth / 1024);
        eprintln!("{name}: {depth} tokens, scored from {from}, repeated {REPEAT_N}-grams {rep:.4}");
        assert!(
            rep <= REPEAT_MAX,
            "{name}: the scored block repeats its context ({rep})"
        );
        set_info.push(
            json!({"set": name, "tokens": depth, "scored_from": from, "repeat_fraction": rep}),
        );
        sets.push(Set {
            name,
            seqs: vec![(s, from)],
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
        set_info.push(json!({"set": "calibration_512", "tokens": CALIB_CTX, "chunks": CALIB_CHUNKS, "scored_from": CALIB_CTX / 2}));
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
    let mut m = load(&gguf, n_ctx, KvFormat::F16, None);
    let n_vocab = m.n_vocab();
    let reference: Vec<Vec<Vec<f32>>> = sets
        .iter()
        .map(|s| {
            s.seqs
                .iter()
                .map(|(t, from)| scored_logits(&mut m, t, *from, UBATCH))
                .collect()
        })
        .collect();
    drop(m);

    let v = |kv, engine, ubatch, role, base| Variant {
        kv,
        engine,
        ubatch,
        role,
        base,
    };
    let gate = |excess, absolute| Role::Gate { excess, absolute };
    // The floor runs first: the gating rows need it. Host memory: the reference, one baseline
    // and the current row, about 3.7 GB each (6144 positions of 151936 F32 logits).
    let (f16, q8, q4) = (KvFormat::F16, KvFormat::Q8_0, KvFormat::Q8_0_Q4_0);
    use Engine::{Builder, Libllama};
    let variants = vec![
        v(f16, Builder(Some(true)), UBATCH, Role::Floor, true),
        v(
            q8,
            Builder(None),
            UBATCH,
            gate((0.003, 0.01), (0.003, 0.99)),
            false,
        ),
        v(
            q4,
            Builder(None),
            UBATCH,
            gate((0.015, 0.03), (0.015, 0.97)),
            false,
        ),
        // The reference again (determinism) and at another ubatch (batching dependence).
        v(f16, Builder(None), UBATCH, Role::Info, false),
        v(f16, Builder(None), UBATCH / 2, Role::Info, false),
        v(q8, Builder(Some(false)), UBATCH, Role::Info, false),
        v(q4, Builder(Some(false)), UBATCH, Role::Info, false),
        v(f16, Libllama, UBATCH, Role::Info, true),
        v(q8, Libllama, UBATCH, Role::Info, false),
        v(q4, Libllama, UBATCH, Role::Info, false),
    ];
    let mut libllama: Option<Arc<Model>> = None;
    let mut results = Vec::new();
    // Per set: the floor row's per-position scores against the reference.
    let mut floor: Option<Vec<Scored>> = None;
    // An engine's own F16 baseline: its name, a label and its logits.
    let mut own_base: Option<(&str, String, Vec<Vec<Vec<f32>>>)> = None;
    for var in &variants {
        let (kv, engine_name, ubatch) = (var.kv, var.engine.name(), var.ubatch);
        if own_base.as_ref().is_some_and(|b| b.0 != engine_name) {
            own_base = None;
        }
        let (rotation, got): (Value, Vec<Vec<Vec<f32>>>) = match &var.engine {
            Engine::Builder(rot) => {
                let mut m = load(&gguf, n_ctx, kv, *rot);
                let (k, v) = m.kv_rotated();
                let got = sets
                    .iter()
                    .map(|s| {
                        s.seqs
                            .iter()
                            .map(|(t, from)| scored_logits(&mut m, t, *from, ubatch))
                            .collect()
                    })
                    .collect();
                (json!({"k": k, "v": v}), got)
            }
            Engine::Libllama => {
                let model: &Arc<Model> = libllama.get_or_insert_with(|| {
                    Arc::new(
                        Model::load(
                            &gguf,
                            &ModelParams {
                                n_gpu_layers: if gpu() { -1 } else { 0 },
                                ..ModelParams::default()
                            },
                        )
                        .expect("libllama loads the model"),
                    )
                });
                let got = sets
                    .iter()
                    .map(|s| {
                        s.seqs
                            .iter()
                            .map(|(t, from)| libllama_logits(model, kv, n_ctx, t, *from))
                            .collect()
                    })
                    .collect();
                (json!("default"), got)
            }
        };
        let label = format!("{engine_name} {kv} rot {rotation} ubatch {ubatch}");
        let row = |s: &Scored, set: &str, reference: &str| {
            let mut r = summary(s);
            eprintln!(
                "{label} {set} vs {reference}: mean KL {:.5}, median {:.6}, p99 {:.4}, max {:.3}, top-1 {:.4}",
                r["mean_kl"].as_f64().unwrap(),
                r["median_kl"].as_f64().unwrap(),
                r["p99_kl"].as_f64().unwrap(),
                r["max_kl"].as_f64().unwrap(),
                r["top1_agreement"].as_f64().unwrap(),
            );
            let o = r.as_object_mut().unwrap();
            o.insert("engine".into(), json!(engine_name));
            o.insert("kv".into(), json!(kv.to_string()));
            o.insert("rotation".into(), rotation.clone());
            o.insert("ubatch".into(), json!(ubatch));
            o.insert("set".into(), json!(set));
            o.insert("reference".into(), json!(reference));
            o.insert("gating".into(), json!(false));
            r
        };
        let mut scored = Vec::new();
        for (i, ((set, refs), got)) in sets.iter().zip(&reference).zip(&got).enumerate() {
            let s = score(refs, got, n_vocab);
            let mut r = row(&s, &set.name, "builder f16");
            if let Role::Gate {
                excess: (ex_kl, ex_top1),
                absolute: (abs_kl, abs_top1),
            } = var.role
            {
                let f = &floor
                    .as_ref()
                    .expect("the floor row runs before gating rows")[i];
                let (d, se, d_top1) = excess(&s, f);
                let (kl, agree) = (
                    r["mean_kl"].as_f64().unwrap(),
                    r["top1_agreement"].as_f64().unwrap(),
                );
                let pass = d <= ex_kl && d_top1 <= ex_top1;
                // The bound lies within two standard errors of the excess: one run cannot tell.
                let marginal = (d - ex_kl).abs() <= 2.0 * se;
                eprintln!(
                    "  gating: excess KL {d:.5} +- {se:.5} (bound {ex_kl}), excess top-1 loss {d_top1:.4} (bound {ex_top1}): {}{}",
                    if pass { "pass" } else { "FAIL" },
                    if marginal { " (marginal)" } else { "" },
                );
                let o = r.as_object_mut().unwrap();
                o.insert("gating".into(), json!(true));
                o.insert(
                    "floor_mean_kl".into(),
                    json!(f.kl.iter().sum::<f64>() / f.kl.len() as f64),
                );
                o.insert("floor_top1".into(), json!(top1(f)));
                o.insert("excess_kl".into(), json!(d));
                o.insert("excess_kl_se".into(), json!(se));
                o.insert("excess_top1_loss".into(), json!(d_top1));
                o.insert("bound_excess_kl".into(), json!(ex_kl));
                o.insert("bound_excess_top1_loss".into(), json!(ex_top1));
                o.insert("marginal".into(), json!(marginal));
                o.insert("pass".into(), json!(pass));
                o.insert("abs_bound_kl".into(), json!(abs_kl));
                o.insert("abs_bound_top1".into(), json!(abs_top1));
                o.insert("abs_pass".into(), json!(kl <= abs_kl && agree >= abs_top1));
            }
            results.push(r);
            if let Some((_, name, logits)) = &own_base {
                results.push(row(&score(&logits[i], &got[i], n_vocab), &set.name, name));
            }
            scored.push(s);
        }
        if matches!(var.role, Role::Floor) {
            floor = Some(scored);
        }
        if var.base {
            own_base = Some((engine_name, label.clone(), got));
        }
    }
    if let Some(p) = std::env::var_os("TETRAEDER_KV_QUALITY_OUT") {
        let out = json!({
            "model": gguf.display().to_string(),
            "reference": "builder F16/F16 KV, rotation off, ubatch 2048 (rows with another \"reference\" are same-engine diagnostics)",
            "backend": if gpu() { "gpu" } else { "cpu" },
            "prompt": {"tokens": base.len(), "fnv1a64": tokens_hash(&base)},
            "sets": set_info,
            "results": results,
        });
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
    for r in results.iter().filter(|r| r["gating"] == json!(true)) {
        assert!(r["pass"].as_bool().unwrap(), "{r}");
    }
}

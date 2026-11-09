//! Tests of the safe libllama wrapper. They need the Qwen3-0.6B Q8_0 GGUF (`cargo xtask
//! fetch-model`); without it they print a skip reason and pass,
//! unless `TETRAEDER_REQUIRE_MODELS` is set (CI): then they fail. Model dir: `TETRAEDER_MODELS_DIR`,
//! else the nearest `.hf_cache` above this crate.

use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, Ftype, GgmlType, LlamaError, Model, ModelParams, QuantizeParams,
    Sampler, ThreadPool, Token, buffer_type, gguf_tensor_types, quantize,
};

const MODEL: &str = "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";

fn model_path() -> Option<PathBuf> {
    let dir = match std::env::var_os("TETRAEDER_MODELS_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|a| a.join(".hf_cache"))
            .find(|p| p.is_dir())?,
    };
    let p = dir.join(MODEL);
    p.is_file().then_some(p)
}

macro_rules! require_model {
    () => {
        match model_path() {
            Some(p) => p,
            None => {
                let test = std::thread::current().name().unwrap_or("?").to_string();
                let reason = format!("{MODEL} not found (run `cargo xtask fetch-model`)");
                assert!(
                    std::env::var_os("TETRAEDER_REQUIRE_MODELS").is_none(),
                    "{test}: {reason} (TETRAEDER_REQUIRE_MODELS is set)"
                );
                eprintln!("skip: {test}: {reason}");
                return;
            }
        }
    };
}

fn cpu_params() -> ModelParams {
    ModelParams {
        n_gpu_layers: 0,
        ..ModelParams::default()
    }
}

fn small_ctx() -> ContextParams {
    ContextParams {
        n_ctx: 256,
        n_batch: 128,
        n_ubatch: 128,
        n_threads: 2,
        n_threads_batch: 2,
        ..ContextParams::default()
    }
}

fn greedy_tokens(model: &Arc<Model>, prompt: &[Token], n: usize) -> Vec<Token> {
    let mut ctx = Context::new(model, &small_ctx()).unwrap();
    let mut batch = Batch::new(128);
    let mut smpl = Sampler::greedy();
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, i == prompt.len() - 1).unwrap();
    }
    let mut out = Vec::new();
    for pos in (prompt.len() as i32..).take(n) {
        ctx.decode(&batch).unwrap();
        let tok = smpl.sample(&mut ctx, -1).unwrap();
        out.push(tok);
        batch.clear();
        batch.push(tok, pos, 0, true).unwrap();
    }
    out
}

#[test]
fn vocab_only_tokenize_roundtrip() {
    let path = require_model!();
    let params = ModelParams {
        vocab_only: true,
        ..cpu_params()
    };
    let model = Model::load(&path, &params).unwrap();
    let toks = model.tokenize("Hello world", false, true).unwrap();
    assert!(!toks.is_empty());
    let mut bytes = Vec::new();
    for &t in &toks {
        bytes.extend(model.token_to_piece(t, true).unwrap());
    }
    assert_eq!(String::from_utf8(bytes).unwrap(), "Hello world");
    assert!(model.n_vocab() > 100_000);
}

#[test]
fn tokenize_handles_long_text() {
    let path = require_model!();
    let model = Model::load(
        &path,
        &ModelParams {
            vocab_only: true,
            ..cpu_params()
        },
    )
    .unwrap();
    // Longer than the first buffer guess, so the resize protocol runs.
    let text = "a ".repeat(500);
    let toks = model.tokenize(&text, false, false).unwrap();
    assert!(toks.len() >= 500);
}

#[test]
fn full_load_decode_has_finite_logits() {
    let path = require_model!();
    let model = Arc::new(Model::load(&path, &cpu_params()).unwrap());
    assert!(model.n_layer() > 0 && model.n_params() > 0 && model.size_bytes() > 0);
    assert!(!model.desc().is_empty());
    let prompt = model
        .tokenize("The capital of France is", true, true)
        .unwrap();
    let mut ctx = Context::new(&model, &small_ctx()).unwrap();
    assert_eq!(ctx.n_ctx(), 256);
    let mut batch = Batch::new(128);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, i == prompt.len() - 1).unwrap();
    }
    ctx.decode(&batch).unwrap();
    ctx.synchronize();
    let n_vocab = model.n_vocab() as usize;
    let logits = ctx.logits_ith(-1).unwrap();
    assert_eq!(logits.len(), n_vocab);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(ctx.seq_pos_max(0), prompt.len() as i32 - 1);
    ctx.memory_clear();
    assert_eq!(ctx.seq_pos_max(0), -1);
    assert!(ctx.perf().n_p_eval > 0);
}

#[test]
fn logits_of_unrequested_output_is_an_error() {
    let path = require_model!();
    let model = Arc::new(Model::load(&path, &cpu_params()).unwrap());
    let mut ctx = Context::new(&model, &small_ctx()).unwrap();
    let mut batch = Batch::new(4);
    batch.push(model.bos(), 0, 0, true).unwrap();
    ctx.decode(&batch).unwrap();
    assert!(ctx.logits_ith(5).is_err());
}

#[test]
fn greedy_is_deterministic_across_contexts_and_threadpool() {
    let path = require_model!();
    let model = Arc::new(Model::load(&path, &cpu_params()).unwrap());
    let prompt = model.tokenize("Once upon a time", true, true).unwrap();
    let a = greedy_tokens(&model, &prompt, 6);
    let b = greedy_tokens(&model, &prompt, 6);
    assert_eq!(a, b);

    let pool = Arc::new(ThreadPool::new(2).unwrap());
    let mut ctx = Context::new(&model, &small_ctx()).unwrap();
    ctx.attach_threadpool(Arc::clone(&pool));
    ctx.set_threads(2, 2);
    let mut batch = Batch::new(128);
    for (i, &t) in prompt.iter().enumerate() {
        batch.push(t, i as i32, 0, i == prompt.len() - 1).unwrap();
    }
    ctx.decode(&batch).unwrap();
    let first = Sampler::greedy().sample(&mut ctx, -1).unwrap();
    assert_eq!(first, a[0]);
}

#[test]
fn buffer_override_to_cpu_loads() {
    let path = require_model!();
    let params = ModelParams {
        buffer_overrides: vec![(r"blk\.0\..*".into(), buffer_type("CPU").unwrap())],
        ..cpu_params()
    };
    let model = Arc::new(Model::load(&path, &params).unwrap());
    let prompt = model.tokenize("Hi", true, true).unwrap();
    assert!(!greedy_tokens(&model, &prompt, 1).is_empty());
}

#[test]
fn unknown_buffer_type_lists_known_names() {
    assert!(buffer_type("NOPE").is_none());
    let names = tetraeder_ggml_sys::llama::buffer_type_names();
    assert!(names.iter().any(|n| n == "CPU"));
}

#[test]
fn tensor_split_longer_than_max_devices_is_an_error() {
    let path = PathBuf::from("unused.gguf");
    let params = ModelParams {
        // Fails before libllama is called, so no model needs to load.
        tensor_split: vec![1.0; 4096],
        ..cpu_params()
    };
    let e = Model::load(&path, &params).unwrap_err();
    assert!(matches!(e, LlamaError::Invalid(m) if m.contains("tensor_split")));
}

#[test]
fn tensor_split_within_limit_is_padded_and_loads() {
    let path = require_model!();
    let params = ModelParams {
        tensor_split: vec![1.0],
        ..cpu_params()
    };
    Model::load(&path, &params).unwrap();
}

#[test]
fn quantize_applies_per_tensor_type_override() {
    let path = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("q4_0.gguf");
    let mut qp = QuantizeParams::new(Ftype::Q4_0);
    qp.allow_requantize = true;
    qp.nthread = 2;
    // Qwen3-0.6B ties the embeddings, so there is no output.weight; token_embd.weight is the one.
    qp.tensor_types = vec![(r"token_embd\.weight".into(), GgmlType::Q8_0)];
    quantize(&path, &dst, &qp).unwrap();

    let types: std::collections::HashMap<_, _> =
        gguf_tensor_types(&dst).unwrap().into_iter().collect();
    assert_eq!(types["token_embd.weight"], "q8_0");
    assert_eq!(types["blk.0.attn_q.weight"], "q4_0");
    assert_eq!(types["blk.3.ffn_gate.weight"], "q4_0");
    assert_eq!(types["blk.0.attn_norm.weight"], "f32");
}

#[test]
fn invalid_quantize_override_regex_is_an_error() {
    let path = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let mut qp = QuantizeParams::new(Ftype::Q4_0);
    qp.allow_requantize = true;
    qp.tensor_types = vec![("(unclosed".into(), GgmlType::Q8_0)];
    let e = quantize(&path, dir.path().join("x.gguf"), &qp).unwrap_err();
    assert!(matches!(e, LlamaError::Quantize(_)));
}

#[test]
fn gguf_listing_of_a_missing_file_is_an_error() {
    assert!(matches!(
        gguf_tensor_types("no/such/file.gguf"),
        Err(LlamaError::Gguf(_))
    ));
}

#[test]
fn seq_rm_rolls_the_cache_back_and_decoding_resumes() {
    let path = require_model!();
    let model = Arc::new(Model::load(&path, &cpu_params()).unwrap());
    let prompt = model
        .tokenize("The capital of France is", true, true)
        .unwrap();
    let mut ctx = Context::new(&model, &small_ctx()).unwrap();
    let mut batch = Batch::new(128);
    let n = prompt.len();
    let feed = |ctx: &mut Context, batch: &mut Batch, from: usize| {
        batch.clear();
        for (i, &t) in prompt.iter().enumerate().skip(from) {
            batch.push(t, i as i32, 0, i == n - 1).unwrap();
        }
        ctx.decode(batch).unwrap();
        ctx.logits_ith(-1).unwrap().to_vec()
    };
    let first = feed(&mut ctx, &mut batch, 0);
    assert_eq!(ctx.seq_pos_max(0), n as i32 - 1);
    assert!(ctx.seq_rm(0, 2, -1));
    assert_eq!(ctx.seq_pos_max(0), 1);
    let again = feed(&mut ctx, &mut batch, 2);
    assert_eq!(ctx.seq_pos_max(0), n as i32 - 1);
    assert_eq!(first.len(), again.len());
    let top = |l: &[f32]| {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(top(&first), top(&again));
}

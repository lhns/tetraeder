//! Parity against llama.cpp (AGENTS rule 2), Qwen3-0.6B Q8_0 on CPU.
//!
//! `matches_llama_simple`: our greedy text equals what the vendored `llama-simple` prints.
//! `matches_bench_golden`: tokens equal exactly and top-k logits agree within a tolerance.
//! Both skip without the model or the golden files. `cargo xtask golden` records the files.

mod common;

use serde::Deserialize;
use std::ops::ControlFlow;
use std::path::PathBuf;
use tetraeder_bench::golden::{self, Golden, Tolerance, TopLogit};
use tetraeder_engine::{ContextConfig, Engine, GenerateRequest, Placement, Prompt};

const TOP_K: usize = 5;

#[derive(Deserialize)]
struct SimpleGolden {
    model: String,
    n_predict: u32,
    cases: Vec<SimpleCase>,
}

#[derive(Deserialize)]
struct SimpleCase {
    id: String,
    prompt: String,
    text: String,
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/golden")
}

fn simple_path() -> PathBuf {
    golden_dir().join("qwen3-0.6b-q8_0.llama-simple.json")
}

fn bench_path() -> PathBuf {
    golden_dir().join("qwen3-0.6b-q8_0-cpu.json")
}

fn load_simple() -> Option<SimpleGolden> {
    let text = std::fs::read_to_string(simple_path()).ok()?;
    Some(serde_json::from_str(&text).expect("parse llama-simple golden"))
}

fn engine(path: &std::path::Path) -> Engine {
    let cfg = ContextConfig {
        n_ctx: 512,
        threads: Some(2),
        ..ContextConfig::default()
    };
    Engine::load(path, &Placement::cpu(), &cfg, None).unwrap()
}

fn request(prompt: &str, n: u32) -> GenerateRequest {
    // add_special and parse_special on, stop at end-of-generation: what llama-simple does.
    let mut r = GenerateRequest::new(Prompt::Text(prompt.into()), n);
    r.top_k_logits = TOP_K;
    r
}

fn bench_case(eng: &mut Engine, c: &SimpleCase, n: u32) -> golden::Case {
    let rep = eng
        .generate(&request(&c.prompt, n), |_| ControlFlow::Continue(()))
        .unwrap();
    golden::Case {
        id: c.id.clone(),
        prompt_tokens: rep.prompt_tokens.iter().map(|&t| t as u32).collect(),
        tokens: rep.tokens.iter().map(|&t| t as u32).collect(),
        top: rep
            .top
            .iter()
            .map(|t| {
                t.iter()
                    .map(|&(id, logit)| TopLogit {
                        id: id as u32,
                        logit,
                    })
                    .collect()
            })
            .collect(),
    }
}

#[test]
fn matches_llama_simple() {
    let path = require_model!();
    let (Some(simple), Ok(bench)) = (load_simple(), Golden::load(&bench_path())) else {
        eprintln!("skip: golden files not recorded (run `cargo xtask golden`)");
        return;
    };
    let mut eng = engine(&path);
    for c in &simple.cases {
        let expected_prompt = bench
            .cases
            .iter()
            .find(|b| b.id == c.id)
            .unwrap_or_else(|| panic!("case {} missing in bench golden", c.id));
        let prompt_tokens = eng.tokenize(&c.prompt, true, true).unwrap();
        let ids: Vec<u32> = prompt_tokens.iter().map(|&t| t as u32).collect();
        assert_eq!(
            ids, expected_prompt.prompt_tokens,
            "prompt tokens of {}",
            c.id
        );

        let rep = eng
            .generate(&request(&c.prompt, simple.n_predict), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        // llama-simple prints the prompt token by token, then the generated pieces.
        let ours = format!(
            "{}{}",
            eng.detokenize(&rep.prompt_tokens).unwrap(),
            rep.text
        );
        assert_eq!(ours, c.text, "case {}", c.id);
    }
}

#[test]
fn matches_bench_golden() {
    let path = require_model!();
    let (Some(simple), Ok(expected)) = (load_simple(), Golden::load(&bench_path())) else {
        eprintln!("skip: golden files not recorded (run `cargo xtask golden`)");
        return;
    };
    let mut eng = engine(&path);
    let cases = simple
        .cases
        .iter()
        .map(|c| bench_case(&mut eng, c, simple.n_predict))
        .collect();
    let actual = Golden {
        cases,
        ..expected.clone()
    };
    // Tokens must match exactly; logits get 0.05 absolute because the golden was recorded
    // with a native-ISA ggml build and CI builds portable (GGML_NATIVE=OFF): different SIMD
    // kernels sum in a different order.
    let cmp = expected.compare(&actual, Tolerance { logit_abs: 0.05 });
    assert!(cmp.ok(), "mismatches: {:#?}", cmp.mismatches);
    eprintln!(
        "golden: {} positions, max logit diff {}",
        cmp.positions, cmp.max_logit_diff
    );
}

/// Writes the bench-format golden from our own greedy run. Driven by `cargo xtask golden`
/// (sets `TETRAEDER_RECORD_GOLDEN=1`), after the llama-simple golden exists.
#[test]
#[ignore = "records golden data; run through `cargo xtask golden`"]
fn record_golden() {
    if std::env::var_os("TETRAEDER_RECORD_GOLDEN").is_none() {
        eprintln!("skip: TETRAEDER_RECORD_GOLDEN is not set");
        return;
    }
    let path = require_model!();
    let simple = load_simple().expect("llama-simple golden missing; run `cargo xtask golden`");
    let mut eng = engine(&path);
    let cases = simple
        .cases
        .iter()
        .map(|c| bench_case(&mut eng, c, simple.n_predict))
        .collect();
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_owned(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        );
    let g = Golden {
        schema: golden::SCHEMA,
        model: simple.model,
        engine: format!("tetraeder@{sha} on libllama b11976"),
        backend: "cpu".into(),
        top_k: TOP_K,
        cases,
    };
    g.save(&bench_path()).unwrap();
    eprintln!("wrote {}", bench_path().display());
}

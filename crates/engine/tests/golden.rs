//! Parity against llama.cpp (AGENTS rule 2), Qwen3-0.6B Q8_0 on CPU.
//!
//! `matches_llama_simple`: our greedy text equals what the vendored `llama-simple` prints.
//! `matches_bench_golden`: tokens equal exactly and top-k logits agree within a tolerance.
//! Both skip without the model or the golden files (fail if `TETRAEDER_REQUIRE_MODELS` is set). `cargo xtask golden` records the files.

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

/// `<variant>-<os>`, the suffix of the golden file names. Float results differ per CPU tier (ADR 0052)
/// and per OS (libm differs in the last ulp, which can flip a near-tied greedy token). The variant is
/// `TETRAEDER_GGML_CPU_VARIANT` (default `sse42`, the tier the first golden was recorded on), the OS
/// `std::env::consts::OS`; `TETRAEDER_GOLDEN_OS` overrides it to compare against another OS's file.
fn tag() -> String {
    let variant = std::env::var("TETRAEDER_GGML_CPU_VARIANT").unwrap_or_else(|_| "sse42".into());
    let os = std::env::var("TETRAEDER_GOLDEN_OS").unwrap_or_else(|_| std::env::consts::OS.into());
    format!("{variant}-{os}")
}

fn simple_path() -> PathBuf {
    golden_dir().join(format!("qwen3-0.6b-q8_0.llama-simple-{}.json", tag()))
}

fn bench_path() -> PathBuf {
    golden_dir().join(format!("qwen3-0.6b-q8_0-cpu-{}.json", tag()))
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
        common::skip_or_fail(&format!(
            "golden files for {} not recorded (run `cargo xtask golden`)",
            tag()
        ));
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
        common::skip_or_fail(&format!(
            "golden files for {} not recorded (run `cargo xtask golden`)",
            tag()
        ));
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
    // A flipped greedy token is a near-tie if the two logits at that position are close: print both top-k
    // lists, which tells a last-ulp difference from a real bug.
    for m in &cmp.mismatches {
        if let golden::Mismatch::TokenDiverged { case, pos, .. } = m {
            let top = |g: &Golden| {
                g.cases.iter().find(|c| &c.id == case).map(|c| {
                    c.top[*pos]
                        .iter()
                        .map(|t| (t.id, t.logit))
                        .collect::<Vec<_>>()
                })
            };
            eprintln!(
                "diverged {case} at {pos}\n  expected top-k {:?}\n  actual   top-k {:?}",
                top(&expected),
                top(&actual)
            );
        }
    }
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

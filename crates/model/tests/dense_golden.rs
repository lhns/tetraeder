//! The dense builder on Qwen3-0.6B Q8_0 against the greedy golden of `engine/tests/golden.rs`
//! (ADR 0066): tokens equal exactly and the top-k logits within that file's tolerance.
//!
//! Hardware label: `cpu_*`. The model is found through `TETRAEDER_MODELS_DIR`, else the nearest
//! `.hf_cache` above this crate; the test skips without it unless `TETRAEDER_REQUIRE_MODELS` is
//! set, then it fails.

mod common;

use common::{argmax, top_k};
use std::path::PathBuf;
use tetraeder_bench::golden::{Case, Golden, Mismatch, Tolerance, TopLogit};
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};

const MODEL: &str = "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const TOP_K: usize = 5;

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

/// `<variant>-<os>`, as in `engine/tests/golden.rs`.
fn tag() -> String {
    let variant = std::env::var("TETRAEDER_GGML_CPU_VARIANT").unwrap_or_else(|_| "sse42".into());
    let os = std::env::var("TETRAEDER_GOLDEN_OS").unwrap_or_else(|_| std::env::consts::OS.into());
    format!("{variant}-{os}")
}

fn skip_or_fail(reason: &str) {
    let test = std::thread::current().name().unwrap_or("?").to_string();
    assert!(
        std::env::var_os("TETRAEDER_REQUIRE_MODELS").is_none(),
        "{test}: {reason} (TETRAEDER_REQUIRE_MODELS is set)"
    );
    eprintln!("skip: {test}: {reason}");
}

/// Greedy decode of the golden's prompt on one sequence, with the top-k logits of every step.
fn run_case(m: &mut Qwen3, expected: &Case) -> Case {
    let prompt: Vec<i32> = expected.prompt_tokens.iter().map(|&t| t as i32).collect();
    m.reset();
    let mut logits = m.eval_last(&prompt).unwrap();
    let (mut tokens, mut top) = (Vec::new(), Vec::new());
    for _ in 0..expected.tokens.len() {
        let t = argmax(&logits);
        top.push(
            top_k(&logits, TOP_K)
                .into_iter()
                .map(|id| TopLogit {
                    id: id as u32,
                    logit: logits[id],
                })
                .collect::<Vec<_>>(),
        );
        tokens.push(t as u32);
        logits = m.eval_last(&[t as i32]).unwrap();
    }
    Case {
        id: expected.id.clone(),
        prompt_tokens: expected.prompt_tokens.clone(),
        tokens,
        top,
    }
}

/// hardware: cpu, model. Tokens equal exactly; logits get 0.05 absolute because the golden was
/// recorded with a native-ISA ggml build and CI builds portable.
#[test]
fn cpu_dense_matches_bench_golden() {
    let Some(path) = model_path() else {
        skip_or_fail(&format!(
            "{MODEL} not found (run `cargo xtask fetch-model`)"
        ));
        return;
    };
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/golden")
        .join(format!("qwen3-0.6b-q8_0-cpu-{}.json", tag()));
    let Ok(expected) = Golden::load(&golden) else {
        skip_or_fail(&format!("golden {} not recorded", golden.display()));
        return;
    };
    let opts = DenseOptions {
        n_ctx_pool: 512,
        threads: Some(2),
        ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
    };
    let mut m = Qwen3::load(&path, &opts).unwrap();
    assert!(m.hparams().tied_output, "Qwen3-0.6B ties its output");
    let actual = Golden {
        cases: expected.cases.iter().map(|c| run_case(&mut m, c)).collect(),
        ..expected.clone()
    };
    let cmp = expected.compare(&actual, Tolerance { logit_abs: 0.05 });
    for mm in &cmp.mismatches {
        if let Mismatch::TokenDiverged { case, pos, .. } = mm {
            eprintln!("diverged {case} at {pos}");
        }
    }
    assert!(cmp.ok(), "mismatches: {:#?}", cmp.mismatches);
    eprintln!(
        "dense golden: {} positions, max logit diff {}",
        cmp.positions, cmp.max_logit_diff
    );
}

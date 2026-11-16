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
use tetraeder_model::{BackendChoice, DenseHparams, DenseOptions, Gguf, Qwen3};

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

/// hardware: cpu, model. Two CPU ranks (`tp-2`, ADR 0070): the tokens equal the golden's except
/// where the golden's top two logits are a near tie (twice `tol_batch`, ADR 0043), reported;
/// logits within the golden's tolerance up to there. Rank slices of a Q8_0 model cut whole
/// blocks (`ffn_down` 96 blocks, `wo` 32 blocks per rank pair), and the tied head is a vocab
/// slice of the token embedding.
#[test]
fn cpu_dense_tp2_matches_bench_golden() {
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
    let n_layer = DenseHparams::from_gguf(&Gguf::open(&path).unwrap())
        .unwrap()
        .attn
        .n_layer;
    let opts = DenseOptions {
        n_ctx_pool: 512,
        threads: Some(2),
        ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
    };
    let mut m = Qwen3::load_sharded(&path, &common::shapes::tp(n_layer, 2), &opts).unwrap();
    assert_eq!(m.n_ranks(), 2);
    let actual = Golden {
        cases: expected.cases.iter().map(|c| run_case(&mut m, c)).collect(),
        ..expected.clone()
    };
    // Q8_0 quantizes every activation block; a partial sum in another order moves some blocks
    // across a rounding step, and 28 layers carry that to about 1% of a logit (measured: 0.10 on
    // 13.6). So 0.3 absolute here, against 0.05 for the unsharded run.
    let cmp = expected.compare(&actual, Tolerance { logit_abs: 0.3 });
    for mm in &cmp.mismatches {
        let Mismatch::TokenDiverged {
            case,
            pos,
            actual: got,
            ..
        } = mm
        else {
            panic!("mismatch: {mm:?}");
        };
        let top = &expected
            .cases
            .iter()
            .find(|c| &c.id == case)
            .expect("case of the golden")
            .top[*pos];
        let gap = top[0].logit - top[1].logit;
        let bound = 2.0 * (1e-2 * top[0].logit.abs());
        eprintln!("near tie in {case} at {pos}: gap {gap:e}, bound {bound:e}");
        assert!(
            *got == top[1].id && gap <= bound,
            "{case} diverged at {pos} to {got}, golden top two {:?}",
            &top[..2]
        );
    }
    eprintln!(
        "dense tp-2 golden: {} positions, max logit diff {}",
        cmp.positions, cmp.max_logit_diff
    );
}

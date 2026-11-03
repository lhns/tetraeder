//! Speculative decoding on the real builder (ADR 0043), on the K-quant fixture of
//! `crates/model/tests/multipass.rs`: top-8 of 32 experts, Q4_K_M, resident and with streamed
//! experts (`MissPolicy::Cpu`). Prompt and generated tokens stay below 512 KV entries: from 512
//! the CPU flash attention takes a split-KV path for single rows that a verify step does not.
//!
//! Bit equality is asserted where ggml's kernels are batch-invariant (no AVX2). With AVX2 the
//! multi-row step takes other kernels for some ops (llamafile sgemm for F32 from 2 rows, ADR
//! 0043) and its logits differ in the last bits (seen on the CI runners, 2026-11-03), so there
//! the logits must agree to a relative 1e-4 and speculative output may leave plain greedy only
//! at a near-tie of the plain run (top-2 margin below `NEAR_TIE`).
//!
//! Hardware label (AGENTS rule 9): `cpu_*` tests need no GPU.

#[allow(dead_code)]
#[path = "../../model/tests/common/synth.rs"]
mod synth;

use std::path::{Path, PathBuf};
use synth::{MoeSpec, quantized_moe};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};
use tetraeder_scheduler::spec::{
    DraftLen, DraftModel, Drafter, MAX_VERIFY, NgramDrafter, Sampling, SpecError, Speculator,
    argmax,
};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_GEN: usize = 48;

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "spec")
}

fn load(path: &Path, streaming: bool) -> Qwen3Moe {
    let mut o = LoadOptions::new(CPU);
    o.n_ctx = 256;
    if streaming {
        o.expert_slots = Some(8);
        o.ram_slots = Some(12);
        o.miss = MissPolicy::Cpu;
    }
    Qwen3Moe::load(path, &o).unwrap()
}

fn prompt() -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect()
}

/// Prefills the prompt and returns the history (prompt plus the first greedy token).
fn prefill(m: &mut Qwen3Moe) -> Vec<i32> {
    let p = prompt();
    m.rewind(0).unwrap();
    let l = m.eval_last(&p).unwrap();
    let mut h = p;
    h.push(argmax(&l) as i32);
    h
}

const NEAR_TIE: f32 = 1e-3;

/// ggml's CPU kernels give the same bits for one row and for a few rows.
fn batch_invariant_kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Plain greedy history by one-row steps, plus the top-2 margin of the logits that chose each
/// history position (0 for the prompt); leaves the model rewound to the prefill.
fn plain_with_margins(m: &mut Qwen3Moe) -> (Vec<i32>, Vec<f32>) {
    m.rewind(0).unwrap();
    let p = prompt();
    let mut margins = vec![0.0; N_PROMPT];
    let mut l = m.eval_last(&p).unwrap();
    let mut h = p;
    while h.len() < N_PROMPT + 1 + N_GEN + MAX_VERIFY {
        let top = argmax(&l);
        let second = l
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != top)
            .map(|(_, &x)| x)
            .fold(f32::MIN, f32::max);
        margins.push(l[top] - second);
        h.push(top as i32);
        l = m.eval(&[top as i32]).unwrap();
    }
    m.rewind(N_PROMPT).unwrap();
    (h, margins)
}

/// `got` equals plain greedy `want` over their common length, or (without batch-invariant
/// kernels) first differs where plain greedy was a near-tie.
fn assert_agrees(got: &[i32], want: &[i32], margins: &[f32], what: &str) {
    let n = got.len().min(want.len());
    let Some(i) = (0..n).find(|&i| got[i] != want[i]) else {
        return;
    };
    assert!(
        !batch_invariant_kernels() && margins[i] < NEAR_TIE,
        "{what}: differs from plain greedy at {i} (plain top-2 margin {})",
        margins[i]
    );
    eprintln!(
        "{what}: leaves plain greedy at a near-tie, position {i}, margin {}",
        margins[i]
    );
}

struct Oracle(Vec<i32>);

impl Drafter for Oracle {
    fn draft(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        let end = (history.len() + max).min(self.0.len());
        Ok(self.0[history.len()..end].to_vec())
    }
    fn method(&self) -> &'static str {
        "oracle"
    }
}

struct Random(u64);

impl Drafter for Random {
    fn draft(&mut self, _: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        Ok((0..max)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 33) % 256) as i32
            })
            .collect())
    }
    fn method(&self) -> &'static str {
        "random"
    }
}

fn bits(l: &[f32]) -> Vec<u32> {
    l.iter().map(|x| x.to_bits()).collect()
}

fn batch_invariance(streaming: bool) {
    let mut m = load(&fixture(), streaming);
    prefill(&mut m);
    let toks: Vec<i32> = (0..MAX_VERIFY as i32).map(|i| (i * 11 + 3) % 256).collect();
    let nv = m.n_vocab();
    let batched = m.eval(&toks).unwrap();
    m.rewind(N_PROMPT).unwrap();
    for (i, &t) in toks.iter().enumerate() {
        let one = m.eval(&[t]).unwrap();
        let row = &batched[i * nv..(i + 1) * nv];
        if batch_invariant_kernels() {
            assert_eq!(
                bits(&one),
                bits(row),
                "row {i} of the {MAX_VERIFY}-row step differs from its one-row step"
            );
        } else {
            let scale = one.iter().fold(0f32, |a, x| a.max(x.abs()));
            let diff = one
                .iter()
                .zip(row)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(
                diff <= 1e-4 * scale,
                "row {i}: max difference {diff} of {scale}"
            );
        }
    }
}

#[test]
fn cpu_batch_invariance_resident() {
    batch_invariance(false);
}

#[test]
fn cpu_batch_invariance_streaming() {
    batch_invariance(true);
}

fn speculative_equals_plain(streaming: bool) {
    let mut m = load(&fixture(), streaming);
    let (want, margins) = plain_with_margins(&mut m);
    let lens = [
        DraftLen::Fixed(2),
        DraftLen::Fixed(6),
        DraftLen::Adaptive { max: 6 },
    ];
    for len in lens {
        let mut drafters: Vec<(Box<dyn Drafter>, bool)> = vec![
            (Box::new(Oracle(want.clone())), true),
            (Box::new(Random(9)), false),
            (Box::new(NgramDrafter::default()), false),
        ];
        for (d, all_accepted) in &mut drafters {
            let mut h = prefill(&mut m);
            let mut sp = Speculator::new(len, Sampling::GREEDY, None);
            sp.generate(&mut m, Some(d.as_mut()), &mut h, N_GEN)
                .unwrap();
            let what = format!("{len:?} {} (streaming {streaming})", d.method());
            assert_agrees(&h, &want, &margins, &what);
            assert_eq!(m.n_past(), h.len() - 1);
            let st = sp.stats();
            assert_eq!(st.emitted, st.accepted + st.steps);
            if *all_accepted {
                assert_eq!(st.accepted, st.drafted);
                assert!(st.steps < N_GEN as u64, "the oracle must save steps");
            }
            m.rewind(N_PROMPT).unwrap();
        }
    }
}

#[test]
fn cpu_speculative_equals_plain_resident() {
    speculative_equals_plain(false);
}

#[test]
fn cpu_speculative_equals_plain_streaming() {
    speculative_equals_plain(true);
}

#[test]
fn cpu_draft_model_equals_plain_and_is_accepted() {
    let path = fixture();
    let mut m = load(&path, false);
    let (want, margins) = plain_with_margins(&mut m);
    let mut d = DraftModel::load(&path, 256, 0, 2, 0.0).unwrap();
    d.check_vocab(m.n_vocab()).unwrap();
    assert!(d.check_vocab(m.n_vocab() + 1).is_err());
    let mut h = prefill(&mut m);
    let mut sp = Speculator::new(DraftLen::Fixed(4), Sampling::GREEDY, None);
    sp.generate(&mut m, Some(&mut d), &mut h, N_GEN).unwrap();
    assert_agrees(&h, &want, &margins, "draft model");
    let st = sp.stats();
    assert!(st.drafted > 0);
    let rate = st.accepted as f64 / st.drafted as f64;
    assert!(rate > 0.5, "acceptance {rate}");
}

//! Speculative decoding on the real builder (ADR 0043), on the K-quant fixture of
//! `crates/model/tests/multipass.rs`: top-8 of 32 experts, Q4_K_M, resident and with streamed
//! experts (`MissPolicy::Cpu`). Prompt and generated tokens stay below 512 KV entries: from 512
//! the CPU flash attention takes a split-KV path for single rows that a verify step does not.
//!
//! The fixture's F32 router may go through llamafile sgemm for 2+ rows on AVX machines, which
//! would break the bitwise batch-invariance test there; it asserts bit equality anyway.
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

/// Plain greedy continuation by one-row steps; leaves the model rewound to the prefill.
fn plain(m: &mut Qwen3Moe) -> Vec<i32> {
    let mut h = prefill(m);
    let mut sp = Speculator::new(DraftLen::Fixed(0), Sampling::GREEDY, None);
    sp.generate(m, None, &mut h, N_GEN).unwrap();
    m.rewind(N_PROMPT).unwrap();
    h
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
        assert_eq!(
            bits(&one),
            bits(&batched[i * nv..(i + 1) * nv]),
            "row {i} of the {MAX_VERIFY}-row step differs from its one-row step"
        );
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
    let want = plain(&mut m);
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
            let n = want.len();
            assert_eq!(
                &h[..n],
                &want[..],
                "{len:?} {} (streaming {streaming})",
                d.method()
            );
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
    let want = plain(&mut m);
    let mut d = DraftModel::load(&path, 256, 0, 2, 0.0).unwrap();
    d.check_vocab(m.n_vocab()).unwrap();
    assert!(d.check_vocab(m.n_vocab() + 1).is_err());
    let mut h = prefill(&mut m);
    let mut sp = Speculator::new(DraftLen::Fixed(4), Sampling::GREEDY, None);
    sp.generate(&mut m, Some(&mut d), &mut h, N_GEN).unwrap();
    assert_eq!(&h[..want.len()], &want[..]);
    let st = sp.stats();
    assert!(st.drafted > 0);
    let rate = st.accepted as f64 / st.drafted as f64;
    assert!(rate > 0.5, "acceptance {rate}");
}

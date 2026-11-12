//! ADR 0064 steps 0 and 1 on the K-quant fixture of `prefetch.rs` (top-8 of 32 experts,
//! Q4_K_M, `MissPolicy::Cpu`): a scheduler sized for one segment gives the logits of the
//! whole-step size bit for bit, and `seg_pairs` counts the RAM pairs computed in segments.
//!
//! Hardware label: `cpu_*`, no GPU needed.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe, StepProfile};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
/// 12 layers: the whole-step size is 128 x 13 x 4 nodes, above the segment size of 1,024 x 4.
/// With the 4 layers of `REALISH` both sizes are 4,096 and the test would compare a size with
/// itself.
const SPEC: MoeSpec = MoeSpec {
    layers: 12,
    ..MoeSpec::REALISH
};
const N_PROMPT: usize = 24;
const N_DECODE: usize = 6;

fn opts(whole_step_sched: bool) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(8),
        ram_slots: Some(12),
        miss: MissPolicy::Cpu,
        whole_step_sched,
        ..LoadOptions::new(CPU)
    }
}

/// Prefill, greedy decodes; every eval's logits.
fn run(m: &mut Qwen3Moe) -> (Vec<Vec<f32>>, StepProfile) {
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let l = m.eval(&prompt).unwrap();
    let mut next = argmax(&l[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
    let mut all = vec![l];
    m.take_profile();
    for _ in 0..N_DECODE {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        all.push(l);
    }
    let prof = m.profile();
    (all, prof)
}

/// hardware: cpu. The segment-sized scheduler changes no logit.
#[test]
fn cpu_segment_sized_scheduler_is_bitwise_equal() {
    let path = quantized_moe(&SPEC, Ftype::Q4KM, "q4km-schedsize");
    let (want, _) = run(&mut Qwen3Moe::load(&path, &opts(true)).unwrap());
    let (got, _) = run(&mut Qwen3Moe::load(&path, &opts(false)).unwrap());
    assert_eq!(want.len(), got.len());
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "eval {i} differs"
        );
    }
}

/// hardware: cpu. Every RAM pair is computed in a segment or a group graph, so `seg_pairs` is
/// at most `cpu_pairs`; with 12 RAM slots over top-8 most layers have a single chunk, so most
/// pairs are in segments.
#[test]
fn cpu_seg_pairs_is_part_of_cpu_pairs() {
    let path = quantized_moe(&SPEC, Ftype::Q4KM, "q4km-schedsize");
    let mut m = Qwen3Moe::load(&path, &opts(false)).unwrap();
    let (_, p) = run(&mut m);
    eprintln!("cpu_pairs {} seg_pairs {}", p.cpu_pairs, p.seg_pairs);
    assert!(p.cpu_pairs > 0);
    assert!(p.seg_pairs > 0);
    assert!(p.seg_pairs <= p.cpu_pairs);
}

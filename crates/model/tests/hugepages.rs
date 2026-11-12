//! `LoadOptions::ram_hugepages` (ADR 0035 addendum) only advises the kernel about the RAM tier's
//! pages, so the logits must equal the run without it bit for bit. Fixture of `admit.rs`.
//!
//! Hardware label (AGENTS rule 9): `cpu_*`, no GPU; the advice itself is Linux only and the test
//! reports what it got elsewhere.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

fn opts(hugepages: bool) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(8),
        ram_slots: Some(12),
        miss: MissPolicy::Cpu,
        ram_hugepages: hugepages,
        ..LoadOptions::new(CPU)
    }
}

/// Prefill and four decode steps, every eval's logits.
fn run(m: &mut Qwen3Moe) -> Vec<Vec<f32>> {
    let prompt: Vec<i32> = (0..24).map(|i| (i * 37 + 5) % 256).collect();
    let l = m.eval(&prompt).unwrap();
    let mut next = argmax(&l[23 * m.n_vocab()..]) as i32;
    let mut all = vec![l];
    for _ in 0..4 {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        all.push(l);
    }
    all
}

/// hardware: cpu.
#[test]
fn cpu_ram_hugepages_keep_logits_bitwise() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-hugepages");
    let mut off = Qwen3Moe::load(&path, &opts(false)).unwrap();
    assert!(off.ram_hugepages().is_none());
    let mut on = Qwen3Moe::load(&path, &opts(true)).unwrap();
    let advice = on.ram_hugepages().expect("the option was on");
    eprintln!("ram_hugepages: {advice:?}");
    // The fixture's tier can be smaller than one huge page, which leaves nothing to advise.
    if cfg!(target_os = "linux") && advice.bytes > 0 {
        assert!(advice.applied, "madvise failed with errno {}", advice.errno);
    }
    let (want, got) = (run(&mut off), run(&mut on));
    assert_eq!(want.len(), got.len());
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "eval {i} differs"
        );
    }
}

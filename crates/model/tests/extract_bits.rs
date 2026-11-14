//! Bit-equality guard for the attention extraction (phase 4, C1.5): logits of every prompt
//! position, 12 greedy `eval_last` steps and the final KV snapshot must not change by one bit.
//! `TETRAEDER_BITS_RECORD=<file>` writes them, `TETRAEDER_BITS_CHECK=<file>` asserts equality,
//! neither prints `skip:` and passes. The record depends on the CPU and OS and is not committed.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use common::*;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

fn capture(m: &mut Qwen3Moe, out: &mut Vec<u8>) {
    m.reset();
    let all = m.eval(&PROMPT).unwrap();
    out.extend(all.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    m.reset();
    let mut next = argmax(&m.eval_last(&PROMPT).unwrap()) as i32;
    for _ in 0..N_GREEDY {
        let l = m.eval_last(&[next]).unwrap();
        out.extend(l.iter().flat_map(|x| x.to_bits().to_le_bytes()));
        next = argmax(&l) as i32;
    }
    let snap = m.kv_snapshot().unwrap();
    out.extend((snap.len() as u64).to_le_bytes());
    out.extend(snap);
}

fn record() -> Vec<u8> {
    let mut out = Vec::new();
    let mut m = Qwen3Moe::load(fixture(), &LoadOptions::new(CPU)).unwrap();
    capture(&mut m, &mut out);
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8");
    for streamed in [false, true] {
        let o = LoadOptions {
            kv: KvFormat::Q8_0,
            kv_rotation: Some(true),
            ..LoadOptions::new(CPU)
        };
        let o = if streamed {
            LoadOptions {
                expert_slots: Some(8),
                ram_slots: Some(16),
                miss: MissPolicy::Cpu,
                ..o
            }
        } else {
            o
        };
        let mut m = Qwen3Moe::load(&path, &o).unwrap();
        capture(&mut m, &mut out);
    }
    out
}

/// hardware: cpu.
#[test]
fn cpu_extraction_is_bit_equal() {
    let rec = std::env::var_os("TETRAEDER_BITS_RECORD");
    let chk = std::env::var_os("TETRAEDER_BITS_CHECK");
    if rec.is_none() && chk.is_none() {
        eprintln!("skip: set TETRAEDER_BITS_RECORD or TETRAEDER_BITS_CHECK");
        return;
    }
    let got = record();
    if let Some(f) = rec {
        std::fs::write(f, &got).unwrap();
    }
    if let Some(f) = chk {
        let want = std::fs::read(f).unwrap();
        assert_eq!(want.len(), got.len(), "record length");
        let at = want.iter().zip(&got).position(|(a, b)| a != b);
        assert!(at.is_none(), "first differing byte at {at:?}");
    }
}

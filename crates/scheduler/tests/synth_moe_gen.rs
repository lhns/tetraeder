//! Writes the synthetic MoE model of `moe_shapes.rs` (8 layers, 32 experts, top-8, Q8_0, 32768-token
//! vocabulary, 8192-token context) to the path in `TETRAEDER_SYNTH_MOE_OUT`. The Phase 4 CPU
//! rehearsal (`.github/workflows/phase4-rehearsal.yml`) serves it as the Qwen3-30B-A3B stand-in on
//! `--engine moe`: it is the one MoE model here whose context fits four sequences.
//!
//! Hardware label: `cpu`. Without the variable the test skips; it writes a file of about 450 MB.

#[allow(dead_code)]
#[path = "../../model/tests/common/synth.rs"]
mod synth;

use std::path::PathBuf;
use synth::{MoeSpec, quantized_moe_at_with};
use tetraeder_ggml_sys::llama::Ftype;

const SPEC: MoeSpec = MoeSpec {
    layers: 8,
    embd: 512,
    heads: 8,
    heads_kv: 2,
    head_dim: 64,
    experts: 32,
    used: 8,
    ff: 256,
    seed: 20261115,
};

#[test]
fn write_synthetic_moe() {
    let Some(out) = std::env::var_os("TETRAEDER_SYNTH_MOE_OUT").map(PathBuf::from) else {
        eprintln!("skip: TETRAEDER_SYNTH_MOE_OUT is not set");
        return;
    };
    let p = quantized_moe_at_with(&SPEC, Ftype::Q8_0, &out, 8192, 32768);
    eprintln!("wrote {}", p.display());
    assert!(p.is_file());
}

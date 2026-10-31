//! Model description and tensor index of the synthetic fixture (hardware: none).

use std::path::PathBuf;
use tetraeder_model::{ModelFile, hparams::EXPERT_TENSORS};
use tetraeder_plan::{Format, UnitId};

fn open() -> ModelFile {
    ModelFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf"),
    )
    .expect("fixture (cargo xtask tiny-moe)")
}

#[test]
fn hparams_from_gguf() {
    let f = open();
    let h = &f.hparams;
    assert_eq!(
        (h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim),
        (2, 64, 4, 2, 16)
    );
    assert_eq!((h.n_expert, h.n_expert_used, h.n_ff_exp), (8, 2, 32));
    assert_eq!(h.n_vocab, 260);
    assert!(!h.tied_output);
}

#[test]
fn model_desc_counts() {
    let d = open().model_desc().unwrap();
    assert_eq!(
        (d.layers, d.heads, d.kv_heads, d.experts, d.experts_used),
        (2, 4, 2, 8, 2)
    );
    assert_eq!(d.embed.params, 260 * 64);
    // norm + q + k + v + o + q_norm + k_norm
    assert_eq!(d.attn.params, 64 + 64 * 64 + 2 * 64 * 32 + 64 * 64 + 2 * 16);
    // ffn_norm + router
    assert_eq!(d.mlp.params, 64 + 8 * 64);
    let e = d.expert.unwrap();
    assert_eq!(e.params, 3 * 32 * 64);
    assert_eq!(e.flops_per_token, 2.0 * e.params as f64);
    // output_norm + output.weight
    assert_eq!(d.head.params, 64 + 260 * 64);
    assert_eq!(d.attn.source_format, Format::F16);
    assert_eq!(d.expert.unwrap().source_format, Format::F16);
    assert_eq!(d.mlp.source_format, Format::F32);
    // K and V of 2 heads * 16, F16.
    assert_eq!(d.kv_bytes_per_token_per_layer, 2 * 2 * 32);
    // The description serializes and names the units the plan addresses.
    assert!(toml_roundtrip(&d));
    assert_eq!(UnitId::Expert(1, 7).to_string(), "layer.1.expert.7");
}

fn toml_roundtrip(d: &tetraeder_plan::ModelDesc) -> bool {
    let s = toml::to_string(d).unwrap();
    toml::from_str::<tetraeder_plan::ModelDesc>(&s).unwrap() == *d
}

#[test]
fn expert_slabs_tile_the_stacked_tensors() {
    let f = open();
    for il in 0..2 {
        for name in EXPERT_TENSORS {
            let t = f.layer_tensor(il, name).unwrap();
            assert_eq!(t.ne[2], 8);
            let mut end = t.offset;
            for e in 0..8 {
                let (off, len) = t.slab(e).unwrap();
                assert_eq!(off, end, "slabs are contiguous");
                assert_eq!(len, t.nbytes / 8);
                end = off + len;
            }
            assert_eq!(end, t.offset + t.nbytes);
            assert!(t.slab(8).is_none());
        }
    }
    let s = f.expert_slabs(1, 3).unwrap();
    assert_eq!(s.bytes(), 3 * 32 * 64 * 2);
    assert!(f.expert_slabs(0, 8).is_err());
    assert!(f.expert_slabs(2, 0).is_err());
}

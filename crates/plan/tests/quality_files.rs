//! The measured quality profiles in bench/quality/ (written by scripts/quality/calibrate.py) load
//! through the Rust type and validate against their model's shape (ADR 0048). Hardware: none.

use std::path::PathBuf;
use tetraeder_plan::{Format, ModelDesc, QualityProfile, UnitSize};

fn size() -> UnitSize {
    UnitSize {
        params: 1,
        flops_per_token: 0.0,
        source_format: Format::BF16,
        roles: Vec::new(),
    }
}

fn desc(layers: u32, experts: u32) -> ModelDesc {
    ModelDesc {
        name: "x".into(),
        layers,
        heads: 32,
        kv_heads: 4,
        experts,
        experts_used: if experts > 0 { 8 } else { 0 },
        embed: size(),
        attn: size(),
        mlp: size(),
        expert: (experts > 0).then(size),
        head: size(),
        kv_bytes_per_token_per_layer: 1,
        activation_bytes_per_token: 1,
    }
}

fn load(name: &str) -> QualityProfile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/quality")
        .join(name);
    QualityProfile::from_json(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

#[test]
fn measured_profiles_load_and_validate() {
    for (file, layers, experts, groups) in [
        ("qwen3-8b.quality.json", 36, 0, 32),
        ("qwen3-30b.quality.json", 48, 128, 22),
    ] {
        let q = load(file);
        assert_eq!(q.groups.len(), groups, "{file}");
        assert_eq!(
            (q.model.layers, q.model.experts),
            (layers, experts),
            "{file}"
        );
        assert!(q.groups.iter().all(|g| g.entries.len() >= 5), "{file}");
        let issues = q.validate(&desc(layers, experts));
        assert!(issues.is_empty(), "{file}: {issues:?}");
    }
}

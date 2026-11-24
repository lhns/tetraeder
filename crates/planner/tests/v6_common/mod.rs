//! Shared by the cost model 6 tests (ADR 0092): replay outcomes built by hand, the committed
//! `stream-predict-20261124` session and its graphs.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use tetraeder_costmodel::predict::{ReadPath, RunFeatures};
use tetraeder_planner::ModelLayout;
use tetraeder_planner::score::{
    BUILDER_DISK, RTX3090_PRIOR, TierLayer, TierOutcome, TierPhase, builder_features_tier,
};
use tetraeder_profiler::HardwareGraph;

pub const KNOBS: (u32, u32, bool) = (12, 2048, true);

pub fn outcome(vram: u32, ram: u32, share: f64, layers: Option<Vec<TierLayer>>) -> TierOutcome {
    TierOutcome {
        vram_slots: vram,
        ram_slots: ram,
        policy: "lru".into(),
        prefetch: true,
        decode: TierPhase {
            vram_share: share,
            full_hit_share: Some(share),
            sync_miss_bytes_per_token: 20e6,
            prefetch_bytes_per_token: 40e6,
            disk_bytes_per_token: 60e6,
            ram_hit_rate_engine_mean: 0.9,
            layers,
        },
        prefill: TierPhase {
            disk_bytes_per_token: 1e6,
            ..TierPhase::default()
        },
    }
}

pub fn rows(first: u32, shares: &[f64]) -> Vec<TierLayer> {
    shares
        .iter()
        .enumerate()
        .map(|(i, &s)| TierLayer {
            layer: first + i as u32,
            vram_share: s,
            full_hit_share: s,
            ..TierLayer::default()
        })
        .collect()
}

pub fn features(l: &ModelLayout, o: &TierOutcome) -> Result<RunFeatures, String> {
    builder_features_tier(
        l,
        o.vram_slots,
        o,
        KNOBS,
        ReadPath::Reader,
        (16, 16),
        BUILDER_DISK,
    )
}

pub fn prior() -> HardwareGraph {
    HardwareGraph::from_toml(RTX3090_PRIOR).unwrap()
}

pub fn cpu_resident(f: &RunFeatures) -> f64 {
    f.parts
        .iter()
        .find(|p| p.device == "cpu0")
        .unwrap()
        .resident_matmul_bytes
}

// ---- the session `stream-predict-20261124`, re-priced

pub fn session() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // a box checkout (no .git) gets no bench/results (taxameter.toml [push])
    if !root.join(".git").exists() {
        eprintln!("skip: box checkout, bench/results not pushed");
        return None;
    }
    Some(root.join("bench/results/stream-predict-20261124"))
}

pub fn json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

pub fn layout_of(dir: &Path) -> ModelLayout {
    serde_json::from_str(&std::fs::read_to_string(dir.join("predictions/layout.json")).unwrap())
        .unwrap()
}

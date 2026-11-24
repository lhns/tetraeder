//! Cost model 6 (ADR 0092), step 6: the prefill offload copy moves the RAM slots once per chunk
//! when the RAM tier does not cover the experts outside VRAM.
//!
//! Hardware label: none.

mod v6_common;

use tetraeder_costmodel::predict::{RunFeatures, predict};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::score::builder_features;
use v6_common::{KNOBS, cpu_resident, features, outcome, prior};

#[test]
fn the_offload_copy_moves_the_ram_slots_once_per_chunk() {
    let l = fixtures::moe();
    let experts: f64 = l.layers.iter().map(|b| b.expert_bytes as f64).sum();
    // (16, 52): 112 experts outside VRAM, 3 chunks of 52 rows, 156 rows per layer
    let chunked = features(&l, &outcome(16, 52, 0.25, None)).unwrap();
    let want = experts * 156.0 / 128.0;
    assert!((cpu_resident(&chunked) - want).abs() <= 1e-9 * want);
    // (32, 96) covers: today's bits
    let covering = features(&l, &outcome(32, 96, 0.25, None)).unwrap();
    assert_eq!(
        cpu_resident(&covering),
        cpu_resident(&builder_features(&l, 32, 0.25, KNOBS))
    );
    // the copy term of the prediction grows with the chunks' rows
    let hw = prior();
    let copy = |f: &RunFeatures| {
        let p = predict(&hw, f, 2048, 8).unwrap();
        p.pp_breakdown
            .iter()
            .filter(|e| e.0.starts_with("offload_copy"))
            .map(|e| e.1)
            .sum::<f64>()
    };
    let unchunked = builder_features(&l, 16, 0.25, KNOBS);
    assert!(copy(&chunked) > copy(&unchunked) * (156.0 / 112.0) * 0.99);
}

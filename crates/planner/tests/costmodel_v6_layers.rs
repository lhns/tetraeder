//! Cost model 6 (ADR 0092), step 3: per-layer routing rows price each layer of a segment, an
//! outcome without them keeps cost model 5's features, rows that do not cover the layout are an
//! error, and more VRAM share is never slower.
//!
//! Hardware label: none.

mod v6_common;

use proptest::prelude::*;
use tetraeder_costmodel::predict::{ReadPath, RunFeatures, predict};
use tetraeder_plan::Span;
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::score::{BUILDER_DISK, builder_features, builder_features_tier};
use tetraeder_planner::segment_layout;
use v6_common::{KNOBS, features, outcome, prior, rows};

#[test]
fn without_layer_rows_a_covering_tier_keeps_cost_model_5s_features() {
    let l = fixtures::moe();
    assert_eq!(l.n_expert, 128);
    // 32 + 96 slots cover the 128 experts
    let f = features(&l, &outcome(32, 96, 0.4, None)).unwrap();
    let old = builder_features(&l, 32, 0.4, KNOBS);
    assert!(f.tier.is_some());
    assert_eq!(RunFeatures { tier: None, ..f }, old);
}

#[test]
fn equal_layer_rows_give_the_aggregate_prediction() {
    let l = fixtures::moe();
    let hw = prior();
    let n = l.layers.len();
    for ram in [30, 96] {
        let agg = features(&l, &outcome(32, ram, 0.4, None)).unwrap();
        let per = features(&l, &outcome(32, ram, 0.4, Some(rows(0, &vec![0.4; n])))).unwrap();
        let (a, b) = (
            predict(&hw, &agg, 2048, 128).unwrap(),
            predict(&hw, &per, 2048, 128).unwrap(),
        );
        for (x, y) in [(a.tg_tok_s, b.tg_tok_s), (a.pp_tok_s, b.pp_tok_s)] {
            assert!((x - y).abs() <= 1e-12 * x.abs(), "{x} vs {y}");
        }
    }
}

#[test]
fn layer_rows_must_cover_exactly_the_layers_of_the_layout() {
    let l = fixtures::moe();
    let n = l.layers.len();
    let whole = outcome(32, 96, 0.4, Some(rows(0, &vec![0.4; n])));
    assert!(features(&l, &whole).is_ok());
    // one row short, a gap, and the whole model's rows for a segment
    let short = outcome(32, 96, 0.4, Some(rows(0, &vec![0.4; n - 1])));
    assert!(features(&l, &short).is_err());
    let mut gap = rows(0, &vec![0.4; n]);
    gap[5].layer = 99;
    assert!(features(&l, &outcome(32, 96, 0.4, Some(gap))).is_err());
    let seg = segment_layout(&l, Span(10, 20), false, false);
    assert!(features(&seg, &whole).is_err());
    // a replay of that span covers it
    let span = outcome(32, 96, 0.4, Some(rows(10, &[0.4; 10])));
    assert!(features(&seg, &span).is_ok());
}

#[test]
fn per_layer_shares_move_the_cpu_work_layer_by_layer() {
    let l = fixtures::moe();
    let n = l.layers.len();
    let mut shares = vec![1.0; n];
    let all_vram = features(&l, &outcome(32, 96, 1.0, Some(rows(0, &shares)))).unwrap();
    assert_eq!(
        all_vram.parts.len(),
        1,
        "no CPU part when every layer is in VRAM"
    );
    assert_eq!(all_vram.switches_per_token, 1.0);
    shares[3] = 0.0;
    let one = features(&l, &outcome(32, 96, 1.0, Some(rows(0, &shares)))).unwrap();
    assert_eq!(
        one.parts.len(),
        2,
        "one layer with misses brings the CPU part back"
    );
    assert_eq!(one.switches_per_token, 3.0);
}

proptest! {
    // Rule 5: more VRAM share in any layer is never slower, on the reader and the service path.
    #[test]
    fn a_higher_per_layer_share_is_never_slower(
        base in prop::collection::vec(0.0f64..1.0, 48),
        bump in prop::collection::vec(0.0f64..0.5, 48),
        service in any::<bool>(),
        ram in prop::sample::select(vec![30u32, 60, 96]),
    ) {
        let l = fixtures::moe();
        prop_assert_eq!(l.layers.len(), 48);
        let hw = prior();
        let hi: Vec<f64> = base.iter().zip(&bump).map(|(b, d)| (b + d).min(1.0)).collect();
        let path = if service { ReadPath::Service } else { ReadPath::Reader };
        let tg = |shares: &[f64]| {
            let o = outcome(32, ram, 0.5, Some(rows(0, shares)));
            let f = builder_features_tier(&l, 32, &o, KNOBS, path, (16, 16), BUILDER_DISK).unwrap();
            predict(&hw, &f, 512, 64).unwrap().tg_tok_s
        };
        prop_assert!(tg(&hi) >= tg(&base) * (1.0 - 1e-12));
    }
}

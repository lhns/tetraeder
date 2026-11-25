//! `tetraeder-plan segment-predict` (ADR 0092 section 6, tree node `bench-235b-segment-confirm`):
//! on the whole model it prices as `node-predict` does, and for a segment of a cluster plan it
//! prices as `predict_cluster` does. Both compare the unfitted prediction, bit for bit: the fit
//! moves the graph, the features do not.
//!
//! Hardware label: none (simulated hardware fixtures only).

use serde_json::{Value, json};
use std::collections::BTreeSet;
use tetraeder_plan::{Hints, Phase, SegmentShape, Span, segment_plan, segments};
use tetraeder_planner::layout::{ModelLayout, fixtures};
use tetraeder_planner::segment::FIT_BAND_LIMIT;
use tetraeder_planner::{
    Replay, ReplayMeta, SegmentAnchor, TierLayer, TierOutcome, Workload, enumerate_cluster,
    segment_layout, segment_predict, single_node_runs, single_node_runs_replay, uniform_tier,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

const LAN: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");

/// The uniform prior of `uniform_tier`, as per-layer rows of the span (what a replay of the span
/// holds): every layer takes the share `slots / n_expert`.
fn replay_of(layout: &ModelLayout, span: Span, v: u32, r: u32) -> Replay {
    Replay {
        meta: ReplayMeta {
            replay_version: 2,
            fnv1a64: "00".into(),
            trace_ubatch: 2048,
            prefill_tokens: 2048,
            layers: Some((span.0, span.1)),
            warm: 1,
            prompt_hash: "909eb7a3c358d2a8".into(),
        },
        outcomes: vec![tier_of(layout, span, v, r)],
        judged_prompt: Some("909eb7a3c358d2a8".into()),
    }
}

fn tier_of(layout: &ModelLayout, span: Span, v: u32, r: u32) -> TierOutcome {
    let sub = segment_layout(layout, span, false, false);
    let mut t = uniform_tier(&sub, &[(v, r)]).remove(0);
    let share = f64::from(v) / f64::from(layout.n_expert);
    t.decode.layers = Some(
        (span.0..span.1)
            .map(|layer| TierLayer {
                layer,
                vram_share: share,
                full_hit_share: share,
                ..TierLayer::default()
            })
            .collect(),
    );
    t
}

/// An `exit_regime` result with the fields an anchor reads.
fn anchor_file(pp: f64, tg: f64, share: f64) -> Value {
    json!({"runs": [{
        "prefill_tok_s": pp,
        "decode_tok_s": tg,
        "decode_tokens": 128,
        "decode_layers": {
            "vram_pair_share": share,
            "full_hit_rate_mean": share,
            "ram_hit_rate_mean": 1.0,
        },
        "decode_io": {"bytes_per_token": 2.0e8},
        "prefill_io": {"bytes_per_token": 0.0},
    }]})
}

fn anchors(share: f64) -> Vec<SegmentAnchor> {
    vec![
        SegmentAnchor {
            id: "a1".into(),
            span: Span(0, 12),
            files: vec![anchor_file(900.0, 30.0, share)],
        },
        SegmentAnchor {
            id: "a2".into(),
            span: Span(0, 24),
            files: vec![anchor_file(520.0, 17.0, share)],
        },
    ]
}

fn cpu_threads(g: &HardwareGraph) -> u32 {
    g.devices
        .iter()
        .find(|d| d.kind == DeviceKind::Cpu)
        .and_then(|d| d.threads)
        .unwrap()
}

/// On the full span with the embedding and the head, `segment-predict` is `node-predict` with the
/// same pair, ubatch and replay.
#[test]
fn the_whole_model_prices_as_node_predict() {
    let layout = fixtures::moe_235b();
    let n = layout.n_layer;
    let hw = HardwareGraph::from_toml(LAN).unwrap();
    let node = hw.node_graph(Some("a"));
    let (all, _) = single_node_runs(&node, &layout, &W, None, None).unwrap();
    assert!(!all.is_empty());
    let mut checked = 0;
    for pick in all.iter().take(6) {
        let (v, r) = (pick.vram_slots, pick.ram_slots);
        let replay = replay_of(&layout, Span(0, n), v, r);
        let (runs, _) =
            single_node_runs_replay(&node, &layout, &W, Some((v, r)), Some(&replay)).unwrap();
        let want = runs.iter().find(|x| x.ubatch == pick.ubatch).unwrap();
        let (got, fitted) = segment_predict(
            &node,
            &layout,
            SegmentShape {
                layers: Span(0, n),
                embed: true,
                head: true,
            },
            (v, r),
            &W,
            cpu_threads(&node),
            pick.ubatch,
            &anchors(f64::from(v) / f64::from(layout.n_expert)),
            &replay,
        )
        .unwrap();
        let tag = format!("v{v} r{r} ub{}", pick.ubatch);
        assert_eq!(
            got.uncalibrated_pp_tps.to_bits(),
            want.pp_tps.to_bits(),
            "{tag}"
        );
        assert_eq!(
            got.uncalibrated_tg_tps.to_bits(),
            want.tg_tps.to_bits(),
            "{tag}"
        );
        assert_eq!(
            got.uncalibrated_request_s.to_bits(),
            want.request_s.to_bits(),
            "{tag}"
        );
        // the fit changes the graph and nothing else
        assert!(got.pp_tps.is_finite() && got.tg_tps.is_finite(), "{tag}");
        assert_ne!(fitted.to_toml().unwrap(), node.to_toml().unwrap(), "{tag}");
        checked += 1;
    }
    assert!(checked > 0);
}

/// Without the head and the embedding the same span is cheaper on the GPU: the shape is an input,
/// not a default (the confirmation segment 47..94 has neither).
#[test]
fn the_shape_of_the_segment_is_priced() {
    let layout = fixtures::moe_235b();
    let n = layout.n_layer;
    let hw = HardwareGraph::from_toml(LAN).unwrap();
    let node = hw.node_graph(Some("a"));
    let (all, _) = single_node_runs(&node, &layout, &W, None, None).unwrap();
    let (v, r, ub) = (all[0].vram_slots, all[0].ram_slots, all[0].ubatch);
    let replay = replay_of(&layout, Span(0, n), v, r);
    let price = |head: bool| {
        segment_predict(
            &node,
            &layout,
            SegmentShape {
                layers: Span(0, n),
                embed: head,
                head,
            },
            (v, r),
            &W,
            cpu_threads(&node),
            ub,
            &anchors(f64::from(v) / f64::from(layout.n_expert)),
            &replay,
        )
        .unwrap()
        .0
    };
    assert_ne!(
        price(true).uncalibrated_tg_tps.to_bits(),
        price(false).uncalibrated_tg_tps.to_bits()
    );
}

/// For every builder segment of a cluster plan, `segment-predict`'s segment time is the segment
/// term of `predict_cluster`.
#[test]
fn a_cluster_segment_prices_as_predict_cluster_does() {
    let layout = fixtures::moe_235b();
    let hw = HardwareGraph::from_toml(LAN).unwrap();
    let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    assert!(!cands.is_empty());
    let mut checked = 0;
    let mut spans = BTreeSet::new();
    for c in cands.iter().take(8) {
        let segs = segments(&c.plan, 0, &hw).unwrap();
        for (i, seg) in segs.iter().enumerate() {
            if seg.stages.is_empty() || seg.layers.0 >= seg.layers.1 {
                continue; // a head-only segment has no layers to price from a replay
            }
            let node = hw.node_graph(seg.node.as_deref());
            if node.device("nvme0").is_none() {
                continue;
            }
            let sp = segment_plan(&c.plan, 0, seg);
            let st = &sp.replicas[0].stages[0];
            let slots = |id: &str| {
                st.expert_cache
                    .iter()
                    .find(|e| e.device == id)
                    .map(|e| e.slots)
            };
            let (Some(v), Some(r)) = (slots("gpu0"), slots("cpu0")) else {
                continue; // an all-CPU segment
            };
            let ub = c.plan.knobs.ubatch.unwrap();
            let threads = c.plan.knobs.threads.unwrap_or_else(|| cpu_threads(&node));
            let replay = replay_of(&layout, seg.layers, v, r);
            let (got, _) = segment_predict(
                &node,
                &layout,
                SegmentShape {
                    layers: seg.layers,
                    embed: seg.embed,
                    head: seg.head,
                },
                (v, r),
                &W,
                threads,
                ub,
                &anchors(f64::from(v) / f64::from(layout.n_expert)),
                &replay,
            )
            .unwrap();
            let want = &c.prediction.segments[i];
            let tag = format!("{} segment {i} {:?} v{v} r{r}", c.id, seg.layers);
            assert_eq!(
                got.uncalibrated_pp_tps.to_bits(),
                want.pp_tok_s.to_bits(),
                "{tag}"
            );
            assert_eq!(
                got.uncalibrated_tg_tps.to_bits(),
                want.tg_tok_s.to_bits(),
                "{tag}"
            );
            assert_eq!(
                got.uncalibrated_request_s.to_bits(),
                want.request_s.to_bits(),
                "{tag}"
            );
            spans.insert((seg.layers.0, seg.layers.1, seg.embed, seg.head));
            checked += 1;
        }
    }
    assert!(checked > 0, "no cluster segment had a GPU and a disk");
    assert!(spans.len() > 1, "only one segment shape was checked");
}

/// A prediction is not frozen on a fit whose band is over the limit (ADR 0091 owner answer 7).
#[test]
fn a_fit_band_over_the_limit_is_refused() {
    let layout = fixtures::moe_235b();
    let n = layout.n_layer;
    let hw = HardwareGraph::from_toml(LAN).unwrap();
    let node = hw.node_graph(Some("a"));
    let (all, _) = single_node_runs(&node, &layout, &W, None, None).unwrap();
    let (v, r, ub) = (all[0].vram_slots, all[0].ram_slots, all[0].ubatch);
    let replay = replay_of(&layout, Span(0, n), v, r);
    let (mut p, _) = segment_predict(
        &node,
        &layout,
        SegmentShape {
            layers: Span(0, n),
            embed: true,
            head: true,
        },
        (v, r),
        &W,
        cpu_threads(&node),
        ub,
        &anchors(f64::from(v) / f64::from(layout.n_expert)),
        &replay,
    )
    .unwrap();
    p.band = FIT_BAND_LIMIT;
    assert!(p.check_fit_band(FIT_BAND_LIMIT).is_ok());
    p.band = FIT_BAND_LIMIT + 1e-9;
    assert!(
        p.check_fit_band(FIT_BAND_LIMIT)
            .unwrap_err()
            .contains("nothing is frozen")
    );
    p.band = f64::NAN;
    assert!(p.check_fit_band(FIT_BAND_LIMIT).is_err());
}

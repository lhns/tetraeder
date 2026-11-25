//! `tetraeder-plan predict --cluster` (ADR 0092 section 1, tree node `planner-costmodel-v6-rest`):
//! a judged prediction of a plan across nodes refuses anything but routing `trace`, carries the
//! routing source and the trace hash, is priced per segment from the replay of that segment's
//! span (the prefill too when the trace header matches), and equals `segment-predict` bit for bit
//! where the two overlap.
//!
//! Hardware label: none (simulated hardware fixtures only).

use serde_json::{Value, json};
use std::path::Path;
use tetraeder_plan::{Hints, Phase, SegmentShape, Span, segment_plan, segments};
use tetraeder_planner::layout::{ModelLayout, fixtures};
use tetraeder_planner::{
    ClusterCandidate, Replay, ReplayMeta, RoutingSource, SegmentAnchor, SegmentReplay, TierLayer,
    TierOutcome, Workload, enumerate_cluster, predict_cluster_judged, predict_cluster_routed,
    segment_layout, segment_predict, uniform_tier,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const W: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

const LAN: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");

/// A builder segment of a plan: its index among the plan's segments and what it runs.
#[derive(Clone)]
struct Seg {
    index: usize,
    node: Option<String>,
    shape: SegmentShape,
    pair: (u32, u32),
}

fn setup() -> (HardwareGraph, ModelLayout, ClusterCandidate, Vec<Seg>) {
    let layout = fixtures::moe_235b();
    let hw = HardwareGraph::from_toml(LAN).unwrap();
    let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    for c in cands {
        let segs = builder_segments(&hw, &c);
        if segs.len() >= 2 {
            return (hw, layout, c, segs);
        }
    }
    panic!("no candidate with two builder segments");
}

fn builder_segments(hw: &HardwareGraph, c: &ClusterCandidate) -> Vec<Seg> {
    let mut out = Vec::new();
    for (index, seg) in segments(&c.plan, 0, hw).unwrap().iter().enumerate() {
        if seg.stages.is_empty() {
            continue;
        }
        let sp = segment_plan(&c.plan, 0, seg);
        let st = &sp.replicas[0].stages[0];
        let slots = |id: &str| st.expert_cache.iter().find(|e| e.device == id).map(|e| e.slots);
        if let (Some(v), Some(r)) = (slots("gpu0"), slots("cpu0")) {
            out.push(Seg {
                index,
                node: seg.node.clone(),
                shape: SegmentShape {
                    layers: seg.layers,
                    embed: seg.embed,
                    head: seg.head,
                },
                pair: (v, r),
            });
        }
    }
    out
}

/// Prefill bytes per token a replay of the span says beyond the uniform prior.
const EXTRA_PREFILL: f64 = 4.0e6;

/// The uniform prior as per-layer rows of the span, with `extra` more prefill disk bytes per
/// token: what a replay of the span holds when its prefill reads more than the prior.
fn tier_of(layout: &ModelLayout, span: Span, (v, r): (u32, u32), extra: f64) -> TierOutcome {
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
    t.prefill.disk_bytes_per_token += extra;
    t.prefill.sync_miss_bytes_per_token += extra;
    t
}

fn replay_of(layout: &ModelLayout, s: &Seg, ubatch: u32, tokens: u64, extra: f64) -> Replay {
    let span = s.shape.layers;
    Replay {
        meta: ReplayMeta {
            replay_version: 2,
            fnv1a64: "ab12".into(),
            trace_ubatch: ubatch,
            prefill_tokens: tokens,
            layers: Some((span.0, span.1)),
            warm: 1,
        },
        outcomes: vec![tier_of(layout, span, s.pair, extra)],
    }
}

fn replays(layout: &ModelLayout, segs: &[Seg], ubatch: u32, tokens: u64, extra: f64) -> Vec<SegmentReplay> {
    segs.iter()
        .map(|s| SegmentReplay {
            span: s.shape.layers,
            replay: replay_of(layout, s, ubatch, tokens, extra),
        })
        .collect()
}

#[test]
fn a_judged_cluster_prediction_needs_a_trace_for_every_builder_segment() {
    let (hw, layout, c, segs) = setup();
    let ub = c.plan.knobs.ubatch.unwrap();
    let all = replays(&layout, &segs, ub, 2048, 0.0);

    let none = predict_cluster_judged(&hw, &layout, &c.plan, &W, &[]).unwrap_err();
    assert!(none.contains("routing trace"), "{none}");
    let part = predict_cluster_judged(&hw, &layout, &c.plan, &W, &all[..1]).unwrap_err();
    assert!(part.contains("routing trace") && part.contains("no replay for the segments"), "{part}");

    let p = predict_cluster_judged(&hw, &layout, &c.plan, &W, &all).unwrap();
    assert_eq!(p.routing, RoutingSource::Trace);
    assert_eq!(p.segments.len(), segs.len());
    assert!(p.segments.iter().all(|s| s.routing == RoutingSource::Trace));
    assert!(p.segments.iter().all(|s| s.fnv1a64.as_deref() == Some("ab12")));

    // Without --judged the same plan is priced and marked: no replay, uniform, today's bits.
    let u = predict_cluster_routed(&hw, &layout, &c.plan, &W, &[]).unwrap();
    assert_eq!(u.routing, RoutingSource::Uniform);
    assert_eq!(u.prediction, c.prediction);
    let mixed = predict_cluster_routed(&hw, &layout, &c.plan, &W, &all[..1]).unwrap();
    assert_eq!(mixed.routing, RoutingSource::Uniform);
}

fn cpu_threads(g: &HardwareGraph) -> u32 {
    g.devices
        .iter()
        .find(|d| d.kind == DeviceKind::Cpu)
        .and_then(|d| d.threads)
        .unwrap()
}

fn anchor_file(pp: f64, tg: f64, share: f64) -> Value {
    json!({"runs": [{
        "prefill_tok_s": pp, "decode_tok_s": tg, "decode_tokens": 128,
        "decode_layers": {"vram_pair_share": share, "full_hit_rate_mean": share, "ram_hit_rate_mean": 1.0},
        "decode_io": {"bytes_per_token": 2.0e8},
        "prefill_io": {"bytes_per_token": 0.0},
    }]})
}

/// The judged cluster prediction of a segment is `segment-predict`'s uncalibrated prediction of
/// that segment from the same replay, bit for bit, with the replay's own prefill bytes.
#[test]
fn a_judged_segment_is_bit_identical_to_segment_predict() {
    let (hw, layout, c, segs) = setup();
    let ub = c.plan.knobs.ubatch.unwrap();
    let reps = replays(&layout, &segs, ub, 2048, EXTRA_PREFILL);
    let judged = predict_cluster_judged(&hw, &layout, &c.plan, &W, &reps).unwrap();
    let plain = predict_cluster_judged(&hw, &layout, &c.plan, &W, &replays(&layout, &segs, ub, 2048, 0.0)).unwrap();
    for (s, rep) in segs.iter().zip(&reps) {
        let node = hw.node_graph(s.node.as_deref());
        let share = f64::from(s.pair.0) / f64::from(layout.n_expert);
        let anchors = vec![
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
        ];
        let threads = c.plan.knobs.threads.unwrap_or_else(|| cpu_threads(&node));
        let (got, _) = segment_predict(
            &node,
            &layout,
            s.shape,
            s.pair,
            &W,
            threads,
            ub,
            &anchors,
            &rep.replay,
        )
        .unwrap();
        let want = &judged.prediction.segments[s.index];
        let tag = format!("segment {} {:?} v{} r{}", s.index, s.shape.layers, s.pair.0, s.pair.1);
        assert_eq!(got.prefill_source, "replay", "{tag}");
        assert_eq!(got.uncalibrated_pp_tps.to_bits(), want.pp_tok_s.to_bits(), "{tag}");
        assert_eq!(got.uncalibrated_tg_tps.to_bits(), want.tg_tok_s.to_bits(), "{tag}");
        assert_eq!(got.uncalibrated_request_s.to_bits(), want.request_s.to_bits(), "{tag}");
        // the replay's extra prefill bytes are in the prediction (item 2)
        let base = &plain.prediction.segments[s.index];
        assert!(want.pp_tok_s < base.pp_tok_s, "{tag}: {} vs {}", want.pp_tok_s, base.pp_tok_s);
        assert_eq!(want.tg_tok_s.to_bits(), base.tg_tok_s.to_bits(), "{tag}: decode is untouched");
    }
}

/// The replay's prefill prices the prefill only when its trace header says it recorded this
/// prompt in this ubatch (ADR 0092 section 1): a 32K trace or a 512 ubatch keeps the prior.
#[test]
fn prefill_disk_bytes_come_from_the_replay_only_when_the_trace_header_matches() {
    let (hw, layout, c, segs) = setup();
    let ub = c.plan.knobs.ubatch.unwrap();
    let pp = |ubatch: u32, tokens: u64, version: u32| {
        let mut r = replays(&layout, &segs, ubatch, tokens, EXTRA_PREFILL);
        for x in &mut r {
            x.replay.meta.replay_version = version;
        }
        predict_cluster_judged(&hw, &layout, &c.plan, &W, &r).unwrap().prediction.pp_tps
    };
    let prior = predict_cluster_judged(&hw, &layout, &c.plan, &W, &replays(&layout, &segs, ub, 2048, 0.0))
        .unwrap()
        .prediction
        .pp_tps;
    let matching = pp(ub, 2048, 2);
    assert!(matching < prior * (1.0 - 1e-6), "{matching} vs {prior}");
    for (what, got) in [
        ("other ubatch", pp(ub + 1, 2048, 2)),
        ("other prompt length", pp(ub, 32768, 2)),
        ("replay version 1", pp(ub, 2048, 1)),
    ] {
        assert!((got - prior).abs() <= 1e-9 * prior, "{what}: {got} vs {prior}");
    }
}

#[test]
fn a_replay_must_describe_exactly_the_segments_span() {
    let (hw, layout, c, segs) = setup();
    let ub = c.plan.knobs.ubatch.unwrap();
    let good = replays(&layout, &segs, ub, 2048, 0.0);
    let run = |r: &[SegmentReplay]| predict_cluster_judged(&hw, &layout, &c.plan, &W, r);
    assert!(run(&good).is_ok());

    // made for other layers
    let mut wrong_scope = good.clone();
    wrong_scope[0].replay.meta.layers = Some((0, 1));
    assert!(run(&wrong_scope).unwrap_err().contains("was made for layers"));

    // rows that start at another layer, with the right count
    let mut shifted = good.clone();
    for row in shifted[0].replay.outcomes[0].decode.layers.as_mut().unwrap() {
        row.layer += 1;
    }
    assert!(run(&shifted).unwrap_err().contains("layer rows"));

    // another slot pair
    let mut other_pair = good.clone();
    other_pair[0].replay.outcomes[0].vram_slots += 1;
    assert!(run(&other_pair).unwrap_err().contains("no lru replay outcome"));

    // a span of no segment
    let mut extra = good.clone();
    extra.push(SegmentReplay {
        span: Span(1, 2),
        replay: good[0].replay.clone(),
    });
    assert!(run(&extra).unwrap_err().contains("matches no segment"));
}

// ---- the command

fn bin() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_tetraeder-plan"))
}

fn replay_file(dir: &Path, name: &str, layout: &ModelLayout, s: &Seg, ubatch: u32) -> String {
    let span = s.shape.layers;
    let mut o = serde_json::to_value(tier_of(layout, span, s.pair, 0.0)).unwrap();
    o["prefill"]["tokens"] = json!(2048);
    let doc = json!({
        "replay_version": 2,
        "trace": {"fnv1a64": "ab12", "header": {"ubatch": ubatch}},
        "scope": {"layers": [span.0, span.1], "warm": 1},
        "outcomes": [o],
    });
    let path = dir.join(name);
    std::fs::write(&path, doc.to_string()).unwrap();
    format!("{}..{}={}", span.0, span.1, path.display())
}

#[test]
fn predict_cluster_judged_exits_non_zero_without_a_trace_and_carries_the_hash() {
    let (_, layout, c, segs) = setup();
    let ub = c.plan.knobs.ubatch.unwrap();
    let dir = std::env::temp_dir().join(format!("tetraeder-pc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hw.toml"), LAN).unwrap();
    std::fs::write(dir.join("layout.json"), serde_json::to_string(&layout).unwrap()).unwrap();
    std::fs::write(dir.join("plan.toml"), c.plan.to_toml().unwrap()).unwrap();
    let specs: Vec<String> = segs
        .iter()
        .enumerate()
        .map(|(i, s)| replay_file(&dir, &format!("r{i}.json"), &layout, s, ub))
        .collect();
    let out = dir.join("pred.json");
    let run = |extra: &[&str]| {
        let mut cmd = bin();
        cmd.arg("predict")
            .args(["--hw", dir.join("hw.toml").to_str().unwrap()])
            .args(["--layout", dir.join("layout.json").to_str().unwrap()])
            .args(["--cluster", dir.join("plan.toml").to_str().unwrap()])
            .args(["--out", out.to_str().unwrap()])
            .args(extra);
        cmd.output().unwrap()
    };
    let replay_args: Vec<&str> = specs.iter().flat_map(|s| ["--tier-replay", s.as_str()]).collect();

    // judged without a trace: refused, nothing written
    let no_trace = run(&["--judged", "--trace-sha256", "abc123"]);
    assert!(!no_trace.status.success());
    assert!(String::from_utf8_lossy(&no_trace.stderr).contains("routing trace"));
    assert!(!out.exists());
    // judged with replays but without the trace's hash: refused
    let mut args = vec!["--judged"];
    args.extend(&replay_args);
    let no_hash = run(&args);
    assert!(!no_hash.status.success());
    assert!(String::from_utf8_lossy(&no_hash.stderr).contains("sha256"));
    // judged with both
    let mut args = vec!["--judged", "--trace-sha256", "abc123"];
    args.extend(&replay_args);
    let ok = run(&args);
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stderr));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(doc["routing"], "trace");
    assert_eq!(doc["judged"], true);
    assert_eq!(doc["routing_trace"]["sha256"], "abc123");
    assert_eq!(doc["routing_trace"]["segments"].as_array().unwrap().len(), segs.len());
    assert_eq!(doc["routing_trace"]["segments"][0]["fnv1a64"], "ab12");
    assert!(doc["prediction"]["pp_tps"].as_f64().unwrap() > 0.0);
    // not judged and no replay: priced, marked uniform, equal to the enumeration's prediction
    std::fs::remove_file(&out).unwrap();
    let plain = run(&[]);
    assert!(plain.status.success(), "{}", String::from_utf8_lossy(&plain.stderr));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(doc["routing"], "uniform");
    assert_eq!(doc["prediction"]["pp_tps"].as_f64().unwrap().to_bits(), c.prediction.pp_tps.to_bits());
    std::fs::remove_dir_all(&dir).ok();
}

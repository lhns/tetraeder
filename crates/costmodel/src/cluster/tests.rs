//! Tests of `predict_cluster` on every simulated cluster fixture (AGENTS.md rules 4 and 5).
//!
//! Hardware label: none (simulated hardware fixtures only).

use super::*;
use crate::predict::fixture::features;
use proptest::prelude::*;
use tetraeder_profiler::{CollectiveCurve, CollectiveOp, HardwareGraph, LinkKind};

const FIXTURES: [(&str, &str); 7] = [
    (
        "lan",
        include_str!("../../fixtures/cluster/cluster_two_3090_lan.toml"),
    ),
    (
        "wan",
        include_str!("../../fixtures/cluster/cluster_two_3090_wan.toml"),
    ),
    (
        "samehost",
        include_str!("../../fixtures/cluster/cluster_two_3090_samehost.toml"),
    ),
    (
        "three",
        include_str!("../../fixtures/cluster/cluster_three_mixed.toml"),
    ),
    (
        "lan_slow",
        include_str!("../../fixtures/cluster/cluster_two_3090_lan_slowlink.toml"),
    ),
    (
        "wan_slow",
        include_str!("../../fixtures/cluster/cluster_two_3090_wan_slowlink.toml"),
    ),
    (
        "lan_degraded",
        include_str!("../../fixtures/cluster/cluster_two_3090_lan_degraded.toml"),
    ),
];

fn graphs() -> Vec<(&'static str, HardwareGraph)> {
    FIXTURES
        .iter()
        .map(|(n, t)| (*n, HardwareGraph::from_toml(t).unwrap()))
        .collect()
}

fn seg(node: &str, gb: f64, embed: bool, head: bool) -> SegmentFeatures {
    SegmentFeatures {
        node: Some(node.into()),
        // Node c of the three-node fixture has no GPU.
        run: if node == "c" {
            features(0.0, gb)
        } else {
            features(gb, 0.0)
        },
        embed,
        head,
    }
}

/// Qwen3-235B-sized rows: 16 KiB hidden, 594 KiB logits.
fn two(gb: (f64, f64), micro: u32, head_on_leader: bool) -> ClusterFeatures {
    let mut segments = vec![seg("a", gb.0, true, false), seg("b", gb.1, false, true)];
    if head_on_leader {
        segments[1].head = false;
        segments.push(seg("a", 0.5, false, true));
    }
    ClusterFeatures {
        segments,
        n_embd: 4096,
        n_vocab: 151_936,
        prefill_micro_batches: micro,
    }
}

fn three(micro: u32) -> ClusterFeatures {
    ClusterFeatures {
        segments: vec![
            seg("a", 6.0, true, false),
            seg("b", 6.0, false, false),
            seg("c", 6.0, false, true),
        ],
        n_embd: 4096,
        n_vocab: 151_936,
        prefill_micro_batches: micro,
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-12), "{a} vs {b}");
}

#[test]
fn every_cluster_fixture_is_a_valid_graph_of_whole_nodes() {
    for (name, g) in graphs() {
        g.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        let nodes = g.nodes();
        assert!(
            nodes.len() >= 2 && nodes.iter().all(Option::is_some),
            "{name}"
        );
        for n in &nodes {
            let ng = g.node_graph(*n);
            ng.validate()
                .unwrap_or_else(|e| panic!("{name} {n:?}: {e}"));
            assert!(ng.devices.iter().all(|d| d.node.is_none()), "{name}");
            assert!(
                ng.device("cpu0").is_some() && ng.device("nvme0").is_some(),
                "{name}"
            );
        }
        // Every cross-node link is a network link and every pair of nodes has one.
        for l in &g.links {
            let cross = g.node_of(&l.a) != g.node_of(&l.b);
            assert_eq!(cross, l.kind == LinkKind::Network, "{name}");
        }
        for (i, a) in nodes.iter().enumerate() {
            for b in &nodes[i + 1..] {
                let (a, b) = (a.unwrap(), b.unwrap());
                assert!(g.network_link(a, b).is_some(), "{name}: {a}-{b}");
            }
        }
    }
}

#[test]
fn hop_time_prefers_the_curve_and_falls_back_to_the_link() {
    let all = graphs();
    let wan = &all[1].1;
    // The measured points, one way: half of the 91 ms echo at 8 KB.
    assert_eq!(hop_time(wan, "a", "b", 8192.0), Some(0.0455));
    // Between the 8 KB and 1 MB points, linear.
    let mid = hop_time(wan, "a", "b", 524_288.0 + 4096.0).unwrap();
    assert!(mid > 0.0455 && mid < 0.1435);
    // The curve, not alpha and beta: changing the link changes nothing.
    let mut g = wan.clone();
    for l in &mut g.links {
        l.alpha_s.value = 99.0;
    }
    assert_eq!(hop_time(&g, "a", "b", 1_048_576.0), Some(0.1435));
    // Without the curve the link formula answers, in both directions.
    g.collectives.clear();
    for l in &mut g.links {
        if l.kind == LinkKind::Network {
            l.alpha_s.value = 0.01;
            l.beta_bytes_per_s.value = 1.0e6;
        }
    }
    close(hop_time(&g, "b", "a", 2.0e6).unwrap(), 0.01 + 2.0);
    // A node with itself costs nothing; two nodes with no link cannot be priced.
    assert_eq!(hop_time(&g, "a", "a", 1e9), Some(0.0));
    g.links.retain(|l| l.kind != LinkKind::Network);
    assert_eq!(hop_time(&g, "a", "b", 1.0), None);
    // The curve is per direction: the reverse falls back to the link.
    let mut one_way = all[0].1.clone();
    one_way.collectives.push(CollectiveCurve {
        op: CollectiveOp::Copy,
        ranks: vec!["a/cpu0".into(), "b/cpu0".into()],
        comm: "tls".into(),
        points: vec![(1.0, 5.0)],
    });
    assert_eq!(hop_time(&one_way, "a", "b", 1.0), Some(5.0));
    assert!(hop_time(&one_way, "b", "a", 1.0).unwrap() < 1.0);
}

#[test]
fn one_segment_predicts_as_predict_does() {
    for (name, g) in graphs() {
        let run = features(12.0, 0.0);
        let f = ClusterFeatures {
            segments: vec![SegmentFeatures {
                node: Some("a".into()),
                run: run.clone(),
                embed: true,
                head: true,
            }],
            n_embd: 4096,
            n_vocab: 151_936,
            prefill_micro_batches: 4,
        };
        let c = predict_cluster(&g, &f, 2048, 128).unwrap();
        let p = predict(&g.node_graph(Some("a")), &run, 2048, 128).unwrap();
        assert_eq!(c.pp_tps.to_bits(), p.pp_tok_s.to_bits(), "{name}");
        assert_eq!(c.tg_tps.to_bits(), p.tg_tok_s.to_bits(), "{name}");
        assert_eq!(c.request_s.to_bits(), p.request_s.to_bits(), "{name}");
        assert_eq!(c.segments, vec![p]);
        assert_eq!(c.hop_s_per_token, 0.0);
    }
    // A graph with no nodes at all predicts through `None` exactly as before.
    let hw = crate::predict::fixture::hw();
    let run = features(12.0, 4.0);
    let f = ClusterFeatures {
        segments: vec![SegmentFeatures {
            node: None,
            run: run.clone(),
            embed: true,
            head: true,
        }],
        n_embd: 4096,
        n_vocab: 151_936,
        prefill_micro_batches: 1,
    };
    let c = predict_cluster(&hw, &f, 512, 64).unwrap();
    assert_eq!(c.segments, vec![predict(&hw, &run, 512, 64).unwrap()]);
}

/// Every network link costs nothing.
fn free_links(g: &HardwareGraph) -> HardwareGraph {
    let mut g = g.clone();
    g.collectives.clear();
    for l in &mut g.links {
        if l.kind == LinkKind::Network {
            l.alpha_s.value = 0.0;
            l.beta_bytes_per_s.value = f64::INFINITY;
        }
    }
    g
}

#[test]
fn free_links_give_the_sum_of_the_segment_times() {
    for (name, g) in graphs() {
        let g = free_links(&g);
        for head_on_leader in [false, true] {
            let f = two((14.0, 11.0), 1, head_on_leader);
            let c = predict_cluster(&g, &f, 2048, 128).unwrap();
            let tg: f64 = c.segments.iter().map(|p| 1.0 / p.tg_tok_s).sum();
            close(1.0 / c.tg_tps, tg);
            let pp: f64 = c
                .segments
                .iter()
                .map(|p| p.pp_breakdown.iter().map(|e| e.1).sum::<f64>())
                .sum();
            close(2048.0 / c.pp_tps, pp);
            assert_eq!(c.hop_s_per_token, 0.0, "{name}");
            // Micro-batches pipeline the segments: never slower than running them in turn.
            let m = predict_cluster(&g, &two((14.0, 11.0), 4, head_on_leader), 2048, 128).unwrap();
            assert!(m.pp_tps >= c.pp_tps * (1.0 - 1e-9), "{name}");
            close(m.tg_tps, c.tg_tps);
        }
    }
}

#[test]
fn the_star_pays_one_round_trip_per_decode_token() {
    let all = graphs();
    let wan = &all[1].1;
    let hid = 4.0 * 4096.0;
    let logits = 4.0 * 151_936.0;
    let hop = |bytes: f64| hop_time(wan, "a", "b", bytes).unwrap();
    // Head on the last node: hidden rows out, the logits row back.
    let last = predict_cluster(wan, &two((12.0, 12.0), 1, false), 512, 64).unwrap();
    close(last.hop_s_per_token, hop(hid) + hop(logits));
    // Head on the leader: hidden rows both ways.
    let lead = predict_cluster(wan, &two((12.0, 12.0), 1, true), 512, 64).unwrap();
    close(lead.hop_s_per_token, 2.0 * hop(hid));
    assert!(lead.hop_s_per_token < last.hop_s_per_token);
    // Three nodes: b and c both go through a.
    let g = &all[3].1;
    let p = predict_cluster(g, &three(1), 512, 64).unwrap();
    let h = |a: &str, b: &str, bytes: f64| hop_time(g, a, b, bytes).unwrap();
    close(
        p.hop_s_per_token,
        h("a", "b", hid) + h("b", "a", hid) + h("a", "c", hid) + h("c", "a", logits),
    );
}

#[test]
fn a_missing_link_or_no_segment_is_an_error() {
    let all = graphs();
    let mut g = all[0].1.clone();
    g.links.retain(|l| l.kind != LinkKind::Network);
    let e = predict_cluster(&g, &two((8.0, 8.0), 1, false), 512, 64).unwrap_err();
    assert!(matches!(e, PredictError::NoLink { .. }));
    let none = ClusterFeatures {
        segments: vec![],
        ..two((1.0, 1.0), 1, false)
    };
    assert!(predict_cluster(&all[0].1, &none, 512, 64).is_err());
}

/// The links slowed down: latency scales the part of a curve that does not grow with the size
/// (its first point), bandwidth the rest, so both act on a measured curve as on `alpha + bytes /
/// beta`. Curve points are scaled, or the property would pass without reading them.
fn slower(g: &HardwareGraph, latency: f64, bandwidth: f64) -> HardwareGraph {
    let mut g = g.clone();
    for c in &mut g.collectives {
        let base = c.points[0].1;
        for p in &mut c.points {
            p.1 = base * latency + (p.1 - base) * bandwidth;
        }
    }
    for l in &mut g.links {
        if l.kind == LinkKind::Network {
            l.alpha_s.value *= latency;
            l.beta_bytes_per_s.value /= bandwidth;
        }
    }
    g
}

proptest! {
    /// More latency or less bandwidth on the links never raises predicted tokens per second.
    #[test]
    fn slower_links_never_raise_the_rates(
        which in 0usize..7,
        l1 in 1.0f64..20.0, l2 in 1.0f64..20.0,
        b1 in 1.0f64..20.0, b2 in 1.0f64..20.0,
        micro in prop::sample::select(vec![1u32, 4]),
        lead in any::<bool>(),
    ) {
        let all = graphs();
        let g = &all[which].1;
        let f = if all[which].0 == "three" { three(micro) } else { two((10.0, 10.0), micro, lead) };
        let (l_lo, l_hi) = (l1.min(l2), l1.max(l2));
        let (b_lo, b_hi) = (b1.min(b2), b1.max(b2));
        let fast = predict_cluster(&slower(g, l_lo, b_lo), &f, 2048, 128).unwrap();
        let slow = predict_cluster(&slower(g, l_hi, b_hi), &f, 2048, 128).unwrap();
        prop_assert!(slow.tg_tps <= fast.tg_tps * (1.0 + 1e-12));
        prop_assert!(slow.pp_tps <= fast.pp_tps * (1.0 + 1e-12));
        // The identity point is the fixture itself.
        let same = predict_cluster(&slower(g, 1.0, 1.0), &f, 2048, 128).unwrap();
        let base = predict_cluster(g, &f, 2048, 128).unwrap();
        prop_assert!((same.tg_tps - base.tg_tps).abs() <= 1e-9 * base.tg_tps);
    }
}

#[test]
fn the_slow_link_fixtures_are_slower_than_their_bases() {
    let all = graphs();
    let f = two((10.0, 10.0), 1, false);
    let rate = |i: usize| predict_cluster(&all[i].1, &f, 2048, 128).unwrap().tg_tps;
    assert!(rate(4) < rate(0), "lan slowlink");
    assert!(rate(5) < rate(1), "wan slowlink");
    assert!(rate(2) > rate(0) && rate(0) > rate(1), "samehost, lan, wan");
}

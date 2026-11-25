//! The cluster planner's inputs (tree node `cluster-planner-hardening`, ADR 0091 step 2 review):
//! node names, the node-order search, and malformed layouts or spans give errors, not panics.
//!
//! Hardware label: none (simulated hardware fixtures only).

use tetraeder_plan::{Hints, Phase, Span};
use tetraeder_planner::cluster::{
    MAX_EXHAUSTIVE_NODES, MAX_SEARCHED_NODES, node_orders, try_segment_layout,
};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::multi_cli::parse_hw;
use tetraeder_planner::{
    Workload, cluster_features, enumerate_cluster, single_node_references, single_node_runs,
};
use tetraeder_profiler::{Device, DeviceKind, HardwareGraph, Tier};

const LAN: &str = include_str!("../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml");

fn lan() -> HardwareGraph {
    HardwareGraph::from_toml(LAN).unwrap()
}

/// The two-node LAN fixture with its nodes renamed.
fn renamed(a: &str, b: &str) -> HardwareGraph {
    let mut g = lan();
    for d in &mut g.devices {
        d.node = Some(if d.node.as_deref() == Some("a") { a } else { b }.to_string());
    }
    g
}

#[test]
fn a_node_name_is_one_path_segment() {
    let layout = fixtures::moe_235b();
    for bad in ["", ".", "..", "x/y", "x\\y"] {
        let g = renamed(bad, "b");
        assert!(g.check_node_names().is_err(), "`{bad}`");
        // refused when the graph is loaded from a file
        let text = g.to_toml().unwrap();
        assert!(parse_hw(&text).unwrap_err().contains("not a node name"), "`{bad}`");
        // and by the search, which prunes everything with the reason instead of building plans
        let (cands, pruned) = enumerate_cluster(&g, &layout, &Hints::default(), Phase(6));
        assert!(cands.is_empty(), "`{bad}`");
        assert!(pruned.iter().any(|(_, why)| why.contains("not a node name")), "`{bad}`");
    }
    assert!(parse_hw(&lan().to_toml().unwrap()).is_ok());
}

#[test]
fn two_nodes_with_one_name_are_refused() {
    let layout = fixtures::moe_235b();
    // every device of b now says it is on a: the ids repeat
    let mut g = lan();
    for d in &mut g.devices {
        if d.node.as_deref() == Some("b") {
            d.id = d.id.replacen("b/", "a/", 1);
            d.node = Some("a".into());
        }
    }
    assert!(g.check_node_names().unwrap_err().contains("repeats"));
    let (cands, pruned) = enumerate_cluster(&g, &layout, &Hints::default(), Phase(6));
    assert!(cands.is_empty());
    assert!(pruned.iter().any(|(_, why)| why.contains("repeats")));
}

#[test]
fn the_node_order_search_is_exhaustive_to_six_nodes_and_bounded_above() {
    for m in 1..=MAX_EXHAUSTIVE_NODES {
        let (orders, rule) = node_orders(m);
        let fact: usize = (1..=m).product();
        assert_eq!(orders.len(), fact, "{m} nodes");
        assert!(rule.is_none());
    }
    assert_eq!(MAX_EXHAUSTIVE_NODES, 6);
    for m in [MAX_EXHAUSTIVE_NODES + 1, 10, MAX_SEARCHED_NODES] {
        let (orders, rule) = node_orders(m);
        assert_eq!(orders.len(), 2 * m, "{m} nodes");
        assert!(rule.unwrap().contains("rotations"));
        let mut seen = std::collections::BTreeSet::new();
        for o in &orders {
            let mut sorted = o.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..m).collect::<Vec<_>>(), "an order is a permutation");
            assert!(seen.insert(o.clone()), "orders are distinct");
        }
        // every node leads at least once
        assert_eq!(orders.iter().map(|o| o[0]).collect::<std::collections::BTreeSet<_>>().len(), m);
        assert_eq!(node_orders(m).0, orders, "deterministic");
    }
}

#[test]
fn more_nodes_than_the_search_covers_are_refused_with_the_reason() {
    let layout = fixtures::moe_235b();
    let n = MAX_SEARCHED_NODES + 1;
    let nodes: Vec<(String, HardwareGraph)> = (0..n)
        .map(|i| {
            let cpu = Device::new("cpu0", DeviceKind::Cpu, "cpu", Tier::Ram, 1 << 36);
            (
                format!("n{i}"),
                HardwareGraph {
                    devices: vec![cpu],
                    ..HardwareGraph::default()
                },
            )
        })
        .collect();
    let g = HardwareGraph::merge_cluster(&nodes, vec![], vec![]);
    let (cands, pruned) = enumerate_cluster(&g, &layout, &Hints::default(), Phase(6));
    assert!(cands.is_empty());
    assert!(
        pruned.iter().any(|(_, why)| why.contains("at most 16")),
        "{pruned:?}"
    );
}

#[test]
fn spans_and_layouts_that_do_not_fit_are_errors_not_panics() {
    let layout = fixtures::moe_235b();
    let n = layout.n_layer;
    assert!(try_segment_layout(&layout, Span(0, n), true, true).is_ok());
    assert!(try_segment_layout(&layout, Span(5, 5), false, false).is_ok());
    for (a, b) in [(5, 3), (0, n + 1), (n + 1, n + 2)] {
        let e = try_segment_layout(&layout, Span(a, b), false, false).unwrap_err();
        assert!(e.contains("outside the layout"), "{a}..{b}: {e}");
    }
    // a layout file whose layer list is shorter than its n_layer
    let mut short = layout.clone();
    short.layers.pop();
    let e = try_segment_layout(&short, Span(0, 1), false, false).unwrap_err();
    assert!(e.contains("malformed layout"), "{e}");
    let hw = lan();
    let (cands, pruned) = enumerate_cluster(&hw, &short, &Hints::default(), Phase(6));
    assert!(cands.is_empty());
    assert!(pruned.iter().any(|(_, why)| why.contains("malformed layout")));
    assert!(single_node_references(&hw, &short, &Hints::default()).is_empty());
    let node = hw.node_graph(Some("a"));
    let w = Workload {
        prompt: 2048,
        r#gen: 128,
    };
    assert!(single_node_runs(&node, &short, &w, None, None).unwrap_err().contains("malformed layout"));
    // a plan whose last stage runs past the layers of the layout
    let (good, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    let mut plan = good[0].plan.clone();
    plan.replicas[0].stages.last_mut().unwrap().layers.1 = n + 5;
    assert!(cluster_features(&hw, &layout, &plan).is_err());
}

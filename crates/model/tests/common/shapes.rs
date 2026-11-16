//! Topologies of the executor's shapes (ADR 0070), hand-built for CPU ranks: `Topology::from_plan`
//! accepts GPU stages only. Shared by the sharded builders' tests.
#![allow(dead_code)]

use tetraeder_plan::{RankSpec, Span, StageSpec, Topology};

pub fn stage(layers: (u32, u32), ranks: &[usize], embed: bool, head: bool) -> StageSpec {
    StageSpec {
        layers: Span(layers.0, layers.1),
        ranks: ranks.to_vec(),
        tp: ranks.len() as u32,
        ep: 1,
        embed,
        head,
    }
}

pub fn topology(n_layer: u32, n_ranks: usize, stages: Vec<StageSpec>) -> Topology {
    Topology {
        ranks: (0..n_ranks)
            .map(|i| RankSpec {
                device: format!("cpu{i}"),
                gpu_index: None,
            })
            .collect(),
        stages,
        n_layer,
    }
}

/// One stage, one rank.
pub fn single(n_layer: u32) -> Topology {
    tp(n_layer, 1)
}

/// `tp-g`: one stage over `g` ranks.
pub fn tp(n_layer: u32, g: usize) -> Topology {
    let ranks: Vec<usize> = (0..g).collect();
    topology(n_layer, g, vec![stage((0, n_layer), &ranks, true, true)])
}

/// `layer-2`: layers `[0, at)` on rank 0, the rest on rank 1.
pub fn layer_split(n_layer: u32, at: u32) -> Topology {
    topology(
        n_layer,
        2,
        vec![
            stage((0, at), &[0], true, false),
            stage((at, n_layer), &[1], false, true),
        ],
    )
}

/// `tp2-pp2`: two stages of two ranks, split before layer `at`.
pub fn tp2pp2(n_layer: u32, at: u32) -> Topology {
    topology(
        n_layer,
        4,
        vec![
            stage((0, at), &[0, 1], true, false),
            stage((at, n_layer), &[2, 3], false, true),
        ],
    )
}

/// `layer-g`: `g` stages of one rank each, the layers dealt as evenly as possible (the first
/// stages one layer larger).
pub fn layer_g(n_layer: u32, g: usize) -> Topology {
    let (base, extra) = (n_layer / g as u32, n_layer % g as u32);
    let mut at = 0;
    let stages = (0..g)
        .map(|i| {
            let len = base + u32::from((i as u32) < extra);
            let st = stage((at, at + len), &[i], i == 0, i + 1 == g);
            at += len;
            st
        })
        .collect();
    topology(n_layer, g, stages)
}

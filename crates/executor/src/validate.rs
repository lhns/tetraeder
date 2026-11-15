use crate::{Op, Program, RankGraph, RankId};
use std::collections::HashMap;
use tetraeder_ggml_sys::ggml::{Tensor, Type};

/// Checks a program against the rules of ADR 0069 and returns every violation:
/// - each rank has at most one graph, cut into strictly increasing ranges that end at its last
///   node (so every node is in exactly one range);
/// - every range is computed exactly once, the ranges of a rank in node order;
/// - an all-reduce names one F32 tensor of one shape per distinct rank, each the last node of a
///   range, computed before it and read only by ranges computed after it;
/// - a copy goes between two ranks with the same layout, its source (when it is a node) is the
///   last node of a range computed before it, its destination is an input (not a node) of its
///   rank's graph and is read only by ranges computed after it.
pub fn validate(p: &Program<'_>, n_ranks: usize) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    let mut graph_of: Vec<Option<&RankGraph<'_>>> = vec![None; n_ranks];
    for (i, g) in p.graphs.iter().enumerate() {
        if g.rank >= n_ranks {
            errs.push(format!(
                "graph {i} is for rank {}, but there are {n_ranks} ranks",
                g.rank
            ));
        } else if graph_of[g.rank].is_some() {
            errs.push(format!("rank {} has more than one graph", g.rank));
        } else if let Err(e) = check_cuts(&g.cuts, g.graph.n_nodes()) {
            errs.push(format!("rank {}: {e}", g.rank));
        } else {
            graph_of[g.rank] = Some(g);
        }
    }

    // Op index of each range's compute.
    let mut computed: HashMap<(RankId, usize), usize> = HashMap::new();
    let mut last_range: Vec<Option<usize>> = vec![None; n_ranks];
    for (i, op) in p.ops.iter().enumerate() {
        let Op::Compute { rank, range } = *op else {
            continue;
        };
        let Some(g) = graph_of.get(rank).copied().flatten() else {
            errs.push(format!(
                "op {i}: compute on rank {rank}, which has no usable graph"
            ));
            continue;
        };
        if range >= g.cuts.len() {
            errs.push(format!(
                "op {i}: rank {rank} has {} ranges, not {range}",
                g.cuts.len()
            ));
        } else if computed.insert((rank, range), i).is_some() {
            errs.push(format!(
                "op {i}: range {range} of rank {rank} is computed twice"
            ));
        } else if last_range[rank].is_some_and(|l| l > range) {
            errs.push(format!(
                "op {i}: range {range} of rank {rank} is computed after a later range"
            ));
        }
        last_range[rank] = Some(range);
    }
    for g in graph_of.iter().flatten() {
        for k in 0..g.cuts.len() {
            if !computed.contains_key(&(g.rank, k)) {
                errs.push(format!("range {k} of rank {} is never computed", g.rank));
            }
        }
    }

    for (i, op) in p.ops.iter().enumerate() {
        match op {
            Op::Compute { .. } => {}
            Op::AllReduce { tensors } => {
                check_all_reduce(i, tensors, &graph_of, &computed, &mut errs)
            }
            Op::Copy { src, dst } => check_copy(i, *src, *dst, &graph_of, &computed, &mut errs),
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

fn check_cuts(cuts: &[usize], n_nodes: usize) -> Result<(), String> {
    let mut prev = 0;
    for &c in cuts {
        if c <= prev {
            return Err(format!("cuts {cuts:?} are not strictly increasing from 0"));
        }
        prev = c;
    }
    if prev != n_nodes {
        return Err(format!("cuts {cuts:?} do not end at the {n_nodes} nodes"));
    }
    Ok(())
}

fn node_index(g: &RankGraph<'_>, t: Tensor<'_>) -> Option<usize> {
    (0..g.graph.n_nodes()).find(|&i| g.graph.node(i).is(t))
}

/// The range whose last node is `t`: `Ok(Some(k))`; `Ok(None)` when `t` is not a node of the
/// graph at all; `Err` when it is a node inside a range.
fn boundary_range(g: &RankGraph<'_>, t: Tensor<'_>) -> Result<Option<usize>, ()> {
    match node_index(g, t) {
        None => Ok(None),
        Some(i) => g.cuts.iter().position(|&c| c == i + 1).map(Some).ok_or(()),
    }
}

/// Ranges of `g` holding a node that reads `t` directly.
fn reader_ranges(g: &RankGraph<'_>, t: Tensor<'_>) -> Vec<usize> {
    let mut out: Vec<usize> = (0..g.graph.n_nodes())
        .filter(|&i| {
            (0..16)
                .filter_map(|s| g.graph.node(i).src(s))
                .any(|s| s.is(t))
        })
        .map(|i| g.cuts.partition_point(|&c| c <= i))
        .collect();
    out.dedup();
    out
}

fn check_all_reduce(
    i: usize,
    tensors: &[(RankId, Tensor<'_>)],
    graph_of: &[Option<&RankGraph<'_>>],
    computed: &HashMap<(RankId, usize), usize>,
    errs: &mut Vec<String>,
) {
    if tensors.len() < 2 {
        errs.push(format!("op {i}: all-reduce of {} tensors", tensors.len()));
    }
    let mut ranks: Vec<RankId> = tensors.iter().map(|&(r, _)| r).collect();
    ranks.sort_unstable();
    if ranks.windows(2).any(|w| w[0] == w[1]) {
        errs.push(format!("op {i}: all-reduce names a rank twice"));
    }
    if let Some(&(_, first)) = tensors.first() {
        for &(r, t) in tensors {
            if t.ty() != Type::F32 {
                errs.push(format!("op {i}: all-reduce tensor of rank {r} is not F32"));
            }
            if t.ne() != first.ne() {
                errs.push(format!(
                    "op {i}: all-reduce tensor of rank {r} has shape {:?}, not {:?}",
                    t.ne(),
                    first.ne()
                ));
            }
        }
    }
    for &(r, t) in tensors {
        let Some(g) = graph_of.get(r).copied().flatten() else {
            errs.push(format!(
                "op {i}: all-reduce on rank {r}, which has no usable graph"
            ));
            continue;
        };
        match boundary_range(g, t) {
            Ok(Some(k)) => {
                if computed.get(&(r, k)).is_some_and(|&c| c > i) {
                    errs.push(format!(
                        "op {i}: all-reduce before range {k} of rank {r} computes its tensor"
                    ));
                }
                check_readers_after(i, "all-reduce", r, g, t, computed, errs);
            }
            Ok(None) | Err(()) => errs.push(format!(
                "op {i}: all-reduce tensor of rank {r} is not the last node of a range"
            )),
        }
    }
}

fn check_copy(
    i: usize,
    src: (RankId, Tensor<'_>),
    dst: (RankId, Tensor<'_>),
    graph_of: &[Option<&RankGraph<'_>>],
    computed: &HashMap<(RankId, usize), usize>,
    errs: &mut Vec<String>,
) {
    if src.0 == dst.0 {
        errs.push(format!("op {i}: copy from rank {0} to itself", src.0));
    }
    if src.1.ty() != dst.1.ty() || src.1.ne() != dst.1.ne() || src.1.nb() != dst.1.nb() {
        errs.push(format!("op {i}: copy between tensors of different layouts"));
    }
    match graph_of.get(src.0).copied().flatten() {
        None => errs.push(format!(
            "op {i}: copy from rank {}, which has no usable graph",
            src.0
        )),
        Some(g) => match boundary_range(g, src.1) {
            // Data that exists before the program runs (an input, a weight).
            Ok(None) => {}
            Ok(Some(k)) => {
                if computed.get(&(src.0, k)).is_some_and(|&c| c > i) {
                    errs.push(format!(
                        "op {i}: copy before range {k} of rank {} computes its source",
                        src.0
                    ));
                }
            }
            Err(()) => errs.push(format!(
                "op {i}: copy source on rank {} is not the last node of a range",
                src.0
            )),
        },
    }
    match graph_of.get(dst.0).copied().flatten() {
        None => errs.push(format!(
            "op {i}: copy to rank {}, which has no usable graph",
            dst.0
        )),
        Some(g) => {
            if node_index(g, dst.1).is_some() {
                errs.push(format!(
                    "op {i}: copy destination on rank {} is computed by its graph",
                    dst.0
                ));
            }
            check_readers_after(i, "copy", dst.0, g, dst.1, computed, errs);
        }
    }
}

/// Every range of `g` that reads `t` is computed after op `i`.
fn check_readers_after(
    i: usize,
    what: &str,
    rank: RankId,
    g: &RankGraph<'_>,
    t: Tensor<'_>,
    computed: &HashMap<(RankId, usize), usize>,
    errs: &mut Vec<String>,
) {
    for k in reader_ranges(g, t) {
        if computed.get(&(rank, k)).is_some_and(|&c| c < i) {
            errs.push(format!(
                "op {i}: range {k} of rank {rank} reads the {what} tensor before op {i}"
            ));
        }
    }
}

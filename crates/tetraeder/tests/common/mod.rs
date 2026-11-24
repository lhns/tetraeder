//! Shared by the pipeline tests: the plan of a two-stage replica across two nodes and the logit
//! comparisons of rule 2.

#![allow(dead_code)]

use tetraeder_plan::{Assign, Plan, Replica, Span, Stage, UnitKind, UnitSel};

fn rule(kinds: Vec<UnitKind>, layers: Option<Span>, dev: Option<&str>) -> Assign {
    let d = dev.map(|d| vec![d.to_owned()]);
    Assign {
        units: UnitSel {
            kinds,
            layers,
            experts: None,
        },
        home: d.clone(),
        compute: d,
        ..Assign::default()
    }
}

/// One replica of two stages: layers `0..cut` on device `first`, `cut..n_layer` on device
/// `second` (ids of a merged cluster graph, `<node>/<device>`). The head sits on the last
/// stage's node, or on `head_on` as a rule of its own (a head-only segment on that node).
pub fn two_stage_plan(
    model: &str,
    n_layer: u32,
    cut: u32,
    first: &str,
    second: &str,
    head_on: Option<&str>,
) -> Plan {
    let stage = |layers: Span, dev: &str, extra: Vec<Assign>| {
        let mut assign = vec![rule(vec![], Some(layers), Some(dev))];
        assign.extend(extra);
        Stage {
            layers,
            devices: vec![dev.to_owned()],
            tp: Some(1),
            ep: Some(1),
            assign,
            ..Stage::default()
        }
    };
    let head = rule(vec![UnitKind::Head], None, head_on);
    Plan {
        model: model.to_owned(),
        knobs: Default::default(),
        replicas: vec![Replica {
            stages: vec![
                stage(
                    Span(0, cut),
                    first,
                    vec![rule(vec![UnitKind::Embed], None, Some(first))],
                ),
                stage(Span(cut, n_layer), second, vec![head]),
            ],
        }],
    }
}

pub fn largest(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

/// `max(1e-4, 1e-3 * largest logit)`: the batched-against-alone rule of the model tests.
pub fn tol_batch(want: &[f32]) -> f32 {
    1e-4f32.max(1e-3 * largest(want))
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|p| p.0)
        .unwrap()
}

/// The `k` largest, largest first, ties by the lower index.
pub fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// What a comparison found.
pub struct Run {
    pub max_diff: f32,
    pub bit_equal: bool,
}

/// Checks `got` against `want`: the whole row and the five largest within `tol_batch`.
pub fn check_row(run: &mut Run, got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let tol = tol_batch(want);
    let d = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    run.max_diff = run.max_diff.max(d);
    assert!(d <= tol, "{what}: {d:e} above {tol:e}");
    for id in top_k(want, 5) {
        assert!(
            (got[id] - want[id]).abs() <= tol,
            "{what}: top-5 token {id}"
        );
    }
    run.bit_equal &= got
        .iter()
        .map(|x| x.to_bits())
        .eq(want.iter().map(|x| x.to_bits()));
}

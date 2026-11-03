//! The objective (ADR 0025): hard limits, then quality, then resource cost, then speed; the
//! Pareto front of speed against quality; `tradeoff` as a position on that front.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use tetraeder_plan::{Hints, Plan, Span};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// Predicted time of one pipeline stage for a decode token, for `plan explain`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTime {
    pub stage: u32,
    pub layers: Span,
    /// Compute device group plus helper devices the stage uses.
    pub devices: Vec<String>,
    /// Batch 1 decode compute time of the stage, including its expert streaming.
    pub compute_s: f64,
    /// Hop to the next stage (the last one hops back to the first): `alpha + a / beta`.
    pub hop_s: f64,
}

/// A candidate plan with its predicted numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    /// The skeleton (devices per stage, expert mode) and compression level it came from.
    pub desc: String,
    pub plan: Plan,
    /// Decode speed at batch 1 (analytic).
    pub tok_s: f64,
    /// p95 time to first token for the workload: the simulator's p95 where `simulated`, never
    /// below the uncontended analytic prefill time.
    pub ttft_s: f64,
    /// Mean time per output token under the workload (simulator only).
    pub tpot_s: Option<f64>,
    pub simulated: bool,
    /// KL divergence: the quality model's base KL plus the per-unit losses of the plan.
    pub quality_loss: f64,
    /// Sum of the hint `cost` of the devices the plan uses ([`resource_cost`]).
    pub resource_cost: f64,
    /// Smallest free memory over the devices the plan uses, after the reserve.
    pub headroom_bytes: u64,
    pub stage_times: Vec<StageTime>,
}

/// Default price of a device without a `cost` hint: 1 per GPU, 0.5 per CPU, 0 per disk (storage
/// is not priced unless a hint says so).
pub fn default_cost(kind: DeviceKind) -> f64 {
    match kind {
        DeviceKind::Gpu => 1.0,
        DeviceKind::Cpu => 0.5,
        DeviceKind::Disk => 0.0,
    }
}

/// Every device a plan names: stage groups, homes, compute devices, KV, workspace and slots.
pub fn devices_used(plan: &Plan) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |d: &String| {
        if !out.contains(d) {
            out.push(d.clone());
        }
    };
    for st in plan.replicas.iter().flat_map(|r| &r.stages) {
        st.devices.iter().for_each(&mut add);
        for a in &st.assign {
            a.home.iter().chain(&a.compute).flatten().for_each(&mut add);
        }
        st.kv.iter().flatten().for_each(|d| add(&d.device));
        st.workspace.iter().for_each(|d| add(&d.device));
        st.expert_cache.iter().for_each(|c| add(&c.device));
    }
    out.sort();
    out
}

/// Sum of the hint cost of the devices the plan uses.
pub fn resource_cost(hw: &HardwareGraph, hints: &Hints, plan: &Plan) -> f64 {
    devices_used(plan)
        .iter()
        .map(|id| {
            hints
                .cost
                .get(id)
                .copied()
                .or_else(|| hw.device(id).map(|d| default_cost(d.kind)))
                .unwrap_or(0.0)
        })
        .sum()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// The plan to run: the best one meeting every limit, else the closest to meeting them.
    /// `None` only when nothing was scored.
    pub chosen: Option<usize>,
    /// Some plan meets every limit.
    pub feasible: bool,
    /// Pareto front of speed against quality over the plans that meet the limits (all plans when
    /// none does), fastest first.
    pub front: Vec<usize>,
    /// Empty when feasible. Otherwise each limit the closest plan misses, with its numbers.
    pub failing: Vec<String>,
    /// The closest plan when infeasible (equal to `chosen`).
    pub closest: Option<usize>,
}

/// Each limit `s` misses, with the shortfall relative to the limit.
pub fn violations(s: &Scored, h: &Hints) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    if let Some(t) = h.target_tok_s
        && s.tok_s < t
    {
        out.push((
            format!("target_tok_s: {:.3} tok/s < {t}", s.tok_s),
            (t - s.tok_s) / t,
        ));
    }
    if let Some(ms) = h.p95_ttft_ms
        && s.ttft_s * 1e3 > ms
    {
        out.push((
            format!("p95_ttft_ms: {:.3} ms > {ms}", s.ttft_s * 1e3),
            (s.ttft_s * 1e3 - ms) / ms,
        ));
    }
    if let Some(q) = h.max_quality_loss
        && s.quality_loss > q + QEPS
    {
        out.push((
            format!("max_quality_loss: {:.6} > {q}", s.quality_loss),
            (s.quality_loss - q) / q.max(1e-6),
        ));
    }
    out
}

/// Losses closer than this are equal (summation order noise).
const QEPS: f64 = 1e-12;

fn cmp_loss(a: f64, b: f64) -> Ordering {
    if (a - b).abs() <= QEPS * (1.0 + a.abs() + b.abs()) {
        Ordering::Equal
    } else {
        a.total_cmp(&b)
    }
}

/// Quality first, then resource cost, then speed, then index.
fn cmp_objective(scored: &[Scored], a: usize, b: usize) -> Ordering {
    let (x, y) = (&scored[a], &scored[b]);
    cmp_loss(x.quality_loss, y.quality_loss)
        .then(cmp_loss(x.resource_cost, y.resource_cost))
        .then(y.tok_s.total_cmp(&x.tok_s))
        .then(a.cmp(&b))
}

/// Indices of `pool` not dominated in (tok/s up, loss down), fastest first. Of plans with the
/// same speed and loss the cheapest comes first.
pub fn pareto_front(scored: &[Scored], pool: &[usize]) -> Vec<usize> {
    let mut order = pool.to_vec();
    order.sort_by(|&a, &b| {
        let (x, y) = (&scored[a], &scored[b]);
        y.tok_s
            .total_cmp(&x.tok_s)
            .then(cmp_loss(x.quality_loss, y.quality_loss))
            .then(cmp_loss(x.resource_cost, y.resource_cost))
            .then(a.cmp(&b))
    });
    let mut front: Vec<usize> = Vec::new();
    for i in order {
        let better = front
            .last()
            .is_none_or(|&l| cmp_loss(scored[i].quality_loss, scored[l].quality_loss).is_lt());
        if better {
            front.push(i);
        }
    }
    front
}

/// Pick the plan (module docs). `tradeoff` selects the front point whose normalized quality
/// position `(loss_max - loss) / (loss_max - loss_min)` is closest to it: 0 is the fastest
/// point, 1 the best-quality one.
pub fn choose(scored: &[Scored], hints: &Hints) -> Choice {
    if scored.is_empty() {
        return Choice {
            chosen: None,
            feasible: false,
            front: Vec::new(),
            failing: vec!["no plan found".into()],
            closest: None,
        };
    }
    let all: Vec<usize> = (0..scored.len()).collect();
    let ok: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&i| violations(&scored[i], hints).is_empty())
        .collect();
    if ok.is_empty() {
        let key = |i: usize| -> f64 { violations(&scored[i], hints).iter().map(|v| v.1).sum() };
        let closest = all
            .iter()
            .copied()
            .min_by(|&a, &b| key(a).total_cmp(&key(b)).then(cmp_objective(scored, a, b)))
            .expect("scored is not empty");
        return Choice {
            chosen: Some(closest),
            feasible: false,
            front: pareto_front(scored, &all),
            failing: violations(&scored[closest], hints)
                .into_iter()
                .map(|v| v.0)
                .collect(),
            closest: Some(closest),
        };
    }
    let front = pareto_front(scored, &ok);
    let by_objective = ok
        .iter()
        .copied()
        .min_by(|&a, &b| cmp_objective(scored, a, b))
        .expect("ok is not empty");
    let chosen = match hints.tradeoff {
        None => by_objective,
        Some(t) => {
            let loss = |i: usize| scored[i].quality_loss;
            let hi = front.iter().map(|&i| loss(i)).fold(f64::MIN, f64::max);
            let lo = front.iter().map(|&i| loss(i)).fold(f64::MAX, f64::min);
            let span = hi - lo;
            let pos = |i: usize| {
                if span > 0.0 {
                    (hi - loss(i)) / span
                } else {
                    0.0
                }
            };
            front
                .iter()
                .copied()
                .min_by(|&a, &b| (pos(a) - t).abs().total_cmp(&(pos(b) - t).abs()))
                .unwrap_or(by_objective)
        }
    };
    Choice {
        chosen: Some(chosen),
        feasible: true,
        front,
        failing: Vec::new(),
        closest: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(tok_s: f64, loss: f64, cost: f64) -> Scored {
        Scored {
            desc: String::new(),
            plan: Plan {
                model: "m".into(),
                knobs: Default::default(),
                replicas: Vec::new(),
            },
            tok_s,
            ttft_s: 0.1,
            tpot_s: None,
            simulated: false,
            quality_loss: loss,
            resource_cost: cost,
            headroom_bytes: 0,
            stage_times: Vec::new(),
        }
    }

    #[test]
    fn order_is_quality_then_cost_then_speed() {
        let v = [
            s(50.0, 0.02, 1.0),
            s(10.0, 0.01, 2.0),
            s(20.0, 0.01, 1.0),
            s(30.0, 0.01, 1.0),
        ];
        let c = choose(&v, &Hints::default());
        assert_eq!(c.chosen, Some(3));
        assert!(c.feasible && c.failing.is_empty() && c.closest.is_none());
        assert_eq!(c.front, vec![0, 3]);
    }

    #[test]
    fn limits_filter_before_the_order() {
        let v = [s(50.0, 0.02, 1.0), s(10.0, 0.01, 1.0)];
        let h = Hints {
            target_tok_s: Some(20.0),
            ..Default::default()
        };
        assert_eq!(choose(&v, &h).chosen, Some(0));
        let h = Hints {
            max_quality_loss: Some(0.015),
            ..Default::default()
        };
        assert_eq!(choose(&v, &h).chosen, Some(1));
        let mut w = v.clone();
        w[0].ttft_s = 3.0;
        let h = Hints {
            p95_ttft_ms: Some(1000.0),
            target_tok_s: Some(20.0),
            ..Default::default()
        };
        let c = choose(&w, &h);
        assert!(!c.feasible);
        // plan 0 misses the ttft limit by 200%, plan 1 the speed limit by 50%
        assert_eq!(c.chosen, Some(1));
        assert_eq!(c.failing.len(), 1);
        assert!(c.failing[0].starts_with("target_tok_s"));
    }

    #[test]
    fn infeasible_names_each_failing_limit_and_returns_the_closest() {
        let v = [s(5.0, 0.05, 1.0), s(9.0, 0.03, 1.0)];
        let h = Hints {
            target_tok_s: Some(20.0),
            max_quality_loss: Some(0.01),
            ..Default::default()
        };
        let c = choose(&v, &h);
        assert!(!c.feasible);
        assert_eq!((c.chosen, c.closest), (Some(1), Some(1)));
        assert_eq!(c.failing.len(), 2);
        assert!(!c.front.is_empty());
    }

    #[test]
    fn tradeoff_walks_the_front() {
        let v = [
            s(100.0, 0.10, 1.0),
            s(80.0, 0.06, 1.0),
            s(50.0, 0.02, 1.0),
            s(40.0, 0.03, 1.0),
            s(20.0, 0.00, 1.0),
        ];
        let at = |t| {
            let h = Hints {
                tradeoff: Some(t),
                ..Default::default()
            };
            choose(&v, &h).chosen.unwrap()
        };
        assert_eq!(choose(&v, &Hints::default()).front, vec![0, 1, 2, 4]);
        assert_eq!((at(0.0), at(1.0)), (0, 4));
        assert_eq!(at(0.5), 1);
    }

    #[test]
    fn equal_points_keep_the_cheapest() {
        let v = [s(10.0, 0.01, 3.0), s(10.0, 0.01, 1.0)];
        assert_eq!(pareto_front(&v, &[0, 1]), vec![1]);
    }

    #[test]
    fn empty_input_has_no_choice() {
        let c = choose(&[], &Hints::default());
        assert_eq!(c.chosen, None);
        assert!(!c.failing.is_empty());
    }
}

//! Layer split across uneven pipeline stages by dynamic programming (docs/plan.md "Search",
//! step 3): minimize the slowest stage.

use std::cmp::Ordering;

/// Result of [`split_layers`].
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// `stages + 1` layer boundaries, starting at 0 and ending at the layer count.
    pub bounds: Vec<u32>,
    /// Cost of the slowest stage; optimal.
    pub bottleneck: f64,
    /// Sum of the stage costs. Among splits with the optimal bottleneck the DP keeps the smaller
    /// sum only as a heuristic tie-break (it is not an exact second criterion).
    pub total: f64,
}

fn less(a: (f64, f64), b: (f64, f64)) -> bool {
    match a.0.total_cmp(&b.0) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.1.total_cmp(&b.1).is_lt(),
    }
}

/// Split `layers` into `stages` non-empty consecutive spans. `cost(stage, start, end)` is the
/// time of the span `[start, end)` on that stage (hop included); a non-finite cost forbids it.
/// `None` when no split avoids forbidden spans or there are more stages than layers.
pub fn split_layers(
    layers: u32,
    stages: usize,
    cost: impl Fn(usize, u32, u32) -> f64,
) -> Option<Split> {
    let n = layers as usize;
    if stages == 0 || stages > n {
        return None;
    }
    // best[i][j]: (bottleneck, total) of the first i stages covering j layers.
    let mut best: Vec<Vec<Option<(f64, f64)>>> = vec![vec![None; n + 1]; stages + 1];
    let mut from = vec![vec![0usize; n + 1]; stages + 1];
    best[0][0] = Some((0.0, 0.0));
    for i in 1..=stages {
        for j in i..=n {
            for k in (i - 1)..j {
                let Some((bottleneck, total)) = best[i - 1][k] else {
                    continue;
                };
                let c = cost(i - 1, k as u32, j as u32);
                if !c.is_finite() {
                    continue;
                }
                let cand = (bottleneck.max(c), total + c);
                if best[i][j].is_none_or(|cur| less(cand, cur)) {
                    best[i][j] = Some(cand);
                    from[i][j] = k;
                }
            }
        }
    }
    let (bottleneck, total) = best[stages][n]?;
    let mut bounds = vec![0u32; stages + 1];
    bounds[stages] = layers;
    let mut j = n;
    for i in (1..=stages).rev() {
        j = from[i][j];
        bounds[i - 1] = j as u32;
    }
    Some(Split {
        bounds,
        bottleneck,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Every split into `stages` non-empty spans, as boundary vectors.
    fn all_splits(layers: u32, stages: usize) -> Vec<Vec<u32>> {
        fn go(layers: u32, stages: usize, at: u32, acc: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            if stages == 1 {
                let mut b = acc.clone();
                b.push(layers);
                out.push(b);
                return;
            }
            for next in at + 1..=layers - (stages as u32 - 1) {
                acc.push(next);
                go(layers, stages - 1, next, acc, out);
                acc.pop();
            }
        }
        let mut out = Vec::new();
        go(layers, stages, 0, &mut vec![0], &mut out);
        out
    }

    fn eval(b: &[u32], cost: &impl Fn(usize, u32, u32) -> f64) -> Option<(f64, f64)> {
        let mut bn = 0.0f64;
        let mut total = 0.0;
        for (i, w) in b.windows(2).enumerate() {
            let c = cost(i, w[0], w[1]);
            if !c.is_finite() {
                return None;
            }
            bn = bn.max(c);
            total += c;
        }
        Some((bn, total))
    }

    #[test]
    fn uneven_stages_balance_the_slowest() {
        // Stage 0 is twice as fast as stage 1: 2 us per layer against 1 us... per layer cost.
        let cost = |s: usize, a: u32, b: u32| f64::from(b - a) * if s == 0 { 1.0 } else { 2.0 };
        let s = split_layers(9, 2, cost).unwrap();
        assert_eq!(s.bounds, vec![0, 6, 9]);
        assert_eq!(s.bottleneck, 6.0);
        assert_eq!(split_layers(1, 2, cost), None);
    }

    #[test]
    fn forbidden_spans_are_avoided() {
        // stage 1 cannot take layer 0..3
        let cost = |s: usize, a: u32, b: u32| {
            if s == 1 && a < 3 {
                f64::INFINITY
            } else {
                f64::from(b - a)
            }
        };
        let s = split_layers(6, 2, cost).unwrap();
        assert_eq!(s.bounds, vec![0, 3, 6]);
        let none = |_: usize, _: u32, _: u32| f64::INFINITY;
        assert_eq!(split_layers(4, 2, none), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn dp_matches_brute_force(
            layers in 1u32..9,
            stages in 1usize..5,
            table in proptest::collection::vec(0u32..30, 4 * 9 * 10),
        ) {
            // cost(stage, a, b) from a random table; ~1 in 8 entries forbidden.
            let cost = |s: usize, a: u32, b: u32| {
                let v = table[(s * 9 + a as usize) * 10 + (b - a) as usize % 10];
                if v % 8 == 7 { f64::INFINITY } else { f64::from(v) }
            };
            let dp = split_layers(layers, stages, cost);
            let brute = if stages > layers as usize {
                None
            } else {
                all_splits(layers, stages)
                    .iter()
                    .filter_map(|b| eval(b, &cost))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
            };
            match (dp, brute) {
                (None, None) => {}
                (Some(d), Some(b)) => {
                    prop_assert_eq!(d.bottleneck, b.0);
                    // the reported numbers are those of the reported split
                    prop_assert_eq!(eval(&d.bounds, &cost), Some((d.bottleneck, d.total)));
                    prop_assert_eq!(d.bounds.len(), stages + 1);
                    prop_assert!(d.bounds.windows(2).all(|w| w[0] < w[1]));
                }
                (d, b) => prop_assert!(false, "dp {:?} brute {:?}", d, b),
            }
        }
    }
}

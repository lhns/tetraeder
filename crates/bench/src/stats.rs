//! Median, spread and a paired bootstrap confidence interval.

use serde::{Deserialize, Serialize};

/// Summary of one sample set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub n: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    /// Interquartile range over the median; 0 for an empty or all-zero set.
    pub rel_iqr: f64,
}

/// Paired ratio b/a with a bootstrap confidence interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ratio {
    pub n: usize,
    pub median: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub confidence: f64,
}

pub fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// Linear-interpolated quantile; NaN for an empty slice.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

pub fn summarize(v: &[f64]) -> Summary {
    let n = v.len();
    let med = median(v);
    let mean = v.iter().sum::<f64>() / n.max(1) as f64;
    let var = if n > 1 {
        v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let iqr = quantile(v, 0.75) - quantile(v, 0.25);
    Summary {
        n,
        median: med,
        min: v.iter().copied().fold(f64::INFINITY, f64::min),
        max: v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        stddev: var.sqrt(),
        rel_iqr: if med != 0.0 && iqr.is_finite() {
            iqr / med
        } else {
            0.0
        },
    }
}

/// Small deterministic generator (splitmix64) so results are reproducible from the spec seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Median of per-pair ratios b[i]/a[i] with a percentile bootstrap over pairs.
/// `a` and `b` are the interleaved repeats of one session, so index i is one pair.
pub fn paired_ratio(
    a: &[f64],
    b: &[f64],
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> Option<Ratio> {
    let n = a.len().min(b.len());
    if n == 0 {
        return None;
    }
    let ratios: Vec<f64> = (0..n)
        .map(|i| b[i] / a[i])
        .filter(|r| r.is_finite())
        .collect();
    if ratios.is_empty() {
        return None;
    }
    let mut rng = Rng::new(seed);
    let mut meds = Vec::with_capacity(resamples);
    let mut buf = vec![0.0; ratios.len()];
    for _ in 0..resamples {
        for slot in buf.iter_mut() {
            *slot = ratios[rng.below(ratios.len())];
        }
        meds.push(median(&buf));
    }
    let tail = (1.0 - confidence) / 2.0;
    Some(Ratio {
        n: ratios.len(),
        median: median(&ratios),
        ci_lo: quantile(&meds, tail),
        ci_hi: quantile(&meds, 1.0 - tail),
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_quantiles() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
        assert_eq!(quantile(&[0.0, 10.0], 0.25), 2.5);
    }

    #[test]
    fn summary_fields() {
        let s = summarize(&[10.0, 12.0, 11.0, 13.0, 9.0]);
        assert_eq!(s.n, 5);
        assert_eq!(s.median, 11.0);
        assert_eq!((s.min, s.max), (9.0, 13.0));
        assert!((s.stddev - 1.5811).abs() < 1e-3);
        assert!((s.rel_iqr - 2.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn constant_ratio_has_tight_interval() {
        let a = [100.0, 98.0, 102.0, 101.0, 99.0];
        let b: Vec<f64> = a.iter().map(|x| x * 0.9).collect();
        let r = paired_ratio(&a, &b, 2000, 0.95, 1).unwrap();
        assert!((r.median - 0.9).abs() < 1e-12);
        assert!((r.ci_lo - 0.9).abs() < 1e-12 && (r.ci_hi - 0.9).abs() < 1e-12);
    }

    #[test]
    fn noisy_ratio_interval_contains_median_and_is_deterministic() {
        let a = [100.0, 90.0, 110.0, 100.0, 95.0, 105.0];
        let b = [101.0, 88.0, 112.0, 97.0, 96.0, 104.0];
        let r1 = paired_ratio(&a, &b, 2000, 0.95, 7).unwrap();
        let r2 = paired_ratio(&a, &b, 2000, 0.95, 7).unwrap();
        assert_eq!(r1, r2);
        assert!(r1.ci_lo <= r1.median && r1.median <= r1.ci_hi);
        assert!(r1.ci_lo < 1.0 && r1.ci_hi > 1.0);
    }

    #[test]
    fn empty_input_gives_none() {
        assert!(paired_ratio(&[], &[], 10, 0.95, 1).is_none());
    }
}

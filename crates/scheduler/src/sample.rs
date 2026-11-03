//! Sampling over a logits row (ADR 0040): mask, greedy, or temperature with top-k and top-p.

use std::sync::atomic::{AtomicU64, Ordering};
use tetraeder_gateway::engine::TokenMask;

/// SplitMix64: a tiny seedable generator, enough for sampling.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Seed for requests that send none: distinct per request, reproducible per process start.
pub fn counter_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Resolved sampling settings of one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleParams {
    /// 0 or less: greedy.
    pub temperature: f32,
    /// 0: off.
    pub top_k: u32,
    /// 1 or more: off.
    pub top_p: f32,
}

/// One request's sampler; keeps a scratch buffer between steps.
pub struct Sampler {
    params: SampleParams,
    rng: SplitMix64,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(params: SampleParams, seed: u64) -> Self {
        Self {
            params,
            rng: SplitMix64::new(seed),
            cand: Vec::new(),
        }
    }

    /// The next token, or `None` when the mask leaves no token with a finite logit.
    pub fn sample(&mut self, logits: &[f32], mask: Option<&TokenMask>) -> Option<u32> {
        let allowed = |i: usize| mask.is_none_or(|m| m.allows(i as u32));
        if self.params.temperature <= 0.0 {
            return greedy(logits, allowed);
        }
        let c = &mut self.cand;
        c.clear();
        c.extend(
            logits
                .iter()
                .enumerate()
                .filter(|&(i, l)| l.is_finite() && allowed(i))
                .map(|(i, &l)| (l, i as u32)),
        );
        if c.is_empty() {
            return None;
        }
        // Descending by logit; lower id first on ties, so greedy-like order is stable.
        let by_logit = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        let k = self.params.top_k as usize;
        if k > 0 && k < c.len() {
            c.select_nth_unstable_by(k - 1, by_logit);
            c.truncate(k);
        }
        c.sort_unstable_by(by_logit);
        let max = c[0].0;
        let inv_t = 1.0 / self.params.temperature;
        let mut sum = 0.0f32;
        for e in c.iter_mut() {
            e.0 = ((e.0 - max) * inv_t).exp();
            sum += e.0;
        }
        // Smallest prefix whose mass reaches top_p (at least one token).
        let mut keep = c.len();
        if self.params.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, e) in c.iter().enumerate() {
                acc += e.0 / sum;
                if acc >= self.params.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        c.truncate(keep);
        let total: f32 = c.iter().map(|e| e.0).sum();
        let mut r = self.rng.next_f32() * total;
        for e in c.iter() {
            r -= e.0;
            if r < 0.0 {
                return Some(e.1);
            }
        }
        c.last().map(|e| e.1)
    }
}

/// Argmax over the allowed ids; the lowest id wins ties, like libllama's greedy sampler.
pub fn greedy(logits: &[f32], allowed: impl Fn(usize) -> bool) -> Option<u32> {
    let mut best: Option<(f32, usize)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() || !allowed(i) {
            continue;
        }
        if best.is_none_or(|(b, _)| l > b) {
            best = Some((l, i));
        }
    }
    best.map(|(_, i)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(allowed: &[u32], n: usize) -> TokenMask {
        let mut words = vec![0u32; n.div_ceil(32)];
        for &a in allowed {
            words[a as usize / 32] |= 1 << (a % 32);
        }
        TokenMask { words }
    }

    #[test]
    fn greedy_takes_the_lowest_id_on_ties() {
        assert_eq!(greedy(&[1.0, 3.0, 3.0, 2.0], |_| true), Some(1));
    }

    #[test]
    fn mask_restricts_greedy_and_random() {
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0];
        let m = mask(&[3, 4], 5);
        let mut g = Sampler::new(
            SampleParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
            },
            1,
        );
        assert_eq!(g.sample(&logits, Some(&m)), Some(3));
        let mut r = Sampler::new(
            SampleParams {
                temperature: 1.0,
                top_k: 40,
                top_p: 1.0,
            },
            7,
        );
        for _ in 0..50 {
            assert!(matches!(r.sample(&logits, Some(&m)), Some(3 | 4)));
        }
        assert_eq!(r.sample(&logits, Some(&mask(&[], 5))), None);
    }

    #[test]
    fn top_k_one_and_tiny_top_p_are_greedy() {
        let logits = [0.5, 2.0, 1.0, 1.5];
        for p in [
            SampleParams {
                temperature: 2.0,
                top_k: 1,
                top_p: 1.0,
            },
            SampleParams {
                temperature: 2.0,
                top_k: 0,
                top_p: 0.01,
            },
        ] {
            let mut s = Sampler::new(p, 3);
            for _ in 0..20 {
                assert_eq!(s.sample(&logits, None), Some(1));
            }
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let logits: Vec<f32> = (0..100).map(|i| (i % 7) as f32 * 0.3).collect();
        let p = SampleParams {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
        };
        let run = |seed| {
            let mut s = Sampler::new(p, seed);
            (0..30)
                .map(|_| s.sample(&logits, None).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }
}

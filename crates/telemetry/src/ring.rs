use crate::metrics::StepSample;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Fixed-size buffer of recent step samples.
#[derive(Debug)]
pub struct StepRing {
    cap: usize,
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    q: VecDeque<StepSample>,
    /// Samples pushed since creation (the deque forgets evicted ones).
    pushed: u64,
}

/// Summary of the samples currently in the ring.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepStats {
    pub count: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
    pub tokens: u64,
}

impl StepRing {
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            cap,
            inner: Mutex::new(Inner {
                q: VecDeque::with_capacity(cap),
                pushed: 0,
            }),
        }
    }

    pub fn push(&self, s: StepSample) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.q.len() == self.cap {
            g.q.pop_front();
        }
        g.q.push_back(s);
        g.pushed += 1;
    }

    /// Samples pushed since creation; a reader that remembers it knows how many of the newest
    /// samples in [`Self::snapshot`] it has not seen.
    pub fn pushed(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).pushed
    }

    /// Copy of the buffer, oldest first.
    pub fn snapshot(&self) -> Vec<StepSample> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.q.iter().copied().collect()
    }

    pub fn stats(&self) -> StepStats {
        let samples = self.snapshot();
        let tokens = samples.iter().map(|s| u64::from(s.tokens)).sum();
        let mut ms: Vec<f64> = samples
            .iter()
            .map(|s| s.duration.as_secs_f64() * 1e3)
            .collect();
        if ms.is_empty() {
            return StepStats::default();
        }
        ms.sort_by(f64::total_cmp);
        let pick = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
        StepStats {
            count: ms.len(),
            mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
            p50_ms: pick(0.5),
            p95_ms: pick(0.95),
            max_ms: ms[ms.len() - 1],
            tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Phase;
    use std::time::Duration;

    fn step(ms: u64) -> StepSample {
        StepSample {
            phase: Phase::Decode,
            stage: 0,
            graphs: true,
            duration: Duration::from_millis(ms),
            tokens: 1,
            segments: 1,
        }
    }

    #[test]
    fn ring_evicts_oldest_and_computes_stats() {
        let r = StepRing::new(3);
        for ms in [100, 1, 2, 3] {
            r.push(step(ms));
        }
        let snap = r.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].duration, Duration::from_millis(1));
        let st = r.stats();
        assert_eq!(st.count, 3);
        assert_eq!(st.p50_ms, 2.0);
        assert_eq!(st.max_ms, 3.0);
        assert_eq!(st.tokens, 3);
    }

    #[test]
    fn pushed_counts_evicted_samples_too() {
        let r = StepRing::new(2);
        assert_eq!(r.pushed(), 0);
        for ms in 1..=5 {
            r.push(step(ms));
        }
        assert_eq!((r.pushed(), r.snapshot().len()), (5, 2));
    }
}

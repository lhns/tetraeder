//! Adaptive draft length (ADR 0043): maximize expected tokens per second of the step.

use super::MAX_VERIFY;

const ACCEPT_ALPHA: f64 = 0.2;
const TIME_ALPHA: f64 = 0.3;

/// The k in `0..=max` that maximizes `(1 - a^(k+1)) / (1 - a) / t(k+1)`, the expected tokens of
/// a step per second. Ties go to the smaller k. `t(n)` is the step time of an `n`-row batch.
pub fn choose_k(a: f64, max: usize, t: impl Fn(usize) -> f64) -> usize {
    let a = a.clamp(0.05, 0.95);
    let mut best = (0, f64::MIN);
    for k in 0..=max.min(MAX_VERIFY - 1) {
        let tokens = (1.0 - a.powi(k as i32 + 1)) / (1.0 - a);
        let score = tokens / t(k + 1).max(1e-9);
        if score > best.1 {
            best = (k, score);
        }
    }
    best.0
}

/// Acceptance and step-time estimates behind `DraftLen::Adaptive`.
#[derive(Debug, Clone)]
pub(super) struct Adaptive {
    max: usize,
    /// Moving averages of accepted drafts and rejections per step; start at a = 0.5.
    accepted: f64,
    rejected: f64,
    /// Moving average of the step time per batch size (index n - 1).
    time: [Option<f64>; MAX_VERIFY],
}

impl Adaptive {
    pub fn new(max: usize) -> Self {
        Self {
            max: max.min(MAX_VERIFY - 1),
            accepted: 1.0,
            rejected: 1.0,
            time: [None; MAX_VERIFY],
        }
    }

    pub fn acceptance(&self) -> f64 {
        (self.accepted / (self.accepted + self.rejected)).clamp(0.05, 0.95)
    }

    /// Step time for `n` rows: measured, else interpolated (extrapolated beyond the seen
    /// sizes), else flat at the one measured value, else 1 (optimistic: explores).
    pub fn time_at(&self, n: usize) -> f64 {
        let seen: Vec<(f64, f64)> = self
            .time
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.map(|t| ((i + 1) as f64, t)))
            .collect();
        let x = n as f64;
        match seen.as_slice() {
            [] => 1.0,
            [(_, t)] => *t,
            _ => {
                if let Some(&(_, t)) = seen.iter().find(|p| p.0 == x) {
                    return t;
                }
                let i = seen.partition_point(|p| p.0 < x).clamp(1, seen.len() - 1);
                let ((x0, t0), (x1, t1)) = (seen[i - 1], seen[i]);
                let t = t0 + (t1 - t0) * (x - x0) / (x1 - x0);
                let floor = seen.iter().map(|p| p.1).fold(f64::MAX, f64::min);
                t.max(floor)
            }
        }
    }

    pub fn next_k(&self) -> usize {
        choose_k(self.acceptance(), self.max, |n| self.time_at(n))
    }

    /// A step of `drafted` > 0 drafts of which `accepted` matched.
    pub fn observe(&mut self, drafted: usize, accepted: usize) {
        let rejected = usize::from(accepted < drafted);
        self.accepted += ACCEPT_ALPHA * (accepted as f64 - self.accepted);
        self.rejected += ACCEPT_ALPHA * (rejected as f64 - self.rejected);
    }

    /// Eval time of a step with `n` rows.
    pub fn observe_time(&mut self, n: usize, secs: f64) {
        let t = &mut self.time[n - 1];
        *t = Some(t.map_or(secs, |old| old + TIME_ALPHA * (secs - old)));
    }
}

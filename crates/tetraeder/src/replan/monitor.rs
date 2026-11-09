//! The residual of the decode step time against the plan's prediction (ADR 0055, 3).

use tetraeder_control::Trigger;
use tetraeder_telemetry::{Metrics, Phase, StepRing};

/// Decode samples of one tick needed before the tick counts.
const MIN_SAMPLES: usize = 16;
/// Consecutive ticks over the threshold before a drift is reported.
const TICKS_OVER: u32 = 5;
/// Residuals below this are noise whatever the graph's spread says.
const MIN_THRESHOLD: f64 = 0.2;

/// Reads the new one-token decode steps from the step ring each tick, compares their median with
/// the predicted step and reports `Trigger::Drift` after [`TICKS_OVER`] ticks over the threshold.
/// Prefill steps are chunked and depend on the context, so they are not compared.
pub struct ResidualMonitor {
    predicted_step_s: Option<f64>,
    /// Residual band of the graph (largest measurement spread), 0 when unknown.
    band: f64,
    seen: u64,
    over: u32,
    /// Median decode step of the last tick that had enough samples.
    pub last_median_s: Option<f64>,
}

impl ResidualMonitor {
    pub fn new(band: f64) -> Self {
        Self {
            predicted_step_s: None,
            band,
            seen: 0,
            over: 0,
            last_median_s: None,
        }
    }

    pub fn threshold(&self) -> f64 {
        MIN_THRESHOLD.max(2.0 * self.band)
    }

    /// A new running plan (or the first one): records the prediction and restarts the count.
    /// `None` switches the monitor off (no prediction to compare with).
    pub fn set_prediction(&mut self, step_s: Option<f64>, m: Option<&Metrics>) {
        self.predicted_step_s = step_s;
        self.over = 0;
        self.last_median_s = None;
        if let (Some(s), Some(m)) = (step_s, m) {
            m.plan_predicted_step(0, s);
        }
    }

    pub fn tick(&mut self, ring: &StepRing, m: Option<&Metrics>) -> Option<Trigger> {
        let pushed = ring.pushed();
        if pushed <= self.seen {
            return None;
        }
        let new = usize::try_from(pushed - self.seen).unwrap_or(usize::MAX);
        self.seen = pushed;
        let predicted = self.predicted_step_s?;
        let snap = ring.snapshot();
        let mut steps: Vec<f64> = snap[snap.len().saturating_sub(new)..]
            .iter()
            .filter(|s| s.phase == Phase::Decode && s.tokens == 1)
            .map(|s| s.duration.as_secs_f64())
            .collect();
        if steps.len() < MIN_SAMPLES {
            return None;
        }
        steps.sort_by(f64::total_cmp);
        let median = steps[steps.len() / 2];
        self.last_median_s = Some(median);
        let residual = (median - predicted).abs() / predicted;
        if let Some(m) = m {
            m.plan_residual(0, residual);
        }
        if residual > self.threshold() {
            self.over += 1;
        } else {
            self.over = 0;
        }
        if self.over >= TICKS_OVER {
            self.over = 0;
            return Some(Trigger::Drift { residual });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tetraeder_telemetry::StepSample;

    fn sample(phase: Phase, ms: u64, tokens: u32) -> StepSample {
        StepSample {
            phase,
            stage: 0,
            graphs: false,
            duration: Duration::from_millis(ms),
            tokens,
            segments: 1,
        }
    }

    fn push(r: &StepRing, n: usize, ms: u64) {
        for _ in 0..n {
            r.push(sample(Phase::Decode, ms, 1));
        }
    }

    fn monitor() -> ResidualMonitor {
        let mut m = ResidualMonitor::new(0.0);
        m.set_prediction(Some(0.010), None);
        m
    }

    // hardware: none
    #[test]
    fn no_new_samples_or_too_few_do_nothing() {
        let (r, mut m) = (StepRing::new(1000), monitor());
        assert!(m.tick(&r, None).is_none());
        push(&r, 15, 50);
        for _ in 0..8 {
            assert!(m.tick(&r, None).is_none(), "15 samples");
        }
        assert_eq!(m.last_median_s, None);
        // no new samples since: still nothing, however often it ticks
        push(&r, 16, 50);
        for _ in 0..4 {
            m.tick(&r, None);
        }
        assert!(m.tick(&r, None).is_none());
    }

    // hardware: none
    #[test]
    fn five_ticks_over_the_band_report_one_drift() {
        let (r, mut m) = (StepRing::new(1000), monitor());
        for i in 1..=5 {
            push(&r, 20, 50);
            let t = m.tick(&r, None);
            if i < 5 {
                assert!(t.is_none(), "tick {i}");
            } else {
                match t {
                    Some(Trigger::Drift { residual }) => assert!((residual - 4.0).abs() < 1e-9),
                    other => panic!("{other:?}"),
                }
            }
        }
        // the count restarted
        push(&r, 20, 50);
        assert!(m.tick(&r, None).is_none());
        assert_eq!(m.last_median_s, Some(0.05));
    }

    // hardware: none
    #[test]
    fn a_tick_under_the_band_resets_the_count() {
        let (r, mut m) = (StepRing::new(1000), monitor());
        for _ in 0..4 {
            push(&r, 20, 50);
            assert!(m.tick(&r, None).is_none());
        }
        push(&r, 20, 11); // residual 0.1 < 0.2
        assert!(m.tick(&r, None).is_none());
        for i in 1..=5 {
            push(&r, 20, 50);
            assert_eq!(m.tick(&r, None).is_some(), i == 5, "tick {i}");
        }
    }

    // hardware: none
    #[test]
    fn prefill_and_batched_steps_are_ignored() {
        let (r, mut m) = (StepRing::new(1000), monitor());
        for _ in 0..8 {
            for _ in 0..40 {
                r.push(sample(Phase::Prefill, 500, 512));
                r.push(sample(Phase::Decode, 500, 4));
            }
            assert!(m.tick(&r, None).is_none());
        }
        assert_eq!(m.last_median_s, None);
    }

    // hardware: none
    #[test]
    fn the_band_widens_the_threshold() {
        let mut m = ResidualMonitor::new(0.4);
        assert!((m.threshold() - 0.8).abs() < 1e-12);
        m.set_prediction(Some(0.010), None);
        let r = StepRing::new(1000);
        for _ in 0..6 {
            push(&r, 20, 15); // residual 0.5, under 0.8
            assert!(m.tick(&r, None).is_none());
        }
        assert!((ResidualMonitor::new(0.0).threshold() - 0.2).abs() < 1e-12);
    }

    // hardware: none
    #[test]
    fn without_a_prediction_nothing_is_reported() {
        let r = StepRing::new(1000);
        let mut m = ResidualMonitor::new(0.0);
        for _ in 0..8 {
            push(&r, 20, 500);
            assert!(m.tick(&r, None).is_none());
        }
    }
}

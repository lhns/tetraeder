use super::policy::Adaptive;
use super::{DraftLen, Drafter, MAX_VERIFY, Sampling, SpecError, Target, choose};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_telemetry::Telemetry;

/// Steps of one verify batch size and their summed eval time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BatchStat {
    pub steps: u64,
    pub time: Duration,
}

/// Cumulative counters of a [`Speculator`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpecStats {
    pub steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Tokens appended to the history (accepted drafts plus one per step).
    pub emitted: u64,
    /// Target eval time.
    pub verify_time: Duration,
    pub draft_time: Duration,
    /// Index `n - 1` holds the steps that evaluated `n` rows.
    pub by_batch: [BatchStat; MAX_VERIFY],
}

pub struct Speculator {
    len: DraftLen,
    adaptive: Option<Adaptive>,
    sampling: Sampling,
    telemetry: Option<Arc<Telemetry>>,
    stats: SpecStats,
}

impl Speculator {
    pub fn new(len: DraftLen, sampling: Sampling, telemetry: Option<Arc<Telemetry>>) -> Self {
        let adaptive = match len {
            DraftLen::Adaptive { max } => Some(Adaptive::new(max)),
            DraftLen::Fixed(_) => None,
        };
        Self {
            len,
            adaptive,
            sampling,
            telemetry,
            stats: SpecStats::default(),
        }
    }

    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    fn draft_len(&self) -> usize {
        let k = match (&self.adaptive, self.len) {
            (Some(a), _) => a.next_k(),
            (None, DraftLen::Fixed(k)) => k,
            (None, DraftLen::Adaptive { max }) => max,
        };
        k.min(MAX_VERIFY - 1)
    }

    /// One step. Precondition: `target.n_past() == history.len() - 1` (the last history token is
    /// pending). Drafts k tokens unless `drafter` is `None`, evaluates `[last, drafts..]`,
    /// appends the accepted drafts and one chosen token to `history` and rewinds the target to
    /// the accepted prefix. Returns the number of tokens appended (at least 1).
    pub fn step<'d>(
        &mut self,
        target: &mut dyn Target,
        drafter: Option<&mut (dyn Drafter + 'd)>,
        history: &mut Vec<i32>,
    ) -> Result<usize, SpecError> {
        let Some(&last) = history.last() else {
            return Err(SpecError::Invalid("empty history".into()));
        };
        let n_past = target.n_past();
        if n_past + 1 != history.len() {
            return Err(SpecError::Invalid(format!(
                "target has {n_past} tokens cached, history {} (the last one must be pending)",
                history.len()
            )));
        }

        let t0 = Instant::now();
        let mut method = "";
        let mut drafts = Vec::new();
        if let Some(d) = drafter {
            let k = self.draft_len();
            if k > 0 {
                drafts = d.draft(history, k)?;
                drafts.truncate(k);
                method = d.method();
            }
        }
        self.stats.draft_time += t0.elapsed();

        let mut batch = Vec::with_capacity(drafts.len() + 1);
        batch.push(last);
        batch.extend_from_slice(&drafts);
        let t0 = Instant::now();
        let logits = target.eval(&batch)?;
        let eval_time = t0.elapsed();
        let nv = target.n_vocab();
        if logits.len() != batch.len() * nv {
            return Err(SpecError::Invalid(format!(
                "target returned {} logits for {} rows of {nv}",
                logits.len(),
                batch.len()
            )));
        }

        let start = history.len();
        let mut accepted = 0;
        for i in 0..batch.len() {
            let tok = choose(&logits[i * nv..(i + 1) * nv], start + i, &self.sampling);
            history.push(tok);
            if drafts.get(i) == Some(&tok) {
                accepted += 1;
            } else {
                break;
            }
        }
        let emitted = history.len() - start;
        debug_assert_eq!(emitted, accepted + 1);
        let keep = n_past + 1 + accepted;
        if keep < target.n_past() {
            target.rewind(keep)?;
        }

        let s = &mut self.stats;
        s.steps += 1;
        s.drafted += drafts.len() as u64;
        s.accepted += accepted as u64;
        s.emitted += emitted as u64;
        s.verify_time += eval_time;
        let b = &mut s.by_batch[batch.len() - 1];
        b.steps += 1;
        b.time += eval_time;
        if let Some(a) = &mut self.adaptive {
            a.observe_time(batch.len(), eval_time.as_secs_f64());
            if !drafts.is_empty() {
                a.observe(drafts.len(), accepted);
            }
        }
        if !drafts.is_empty()
            && let Some(t) = &self.telemetry
        {
            t.metrics()
                .spec_step(method, drafts.len() as u64, accepted as u64);
        }
        Ok(emitted)
    }

    /// Steps until at least `n` tokens were appended; returns how many (it can overshoot by
    /// up to k).
    pub fn generate<'d>(
        &mut self,
        target: &mut dyn Target,
        mut drafter: Option<&mut (dyn Drafter + 'd)>,
        history: &mut Vec<i32>,
        n: usize,
    ) -> Result<usize, SpecError> {
        let mut total = 0;
        while total < n {
            total += self.step(target, drafter.as_deref_mut(), history)?;
        }
        Ok(total)
    }
}

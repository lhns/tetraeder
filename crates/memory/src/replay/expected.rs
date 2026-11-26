//! Expected uses of an expert within one tier turnover (Magnitude's bank ranking, `docs/plan.md`
//! references), the value behind [`super::sim::Policy::GlobalExpected`].
//!
//! History: a per-(layer, expert) count of decode steps that routed it, decayed with time
//! constant `horizon` decode steps. At steady use rate `p` the count settles near `p * horizon`,
//! which is the expected number of uses in the next `horizon` steps (the tier turnover). It is
//! seeded at the first decode step from the prefill routing counts, normalised per token.
//! Lookahead: an expert named by an unconsumed router prediction (`predicted`, `predicted2` of
//! the trace) is expected to be used once more, weighted by the predictions' running precision.
//! The trace holds no next-token routing, so that is the only lookahead.

/// Decayed use counts and pending predictions of every (layer, expert).
pub struct Expected {
    horizon: f64,
    now: u32,
    uses: Vec<Vec<f64>>,
    last: Vec<Vec<u32>>,
    /// Named by a prediction for this layer that its step has not consumed yet.
    look: Vec<Vec<bool>>,
    /// Predicted experts that were routed, and all predicted (with a prior of 4 of 5).
    right: u64,
    named: u64,
    /// `exp(-lag / horizon)` by lag in decode steps.
    decay: Vec<f64>,
}

impl Expected {
    pub fn new(layers: usize, experts: usize, horizon: u32) -> Expected {
        Expected {
            horizon: f64::from(horizon.max(1)),
            now: 0,
            uses: vec![vec![0.0; experts]; layers],
            last: vec![vec![0; experts]; layers],
            look: vec![vec![false; experts]; layers],
            right: 4,
            named: 5,
            decay: Vec::new(),
        }
    }

    /// Seeds the history as decode starts: per layer, `counts / total * n_used` is the share of
    /// tokens that route the expert, times the horizon.
    pub fn seed(&mut self, counts: &[Vec<u64>], n_used: u32) {
        for (il, c) in counts.iter().enumerate() {
            let total = c.iter().sum::<u64>().max(1) as f64;
            for (e, &n) in c.iter().enumerate() {
                self.uses[il][e] = n as f64 / total * f64::from(n_used) * self.horizon;
                self.last[il][e] = self.now;
            }
        }
    }

    /// A new decode step `now` begins.
    pub fn advance(&mut self, now: u32) {
        self.now = now;
        while self.decay.len() <= now as usize {
            self.decay.push((-(self.decay.len() as f64) / self.horizon).exp());
        }
    }

    fn history(&self, il: usize, e: usize) -> f64 {
        self.uses[il][e] * self.decay[(self.now - self.last[il][e]) as usize]
    }

    /// Expected uses within the next `horizon` decode steps.
    pub fn value(&self, layer: u32, expert: u32) -> f64 {
        let (il, e) = (layer as usize, expert as usize);
        let look = if self.look[il][e] {
            self.right as f64 / self.named as f64
        } else {
            0.0
        };
        self.history(il, e) + look
    }

    /// The decode step of `layer` routed `experts` (distinct): score the predictions made for
    /// it, then count the uses.
    pub fn routed(&mut self, layer: u32, experts: &[u32]) {
        let il = layer as usize;
        for e in 0..self.look[il].len() {
            if std::mem::take(&mut self.look[il][e]) {
                self.named += 1;
                self.right += u64::from(experts.contains(&(e as u32)));
            }
        }
        for &e in experts {
            let e = e as usize;
            self.uses[il][e] = self.history(il, e) + 1.0;
            self.last[il][e] = self.now;
        }
    }

    /// A prediction names `experts` for `layer`.
    pub fn name(&mut self, layer: u32, experts: &[u16]) {
        for &e in experts {
            if let Some(x) = self.look[layer as usize].get_mut(usize::from(e)) {
                *x = true;
            }
        }
    }
}

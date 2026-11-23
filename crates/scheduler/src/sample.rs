//! Sampling over a logits row (ADR 0046, 0083 decision 7): the host sampler is libllama's own
//! sampler chain, built in llama-server's order with its defaults, so seeded and sampled outputs
//! follow `llama-server` at the pin. The token mask (tool calls, grammars) is applied first by
//! setting masked logits to `-inf`.

use std::sync::Arc;
use tetraeder_gateway::engine::{
    GrammarSource, SamplerKind, SamplingParams, TokenId, TokenMask, TokenProbs,
};
use tetraeder_ggml_sys::llama::{Model, Sampler as Chain};

/// `LLAMA_DEFAULT_SEED`: libllama draws a random seed.
const DEFAULT_SEED: u32 = u32::MAX;

/// Resolved sampling settings of one request; `Default` is llama-server's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    /// 0 or less: greedy.
    pub temperature: f32,
    /// 0: off.
    pub top_k: u32,
    /// 1 or more: off.
    pub top_p: f32,
    /// 0: off.
    pub min_p: f32,
    /// 1 or more: off.
    pub typical_p: f32,
    /// 0 or less: off.
    pub top_n_sigma: f32,
    /// 0: off; a threshold above 0.5 also turns XTC off.
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    /// 0: penalties off.
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    /// 0: DRY off.
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Arc<[String]>,
    /// 0 off, 1 or 2: Mirostat, which replaces the `samplers` list.
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub min_keep: usize,
    pub samplers: Arc<[SamplerKind]>,
    pub logit_bias: Arc<[(i32, f32)]>,
    pub ignore_eos: bool,
    /// GBNF the output must follow; the gateway has turned a JSON schema into one.
    pub grammar: Option<Arc<str>>,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            top_n_sigma: -1.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_sequence_breakers: ["\n", ":", "\"", "*"].map(String::from).to_vec().into(),
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            min_keep: 0,
            samplers: SamplerKind::DEFAULT_ORDER.to_vec().into(),
            logit_bias: Arc::default(),
            ignore_eos: false,
            grammar: None,
        }
    }
}

impl SampleParams {
    /// These server-wide settings with the request's fields on top. Soft limits clamp as
    /// llama-server's do; the gateway has rejected the hard-limit violations. The penalty
    /// windows are capped at the context of a sequence, which the history cannot exceed: libllama
    /// allocates a ring buffer of that size, so a request cannot ask for gigabytes
    /// (llama-server does not cap them).
    pub fn resolve(&self, r: &SamplingParams, n_ctx_seq: usize) -> SampleParams {
        let e = &r.extra;
        let mut p = self.clone();
        if let Some(v) = r.temperature {
            p.temperature = v.max(0.0);
        }
        if let Some(v) = r.top_k {
            p.top_k = v;
        }
        if let Some(v) = r.top_p {
            p.top_p = v.clamp(0.0, 1.0);
        }
        macro_rules! set {
            ($($f:ident),*) => {$( if let Some(v) = e.$f { p.$f = v; } )*};
        }
        set!(
            min_p,
            typical_p,
            top_n_sigma,
            xtc_probability,
            xtc_threshold,
            dynatemp_range,
            dynatemp_exponent,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            ignore_eos
        );
        if let Some(v) = e.repeat_penalty {
            p.penalty_repeat = v;
        }
        if let Some(v) = e.repeat_last_n {
            p.penalty_last_n = v;
        }
        if let Some(v) = e.presence_penalty {
            p.penalty_present = v;
        }
        if let Some(v) = e.frequency_penalty {
            p.penalty_freq = v;
        }
        if let Some(v) = e.min_keep {
            p.min_keep = v as usize;
        }
        if let Some(v) = &e.dry_sequence_breakers {
            p.dry_sequence_breakers = v.clone().into();
        }
        if !e.logit_bias.is_empty() {
            p.logit_bias = e.logit_bias.clone().into();
        }
        if let Some(v) = &e.samplers {
            p.samplers = v.clone().into();
        }
        if let Some(GrammarSource::Gbnf(g)) = &e.grammar {
            p.grammar = Some(g.as_str().into());
        }
        let cap = i32::try_from(n_ctx_seq).unwrap_or(i32::MAX);
        p.penalty_last_n = p.penalty_last_n.min(cap);
        p.dry_penalty_last_n = p.dry_penalty_last_n.min(cap);
        p
    }

    /// llama-server picks the argmax when the list ends in temperature 0 (no dynamic range) or in
    /// `top_k` 1; Mirostat never does.
    fn is_greedy(&self) -> bool {
        self.mirostat == 0
            && match self.samplers.last() {
                Some(SamplerKind::Temperature) => {
                    self.temperature <= 0.0 && self.dynatemp_range == 0.0
                }
                Some(SamplerKind::TopK) => self.top_k == 1,
                _ => false,
            }
    }

    /// Whether a greedy chain may leave out its truncating stages (top_n_sigma, top_k, top_p,
    /// min_p, temperature): they keep the best token, so they only matter when a later stage
    /// can remove it (typical_p, XTC) or sees the truncated candidates (penalties and DRY
    /// placed after one of them in a custom order).
    fn greedy_can_elide(&self) -> bool {
        if self.typical_p < 1.0 || self.xtc_on() {
            return false;
        }
        let truncates = |k: &SamplerKind| match k {
            SamplerKind::TopNSigma => self.top_n_sigma > 0.0,
            SamplerKind::TopK => self.top_k > 0,
            SamplerKind::TopP => self.top_p < 1.0,
            SamplerKind::MinP => self.min_p > 0.0,
            _ => false,
        };
        let first = self.samplers.iter().position(truncates);
        first.is_none_or(|f| {
            !self.samplers[f..].iter().any(|k| match k {
                SamplerKind::Penalties => self.penalties_on(),
                SamplerKind::Dry => self.dry_on(),
                _ => false,
            })
        })
    }

    fn penalties_on(&self) -> bool {
        self.penalty_last_n != 0
            && (self.penalty_repeat != 1.0
                || self.penalty_freq != 0.0
                || self.penalty_present != 0.0)
    }

    fn dry_on(&self) -> bool {
        self.dry_multiplier != 0.0 && self.dry_base >= 1.0 && self.dry_penalty_last_n != 0
    }

    fn xtc_on(&self) -> bool {
        self.xtc_probability > 0.0 && self.xtc_threshold <= 0.5
    }

    /// Greedy and with no sampler that looks at history or removes the best token: the chosen
    /// token is the argmax of the logits, so speculation may verify drafts against it.
    pub fn is_plain_greedy(&self) -> bool {
        self.is_greedy()
            && !self.penalties_on()
            && !self.dry_on()
            && !self.xtc_on()
            && self.typical_p >= 1.0
            && self.logit_bias.is_empty()
            && !self.ignore_eos
            && self.grammar.is_none()
    }
}

/// What building a chain needs from the model.
#[derive(Clone, Copy, Default)]
pub struct SamplerEnv<'a> {
    pub n_vocab: usize,
    /// End-of-generation token ids, banned by `ignore_eos`.
    pub eog: &'a [u32],
    /// The model whose vocabulary DRY reads its breakers from.
    pub model: Option<&'a Model>,
}

/// One request's sampler: libllama's chain plus its history.
pub struct Sampler {
    /// `None`: plain greedy, which needs no chain and no history.
    chain: Option<Chain>,
}

impl Sampler {
    /// The chain of `common_sampler_init` for `p`. `seed` `None` is a random seed per request.
    /// Samplers that cannot change the result with these settings are left out, which keeps
    /// the cost of the default chain down without changing any token.
    pub fn new(p: &SampleParams, seed: Option<u32>, env: &SamplerEnv) -> Result<Self, String> {
        // Plain greedy is an argmax over the logits: no candidate array, no history. The model's
        // suppress tokens would still need the bias stage.
        if p.is_plain_greedy() && env.model.is_none_or(|m| m.suppress_tokens().is_empty()) {
            return Ok(Self { chain: None });
        }
        let seed = seed.unwrap_or(DEFAULT_SEED);
        let n_vocab = env.n_vocab as i32;
        let mut chain = Chain::chain();
        if let Some(g) = &p.grammar {
            let model = env.model.ok_or(
                "a grammar needs the model's vocabulary, which this engine has not loaded",
            )?;
            chain = chain
                .with_grammar(model, g)
                .map_err(|e| format!("invalid grammar: {e}"))?;
        }

        let mut bias: Vec<(i32, f32)> = p
            .logit_bias
            .iter()
            .copied()
            .filter(|&(t, _)| t >= 0 && t < n_vocab)
            .collect();
        if p.ignore_eos {
            bias.extend(env.eog.iter().map(|&t| (t as i32, f32::NEG_INFINITY)));
        }
        // The model's own suppress tokens come last, as in `common_sampler_init`.
        if let Some(m) = env.model {
            bias.extend(
                m.suppress_tokens()
                    .into_iter()
                    .map(|t| (t, f32::NEG_INFINITY)),
            );
        }
        if !bias.is_empty() {
            chain = chain.logit_bias(n_vocab, &bias);
        }

        if p.mirostat == 1 || p.mirostat == 2 {
            chain = chain.temperature(p.temperature);
            chain = if p.mirostat == 1 {
                chain.mirostat(n_vocab, seed, p.mirostat_tau, p.mirostat_eta, 100)
            } else {
                chain.mirostat_v2(seed, p.mirostat_tau, p.mirostat_eta)
            };
            return Ok(Self { chain: Some(chain) });
        }

        let greedy = p.is_greedy();
        let elide = greedy && p.greedy_can_elide();
        let k = p.min_keep;
        for kind in p.samplers.iter() {
            chain = match kind {
                SamplerKind::Penalties if p.penalties_on() => chain.penalties(
                    n_vocab,
                    p.penalty_last_n,
                    p.penalty_repeat,
                    p.penalty_freq,
                    p.penalty_present,
                ),
                SamplerKind::Dry if p.dry_on() => {
                    let model = env.model.ok_or(
                        "DRY needs the model's vocabulary, which this engine has not loaded",
                    )?;
                    chain
                        .dry(
                            model,
                            p.dry_multiplier,
                            p.dry_base,
                            p.dry_allowed_length,
                            p.dry_penalty_last_n,
                            &p.dry_sequence_breakers,
                        )
                        .map_err(|e| e.to_string())?
                }
                SamplerKind::TypicalP if p.typical_p < 1.0 => chain.typical(p.typical_p, k),
                SamplerKind::Xtc if p.xtc_on() => {
                    chain.xtc(p.xtc_probability, p.xtc_threshold, k, seed)
                }
                // With nothing before the end that removes the best token or reads the
                // candidates after truncation, the rest cannot change an argmax.
                _ if elide => chain,
                SamplerKind::TopNSigma if p.top_n_sigma > 0.0 => chain.top_n_sigma(p.top_n_sigma),
                SamplerKind::TopK if p.top_k > 0 => {
                    chain.top_k(p.top_k.min(i32::MAX as u32) as i32)
                }
                SamplerKind::TopP if p.top_p < 1.0 => chain.top_p(p.top_p, k),
                SamplerKind::MinP if p.min_p > 0.0 => chain.min_p(p.min_p, k),
                SamplerKind::Temperature => {
                    chain.temperature_ext(p.temperature, p.dynatemp_range, p.dynatemp_exponent)
                }
                _ => chain,
            };
        }
        chain = if greedy {
            chain.then_greedy()
        } else {
            chain.dist(seed)
        };
        Ok(Self { chain: Some(chain) })
    }

    /// Feeds prompt tokens into the history (penalties, DRY), as llama-server does at admission.
    pub fn accept_all(&mut self, tokens: &[i32]) {
        for &t in tokens {
            if let Some(c) = &mut self.chain {
                c.accept(t);
            }
        }
    }

    /// The next token, or `None` when the mask leaves no token with a finite logit. The token is
    /// accepted into the history.
    pub fn sample(&mut self, logits: &[f32], mask: Option<&TokenMask>) -> Option<u32> {
        let Some(chain) = &mut self.chain else {
            let best = greedy(logits, |i| mask.is_none_or(|m| m.allows(i as u32)))?;
            // As the chain: nothing to pick when no allowed logit is finite.
            return logits[best as usize].is_finite().then_some(best);
        };
        let t = match mask {
            Some(m) => chain.sample_masked(logits, |i| m.allows(i as u32)),
            None => chain.sample_logits(logits),
        }?;
        u32::try_from(t).ok()
    }
}

/// The probabilities of `token` under the softmax of the raw `logits`, with the `n_top` most
/// likely tokens (ties by lower id), as llama-server's `get_token_probabilities` reports them.
pub fn token_probs(logits: &[f32], token: u32, n_top: usize) -> TokenProbs {
    let max = logits
        .iter()
        .copied()
        .filter(|l| !l.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - max).exp();
    }
    let logprob = |id: usize| {
        let p = (logits[id] - max).exp() / sum;
        // JSON has no -inf: the lowest float stands for log(0).
        if p > 0.0 { p.ln() } else { f32::MIN }
    };
    let n = n_top.min(logits.len());
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let better = |a: &u32, b: &u32| {
        logits[*b as usize]
            .total_cmp(&logits[*a as usize])
            .then(a.cmp(b))
    };
    if n < ids.len() {
        ids.select_nth_unstable_by(n, better);
        ids.truncate(n);
    }
    ids.sort_unstable_by(better);
    TokenProbs {
        token: token as TokenId,
        logprob: logprob(token as usize),
        top: ids.into_iter().map(|i| (i, logprob(i as usize))).collect(),
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
    //! Pure CPU: no model, no hardware (rule 9). Each case compares the scheduler's `Sampler`
    //! against a chain built by hand from the same libllama constructors, over 200 draws.
    use super::*;

    fn mask(allowed: &[u32], n: usize) -> TokenMask {
        let mut words = vec![0u32; n.div_ceil(32)];
        for &a in allowed {
            words[a as usize / 32] |= 1 << (a % 32);
        }
        TokenMask { words }
    }

    const N: usize = 300;

    /// A deterministic, varied logits row for step `i`.
    fn row(i: usize) -> Vec<f32> {
        let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..N)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 2.0
            })
            .collect()
    }

    fn env() -> SamplerEnv<'static> {
        SamplerEnv {
            n_vocab: N,
            ..SamplerEnv::default()
        }
    }

    fn draws(mut next: impl FnMut(&[f32]) -> u32) -> Vec<u32> {
        (0..200).map(|i| next(&row(i % 17))).collect()
    }

    fn ours(p: &SampleParams, seed: u32) -> Vec<u32> {
        let mut s = Sampler::new(p, Some(seed), &env()).unwrap();
        draws(|l| s.sample(l, None).unwrap())
    }

    fn reference(c: Chain) -> Vec<u32> {
        let mut c = c;
        draws(|l| c.sample_logits(l).unwrap() as u32)
    }

    #[test]
    fn token_probs_are_a_softmax_with_the_best_first() {
        let l = [1.0f32, 3.0, 2.0, 3.0];
        let p = token_probs(&l, 2, 3);
        // ties go to the lower id
        let ids: Vec<u32> = p.top.iter().map(|t| t.0).collect();
        assert_eq!(ids, [1, 3, 2]);
        let total: f32 = (0..4).map(|i| token_probs(&l, i, 0).logprob.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        let z: f32 = l.iter().map(|x| x.exp()).sum();
        assert!((p.logprob - (2.0f32.exp() / z).ln()).abs() < 1e-6);
        assert!((p.top[0].1 - (3.0f32.exp() / z).ln()).abs() < 1e-6);
        // more asked than the vocabulary has
        assert_eq!(token_probs(&l, 0, 10).top.len(), 4);
    }

    #[test]
    fn a_banned_token_has_the_lowest_float_as_log_probability() {
        let l = [0.0f32, f32::NEG_INFINITY];
        assert_eq!(token_probs(&l, 1, 2).logprob, f32::MIN);
    }

    fn params(f: impl FnOnce(&mut SampleParams)) -> SampleParams {
        let mut p = SampleParams::default();
        f(&mut p);
        p
    }

    const SEED: u32 = 1234;

    #[test]
    fn default_chain_is_llama_servers() {
        let want = reference(
            Chain::chain()
                .top_k(40)
                .top_p(0.95, 0)
                .min_p(0.05, 0)
                .temperature_ext(0.8, 0.0, 1.0)
                .dist(SEED),
        );
        assert_eq!(ours(&SampleParams::default(), SEED), want);
        assert_ne!(ours(&SampleParams::default(), SEED + 1), want);
    }

    #[test]
    fn disabled_stages_left_out_change_nothing() {
        // The literal list of llama-server with every stage present, disabled ones included.
        let want = reference(
            Chain::chain()
                .penalties(N as i32, 64, 1.0, 0.0, 0.0)
                .top_n_sigma(-1.0)
                .top_k(40)
                .typical(1.0, 0)
                .top_p(0.95, 0)
                .min_p(0.05, 0)
                .xtc(0.0, 0.1, 0, SEED)
                .temperature_ext(0.8, 0.0, 1.0)
                .dist(SEED),
        );
        assert_eq!(ours(&SampleParams::default(), SEED), want);
    }

    #[test]
    fn greedy_keeps_stages_that_can_remove_the_best_token() {
        // Typical sampling and XTC may drop the argmax, so the stages before them must run.
        let typical = params(|p| {
            p.temperature = 0.0;
            p.typical_p = 0.3;
        });
        let want = reference(
            Chain::chain()
                .top_k(40)
                .typical(0.3, 0)
                .top_p(0.95, 0)
                .min_p(0.05, 0)
                .temperature_ext(0.0, 0.0, 1.0)
                .then_greedy(),
        );
        assert_eq!(ours(&typical, SEED), want);
        let xtc = params(|p| {
            p.temperature = 0.0;
            p.xtc_probability = 1.0;
            p.xtc_threshold = 0.05;
        });
        let want = reference(
            Chain::chain()
                .top_k(40)
                .top_p(0.95, 0)
                .min_p(0.05, 0)
                .xtc(1.0, 0.05, 0, SEED)
                .temperature_ext(0.0, 0.0, 1.0)
                .then_greedy(),
        );
        assert_eq!(ours(&xtc, SEED), want);
        // Penalties after top_k see the truncated candidates only.
        let late = params(|p| {
            p.temperature = 0.0;
            p.penalty_repeat = 1.5;
            p.penalty_last_n = 16;
            p.samplers = [
                SamplerKind::TopK,
                SamplerKind::Penalties,
                SamplerKind::Temperature,
            ]
            .to_vec()
            .into();
        });
        let want = reference(
            Chain::chain()
                .top_k(40)
                .penalties(N as i32, 16, 1.5, 0.0, 0.0)
                .temperature_ext(0.0, 0.0, 1.0)
                .then_greedy(),
        );
        assert_eq!(ours(&late, SEED), want);
    }

    #[test]
    fn min_p_typical_top_n_sigma_and_min_keep() {
        let p = params(|p| {
            p.min_p = 0.2;
            p.typical_p = 0.9;
            p.top_n_sigma = 1.5;
            p.min_keep = 2;
        });
        let want = reference(
            Chain::chain()
                .top_n_sigma(1.5)
                .top_k(40)
                .typical(0.9, 2)
                .top_p(0.95, 2)
                .min_p(0.2, 2)
                .temperature_ext(0.8, 0.0, 1.0)
                .dist(SEED),
        );
        assert_eq!(ours(&p, SEED), want);
    }

    #[test]
    fn xtc_draws_with_the_request_seed() {
        let p = params(|p| {
            p.xtc_probability = 0.7;
            p.xtc_threshold = 0.15;
        });
        let want = reference(
            Chain::chain()
                .top_k(40)
                .top_p(0.95, 0)
                .min_p(0.05, 0)
                .xtc(0.7, 0.15, 0, SEED)
                .temperature_ext(0.8, 0.0, 1.0)
                .dist(SEED),
        );
        assert_eq!(ours(&p, SEED), want);
    }

    fn penalty_chain() -> Chain {
        Chain::chain()
            .penalties(N as i32, 32, 1.4, 0.3, 0.2)
            .top_k(40)
            .top_p(0.95, 0)
            .min_p(0.05, 0)
            .temperature_ext(0.8, 0.0, 1.0)
            .dist(SEED)
    }

    #[test]
    fn penalties_see_the_accepted_tokens() {
        let p = params(|p| {
            p.penalty_repeat = 1.4;
            p.penalty_freq = 0.3;
            p.penalty_present = 0.2;
            p.penalty_last_n = 32;
        });
        assert_eq!(ours(&p, SEED), reference(penalty_chain()));
        // The prompt counts as history, as in llama-server.
        let mut with = Sampler::new(&p, Some(SEED), &env()).unwrap();
        with.accept_all(&[1, 2, 3]);
        let mut by_hand = penalty_chain();
        for t in [1, 2, 3] {
            by_hand.accept(t);
        }
        assert_eq!(
            draws(|l| with.sample(l, None).unwrap()),
            draws(|l| by_hand.sample_logits(l).unwrap() as u32)
        );
    }

    #[test]
    fn dynatemp_and_samplers_order() {
        let p = params(|p| {
            p.dynatemp_range = 0.5;
            p.dynatemp_exponent = 1.5;
            p.samplers = [
                SamplerKind::Temperature,
                SamplerKind::MinP,
                SamplerKind::TopK,
            ]
            .to_vec()
            .into();
        });
        let want = reference(
            Chain::chain()
                .temperature_ext(0.8, 0.5, 1.5)
                .min_p(0.05, 0)
                .top_k(40)
                .dist(SEED),
        );
        assert_eq!(ours(&p, SEED), want);
        let default_order = params(|p| {
            p.dynatemp_range = 0.5;
            p.dynatemp_exponent = 1.5;
        });
        assert_ne!(ours(&default_order, SEED), want, "the order must matter");
    }

    #[test]
    fn mirostat_replaces_the_list() {
        for (m, tau, eta) in [(1, 4.0, 0.2), (2, 6.0, 0.05)] {
            let p = params(|p| {
                p.mirostat = m;
                p.mirostat_tau = tau;
                p.mirostat_eta = eta;
            });
            let c = Chain::chain().temperature(0.8);
            let c = if m == 1 {
                c.mirostat(N as i32, SEED, tau, eta, 100)
            } else {
                c.mirostat_v2(SEED, tau, eta)
            };
            assert_eq!(ours(&p, SEED), reference(c), "mirostat {m}");
        }
    }

    #[test]
    fn logit_bias_and_ignore_eos_ban_tokens() {
        let eog = [7u32];
        let p = params(|p| {
            p.temperature = 0.0;
            p.logit_bias = vec![(3, f32::NEG_INFINITY), (9999, 5.0)].into();
            p.ignore_eos = true;
        });
        let e = SamplerEnv {
            n_vocab: N,
            eog: &eog,
            model: None,
        };
        let mut s = Sampler::new(&p, None, &e).unwrap();
        let mut l = vec![0.0f32; N];
        l[3] = 10.0;
        l[7] = 9.0;
        l[11] = 8.0;
        // 3 and 7 are banned; id 9999 is outside the vocabulary and dropped.
        assert_eq!(s.sample(&l, None), Some(11));
    }

    #[test]
    fn greedy_is_the_argmax_after_penalties() {
        let g = params(|p| p.temperature = 0.0);
        let mut s = Sampler::new(&g, None, &env()).unwrap();
        for i in 0..20 {
            let l = row(i);
            assert_eq!(s.sample(&l, None), greedy(&l, |_| true));
        }
        let pen = params(|p| {
            p.temperature = 0.0;
            p.penalty_repeat = 100.0;
        });
        assert!(g.is_plain_greedy() && !pen.is_plain_greedy());
        let mut s = Sampler::new(&pen, None, &env()).unwrap();
        let l = [5.0, 4.0, 1.0];
        assert_eq!(s.sample(&l, None), Some(0));
        // Token 0 is now in the history, so the heavy penalty moves the pick.
        assert_eq!(s.sample(&l, None), Some(1));
    }

    #[test]
    fn greedy_fast_path_agrees_with_the_chain() {
        let g = params(|p| p.temperature = 0.0);
        assert!(g.is_plain_greedy());
        let mut fast = Sampler::new(&g, None, &env()).unwrap();
        assert!(fast.chain.is_none());
        let mut chain = Chain::chain().then_greedy();
        for i in 0..200 {
            // Few distinct values, so ties are common.
            let l: Vec<f32> = row(i).iter().map(|v| (v * 0.7).round()).collect();
            assert_eq!(
                fast.sample(&l, None),
                chain.sample_logits(&l).map(|t| t as u32)
            );
            let allowed: Vec<u32> = (0..N as u32)
                .filter(|t| (t + i as u32).is_multiple_of(3))
                .collect();
            let m = mask(&allowed, N);
            assert_eq!(
                fast.sample(&l, Some(&m)),
                chain
                    .sample_masked(&l, |t| m.allows(t as u32))
                    .map(|t| t as u32)
            );
        }
        // Nothing allowed: no token from either.
        let none = mask(&[], N);
        let l = row(1);
        assert_eq!(fast.sample(&l, Some(&none)), None);
        assert_eq!(chain.sample_masked(&l, |t| none.allows(t as u32)), None);
    }

    #[test]
    fn greedy_takes_the_lowest_id_on_ties() {
        assert_eq!(greedy(&[1.0, 3.0, 3.0, 2.0], |_| true), Some(1));
        let g = params(|p| p.temperature = 0.0);
        let mut s = Sampler::new(&g, None, &env()).unwrap();
        assert_eq!(s.sample(&[1.0, 3.0, 3.0, 2.0], None), Some(1));
    }

    #[test]
    fn mask_restricts_greedy_and_random() {
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0];
        let m = mask(&[3, 4], 5);
        let e = SamplerEnv {
            n_vocab: 5,
            ..SamplerEnv::default()
        };
        let g = params(|p| p.temperature = 0.0);
        let mut s = Sampler::new(&g, Some(1), &e).unwrap();
        assert_eq!(s.sample(&logits, Some(&m)), Some(3));
        let r = params(|p| p.temperature = 1.0);
        let mut s = Sampler::new(&r, Some(7), &e).unwrap();
        for _ in 0..50 {
            assert!(matches!(s.sample(&logits, Some(&m)), Some(3 | 4)));
        }
        assert_eq!(s.sample(&logits, Some(&mask(&[], 5))), None);
    }

    #[test]
    fn top_k_one_and_tiny_top_p_are_greedy() {
        let logits = [0.5, 2.0, 1.0, 1.5];
        let e = SamplerEnv {
            n_vocab: 4,
            ..SamplerEnv::default()
        };
        for p in [
            params(|p| {
                p.temperature = 2.0;
                p.top_k = 1;
            }),
            params(|p| {
                p.temperature = 2.0;
                p.top_k = 0;
                p.top_p = 0.01;
                p.min_p = 0.0;
            }),
        ] {
            let mut s = Sampler::new(&p, Some(3), &e).unwrap();
            for _ in 0..20 {
                assert_eq!(s.sample(&logits, None), Some(1));
            }
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let p = SampleParams::default();
        assert_eq!(ours(&p, 5), ours(&p, 5));
        assert_ne!(ours(&p, 5), ours(&p, 6));
    }

    #[test]
    fn dry_without_a_model_is_an_error() {
        let p = params(|p| p.dry_multiplier = 0.8);
        assert!(Sampler::new(&p, None, &env()).is_err());
    }

    #[test]
    fn request_fields_override_the_defaults() {
        use tetraeder_gateway::engine::SamplingExtras;
        let d = SampleParams::default();
        let r = SamplingParams {
            temperature: Some(0.0),
            extra: SamplingExtras {
                min_p: Some(0.3),
                repeat_penalty: Some(1.2),
                ..SamplingExtras::default()
            },
            ..SamplingParams::default()
        };
        let p = d.resolve(&r, 4096);
        assert_eq!((p.temperature, p.min_p, p.penalty_repeat), (0.0, 0.3, 1.2));
        assert_eq!(p.top_k, 40);
        assert!(
            !p.is_plain_greedy(),
            "a penalty makes greedy history-dependent"
        );
        let g = SamplingParams {
            temperature: Some(0.0),
            ..SamplingParams::default()
        };
        assert!(d.resolve(&g, 4096).is_plain_greedy());
    }

    #[test]
    fn penalty_windows_are_capped_at_the_context() {
        use tetraeder_gateway::engine::SamplingExtras;
        let d = SampleParams::default();
        let r = SamplingParams {
            extra: SamplingExtras {
                repeat_last_n: Some(i32::MAX),
                dry_penalty_last_n: Some(i32::MAX),
                ..SamplingExtras::default()
            },
            ..SamplingParams::default()
        };
        let p = d.resolve(&r, 4096);
        assert_eq!((p.penalty_last_n, p.dry_penalty_last_n), (4096, 4096));
        // smaller windows are left as they are
        let p = d.resolve(&SamplingParams::default(), 4096);
        assert_eq!((p.penalty_last_n, p.dry_penalty_last_n), (64, 64));
    }

    /// `/props` reports the same defaults the sampler uses (it once said -1 for the DRY window).
    #[test]
    fn the_props_defaults_are_the_samplers() {
        use serde_json::json;
        use tetraeder_gateway::gateway::GatewayConfig;
        let reported = tetraeder_gateway::settings::defaults(&GatewayConfig::default());
        let d = SampleParams::default();
        let f = |x: f32| json!(x.to_string().parse::<f64>().unwrap());
        for (name, want) in [
            ("temperature", f(d.temperature)),
            ("top_k", json!(d.top_k)),
            ("top_p", f(d.top_p)),
            ("min_p", f(d.min_p)),
            ("top_n_sigma", f(d.top_n_sigma)),
            ("xtc_probability", f(d.xtc_probability)),
            ("xtc_threshold", f(d.xtc_threshold)),
            ("typical_p", f(d.typical_p)),
            ("dynatemp_range", f(d.dynatemp_range)),
            ("dynatemp_exponent", f(d.dynatemp_exponent)),
            ("repeat_last_n", json!(d.penalty_last_n)),
            ("repeat_penalty", f(d.penalty_repeat)),
            ("presence_penalty", f(d.penalty_present)),
            ("frequency_penalty", f(d.penalty_freq)),
            ("dry_multiplier", f(d.dry_multiplier)),
            ("dry_base", f(d.dry_base)),
            ("dry_allowed_length", json!(d.dry_allowed_length)),
            ("dry_penalty_last_n", json!(d.dry_penalty_last_n)),
            ("dry_sequence_breakers", json!(&*d.dry_sequence_breakers)),
            ("mirostat", json!(d.mirostat)),
            ("mirostat_tau", f(d.mirostat_tau)),
            ("mirostat_eta", f(d.mirostat_eta)),
            ("min_keep", json!(d.min_keep)),
            ("ignore_eos", json!(d.ignore_eos)),
        ] {
            assert_eq!(reported[name], want, "{name}");
        }
        let samplers: Vec<_> = reported["samplers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| SamplerKind::from_name(n.as_str().unwrap()))
            .collect();
        let want: Vec<_> = d.samplers.iter().map(|k| Some(*k)).collect();
        assert_eq!(samplers, want);
    }
}

//! Sampling over a logits row (ADR 0046, 0083 decision 7): the host sampler is libllama's own
//! sampler chain, built in llama-server's order with its defaults, so seeded and sampled outputs
//! follow `llama-server` at the pin. The token mask (tool calls, grammars) is applied first by
//! setting masked logits to `-inf`.

use std::sync::Arc;
use tetraeder_gateway::engine::{SamplerKind, SamplingParams, TokenMask};
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
        }
    }
}

impl SampleParams {
    /// These server-wide settings with the request's fields on top. Soft limits clamp as
    /// llama-server's do; the gateway has rejected the hard-limit violations.
    pub fn resolve(&self, r: &SamplingParams) -> SampleParams {
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
    chain: Chain,
}

impl Sampler {
    /// The chain of `common_sampler_init` for `p`. `seed` `None` is a random seed per request.
    /// Samplers that cannot change the result with these settings are left out, which keeps
    /// the cost of the default chain down without changing any token.
    pub fn new(p: &SampleParams, seed: Option<u32>, env: &SamplerEnv) -> Result<Self, String> {
        let seed = seed.unwrap_or(DEFAULT_SEED);
        let n_vocab = env.n_vocab as i32;
        let mut chain = Chain::chain();

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
            return Ok(Self { chain });
        }

        let greedy = p.is_greedy();
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
                // The rest keep the best token, so they cannot change an argmax.
                _ if greedy => chain,
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
        Ok(Self { chain })
    }

    /// Feeds prompt tokens into the history (penalties, DRY), as llama-server does at admission.
    pub fn accept_all(&mut self, tokens: &[i32]) {
        for &t in tokens {
            self.chain.accept(t);
        }
    }

    /// The next token, or `None` when the mask leaves no token with a finite logit. The token is
    /// accepted into the history.
    pub fn sample(&mut self, logits: &[f32], mask: Option<&TokenMask>) -> Option<u32> {
        let t = match mask {
            Some(m) => self.chain.sample_masked(logits, |i| m.allows(i as u32)),
            None => self.chain.sample_logits(logits),
        }?;
        u32::try_from(t).ok()
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
        let p = d.resolve(&r);
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
        assert!(d.resolve(&g).is_plain_greedy());
    }
}

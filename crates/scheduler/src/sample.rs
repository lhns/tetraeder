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
            && (self.penalty_repeat != 1.0 || self.penalty_freq != 0.0 || self.penalty_present != 0.0)
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
                    let model = env
                        .model
                        .ok_or("DRY needs the model's vocabulary, which this engine has not loaded")?;
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
                SamplerKind::TopK if p.top_k > 0 => chain.top_k(p.top_k.min(i32::MAX as u32) as i32),
                SamplerKind::TopP if p.top_p < 1.0 => chain.top_p(p.top_p, k),
                SamplerKind::MinP if p.min_p > 0.0 => chain.min_p(p.min_p, k),
                SamplerKind::Temperature => {
                    chain.temperature_ext(p.temperature, p.dynatemp_range, p.dynatemp_exponent)
                }
                _ => chain,
            };
        }
        chain = if greedy { chain.then_greedy() } else { chain.dist(seed) };
        Ok(Self { chain })
    }

    /// Feeds prompt tokens into the history (penalties, DRY), as llama-server does at admission.
    pub fn accept_all(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.chain.accept(t as i32);
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

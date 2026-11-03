//! Speculative decoding (ADR 0043): a drafter proposes tokens, the target verifies them in one
//! multi-token step and keeps the prefix it agrees with.

mod draft;
mod ngram;
mod policy;
mod speculator;
mod target;
#[cfg(test)]
mod tests;

pub use draft::DraftModel;
pub use ngram::NgramDrafter;
pub use policy::choose_k;
pub use speculator::{BatchStat, SpecStats, Speculator};

use tetraeder_ggml_sys::llama::LlamaError;
use tetraeder_model::ModelError;

/// Largest verify batch. K-quant matmuls take the single-row `vec_dot` kernel up to 7 rows and
/// the tiled kernel from 8, which rounds differently (ADR 0043), so k is at most 6.
pub const MAX_VERIFY: usize = 7;

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Llama(#[from] LlamaError),
    #[error("{0}")]
    Invalid(String),
}

/// The model whose output is kept.
pub trait Target {
    fn n_vocab(&self) -> usize;
    /// Tokens in the KV cache.
    fn n_past(&self) -> usize;
    /// Runs `tokens` after the cached ones; logits of every position, position-major.
    fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>, SpecError>;
    /// Drops the cached tokens above `n_past`.
    fn rewind(&mut self, n_past: usize) -> Result<(), SpecError>;
}

pub trait Drafter {
    /// Up to `max` tokens continuing `history` (prompt and output so far; its last token is not
    /// yet in the target's KV). May return fewer or none.
    fn draft(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, SpecError>;
    /// Name for the metric attribute: `ngram`, `draft`.
    fn method(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftLen {
    Fixed(usize),
    /// Up to `max` tokens, chosen per step from the measured acceptance and step times.
    Adaptive {
        max: usize,
    },
}

/// Temperature 0 is greedy. Above 0 the token at absolute position p is sampled with an RNG
/// seeded from `(seed, p)`, so speculative and plain decoding agree token for token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub seed: u64,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling {
        temperature: 0.0,
        seed: 0,
    };
}

/// Index of the largest logit; the first one wins a tie. NaN never wins.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] || logits[best].is_nan() {
            best = i;
        }
    }
    best
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The token chosen from `logits` for absolute position `pos` (its index in the history).
pub fn choose(logits: &[f32], pos: usize, s: &Sampling) -> i32 {
    if s.temperature <= 0.0 {
        return argmax(logits) as i32;
    }
    let t = f64::from(s.temperature);
    let max = f64::from(logits[argmax(logits)]);
    let w: Vec<f64> = logits
        .iter()
        .map(|&l| ((f64::from(l) - max) / t).exp())
        .map(|w| if w.is_finite() { w } else { 0.0 })
        .collect();
    let sum: f64 = w.iter().sum();
    let u = (splitmix64(splitmix64(s.seed) ^ pos as u64) >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0.0;
    let mut last = 0;
    for (i, &w) in w.iter().enumerate() {
        if w > 0.0 {
            last = i;
        }
        acc += w;
        if u * sum < acc {
            return i as i32;
        }
    }
    last as i32
}

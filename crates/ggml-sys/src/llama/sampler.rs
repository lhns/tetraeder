//! Sampler chains.

use super::context::Context;
use super::model::Model;
use super::{LlamaError, Result, Token, init};
use crate::raw as sys;
use std::ffi::CString;

/// A sampler chain. Build with [`Sampler::chain`] and the adders, or use [`Sampler::greedy`].
pub struct Sampler {
    ptr: *mut sys::llama_sampler,
    /// Candidate buffer of `sample_logits`, kept between calls.
    cand: Vec<sys::llama_token_data>,
}

// SAFETY: a sampler chain owns its state and has no thread affinity; it is not Sync.
unsafe impl Send for Sampler {}

impl Sampler {
    /// An empty chain.
    pub fn chain() -> Self {
        init();
        // SAFETY: default params are plain values; the chain is owned by the returned value.
        let ptr = unsafe {
            let mut p = sys::llama_sampler_chain_default_params();
            p.no_perf = true;
            sys::llama_sampler_chain_init(p)
        };
        Self {
            ptr,
            cand: Vec::new(),
        }
    }

    fn add(self, s: *mut sys::llama_sampler) -> Self {
        // SAFETY: the chain takes ownership of `s`.
        unsafe { sys::llama_sampler_chain_add(self.ptr, s) };
        self
    }

    pub fn greedy() -> Self {
        // SAFETY: constructor returning an owned sampler.
        Self::chain().add(unsafe { sys::llama_sampler_init_greedy() })
    }

    pub fn top_k(self, k: i32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_top_k(k) })
    }

    pub fn top_p(self, p: f32, min_keep: usize) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_top_p(p, min_keep) })
    }

    pub fn temperature(self, t: f32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_temp(t) })
    }

    /// Final stage that picks the highest logit (the first one on ties).
    pub fn then_greedy(self) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_greedy() })
    }

    pub fn min_p(self, p: f32, min_keep: usize) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_min_p(p, min_keep) })
    }

    pub fn typical(self, p: f32, min_keep: usize) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_typical(p, min_keep) })
    }

    pub fn top_n_sigma(self, n: f32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_top_n_sigma(n) })
    }

    /// Exclude-top-choices sampling; draws from its own RNG seeded with `seed`.
    pub fn xtc(self, probability: f32, threshold: f32, min_keep: usize, seed: u32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_xtc(probability, threshold, min_keep, seed) })
    }

    /// Temperature with optional dynamic range (`range` 0: plain temperature).
    pub fn temperature_ext(self, t: f32, range: f32, exponent: f32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_temp_ext(t, range, exponent) })
    }

    /// Repeat, frequency and presence penalties over the last `last_n` accepted tokens.
    pub fn penalties(
        self,
        n_vocab: i32,
        last_n: i32,
        repeat: f32,
        freq: f32,
        present: f32,
    ) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe {
            sys::llama_sampler_init_penalties(n_vocab, last_n, repeat, freq, present)
        })
    }

    /// DRY repetition penalty. Needs the vocabulary of `model` to find the sequence breakers.
    pub fn dry(
        self,
        model: &Model,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        breakers: &[String],
    ) -> Result<Self> {
        let c: Vec<CString> = breakers
            .iter()
            .map(|b| {
                CString::new(b.as_str())
                    .map_err(|_| LlamaError::Invalid("DRY breaker contains a NUL byte".into()))
            })
            .collect::<Result<_>>()?;
        let ptrs: Vec<*const std::ffi::c_char> = c.iter().map(|b| b.as_ptr()).collect();
        // SAFETY: the vocab belongs to the live model; the breaker strings are NUL-terminated and
        // libllama copies them during init, before `c` is dropped.
        let s = unsafe {
            sys::llama_sampler_init_dry(
                model.vocab,
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                ptrs.as_ptr().cast_mut(),
                ptrs.len(),
            )
        };
        if s.is_null() {
            return Err(LlamaError::Invalid("cannot create the DRY sampler".into()));
        }
        Ok(self.add(s))
    }

    /// Adds `bias` to the logit of each listed token (`-inf` bans it). Ids outside the
    /// vocabulary are the caller's to drop.
    pub fn logit_bias(self, n_vocab: i32, bias: &[(Token, f32)]) -> Self {
        let b: Vec<sys::llama_logit_bias> = bias
            .iter()
            .map(|&(token, bias)| sys::llama_logit_bias { token, bias })
            .collect();
        // SAFETY: `b` holds `b.len()` entries; libllama copies them during init.
        self.add(unsafe { sys::llama_sampler_init_logit_bias(n_vocab, b.len() as i32, b.as_ptr()) })
    }

    /// Mirostat 1. Selects the token itself, so it ends the chain.
    pub fn mirostat(self, n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) })
    }

    /// Mirostat 2. Selects the token itself, so it ends the chain.
    pub fn mirostat_v2(self, seed: u32, tau: f32, eta: f32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_mirostat_v2(seed, tau, eta) })
    }

    /// Final stage of a random chain: draws from the distribution with a seeded RNG.
    pub fn dist(self, seed: u32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_dist(seed) })
    }

    /// Samples from output `idx` of the last decode (`-1` = last) and accepts the token into
    /// the chain's state.
    pub fn sample(&mut self, ctx: &mut Context, idx: i32) -> Result<Token> {
        ctx.check_output(idx)?;
        // SAFETY: both handles are live; `idx` was validated against the last batch's outputs.
        Ok(unsafe { sys::llama_sampler_sample(self.ptr, ctx.ptr, idx) })
    }

    /// Runs the chain on a logits row without a context and returns the chosen token, which it
    /// also accepts into the chain's state.
    pub fn sample_logits(&mut self, logits: &[f32]) -> Option<Token> {
        self.sample_masked(logits, |_| true)
    }

    /// As `sample_logits`, with the ids `allow` rejects set to `-inf` first. `None` when no
    /// allowed id has a finite logit (the chain would draw from nothing).
    pub fn sample_masked(
        &mut self,
        logits: &[f32],
        allow: impl Fn(usize) -> bool,
    ) -> Option<Token> {
        self.cand.clear();
        self.cand.reserve(logits.len());
        let mut any = false;
        for (i, &l) in logits.iter().enumerate() {
            let l = if allow(i) { l } else { f32::NEG_INFINITY };
            any |= l.is_finite();
            self.cand.push(sys::llama_token_data {
                id: i as Token,
                logit: l,
                p: 0.0,
            });
        }
        if !any {
            return None;
        }
        let mut arr = sys::llama_token_data_array {
            data: self.cand.as_mut_ptr(),
            size: self.cand.len(),
            selected: -1,
            sorted: false,
        };
        // SAFETY: `arr` points at `cand`, which outlives the call; samplers only reorder or
        // shrink the array in place (they may repoint `data` inside `cand`'s buffer).
        unsafe { sys::llama_sampler_apply(self.ptr, &mut arr) };
        let sel = usize::try_from(arr.selected)
            .ok()
            .filter(|&s| s < arr.size)?;
        // SAFETY: `sel < arr.size` and `arr.data` is valid for `arr.size` entries.
        let tok = unsafe { (*arr.data.add(sel)).id };
        self.accept(tok);
        Some(tok)
    }

    /// Feeds a token into the chain's state (for example repetition history). `sample`
    /// already does this for the token it returns.
    pub fn accept(&mut self, token: Token) {
        // SAFETY: live sampler.
        unsafe { sys::llama_sampler_accept(self.ptr, token) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: owned chain; its members are freed with it.
        unsafe { sys::llama_sampler_free(self.ptr) }
    }
}

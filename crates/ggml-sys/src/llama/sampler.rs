//! Sampler chains.

use super::context::Context;
use super::{Result, Token, init};
use crate::raw as sys;

/// A sampler chain. Build with [`Sampler::chain`] and the adders, or use [`Sampler::greedy`].
pub struct Sampler {
    ptr: *mut sys::llama_sampler,
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
        Self { ptr }
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

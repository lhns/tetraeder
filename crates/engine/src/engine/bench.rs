//! llama-bench adapter: loops that mirror `test_prompt` and `test_gen`.

use super::Engine;
use crate::Result;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::llama::Token;
use tetraeder_telemetry::Phase;

impl Engine {
    fn next_random_token(&mut self) -> Token {
        // xorshift64*: a deterministic stand-in for llama-bench's std::rand().
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33;
        (r % self.model.n_vocab().max(1) as u64) as Token
    }

    /// Mirrors llama-bench `test_prompt`: `n_prompt` random tokens (BOS first if the model adds
    /// one) in `n_batch` chunks, no sampler. Returns the wall time of the whole run.
    pub fn bench_prompt(&mut self, n_prompt: usize) -> Result<Duration> {
        let t0 = Instant::now();
        self.ctx.set_threads(self.threads, self.threads);
        let mut pos = self.ctx.seq_pos_max(0) + 1;
        let mut done = 0;
        while done < n_prompt {
            let n = (n_prompt - done).min(self.chunk);
            self.batch.clear();
            for i in 0..n {
                let tok = if done == 0 && i == 0 && self.model.add_bos() {
                    self.model.bos()
                } else {
                    self.next_random_token()
                };
                self.batch.push(tok, pos + i as i32, 0, i == n - 1)?;
            }
            self.step(Phase::Prefill)?;
            done += n;
            pos += n as i32;
        }
        Ok(t0.elapsed())
    }

    /// Mirrors llama-bench `test_gen`: `n_gen` single-token decodes starting after the last
    /// stored position, each synchronized, logits requested, no sampler.
    pub fn bench_gen(&mut self, n_gen: usize) -> Result<Duration> {
        let t0 = Instant::now();
        self.ctx.set_threads(self.threads, self.threads);
        let mut token = if self.model.add_bos() {
            self.model.bos()
        } else {
            self.next_random_token()
        };
        let first_pos = self.ctx.seq_pos_max(0) + 1;
        for i in 0..n_gen {
            self.batch.clear();
            self.batch.push(token, first_pos + i as i32, 0, true)?;
            self.step(Phase::Decode)?;
            token = self.next_random_token();
        }
        Ok(t0.elapsed())
    }

    /// llama-bench warmup: one prompt run, then one generated token, if each is requested.
    pub fn bench_warmup(&mut self, n_prompt: usize, n_gen: usize) -> Result<()> {
        if n_prompt > 0 {
            self.bench_prompt(n_prompt)?;
        }
        if n_gen > 0 {
            self.bench_gen(1)?;
        }
        Ok(())
    }

    /// Empties the KV cache between repetitions.
    pub fn bench_reset(&mut self) {
        self.ctx.memory_clear();
    }
}

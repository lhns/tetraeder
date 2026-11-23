//! Embeddings and reranking over libllama (ADR 0090): a second context on the model the engine
//! already holds, built with `embeddings = true` and a pooling type. It never decodes text
//! (that stays with the generation context), so the two do not share a KV cache.

use crate::{EngineError, Result};
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Model, Pooling, ThreadPool, Token,
};

/// Settings of the embedding context.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// `Unspecified` takes the model's own pooling type (`none` when it has none); `Rank` is a
    /// reranker.
    pub pooling: Pooling,
    /// Most tokens one input may have (`n_batch` and `n_ubatch` of the context: an embedding
    /// input is evaluated in one batch).
    pub n_batch: u32,
    /// `None`: all available hardware threads.
    pub threads: Option<i32>,
    pub flash_attn: FlashAttn,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            pooling: Pooling::Unspecified,
            n_batch: 2048,
            threads: None,
            flash_attn: FlashAttn::Auto,
        }
    }
}

/// What one input embeds to.
#[derive(Debug, Clone, PartialEq)]
pub enum Embedding {
    /// One vector for the whole input (mean, cls or last pooling).
    Pooled(Vec<f32>),
    /// One vector per token (pooling `none`).
    Tokens(Vec<Vec<f32>>),
}

pub struct Embedder {
    ctx: Context,
    batch: Batch,
    pooling: Pooling,
    n_embd: usize,
    n_cls: usize,
    n_batch: usize,
    threads: i32,
}

impl Embedder {
    /// Creates the embedding context over `model`. A reranker (`Rank`) scores with the model's
    /// classification head; a model without one gives the first value of its pooled hidden
    /// state, as in llama-server.
    pub fn new(model: &Arc<Model>, cfg: &EmbedConfig) -> Result<Self> {
        if cfg.n_batch == 0 {
            return Err(EngineError::InvalidRequest(
                "n_batch must be positive".into(),
            ));
        }
        let threads = cfg
            .threads
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(4, |n| i32::try_from(n.get()).unwrap_or(4))
            })
            .max(1);
        let pooled = !matches!(cfg.pooling, Pooling::None);
        let params = ContextParams {
            n_ctx: cfg.n_batch,
            n_batch: cfg.n_batch,
            n_ubatch: cfg.n_batch,
            n_threads: threads,
            n_threads_batch: threads,
            flash_attn: cfg.flash_attn,
            n_seq_max: 1,
            kv_unified: true,
            embeddings: true,
            pooling: cfg.pooling,
            // a pooled sequence has one output, so the context need not hold one per token
            n_outputs_max_per_seq: u32::from(pooled),
            ..ContextParams::default()
        };
        let mut ctx = Context::new(model, &params)?;
        ctx.attach_threadpool(Arc::new(ThreadPool::new(threads)?));
        let pooling = ctx.pooling();
        if cfg.pooling == Pooling::Rank && pooling != Pooling::Rank {
            return Err(EngineError::InvalidRequest(
                "libllama did not take the rank pooling".into(),
            ));
        }
        let n_batch = ctx.n_batch() as usize;
        Ok(Self {
            ctx,
            batch: Batch::new(n_batch),
            pooling,
            n_embd: model.n_embd_out(),
            n_cls: model.n_cls_out().max(1),
            n_batch,
            threads,
        })
    }

    /// The pooling the context runs with (the model's own when none was asked for).
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Most tokens of one input.
    pub fn n_batch(&self) -> usize {
        self.n_batch
    }

    /// Evaluates `tokens` as one sequence from an empty cache with an output on every token.
    fn run(&mut self, tokens: &[Token]) -> Result<()> {
        if tokens.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if tokens.len() > self.n_batch {
            return Err(EngineError::ContextTooSmall {
                needed: tokens.len(),
                n_ctx: self.n_batch,
            });
        }
        self.ctx.memory_clear();
        self.ctx.set_threads(self.threads, self.threads);
        self.batch.clear();
        for (i, &t) in tokens.iter().enumerate() {
            self.batch.push(t, i as i32, 0, true)?;
        }
        self.ctx.decode(&self.batch)?;
        Ok(())
    }

    /// The embedding of `tokens`, not normalized.
    pub fn embed(&mut self, tokens: &[Token]) -> Result<Embedding> {
        self.run(tokens)?;
        if self.pooling == Pooling::None {
            let mut out = Vec::with_capacity(tokens.len());
            for i in 0..tokens.len() {
                out.push(self.ctx.embeddings_ith(i as i32)?.to_vec());
            }
            return Ok(Embedding::Tokens(out));
        }
        let n = if self.pooling == Pooling::Rank {
            self.n_cls
        } else {
            self.n_embd
        };
        Ok(Embedding::Pooled(self.ctx.embeddings_seq(0, n)?.to_vec()))
    }

    /// The relevance score of a query and document pair already joined into `tokens`: the first
    /// output of the classification head. Needs `Rank` pooling.
    pub fn score(&mut self, tokens: &[Token]) -> Result<f32> {
        if self.pooling != Pooling::Rank {
            return Err(EngineError::InvalidRequest(
                "the context does not run the rank pooling".into(),
            ));
        }
        self.run(tokens)?;
        let n = self.n_cls;
        match self.ctx.embeddings_seq(0, n) {
            Ok(v) => Ok(v[0]),
            // as llama-server: the last output when the sequence has none
            Err(_) => Ok(self.ctx.embeddings_ith(-1)?[0]),
        }
    }
}

/// `common_embd_normalize` of llama.cpp: `-1` none, `0` max absolute scaled to the int16 range,
/// `1` taxicab, `2` euclidean (llama-server's default), above 2 the p-norm.
pub fn normalize(v: &[f32], mode: i32) -> Vec<f32> {
    let sum: f64 = match mode {
        -1 => 1.0,
        0 => v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x.abs()))) / 32760.0,
        2 => v
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt(),
        p => v
            .iter()
            .map(|&x| f64::from(x.abs()).powi(p))
            .sum::<f64>()
            .powf(1.0 / f64::from(p)),
    };
    let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    v.iter().map(|&x| x * norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn normalize_follows_common_embd_normalize() {
        let v = [3.0f32, -4.0];
        assert_eq!(normalize(&v, -1), v);
        let e = normalize(&v, 2);
        assert!((e[0] - 0.6).abs() < 1e-6 && (e[1] + 0.8).abs() < 1e-6);
        let t = normalize(&v, 1);
        assert!((t[0] - 3.0 / 7.0).abs() < 1e-6 && (t[1] + 4.0 / 7.0).abs() < 1e-6);
        let m = normalize(&v, 0);
        assert!(
            (m[1] + 32760.0).abs() < 0.1,
            "the largest maps to the int16 range: {m:?}"
        );
        let p3 = normalize(&[2.0, 0.0], 3);
        assert!((p3[0] - 1.0).abs() < 1e-6);
        // the zero vector stays zero
        assert_eq!(normalize(&[0.0, 0.0], 2), [0.0, 0.0]);
    }
}

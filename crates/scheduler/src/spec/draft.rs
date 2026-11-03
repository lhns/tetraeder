use super::{Drafter, SpecError, argmax};
use std::path::Path;
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{Batch, Context, ContextParams, Model, ModelParams, ThreadPool};

/// A small model of the target's tokenizer run through libllama (ADR 0043). It keeps its own
/// KV of the history, rolls it back to the common prefix with each new history and drafts
/// greedily while its top probability stays at or above `p_min`.
pub struct DraftModel {
    ctx: Context,
    batch: Batch,
    n_batch: usize,
    n_vocab: usize,
    p_min: f32,
    /// Tokens in the KV cache of sequence 0, in order.
    cached: Vec<i32>,
}

impl DraftModel {
    pub fn load(
        path: impl AsRef<Path>,
        n_ctx: u32,
        n_gpu_layers: i32,
        threads: i32,
        p_min: f32,
    ) -> Result<Self, SpecError> {
        let threads = threads.max(1);
        let model = Arc::new(Model::load(
            path,
            &ModelParams {
                n_gpu_layers,
                ..ModelParams::default()
            },
        )?);
        let n_batch = n_ctx.clamp(1, 512);
        let params = ContextParams {
            n_ctx,
            n_batch,
            n_ubatch: n_batch,
            n_threads: threads,
            n_threads_batch: threads,
            ..ContextParams::default()
        };
        let mut ctx = Context::new(&model, &params)?;
        ctx.attach_threadpool(Arc::new(ThreadPool::new(threads)?));
        let n_batch = (ctx.n_batch() as usize).max(1);
        Ok(Self {
            n_vocab: usize::try_from(model.n_vocab()).unwrap_or(0),
            batch: Batch::new(n_batch),
            n_batch,
            ctx,
            p_min,
            cached: Vec::new(),
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The draft and the target must share the vocabulary.
    pub fn check_vocab(&self, target_n_vocab: usize) -> Result<(), SpecError> {
        if self.n_vocab == target_n_vocab {
            Ok(())
        } else {
            Err(SpecError::Invalid(format!(
                "draft vocab {} differs from the target's {target_n_vocab}",
                self.n_vocab
            )))
        }
    }

    /// Decodes `tokens` at the positions after `cached`, logits for the last one only.
    fn decode(&mut self, tokens: &[i32]) -> Result<(), SpecError> {
        for chunk in tokens.chunks(self.n_batch) {
            self.batch.clear();
            let base = self.cached.len();
            for (i, &t) in chunk.iter().enumerate() {
                let is_last = i + 1 == chunk.len();
                self.batch.push(t, (base + i) as i32, 0, is_last)?;
            }
            self.ctx.decode(&self.batch)?;
            self.cached.extend_from_slice(chunk);
        }
        Ok(())
    }

    fn fill(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        let n = history.len();
        let mut c = self
            .cached
            .iter()
            .zip(history)
            .take_while(|(a, b)| a == b)
            .count()
            .min(n - 1);
        if !self.ctx.seq_rm(0, c as i32, -1) {
            self.ctx.memory_clear();
            c = 0;
        }
        self.cached.truncate(c);
        self.decode(&history[c..])?;
        let mut out = Vec::new();
        while out.len() < max {
            let logits = self.ctx.logits_ith(-1)?;
            let top = argmax(logits);
            // Softmax probability of the top token: 1 / sum(exp(l - top)).
            let z: f32 = logits.iter().map(|&l| (l - logits[top]).exp()).sum();
            if 1.0 / z < self.p_min {
                break;
            }
            out.push(top as i32);
            if out.len() < max {
                self.decode(&[top as i32])?;
            }
        }
        Ok(out)
    }
}

impl Drafter for DraftModel {
    fn draft(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        // Room for the history and the drafts we feed back (the last draft is not decoded).
        let room = (self.ctx.n_ctx() as usize).saturating_sub(history.len());
        let max = max.min(room);
        if history.is_empty() || max == 0 {
            return Ok(Vec::new());
        }
        let r = self.fill(history, max);
        if r.is_err() {
            // The KV state is unknown after a failed decode; start over next time.
            self.ctx.memory_clear();
            self.cached.clear();
        }
        r
    }

    fn method(&self) -> &'static str {
        "draft"
    }
}

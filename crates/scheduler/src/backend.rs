//! The engines as seen by the scheduler: tokens in, last-position logits out (ADR 0046).

use tetraeder_engine::Engine;
use tetraeder_model::Qwen3Moe;

/// Positions are token counts of the single cached sequence.
pub trait Backend {
    /// Context length in tokens.
    fn n_ctx(&self) -> usize;
    /// Positions currently cached.
    fn n_past(&self) -> usize;
    /// Keeps the first `keep` cached positions.
    fn truncate(&mut self, keep: usize) -> Result<(), String>;
    /// Appends `tokens` and returns the logits of the last one (`n_vocab` values, valid until
    /// the next call). Prefill is chunked inside.
    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String>;
    /// The cached positions as host bytes, for the prefix cache (ADR 0049).
    fn snapshot(&mut self) -> Result<Vec<u8>, String>;
    /// Replaces the cache with a [`Backend::snapshot`] of `n` positions.
    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String>;
    /// Whether steps run on a GPU, so wall time counts as device time.
    fn on_gpu(&self) -> bool;
    /// Weight bytes read from storage since load; `None` when the backend does not stream.
    fn bytes_streamed_total(&self) -> Option<u64>;
}

/// Tokens per `eval_last` call of the MoE builder; the scheduler also chunks to stay
/// cancellable, this keeps the backend safe on its own.
pub const MOE_CHUNK: usize = 512;

/// The libllama engine, loaded on CPU or GPU.
pub struct LlamaBackend {
    engine: Engine,
    on_gpu: bool,
}

impl LlamaBackend {
    pub fn new(engine: Engine, on_gpu: bool) -> Self {
        Self { engine, on_gpu }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

impl Backend for LlamaBackend {
    fn n_ctx(&self) -> usize {
        self.engine.n_ctx()
    }

    fn n_past(&self) -> usize {
        self.engine.n_past()
    }

    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        self.engine.truncate(keep).map_err(|e| e.to_string())
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        self.engine.eval(tokens).map_err(|e| e.to_string())
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.engine
            .snapshot()
            .map(|(bytes, _)| bytes)
            .map_err(|e| e.to_string())
    }

    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.engine.restore(bytes, n).map_err(|e| e.to_string())
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
}

/// The Qwen3 MoE builder.
pub struct MoeBackend {
    model: Qwen3Moe,
    on_gpu: bool,
    logits: Vec<f32>,
}

impl MoeBackend {
    pub fn new(model: Qwen3Moe, on_gpu: bool) -> Self {
        Self {
            model,
            on_gpu,
            logits: Vec::new(),
        }
    }

    pub fn model(&self) -> &Qwen3Moe {
        &self.model
    }
}

impl Backend for MoeBackend {
    fn n_ctx(&self) -> usize {
        self.model.n_ctx()
    }

    fn n_past(&self) -> usize {
        self.model.n_past()
    }

    fn truncate(&mut self, keep: usize) -> Result<(), String> {
        let keep = keep.min(self.model.n_past());
        self.model.rewind(keep).map_err(|e| e.to_string())
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
        if tokens.is_empty() {
            return Err("no tokens".into());
        }
        for chunk in tokens.chunks(MOE_CHUNK) {
            self.logits = self.model.eval_last(chunk).map_err(|e| e.to_string())?;
        }
        Ok(&self.logits)
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.model.kv_snapshot().map_err(|e| e.to_string())
    }

    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.model.kv_restore(bytes, n).map_err(|e| e.to_string())
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.model.expert_stats().map(|s| s.io.bytes_read)
    }
}

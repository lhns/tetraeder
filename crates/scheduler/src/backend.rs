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
    /// The KV cache format the snapshots are in, keying the prefix cache (ADR 0050 b).
    fn kv_format(&self) -> String {
        "f16/f16".into()
    }
    /// Whether steps run on a GPU, so wall time counts as device time.
    fn on_gpu(&self) -> bool;
    /// Weight bytes read from storage since load; `None` when the backend does not stream.
    fn bytes_streamed_total(&self) -> Option<u64>;
    /// Cumulative counters for the decode profile (`TETRAEDER_DECODE_PROFILE=1`), such as graph
    /// builds against reuses; the scheduler reports their change over a request's decode.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
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

    /// libllama's perf counters: `n_reused` of `n_eval` decodes reused the previous graph.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        let p = self.engine.perf();
        vec![
            ("n_eval", f64::from(p.n_eval)),
            ("n_reused", f64::from(p.n_reused)),
            ("eval_s", p.eval_ms / 1e3),
        ]
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
            // Into the kept buffer: no 600 KB allocation per decode token.
            self.model
                .eval_last_into(chunk, &mut self.logits)
                .map_err(|e| e.to_string())?;
        }
        Ok(&self.logits)
    }

    fn snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.model.kv_snapshot().map_err(|e| e.to_string())
    }

    fn restore(&mut self, bytes: &[u8], n: usize) -> Result<(), String> {
        self.model.kv_restore(bytes, n).map_err(|e| e.to_string())
    }

    fn kv_format(&self) -> String {
        self.model.kv_format().to_string()
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.model.expert_stats().map(|s| s.io.bytes_read)
    }

    /// The builder's step profile (host timers) and whole-step graph builds.
    fn counters(&self) -> Vec<(&'static str, f64)> {
        let p = self.model.profile();
        vec![
            ("steps", p.steps as f64),
            ("graphs", p.graphs as f64),
            ("step_graphs_built", self.model.step_graphs_built() as f64),
            ("inputs_s", p.inputs.as_secs_f64()),
            ("build_s", p.build.as_secs_f64()),
            ("alloc_s", p.alloc.as_secs_f64()),
            ("compute_s", p.compute.as_secs_f64()),
            ("readback_s", p.readback.as_secs_f64()),
            ("streamer_s", p.streamer.as_secs_f64()),
            ("total_s", p.total.as_secs_f64()),
        ]
    }
}

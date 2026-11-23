//! Phase 1 engine: the generation loop and the llama-bench adapter over libllama (ADR 0026).
//!
//! One [`Engine`] owns a model, a context, a threadpool and a batch. Every `llama_decode` goes
//! through one private step function that times decode plus synchronize and feeds telemetry.

mod config;
mod embed;
mod engine;
mod stream;

pub use config::{
    BufferOverride, ContextConfig, FFN_EXPS_REGEX, KV_CACHE_TYPES, Placement, Sampling,
    cpu_moe_overrides, parse_kv_cache_type,
};
pub use embed::{EmbedConfig, Embedder, Embedding, normalize};
pub use engine::{
    Engine, GenerateReport, GenerateRequest, Prompt, SeqEval, StopReason, TokenEvent,
};
pub use tetraeder_ggml_sys::ggml::Type as CacheType;
pub use tetraeder_ggml_sys::llama::{FlashAttn, LoadMode, Pooling, SplitMode, Token};

use tetraeder_ggml_sys::llama::LlamaError;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Llama(#[from] LlamaError),
    #[error("placement: {0}")]
    Placement(String),
    #[error("the prompt has no tokens")]
    EmptyPrompt,
    #[error("request needs {needed} context positions but the context has {n_ctx}")]
    ContextTooSmall { needed: usize, n_ctx: usize },
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

impl EngineError {
    /// Short stable label for the `error.type` metric attribute.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Llama(_) => "llama",
            Self::Placement(_) => "placement",
            Self::EmptyPrompt => "empty_prompt",
            Self::ContextTooSmall { .. } => "context_too_small",
            Self::InvalidRequest(_) => "invalid_request",
        }
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;

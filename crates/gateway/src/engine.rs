//! The contract between the gateway and an inference engine (ADR 0028).
//!
//! Tokens in, tokens out. The gateway renders templates, tokenizes through the engine,
//! detokenizes from [`Vocab`] bytes, parses tool calls and builds constraints; the engine
//! samples, applies the constraint mask and reports per-request stats.

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub type TokenId = u32;

/// One served model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    /// Maximum prompt plus generated tokens.
    pub context_length: u32,
    /// Jinja chat template from GGUF `tokenizer.chat_template`; `None` for base models.
    pub chat_template: Option<String>,
    /// Token strings passed to the template as `bos_token` and `eos_token`.
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    /// Unix seconds, reported by the models endpoints.
    pub created: u64,
}

/// The tokenizer's vocabulary as bytes per token id.
#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    /// Bytes of each token, indexed by id. Special tokens hold their text form (`<|im_end|>`).
    pub tokens: Vec<Vec<u8>>,
    /// Ids that are special (control) tokens; their text is never constrained as bytes.
    pub special: Vec<TokenId>,
    /// End-of-generation ids (EOS, EOT, `<|im_end|>`, ...). The first one is the primary EOS.
    pub eog: Vec<TokenId>,
}

impl Vocab {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn bytes(&self, id: TokenId) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn is_eog(&self, id: TokenId) -> bool {
        self.eog.contains(&id)
    }
}

/// Sampling parameters. `None` means the engine's default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingParams {
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
}

/// Allowed next tokens as a bitset over the vocabulary (bit `id % 32` of word `id / 32`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMask {
    pub words: Vec<u32>,
}

impl TokenMask {
    pub fn allows(&self, id: TokenId) -> bool {
        self.words
            .get(id as usize / 32)
            .is_some_and(|w| w & (1 << (id % 32)) != 0)
    }
}

/// A per-request constraint on sampling (structured output, forced tool calls).
///
/// The engine calls `mask` before each sampling step and `commit` with the sampled token.
/// An error ends generation with [`GenEvent::Error`].
pub trait TokenConstraint: Send {
    /// Allowed tokens for the next step; `None` means unconstrained.
    fn mask(&mut self) -> Result<Option<TokenMask>, String>;
    fn commit(&mut self, token: TokenId) -> Result<(), String>;
}

pub struct GenerateRequest {
    pub model: String,
    pub prompt: Vec<TokenId>,
    pub params: SamplingParams,
    pub constraint: Option<Box<dyn TokenConstraint>>,
    /// Cancelled by the gateway on client disconnect, stop sequence or finished tool call.
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An end-of-generation token was sampled (not sent as a `Token` event).
    EndOfGeneration,
    /// `max_tokens` reached.
    Length,
    Cancelled,
}

/// What one request cost the engine. Fields the engine cannot measure stay `None`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineStats {
    pub prompt_tokens: u32,
    /// Prompt tokens served from a prefix cache.
    pub cached_tokens: u32,
    pub generated_tokens: u32,
    /// Device time attributed to this request.
    pub gpu_seconds: Option<f64>,
    /// Weight bytes moved between memory tiers for this request.
    pub bytes_streamed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenEvent {
    Token(TokenId),
    Finished {
        reason: FinishReason,
        stats: EngineStats,
    },
    Error(String),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("engine overloaded")]
    Overloaded,
    #[error("{0}")]
    Other(String),
}

/// The engine as seen by the gateway. The Phase 1 adapter implements this on libllama.
pub trait Engine: Send + Sync + 'static {
    fn models(&self) -> Vec<ModelInfo>;
    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError>;
    /// Special tokens are parsed; no BOS is added (templates add it).
    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError>;
    /// Starts generation. The channel is bounded; the engine stops when `req.cancel` fires or
    /// the receiver is dropped, and always ends with `Finished` or `Error` unless the
    /// receiver is gone.
    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError>;
}

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
    pub extra: SamplingExtras,
}

/// One stage of llama.cpp's sampler chain, as the `samplers` request field names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    Penalties,
    Dry,
    TopNSigma,
    TopK,
    TypicalP,
    TopP,
    MinP,
    Xtc,
    Temperature,
}

impl SamplerKind {
    /// llama-server's default order.
    pub const DEFAULT_ORDER: [SamplerKind; 9] = [
        SamplerKind::Penalties,
        SamplerKind::Dry,
        SamplerKind::TopNSigma,
        SamplerKind::TopK,
        SamplerKind::TypicalP,
        SamplerKind::TopP,
        SamplerKind::MinP,
        SamplerKind::Xtc,
        SamplerKind::Temperature,
    ];

    /// A name as llama.cpp reads it: canonical, kebab-case, without separators, or an alias.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        Some(match n.as_str() {
            "dry" => Self::Dry,
            "top_k" | "top-k" | "topk" => Self::TopK,
            "top_p" | "top-p" | "topp" | "nucleus" => Self::TopP,
            "top_n_sigma" | "top-n-sigma" | "topnsigma" => Self::TopNSigma,
            "typ_p" | "typ-p" | "typp" | "typ" => Self::TypicalP,
            "min_p" | "min-p" | "minp" => Self::MinP,
            "temperature" | "temp" => Self::Temperature,
            "xtc" => Self::Xtc,
            "penalties" => Self::Penalties,
            _ => return None,
        })
    }

    /// The one-letter form of the `samplers` string.
    pub fn from_char(c: char) -> Option<Self> {
        Some(match c {
            'd' => Self::Dry,
            'k' => Self::TopK,
            'y' => Self::TypicalP,
            'p' => Self::TopP,
            's' => Self::TopNSigma,
            'm' => Self::MinP,
            't' => Self::Temperature,
            'x' => Self::Xtc,
            'e' => Self::Penalties,
            _ => return None,
        })
    }
}

/// Sampling fields beyond temperature, top-p, top-k and seed (llama-server's group A). `None`
/// means the engine's default; values are already range-checked by the gateway.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingExtras {
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    /// Below 1.0 the request means the server default, so the gateway sends `None`.
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<i32>,
    pub dry_penalty_last_n: Option<i32>,
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// (token id, bias); `-inf` bans the token. Ids outside the vocabulary are dropped by the
    /// engine, as llama-server does.
    pub logit_bias: Vec<(i32, f32)>,
    pub ignore_eos: Option<bool>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    pub min_keep: Option<u32>,
    pub samplers: Option<Vec<SamplerKind>>,
    /// Report the log-probabilities of the sampled token and this many top tokens (0 or `None`:
    /// off), from the raw logits before the token mask and the chain, as llama-server does.
    pub logprobs: Option<u32>,
    /// Constrains the output like `grammar` or `json_schema` of llama-server. The gateway turns a
    /// schema into GBNF before the engine sees the request.
    pub grammar: Option<GrammarSource>,
    /// Added to the seed (`n` above 1: choice `i` samples with seed + i, as llama-server does).
    pub seed_offset: u32,
}

/// What a request's grammar is written in.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarSource {
    /// GBNF, root rule `root`.
    Gbnf(String),
    /// A JSON schema as JSON text; converted with [`Engine::json_schema_to_grammar`].
    JsonSchema(String),
}

/// What the sampler saw for one sampled token (see [`SamplingExtras::logprobs`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenProbs {
    pub token: TokenId,
    /// Natural log of the token's probability under the softmax of the raw logits.
    pub logprob: f32,
    /// The most likely tokens with their log-probabilities, best first.
    pub top: Vec<(TokenId, f32)>,
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
    /// From the start of the request's prompt processing to its first sampled token.
    pub prompt_ms: f64,
    /// From the first sampled token to the last one (the end-of-generation token included).
    pub predicted_ms: f64,
    /// Device time attributed to this request: its token share of the steps it took part in, so it
    /// depends on who ran beside it. No gate may use it per request.
    pub gpu_seconds: Option<f64>,
    /// Weight bytes moved between memory tiers for this request.
    pub bytes_streamed: Option<u64>,
    /// Where the time per token went; only with `TETRAEDER_DECODE_PROFILE=1`.
    pub profile: Option<DecodeProfile>,
}

/// Host-side time split of one request's decode (A6000 gap diagnosis, docs/m1-open.md item 4).
/// Seconds summed over the decode steps (every step after the one that ends the prefill).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodeProfile {
    /// The step that evaluates the prompt suffix and samples the first token.
    pub prefill_s: f64,
    pub decode_steps: u32,
    /// Backend `eval` of one token, logits readback included. The wall time of the steps the
    /// request took part in: a step shared with other requests counts in full for each, so this
    /// is not the request's own cost.
    pub eval_s: f64,
    /// Constraint mask and sampling over the logits.
    pub sample_s: f64,
    /// Handing each emitted token to the gateway (waits when its buffer is full).
    pub emit_s: f64,
    /// Gateway work per token: detokenize, stop strings, tool-call parsing, queueing the
    /// segments for the HTTP response. Filled by the gateway, over all tokens.
    pub gateway_s: f64,
    /// Backend counters over the decode steps, e.g. graph builds against reuses.
    pub backend: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenEvent {
    Token(TokenId),
    /// The probabilities of a sampled token, sent before its `Token` (also for the
    /// end-of-generation token, which has no `Token`). Only with `SamplingExtras::logprobs`.
    Probs(TokenProbs),
    Finished {
        reason: FinishReason,
        stats: EngineStats,
    },
    Error(String),
    /// The prompt does not fit the context or the KV cache: the client's error (HTTP 400), which
    /// a retry of the same request cannot fix.
    ContextExceeded(String),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("engine overloaded")]
    Overloaded,
    /// The client's request is wrong (HTTP 400).
    #[error("{0}")]
    Invalid(String),
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
    /// GBNF for a JSON schema (JSON text), built as llama-server builds it.
    fn json_schema_to_grammar(&self, _model: &str, _schema: &str) -> Result<String, EngineError> {
        Err(EngineError::Invalid(
            "json_schema is not supported by this engine".into(),
        ))
    }
    /// Checks that a GBNF grammar parses, so a bad one is a 400 before generation starts.
    fn check_grammar(&self, _model: &str, _grammar: &str) -> Result<(), EngineError> {
        Err(EngineError::Invalid(
            "grammar is not supported by this engine".into(),
        ))
    }
}

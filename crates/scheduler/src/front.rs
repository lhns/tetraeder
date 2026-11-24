//! The tokenizer-only engine of a cluster head that leads no replica (ADR 0091, step 6): it
//! answers models, tokenization, templates' special tokens and grammar checks from the model's
//! vocabulary, and refuses to generate (the replicas do that). The same functions serve the
//! [`Scheduler`](crate::Scheduler), so a head answers exactly as a single node does.

use crate::load::model_id;
use crate::scheduler::build_vocab;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tetraeder_gateway::engine::{
    Engine, EngineError, FimTokens, GenEvent, GenerateRequest, ModelInfo, TokenId, Vocab,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tokio::sync::mpsc;

pub(crate) fn tokenize(
    m: &Model,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<TokenId>, EngineError> {
    let toks = m
        .tokenize(text, add_special, parse_special)
        .map_err(|e| EngineError::Other(e.to_string()))?;
    Ok(toks.into_iter().map(|t| t as TokenId).collect())
}

pub(crate) fn fim_tokens(m: &Model) -> FimTokens {
    let id = |t: Option<i32>| t.map(|t| t as TokenId);
    FimTokens {
        pre: id(m.fim_pre()),
        suf: id(m.fim_suf()),
        mid: id(m.fim_mid()),
        rep: id(m.fim_rep()),
        sep: id(m.fim_sep()),
        bos: (m.add_bos() && m.bos() >= 0).then(|| m.bos() as TokenId),
    }
}

pub(crate) fn check_grammar(m: &Model, grammar: &str) -> Result<(), EngineError> {
    tetraeder_ggml_sys::llama::Sampler::check_grammar(m, grammar)
        .map_err(|e| EngineError::Invalid(format!("invalid grammar: {e}")))
}

/// An [`Engine`] over a vocabulary only. `generate` fails: a head with this engine routes every
/// request to a replica.
pub struct FrontEngine {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    tokenizer: Arc<Model>,
}

impl FrontEngine {
    /// Reads the vocabulary of `model_path` (no weights). `alias` and `chat_template` are what
    /// `serve` would use for the model id and template; `context_length` is reported as the
    /// model's.
    pub fn open(
        model_path: &Path,
        alias: Option<String>,
        chat_template: Option<String>,
        context_length: u32,
    ) -> Result<Self, String> {
        let tokenizer = Arc::new(
            Model::load(
                model_path,
                &ModelParams {
                    vocab_only: true,
                    ..ModelParams::default()
                },
            )
            .map_err(|e| format!("{}: {e}", model_path.display()))?,
        );
        let vocab = Arc::new(build_vocab(&tokenizer));
        let piece = |t: i32| {
            (t >= 0 && (t as usize) < vocab.len())
                .then(|| String::from_utf8_lossy(vocab.bytes(t as u32)).into_owned())
        };
        let created = std::fs::metadata(model_path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|_| SystemTime::now())
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let info = ModelInfo {
            id: alias.unwrap_or_else(|| model_id(model_path)),
            context_length,
            chat_template,
            bos_token: piece(tokenizer.bos()),
            eos_token: piece(tokenizer.eos()),
            created,
            path: Some(model_path.display().to_string()),
        };
        Ok(Self {
            info,
            vocab,
            tokenizer,
        })
    }

    fn check(&self, model: &str) -> Result<(), EngineError> {
        if model == self.info.id {
            Ok(())
        } else {
            Err(EngineError::ModelNotFound(model.into()))
        }
    }
}

impl Engine for FrontEngine {
    fn models(&self) -> Vec<ModelInfo> {
        vec![self.info.clone()]
    }

    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        self.check(model)?;
        Ok(Arc::clone(&self.vocab))
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError> {
        self.check(model)?;
        tokenize(&self.tokenizer, text, false, true)
    }

    fn tokenize_with(
        &self,
        model: &str,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>, EngineError> {
        self.check(model)?;
        tokenize(&self.tokenizer, text, add_special, parse_special)
    }

    fn fim_tokens(&self, _model: &str) -> FimTokens {
        fim_tokens(&self.tokenizer)
    }

    fn json_schema_to_grammar(&self, model: &str, schema: &str) -> Result<String, EngineError> {
        self.check(model)?;
        tetraeder_ggml_sys::llama::json_schema_to_grammar(schema).map_err(EngineError::Invalid)
    }

    fn check_grammar(&self, model: &str, grammar: &str) -> Result<(), EngineError> {
        self.check(model)?;
        check_grammar(&self.tokenizer, grammar)
    }

    fn generate(&self, _req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        Err(EngineError::Other(
            "this head leads no replica; requests go to the replicas".into(),
        ))
    }
}

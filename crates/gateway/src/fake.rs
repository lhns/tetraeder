//! A scripted engine for tests: byte-level vocabulary plus a few special tokens.
//!
//! It "generates" the text a script returns for the decoded prompt, one token per step, on
//! its own thread like the Phase 1 adapter will. It honours constraint masks: a scripted
//! token the mask forbids is skipped; when the script runs out it picks an end-of-generation
//! token if allowed, else the lowest allowed id.

use crate::engine::{
    Engine, EngineError, EngineStats, FinishReason, GenEvent, GenerateRequest, ModelInfo, TokenId,
    Vocab,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// Special tokens after the 256 byte tokens, in id order.
pub const SPECIALS: [&str; 6] = [
    "<|im_end|>",
    "<|im_start|>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
];
pub const IM_END: TokenId = 256;

/// ChatML template, the shape Qwen and Hermes models use, with tools in the system turn.
pub const CHATML_TEMPLATE: &str = r##"{%- if tools %}{{ '<|im_start|>system\n' }}{% if messages[0].role == 'system' %}{{ messages[0].content + '\n\n' }}{% endif %}{{ '# Tools\n\n<tools>' }}{% for tool in tools %}{{ '\n' + (tool | tojson) }}{% endfor %}{{ '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' }}
{%- elif messages[0].role == 'system' %}{{ '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'system' %}
{%- elif message.role == 'assistant' %}{{ '<|im_start|>assistant\n' + (message.content or '') }}{% for tc in message.tool_calls or [] %}{{ '\n<tool_call>\n{"name": "' + tc.function.name + '", "arguments": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}{% endfor %}{{ '<|im_end|>\n' }}
{%- elif message.role == 'tool' %}{{ '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"##;

type Script = dyn Fn(&str) -> String + Send + Sync;

pub struct FakeEngine {
    models: Vec<ModelInfo>,
    vocab: Arc<Vocab>,
    script: Arc<Script>,
    /// Pause per generated token (cancellation and backpressure tests).
    pub token_delay: Duration,
    /// Channel capacity per request.
    pub channel_capacity: usize,
    /// Make `generate` fail with this error.
    pub reject: Option<EngineError>,
    /// Send `GenEvent::Error` after this many tokens.
    pub fail_after: Option<usize>,
    /// Stats the engine reports besides token counts.
    pub gpu_seconds: Option<f64>,
    pub bytes_streamed: Option<u64>,
    /// Generations that ended by cancellation (or a dropped receiver).
    pub cancelled: Arc<AtomicUsize>,
    /// Generations started.
    pub started: Arc<AtomicUsize>,
    /// Prompts seen, decoded, in order.
    pub prompts: Arc<Mutex<Vec<String>>>,
}

impl FakeEngine {
    /// One model `fake` with the ChatML template, 4096 tokens of context, and a script.
    pub fn new(script: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self::with_template(Some(CHATML_TEMPLATE.to_owned()), script)
    }

    pub fn with_template(
        template: Option<String>,
        script: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        tokens.extend(SPECIALS.iter().map(|s| s.as_bytes().to_vec()));
        // Like Qwen: only the turn markers are special; `<tool_call>` and `<think>` are
        // ordinary added tokens whose bytes are their text.
        let special = vec![256, 257];
        Self {
            models: vec![ModelInfo {
                id: "fake".into(),
                context_length: 4096,
                chat_template: template,
                bos_token: None,
                eos_token: Some("<|im_end|>".into()),
                created: 1_700_000_000,
            }],
            vocab: Arc::new(Vocab {
                tokens,
                special,
                eog: vec![IM_END],
            }),
            script: Arc::new(script),
            token_delay: Duration::ZERO,
            channel_capacity: 4,
            reject: None,
            fail_after: None,
            gpu_seconds: Some(0.5),
            bytes_streamed: Some(1024),
            cancelled: Arc::default(),
            started: Arc::default(),
            prompts: Arc::default(),
        }
    }

    /// Always answers `text`.
    pub fn fixed(text: &str) -> Self {
        let t = text.to_owned();
        Self::new(move |_| t.clone())
    }

    pub fn set_context_length(&mut self, n: u32) {
        self.models[0].context_length = n;
    }

    fn check(&self, model: &str) -> Result<(), EngineError> {
        if self.models.iter().any(|m| m.id == model) {
            Ok(())
        } else {
            Err(EngineError::ModelNotFound(model.into()))
        }
    }

    fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (i, s) in SPECIALS.iter().enumerate() {
                if let Some(r) = rest.strip_prefix(s) {
                    out.push(256 + i as TokenId);
                    rest = r;
                    continue 'outer;
                }
            }
            let c = rest.chars().next().unwrap_or_default();
            let mut buf = [0u8; 4];
            out.extend(c.encode_utf8(&mut buf).bytes().map(TokenId::from));
            rest = &rest[c.len_utf8()..];
        }
        out
    }

    fn decode(&self, ids: &[TokenId]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.vocab.bytes(i).to_vec())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Engine for FakeEngine {
    fn models(&self) -> Vec<ModelInfo> {
        self.models.clone()
    }

    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        self.check(model)?;
        Ok(Arc::clone(&self.vocab))
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError> {
        self.check(model)?;
        Ok(self.encode(text))
    }

    fn generate(&self, mut req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        self.check(&req.model)?;
        if let Some(e) = &self.reject {
            return Err(e.clone());
        }
        let prompt = self.decode(&req.prompt);
        self.prompts.lock().unwrap().push(prompt.clone());
        self.started.fetch_add(1, Ordering::SeqCst);
        let script = self.encode(&(self.script)(&prompt));
        let (tx, rx) = mpsc::channel(self.channel_capacity.max(1));
        let delay = self.token_delay;
        let fail_after = self.fail_after;
        let cancelled = Arc::clone(&self.cancelled);
        let vocab_len = self.vocab.len() as TokenId;
        let mut stats = EngineStats {
            prompt_tokens: req.prompt.len() as u32,
            cached_tokens: 0,
            generated_tokens: 0,
            gpu_seconds: self.gpu_seconds,
            bytes_streamed: self.bytes_streamed,
        };
        std::thread::spawn(move || {
            let mut pos = 0;
            let reason = loop {
                if req.cancel.is_cancelled() || tx.is_closed() {
                    break FinishReason::Cancelled;
                }
                if stats.generated_tokens >= req.params.max_tokens {
                    break FinishReason::Length;
                }
                if fail_after.is_some_and(|n| stats.generated_tokens as usize >= n) {
                    let _ = tx.blocking_send(GenEvent::Error("injected failure".into()));
                    return;
                }
                let mask = match req.constraint.as_mut().map(|c| c.mask()) {
                    Some(Err(e)) => {
                        let _ = tx.blocking_send(GenEvent::Error(e));
                        return;
                    }
                    Some(Ok(m)) => m,
                    None => None,
                };
                let allowed = |t: TokenId| mask.as_ref().is_none_or(|m| m.allows(t));
                while pos < script.len() && !allowed(script[pos]) {
                    pos += 1;
                }
                let tok = if pos < script.len() {
                    pos += 1;
                    script[pos - 1]
                } else if allowed(IM_END) {
                    IM_END
                } else {
                    match (0..vocab_len).find(|&t| allowed(t)) {
                        Some(t) => t,
                        None => {
                            let _ = tx.blocking_send(GenEvent::Error("empty mask".into()));
                            return;
                        }
                    }
                };
                if let Some(c) = req.constraint.as_mut()
                    && let Err(e) = c.commit(tok)
                {
                    let _ = tx.blocking_send(GenEvent::Error(e));
                    return;
                }
                if tok == IM_END {
                    break FinishReason::EndOfGeneration;
                }
                stats.generated_tokens += 1;
                if tx.blocking_send(GenEvent::Token(tok)).is_err() {
                    break FinishReason::Cancelled;
                }
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
            };
            if reason == FinishReason::Cancelled {
                cancelled.fetch_add(1, Ordering::SeqCst);
            }
            let _ = tx.blocking_send(GenEvent::Finished { reason, stats });
        });
        Ok(rx)
    }
}

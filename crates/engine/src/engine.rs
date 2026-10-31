use crate::config::{ContextConfig, Placement, Sampling};
use crate::stream::{Utf8Stream, top_k};
use crate::{EngineError, Result};
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::llama::{
    self, Batch, Context, ContextParams, DeviceKind, Model, Sampler, ThreadPool, Token,
};
use tetraeder_telemetry::{CacheResult, Direction, Phase, StepSample, Telemetry};
use tracing::Span;

mod bench;

const OPERATION: &str = "text_completion";

pub enum Prompt {
    Text(String),
    Tokens(Vec<Token>),
}

pub struct GenerateRequest {
    pub prompt: Prompt,
    pub max_tokens: u32,
    pub sampling: Sampling,
    /// Let the tokenizer add BOS/EOS as the model asks (llama-simple does).
    pub add_special: bool,
    /// Turn special-token text such as `<|im_start|>` into token ids.
    pub parse_special: bool,
    /// Stop (without emitting it) at an end-of-generation token.
    pub stop_at_eog: bool,
    /// 0 = off; else the report gets the top-k (token, logit) per generated position.
    pub top_k_logits: usize,
}

impl GenerateRequest {
    /// Greedy request with llama-simple's tokenizer settings.
    pub fn new(prompt: Prompt, max_tokens: u32) -> Self {
        Self {
            prompt,
            max_tokens,
            sampling: Sampling::Greedy,
            add_special: true,
            parse_special: true,
            stop_at_eog: true,
            top_k_logits: 0,
        }
    }
}

/// One generated token as streamed to the callback. `piece` holds only complete UTF-8; it is
/// empty while a multi-byte character is still incomplete. After the last token one more
/// event with `flush` set carries any leftover bytes (lossily decoded).
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token: Token,
    pub index: usize,
    pub piece: String,
    pub flush: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    Eog,
    /// The callback returned `Break`.
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GenerateReport {
    pub prompt_tokens: Vec<Token>,
    /// Generated tokens (an end-of-generation token that stopped the run is not included).
    pub tokens: Vec<Token>,
    pub text: String,
    /// `top[i]` = top-k (token, logit) at the position that produced `tokens[i]`.
    pub top: Vec<Vec<(Token, f32)>>,
    /// Request start to the first sampled token.
    pub ttft: Duration,
    pub total: Duration,
    pub prefill_steps: usize,
    pub decode_steps: usize,
    pub stop_reason: StopReason,
}

pub struct Engine {
    model: Arc<Model>,
    ctx: Context,
    batch: Batch,
    name: String,
    telemetry: Option<Arc<Telemetry>>,
    graphs: bool,
    threads: i32,
    threads_batch: i32,
    chunk: usize,
    rng: u64,
}

fn default_threads() -> i32 {
    std::thread::available_parallelism().map_or(4, |n| i32::try_from(n.get()).unwrap_or(4))
}

impl Engine {
    /// Loads the model, creates the context, attaches a threadpool and records the weights
    /// size. The metrics model name is the file stem.
    pub fn load(
        model_path: impl AsRef<Path>,
        placement: &Placement,
        cfg: &ContextConfig,
        telemetry: Option<Arc<Telemetry>>,
    ) -> Result<Engine> {
        let path = model_path.as_ref();
        let mparams = placement.to_model_params()?;
        if cfg.n_batch == 0 || cfg.n_ubatch == 0 {
            return Err(EngineError::InvalidRequest(
                "n_batch and n_ubatch must be positive".into(),
            ));
        }
        let model = Arc::new(Model::load(path, &mparams)?);

        let threads = cfg.threads.unwrap_or_else(default_threads).max(1);
        let threads_batch = cfg.threads_batch.unwrap_or(threads).max(1);
        let cparams = ContextParams {
            n_ctx: cfg.n_ctx,
            n_batch: cfg.n_batch,
            n_ubatch: cfg.n_ubatch,
            n_threads: threads,
            n_threads_batch: threads_batch,
            flash_attn: cfg.flash_attn,
            ..ContextParams::default()
        };
        let mut ctx = Context::new(&model, &cparams)?;
        ctx.attach_threadpool(Arc::new(ThreadPool::new(threads)?));
        let chunk = (ctx.n_batch() as usize).max(1);
        let batch = Batch::new(chunk);

        let devices = llama::devices();
        let gpu_devices = devices
            .iter()
            .any(|d| matches!(d.kind, DeviceKind::Gpu | DeviceKind::IGpu));
        let cuda = devices.iter().any(|d| d.name.starts_with("CUDA"));
        let offloaded = placement.n_gpu_layers != 0 && gpu_devices;
        let graphs = placement.n_gpu_layers != 0
            && cuda
            && std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_none();

        let name = path
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        if let Some(t) = &telemetry {
            t.metrics().memory_used(
                "model",
                if offloaded { "vram" } else { "ram" },
                "weights",
                model.size_bytes(),
            );
        }
        Ok(Engine {
            model,
            ctx,
            batch,
            name,
            telemetry,
            graphs,
            threads,
            threads_batch,
            chunk,
            rng: 0x9E37_79B9_7F4A_7C15,
        })
    }

    /// Model name used for metric and span attributes.
    pub fn model_name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Whether steps are expected to run as CUDA graphs (layers offloaded to CUDA, not disabled by env).
    pub fn graphs(&self) -> bool {
        self.graphs
    }

    /// libllama's cumulative perf counters; `n_eval` and `n_reused` show how many decodes reused the graph.
    pub fn perf(&self) -> llama::PerfData {
        self.ctx.perf()
    }

    pub fn n_ctx(&self) -> usize {
        self.ctx.n_ctx() as usize
    }

    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>> {
        Ok(self.model.tokenize(text, add_special, parse_special)?)
    }

    /// Concatenated token pieces (special tokens rendered), lossy for broken UTF-8.
    pub fn detokenize(&self, tokens: &[Token]) -> Result<String> {
        let mut bytes = Vec::new();
        for &t in tokens {
            bytes.extend(self.model.token_to_piece(t, true)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Child span of the request span; none when telemetry is off.
    fn span(&self, phase: Phase) -> Span {
        if self.telemetry.is_none() {
            Span::none()
        } else if phase == Phase::Prefill {
            tracing::info_span!("prefill")
        } else {
            tracing::info_span!("decode")
        }
    }

    /// Decode the current batch and wait for it, timed on the host (ADR 0020).
    fn step(&mut self, phase: Phase) -> Result<Duration> {
        let tokens = u32::try_from(self.batch.len()).unwrap_or(u32::MAX);
        let t0 = Instant::now();
        self.ctx.decode(&self.batch)?;
        self.ctx.synchronize();
        let duration = t0.elapsed();
        if let Some(t) = &self.telemetry {
            t.record_step(StepSample {
                phase,
                stage: 0,
                graphs: self.graphs,
                duration,
                tokens,
            });
        }
        Ok(duration)
    }

    /// Generates a completion, streaming text pieces to `on_piece`. Clears the KV cache first.
    pub fn generate(
        &mut self,
        req: &GenerateRequest,
        mut on_piece: impl FnMut(&TokenEvent) -> ControlFlow<()>,
    ) -> Result<GenerateReport> {
        let start = Instant::now();
        let span = if self.telemetry.is_some() {
            tracing::info_span!(
                "text_completion",
                "gen_ai.operation.name" = OPERATION,
                "gen_ai.request.model" = %self.name,
                "gen_ai.usage.input_tokens" = tracing::field::Empty,
                "gen_ai.usage.output_tokens" = tracing::field::Empty,
            )
        } else {
            Span::none()
        };
        let result = {
            let _entered = span.enter();
            self.generate_inner(req, &mut on_piece, start)
        };
        match &result {
            Ok(r) => {
                span.record("gen_ai.usage.input_tokens", r.prompt_tokens.len() as u64);
                span.record("gen_ai.usage.output_tokens", r.tokens.len() as u64);
                if let Some(t) = &self.telemetry {
                    let m = t.metrics();
                    if !r.tokens.is_empty() {
                        m.time_to_first_token(r.ttft, &self.name, OPERATION);
                    }
                    if r.tokens.len() > 1 {
                        let rest = r.total.saturating_sub(r.ttft);
                        m.time_per_output_token(rest / (r.tokens.len() as u32 - 1), &self.name);
                    }
                    m.request_duration(r.total, &self.name, None);
                    m.tokens(
                        r.prompt_tokens.len() as u64,
                        &self.name,
                        Direction::Input,
                        CacheResult::Miss,
                    );
                    m.tokens(
                        r.tokens.len() as u64,
                        &self.name,
                        Direction::Output,
                        CacheResult::Miss,
                    );
                }
            }
            Err(e) => {
                if let Some(t) = &self.telemetry {
                    t.metrics()
                        .request_duration(start.elapsed(), &self.name, Some(e.kind()));
                }
            }
        }
        result
    }

    fn generate_inner(
        &mut self,
        req: &GenerateRequest,
        on_piece: &mut dyn FnMut(&TokenEvent) -> ControlFlow<()>,
        start: Instant,
    ) -> Result<GenerateReport> {
        let prompt_tokens = match &req.prompt {
            Prompt::Text(s) => self.tokenize(s, req.add_special, req.parse_special)?,
            Prompt::Tokens(t) => t.clone(),
        };
        if prompt_tokens.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        let max_tokens = req.max_tokens as usize;
        // The last sampled token is never decoded, like in llama-simple.
        let needed = prompt_tokens.len() + max_tokens.saturating_sub(1);
        if needed > self.n_ctx() {
            return Err(EngineError::ContextTooSmall {
                needed,
                n_ctx: self.n_ctx(),
            });
        }

        let mut sampler = match req.sampling {
            Sampling::Greedy => Sampler::greedy(),
            Sampling::Random {
                seed,
                temperature,
                top_k,
                top_p,
            } => {
                if temperature <= 0.0 {
                    Sampler::greedy()
                } else {
                    let mut s = Sampler::chain();
                    if top_k > 0 {
                        s = s.top_k(top_k);
                    }
                    if top_p < 1.0 {
                        s = s.top_p(top_p, 1);
                    }
                    s.temperature(temperature).dist(seed)
                }
            }
        };

        self.ctx.memory_clear();
        self.ctx.set_threads(self.threads, self.threads_batch);

        // Prefill in n_batch chunks; logits only for the last prompt token.
        let mut prefill_steps = 0;
        {
            let span = self.span(Phase::Prefill);
            let _e = span.enter();
            let n = prompt_tokens.len();
            let mut pos = 0;
            while pos < n {
                let end = (pos + self.chunk).min(n);
                self.batch.clear();
                for (i, &t) in prompt_tokens[pos..end].iter().enumerate() {
                    self.batch.push(t, (pos + i) as i32, 0, pos + i == n - 1)?;
                }
                self.step(Phase::Prefill)?;
                prefill_steps += 1;
                pos = end;
            }
        }

        let mut tokens = Vec::new();
        let mut top = Vec::new();
        let mut text = String::new();
        let mut utf8 = Utf8Stream::default();
        let mut ttft = Duration::ZERO;
        let mut decode_steps = 0;
        let mut stop_reason = StopReason::MaxTokens;
        let mut pos = prompt_tokens.len() as i32;
        {
            let span = self.span(Phase::Decode);
            let _e = span.enter();
            for index in 0..max_tokens {
                let candidates = if req.top_k_logits > 0 {
                    top_k(self.ctx.logits_ith(-1)?, req.top_k_logits)
                } else {
                    Vec::new()
                };
                let tok = sampler.sample(&mut self.ctx, -1)?;
                if index == 0 {
                    ttft = start.elapsed();
                }
                if req.stop_at_eog && self.model.is_eog(tok) {
                    stop_reason = StopReason::Eog;
                    break;
                }
                tokens.push(tok);
                if req.top_k_logits > 0 {
                    top.push(candidates);
                }
                let piece = utf8.push(&self.model.token_to_piece(tok, true)?);
                text.push_str(&piece);
                let event = TokenEvent {
                    token: tok,
                    index,
                    piece,
                    flush: false,
                };
                if on_piece(&event).is_break() {
                    stop_reason = StopReason::Cancelled;
                    break;
                }
                if index + 1 == max_tokens {
                    break;
                }
                self.batch.clear();
                self.batch.push(tok, pos, 0, true)?;
                pos += 1;
                self.step(Phase::Decode)?;
                decode_steps += 1;
            }
        }
        let tail = utf8.finish();
        if !tail.is_empty() {
            text.push_str(&tail);
            // Cancel is moot at the end of the stream.
            let _ = on_piece(&TokenEvent {
                token: tokens.last().copied().unwrap_or_default(),
                index: tokens.len().saturating_sub(1),
                piece: tail,
                flush: true,
            });
        }
        Ok(GenerateReport {
            prompt_tokens,
            tokens,
            text,
            top,
            ttft,
            total: start.elapsed(),
            prefill_steps,
            decode_steps,
            stop_reason,
        })
    }
}

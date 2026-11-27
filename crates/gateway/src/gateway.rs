//! The API-neutral core: prompt building, admission, generation, output parsing, metrics.

use crate::chat::{
    ChatMessage, Finish, LogprobEntry, OutputFormat, Segment, ToolCall, ToolChoice, ToolDef, Usage,
};
use crate::constraint::{ConstraintFactory, Reasoning};
use crate::decode::StopMatcher;
use crate::engine::{
    Embedded, EmbeddingCaps, Engine, EngineError, FimTokens, GenerateRequest, GrammarSource,
    ModelInfo, Reranked, SamplingExtras, SamplingParams, SlotAction, SlotWait, TokenConstraint,
    TokenId, Vocab,
};
use crate::error::GatewayError;
use crate::template::ChatTemplate;
use crate::toolcall::{OutputParser, ToolFormat};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_util::sync::{CancellationToken, DropGuard};
use tracing::Instrument;

mod cost;
mod counters;
mod pump;

pub use cost::{CostSink, RequestCost, attr};
use counters::Counters;
use pump::Run;

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Requests generating at once.
    pub max_concurrent: usize,
    /// Requests waiting for a slot; more are rejected as overloaded.
    pub max_queued: usize,
    /// `Retry-After` on overload rejections.
    pub retry_after_secs: u64,
    /// Output events buffered per request before generation waits for the client.
    pub stream_buffer: usize,
    /// Model that unknown ids resolve to (Claude Code sends its own `claude-*` ids).
    pub default_model: Option<String>,
    /// Keys that open every route but `/health`; empty: no authentication (`--api-key`).
    pub api_keys: Vec<String>,
    /// Serve `/metrics` (`--metrics`); without it the route answers 501, as llama-server.
    pub metrics: bool,
    /// The server's sampling defaults under llama-server's names, laid over llama.cpp's own
    /// defaults for `/props` and `generation_settings`.
    pub defaults: serde_json::Map<String, serde_json::Value>,
    /// `--context-shift`: a generation may run past the context (the engine shifts it), so
    /// `max_tokens` is not clamped to the room left.
    pub context_shift: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1,
            max_queued: 16,
            retry_after_secs: 1,
            stream_buffer: 32,
            default_model: None,
            api_keys: Vec::new(),
            metrics: false,
            defaults: serde_json::Map::new(),
            context_shift: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Chat,
    TextCompletion,
}

impl Operation {
    /// `gen_ai.operation.name` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::Chat => "chat",
            Operation::TextCompletion => "text_completion",
        }
    }
}

/// What a continued message continues (llama-server's `common_chat_continuation`, resolved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Continuation {
    /// The reply continues the message's content (its reasoning, if any, is closed).
    Content,
    /// The reply continues the message's reasoning, inside the think block.
    Reasoning,
}

#[derive(Clone)]
pub enum Prompt {
    Chat {
        messages: Vec<ChatMessage>,
        /// Text of a final assistant message the reply continues (Anthropic prefill).
        prefill: Option<String>,
        /// The reply continues the last message (`continue_final_message`, or a trailing
        /// assistant message on the OpenAI routes), built as llama-server builds it.
        continuation: Option<Continuation>,
        /// `add_generation_prompt` (OpenAI routes; true elsewhere).
        add_generation_prompt: bool,
        tools: Vec<ToolDef>,
        tool_choice: ToolChoice,
        parallel_tool_calls: bool,
        /// Extra template variables, for example `enable_thinking`.
        kwargs: serde_json::Map<String, serde_json::Value>,
    },
    Text(String),
    /// Token ids as they are (a native `prompt` of ids, `/infill`).
    Tokens(Vec<TokenId>),
}

/// One generation request after API-specific parsing.
#[derive(Clone)]
pub struct GenerateSpec {
    pub model: String,
    pub prompt: Prompt,
    /// `None`: up to the end of the context.
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub extra: SamplingExtras,
    pub stop: Vec<String>,
    pub format: Option<OutputFormat>,
    /// Who pays: OpenAI `user`, Anthropic `metadata.user_id`.
    pub caller: Option<String>,
}

impl GenerateSpec {
    pub fn new(model: impl Into<String>, prompt: Prompt) -> Self {
        Self {
            model: model.into(),
            prompt,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            extra: SamplingExtras::default(),
            stop: Vec::new(),
            format: None,
            caller: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutEvent {
    Segment(Segment),
    /// The probabilities of a generated token (only with `SamplingExtras::logprobs`), sent before
    /// the text segments of the token.
    Logprobs(LogprobEntry),
    Done {
        finish: Finish,
        usage: Usage,
    },
    /// Generation failed after it started; nothing follows.
    Error(GatewayError),
}

/// A running generation. Dropping it cancels the engine work.
pub struct Generation {
    /// Random id; the API layers add their prefix (`chatcmpl-`, `msg_`).
    pub id: String,
    pub model: String,
    /// Unix seconds.
    pub created: u64,
    pub prompt_tokens: u32,
    rx: mpsc::Receiver<OutEvent>,
    _cancel_on_drop: DropGuard,
}

/// A whole generation, for non-streaming responses.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Collected {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish: Option<Finish>,
    pub usage: Usage,
    pub logprobs: Vec<LogprobEntry>,
}

impl Generation {
    /// A generation that replays `events` and ends (tests of the stream encoders).
    #[cfg(test)]
    pub(crate) fn scripted(events: Vec<OutEvent>) -> Self {
        let (tx, rx) = mpsc::channel(events.len().max(1));
        for e in events {
            assert!(tx.try_send(e).is_ok(), "channel sized for the script");
        }
        Self {
            id: "scripted".into(),
            model: "fake".into(),
            created: 0,
            prompt_tokens: 1,
            rx,
            _cancel_on_drop: CancellationToken::new().drop_guard(),
        }
    }

    pub async fn next(&mut self) -> Option<OutEvent> {
        self.rx.recv().await
    }

    pub async fn collect(mut self) -> Result<Collected, GatewayError> {
        let mut c = Collected::default();
        while let Some(ev) = self.next().await {
            match ev {
                OutEvent::Segment(Segment::Text(t)) => c.text.push_str(&t),
                OutEvent::Segment(Segment::Reasoning(r)) => {
                    c.reasoning.get_or_insert_with(String::new).push_str(&r)
                }
                OutEvent::Segment(Segment::ToolCall(tc)) => c.tool_calls.push(tc),
                OutEvent::Logprobs(e) => c.logprobs.push(e),
                OutEvent::Done { finish, usage } => {
                    c.finish = Some(finish);
                    c.usage = usage;
                }
                OutEvent::Error(e) => return Err(e),
            }
        }
        if c.finish.is_none() {
            return Err(GatewayError::Internal("generation ended early".into()));
        }
        Ok(c)
    }
}

struct GatewayMetrics {
    genai: tetraeder_telemetry::Metrics,
    request_cost: Histogram<f64>,
    queue_depth: Gauge<u64>,
    /// Tool calls by parser outcome and class (ADR 0051).
    tool_calls: Counter<u64>,
    /// HTTP requests by matched route and status (ADR 0089).
    requests: Counter<u64>,
}

impl GatewayMetrics {
    fn new(meter: &Meter) -> Self {
        Self {
            genai: tetraeder_telemetry::Metrics::new(meter),
            request_cost: meter
                .f64_histogram("tetraeder.request.cost")
                .with_description("Resources used by one request, per resource")
                .build(),
            queue_depth: meter
                .u64_gauge("tetraeder.scheduler.queue.depth")
                .with_unit("{request}")
                .with_description("Requests waiting for a generation slot")
                .build(),
            tool_calls: meter
                .u64_counter("tetraeder.gateway.tool_calls")
                .with_unit("{call}")
                .with_description("Tool calls by parser outcome and class")
                .build(),
            requests: meter
                .u64_counter("tetraeder.gateway.requests")
                .with_unit("{request}")
                .with_description("HTTP requests by matched route and response status")
                .build(),
        }
    }
}

struct Admission {
    slots: Arc<Semaphore>,
    waiting: AtomicUsize,
    max_queued: usize,
}

struct ModelState {
    info: ModelInfo,
    template: Option<ChatTemplate>,
    template_error: Option<String>,
    tool_format: ToolFormat,
    /// The template knows `<think>` blocks.
    think: bool,
    vocab: Arc<Vocab>,
    constraints: OnceLock<Result<Arc<ConstraintFactory>, String>>,
}

impl ModelState {
    fn constraints(&self) -> Result<Arc<ConstraintFactory>, GatewayError> {
        self.constraints
            .get_or_init(|| ConstraintFactory::new(&self.vocab).map(Arc::new))
            .clone()
            .map_err(|e| GatewayError::Internal(format!("constraint setup failed: {e}")))
    }
}

pub struct Gateway {
    engine: Arc<dyn Engine>,
    cfg: GatewayConfig,
    admission: Admission,
    metrics: Option<GatewayMetrics>,
    cost_sink: Option<Arc<dyn CostSink>>,
    models: Mutex<HashMap<String, Arc<ModelState>>>,
    callers: Mutex<HashSet<String>>,
    counters: Counters,
}

/// Response header carrying the generation id (the request log joins costs on it).
pub const GENERATION_ID_HEADER: &str = "x-tetraeder-generation-id";

/// Adds [`GENERATION_ID_HEADER`] to a response.
pub fn tag_generation(mut r: axum::response::Response, id: &str) -> axum::response::Response {
    if let Ok(v) = axum::http::HeaderValue::from_str(id) {
        r.headers_mut().insert(GENERATION_ID_HEADER, v);
    }
    r
}

/// Distinct caller ids kept as metric label values; later ones count as `other`.
const MAX_CALLERS: usize = 100;
const MAX_CALLER_LEN: usize = 64;

/// Seconds since the Unix epoch (0 if the clock is before it).
pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// 24 random lowercase hex characters.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..24].to_owned()
}

impl Gateway {
    pub fn new(engine: Arc<dyn Engine>, cfg: GatewayConfig) -> Self {
        Self {
            admission: Admission {
                slots: Arc::new(Semaphore::new(cfg.max_concurrent.max(1))),
                waiting: AtomicUsize::new(0),
                max_queued: cfg.max_queued,
            },
            engine,
            cfg,
            metrics: None,
            cost_sink: None,
            models: Mutex::default(),
            callers: Mutex::default(),
            counters: Counters::new(),
        }
    }

    /// Records `gen_ai.*` and gateway metrics on this meter.
    pub fn with_meter(mut self, meter: &Meter) -> Self {
        self.metrics = Some(GatewayMetrics::new(meter));
        self
    }

    pub fn with_cost_sink(mut self, sink: Arc<dyn CostSink>) -> Self {
        self.cost_sink = Some(sink);
        self
    }

    pub fn models(&self) -> Vec<ModelInfo> {
        self.engine.models()
    }

    pub fn config(&self) -> &GatewayConfig {
        &self.cfg
    }

    /// False while the engine cannot take requests (`/health` answers 503).
    pub fn ready(&self) -> bool {
        self.engine.ready()
    }

    /// Counts one HTTP request by matched route and response status.
    pub fn count_request(&self, route: &str, status: u16) {
        if let Some(m) = &self.metrics {
            m.requests.add(
                1,
                &[
                    KeyValue::new("http.route", route.to_owned()),
                    KeyValue::new("http.response.status_code", i64::from(status)),
                ],
            );
        }
    }

    /// Parallel slots: the most completions one request may ask for.
    pub(crate) fn parallel_slots(&self) -> usize {
        self.cfg.max_concurrent.max(1)
    }

    /// llama-server's `/metrics` text, then the engine's own ([`Engine::metrics_text`]), and the
    /// server's start time (Unix seconds).
    pub fn render_metrics(&self) -> (String, u64) {
        let processing = self
            .cfg
            .max_concurrent
            .max(1)
            .saturating_sub(self.admission.slots.available_permits());
        let mut text = self.counters.render(processing, self.queue_depth());
        text += &self.engine.metrics_text();
        (text, self.counters.started_unix())
    }

    /// Tokenizes `text` as llama.cpp's `llama_tokenize` with the two switches.
    pub fn tokenize_with(
        &self,
        model: &str,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>, GatewayError> {
        let id = self.model(model)?.id;
        self.engine
            .tokenize_with(&id, text, add_special, parse_special)
            .map_err(|e| self.engine_err(e))
    }

    /// The model's vocabulary (token bytes, special and end-of-generation ids).
    pub fn vocab(&self, model: &str) -> Result<Arc<Vocab>, GatewayError> {
        let id = self.model(model)?.id;
        self.engine.vocab(&id).map_err(|e| self.engine_err(e))
    }

    /// The model's fill-in-the-middle tokens.
    pub fn fim_tokens(&self, model: &str) -> Result<FimTokens, GatewayError> {
        let id = self.model(model)?.id;
        Ok(self.engine.fim_tokens(&id))
    }

    /// What the served model offers for embeddings and reranking; `None` without `--embeddings`
    /// or `--reranking`.
    pub fn embedding_caps(&self) -> Option<EmbeddingCaps> {
        let id = self.model("").ok()?.id;
        self.engine.embedding_caps(&id)
    }

    /// Embeds one input (blocking: run it off the async runtime).
    pub fn embed(
        &self,
        model: &str,
        tokens: &[TokenId],
        normalize: i32,
    ) -> Result<Embedded, GatewayError> {
        let id = self.model(model)?.id;
        self.engine
            .embed(&id, tokens, normalize)
            .map_err(|e| self.engine_err(e))
    }

    /// Scores one document against a query (blocking, like [`Gateway::embed`]).
    pub fn rerank(
        &self,
        model: &str,
        query: &str,
        document: &str,
    ) -> Result<Reranked, GatewayError> {
        let id = self.model(model)?.id;
        self.engine
            .rerank(&id, query, document)
            .map_err(|e| self.engine_err(e))
    }

    /// Whether `--slot-save-path` is on.
    pub fn slot_actions(&self) -> bool {
        self.engine.slot_actions()
    }

    /// Queues a slot action; await the receiver for its answer. Dropping it cancels the action
    /// if it has not started.
    pub fn slot_action(&self, id: i32, action: SlotAction) -> Result<SlotWait, GatewayError> {
        self.engine
            .slot_action(id, action)
            .map_err(|e| self.engine_err(e))
    }

    /// Maps an engine error of a slot action to the gateway's.
    pub fn slot_err(&self, e: EngineError) -> GatewayError {
        self.engine_err(e)
    }

    /// The prompt text a request renders to, without generating (`/apply-template`).
    pub fn render_prompt(&self, spec: &GenerateSpec) -> Result<String, GatewayError> {
        let st = self.state(&spec.model)?;
        self.render(&st, &spec.prompt).map(|(text, _)| text)
    }

    /// The model `id` names, or the configured default model for an unknown id.
    pub fn model(&self, id: &str) -> Result<ModelInfo, GatewayError> {
        let models = self.engine.models();
        let default = self.cfg.default_model.as_deref();
        let find = |want: &str| models.iter().find(|m| m.id == want).cloned();
        // No model named: the served one, as llama-server answers every request with its model.
        if id.is_empty() {
            return models
                .first()
                .cloned()
                .ok_or_else(|| GatewayError::NotFound("no model is served".into()));
        }
        find(id)
            .or_else(|| default.and_then(find))
            .ok_or_else(|| GatewayError::NotFound(format!("model '{id}' not found")))
    }

    fn state(&self, id: &str) -> Result<Arc<ModelState>, GatewayError> {
        let info = self.model(id)?;
        let id = info.id.clone();
        let id = id.as_str();
        if let Some(s) = self
            .models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
        {
            return Ok(Arc::clone(s));
        }
        let vocab = self.engine.vocab(id).map_err(|e| self.engine_err(e))?;
        let (template, template_error) = match &info.chat_template {
            Some(src) => {
                match ChatTemplate::new(src, info.bos_token.as_deref(), info.eos_token.as_deref()) {
                    Ok(t) => (Some(t), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            }
            None => (None, None),
        };
        let src = info.chat_template.as_deref().unwrap_or_default();
        let st = Arc::new(ModelState {
            tool_format: ToolFormat::detect(src),
            think: src.contains("<think>"),
            info,
            template,
            template_error,
            vocab,
            constraints: OnceLock::new(),
        });
        self.models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.to_owned(), Arc::clone(&st));
        Ok(st)
    }

    /// Renders and tokenizes a chat prompt (`count_tokens`).
    pub fn count_tokens(&self, spec: &GenerateSpec) -> Result<u32, GatewayError> {
        let st = self.state(&spec.model)?;
        if let Prompt::Tokens(t) = &spec.prompt {
            return Ok(t.len() as u32);
        }
        let (text, _) = self.render(&st, &spec.prompt)?;
        let toks = self
            .engine
            .tokenize(&st.info.id, &text)
            .map_err(|e| self.engine_err(e))?;
        Ok(toks.len() as u32)
    }

    /// Returns the prompt text and whether output starts inside a think block.
    fn render(&self, st: &ModelState, prompt: &Prompt) -> Result<(String, bool), GatewayError> {
        match prompt {
            Prompt::Text(t) => Ok((t.clone(), false)),
            Prompt::Tokens(_) => Err(GatewayError::Internal(
                "a token prompt has no text to render".into(),
            )),
            Prompt::Chat {
                messages,
                prefill,
                continuation,
                add_generation_prompt,
                tools,
                tool_choice,
                kwargs,
                ..
            } => {
                let Some(tpl) = &st.template else {
                    let why = st
                        .template_error
                        .as_deref()
                        .map_or("it has no chat template".to_owned(), |e| {
                            format!("its chat template failed to load: {e}")
                        });
                    return Err(GatewayError::InvalidRequest(format!(
                        "model '{}' does not support chat: {why}",
                        st.info.id
                    )));
                };
                let tools: &[ToolDef] = if *tool_choice == ToolChoice::None {
                    &[]
                } else {
                    tools
                };
                let render = |msgs: &[ChatMessage], gen_prompt: bool| {
                    tpl.render(msgs, tools, gen_prompt, kwargs)
                        .map_err(|e| GatewayError::InvalidRequest(e.to_string()))
                };
                let text = match (continuation, prefill, messages.split_last()) {
                    // llama-server's continuation (`peg_generator::generate_parser`): the
                    // conversation before the message, the template's generation prompt (cut
                    // before its think block, which is reopened with the message's reasoning),
                    // then the message's content.
                    (Some(c), _, Some((last, rest))) => {
                        let base = render(rest, false)?;
                        let full = render(rest, true)?;
                        let mut gen_prompt = full.strip_prefix(&base).unwrap_or(&full).to_owned();
                        if st.think {
                            if let Some(i) = gen_prompt.find("<think>") {
                                gen_prompt.truncate(i);
                            }
                            gen_prompt.push_str("<think>");
                            gen_prompt.push_str(last.reasoning_content.as_deref().unwrap_or(""));
                            if *c == Continuation::Content {
                                gen_prompt.push_str("</think>");
                            }
                        }
                        if *c == Continuation::Content {
                            gen_prompt.push_str(&last.content);
                        }
                        base + &gen_prompt
                    }
                    // Render the final assistant message and cut after its text; if the
                    // template altered the text, fall back to the conversation without it.
                    (None, Some(p), _) => {
                        let full = render(messages, false)?;
                        match full.rfind(p.as_str()) {
                            Some(i) => full[..i + p.len()].to_owned(),
                            None => render(&messages[..messages.len() - 1], true)?,
                        }
                    }
                    _ => render(messages, *add_generation_prompt)?,
                };
                let open = st.think
                    && (*continuation == Some(Continuation::Reasoning)
                        || text.trim_end().ends_with("<think>"));
                Ok((text, open))
            }
        }
    }

    /// Validates, admits and starts a generation. Errors here happen before any output.
    pub async fn generate(
        self: &Arc<Self>,
        spec: GenerateSpec,
        op: Operation,
    ) -> Result<Generation, GatewayError> {
        let started = Instant::now();
        let id = new_id();
        let span = tracing::info_span!(
            "gen_ai",
            otel.name = format!("{} {}", op.as_str(), spec.model),
            otel.kind = "server",
            gen_ai.operation.name = op.as_str(),
            gen_ai.request.model = spec.model.as_str(),
            gen_ai.request.max_tokens = spec.max_tokens,
            gen_ai.request.temperature = spec.temperature,
            gen_ai.request.top_p = spec.top_p,
            gen_ai.response.id = id.as_str(),
            gen_ai.response.model = spec.model.as_str(),
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.finish_reasons = tracing::field::Empty,
            error.type = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
        );
        let res = self
            .start(&spec, op, id, started, span.clone(), None)
            .instrument(span.clone())
            .await;
        if let Err(e) = &res {
            record_error(&span, e.otel_type());
            if let Some(m) = &self.metrics {
                // A model the engine does not know is client input: not a label value.
                let model = match e {
                    GatewayError::NotFound(_) => "unknown",
                    _ => spec.model.as_str(),
                };
                m.genai.request_duration(
                    started.elapsed(),
                    model,
                    op.as_str(),
                    Some(e.otel_type()),
                );
            }
        }
        res
    }

    /// Starts `n` generations of one request (`n` above 1 in the OpenAI APIs), the slots taken
    /// together so that two such requests cannot each hold half of them. `specs` differ in
    /// `extra.seed_offset` only; `n` above the parallel slots is rejected, as in llama-server.
    pub async fn generate_many(
        self: &Arc<Self>,
        mut specs: Vec<GenerateSpec>,
        op: Operation,
    ) -> Result<Vec<Generation>, GatewayError> {
        let n = specs.len();
        let slots = self.cfg.max_concurrent.max(1);
        if n > slots {
            return Err(GatewayError::InvalidRequest(format!(
                "n must be at most {slots}, the number of parallel slots"
            )));
        }
        // llama-server wraps an id_slot out of bounds around the slots (`get_slot_by_id`).
        for s in &mut specs {
            s.extra.slot = s.extra.slot.map(|id| id % slots);
        }
        if n <= 1 {
            let mut out = Vec::with_capacity(n);
            for spec in specs {
                out.push(self.generate(spec, op).await?);
            }
            return Ok(out);
        }
        let mut permits = self.admit_many(n).await?;
        let mut out = Vec::with_capacity(n);
        for spec in specs {
            let started = Instant::now();
            let id = new_id();
            let span = tracing::info_span!(
                "gen_ai",
                otel.name = format!("{} {}", op.as_str(), spec.model),
                otel.kind = "server",
                gen_ai.operation.name = op.as_str(),
                gen_ai.request.model = spec.model.as_str(),
                gen_ai.response.id = id.as_str(),
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.response.finish_reasons = tracing::field::Empty,
                error.type = tracing::field::Empty,
                otel.status_code = tracing::field::Empty,
            );
            let permit = permits.pop();
            let res = self
                .start(&spec, op, id, started, span.clone(), permit)
                .instrument(span.clone())
                .await;
            match res {
                Ok(g) => out.push(g),
                Err(e) => {
                    record_error(&span, e.otel_type());
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// The number of tokens `text` makes (llama-server removes that many log-probability entries
    /// when a stop string ends the output).
    /// How many tokens `text` is as llama-server counts a stop word's (`common_tokenize` with no
    /// special tokens added and none parsed).
    pub fn token_count(&self, model: &str, text: &str) -> usize {
        self.model(model)
            .ok()
            .and_then(|m| self.engine.tokenize_with(&m.id, text, false, false).ok())
            .map_or(0, |t| t.len())
    }

    async fn start(
        self: &Arc<Self>,
        spec: &GenerateSpec,
        op: Operation,
        id: String,
        started: Instant,
        span: tracing::Span,
        permit: Option<OwnedSemaphorePermit>,
    ) -> Result<Generation, GatewayError> {
        let st = self.state(&spec.model)?;
        let model = st.info.id.clone();
        let (prompt, in_reasoning) = match &spec.prompt {
            Prompt::Tokens(t) => (t.clone(), false),
            p => {
                let (text, in_reasoning) = self.render(&st, p)?;
                let tokens = self
                    .engine
                    .tokenize(&model, &text)
                    .map_err(|e| self.engine_err(e))?;
                (tokens, in_reasoning)
            }
        };
        let ctx = st.info.context_length;
        let n_prompt = prompt.len() as u32;
        if n_prompt >= ctx {
            return Err(GatewayError::ContextExceeded(format!(
                "{n_prompt} tokens, the context length of '{model}' is {ctx}"
            )));
        }
        let max_tokens = match spec.max_tokens {
            Some(0) => {
                return Err(GatewayError::InvalidRequest(
                    "max_tokens must be at least 1".into(),
                ));
            }
            // With the context shift the request may outlive the context, as in llama-server.
            Some(m) if self.cfg.context_shift => m,
            None if self.cfg.context_shift => u32::MAX,
            // Clamped to the room left, as llama-server does.
            Some(m) => m.min(ctx - n_prompt),
            None => ctx - n_prompt,
        };

        let mut extra = spec.extra.clone();
        let schema_grammar = |s: &str| {
            self.engine
                .json_schema_to_grammar(&model, s)
                .map_err(|e| match self.engine_err(e) {
                    GatewayError::InvalidRequest(m) => {
                        GatewayError::InvalidRequest(format!("json_schema: {m}"))
                    }
                    e => e,
                })
        };
        // A chat response_format becomes llama-server's grammar when nothing else shapes the
        // output: no tools, no think block, no grammar of the request. Otherwise the token mask
        // keeps the think block and tool calls apart.
        let mut format = spec.format.clone();
        let mut unfence = None;
        if let (
            Some(f),
            Prompt::Chat {
                tools, tool_choice, ..
            },
        ) = (&format, &spec.prompt)
            && !st.think
            && (tools.is_empty() || *tool_choice == ToolChoice::None)
            && extra.grammar.is_none()
        {
            let schema = match f {
                OutputFormat::JsonObject => serde_json::json!({"type": "object"}),
                OutputFormat::JsonSchema(s) => s.clone(),
            };
            let g = response_format_grammar(&schema_grammar(&schema.to_string())?);
            extra.grammar = Some(GrammarSource::Gbnf(g));
            format = None;
            unfence = Some(pump::Unfence::default());
        }
        let (parser, constraint) = self.output_setup(&st, &spec.prompt, &format, in_reasoning)?;

        extra.grammar = match extra.grammar.take() {
            Some(GrammarSource::JsonSchema(s)) => Some(GrammarSource::Gbnf(schema_grammar(&s)?)),
            Some(GrammarSource::Gbnf(g)) => {
                self.engine
                    .check_grammar(&model, &g)
                    .map_err(|e| self.engine_err(e))?;
                Some(GrammarSource::Gbnf(g))
            }
            None => None,
        };
        let want_probs = extra.logprobs.is_some_and(|n| n > 0);

        let permit = match permit {
            Some(p) => p,
            None => self.admit().await?,
        };
        let cancel = CancellationToken::new();
        let erx = self
            .engine
            .generate(GenerateRequest {
                model: model.clone(),
                prompt,
                params: SamplingParams {
                    max_tokens,
                    temperature: spec.temperature,
                    top_p: spec.top_p,
                    top_k: spec.top_k,
                    seed: spec.seed,
                    extra,
                },
                constraint,
                cancel: cancel.clone(),
            })
            .map_err(|e| self.engine_err(e))?;

        let (tx, rx) = mpsc::channel(self.cfg.stream_buffer.max(1));
        let run = Run {
            gw: Arc::clone(self),
            id: id.clone(),
            model: model.clone(),
            caller: spec.caller.clone().unwrap_or_else(|| "anonymous".into()),
            op,
            started,
            prompt_tokens: n_prompt,
            vocab: Arc::clone(&st.vocab),
            parser,
            unfence,
            single_call: matches!(
                &spec.prompt,
                Prompt::Chat { parallel_tool_calls: false, tools, tool_choice, .. }
                    if !tools.is_empty() && *tool_choice != ToolChoice::None
            ),
            stop: StopMatcher::new(spec.stop.clone()),
            special: want_probs.then(|| st.vocab.special.iter().copied().collect()),
            tool_outcomes: Default::default(),
            cancel: cancel.clone(),
            span: span.clone(),
            _permit: permit,
        };
        tokio::spawn(run.pump(erx, tx).instrument(span));
        Ok(Generation {
            id,
            model,
            created: now_unix(),
            prompt_tokens: n_prompt,
            rx,
            _cancel_on_drop: cancel.drop_guard(),
        })
    }

    fn output_setup(
        &self,
        st: &ModelState,
        prompt: &Prompt,
        format: &Option<OutputFormat>,
        in_reasoning: bool,
    ) -> Result<OutputSetup, GatewayError> {
        let Prompt::Chat {
            tools,
            tool_choice,
            parallel_tool_calls,
            ..
        } = prompt
        else {
            let c = match format {
                Some(f) => Some(
                    st.constraints()?
                        .json(f, Reasoning::None)
                        .map_err(GatewayError::InvalidRequest)?,
                ),
                None => None,
            };
            return Ok((None, c));
        };
        let reasoning = match (st.think, in_reasoning) {
            (false, _) => Reasoning::None,
            (true, true) => Reasoning::Open,
            (true, false) => Reasoning::Optional,
        };
        let tools: &[ToolDef] = if *tool_choice == ToolChoice::None {
            &[]
        } else {
            tools
        };
        if let ToolChoice::Named(n) = tool_choice
            && !tools.iter().any(|t| &t.name == n)
        {
            return Err(GatewayError::InvalidRequest(format!(
                "tool_choice names '{n}', which is not in tools"
            )));
        }
        let forced = matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_));
        if forced && tools.is_empty() {
            return Err(GatewayError::InvalidRequest(
                "tool_choice requires a tool call but no tools are given".into(),
            ));
        }
        let constraint = if let Some(f) = format {
            Some(
                st.constraints()?
                    .json(f, reasoning)
                    .map_err(GatewayError::InvalidRequest)?,
            )
        } else if forced {
            Some(
                st.constraints()?
                    .tool_calls(
                        st.tool_format,
                        tools,
                        tool_choice,
                        *parallel_tool_calls,
                        reasoning,
                    )
                    .map_err(GatewayError::InvalidRequest)?,
            )
        } else {
            None
        };
        let parser = OutputParser::new(st.tool_format, tools, st.think, in_reasoning);
        Ok((Some(parser), constraint))
    }

    /// A generation slot held for the length of an embedding or rerank request, so those routes
    /// queue and are refused like generation (503 when the queue is full).
    pub async fn admit_auxiliary(&self) -> Result<OwnedSemaphorePermit, GatewayError> {
        self.admit().await
    }

    async fn admit(&self) -> Result<OwnedSemaphorePermit, GatewayError> {
        let mut p = self.admit_many(1).await?;
        p.pop()
            .ok_or_else(|| GatewayError::Internal("no admission permit".into()))
    }

    /// `n` generation slots at once, as `n` permits; waits in the queue as one request.
    async fn admit_many(&self, n: usize) -> Result<Vec<OwnedSemaphorePermit>, GatewayError> {
        let a = &self.admission;
        let n32 = u32::try_from(n).unwrap_or(u32::MAX);
        let split = |mut whole: OwnedSemaphorePermit| {
            let mut out = Vec::with_capacity(n);
            while let Some(one) = whole.split(1) {
                out.push(one);
            }
            out
        };
        if let Ok(p) = Arc::clone(&a.slots).try_acquire_many_owned(n32) {
            return Ok(split(p));
        }
        let overloaded = || GatewayError::Overloaded {
            retry_after_secs: self.cfg.retry_after_secs,
        };
        if a.waiting.fetch_add(1, Ordering::SeqCst) >= a.max_queued {
            a.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(overloaded());
        }
        self.record_queue_depth();
        // Leaves the queue on success, failure or cancellation (client gone while waiting).
        struct Leave<'a>(&'a Gateway);
        impl Drop for Leave<'_> {
            fn drop(&mut self) {
                self.0.admission.waiting.fetch_sub(1, Ordering::SeqCst);
                self.0.record_queue_depth();
            }
        }
        let _leave = Leave(self);
        Arc::clone(&a.slots)
            .acquire_many_owned(n32)
            .await
            .map(split)
            .map_err(|_| overloaded())
    }

    fn record_queue_depth(&self) {
        if let Some(m) = &self.metrics {
            m.queue_depth.record(
                self.admission.waiting.load(Ordering::SeqCst) as u64,
                &[KeyValue::new(attr::PRIORITY, "default")],
            );
        }
    }

    fn engine_err(&self, e: crate::engine::EngineError) -> GatewayError {
        use crate::engine::EngineError as E;
        match e {
            E::ModelNotFound(m) => GatewayError::NotFound(format!("model '{m}' not found")),
            E::Overloaded => GatewayError::Overloaded {
                retry_after_secs: self.cfg.retry_after_secs,
            },
            E::Invalid(m) => GatewayError::InvalidRequest(m),
            E::Other(m) => GatewayError::Internal(m),
        }
    }

    /// The caller as a metric label: the first `MAX_CALLERS` distinct ids, then `other`.
    /// Callers are client-supplied until auth exists, so they must not grow the series count.
    fn caller_label(&self, caller: &str) -> String {
        let caller: String = caller.chars().take(MAX_CALLER_LEN).collect();
        let mut seen = self.callers.lock().unwrap_or_else(|e| e.into_inner());
        if seen.contains(&caller) || (seen.len() < MAX_CALLERS && seen.insert(caller.clone())) {
            caller
        } else {
            "other".to_owned()
        }
    }

    /// Slots held by running requests (generation, embedding, rerank).
    pub fn busy_slots(&self) -> usize {
        self.parallel_slots()
            .saturating_sub(self.admission.slots.available_permits())
    }

    /// Requests waiting for a slot.
    pub fn queue_depth(&self) -> usize {
        self.admission.waiting.load(Ordering::SeqCst)
    }
}

/// The output parser (chat only) and the sampling constraint of one request.
type OutputSetup = (Option<OutputParser>, Option<Box<dyn TokenConstraint>>);

/// llama-server's grammar for a chat `response_format` on a template without reasoning
/// (`analyze_*::build_parser` at the pin, `chat-auto-parser-generator.cpp:119-133`, rendered
/// by `common_peg_arena::build_grammar`): the schema, bare or in a ```` ```json ```` fence, with
/// the converter's `space` around it. Its grammar starts with the generation prompt, which its
/// sampler is fed before the first token; here the grammar starts after it. `schema` is the
/// converter's output for the schema (`root` is renamed; a schema that refers to itself with
/// `"$ref": "#"` is not supported).
fn response_format_grammar(schema: &str) -> String {
    let mut out = String::from(
        "root ::= space space (\"```json\" space response-format space \"```\" | space response-format space)\n\
         response-format ::= response-format-schema\n",
    );
    for line in schema.lines() {
        match line.strip_prefix("root ::= ") {
            Some(body) => out.push_str(&format!("response-format-schema ::= {body}\n")),
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// Sets `error.type`; server failures also mark the span as an error, so tail sampling keeps it.
fn record_error(span: &tracing::Span, error_type: &'static str) {
    span.record("error.type", error_type);
    if error_type == "internal" {
        span.record("otel.status_code", "ERROR");
    }
}

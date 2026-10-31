//! The API-neutral core: prompt building, admission, generation, output parsing, metrics.

use crate::chat::{
    ChatMessage, Finish, OutputFormat, Segment, ToolCall, ToolChoice, ToolDef, Usage,
};
use crate::constraint::{ConstraintFactory, Reasoning};
use crate::decode::StopMatcher;
use crate::engine::{Engine, GenerateRequest, ModelInfo, SamplingParams, TokenConstraint, Vocab};
use crate::error::GatewayError;
use crate::template::ChatTemplate;
use crate::toolcall::{OutputParser, ToolFormat};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Gauge, Histogram, Meter};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_util::sync::{CancellationToken, DropGuard};
use tracing::Instrument;

mod cost;
mod pump;

pub use cost::{CostSink, RequestCost, attr};
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
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1,
            max_queued: 16,
            retry_after_secs: 1,
            stream_buffer: 32,
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

pub enum Prompt {
    Chat {
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDef>,
        tool_choice: ToolChoice,
        parallel_tool_calls: bool,
        /// Extra template variables, for example `enable_thinking`.
        kwargs: serde_json::Map<String, serde_json::Value>,
    },
    Text(String),
}

/// One generation request after API-specific parsing.
pub struct GenerateSpec {
    pub model: String,
    pub prompt: Prompt,
    /// `None`: up to the end of the context.
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
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
            stop: Vec::new(),
            format: None,
            caller: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutEvent {
    Segment(Segment),
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
}

impl Generation {
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
}

/// Distinct caller ids kept as metric label values; later ones count as `other`.
const MAX_CALLERS: usize = 100;
const MAX_CALLER_LEN: usize = 64;

fn now_unix() -> u64 {
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

    pub fn model(&self, id: &str) -> Result<ModelInfo, GatewayError> {
        self.engine
            .models()
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| GatewayError::NotFound(format!("model '{id}' not found")))
    }

    fn state(&self, id: &str) -> Result<Arc<ModelState>, GatewayError> {
        if let Some(s) = self
            .models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
        {
            return Ok(Arc::clone(s));
        }
        let info = self.model(id)?;
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
        let (text, _) = self.render(&st, &spec.prompt)?;
        let toks = self
            .engine
            .tokenize(&spec.model, &text)
            .map_err(|e| self.engine_err(e))?;
        Ok(toks.len() as u32)
    }

    /// Returns the prompt text and whether output starts inside a think block.
    fn render(&self, st: &ModelState, prompt: &Prompt) -> Result<(String, bool), GatewayError> {
        match prompt {
            Prompt::Text(t) => Ok((t.clone(), false)),
            Prompt::Chat {
                messages,
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
                let text = tpl
                    .render(messages, tools, true, kwargs)
                    .map_err(|e| GatewayError::InvalidRequest(e.to_string()))?;
                let open = st.think && text.trim_end().ends_with("<think>");
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
            .start(&spec, op, id, started, span.clone())
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
                m.genai
                    .request_duration(started.elapsed(), model, Some(e.otel_type()));
            }
        }
        res
    }

    async fn start(
        self: &Arc<Self>,
        spec: &GenerateSpec,
        op: Operation,
        id: String,
        started: Instant,
        span: tracing::Span,
    ) -> Result<Generation, GatewayError> {
        let st = self.state(&spec.model)?;
        let (text, in_reasoning) = self.render(&st, &spec.prompt)?;
        let prompt = self
            .engine
            .tokenize(&spec.model, &text)
            .map_err(|e| self.engine_err(e))?;
        let ctx = st.info.context_length;
        let n_prompt = prompt.len() as u32;
        if n_prompt >= ctx {
            return Err(GatewayError::InvalidRequest(format!(
                "prompt is {n_prompt} tokens, the context length of '{}' is {ctx}",
                spec.model
            )));
        }
        let max_tokens = match spec.max_tokens {
            Some(0) => {
                return Err(GatewayError::InvalidRequest(
                    "max_tokens must be at least 1".into(),
                ));
            }
            Some(m) if n_prompt.saturating_add(m) > ctx => {
                return Err(GatewayError::InvalidRequest(format!(
                    "prompt ({n_prompt} tokens) plus max_tokens ({m}) exceeds the context length of '{}' ({ctx})",
                    spec.model
                )));
            }
            Some(m) => m,
            None => ctx - n_prompt,
        };

        let (parser, constraint) =
            self.output_setup(&st, &spec.prompt, &spec.format, in_reasoning)?;

        let permit = self.admit().await?;
        let cancel = CancellationToken::new();
        let erx = self
            .engine
            .generate(GenerateRequest {
                model: spec.model.clone(),
                prompt,
                params: SamplingParams {
                    max_tokens,
                    temperature: spec.temperature,
                    top_p: spec.top_p,
                    top_k: spec.top_k,
                    seed: spec.seed,
                },
                constraint,
                cancel: cancel.clone(),
            })
            .map_err(|e| self.engine_err(e))?;

        let (tx, rx) = mpsc::channel(self.cfg.stream_buffer.max(1));
        let run = Run {
            gw: Arc::clone(self),
            id: id.clone(),
            model: spec.model.clone(),
            caller: spec.caller.clone().unwrap_or_else(|| "anonymous".into()),
            op,
            started,
            prompt_tokens: n_prompt,
            vocab: Arc::clone(&st.vocab),
            parser,
            single_call: matches!(
                &spec.prompt,
                Prompt::Chat { parallel_tool_calls: false, tools, tool_choice, .. }
                    if !tools.is_empty() && *tool_choice != ToolChoice::None
            ),
            stop: StopMatcher::new(spec.stop.clone()),
            cancel: cancel.clone(),
            span: span.clone(),
            _permit: permit,
        };
        tokio::spawn(run.pump(erx, tx).instrument(span));
        Ok(Generation {
            id,
            model: spec.model.clone(),
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

    async fn admit(&self) -> Result<OwnedSemaphorePermit, GatewayError> {
        let a = &self.admission;
        if let Ok(p) = Arc::clone(&a.slots).try_acquire_owned() {
            return Ok(p);
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
            .acquire_owned()
            .await
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

    /// Requests waiting for a slot.
    pub fn queue_depth(&self) -> usize {
        self.admission.waiting.load(Ordering::SeqCst)
    }
}

/// The output parser (chat only) and the sampling constraint of one request.
type OutputSetup = (Option<OutputParser>, Option<Box<dyn TokenConstraint>>);

/// Sets `error.type`; server failures also mark the span as an error, so tail sampling keeps it.
fn record_error(span: &tracing::Span, error_type: &'static str) {
    span.record("error.type", error_type);
    if error_type == "internal" {
        span.record("otel.status_code", "ERROR");
    }
}

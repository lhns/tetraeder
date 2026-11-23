//! OpenAI-compatible API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
//!
//! Shapes follow the official OpenAPI spec (openai/openai-openapi). Extensions accepted:
//! `top_k`, `chat_template_kwargs`, `reasoning_content` on assistant messages (as in
//! llama.cpp and vLLM).

use crate::chat::{
    ChatMessage, Finish, LogprobEntry, OutputFormat, Segment, ToolCall, ToolChoice, ToolDef, Usage,
    logprobs_json,
};
use crate::error::{GatewayError, OpenAiError};
use crate::gateway::{
    Collected, Gateway, GenerateSpec, Generation, Operation, OutEvent, Prompt, tag_generation,
};
use crate::sampling::{self, SamplingFields};
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use futures_util::stream::{self, Stream};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, OpenAiError>;

fn bad(msg: impl Into<String>) -> OpenAiError {
    OpenAiError(GatewayError::InvalidRequest(msg.into()))
}

/// Parses a JSON body into `T`, reporting errors in the OpenAI format.
pub(crate) fn parse_body<T: serde::de::DeserializeOwned>(
    body: &Bytes,
) -> std::result::Result<T, GatewayError> {
    serde_json::from_slice(body)
        .map_err(|e| GatewayError::InvalidRequest(format!("invalid request body: {e}")))
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    /// Missing: the served model.
    #[serde(default)]
    model: String,
    messages: Vec<Value>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    seed: Option<i64>,
    n: Option<u32>,
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<Value>,
    user: Option<String>,
    logprobs: Option<bool>,
    top_logprobs: Option<Value>,
    n_probs: Option<i64>,
    reasoning_effort: Option<String>,
    chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

#[derive(Deserialize)]
struct CompletionRequest {
    /// Missing: the served model.
    #[serde(default)]
    model: String,
    prompt: Value,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    seed: Option<i64>,
    n: Option<u32>,
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    suffix: Option<String>,
    logprobs: Option<i64>,
    n_probs: Option<i64>,
    user: Option<String>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// Text of a string or an array of `{"type": "text", "text"}` parts; other parts are rejected.
fn content_text(v: &Value) -> std::result::Result<String, GatewayError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.push_str(p.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some(t) => {
                        return Err(GatewayError::InvalidRequest(format!(
                            "content part type '{t}' is not supported, only text"
                        )));
                    }
                    None => {
                        return Err(GatewayError::InvalidRequest(
                            "content part without type".into(),
                        ));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(GatewayError::InvalidRequest(
            "content must be a string or an array of parts".into(),
        )),
    }
}

pub(crate) fn stop_list(v: Option<Value>) -> std::result::Result<Vec<String>, GatewayError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s]),
        Some(Value::Array(a)) => a
            .into_iter()
            .map(|s| match s {
                Value::String(s) => Ok(s),
                _ => Err(GatewayError::InvalidRequest(
                    "stop must hold strings".into(),
                )),
            })
            .collect(),
        Some(_) => Err(GatewayError::InvalidRequest(
            "stop must be a string or an array".into(),
        )),
    }
}

fn messages(raw: &[Value]) -> std::result::Result<Vec<ChatMessage>, GatewayError> {
    if raw.is_empty() {
        return Err(GatewayError::InvalidRequest(
            "messages must not be empty".into(),
        ));
    }
    let mut out: Vec<ChatMessage> = Vec::new();
    for m in raw {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::InvalidRequest("message without role".into()))?;
        let content = content_text(m.get("content").unwrap_or(&Value::Null))?;
        let mut msg = match role {
            "system" | "developer" => ChatMessage::new("system", content),
            "user" => ChatMessage::new("user", content),
            "assistant" => {
                let mut a = ChatMessage::new("assistant", content);
                a.reasoning_content = m
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                for tc in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let f = tc.get("function").unwrap_or(&Value::Null);
                    let args = f.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                    a.tool_calls.push(ToolCall {
                        id: tc
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        name: f
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        // Templates want an object; keep unparsable arguments as a string.
                        arguments: serde_json::from_str(args)
                            .unwrap_or_else(|_| Value::String(args.to_owned())),
                    });
                }
                a
            }
            "tool" => {
                let mut t = ChatMessage::new("tool", content);
                let id = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                t.name = id.as_ref().and_then(|id| {
                    out.iter()
                        .flat_map(|p| &p.tool_calls)
                        .find(|c| &c.id == id)
                        .map(|c| c.name.clone())
                });
                t.tool_call_id = id;
                t
            }
            other => {
                return Err(GatewayError::InvalidRequest(format!(
                    "unsupported role '{other}'"
                )));
            }
        };
        if let Some(n) = m.get("name").and_then(Value::as_str)
            && msg.role != "tool"
        {
            msg.name = Some(n.to_owned());
        }
        out.push(msg);
    }
    Ok(out)
}

fn tools(raw: Vec<Value>) -> std::result::Result<Vec<ToolDef>, GatewayError> {
    raw.into_iter()
        .map(|t| {
            if t.get("type").and_then(Value::as_str) != Some("function") {
                return Err(GatewayError::InvalidRequest(
                    "only function tools are supported".into(),
                ));
            }
            let f = t.get("function").cloned().unwrap_or(Value::Null);
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| GatewayError::InvalidRequest("tool function without name".into()))?;
            Ok(ToolDef {
                name: name.to_owned(),
                description: f
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters: f
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            })
        })
        .collect()
}

fn tool_choice(v: Option<Value>, has_tools: bool) -> std::result::Result<ToolChoice, GatewayError> {
    Ok(match v {
        None | Some(Value::Null) => {
            if has_tools {
                ToolChoice::Auto
            } else {
                ToolChoice::None
            }
        }
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            o => {
                return Err(GatewayError::InvalidRequest(format!(
                    "unknown tool_choice '{o}'"
                )));
            }
        },
        Some(o) => match o.pointer("/function/name").and_then(Value::as_str) {
            Some(n) if o.get("type").and_then(Value::as_str) == Some("function") => {
                ToolChoice::Named(n.to_owned())
            }
            _ => {
                return Err(GatewayError::InvalidRequest(
                    "unsupported tool_choice".into(),
                ));
            }
        },
    })
}

fn response_format(v: Option<Value>) -> std::result::Result<Option<OutputFormat>, GatewayError> {
    let Some(v) = v else { return Ok(None) };
    match v.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(OutputFormat::JsonObject)),
        Some("json_schema") => {
            let schema = v
                .pointer("/json_schema/schema")
                .cloned()
                .unwrap_or_else(|| json!({}));
            Ok(Some(OutputFormat::JsonSchema(schema)))
        }
        Some(t) => Err(GatewayError::InvalidRequest(format!(
            "unsupported response_format type '{t}'"
        ))),
    }
}

/// What a stream or a response calls this server, as llama-server's `system_fingerprint` does
/// with its build.
pub(crate) const FINGERPRINT: &str = concat!("tetraeder-", env!("CARGO_PKG_VERSION"));

/// Number of choices: `n`, from 1 to the parallel slots (llama-server's hard limits on
/// `n_cmpl`). Checked before the spec is cloned `n` times.
pub(crate) fn choice_count(
    n: Option<u32>,
    slots: usize,
) -> std::result::Result<usize, GatewayError> {
    match n {
        None => Ok(1),
        Some(0) => Err(GatewayError::InvalidRequest("n must be at least 1".into())),
        Some(n) if n as usize > slots => Err(GatewayError::InvalidRequest(format!(
            "n must be at most {slots}, the number of parallel slots"
        ))),
        Some(n) => Ok(n as usize),
    }
}

/// `n` copies of the spec; choice `i` samples with seed + i, as llama-server's child tasks do.
pub(crate) fn choice_specs(spec: GenerateSpec, n: usize) -> Vec<GenerateSpec> {
    (0..n)
        .map(|i| {
            let mut s = spec.clone();
            s.extra.seed_offset = i as u32;
            s
        })
        .collect()
}

/// llama-server's `n_probs`: how many top tokens to report; 0 is off.
pub(crate) fn probs_count(n: Option<i64>) -> std::result::Result<Option<u32>, GatewayError> {
    match n {
        None => Ok(None),
        Some(n) => match u32::try_from(n) {
            Ok(0) => Ok(None),
            Ok(n) => Ok(Some(n)),
            Err(_) => Err(GatewayError::InvalidRequest(
                "n_probs must not be negative".into(),
            )),
        },
    }
}

/// `logprobs: true` takes `top_logprobs` (default 20) as `n_probs`; `top_logprobs` alone is an
/// error (llama-server, `oaicompat_chat_params_parse`).
fn chat_n_probs(
    logprobs: bool,
    top_logprobs: Option<Value>,
    n_probs: Option<i64>,
    tools_and_stream: bool,
) -> std::result::Result<Option<u32>, GatewayError> {
    let top = top_logprobs.filter(|v| !v.is_null());
    if logprobs {
        if tools_and_stream {
            return Err(GatewayError::InvalidRequest(
                "logprobs is not supported with tools + stream".into(),
            ));
        }
        let n = match top {
            None => 20,
            Some(v) => v.as_i64().ok_or_else(|| {
                GatewayError::InvalidRequest("top_logprobs must be an integer".into())
            })?,
        };
        return probs_count(Some(n));
    }
    if top.is_some() {
        return Err(GatewayError::InvalidRequest(
            "top_logprobs requires logprobs to be set to true".into(),
        ));
    }
    probs_count(n_probs)
}

/// `reasoning_effort` as llama-server reads it: `none` turns thinking off, any other value goes
/// to the template as the `reasoning_effort` variable.
pub(crate) fn reasoning_effort(kwargs: &mut Map<String, Value>, effort: Option<&str>) {
    match effort {
        Some("none") => {
            kwargs.insert("enable_thinking".into(), Value::Bool(false));
            kwargs.remove("reasoning_effort");
        }
        Some(e) if !e.is_empty() => {
            kwargs.insert("reasoning_effort".into(), Value::String(e.to_owned()));
        }
        _ => {}
    }
}

pub(crate) fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
    })
}

pub(crate) fn tool_call_json(tc: &ToolCall) -> Value {
    let args = match &tc.arguments {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    json!({"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": args}})
}

/// An SSE frame carrying `v` as its data.
pub(crate) fn data(v: &Value) -> Event {
    Event::default().data(v.to_string())
}

pub(crate) fn sse(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    Sse::new(events.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// A parsed chat request: the generation, the number of choices and the stream options.
pub(crate) struct ChatPlan {
    pub spec: GenerateSpec,
    pub n: usize,
    pub stream: bool,
    pub include_usage: bool,
}

/// Validates a chat body and maps it onto a [`GenerateSpec`]; the chat route, `/apply-template`
/// and `/v1/responses` share it.
pub(crate) fn chat_plan(body: Value, slots: usize) -> std::result::Result<ChatPlan, GatewayError> {
    let req: ChatRequest = serde_json::from_value(body)
        .map_err(|e| GatewayError::InvalidRequest(format!("invalid request body: {e}")))?;
    let n = choice_count(req.n, slots)?;
    if req.sampling.has_grammar_and_schema() {
        return Err(GatewayError::InvalidRequest(
            "Cannot use both json_schema and grammar".into(),
        ));
    }
    let tools = tools(req.tools)?;
    let n_probs = chat_n_probs(
        req.logprobs.unwrap_or(false),
        req.top_logprobs,
        req.n_probs,
        req.stream && !tools.is_empty(),
    )?;
    let choice = tool_choice(req.tool_choice, !tools.is_empty())?;
    let mut kwargs = req.chat_template_kwargs.unwrap_or_default();
    reasoning_effort(&mut kwargs, req.reasoning_effort.as_deref());
    let mut spec = GenerateSpec::new(
        req.model,
        Prompt::Chat {
            messages: messages(&req.messages)?,
            prefill: None,
            tools,
            tool_choice: choice,
            parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
            kwargs,
        },
    );
    spec.max_tokens = req.max_completion_tokens.or(req.max_tokens);
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = sampling::top_k(req.top_k);
    spec.seed = sampling::seed(req.seed);
    spec.extra = req.sampling.extras()?;
    spec.extra.logprobs = n_probs;
    spec.stop = stop_list(req.stop)?;
    spec.format = response_format(req.response_format)?;
    spec.caller = req.user;
    Ok(ChatPlan {
        spec,
        n,
        stream: req.stream,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
    })
}

pub async fn chat_completions(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let ChatPlan {
        spec,
        n,
        stream,
        include_usage,
    } = chat_plan(parse_body(&body)?, gw.parallel_slots())?;
    let generations = gw
        .generate_many(choice_specs(spec, n), Operation::Chat)
        .await?;
    let gid = generations[0].id.clone();
    if stream {
        return Ok(tag_generation(
            sse(oai_stream(generations, true, include_usage)),
            &gid,
        ));
    }
    let (id, model, created) = (
        format!("chatcmpl-{gid}"),
        generations[0].model.clone(),
        generations[0].created,
    );
    let all = try_join_all(generations.into_iter().map(Generation::collect)).await?;
    let usage = total_usage(&all);
    let choices: Vec<Value> = all
        .into_iter()
        .enumerate()
        .map(|(i, c)| chat_choice(&gw, &model, i, c))
        .collect();
    let mut body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": FINGERPRINT,
        "choices": choices,
        "usage": usage_json(&usage),
    });
    if let Some(t) = &usage.timings {
        body["timings"] = t.to_json();
    }
    Ok(tag_generation(Json(body).into_response(), &gid))
}

fn chat_choice(gw: &Gateway, model: &str, index: usize, c: Collected) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": if c.text.is_empty() && !c.tool_calls.is_empty() { Value::Null } else { Value::String(c.text) },
        "refusal": null,
    });
    if !c.tool_calls.is_empty() {
        message["tool_calls"] = c.tool_calls.iter().map(tool_call_json).collect();
    }
    if let Some(r) = c.reasoning {
        message["reasoning_content"] = Value::String(r);
    }
    let finish = c.finish.as_ref().map_or("stop", Finish::openai_reason);
    let logprobs = response_logprobs(gw, model, c.logprobs, c.finish.as_ref());
    json!({"index": index, "message": message, "logprobs": logprobs, "finish_reason": finish})
}

/// When a stop string ended the output, drops the last entries, as many as the stop string has
/// tokens: what llama-server does in a non-streaming response (`server_slot::to_final`).
pub(crate) fn trim_stop_logprobs(
    gw: &Gateway,
    model: &str,
    entries: &mut Vec<LogprobEntry>,
    finish: Option<&Finish>,
) {
    if let Some(Finish::StopSequence(s)) = finish {
        let k = gw.token_count(model, s);
        entries.truncate(entries.len().saturating_sub(k));
    }
}

/// The `logprobs` of a non-streaming choice: null without entries.
fn response_logprobs(
    gw: &Gateway,
    model: &str,
    mut entries: Vec<LogprobEntry>,
    finish: Option<&Finish>,
) -> Value {
    trim_stop_logprobs(gw, model, &mut entries, finish);
    if entries.is_empty() {
        Value::Null
    } else {
        logprobs_json(&entries)
    }
}

/// A stream of one or several generations as OpenAI chunks, choices interleaved as they come.
/// A completions stream carries its usage on the finish chunk whatever `stream_options` say, as
/// llama-server's final result does; a chat stream adds the empty-choices chunk when asked.
fn oai_stream(
    gens: Vec<Generation>,
    chat: bool,
    usage: bool,
) -> impl Stream<Item = Event> + Send + 'static {
    let n = gens.len();
    let prefix = if chat { "chatcmpl-" } else { "cmpl-" };
    let s = OaiStream {
        id: format!("{prefix}{}", gens[0].id),
        created: gens[0].created,
        model: gens[0].model.clone(),
        chat,
        usage,
        started: false,
        tool_index: vec![0; n],
        lp: vec![Vec::new(); n],
        tally: Tally::new(n),
        done: false,
        events: merge(gens),
    };
    stream::unfold(s, |mut s| async move {
        if s.done {
            return None;
        }
        let mut out = Vec::new();
        if !s.started {
            s.started = true;
            for i in 0..s.tally.n {
                if s.chat {
                    let c = s.chunk(i, json!({"role": "assistant", "content": ""}), None);
                    out.push(data(&c));
                }
            }
            if !out.is_empty() {
                return Some((stream::iter(out), s));
            }
        }
        match s.events.next().await {
            Some((i, OutEvent::Segment(Segment::Text(t)))) => {
                let c = if s.chat {
                    s.chunk(i, json!({"content": t}), None)
                } else {
                    s.text_chunk(i, &t, None)
                };
                out.push(data(&c));
            }
            Some((i, OutEvent::Segment(Segment::Reasoning(r)))) if s.chat => {
                let c = s.chunk(i, json!({"reasoning_content": r}), None);
                out.push(data(&c));
            }
            Some((i, OutEvent::Segment(Segment::ToolCall(tc)))) if s.chat => {
                let mut call = tool_call_json(&tc);
                call["index"] = json!(s.tool_index[i]);
                s.tool_index[i] += 1;
                let c = s.chunk(i, json!({"tool_calls": [call]}), None);
                out.push(data(&c));
            }
            Some((_, OutEvent::Segment(_))) => {}
            Some((i, OutEvent::Logprobs(e))) => s.lp[i].push(e),
            Some((i, OutEvent::Done { finish, usage })) => {
                let mut fin = if s.chat {
                    s.chunk(i, json!({}), Some(finish.openai_reason()))
                } else {
                    s.text_chunk(i, "", Some(finish.openai_reason()))
                };
                s.tally.add(i, usage);
                if !s.tally.complete() {
                    out.push(data(&fin));
                } else {
                    let total = s.tally.total();
                    let timings = total.timings.map(|t| t.to_json());
                    if !s.chat {
                        // llama-server's final completions chunk: the finish and the usage
                        fin["usage"] = usage_json(&total);
                        if let Some(t) = timings {
                            fin["timings"] = t;
                        }
                        out.push(data(&fin));
                    } else if s.usage {
                        out.push(data(&fin));
                        let mut last = json!({
                            "id": s.id,
                            "object": if s.chat { "chat.completion.chunk" } else { "text_completion" },
                            "created": s.created,
                            "model": s.model,
                            "system_fingerprint": FINGERPRINT,
                            "choices": [],
                            "usage": usage_json(&total),
                        });
                        if let Some(t) = timings {
                            last["timings"] = t;
                        }
                        out.push(data(&last));
                    } else {
                        if let Some(t) = timings {
                            fin["timings"] = t;
                        }
                        out.push(data(&fin));
                    }
                    out.push(Event::default().data("[DONE]"));
                    s.done = true;
                }
            }
            Some((_, OutEvent::Error(e))) => {
                // A data frame carrying an `error` object, then [DONE], as llama-server ends it.
                out.push(data(&e.openai_body()));
                out.push(Event::default().data("[DONE]"));
                s.done = true;
            }
            None => {
                // Every choice finishes or fails; a stream that ends without that is cut short.
                let e = GatewayError::Internal("the generation ended without finishing".into());
                out.push(data(&e.openai_body()));
                out.push(Event::default().data("[DONE]"));
                s.done = true;
            }
        }
        Some((stream::iter(out), s))
    })
    .flatten()
}

pub(crate) type Merged = Pin<Box<dyn Stream<Item = (usize, OutEvent)> + Send>>;

/// The events of every generation as one stream tagged with the choice index. Dropping it drops
/// the generations, which cancels them.
pub(crate) fn merge(gens: Vec<Generation>) -> Merged {
    let streams = gens.into_iter().enumerate().map(|(i, g)| {
        stream::unfold(g, move |mut g| async move {
            g.next().await.map(|e| ((i, e), g))
        })
        .boxed()
    });
    Box::pin(stream::select_all(streams))
}

struct OaiStream {
    events: Merged,
    id: String,
    created: u64,
    model: String,
    chat: bool,
    usage: bool,
    started: bool,
    tool_index: Vec<usize>,
    /// Log-probability entries waiting for the next chunk of their choice.
    lp: Vec<Vec<LogprobEntry>>,
    tally: Tally,
    done: bool,
}

impl OaiStream {
    fn envelope(&self, choice: Value) -> Value {
        let mut c = json!({
            "id": self.id,
            "object": if self.chat { "chat.completion.chunk" } else { "text_completion" },
            "created": self.created,
            "model": self.model,
            "system_fingerprint": FINGERPRINT,
            "choices": [choice],
        });
        if self.usage && self.chat {
            c["usage"] = Value::Null;
        }
        c
    }

    fn logprobs(&mut self, index: usize) -> Value {
        let lp = std::mem::take(&mut self.lp[index]);
        if lp.is_empty() {
            Value::Null
        } else {
            logprobs_json(&lp)
        }
    }

    fn chunk(&mut self, index: usize, delta: Value, finish: Option<&str>) -> Value {
        let logprobs = self.logprobs(index);
        self.envelope(
            json!({"index": index, "delta": delta, "logprobs": logprobs, "finish_reason": finish}),
        )
    }

    fn text_chunk(&mut self, index: usize, text: &str, finish: Option<&str>) -> Value {
        let logprobs = self.logprobs(index);
        self.envelope(
            json!({"text": text, "index": index, "logprobs": logprobs, "finish_reason": finish}),
        )
    }
}

/// Usage of a request with several choices: one prompt, the completions added up (llama-server
/// takes the prompt, cache and timings of the first result).
struct Tally {
    n: usize,
    done: usize,
    total: Usage,
}

impl Tally {
    fn new(n: usize) -> Self {
        Self {
            n,
            done: 0,
            total: Usage::default(),
        }
    }

    fn add(&mut self, index: usize, u: Usage) {
        let sum = self.total.completion_tokens + u.completion_tokens;
        if index == 0 {
            self.total = u;
        }
        self.total.completion_tokens = sum;
        self.done += 1;
    }

    fn complete(&self) -> bool {
        self.done >= self.n
    }

    fn total(&self) -> Usage {
        self.total
    }
}

pub(crate) fn total_usage(all: &[Collected]) -> Usage {
    let mut t = Tally::new(all.len());
    for (i, c) in all.iter().enumerate() {
        t.add(i, c.usage);
    }
    t.total()
}

pub async fn completions(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let req: CompletionRequest = parse_body(&body)?;
    let n = choice_count(req.n, gw.parallel_slots())?;
    let n_probs = probs_count(req.n_probs.or(req.logprobs))?;
    if req.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(bad("suffix is not supported"));
    }
    let prompt = match req.prompt {
        Value::String(s) => s,
        Value::Array(a) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap_or_default().to_owned()
        }
        Value::Array(_) => {
            return Err(bad(
                "prompt must be one string; batches and token arrays are not supported",
            ));
        }
        _ => return Err(bad("prompt must be a string")),
    };
    let mut spec = GenerateSpec::new(req.model, Prompt::Text(prompt));
    // llama-server's `n_predict` default is -1, as for chat: no limit but the context.
    spec.max_tokens = req.max_tokens;
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = sampling::top_k(req.top_k);
    spec.seed = sampling::seed(req.seed);
    spec.extra = req.sampling.extras()?;
    spec.extra.logprobs = n_probs;
    spec.stop = stop_list(req.stop)?;
    spec.caller = req.user;
    let generations = gw
        .generate_many(choice_specs(spec, n), Operation::TextCompletion)
        .await?;
    let gid = generations[0].id.clone();
    if req.stream {
        return Ok(tag_generation(
            sse(oai_stream(generations, false, false)),
            &gid,
        ));
    }
    let (id, model, created) = (
        format!("cmpl-{gid}"),
        generations[0].model.clone(),
        generations[0].created,
    );
    let all = try_join_all(generations.into_iter().map(Generation::collect)).await?;
    let usage = total_usage(&all);
    let choices: Vec<Value> = all
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let text = c.text;
            let finish = c.finish.as_ref().map_or("stop", Finish::openai_reason);
            let logprobs = response_logprobs(&gw, &model, c.logprobs, c.finish.as_ref());
            json!({"text": text, "index": i, "logprobs": logprobs, "finish_reason": finish})
        })
        .collect();
    let mut body = json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": FINGERPRINT,
        "choices": choices,
        "usage": usage_json(&usage),
    });
    if let Some(t) = &usage.timings {
        body["timings"] = t.to_json();
    }
    Ok(tag_generation(Json(body).into_response(), &gid))
}

pub(crate) fn model_json(m: &crate::engine::ModelInfo) -> Value {
    json!({"id": m.id, "object": "model", "created": m.created, "owned_by": "tetraeder"})
}

pub async fn list_models(gw: &Gateway) -> Response {
    let data: Vec<Value> = gw.models().iter().map(model_json).collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

pub async fn get_model(gw: &Gateway, Path(id): Path<String>) -> Result<Response> {
    let m = gw.model(&id)?;
    Ok(Json(model_json(&m)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_none_turns_thinking_off() {
        let mut k = Map::new();
        k.insert("reasoning_effort".into(), json!("high"));
        reasoning_effort(&mut k, Some("none"));
        assert_eq!(k.get("enable_thinking"), Some(&json!(false)));
        assert!(!k.contains_key("reasoning_effort"));
        let mut k = Map::new();
        reasoning_effort(&mut k, Some("low"));
        assert_eq!(k.get("reasoning_effort"), Some(&json!("low")));
        reasoning_effort(&mut k, None);
        reasoning_effort(&mut k, Some(""));
        assert_eq!(k.len(), 1);
    }

    #[test]
    fn n_probs_follow_llama_server() {
        assert_eq!(chat_n_probs(false, None, None, false).unwrap(), None);
        assert_eq!(chat_n_probs(true, None, None, false).unwrap(), Some(20));
        assert_eq!(
            chat_n_probs(true, Some(json!(3)), None, false).unwrap(),
            Some(3)
        );
        // zero is off, as `n_probs` 0 is
        assert_eq!(
            chat_n_probs(true, Some(json!(0)), None, false).unwrap(),
            None
        );
        assert_eq!(chat_n_probs(false, None, Some(5), false).unwrap(), Some(5));
        assert!(chat_n_probs(false, Some(json!(3)), None, false).is_err());
        assert!(chat_n_probs(true, None, None, true).is_err());
        assert!(chat_n_probs(true, Some(json!(-1)), None, false).is_err());
    }

    /// Every frame of `s` as text.
    async fn frames(s: impl Stream<Item = Event>) -> Vec<String> {
        s.map(|e| format!("{e:?}")).collect().await
    }

    #[tokio::test]
    async fn a_stream_cut_short_ends_with_an_error_and_done() {
        let done = OutEvent::Done {
            finish: Finish::Stop,
            usage: Usage::default(),
        };
        // choice 1 never finishes
        let gens = vec![
            Generation::scripted(vec![done]),
            Generation::scripted(vec![]),
        ];
        let f = frames(oai_stream(gens, true, false)).await;
        let n = f.len();
        assert!(f[n - 2].contains("server_error"), "{f:?}");
        assert!(f[n - 1].contains("[DONE]"), "{f:?}");
    }

    #[test]
    fn choices_start_at_one_and_total_their_completions() {
        assert_eq!(choice_count(None, 2).unwrap(), 1);
        assert!(choice_count(Some(0), 2).is_err());
        assert_eq!(choice_count(Some(2), 2).unwrap(), 2);
        assert!(choice_count(Some(3), 2).is_err());
        assert!(choice_count(Some(u32::MAX), 2).is_err());
        let u = |p, c| Usage {
            prompt_tokens: p,
            completion_tokens: c,
            ..Usage::default()
        };
        let mut t = Tally::new(2);
        // the second choice finishes first
        t.add(1, u(9, 4));
        assert!(!t.complete());
        t.add(0, u(7, 5));
        assert!(t.complete());
        assert_eq!(t.total().prompt_tokens, 7);
        assert_eq!(t.total().completion_tokens, 9);
    }
}

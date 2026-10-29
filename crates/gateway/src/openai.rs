//! OpenAI-compatible API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
//!
//! Shapes follow the official OpenAPI spec (openai/openai-openapi). Extensions accepted:
//! `top_k`, `chat_template_kwargs`, `reasoning_content` on assistant messages (as in
//! llama.cpp and vLLM).

use crate::chat::{ChatMessage, Finish, OutputFormat, Segment, ToolCall, ToolChoice, ToolDef, Usage};
use crate::error::{GatewayError, OpenAiError};
use crate::gateway::{Gateway, GenerateSpec, Generation, OutEvent, Operation, Prompt};
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use futures_util::stream::{self, Stream};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::convert::Infallible;
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
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
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
    #[serde(default)]
    logprobs: bool,
    chat_template_kwargs: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: Value,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    n: Option<u32>,
    stop: Option<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    echo: bool,
    suffix: Option<String>,
    logprobs: Option<u32>,
    user: Option<String>,
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
                    Some("text") => out.push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some(t) => {
                        return Err(GatewayError::InvalidRequest(format!(
                            "content part type '{t}' is not supported, only text"
                        )));
                    }
                    None => return Err(GatewayError::InvalidRequest("content part without type".into())),
                }
            }
            Ok(out)
        }
        _ => Err(GatewayError::InvalidRequest("content must be a string or an array of parts".into())),
    }
}

fn stop_list(v: Option<Value>) -> std::result::Result<Vec<String>, GatewayError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s]),
        Some(Value::Array(a)) => a
            .into_iter()
            .map(|s| match s {
                Value::String(s) => Ok(s),
                _ => Err(GatewayError::InvalidRequest("stop must hold strings".into())),
            })
            .collect(),
        Some(_) => Err(GatewayError::InvalidRequest("stop must be a string or an array".into())),
    }
}

fn messages(raw: &[Value]) -> std::result::Result<Vec<ChatMessage>, GatewayError> {
    if raw.is_empty() {
        return Err(GatewayError::InvalidRequest("messages must not be empty".into()));
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
                for tc in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    let f = tc.get("function").unwrap_or(&Value::Null);
                    let args = f.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                    a.tool_calls.push(ToolCall {
                        id: tc.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        name: f.get("name").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        // Templates want an object; keep unparsable arguments as a string.
                        arguments: serde_json::from_str(args)
                            .unwrap_or_else(|_| Value::String(args.to_owned())),
                    });
                }
                a
            }
            "tool" => {
                let mut t = ChatMessage::new("tool", content);
                let id = m.get("tool_call_id").and_then(Value::as_str).map(str::to_owned);
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
                return Err(GatewayError::InvalidRequest(format!("unsupported role '{other}'")));
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
                return Err(GatewayError::InvalidRequest("only function tools are supported".into()));
            }
            let f = t.get("function").cloned().unwrap_or(Value::Null);
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| GatewayError::InvalidRequest("tool function without name".into()))?;
            Ok(ToolDef {
                name: name.to_owned(),
                description: f.get("description").and_then(Value::as_str).map(str::to_owned),
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
            o => return Err(GatewayError::InvalidRequest(format!("unknown tool_choice '{o}'"))),
        },
        Some(o) => match o.pointer("/function/name").and_then(Value::as_str) {
            Some(n) if o.get("type").and_then(Value::as_str) == Some("function") => {
                ToolChoice::Named(n.to_owned())
            }
            _ => return Err(GatewayError::InvalidRequest("unsupported tool_choice".into())),
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
        Some(t) => Err(GatewayError::InvalidRequest(format!("unsupported response_format type '{t}'"))),
    }
}

fn check_n(n: Option<u32>) -> std::result::Result<(), GatewayError> {
    match n {
        None | Some(1) => Ok(()),
        Some(_) => Err(GatewayError::InvalidRequest("only n = 1 is supported".into())),
    }
}

fn finish_reason(f: &Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::StopSequence(_) => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
    })
}

fn tool_call_json(tc: &ToolCall) -> Value {
    let args = match &tc.arguments {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    json!({"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": args}})
}

fn data(v: &Value) -> Event {
    Event::default().data(v.to_string())
}

pub(crate) fn sse(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    Sse::new(events.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

pub async fn chat_completions(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let req: ChatRequest = parse_body(&body)?;
    check_n(req.n)?;
    if req.logprobs {
        return Err(bad("logprobs are not supported"));
    }
    let tools = tools(req.tools)?;
    let choice = tool_choice(req.tool_choice, !tools.is_empty())?;
    let mut spec = GenerateSpec::new(
        req.model,
        Prompt::Chat {
            messages: messages(&req.messages)?,
            tools,
            tool_choice: choice,
            parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
            kwargs: req.chat_template_kwargs.unwrap_or_default(),
        },
    );
    spec.max_tokens = req.max_completion_tokens.or(req.max_tokens);
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = req.top_k;
    spec.seed = req.seed;
    spec.stop = stop_list(req.stop)?;
    spec.format = response_format(req.response_format)?;
    spec.caller = req.user;
    let generation = gw.generate(spec, Operation::Chat).await?;
    if req.stream {
        let usage = req.stream_options.is_some_and(|o| o.include_usage);
        return Ok(sse(chat_stream(generation, usage)));
    }
    let (id, model, created) = (
        format!("chatcmpl-{}", generation.id),
        generation.model.clone(),
        generation.created,
    );
    let c = generation.collect().await?;
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
    let finish = c.finish.as_ref().map_or("stop", finish_reason);
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": null, "finish_reason": finish}],
        "usage": usage_json(&c.usage),
    }))
    .into_response())
}

struct ChatStream {
    g: Generation,
    id: String,
    usage: bool,
    started: bool,
    tool_index: usize,
    done: bool,
}

impl ChatStream {
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        let mut c = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.g.created,
            "model": self.g.model,
            "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish}],
        });
        if self.usage {
            c["usage"] = Value::Null;
        }
        c
    }
}

fn chat_stream(g: Generation, usage: bool) -> impl Stream<Item = Event> + Send + 'static {
    let s = ChatStream {
        id: format!("chatcmpl-{}", g.id),
        g,
        usage,
        started: false,
        tool_index: 0,
        done: false,
    };
    stream::unfold(s, |mut s| async move {
        if s.done {
            return None;
        }
        let mut out = Vec::new();
        if !s.started {
            s.started = true;
            out.push(data(&s.chunk(json!({"role": "assistant", "content": ""}), None)));
            return Some((stream::iter(out), s));
        }
        match s.g.next().await {
            Some(OutEvent::Segment(Segment::Text(t))) => {
                out.push(data(&s.chunk(json!({"content": t}), None)));
            }
            Some(OutEvent::Segment(Segment::Reasoning(r))) => {
                out.push(data(&s.chunk(json!({"reasoning_content": r}), None)));
            }
            Some(OutEvent::Segment(Segment::ToolCall(tc))) => {
                let mut call = tool_call_json(&tc);
                call["index"] = json!(s.tool_index);
                s.tool_index += 1;
                out.push(data(&s.chunk(json!({"tool_calls": [call]}), None)));
            }
            Some(OutEvent::Done { finish, usage }) => {
                out.push(data(&s.chunk(json!({}), Some(finish_reason(&finish)))));
                if s.usage {
                    out.push(data(&json!({
                        "id": s.id,
                        "object": "chat.completion.chunk",
                        "created": s.g.created,
                        "model": s.g.model,
                        "choices": [],
                        "usage": usage_json(&usage),
                    })));
                }
                out.push(Event::default().data("[DONE]"));
                s.done = true;
            }
            Some(OutEvent::Error(e)) => {
                // The spec allows a data frame carrying an `error` object; no [DONE] follows.
                out.push(data(&e.openai_body()));
                s.done = true;
            }
            None => return None,
        }
        Some((stream::iter(out), s))
    })
    .flatten()
}

pub async fn completions(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let req: CompletionRequest = parse_body(&body)?;
    check_n(req.n)?;
    if req.logprobs.is_some_and(|n| n > 0) {
        return Err(bad("logprobs are not supported"));
    }
    if req.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(bad("suffix is not supported"));
    }
    let prompt = match req.prompt {
        Value::String(s) => s,
        Value::Array(a) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap_or_default().to_owned()
        }
        Value::Array(_) => {
            return Err(bad("prompt must be one string; batches and token arrays are not supported"));
        }
        _ => return Err(bad("prompt must be a string")),
    };
    let mut spec = GenerateSpec::new(req.model, Prompt::Text(prompt.clone()));
    // The spec's default.
    spec.max_tokens = Some(req.max_tokens.unwrap_or(16));
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = req.top_k;
    spec.seed = req.seed;
    spec.stop = stop_list(req.stop)?;
    spec.caller = req.user;
    let generation = gw.generate(spec, Operation::TextCompletion).await?;
    let echo = req.echo.then_some(prompt);
    if req.stream {
        let usage = req.stream_options.is_some_and(|o| o.include_usage);
        return Ok(sse(completion_stream(generation, usage, echo)));
    }
    let (id, model, created) = (
        format!("cmpl-{}", generation.id),
        generation.model.clone(),
        generation.created,
    );
    let c = generation.collect().await?;
    let text = echo.unwrap_or_default() + &c.text;
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": null,
                     "finish_reason": c.finish.as_ref().map_or("stop", finish_reason)}],
        "usage": usage_json(&c.usage),
    }))
    .into_response())
}

fn completion_stream(
    g: Generation,
    usage: bool,
    echo: Option<String>,
) -> impl Stream<Item = Event> + Send + 'static {
    let id = format!("cmpl-{}", g.id);
    let chunk = move |g: &Generation, text: &str, finish: Option<&str>| {
        let mut c = json!({
            "id": id,
            "object": "text_completion",
            "created": g.created,
            "model": g.model,
            "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}],
        });
        if usage {
            c["usage"] = Value::Null;
        }
        c
    };
    stream::unfold((g, echo, false), move |(mut g, mut echo, done)| {
        let chunk = chunk.clone();
        async move {
            if done {
                return None;
            }
            let mut out = Vec::new();
            if let Some(e) = echo.take() {
                out.push(data(&chunk(&g, &e, None)));
                return Some((stream::iter(out), (g, None, false)));
            }
            let mut done = false;
            match g.next().await {
                Some(OutEvent::Segment(Segment::Text(t))) => out.push(data(&chunk(&g, &t, None))),
                Some(OutEvent::Segment(_)) => {}
                Some(OutEvent::Done { finish, usage: u }) => {
                    out.push(data(&chunk(&g, "", Some(finish_reason(&finish)))));
                    if usage {
                        let mut last = chunk(&g, "", None);
                        last["choices"] = json!([]);
                        last["usage"] = usage_json(&u);
                        out.push(data(&last));
                    }
                    out.push(Event::default().data("[DONE]"));
                    done = true;
                }
                Some(OutEvent::Error(e)) => {
                    out.push(data(&e.openai_body()));
                    done = true;
                }
                None => return None,
            }
            Some((stream::iter(out), (g, None, done)))
        }
    })
    .flatten()
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

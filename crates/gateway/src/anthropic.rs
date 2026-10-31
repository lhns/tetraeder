//! Anthropic-compatible API: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`.
//!
//! Shapes follow platform.claude.com/docs/en/api (messages, streaming, errors, models).
//! Errors carry a `request_id` matching the `request-id` header.

use crate::chat::{
    ChatMessage, Finish, OutputFormat, Segment, ToolCall, ToolChoice, ToolDef, Usage,
};
use crate::error::{AnthropicError, GatewayError};
use crate::gateway::{Gateway, GenerateSpec, Generation, Operation, OutEvent, Prompt, new_id};
use crate::openai::{parse_body, sse};
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::HeaderValue;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use futures_util::stream::{self, Stream};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::sync::Arc;

#[derive(Deserialize)]
struct MessagesRequest {
    model: String,
    max_tokens: Option<u32>,
    messages: Vec<Value>,
    system: Option<Value>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
    metadata: Option<Value>,
    output_config: Option<Value>,
    /// Deprecated form of `output_config.format`.
    output_format: Option<Value>,
}

/// Text of a string or an array of text blocks; other blocks are rejected.
fn text_of(v: &Value, what: &str) -> Result<String, GatewayError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some(t) => {
                        return Err(GatewayError::InvalidRequest(format!(
                            "{what}: block type '{t}' is not supported here"
                        )));
                    }
                    None => {
                        return Err(GatewayError::InvalidRequest(format!(
                            "{what}: block without type"
                        )));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(GatewayError::InvalidRequest(format!(
            "{what} must be a string or an array of blocks"
        ))),
    }
}

/// Converts Anthropic messages to the HF shape: `tool_result` blocks become `tool` messages,
/// `tool_use` blocks become `tool_calls`, `thinking` becomes `reasoning_content`.
fn messages(system: Option<&Value>, raw: &[Value]) -> Result<Vec<ChatMessage>, GatewayError> {
    if raw.is_empty() {
        return Err(GatewayError::InvalidRequest(
            "messages: at least one message is required".into(),
        ));
    }
    let mut out = Vec::new();
    if let Some(s) = system {
        let s = text_of(s, "system")?;
        if !s.is_empty() {
            out.push(ChatMessage::new("system", s));
        }
    }
    let mut names: Vec<(String, String)> = Vec::new();
    for m in raw {
        let role = m.get("role").and_then(Value::as_str).unwrap_or_default();
        let blocks: Vec<Value> = match m.get("content") {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(a)) => a.clone(),
            _ => {
                return Err(GatewayError::InvalidRequest(
                    "messages: content must be a string or an array of blocks".into(),
                ));
            }
        };
        match role {
            "user" => {
                let mut text = String::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            text.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""))
                        }
                        Some("tool_result") => {
                            // Keep the order of the blocks: earlier text is its own message.
                            if !text.is_empty() {
                                out.push(ChatMessage::new("user", std::mem::take(&mut text)));
                            }
                            let id = b
                                .get("tool_use_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            let mut t = ChatMessage::new(
                                "tool",
                                text_of(
                                    b.get("content").unwrap_or(&Value::Null),
                                    "tool_result content",
                                )?,
                            );
                            t.tool_call_id = Some(id.to_owned());
                            t.name = names.iter().find(|(i, _)| i == id).map(|(_, n)| n.clone());
                            out.push(t);
                        }
                        Some(t) => {
                            return Err(GatewayError::InvalidRequest(format!(
                                "messages: block type '{t}' is not supported"
                            )));
                        }
                        None => {
                            return Err(GatewayError::InvalidRequest(
                                "messages: block without type".into(),
                            ));
                        }
                    }
                }
                if !text.is_empty() {
                    out.push(ChatMessage::new("user", text));
                }
            }
            "assistant" => {
                let mut a = ChatMessage::new("assistant", "");
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => a
                            .content
                            .push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                        Some("thinking") => {
                            let t = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                            a.reasoning_content
                                .get_or_insert_with(String::new)
                                .push_str(t);
                        }
                        Some("redacted_thinking") => {}
                        Some("tool_use") => {
                            let id = b
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            let name = b
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            names.push((id.clone(), name.clone()));
                            a.tool_calls.push(ToolCall {
                                id,
                                name,
                                arguments: b.get("input").cloned().unwrap_or_else(|| json!({})),
                            });
                        }
                        Some(t) => {
                            return Err(GatewayError::InvalidRequest(format!(
                                "messages: block type '{t}' is not supported"
                            )));
                        }
                        None => {
                            return Err(GatewayError::InvalidRequest(
                                "messages: block without type".into(),
                            ));
                        }
                    }
                }
                out.push(a);
            }
            r => {
                return Err(GatewayError::InvalidRequest(format!(
                    "messages: unsupported role '{r}'"
                )));
            }
        }
    }
    if raw
        .last()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
    {
        return Err(GatewayError::InvalidRequest(
            "This model does not support assistant message prefill. The conversation must end with a user message."
                .into(),
        ));
    }
    Ok(out)
}

/// Custom tools only; Anthropic-defined tools (with a `type` other than `custom`) are skipped.
fn tools(raw: &[Value]) -> Result<Vec<ToolDef>, GatewayError> {
    let mut out = Vec::new();
    for t in raw {
        match t.get("type").and_then(Value::as_str) {
            None | Some("custom") => {}
            Some(other) => {
                tracing::warn!(tool_type = other, "skipping Anthropic-defined tool");
                continue;
            }
        }
        let name = t
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::InvalidRequest("tools: name is required".into()))?;
        out.push(ToolDef {
            name: name.to_owned(),
            description: t
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            parameters: t.get("input_schema").cloned().ok_or_else(|| {
                GatewayError::InvalidRequest(format!("tools.{name}: input_schema is required"))
            })?,
        });
    }
    Ok(out)
}

/// Returns the choice and whether parallel calls are allowed.
fn tool_choice(v: Option<&Value>, has_tools: bool) -> Result<(ToolChoice, bool), GatewayError> {
    let Some(v) = v else {
        return Ok((
            if has_tools {
                ToolChoice::Auto
            } else {
                ToolChoice::None
            },
            true,
        ));
    };
    let parallel = !v
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let c = match v.get("type").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => ToolChoice::Named(
            v.get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    GatewayError::InvalidRequest("tool_choice: name is required".into())
                })?
                .to_owned(),
        ),
        _ => {
            return Err(GatewayError::InvalidRequest(
                "tool_choice: unknown type".into(),
            ));
        }
    };
    Ok((c, parallel))
}

fn output_format(req: &MessagesRequest) -> Result<Option<OutputFormat>, GatewayError> {
    let f = req
        .output_config
        .as_ref()
        .and_then(|c| c.get("format"))
        .or(req.output_format.as_ref());
    let Some(f) = f else { return Ok(None) };
    match f.get("type").and_then(Value::as_str) {
        Some("json_schema") => Ok(Some(OutputFormat::JsonSchema(
            f.get("schema").cloned().unwrap_or_else(|| json!({})),
        ))),
        _ => Err(GatewayError::InvalidRequest(
            "output_config.format: type must be json_schema".into(),
        )),
    }
}

fn spec(req: MessagesRequest) -> Result<GenerateSpec, GatewayError> {
    let tools = tools(&req.tools)?;
    let (choice, parallel) = tool_choice(req.tool_choice.as_ref(), !tools.is_empty())?;
    let mut kwargs = Map::new();
    match req
        .thinking
        .as_ref()
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
    {
        Some("enabled") | Some("adaptive") => {
            kwargs.insert("enable_thinking".into(), Value::Bool(true));
        }
        Some("disabled") => {
            kwargs.insert("enable_thinking".into(), Value::Bool(false));
        }
        _ => {}
    }
    let format = output_format(&req)?;
    let mut s = GenerateSpec::new(
        req.model,
        Prompt::Chat {
            messages: messages(req.system.as_ref(), &req.messages)?,
            tools,
            tool_choice: choice,
            parallel_tool_calls: parallel,
            kwargs,
        },
    );
    s.max_tokens = req.max_tokens;
    s.temperature = req.temperature;
    s.top_p = req.top_p;
    s.top_k = req.top_k;
    s.stop = req.stop_sequences;
    s.format = format;
    s.caller = req
        .metadata
        .as_ref()
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(s)
}

fn stop_reason(f: &Finish) -> (&'static str, Value) {
    match f {
        Finish::Stop => ("end_turn", Value::Null),
        Finish::Length => ("max_tokens", Value::Null),
        Finish::StopSequence(s) => ("stop_sequence", Value::String(s.clone())),
        Finish::ToolCalls => ("tool_use", Value::Null),
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens.saturating_sub(u.cached_tokens),
        "output_tokens": u.completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": u.cached_tokens,
    })
}

/// `call_x` ids from the parser become `toolu_x`.
fn tool_id(id: &str) -> String {
    format!("toolu_{}", id.strip_prefix("call_").unwrap_or(id))
}

fn with_request_id(mut r: Response, request_id: &str) -> Response {
    if let Ok(v) = HeaderValue::from_str(request_id) {
        r.headers_mut().insert("request-id", v);
    }
    r
}

pub async fn messages_handler(State(gw): State<Arc<Gateway>>, body: Bytes) -> Response {
    let request_id = format!("req_{}", new_id());
    match messages_inner(gw, body, &request_id).await {
        Ok(r) => with_request_id(r, &request_id),
        Err(e) => AnthropicError(e, request_id).into_response(),
    }
}

async fn messages_inner(
    gw: Arc<Gateway>,
    body: Bytes,
    request_id: &str,
) -> Result<Response, GatewayError> {
    let req: MessagesRequest = parse_body(&body)?;
    if req.max_tokens.is_none() {
        return Err(GatewayError::InvalidRequest(
            "max_tokens: Field required".into(),
        ));
    }
    let stream = req.stream;
    let generation = gw.generate(spec(req)?, Operation::Chat).await?;
    if stream {
        return Ok(sse(message_stream(generation, request_id.to_owned())));
    }
    let (id, model) = (format!("msg_{}", generation.id), generation.model.clone());
    let mut content: Vec<Value> = Vec::new();
    let mut finish = None;
    let mut usage = Usage::default();
    let mut g = generation;
    while let Some(ev) = g.next().await {
        match ev {
            OutEvent::Segment(s) => push_block(&mut content, s),
            OutEvent::Done {
                finish: f,
                usage: u,
            } => {
                finish = Some(f);
                usage = u;
            }
            OutEvent::Error(e) => return Err(e),
        }
    }
    let Some(finish) = finish else {
        return Err(GatewayError::Internal("generation ended early".into()));
    };
    let (reason, seq) = stop_reason(&finish);
    Ok(Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": seq,
        "usage": usage_json(&usage),
    }))
    .into_response())
}

/// Appends a segment to the content, merging with the previous block of the same kind.
fn push_block(content: &mut Vec<Value>, s: Segment) {
    let last_type = content
        .last()
        .and_then(|b| b.get("type"))
        .and_then(Value::as_str);
    match s {
        Segment::Text(t) => {
            if last_type == Some("text") {
                let b = content.last_mut().expect("checked");
                b["text"] = Value::String(b["text"].as_str().unwrap_or_default().to_owned() + &t);
            } else {
                content.push(json!({"type": "text", "text": t}));
            }
        }
        Segment::Reasoning(r) => {
            if last_type == Some("thinking") {
                let b = content.last_mut().expect("checked");
                b["thinking"] =
                    Value::String(b["thinking"].as_str().unwrap_or_default().to_owned() + &r);
            } else {
                content.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
            }
        }
        Segment::ToolCall(tc) => content.push(json!({
            "type": "tool_use",
            "id": tool_id(&tc.id),
            "name": tc.name,
            "input": tc.arguments,
        })),
    }
}

fn event(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

#[derive(Clone, Copy, PartialEq)]
enum Open {
    None,
    Text,
    Thinking,
}

struct MsgStream {
    g: Generation,
    request_id: String,
    started: bool,
    index: usize,
    open: Open,
    done: bool,
}

impl MsgStream {
    fn close(&mut self, out: &mut Vec<Event>) {
        if self.open == Open::Thinking {
            out.push(event(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": self.index,
                       "delta": {"type": "signature_delta", "signature": ""}}),
            ));
        }
        if self.open != Open::None {
            out.push(event(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.index}),
            ));
            self.index += 1;
            self.open = Open::None;
        }
    }

    fn open(&mut self, kind: Open, out: &mut Vec<Event>) {
        if self.open == kind {
            return;
        }
        self.close(out);
        let block = match kind {
            Open::Text => json!({"type": "text", "text": ""}),
            Open::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            Open::None => return,
        };
        out.push(event(
            "content_block_start",
            json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        ));
        self.open = kind;
    }

    fn delta(&self, delta: Value) -> Event {
        event(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
    }
}

fn message_stream(g: Generation, request_id: String) -> impl Stream<Item = Event> + Send + 'static {
    let s = MsgStream {
        g,
        request_id,
        started: false,
        index: 0,
        open: Open::None,
        done: false,
    };
    stream::unfold(s, |mut s| async move {
        if s.done {
            return None;
        }
        let mut out = Vec::new();
        if !s.started {
            s.started = true;
            let usage = usage_json(&Usage {
                prompt_tokens: s.g.prompt_tokens,
                ..Usage::default()
            });
            out.push(event(
                "message_start",
                json!({"type": "message_start", "message": {
                    "id": format!("msg_{}", s.g.id), "type": "message", "role": "assistant",
                    "content": [], "model": s.g.model, "stop_reason": null, "stop_sequence": null,
                    "usage": usage}}),
            ));
            out.push(event("ping", json!({"type": "ping"})));
            return Some((stream::iter(out), s));
        }
        match s.g.next().await {
            Some(OutEvent::Segment(Segment::Text(t))) => {
                s.open(Open::Text, &mut out);
                out.push(s.delta(json!({"type": "text_delta", "text": t})));
            }
            Some(OutEvent::Segment(Segment::Reasoning(r))) => {
                s.open(Open::Thinking, &mut out);
                out.push(s.delta(json!({"type": "thinking_delta", "thinking": r})));
            }
            Some(OutEvent::Segment(Segment::ToolCall(tc))) => {
                s.close(&mut out);
                out.push(event(
                    "content_block_start",
                    json!({"type": "content_block_start", "index": s.index, "content_block": {
                        "type": "tool_use", "id": tool_id(&tc.id), "name": tc.name, "input": {}}}),
                ));
                out.push(s.delta(
                    json!({"type": "input_json_delta", "partial_json": tc.arguments.to_string()}),
                ));
                out.push(event(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": s.index}),
                ));
                s.index += 1;
            }
            Some(OutEvent::Done { finish, usage }) => {
                s.close(&mut out);
                let (reason, seq) = stop_reason(&finish);
                out.push(event(
                    "message_delta",
                    json!({"type": "message_delta",
                           "delta": {"stop_reason": reason, "stop_sequence": seq},
                           "usage": usage_json(&usage)}),
                ));
                out.push(event("message_stop", json!({"type": "message_stop"})));
                s.done = true;
            }
            Some(OutEvent::Error(e)) => {
                let body = e.anthropic_body(&s.request_id);
                out.push(event(
                    "error",
                    json!({"type": "error", "error": body["error"]}),
                ));
                s.done = true;
            }
            None => return None,
        }
        Some((stream::iter(out), s))
    })
    .flatten()
}

pub async fn count_tokens(State(gw): State<Arc<Gateway>>, body: Bytes) -> Response {
    let request_id = format!("req_{}", new_id());
    let res = (|| {
        let mut req: MessagesRequest = parse_body(&body)?;
        req.max_tokens = Some(0);
        req.stream = false;
        let n = gw.count_tokens(&spec(req)?)?;
        Ok::<_, GatewayError>(Json(json!({"input_tokens": n})).into_response())
    })();
    match res {
        Ok(r) => with_request_id(r, &request_id),
        Err(e) => AnthropicError(e, request_id).into_response(),
    }
}

/// RFC 3339 UTC timestamp from Unix seconds.
fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn model_json(m: &crate::engine::ModelInfo) -> Value {
    json!({
        "type": "model",
        "id": m.id,
        "display_name": m.id,
        "created_at": rfc3339(m.created),
        "max_input_tokens": m.context_length,
        "max_tokens": m.context_length,
    })
}

pub async fn list_models(gw: &Gateway) -> Response {
    let data: Vec<Value> = gw.models().iter().map(model_json).collect();
    let first = data.first().map(|m| m["id"].clone()).unwrap_or(Value::Null);
    let last = data.last().map(|m| m["id"].clone()).unwrap_or(Value::Null);
    Json(json!({"data": data, "has_more": false, "first_id": first, "last_id": last}))
        .into_response()
}

pub async fn get_model(gw: &Gateway, Path(id): Path<String>) -> Response {
    let request_id = format!("req_{}", new_id());
    match gw.model(&id) {
        Ok(m) => with_request_id(Json(model_json(&m)).into_response(), &request_id),
        Err(e) => AnthropicError(e, request_id).into_response(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rfc3339_dates() {
        assert_eq!(super::rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::rfc3339(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(super::rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }
}

//! OpenAI Responses API as llama-server serves it (ADR 0089): the request is converted to a chat
//! body (`server_chat_convert_responses_to_chatcmpl`), the answer is a `response` object or the
//! `response.*` event stream. No `[DONE]`.

use crate::chat::{Segment, ToolCall, Usage};
use crate::error::LlamaError;
use crate::gateway::{Collected, Gateway, Generation, Operation, OutEvent, new_id, tag_generation};
use crate::openai::{ChatPlan, chat_plan, sse};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures_util::stream::{self, Stream};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, LlamaError>;

fn bad(msg: impl Into<String>) -> LlamaError {
    LlamaError::invalid(msg)
}

fn is_str(v: &Value, key: &str) -> bool {
    v.get(key).is_some_and(Value::is_string)
}

fn is_array(v: &Value, key: &str) -> bool {
    v.get(key).is_some_and(Value::is_array)
}

fn text_part(text: &Value) -> Value {
    json!({"text": text, "type": "text"})
}

/// An `input_text` or `input_image` part of a message as a chat content part.
fn input_part(p: &Value) -> Result<Value> {
    match p.get("type").and_then(Value::as_str).unwrap_or_default() {
        "input_text" => match p.get("text") {
            Some(t) => Ok(text_part(t)),
            None => Err(bad("'Input text' requires 'text'")),
        },
        "input_image" => match p.get("image_url") {
            Some(u) => Ok(json!({"image_url": {"url": u}, "type": "image_url"})),
            None => Err(bad("'image_url' is required")),
        },
        "input_file" => Err(bad(
            "'input_file' is not supported by this server at this moment",
        )),
        _ => Err(bad(
            "'type' must be one of 'input_text', 'input_image', or 'input_file'",
        )),
    }
}

/// The output parts of an assistant message as chat content parts.
fn output_parts(item: &Value) -> Result<Vec<Value>> {
    match item.get("content") {
        Some(Value::String(s)) => Ok(vec![text_part(&json!(s))]),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(
                |p| match p.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "output_text" | "input_text" => {
                        if is_str(p, "text") {
                            Ok(text_part(&p["text"]))
                        } else {
                            Err(bad("'Output text' requires 'text'"))
                        }
                    }
                    "refusal" => {
                        if is_str(p, "refusal") {
                            Ok(json!({"refusal": p["refusal"], "type": "refusal"}))
                        } else {
                            Err(bad("'Refusal' requires 'refusal'"))
                        }
                    }
                    _ => Err(bad("'type' must be one of 'output_text' or 'refusal'")),
                },
            )
            .collect(),
        _ => Ok(Vec::new()),
    }
}

/// The tool-call output parts (`input_text`, `input_image`) as chat parts.
fn tool_output_parts(parts: &[Value]) -> Result<Value> {
    let mut out = Vec::new();
    for p in parts {
        let mut p = p.clone();
        match p.get("type").and_then(Value::as_str) {
            None => return Err(bad("Output of tool call missing 'type' field")),
            Some("input_text") => p["type"] = json!("text"),
            Some("input_image") => {
                let Some(url) = p.get("image_url").cloned() else {
                    return Err(bad("'image_url' is required"));
                };
                p["type"] = json!("image_url");
                p["image_url"] = json!({"url": url});
            }
            Some(_) => {
                return Err(bad(
                    "Output of tool call should be 'Input text' or 'Input image'",
                ));
            }
        }
        out.push(p);
    }
    Ok(Value::Array(out))
}

fn tools(tools: &Value) -> Result<Vec<Value>> {
    let list = tools
        .as_array()
        .ok_or_else(|| bad("'tools' must be an array of objects"))?;
    // Tools that are not functions have no chat equivalent and are skipped, as llama-server does.
    Ok(list
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("function"))
        .map(|t| {
            let mut f = t.clone();
            if let Some(o) = f.as_object_mut() {
                o.remove("type");
                o.entry("strict").or_insert(json!(true));
            }
            json!({"type": "function", "function": f})
        })
        .collect())
}

/// Appends to the last message when that is the assistant's (items of one turn share a message).
fn last_is_assistant(messages: &[Value]) -> bool {
    messages
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
}

fn convert_items(items: &[Value], messages: &mut Vec<Value>) -> Result<()> {
    for item in items {
        let mut item = item.clone();
        let merge = last_is_assistant(messages);
        if let Some(Value::String(s)) = item.get("content") {
            item["content"] = json!([{"text": s, "type": "input_text"}]);
        }
        let role = item.get("role").and_then(Value::as_str).unwrap_or_default();
        let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
        if is_array(&item, "content") && matches!(role, "user" | "system" | "developer") {
            let parts = item["content"]
                .as_array()
                .into_iter()
                .flatten()
                .map(input_part)
                .collect::<Result<Vec<_>>>()?;
            if let Some(o) = item.as_object_mut() {
                o.remove("type");
                o.remove("status");
            }
            item["content"] = Value::Array(parts);
            messages.push(item);
        } else if role == "assistant" && kind == "message" {
            let parts = output_parts(&item)?;
            if merge {
                if let Some(prev) = messages.last_mut() {
                    match prev.get_mut("content").and_then(Value::as_array_mut) {
                        Some(c) => c.extend(parts),
                        None => prev["content"] = Value::Array(parts),
                    }
                }
            } else {
                if let Some(o) = item.as_object_mut() {
                    o.remove("status");
                    o.remove("type");
                }
                item["content"] = Value::Array(parts);
                messages.push(item);
            }
        } else if kind == "function_call"
            && is_str(&item, "arguments")
            && is_str(&item, "call_id")
            && is_str(&item, "name")
        {
            let call = json!({
                "function": {"arguments": item["arguments"], "name": item["name"]},
                "id": item["call_id"],
                "type": "function",
            });
            if merge {
                if let Some(prev) = messages.last_mut() {
                    match prev.get_mut("tool_calls").and_then(Value::as_array_mut) {
                        Some(c) => c.push(call),
                        None => prev["tool_calls"] = json!([call]),
                    }
                }
            } else {
                messages.push(json!({"role": "assistant", "tool_calls": [call]}));
            }
        } else if kind == "function_call_output"
            && is_str(&item, "call_id")
            && (is_str(&item, "output") || is_array(&item, "output"))
        {
            let content = match &item["output"] {
                Value::Array(parts) => tool_output_parts(parts)?,
                s => s.clone(),
            };
            messages
                .push(json!({"content": content, "role": "tool", "tool_call_id": item["call_id"]}));
        } else if kind == "reasoning" && is_array(&item, "summary") {
            let text = item
                .get("content")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("item['content'] is not an array"))?
                .first()
                .ok_or_else(|| bad("item['content'] is empty"))?
                .get("text")
                .filter(|t| t.is_string())
                .ok_or_else(|| bad("item['content']['text'] is not a string"))?
                .clone();
            if merge {
                if let Some(prev) = messages.last_mut() {
                    prev["reasoning_content"] = text;
                }
            } else {
                messages
                    .push(json!({"role": "assistant", "content": [], "reasoning_content": text}));
            }
        } else {
            return Err(bad("Cannot determine type of 'item'"));
        }
    }
    Ok(())
}

/// llama-server's `server_chat_convert_responses_to_chatcmpl`.
pub(crate) fn to_chat_body(body: &Value) -> Result<Value> {
    let Some(obj) = body.as_object() else {
        return Err(bad("request body must be an object"));
    };
    let Some(input) = obj.get("input") else {
        return Err(bad("'input' is required"));
    };
    if obj
        .get("previous_response_id")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return Err(bad("'previous_response_id' is not supported."));
    }
    let mut chat: Map<String, Value> = obj.clone();
    chat.remove("input");
    let mut messages: Vec<Value> = Vec::new();
    if let Some(ins) = obj.get("instructions") {
        let text = ins.as_str().unwrap_or_default();
        messages.push(json!({"role": "system", "content": text}));
        chat.remove("instructions");
    }
    match input {
        Value::String(s) => messages.push(json!({"role": "user", "content": s})),
        Value::Array(items) => convert_items(items, &mut messages)?,
        _ => return Err(bad("'input' must be a string or array of objects")),
    }
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(t) = obj.get("tools") {
        let converted = tools(t)?;
        chat.remove("tools");
        if !converted.is_empty() {
            chat.insert("tools".into(), Value::Array(converted));
        }
    }
    // `{"type": "function", "name": ...}` names a tool without the chat route's `function` level.
    if let Some(Value::Object(tc)) = obj.get("tool_choice")
        && tc.get("type").and_then(Value::as_str) == Some("function")
        && let Some(name) = tc.get("name")
    {
        chat.insert(
            "tool_choice".into(),
            json!({"type": "function", "function": {"name": name}}),
        );
    }
    if let Some(m) = obj.get("max_output_tokens") {
        chat.remove("max_output_tokens");
        chat.insert("max_tokens".into(), m.clone());
    }
    if let Some(r) = obj.get("reasoning") {
        if let Some(e) = r.get("effort") {
            chat.insert("reasoning_effort".into(), e.clone());
        }
        chat.remove("reasoning");
    }
    if let Some(text) = obj.get("text") {
        let Some(text) = text.as_object() else {
            return Err(bad("'text' must be an object"));
        };
        chat.remove("text");
        if let Some(format) = text.get("format") {
            let Some(format) = format.as_object() else {
                return Err(bad("'text.format' must be an object"));
            };
            match format
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "json_schema" => {
                    let mut schema = format.clone();
                    schema.remove("type");
                    chat.insert(
                        "response_format".into(),
                        json!({"type": "json_schema", "json_schema": schema}),
                    );
                }
                "json_object" | "text" => {
                    chat.insert("response_format".into(), Value::Object(format.clone()));
                }
                "" => {}
                _ => {
                    return Err(bad(
                        "'text.format.type' must be one of 'text', 'json_object', or 'json_schema'",
                    ));
                }
            }
        }
    }
    // llama.cpp parameters that have no Responses field travel in `extra_params`.
    if let Some(extra) = obj.get("extra_params") {
        chat.remove("extra_params");
        if let Some(o) = extra.as_object() {
            for (k, v) in o {
                chat.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(Value::Object(chat))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The id of a tool call without the `call_` the chat parser gave it, so `fc_` and `call_`
/// prefixes are not doubled.
fn bare(id: &str) -> &str {
    id.strip_prefix("call_").unwrap_or(id)
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "input_tokens_details": {"cached_tokens": u.cached_tokens},
    })
}

fn reasoning_item(id: &str, text: &str, status: Option<&str>) -> Value {
    let mut v = json!({
        "id": id,
        "summary": [],
        "type": "reasoning",
        "content": [{"text": text, "type": "reasoning_text"}],
        "encrypted_content": "",
    });
    if let Some(s) = status {
        v["status"] = json!(s);
    }
    v
}

fn message_item(id: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "status": "completed",
        "id": id,
        "content": [output_text(text)],
        "role": "assistant",
    })
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "annotations": [], "logprobs": [], "text": text})
}

fn call_item(tc: &ToolCall, status: &str, with_args: bool) -> Value {
    let args = match &tc.arguments {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    json!({
        "id": format!("fc_{}", bare(&tc.id)),
        "type": "function_call",
        "status": status,
        "arguments": if with_args { args } else { String::new() },
        "call_id": format!("call_{}", bare(&tc.id)),
        "name": tc.name,
    })
}

/// The `response` object of a finished non-streaming request.
fn response_json(id: &str, model: &str, c: &Collected) -> Value {
    let mut output = Vec::new();
    if let Some(r) = c.reasoning.as_deref().filter(|r| !r.is_empty()) {
        output.push(reasoning_item(
            &format!("rs_{}", new_id()),
            r,
            Some("completed"),
        ));
    }
    if !c.text.is_empty() {
        output.push(message_item(&format!("msg_{}", new_id()), &c.text));
    }
    output.extend(
        c.tool_calls
            .iter()
            .map(|tc| call_item(tc, "completed", true)),
    );
    let t = now();
    json!({
        "completed_at": t,
        "created_at": t,
        "id": id,
        "model": model,
        "object": "response",
        "output": output,
        "status": "completed",
        "usage": usage_json(&c.usage),
    })
}

pub async fn responses(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let body: Value =
        serde_json::from_slice(&body).map_err(|e| bad(format!("invalid request body: {e}")))?;
    let ChatPlan { spec, stream, .. } = chat_plan(to_chat_body(&body)?)?;
    let generation = gw.generate(spec, Operation::Chat).await?;
    let gid = generation.id.clone();
    let resp_id = format!("resp_{gid}");
    if stream {
        return Ok(tag_generation(sse(event_stream(generation, resp_id)), &gid));
    }
    let model = generation.model.clone();
    let c = generation.collect().await?;
    Ok(tag_generation(
        Json(response_json(&resp_id, &model, &c)).into_response(),
        &gid,
    ))
}

fn event(name: &str, mut data: Value) -> Event {
    data["type"] = json!(name);
    Event::default().event(name).data(data.to_string())
}

/// An output item of a stream, in output order.
enum Item {
    Reasoning { id: String, text: String },
    Message { id: String, text: String },
    Call(ToolCall),
}

struct Stream1 {
    gen_: Generation,
    id: String,
    started: bool,
    items: Vec<Item>,
    done: bool,
}

impl Stream1 {
    fn index_of(&self, pred: impl Fn(&Item) -> bool) -> Option<usize> {
        self.items.iter().position(pred)
    }

    fn response(&self, status: &str) -> Value {
        json!({"id": self.id, "object": "response", "status": status})
    }

    /// The events a text or reasoning delta needs: the item and part openers when it is the first.
    fn delta(&mut self, reasoning: bool, text: &str, out: &mut Vec<Event>) {
        let existing = self.index_of(|i| {
            if reasoning {
                matches!(i, Item::Reasoning { .. })
            } else {
                matches!(i, Item::Message { .. })
            }
        });
        let index = match existing {
            Some(i) => i,
            None => {
                let index = self.items.len();
                if reasoning {
                    let id = format!("rs_{}", new_id());
                    let mut item = reasoning_item(&id, "", Some("in_progress"));
                    item["content"] = json!([]);
                    out.push(event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": item}),
                    ));
                    self.items.push(Item::Reasoning {
                        id,
                        text: String::new(),
                    });
                } else {
                    let id = format!("msg_{}", new_id());
                    out.push(event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": {
                            "content": [], "id": id, "role": "assistant",
                            "status": "in_progress", "type": "message"}}),
                    ));
                    out.push(event(
                        "response.content_part.added",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                               "part": {"type": "output_text", "text": ""}}),
                    ));
                    self.items.push(Item::Message {
                        id,
                        text: String::new(),
                    });
                }
                index
            }
        };
        let (id, acc) = match &mut self.items[index] {
            Item::Reasoning { id, text } | Item::Message { id, text } => (id.clone(), text),
            Item::Call(_) => return,
        };
        acc.push_str(text);
        let name = if reasoning {
            "response.reasoning_text.delta"
        } else {
            "response.output_text.delta"
        };
        out.push(event(
            name,
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": text}),
        ));
    }

    fn call(&mut self, tc: ToolCall, out: &mut Vec<Event>) {
        let index = self.items.len();
        let item = call_item(&tc, "in_progress", false);
        let args = call_item(&tc, "in_progress", true)["arguments"].clone();
        out.push(event(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        ));
        out.push(event(
            "response.function_call_arguments.delta",
            json!({"delta": args, "item_id": format!("fc_{}", bare(&tc.id)), "output_index": index}),
        ));
        self.items.push(Item::Call(tc));
    }

    /// The closing events, then `response.completed`.
    fn finish(&mut self, model: &str, usage: &Usage, out: &mut Vec<Event>) {
        let mut output = Vec::new();
        for (index, item) in self.items.iter().enumerate() {
            let done = match item {
                Item::Reasoning { id, text } => reasoning_item(id, text, None),
                Item::Message { id, text } => {
                    out.push(event(
                        "response.output_text.done",
                        json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                    ));
                    out.push(event(
                        "response.content_part.done",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                               "part": output_text(text)}),
                    ));
                    message_item(id, text)
                }
                Item::Call(tc) => call_item(tc, "completed", true),
            };
            out.push(event(
                "response.output_item.done",
                json!({"output_index": index, "item": done}),
            ));
            output.push(done);
        }
        let mut completed = json!({"response": {
            "id": self.id,
            "object": "response",
            "created_at": now(),
            "status": "completed",
            "model": model,
            "output": output,
            "usage": usage_json(usage),
        }});
        if let Some(t) = &usage.timings {
            completed["timings"] = t.to_json();
        }
        out.push(event("response.completed", completed));
    }
}

fn event_stream(gen_: Generation, id: String) -> impl Stream<Item = Event> + Send + 'static {
    let model = gen_.model.clone();
    let s = Stream1 {
        gen_,
        id,
        started: false,
        items: Vec::new(),
        done: false,
    };
    use futures_util::StreamExt;
    stream::unfold(s, move |mut s| {
        let model = model.clone();
        async move {
            if s.done {
                return None;
            }
            let ev = s.gen_.next().await?;
            let mut out = Vec::new();
            if !s.started {
                s.started = true;
                for name in ["response.created", "response.in_progress"] {
                    out.push(event(name, json!({"response": s.response("in_progress")})));
                }
            }
            match ev {
                OutEvent::Segment(Segment::Text(t)) => s.delta(false, &t, &mut out),
                OutEvent::Segment(Segment::Reasoning(r)) => s.delta(true, &r, &mut out),
                OutEvent::Segment(Segment::ToolCall(tc)) => s.call(tc, &mut out),
                OutEvent::Logprobs(_) => {}
                OutEvent::Done { usage, .. } => {
                    s.finish(&model, &usage, &mut out);
                    s.done = true;
                }
                OutEvent::Error(e) => {
                    // a plain data frame with the error object, as llama-server ends a failed stream
                    out.push(
                        Event::default()
                            .data(json!({"error": LlamaError::from(e).object()}).to_string()),
                    );
                    s.done = true;
                }
            }
            Some((stream::iter(out), s))
        }
    })
    .flatten()
}

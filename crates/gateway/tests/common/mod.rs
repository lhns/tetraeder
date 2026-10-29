//! Test helpers: an in-process router over a fake engine, JSON and SSE requests.
#![allow(dead_code)]

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::Arc;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::server::router;
use tower::ServiceExt;

pub const QWEN3: &str = include_str!("../fixtures/templates/Qwen-Qwen3-0.6B.jinja");

pub fn app(engine: FakeEngine) -> (Router, Arc<Gateway>) {
    app_with(engine, GatewayConfig::default())
}

pub fn app_with(engine: FakeEngine, cfg: GatewayConfig) -> (Router, Arc<Gateway>) {
    let gw = Arc::new(Gateway::new(Arc::new(engine), cfg));
    (router(Arc::clone(&gw)), gw)
}

pub fn request(method: &str, path: &str, body: Option<&Value>, anthropic: bool) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(path);
    if anthropic {
        b = b
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", "test");
    }
    match body {
        Some(v) => b
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

pub async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}

pub async fn post(
    app: &Router,
    path: &str,
    body: &Value,
    anthropic: bool,
) -> (StatusCode, HeaderMap, Value) {
    let (s, h, text) = send(app, request("POST", path, Some(body), anthropic)).await;
    let v = serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}"));
    (s, h, v)
}

pub async fn get(app: &Router, path: &str, anthropic: bool) -> (StatusCode, HeaderMap, Value) {
    let (s, h, text) = send(app, request("GET", path, None, anthropic)).await;
    (s, h, serde_json::from_str(&text).unwrap())
}

/// One server-sent event: the `event:` name (if any) and the joined `data:` lines.
#[derive(Debug, Clone)]
pub struct Sse {
    pub event: Option<String>,
    pub data: String,
}

impl Sse {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.data).unwrap_or_else(|e| panic!("bad JSON ({e}): {}", self.data))
    }
}

/// Parses an SSE body; comment-only frames (keep-alives) are dropped.
pub fn parse_sse(body: &str) -> Vec<Sse> {
    body.replace("\r\n", "\n")
        .split("\n\n")
        .filter_map(|frame| {
            let mut event = None;
            let mut data: Vec<&str> = Vec::new();
            for line in frame.lines() {
                if let Some(v) = line.strip_prefix("event:") {
                    event = Some(v.trim_start().to_owned());
                } else if let Some(v) = line.strip_prefix("data:") {
                    data.push(v.strip_prefix(' ').unwrap_or(v));
                }
            }
            (event.is_some() || !data.is_empty()).then(|| Sse {
                event,
                data: data.join("\n"),
            })
        })
        .collect()
}

pub async fn post_sse(
    app: &Router,
    path: &str,
    body: &Value,
    anthropic: bool,
) -> (StatusCode, HeaderMap, Vec<Sse>) {
    let (s, h, text) = send(app, request("POST", path, Some(body), anthropic)).await;
    (s, h, parse_sse(&text))
}

/// The ChatML prompt the fake template renders for one user message.
pub fn chatml_user(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn weather_tool_openai() -> Value {
    serde_json::json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }})
}

pub fn weather_tool_anthropic() -> Value {
    serde_json::json!({
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    })
}

pub const WEATHER_CALL: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

//! Request log (ADR 0046): one JSON line per generation or `count_tokens` request.
//!
//! An axum middleware reads the request body, wraps the response body to time the first
//! byte and capture the text, and writes the line when the body ends or is dropped (client
//! gone). Usage and cost come from the gateway's [`CostSink`], joined on the
//! `x-tetraeder-generation-id` response header. Only three request headers are logged;
//! keys and authorization never are.

use crate::gateway::{CostSink, GENERATION_ID_HEADER, RequestCost};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method};
use axum::middleware::Next;
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Paths that are logged (POST only).
const LOGGED: [&str; 4] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
];
/// The only request headers that reach the log.
const HEADERS: [&str; 3] = ["anthropic-version", "anthropic-beta", "user-agent"];
/// Costs waiting for their response body; a safety cap against leaks.
const MAX_PENDING: usize = 4096;

pub struct RequestLog {
    out: Mutex<BufWriter<std::fs::File>>,
    costs: Mutex<HashMap<String, RequestCost>>,
}

impl RequestLog {
    /// Opens `path` for appending, creating it.
    pub fn open(path: &Path) -> std::io::Result<Arc<Self>> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Arc::new(Self {
            out: Mutex::new(BufWriter::new(file)),
            costs: Mutex::default(),
        }))
    }

    fn take_cost(&self, id: &str) -> Option<RequestCost> {
        self.costs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
    }

    fn write(&self, line: &Value) {
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        if writeln!(out, "{line}").and_then(|()| out.flush()).is_err() {
            tracing::warn!("request log write failed");
        }
    }
}

impl CostSink for RequestLog {
    fn record(&self, cost: &RequestCost) {
        let mut m = self.costs.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() >= MAX_PENDING {
            m.clear();
        }
        m.insert(cost.request_id.clone(), cost.clone());
    }
}

fn json_or_text(b: &[u8]) -> Value {
    serde_json::from_slice(b).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(b).into()))
}

fn headers_json(h: &HeaderMap) -> Value {
    let mut m = serde_json::Map::new();
    for name in HEADERS {
        if let Some(v) = h.get(name).and_then(|v| v.to_str().ok()) {
            m.insert(name.to_owned(), Value::String(v.to_owned()));
        }
    }
    Value::Object(m)
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// Writes the line when dropped, so a client that disconnects mid-stream is logged too.
struct Pending {
    log: Arc<RequestLog>,
    ts_ms: u64,
    start: Instant,
    method: String,
    path: String,
    query: Option<String>,
    headers: Value,
    request: Value,
    stream: bool,
    status: u16,
    sse: bool,
    generation_id: Option<String>,
    first_byte: Option<f64>,
    response: Vec<u8>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        let total_ms = ms(self.start);
        let cost = self
            .generation_id
            .as_deref()
            .and_then(|id| self.log.take_cost(id));
        let response = if self.sse {
            Value::String(String::from_utf8_lossy(&self.response).into())
        } else {
            json_or_text(&self.response)
        };
        let error_type = cost
            .as_ref()
            .and_then(|c| c.error_type.map(str::to_owned))
            .or_else(|| response["error"]["type"].as_str().map(str::to_owned));
        let line = json!({
            "v": 1,
            "ts_ms": self.ts_ms,
            "method": self.method,
            "path": self.path,
            "query": self.query,
            "headers": self.headers,
            "status": self.status,
            "stream": self.stream,
            "request": self.request,
            "first_byte_ms": self.first_byte.unwrap_or(total_ms),
            "total_ms": total_ms,
            "generation_id": self.generation_id,
            "usage": cost.as_ref().map(|c| json!({
                "prompt_tokens": c.prompt_tokens,
                "cached_tokens": c.cached_tokens,
                "completion_tokens": c.completion_tokens,
            })),
            "cost": cost.as_ref().map(|c| json!({
                "gpu_seconds": c.gpu_seconds,
                "bytes_streamed": c.bytes_streamed,
            })),
            "error_type": error_type,
            "response": response,
        });
        self.log.write(&line);
    }
}

/// The middleware; add with `axum::middleware::from_fn_with_state`.
pub async fn log_requests(
    State(log): State<Arc<RequestLog>>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() != Method::POST || !LOGGED.contains(&req.uri().path()) {
        return next.run(req).await;
    }
    let start = Instant::now();
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let (parts, body) = req.into_parts();
    let bytes: Bytes = match axum::body::to_bytes(body, crate::server::BODY_LIMIT).await {
        Ok(b) => b,
        // Too large or broken: the handler's own body handling answers.
        Err(_) => return next.run(Request::from_parts(parts, Body::empty())).await,
    };
    let request = json_or_text(&bytes);
    let mut pending = Pending {
        log,
        ts_ms,
        start,
        method: parts.method.to_string(),
        path: parts.uri.path().to_owned(),
        query: parts.uri.query().map(str::to_owned),
        headers: headers_json(&parts.headers),
        stream: request["stream"].as_bool().unwrap_or(false),
        request,
        status: 0,
        sse: false,
        generation_id: None,
        first_byte: None,
        response: Vec::new(),
    };
    let resp = next
        .run(Request::from_parts(parts, Body::from(bytes)))
        .await;
    let (parts, body) = resp.into_parts();
    pending.status = parts.status.as_u16();
    pending.sse = parts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    pending.generation_id = parts
        .headers
        .get(GENERATION_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let stream = body.into_data_stream().map(move |chunk| {
        if let Ok(c) = &chunk {
            if pending.first_byte.is_none() && !c.is_empty() {
                pending.first_byte = Some(ms(pending.start));
            }
            pending.response.extend_from_slice(c);
        }
        chunk
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

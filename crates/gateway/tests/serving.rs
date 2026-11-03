//! ADR 0046: context clamp, default model, mid-conversation system messages, generation id
//! header and the request log.

mod common;

use axum::Router;
use common::*;
use serde_json::{Value, json};
use std::sync::Arc;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::reqlog::RequestLog;
use tetraeder_gateway::server::router_with;

fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

#[tokio::test]
async fn max_tokens_beyond_context_is_clamped() {
    let n = chatml_user_tokens("hi") as u32;
    let mut e = FakeEngine::fixed(&"a".repeat(500));
    e.set_context_length(n + 10);
    let (app, _) = app(e);
    for max in [100_000u32, u32::MAX] {
        let body = json!({"model": "fake", "messages": user("hi"), "max_tokens": max});
        let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["completion_tokens"], 10);
    }
    let body = json!({"model": "fake", "messages": user("hi"), "max_tokens": 0});
    let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 400);
}

#[tokio::test]
async fn prompt_filling_the_context_is_rejected() {
    let mut e = FakeEngine::fixed("x");
    e.set_context_length(chatml_user_tokens("hi") as u32);
    let (app, _) = app(e);
    let body = json!({"model": "fake", "messages": user("hi"), "max_tokens": 1});
    let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 400);
}

#[tokio::test]
async fn default_model_resolves_unknown_ids() {
    let cfg = GatewayConfig {
        default_model: Some("fake".into()),
        ..GatewayConfig::default()
    };
    let (app, _) = app_with(FakeEngine::fixed("ok"), cfg);
    let body = json!({"model": "claude-haiku-4-5", "max_tokens": 5, "messages": user("hi")});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "fake");
    let (s, _, v) = post(&app, "/v1/messages/count_tokens", &body, true).await;
    assert_eq!(s, 200, "{v}");
    let (s, _, v) = get(&app, "/v1/models/claude-sonnet-4-5", true).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["id"], "fake");
    let body = json!({"model": "gpt-4o", "messages": user("hi")});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "fake");
}

#[tokio::test]
async fn unknown_model_without_default_is_404() {
    let (app, _) = app(FakeEngine::fixed("ok"));
    let body = json!({"model": "claude-haiku-4-5", "max_tokens": 5, "messages": user("hi")});
    let (s, _, _) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 404);
}

#[tokio::test]
async fn system_message_in_messages_becomes_user_text() {
    let engine = FakeEngine::fixed("ok");
    let prompts = Arc::clone(&engine.prompts);
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "max_tokens": 5, "messages": [
        {"role": "user", "content": "one"},
        {"role": "system", "content": "be brief"},
        {"role": "assistant", "content": "two"},
        {"role": "system", "content": [{"type": "text", "text": "late"}]},
        {"role": "user", "content": "three"},
    ]});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    let p = prompts.lock().unwrap()[0].clone();
    assert!(
        p.contains("<|im_start|>user\none\n\n<system>\nbe brief\n</system><|im_end|>"),
        "{p}"
    );
    assert!(
        p.contains("<|im_start|>user\n<system>\nlate\n</system>\n\nthree<|im_end|>"),
        "{p}"
    );
}

#[tokio::test]
async fn generation_id_header_on_both_apis() {
    let (app, _) = app(FakeEngine::fixed("ok"));
    for stream in [false, true] {
        let body = json!({"model": "fake", "messages": user("hi"), "stream": stream});
        let req = request("POST", "/v1/chat/completions", Some(&body), false);
        let (_, h, _) = send(&app, req).await;
        assert!(h.contains_key("x-tetraeder-generation-id"));
        let body =
            json!({"model": "fake", "max_tokens": 5, "messages": user("hi"), "stream": stream});
        let (_, h, _) = send(&app, request("POST", "/v1/messages", Some(&body), true)).await;
        assert!(h.contains_key("x-tetraeder-generation-id"));
    }
}

fn logged_app(engine: FakeEngine) -> (Router, std::path::PathBuf) {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tetraeder-reqlog-{t}.jsonl"));
    let log = RequestLog::open(&path).unwrap();
    let gw = Gateway::new(Arc::new(engine), GatewayConfig::default()).with_cost_sink(log.clone());
    (router_with(Arc::new(gw), Some(log)), path)
}

#[tokio::test]
async fn request_log_has_one_line_per_request() {
    let (app, path) = logged_app(FakeEngine::fixed("hello"));
    // Not logged: GET.
    let _ = get(&app, "/v1/models", false).await;

    let body = json!({"model": "fake", "messages": user("hi"), "stream": false});
    let mut req = request("POST", "/v1/chat/completions?x=1", Some(&body), false);
    let h = req.headers_mut();
    h.insert("authorization", "Bearer sekrit-token".parse().unwrap());
    h.insert("user-agent", "tests/1".parse().unwrap());
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, 200);

    let body = json!({"model": "fake", "max_tokens": 5, "messages": user("hi"), "stream": true});
    let mut req = request("POST", "/v1/messages", Some(&body), true);
    req.headers_mut()
        .insert("x-api-key", "sekrit-key".parse().unwrap());
    let (s, _, sse) = send(&app, req).await;
    assert_eq!(s, 200);

    let body = json!({"model": "fake", "messages": user("hi"), "max_tokens": 1});
    let (s, _, _) = post(&app, "/v1/messages/count_tokens", &body, true).await;
    assert_eq!(s, 200);

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains("sekrit"), "{text}");
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3, "{text}");

    let a = &lines[0];
    assert_eq!(a["v"], 1);
    assert_eq!(a["method"], "POST");
    assert_eq!(a["path"], "/v1/chat/completions");
    assert_eq!(a["query"], "x=1");
    assert_eq!(a["headers"], json!({"user-agent": "tests/1"}));
    assert_eq!(a["status"], 200);
    assert_eq!(a["stream"], false);
    assert_eq!(a["request"]["model"], "fake");
    assert!(a["generation_id"].as_str().is_some());
    assert_eq!(a["usage"]["completion_tokens"], 5);
    assert_eq!(a["cost"]["bytes_streamed"], 1024);
    assert!(a["error_type"].is_null());
    assert_eq!(a["response"]["choices"][0]["message"]["content"], "hello");
    assert!(a["total_ms"].as_f64().unwrap() >= a["first_byte_ms"].as_f64().unwrap());
    assert!(a["ts_ms"].as_u64().unwrap() > 0);

    let b = &lines[1];
    assert_eq!(b["path"], "/v1/messages");
    assert_eq!(b["stream"], true);
    assert_eq!(b["headers"]["anthropic-version"], "2023-06-01");
    assert_eq!(b["response"].as_str().unwrap(), sse);
    assert_eq!(b["usage"]["completion_tokens"], 5);

    assert_eq!(lines[2]["path"], "/v1/messages/count_tokens");
    assert!(lines[2]["generation_id"].is_null());
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn request_log_records_errors() {
    let (app, path) = logged_app(FakeEngine::fixed("x"));
    let body = json!({"model": "nope", "messages": user("hi")});
    let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 404);
    let text = std::fs::read_to_string(&path).unwrap();
    let l: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(l["status"], 404);
    assert!(l["error_type"].is_string(), "{l}");
    let _ = std::fs::remove_file(path);
}

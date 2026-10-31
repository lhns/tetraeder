//! Anthropic Messages API conformance against platform.claude.com/docs/en/api/messages,
//! /build-with-claude/streaming, /api/errors, /api/messages/count_tokens and /api/models.

mod common;

use common::*;
use serde_json::{Value, json};
use tetraeder_gateway::fake::FakeEngine;

fn req(text: &str) -> Value {
    json!({"model": "fake", "max_tokens": 256, "messages": [{"role": "user", "content": text}]})
}

fn assert_error(v: &Value, ty: &str, headers: &axum::http::HeaderMap) {
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], ty, "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty())
    );
    let rid = v["request_id"].as_str().unwrap();
    assert!(rid.starts_with("req_"));
    assert_eq!(headers["request-id"], rid);
}

/// Event names must equal the `type` of their data.
fn typed(ev: &[Sse]) -> Vec<(String, Value)> {
    ev.iter()
        .map(|e| {
            let v = e.json();
            let name = e.event.clone().expect("Anthropic events are named");
            assert_eq!(v["type"], name.as_str(), "{v}");
            (name, v)
        })
        .collect()
}

#[tokio::test]
async fn message_shape() {
    let (app, _) = app(FakeEngine::fixed("Hello!"));
    let (s, h, v) = post(&app, "/v1/messages", &req("hi"), true).await;
    assert_eq!(s, 200, "{v}");
    assert!(h["request-id"].to_str().unwrap().starts_with("req_"));
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "fake");
    assert_eq!(v["content"], json!([{"type": "text", "text": "Hello!"}]));
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["stop_sequence"].is_null());
    assert_eq!(v["usage"]["input_tokens"], chatml_user_tokens("hi") as u64);
    assert_eq!(v["usage"]["output_tokens"], 6);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 0);
}

#[tokio::test]
async fn system_and_blocks_reach_the_template() {
    let engine = FakeEngine::fixed("ok");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "max_tokens": 10,
        "system": [{"type": "text", "text": "Be brief."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]}]});
    let (s, _, _) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200);
    assert_eq!(
        prompts.lock().unwrap()[0],
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn stream_event_sequence() {
    let (app, _) = app(FakeEngine::fixed("Hi you"));
    let mut body = req("hi");
    body["stream"] = json!(true);
    let (s, h, ev) = post_sse(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200);
    assert!(
        h["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let ev = typed(&ev);
    let names: Vec<&str> = ev.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names[..3], ["message_start", "ping", "content_block_start"]);
    assert_eq!(
        names[names.len() - 3..],
        ["content_block_stop", "message_delta", "message_stop"]
    );
    assert!(
        names[3..names.len() - 3]
            .iter()
            .all(|n| *n == "content_block_delta")
    );

    let m = &ev[0].1["message"];
    assert!(m["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(m["type"], "message");
    assert_eq!(m["role"], "assistant");
    assert_eq!(m["content"], json!([]));
    assert!(m["stop_reason"].is_null() && m["stop_sequence"].is_null());
    assert!(m["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        ev[2].1,
        json!({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}})
    );
    let text: String = ev[3..ev.len() - 3]
        .iter()
        .map(|(_, v)| {
            assert_eq!(v["index"], 0);
            assert_eq!(v["delta"]["type"], "text_delta");
            v["delta"]["text"].as_str().unwrap().to_owned()
        })
        .collect();
    assert_eq!(text, "Hi you");
    assert_eq!(
        ev[ev.len() - 3].1,
        json!({"type": "content_block_stop", "index": 0})
    );
    let d = &ev[ev.len() - 2].1;
    assert_eq!(
        d["delta"],
        json!({"stop_reason": "end_turn", "stop_sequence": null})
    );
    assert_eq!(d["usage"]["output_tokens"], 6);
    assert_eq!(ev[ev.len() - 1].1, json!({"type": "message_stop"}));
}

#[tokio::test]
async fn tool_use_message() {
    let (app, _) = app(FakeEngine::fixed(&format!("Checking.{WEATHER_CALL}")));
    let mut body = req("weather?");
    body["tools"] = json!([weather_tool_anthropic()]);
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["stop_reason"], "tool_use");
    assert_eq!(
        v["content"][0],
        json!({"type": "text", "text": "Checking."})
    );
    let t = &v["content"][1];
    assert_eq!(t["type"], "tool_use");
    assert!(t["id"].as_str().unwrap().starts_with("toolu_"));
    assert_eq!(t["name"], "get_weather");
    assert_eq!(t["input"], json!({"city": "Paris"}));
}

#[tokio::test]
async fn tool_use_stream() {
    let (app, _) = app(FakeEngine::fixed(&format!("Checking.{WEATHER_CALL}")));
    let mut body = req("weather?");
    body["tools"] = json!([weather_tool_anthropic()]);
    body["stream"] = json!(true);
    let (_, _, ev) = post_sse(&app, "/v1/messages", &body, true).await;
    let ev = typed(&ev);
    let starts: Vec<&Value> = ev
        .iter()
        .filter(|(n, _)| n == "content_block_start")
        .map(|(_, v)| v)
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0]["index"], 0);
    assert_eq!(starts[1]["index"], 1);
    let tb = &starts[1]["content_block"];
    assert_eq!(tb["type"], "tool_use");
    assert!(tb["id"].as_str().unwrap().starts_with("toolu_"));
    assert_eq!(tb["name"], "get_weather");
    assert_eq!(tb["input"], json!({}));
    let partial: String = ev
        .iter()
        .filter(|(n, v)| n == "content_block_delta" && v["index"] == 1)
        .map(|(_, v)| {
            assert_eq!(v["delta"]["type"], "input_json_delta");
            v["delta"]["partial_json"].as_str().unwrap().to_owned()
        })
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&partial).unwrap(),
        json!({"city": "Paris"})
    );
    let stops: Vec<u64> = ev
        .iter()
        .filter(|(n, _)| n == "content_block_stop")
        .map(|(_, v)| v["index"].as_u64().unwrap())
        .collect();
    assert_eq!(stops, [0, 1]);
    let d = ev.iter().find(|(n, _)| n == "message_delta").unwrap();
    assert_eq!(d.1["delta"]["stop_reason"], "tool_use");
}

#[tokio::test]
async fn tool_result_history_reaches_the_template() {
    let engine = FakeEngine::fixed("Sunny.");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "max_tokens": 64, "tools": [weather_tool_anthropic()], "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "sunny"}]}]},
    ]});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    let p = prompts.lock().unwrap()[0].clone();
    assert!(p.contains("Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"), "{p}");
    assert!(
        p.contains("<tool_response>\nsunny\n</tool_response>"),
        "{p}"
    );
}

#[tokio::test]
async fn thinking_blocks() {
    let script = |_: &str| "<think>\nponder\n</think>\n\nDone.".to_owned();
    let (app, _) = app(FakeEngine::with_template(Some(QWEN3.into()), script));
    let (s, _, v) = post(&app, "/v1/messages", &req("q"), true).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        v["content"][0],
        json!({"type": "thinking", "thinking": "ponder", "signature": ""})
    );
    assert_eq!(v["content"][1]["type"], "text");
    assert_eq!(v["content"][1]["text"].as_str().unwrap().trim(), "Done.");

    let mut body = req("q");
    body["stream"] = json!(true);
    let (_, _, ev) = post_sse(&app, "/v1/messages", &body, true).await;
    let ev = typed(&ev);
    let start = ev.iter().find(|(n, _)| n == "content_block_start").unwrap();
    assert_eq!(
        start.1["content_block"],
        json!({"type": "thinking", "thinking": "", "signature": ""})
    );
    let kinds: Vec<&str> = ev
        .iter()
        .filter(|(n, v)| n == "content_block_delta" && v["index"] == 0)
        .map(|(_, v)| v["delta"]["type"].as_str().unwrap())
        .collect();
    assert!(
        kinds
            .iter()
            .all(|k| *k == "thinking_delta" || *k == "signature_delta")
    );
    assert_eq!(kinds.last(), Some(&"signature_delta"));
}

#[tokio::test]
async fn thinking_disabled_reaches_the_template() {
    let engine = FakeEngine::with_template(Some(QWEN3.into()), |_| "ok".into());
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let mut body = req("q");
    body["thinking"] = json!({"type": "disabled"});
    let (s, _, _) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200);
    assert!(prompts.lock().unwrap()[0].ends_with("<think>\n\n</think>\n\n"));
}

#[tokio::test]
async fn stop_sequence_and_max_tokens() {
    let (app, _) = app(FakeEngine::fixed("one two STOP three"));
    let mut body = req("hi");
    body["stop_sequences"] = json!(["STOP"]);
    let (_, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(v["content"][0]["text"], "one two ");
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "STOP");

    let mut body = req("hi");
    body["max_tokens"] = json!(3);
    let (_, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(v["content"][0]["text"], "one");
    assert_eq!(v["stop_reason"], "max_tokens");
}

#[tokio::test]
async fn output_config_format_is_enforced() {
    let (app, _) = app(FakeEngine::fixed("Result: {\"ok\": true}"));
    let mut body = req("json");
    body["output_config"] = json!({"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
        "additionalProperties": false}}});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    let out: Value = serde_json::from_str(v["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(out, json!({"ok": true}));
}

#[tokio::test]
async fn forced_tool_choice() {
    let (app, _) = app(FakeEngine::fixed(&format!(
        "I will call it. {WEATHER_CALL}"
    )));
    let mut body = req("weather?");
    body["tools"] = json!([weather_tool_anthropic()]);
    body["tool_choice"] = json!({"type": "tool", "name": "get_weather"});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["stop_reason"], "tool_use");
    let blocks = v["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 1, "{v}");
    assert_eq!(blocks[0]["name"], "get_weather");
}

#[tokio::test]
async fn errors_use_the_anthropic_envelope() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let mut no_max = req("hi");
    no_max.as_object_mut().unwrap().remove("max_tokens");
    let mut unknown = req("hi");
    unknown["model"] = json!("nope");
    let image = json!({"model": "fake", "max_tokens": 5, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]});
    let prefill = json!({"model": "fake", "max_tokens": 5, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure"}]});
    let empty = json!({"model": "fake", "max_tokens": 5, "messages": []});
    for (body, status, ty) in [
        (no_max, 400, "invalid_request_error"),
        (unknown, 404, "not_found_error"),
        (image, 400, "invalid_request_error"),
        (prefill, 400, "invalid_request_error"),
        (empty, 400, "invalid_request_error"),
    ] {
        let (s, h, v) = post(&app, "/v1/messages", &body, true).await;
        assert_eq!(s, status, "{body} -> {v}");
        assert_error(&v, ty, &h);
    }
    let (s, h, text) = send(&app, request("POST", "/v1/messages", None, true)).await;
    assert_eq!(s, 400);
    assert_error(
        &serde_json::from_str(&text).unwrap(),
        "invalid_request_error",
        &h,
    );
}

#[tokio::test]
async fn mid_stream_error_event() {
    let mut engine = FakeEngine::fixed("abcdef");
    engine.fail_after = Some(2);
    let (app, _) = app(engine);
    let mut body = req("hi");
    body["stream"] = json!(true);
    let (_, _, ev) = post_sse(&app, "/v1/messages", &body, true).await;
    let ev = typed(&ev);
    let (name, v) = ev.last().unwrap();
    assert_eq!(name, "error");
    assert_eq!(v["error"]["type"], "api_error");
    assert!(v["error"]["message"].is_string());
}

#[tokio::test]
async fn count_tokens_matches_usage() {
    let (app, _) = app(FakeEngine::fixed("ok"));
    let mut body = req("count me");
    body.as_object_mut().unwrap().remove("max_tokens");
    let (s, h, v) = post(&app, "/v1/messages/count_tokens", &body, true).await;
    assert_eq!(s, 200, "{v}");
    assert!(h.contains_key("request-id"));
    assert_eq!(v, json!({"input_tokens": chatml_user_tokens("count me")}));
}

#[tokio::test]
async fn models_in_anthropic_shape() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = get(&app, "/v1/models", true).await;
    assert_eq!(s, 200);
    let m = &v["data"][0];
    assert_eq!(m["type"], "model");
    assert_eq!(m["id"], "fake");
    assert_eq!(m["display_name"], "fake");
    assert_eq!(m["created_at"], "2023-11-14T22:13:20Z");
    assert_eq!(v["has_more"], false);
    assert_eq!(v["first_id"], "fake");
    assert_eq!(v["last_id"], "fake");
    let (s, _, one) = get(&app, "/v1/models/fake", true).await;
    assert_eq!(s, 200);
    assert_eq!(&one, m);
    let (s, h, e) = get(&app, "/v1/models/nope", true).await;
    assert_eq!(s, 404);
    assert_error(&e, "not_found_error", &h);
}

#[tokio::test]
async fn text_before_tool_result_keeps_its_order() {
    let engine = FakeEngine::fixed("ok");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "max_tokens": 64, "tools": [weather_tool_anthropic()], "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [
            {"type": "text", "text": "BEFORE-RESULT"},
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]},
    ]});
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{v}");
    let p = prompts.lock().unwrap()[0].clone();
    let (a, b) = (
        p.find("BEFORE-RESULT").unwrap(),
        p.find("<tool_response>").unwrap(),
    );
    assert!(a < b, "{p}");
}

#[tokio::test]
async fn zero_max_tokens_is_rejected() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let mut body = req("hi");
    body["max_tokens"] = json!(0);
    let (s, h, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 400, "{v}");
    assert_error(&v, "invalid_request_error", &h);
}

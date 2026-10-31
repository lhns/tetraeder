//! OpenAI API conformance against the official OpenAPI spec (openai/openai-openapi):
//! CreateChatCompletionResponse, CreateChatCompletionStreamResponse, CreateCompletionResponse,
//! ListModelsResponse, Model, ErrorResponse.

mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use tetraeder_gateway::fake::FakeEngine;

fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

fn assert_error(v: &Value, ty: &str) {
    let e = &v["error"];
    assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{v}");
    assert_eq!(e["type"], ty, "{v}");
    assert!(e.get("param").is_some() && e.get("code").is_some(), "{v}");
}

#[tokio::test]
async fn chat_completion_shape() {
    let (app, _) = app(FakeEngine::fixed("Hello there"));
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("hi")}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["object"], "chat.completion");
    assert!(v["created"].is_u64());
    assert_eq!(v["model"], "fake");
    let c = &v["choices"][0];
    assert_eq!(c["index"], 0);
    assert_eq!(c["message"]["role"], "assistant");
    assert_eq!(c["message"]["content"], "Hello there");
    assert!(c["message"]["refusal"].is_null());
    assert!(c.get("logprobs").is_some_and(Value::is_null));
    assert_eq!(c["finish_reason"], "stop");
    let u = &v["usage"];
    let prompt = chatml_user_tokens("hi") as u64;
    assert_eq!(u["prompt_tokens"], prompt);
    assert_eq!(u["completion_tokens"], 11);
    assert_eq!(u["total_tokens"], prompt + 11);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn chat_prompt_uses_the_template() {
    let engine = FakeEngine::fixed("ok");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]});
    let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200);
    assert_eq!(
        prompts.lock().unwrap()[0],
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nab<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn chat_stream_sequence() {
    let (app, _) = app(FakeEngine::fixed("Hi you"));
    let body = json!({"model": "fake", "messages": user("hi"), "stream": true});
    let (s, h, ev) = post_sse(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200);
    assert!(
        h["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    assert!(
        ev.iter().all(|e| e.event.is_none()),
        "OpenAI streams have no event names"
    );
    assert_eq!(ev.last().unwrap().data, "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(Sse::json).collect();
    let id = chunks[0]["id"].clone();
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], id);
        assert_eq!(c["created"], chunks[0]["created"]);
        assert_eq!(c["model"], "fake");
        assert!(
            c.get("usage").is_none(),
            "no usage without include_usage: {c}"
        );
    }
    assert_eq!(
        chunks[0]["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    let text: String = chunks[1..chunks.len() - 1]
        .iter()
        .map(|c| {
            assert!(c["choices"][0]["finish_reason"].is_null());
            c["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(text, "Hi you");
    let last = &chunks[chunks.len() - 1]["choices"][0];
    assert_eq!(last["delta"], json!({}));
    assert_eq!(last["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_stream_include_usage() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let body = json!({"model": "fake", "messages": user("hi"), "stream": true,
                      "stream_options": {"include_usage": true}});
    let (_, _, ev) = post_sse(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(ev.last().unwrap().data, "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(Sse::json).collect();
    let (last, rest) = chunks.split_last().unwrap();
    for c in rest {
        assert!(c.get("usage").is_some_and(Value::is_null), "{c}");
    }
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["completion_tokens"], 3);
    assert_eq!(
        last["usage"]["total_tokens"],
        last["usage"]["prompt_tokens"].as_u64().unwrap() + 3
    );
}

#[tokio::test]
async fn tool_call_response() {
    let (app, _) = app(FakeEngine::fixed(WEATHER_CALL));
    let body = json!({"model": "fake", "messages": user("weather in Paris?"),
                      "tools": [weather_tool_openai()]});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    assert!(c["message"]["content"].is_null());
    let tc = &c["message"]["tool_calls"][0];
    assert!(tc["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris"}));
}

#[tokio::test]
async fn tool_call_stream() {
    let (app, _) = app(FakeEngine::fixed(&format!("Checking.{WEATHER_CALL}")));
    let body = json!({"model": "fake", "messages": user("weather?"), "stream": true,
                      "tools": [weather_tool_openai()]});
    let (_, _, ev) = post_sse(&app, "/v1/chat/completions", &body, false).await;
    let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(Sse::json).collect();
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "Checking.");
    let calls: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"].get("tool_calls"))
        .collect();
    assert_eq!(calls.len(), 1);
    let tc = &calls[0][0];
    assert_eq!(tc["index"], 0);
    assert!(tc["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris"}));
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[tokio::test]
async fn tool_history_reaches_the_template() {
    let engine = FakeEngine::fixed("It is sunny.");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "tools": [weather_tool_openai()], "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let p = prompts.lock().unwrap()[0].clone();
    assert!(p.contains("\"name\": \"get_weather\""), "{p}");
    assert!(
        p.contains(r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#),
        "{p}"
    );
    assert!(
        p.contains("<tool_response>\nsunny\n</tool_response>"),
        "{p}"
    );
}

#[tokio::test]
async fn stop_sequence_ends_generation() {
    let engine = FakeEngine::fixed("Hello END world");
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "messages": user("hi"), "stop": "END"});
    let (_, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn max_tokens_gives_length() {
    let (app, _) = app(FakeEngine::fixed("abcdefgh"));
    for field in ["max_tokens", "max_completion_tokens"] {
        let body = json!({"model": "fake", "messages": user("hi"), field: 3});
        let (_, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
        assert_eq!(v["choices"][0]["message"]["content"], "abc");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["completion_tokens"], 3);
    }
}

#[tokio::test]
async fn reasoning_content_is_split() {
    let (app, _) = app(FakeEngine::with_template(Some(QWEN3.into()), |_| {
        "<think>\nhmm\n</think>\n\nAnswer".into()
    }));
    let body = json!({"model": "fake", "messages": user("q")});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "hmm");
    assert_eq!(m["content"].as_str().unwrap().trim(), "Answer");
}

#[tokio::test]
async fn json_schema_response_format_is_enforced() {
    let (app, _) = app(FakeEngine::fixed("Sure! {\"a\": 1}"));
    let body = json!({"model": "fake", "messages": user("json please"), "response_format": {
        "type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {
            "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
            "additionalProperties": false}}}});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let out: Value =
        serde_json::from_str(v["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert_eq!(out, json!({"a": 1}));
}

#[tokio::test]
async fn json_object_response_format() {
    let (app, _) = app(FakeEngine::fixed("Here: {\"k\": \"v\"}"));
    let body = json!({"model": "fake", "messages": user("json"), "response_format": {"type": "json_object"}});
    let (_, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    let out: Value =
        serde_json::from_str(v["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert_eq!(out, json!({"k": "v"}));
}

#[tokio::test]
async fn required_tool_choice_forces_a_call() {
    let (app, _) = app(FakeEngine::fixed(&format!(
        "Let me look that up. {WEATHER_CALL}"
    )));
    let body = json!({"model": "fake", "messages": user("weather?"), "tools": [weather_tool_openai()],
                      "tool_choice": "required"});
    let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    assert_eq!(
        c["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert!(c["message"]["content"].is_null(), "{v}");
}

#[tokio::test]
async fn errors_use_the_openai_shape() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let cases = [
        (
            json!({"model": "nope", "messages": user("hi")}),
            404,
            "invalid_request_error",
        ),
        (
            json!({"model": "fake", "messages": []}),
            400,
            "invalid_request_error",
        ),
        (json!({"model": "fake"}), 400, "invalid_request_error"),
        (
            json!({"model": "fake", "messages": user("hi"), "n": 2}),
            400,
            "invalid_request_error",
        ),
        (
            json!({"model": "fake", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]}),
            400,
            "invalid_request_error",
        ),
        (
            json!({"model": "fake", "messages": user("hi"), "max_tokens": 100000}),
            400,
            "invalid_request_error",
        ),
        (
            // u32::MAX must not overflow the context check.
            json!({"model": "fake", "messages": user("hi"), "max_tokens": 4294967295u32}),
            400,
            "invalid_request_error",
        ),
        (
            json!({"model": "fake", "messages": user("hi"), "tool_choice": "required"}),
            400,
            "invalid_request_error",
        ),
    ];
    for (body, status, ty) in cases {
        let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
        assert_eq!(s, status, "{body} -> {v}");
        assert_error(&v, ty);
    }
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "nope", "messages": user("hi")}),
        false,
    )
    .await;
    assert_eq!(
        (s.as_u16(), v["error"]["code"].clone()),
        (404, json!("model_not_found"))
    );
    let (s, _, text) = send(&app, request("POST", "/v1/chat/completions", None, false)).await;
    assert_eq!(s, 400);
    assert_error(
        &serde_json::from_str(&text).unwrap(),
        "invalid_request_error",
    );
}

#[tokio::test]
async fn models_endpoints() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = get(&app, "/v1/models", false).await;
    assert_eq!(s, 200);
    assert_eq!(v["object"], "list");
    assert_eq!(
        v["data"][0],
        json!({"id": "fake", "object": "model", "created": 1_700_000_000u64, "owned_by": "tetraeder"})
    );
    let (s, _, m) = get(&app, "/v1/models/fake", false).await;
    assert_eq!(s, 200);
    assert_eq!(m, v["data"][0]);
    let (s, _, e) = get(&app, "/v1/models/nope", false).await;
    assert_eq!(s, 404);
    assert_error(&e, "invalid_request_error");
}

#[tokio::test]
async fn completions_shape_and_default_max_tokens() {
    let engine = FakeEngine::fixed("0123456789abcdefghijXYZ");
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "Count:"}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(v["object"], "text_completion");
    assert!(v["created"].is_u64());
    let c = &v["choices"][0];
    assert_eq!(c["text"], "0123456789abcdef", "default max_tokens is 16");
    assert_eq!(c["index"], 0);
    assert!(c["logprobs"].is_null());
    assert_eq!(c["finish_reason"], "length");
    assert_eq!(v["usage"]["prompt_tokens"], 6);
    assert_eq!(v["usage"]["completion_tokens"], 16);
    assert_eq!(
        prompts.lock().unwrap()[0],
        "Count:",
        "no template on completions"
    );

    let body = json!({"model": "fake", "prompt": ["P"], "max_tokens": 2, "echo": true});
    let (_, _, v) = post(&app, "/v1/completions", &body, false).await;
    assert_eq!(v["choices"][0]["text"], "P01");
}

#[tokio::test]
async fn completions_stream() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let body = json!({"model": "fake", "prompt": "x", "stream": true, "max_tokens": 10,
                      "stream_options": {"include_usage": true}});
    let (_, _, ev) = post_sse(&app, "/v1/completions", &body, false).await;
    assert_eq!(ev.last().unwrap().data, "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(Sse::json).collect();
    let (usage, rest) = chunks.split_last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 3);
    for c in rest {
        assert_eq!(c["object"], "text_completion");
    }
    let text: String = rest
        .iter()
        .map(|c| c["choices"][0]["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "abc");
    assert_eq!(rest.last().unwrap()["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn mid_stream_error_is_a_data_frame() {
    let mut engine = FakeEngine::fixed("abcdef");
    engine.fail_after = Some(2);
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "messages": user("hi"), "stream": true});
    let (s, _, ev) = post_sse(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 200);
    let last = ev.last().unwrap();
    assert_ne!(last.data, "[DONE]");
    assert_error(&last.json(), "server_error");
}

#[tokio::test]
async fn stop_cancels_the_engine() {
    let engine = FakeEngine::fixed("Hello END and much more text after the stop");
    let cancelled = engine.cancelled.clone();
    let (app, _) = app(engine);
    let body = json!({"model": "fake", "messages": user("hi"), "stop": ["END"]});
    let (_, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello ");
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn zero_max_tokens_is_rejected() {
    let (app, _) = app(FakeEngine::fixed("x"));
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model": "fake", "messages": user("hi"), "max_tokens": 0}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "fake", "messages": user("hi"), "max_completion_tokens": 0}),
        ),
        (
            "/v1/completions",
            json!({"model": "fake", "prompt": "hi", "max_tokens": 0}),
        ),
    ] {
        let (s, _, v) = post(&app, path, &body, false).await;
        assert_eq!(s, 400, "{body} -> {v}");
        assert_error(&v, "invalid_request_error");
    }
}

//! Endpoints of llama-server's group C (docs/notes/llamacpp-parity.md, ADR 0083 and 0089):
//! health, auth, metrics, props, tokenize, detokenize, apply-template, native completion,
//! Responses, infill and the unprefixed aliases. Fake engine, no model.

mod common;

use axum::body::Body;
use axum::http::Request;
use common::*;
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::server::router;

fn keyed(keys: &[&str]) -> GatewayConfig {
    GatewayConfig {
        api_keys: keys.iter().map(|k| (*k).to_owned()).collect(),
        ..GatewayConfig::default()
    }
}

fn with_header(method: &str, path: &str, body: Option<&Value>, h: (&str, &str)) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(path).header(h.0, h.1);
    if body.is_some() {
        b = b.header("content-type", "application/json");
    }
    b.body(body.map_or_else(Body::empty, |v| Body::from(v.to_string())))
        .unwrap()
}

fn chat_body() -> Value {
    json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
}

// ---- health and auth ----

#[tokio::test]
async fn health_answers_in_llama_servers_shape() {
    let (app, _) = app(FakeEngine::fixed("x"));
    for path in ["/health", "/v1/health"] {
        let (s, _, v) = get(&app, path, false).await;
        assert_eq!(s, 200, "{path}");
        assert_eq!(v, json!({"status": "ok"}));
    }
}

#[tokio::test]
async fn health_is_503_while_the_engine_is_not_ready() {
    let mut e = FakeEngine::fixed("x");
    e.ready = false;
    let (app, _) = app(e);
    let (s, _, v) = get(&app, "/health", false).await;
    assert_eq!(s, 503);
    assert_eq!(v["error"]["type"], "unavailable_error");
    assert_eq!(v["error"]["message"], "Loading model");
    assert_eq!(v["error"]["code"], 503);
}

#[tokio::test]
async fn api_key_guards_every_route_but_health() {
    let (app, _) = app_with(FakeEngine::fixed("abc"), keyed(&["k1", "k2"]));
    // no key, wrong key, unknown path, GET routes, native routes
    for (m, p, b) in [
        ("POST", "/v1/chat/completions", Some(chat_body())),
        ("POST", "/chat/completions", Some(chat_body())),
        ("GET", "/models", None),
        ("GET", "/v1/models", None),
        ("GET", "/props", None),
        ("GET", "/metrics", None),
        ("POST", "/tokenize", Some(json!({"content": "a"}))),
        ("POST", "/nope", None),
    ] {
        let (s, _, text) = send(&app, request(m, p, b.as_ref(), false)).await;
        assert_eq!(s, 401, "{m} {p}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v,
            json!({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}})
        );
    }
    let (s, _, _) = send(
        &app,
        with_header("GET", "/models", None, ("authorization", "Bearer nope")),
    )
    .await;
    assert_eq!(s, 401);
    // Public: no key needed.
    let (s, _, _) = send(&app, request("GET", "/health", None, false)).await;
    assert_eq!(s, 200);
    let (s, _, _) = send(&app, request("GET", "/v1/health", None, false)).await;
    assert_eq!(s, 200);
}

#[tokio::test]
async fn api_key_is_taken_from_authorization_or_x_api_key() {
    let (app, _) = app_with(FakeEngine::fixed("abc"), keyed(&["k1", "k2"]));
    let body = chat_body();
    for h in [
        ("authorization", "Bearer k1"),
        ("authorization", "Bearer k2"),
        // the prefix is optional
        ("authorization", "k1"),
        ("x-api-key", "k2"),
    ] {
        let (s, _, text) = send(
            &app,
            with_header("POST", "/v1/chat/completions", Some(&body), h),
        )
        .await;
        assert_eq!(s, 200, "{h:?}: {text}");
    }
    // x-api-key is a fallback only when Authorization is empty
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer wrong")
        .header("x-api-key", "k1")
        .body(Body::from(body.to_string()))
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, 401);
    // the Anthropic route accepts the key in x-api-key
    let (s, _, _) = send(
        &app,
        with_header(
            "POST",
            "/v1/messages",
            Some(&json!({"model": "fake", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})),
            ("x-api-key", "k1"),
        ),
    )
    .await;
    // anthropic-version is missing, which the route does not need
    assert_eq!(s, 200);
}

// ---- aliases ----

#[tokio::test]
async fn unprefixed_models_and_chat_completions() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let (s, _, v) = get(&app, "/models", false).await;
    assert_eq!(s, 200);
    assert_eq!(v["data"][0]["id"], "fake");
    let (s, _, v) = post(&app, "/chat/completions", &chat_body(), false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "abc");
}

// ---- tokenize and detokenize ----

#[tokio::test]
async fn tokenize_defaults_and_options() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = post(&app, "/tokenize", &json!({"content": "hi"}), false).await;
    assert_eq!(s, 200);
    assert_eq!(v, json!({"tokens": [104, 105]}));
    // special tokens are parsed by default
    let (_, _, v) = post(&app, "/tokenize", &json!({"content": "<|im_end|>"}), false).await;
    assert_eq!(v, json!({"tokens": [256]}));
    let body = json!({"content": "<|im_end|>", "parse_special": false});
    let (_, _, v) = post(&app, "/tokenize", &body, false).await;
    assert_eq!(v["tokens"].as_array().unwrap().len(), 10);
    // mixed array: strings and token ids
    let (_, _, v) = post(
        &app,
        "/tokenize",
        &json!({"content": ["a", 98, "c"]}),
        false,
    )
    .await;
    assert_eq!(v, json!({"tokens": [97, 98, 99]}));
    // no content: an empty list, as llama-server
    let (s, _, v) = post(&app, "/tokenize", &json!({}), false).await;
    assert_eq!(s, 200);
    assert_eq!(v, json!({"tokens": []}));
}

#[tokio::test]
async fn tokenize_with_pieces_keeps_bytes_that_are_not_utf8() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let body = json!({"content": "h\u{e9}", "with_pieces": true});
    let (_, _, v) = post(&app, "/tokenize", &body, false).await;
    // h, then the two bytes of U+00E9 as separate tokens
    assert_eq!(
        v,
        json!({"tokens": [{"id": 104, "piece": "h"}, {"id": 195, "piece": [195]}, {"id": 169, "piece": [169]}]})
    );
}

#[tokio::test]
async fn detokenize_joins_pieces() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = post(&app, "/detokenize", &json!({"tokens": [104, 105]}), false).await;
    assert_eq!(s, 200);
    assert_eq!(v, json!({"content": "hi"}));
    // control tokens are rendered
    let (_, _, v) = post(&app, "/detokenize", &json!({"tokens": [256, 104]}), false).await;
    assert_eq!(v, json!({"content": "<|im_end|>h"}));
    let (_, _, v) = post(&app, "/detokenize", &json!({}), false).await;
    assert_eq!(v, json!({"content": ""}));
    // not an array of numbers: 400 in llama-server's error object
    let (s, _, v) = post(&app, "/detokenize", &json!({"tokens": "x"}), false).await;
    assert_eq!(s, 400);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], 400);
}

// ---- apply-template ----

#[tokio::test]
async fn apply_template_renders_the_chat_prompt() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let body = json!({"messages": [{"role": "user", "content": "hi"}]});
    let (s, _, v) = post(&app, "/apply-template", &body, false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v, json!({"prompt": chatml_user("hi")}));
    let (s, _, v) = post(&app, "/apply-template", &json!({}), false).await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

// ---- native completion ----

#[tokio::test]
async fn native_completion_response_fields() {
    let e = FakeEngine::fixed("abc");
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    let body = json!({"prompt": "hi", "n_predict": 8, "temperature": 0});
    for path in ["/completion", "/completions"] {
        let (s, h, v) = post(&app, path, &body, false).await;
        assert_eq!(s, 200, "{v}");
        assert!(h.contains_key("x-tetraeder-generation-id"));
        assert_eq!(v["content"], "abc");
        assert_eq!(v["stop"], true);
        assert_eq!(v["stop_type"], "eos");
        assert_eq!(v["stopping_word"], "");
        assert_eq!(v["model"], "fake");
        assert_eq!(v["index"], 0);
        assert_eq!(v["tokens"], json!([]));
        // a, b, c and the end-of-generation token
        assert_eq!(v["tokens_predicted"], 4);
        assert_eq!(v["tokens_evaluated"], 2);
        assert_eq!(v["prompt"], "hi");
        assert_eq!(v["has_new_line"], false);
        assert_eq!(v["truncated"], false);
        assert!(v["tokens_cached"].is_number());
        assert!(v["timings"]["prompt_n"].is_number());
        assert_eq!(v["generation_settings"]["n_predict"], 8);
        assert_eq!(v["generation_settings"]["max_tokens"], 8);
        assert_eq!(v["generation_settings"]["temperature"], 0.0);
        assert_eq!(v["generation_settings"]["stream"], false);
    }
    assert_eq!(prompts.lock().unwrap().as_slice(), ["hi", "hi"]);
}

#[tokio::test]
async fn native_completion_limit_and_stop_word() {
    let (app, _) = app(FakeEngine::fixed("abcdef\n"));
    let (_, _, v) = post(
        &app,
        "/completion",
        &json!({"prompt": "x", "n_predict": 3}),
        false,
    )
    .await;
    assert_eq!(v["content"], "abc");
    assert_eq!(v["stop_type"], "limit");
    assert_eq!(v["tokens_predicted"], 3);
    let (_, _, v) = post(
        &app,
        "/completion",
        &json!({"prompt": "x", "stop": ["d"]}),
        false,
    )
    .await;
    assert_eq!(v["content"], "abc");
    assert_eq!(v["stop_type"], "word");
    assert_eq!(v["stopping_word"], "d");
    assert_eq!(v["generation_settings"]["stop"], json!(["d"]));
    let (_, _, v) = post(&app, "/completion", &json!({"prompt": "x"}), false).await;
    assert_eq!(v["has_new_line"], true);
    // n_predict -1: up to the end of the context
    let (s, _, v) = post(
        &app,
        "/completion",
        &json!({"prompt": "x", "n_predict": -1}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["stop_type"], "eos");
}

#[tokio::test]
async fn native_completion_prompt_forms() {
    let e = FakeEngine::fixed("a");
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    // token ids, and a mix of text and ids
    for p in [json!([104, 105]), json!(["h", 105]), json!("hi")] {
        let (s, _, v) = post(
            &app,
            "/completion",
            &json!({"prompt": p, "n_predict": 1}),
            false,
        )
        .await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["tokens_evaluated"], 2);
    }
    assert_eq!(prompts.lock().unwrap().as_slice(), ["hi", "hi", "hi"]);
    // a missing prompt and a wrong type are 400 in llama-server's error object
    for b in [json!({}), json!({"prompt": 5}), json!({"prompt": {"a": 1}})] {
        let (s, _, v) = post(&app, "/completion", &b, false).await;
        assert_eq!(s, 400, "{b}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], 400);
    }
}

#[tokio::test]
async fn native_completion_rejects_what_the_chat_routes_reject() {
    let (app, _) = app(FakeEngine::fixed("a"));
    let body = json!({"prompt": "x", "grammar": "root ::= BAD"});
    let (s, _, v) = post(&app, "/completion", &body, false).await;
    assert_eq!(s, 400, "{v}");
    let body = json!({"prompt": "x", "lora": [{"id": 0, "scale": 1.0}]});
    let (s, _, _) = post(&app, "/completion", &body, false).await;
    assert_eq!(s, 400);
}

#[tokio::test]
async fn native_completion_stream_has_no_done_and_a_final_stop_chunk() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let body = json!({"prompt": "hi", "n_predict": 8, "stream": true});
    let (s, _, events) = post_sse(&app, "/completion", &body, false).await;
    assert_eq!(s, 200);
    assert!(events.iter().all(|e| e.data != "[DONE]"), "{events:?}");
    let chunks: Vec<Value> = events.iter().map(Sse::json).collect();
    let (last, partial) = chunks.split_last().unwrap();
    assert!(partial.iter().all(|c| c["stop"] == false));
    let text: String = partial
        .iter()
        .map(|c| c["content"].as_str().unwrap())
        .collect();
    assert_eq!(text, "abc");
    assert_eq!(partial[0]["index"], 0);
    assert!(partial[0]["tokens_predicted"].is_number());
    // the final chunk repeats no content in stream mode
    assert_eq!(last["stop"], true);
    assert_eq!(last["content"], "");
    assert_eq!(last["stop_type"], "eos");
    assert_eq!(last["tokens_predicted"], 4);
    assert_eq!(last["generation_settings"]["stream"], true);
}

// ---- props ----

#[tokio::test]
async fn props_report_what_the_server_knows() {
    let cfg = GatewayConfig {
        max_concurrent: 3,
        ..GatewayConfig::default()
    };
    let (app, _) = app_with(FakeEngine::fixed("x"), cfg);
    let (s, _, v) = get(&app, "/props", false).await;
    assert_eq!(s, 200);
    assert_eq!(v["total_slots"], 3);
    assert_eq!(v["model_alias"], "fake");
    assert_eq!(v["model_path"], "/models/fake.gguf");
    assert_eq!(v["default_generation_settings"]["n_ctx"], 4096);
    assert!(v["default_generation_settings"]["params"].is_object());
    assert!(v["chat_template"].as_str().unwrap().contains("im_start"));
    assert_eq!(v["eos_token"], "<|im_end|>");
    assert_eq!(v["bos_token"], "");
    assert_eq!(v["endpoint_metrics"], false);
    assert_eq!(v["endpoint_props"], false);
    assert_eq!(v["is_sleeping"], false);
    assert!(v["build_info"].as_str().unwrap().starts_with("tetraeder-"));
}

#[tokio::test]
async fn post_props_is_not_supported_without_the_flag() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = post(&app, "/props", &json!({}), false).await;
    assert_eq!(s, 501);
    assert_eq!(v["error"]["type"], "not_supported_error");
    assert_eq!(v["error"]["code"], 501);
    assert!(v["error"]["message"].as_str().unwrap().contains("--props"));
}

// ---- metrics ----

#[tokio::test]
async fn metrics_need_the_flag() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = get(&app, "/metrics", false).await;
    assert_eq!(s, 501);
    assert_eq!(v["error"]["type"], "not_supported_error");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--metrics")
    );
}

#[tokio::test]
async fn metrics_export_llama_servers_names() {
    let cfg = GatewayConfig {
        metrics: true,
        ..GatewayConfig::default()
    };
    let (app, _) = app_with(FakeEngine::fixed("abc"), cfg);
    let (s, _, _) = post(&app, "/v1/completions", &json!({"prompt": "hi"}), false).await;
    assert_eq!(s, 200);
    // the counters are added when the generation ends
    let mut text = String::new();
    for _ in 0..200 {
        let (s, h, t) = send(&app, request("GET", "/metrics", None, false)).await;
        assert_eq!(s, 200);
        assert_eq!(h["content-type"], "text/plain; version=0.0.4");
        assert!(h.contains_key("process-start-time-unix"));
        text = t;
        if text.contains("llamacpp:tokens_predicted_total 4\n") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    for want in [
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens\n",
        "# TYPE llamacpp:prompt_tokens_total counter\n",
        "llamacpp:prompt_tokens_total 2\n",
        "llamacpp:prompt_tokens_cached_total 0\n",
        "llamacpp:tokens_predicted_total 4\n",
        "# TYPE llamacpp:requests_processing gauge\n",
        "llamacpp:requests_processing 0\n",
        "llamacpp:requests_deferred 0\n",
        "# TYPE llamacpp:prompt_tokens_seconds gauge\n",
        "# TYPE llamacpp:predicted_tokens_seconds gauge\n",
        "# TYPE llamacpp:prompt_seconds_total counter\n",
        "# TYPE llamacpp:tokens_predicted_seconds_total counter\n",
        "# TYPE llamacpp:n_tokens_max counter\n",
    ] {
        assert!(text.contains(want), "missing {want:?} in\n{text}");
    }
    // what the gateway cannot measure is not exported
    assert!(!text.contains("spec_decode"));
    assert!(!text.contains("n_decode_total"));
}

// ---- Responses ----

fn responses_body() -> Value {
    json!({"input": "hi", "instructions": "be brief", "max_output_tokens": 8})
}

#[tokio::test]
async fn responses_answers_in_the_response_shape() {
    let e = FakeEngine::fixed("abc");
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    for path in ["/v1/responses", "/responses"] {
        let (s, _, v) = post(&app, path, &responses_body(), false).await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["model"], "fake");
        assert!(v["id"].as_str().unwrap().starts_with("resp_"));
        let msg = &v["output"][0];
        assert_eq!(msg["type"], "message");
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["status"], "completed");
        assert!(msg["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(msg["content"][0]["type"], "output_text");
        assert_eq!(msg["content"][0]["text"], "abc");
        assert_eq!(v["usage"]["output_tokens"], 4);
        assert_eq!(
            v["usage"]["total_tokens"],
            v["usage"]["input_tokens"].as_u64().unwrap() + 4
        );
        assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 0);
    }
    // instructions become the system turn, the string input the user turn
    let want = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(prompts.lock().unwrap()[0], want);
}

#[tokio::test]
async fn responses_input_items() {
    let e = FakeEngine::fixed("ok");
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    let body = json!({"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "one"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "two"}]},
        {"role": "user", "content": "three"},
    ]});
    let (s, _, v) = post(&app, "/v1/responses", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let want = "<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\ntwo<|im_end|>\n<|im_start|>user\nthree<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(prompts.lock().unwrap()[0], want);
}

#[tokio::test]
async fn responses_function_call_output() {
    let (app, _) = app(FakeEngine::fixed(WEATHER_CALL));
    let body = json!({
        "input": "weather?",
        "tools": [{"type": "function", "name": "get_weather", "description": "Weather for a city",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    });
    let (s, _, v) = post(&app, "/v1/responses", &body, false).await;
    assert_eq!(s, 200, "{v}");
    let call = &v["output"][0];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["status"], "completed");
    assert_eq!(call["name"], "get_weather");
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    assert!(call["id"].as_str().unwrap().starts_with("fc_"));
    let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris"}));
}

#[tokio::test]
async fn responses_stream_events() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let mut body = responses_body();
    body["stream"] = json!(true);
    let (s, _, events) = post_sse(&app, "/v1/responses", &body, false).await;
    assert_eq!(s, 200);
    assert!(events.iter().all(|e| e.data != "[DONE]"));
    let names: Vec<&str> = events.iter().map(|e| e.event.as_deref().unwrap()).collect();
    assert_eq!(
        &names[..4],
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added"
        ]
    );
    let deltas: Vec<&Sse> = events
        .iter()
        .filter(|e| e.event.as_deref() == Some("response.output_text.delta"))
        .collect();
    let text: String = deltas
        .iter()
        .map(|e| e.json()["delta"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(text, "abc");
    assert_eq!(
        &names[names.len() - 4..],
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    let done = events
        .iter()
        .find(|e| e.event.as_deref() == Some("response.output_text.done"))
        .unwrap()
        .json();
    assert_eq!(done["text"], "abc");
    let completed = events.last().unwrap().json();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "abc"
    );
    assert_eq!(completed["response"]["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn responses_reject_what_is_not_supported() {
    let (app, _) = app(FakeEngine::fixed("x"));
    for (b, msg) in [
        (json!({}), "'input' is required"),
        (
            json!({"input": "x", "previous_response_id": "resp_1"}),
            "previous_response_id",
        ),
        (
            json!({"input": 5}),
            "'input' must be a string or array of objects",
        ),
        (
            json!({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]}),
            "input_file",
        ),
        (json!({"input": "x", "text": 5}), "'text' must be an object"),
    ] {
        let (s, _, v) = post(&app, "/v1/responses", &b, false).await;
        assert_eq!(s, 400, "{b}: {v}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(msg),
            "{b}: {v}"
        );
    }
}

// ---- infill ----

#[tokio::test]
async fn infill_needs_fim_tokens() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = post(
        &app,
        "/infill",
        &json!({"input_prefix": "a", "input_suffix": "b"}),
        false,
    )
    .await;
    assert_eq!(s, 501, "{v}");
    assert_eq!(v["error"]["type"], "not_supported_error");
    assert_eq!(
        v["error"]["message"],
        "Infill is not supported by this model: prefix token is missing. suffix token is missing. middle token is missing. "
    );
}

#[tokio::test]
async fn infill_builds_the_fim_prompt() {
    let e = FakeEngine::fixed("body").with_fim(false);
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    let body = json!({"input_prefix": "ab", "input_suffix": "cd", "n_predict": 8});
    let (s, _, v) = post(&app, "/infill", &body, false).await;
    assert_eq!(s, 200, "{v}");
    // native completion body
    assert_eq!(v["content"], "body");
    assert_eq!(v["stop"], true);
    assert_eq!(
        prompts.lock().unwrap()[0],
        "<fim_prefix>ab<fim_suffix>cd<fim_middle>"
    );
    // the prompt field goes between the prefix and the suffix
    let body = json!({"input_prefix": "ab", "input_suffix": "cd", "prompt": "X"});
    let (s, _, _) = post(&app, "/infill", &body, false).await;
    assert_eq!(s, 200);
    assert_eq!(
        prompts.lock().unwrap()[1],
        "<fim_prefix>abX<fim_suffix>cd<fim_middle>"
    );
}

#[tokio::test]
async fn infill_puts_extra_chunks_before_the_prefix() {
    let e = FakeEngine::fixed("x").with_fim(true);
    let prompts = Arc::clone(&e.prompts);
    let (app, _) = app(e);
    let body = json!({
        "input_prefix": "a", "input_suffix": "b",
        "input_extra": [{"filename": "f.rs", "text": "XX"}, {"text": "YY"}],
    });
    let (s, _, v) = post(&app, "/infill", &body, false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        prompts.lock().unwrap()[0],
        "<fim_repo>myproject\n<fim_sep>f.rs\nXX<fim_sep>tmp\nYY<fim_sep>filename\n<fim_prefix>a<fim_suffix>b<fim_middle>"
    );
}

#[tokio::test]
async fn infill_validates_its_input() {
    let (app, _) = app(FakeEngine::fixed("x").with_fim(false));
    for (b, msg) in [
        (json!({"input_suffix": "b"}), "\"input_prefix\" is required"),
        (json!({"input_prefix": "a"}), "\"input_suffix\" is required"),
        (
            json!({"input_prefix": "a", "input_suffix": "b", "prompt": 5}),
            "\"prompt\" must be a string",
        ),
        (
            json!({"input_prefix": "a", "input_suffix": "b", "input_extra": "x"}),
            "\"input_extra\" must be an array",
        ),
        (
            json!({"input_prefix": "a", "input_suffix": "b", "input_extra": [{"filename": "f"}]}),
            "extra_context chunk must contain a \"text\" field",
        ),
    ] {
        let (s, _, v) = post(&app, "/infill", &b, false).await;
        assert_eq!(s, 400, "{b}: {v}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(msg),
            "{b}: {v}"
        );
    }
}

// ---- request counter (rule 6) ----

fn request_points(exp: &InMemoryMetricExporter) -> Vec<(Vec<(String, String)>, u64)> {
    let mut out = Vec::new();
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for m in sm
                .metrics()
                .filter(|m| m.name() == "tetraeder.gateway.requests")
            {
                if let AggregatedMetrics::U64(MetricData::Sum(s)) = m.data() {
                    for dp in s.data_points() {
                        let mut a: Vec<(String, String)> = dp
                            .attributes()
                            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                            .collect();
                        a.sort();
                        out.push((a, dp.value()));
                    }
                }
            }
        }
    }
    out
}

#[tokio::test]
async fn requests_are_counted_per_route_and_status() {
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let gw = Arc::new(
        Gateway::new(Arc::new(FakeEngine::fixed("abc")), keyed(&["k"]))
            .with_meter(&mp.meter("test")),
    );
    let app = router(gw);
    send(&app, request("GET", "/health", None, false)).await;
    send(&app, request("GET", "/health", None, false)).await;
    send(&app, request("GET", "/models", None, false)).await; // 401
    send(&app, request("GET", "/some/raw/path/123", None, false)).await; // 401, unmatched
    let ok = with_header("GET", "/models", None, ("authorization", "Bearer k"));
    send(&app, ok).await;
    mp.force_flush().unwrap();
    let pts = request_points(&exp);
    let find = |route: &str, status: &str| {
        pts.iter()
            .find(|(a, _)| {
                has(a, "http.route", route) && has(a, "http.response.status_code", status)
            })
            .map(|p| p.1)
    };
    assert_eq!(find("/health", "200"), Some(2), "{pts:?}");
    assert_eq!(find("/models", "401"), Some(1), "{pts:?}");
    assert_eq!(find("/models", "200"), Some(1), "{pts:?}");
    // an unknown path is one label, not a series per path
    assert_eq!(find("unmatched", "401"), Some(1), "{pts:?}");
}

fn has(attrs: &[(String, String)], k: &str, v: &str) -> bool {
    attrs.iter().any(|(a, b)| a == k && b == v)
}

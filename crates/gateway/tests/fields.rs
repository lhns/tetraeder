//! Request and response fields of llama-server's group B (docs/notes/llamacpp-parity.md, ADR
//! 0083): optional model, EOS in the token count, log-probabilities, `n`, grammars, fingerprint
//! and timings. Fake engine, no model.

mod common;

use common::*;
use serde_json::{Value, json};
use tetraeder_gateway::engine::GrammarSource;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::GatewayConfig;

fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

fn slots(n: usize) -> GatewayConfig {
    GatewayConfig {
        max_concurrent: n,
        ..GatewayConfig::default()
    }
}

#[tokio::test]
async fn model_is_optional() {
    let (app, _) = app(FakeEngine::fixed("hi"));
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"messages": user("x")}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "fake");
    let (s, _, v) = post(&app, "/v1/completions", &json!({"prompt": "x"}), false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "fake");
}

#[tokio::test]
async fn end_of_generation_token_counts() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let body = json!({"model": "fake", "prompt": "x", "max_tokens": 10});
    let (_, _, v) = post(&app, "/v1/completions", &body, false).await;
    // a, b, c and the end-of-generation token
    assert_eq!(v["usage"]["completion_tokens"], 4, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let body = json!({"model": "fake", "prompt": "x", "max_tokens": 3});
    let (_, _, v) = post(&app, "/v1/completions", &body, false).await;
    assert_eq!(v["usage"]["completion_tokens"], 3, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn fingerprint_and_timings_on_responses() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    let (_, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("x")}),
        false,
    )
    .await;
    assert!(
        v["system_fingerprint"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "{v}"
    );
    let t = &v["timings"];
    for k in [
        "cache_n",
        "prompt_n",
        "prompt_ms",
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_ms",
        "predicted_per_token_ms",
        "predicted_per_second",
    ] {
        assert!(t[k].is_number(), "{k} in {t}");
    }
    assert_eq!(t["predicted_n"], 4);
    assert_eq!(t["prompt_n"], chatml_user_tokens("x"));
    // the rates count the steps after the first token
    let per_token = t["predicted_per_token_ms"].as_f64().unwrap();
    assert!((per_token - 2.5 / 3.0).abs() < 1e-9, "{t}");
}

#[tokio::test]
async fn timings_ride_the_last_stream_chunk() {
    let (app, _) = app(FakeEngine::fixed("abc"));
    for usage in [false, true] {
        let (_, _, sse) = post_sse(
            &app,
            "/v1/chat/completions",
            &json!({"model": "fake", "messages": user("x"), "stream": true,
                    "stream_options": {"include_usage": usage}}),
            false,
        )
        .await;
        let chunks: Vec<Value> = sse
            .iter()
            .filter(|e| e.data != "[DONE]")
            .map(Sse::json)
            .collect();
        let last = chunks.last().unwrap();
        assert!(last["timings"]["predicted_n"].is_number(), "{last}");
        assert!(chunks.iter().all(|c| c["system_fingerprint"].is_string()));
        let with_timings = chunks.iter().filter(|c| c.get("timings").is_some()).count();
        assert_eq!(with_timings, 1);
    }
}

#[tokio::test]
async fn chat_logprobs_shape() {
    let (app, _) = app(FakeEngine::fixed("ab"));
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("x"), "logprobs": true, "top_logprobs": 2}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    // a, b and the end-of-generation token
    assert_eq!(content.len(), 3, "{v}");
    let a = &content[0];
    assert_eq!(a["token"], "a");
    assert_eq!(a["bytes"], json!([97]));
    assert_eq!(a["id"], 97);
    assert!(a["logprob"].is_number());
    let top = a["top_logprobs"].as_array().unwrap();
    assert_eq!(top.len(), 2);
    for t in top {
        for k in ["id", "token", "bytes", "logprob"] {
            assert!(t.get(k).is_some(), "{k} in {t}");
        }
    }
    // the end-of-generation token is a control token: no text
    assert_eq!(content[2]["token"], "");
    assert_eq!(content[2]["bytes"], json!([]));
}

#[tokio::test]
async fn logprobs_default_to_twenty_top_tokens() {
    let engine = FakeEngine::fixed("a");
    let seen = engine.params.clone();
    let (app, _) = app(engine);
    post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("x"), "logprobs": true}),
        false,
    )
    .await;
    assert_eq!(seen.lock().unwrap()[0].extra.logprobs, Some(20));
}

#[tokio::test]
async fn logprobs_field_errors() {
    let (app, _) = app(FakeEngine::fixed("a"));
    let chat = |extra: Value| {
        let mut b = json!({"model": "fake", "messages": user("x")});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    for body in [
        chat(json!({"top_logprobs": 2})),
        chat(json!({"logprobs": true, "post_sampling_probs": true})),
        chat(json!({"logprobs": true, "top_logprobs": -1})),
        chat(json!({"logprobs": true, "stream": true, "tools": [weather_tool_openai()]})),
    ] {
        let (s, _, v) = post(&app, "/v1/chat/completions", &body, false).await;
        assert_eq!(s, 400, "{body} gave {v}");
    }
    // logprobs: false and top_logprobs: null are no request for logprobs
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &chat(json!({"logprobs": false, "top_logprobs": null})),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert!(v["choices"][0]["logprobs"].is_null());
}

#[tokio::test]
async fn completions_logprobs() {
    let (app, _) = app(FakeEngine::fixed("a"));
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "logprobs": 3}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2, "{v}");
    assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn stream_logprobs_ride_the_text_chunks() {
    let (app, _) = app(FakeEngine::fixed("ab"));
    let (_, _, sse) = post_sse(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("x"), "stream": true, "logprobs": true,
                "top_logprobs": 1}),
        false,
    )
    .await;
    let chunks: Vec<Value> = sse
        .iter()
        .filter(|e| e.data != "[DONE]")
        .map(Sse::json)
        .collect();
    let mut entries = Vec::new();
    for c in &chunks {
        if let Some(content) = c["choices"][0]["logprobs"]["content"].as_array() {
            entries.extend(content.iter().cloned());
        }
    }
    // a, b, end-of-generation
    assert_eq!(entries.len(), 3, "{chunks:?}");
    let first_text = chunks
        .iter()
        .find(|c| c["choices"][0]["delta"]["content"] == "a")
        .unwrap();
    assert_eq!(
        first_text["choices"][0]["logprobs"]["content"][0]["token"],
        "a"
    );
}

#[tokio::test]
async fn n_gives_indexed_choices_and_summed_usage() {
    let engine = FakeEngine::fixed("ab");
    let seen = engine.params.clone();
    let (app, _) = app_with(engine, slots(4));
    let (s, _, v) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "fake", "messages": user("x"), "n": 3, "seed": 10}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    let choices = v["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3);
    for (i, c) in choices.iter().enumerate() {
        assert_eq!(c["index"], i);
        assert_eq!(c["message"]["content"], "ab");
    }
    // one prompt, the completions of all choices
    assert_eq!(v["usage"]["prompt_tokens"], chatml_user_tokens("x"));
    assert_eq!(v["usage"]["completion_tokens"], 9);
    let mut offsets: Vec<u32> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|p| p.extra.seed_offset)
        .collect();
    offsets.sort_unstable();
    assert_eq!(offsets, [0, 1, 2]);
    assert!(seen.lock().unwrap().iter().all(|p| p.seed == Some(10)));
}

#[tokio::test]
async fn n_above_the_parallel_slots_is_rejected() {
    let (app, _) = app_with(FakeEngine::fixed("a"), slots(2));
    for n in [3, 0] {
        let (s, _, v) = post(
            &app,
            "/v1/completions",
            &json!({"model": "fake", "prompt": "x", "n": n}),
            false,
        )
        .await;
        assert_eq!(s, 400, "n={n}: {v}");
    }
}

#[tokio::test]
async fn n_streams_interleave_by_index() {
    let (app, _) = app_with(FakeEngine::fixed("ab"), slots(2));
    let (s, _, sse) = post_sse(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "n": 2, "stream": true,
                "stream_options": {"include_usage": true}}),
        false,
    )
    .await;
    assert_eq!(s, 200);
    let chunks: Vec<Value> = sse
        .iter()
        .filter(|e| e.data != "[DONE]")
        .map(Sse::json)
        .collect();
    for i in 0..2 {
        let text: String = chunks
            .iter()
            .filter(|c| c["choices"][0]["index"] == i)
            .filter_map(|c| c["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(text, "ab", "choice {i}");
        let finished = chunks
            .iter()
            .filter(|c| c["choices"][0]["index"] == i && c["choices"][0]["finish_reason"] == "stop")
            .count();
        assert_eq!(finished, 1, "choice {i}");
    }
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 6);
    assert!(usage["timings"].is_object());
}

#[tokio::test]
async fn grammar_reaches_the_engine_and_a_bad_one_is_a_400() {
    let engine = FakeEngine::fixed("a");
    let seen = engine.params.clone();
    let (app, _) = app(engine);
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "grammar": "root ::= \"ab\""}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        seen.lock().unwrap()[0].extra.grammar,
        Some(GrammarSource::Gbnf("root ::= \"ab\"".into()))
    );
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "grammar": "root ::= BAD"}),
        false,
    )
    .await;
    assert_eq!(s, 400, "{v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("grammar"),
        "{v}"
    );
}

#[tokio::test]
async fn json_schema_becomes_a_grammar() {
    let engine = FakeEngine::fixed("a");
    let seen = engine.params.clone();
    let (app, _) = app(engine);
    for schema in [json!({"type": "object"}), json!({})] {
        let (s, _, v) = post(
            &app,
            "/v1/completions",
            &json!({"model": "fake", "prompt": "x", "json_schema": schema}),
            false,
        )
        .await;
        assert_eq!(s, 200, "{v}");
    }
    assert!(matches!(
        seen.lock().unwrap()[0].extra.grammar,
        Some(GrammarSource::Gbnf(_))
    ));
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "json_schema": {"title": "BAD"}}),
        false,
    )
    .await;
    assert_eq!(s, 400, "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("json_schema"),
        "{v}"
    );
}

#[tokio::test]
async fn chat_rejects_grammar_with_json_schema_but_completions_prefers_the_grammar() {
    let engine = FakeEngine::fixed("a");
    let seen = engine.params.clone();
    let (app, _) = app(engine);
    let both = json!({"grammar": "root ::= \"a\"", "json_schema": {"type": "object"}});
    let mut chat = json!({"model": "fake", "messages": user("x")});
    chat.as_object_mut()
        .unwrap()
        .extend(both.as_object().unwrap().clone());
    let (s, _, v) = post(&app, "/v1/chat/completions", &chat, false).await;
    assert_eq!(s, 400, "{v}");
    let mut comp = json!({"model": "fake", "prompt": "x"});
    comp.as_object_mut()
        .unwrap()
        .extend(both.as_object().unwrap().clone());
    let (s, _, v) = post(&app, "/v1/completions", &comp, false).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        seen.lock().unwrap()[0].extra.grammar,
        Some(GrammarSource::Gbnf("root ::= \"a\"".into()))
    );
}

#[tokio::test]
async fn unsupported_fields_are_rejected_unless_at_their_default() {
    let (app, _) = app(FakeEngine::fixed("a"));
    for (field, bad, ok) in [
        ("return_tokens", json!(true), json!(false)),
        ("timings_per_token", json!(true), json!(false)),
        ("response_fields", json!(["content"]), json!([])),
        ("t_max_predict_ms", json!(100), json!(0)),
        ("n_indent", json!(4), json!(0)),
        ("lora", json!([{"id": 0, "scale": 1.0}]), json!([])),
        ("adaptive_target", json!(0.5), json!(-1)),
        ("grammar_lazy", json!(true), json!(false)),
    ] {
        for (value, status) in [(bad, 400), (ok, 200)] {
            let (s, _, v) = post(
                &app,
                "/v1/completions",
                &json!({"model": "fake", "prompt": "x", field: value}),
                false,
            )
            .await;
            assert_eq!(s, status, "{field}={value}: {v}");
        }
    }
    // accepted and without effect: prefix reuse is automatic. `id_slot` and `n_keep` act with the
    // scheduler (ADR 0090); a slot that does not exist is a 400.
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "cache_prompt": false, "id_slot": 0, "n_keep": 4}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    let (s, _, v) = post(
        &app,
        "/v1/completions",
        &json!({"model": "fake", "prompt": "x", "id_slot": 2}),
        false,
    )
    .await;
    assert_eq!(s, 400, "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Invalid slot ID")
    );
}

#[tokio::test]
async fn anthropic_model_is_optional_and_effort_is_read() {
    let (app, _) = app(FakeEngine::fixed("hi"));
    let (s, _, v) = post(
        &app,
        "/v1/messages",
        &json!({"max_tokens": 5, "messages": user("x"), "output_config": {"effort": "none"}}),
        true,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "fake");
}

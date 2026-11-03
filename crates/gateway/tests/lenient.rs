//! Claude-Code-shaped `/v1/messages` bodies through the Qwen3-Coder chat template (ADR 0049).
//! llama-server accepts all of them; each must give 200 and a sensible prompt.

mod common;

use common::*;
use serde_json::{Value, json};
use tetraeder_gateway::fake::FakeEngine;

const CODER: &str = include_str!("fixtures/templates/Qwen-Qwen3-Coder-30B-A3B-Instruct.jinja");

/// Posts `body` and returns the rendered prompt; fails unless the answer is 200.
async fn prompt_of(body: Value) -> String {
    let engine = FakeEngine::with_template(Some(CODER.to_owned()), |_| "ok".to_owned());
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let (s, _, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s, 200, "{body} -> {v}");
    let p = prompts.lock().unwrap()[0].clone();
    p
}

fn body(messages: Value) -> Value {
    json!({"model": "fake", "max_tokens": 16, "messages": messages})
}

fn bash_use(id: &str, input: Value) -> Value {
    json!({"type": "tool_use", "id": id, "name": "Bash", "input": input})
}

#[tokio::test]
async fn tool_reference_in_tool_result_becomes_text() {
    // The shape behind the 12 HTTP 400s of the Claude Code proxy run (tool search results).
    let p = prompt_of(body(json!([
        {"role": "user", "content": "find a tool"},
        {"role": "assistant", "content": [bash_use("t1", json!({"command": "ls"}))]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "tool_reference", "tool_name": "WebFetch"},
            {"type": "tool_reference", "tool_name": "NotebookEdit"}]}]}
    ])))
    .await;
    assert!(
        p.contains(
            "<tool_response>\nTool WebFetch loaded.\nTool NotebookEdit loaded.\n</tool_response>"
        ),
        "{p}"
    );
}

#[tokio::test]
async fn deferred_tools_are_rendered() {
    let engine = FakeEngine::with_template(Some(CODER.to_owned()), |_| "ok".to_owned());
    let prompts = engine.prompts.clone();
    let (app, _) = app(engine);
    let mut b = body(json!([{"role": "user", "content": "hi"}]));
    b["tools"] = json!([
        {"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {}}},
        {"name": "WebFetch", "description": "fetch", "defer_loading": true,
         "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}}}]);
    let (s, _, v) = post(&app, "/v1/messages", &b, true).await;
    assert_eq!(s, 200, "{v}");
    let p = prompts.lock().unwrap()[0].clone();
    assert!(
        p.contains("<name>Bash</name>") && p.contains("<name>WebFetch</name>"),
        "{p}"
    );
}

#[tokio::test]
async fn tool_use_shapes() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Running."},
            bash_use("a", json!({})),
            bash_use("b", json!({"command": "ls", "opts": {"all": true, "n": [1, 2]}})),
            {"type": "text", "text": ""}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": [
                {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
            {"type": "tool_result", "tool_use_id": "b", "is_error": true, "content": "boom"},
            {"type": "text", "text": "continue"}]}
    ])))
    .await;
    assert!(p.contains("Running."), "{p}");
    assert!(p.contains("<function=Bash>\n</function>"), "{p}");
    assert!(
        p.contains("<parameter=opts>\n{\"all\": true, \"n\": [1, 2]}\n</parameter>"),
        "{p}"
    );
    assert!(
        p.contains("<tool_response>\none\ntwo\n</tool_response>"),
        "{p}"
    );
    assert!(
        p.contains("<tool_response>\nError: boom\n</tool_response>"),
        "{p}"
    );
    assert!(p.contains("<|im_start|>user\ncontinue<|im_end|>"), "{p}");
}

#[tokio::test]
async fn non_object_tool_input_is_wrapped() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": "ls -l"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]}
    ])))
    .await;
    assert!(p.contains("<parameter=input>\nls -l\n</parameter>"), "{p}");
}

#[tokio::test]
async fn system_shapes() {
    let mut b = body(json!([
        {"role": "user", "content": "a"},
        {"role": "system", "content": "mid"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}]));
    b["system"] = json!([
        {"type": "text", "text": "S1.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": ""},
        {"type": "text", "text": "S2."}]);
    let p = prompt_of(b).await;
    assert!(
        p.starts_with("<|im_start|>system\nS1.S2.<|im_end|>\n"),
        "{p}"
    );
    assert!(
        p.contains("<|im_start|>user\na\n\n<system>\nmid\n</system><|im_end|>"),
        "{p}"
    );
}

#[tokio::test]
async fn leading_system_message_without_system_field() {
    let p = prompt_of(body(json!([
        {"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}])))
    .await;
    assert!(
        p.starts_with("<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nhi"),
        "{p}"
    );
}

#[tokio::test]
async fn thinking_blocks_and_unknown_blocks() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "mystery", "x": 1}, {"foo": 1}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "s"},
            {"type": "redacted_thinking", "data": "zzz"},
            {"type": "server_tool_use", "id": "x"},
            {"type": "text", "text": "answer"}]},
        {"role": "user", "content": "next"}])))
    .await;
    assert!(p.contains("<|im_start|>user\nq<|im_end|>"), "{p}");
    assert!(p.contains("<|im_start|>assistant\nanswer<|im_end|>"), "{p}");
    assert!(!p.contains("mystery") && !p.contains("hmm"), "{p}");
}

#[tokio::test]
async fn image_and_document_blocks_become_text() {
    let p = prompt_of(body(json!([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": " see "},
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "DOC TEXT"}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA"}}]}])))
    .await;
    assert!(p.contains("[image] see DOC TEXT[document]"), "{p}");
}

#[tokio::test]
async fn images_inside_tool_results() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [bash_use("a", json!({"command": "shot"}))]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]}]}])))
    .await;
    assert!(
        p.contains("<tool_response>\n[image]\n</tool_response>"),
        "{p}"
    );
}

#[tokio::test]
async fn consecutive_and_empty_messages_are_normalized() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": []},
        {"role": "user", "content": "three"}])))
    .await;
    assert!(
        p.contains("<|im_start|>user\none\n\ntwo\n\nthree<|im_end|>\n<|im_start|>assistant\n"),
        "{p}"
    );
}

#[tokio::test]
async fn prefill_continues_the_final_assistant_message() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "json please"},
        {"role": "assistant", "content": "{\"a\":"}])))
    .await;
    assert!(
        p.ends_with("<|im_start|>user\njson please<|im_end|>\n<|im_start|>assistant\n{\"a\":"),
        "{p}"
    );
}

#[tokio::test]
async fn empty_final_assistant_message_is_dropped() {
    let p = prompt_of(body(json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""}])))
    .await;
    assert!(
        p.ends_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"),
        "{p}"
    );
}

#[tokio::test]
async fn count_tokens_accepts_the_same_shapes() {
    let engine = FakeEngine::with_template(Some(CODER.to_owned()), |_| "ok".to_owned());
    let (app, _) = app(engine);
    let b = body(json!([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [bash_use("a", json!({}))]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [
            {"type": "tool_reference", "tool_name": "T"}]}]},
        {"role": "assistant", "content": "pre"}]));
    let (s, _, v) = post(&app, "/v1/messages/count_tokens", &b, true).await;
    assert_eq!(s, 200, "{v}");
}

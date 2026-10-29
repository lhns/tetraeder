//! Chat templates rendered by minijinja must equal Python jinja2 (golden.json, made by
//! fixtures/render_golden.py) byte for byte, and fail where jinja2 fails.

use std::path::PathBuf;

use serde_json::Value;
use tetraeder_gateway::chat::{ChatMessage, ToolCall, ToolDef};
use tetraeder_gateway::template::{ChatTemplate, TemplateError};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("golden.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn str_of(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(String::from)
}

fn message(v: &Value) -> ChatMessage {
    let tool_calls = v
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|c| ToolCall {
                    id: str_of(c, "id").unwrap(),
                    name: str_of(&c["function"], "name").unwrap(),
                    arguments: c["function"]["arguments"].clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    ChatMessage {
        role: str_of(v, "role").unwrap(),
        content: str_of(v, "content").unwrap(),
        reasoning_content: str_of(v, "reasoning_content"),
        tool_calls,
        tool_call_id: str_of(v, "tool_call_id"),
        name: str_of(v, "name"),
    }
}

fn tool(v: &Value) -> ToolDef {
    let f = &v["function"];
    ToolDef {
        name: str_of(f, "name").unwrap(),
        description: str_of(f, "description"),
        parameters: f["parameters"].clone(),
    }
}

#[test]
fn golden_parity_with_python_jinja2() {
    let g = golden();
    let time = g["fixed_time"].as_i64().unwrap();
    let mut checked = 0;
    let mut errors = 0;
    let mut failures = Vec::new();
    for entry in g["renders"].as_array().unwrap() {
        let name = entry["template"].as_str().unwrap();
        let case_name = entry["case"].as_str().unwrap();
        if let Some(reason) = entry.get("known_difference").and_then(Value::as_str) {
            eprintln!("skip {name}/{case_name}: {reason}");
            continue;
        }
        let case = &g["cases"][case_name];
        let src =
            std::fs::read_to_string(fixtures().join(format!("templates/{name}.jinja"))).unwrap();
        let bos = entry["bos"].as_str().unwrap();
        let eos = entry["eos"].as_str().unwrap();
        let tmpl = ChatTemplate::new(&src, Some(bos), Some(eos))
            .unwrap()
            .with_fixed_time(time);
        let messages: Vec<_> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(message)
            .collect();
        let tools: Vec<_> = case["tools"].as_array().unwrap().iter().map(tool).collect();
        let kwargs = case["kwargs"].as_object().unwrap();
        let got = tmpl.render(
            &messages,
            &tools,
            case["add_generation_prompt"].as_bool().unwrap(),
            kwargs,
        );
        checked += 1;
        if entry.get("error").and_then(Value::as_bool) == Some(true) {
            errors += 1;
            let want = entry["message"].as_str().unwrap();
            match got {
                Err(TemplateError::Render(m)) if m == want => {}
                other => failures.push(format!(
                    "{name}/{case_name}: want error {want:?}, got {other:?}"
                )),
            }
        } else {
            let want = entry["expected"].as_str().unwrap();
            match got {
                Ok(s) if s == want => {}
                Ok(s) => failures.push(format!(
                    "{name}/{case_name}: output differs\n--- got\n{s}\n--- want\n{want}"
                )),
                Err(e) => failures.push(format!("{name}/{case_name}: unexpected error {e}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} failed:\n{}",
        failures.len(),
        failures.join("\n=====\n")
    );
    assert!(
        checked >= 40 && errors >= 1,
        "checked {checked}, errors {errors}"
    );
}

#[test]
fn every_tool_template_renders_the_tool_case() {
    let g = golden();
    for name in [
        "Qwen-Qwen3-0.6B",
        "Qwen3-Coder",
        "Qwen-Qwen2.5-7B-Instruct",
        "meta-llama-Llama-3.1-8B-Instruct",
        "meta-llama-Llama-3.2-3B-Instruct",
        "Mistral-Small-3.2-24B-Instruct-2506",
        "NousResearch-Hermes-3-Llama-3.1-8B-tool_use",
    ] {
        let e = g["renders"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["template"] == name && e["case"] == "tool_round")
            .unwrap();
        assert!(e.get("error").is_none(), "{name} should render tool_round");
        assert!(
            e["expected"].as_str().unwrap().contains("get_weather"),
            "{name}"
        );
    }
}

#[test]
fn undefined_tools_when_empty_and_none_tokens_are_empty() {
    let t = ChatTemplate::new(
        "{{ bos_token }}|{{ eos_token }}|{{ tools is defined }}|{{ add_generation_prompt }}|{{ enable_thinking }}",
        None,
        None,
    )
    .unwrap();
    let mut kw = serde_json::Map::new();
    kw.insert("enable_thinking".into(), Value::Bool(false));
    let m = [ChatMessage::new("user", "x")];
    assert_eq!(t.render(&m, &[], true, &kw).unwrap(), "||False|True|False");
    assert_eq!(
        t.source(),
        "{{ bos_token }}|{{ eos_token }}|{{ tools is defined }}|{{ add_generation_prompt }}|{{ enable_thinking }}"
    );
}

#[test]
fn strftime_now_uses_system_time_unless_fixed() {
    let src = "{{ strftime_now('%Y') }}";
    let none = serde_json::Map::new();
    let now = ChatTemplate::new(src, None, None)
        .unwrap()
        .render(&[], &[], false, &none)
        .unwrap();
    assert!(now.parse::<i32>().unwrap() >= 2026);
    let fixed = ChatTemplate::new(src, None, None)
        .unwrap()
        .with_fixed_time(0)
        .render(&[], &[], false, &none)
        .unwrap();
    assert_eq!(fixed, "1970");
}

#[test]
fn crlf_in_source_is_normalised_like_jinja2() {
    let t = ChatTemplate::new("a{{ 'x\r\ny' }}\r\nb\r\n", None, None).unwrap();
    let out = t
        .render(&[], &[], false, &serde_json::Map::new())
        .unwrap();
    assert_eq!(out, "ax\ny\nb");
}

#[test]
fn syntax_errors_are_reported_at_construction() {
    assert!(matches!(
        ChatTemplate::new("{% for %}", None, None),
        Err(TemplateError::Syntax(_))
    ));
}

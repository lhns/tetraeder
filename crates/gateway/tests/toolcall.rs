//! Output parsing: formats, streaming at every chunking, fallbacks and reasoning.

use serde_json::{Value, json};
use tetraeder_gateway::chat::{Segment, ToolCall, ToolDef};
use tetraeder_gateway::toolcall::{OutputParser, ToolFormat};

fn tool(name: &str, parameters: Value) -> ToolDef {
    ToolDef {
        name: name.into(),
        description: None,
        parameters,
    }
}

fn weather() -> ToolDef {
    tool(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )
}

fn t(s: &str) -> Segment {
    Segment::Text(s.into())
}

fn r(s: &str) -> Segment {
    Segment::Reasoning(s.into())
}

/// Ids are random, so expected calls carry an empty one and `norm` blanks the parsed ones.
fn call(name: &str, arguments: Value) -> Segment {
    Segment::ToolCall(ToolCall {
        id: String::new(),
        name: name.into(),
        arguments,
    })
}

/// Merges adjacent pieces of one kind and blanks call ids.
fn norm(segs: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for s in segs {
        match (out.last_mut(), s) {
            (Some(Segment::Text(a)), Segment::Text(b)) => a.push_str(&b),
            (Some(Segment::Reasoning(a)), Segment::Reasoning(b)) => a.push_str(&b),
            (_, Segment::ToolCall(mut c)) => {
                c.id.clear();
                out.push(Segment::ToolCall(c));
            }
            (_, s) => out.push(s),
        }
    }
    out
}

struct Setup {
    format: ToolFormat,
    tools: Vec<ToolDef>,
    reasoning: bool,
    in_reasoning: bool,
}

impl Setup {
    fn new(format: ToolFormat) -> Self {
        Self {
            format,
            tools: vec![weather()],
            reasoning: false,
            in_reasoning: false,
        }
    }

    fn reasoning(mut self, in_reasoning: bool) -> Self {
        self.reasoning = true;
        self.in_reasoning = in_reasoning;
        self
    }

    fn tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }

    fn run<'a>(&self, chunks: impl IntoIterator<Item = &'a str>) -> Vec<Segment> {
        let mut p = OutputParser::new(self.format, &self.tools, self.reasoning, self.in_reasoning);
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c));
        }
        out.extend(p.finish());
        norm(out)
    }

    /// Parses `input` whole, one char at a time and in random chunkings; all must agree.
    fn parse(&self, input: &str) -> Vec<Segment> {
        let whole = self.run([input]);
        let chars: Vec<String> = input.chars().map(String::from).collect();
        assert_eq!(
            self.run(chars.iter().map(String::as_str)),
            whole,
            "per char: {input:?}"
        );
        for seed in 1..=40u64 {
            let chunks = random_chunks(input, seed);
            assert_eq!(
                self.run(chunks.iter().map(String::as_str)),
                whole,
                "chunks {chunks:?}"
            );
        }
        whole
    }
}

fn random_chunks(s: &str, seed: u64) -> Vec<String> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let n = 1 + next() % 7;
        let end = (i + n).min(chars.len());
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

const HERMES_A: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
const HERMES_B: &str =
    "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"CET\"}}\n</tool_call>";

#[test]
fn hermes_single_call() {
    let got = Setup::new(ToolFormat::Hermes).parse(HERMES_A);
    assert_eq!(got, vec![call("get_weather", json!({"city": "Paris"}))]);
}

#[test]
fn hermes_text_before_and_two_calls() {
    let input = format!("Let me check.\n{HERMES_A}\n{HERMES_B}");
    let got = Setup::new(ToolFormat::Hermes).parse(&input);
    assert_eq!(
        got,
        vec![
            t("Let me check.\n"),
            call("get_weather", json!({"city": "Paris"})),
            call("get_time", json!({"tz": "CET"})),
        ]
    );
}

#[test]
fn hermes_text_after_call_drops_leading_whitespace() {
    let input = format!("{HERMES_A}\n\nDone.");
    let got = Setup::new(ToolFormat::Hermes).parse(&input);
    assert_eq!(
        got,
        vec![call("get_weather", json!({"city": "Paris"})), t("Done.")]
    );
}

#[test]
fn hermes_unknown_tool_and_string_arguments_pass_through() {
    let input = "<tool_call>{\"name\": \"mystery\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>";
    let got = Setup::new(ToolFormat::Hermes).parse(input);
    assert_eq!(got, vec![call("mystery", json!({"a": 1}))]);
}

#[test]
fn hermes_missing_close_tag_still_parses_at_end() {
    let input = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}";
    let got = Setup::new(ToolFormat::Hermes).parse(input);
    assert_eq!(got, vec![call("get_weather", json!({}))]);
}

#[test]
fn hermes_malformed_call_becomes_text() {
    let bad = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": }\n</tool_call>";
    let got = Setup::new(ToolFormat::Hermes).parse(bad);
    assert_eq!(got, vec![t(bad)]);
    let input = format!("{bad} and then {HERMES_A}");
    let got = Setup::new(ToolFormat::Hermes).parse(&input);
    assert_eq!(
        got,
        vec![
            t(&format!("{bad} and then ")),
            call("get_weather", json!({"city": "Paris"}))
        ]
    );
    // JSON without a name is not a call either.
    let nameless = "<tool_call>{\"x\": 1}</tool_call>";
    assert_eq!(
        Setup::new(ToolFormat::Hermes).parse(nameless),
        vec![t(nameless)]
    );
}

#[test]
fn unterminated_garbage_block_is_text_at_end() {
    let input = "<tool_call>\n{\"name\": \"get_wea";
    assert_eq!(Setup::new(ToolFormat::Hermes).parse(input), vec![t(input)]);
}

#[test]
fn without_tools_everything_is_text() {
    let got = Setup::new(ToolFormat::Hermes).tools(vec![]).parse(HERMES_A);
    assert_eq!(got, vec![t(HERMES_A)]);
    let got = Setup::new(ToolFormat::Mistral)
        .tools(vec![])
        .parse("[TOOL_CALLS]a[ARGS]{}");
    assert_eq!(got, vec![t("[TOOL_CALLS]a[ARGS]{}")]);
}

fn search_tool() -> ToolDef {
    tool(
        "search",
        json!({"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "exact": {"type": "boolean"},
            "tags": {"type": "array"},
            "opts": {"type": "object"},
            "note": {"type": "string"},
            "ratio": {"type": "number"},
            "maybe": {"type": ["integer", "null"]},
        }}),
    )
}

#[test]
fn qwen_xml_converts_parameters_by_schema() {
    let input = concat!(
        "<tool_call>\n<function=search>\n",
        "<parameter=query>\nline1\nline2\n</parameter>\n",
        "<parameter=limit>\n5\n</parameter>\n",
        "<parameter=exact>\ntrue\n</parameter>\n",
        "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n",
        "<parameter=opts>\n{\"k\": 1}\n</parameter>\n",
        "<parameter=note>\n42\n</parameter>\n",
        "<parameter=ratio>\n0.5\n</parameter>\n",
        "<parameter=maybe>\nnull\n</parameter>\n",
        "<parameter=extra>\n7\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let got = Setup::new(ToolFormat::QwenXml)
        .tools(vec![search_tool()])
        .parse(input);
    assert_eq!(
        got,
        vec![call(
            "search",
            json!({
                "query": "line1\nline2", "limit": 5, "exact": true, "tags": ["a", "b"],
                "opts": {"k": 1}, "note": "42", "ratio": 0.5, "maybe": null, "extra": "7"
            })
        )]
    );
}

#[test]
fn qwen_xml_bad_value_falls_back_to_string_and_two_calls() {
    let one = "<tool_call>\n<function=search>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let input = format!("Searching.\n{one}\n{one}");
    let got = Setup::new(ToolFormat::QwenXml)
        .tools(vec![search_tool()])
        .parse(&input);
    let c = call("search", json!({"limit": "many"}));
    assert_eq!(got, vec![t("Searching.\n"), c.clone(), c]);
}

#[test]
fn qwen_xml_no_parameters_and_malformed() {
    let ok = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
    assert_eq!(
        Setup::new(ToolFormat::QwenXml).parse(ok),
        vec![call("ping", json!({}))]
    );
    let bad = "<tool_call>\n<function=ping>\n<parameter=a>\n1\n</function>\n</tool_call>";
    assert_eq!(Setup::new(ToolFormat::QwenXml).parse(bad), vec![t(bad)]);
}

#[test]
fn qwen_xml_accepts_json_call_too() {
    let got = Setup::new(ToolFormat::QwenXml).parse(HERMES_A);
    assert_eq!(got, vec![call("get_weather", json!({"city": "Paris"}))]);
}

#[test]
fn llama3_bare_json() {
    let s = Setup::new(ToolFormat::Llama3);
    let a = r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#;
    assert_eq!(
        s.parse(a),
        vec![call("get_weather", json!({"city": "Paris"}))]
    );
    assert_eq!(
        s.parse(&format!("\n {a}\n")),
        vec![call("get_weather", json!({"city": "Paris"}))]
    );
    assert_eq!(
        s.parse(r#"{"name": "f", "arguments": {"x": 1}}"#),
        vec![call("f", json!({"x": 1}))]
    );
}

#[test]
fn llama3_python_tag_and_multiple() {
    let s = Setup::new(ToolFormat::Llama3);
    let input =
        r#"<|python_tag|>{"name": "a", "parameters": {"x": ";"}}; {"name": "b", "parameters": {}}"#;
    assert_eq!(
        s.parse(input),
        vec![call("a", json!({"x": ";"})), call("b", json!({}))]
    );
}

#[test]
fn llama3_other_output_is_text() {
    let s = Setup::new(ToolFormat::Llama3);
    for input in [
        "Hello, how can I help?",
        "  Sure {\"name\": \"a\", \"parameters\": {}}",
        "{\"a\": 1}",
        "{\"name\": \"a\", \"parameters\": {}} and more",
        "{\"name\": \"a\"}",
        "<|python_tag|>brave_search.call(query=\"x\")",
        "<|python",
        "",
        "  ",
    ] {
        let got = s.parse(input);
        let want = if input.is_empty() {
            vec![]
        } else {
            vec![t(input)]
        };
        assert_eq!(got, want, "{input:?}");
    }
}

#[test]
fn mistral_array_form() {
    let s = Setup::new(ToolFormat::Mistral);
    let input = r#"Sure.[TOOL_CALLS][{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]"#;
    assert_eq!(
        s.parse(input),
        vec![t("Sure."), call("a", json!({"x": 1})), call("b", json!({}))]
    );
}

#[test]
fn mistral_name_args_form() {
    let s = Setup::new(ToolFormat::Mistral);
    assert_eq!(
        s.parse(r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#),
        vec![call("get_weather", json!({"city": "Paris"}))]
    );
    let two = concat!(
        r#"Checking.[TOOL_CALLS]a[CALL_ID]abc123xyz[ARGS]{"x": {"y": "}"}}"#,
        r#"[TOOL_CALLS]b[ARGS]{}"#
    );
    assert_eq!(
        s.parse(two),
        vec![
            t("Checking."),
            call("a", json!({"x": {"y": "}"}})),
            call("b", json!({}))
        ]
    );
}

#[test]
fn mistral_malformed_is_text() {
    let s = Setup::new(ToolFormat::Mistral);
    for input in [
        "[TOOL_CALLS]not a call",
        "[TOOL_CALLS]a[ARGS]{\"x\": ",
        "[TOOL_CALLS]a[ARGS]nope",
        "[TOOL_CALLS][1, 2]",
        "[TOOL_CALLS]",
    ] {
        assert_eq!(s.parse(input), vec![t(input)], "{input:?}");
    }
}

#[test]
fn reasoning_block_then_answer() {
    let s = Setup::new(ToolFormat::Hermes).reasoning(false);
    assert_eq!(
        s.parse("<think>\nplan it\nstep two\n</think>\n\nAnswer"),
        vec![r("plan it\nstep two"), t("Answer")]
    );
    assert_eq!(
        s.parse("Pre <think>x</think>post"),
        vec![t("Pre "), r("x"), t("post")]
    );
    assert_eq!(s.parse("<think>\n\n</think>\n\nHi"), vec![t("Hi")]);
    assert_eq!(s.parse("<think>unfinished\n"), vec![r("unfinished\n")]);
}

#[test]
fn reasoning_when_prompt_opened_the_block() {
    let s = Setup::new(ToolFormat::Hermes).reasoning(true);
    assert_eq!(
        s.parse("plan\n</think>\nAnswer"),
        vec![r("plan"), t("Answer")]
    );
    assert_eq!(
        s.parse("\nplan</think>Answer"),
        vec![r("plan"), t("Answer")]
    );
    assert_eq!(s.parse("still thinking"), vec![r("still thinking")]);
}

#[test]
fn reasoning_off_leaves_think_tags_as_text() {
    let s = Setup::new(ToolFormat::Hermes);
    assert_eq!(s.parse("<think>x</think>y"), vec![t("<think>x</think>y")]);
}

#[test]
fn reasoning_then_call() {
    let input = format!("<think>\nneed weather\n</think>\n\n{HERMES_A}");
    let got = Setup::new(ToolFormat::Hermes)
        .reasoning(false)
        .parse(&input);
    assert_eq!(
        got,
        vec![
            r("need weather"),
            call("get_weather", json!({"city": "Paris"}))
        ]
    );
    let llama = r#"{"name": "a", "parameters": {}}"#;
    let got = Setup::new(ToolFormat::Llama3)
        .reasoning(false)
        .parse(&format!("<think>hm</think>\n{llama}"));
    assert_eq!(got, vec![r("hm"), call("a", json!({}))]);
}

#[test]
fn text_is_emitted_as_soon_as_it_cannot_start_a_marker() {
    let mut p = OutputParser::new(ToolFormat::Hermes, &[weather()], true, false);
    assert_eq!(p.push("Hello <tool"), vec![t("Hello ")]);
    assert_eq!(p.push("_ca"), vec![]);
    assert_eq!(p.push("x"), vec![t("<tool_cax")]);
    assert_eq!(p.push("a <"), vec![t("a ")]);
    assert_eq!(p.finish(), vec![t("<")]);

    let mut p = OutputParser::new(ToolFormat::Hermes, &[weather()], true, true);
    assert_eq!(p.push("thinking"), vec![r("thinking")]);
    assert_eq!(p.push(" more\n"), vec![r(" more")]);
    assert_eq!(p.push("</think>"), vec![]);
    assert_eq!(p.push("\n\nok"), vec![t("ok")]);
}

#[test]
fn call_is_emitted_when_complete() {
    let mut p = OutputParser::new(ToolFormat::Hermes, &[weather()], false, false);
    let (head, tail) = HERMES_A.split_at(HERMES_A.len() - 3);
    assert_eq!(p.push(head), vec![]);
    let segs = p.push(tail);
    assert_eq!(
        norm(segs),
        vec![call("get_weather", json!({"city": "Paris"}))]
    );
    assert_eq!(p.finish(), vec![]);

    let mut p = OutputParser::new(ToolFormat::Mistral, &[weather()], false, false);
    assert_eq!(
        p.push("[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Pa"),
        vec![]
    );
    assert_eq!(norm(p.push("ris\"}")).len(), 1);
}

#[test]
fn call_ids_look_like_openai_ids() {
    let mut p = OutputParser::new(ToolFormat::Hermes, &[weather()], false, false);
    let mut ids = Vec::new();
    for s in p.push(&format!("{HERMES_A}{HERMES_B}")) {
        let Segment::ToolCall(c) = s else { panic!() };
        ids.push(c.id);
    }
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    for id in ids {
        let body = id.strip_prefix("call_").unwrap();
        assert_eq!(body.len(), 24);
        assert!(
            body.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        );
    }
}

#[test]
fn detect_on_fixture_templates() {
    macro_rules! tpl {
        ($n:literal) => {
            include_str!(concat!("fixtures/templates/", $n, ".jinja"))
        };
    }
    use ToolFormat::*;
    for (name, src, want) in [
        ("Qwen3-0.6B", tpl!("Qwen-Qwen3-0.6B"), Hermes),
        ("Qwen2.5", tpl!("Qwen-Qwen2.5-7B-Instruct"), Hermes),
        ("Qwen3-Coder", tpl!("Qwen3-Coder"), QwenXml),
        (
            "Llama-3.1",
            tpl!("meta-llama-Llama-3.1-8B-Instruct"),
            Llama3,
        ),
        (
            "Llama-3.2",
            tpl!("meta-llama-Llama-3.2-3B-Instruct"),
            Llama3,
        ),
        (
            "Mistral-Small",
            tpl!("Mistral-Small-3.2-24B-Instruct-2506"),
            Mistral,
        ),
        (
            "Hermes-3",
            tpl!("NousResearch-Hermes-3-Llama-3.1-8B-tool_use"),
            Hermes,
        ),
        (
            "R1-distill",
            tpl!("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B"),
            Hermes,
        ),
        ("gemma", tpl!("google-gemma-2-2b-it"), Hermes),
        ("phi", tpl!("microsoft-Phi-3.5-mini-instruct"), Hermes),
    ] {
        assert_eq!(ToolFormat::detect(src), want, "{name}");
    }
}

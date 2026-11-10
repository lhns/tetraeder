//! Output parsing: formats, streaming at every chunking, fallbacks and reasoning.
//!
//! Hardware label: none (pure host code).

use serde_json::{Value, json};
use tetraeder_gateway::chat::{Segment, ToolCall, ToolDef};
use tetraeder_gateway::toolcall::{CallOutcome, OutputParser, ToolFormat};

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

/// Claude Code's Edit tool as it sends it (ADR 0057 fixtures).
fn cc_edit() -> ToolDef {
    tool(
        "Edit",
        json!({"type": "object", "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"default": false, "type": "boolean"},
        }, "required": ["file_path", "old_string", "new_string"], "additionalProperties": false}),
    )
}

fn cc_schedule_wakeup() -> ToolDef {
    tool(
        "ScheduleWakeup",
        json!({"type": "object", "properties": {
            "delaySeconds": {"type": "number"},
            "reason": {"type": "string"},
            "prompt": {"type": "string"},
            "stop": {"type": "boolean"},
            "noop": {"type": "boolean"},
        }, "additionalProperties": false}),
    )
}

fn cc_report_findings() -> ToolDef {
    tool(
        "ReportFindings",
        json!({"type": "object", "properties": {
            "level": {"type": "string", "enum": ["low", "medium", "high", "xhigh", "max"]},
            "findings": {"type": "array", "maxItems": 32, "items": {"type": "object", "properties": {
                "file": {"type": "string"},
                "line": {"type": "integer"},
                "summary": {"type": "string"},
                "short_summary": {"type": "string", "maxLength": 60},
                "failure_scenario": {"type": "string"},
                "category": {"type": "string", "maxLength": 40},
            }}},
        }}),
    )
}

fn claude_code_tools() -> Vec<ToolDef> {
    vec![cc_edit(), cc_schedule_wakeup(), cc_report_findings()]
}

/// 58 of the 60 InputValidationErrors of bench/results/20261110/claude-code-proxy: Python-cased
/// booleans (`False` 52, `True` 6) on boolean parameters. Raw text reconstructed from the request
/// log, which holds the parsed calls only.
#[test]
fn qwen_xml_python_booleans_follow_schema() {
    let edit = concat!(
        "<tool_call>\n<function=Edit>\n",
        "<parameter=replace_all>\nFalse\n</parameter>\n",
        "<parameter=file_path>\n/tmp/repo/invoicing/cli.py\n</parameter>\n",
        "<parameter=old_string>\nfrom invoicing.models import Invoice, compute_total\n</parameter>\n",
        "<parameter=new_string>\nfrom invoicing.models import Invoice, invoice_total\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let wake = concat!(
        "<tool_call>\n<function=ScheduleWakeup>\n",
        "<parameter=delaySeconds>\n300\n</parameter>\n",
        "<parameter=reason>\nFinal verification complete\n</parameter>\n",
        "<parameter=prompt>\nFinal verification complete\n</parameter>\n",
        "<parameter=stop>\nTrue\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let s = Setup::new(ToolFormat::QwenXml).tools(claude_code_tools());
    assert_eq!(
        s.parse(edit),
        vec![call(
            "Edit",
            json!({
                "replace_all": false,
                "file_path": "/tmp/repo/invoicing/cli.py",
                "old_string": "from invoicing.models import Invoice, compute_total",
                "new_string": "from invoicing.models import Invoice, invoice_total",
            })
        )]
    );
    assert_eq!(
        s.parse(wake),
        vec![call(
            "ScheduleWakeup",
            json!({
                "delaySeconds": 300, "reason": "Final verification complete",
                "prompt": "Final verification complete", "stop": true,
            })
        )]
    );
    assert_eq!(s.outcomes(edit), vec![o("parsed", "ok")]);
    assert_eq!(s.outcomes(wake), vec![o("parsed", "ok")]);
}

#[test]
fn qwen_xml_boolean_casing_only_for_boolean_parameters() {
    let input = concat!(
        "<tool_call>\n<function=Edit>\n",
        "<parameter=file_path>\nTrue\n</parameter>\n",
        "<parameter=old_string>\nFalse\n</parameter>\n",
        "<parameter=new_string>\nNone\n</parameter>\n",
        "<parameter=replace_all>\nTRUE\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let got = Setup::new(ToolFormat::QwenXml)
        .tools(claude_code_tools())
        .parse(input);
    assert_eq!(
        got,
        vec![call(
            "Edit",
            json!({"file_path": "True", "old_string": "False", "new_string": "None", "replace_all": true})
        )]
    );
}

/// The other 2 InputValidationErrors: a Python `repr` list on an array parameter.
#[test]
fn qwen_xml_python_literal_array_follows_schema() {
    let findings = concat!(
        "[{'category': 'correctness', 'file': '/tmp/repo/legacy/pricing/volume.py', 'line': 15, ",
        "'summary': 'Volume pricing select_tier function incorrectly excludes boundary values from tier selection', ",
        "'short_summary': 'Tier boundary exclusion bug in volume pricing', ",
        "'failure_scenario': \"When customer usage exactly equals a tier's upper limit (e.g., 100 GB for a tier ",
        "with up_to=100), the function selects the next tier instead of the correct tier\"}]"
    );
    let input = format!(
        "<tool_call>\n<function=ReportFindings>\n<parameter=findings>\n{findings}\n</parameter>\n\
         <parameter=level>\nhigh\n</parameter>\n</function>\n</tool_call>"
    );
    let got = Setup::new(ToolFormat::QwenXml)
        .tools(claude_code_tools())
        .parse(&input);
    assert_eq!(
        got,
        vec![call(
            "ReportFindings",
            json!({"findings": [{
                "category": "correctness",
                "file": "/tmp/repo/legacy/pricing/volume.py",
                "line": 15,
                "summary": "Volume pricing select_tier function incorrectly excludes boundary values from tier selection",
                "short_summary": "Tier boundary exclusion bug in volume pricing",
                "failure_scenario": "When customer usage exactly equals a tier's upper limit (e.g., 100 GB for a tier with up_to=100), the function selects the next tier instead of the correct tier",
            }], "level": "high"})
        )]
    );
}

#[test]
fn qwen_xml_python_literals_edge_cases() {
    let s = Setup::new(ToolFormat::QwenXml).tools(vec![search_tool()]);
    let one = |name: &str, value: &str| {
        let input = format!(
            "<tool_call>\n<function=search>\n<parameter={name}>\n{value}\n</parameter>\n</function>\n</tool_call>"
        );
        match s.parse(&input).as_slice() {
            [Segment::ToolCall(c)] => c.arguments[name].clone(),
            other => panic!("{other:?}"),
        }
    };
    // Keywords inside strings stay text; escapes, tuples, nesting and a trailing comma.
    assert_eq!(
        one(
            "tags",
            r#"['True', "it's", 'a\'b', 'x\ny', None, False, -1.5, (1, 2), {'k': [True]},]"#
        ),
        json!(["True", "it's", "a'b", "x\ny", null, false, -1.5, [1, 2], {"k": [true]}])
    );
    assert_eq!(one("opts", "{'a': {'b': None}}"), json!({"a": {"b": null}}));
    assert_eq!(one("maybe", "None"), json!(null));
    assert_eq!(one("exact", "false"), json!(false));
    // Not a whole literal, or the wrong type: the text is kept, as before.
    assert_eq!(one("tags", "[1, 2"), json!("[1, 2"));
    assert_eq!(one("tags", "['a'] trailing"), json!("['a'] trailing"));
    assert_eq!(one("tags", "{'a': 1}"), json!("{'a': 1}"));
    assert_eq!(one("opts", "{1: 'a'}"), json!("{1: 'a'}"));
    assert_eq!(one("tags", "[Truth]"), json!("[Truth]"));
    assert_eq!(one("exact", "yes"), json!("yes"));
    assert_eq!(one("ratio", "nan"), json!("nan"));
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
    ] {
        assert_eq!(s.parse(input), vec![t(input)], "{input:?}");
    }
}

#[test]
fn mistral_bad_array_with_non_ascii_is_text() {
    // A nameless item or an empty array falls back to text at the block length, not the JSON
    // length; the wrong offset lands inside a multi-byte char and panics.
    let s = Setup::new(ToolFormat::Mistral);
    for input in [
        "[TOOL_CALLS][{\"arguments\":{}}] \u{e9}\u{e9}\u{e9}",
        "[TOOL_CALLS][{\"arguments\":{\"q\":\"\u{4e16}\u{754c}\"}}]",
        "[TOOL_CALLS][] \u{1f600}",
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

// Tool-call outcomes and the empty-call repair (ADR 0051).

fn o(outcome: &'static str, class: &'static str) -> CallOutcome {
    CallOutcome { outcome, class }
}

impl Setup {
    /// Outcomes for one chunking.
    fn outcomes_of<'a>(&self, chunks: impl IntoIterator<Item = &'a str>) -> Vec<CallOutcome> {
        let mut p = OutputParser::new(self.format, &self.tools, self.reasoning, self.in_reasoning);
        let mut out = Vec::new();
        for c in chunks {
            p.push(c);
            out.extend(p.take_outcomes());
        }
        p.finish();
        out.extend(p.take_outcomes());
        out
    }

    /// Outcomes of `input`; every chunking must agree.
    fn outcomes(&self, input: &str) -> Vec<CallOutcome> {
        let whole = self.outcomes_of([input]);
        let chars: Vec<String> = input.chars().map(String::from).collect();
        assert_eq!(self.outcomes_of(chars.iter().map(String::as_str)), whole);
        for seed in 1..=40u64 {
            let chunks = random_chunks(input, seed);
            assert_eq!(
                self.outcomes_of(chunks.iter().map(String::as_str)),
                whole,
                "chunks {chunks:?}"
            );
        }
        whole
    }
}

#[test]
fn qwen_xml_trailing_empty_opener_is_dropped() {
    let text = "The test suite is now ready to ensure the shipping module works correctly and to catch any future changes that might break existing functionality.\n";
    let input = format!("{text}<tool_call>");
    let s = Setup::new(ToolFormat::QwenXml);
    assert_eq!(s.parse(&input), vec![t(text)]);
    assert_eq!(s.outcomes(&input), vec![o("repaired", "empty_call")]);
}

#[test]
fn empty_closed_call_is_dropped() {
    let h = Setup::new(ToolFormat::Hermes);
    assert_eq!(h.parse("<tool_call>\n</tool_call>"), vec![]);
    assert_eq!(
        h.outcomes("<tool_call>\n</tool_call>"),
        vec![o("repaired", "empty_call")]
    );
    let q = Setup::new(ToolFormat::QwenXml);
    assert_eq!(q.parse("<tool_call> </tool_call>after"), vec![t("after")]);
    assert_eq!(
        q.outcomes("<tool_call> </tool_call>after"),
        vec![o("repaired", "empty_call")]
    );
}

#[test]
fn mistral_and_llama3_empty_calls_are_dropped() {
    let m = Setup::new(ToolFormat::Mistral);
    assert_eq!(m.parse("[TOOL_CALLS]  "), vec![]);
    assert_eq!(
        m.outcomes("[TOOL_CALLS]  "),
        vec![o("repaired", "empty_call")]
    );
    let l = Setup::new(ToolFormat::Llama3);
    assert_eq!(l.parse("<|python_tag|>"), vec![]);
    assert_eq!(
        l.outcomes("<|python_tag|>"),
        vec![o("repaired", "empty_call")]
    );
}

#[test]
fn outcomes_by_class() {
    let h = Setup::new(ToolFormat::Hermes);
    assert_eq!(h.outcomes(HERMES_A), vec![o("parsed", "ok")]);
    assert_eq!(h.outcomes(HERMES_B), vec![o("parsed", "unknown_tool")]);
    let two = format!("{HERMES_A}{HERMES_B}");
    assert_eq!(
        h.outcomes(&two),
        vec![o("parsed", "ok"), o("parsed", "unknown_tool")]
    );
    let bad_json = "<tool_call>{\"name\": </tool_call>";
    assert_eq!(h.outcomes(bad_json), vec![o("failed", "bad_json")]);
    assert_eq!(h.parse(bad_json), vec![t(bad_json)]);

    let q = Setup::new(ToolFormat::QwenXml);
    let unclosed = "<tool_call>\n<function=read>\n<parameter=path>\na";
    assert_eq!(q.outcomes(unclosed), vec![o("failed", "unclosed")]);
    assert_eq!(q.parse(unclosed), vec![t(unclosed)]);
    let bad_xml = "<tool_call><function=read><parameter=path>x</tool_call>";
    assert_eq!(q.outcomes(bad_xml), vec![o("failed", "bad_xml")]);
    assert_eq!(q.parse(bad_xml), vec![t(bad_xml)]);

    let m = Setup::new(ToolFormat::Mistral);
    assert_eq!(
        m.outcomes("[TOOL_CALLS]get_weather[ARGS]{\"city\": "),
        vec![o("failed", "unclosed")]
    );
    assert_eq!(
        m.outcomes("[TOOL_CALLS][1, 2]"),
        vec![o("failed", "bad_json")]
    );
    let l = Setup::new(ToolFormat::Llama3);
    assert_eq!(
        l.outcomes("{\"name\": \"get_weather\", \"parameters\": "),
        vec![o("failed", "unclosed")]
    );
    assert_eq!(l.outcomes("{\"a\": 1}"), vec![o("failed", "bad_json")]);
    assert_eq!(l.outcomes("Hello there"), vec![]);
}

//! Parses model output into text, reasoning and tool calls, incrementally.
//!
//! One state machine serves every family. Text and reasoning go out as soon as they cannot be
//! the start of a marker; a tool call goes out when it is complete; a call that does not parse
//! goes out as the raw text it was. Results do not depend on how the input is chunked, except
//! that adjacent pieces of one kind may be split differently.

use serde_json::{Map, Value};

use crate::chat::{Segment, ToolCall, ToolDef};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const MISTRAL_CALLS: &str = "[TOOL_CALLS]";
const MISTRAL_ARGS: &str = "[ARGS]";
const MISTRAL_ID: &str = "[CALL_ID]";
const PYTHON_TAG: &str = "<|python_tag|>";
const PARAM_CLOSE: &str = "</parameter>";

/// The tool-call syntax a model family uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Hermes, Qwen2.5, Qwen3).
    Hermes,
    /// `<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`.
    QwenXml,
    /// A bare JSON object `{"name": ..., "parameters": {...}}`, optionally after `<|python_tag|>`.
    Llama3,
    /// `[TOOL_CALLS][{...}]` or `[TOOL_CALLS]NAME[ARGS]{...}`.
    Mistral,
}

impl ToolFormat {
    /// Picks the family from the chat template text; unknown templates map to Hermes.
    pub fn detect(template: &str) -> ToolFormat {
        if template.contains("<function=") {
            ToolFormat::QwenXml
        } else if template.contains(MISTRAL_CALLS) {
            ToolFormat::Mistral
        } else if template.contains(TOOL_OPEN) {
            ToolFormat::Hermes
        } else if template.contains(PYTHON_TAG)
            || template.contains("Environment: ipython")
            || template.contains("<|start_header_id|>")
        {
            ToolFormat::Llama3
        } else {
            ToolFormat::Hermes
        }
    }
}

enum Mode {
    Text,
    Think,
    /// Collecting the raw text of a tool call in `block`.
    Block,
}

pub struct OutputParser {
    format: ToolFormat,
    tools: Vec<ToolDef>,
    parse_tools: bool,
    reasoning: bool,
    mode: Mode,
    /// Received but undecided text.
    pending: String,
    /// Raw text of the tool call being collected, start marker included.
    block: String,
    /// Drop whitespace before the next text (after a call or a closed think block).
    strip_ws: bool,
    /// Drop one newline at the start of a think block.
    think_lead: bool,
    /// Llama 3: the next non-whitespace character may start a bare JSON call.
    armed: bool,
    out: Vec<Segment>,
}

impl OutputParser {
    /// `tools`: parse tool calls (else all non-reasoning output is text). `reasoning`: split
    /// `<think>...</think>`. `in_reasoning`: the output starts inside a think block.
    pub fn new(format: ToolFormat, tools: &[ToolDef], reasoning: bool, in_reasoning: bool) -> Self {
        let in_think = reasoning && in_reasoning;
        Self {
            format,
            tools: tools.to_vec(),
            parse_tools: !tools.is_empty(),
            reasoning,
            mode: if in_think { Mode::Think } else { Mode::Text },
            pending: String::new(),
            block: String::new(),
            strip_ws: false,
            think_lead: in_think,
            armed: true,
            out: Vec::new(),
        }
    }

    /// Feeds decoded text; returns the segments that are final so far.
    pub fn push(&mut self, text: &str) -> Vec<Segment> {
        self.pending.push_str(text);
        self.drain(false);
        std::mem::take(&mut self.out)
    }

    /// End of output: flushes held-back text, closes open blocks, parses pending calls.
    pub fn finish(&mut self) -> Vec<Segment> {
        self.drain(true);
        std::mem::take(&mut self.out)
    }

    fn drain(&mut self, finish: bool) {
        loop {
            let again = match self.mode {
                Mode::Text => self.drain_text(finish),
                Mode::Think => self.drain_think(finish),
                Mode::Block => self.drain_block(finish),
            };
            if !again {
                break;
            }
        }
    }

    fn emit_text(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.armed = false;
        match self.out.last_mut() {
            Some(Segment::Text(t)) => t.push_str(s),
            _ => self.out.push(Segment::Text(s.to_owned())),
        }
    }

    fn emit_reasoning(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        match self.out.last_mut() {
            Some(Segment::Reasoning(t)) => t.push_str(s),
            _ => self.out.push(Segment::Reasoning(s.to_owned())),
        }
    }

    /// Returns true when the mode changed and the loop should run again.
    fn drain_text(&mut self, finish: bool) -> bool {
        if self.strip_ws {
            let n = self.pending.len() - self.pending.trim_start().len();
            self.pending.drain(..n);
            if self.pending.is_empty() {
                return false;
            }
            self.strip_ws = false;
        }
        if self.parse_tools && self.format == ToolFormat::Llama3 && self.armed {
            let t = self.pending.trim_start();
            if t.is_empty() {
                if finish {
                    let s = std::mem::take(&mut self.pending);
                    self.emit_text(&s);
                }
                return false;
            }
            if t.starts_with(PYTHON_TAG) || t.starts_with('{') {
                self.block = std::mem::take(&mut self.pending);
                self.mode = Mode::Block;
                return true;
            }
            if !finish && PYTHON_TAG.starts_with(t) {
                return false;
            }
            self.armed = false;
        }
        let mut markers: Vec<&'static str> = Vec::new();
        if self.reasoning {
            markers.push(THINK_OPEN);
        }
        if self.parse_tools {
            match self.format {
                ToolFormat::Hermes | ToolFormat::QwenXml => markers.push(TOOL_OPEN),
                ToolFormat::Mistral => markers.push(MISTRAL_CALLS),
                ToolFormat::Llama3 => {}
            }
        }
        let found = markers
            .iter()
            .filter_map(|m| self.pending.find(m).map(|i| (i, *m)))
            .min_by_key(|(i, _)| *i);
        if let Some((i, marker)) = found {
            let before: String = self.pending.drain(..i).collect();
            self.emit_text(&before);
            self.pending.drain(..marker.len());
            if marker == THINK_OPEN {
                self.mode = Mode::Think;
                self.think_lead = true;
            } else {
                self.mode = Mode::Block;
                self.block = marker.to_owned();
            }
            return true;
        }
        let hold = if finish {
            0
        } else {
            hold_len(&self.pending, &markers)
        };
        let n = self.pending.len() - hold;
        let text: String = self.pending.drain(..n).collect();
        self.emit_text(&text);
        false
    }

    fn drain_think(&mut self, finish: bool) -> bool {
        if self.think_lead && !self.pending.is_empty() {
            if self.pending.starts_with('\n') {
                self.pending.remove(0);
            }
            self.think_lead = false;
        }
        if let Some(i) = self.pending.find(THINK_CLOSE) {
            let mut body: String = self.pending.drain(..i).collect();
            self.pending.drain(..THINK_CLOSE.len());
            if body.ends_with('\n') {
                body.pop();
            }
            self.emit_reasoning(&body);
            self.mode = Mode::Text;
            self.strip_ws = true;
            self.armed = true;
            return true;
        }
        let hold = if finish {
            0
        } else {
            hold_len(&self.pending, &[THINK_CLOSE, "\n</think>"])
        };
        let n = self.pending.len() - hold;
        let text: String = self.pending.drain(..n).collect();
        self.emit_reasoning(&text);
        false
    }

    fn drain_block(&mut self, finish: bool) -> bool {
        let moved = std::mem::take(&mut self.pending);
        self.block.push_str(&moved);
        let step = match self.format {
            ToolFormat::Hermes | ToolFormat::QwenXml => self.step_tagged(finish),
            ToolFormat::Mistral => self.step_mistral(finish),
            ToolFormat::Llama3 => self.step_llama(finish),
        };
        match step {
            Step::Wait => false,
            Step::Calls(calls, consumed) => {
                self.leave_block(consumed);
                self.strip_ws = true;
                self.out.extend(calls.into_iter().map(Segment::ToolCall));
                true
            }
            Step::Fail(consumed) => {
                let raw = self.block[..consumed].to_owned();
                self.leave_block(consumed);
                self.emit_text(&raw);
                true
            }
        }
    }

    /// Returns to text mode with the unconsumed rest of the block as pending text.
    fn leave_block(&mut self, consumed: usize) {
        self.pending = self.block.split_off(consumed);
        self.block.clear();
        self.mode = Mode::Text;
    }

    /// Hermes and Qwen XML: the call ends at `</tool_call>` (or at the end of the output).
    fn step_tagged(&self, finish: bool) -> Step {
        let (content_end, consumed) = match self.block.find(TOOL_CLOSE) {
            Some(i) => (i, i + TOOL_CLOSE.len()),
            None if finish => (self.block.len(), self.block.len()),
            None => return Step::Wait,
        };
        let content = &self.block[TOOL_OPEN.len()..content_end];
        let call = match self.format {
            ToolFormat::QwenXml if content.trim_start().starts_with("<function=") => {
                parse_xml_call(content, &self.tools)
            }
            _ => serde_json::from_str::<Value>(content.trim())
                .ok()
                .and_then(|v| call_from_json(&v, false)),
        };
        match call {
            Some(c) => Step::Calls(vec![c], consumed),
            None => Step::Fail(consumed),
        }
    }

    /// Llama 3: the whole output must be calls, so decide at the end.
    fn step_llama(&self, finish: bool) -> Step {
        if !finish {
            return Step::Wait;
        }
        match parse_llama(&self.block) {
            Some(calls) => Step::Calls(calls, self.block.len()),
            None => Step::Fail(self.block.len()),
        }
    }

    /// Mistral: `[TOOL_CALLS][{...}, ...]` or `[TOOL_CALLS]NAME[CALL_ID]id[ARGS]{...}`.
    fn step_mistral(&self, finish: bool) -> Step {
        let body = &self.block[MISTRAL_CALLS.len()..];
        let lead = body.len() - body.trim_start().len();
        let text = &body[lead..];
        let base = MISTRAL_CALLS.len() + lead;
        let incomplete = || {
            if finish {
                Step::Fail(self.block.len())
            } else {
                Step::Wait
            }
        };
        if text.is_empty() {
            return incomplete();
        }
        if text.starts_with('[') {
            return match first_json(text) {
                Json::Done(Value::Array(items), n) => {
                    let calls: Option<Vec<ToolCall>> =
                        items.iter().map(|v| call_from_json(v, false)).collect();
                    match calls {
                        Some(c) if !c.is_empty() => Step::Calls(c, base + n),
                        _ => Step::Fail(self.block.len()),
                    }
                }
                Json::Partial => incomplete(),
                _ => Step::Fail(self.block.len()),
            };
        }
        let Some(ai) = text.find(MISTRAL_ARGS) else {
            let plausible = text
                .chars()
                .all(|c| is_name_char(c) || c == '[' || c == ']');
            return if plausible {
                incomplete()
            } else {
                Step::Fail(self.block.len())
            };
        };
        let head = &text[..ai];
        let name = head.split_once(MISTRAL_ID).map_or(head, |(n, _)| n);
        if name.is_empty() || !name.chars().all(is_name_char) {
            return Step::Fail(self.block.len());
        }
        let after = &text[ai + MISTRAL_ARGS.len()..];
        let skipped = after.len() - after.trim_start().len();
        match first_json(&after[skipped..]) {
            Json::Done(args @ Value::Object(_), n) => {
                let used = base + ai + MISTRAL_ARGS.len() + skipped + n;
                Step::Calls(vec![make_call(name, args)], used)
            }
            Json::Partial => incomplete(),
            _ => Step::Fail(self.block.len()),
        }
    }
}

enum Step {
    Wait,
    /// Calls parsed; this many bytes of the block are used up.
    Calls(Vec<ToolCall>, usize),
    /// Not a call; this many bytes of the block are plain text.
    Fail(usize),
}

enum Json {
    /// A complete value and the bytes it took.
    Done(Value, usize),
    /// A valid prefix of a value.
    Partial,
    Invalid,
}

fn first_json(s: &str) -> Json {
    let mut it = serde_json::Deserializer::from_str(s).into_iter::<Value>();
    match it.next() {
        Some(Ok(v)) => Json::Done(v, it.byte_offset()),
        Some(Err(e)) if e.is_eof() => Json::Partial,
        None => Json::Partial,
        Some(Err(_)) => Json::Invalid,
    }
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')
}

/// Length of the longest suffix of `s` that is a proper prefix of one of `markers`.
fn hold_len(s: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        let max = (m.len() - 1).min(s.len());
        for n in (best + 1..=max).rev() {
            if s.is_char_boundary(s.len() - n) && m.starts_with(&s[s.len() - n..]) {
                best = n;
                break;
            }
        }
    }
    best
}

fn new_call_id() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("call_{}", &id[..24])
}

fn make_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: new_call_id(),
        name: name.to_owned(),
        arguments,
    }
}

/// `{"name": ..., "arguments"|"parameters": {...}}`. Arguments given as a JSON string are decoded.
fn call_from_json(v: &Value, require_args: bool) -> Option<ToolCall> {
    let o = v.as_object()?;
    let name = o.get("name")?.as_str().filter(|n| !n.is_empty())?;
    let args = match o.get("arguments").or_else(|| o.get("parameters")) {
        None if require_args => return None,
        None => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
            .ok()
            .filter(Value::is_object)?,
        Some(a) if a.is_object() => a.clone(),
        Some(_) => return None,
    };
    Some(make_call(name, args))
}

/// Whole output is one or more JSON calls, optionally after `<|python_tag|>`, split by `;`.
fn parse_llama(block: &str) -> Option<Vec<ToolCall>> {
    let mut rest = block.trim_start();
    rest = rest.strip_prefix(PYTHON_TAG).unwrap_or(rest);
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ';');
        if rest.is_empty() {
            break;
        }
        let Json::Done(v, n) = first_json(rest) else {
            return None;
        };
        calls.push(call_from_json(&v, true)?);
        rest = &rest[n..];
    }
    (!calls.is_empty()).then_some(calls)
}

/// Qwen3-Coder: `<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>`.
fn parse_xml_call(content: &str, tools: &[ToolDef]) -> Option<ToolCall> {
    let rest = content.trim().strip_prefix("<function=")?;
    let gt = rest.find('>')?;
    let name = rest[..gt].trim();
    if name.is_empty() {
        return None;
    }
    let schema = tools.iter().find(|t| t.name == name).map(|t| &t.parameters);
    let mut body = &rest[gt + 1..];
    let mut args = Map::new();
    loop {
        body = body.trim_start();
        let Some(p) = body.strip_prefix("<parameter=") else {
            let tail = body.trim();
            if tail.is_empty() || tail == "</function>" {
                break;
            }
            return None;
        };
        let gt = p.find('>')?;
        let pname = p[..gt].trim();
        let p = &p[gt + 1..];
        let end = p.find(PARAM_CLOSE)?;
        let raw = &p[..end];
        let raw = raw
            .strip_prefix("\r\n")
            .or_else(|| raw.strip_prefix('\n'))
            .unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let ty = schema.and_then(|s| s.get("properties")?.get(pname)?.get("type"));
        args.insert(pname.to_owned(), convert_param(raw, ty));
        body = &p[end + PARAM_CLOSE.len()..];
    }
    Some(make_call(name, Value::Object(args)))
}

/// Parameter values are text; use the schema type to turn them into JSON, else keep the string.
fn convert_param(raw: &str, ty: Option<&Value>) -> Value {
    let types: Vec<&str> = match ty {
        Some(Value::String(s)) => vec![s.as_str()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    if types.is_empty() || types.contains(&"string") {
        return Value::String(raw.to_owned());
    }
    let parsed = serde_json::from_str::<Value>(raw.trim()).ok();
    let fits = |v: &Value| {
        types.iter().any(|t| match *t {
            "integer" => v.is_i64() || v.is_u64(),
            "number" => v.is_number(),
            "boolean" => v.is_boolean(),
            "object" => v.is_object(),
            "array" => v.is_array(),
            "null" => v.is_null(),
            _ => false,
        })
    };
    match parsed {
        Some(v) if fits(&v) => v,
        _ => Value::String(raw.to_owned()),
    }
}

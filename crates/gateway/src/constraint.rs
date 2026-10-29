//! Structured output and forced tool calls as llguidance token constraints (ADR 0028).
//!
//! Every constraint is a Lark grammar: an optional think block, then the payload (JSON for
//! structured output, the model family's call syntax for tool calls). JSON parts use
//! llguidance's `%json { schema }`.

use crate::chat::{OutputFormat, ToolChoice, ToolDef};
use crate::engine::{TokenConstraint, TokenId, TokenMask, Vocab};
use crate::toolcall::ToolFormat;
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use serde_json::{Value, json};
use std::sync::Arc;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

/// How the model's output may start with respect to a `<think>` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reasoning {
    /// No reasoning block.
    None,
    /// An optional `<think>...</think>` block may come first.
    Optional,
    /// Output starts inside a think block (the prompt ended with `<think>`); it must close it first.
    Open,
}

/// Builds constraints for one model's vocabulary. Built once per model and cached.
pub struct ConstraintFactory {
    factory: ParserFactory,
    /// All end-of-generation ids; the grammar knows only the first as EOS.
    eog: Vec<TokenId>,
    /// Special tokens by text, for grammar markers that are special tokens (Mistral).
    special: Vec<(String, TokenId)>,
}

impl ConstraintFactory {
    pub fn new(vocab: &Vocab) -> Result<Self, String> {
        let eos = *vocab.eog.first().ok_or("vocabulary has no end-of-generation token")?;
        let mut words = vocab.tokens.clone();
        let mut special = Vec::new();
        for &id in &vocab.special {
            if let Some(w) = words.get_mut(id as usize) {
                special.push((String::from_utf8_lossy(w).into_owned(), id));
                // toktrie marks special tokens with a 0xFF prefix; they never match text.
                w.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
            }
        }
        let info = TokRxInfo::new(words.len() as u32, eos);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        factory.quiet();
        Ok(Self {
            factory,
            eog: vocab.eog.clone(),
            special,
        })
    }

    /// Output is JSON matching `format`, after a think block as `reasoning` says.
    pub fn json(
        &self,
        format: &OutputFormat,
        reasoning: Reasoning,
    ) -> Result<Box<dyn TokenConstraint>, String> {
        let schema = match format {
            OutputFormat::JsonObject => json!({"type": "object"}),
            OutputFormat::JsonSchema(s) => s.clone(),
        };
        self.build(&format!("body: %json {schema}\n"), reasoning)
    }

    /// Output is one or more tool calls in `format` syntax (one if `parallel` is false), for
    /// `choice` `Required` (any of `tools`) or `Named`.
    pub fn tool_calls(
        &self,
        format: ToolFormat,
        tools: &[ToolDef],
        choice: &ToolChoice,
        parallel: bool,
        reasoning: Reasoning,
    ) -> Result<Box<dyn TokenConstraint>, String> {
        let allowed: Vec<&ToolDef> = match choice {
            ToolChoice::Named(n) => tools.iter().filter(|t| &t.name == n).collect(),
            _ => tools.iter().collect(),
        };
        if allowed.is_empty() {
            return Err("no tool to call".into());
        }
        let repeat = if parallel { "+" } else { "" };
        let body = match format {
            ToolFormat::Hermes => format!(
                "body: (call /[ \\t\\r\\n]*/){repeat}\ncall: \"<tool_call>\\n\" %json {} \"\\n</tool_call>\"\n",
                call_schema(&allowed, "arguments")
            ),
            ToolFormat::Llama3 => {
                let one = call_schema(&allowed, "parameters");
                if parallel {
                    format!("body: %json {one} (\"; \" %json {one})*\n")
                } else {
                    format!("body: %json {one}\n")
                }
            }
            ToolFormat::Mistral => {
                let mut arr = json!({"type": "array", "items": call_schema(&allowed, "arguments"), "minItems": 1});
                if !parallel {
                    arr["maxItems"] = json!(1);
                }
                format!("body: {} %json {arr}\n", self.marker("[TOOL_CALLS]"))
            }
            ToolFormat::QwenXml => qwen_xml_grammar(&allowed, repeat),
        };
        self.build(&body, reasoning)
    }

    /// A grammar literal for `text`: the special token if the vocabulary has one, else the text.
    fn marker(&self, text: &str) -> String {
        match self.special.iter().find(|(t, _)| t == text) {
            Some((_, id)) => format!("<[{id}]>"),
            None => lark_string(text),
        }
    }

    fn build(&self, body: &str, reasoning: Reasoning) -> Result<Box<dyn TokenConstraint>, String> {
        let start = match reasoning {
            Reasoning::None => "start: body\n",
            Reasoning::Optional => "start: think? body\nthink: \"<think>\" think_text WS\n",
            Reasoning::Open => "start: think_text WS body\n",
        };
        let think = if reasoning == Reasoning::None {
            ""
        } else {
            "think_text[suffix=\"</think>\"]: /(?s:.*)/\nWS: /[ \\t\\r\\n]*/\n"
        };
        let lark = format!("{start}{think}{body}");
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_lark(lark))
            .map_err(|e| format!("grammar: {e}"))?;
        Ok(Box::new(Constraint {
            matcher: Matcher::new(Ok(parser)),
            eog: self.eog.clone(),
        }))
    }
}

/// Schema of one call object: `{"name": <one of the tools>, <args_key>: <its parameters>}`.
fn call_schema(tools: &[&ToolDef], args_key: &str) -> Value {
    let variants: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "object",
                "properties": {"name": {"const": t.name}, args_key: t.parameters},
                "required": ["name", args_key],
                "additionalProperties": false,
            })
        })
        .collect();
    if variants.len() == 1 {
        variants.into_iter().next().unwrap_or_default()
    } else {
        json!({"anyOf": variants})
    }
}

/// Qwen3-Coder calls: `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
/// Values are free text up to `</parameter>`; names are constrained.
fn qwen_xml_grammar(tools: &[&ToolDef], repeat: &str) -> String {
    let funcs: Vec<String> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let params: Vec<String> = t
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .map(|p| p.keys().map(|k| lark_string(k)).collect())
                .unwrap_or_default();
            let param_rule = if params.is_empty() {
                String::new()
            } else {
                format!(
                    "fparam{i}: \"<parameter=\" ({}) \">\\n\" value \"\\n\"\n",
                    params.join(" | ")
                )
            };
            let body = if params.is_empty() {
                format!("func{i}: \"<function=\" {} \">\\n\" \"</function>\"\n", lark_string(&t.name))
            } else {
                format!(
                    "func{i}: \"<function=\" {} \">\\n\" fparam{i}* \"</function>\"\n{param_rule}",
                    lark_string(&t.name)
                )
            };
            body
        })
        .collect();
    let alts: Vec<String> = (0..tools.len()).map(|i| format!("func{i}")).collect();
    format!(
        "body: (call /[ \\t\\r\\n]*/){repeat}\ncall: \"<tool_call>\\n\" ({}) \"\\n</tool_call>\"\nvalue[suffix=\"\\n</parameter>\"]: /(?s:.*)/\n{}",
        alts.join(" | "),
        funcs.concat()
    )
}

/// A Lark string literal (JSON escaping is valid Lark escaping).
fn lark_string(s: &str) -> String {
    Value::String(s.to_owned()).to_string()
}

struct Constraint {
    matcher: Matcher,
    eog: Vec<TokenId>,
}

impl TokenConstraint for Constraint {
    fn mask(&mut self) -> Result<Option<TokenMask>, String> {
        let v = self.matcher.compute_mask_or_eos().map_err(|e| e.to_string())?;
        let mut words = v.as_slice().to_vec();
        // The grammar knows one EOS; any end-of-generation token ends the output equally.
        if self.eog.first().is_some_and(|&e| v.is_allowed(e)) {
            for &id in &self.eog {
                let w = id as usize / 32;
                if w >= words.len() {
                    words.resize(w + 1, 0);
                }
                words[w] |= 1 << (id % 32);
            }
        }
        Ok(Some(TokenMask { words }))
    }

    fn commit(&mut self, token: TokenId) -> Result<(), String> {
        if self.eog.contains(&token) {
            return Ok(());
        }
        self.matcher
            .consume_token(token)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeEngine, IM_END};
    use crate::engine::Engine;

    fn factory() -> (ConstraintFactory, FakeEngine) {
        let e = FakeEngine::fixed("");
        let v = e.vocab("fake").unwrap();
        (ConstraintFactory::new(&v).unwrap(), e)
    }

    /// Feeds `text` token by token, asserting each token is allowed; returns the final mask.
    fn feed(c: &mut dyn TokenConstraint, e: &FakeEngine, text: &str) -> TokenMask {
        for t in e.tokenize("fake", text).unwrap() {
            let m = c.mask().unwrap().unwrap();
            assert!(m.allows(t), "token {t} ({:?}) rejected after part of {text:?}", char::from_u32(t));
            c.commit(t).unwrap();
        }
        c.mask().unwrap().unwrap()
    }

    #[test]
    fn json_schema_mask() {
        let (f, e) = factory();
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}},
                            "required": ["a"], "additionalProperties": false});
        let mut c = f.json(&OutputFormat::JsonSchema(schema), Reasoning::None).unwrap();
        let m = c.mask().unwrap().unwrap();
        assert!(!m.allows(u32::from(b'S')));
        assert!(!m.allows(IM_END));
        let end = feed(c.as_mut(), &e, "{\"a\": 12}");
        assert!(end.allows(IM_END));
    }

    #[test]
    fn json_after_open_think() {
        let (f, e) = factory();
        let mut c = f.json(&OutputFormat::JsonObject, Reasoning::Open).unwrap();
        let end = feed(c.as_mut(), &e, "let me think { not json\n</think>\n\n{\"k\": 1}");
        assert!(end.allows(IM_END));
    }

    #[test]
    fn optional_think_then_json() {
        let (f, e) = factory();
        let mut c = f.json(&OutputFormat::JsonObject, Reasoning::Optional).unwrap();
        feed(c.as_mut(), &e, "<think>hm</think>\n{}");
        let mut c = f.json(&OutputFormat::JsonObject, Reasoning::Optional).unwrap();
        let end = feed(c.as_mut(), &e, "{}");
        assert!(end.allows(IM_END));
    }

    fn tools() -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "get_weather".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
            },
            ToolDef {
                name: "now".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {}}),
            },
        ]
    }

    #[test]
    fn hermes_tool_call() {
        let (f, e) = factory();
        let t = tools();
        let mut c = f
            .tool_calls(ToolFormat::Hermes, &t, &ToolChoice::Required, true, Reasoning::None)
            .unwrap();
        let m = c.mask().unwrap().unwrap();
        assert!(!m.allows(u32::from(b'I')));
        let end = feed(
            c.as_mut(),
            &e,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>",
        );
        assert!(end.allows(IM_END));
    }

    #[test]
    fn named_tool_excludes_others() {
        let (f, e) = factory();
        let t = tools();
        let mut c = f
            .tool_calls(ToolFormat::Hermes, &t, &ToolChoice::Named("now".into()), false, Reasoning::None)
            .unwrap();
        let m = feed(c.as_mut(), &e, "<tool_call>\n{\"name\": \"");
        assert!(m.allows(u32::from(b'n')));
        assert!(!m.allows(u32::from(b'g')));
    }

    #[test]
    fn qwen_xml_tool_call() {
        let (f, e) = factory();
        let t = tools();
        let mut c = f
            .tool_calls(ToolFormat::QwenXml, &t, &ToolChoice::Required, false, Reasoning::None)
            .unwrap();
        let end = feed(
            c.as_mut(),
            &e,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew <York>\n</parameter>\n</function>\n</tool_call>",
        );
        assert!(end.allows(IM_END));
    }

    #[test]
    fn llama3_and_mistral_calls() {
        let (f, e) = factory();
        let t = tools();
        let mut c = f
            .tool_calls(ToolFormat::Llama3, &t, &ToolChoice::Required, false, Reasoning::None)
            .unwrap();
        let end = feed(c.as_mut(), &e, "{\"name\": \"now\", \"parameters\": {}}");
        assert!(end.allows(IM_END));
        let mut c = f
            .tool_calls(ToolFormat::Mistral, &t, &ToolChoice::Required, true, Reasoning::None)
            .unwrap();
        let end = feed(c.as_mut(), &e, "[TOOL_CALLS][{\"name\": \"now\", \"arguments\": {}}]");
        assert!(end.allows(IM_END));
    }
}

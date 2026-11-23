//! llama-server's sampling request fields (groups A and B of docs/notes/llamacpp-parity.md),
//! shared by the OpenAI and Anthropic handlers. Each field is honoured or rejected with 400 (ADR
//! 0083 decision 8). Soft-limit fields are clamped as llama-server does; hard limits are rejected.
//!
//! Fields llama-server has and this server does not implement are rejected when set to a
//! non-default value (`UNSUPPORTED`), so a client that sends the defaults still works. A few are
//! accepted and have no effect, each for a stated reason (`cache_prompt`, `n_cache_reuse`,
//! `return_progress`, `sse_ping_interval`, `verbose`: prefix reuse is automatic, streams
//! keep-alive on their own). `id_slot`, `n_keep` and `n_discard` act with `--context-shift` and
//! the slot routes (ADR 0090): the slot a request runs on, the tokens a context shift keeps and
//! drops.

use crate::engine::{GrammarSource, SamplerKind, SamplingExtras};
use crate::error::GatewayError;
use serde::Deserialize;
use serde_json::{Map, Value};

/// At most this many distinct DRY breakers (`LLAMA_DRY_MAX_SEQ_BREAKERS`).
const DRY_MAX_BREAKERS: usize = 64;

/// The fields as sent; flatten into a request struct and call [`SamplingFields::extras`].
#[derive(Debug, Default, Deserialize)]
pub struct SamplingFields {
    min_p: Option<f32>,
    typical_p: Option<f32>,
    top_n_sigma: Option<f32>,
    xtc_probability: Option<f32>,
    xtc_threshold: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<i64>,
    dry_penalty_last_n: Option<i64>,
    dry_sequence_breakers: Option<Value>,
    logit_bias: Option<Value>,
    ignore_eos: Option<bool>,
    mirostat: Option<i64>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    dynatemp_range: Option<f32>,
    dynatemp_exponent: Option<f32>,
    min_keep: Option<i64>,
    n_keep: Option<i64>,
    n_discard: Option<i64>,
    id_slot: Option<i64>,
    samplers: Option<Value>,
    grammar: Option<Value>,
    json_schema: Option<Value>,
    return_tokens: Option<Value>,
    timings_per_token: Option<Value>,
    response_fields: Option<Value>,
    t_max_predict_ms: Option<Value>,
    n_indent: Option<Value>,
    lora: Option<Value>,
    adaptive_target: Option<Value>,
    post_sampling_probs: Option<Value>,
    grammar_lazy: Option<Value>,
}

/// Fields that are rejected when they differ from llama-server's default, with the reason.
const UNSUPPORTED: [(&str, &str); 9] = [
    ("return_tokens", "token ids are not returned"),
    (
        "timings_per_token",
        "timings are reported once per response",
    ),
    ("response_fields", "responses are not filtered"),
    ("t_max_predict_ms", "there is no time limit on prediction"),
    ("n_indent", "there is no indentation limit"),
    ("lora", "LoRA adapters are not supported"),
    ("adaptive_target", "the adaptive-p sampler is not supported"),
    (
        "post_sampling_probs",
        "only probabilities from the raw logits are reported",
    ),
    ("grammar_lazy", "lazy grammars are not supported"),
];

fn is_default(name: &str, v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => true,
        Value::Bool(true) => false,
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            // `adaptive_target` is off below zero, the others at zero.
            if name == "adaptive_target" {
                x < 0.0
            } else {
                x == 0.0
            }
        }
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

fn bad(msg: impl Into<String>) -> GatewayError {
    GatewayError::InvalidRequest(msg.into())
}

fn unit(v: Option<f32>) -> Option<f32> {
    v.map(|x| x.clamp(0.0, 1.0))
}

fn int(name: &str, v: Option<i64>) -> Result<Option<i32>, GatewayError> {
    match v {
        None => Ok(None),
        Some(x) if (0..=i64::from(i32::MAX)).contains(&x) => Ok(Some(x as i32)),
        Some(x) => Err(bad(format!(
            "{name}: value must be between 0 and {}, but got {x}",
            i32::MAX
        ))),
    }
}

fn logit_bias(v: &Value) -> Result<Vec<(i32, f32)>, GatewayError> {
    let bias = |v: &Value| -> Result<f32, GatewayError> {
        match v {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) as f32),
            Value::Bool(false) => Ok(f32::NEG_INFINITY),
            _ => Err(bad(
                "logit_bias: a bias is a number, or false to ban the token",
            )),
        }
    };
    let token = |v: &Value| -> Result<i32, GatewayError> {
        let id = match v {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse::<i64>().ok(),
            _ => None,
        };
        id.and_then(|i| i32::try_from(i).ok())
            .ok_or_else(|| bad("logit_bias: tokens are integer ids; text keys are not supported"))
    };
    match v {
        Value::Null => Ok(Vec::new()),
        Value::Array(a) => a
            .iter()
            .map(|e| match e.as_array().map(Vec::as_slice) {
                Some([t, b]) => Ok((token(t)?, bias(b)?)),
                _ => Err(bad("logit_bias: array entries are [token, bias] pairs")),
            })
            .collect(),
        Value::Object(m) => m
            .iter()
            .map(|(k, b)| Ok((token(&Value::String(k.clone()))?, bias(b)?)))
            .collect(),
        _ => Err(bad("logit_bias must be an array of pairs or an object")),
    }
}

fn samplers(v: &Value) -> Result<Vec<SamplerKind>, GatewayError> {
    let unsupported = |n: &str| {
        bad(format!(
            "samplers: '{n}' is not supported (known: penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature)"
        ))
    };
    match v {
        Value::Array(a) => a
            .iter()
            .map(|n| {
                let n = n
                    .as_str()
                    .ok_or_else(|| bad("samplers must hold strings"))?;
                SamplerKind::from_name(n).ok_or_else(|| unsupported(n))
            })
            .collect(),
        Value::String(s) => s
            .chars()
            .map(|c| SamplerKind::from_char(c).ok_or_else(|| unsupported(&c.to_string())))
            .collect(),
        _ => Err(bad(
            "samplers must be an array of names or a string of letters",
        )),
    }
}

fn breakers(v: &Value) -> Result<Vec<String>, GatewayError> {
    let msg = "dry_sequence_breakers must be a non-empty array of strings";
    let a = v
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| bad(msg))?;
    let mut out: Vec<String> = Vec::new();
    for b in a {
        let s = b.as_str().ok_or_else(|| bad(msg))?;
        if !out.iter().any(|o| o == s) {
            out.push(s.to_owned());
        }
    }
    if out.len() > DRY_MAX_BREAKERS {
        return Err(bad(format!(
            "dry_sequence_breakers must contain at most {DRY_MAX_BREAKERS} distinct entries"
        )));
    }
    Ok(out)
}

impl SamplingFields {
    /// Both `grammar` and `json_schema` are set (chat requests reject that, as llama-server does).
    pub fn has_grammar_and_schema(&self) -> bool {
        let set = |v: &Option<Value>| v.as_ref().is_some_and(|v| !is_default("", v));
        set(&self.grammar) && set(&self.json_schema)
    }

    fn reject_unsupported(&self) -> Result<(), GatewayError> {
        let fields: [(&str, &Option<Value>); 9] = [
            ("return_tokens", &self.return_tokens),
            ("timings_per_token", &self.timings_per_token),
            ("response_fields", &self.response_fields),
            ("t_max_predict_ms", &self.t_max_predict_ms),
            ("n_indent", &self.n_indent),
            ("lora", &self.lora),
            ("adaptive_target", &self.adaptive_target),
            ("post_sampling_probs", &self.post_sampling_probs),
            ("grammar_lazy", &self.grammar_lazy),
        ];
        for (name, v) in fields {
            if let Some(v) = v
                && !is_default(name, v)
            {
                let why = UNSUPPORTED
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map_or("", |(_, w)| *w);
                return Err(bad(format!("{name} is not supported: {why}")));
            }
        }
        Ok(())
    }

    /// `grammar` (GBNF text) or `json_schema`; the grammar wins when both are sent, as in
    /// llama-server's completion schema.
    fn grammar(&self) -> Result<Option<GrammarSource>, GatewayError> {
        let set = |v: &Option<Value>| v.clone().filter(|v| !is_default("", v));
        match (set(&self.grammar), set(&self.json_schema)) {
            (Some(Value::String(g)), _) => Ok(Some(GrammarSource::Gbnf(g))),
            (Some(_), _) => Err(bad("grammar must be a string")),
            (None, Some(mut schema)) => {
                // An empty schema means any object.
                if schema.as_object().is_some_and(Map::is_empty) {
                    schema = serde_json::json!({"type": "object"});
                }
                Ok(Some(GrammarSource::JsonSchema(schema.to_string())))
            }
            (None, None) => Ok(None),
        }
    }

    pub fn extras(self) -> Result<SamplingExtras, GatewayError> {
        self.reject_unsupported()?;
        let grammar = self.grammar()?;
        if let Some(r) = self.repeat_penalty
            && (!r.is_finite() || r <= 0.0 || !(1.0 / r).is_finite())
        {
            return Err(bad("repeat_penalty must be finite and greater than 0"));
        }
        let none_if_null = |v: Option<Value>| v.filter(|v| !v.is_null());
        let dry_breakers = none_if_null(self.dry_sequence_breakers)
            .map(|v| breakers(&v))
            .transpose()?;
        Ok(SamplingExtras {
            min_p: unit(self.min_p),
            typical_p: self.typical_p,
            top_n_sigma: self.top_n_sigma,
            xtc_probability: unit(self.xtc_probability),
            xtc_threshold: unit(self.xtc_threshold),
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: int("repeat_last_n", self.repeat_last_n)?,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            dry_multiplier: self.dry_multiplier,
            // Below 1.0 llama-server falls back to its default.
            dry_base: self.dry_base.filter(|b| *b >= 1.0),
            dry_allowed_length: int("dry_allowed_length", self.dry_allowed_length)?,
            dry_penalty_last_n: int("dry_penalty_last_n", self.dry_penalty_last_n)?,
            dry_sequence_breakers: dry_breakers,
            logit_bias: none_if_null(self.logit_bias)
                .map(|v| logit_bias(&v))
                .transpose()?
                .unwrap_or_default(),
            ignore_eos: self.ignore_eos,
            mirostat: self.mirostat.map(|m| m.clamp(0, 2) as i32),
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            dynatemp_range: self.dynatemp_range,
            dynatemp_exponent: self.dynatemp_exponent,
            min_keep: int("min_keep", self.min_keep)?.map(|k| k as u32),
            samplers: none_if_null(self.samplers)
                .map(|v| samplers(&v))
                .transpose()?,
            grammar,
            n_keep: self
                .n_keep
                .map(|k| i32::try_from(k).unwrap_or(if k < 0 { -1 } else { i32::MAX })),
            n_discard: self
                .n_discard
                .filter(|&d| d > 0)
                .map(|d| u32::try_from(d).unwrap_or(u32::MAX)),
            // a negative id means any slot
            slot: self.id_slot.and_then(|s| usize::try_from(s).ok()),
            ..SamplingExtras::default()
        })
    }
}

/// A request `seed`: negative means random, like llama-server.
pub fn seed(v: Option<i64>) -> Option<u64> {
    v.and_then(|s| u64::try_from(s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Result<SamplingExtras, GatewayError> {
        serde_json::from_value::<SamplingFields>(v)
            .map_err(|e| bad(e.to_string()))?
            .extras()
    }

    // hardware: none
    #[test]
    fn slot_and_shift_fields_reach_the_extras() {
        let e = parse(json!({"id_slot": 1, "n_keep": -1, "n_discard": 7})).unwrap();
        assert_eq!(
            (e.slot, e.n_keep, e.n_discard),
            (Some(1), Some(-1), Some(7))
        );
        let e = parse(json!({"id_slot": -1, "n_discard": 0})).unwrap();
        assert_eq!((e.slot, e.n_keep, e.n_discard), (None, None, None));
    }

    #[test]
    fn nothing_sent_is_all_default() {
        assert_eq!(parse(json!({})).unwrap(), SamplingExtras::default());
    }

    #[test]
    fn logit_bias_takes_both_forms_and_false() {
        let want = vec![(5, -2.0), (7, f32::NEG_INFINITY)];
        let m = parse(json!({"logit_bias": {"5": -2, "7": false}})).unwrap();
        assert_eq!(m.logit_bias, want);
        let a = parse(json!({"logit_bias": [[5, -2.0], [7, false]]})).unwrap();
        assert_eq!(a.logit_bias, want);
        assert!(parse(json!({"logit_bias": {"hello": 1}})).is_err());
        assert!(parse(json!({"logit_bias": [[1, true]]})).is_err());
    }

    #[test]
    fn limits_clamp_or_reject_like_llama_server() {
        let e = parse(json!({"min_p": 3.0, "xtc_probability": -1, "mirostat": 9, "dry_base": 0.5}))
            .unwrap();
        assert_eq!(e.min_p, Some(1.0));
        assert_eq!(e.xtc_probability, Some(0.0));
        assert_eq!(e.mirostat, Some(2));
        assert_eq!(e.dry_base, None);
        assert!(parse(json!({"repeat_last_n": -1})).is_err());
        assert!(parse(json!({"repeat_penalty": 0})).is_err());
        assert!(parse(json!({"dry_sequence_breakers": []})).is_err());
    }

    #[test]
    fn samplers_by_name_and_by_letter() {
        let a = parse(json!({"samplers": ["top-k", "Min_P", "temp"]})).unwrap();
        let b = parse(json!({"samplers": "kmt"})).unwrap();
        assert_eq!(a.samplers, b.samplers);
        assert_eq!(
            a.samplers,
            Some(vec![
                SamplerKind::TopK,
                SamplerKind::MinP,
                SamplerKind::Temperature
            ])
        );
        assert!(parse(json!({"samplers": ["adaptive_p"]})).is_err());
    }

    #[test]
    fn negative_seed_is_random() {
        assert_eq!(seed(Some(-1)), None);
        assert_eq!(seed(Some(7)), Some(7));
    }
}

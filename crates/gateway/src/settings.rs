//! The `generation_settings` of llama-server (`task_params::to_json`), for `/props` and the
//! native completion response (ADR 0089).
//!
//! Values are the server's defaults (llama.cpp's own, replaced by what `tetraeder serve` was
//! started with, [`crate::gateway::GatewayConfig::defaults`]) with the request's own values on
//! top. Fields the engine has no notion of carry llama-server's default values.

use crate::gateway::GatewayConfig;
use serde_json::{Map, Value, json};

/// Request fields whose value, when it has the type of the default, is reported as sent.
const FROM_REQUEST: [&str; 21] = [
    "temperature",
    "dynatemp_range",
    "dynatemp_exponent",
    "top_k",
    "top_p",
    "min_p",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "typical_p",
    "repeat_last_n",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "mirostat",
    "ignore_eos",
    "min_keep",
];

/// Fields of `/props` `default_generation_settings.params` (llama-server's `only_metrics`
/// set): the others describe one request.
const REQUEST_ONLY: [&str; 6] = [
    "stop",
    "logit_bias",
    "grammar",
    "grammar_lazy",
    "grammar_triggers",
    "preserved_tokens",
];

/// llama.cpp's defaults under llama-server's names.
fn builtin() -> Map<String, Value> {
    let entries: [(&str, Value); 49] = [
        ("seed", json!(4_294_967_295u64)),
        ("temperature", json!(0.8)),
        ("dynatemp_range", json!(0.0)),
        ("dynatemp_exponent", json!(1.0)),
        ("top_k", json!(40)),
        ("top_p", json!(0.95)),
        ("min_p", json!(0.05)),
        ("top_n_sigma", json!(-1.0)),
        ("xtc_probability", json!(0.0)),
        ("xtc_threshold", json!(0.1)),
        ("typical_p", json!(1.0)),
        ("repeat_last_n", json!(64)),
        ("repeat_penalty", json!(1.0)),
        ("presence_penalty", json!(0.0)),
        ("frequency_penalty", json!(0.0)),
        ("dry_multiplier", json!(0.0)),
        ("dry_base", json!(1.75)),
        ("dry_allowed_length", json!(2)),
        ("dry_penalty_last_n", json!(-1)),
        ("dry_sequence_breakers", json!(["\n", ":", "\"", "*"])),
        ("mirostat", json!(0)),
        ("mirostat_tau", json!(5.0)),
        ("mirostat_eta", json!(0.1)),
        ("adaptive_target", json!(-1.0)),
        ("adaptive_decay", json!(0.9)),
        ("stop", json!([])),
        ("max_tokens", json!(-1)),
        ("n_predict", json!(-1)),
        ("n_keep", json!(0)),
        ("n_discard", json!(0)),
        ("ignore_eos", json!(false)),
        ("stream", json!(true)),
        ("logit_bias", json!([])),
        ("n_probs", json!(0)),
        ("min_keep", json!(0)),
        ("grammar", json!("")),
        ("grammar_lazy", json!(false)),
        ("grammar_triggers", json!([])),
        ("preserved_tokens", json!([])),
        ("chat_format", json!("Content-only")),
        ("reasoning_format", json!("auto")),
        ("reasoning_in_content", json!(false)),
        ("generation_prompt", json!("")),
        (
            "samplers",
            json!([
                "penalties",
                "dry",
                "top_n_sigma",
                "top_k",
                "typ_p",
                "top_p",
                "min_p",
                "xtc",
                "temperature"
            ]),
        ),
        ("speculative.types", json!("none")),
        ("timings_per_token", json!(false)),
        ("post_sampling_probs", json!(false)),
        ("backend_sampling", json!(false)),
        ("lora", json!([])),
    ];
    entries
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
}

/// The server's defaults: llama.cpp's, then what the server was started with.
pub fn defaults(cfg: &GatewayConfig) -> Map<String, Value> {
    let mut m = builtin();
    for (k, v) in &cfg.defaults {
        m.insert(k.clone(), v.clone());
    }
    m
}

/// What the request itself decided.
pub struct Request<'a> {
    /// The body as sent (`Null` for `/props`).
    pub body: &'a Value,
    pub n_predict: i64,
    pub stream: bool,
    pub stop: &'a [String],
    pub n_probs: u32,
}

fn same_kind(a: &Value, b: &Value) -> bool {
    matches!(
        (a, b),
        (Value::Number(_), Value::Number(_))
            | (Value::Bool(_), Value::Bool(_))
            | (Value::String(_), Value::String(_))
            | (Value::Array(_), Value::Array(_))
    )
}

/// `generation_settings` of a response (`req` given) or the `params` of `/props`.
pub fn params(cfg: &GatewayConfig, req: Option<&Request>) -> Value {
    let mut m = defaults(cfg);
    if let Some(r) = req {
        for k in FROM_REQUEST {
            if let (Some(v), Some(d)) = (r.body.get(k), m.get(k))
                && same_kind(v, d)
            {
                m.insert(k.to_owned(), v.clone());
            }
        }
        // A negative seed asks for a random one, which the defaults already say.
        if let Some(s) = r.body.get("seed").and_then(Value::as_u64) {
            m.insert("seed".into(), json!(s));
        }
        if let Some(Value::Array(a)) = r.body.get("dry_sequence_breakers") {
            m.insert("dry_sequence_breakers".into(), Value::Array(a.clone()));
        }
        m.insert("n_predict".into(), json!(r.n_predict));
        m.insert("max_tokens".into(), json!(r.n_predict));
        m.insert("stream".into(), json!(r.stream));
        m.insert("stop".into(), json!(r.stop));
        m.insert("n_probs".into(), json!(r.n_probs));
        if let Some(g) = r.body.get("grammar").and_then(Value::as_str) {
            m.insert("grammar".into(), json!(g));
        }
    } else {
        for k in REQUEST_ONLY {
            m.remove(k);
        }
        m.remove("dry_sequence_breakers");
    }
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_defaults_override_llama_cpps_and_the_request_overrides_both() {
        let mut cfg = GatewayConfig::default();
        cfg.defaults.insert("top_k".into(), json!(7));
        let body =
            json!({"temperature": 0.2, "top_p": "x", "seed": 5, "dry_sequence_breakers": ["a"]});
        let r = Request {
            body: &body,
            n_predict: 9,
            stream: false,
            stop: &["z".to_owned()],
            n_probs: 2,
        };
        let p = params(&cfg, Some(&r));
        assert_eq!(p["top_k"], 7);
        assert_eq!(p["temperature"], 0.2);
        // a value of the wrong type is not reported
        assert_eq!(p["top_p"], 0.95);
        assert_eq!(p["seed"], 5);
        assert_eq!(p["n_predict"], 9);
        assert_eq!(p["stop"], json!(["z"]));
        assert_eq!(p["dry_sequence_breakers"], json!(["a"]));
        let props = params(&cfg, None);
        assert_eq!(props["top_k"], 7);
        assert!(props.get("stop").is_none());
    }
}

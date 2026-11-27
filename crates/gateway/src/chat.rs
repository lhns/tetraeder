//! API-neutral request and output types. Both API layers map onto these.

use serde::Serialize;
use serde_json::Value;

/// A message in the Hugging Face chat-template shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For `tool` messages: the call this answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For `tool` messages: the function name, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Self::default()
        }
    }
}

/// A function call made by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: Value,
}

/// Serialised as HF templates expect: `{"id", "type": "function", "function": {"name", "arguments"}}`
/// with `arguments` a JSON object, not a string.
impl Serialize for ToolCall {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        })
        .serialize(s)
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON schema of the arguments object.
    pub parameters: Value,
}

/// Serialised in the OpenAI shape that HF templates expect:
/// `{"type": "function", "function": {"name", "description", "parameters"}}`.
impl Serialize for ToolDef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut f = serde_json::Map::new();
        f.insert("name".into(), Value::String(self.name.clone()));
        // llama-server's template input always has a description, empty when the request gave none
        f.insert(
            "description".into(),
            Value::String(self.description.clone().unwrap_or_default()),
        );
        f.insert("parameters".into(), self.parameters.clone());
        serde_json::json!({"type": "function", "function": f}).serialize(s)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    /// Some tool must be called.
    Required,
    /// This tool must be called.
    Named(String),
}

/// Constrained output format.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    /// Any JSON object.
    JsonObject,
    /// JSON matching this schema.
    JsonSchema(Value),
}

/// Why generation ended, before mapping to an API's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// End of generation.
    Stop,
    Length,
    /// A caller stop sequence matched; it is not part of the output.
    StopSequence(String),
    /// At least one tool call was parsed.
    ToolCalls,
}

impl Finish {
    /// OpenAI's `finish_reason` vocabulary, also used by the GenAI semantic conventions.
    pub fn openai_reason(&self) -> &'static str {
        match self {
            Finish::Stop | Finish::StopSequence(_) => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    /// What llama-server reports as `timings`; `None` when the request did not finish.
    pub timings: Option<Timings>,
}

/// The `timings` object of llama-server (`server_slot_stats::to_json`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// Prompt tokens served from the cache.
    pub cache_n: u32,
    /// Prompt tokens processed.
    pub prompt_n: u32,
    pub prompt_ms: f64,
    /// Sampled tokens, the end-of-generation token included.
    pub predicted_n: u32,
    pub predicted_ms: f64,
}

impl Timings {
    pub fn to_json(&self) -> Value {
        // The first token comes from the prompt step, so the rates count the steps after it.
        let steps = f64::from(self.predicted_n.saturating_sub(1));
        let per = |ms: f64, n: f64| if n > 0.0 { ms / n } else { 0.0 };
        let rate = |ms: f64, n: f64| if ms > 0.0 { 1e3 / ms * n } else { 0.0 };
        let prompt_n = f64::from(self.prompt_n);
        serde_json::json!({
            "cache_n": self.cache_n,
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_token_ms": per(self.prompt_ms, prompt_n),
            "prompt_per_second": rate(self.prompt_ms, prompt_n),
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_token_ms": per(self.predicted_ms, steps),
            "predicted_per_second": rate(self.predicted_ms, steps),
        })
    }
}

/// One generated token with its probabilities, as llama-server's `completion_probabilities`
/// entry (`content[]` of `logprobs`).
#[derive(Debug, Clone, PartialEq)]
pub struct LogprobEntry {
    pub id: u32,
    /// The bytes the token added to the output (control tokens add none).
    pub bytes: Vec<u8>,
    pub logprob: f32,
    pub top: Vec<LogprobTop>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogprobTop {
    pub id: u32,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

/// A token's text for JSON: the bytes without an incomplete UTF-8 tail, invalid bytes replaced.
fn token_text(bytes: &[u8]) -> String {
    let end = match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        Err(_) => bytes.len(),
    };
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn finite(x: f32) -> f64 {
    // llama-server writes log(0) as the lowest float, because JSON has no -inf.
    if x.is_finite() {
        f64::from(x)
    } else {
        f64::from(f32::MIN)
    }
}

impl LogprobEntry {
    /// `{"id", "token", "bytes", "logprob", "top_logprobs": [...]}`.
    pub fn to_json(&self) -> Value {
        let top: Vec<Value> = self
            .top
            .iter()
            .map(|t| {
                serde_json::json!({
                    "id": t.id,
                    "token": token_text(&t.bytes),
                    "bytes": t.bytes,
                    "logprob": finite(t.logprob),
                })
            })
            .collect();
        serde_json::json!({
            "id": self.id,
            "token": token_text(&self.bytes),
            "bytes": self.bytes,
            "logprob": finite(self.logprob),
            "top_logprobs": top,
        })
    }
}

/// `{"content": [...]}`, the `logprobs` value of a choice.
pub fn logprobs_json(entries: &[LogprobEntry]) -> Value {
    serde_json::json!({"content": entries.iter().map(LogprobEntry::to_json).collect::<Vec<_>>()})
}

/// One piece of parsed model output, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
}

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
        if let Some(d) = &self.description {
            f.insert("description".into(), Value::String(d.clone()));
        }
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
}

/// One piece of parsed model output, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
}

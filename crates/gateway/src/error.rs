//! Gateway errors and their mapping to each API's error format.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GatewayError {
    /// Malformed or unsupported request (400).
    #[error("{0}")]
    InvalidRequest(String),
    /// The prompt does not fit the model's context or the engine's KV cache (400; OpenAI code
    /// `context_length_exceeded`). The message starts with Anthropic's wording for this error.
    #[error("prompt is too long: {0}")]
    ContextExceeded(String),
    /// Unknown model or route (404).
    #[error("{0}")]
    NotFound(String),
    /// Admission queue full; retry after the given seconds (OpenAI 429, Anthropic 529).
    #[error("server overloaded, retry later")]
    Overloaded { retry_after_secs: u64 },
    /// Engine or internal failure (500).
    #[error("{0}")]
    Internal(String),
}

impl GatewayError {
    /// Value of the `error.type` metric and span attribute.
    pub fn otel_type(&self) -> &'static str {
        match self {
            GatewayError::InvalidRequest(_) | GatewayError::ContextExceeded(_) => "invalid_request",
            GatewayError::NotFound(_) => "not_found",
            GatewayError::Overloaded { .. } => "overloaded",
            GatewayError::Internal(_) => "internal",
        }
    }

    fn retry_after(&self) -> Option<u64> {
        match self {
            GatewayError::Overloaded { retry_after_secs } => Some(*retry_after_secs),
            _ => None,
        }
    }

    /// OpenAI error object: `{"error": {"message", "type", "param", "code"}}`.
    pub fn openai_body(&self) -> serde_json::Value {
        let (ty, code) = match self {
            GatewayError::InvalidRequest(_) => ("invalid_request_error", None),
            GatewayError::ContextExceeded(_) => {
                ("invalid_request_error", Some("context_length_exceeded"))
            }
            GatewayError::NotFound(_) => ("invalid_request_error", Some("model_not_found")),
            GatewayError::Overloaded { .. } => ("rate_limit_error", Some("rate_limit_exceeded")),
            GatewayError::Internal(_) => ("server_error", None),
        };
        json!({"error": {"message": self.to_string(), "type": ty, "param": null, "code": code}})
    }

    pub fn openai_status(&self) -> StatusCode {
        match self {
            GatewayError::InvalidRequest(_) | GatewayError::ContextExceeded(_) => {
                StatusCode::BAD_REQUEST
            }
            GatewayError::NotFound(_) => StatusCode::NOT_FOUND,
            GatewayError::Overloaded { .. } => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Anthropic error type: `invalid_request_error`, `not_found_error`, ...
    pub fn anthropic_type(&self) -> &'static str {
        match self {
            GatewayError::InvalidRequest(_) | GatewayError::ContextExceeded(_) => {
                "invalid_request_error"
            }
            GatewayError::NotFound(_) => "not_found_error",
            GatewayError::Overloaded { .. } => "overloaded_error",
            GatewayError::Internal(_) => "api_error",
        }
    }

    /// Anthropic envelope: `{"type": "error", "error": {"type", "message"}, "request_id"}`.
    pub fn anthropic_body(&self, request_id: &str) -> serde_json::Value {
        json!({
            "type": "error",
            "error": {"type": self.anthropic_type(), "message": self.to_string()},
            "request_id": request_id,
        })
    }

    /// The OpenAI statuses but for the full queue: 529 is Anthropic's "overloaded".
    pub fn anthropic_status(&self) -> StatusCode {
        match self {
            GatewayError::Overloaded { .. } => {
                StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
            }
            _ => self.openai_status(),
        }
    }
}

fn with_retry_after(mut r: Response, e: &GatewayError) -> Response {
    if let Some(s) = e.retry_after() {
        r.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(s));
    }
    r
}

/// A [`GatewayError`] rendered in the OpenAI format.
pub struct OpenAiError(pub GatewayError);

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let r = (self.0.openai_status(), Json(self.0.openai_body())).into_response();
        with_retry_after(r, &self.0)
    }
}

impl From<GatewayError> for OpenAiError {
    fn from(e: GatewayError) -> Self {
        Self(e)
    }
}

/// An error in llama-server's own format (`format_error_response`), used by the routes that are
/// llama-server's and not OpenAI's: `{"error": {"code", "message", "type"}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
}

impl LlamaError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, "not_supported_error", message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_error",
            message,
        )
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Invalid API Key",
        )
    }

    /// The error object, also the body of a mid-stream error frame.
    pub fn object(&self) -> serde_json::Value {
        json!({"code": self.status.as_u16(), "message": self.message, "type": self.kind})
    }
}

impl From<GatewayError> for LlamaError {
    fn from(e: GatewayError) -> Self {
        let kind = match &e {
            GatewayError::InvalidRequest(_) => "invalid_request_error",
            GatewayError::ContextExceeded(_) => "exceed_context_size_error",
            GatewayError::NotFound(_) => "not_found_error",
            GatewayError::Overloaded { .. } => "unavailable_error",
            GatewayError::Internal(_) => "server_error",
        };
        // Status codes are the OpenAI ones but for the full queue, which llama-server answers
        // with 503.
        let status = match &e {
            GatewayError::Overloaded { .. } => StatusCode::SERVICE_UNAVAILABLE,
            _ => e.openai_status(),
        };
        Self::new(status, kind, e.to_string())
    }
}

/// A 400 in llama-server's error format, for the routes that are its own.
pub(crate) fn bad(msg: impl Into<String>) -> LlamaError {
    LlamaError::invalid(msg)
}

impl IntoResponse for LlamaError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.object()}))).into_response()
    }
}

/// A [`GatewayError`] rendered in the Anthropic format, with its request id.
pub struct AnthropicError(pub GatewayError, pub String);

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let r = (
            self.0.anthropic_status(),
            Json(self.0.anthropic_body(&self.1)),
        )
            .into_response();
        let mut r = with_retry_after(r, &self.0);
        if let Ok(v) = HeaderValue::from_str(&self.1) {
            r.headers_mut().insert("request-id", v);
        }
        r
    }
}

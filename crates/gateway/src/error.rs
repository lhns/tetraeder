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
            GatewayError::InvalidRequest(_) => "invalid_request",
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
            GatewayError::NotFound(_) => ("invalid_request_error", Some("model_not_found")),
            GatewayError::Overloaded { .. } => ("rate_limit_error", Some("rate_limit_exceeded")),
            GatewayError::Internal(_) => ("server_error", None),
        };
        json!({"error": {"message": self.to_string(), "type": ty, "param": null, "code": code}})
    }

    pub fn openai_status(&self) -> StatusCode {
        match self {
            GatewayError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::NotFound(_) => StatusCode::NOT_FOUND,
            GatewayError::Overloaded { .. } => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Anthropic error type: `invalid_request_error`, `not_found_error`, ...
    pub fn anthropic_type(&self) -> &'static str {
        match self {
            GatewayError::InvalidRequest(_) => "invalid_request_error",
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

    pub fn anthropic_status(&self) -> StatusCode {
        match self {
            GatewayError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::NotFound(_) => StatusCode::NOT_FOUND,
            // 529 is Anthropic's "overloaded" status.
            GatewayError::Overloaded { .. } => {
                StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
            }
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
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

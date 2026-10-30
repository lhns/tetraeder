//! The HTTP router serving both APIs.

use crate::error::{AnthropicError, GatewayError, OpenAiError};
use crate::gateway::{Gateway, new_id};
use crate::{anthropic, openai};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use std::sync::Arc;

/// Request bodies up to this size are accepted (long agent conversations).
pub const BODY_LIMIT: usize = 64 * 1024 * 1024;

/// Anthropic clients send `anthropic-version`; OpenAI clients do not.
fn is_anthropic(h: &HeaderMap) -> bool {
    h.contains_key("anthropic-version")
}

async fn models(State(gw): State<Arc<Gateway>>, h: HeaderMap) -> Response {
    if is_anthropic(&h) {
        anthropic::list_models(&gw).await
    } else {
        openai::list_models(&gw).await
    }
}

async fn model(State(gw): State<Arc<Gateway>>, h: HeaderMap, id: Path<String>) -> Response {
    if is_anthropic(&h) {
        anthropic::get_model(&gw, id).await
    } else {
        openai::get_model(&gw, id).await.into_response()
    }
}

async fn not_found(h: HeaderMap, uri: Uri) -> Response {
    let e = GatewayError::NotFound(format!("no route for {}", uri.path()));
    if is_anthropic(&h) {
        AnthropicError(e, format!("req_{}", new_id())).into_response()
    } else {
        OpenAiError(e).into_response()
    }
}

/// Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
/// `/v1/messages/count_tokens`, `/v1/models`, `/v1/models/{id}`.
pub fn router(gw: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/messages", post(anthropic::messages_handler))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(gw)
}

/// Serves the router on `listener` until the future is dropped or fails.
pub async fn serve(listener: tokio::net::TcpListener, gw: Arc<Gateway>) -> std::io::Result<()> {
    axum::serve(listener, router(gw)).await
}

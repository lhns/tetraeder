//! The HTTP router serving both APIs.

use crate::error::LlamaError;
use crate::error::{AnthropicError, GatewayError, OpenAiError};
use crate::gateway::{Gateway, new_id};
use crate::reqlog::{RequestLog, log_requests};
use crate::{anthropic, embeddings, native, openai, responses, slots};
use axum::Router;
use axum::extract::{DefaultBodyLimit, MatchedPath, Path, Request, State};
use axum::http::{HeaderMap, Uri};
use axum::middleware::{Next, from_fn_with_state};
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

/// Whether `given` is one of `keys`, comparing every byte of every key.
fn key_matches(keys: &[String], given: &str) -> bool {
    let mut ok = false;
    for k in keys {
        let same = k.len() == given.len()
            && k.bytes()
                .zip(given.bytes())
                .fold(0u8, |d, (a, b)| d | (a ^ b))
                == 0;
        ok |= same;
    }
    ok
}

/// `--api-key`: every route but `/health` wants a key in `Authorization` (with or without
/// `Bearer `) or, when that is empty, in `X-Api-Key`, as llama-server's middleware does. It runs
/// before routing, so an unknown path is a 401 too.
async fn require_key(State(gw): State<Arc<Gateway>>, req: Request, next: Next) -> Response {
    let keys = &gw.config().api_keys;
    if keys.is_empty() || matches!(req.uri().path(), "/health" | "/v1/health") {
        return next.run(req).await;
    }
    let given = {
        let h = req.headers();
        let get = |name: &str| {
            h.get(name)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
        };
        let given = match get("authorization") {
            "" => get("x-api-key"),
            a => a,
        };
        given.strip_prefix("Bearer ").unwrap_or(given).to_owned()
    };
    if key_matches(keys, &given) {
        next.run(req).await
    } else {
        LlamaError::unauthorized().into_response()
    }
}

/// Counts every request by matched route (never the raw path: a client picks that) and status.
async fn count_request(State(gw): State<Arc<Gateway>>, req: Request, next: Next) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_owned(), |m| m.as_str().to_owned());
    let resp = next.run(req).await;
    gw.count_request(&route, resp.status().as_u16());
    resp
}

/// Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
/// `/v1/messages/count_tokens`, `/v1/models`, `/v1/models/{id}`, `/v1/responses`, and
/// llama-server's own: `/health`, `/metrics`, `/props`, `/tokenize`, `/detokenize`,
/// `/apply-template`, `/completion`, `/infill`, with `/models`, `/chat/completions` and
/// `/responses` as aliases.
pub fn router(gw: Arc<Gateway>) -> Router {
    router_with(gw, None)
}

/// [`router`] with an optional request log (ADR 0046). The log must also be the gateway's
/// cost sink to fill `usage` and `cost`.
pub fn router_with(gw: Arc<Gateway>, log: Option<Arc<RequestLog>>) -> Router {
    let app = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/messages", post(anthropic::messages_handler))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/v1/models/{id}", get(model))
        .route("/v1/responses", post(responses::responses))
        .route("/responses", post(responses::responses))
        .route("/health", get(native::health))
        .route("/v1/health", get(native::health))
        .route("/metrics", get(native::metrics))
        .route("/props", get(native::props).post(native::post_props))
        .route("/tokenize", post(native::tokenize))
        .route("/detokenize", post(native::detokenize))
        .route("/apply-template", post(native::apply_template))
        .route("/completion", post(native::completion))
        .route("/completions", post(native::completion))
        .route("/infill", post(native::infill))
        .route("/embedding", post(embeddings::embeddings))
        .route("/embeddings", post(embeddings::embeddings))
        .route("/v1/embeddings", post(embeddings::embeddings_oai))
        .route("/rerank", post(embeddings::rerank))
        .route("/reranking", post(embeddings::rerank))
        .route("/v1/rerank", post(embeddings::rerank))
        .route("/v1/reranking", post(embeddings::rerank))
        .route("/slots/{id_slot}", post(slots::slot_action))
        .fallback(not_found)
        .layer(from_fn_with_state(Arc::clone(&gw), require_key))
        .layer(from_fn_with_state(Arc::clone(&gw), count_request))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(gw);
    match log {
        Some(log) => app.layer(axum::middleware::from_fn_with_state(log, log_requests)),
        None => app,
    }
}

/// Serves the router on `listener` until the future is dropped or fails.
pub async fn serve(listener: tokio::net::TcpListener, gw: Arc<Gateway>) -> std::io::Result<()> {
    axum::serve(listener, router(gw)).await
}

/// [`serve`] with a request log.
pub async fn serve_with(
    listener: tokio::net::TcpListener,
    gw: Arc<Gateway>,
    log: Option<Arc<RequestLog>>,
) -> std::io::Result<()> {
    axum::serve(listener, router_with(gw, log)).await
}

//! `/embedding`, `/embeddings`, `/v1/embeddings` and the four rerank routes (ADR 0090), with the
//! bodies and errors of llama-server's `handle_embeddings_impl` and `handle_rerank_impl`.

use crate::engine::{PoolingKind, TokenId};
use crate::error::{LlamaError, bad};
use crate::gateway::Gateway;
use crate::native::{json_body, token_id, tokenize_mixed};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use serde_json::{Map, Value, json};
use std::sync::Arc;

type Result<T> = std::result::Result<T, LlamaError>;

/// Inputs of one embedding request and documents of one rerank request. llama-server has no
/// limit; this server shares its workers with generation.
pub const MAX_ITEMS: usize = 1024;

/// llama.cpp's default `embd_normalize`: euclidean.
const DEFAULT_NORMALIZE: i32 = 2;

fn embeddings_off() -> LlamaError {
    LlamaError::not_supported(
        "This server does not support embeddings. Start it with `--embeddings`",
    )
}

/// `POST /embedding` and `/embeddings`: a list of `{index, embedding}`, `embedding` holding one
/// vector, or one per token without pooling.
pub async fn embeddings(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    embeddings_impl(gw, body, false).await
}

/// `POST /v1/embeddings`: the OpenAI list object.
pub async fn embeddings_oai(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    embeddings_impl(gw, body, true).await
}

async fn embeddings_impl(gw: Arc<Gateway>, body: Bytes, mut oai: bool) -> Result<Json<Value>> {
    let caps = gw.embedding_caps().ok_or_else(embeddings_off)?;
    if oai && caps.pooling == PoolingKind::None {
        return Err(bad(
            "Pooling type 'none' is not OAI compatible. Please use a different pooling type",
        ));
    }
    let body = json_body(&body)?;
    let prompt = if let Some(p) = body.get("input") {
        p
    } else if let Some(p) = body.get("content") {
        // `content` is not OAI: the native shape answers
        oai = false;
        p
    } else {
        return Err(bad("\"input\" or \"content\" must be provided"));
    };
    let mut base64 = false;
    if let Some(f) = body.get("encoding_format") {
        match f.as_str() {
            Some("base64") => base64 = true,
            Some("float") => {}
            _ => {
                return Err(bad(
                    "The format to return the embeddings in. Can be either float or base64",
                ));
            }
        }
    }
    // One generation slot for the request, then the tokenizing off the async runtime.
    let _permit = gw.admit_auxiliary().await?;
    let inputs = {
        let (gw, prompt) = (Arc::clone(&gw), prompt.clone());
        tokio::task::spawn_blocking(move || tokenize_inputs(&gw, &prompt))
            .await
            .map_err(|e| LlamaError::from(crate::error::GatewayError::Internal(e.to_string())))??
    };
    if inputs.is_empty() {
        return Err(bad("\"input\" must not be empty"));
    }
    if inputs.iter().any(Vec::is_empty) {
        return Err(bad("Input content cannot be empty"));
    }
    let normalize = body
        .get("embd_normalize")
        .and_then(Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
        .unwrap_or(DEFAULT_NORMALIZE);

    let mut results = Vec::with_capacity(inputs.len());
    for tokens in inputs {
        let gw = Arc::clone(&gw);
        let out = tokio::task::spawn_blocking(move || gw.embed("", &tokens, normalize))
            .await
            .map_err(|e| LlamaError::from(crate::error::GatewayError::Internal(e.to_string())))??;
        results.push(out);
    }

    if !oai {
        let list: Vec<Value> = results
            .iter()
            .enumerate()
            .map(|(i, r)| json!({"index": i, "embedding": r.vectors}))
            .collect();
        return Ok(Json(Value::Array(list)));
    }
    let mut n_tokens = 0u64;
    let data: Vec<Value> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            n_tokens += u64::from(r.tokens);
            let vec = r.vectors.first().map_or(&[][..], Vec::as_slice);
            if base64 {
                json!({
                    "embedding": base64_floats(vec),
                    "index": i,
                    "object": "embedding",
                    "encoding_format": "base64",
                })
            } else {
                json!({"embedding": vec, "index": i, "object": "embedding"})
            }
        })
        .collect();
    let model = match body.get("model") {
        Some(m) => m.clone(),
        None => json!(gw.model("")?.id),
    };
    Ok(Json(json!({
        "model": model,
        "object": "list",
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
        "data": data,
    })))
}

/// llama-server's `tokenize_input_prompts` for an embedding request: a string, a list of token
/// ids (or of strings and ids) is one input, any other array is a list of inputs, each a string,
/// token ids, mixed, or an object with `prompt_string` or OpenAI `content` parts. Special tokens
/// are added and parsed.
fn tokenize_inputs(gw: &Gateway, prompt: &Value) -> Result<Vec<Vec<TokenId>>> {
    let n_vocab = gw.vocab("")?.len();
    let one = |p: &Value| -> Result<Vec<TokenId>> {
        match p {
            Value::String(_) => tokenize_mixed(gw, p, true, true),
            Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_i64) => {
                items.iter().map(|v| token_id(v, n_vocab)).collect()
            }
            Value::Array(_) => tokenize_mixed(gw, p, true, true),
            Value::Object(o) => {
                if let Some(s) = o.get("prompt_string") {
                    return tokenize_mixed(gw, s, true, true);
                }
                if let Some(parts) = o.get("content") {
                    return tokenize_mixed(gw, &Value::String(content_text(parts)?), true, true);
                }
                Err(bad(ELEMENTS))
            }
            _ => Err(bad(ELEMENTS)),
        }
    };
    match prompt {
        Value::Array(items) if !items.iter().any(Value::is_i64) => {
            if items.len() > MAX_ITEMS {
                return Err(too_many("inputs", items.len()));
            }
            items.iter().map(one).collect()
        }
        p => Ok(vec![one(p)?]),
    }
}

fn too_many(what: &str, n: usize) -> LlamaError {
    bad(format!(
        "too many {what}: {n}, at most {MAX_ITEMS} per request"
    ))
}

const ELEMENTS: &str = "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.";

/// The text of OpenAI content parts (`[{"type": "text", "text": ...}]`) or of a plain string.
fn content_text(parts: &Value) -> Result<String> {
    match parts {
        Value::String(s) => Ok(s.clone()),
        Value::Array(items) => {
            let mut out = String::new();
            for p in items {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.push_str(p.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some(t) => {
                        return Err(bad(format!(
                            "content part of type '{t}' is not supported: this server has no multimodal projector"
                        )));
                    }
                    None => return Err(bad("content parts need a \"type\"")),
                }
            }
            Ok(out)
        }
        _ => Err(bad(
            "\"content\" must be a string or an array of content parts",
        )),
    }
}

/// Base64 of the little-endian f32 bytes, as llama-server's `encoding_format: base64`.
fn base64_floats(v: &[f32]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = u32::from(chunk[0]) << 16
            | u32::from(*chunk.get(1).unwrap_or(&0)) << 8
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// One rerank item (query or document): a string, or an object whose `text` is a string or an
/// array of strings. Images and video need a multimodal projector, which this server has not.
fn rerank_item(item: &Value) -> Result<String> {
    match item {
        Value::String(s) => Ok(s.clone()),
        Value::Object(o) => {
            if o.contains_key("video") || o.contains_key("fps") || o.contains_key("max_frames") {
                return Err(bad(
                    "video/fps/max_frames are not supported for rerank input",
                ));
            }
            if o.get("image").is_some_and(|v| !v.is_null()) {
                return Err(bad("multimodal rerank input requires --mmproj"));
            }
            match o.get("text") {
                Some(Value::String(s)) => Ok(s.clone()),
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|t| {
                        t.as_str()
                            .ok_or_else(|| bad("\"text\" array elements must be strings"))
                    })
                    .collect(),
                None | Some(Value::Null) => Ok(String::new()),
                Some(_) => Err(bad("\"text\" must be a string or an array of strings")),
            }
        }
        _ => Err(bad(
            "rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}",
        )),
    }
}

/// `POST /rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking`: Jina's shape, or TEI's when the
/// body has `texts`.
pub async fn rerank(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    if gw.embedding_caps().map(|c| c.pooling) != Some(PoolingKind::Rank) {
        return Err(LlamaError::not_supported(
            "This server does not support reranking. Start it with `--reranking`",
        ));
    }
    let body = json_body(&body)?;
    let tei = body.get("texts").is_some();
    let query = match body.get("query") {
        Some(q) => rerank_item(q)?,
        None => return Err(bad("\"query\" must be provided")),
    };
    let docs = body.get("documents").or_else(|| body.get("texts"));
    let docs = docs
        .and_then(Value::as_array)
        .filter(|d| !d.is_empty())
        .ok_or_else(|| bad("\"documents\" must be a non-empty array of strings or objects"))?;
    if docs.len() > MAX_ITEMS {
        return Err(too_many("documents", docs.len()));
    }
    let documents = docs.iter().map(rerank_item).collect::<Result<Vec<_>>>()?;
    let top_n = match body.get("top_n").and_then(Value::as_i64) {
        Some(n) if n < 0 => return Err(bad("\"top_n\" must be >= 0")),
        Some(n) => usize::try_from(n).unwrap_or(usize::MAX),
        None => documents.len(),
    };

    let _permit = gw.admit_auxiliary().await?;
    let mut ranks = Vec::with_capacity(documents.len());
    let mut n_tokens = 0u64;
    for (index, doc) in documents.iter().enumerate() {
        let (gw, query, doc) = (Arc::clone(&gw), query.clone(), doc.clone());
        let r = tokio::task::spawn_blocking(move || gw.rerank("", &query, &doc))
            .await
            .map_err(|e| LlamaError::from(crate::error::GatewayError::Internal(e.to_string())))??;
        n_tokens += u64::from(r.tokens);
        ranks.push((index, r.score));
    }
    ranks.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranks.truncate(top_n);

    let label = if tei { "score" } else { "relevance_score" };
    let return_text = tei && body.get("return_text").and_then(Value::as_bool) == Some(true);
    let elements: Vec<Value> = ranks
        .iter()
        .map(|&(index, score)| {
            let mut e = Map::new();
            e.insert("index".into(), json!(index));
            e.insert(label.into(), json!(score));
            if return_text {
                e.insert("text".into(), json!(documents[index]));
            }
            Value::Object(e)
        })
        .collect();
    if tei {
        return Ok(Json(Value::Array(elements)));
    }
    let model = match body.get("model") {
        Some(m) => m.clone(),
        None => json!(gw.model("")?.id),
    };
    Ok(Json(json!({
        "model": model,
        "object": "list",
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
        "results": elements,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn base64_matches_the_standard_alphabet_and_padding() {
        // 1.0f32 is 00 00 80 3f: "AACAPw==" in base64
        assert_eq!(base64_floats(&[1.0]), "AACAPw==");
        assert_eq!(base64_floats(&[]), "");
        // 12 bytes fill the groups of three: no padding
        assert_eq!(base64_floats(&[1.0, 1.0, 1.0]), "AACAPwAAgD8AAIA/");
        // 8 bytes: one padding character
        assert_eq!(base64_floats(&[1.0, 1.0]), "AACAPwAAgD8=");
    }

    // hardware: none
    #[test]
    fn rerank_items_are_strings_or_text_objects() {
        assert_eq!(rerank_item(&json!("a")).unwrap(), "a");
        assert_eq!(rerank_item(&json!({"text": "a"})).unwrap(), "a");
        assert_eq!(rerank_item(&json!({"text": ["a", "b"]})).unwrap(), "ab");
        assert_eq!(rerank_item(&json!({})).unwrap(), "");
        assert!(rerank_item(&json!(3)).is_err());
        assert!(rerank_item(&json!({"text": 3})).is_err());
        assert!(rerank_item(&json!({"text": "a", "image": "http://x"})).is_err());
        assert!(rerank_item(&json!({"text": "a", "fps": 1})).is_err());
    }
}

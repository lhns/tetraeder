//! llama-server's own routes (ADR 0089): `/health`, `/metrics`, `/props`, `/tokenize`,
//! `/detokenize`, `/apply-template`, native `/completion` and `/infill`. Bodies, status codes
//! and the error object follow `tools/server` at the pin.

use crate::chat::{Finish, LogprobEntry, Segment, Usage};
use crate::engine::{FimTokens, TokenId};
use crate::error::LlamaError;
use crate::gateway::{
    Collected, Gateway, GenerateSpec, Generation, Operation, OutEvent, Prompt, tag_generation,
};
use crate::openai::{
    ChatPlan, FINGERPRINT, Merged, chat_plan, choice_count, choice_specs, merge, probs_count, sse,
    stop_list,
};
use crate::sampling::{self, SamplingFields};
use crate::settings;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use futures_util::stream::{self, Stream};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

type Result<T> = std::result::Result<T, LlamaError>;

/// The `n_batch` llama-server's `/infill` fits the prompt into (its default, as `--step-tokens`).
const INFILL_N_BATCH: usize = 2048;

/// Separates extra context chunks when the vocabulary has no FIM separator token.
const CHUNK_SEPARATOR: &str = "\n\n--- snippet ---\n\n";

fn json_body(body: &Bytes) -> Result<Value> {
    serde_json::from_slice(body)
        .map_err(|e| LlamaError::invalid(format!("invalid request body: {e}")))
}

fn bad(msg: impl Into<String>) -> LlamaError {
    LlamaError::invalid(msg)
}

// ---- health, metrics, props ----

pub async fn health(State(gw): State<Arc<Gateway>>) -> Result<Json<Value>> {
    if gw.ready() {
        Ok(Json(json!({"status": "ok"})))
    } else {
        Err(LlamaError::unavailable("Loading model"))
    }
}

pub async fn metrics(State(gw): State<Arc<Gateway>>) -> Result<Response> {
    if !gw.config().metrics {
        return Err(LlamaError::not_supported(
            "This server does not support metrics endpoint. Start it with `--metrics`",
        ));
    }
    let (text, started) = gw.render_metrics();
    Ok((
        [
            ("content-type", "text/plain; version=0.0.4".to_owned()),
            ("process-start-time-unix", started.to_string()),
        ],
        text,
    )
        .into_response())
}

pub async fn props(State(gw): State<Arc<Gateway>>) -> Result<Json<Value>> {
    let m = gw.model("")?;
    let cfg = gw.config();
    Ok(Json(json!({
        "default_generation_settings": {
            "params": settings::params(cfg, None),
            "n_ctx": m.context_length,
        },
        "total_slots": cfg.max_concurrent,
        "model_alias": m.id,
        "model_path": m.path.unwrap_or_default(),
        "modalities": {"vision": false, "video": false, "audio": false},
        "endpoint_slots": false,
        "endpoint_props": false,
        "endpoint_metrics": cfg.metrics,
        "ui": false,
        "chat_template": m.chat_template.unwrap_or_default(),
        "bos_token": m.bos_token.unwrap_or_default(),
        "eos_token": m.eos_token.unwrap_or_default(),
        "build_info": FINGERPRINT,
        "is_sleeping": false,
        "cors_proxy_enabled": false,
    })))
}

/// There is no property to change, so `--props` does not exist: always the answer without it.
pub async fn post_props() -> Result<Json<Value>> {
    Err(LlamaError::not_supported(
        "This server does not support changing global properties. Start it with `--props`",
    ))
}

// ---- tokenize, detokenize, apply-template ----

/// A token id from JSON: an integer inside the vocabulary.
fn token_id(v: &Value, n_vocab: usize) -> Result<TokenId> {
    v.as_i64()
        .and_then(|i| TokenId::try_from(i).ok())
        .filter(|&i| (i as usize) < n_vocab)
        .ok_or_else(|| {
            bad(format!(
                "token ids must be integers below {n_vocab}, got {v}"
            ))
        })
}

/// llama-server's `tokenize_mixed`: a string, or an array of strings and token ids. The BOS
/// (`add_special`) goes to the first element only, and only if that is a string.
fn tokenize_mixed(
    gw: &Gateway,
    content: &Value,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<TokenId>> {
    let n_vocab = gw.vocab("")?.len();
    match content {
        Value::String(s) => Ok(gw.tokenize_with("", s, add_special, parse_special)?),
        Value::Array(items) => {
            let mut out = Vec::new();
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => {
                        out.extend(gw.tokenize_with("", s, add_special && i == 0, parse_special)?)
                    }
                    v => out.push(token_id(v, n_vocab)?),
                }
            }
            Ok(out)
        }
        _ => Err(bad(
            "content must be a string or an array of strings and token ids",
        )),
    }
}

fn flag(body: &Value, key: &str, default: bool) -> bool {
    body.get(key).and_then(Value::as_bool).unwrap_or(default)
}

pub async fn tokenize(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    let body = json_body(&body)?;
    let mut tokens = Value::Array(Vec::new());
    if let Some(content) = body.get("content") {
        let ids = tokenize_mixed(
            &gw,
            content,
            flag(&body, "add_special", false),
            flag(&body, "parse_special", true),
        )?;
        tokens = if flag(&body, "with_pieces", false) {
            let vocab = gw.vocab("")?;
            ids.iter()
                .map(|&id| {
                    let bytes = vocab.bytes(id);
                    let piece = match std::str::from_utf8(bytes) {
                        Ok(s) => json!(s),
                        Err(_) => json!(bytes),
                    };
                    json!({"id": id, "piece": piece})
                })
                .collect()
        } else {
            json!(ids)
        };
    }
    Ok(Json(json!({"tokens": tokens})))
}

pub async fn detokenize(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    let body = json_body(&body)?;
    let mut content = String::new();
    if let Some(tokens) = body.get("tokens") {
        let vocab = gw.vocab("")?;
        let list = tokens
            .as_array()
            .ok_or_else(|| bad("tokens must be an array of token ids"))?;
        let mut bytes = Vec::new();
        for t in list {
            let id = t
                .as_i64()
                .ok_or_else(|| bad("tokens must be an array of token ids"))?;
            // an id outside the vocabulary has no piece, as in libllama
            if let Ok(id) = TokenId::try_from(id) {
                bytes.extend_from_slice(vocab.bytes(id));
            }
        }
        content = String::from_utf8_lossy(&bytes).into_owned();
    }
    Ok(Json(json!({"content": content})))
}

pub async fn apply_template(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Json<Value>> {
    let ChatPlan { spec, .. } = chat_plan(json_body(&body)?)?;
    Ok(Json(json!({"prompt": gw.render_prompt(&spec)?})))
}

// ---- native completion ----

#[derive(Deserialize)]
struct NativeRequest {
    /// Ignored by llama-server; accepted so a client that names the model works.
    #[serde(default)]
    model: String,
    prompt: Option<Value>,
    n_predict: Option<i64>,
    max_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    n_cmpl: Option<u32>,
    n: Option<u32>,
    #[serde(default)]
    stream: bool,
    stop: Option<Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<i64>,
    n_probs: Option<i64>,
    user: Option<String>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// llama-server's `tokenize_input_prompts` for one prompt: a string or mixed array is tokenized
/// with the BOS and special tokens parsed, an array of numbers is taken as it is.
fn prompt_tokens(gw: &Gateway, prompt: &Value) -> Result<Vec<TokenId>> {
    let sub = |v: &Value| -> Result<Vec<TokenId>> {
        match v {
            Value::String(_) => tokenize_mixed(gw, v, true, true),
            Value::Array(a) if a.iter().all(Value::is_number) => {
                let n_vocab = gw.vocab("")?.len();
                a.iter().map(|t| token_id(t, n_vocab)).collect()
            }
            Value::Array(a) if a.iter().all(|t| t.is_number() || t.is_string()) => {
                tokenize_mixed(gw, v, true, true)
            }
            Value::Object(o) if o.contains_key("prompt_string") => {
                if o.contains_key("multimodal_data") {
                    return Err(bad(
                        "multimodal data is not supported, this model is text only",
                    ));
                }
                tokenize_mixed(gw, &o["prompt_string"], true, true)
            }
            _ => Err(bad(
                "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.",
            )),
        }
    };
    let tokens = match prompt {
        // An array with no number in it is a batch of prompts.
        Value::Array(a) if !a.iter().any(Value::is_number) => match a.as_slice() {
            [] => return Err(bad("\"prompt\" must not be empty")),
            [one] => sub(one)?,
            _ => return Err(bad("batches of prompts are not supported, send one prompt")),
        },
        v => sub(v)?,
    };
    if tokens.is_empty() {
        return Err(bad("\"prompt\" must not be empty"));
    }
    Ok(tokens)
}

/// `n_predict` and its aliases: -1 is up to the end of the context.
fn n_predict(req: &NativeRequest) -> Result<(i64, Option<u32>)> {
    let n = req
        .n_predict
        .or(req.max_completion_tokens)
        .or(req.max_tokens)
        .unwrap_or(-1);
    match n {
        -1 => Ok((-1, None)),
        0 => Err(bad(
            "n_predict must be at least 1 (or -1); evaluating the prompt only is not supported",
        )),
        n => u32::try_from(n)
            .map(|m| (n, Some(m)))
            .map_err(|_| bad("n_predict must be -1 or a positive number")),
    }
}

struct Plan {
    spec: GenerateSpec,
    n: usize,
    stream: bool,
    /// The `generation_settings` of every result.
    settings: Value,
    prompt_text: String,
}

fn native_plan(gw: &Gateway, body: Value) -> Result<Plan> {
    let req: NativeRequest = serde_json::from_value(body.clone())
        .map_err(|e| bad(format!("invalid request body: {e}")))?;
    let prompt = req
        .prompt
        .as_ref()
        .ok_or_else(|| bad("\"prompt\" is required"))?;
    let tokens = prompt_tokens(gw, prompt)?;
    let (n_predict, max_tokens) = n_predict(&req)?;
    let n = choice_count(req.n_cmpl.or(req.n))?;
    let n_probs = probs_count(req.n_probs)?;
    let prompt_text = text_of(gw, &tokens)?;
    let mut spec = GenerateSpec::new(req.model, Prompt::Tokens(tokens));
    spec.max_tokens = max_tokens;
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = req.top_k;
    spec.seed = sampling::seed(req.seed);
    spec.extra = req.sampling.extras()?;
    spec.extra.logprobs = n_probs;
    spec.stop = stop_list(req.stop)?;
    spec.caller = req.user;
    let settings = settings::params(
        gw.config(),
        Some(&settings::Request {
            body: &body,
            n_predict,
            stream: req.stream,
            stop: &spec.stop,
            n_probs: n_probs.unwrap_or(0),
        }),
    );
    Ok(Plan {
        spec,
        n,
        stream: req.stream,
        settings,
        prompt_text,
    })
}

fn text_of(gw: &Gateway, tokens: &[TokenId]) -> Result<String> {
    let vocab = gw.vocab("")?;
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|&t| vocab.bytes(t).to_vec())
        .collect();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// llama-server's `stop_type`, and the stop word for `stopping_word`.
fn stop_type(finish: &Finish) -> (&'static str, &str) {
    match finish {
        Finish::Stop => ("eos", ""),
        Finish::Length => ("limit", ""),
        Finish::StopSequence(s) => ("word", s),
        Finish::ToolCalls => ("none", ""),
    }
}

/// The body of a finished native completion (`server_task_result_cmpl_final::to_json`).
fn final_json(plan: &Plan, model: &str, index: usize, c: &Collected, stream: bool) -> Value {
    let finish = c.finish.clone().unwrap_or(Finish::Stop);
    let (stop_type, stopping_word) = stop_type(&finish);
    let content = if stream { "" } else { c.text.as_str() };
    let mut v = json!({
        "index": index,
        "content": content,
        "tokens": [],
        "id_slot": 0,
        "stop": true,
        "model": model,
        "tokens_predicted": c.usage.completion_tokens,
        "tokens_evaluated": c.usage.prompt_tokens,
        "generation_settings": plan.settings,
        "prompt": plan.prompt_text,
        "has_new_line": c.text.contains('\n'),
        "truncated": false,
        "stop_type": stop_type,
        "stopping_word": stopping_word,
        "tokens_cached": c.usage.prompt_tokens + c.usage.completion_tokens.saturating_sub(1),
        "timings": c.usage.timings.map_or(Value::Null, |t| t.to_json()),
    });
    if !stream && !c.logprobs.is_empty() {
        v["completion_probabilities"] = c.logprobs.iter().map(LogprobEntry::to_json).collect();
    }
    v
}

async fn run(gw: Arc<Gateway>, plan: Plan) -> Result<Response> {
    let generations = gw
        .generate_many(
            choice_specs(plan.spec.clone(), plan.n),
            Operation::TextCompletion,
        )
        .await?;
    let gid = generations[0].id.clone();
    let model = generations[0].model.clone();
    if plan.stream {
        let events = native_stream(generations, plan, model);
        return Ok(tag_generation(sse(events), &gid));
    }
    let all = try_join_all(generations.into_iter().map(Generation::collect))
        .await
        .map_err(LlamaError::from)?;
    let mut results: Vec<Value> = all
        .iter()
        .enumerate()
        .map(|(i, c)| final_json(&plan, &model, i, c, false))
        .collect();
    let body = if results.len() == 1 {
        results.remove(0)
    } else {
        Value::Array(results)
    };
    Ok(tag_generation(Json(body).into_response(), &gid))
}

fn data(v: &Value) -> Event {
    Event::default().data(v.to_string())
}

struct NativeStream {
    events: Merged,
    plan: Plan,
    model: String,
    /// Text chunks sent per choice, the `tokens_predicted` of a partial chunk.
    sent: Vec<u32>,
    /// Log-probability entries waiting for the next chunk of their choice.
    lp: Vec<Vec<LogprobEntry>>,
    prompt_tokens: u32,
    usage: Vec<Option<Usage>>,
    done: bool,
}

/// The chunks of a native stream: partial chunks, then one final chunk per choice. No `[DONE]`,
/// as in llama-server.
fn native_stream(
    gens: Vec<Generation>,
    plan: Plan,
    model: String,
) -> impl Stream<Item = Event> + Send + 'static {
    let n = gens.len();
    let s = NativeStream {
        prompt_tokens: gens[0].prompt_tokens,
        events: merge(gens),
        plan,
        model,
        sent: vec![0; n],
        lp: vec![Vec::new(); n],
        usage: vec![None; n],
        done: false,
    };
    stream::unfold(s, |mut s| async move {
        if s.done {
            return None;
        }
        let mut out = Vec::new();
        match s.events.next().await {
            Some((i, OutEvent::Segment(Segment::Text(t)))) => {
                s.sent[i] += 1;
                let mut c = json!({
                    "index": i,
                    "content": t,
                    "tokens": [],
                    "stop": false,
                    "id_slot": 0,
                    "tokens_predicted": s.sent[i],
                    "tokens_evaluated": s.prompt_tokens,
                });
                let lp = std::mem::take(&mut s.lp[i]);
                if !lp.is_empty() {
                    c["completion_probabilities"] = lp.iter().map(LogprobEntry::to_json).collect();
                }
                out.push(data(&c));
            }
            Some((_, OutEvent::Segment(_))) => {}
            Some((i, OutEvent::Logprobs(e))) => s.lp[i].push(e),
            Some((i, OutEvent::Done { finish, usage })) => {
                s.usage[i] = Some(usage);
                let c = Collected {
                    finish: Some(finish),
                    usage,
                    ..Collected::default()
                };
                out.push(data(&final_json(&s.plan, &s.model, i, &c, true)));
                s.done = s.usage.iter().all(Option::is_some);
            }
            Some((_, OutEvent::Error(e))) => {
                out.push(data(&json!({"error": LlamaError::from(e).object()})));
                s.done = true;
            }
            None => return None,
        }
        Some((stream::iter(out), s))
    })
    .flatten()
}

pub async fn completion(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let plan = native_plan(&gw, json_body(&body)?)?;
    run(gw, plan).await
}

// ---- infill ----

/// The FIM prompt of llama-server's `format_prompt_infill`: extra context chunks, then
/// `[PRE] prefix prompt [SUF] suffix [MID]`, with a BOS first when the vocabulary adds one.
fn infill_prompt(
    gw: &Gateway,
    fim: &FimTokens,
    (pre, suf, mid): (TokenId, TokenId, TokenId),
    data: &Value,
    prompt: Vec<TokenId>,
) -> Result<Vec<TokenId>> {
    let tok = |text: &str| gw.tokenize_with("", text, false, false);
    let mut prefix = tokenize_mixed(gw, &data["input_prefix"], false, false)?;
    let mut suffix = tokenize_mixed(gw, &data["input_suffix"], false, false)?;
    let mut extra: Vec<TokenId> = Vec::new();
    if let Some(rep) = fim.rep {
        extra.push(rep);
        extra.extend(tok("myproject\n")?);
    }
    for chunk in data["input_extra"].as_array().into_iter().flatten() {
        let text = chunk["text"].as_str().unwrap_or_default();
        let filename = chunk["filename"].as_str().unwrap_or("tmp");
        match fim.sep {
            Some(sep) => {
                extra.push(sep);
                extra.extend(tok(&format!("{filename}\n"))?);
            }
            None => extra.extend(tok(CHUNK_SEPARATOR)?),
        }
        extra.extend(tok(text)?);
    }
    if let Some(sep) = fim.sep {
        extra.push(sep);
        extra.extend(tok("filename\n")?);
    }
    // The context to fit: a batch for prefix and suffix (3:1), the rest of the context for extras.
    let n_ctx = gw.model("")?.context_length as usize;
    let take_prefix = prefix.len().min(3 * (INFILL_N_BATCH / 4));
    let take_suffix = suffix
        .len()
        .min((INFILL_N_BATCH / 4).saturating_sub(2 + prompt.len()));
    // the server's n_predict is -1: room for two more tokens
    let take_extra = (n_ctx + 2).saturating_sub(INFILL_N_BATCH).min(extra.len());
    prefix.drain(..prefix.len() - take_prefix);
    suffix.truncate(take_suffix);
    prefix.insert(0, pre);
    prefix.extend(prompt);
    suffix.insert(0, suf);
    let mut out: Vec<TokenId> = Vec::new();
    out.extend_from_slice(&extra[extra.len() - take_extra..]);
    out.extend(fim.bos);
    out.extend(prefix);
    out.extend(suffix);
    out.push(mid);
    Ok(out)
}

pub async fn infill(State(gw): State<Arc<Gateway>>, body: Bytes) -> Result<Response> {
    let fim = gw.fim_tokens("")?;
    let mut missing = String::new();
    for (token, what) in [
        (fim.pre, "prefix"),
        (fim.suf, "suffix"),
        (fim.mid, "middle"),
    ] {
        if token.is_none() {
            missing.push_str(&format!("{what} token is missing. "));
        }
    }
    let (Some(pre), Some(suf), Some(mid)) = (fim.pre, fim.suf, fim.mid) else {
        return Err(LlamaError::not_supported(format!(
            "Infill is not supported by this model: {missing}"
        )));
    };
    let mut data = json_body(&body)?;
    if data.get("prompt").is_some_and(|p| !p.is_string()) {
        return Err(bad("\"prompt\" must be a string"));
    }
    for key in ["input_prefix", "input_suffix"] {
        if data.get(key).is_none() {
            return Err(bad(format!("\"{key}\" is required")));
        }
    }
    match data.get("input_extra") {
        None | Some(Value::Null) => {}
        Some(Value::Array(chunks)) => {
            for chunk in chunks {
                if !chunk.get("text").is_some_and(Value::is_string) {
                    return Err(bad(
                        "extra_context chunk must contain a \"text\" field with a string value",
                    ));
                }
                if chunk.get("filename").is_some_and(|f| !f.is_string()) {
                    return Err(bad(
                        "extra_context chunk's \"filename\" field must be a string",
                    ));
                }
            }
        }
        Some(_) => {
            return Err(bad(
                "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
            ));
        }
    }
    let prompt = data
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let prompt = gw.tokenize_with("", prompt, false, true)?;
    let tokens = infill_prompt(&gw, &fim, (pre, suf, mid), &data, prompt)?;
    if let Some(o) = data.as_object_mut() {
        o.insert("prompt".into(), json!(tokens));
    }
    let plan = native_plan(&gw, data)?;
    run(gw, plan).await
}

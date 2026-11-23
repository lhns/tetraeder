//! The embedding, rerank and slot routes (ADR 0090): bodies, errors and status codes as
//! llama-server answers them, over a fake engine. No model.

mod common;

use common::*;
use serde_json::{Value, json};
use tetraeder_gateway::engine::{PoolingKind, SlotAction};
use tetraeder_gateway::fake::FakeEngine;

fn embedding_engine(p: PoolingKind) -> FakeEngine {
    let mut e = FakeEngine::fixed("x");
    e.embeddings = Some(p);
    e
}

fn error(v: &Value) -> (u64, &str, &str) {
    let e = &v["error"];
    (
        e["code"].as_u64().unwrap(),
        e["type"].as_str().unwrap(),
        e["message"].as_str().unwrap(),
    )
}

// ---- embeddings ----

#[tokio::test]
async fn embeddings_are_not_supported_without_the_flag() {
    let (app, _) = app(FakeEngine::fixed("x"));
    for path in ["/embedding", "/embeddings", "/v1/embeddings"] {
        let (s, _, v) = post(&app, path, &json!({"input": "a"}), false).await;
        assert_eq!(s, 501, "{path}");
        assert_eq!(
            error(&v),
            (
                501,
                "not_supported_error",
                "This server does not support embeddings. Start it with `--embeddings`"
            )
        );
    }
}

#[tokio::test]
async fn openai_embeddings_answer_with_the_list_object() {
    let (app, gw) = app(embedding_engine(PoolingKind::Mean));
    let (s, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": "ab", "model": "m"}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "m");
    assert_eq!(v["usage"], json!({"prompt_tokens": 2, "total_tokens": 2}));
    assert_eq!(
        v["data"],
        json!([{"embedding": [2.0, 195.0, 1.0], "index": 0, "object": "embedding"}])
    );
    // no model in the request: the served one
    let (_, _, v) = post(&app, "/v1/embeddings", &json!({"input": "a"}), false).await;
    assert_eq!(v["model"], "fake");
    drop(gw);
}

#[tokio::test]
async fn embedding_inputs_take_llama_servers_shapes() {
    let engine = embedding_engine(PoolingKind::Mean);
    let log = std::sync::Arc::clone(&engine.embedded);
    let (app, _) = app(engine);
    let tokens = |v: &Value| -> Vec<usize> {
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["embedding"][0].as_f64().unwrap() as usize)
            .collect()
    };
    // a list of strings is a list of inputs
    let (_, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": ["a", "bcd"]}),
        false,
    )
    .await;
    assert_eq!(tokens(&v), [1, 3]);
    assert_eq!(v["data"][1]["index"], 1);
    assert_eq!(v["usage"]["prompt_tokens"], 4);
    // a list of ids is one input; a list of lists is several
    let (_, _, v) = post(&app, "/v1/embeddings", &json!({"input": [1, 2, 3]}), false).await;
    assert_eq!(tokens(&v), [3]);
    let (_, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": [[1, 2], [3]]}),
        false,
    )
    .await;
    assert_eq!(tokens(&v), [2, 1]);
    // `content` is the native field and answers the native shape
    let (_, _, v) = post(&app, "/v1/embeddings", &json!({"content": "ab"}), false).await;
    assert_eq!(v, json!([{"index": 0, "embedding": [[2.0, 195.0, 1.0]]}]));
    // OpenAI content parts
    let (s, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": [{"content": [{"type": "text", "text": "ab"}]}]}),
        false,
    )
    .await;
    assert_eq!((s, tokens(&v)), (200, vec![2]));
    // the default normalization is euclidean (2); the request may change it
    let seen = log.lock().unwrap().clone();
    assert!(seen.iter().all(|(_, n)| *n == 2));
    post(
        &app,
        "/v1/embeddings",
        &json!({"input": "a", "embd_normalize": -1}),
        false,
    )
    .await;
    assert_eq!(log.lock().unwrap().last().unwrap().1, -1);
}

#[tokio::test]
async fn native_embeddings_are_a_list_of_index_and_vectors() {
    let (app, _) = app(embedding_engine(PoolingKind::Mean));
    for path in ["/embedding", "/embeddings"] {
        let (s, _, v) = post(&app, path, &json!({"content": ["a", "b"]}), false).await;
        assert_eq!(s, 200, "{path}");
        assert_eq!(v.as_array().unwrap().len(), 2);
        assert_eq!(v[1]["index"], 1);
        assert_eq!(
            v[1]["embedding"].as_array().unwrap().len(),
            1,
            "one pooled vector"
        );
    }
    // `input` works on the native route too
    let (s, _, v) = post(&app, "/embeddings", &json!({"input": "a"}), false).await;
    assert_eq!(s, 200);
    assert!(v[0]["embedding"][0].is_array());
}

#[tokio::test]
async fn without_pooling_the_native_route_gives_a_vector_per_token_and_openai_refuses() {
    let (app, _) = app(embedding_engine(PoolingKind::None));
    let (s, _, v) = post(&app, "/embeddings", &json!({"content": "ab"}), false).await;
    assert_eq!(s, 200);
    assert_eq!(v[0]["embedding"], json!([[97.0, 0.0], [98.0, 1.0]]));
    let (s, _, v) = post(&app, "/v1/embeddings", &json!({"input": "ab"}), false).await;
    assert_eq!(s, 400);
    assert_eq!(
        error(&v),
        (
            400,
            "invalid_request_error",
            "Pooling type 'none' is not OAI compatible. Please use a different pooling type"
        )
    );
}

#[tokio::test]
async fn base64_embeddings_carry_the_float_bytes() {
    let (app, _) = app(embedding_engine(PoolingKind::Mean));
    let (_, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": "a", "encoding_format": "base64"}),
        false,
    )
    .await;
    let d = &v["data"][0];
    assert_eq!(d["encoding_format"], "base64");
    // [1.0, 97.0, 1.0] as little-endian f32
    assert_eq!(d["embedding"], "AACAP wAAwkIAAIA/".replace(' ', ""));
    let (s, _, v) = post(
        &app,
        "/v1/embeddings",
        &json!({"input": "a", "encoding_format": "xml"}),
        false,
    )
    .await;
    assert_eq!(s, 400);
    assert_eq!(
        error(&v).2,
        "The format to return the embeddings in. Can be either float or base64"
    );
}

#[tokio::test]
async fn bad_embedding_requests_are_400s_with_llama_servers_words() {
    let (app, _) = app(embedding_engine(PoolingKind::Mean));
    for (body, msg) in [
        (json!({}), "\"input\" or \"content\" must be provided"),
        (json!({"input": []}), "\"input\" must not be empty"),
        (json!({"input": ""}), "Input content cannot be empty"),
        (json!({"input": ["a", ""]}), "Input content cannot be empty"),
    ] {
        let (s, _, v) = post(&app, "/v1/embeddings", &body, false).await;
        assert_eq!(s, 400, "{body}");
        assert_eq!(error(&v).2, msg, "{body}");
    }
    let (s, _, _) = post(&app, "/v1/embeddings", &json!({"input": {"x": 1}}), false).await;
    assert_eq!(s, 400);
}

// ---- rerank ----

#[tokio::test]
async fn rerank_needs_the_rank_pooling() {
    for engine in [FakeEngine::fixed("x"), embedding_engine(PoolingKind::Mean)] {
        let (app, _) = app(engine);
        for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
            let (s, _, v) = post(
                &app,
                path,
                &json!({"query": "q", "documents": ["a"]}),
                false,
            )
            .await;
            assert_eq!(s, 501, "{path}");
            assert_eq!(
                error(&v),
                (
                    501,
                    "not_supported_error",
                    "This server does not support reranking. Start it with `--reranking`"
                )
            );
        }
    }
}

#[tokio::test]
async fn rerank_sorts_by_score_and_cuts_to_top_n() {
    let (app, _) = app(embedding_engine(PoolingKind::Rank));
    let body = json!({"query": "q", "documents": ["aa", "a", "aaaa"], "top_n": 2, "model": "m"});
    for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        let (s, _, v) = post(&app, path, &body, false).await;
        assert_eq!(s, 200, "{path}: {v}");
        assert_eq!(v["object"], "list");
        assert_eq!(v["model"], "m");
        assert_eq!(
            v["results"],
            json!([
                {"index": 2, "relevance_score": 4.0},
                {"index": 0, "relevance_score": 2.0}
            ])
        );
        // query 1 byte + documents 2 + 1 + 4 bytes
        assert_eq!(v["usage"], json!({"prompt_tokens": 10, "total_tokens": 10}));
    }
    // objects with text parts; top_n 0 keeps none
    let (_, _, v) = post(
        &app,
        "/rerank",
        &json!({"query": {"text": ["q", "r"]}, "documents": [{"text": "a"}], "top_n": 0}),
        false,
    )
    .await;
    assert_eq!(v["results"], json!([]));
}

#[tokio::test]
async fn tei_rerank_answers_a_bare_list() {
    let (app, _) = app(embedding_engine(PoolingKind::Rank));
    let (s, _, v) = post(
        &app,
        "/rerank",
        &json!({"query": "q", "texts": ["a", "bbb"], "return_text": true}),
        false,
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(
        v,
        json!([
            {"index": 1, "score": 3.0, "text": "bbb"},
            {"index": 0, "score": 1.0, "text": "a"}
        ])
    );
    let (_, _, v) = post(
        &app,
        "/rerank",
        &json!({"query": "q", "texts": ["a"]}),
        false,
    )
    .await;
    assert_eq!(v, json!([{"index": 0, "score": 1.0}]));
}

#[tokio::test]
async fn bad_rerank_requests_are_400s() {
    let (app, _) = app(embedding_engine(PoolingKind::Rank));
    for (body, msg) in [
        (json!({"documents": ["a"]}), "\"query\" must be provided"),
        (
            json!({"query": "q"}),
            "\"documents\" must be a non-empty array of strings or objects",
        ),
        (
            json!({"query": "q", "documents": []}),
            "\"documents\" must be a non-empty array of strings or objects",
        ),
        (
            json!({"query": "q", "documents": ["a"], "top_n": -1}),
            "\"top_n\" must be >= 0",
        ),
        (
            json!({"query": "q", "documents": [{"text": "a", "image": "http://x/y.png"}]}),
            "multimodal rerank input requires --mmproj",
        ),
    ] {
        let (s, _, v) = post(&app, "/v1/rerank", &body, false).await;
        assert_eq!(s, 400, "{body}");
        assert_eq!(error(&v).2, msg, "{body}");
    }
}

// ---- slots ----

fn slot_engine() -> FakeEngine {
    let mut e = FakeEngine::fixed("x");
    e.slot_actions = true;
    e
}

#[tokio::test]
async fn slot_actions_are_not_supported_without_the_flag() {
    let (app, _) = app(FakeEngine::fixed("x"));
    let (s, _, v) = post(
        &app,
        "/slots/0?action=save",
        &json!({"filename": "s.bin"}),
        false,
    )
    .await;
    assert_eq!(s, 501);
    assert_eq!(
        error(&v),
        (
            501,
            "not_supported_error",
            "This server does not support slots action. Start it with `--slot-save-path`"
        )
    );
}

#[tokio::test]
async fn slot_save_restore_and_erase_answer_in_llama_servers_shape() {
    let engine = slot_engine();
    let log = std::sync::Arc::clone(&engine.slot_log);
    let (app, _) = app(engine);
    let (s, _, v) = post(
        &app,
        "/slots/1?action=save",
        &json!({"filename": "s.bin"}),
        false,
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        (
            &v["id_slot"],
            &v["filename"],
            &v["n_saved"],
            &v["n_written"]
        ),
        (&json!(1), &json!("s.bin"), &json!(7), &json!(70))
    );
    assert_eq!(v["timings"]["save_ms"], 1.5);
    let (s, _, v) = post(
        &app,
        "/slots/0?action=restore",
        &json!({"filename": "s.bin"}),
        false,
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(
        (&v["id_slot"], &v["n_restored"], &v["n_read"]),
        (&json!(0), &json!(7), &json!(70))
    );
    assert_eq!(v["timings"]["restore_ms"], 1.5);
    let (s, _, v) = post(&app, "/slots/0?action=erase", &json!({}), false).await;
    assert_eq!((s, v), (200, json!({"id_slot": 0, "n_erased": 7})));
    // erase needs no body at all
    let (s, _, _) = send(&app, request("POST", "/slots/1?action=erase", None, false)).await;
    assert_eq!(s, 200);
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen.len(), 4);
    assert_eq!(
        seen[0],
        (
            1,
            SlotAction::Save {
                filename: "s.bin".into()
            }
        )
    );
}

#[tokio::test]
async fn bad_slot_requests_are_400s() {
    let (app, _) = app(slot_engine());
    let body = json!({"filename": "s.bin"});
    for (path, b, msg) in [
        ("/slots/x?action=save", body.clone(), "Invalid slot ID"),
        ("/slots/0?action=bogus", body.clone(), "Invalid action"),
        ("/slots/0", body.clone(), "Invalid action"),
        (
            "/slots/0?action=save",
            json!({"filename": "../s.bin"}),
            "Invalid filename",
        ),
        (
            "/slots/0?action=save",
            json!({"filename": "a/b"}),
            "Invalid filename",
        ),
        (
            "/slots/0?action=restore",
            json!({"filename": ""}),
            "Invalid filename",
        ),
        ("/slots/7?action=erase", json!({}), "Invalid slot ID"),
    ] {
        let (s, _, v) = post(&app, path, &b, false).await;
        assert_eq!(s, 400, "{path} {b}");
        assert_eq!(error(&v).2, msg, "{path} {b}");
    }
}

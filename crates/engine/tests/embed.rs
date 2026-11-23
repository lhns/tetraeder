//! The embedding context over the tiny Qwen3 fixture (ADR 0090). The fixture has no pooling type
//! of its own (it is `none`) and no classification head, so the pooled values are checked against
//! the per-token ones, and the reranker is checked to be refused.
//!
//! Hardware label: none.

use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_engine::{EmbedConfig, Embedder, Embedding, EngineError, Pooling};
use tetraeder_ggml_sys::llama::{Model, ModelParams};

fn model() -> Arc<Model> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-f16.gguf");
    let params = ModelParams {
        n_gpu_layers: 0,
        ..ModelParams::default()
    };
    Arc::new(Model::load(path, &params).expect("fixture loads"))
}

fn embedder(m: &Arc<Model>, pooling: Pooling) -> Embedder {
    let cfg = EmbedConfig {
        pooling,
        n_batch: 64,
        threads: Some(2),
        ..EmbedConfig::default()
    };
    Embedder::new(m, &cfg).expect("embedding context")
}

fn tokens(m: &Model, text: &str) -> Vec<i32> {
    m.tokenize(text, true, true).expect("tokenizes")
}

#[test]
fn the_models_own_pooling_is_none_and_gives_one_vector_per_token() {
    let m = model();
    let mut e = embedder(&m, Pooling::Unspecified);
    assert_eq!(e.pooling(), Pooling::None);
    let t = tokens(&m, "Hello world, once upon a time");
    let Embedding::Tokens(v) = e.embed(&t).unwrap() else {
        panic!("not per token")
    };
    assert_eq!(v.len(), t.len());
    assert!(
        v.iter()
            .all(|x| x.len() == m.n_embd_out() && x.len() == 128)
    );
    assert!(v.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn mean_pooling_is_the_mean_of_the_per_token_vectors() {
    let m = model();
    let t = tokens(&m, "Hello world, once upon a time");
    let Embedding::Tokens(per_token) = embedder(&m, Pooling::None).embed(&t).unwrap() else {
        panic!("not per token")
    };
    let mut mean = embedder(&m, Pooling::Mean);
    assert_eq!(mean.pooling(), Pooling::Mean);
    let Embedding::Pooled(pooled) = mean.embed(&t).unwrap() else {
        panic!("not pooled")
    };
    assert_eq!(pooled.len(), 128);
    for (d, p) in pooled.iter().enumerate() {
        let want: f32 = per_token.iter().map(|v| v[d]).sum::<f32>() / t.len() as f32;
        assert!((p - want).abs() < 1e-4, "dim {d}: {p} against {want}");
    }
    // the same input twice gives the same vector: no state is carried between inputs
    let Embedding::Pooled(again) = mean.embed(&t).unwrap() else {
        panic!("not pooled")
    };
    assert_eq!(pooled, again);
}

#[test]
fn last_and_cls_pooling_pick_one_token() {
    let m = model();
    let t = tokens(&m, "Hello world");
    let Embedding::Tokens(per_token) = embedder(&m, Pooling::None).embed(&t).unwrap() else {
        panic!("not per token")
    };
    let Embedding::Pooled(last) = embedder(&m, Pooling::Last).embed(&t).unwrap() else {
        panic!("not pooled")
    };
    let end = per_token.last().unwrap();
    assert!(last.iter().zip(end).all(|(a, b)| (a - b).abs() < 1e-4));
    let Embedding::Pooled(cls) = embedder(&m, Pooling::Cls).embed(&t).unwrap() else {
        panic!("not pooled")
    };
    assert!(
        cls.iter()
            .zip(&per_token[0])
            .all(|(a, b)| (a - b).abs() < 1e-4)
    );
}

#[test]
fn inputs_beyond_the_batch_and_empty_ones_are_refused() {
    let m = model();
    let mut e = embedder(&m, Pooling::Mean);
    assert!(matches!(e.embed(&[]), Err(EngineError::EmptyPrompt)));
    let long = vec![10; 65];
    assert!(matches!(
        e.embed(&long),
        Err(EngineError::ContextTooSmall {
            needed: 65,
            n_ctx: 64
        })
    ));
    // and the context still works afterwards
    assert!(e.embed(&[10, 11]).is_ok());
}

#[test]
fn a_model_without_a_classification_head_cannot_rerank() {
    let m = model();
    assert_eq!(m.n_cls_out(), 0);
    let cfg = EmbedConfig {
        pooling: Pooling::Rank,
        n_batch: 64,
        threads: Some(2),
        ..EmbedConfig::default()
    };
    let err = Embedder::new(&m, &cfg).err().expect("refused").to_string();
    assert!(err.contains("classification head"), "{err}");
}

//! What crosses to a replica on another node (ADR 0091, step 5): the request's sampling
//! parameters and the engine's events as JSON, and a constraint as the grammar it was built from.
//!
//! Hardware label: none.

use tetraeder_gateway::chat::OutputFormat;
use tetraeder_gateway::constraint::{ConstraintFactory, Reasoning};
use tetraeder_gateway::engine::{
    Engine, EngineStats, FinishReason, GenEvent, GrammarSource, SamplerKind, SamplingExtras,
    SamplingParams, TokenProbs,
};
use tetraeder_gateway::fake::FakeEngine;

// hardware: none
#[test]
fn sampling_parameters_survive_json_including_a_banned_token() {
    let p = SamplingParams {
        max_tokens: 256,
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        seed: Some(u64::MAX - 1),
        extra: SamplingExtras {
            logit_bias: vec![(5, f32::NEG_INFINITY), (7, 1.5), (9, f32::INFINITY)],
            samplers: Some(vec![SamplerKind::TopK, SamplerKind::Temperature]),
            grammar: Some(GrammarSource::Gbnf("root ::= \"a\"".into())),
            dry_sequence_breakers: Some(vec!["\n".into()]),
            logprobs: Some(3),
            slot: Some(1),
            ..SamplingExtras::default()
        },
    };
    let text = serde_json::to_string(&p).unwrap();
    let back: SamplingParams = serde_json::from_str(&text).unwrap();
    assert_eq!(back, p);
}

// hardware: none
#[test]
fn engine_events_survive_json() {
    let events = vec![
        GenEvent::Token(42),
        GenEvent::Probs(TokenProbs {
            token: 3,
            logprob: -0.5,
            top: vec![(3, -0.5), (4, -1.5)],
        }),
        GenEvent::Finished {
            reason: FinishReason::Length,
            stats: EngineStats {
                prompt_tokens: 512,
                cached_tokens: 480,
                generated_tokens: 256,
                prompt_ms: 12.5,
                predicted_ms: 800.0,
                gpu_seconds: Some(0.9),
                bytes_streamed: None,
                profile: None,
            },
        },
        GenEvent::Error("replica 1 failed".into()),
        GenEvent::ContextExceeded("too long".into()),
    ];
    for e in events {
        let back: GenEvent = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(back, e);
    }
}

// hardware: none
#[test]
fn a_constraint_is_rebuilt_from_its_source_and_masks_alike() {
    let fake = FakeEngine::fixed("");
    let vocab = fake.vocab("fake").unwrap();
    // The head builds it; the leader rebuilds it from the text, with its own factory.
    let head = ConstraintFactory::new(&vocab).unwrap();
    let mut a = head
        .json(&OutputFormat::JsonObject, Reasoning::None)
        .unwrap();
    let source = a.source().expect("a gateway constraint knows its grammar");
    let leader = ConstraintFactory::new(&vocab).unwrap();
    let mut b = leader.from_lark(&source).unwrap();
    assert_eq!(b.source().as_deref(), Some(source.as_str()));
    for tok in [u32::from(b'{'), u32::from(b'}')] {
        assert_eq!(a.mask().unwrap(), b.mask().unwrap());
        a.commit(tok).unwrap();
        b.commit(tok).unwrap();
    }
    assert_eq!(a.mask().unwrap(), b.mask().unwrap());
    assert!(leader.from_lark("start: (").is_err());
}

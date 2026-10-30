//! The request span. Its own test binary: tracing caches callsite interest per process, and
//! tests without a subscriber running in parallel would hide the span.

mod common;

use common::*;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use serde_json::{Value, json};
use std::time::Duration;
use tetraeder_gateway::fake::FakeEngine;

fn chat() -> Value {
    json!({"model": "fake", "messages": [{"role": "user", "content": "hi"}]})
}

#[test]
fn request_span_has_gen_ai_attributes() {
    use tracing_subscriber::layer::SubscriberExt;
    let exp = InMemorySpanExporter::default();
    let tp = SdkTracerProvider::builder()
        .with_simple_exporter(exp.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tp.tracer("test")));
    let _g = tracing::subscriber::set_default(subscriber);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (app, _) = app(FakeEngine::fixed("abc"));
        let (s, _, _) = post(&app, "/v1/chat/completions", &chat(), false).await;
        assert_eq!(s, 200);
        for _ in 0..500 {
            if !exp.get_finished_spans().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    let spans = exp.get_finished_spans().unwrap();
    let s = spans
        .iter()
        .find(|s| s.name == "chat fake")
        .expect("span named '{op} {model}'");
    assert_eq!(s.span_kind, opentelemetry::trace::SpanKind::Server);
    let attr = |k: &str| {
        s.attributes
            .iter()
            .find(|kv| kv.key.as_str() == k)
            .map(|kv| kv.value.to_string())
    };
    assert_eq!(attr("gen_ai.operation.name").as_deref(), Some("chat"));
    assert_eq!(attr("gen_ai.request.model").as_deref(), Some("fake"));
    assert_eq!(attr("gen_ai.usage.output_tokens").as_deref(), Some("3"));
    assert_eq!(
        attr("gen_ai.usage.input_tokens"),
        Some(chatml_user_tokens("hi").to_string())
    );
    assert_eq!(
        attr("gen_ai.response.finish_reasons").as_deref(),
        Some("stop")
    );
    assert!(attr("gen_ai.response.id").is_some());
}

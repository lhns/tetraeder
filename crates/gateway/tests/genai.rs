//! GenAI semantic-convention request metrics (ADR 0063): one TTFT per request that produced a
//! token, one TPOT over the tokens after the first, on both API routes, streamed or not.
//! The same code serves every engine behind the `Engine` trait, native or proxied (ADR 0046).

mod common;

use common::*;
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::server::router;

type Attrs = Vec<(String, String)>;

/// (attributes, count, sum in seconds) of every data point of an f64 histogram.
fn hist(exp: &InMemoryMetricExporter, name: &str) -> Vec<(Attrs, u64, f64)> {
    let mut out = Vec::new();
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for m in sm.metrics().filter(|m| m.name() == name) {
                if let AggregatedMetrics::F64(MetricData::Histogram(h)) = m.data() {
                    for dp in h.data_points() {
                        let mut a: Attrs = dp
                            .attributes()
                            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                            .collect();
                        a.sort();
                        out.push((a, dp.count(), dp.sum()));
                    }
                }
            }
        }
    }
    out
}

fn has(a: &Attrs, k: &str, v: &str) -> bool {
    a.iter().any(|(x, y)| x == k && y == v)
}

fn setup(
    mut engine: FakeEngine,
    delay: Duration,
) -> (axum::Router, SdkMeterProvider, InMemoryMetricExporter) {
    engine.token_delay = delay;
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let gw = Arc::new(
        Gateway::new(Arc::new(engine), GatewayConfig::default()).with_meter(&mp.meter("test")),
    );
    (router(gw), mp, exp)
}

/// Sends one request on the route and reads the whole body.
async fn run(app: &axum::Router, anthropic: bool, stream: bool) {
    let (path, body): (&str, Value) = if anthropic {
        (
            "/v1/messages",
            json!({"model": "fake", "max_tokens": 64, "stream": stream,
                   "messages": [{"role": "user", "content": "hi"}]}),
        )
    } else {
        (
            "/v1/chat/completions",
            json!({"model": "fake", "stream": stream,
                   "messages": [{"role": "user", "content": "hi"}]}),
        )
    };
    let (s, _, text) = send(app, request("POST", path, Some(&body), anthropic)).await;
    assert_eq!(s, 200, "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ttft_and_tpot_on_every_route_and_mode() {
    let delay = Duration::from_millis(20);
    for anthropic in [false, true] {
        for stream in [false, true] {
            let case = format!("anthropic={anthropic} stream={stream}");
            // 5 output tokens: 4 intervals after the first.
            let (app, mp, exp) = setup(FakeEngine::fixed("abcde"), delay);
            run(&app, anthropic, stream).await;
            mp.force_flush().unwrap();

            let ttft = hist(&exp, "gen_ai.server.time_to_first_token");
            assert_eq!(ttft.len(), 1, "{case}: {ttft:?}");
            assert_eq!(ttft[0].1, 1, "{case}: one TTFT per request");
            assert!(has(&ttft[0].0, "gen_ai.operation.name", "chat"), "{case}");
            assert!(has(&ttft[0].0, "gen_ai.request.model", "fake"), "{case}");

            let tpot = hist(&exp, "gen_ai.server.time_per_output_token");
            assert_eq!(tpot.len(), 1, "{case}: {tpot:?}");
            assert_eq!(tpot[0].1, 1, "{case}: one TPOT per request");
            assert!(has(&tpot[0].0, "gen_ai.operation.name", "chat"), "{case}");
            assert!(has(&tpot[0].0, "gen_ai.request.model", "fake"), "{case}");
            // Mean over the 4 intervals: at least the delay, far below the whole request.
            assert!(
                (0.015..0.5).contains(&tpot[0].2),
                "{case}: TPOT {} s",
                tpot[0].2
            );

            let dur = hist(&exp, "gen_ai.server.request.duration");
            assert_eq!(dur.len(), 1, "{case}: {dur:?}");
            assert!(has(&dur[0].0, "gen_ai.operation.name", "chat"), "{case}");
            assert!(!dur[0].0.iter().any(|(k, _)| k == "error.type"), "{case}");
            // The request covers the first token and the four intervals after it.
            assert!(dur[0].2 >= ttft[0].2 + 4.0 * 0.015, "{case}: {dur:?}");
        }
    }
}

#[tokio::test]
async fn one_token_has_a_ttft_but_no_tpot() {
    for anthropic in [false, true] {
        let (app, mp, exp) = setup(FakeEngine::fixed("a"), Duration::ZERO);
        run(&app, anthropic, true).await;
        mp.force_flush().unwrap();
        assert_eq!(hist(&exp, "gen_ai.server.time_to_first_token").len(), 1);
        assert!(hist(&exp, "gen_ai.server.time_per_output_token").is_empty());
    }
}

#[tokio::test]
async fn rejected_request_has_error_type_and_operation_but_no_ttft() {
    let (app, mp, exp) = setup(FakeEngine::fixed("abc"), Duration::ZERO);
    let body = json!({"model": "nope", "messages": [{"role": "user", "content": "hi"}]});
    let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
    assert_eq!(s, 404);
    mp.force_flush().unwrap();
    let dur = hist(&exp, "gen_ai.server.request.duration");
    assert_eq!(dur.len(), 1, "{dur:?}");
    assert!(has(&dur[0].0, "error.type", "not_found"));
    assert!(has(&dur[0].0, "gen_ai.operation.name", "chat"));
    assert!(hist(&exp, "gen_ai.server.time_to_first_token").is_empty());
}

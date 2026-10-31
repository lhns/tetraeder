//! Backpressure, cancellation, cost accounting, metrics and traces.

mod common;

use common::*;
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_gateway::gateway::{CostSink, Gateway, GatewayConfig, RequestCost};
use tetraeder_gateway::server::{router, serve};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn chat(stream: bool) -> Value {
    json!({"model": "fake", "messages": [{"role": "user", "content": "hi"}], "stream": stream, "user": "alice"})
}

/// Waits up to 5 s for `f` to hold.
async fn eventually(mut f: impl FnMut() -> bool) {
    for _ in 0..500 {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("condition not reached within 5 s");
}

#[derive(Default)]
struct Costs(Mutex<Vec<RequestCost>>);

impl CostSink for Costs {
    fn record(&self, c: &RequestCost) {
        self.0.lock().unwrap().push(c.clone());
    }
}

fn slow_engine() -> FakeEngine {
    let mut e = FakeEngine::fixed(&"x".repeat(2000));
    e.token_delay = Duration::from_millis(5);
    e
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_queue_is_rejected_per_api() {
    let cfg = GatewayConfig {
        max_concurrent: 1,
        max_queued: 1,
        retry_after_secs: 3,
        ..GatewayConfig::default()
    };
    let (app, gw) = app_with(slow_engine(), cfg);
    // A holds the slot: its streaming body is never read.
    let a = app
        .clone()
        .oneshot_req(request(
            "POST",
            "/v1/chat/completions",
            Some(&chat(true)),
            false,
        ))
        .await;
    assert_eq!(a.status(), 200);
    // B waits in the queue.
    let app_b = app.clone();
    let b = tokio::spawn(async move {
        let mut body = chat(false);
        body["max_tokens"] = json!(5);
        send(
            &app_b,
            request("POST", "/v1/chat/completions", Some(&body), false),
        )
        .await
    });
    eventually(|| gw.queue_depth() == 1).await;

    let (s, h, v) = post(&app, "/v1/chat/completions", &chat(false), false).await;
    assert_eq!(s, 429, "{v}");
    assert_eq!(h["retry-after"], "3");
    assert_eq!(v["error"]["type"], "rate_limit_error");

    let body =
        json!({"model": "fake", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]});
    let (s, h, v) = post(&app, "/v1/messages", &body, true).await;
    assert_eq!(s.as_u16(), 529, "{v}");
    assert_eq!(h["retry-after"], "3");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "overloaded_error");

    // Dropping A frees the slot; B then runs to completion.
    drop(a);
    let (s, _, _) = b.await.unwrap();
    assert_eq!(s, 200);
    assert_eq!(gw.queue_depth(), 0);
}

/// `oneshot` returning the response without reading the body.
trait Oneshot {
    async fn oneshot_req(
        self,
        r: axum::http::Request<axum::body::Body>,
    ) -> axum::http::Response<axum::body::Body>;
}

impl Oneshot for axum::Router {
    async fn oneshot_req(
        self,
        r: axum::http::Request<axum::body::Body>,
    ) -> axum::http::Response<axum::body::Body> {
        use tower::ServiceExt;
        self.oneshot(r).await.unwrap()
    }
}

async fn spawn_server(engine: FakeEngine) -> (std::net::SocketAddr, Arc<Gateway>, Arc<Costs>) {
    let costs = Arc::new(Costs::default());
    let gw = Arc::new(
        Gateway::new(Arc::new(engine), GatewayConfig::default()).with_cost_sink(costs.clone()),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, Arc::clone(&gw)));
    (addr, gw, costs)
}

async fn raw_post(addr: std::net::SocketAddr, path: &str, body: &Value) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let b = body.to_string();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
        b.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_disconnect_cancels_streaming_generation() {
    let engine = slow_engine();
    let (cancelled, started) = (engine.cancelled.clone(), engine.started.clone());
    let (addr, _gw, costs) = spawn_server(engine).await;
    let mut s = raw_post(addr, "/v1/chat/completions", &chat(true)).await;
    let mut buf = [0u8; 512];
    let n = s.read(&mut buf).await.unwrap();
    assert!(String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"));
    drop(s);
    eventually(|| cancelled.load(Ordering::SeqCst) == 1).await;
    assert_eq!(started.load(Ordering::SeqCst), 1);
    eventually(|| costs.0.lock().unwrap().len() == 1).await;
    let c = costs.0.lock().unwrap()[0].clone();
    assert_eq!(c.error_type, Some("cancelled"));
    assert!(c.completion_tokens < 2000);
    assert_eq!(c.caller, "alice");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_disconnect_cancels_non_streaming_generation() {
    let engine = slow_engine();
    let (cancelled, started) = (engine.cancelled.clone(), engine.started.clone());
    let (addr, _gw, _) = spawn_server(engine).await;
    let s = raw_post(addr, "/v1/chat/completions", &chat(false)).await;
    eventually(|| started.load(Ordering::SeqCst) == 1).await;
    drop(s);
    eventually(|| cancelled.load(Ordering::SeqCst) == 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_while_queued_leaves_the_queue() {
    let cfg = GatewayConfig {
        max_concurrent: 1,
        max_queued: 4,
        ..GatewayConfig::default()
    };
    let engine = slow_engine();
    let started = engine.started.clone();
    let gw = Arc::new(Gateway::new(Arc::new(engine), cfg));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, Arc::clone(&gw)));
    let _a = raw_post(addr, "/v1/chat/completions", &chat(true)).await;
    eventually(|| started.load(Ordering::SeqCst) == 1).await;
    let b = raw_post(addr, "/v1/chat/completions", &chat(true)).await;
    eventually(|| gw.queue_depth() == 1).await;
    drop(b);
    eventually(|| gw.queue_depth() == 0).await;
    assert_eq!(started.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cost_record_per_request() {
    let costs = Arc::new(Costs::default());
    let gw = Arc::new(
        Gateway::new(Arc::new(FakeEngine::fixed("abc")), GatewayConfig::default())
            .with_cost_sink(costs.clone()),
    );
    let app = router(gw);
    let (s, _, _) = post(&app, "/v1/chat/completions", &chat(false), false).await;
    assert_eq!(s, 200);
    let c = costs.0.lock().unwrap()[0].clone();
    assert_eq!(c.caller, "alice");
    assert_eq!(c.model, "fake");
    assert_eq!(c.operation, "chat");
    assert_eq!(c.completion_tokens, 3);
    assert_eq!(c.prompt_tokens as usize, chatml_user_tokens("hi"));
    assert_eq!(c.gpu_seconds, Some(0.5));
    assert_eq!(c.bytes_streamed, Some(1024));
    assert_eq!(c.error_type, None);
}

/// (attributes, count) of every data point of a metric.
fn points(exp: &InMemoryMetricExporter, name: &str) -> Vec<(Vec<(String, String)>, u64)> {
    let mut out = Vec::new();
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for m in sm.metrics().filter(|m| m.name() == name) {
                let attrs = |it: &mut dyn Iterator<Item = &opentelemetry::KeyValue>| {
                    let mut v: Vec<(String, String)> = it
                        .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                        .collect();
                    v.sort();
                    v
                };
                match m.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            out.push((attrs(&mut dp.attributes()), dp.count()));
                        }
                    }
                    AggregatedMetrics::U64(MetricData::Sum(s)) => {
                        for dp in s.data_points() {
                            out.push((attrs(&mut dp.attributes()), dp.value()));
                        }
                    }
                    AggregatedMetrics::U64(MetricData::Gauge(g)) => {
                        for dp in g.data_points() {
                            out.push((attrs(&mut dp.attributes()), dp.value()));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

fn has(attrs: &[(String, String)], k: &str, v: &str) -> bool {
    attrs.iter().any(|(a, b)| a == k && b == v)
}

#[tokio::test]
async fn gen_ai_metrics_are_recorded() {
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let gw = Arc::new(
        Gateway::new(
            Arc::new(FakeEngine::fixed("abcd")),
            GatewayConfig::default(),
        )
        .with_meter(&mp.meter("test")),
    );
    let app = router(gw);
    let (s, _, _) = post(&app, "/v1/chat/completions", &chat(false), false).await;
    assert_eq!(s, 200);
    let (s, _, _) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "nope", "messages": [{"role": "user", "content": "hi"}]}),
        false,
    )
    .await;
    assert_eq!(s, 404);
    mp.force_flush().unwrap();

    let ttft = points(&exp, "gen_ai.server.time_to_first_token");
    assert_eq!(ttft.len(), 1);
    assert!(has(&ttft[0].0, "gen_ai.operation.name", "chat"));
    assert!(has(&ttft[0].0, "gen_ai.request.model", "fake"));
    let tpot = points(&exp, "gen_ai.server.time_per_output_token");
    assert_eq!(tpot[0].1, 1);
    let dur = points(&exp, "gen_ai.server.request.duration");
    assert!(
        dur.iter()
            .any(|(a, n)| *n == 1 && !a.iter().any(|(k, _)| k == "error.type"))
    );
    assert!(dur.iter().any(|(a, _)| has(a, "error.type", "not_found")));
    let tokens = points(&exp, "tetraeder.tokens");
    assert!(
        tokens
            .iter()
            .any(|(a, n)| has(a, "tetraeder.token.direction", "output") && *n == 4)
    );
    assert!(tokens.iter().any(|(a, n)| {
        has(a, "tetraeder.token.direction", "input") && *n == chatml_user_tokens("hi") as u64
    }));
    let cost = points(&exp, "tetraeder.request.cost");
    for res in ["gpu_seconds", "bytes_streamed"] {
        assert!(
            cost.iter()
                .any(|(a, _)| has(a, "tetraeder.cost.resource", res)
                    && has(a, "tetraeder.caller", "alice")),
            "{cost:?}"
        );
    }
}

#[tokio::test]
async fn metric_labels_stay_bounded() {
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let gw = Arc::new(
        Gateway::new(Arc::new(FakeEngine::fixed("a")), GatewayConfig::default())
            .with_meter(&mp.meter("test")),
    );
    let app = router(gw);
    // Unknown model names come from the client; they must not become label values.
    let (s, _, _) = post(
        &app,
        "/v1/chat/completions",
        &json!({"model": "random-name-1", "messages": [{"role": "user", "content": "hi"}]}),
        false,
    )
    .await;
    assert_eq!(s, 404);
    // So must caller ids.
    for i in 0..150 {
        let mut body = chat(false);
        body["user"] = json!(format!("caller-{i}"));
        let (s, _, _) = post(&app, "/v1/chat/completions", &body, false).await;
        assert_eq!(s, 200);
    }
    mp.force_flush().unwrap();

    let dur = points(&exp, "gen_ai.server.request.duration");
    assert!(
        !dur.iter()
            .any(|(a, _)| has(a, "gen_ai.request.model", "random-name-1")),
        "{dur:?}"
    );
    let cost = points(&exp, "tetraeder.request.cost");
    let callers: std::collections::HashSet<&str> = cost
        .iter()
        .flat_map(|(a, _)| a.iter())
        .filter(|(k, _)| k == "tetraeder.caller")
        .map(|(_, v)| v.as_str())
        .collect();
    assert!(callers.len() <= 101, "{} callers", callers.len());
    assert!(callers.contains("other"), "{callers:?}");
}

#[tokio::test]
async fn engine_overload_uses_the_configured_retry_after() {
    let mut engine = FakeEngine::fixed("a");
    engine.reject = Some(tetraeder_gateway::engine::EngineError::Overloaded);
    let cfg = GatewayConfig {
        retry_after_secs: 7,
        ..GatewayConfig::default()
    };
    let (app, _) = app_with(engine, cfg);
    let (s, h, _) = post(&app, "/v1/chat/completions", &chat(false), false).await;
    assert_eq!(s, 429);
    assert_eq!(h["retry-after"], "7");
}

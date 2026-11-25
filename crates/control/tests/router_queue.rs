//! A request the router cannot place does not wait without end: with no replica that takes
//! requests (none up) it is refused at once as overloaded, and one that waits
//! for a slot past the queue deadline ends with an error (ADR 0091, pre-exit checks).
//!
//! Hardware label: none.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_control::head::{ClusterEngine, ReplicaTarget};
use tetraeder_gateway::engine::{Engine, EngineError, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tokio_util::sync::CancellationToken;

fn engine(cap: u32) -> ClusterEngine {
    let mut slow = FakeEngine::new(|_| "x".repeat(40));
    slow.token_delay = Duration::from_millis(50);
    let slow: Arc<dyn Engine> = Arc::new(slow);
    ClusterEngine::new(
        Arc::clone(&slow),
        vec![ReplicaTarget {
            engine: slow,
            remote: None,
            cap,
        }],
        8,
        None,
    )
}

fn request() -> GenerateRequest {
    GenerateRequest {
        model: "fake".into(),
        prompt: vec![1, 2, 3],
        params: SamplingParams {
            max_tokens: 40,
            ..SamplingParams::default()
        },
        constraint: None,
        cancel: CancellationToken::new(),
    }
}

// hardware: none
#[test]
fn no_replica_up_refuses_at_once() {
    let e = engine(1);
    assert!(matches!(
        e.generate(request()),
        Err(EngineError::Overloaded)
    ));
    e.set_up(0, true);
    assert!(e.generate(request()).is_ok());
}

// hardware: none
#[test]
fn a_draining_replica_keeps_the_requests_waiting_until_the_deadline() {
    // A planned drain is replaced by the replanner: its requests wait, within the deadline.
    let e = engine(1);
    e.set_queue_deadline(Duration::from_millis(200));
    e.set_up(0, true);
    e.set_draining(0, true);
    let mut waiting = e.generate(request()).unwrap();
    assert_eq!(e.queued(), 1);
    std::thread::sleep(Duration::from_millis(600));
    assert!(matches!(waiting.try_recv(), Ok(GenEvent::Error(_))));
}

// hardware: none
#[test]
fn a_request_that_waits_past_the_deadline_ends_with_one_error() {
    let e = engine(1);
    e.set_queue_deadline(Duration::from_millis(200));
    e.set_up(0, true);
    // Takes the only slot for about two seconds.
    let _running = e.generate(request()).unwrap();
    let t0 = Instant::now();
    let mut waiting = e.generate(request()).unwrap();
    assert_eq!(e.queued(), 1);
    let ev = loop {
        match waiting.try_recv() {
            Ok(ev) => break ev,
            Err(_) => {
                assert!(t0.elapsed() < Duration::from_secs(1), "no end after 1 s");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    };
    assert!(
        matches!(&ev, GenEvent::Error(m) if m.contains("no replica took the request")),
        "{ev:?}"
    );
    assert!(t0.elapsed() >= Duration::from_millis(200));
    assert_eq!(e.queued(), 0, "it left the queue");
    // Nothing follows the error.
    std::thread::sleep(Duration::from_millis(100));
    assert!(waiting.try_recv().is_err());
}

// hardware: none
#[test]
fn a_replica_being_replaced_keeps_the_requests_waiting_until_the_new_one_joins() {
    // A planned drain ends with the old replica leaving the router and the new one joining. A
    // request that arrives between the two waits for it; it is not refused as overloaded
    // (cluster_integration, the planned drain: clients streaming through the replan).
    let e = engine(1);
    e.set_up(0, true);
    assert!(e.drain(0, Duration::from_secs(1)));
    e.set_up(0, false);
    let mut waiting = e
        .generate(request())
        .expect("a replica being replaced is coming back: wait, do not refuse");
    assert_eq!(e.queued(), 1);
    e.set_up(0, true);
    let t0 = Instant::now();
    loop {
        match waiting.try_recv() {
            Ok(GenEvent::Finished { .. }) => break,
            Ok(GenEvent::Error(m)) => panic!("{m}"),
            _ => {
                assert!(t0.elapsed() < Duration::from_secs(10), "never finished");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

// hardware: none
#[test]
fn a_replacement_that_is_given_up_refuses_at_once_again() {
    let e = engine(1);
    e.set_up(0, true);
    assert!(e.drain(0, Duration::from_secs(1)));
    e.set_up(0, false);
    e.end_replace(0);
    assert!(matches!(
        e.generate(request()),
        Err(EngineError::Overloaded)
    ));
}

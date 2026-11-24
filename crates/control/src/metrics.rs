//! Metrics of the router (ADR 0091, step 5; rule 6): `tetraeder.router.requests` (counter,
//! `replica` and `result`: `ok`, `failed_in_flight`, `moved`, `affinity`),
//! `tetraeder.router.replicas.up` (gauge) and `tetraeder.kv.prefix.fetch` (counter, `result`:
//! `fetched`, `recomputed`). Dashboard row "Router: requests per replica and result, replicas
//! up, prefix fetches" and alert `TetraederReplicaOut`.

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge};
use std::sync::OnceLock;

struct M {
    requests: Counter<u64>,
    up: Gauge<u64>,
    prefix: Counter<u64>,
}

fn m() -> &'static M {
    static M: OnceLock<M> = OnceLock::new();
    M.get_or_init(|| {
        let meter = opentelemetry::global::meter("tetraeder");
        M {
            requests: meter
                .u64_counter("tetraeder.router.requests")
                .with_description("Requests by replica and what happened to them")
                .build(),
            up: meter
                .u64_gauge("tetraeder.router.replicas.up")
                .with_description("Replicas the router sends requests to")
                .build(),
            prefix: meter
                .u64_counter("tetraeder.kv.prefix.fetch")
                .with_description("Prefixes another replica held: fetched or recomputed")
                .build(),
        }
    })
}

/// `result`: `ok`, `failed_in_flight`, `moved` or `affinity`.
pub fn request(replica: u32, result: &'static str) {
    m().requests.add(
        1,
        &[
            KeyValue::new("replica", i64::from(replica)),
            KeyValue::new("result", result),
        ],
    );
}

pub fn replicas_up(n: usize) {
    m().up.record(n as u64, &[]);
}

/// `result`: `fetched` or `recomputed`.
pub fn prefix_fetch(result: &'static str) {
    m().prefix.add(1, &[KeyValue::new("result", result)]);
}

//! Metrics of the router (ADR 0091, step 5; rule 6): `tetraeder.router.requests` (counter,
//! `replica` and `result`: `ok`, `failed_in_flight`, `moved`, `affinity`),
//! `tetraeder.router.replicas.up` and `.planned` (gauges) and `tetraeder.kv.prefix.fetch` (counter, `result`:
//! `fetched`, `recomputed`), `tetraeder.kv.prefix.served` (counter, `result`: `snapshot`, `gone`;
//! prefixes this leader gave to another), `tetraeder.cluster.drain` (counter, `replica` and
//! `result`: `drained`, `timed_out`; planned drains of a replica) and its duration
//! `tetraeder.cluster.drain.duration`. Dashboard row "Router: requests per replica and result, replicas
//! up, prefix fetches" and alert `TetraederReplicaOut`.

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram};
use std::sync::OnceLock;

struct M {
    requests: Counter<u64>,
    up: Gauge<u64>,
    planned: Gauge<u64>,
    prefix: Counter<u64>,
    served: Counter<u64>,
    drains: Counter<u64>,
    drain_s: Histogram<f64>,
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
            planned: meter
                .u64_gauge("tetraeder.router.replicas.planned")
                .with_description("Replicas the plan has")
                .build(),
            prefix: meter
                .u64_counter("tetraeder.kv.prefix.fetch")
                .with_description("Prefixes another replica held: fetched or recomputed")
                .build(),
            served: meter
                .u64_counter("tetraeder.kv.prefix.served")
                .with_description("Prefixes this leader gave to another replica, or no longer had")
                .build(),
            drains: meter
                .u64_counter("tetraeder.cluster.drain")
                .with_description("Planned drains of a replica by replica and result")
                .build(),
            drain_s: meter
                .f64_histogram("tetraeder.cluster.drain.duration")
                .with_unit("s")
                .with_description("Time a planned drain waited for the replica's requests")
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

/// Replicas the plan has (the baseline of `TetraederReplicaOut`).
pub fn replicas_planned(n: usize) {
    m().planned.record(n as u64, &[]);
}

/// `result`: `fetched` or `recomputed`.
pub fn prefix_fetch(result: &'static str) {
    m().prefix.add(1, &[KeyValue::new("result", result)]);
}

/// `result`: `snapshot` or `gone`.
pub fn prefix_served(result: &'static str) {
    m().served.add(1, &[KeyValue::new("result", result)]);
}

/// A planned drain of `replica` ended: `drained` (nothing left in flight) or `timed_out`.
pub fn drain(replica: u32, result: &'static str, took: std::time::Duration) {
    m().drains.add(
        1,
        &[
            KeyValue::new("replica", i64::from(replica)),
            KeyValue::new("result", result),
        ],
    );
    m().drain_s.record(took.as_secs_f64(), &[]);
}

//! Metrics of the router (ADR 0091, step 5; rule 6): `tetraeder.router.requests` (counter,
//! `replica` and `result`: `ok`, `failed_in_flight`, `moved`, `affinity`),
//! `tetraeder.router.replicas.up` and `.planned` (gauges) and `tetraeder.kv.prefix.fetch` (counter, `result`:
//! `fetched`, `recomputed`), `tetraeder.kv.prefix.served` (counter, `result`: `snapshot`, `gone`;
//! prefixes this leader gave to another), `tetraeder.cluster.drain` (counter, `replica` and
//! `result`: `drained`, `timed_out`; planned drains of a replica) and its duration
//! `tetraeder.cluster.drain.duration`, and `tetraeder.cluster.join.attempts` (counter, `replica` and
//! `result`: `joined`, `failed`; every try of the head to start and connect a remote replica,
//! alert `TetraederReplicaJoinFailing`). Dashboard row "Router: requests per replica and result, replicas
//! up, prefix fetches" and alert `TetraederReplicaOut`.
//!
//! The prefix fetch counter also carries `reason` (why a prefix another replica held was fetched
//! or recomputed, see [`FETCH_REASONS`]), with `tetraeder.kv.prefix.fetch.bytes` (snapshot bytes
//! fetched) and alert `TetraederPrefixFetchFailing`. The head renders both on the gateway's
//! `/metrics` too ([`prefix_text`]), so they are readable without an OTLP collector.

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

struct M {
    requests: Counter<u64>,
    up: Gauge<u64>,
    planned: Gauge<u64>,
    prefix: Counter<u64>,
    prefix_bytes: Counter<u64>,
    served: Counter<u64>,
    drains: Counter<u64>,
    drain_s: Histogram<f64>,
    joins: Counter<u64>,
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
                .with_description("Prefixes another replica held: fetched or recomputed, and why")
                .build(),
            prefix_bytes: meter
                .u64_counter("tetraeder.kv.prefix.fetch.bytes")
                .with_unit("By")
                .with_description("Snapshot bytes fetched from another replica")
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
            joins: meter
                .u64_counter("tetraeder.cluster.join.attempts")
                .with_description("Tries of the head to start and connect a remote replica")
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

/// Why a prefix another replica held was fetched or recomputed. Decided by the head:
/// `no_policy` (no hardware graph to price it), `target_in_process` (the head's own replica
/// cannot be told to fetch), `priced_recompute`, `same_node`, `unknown_kv`, `too_short`,
/// `no_rate`, `no_link` (the rule), `policy_recompute` (another policy), `not_offered` (the
/// holder could not be told). Reported by the fetching replica once asked: `fetched`, `gone`
/// (the holder no longer had it: evicted, or held only by a live sequence), `failed`,
/// `bad_snapshot`, `not_installed` (its cache refused it), `refused` (the replica cannot check
/// or hold it), `unknown` (an older replica).
pub const FETCH_REASONS: &[&str] = &[
    "fetched",
    "no_policy",
    "target_in_process",
    "priced_recompute",
    "same_node",
    "unknown_kv",
    "too_short",
    "no_rate",
    "no_link",
    "policy_recompute",
    "not_offered",
    "gone",
    "failed",
    "bad_snapshot",
    "not_installed",
    "refused",
    "unknown",
];

/// `reason` as one of [`FETCH_REASONS`]: what a replica reports is bounded to them.
pub fn fetch_reason(reason: &str) -> &'static str {
    FETCH_REASONS
        .iter()
        .find(|r| **r == reason)
        .copied()
        .unwrap_or("unknown")
}

/// The counters of [`prefix_text`], kept beside the OTel instruments.
struct Mirror {
    fetch: Mutex<BTreeMap<(&'static str, &'static str), u64>>,
    bytes: AtomicU64,
}

fn mirror() -> &'static Mirror {
    static M: OnceLock<Mirror> = OnceLock::new();
    M.get_or_init(|| Mirror {
        fetch: Mutex::new(BTreeMap::new()),
        bytes: AtomicU64::new(0),
    })
}

/// A prefix another replica held: `fetched` (reason `fetched`) or `recomputed` (reason: why).
pub fn prefix_fetch(result: &'static str, reason: &'static str) {
    m().prefix.add(
        1,
        &[
            KeyValue::new("result", result),
            KeyValue::new("reason", reason),
        ],
    );
    *mirror()
        .fetch
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry((result, reason))
        .or_default() += 1;
}

/// Snapshot bytes a replica fetched from another.
pub fn prefix_fetch_bytes(bytes: u64) {
    m().prefix_bytes.add(bytes, &[]);
    mirror().bytes.fetch_add(bytes, Ordering::Relaxed);
}

/// The prefix fetch counters of this process in Prometheus text, under the names the OTel export
/// gives them (`tetraeder_kv_prefix_fetch_total`, `tetraeder_kv_prefix_fetch_bytes_total`).
pub fn prefix_text() -> String {
    let mut s = String::from(
        "# HELP tetraeder_kv_prefix_fetch_total Prefixes another replica held: fetched or recomputed, and why\n\
         # TYPE tetraeder_kv_prefix_fetch_total counter\n",
    );
    for ((result, reason), n) in mirror()
        .fetch
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
    {
        s += &format!(
            "tetraeder_kv_prefix_fetch_total{{result=\"{result}\",reason=\"{reason}\"}} {n}\n"
        );
    }
    s += &format!(
        "# HELP tetraeder_kv_prefix_fetch_bytes_total Snapshot bytes fetched from another replica\n\
         # TYPE tetraeder_kv_prefix_fetch_bytes_total counter\n\
         tetraeder_kv_prefix_fetch_bytes_total {}\n",
        mirror().bytes.load(Ordering::Relaxed)
    );
    s
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

/// One try of the head's join loop for `replica` ended: `joined` or `failed`.
pub fn join_attempt(replica: u32, result: &'static str) {
    m().joins.add(
        1,
        &[
            KeyValue::new("replica", i64::from(replica)),
            KeyValue::new("result", result),
        ],
    );
}

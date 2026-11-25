//! Link metrics (ADR 0091, rule 6): `tetraeder.link.bytes`, `tetraeder.link.latency`,
//! `tetraeder.cluster.tls.rejected`.

use crate::NodeId;
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram};

/// Bytes sent and round trips of one connection, with the attributes `source`, `destination`,
/// `transport` and `tetraeder.purpose`. Bytes are counted on the sending side only, so a sum
/// over nodes counts each byte once.
#[derive(Clone)]
pub struct LinkMetrics {
    bytes: Counter<u64>,
    latency: Histogram<f64>,
    attrs: Vec<KeyValue>,
}

impl LinkMetrics {
    pub fn new(
        source: &NodeId,
        destination: &NodeId,
        transport: &'static str,
        purpose: &'static str,
    ) -> Self {
        let meter = opentelemetry::global::meter("tetraeder");
        Self {
            bytes: meter
                .u64_counter("tetraeder.link.bytes")
                .with_unit("By")
                .with_description("Bytes sent over a link between nodes")
                .build(),
            latency: meter
                .f64_histogram("tetraeder.link.latency")
                .with_unit("s")
                .with_description("Round trip of one frame over a link between nodes")
                .build(),
            attrs: vec![
                KeyValue::new("source", source.0.clone()),
                KeyValue::new("destination", destination.0.clone()),
                KeyValue::new("transport", transport),
                KeyValue::new("tetraeder.purpose", purpose),
            ],
        }
    }

    /// The same link with another purpose label (`event`, `logits`, ...).
    #[must_use]
    pub fn with_purpose(&self, purpose: &'static str) -> Self {
        let mut m = self.clone();
        m.attrs[3] = KeyValue::new("tetraeder.purpose", purpose);
        m
    }

    pub fn bytes(&self, n: usize) {
        self.bytes.add(n as u64, &self.attrs);
    }

    pub fn latency(&self, secs: f64) {
        self.latency.record(secs, &self.attrs);
    }
}

static REJECTED: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, u64>>,
> = std::sync::LazyLock::new(Default::default);

/// Refusals of this process so far for `reason` (the counter's value, readable by tests).
pub fn rejected_total(reason: &str) -> u64 {
    REJECTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(reason)
        .copied()
        .unwrap_or(0)
}

/// A refused handshake. `reason` is one of `untrusted`, `unknown_node`, `no_identity`,
/// `wrong_peer`, `bad_hello`, `handshake` (no preface within 10 s), `tls_timeout` (TLS not
/// complete within 3 s), `limit` (too many handshakes or connections), `ip_limit` (too many
/// handshakes in flight from one source address) and `peer_limit` (too many connections of one
/// peer). The cluster code adds `not_leader` (a segment connection from a node that is neither
/// the head nor the replica's leader), `no_grant` (a prefix fetch the head did not announce),
/// `bad_snapshot` (a fetched snapshot that does not match its key), `bad_profile` (an
/// implausible hardware graph), `bad_frame` (a frame over the request limit) and `bad_relay`
/// (a connection to a replica's loopback prefix listener without the launch token).
/// `duplicate` is no longer produced: a new connection replaces the old one.
pub fn rejected(reason: &'static str) {
    *REJECTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(reason)
        .or_insert(0) += 1;
    opentelemetry::global::meter("tetraeder")
        .u64_counter("tetraeder.cluster.tls.rejected")
        .with_description("Connections refused during the handshake")
        .build()
        .add(1, &[KeyValue::new("reason", reason)]);
}

/// A writer that counts what it wrote in `tetraeder.link.bytes`.
pub struct CountingWriter<W> {
    inner: W,
    m: LinkMetrics,
}

impl<W> CountingWriter<W> {
    pub fn new(inner: W, m: LinkMetrics) -> Self {
        Self { inner, m }
    }
}

impl<W: std::io::Write> std::io::Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.m.bytes(n);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

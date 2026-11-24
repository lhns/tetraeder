//! Remote KV, Phase 6 scope (ADR 0091): a replica keeps its own prefix cache; the bytes of a
//! prefix cross the network only when a request lands on a replica that lacks a prefix another
//! one holds, and only when fetching is predicted faster than recomputing it.
//!
//! Wire: `Purpose::Prefix` between two replica leaders, one frame per request, the snapshot as
//! the reply's payload. A snapshot is only offered to a replica of the same `kv_format` (the
//! cut points of a pipeline are part of it, step 4), so a foreign one is never restored.

use crate::metrics;
use serde::{Deserialize, Serialize};
use std::io;
use tetraeder_profiler::HardwareGraph;
use tetraeder_transport::frame;
use tetraeder_transport::{Conn, Peer, Purpose, Transport};

/// What a snapshot is keyed by: the cache format and the tokens it holds, as in the prefix
/// cache (`PrefixCache::insert(format, tokens, bytes)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixKey {
    pub format: String,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum PrefixRequest {
    Fetch { key: PrefixKey },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum PrefixReply {
    /// The snapshot is the payload; `tokens` are the tokens it holds. A cache entry is a whole
    /// sequence, so they can run past the ones asked for.
    Snapshot {
        tokens: Vec<u32>,
    },
    /// Evicted or never there: the caller recomputes.
    Gone,
    Error {
        msg: String,
    },
}

/// The replica's prefix cache as the prefix server and the fetching side see it.
pub trait PrefixStore: Send + Sync {
    /// The format of this replica's snapshots (`Backend::kv_format`).
    fn kv_format(&self) -> String;
    /// The snapshot sharing the longest prefix with `key.tokens` (at least the cache's restore
    /// gain); the tokens it holds come back with it and can be more than asked for.
    fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)>;
    /// Installs a fetched snapshot of `key.tokens`, as if this replica had computed it.
    fn put(&self, key: PrefixKey, bytes: Vec<u8>);
    /// Bytes of snapshot per cached token, for the head's fetch rule; 0 when not known.
    fn row_bytes(&self) -> u64 {
        0
    }
}

/// The fetch rule: fetch when the prefix is long enough to matter (`min_gain_tokens`, the
/// prefix cache's `RESTORE_MIN_GAIN`) and `hop_time(bytes) + restore_s` is below the time to
/// recompute `tokens` at the replica's predicted prefill rate. No curve for the link: recompute.
#[allow(clippy::too_many_arguments)]
pub fn fetch_beats_recompute(
    hw: &HardwareGraph,
    from: &str,
    to: &str,
    bytes: u64,
    tokens: u32,
    predicted_pp_tps: f64,
    restore_s: f64,
    min_gain_tokens: u32,
) -> bool {
    if tokens < min_gain_tokens || predicted_pp_tps <= 0.0 || predicted_pp_tps.is_nan() {
        return false;
    }
    let Some(hop) = tetraeder_costmodel::cluster::hop_time(hw, from, to, bytes as f64) else {
        return false;
    };
    hop + restore_s < f64::from(tokens) / predicted_pp_tps
}

/// Serves `Fetch` requests on one inbound `Prefix` connection until it ends.
pub fn serve_prefix(mut conn: Conn, store: &dyn PrefixStore) {
    serve_prefix_io(&mut conn.rx, &mut conn.tx, store);
    (conn.close)();
}

/// [`serve_prefix`] over any byte streams: the daemon relays a `Prefix` connection to the
/// replica's own listener, which serves it from here.
pub fn serve_prefix_io<R: io::Read, W: io::Write>(rx: &mut R, tx: &mut W, store: &dyn PrefixStore) {
    while let Ok(Some(f)) = frame::read_frame_max(rx, Purpose::Prefix.max_frame()) {
        let (reply, payload) = match frame::parse_header::<PrefixRequest>(&f) {
            Ok(PrefixRequest::Fetch { key }) => {
                if key.format != store.kv_format() {
                    (
                        PrefixReply::Error {
                            msg: format!("this replica's cache format is `{}`", store.kv_format()),
                        },
                        Vec::new(),
                    )
                } else {
                    match store.get(&key) {
                        Some((tokens, bytes)) => {
                            metrics::prefix_served("snapshot");
                            (PrefixReply::Snapshot { tokens }, bytes)
                        }
                        None => {
                            metrics::prefix_served("gone");
                            (PrefixReply::Gone, Vec::new())
                        }
                    }
                }
            }
            Err(e) => (PrefixReply::Error { msg: e }, Vec::new()),
        };
        if frame::write_frame(&mut *tx, &reply, &payload).is_err() {
            break;
        }
    }
}

/// Relays an inbound `Prefix` connection to the replica's own prefix listener on `addr` (a
/// loopback port: the replica's cache lives in its child process, which the daemon reaches only
/// through its stdio, so the child listens for this one purpose). Returns when either side ends;
/// the connection is closed then. If the listener is not there, the connection is closed and the
/// fetcher recomputes.
pub fn relay_prefix(conn: Conn, addr: std::net::SocketAddr) {
    let Conn {
        mut rx,
        mut tx,
        close,
        ..
    } = conn;
    let stream =
        match std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2)) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%addr, "prefix listener not reachable: {e}");
                close();
                return;
            }
        };
    let Ok(mut from_replica) = stream.try_clone() else {
        close();
        return;
    };
    let mut to_replica = stream;
    let up = std::thread::spawn(move || {
        let _ = io::copy(&mut rx, &mut to_replica);
        let _ = to_replica.shutdown(std::net::Shutdown::Write);
    });
    let _ = io::copy(&mut from_replica, &mut tx);
    let _ = io::Write::flush(&mut tx);
    close();
    let _ = up.join();
}

/// Asks the leader `peer` for a snapshot: the tokens it holds and its bytes. `Ok(None)`: it is
/// gone, recompute.
pub fn fetch_prefix(
    t: &dyn Transport,
    peer: &Peer,
    key: &PrefixKey,
) -> io::Result<Option<(Vec<u32>, Vec<u8>)>> {
    let mut c = t.connect(peer, Purpose::Prefix)?;
    let r = (|| {
        frame::write_frame(&mut c.tx, &PrefixRequest::Fetch { key: key.clone() }, &[])
            .map_err(io::Error::other)?;
        let f = frame::read_frame_max(&mut c.rx, c.purpose.max_frame())
            .map_err(io::Error::other)?
            .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
        match frame::parse_header::<PrefixReply>(&f).map_err(io::Error::other)? {
            PrefixReply::Snapshot { tokens } => Ok(Some((tokens, f.payload))),
            PrefixReply::Gone => Ok(None),
            PrefixReply::Error { msg } => Err(io::Error::other(msg)),
        }
    })();
    (c.close)();
    r
}

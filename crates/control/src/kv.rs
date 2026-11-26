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
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tetraeder_profiler::HardwareGraph;
use tetraeder_transport::frame;
use tetraeder_transport::{Conn, NodeId, Peer, Purpose, Transport};

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
    /// Installs a fetched snapshot of `key.tokens`, as if this replica had computed it; false
    /// when the cache did not take it (off, too short, already held).
    fn put(&self, key: PrefixKey, bytes: Vec<u8>) -> bool;
    /// Bytes of snapshot per cached token, for the head's fetch rule; 0 when not known. A
    /// replica fetches only when it is known: the size of a reply is checked against it.
    fn row_bytes(&self) -> u64 {
        0
    }
    /// Most tokens a snapshot of this replica can hold (its context length).
    fn max_tokens(&self) -> u64 {
        u64::MAX
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

/// Largest request frame of a `Prefix` connection: a `Fetch` is a key, never bulk (replies
/// carry the snapshot and keep the large limit).
pub const PREFIX_REQUEST_MAX: u64 = 1 << 20;
/// How long an announced grant stays valid, and how many wait at once (in total and per peer):
/// a grant holds a token list, so the table is bounded.
const GRANT_TTL: Duration = Duration::from_secs(30);
const MAX_GRANTS: usize = 64;
const MAX_GRANTS_PER_PEER: usize = 8;
/// How long a fetch waits for the head's announcement, which travels on another connection and
/// can arrive after the fetch.
const GRANT_WAIT: Duration = Duration::from_secs(2);
/// Requests served on one inbound `Prefix` connection (a fetch uses one).
const MAX_REQUESTS_PER_CONN: usize = 16;
/// Longest first line the daemon sends a replica's prefix listener (token, peer id).
const RELAY_PREFACE_MAX: usize = 256;

struct Grant {
    peer: NodeId,
    key: PrefixKey,
    at: Instant,
}

/// What other nodes may fetch from this replica: only what the head announced (`Offer`), once,
/// for the node the head named. A configured node that the head did not send cannot read the
/// prefixes of other users.
#[derive(Default)]
pub struct PrefixGrants {
    grants: Mutex<Vec<Grant>>,
    cv: Condvar,
}

impl PrefixGrants {
    fn list(&self) -> MutexGuard<'_, Vec<Grant>> {
        self.grants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The head announces that `peer` may fetch `key` once.
    pub fn offer(&self, peer: NodeId, key: PrefixKey) {
        let mut g = self.list();
        g.retain(|x| x.at.elapsed() < GRANT_TTL);
        while g.iter().filter(|x| x.peer == peer).count() >= MAX_GRANTS_PER_PEER {
            if let Some(i) = g.iter().position(|x| x.peer == peer) {
                g.remove(i);
            }
        }
        while g.len() >= MAX_GRANTS {
            g.remove(0);
        }
        g.push(Grant {
            peer,
            key,
            at: Instant::now(),
        });
        self.cv.notify_all();
    }

    /// Uses the grant of `peer` for `key`, waiting up to `wait` for it to be announced.
    pub fn take(&self, peer: &NodeId, key: &PrefixKey, wait: Duration) -> bool {
        let end = Instant::now() + wait;
        let mut g = self.list();
        loop {
            g.retain(|x| x.at.elapsed() < GRANT_TTL);
            if let Some(i) = g.iter().position(|x| x.peer == *peer && x.key == *key) {
                g.remove(i);
                return true;
            }
            let left = end.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            g = self
                .cv
                .wait_timeout(g, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// Grants waiting.
    pub fn len(&self) -> usize {
        self.list().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A random token for one launch of a replica: the daemon passes it to the child by environment
/// and puts it first on every connection it relays to the child's loopback listener, so another
/// process of the machine that finds the port cannot read the cache. Seeded by the operating
/// system through the standard library's hasher keys.
pub fn new_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::with_capacity(32);
    for i in 0u64..2 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.write_u32(std::process::id());
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

fn same_token(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// The first line the daemon sends on a relayed connection: `<token> <peer>\n`.
pub fn write_relay_preface<W: io::Write>(w: &mut W, token: &str, peer: &NodeId) -> io::Result<()> {
    w.write_all(format!("{token} {}\n", peer.0).as_bytes())
}

/// What the listener answers to a valid preface, so the daemon knows the listener it dialed is
/// the replica's and not a stale file's port, now owned by another loopback service.
pub const RELAY_ACK: &[u8] = b"tetraeder-prefix ok\n";

/// Reads the first line of a relayed connection. `Some(peer)` when it carries `token`; `None`
/// for a wrong token, a malformed or over-long line, or an end of input. It reads byte by byte
/// so nothing after the line is consumed.
pub fn read_relay_preface<R: io::Read>(r: &mut R, token: &str) -> Option<NodeId> {
    let mut line = Vec::new();
    let mut b = [0u8; 1];
    while line.len() <= RELAY_PREFACE_MAX {
        match r.read(&mut b) {
            Ok(1) if b[0] == b'\n' => {
                let text = std::str::from_utf8(&line).ok()?;
                let (t, peer) = text.split_once(' ')?;
                return (same_token(t, token) && !peer.is_empty() && !peer.contains(' '))
                    .then(|| NodeId::new(peer));
            }
            Ok(1) => line.push(b[0]),
            _ => return None,
        }
    }
    None
}

/// Serves the `Fetch` requests of the connection's peer on one inbound `Prefix` connection until
/// it ends.
pub fn serve_prefix(mut conn: Conn, store: &dyn PrefixStore, grants: &PrefixGrants) {
    let peer = conn.peer.clone();
    serve_prefix_io(&mut conn.rx, &mut conn.tx, store, grants, &peer);
    (conn.close)();
}

/// [`serve_prefix`] over any byte streams: the daemon relays a `Prefix` connection to the
/// replica's own listener, which serves it from here. `peer` is the authenticated node on the
/// other end. A fetch is served only when the head announced it (`grants`); the reply holds the
/// snapshot only when it covers the asked tokens.
pub fn serve_prefix_io<R: io::Read, W: io::Write>(
    rx: &mut R,
    tx: &mut W,
    store: &dyn PrefixStore,
    grants: &PrefixGrants,
    peer: &NodeId,
) {
    for _ in 0..MAX_REQUESTS_PER_CONN {
        let f = match frame::read_frame_max(rx, PREFIX_REQUEST_MAX) {
            Ok(Some(f)) => f,
            Ok(None) => return,
            Err(_) => {
                tetraeder_transport::metrics::rejected("bad_frame");
                return;
            }
        };
        let mut last = false;
        let (reply, payload) = match frame::parse_header::<PrefixRequest>(&f) {
            Ok(PrefixRequest::Fetch { key }) => {
                if key.format != store.kv_format() {
                    (
                        PrefixReply::Error {
                            msg: format!("this replica's cache format is `{}`", store.kv_format()),
                        },
                        Vec::new(),
                    )
                } else if !grants.take(peer, &key, GRANT_WAIT) {
                    tetraeder_transport::metrics::rejected("no_grant");
                    last = true;
                    (
                        PrefixReply::Error {
                            msg: "this fetch was not announced by the head".into(),
                        },
                        Vec::new(),
                    )
                } else {
                    match store.get(&key) {
                        Some((tokens, bytes)) if tokens.starts_with(&key.tokens) => {
                            metrics::prefix_served("snapshot");
                            tracing::info!(
                                peer = %peer,
                                tokens = key.tokens.len(),
                                held = tokens.len(),
                                bytes = bytes.len(),
                                "prefix served"
                            );
                            (PrefixReply::Snapshot { tokens }, bytes)
                        }
                        _ => {
                            metrics::prefix_served("gone");
                            // Only idle sequences and cache entries are served: a prefix that
                            // a running request extends in place is not.
                            tracing::info!(
                                peer = %peer,
                                tokens = key.tokens.len(),
                                "prefix gone: no idle sequence or cache entry holds it"
                            );
                            (PrefixReply::Gone, Vec::new())
                        }
                    }
                }
            }
            Err(e) => {
                last = true;
                (PrefixReply::Error { msg: e }, Vec::new())
            }
        };
        if frame::write_frame(&mut *tx, &reply, &payload).is_err() || last {
            return;
        }
    }
}

/// Relays an inbound `Prefix` connection to the replica's own prefix listener on `addr` (a
/// loopback port: the replica's cache lives in its child process, which the daemon reaches only
/// through its stdio, so the child listens for this one purpose). The first line sent is the
/// launch's `token` and the authenticated peer (`read_relay_preface`). Returns when either side
/// ends; the connection is closed then. If the listener is not there or does not answer the
/// preface with `RELAY_ACK`, the connection is closed and the fetcher recomputes.
pub fn relay_prefix(conn: Conn, addr: std::net::SocketAddr, token: &str) {
    let Conn {
        mut rx,
        mut tx,
        close,
        peer,
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
    if write_relay_preface(&mut to_replica, token, &peer).is_err() {
        close();
        return;
    }
    // Relay nothing until the listener has answered the preface as ours.
    let mut ack = [0u8; RELAY_ACK.len()];
    let _ = to_replica.set_read_timeout(Some(std::time::Duration::from_secs(2)));
    let acked = io::Read::read_exact(&mut from_replica, &mut ack).is_ok() && ack == RELAY_ACK;
    let _ = to_replica.set_read_timeout(None);
    if !acked {
        tracing::warn!(%addr, "the prefix listener did not answer the handshake (a stale port file?); closed");
        close();
        return;
    }
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
/// gone, recompute. A reply frame over `max_reply` bytes is refused before it is read.
pub fn fetch_prefix(
    t: &dyn Transport,
    peer: &Peer,
    key: &PrefixKey,
    max_reply: u64,
) -> io::Result<Option<(Vec<u32>, Vec<u8>)>> {
    let mut c = t.connect(peer, Purpose::Prefix)?;
    let r = (|| {
        frame::write_frame(&mut c.tx, &PrefixRequest::Fetch { key: key.clone() }, &[])
            .map_err(io::Error::other)?;
        let f = frame::read_frame_max(&mut c.rx, max_reply.min(c.purpose.max_frame()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn key(n: u32) -> PrefixKey {
        PrefixKey {
            format: "f16".into(),
            tokens: (0..n).collect(),
        }
    }

    #[test]
    fn a_grant_is_for_one_peer_one_key_and_one_use() {
        let g = PrefixGrants::default();
        let (b, c) = (NodeId::new("b"), NodeId::new("c"));
        g.offer(b.clone(), key(4));
        assert!(!g.take(&c, &key(4), Duration::ZERO), "another node");
        assert!(!g.take(&b, &key(5), Duration::ZERO), "another key");
        assert!(g.take(&b, &key(4), Duration::ZERO));
        assert!(!g.take(&b, &key(4), Duration::ZERO), "used once");
        assert!(g.is_empty());
    }

    #[test]
    fn a_fetch_waits_for_a_grant_that_arrives_late() {
        let g = Arc::new(PrefixGrants::default());
        let g2 = Arc::clone(&g);
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            g2.offer(NodeId::new("b"), key(4));
        });
        assert!(g.take(&NodeId::new("b"), &key(4), Duration::from_secs(5)));
        t.join().unwrap();
    }

    #[test]
    fn grants_are_bounded_per_peer() {
        let g = PrefixGrants::default();
        for n in 0..100 {
            g.offer(NodeId::new("b"), key(n + 1));
        }
        assert_eq!(g.len(), MAX_GRANTS_PER_PEER);
        // The newest survive.
        assert!(g.take(&NodeId::new("b"), &key(100), Duration::ZERO));
    }

    #[test]
    fn the_relay_preface_needs_the_launch_token() {
        let t = new_token();
        assert_eq!(t.len(), 32);
        assert_ne!(t, new_token());
        let mut wire = Vec::new();
        write_relay_preface(&mut wire, &t, &NodeId::new("b")).unwrap();
        wire.extend_from_slice(b"rest");
        let mut r = wire.as_slice();
        assert_eq!(read_relay_preface(&mut r, &t), Some(NodeId::new("b")));
        // Nothing past the line was read.
        assert_eq!(r, b"rest");
        assert_eq!(read_relay_preface(&mut wire.as_slice(), &new_token()), None);
        assert_eq!(read_relay_preface(&mut &b"no newline"[..], &t), None);
        assert_eq!(
            read_relay_preface(&mut vec![b'x'; 1000].as_slice(), &t),
            None
        );
    }

    /// A listener on loopback that reads the preface line and answers `reply`, then echoes.
    fn listener(reply: &'static [u8]) -> std::net::SocketAddr {
        let l = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for mut s in l.incoming().flatten() {
                let mut b = [0u8; 1];
                while io::Read::read(&mut s, &mut b).is_ok_and(|n| n == 1) && b[0] != b'\n' {}
                let _ = io::Write::write_all(&mut s, reply);
                let mut c = s.try_clone().unwrap();
                let _ = io::copy(&mut s, &mut c);
            }
        });
        addr
    }

    struct Sink(Arc<std::sync::Mutex<Vec<u8>>>);

    impl io::Write for Sink {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn relay_to(addr: std::net::SocketAddr) -> (Vec<u8>, bool) {
        let out = Arc::new(std::sync::Mutex::new(Vec::new()));
        let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let c = Arc::clone(&closed);
        relay_prefix(
            Conn {
                peer: NodeId::new("b"),
                purpose: tetraeder_transport::Purpose::Prefix,
                rx: Box::new(&b"payload"[..]),
                tx: Box::new(Sink(Arc::clone(&out))),
                close: Arc::new(move || c.store(true, std::sync::atomic::Ordering::SeqCst)),
            },
            addr,
            "tok",
        );
        let got = out.lock().unwrap().clone();
        (got, closed.load(std::sync::atomic::Ordering::SeqCst))
    }

    #[test]
    fn a_listener_that_does_not_ack_the_preface_gets_no_relay() {
        // A stale port file whose port another loopback service now holds.
        let (got, closed) = relay_to(listener(b"HTTP/1.1 400 Bad Request\r\n\r\n"));
        assert!(closed);
        assert!(got.is_empty(), "nothing the other service said is relayed");
    }

    #[test]
    fn a_listener_that_acks_the_preface_is_relayed_to() {
        let (got, closed) = relay_to(listener(RELAY_ACK));
        assert!(closed);
        assert_eq!(got, b"payload");
    }

    struct One;

    impl PrefixStore for One {
        fn kv_format(&self) -> String {
            "f16".into()
        }
        fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
            Some((key.tokens.clone(), vec![1u8; 8]))
        }
        fn put(&self, _key: PrefixKey, _bytes: Vec<u8>) -> bool {
            false
        }
    }

    fn fetch_frame(k: &PrefixKey) -> Vec<u8> {
        let mut w = Vec::new();
        frame::write_frame(&mut w, &PrefixRequest::Fetch { key: k.clone() }, &[]).unwrap();
        w
    }

    #[test]
    fn only_an_announced_fetch_is_served() {
        let grants = PrefixGrants::default();
        let peer = NodeId::new("b");
        let mut wire = fetch_frame(&key(4));
        let mut out = Vec::new();
        // Not announced: refused (after the wait), nothing of the cache leaves.
        let t0 = Instant::now();
        serve_prefix_io(&mut wire.as_slice(), &mut out, &One, &grants, &peer);
        assert!(t0.elapsed() >= GRANT_WAIT);
        let f = frame::read_frame(&mut out.as_slice()).unwrap().unwrap();
        assert!(matches!(
            frame::parse_header::<PrefixReply>(&f).unwrap(),
            PrefixReply::Error { .. }
        ));
        assert!(f.payload.is_empty());
        // Announced for this node: served once.
        grants.offer(peer.clone(), key(4));
        out.clear();
        wire = fetch_frame(&key(4));
        serve_prefix_io(&mut wire.as_slice(), &mut out, &One, &grants, &peer);
        let f = frame::read_frame(&mut out.as_slice()).unwrap().unwrap();
        assert!(matches!(
            frame::parse_header::<PrefixReply>(&f).unwrap(),
            PrefixReply::Snapshot { .. }
        ));
        assert_eq!(f.payload.len(), 8);
    }

    #[test]
    fn a_request_frame_over_the_limit_ends_the_connection() {
        let mut wire = Vec::new();
        let big = vec![0u8; (PREFIX_REQUEST_MAX as usize) + 1];
        frame::write_frame(&mut wire, &PrefixRequest::Fetch { key: key(1) }, &big).unwrap();
        let mut out = Vec::new();
        serve_prefix_io(
            &mut wire.as_slice(),
            &mut out,
            &One,
            &PrefixGrants::default(),
            &NodeId::new("b"),
        );
        assert!(out.is_empty());
    }
}

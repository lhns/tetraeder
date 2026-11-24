//! The head's view of the cluster (ADR 0091 step 3): one control connection per node, a hello
//! and then a heartbeat every `node_down_after / 4`, each answered by the node's daemon; replies
//! feed [`Membership`]. A connection that has heard nothing for `node_down_after` is closed, so a
//! blocked call returns at once instead of waiting for a transport timeout; a node that goes
//! `Down` has every registered connection closed. A node that comes back is dialed again at the
//! next heartbeat.
//!
//! Metrics (rule 6): `tetraeder.cluster.node.up` (gauge, `node`), `tetraeder.cluster.heartbeat.rtt`
//! (histogram, `node`) and `tetraeder.cluster.node.transitions` (counter, `node`, `to`).

use crate::cluster::ClusterConfig;
use crate::membership::{Membership, NodeEvent, NodeState, Transitions};
use crate::proto::{self, NodeReply, NodeRequest};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetraeder_transport::{NodeId, Purpose, Transport};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

type Closer = Arc<dyn Fn() + Send + Sync>;

struct Metrics {
    up: Gauge<u64>,
    rtt: Histogram<f64>,
    transitions: Counter<u64>,
    baseline: Gauge<f64>,
}

impl Metrics {
    fn new() -> Self {
        let m = opentelemetry::global::meter("tetraeder");
        Self {
            up: m
                .u64_gauge("tetraeder.cluster.node.up")
                .with_description("1 while the node is up, else 0")
                .build(),
            rtt: m
                .f64_histogram("tetraeder.cluster.heartbeat.rtt")
                .with_unit("s")
                .with_description("Round trip of a heartbeat to a node's daemon")
                .build(),
            transitions: m
                .u64_counter("tetraeder.cluster.node.transitions")
                .with_description("State changes of a node, by the state entered")
                .build(),
            baseline: m
                .f64_gauge("tetraeder.cluster.heartbeat.baseline")
                .with_unit("s")
                .with_description("Probed small-frame round trip to a node, for the link alert")
                .build(),
        }
    }
}

struct Conn {
    close: Closer,
    /// Connect time, then the time of the last reply.
    since: Instant,
}

struct Shared {
    cfg: ClusterConfig,
    transport: Arc<dyn Transport>,
    origin: Instant,
    stop: AtomicBool,
    mem: Mutex<Membership>,
    conns: Mutex<HashMap<NodeId, Conn>>,
    extra: Mutex<HashMap<NodeId, Vec<Closer>>>,
    subs: Mutex<Vec<Sender<(NodeId, NodeState)>>>,
    metrics: Metrics,
}

impl Shared {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }

    fn apply(&self, f: impl FnOnce(&mut Membership, Duration) -> Transitions) {
        let out = f(&mut lock(&self.mem), self.now());
        for (id, st) in out {
            tracing::info!(node = %id, to = st.label(), "node state");
            let node = KeyValue::new("node", id.0.clone());
            self.metrics
                .transitions
                .add(1, &[node.clone(), KeyValue::new("to", st.label())]);
            self.metrics
                .up
                .record(u64::from(st == NodeState::Up), &[node]);
            lock(&self.subs).retain(|s| s.send((id.clone(), st)).is_ok());
            if st == NodeState::Down {
                self.close_all(&id);
            }
        }
    }

    fn event(&self, e: NodeEvent) {
        self.apply(|m, now| m.on_event(now, e));
    }

    /// Closes every connection to `id`: a call blocked on it returns at once.
    fn close_all(&self, id: &NodeId) {
        let mut closers: Vec<Closer> = lock(&self.extra).remove(id).unwrap_or_default();
        if let Some(c) = lock(&self.conns).get(id) {
            closers.push(Arc::clone(&c.close));
        }
        for c in closers {
            c();
        }
    }
}

pub struct Monitor {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Monitor {
    /// Dials every configured node (the head's own daemon included).
    pub fn start(transport: Arc<dyn Transport>, cfg: ClusterConfig) -> Self {
        let shared = Arc::new(Shared {
            mem: Mutex::new(Membership::new(&cfg)),
            cfg,
            transport,
            origin: Instant::now(),
            stop: AtomicBool::new(false),
            conns: Mutex::new(HashMap::new()),
            extra: Mutex::new(HashMap::new()),
            subs: Mutex::new(Vec::new()),
            metrics: Metrics::new(),
        });
        for n in &shared.cfg.nodes {
            shared
                .metrics
                .up
                .record(0, &[KeyValue::new("node", n.id.0.clone())]);
        }
        let mut threads = Vec::new();
        for id in shared.cfg.ids() {
            let s = Arc::clone(&shared);
            if let Ok(h) = std::thread::Builder::new()
                .name(format!("monitor-{id}"))
                .spawn(move || node_loop(&s, &id))
            {
                threads.push(h);
            }
        }
        let s = Arc::clone(&shared);
        if let Ok(h) = std::thread::Builder::new()
            .name("monitor-tick".into())
            .spawn(move || ticker(&s))
        {
            threads.push(h);
        }
        Self { shared, threads }
    }

    pub fn state(&self, id: &NodeId) -> Option<NodeState> {
        lock(&self.shared.mem).state(id)
    }

    pub fn states(&self) -> Vec<(NodeId, NodeState)> {
        lock(&self.shared.mem).states()
    }

    /// Every transition from now on.
    pub fn subscribe(&self) -> Receiver<(NodeId, NodeState)> {
        let (tx, rx) = mpsc::channel();
        lock(&self.shared.subs).push(tx);
        rx
    }

    /// `close` is called once when `node` next goes `Down` (the `Conn::close` of a replica or
    /// segment connection to it), so a call blocked on it fails at once.
    pub fn watch_conn(&self, node: &NodeId, close: Closer) {
        lock(&self.shared.extra)
            .entry(node.clone())
            .or_default()
            .push(close);
    }

    /// The probed small-frame round trip to `node` in seconds: the baseline of the link alert.
    pub fn set_baseline(&self, node: &NodeId, secs: f64) {
        self.shared
            .metrics
            .baseline
            .record(secs, &[KeyValue::new("node", node.0.clone())]);
    }

    /// The heartbeat interval in use.
    pub fn heartbeat_every(&self) -> Duration {
        self.shared.cfg.heartbeat_every()
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        let ids: Vec<NodeId> = lock(&self.shared.conns).keys().cloned().collect();
        for id in ids {
            self.shared.close_all(&id);
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Sleeps `d` in short steps; false when the monitor stopped meanwhile.
fn pause(s: &Shared, d: Duration) -> bool {
    let end = Instant::now() + d;
    while Instant::now() < end {
        if s.stop.load(Ordering::SeqCst) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5).min(d));
    }
    !s.stop.load(Ordering::SeqCst)
}

fn ticker(s: &Arc<Shared>) {
    let every = (s.cfg.heartbeat_every() / 4).max(Duration::from_millis(5));
    let after = s.cfg.node_down_after();
    while pause(s, every) {
        s.apply(|m, now| m.tick(now));
        // A connection that heard nothing for `node_down_after` is cut, also before its hello
        // came back (a blackholed link): its blocked call then fails.
        let stale: Vec<Closer> = lock(&s.conns)
            .values()
            .filter(|c| c.since.elapsed() >= after)
            .map(|c| Arc::clone(&c.close))
            .collect();
        for c in stale {
            c();
        }
        for (id, st) in lock(&s.mem).states() {
            s.metrics.up.record(
                u64::from(st == NodeState::Up),
                &[KeyValue::new("node", id.0)],
            );
        }
    }
}

fn node_loop(s: &Arc<Shared>, id: &NodeId) {
    let hb = s.cfg.heartbeat_every();
    let Some(peer) = s.cfg.peer(id) else { return };
    let attrs = [KeyValue::new("node", id.0.clone())];
    let mut n = 0u64;
    while !s.stop.load(Ordering::SeqCst) {
        match s.transport.connect(&peer, Purpose::Control) {
            Err(e) => {
                tracing::debug!(node = %id, "cannot connect: {e}");
                s.event(NodeEvent::Failed(id.clone()));
            }
            Ok(mut conn) => {
                lock(&s.conns).insert(
                    id.clone(),
                    Conn {
                        close: Arc::clone(&conn.close),
                        since: Instant::now(),
                    },
                );
                let hello = NodeRequest::Hello {
                    cluster: s.cfg.cluster.clone(),
                };
                let ok = match proto::call(&mut conn, &hello, &[]) {
                    Ok((NodeReply::Hello { node, .. }, _)) if node == *id => true,
                    Ok((other, _)) => {
                        tracing::warn!(node = %id, "unexpected hello reply {other:?}");
                        false
                    }
                    Err(e) => {
                        tracing::debug!(node = %id, "hello failed: {e}");
                        false
                    }
                };
                if ok {
                    touch(s, id);
                    s.event(NodeEvent::Hello(id.clone()));
                    heartbeats(s, id, &mut conn, &mut n, hb, &attrs);
                }
                s.event(NodeEvent::Failed(id.clone()));
                (conn.close)();
                lock(&s.conns).remove(id);
            }
        }
        if !pause(s, hb) {
            break;
        }
    }
}

fn touch(s: &Shared, id: &NodeId) {
    if let Some(c) = lock(&s.conns).get_mut(id) {
        c.since = Instant::now();
    }
}

fn heartbeats(
    s: &Arc<Shared>,
    id: &NodeId,
    conn: &mut tetraeder_transport::Conn,
    n: &mut u64,
    hb: Duration,
    attrs: &[KeyValue],
) {
    while !s.stop.load(Ordering::SeqCst) {
        let t0 = Instant::now();
        match proto::call(conn, &NodeRequest::Heartbeat { n: *n }, &[]) {
            Ok((NodeReply::Alive { draining, .. }, _)) => {
                touch(s, id);
                s.metrics.rtt.record(t0.elapsed().as_secs_f64(), attrs);
                s.event(NodeEvent::Heard(id.clone()));
                if draining {
                    s.event(NodeEvent::Drain(id.clone()));
                }
                // The node was declared down meanwhile (a late reply): start over.
                if lock(&s.mem).state(id) == Some(NodeState::Down) {
                    return;
                }
            }
            Ok((other, _)) => {
                tracing::warn!(node = %id, "unexpected heartbeat reply {other:?}");
                return;
            }
            Err(e) => {
                tracing::debug!(node = %id, "heartbeat failed: {e}");
                return;
            }
        }
        *n += 1;
        let wait = hb.saturating_sub(t0.elapsed());
        if !pause(s, wait) {
            return;
        }
    }
}

//! The node daemon (ADR 0091 step 3): `tetraeder node`.
//!
//! It listens on one transport, answers `Hello` and `Heartbeat` on the control connection from
//! its own thread (never from a worker, so a long step never looks like a dead node), reports
//! the node's hardware graph, times links to other nodes on request, and starts and supervises
//! the workers of the segments assigned to it. A worker speaks the frame protocol of
//! `RemoteBackend` over its stdio; the daemon relays frames between it and the segment
//! connection of the leader, restarts a dead worker with the backoff and cap of `RemoteConfig`
//! (ADR 0069) and, at the cap, gives the assignment up without ending the daemon.
//!
//! Who may open what: `Control` only from the head; a `Probe` connection is privileged (profile,
//! assign, link probes) only from the head with session 0, other nodes may only send probe
//! frames; `Segment` only from a configured node and only for an assigned segment; the replica
//! purposes belong to step 5 and are closed.

use crate::cluster::ClusterConfig;
use crate::proto::{self, NodeReply, NodeRequest};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetraeder_transport::frame::{self, Frame};
use tetraeder_transport::{Conn, NodeId, Purpose, Transport};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What a segment's worker needs to start.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub replica: u32,
    pub segment: u32,
    pub plan: String,
    pub model_sha256: String,
}

/// A running worker: frames in on `stdin`, out on `stdout`; `kill` ends it and must be safe to
/// call twice. The worker is dead when `stdout` ends.
pub struct WorkerIo {
    pub stdin: Box<dyn Write + Send>,
    pub stdout: Box<dyn Read + Send>,
    pub kill: Arc<dyn Fn() + Send + Sync>,
}

pub trait Launcher: Send + Sync {
    fn launch(&self, a: &Assignment) -> io::Result<WorkerIo>;
}

/// Restart delays are `backoff_base` times 1, 2, 4, 4, ...; `max_restarts` within `window` are
/// allowed, one more gives up. The defaults are `RemoteConfig`'s (1 s, 5 in 10 minutes).
#[derive(Debug, Clone, Copy)]
pub struct RestartPolicy {
    pub backoff_base: Duration,
    pub max_restarts: usize,
    pub window: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            backoff_base: Duration::from_secs(1),
            max_restarts: 5,
            window: Duration::from_secs(600),
        }
    }
}

/// The node's hardware graph as JSON.
pub type ProfileFn = Arc<dyn Fn() -> Result<String, String> + Send + Sync>;
/// A transport to dial from, with explicit socket buffers (send, receive) when given.
pub type DialFn =
    Arc<dyn Fn(Option<(usize, usize)>) -> io::Result<Arc<dyn Transport>> + Send + Sync>;

pub struct NodeOptions {
    pub cfg: ClusterConfig,
    pub launcher: Arc<dyn Launcher>,
    pub profile: ProfileFn,
    pub dial: DialFn,
    pub restart: RestartPolicy,
}

struct Sink {
    generation: u64,
    tx: Box<dyn Write + Send>,
    close: Arc<dyn Fn() + Send + Sync>,
}

struct Worker {
    assign: Assignment,
    stop: AtomicBool,
    running: AtomicBool,
    gave_up: AtomicBool,
    restarts: AtomicU64,
    stdin: Mutex<Option<Box<dyn Write + Send>>>,
    kill: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    sink: Mutex<Option<Sink>>,
}

impl Worker {
    fn end(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(k) = lock(&self.kill).clone() {
            k();
        }
        if let Some(s) = lock(&self.sink).take() {
            (s.close)();
        }
    }
}

struct Shared {
    transport: Arc<dyn Transport>,
    opts: NodeOptions,
    me: NodeId,
    stop: AtomicBool,
    draining: AtomicBool,
    started: Instant,
    next_id: AtomicU64,
    conns: Mutex<HashMap<u64, Arc<dyn Fn() + Send + Sync>>>,
    workers: Mutex<HashMap<(u32, u32), Arc<Worker>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

pub struct Daemon {
    shared: Arc<Shared>,
    accept: Option<JoinHandle<()>>,
}

impl Daemon {
    /// Starts accepting on `transport`.
    pub fn start(transport: Arc<dyn Transport>, opts: NodeOptions) -> Self {
        let shared = Arc::new(Shared {
            me: transport.local().clone(),
            transport,
            opts,
            stop: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            started: Instant::now(),
            next_id: AtomicU64::new(0),
            conns: Mutex::new(HashMap::new()),
            workers: Mutex::new(HashMap::new()),
            threads: Mutex::new(Vec::new()),
        });
        let s = Arc::clone(&shared);
        let accept = std::thread::Builder::new()
            .name("node-accept".into())
            .spawn(move || accept_loop(&s))
            .ok();
        Self { shared, accept }
    }

    /// Workers that are assigned and not given up.
    pub fn workers(&self) -> usize {
        lock(&self.shared.workers)
            .values()
            .filter(|w| !w.gave_up.load(Ordering::SeqCst))
            .count()
    }

    /// Times a worker was restarted since its assignment.
    pub fn worker_restarts(&self, replica: u32, segment: u32) -> Option<u64> {
        lock(&self.shared.workers)
            .get(&(replica, segment))
            .map(|w| w.restarts.load(Ordering::SeqCst))
    }

    pub fn is_draining(&self) -> bool {
        self.shared.draining.load(Ordering::SeqCst)
    }

    /// A killed process: stops accepting, resets every connection and kills the workers.
    pub fn stop(&mut self) {
        let s = &self.shared;
        s.stop.store(true, Ordering::SeqCst);
        s.transport.shutdown();
        let closers: Vec<_> = lock(&s.conns).values().cloned().collect();
        for c in closers {
            c();
        }
        let workers: Vec<_> = lock(&s.workers).drain().map(|(_, w)| w).collect();
        for w in workers {
            w.end();
        }
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
        let threads: Vec<_> = lock(&s.threads).drain(..).collect();
        for t in threads {
            let _ = t.join();
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.stop();
    }
}

fn spawn(s: &Arc<Shared>, name: &str, f: impl FnOnce() + Send + 'static) {
    if let Ok(h) = std::thread::Builder::new().name(name.into()).spawn(f) {
        let mut t = lock(&s.threads);
        t.retain(|h| !h.is_finished());
        t.push(h);
    }
}

fn accept_loop(s: &Arc<Shared>) {
    while !s.stop.load(Ordering::SeqCst) {
        match s.transport.accept() {
            Ok(conn) => {
                let h = Arc::clone(s);
                spawn(s, "node-conn", move || handle_conn(&h, conn));
            }
            Err(e) if e.kind() == io::ErrorKind::NotConnected => break,
            Err(e) => {
                tracing::warn!("accept failed: {e}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn handle_conn(s: &Arc<Shared>, conn: Conn) {
    let id = s.next_id.fetch_add(1, Ordering::SeqCst);
    lock(&s.conns).insert(id, Arc::clone(&conn.close));
    // A shutdown that raced with this connection must still reset it.
    if s.stop.load(Ordering::SeqCst) {
        (conn.close)();
    }
    let head = conn.peer == s.opts.cfg.head;
    let close = Arc::clone(&conn.close);
    match conn.purpose.clone() {
        Purpose::Control if head => serve_requests(s, conn, true, true),
        Purpose::Probe { session } => serve_requests(s, conn, head && session == 0, false),
        Purpose::Segment { replica, segment } => relay_segment(s, conn, (replica, segment)),
        other => {
            tracing::warn!(peer = %conn.peer, purpose = other.label(), "connection not allowed; closed");
            close();
        }
    }
    lock(&s.conns).remove(&id);
}

/// Answers requests until the connection ends. `watched`: a connection that stays silent for
/// twice `node_down_after` is closed, so a head that is gone does not hold the purpose.
fn serve_requests(s: &Arc<Shared>, mut conn: Conn, privileged: bool, watched: bool) {
    let last = Arc::new(AtomicU64::new(s.started.elapsed().as_millis() as u64));
    let done = Arc::new(AtomicBool::new(false));
    if watched {
        let limit = s.opts.cfg.node_down_after() * 2;
        let (last, done, close, s2) = (
            Arc::clone(&last),
            Arc::clone(&done),
            Arc::clone(&conn.close),
            Arc::clone(s),
        );
        spawn(s, "node-idle", move || {
            while !done.load(Ordering::SeqCst) && !s2.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
                let idle = s2
                    .started
                    .elapsed()
                    .saturating_sub(Duration::from_millis(last.load(Ordering::SeqCst)));
                if idle > limit {
                    tracing::warn!("control connection idle for {idle:?}; closed");
                    close();
                    break;
                }
            }
        });
    }
    while let Ok(Some(f)) = proto::recv(&mut conn.rx, &conn.purpose) {
        last.store(s.started.elapsed().as_millis() as u64, Ordering::SeqCst);
        let (reply, payload) = match frame::parse_header::<NodeRequest>(&f) {
            Ok(req) => answer(s, &req, &f, privileged),
            Err(e) => (NodeReply::Error { msg: e }, Vec::new()),
        };
        if proto::send(&mut conn.tx, &reply, &payload).is_err() {
            break;
        }
    }
    done.store(true, Ordering::SeqCst);
    (conn.close)();
}

fn error(msg: impl Into<String>) -> (NodeReply, Vec<u8>) {
    (NodeReply::Error { msg: msg.into() }, Vec::new())
}

fn answer(s: &Arc<Shared>, req: &NodeRequest, f: &Frame, privileged: bool) -> (NodeReply, Vec<u8>) {
    let reply = |r: NodeReply| (r, Vec::new());
    match req {
        NodeRequest::Probe { bytes } => {
            if f.payload.len() as u64 == *bytes {
                reply(NodeReply::Probe)
            } else {
                error("probe payload has the wrong length")
            }
        }
        _ if !privileged => error("not allowed on this connection"),
        NodeRequest::Hello { cluster } => {
            if *cluster == s.opts.cfg.cluster {
                reply(NodeReply::Hello {
                    node: s.me.clone(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                })
            } else {
                error(format!(
                    "this node belongs to cluster `{}`, not `{cluster}`",
                    s.opts.cfg.cluster
                ))
            }
        }
        NodeRequest::Heartbeat { n } => reply(NodeReply::Alive {
            n: *n,
            workers: lock(&s.workers)
                .values()
                .filter(|w| w.running.load(Ordering::SeqCst))
                .count() as u32,
            draining: s.draining.load(Ordering::SeqCst),
        }),
        NodeRequest::Profile => match (s.opts.profile)() {
            Ok(json) => (NodeReply::Profile, json.into_bytes()),
            Err(e) => error(format!("cannot profile: {e}")),
        },
        NodeRequest::Assign {
            replica,
            segment,
            plan,
            model_sha256,
        } => {
            if s.draining.load(Ordering::SeqCst) {
                return error("the node is draining");
            }
            assign(
                s,
                Assignment {
                    replica: *replica,
                    segment: *segment,
                    plan: plan.clone(),
                    model_sha256: model_sha256.clone(),
                },
            );
            reply(NodeReply::Assigned)
        }
        NodeRequest::Release { replica, segment } => {
            if let Some(w) = lock(&s.workers).remove(&(*replica, *segment)) {
                w.end();
            }
            reply(NodeReply::Released)
        }
        NodeRequest::Drain => {
            s.draining.store(true, Ordering::SeqCst);
            reply(NodeReply::Alive {
                n: 0,
                workers: 0,
                draining: true,
            })
        }
        NodeRequest::ProbePeer {
            peer,
            sizes,
            repeats,
            buffers,
        } => match probe_peer(s, peer, sizes, *repeats, *buffers) {
            Ok(r) => reply(r),
            Err(e) => error(format!("probe of {peer} failed: {e}")),
        },
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn probe_peer(
    s: &Arc<Shared>,
    peer: &NodeId,
    sizes: &[u64],
    repeats: u32,
    buffers: Option<(usize, usize)>,
) -> io::Result<NodeReply> {
    let to = s
        .opts
        .cfg
        .peer(peer)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "not a configured node"))?;
    if repeats == 0 || repeats > 1000 || sizes.len() > 32 || sizes.iter().any(|b| *b > 64 << 20) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "repeats 1 to 1000, at most 32 sizes of at most 64 MiB",
        ));
    }
    let t = (s.opts.dial)(buffers)?;
    let mut c = t.connect(&to, Purpose::Probe { session: 1 })?;
    let mut rtt = |bytes: u64, payload: &[u8]| -> io::Result<f64> {
        let mut v = Vec::new();
        for _ in 0..repeats {
            let t0 = Instant::now();
            let (r, _) = proto::call(&mut c, &NodeRequest::Probe { bytes }, payload)?;
            if !matches!(r, NodeReply::Probe) {
                return Err(io::Error::other(format!("unexpected reply {r:?}")));
            }
            v.push(t0.elapsed().as_secs_f64());
        }
        Ok(median(&mut v))
    };
    let empty = rtt(0, &[])?;
    let mut points = Vec::new();
    for &b in sizes {
        let payload = vec![0u8; b as usize];
        let full = rtt(b, &payload)?;
        // One way: the frame's time less half the empty round trip.
        points.push((b as f64, (full - empty / 2.0).max(1e-9)));
    }
    (c.close)();
    Ok(NodeReply::Curve {
        points,
        empty_rtt_s: empty,
    })
}

fn assign(s: &Arc<Shared>, a: Assignment) {
    let key = (a.replica, a.segment);
    let w = Arc::new(Worker {
        assign: a,
        stop: AtomicBool::new(false),
        running: AtomicBool::new(false),
        gave_up: AtomicBool::new(false),
        restarts: AtomicU64::new(0),
        stdin: Mutex::new(None),
        kill: Mutex::new(None),
        sink: Mutex::new(None),
    });
    if let Some(old) = lock(&s.workers).insert(key, Arc::clone(&w)) {
        old.end();
    }
    let s2 = Arc::clone(s);
    spawn(s, "node-worker", move || supervise(&s2, &w));
}

/// Starts the worker, relays its frames to the attached connection, restarts it when it dies.
fn supervise(s: &Arc<Shared>, w: &Arc<Worker>) {
    let pol = s.opts.restart;
    let mut restarts: VecDeque<Instant> = VecDeque::new();
    let mut first = true;
    while !w.stop.load(Ordering::SeqCst) && !s.stop.load(Ordering::SeqCst) {
        if !first {
            let now = Instant::now();
            while restarts
                .front()
                .is_some_and(|t| now.duration_since(*t) > pol.window)
            {
                restarts.pop_front();
            }
            if restarts.len() >= pol.max_restarts {
                tracing::error!(
                    replica = w.assign.replica,
                    segment = w.assign.segment,
                    "the worker died {} times within {:?}; giving up",
                    pol.max_restarts + 1,
                    pol.window
                );
                w.gave_up.store(true, Ordering::SeqCst);
                return;
            }
            let delay = pol.backoff_base * (1u32 << restarts.len().min(2));
            let until = Instant::now() + delay;
            while Instant::now() < until && !w.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(10));
            }
            if w.stop.load(Ordering::SeqCst) {
                return;
            }
            restarts.push_back(Instant::now());
            w.restarts.fetch_add(1, Ordering::SeqCst);
        }
        first = false;
        let io = match s.opts.launcher.launch(&w.assign) {
            Ok(io) => io,
            Err(e) => {
                tracing::warn!("cannot start the worker: {e}");
                continue;
            }
        };
        let WorkerIo {
            stdin,
            mut stdout,
            kill,
        } = io;
        *lock(&w.stdin) = Some(stdin);
        *lock(&w.kill) = Some(Arc::clone(&kill));
        w.running.store(true, Ordering::SeqCst);
        if w.stop.load(Ordering::SeqCst) {
            kill();
        }
        // The worker's frames go to whoever is attached; with nobody, they are dropped.
        while let Ok(Some(f)) = frame::read_frame(&mut stdout) {
            let mut sink = lock(&w.sink);
            if let Some(si) = sink.as_mut() {
                let ok = frame::encode_raw(&f.header, &f.payload)
                    .map_err(io::Error::other)
                    .and_then(|b| si.tx.write_all(&b).and_then(|()| si.tx.flush()))
                    .is_ok();
                if !ok && let Some(si) = sink.take() {
                    (si.close)();
                }
            }
        }
        w.running.store(false, Ordering::SeqCst);
        kill();
        *lock(&w.stdin) = None;
        // The leader sees the failure at once and reconnects after the restart.
        if let Some(si) = lock(&w.sink).take() {
            (si.close)();
        }
    }
}

fn relay_segment(s: &Arc<Shared>, mut conn: Conn, key: (u32, u32)) {
    let Some(w) = lock(&s.workers).get(&key).cloned() else {
        tracing::warn!(peer = %conn.peer, "segment connection without an assignment; closed");
        (conn.close)();
        return;
    };
    let generation = s.next_id.fetch_add(1, Ordering::SeqCst);
    let rx = &mut conn.rx;
    {
        let mut sink = lock(&w.sink);
        if let Some(old) = sink.take() {
            (old.close)();
        }
        *sink = Some(Sink {
            generation,
            tx: std::mem::replace(&mut conn.tx, Box::new(io::sink())),
            close: Arc::clone(&conn.close),
        });
    }
    while let Ok(Some(f)) = proto::recv(rx, &conn.purpose) {
        let Ok(bytes) = frame::encode_raw(&f.header, &f.payload) else {
            break;
        };
        let mut stdin = lock(&w.stdin);
        let Some(si) = stdin.as_mut() else { break };
        if si.write_all(&bytes).and_then(|()| si.flush()).is_err() {
            break;
        }
    }
    let mut sink = lock(&w.sink);
    if sink.as_ref().is_some_and(|x| x.generation == generation) {
        *sink = None;
    }
    (conn.close)();
}

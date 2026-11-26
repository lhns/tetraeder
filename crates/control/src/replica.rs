//! The replica protocol (ADR 0091, step 5): a replica is today's engine on its leader node,
//! reached by the head through [`RemoteEngine`], an `Engine` of the gateway over a
//! `Purpose::Replica` connection.
//!
//! One frame per message, a JSON header and no payload. Per generated token a replica sends one
//! `Event` frame of a few dozen bytes (a token id, its probabilities when asked), never logits.
//! The head sends the prompt tokens, the sampling parameters and the constraint as its Lark
//! source; the leader rebuilds the constraint from the vocabulary it holds.
//!
//! A request id is unique across reconnects. The daemon puts `Attach` and `Detach` frames
//! around each head connection into the replica's stdin, so a replica that outlives one head
//! connection cancels what that connection started (a zombie request after a partition heals
//! would otherwise run to its end for nobody).

use crate::kv::{PrefixGrants, PrefixKey, PrefixStore, fetch_prefix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tetraeder_gateway::constraint::ConstraintFactory;
use tetraeder_gateway::engine::{
    Engine, EngineError, GenEvent, GenerateRequest, ModelInfo, SamplingParams,
};
use tetraeder_transport::frame;
use tetraeder_transport::{Conn, NodeId, Peer, Purpose, Transport};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Frames a replica connection accepts (`Purpose::Replica`).
const MAX_FRAME: u64 = 16 << 20;
/// Events a request's channel holds before the sender waits for the consumer.
const EVENT_BUFFER: usize = 1024;
/// How long a probe step may take before the replica counts as not joined.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the replica's first answer (`Info`) may take.
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests one connection may have running at once, as a multiple of the engine's `cap` (the
/// router keeps to `cap`; some queue inside the engine) and at least this many.
const RUNNING_PER_CAP: usize = 4;
const RUNNING_MIN: usize = 64;

/// The constraint of a request as its source (compiled on the leader).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintSpec {
    /// A complete Lark grammar, as `TokenConstraint::source` returns it.
    Lark(String),
}

/// `GenerateRequest` on the wire. Sampling parameters carry their grammar (GBNF) or JSON schema
/// as text already.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireGenerate {
    pub model: String,
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub constraint: Option<ConstraintSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<String>,
}

/// A prefix to fetch from another leader before generating (the head priced it against
/// recomputing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FetchFrom {
    pub node: NodeId,
    pub key: PrefixKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum ReplicaRequest {
    /// Written by the daemon when a head connection starts and ends.
    Attach {
        session: u64,
    },
    Detach {
        session: u64,
    },
    /// Answered with `Info`.
    Hello,
    Generate {
        id: u64,
        req: Box<WireGenerate>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fetch: Option<FetchFrom>,
    },
    Cancel {
        id: u64,
    },
    /// The head announces that `peer` may fetch `key` from this replica once. Nothing else is
    /// served on the `Prefix` purpose.
    Offer {
        peer: NodeId,
        key: PrefixKey,
    },
    /// One step of the engine; answered with `Probed`. A replica joins the router after it.
    Probe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum ReplicaEvent {
    Info {
        models: Vec<ModelInfo>,
        /// The KV format of the replica's snapshots (empty without a prefix store).
        kv_format: String,
        /// Requests the replica takes at once (`n_seq_max`).
        cap: u32,
        /// Bytes of one cached token in its snapshots (0: no prefix store).
        #[serde(default)]
        kv_row_bytes: u64,
    },
    Event {
        id: u64,
        event: GenEvent,
    },
    /// The outcome of a requested prefix fetch, before the request's first event.
    Fetched {
        id: u64,
        fetched: bool,
        /// `fetched`, or why it was recomputed (`metrics::FETCH_REASONS`); empty from an older
        /// replica.
        #[serde(default)]
        reason: String,
        /// Snapshot bytes installed.
        #[serde(default)]
        bytes: u64,
    },
    Probed {
        ok: bool,
        msg: String,
    },
}

fn send<W: Write, T: Serialize>(w: &Mutex<W>, msg: &T) -> bool {
    frame::write_frame(&mut *lock(w), msg, &[]).is_ok()
}

fn is_terminal(e: &GenEvent) -> bool {
    matches!(
        e,
        GenEvent::Finished { .. } | GenEvent::Error(_) | GenEvent::ContextExceeded(_)
    )
}

/// The address of a node by id (`cluster.toml`).
pub type PeerLookup = Arc<dyn Fn(&NodeId) -> Option<Peer> + Send + Sync>;

/// Where the replica fetches prefixes from: its own prefix store and a way to dial a peer's
/// leader.
pub struct KvHooks {
    pub store: Arc<dyn PrefixStore>,
    pub transport: Arc<dyn Transport>,
    pub peers: PeerLookup,
    /// What other nodes may fetch from `store`: filled by the head's `Offer`s, read by the
    /// prefix server (`kv::serve_prefix_io`).
    pub grants: Arc<PrefixGrants>,
}

/// The replica side: serves one engine over frames.
pub struct ReplicaServer {
    engine: Arc<dyn Engine>,
    cap: u32,
    kv: Option<KvHooks>,
    factories: Mutex<HashMap<String, Arc<ConstraintFactory>>>,
    /// One fetch at a time: a second `Prefix` connection to the same node replaces the first
    /// (the transport keeps one per peer and purpose), which would break the one in progress.
    fetching: Mutex<()>,
}

impl ReplicaServer {
    pub fn new(engine: Arc<dyn Engine>, cap: u32, kv: Option<KvHooks>) -> Arc<Self> {
        Arc::new(Self {
            engine,
            cap,
            kv,
            factories: Mutex::new(HashMap::new()),
            fetching: Mutex::new(()),
        })
    }

    fn constraint(
        &self,
        model: &str,
        spec: &ConstraintSpec,
    ) -> Result<Box<dyn tetraeder_gateway::engine::TokenConstraint>, String> {
        let ConstraintSpec::Lark(lark) = spec;
        let f = {
            let mut m = lock(&self.factories);
            if let Some(f) = m.get(model) {
                Arc::clone(f)
            } else {
                let v = self.engine.vocab(model).map_err(|e| e.to_string())?;
                let f = Arc::new(ConstraintFactory::new(&v)?);
                m.insert(model.to_owned(), Arc::clone(&f));
                f
            }
        };
        f.from_lark(lark)
    }

    /// Serves frames from `rx` and answers on `tx` until `rx` ends; every request still running
    /// then is cancelled.
    pub fn serve<R: Read, W: Write + Send + 'static>(self: &Arc<Self>, mut rx: R, tx: W) {
        let tx = Arc::new(Mutex::new(tx));
        let running: Arc<Mutex<HashMap<u64, CancellationToken>>> = Arc::default();
        let mut current = 0u64;
        let probing = Arc::new(AtomicBool::new(false));
        let max_running = (self.cap as usize * RUNNING_PER_CAP).max(RUNNING_MIN);
        let cancel_all = |running: &Mutex<HashMap<u64, CancellationToken>>| {
            for (_, t) in lock(running).drain() {
                t.cancel();
            }
        };
        while let Ok(Some(f)) = frame::read_frame_max(&mut rx, MAX_FRAME) {
            let Ok(req) = frame::parse_header::<ReplicaRequest>(&f) else {
                break;
            };
            match req {
                ReplicaRequest::Attach { session } => {
                    cancel_all(&running);
                    current = session;
                }
                ReplicaRequest::Detach { session } => {
                    if session == current {
                        cancel_all(&running);
                    }
                }
                ReplicaRequest::Hello => {
                    let info = ReplicaEvent::Info {
                        models: self.engine.models(),
                        kv_format: self
                            .kv
                            .as_ref()
                            .map(|k| k.store.kv_format())
                            .unwrap_or_default(),
                        cap: self.cap,
                        kv_row_bytes: self.kv.as_ref().map_or(0, |k| k.store.row_bytes()),
                    };
                    if !send(&tx, &info) {
                        break;
                    }
                }
                ReplicaRequest::Cancel { id } => {
                    if let Some(t) = lock(&running).get(&id) {
                        t.cancel();
                    }
                }
                ReplicaRequest::Offer { peer, key } => {
                    if let Some(kv) = &self.kv {
                        kv.grants.offer(peer, key);
                    }
                }
                ReplicaRequest::Probe => {
                    // One probe step at a time: a peer that repeats the call does not get a
                    // thread each.
                    if probing.swap(true, Ordering::SeqCst) {
                        let busy = ReplicaEvent::Probed {
                            ok: false,
                            msg: "a probe step is running".into(),
                        };
                        if !send(&tx, &busy) {
                            break;
                        }
                        continue;
                    }
                    let (s, tx, probing) =
                        (Arc::clone(self), Arc::clone(&tx), Arc::clone(&probing));
                    std::thread::spawn(move || {
                        let r = s.probe_step();
                        probing.store(false, Ordering::SeqCst);
                        let _ = send(
                            &tx,
                            &ReplicaEvent::Probed {
                                ok: r.is_ok(),
                                msg: r.err().unwrap_or_default(),
                            },
                        );
                    });
                }
                ReplicaRequest::Generate { id, req, fetch } => {
                    if lock(&running).len() >= max_running {
                        let busy = ReplicaEvent::Event {
                            id,
                            event: GenEvent::Error(
                                "the replica has too many requests running".into(),
                            ),
                        };
                        if !send(&tx, &busy) {
                            break;
                        }
                        continue;
                    }
                    let cancel = CancellationToken::new();
                    lock(&running).insert(id, cancel.clone());
                    let (s, tx, running) =
                        (Arc::clone(self), Arc::clone(&tx), Arc::clone(&running));
                    std::thread::spawn(move || {
                        s.run(id, *req, fetch, cancel, &tx);
                        lock(&running).remove(&id);
                    });
                }
            }
        }
        cancel_all(&running);
    }

    /// One generated token on the first model: the engine loaded, answering, and sampling.
    fn probe_step(&self) -> Result<(), String> {
        let model = self
            .engine
            .models()
            .first()
            .map(|m| m.id.clone())
            .ok_or("the engine serves no model")?;
        let vocab = self.engine.vocab(&model).map_err(|e| e.to_string())?;
        let prompt = vec![vocab.eog.first().copied().unwrap_or(0)];
        let mut rx = self
            .engine
            .generate(GenerateRequest {
                model,
                prompt,
                params: SamplingParams {
                    max_tokens: 1,
                    temperature: Some(0.0),
                    ..SamplingParams::default()
                },
                constraint: None,
                cancel: CancellationToken::new(),
            })
            .map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        loop {
            match rx.blocking_recv() {
                Some(GenEvent::Finished { .. }) => return Ok(()),
                Some(GenEvent::Error(e) | GenEvent::ContextExceeded(e)) => return Err(e),
                Some(_) => {}
                None => return Err("the engine ended the probe without a result".into()),
            }
            if t0.elapsed() > PROBE_TIMEOUT {
                return Err("the probe step timed out".into());
            }
        }
    }

    fn run<W: Write>(
        &self,
        id: u64,
        wire: WireGenerate,
        fetch: Option<FetchFrom>,
        cancel: CancellationToken,
        tx: &Mutex<W>,
    ) {
        let event = |e: GenEvent| send(tx, &ReplicaEvent::Event { id, event: e });
        // A replica holds one model. Its id is the file's name on this node (a sha256 stem) or
        // `--alias`, and the head's is its own: the request means the model the replica serves.
        let Some(model) = self.engine.models().first().map(|m| m.id.clone()) else {
            let _ = event(GenEvent::Error("the engine serves no model".into()));
            return;
        };
        if let Some(f) = &fetch {
            let t0 = Instant::now();
            let (reason, bytes) = self.fetch(f);
            tracing::info!(
                request = id,
                node = %f.node,
                tokens = f.key.tokens.len(),
                bytes,
                took_ms = format_args!("{:.1}", t0.elapsed().as_secs_f64() * 1e3),
                reason,
                "prefix fetch"
            );
            let ev = ReplicaEvent::Fetched {
                id,
                fetched: reason == "fetched",
                reason: reason.into(),
                bytes,
            };
            if !send(tx, &ev) {
                return;
            }
        }
        let constraint = match wire.constraint.as_ref().map(|c| self.constraint(&model, c)) {
            Some(Err(e)) => {
                let _ = event(GenEvent::Error(format!("constraint: {e}")));
                return;
            }
            Some(Ok(c)) => Some(c),
            None => None,
        };
        let rx = self.engine.generate(GenerateRequest {
            model,
            prompt: wire.prompt,
            params: wire.params,
            constraint,
            cancel: cancel.clone(),
        });
        let mut rx = match rx {
            Ok(rx) => rx,
            Err(e) => {
                let _ = event(match e {
                    EngineError::Invalid(m) => GenEvent::ContextExceeded(m),
                    e => GenEvent::Error(e.to_string()),
                });
                return;
            }
        };
        while let Some(ev) = rx.blocking_recv() {
            let end = is_terminal(&ev);
            if !event(ev) {
                // The head is gone: stop the generation.
                cancel.cancel();
                return;
            }
            if end {
                return;
            }
        }
        let _ = event(GenEvent::Error(
            "the engine ended the request without a final event".into(),
        ));
    }

    /// Fetches the snapshot into this replica's store: `fetched` and the bytes installed, or why
    /// it is recomputed (`metrics::FETCH_REASONS`).
    fn fetch(&self, f: &FetchFrom) -> (&'static str, u64) {
        let Some(kv) = &self.kv else {
            return ("refused", 0);
        };
        if f.key.format != kv.store.kv_format() {
            return ("refused", 0);
        }
        let Some(peer) = (kv.peers)(&f.node) else {
            return ("refused", 0);
        };
        // What a snapshot of this replica looks like: a reply is checked against it before it
        // is read and before it is installed. Without a known row size nothing can be checked.
        let row = kv.store.row_bytes();
        let max_tokens = kv.store.max_tokens();
        if row == 0 || f.key.tokens.is_empty() || f.key.tokens.len() as u64 > max_tokens {
            return ("refused", 0);
        }
        let max_reply = max_tokens
            .min(u64::from(u32::MAX))
            .saturating_mul(row)
            .saturating_add(1 << 20);
        let _one = lock(&self.fetching);
        match fetch_prefix(kv.transport.as_ref(), &peer, &f.key, max_reply) {
            Ok(Some((tokens, bytes))) => {
                // The holder is another node: it must send what was asked, the size its rows
                // make, within this replica's context.
                let sized = (tokens.len() as u64).checked_mul(row) == Some(bytes.len() as u64);
                if !tokens.starts_with(&f.key.tokens) || tokens.len() as u64 > max_tokens || !sized
                {
                    tetraeder_transport::metrics::rejected("bad_snapshot");
                    tracing::warn!(node = %f.node, "prefix reply does not match its key; discarded");
                    return ("bad_snapshot", 0);
                }
                let n = bytes.len() as u64;
                // Installed under the tokens the snapshot really holds.
                let put = kv.store.put(
                    PrefixKey {
                        format: f.key.format.clone(),
                        tokens,
                    },
                    bytes,
                );
                if put {
                    ("fetched", n)
                } else {
                    ("not_installed", 0)
                }
            }
            // The holder no longer has it (evicted, or held only by a live sequence).
            Ok(None) => ("gone", 0),
            Err(e) => {
                tracing::warn!(node = %f.node, "prefix fetch failed: {e}");
                ("failed", 0)
            }
        }
    }
}

/// Serves a replica over one transport connection (an in-process replica; a replica child of a
/// daemon speaks the same frames over its stdio).
pub fn serve_conn(server: &Arc<ReplicaServer>, conn: Conn) {
    let close = Arc::clone(&conn.close);
    server.serve(conn.rx, conn.tx);
    close();
}

// ---------------------------------------------------------------------------------------------
// The head's side.
// ---------------------------------------------------------------------------------------------

struct Pend {
    tx: mpsc::Sender<GenEvent>,
    cancel: CancellationToken,
    cancel_sent: bool,
}

struct Link {
    tx: Mutex<Box<dyn Write + Send>>,
    close: Arc<dyn Fn() + Send + Sync>,
    pending: Mutex<HashMap<u64, Pend>>,
    dead: AtomicBool,
    info: Mutex<Option<mpsc::Sender<ReplicaEvent>>>,
    probe: Mutex<Option<std::sync::mpsc::Sender<Result<(), String>>>>,
}

impl Link {
    fn write(&self, msg: &ReplicaRequest) -> Result<(), String> {
        frame::write_frame(&mut *lock(&self.tx), msg, &[]).inspect_err(|_| (self.close)())
    }

    /// Ends every request still open with one error each.
    fn fail_all(&self, why: &str) {
        self.dead.store(true, Ordering::SeqCst);
        let all: Vec<Pend> = lock(&self.pending).drain().map(|(_, p)| p).collect();
        for p in all {
            let _ = deliver(&p.tx, GenEvent::Error(why.to_owned()));
        }
        if let Some(p) = lock(&self.probe).take() {
            let _ = p.send(Err(why.to_owned()));
        }
        // A `connect` still waiting for `Info` holds this link, so the sender only drops here:
        // without it the wait never ends (an engine that loads longer than the hello wait hung
        // the head's join loop for good).
        lock(&self.info).take();
    }
}

/// Sends `ev` into a request's channel, waiting while it is full; false when the receiver is
/// gone or does not drain for a minute.
fn deliver(tx: &mpsc::Sender<GenEvent>, mut ev: GenEvent) -> bool {
    let t0 = Instant::now();
    loop {
        match tx.try_send(ev) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(e)) => {
                if t0.elapsed() > Duration::from_secs(60) {
                    return false;
                }
                ev = e;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn reader(
    link: &Arc<Link>,
    mut rx: Box<dyn Read + Send>,
    replica: u32,
    lost: &LostFn,
    epoch: (u64, &AtomicU64),
) {
    while let Ok(Some(f)) = frame::read_frame_max(&mut rx, MAX_FRAME) {
        let Ok(ev) = frame::parse_header::<ReplicaEvent>(&f) else {
            break;
        };
        match ev {
            ReplicaEvent::Event { id, event } => {
                let end = is_terminal(&event);
                let tx = {
                    let mut p = lock(&link.pending);
                    if end {
                        p.remove(&id).map(|p| p.tx)
                    } else {
                        p.get(&id).map(|p| p.tx.clone())
                    }
                };
                // An id we do not know ended already (cancelled, failed): drop its events.
                if let Some(tx) = tx {
                    let _ = deliver(&tx, event);
                }
            }
            ReplicaEvent::Fetched {
                fetched,
                reason,
                bytes,
                ..
            } => {
                let reason = match (fetched, reason.as_str()) {
                    (true, _) => "fetched",
                    (false, "" | "fetched") => "unknown",
                    (false, r) => crate::metrics::fetch_reason(r),
                };
                tracing::info!(
                    replica,
                    result = if fetched { "fetched" } else { "recomputed" },
                    reason,
                    bytes,
                    "prefix fetch done"
                );
                if fetched {
                    crate::metrics::prefix_fetch("fetched", reason);
                    crate::metrics::prefix_fetch_bytes(bytes);
                } else {
                    crate::metrics::prefix_fetch("recomputed", reason);
                }
            }
            ReplicaEvent::Probed { ok, msg } => {
                if let Some(p) = lock(&link.probe).take() {
                    let _ = p.send(if ok { Ok(()) } else { Err(msg) });
                }
            }
            info @ ReplicaEvent::Info { .. } => {
                if let Some(t) = lock(&link.info).take() {
                    let _ = t.try_send(info);
                }
            }
        }
    }
    (link.close)();
    link.fail_all(&format!("the connection to replica {replica} was lost"));
    // Only the newest connection reports its loss: an older one ends late, after a new join.
    let l = lock(lost)
        .clone()
        .filter(|_| epoch.1.load(Ordering::SeqCst) == epoch.0);
    if let Some(l) = l {
        l();
    }
}

/// Sends `Cancel` for requests whose client gave up, until the link ends.
fn cancel_poller(link: &Arc<Link>) {
    while !link.dead.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(10));
        let todo: Vec<u64> = lock(&link.pending)
            .iter_mut()
            .filter_map(|(id, p)| {
                if !p.cancel_sent && (p.cancel.is_cancelled() || p.tx.is_closed()) {
                    p.cancel_sent = true;
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        for id in todo {
            if link.write(&ReplicaRequest::Cancel { id }).is_err() {
                return;
            }
        }
    }
}

type LostFn = Mutex<Option<Arc<dyn Fn() + Send + Sync>>>;

/// What a joined replica reported in `Info`.
#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    pub models: Vec<ModelInfo>,
    pub kv_format: String,
    pub cap: u32,
    pub kv_row_bytes: u64,
}

/// The gateway's `Engine` for a replica led by another node (ADR 0091). Tokenization, templates
/// and grammar checks stay on the head (`front`); `generate` sends the request to the leader and
/// streams its events. Connect with [`RemoteEngine::connect`]; without a live connection
/// `generate` fails at once, so the router can send the request elsewhere.
pub struct RemoteEngine {
    front: Arc<dyn Engine>,
    transport: Arc<dyn Transport>,
    peer: Peer,
    replica: u32,
    link: Mutex<Option<Arc<Link>>>,
    next_id: AtomicU64,
    lost: Arc<LostFn>,
    epoch: Arc<AtomicU64>,
}

impl RemoteEngine {
    pub fn new(
        front: Arc<dyn Engine>,
        transport: Arc<dyn Transport>,
        peer: Peer,
        replica: u32,
    ) -> Arc<Self> {
        // Ids are unique across reconnects and across head restarts.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
            << 8;
        Arc::new(Self {
            front,
            transport,
            peer,
            replica,
            link: Mutex::new(None),
            next_id: AtomicU64::new(seed),
            lost: Arc::new(Mutex::new(None)),
            epoch: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn peer(&self) -> &Peer {
        &self.peer
    }

    /// `f` runs once each time a connection ends (its requests have been failed by then).
    pub fn on_lost(&self, f: impl Fn() + Send + Sync + 'static) {
        *lock(&self.lost) = Some(Arc::new(f));
    }

    /// Dials the replica and waits for its `Info`. `watch` gets the connection's closer before
    /// anything is sent (the monitor closes it when the node goes `Down`).
    pub fn connect(
        &self,
        watch: &dyn Fn(Arc<dyn Fn() + Send + Sync>),
    ) -> Result<ReplicaInfo, String> {
        self.disconnect();
        let conn = self
            .transport
            .connect(
                &self.peer,
                Purpose::Replica {
                    replica: self.replica,
                },
            )
            .map_err(|e| format!("cannot connect to replica {}: {e}", self.replica))?;
        let Conn { rx, tx, close, .. } = conn;
        watch(Arc::clone(&close));
        let (info_tx, mut info_rx) = mpsc::channel(1);
        let link = Arc::new(Link {
            tx: Mutex::new(tx),
            close: Arc::clone(&close),
            pending: Mutex::new(HashMap::new()),
            dead: AtomicBool::new(false),
            info: Mutex::new(Some(info_tx)),
            probe: Mutex::new(None),
        });
        let my = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        let (l, lost, replica, epoch) = (
            Arc::clone(&link),
            Arc::clone(&self.lost),
            self.replica,
            Arc::clone(&self.epoch),
        );
        std::thread::Builder::new()
            .name(format!("replica-{replica}-rx"))
            .spawn(move || reader(&l, rx, replica, &lost, (my, &epoch)))
            .map_err(|e| e.to_string())?;
        let l = Arc::clone(&link);
        std::thread::Builder::new()
            .name(format!("replica-{replica}-cancel"))
            .spawn(move || cancel_poller(&l))
            .map_err(|e| e.to_string())?;
        // A replica that never answers must not hold the caller: cut the connection.
        let answered = Arc::new(AtomicBool::new(false));
        let (a, c) = (Arc::clone(&answered), Arc::clone(&close));
        std::thread::spawn(move || {
            let t0 = Instant::now();
            while t0.elapsed() < HELLO_TIMEOUT {
                if a.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            c();
        });
        let r = (|| {
            link.write(&ReplicaRequest::Hello)?;
            match info_rx.blocking_recv() {
                Some(ReplicaEvent::Info {
                    models,
                    kv_format,
                    cap,
                    kv_row_bytes,
                }) => Ok(ReplicaInfo {
                    models,
                    kv_format,
                    cap,
                    kv_row_bytes,
                }),
                _ => Err(format!("replica {} did not answer", self.replica)),
            }
        })();
        answered.store(true, Ordering::SeqCst);
        match r {
            Ok(info) => {
                *lock(&self.link) = Some(link);
                Ok(info)
            }
            Err(e) => {
                close();
                Err(e)
            }
        }
    }

    /// One engine step on the replica; a replica joins the router after it succeeds.
    pub fn probe(&self) -> Result<(), String> {
        let link = self.live().ok_or("the replica is not connected")?;
        let (tx, rx) = std::sync::mpsc::channel();
        *lock(&link.probe) = Some(tx);
        link.write(&ReplicaRequest::Probe)?;
        match rx.recv_timeout(PROBE_TIMEOUT) {
            Ok(r) => r,
            Err(_) => {
                (link.close)();
                Err("the probe step timed out".into())
            }
        }
    }

    pub fn disconnect(&self) {
        if let Some(l) = lock(&self.link).take() {
            (l.close)();
        }
    }

    fn live(&self) -> Option<Arc<Link>> {
        lock(&self.link)
            .as_ref()
            .filter(|l| !l.dead.load(Ordering::SeqCst))
            .cloned()
    }

    pub fn is_connected(&self) -> bool {
        self.live().is_some()
    }

    /// Tells the replica's leader that `peer` may fetch `key` from it once (`Offer`); false when
    /// the link is down.
    pub fn offer(&self, peer: NodeId, key: PrefixKey) -> bool {
        self.live()
            .is_some_and(|l| l.write(&ReplicaRequest::Offer { peer, key }).is_ok())
    }

    /// [`Engine::generate`] with a prefix to fetch first. The second value is the request's id
    /// on the wire.
    pub fn generate_with(
        &self,
        req: GenerateRequest,
        fetch: Option<FetchFrom>,
    ) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        let link = self.live().ok_or_else(|| {
            EngineError::Other(format!("replica {} is not connected", self.replica))
        })?;
        let constraint = match &req.constraint {
            None => None,
            Some(c) => Some(ConstraintSpec::Lark(c.source().ok_or_else(|| {
                EngineError::Invalid("this constraint cannot be sent to a replica".into())
            })?)),
        };
        let wire = WireGenerate {
            model: req.model,
            prompt: req.prompt,
            params: req.params,
            constraint,
            trace: None,
        };
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        lock(&link.pending).insert(
            id,
            Pend {
                tx,
                cancel: req.cancel,
                cancel_sent: false,
            },
        );
        if let Err(e) = link.write(&ReplicaRequest::Generate {
            id,
            req: Box::new(wire),
            fetch,
        }) {
            lock(&link.pending).remove(&id);
            return Err(EngineError::Other(format!(
                "replica {} is not reachable: {e}",
                self.replica
            )));
        }
        // The link may have died between the check and the insert: nobody would end this request.
        if link.dead.load(Ordering::SeqCst)
            && let Some(p) = lock(&link.pending).remove(&id)
        {
            let _ = deliver(
                &p.tx,
                GenEvent::Error(format!("replica {} was lost", self.replica)),
            );
        }
        Ok(rx)
    }
}

impl Engine for RemoteEngine {
    fn models(&self) -> Vec<ModelInfo> {
        self.front.models()
    }

    fn vocab(&self, model: &str) -> Result<Arc<tetraeder_gateway::engine::Vocab>, EngineError> {
        self.front.vocab(model)
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<u32>, EngineError> {
        self.front.tokenize(model, text)
    }

    fn tokenize_with(
        &self,
        model: &str,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>, EngineError> {
        self.front
            .tokenize_with(model, text, add_special, parse_special)
    }

    fn fim_tokens(&self, model: &str) -> tetraeder_gateway::engine::FimTokens {
        self.front.fim_tokens(model)
    }

    fn ready(&self) -> bool {
        self.is_connected()
    }

    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        self.generate_with(req, None)
    }

    fn json_schema_to_grammar(&self, model: &str, schema: &str) -> Result<String, EngineError> {
        self.front.json_schema_to_grammar(model, schema)
    }

    fn check_grammar(&self, model: &str, grammar: &str) -> Result<(), EngineError> {
        self.front.check_grammar(model, grammar)
    }
}

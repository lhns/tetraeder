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

use crate::kv::{PrefixKey, PrefixStore, fetch_prefix};
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
    },
    Event {
        id: u64,
        event: GenEvent,
    },
    /// The outcome of a requested prefix fetch, before the request's first event.
    Fetched {
        id: u64,
        fetched: bool,
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
}

/// The replica side: serves one engine over frames.
pub struct ReplicaServer {
    engine: Arc<dyn Engine>,
    cap: u32,
    kv: Option<KvHooks>,
    factories: Mutex<HashMap<String, Arc<ConstraintFactory>>>,
}

impl ReplicaServer {
    pub fn new(engine: Arc<dyn Engine>, cap: u32, kv: Option<KvHooks>) -> Arc<Self> {
        Arc::new(Self {
            engine,
            cap,
            kv,
            factories: Mutex::new(HashMap::new()),
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
                ReplicaRequest::Probe => {
                    let (s, tx) = (Arc::clone(self), Arc::clone(&tx));
                    std::thread::spawn(move || {
                        let r = s.probe_step();
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
        if let Some(f) = &fetch {
            let ok = self.fetch(f);
            if !send(tx, &ReplicaEvent::Fetched { id, fetched: ok }) {
                return;
            }
        }
        let constraint = match wire
            .constraint
            .as_ref()
            .map(|c| self.constraint(&wire.model, c))
        {
            Some(Err(e)) => {
                let _ = event(GenEvent::Error(format!("constraint: {e}")));
                return;
            }
            Some(Ok(c)) => Some(c),
            None => None,
        };
        let rx = self.engine.generate(GenerateRequest {
            model: wire.model,
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

    /// Fetches the snapshot into this replica's store; false: recompute.
    fn fetch(&self, f: &FetchFrom) -> bool {
        let Some(kv) = &self.kv else { return false };
        if f.key.format != kv.store.kv_format() {
            return false;
        }
        let Some(peer) = (kv.peers)(&f.node) else {
            return false;
        };
        match fetch_prefix(kv.transport.as_ref(), &peer, &f.key) {
            Ok(Some(bytes)) => {
                kv.store.put(f.key.clone(), bytes);
                true
            }
            Ok(None) => false,
            Err(e) => {
                tracing::warn!(node = %f.node, "prefix fetch failed: {e}");
                false
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
            ReplicaEvent::Fetched { fetched, .. } => {
                crate::metrics::prefix_fetch(if fetched { "fetched" } else { "recomputed" });
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
                }) => Ok(ReplicaInfo {
                    models,
                    kv_format,
                    cap,
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
